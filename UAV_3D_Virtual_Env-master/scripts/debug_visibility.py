#!/usr/bin/env python
"""
debug_visibility.py — Diagnose why `target_visible=False` during v7.x
training despite the sphere being clearly visible in recorded videos.

Tests three hypotheses against the SAME environment configuration used
by `train_hover_track_v7_2.py`:

  H1. HSV threshold too strict.
      The detector requires H∈[140,170], S∈[100,255], V∈[100,255] and
      pixel_count > 2 on a 32×32 image. At that resolution the magenta
      sphere may be antialiased into pinkish/desaturated pixels outside
      the strict range. We re-run the detector with three relaxed masks
      and compare counts.

  H2. Env state forces target_visible=False.
      We dump `stabilization_only`, `_target_visible_last_step`,
      `reward_version`, and confirm `info['target_visible']` from a
      genuine `step()` call matches what `_detect_target_in_image()`
      returns when invoked directly afterwards. Mismatch -> bug in a
      reward branch overriding the flag.

  H3. The detector runs on a different buffer than the one in videos.
      `EpisodeRecorder` uses `_last_high_freq_image` for FPV. The
      detector also uses `_last_high_freq_image`. We capture once,
      hash both reads, and verify they are byte-identical. We also
      pull the camera buffer DIRECTLY (bypassing the env) and compare
      to confirm the env's cached buffer is current and not stale.

Output (in `experiments/debug_visibility/`):
  - frame_h{Z}_o{O}_raw.png        : original 32×32 captured frame (RGB)
  - frame_h{Z}_o{O}_upscaled.png   : 8× upscaled for inspection
  - frame_h{Z}_o{O}_hsv_h.png      : H channel
  - frame_h{Z}_o{O}_hsv_s.png      : S channel
  - frame_h{Z}_o{O}_hsv_v.png      : V channel
  - frame_h{Z}_o{O}_mask_strict.png: current detector mask
  - frame_h{Z}_o{O}_mask_loose.png : relaxed mask comparison
  - frame_h{Z}_o{O}_external.png   : external bird-view camera (sanity)
  - report.json                    : numerical summary
  - report.txt                     : human-readable verdict
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from panda3d.core import Filename, loadPrcFile, loadPrcFileData

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))
loadPrcFileData('', 'win-size 64 64')
loadPrcFileData('', 'undecorated true')

from direct.showbase.ShowBase import ShowBase

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


# ─────────────────────────────────────────────────────────────────────
# HSV mask variants — same image, different thresholds
# ─────────────────────────────────────────────────────────────────────

HSV_VARIANTS = [
    # (name, lower (H,S,V), upper (H,S,V))
    ("strict_current",   (140, 100, 100), (170, 255, 255)),  # the production rule
    ("loose_h",          (130, 100, 100), (175, 255, 255)),  # wider hue
    ("loose_sv",         (140,  50,  50), (170, 255, 255)),  # tolerate pale/dark magenta
    ("very_loose",       (125,  40,  40), (180, 255, 255)),  # both
    ("anything_pinkish", (120,  20,  20), (180, 255, 255)),  # last-resort sanity
]


def apply_hsv_mask(rgb_img, lower, upper):
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask, hsv


def hsv_stats_of_brightest(rgb_img, top_k=10):
    """Return HSV of the K brightest pixels — should reveal the sphere."""
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].flatten()
    idx = np.argsort(v)[-top_k:][::-1]
    h = hsv.shape[1]
    out = []
    for i in idx:
        y, x = i // h, i % h
        out.append({
            "y": int(y), "x": int(x),
            "H": int(hsv[y, x, 0]),
            "S": int(hsv[y, x, 1]),
            "V": int(hsv[y, x, 2]),
            "RGB": [int(rgb_img[y, x, 0]),
                    int(rgb_img[y, x, 1]),
                    int(rgb_img[y, x, 2])],
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# Diagnostic harness
# ─────────────────────────────────────────────────────────────────────

class VisibilityDebugApp(ShowBase):
    """Mirrors the env setup from `train_hover_track_v7_2.py` exactly."""

    def __init__(self):
        ShowBase.__init__(self)
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("[setup] Loading 3D scene...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)
        self.taskMgr.remove('Camera Movement')

        # FPV camera — pointing DOWN, identical config to training
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        # External camera for sanity reference
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.buffer.setActive(1)

        print("[setup] Creating env (same config as v7.2 training)...")
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
            use_target=True,
            target_mode='moving',
            target_speed=0.0,
            target_radius=0.25,
            lemniscate_scale=2.0,
            filming_mode=True,
            enable_collisions=False,
            n=3000,
            t_step=0.01,
            direct_control=1,
            centroid_obs=True,
            camera_down=True,
            hover_height=1.394,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.2,
            init_vel_range=0.10,
            init_ang_range=0.03,
            reward_version='v3.1',
        )

    def place_drone_over_target(self, drone_z, offset_xy=(0.0, 0.0),
                                 lemniscate_phase=0.0):
        """Force a deterministic configuration for diagnostic capture."""
        # Reset env first to initialise everything
        self.env.reset(seed=42)

        # Compute the lemniscate point we want the target at
        x, y = self.env._lemniscate_point(lemniscate_phase)
        self.env.target_pos = np.array([x, y, 0.0])
        self.env._update_target_marker_pos()

        # Place the drone ABOVE the target at the requested altitude
        state = self.env.base_env.state.copy()
        state[0] = self.env.target_pos[0] + offset_xy[0]   # x
        state[2] = self.env.target_pos[1] + offset_xy[1]   # y
        state[4] = self.env.target_pos[2] + drone_z         # z
        state[1] = 0.0  # vx
        state[3] = 0.0  # vy
        state[5] = 0.0  # vz
        # Identity quaternion: drone level, yaw=0
        state[6] = 1.0
        state[7] = 0.0
        state[8] = 0.0
        state[9] = 0.0
        state[10:13] = 0.0  # angular velocities

        self.env.base_env.state = state.copy()
        self.env.base_env.previous_state = state.copy()
        self.env._update_visualization()

        # Force render + capture
        self.graphicsEngine.renderFrame()
        # Two extra task-manager ticks: Panda3D buffers are double-buffered
        # and the first renderFrame after a parent-reparenting can return
        # the previous frame. Two ticks guarantee the new frame is visible.
        self.taskMgr.step()
        self.taskMgr.step()
        self.env._capture_camera_images(force_capture=True)


# ─────────────────────────────────────────────────────────────────────
# Per-configuration diagnostic
# ─────────────────────────────────────────────────────────────────────

def diagnose_config(app, drone_z, offset_xy, label, out_dir):
    print(f"\n{'='*66}")
    print(f"[{label}]  z={drone_z}m  offset_xy={offset_xy}")
    print('=' * 66)

    app.place_drone_over_target(drone_z, offset_xy)
    env = app.env

    # ── Buffer capture (the same buffer the detector uses) ──
    img = env._last_high_freq_image
    if img is None:
        print("  [FATAL] _last_high_freq_image is None (capture failed).")
        return {"label": label, "fatal": "capture_returned_none"}

    print(f"  Buffer shape: {img.shape}  dtype: {img.dtype}  "
          f"min={img.min()} max={img.max()} mean={img.mean():.1f}")

    # H3 — buffer consistency: pull the camera DIRECTLY and compare
    ok, raw = app.fpv_camera.get_image()
    direct_match = None
    if ok and raw is not None:
        raw_rgb = raw[:, :, :3]
        raw_resized = cv2.resize(raw_rgb, (32, 32),
                                 interpolation=cv2.INTER_AREA)
        direct_match = bool(np.array_equal(raw_resized, img))
    print(f"  H3 buffer-vs-direct match: {direct_match}")

    # External bird-view (for visual sanity that the scene is correct)
    ok_ext, ext = app.ext_camera.get_image()
    if ok_ext and ext is not None:
        cv2.imwrite(str(out_dir / f"frame_{label}_external.png"),
                    cv2.cvtColor(ext[:, :, :3], cv2.COLOR_RGB2BGR))

    # Save raw + upscaled FPV
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / f"frame_{label}_raw.png"), bgr)
    cv2.imwrite(str(out_dir / f"frame_{label}_upscaled.png"),
                cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_NEAREST))

    # Save HSV channels (upscaled greyscale)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    for ci, ch in enumerate(['h', 's', 'v']):
        up = cv2.resize(hsv[:, :, ci], (256, 256),
                        interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"frame_{label}_hsv_{ch}.png"), up)

    # ── H1 — masks under different thresholds ──
    masks_summary = {}
    for name, lo, hi in HSV_VARIANTS:
        mask, _ = apply_hsv_mask(img, lo, hi)
        cnt = int(np.sum(mask > 0))
        masks_summary[name] = {
            "lower": list(lo), "upper": list(hi),
            "pixel_count": cnt,
            "would_be_visible": cnt > 2,
        }
        cv2.imwrite(str(out_dir / f"frame_{label}_mask_{name}.png"),
                    cv2.resize(mask, (256, 256),
                               interpolation=cv2.INTER_NEAREST))
    print("  H1 mask counts:")
    for name, info in masks_summary.items():
        marker = "[ok]" if info["would_be_visible"] else "[--]"
        print(f"     {marker} {name:<20s} -> {info['pixel_count']:>4d}px  "
              f"({info['lower']} -> {info['upper']})")

    # HSV of the brightest pixels (where the sphere SHOULD be)
    brightest = hsv_stats_of_brightest(img, top_k=8)
    print("  Top-8 brightest pixels (HSV / RGB):")
    for p in brightest:
        in_strict = (140 <= p['H'] <= 170 and p['S'] >= 100 and p['V'] >= 100)
        marker = "MAGENTA" if in_strict else "       "
        print(f"     ({p['x']:>2d},{p['y']:>2d})  "
              f"H={p['H']:>3d} S={p['S']:>3d} V={p['V']:>3d}  "
              f"RGB={p['RGB']}  {marker}")

    # ── H2 — env-state diagnostic ──
    print("  H2 env-state flags:")
    print(f"     reward_version            = {env.reward_version}")
    print(f"     stabilization_only        = {env.stabilization_only}")
    print(f"     _target_visible_last_step = "
          f"{env._target_visible_last_step}")
    print(f"     camera_down               = {env.camera_down}")
    print(f"     filming_mode              = {env.filming_mode}")
    print(f"     centroid_obs              = {env.centroid_obs}")

    # Detect via the env's own method, then via a real step()
    cx, cy, frac, vis = env._detect_target_in_image()
    print(f"  Direct _detect_target_in_image(): "
          f"cx={cx:.3f} cy={cy:.3f} frac={frac:.4f} vis={vis}")

    # Full step to see what info dict reports
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, rew, term, trunc, info = env.step(zero_action)
    print(f"  After env.step([0,0,0,0]):")
    print(f"     info.target_visible = {info.get('target_visible')}")
    print(f"     info.target_pixels  = {info.get('target_pixels')}")
    print(f"     info.r_centering    = {info.get('r_centering')}")
    print(f"     info.r_stability    = {info.get('r_stability')}")

    # Did the buffer change after step?
    img_after = env._last_high_freq_image
    same_buffer_after_step = bool(np.array_equal(img_after, img))
    print(f"  Buffer identical pre/post step: {same_buffer_after_step}")

    return {
        "label": label,
        "drone_z": drone_z,
        "offset_xy": list(offset_xy),
        "buffer_shape": list(img.shape),
        "buffer_min_max_mean": [int(img.min()), int(img.max()),
                                round(float(img.mean()), 2)],
        "h3_direct_buffer_matches_env": direct_match,
        "h1_masks": masks_summary,
        "brightest_pixels": brightest,
        "h2_env_flags": {
            "reward_version": env.reward_version,
            "stabilization_only": bool(env.stabilization_only),
            "target_visible_last_step": bool(
                env._target_visible_last_step),
            "camera_down": bool(env.camera_down),
            "filming_mode": bool(env.filming_mode),
            "centroid_obs": bool(env.centroid_obs),
        },
        "direct_detect": {
            "cx": float(cx), "cy": float(cy),
            "frac": float(frac), "vis": float(vis),
        },
        "step_info": {
            "target_visible": bool(info.get('target_visible', False)),
            "target_pixels": int(info.get('target_pixels', 0)),
            "r_centering": float(info.get('r_centering', 0.0)),
            "r_stability": float(info.get('r_stability', 0.0)),
        },
        "buffer_identical_pre_post_step": same_buffer_after_step,
    }


# ─────────────────────────────────────────────────────────────────────
# Verdict synthesizer
# ─────────────────────────────────────────────────────────────────────

def write_verdict(report, out_dir):
    lines = []
    lines.append("=" * 70)
    lines.append("  VISIBILITY DIAGNOSTIC VERDICT")
    lines.append("=" * 70)

    # H3 — buffer integrity
    h3_ok = all(c.get("h3_direct_buffer_matches_env") is True
                for c in report["configs"])
    lines.append(f"\nH3 (buffer mismatch): "
                 f"{'NO BUG' if h3_ok else '[WARN] BUFFER DRIFT DETECTED'}")
    if not h3_ok:
        lines.append("    -> env._last_high_freq_image differs from a fresh "
                     "camera read.\n      The detector is operating on a "
                     "stale buffer.")

    # H2 — env state
    flags = [c.get("h2_env_flags", {}) for c in report["configs"]]
    h2_problems = []
    for f in flags:
        if f.get("stabilization_only"):
            h2_problems.append("stabilization_only=True")
        if f.get("reward_version") != "v3.1":
            h2_problems.append(
                f"reward_version={f.get('reward_version')} != v3.1")
    lines.append(f"\nH2 (env-flag override): "
                 f"{'NO BUG' if not h2_problems else '[WARN] ' + ', '.join(h2_problems)}")

    # Mismatch between direct detect and step info
    h2b_problems = []
    for c in report["configs"]:
        d_vis = c["direct_detect"]["vis"] > 0
        s_vis = c["step_info"]["target_visible"]
        if d_vis != s_vis:
            h2b_problems.append(
                f"{c['label']}: direct={d_vis} but step_info={s_vis}")
    if h2b_problems:
        lines.append(f"\nH2b (info-dict propagation): [WARN] MISMATCH:")
        for p in h2b_problems:
            lines.append(f"    -> {p}")
    else:
        lines.append("\nH2b (info-dict propagation): NO BUG")

    # H1 — HSV thresholds
    lines.append("\nH1 (HSV threshold) — pixel counts per variant:")
    lines.append(f"  {'config':<18s}  " + "  ".join(
        f"{n:<18s}" for n, *_ in HSV_VARIANTS))
    for c in report["configs"]:
        masks = c.get("h1_masks", {})
        row = [f"{c['label']:<18s}"]
        for name, *_ in HSV_VARIANTS:
            cnt = masks.get(name, {}).get("pixel_count", 0)
            row.append(f"{cnt:<18d}")
        lines.append("  " + "  ".join(row))

    strict_zero_loose_nonzero = []
    for c in report["configs"]:
        m = c.get("h1_masks", {})
        if (m.get("strict_current", {}).get("pixel_count", 0) <= 2 and
                m.get("very_loose", {}).get("pixel_count", 0) > 2):
            strict_zero_loose_nonzero.append(c["label"])
    if strict_zero_loose_nonzero:
        lines.append(
            f"\n  [WARN] The strict mask returns 0 but the loose mask DETECTS the "
            f"sphere in:\n     {strict_zero_loose_nonzero}")
        lines.append(
            "    -> H1 CONFIRMED: HSV threshold too tight at 32×32 "
            "resolution.\n      Recommend widening to H∈[125,180] "
            "S≥40 V≥40, or upscaling the\n      detector input to 64×64.")
    else:
        lines.append("\n  -> H1 not confirmed by this run.")

    # Final verdict block
    lines.append("\n" + "=" * 70)
    if (h3_ok and not h2_problems and not h2b_problems and
            strict_zero_loose_nonzero):
        lines.append("  VERDICT: H1 is the bug. Fix the HSV mask.")
    elif not h3_ok:
        lines.append("  VERDICT: H3 is the bug. Detector reads stale buffer.")
    elif h2_problems or h2b_problems:
        lines.append("  VERDICT: H2 is the bug. An env flag or reward branch "
                     "overrides target_visible.")
    else:
        lines.append("  VERDICT: None of H1/H2/H3 reproduced — re-examine "
                     "live training logs or scene lighting.")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(project_root) / "experiments" / "debug_visibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output directory: {out_dir}")

    app = VisibilityDebugApp()

    configs = [
        # (label, drone_z, offset_xy)
        ("h1.4_centered",  1.394, (0.0, 0.0)),    # nominal training spawn
        ("h1.4_offset0.2", 1.394, (0.2, 0.0)),    # mid-FOV
        ("h1.4_offset0.3", 1.394, (0.3, 0.0)),    # near-edge FOV
        ("h1.5_centered",  1.5,   (0.0, 0.0)),    # slightly higher
        ("h2.0_centered",  2.0,   (0.0, 0.0)),    # user-proposed altitude
        ("h2.0_offset0.4", 2.0,   (0.4, 0.0)),    # 2 m + lateral
    ]

    report = {"configs": []}
    for label, z, off in configs:
        try:
            res = diagnose_config(app, z, off, label, out_dir)
            report["configs"].append(res)
        except Exception as e:
            print(f"  [FATAL] exception during {label}: {e}")
            report["configs"].append({"label": label, "exception": str(e)})

    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    write_verdict(report, out_dir)

    print(f"\n[done] artefacts in {out_dir}")


if __name__ == "__main__":
    main()
