#!/usr/bin/env python
"""
Calibration test: vertical distance for 25 % target pixel coverage.

Mounts the FPV camera pointing straight down (HPR 0, -90, 0) and sweeps
through vertical distances to the magenta sphere (r = 0.25 m by default).
At each height the HSV-detected pixel fraction is measured — the same
pipeline the reward system uses.

Aspect-ratio effect
───────────────────
The training window is 1920×1080 (16:9) but the reward image is 32×32
(square).  Panda3D adjusts the lens VFOV to match the buffer aspect
ratio, so a 16:9 buffer has a narrower VFOV than a square one.  Because
both the sphere projection AND the total pixel count scale together
through any resize, the fraction is preserved — but it differs between
aspect ratios because the effective film height changes.

    Window     Effective film   Theoretical d  (r=0.25, frac=0.25)
    ─────────  ───────────────  ──────────────
    1:1        36 × 36 mm       ≈ 1.14 m
    3:2        36 × 24 mm       ≈ 1.38 m   (film native)
    16:9       36 × 20.25 mm    ≈ 1.50 m   (training default)

The test defaults to the training 16:9 aspect.  Pass --square to use a
square window (eliminates aspect distortion for cross-validation).

Output → experiments/altitude_calibration/
  altitude_vs_fraction.png   — fraction-vs-height curve with theory overlay
  calibration_result.txt     — calibrated distance and parameters
  sample_*.png               — FPV image + mask at selected heights

Usage
─────
    python tests/test_altitude_calibration.py
    python tests/test_altitude_calibration.py --square
    python tests/test_altitude_calibration.py --h-min 0.5 --h-max 3.0 --h-step 0.02
"""

import argparse
import math
import os
import sys
import numpy as np
import cv2
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from panda3d.core import Filename, loadPrcFile, loadPrcFileData


# ── CLI (must run before ShowBase) ────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Find vertical distance for 25 %% target pixel coverage")
    p.add_argument('--square', action='store_true',
                   help="Use a 512×512 window (no aspect distortion)")
    p.add_argument('--win-size', type=int, default=None,
                   help="Override square window size (implies --square)")
    p.add_argument('--h-min', type=float, default=0.5,
                   help="Minimum height to test (m)")
    p.add_argument('--h-max', type=float, default=3.0,
                   help="Maximum height to test (m)")
    p.add_argument('--h-step', type=float, default=0.05,
                   help="Height step (m)")
    p.add_argument('--target-fraction', type=float, default=0.25,
                   help="Desired pixel fraction (0-1)")
    p.add_argument('--target-radius', type=float, default=0.25,
                   help="Sphere radius (m)")
    p.add_argument('--output-dir', type=str,
                   default='./experiments/altitude_calibration')
    p.add_argument('--capture-sizes', nargs='+', type=int,
                   default=[32, 128],
                   help="Image sizes to measure (px)")
    return p.parse_args()


args = parse_args()

# Load Panda3D config
loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

# Override window size if requested
if args.win_size is not None:
    loadPrcFileData('', f'win-size {args.win_size} {args.win_size}')
elif args.square:
    loadPrcFileData('', 'win-size 512 512')

from direct.showbase.ShowBase import ShowBase
from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


# ══════════════════════════════════════════════════════════════════════
#  Theoretical model (pinhole)
# ══════════════════════════════════════════════════════════════════════

def effective_film_height(film_w_mm, film_h_mm, buffer_w, buffer_h):
    """Compute effective film height after Panda3D aspect-ratio adjustment.

    Panda3D keeps the horizontal FOV (based on film_w) and adjusts the
    vertical FOV to match the buffer aspect ratio:
        effective_h = film_w / buffer_aspect
    """
    buf_aspect = buffer_w / buffer_h
    film_aspect = film_w_mm / film_h_mm
    if abs(buf_aspect - film_aspect) > 0.01:
        return film_w_mm / buf_aspect
    return film_h_mm


def theoretical_fraction(distance, sphere_radius,
                         focal_mm=45, film_w_mm=36, eff_film_h_mm=24):
    """Pixel fraction at given distance (pinhole model).

    fraction = π · f² · r² / ((d² - r²) · W · H_eff)
    """
    f = focal_mm
    r = sphere_radius * 1000  # m → mm
    d = distance * 1000       # m → mm
    W = film_w_mm
    H = eff_film_h_mm
    if d <= r:
        return 1.0
    return min(1.0, math.pi * f**2 * r**2 / ((d**2 - r**2) * W * H))


def theoretical_distance(target_fraction, sphere_radius,
                         focal_mm=45, film_w_mm=36, eff_film_h_mm=24):
    """Distance for a given fraction (pinhole model).

    d = √(π·f²·r² / (frac·W·H_eff) + r²)
    """
    f = focal_mm
    r = sphere_radius * 1000
    W = film_w_mm
    H = eff_film_h_mm
    d_sq = math.pi * f**2 * r**2 / (target_fraction * W * H) + r**2
    return math.sqrt(d_sq) / 1000  # mm → m


# ══════════════════════════════════════════════════════════════════════
#  Application
# ══════════════════════════════════════════════════════════════════════

class AltitudeCalibrationApp(ShowBase):
    N_SETTLE = 10  # render frames before capture (buffer stabilisation)

    def __init__(self):
        ShowBase.__init__(self)
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone …")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Remove interactive camera task
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)

        # ── FPV camera: pointing STRAIGHT DOWN ──────────────────────
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.05)   # slightly below drone
        self.fpv_camera.cam.setHpr(0, -90, 0)      # straight down
        self.fpv_camera.buffer.setActive(1)

        # ── Read actual lens & buffer info ──────────────────────────
        buf_w = self.fpv_camera.buffer.getXSize()
        buf_h = self.fpv_camera.buffer.getYSize()
        lens  = self.fpv_camera.cam.node().getLens()
        self.buf_w, self.buf_h = buf_w, buf_h
        self.film_w = lens.getFilmSize()[0]
        self.film_h = lens.getFilmSize()[1]
        self.focal  = lens.getFocalLength()
        self.eff_h  = effective_film_height(
            self.film_w, self.film_h, buf_w, buf_h)

        print(f"  Buffer:         {buf_w}×{buf_h}  "
              f"(aspect {buf_w/buf_h:.3f})")
        print(f"  Film (nominal): {self.film_w}×{self.film_h} mm  "
              f"(aspect {self.film_w/self.film_h:.2f})")
        print(f"  Film (effective): {self.film_w}×{self.eff_h:.2f} mm  "
              f"(after aspect adjust)")
        print(f"  Focal length:   {self.focal} mm")
        print(f"  HFOV: {lens.getHfov():.1f}°   VFOV: {lens.getVfov():.1f}°")

        # ── Environment ─────────────────────────────────────────────
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='fixed',
            target_range=3.0,
            target_speed=0.0,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=1000,
            t_step=0.01,
            direct_control=1,
            target_radius=args.target_radius,
            filming_mode=True,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def _force_hover(self, drone_z):
        """Place drone at (0, 0, drone_z) with zero rotation.
        Place target at (0, 0, 0) directly below."""
        state = self.env.base_env.state
        state[:] = 0.0
        state[4] = drone_z             # z position
        state[6] = 1.0                 # quaternion w (scalar-first identity)
        self.env.base_env.ang[:] = 0.0
        self.env._update_visualization()

        self.env.target_pos = np.array([0.0, 0.0, 0.0])
        self.env._update_target_marker_pos()

        # External camera — side view for debugging
        rz = drone_z + 5  # rendering z (offset)
        self.cam.setPos(4, -4, rz)
        self.cam.lookAt(0, 0, 5)

    def _detect_fraction(self, capture_size):
        """Render scene, capture FPV image, detect magenta pixels.

        Uses the same HSV pipeline as the reward system.

        Returns (fraction, rgb_image, binary_mask)
        """
        for _ in range(self.N_SETTLE):
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

        ok, rgba = self.fpv_camera.get_image()
        if not ok or rgba is None:
            return 0.0, None, None

        # First 3 channels from Panda3D buffer (BGRA → BGR)
        img = rgba[:, :, :3]
        img_resized = cv2.resize(img, (capture_size, capture_size),
                                 interpolation=cv2.INTER_AREA)

        # HSV detection — same as _compute_new_reward / _compute_visual_tracking_reward
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))

        count = int(np.sum(mask > 0))
        total = capture_size * capture_size
        return count / total, img_resized, mask

    # ── Main sweep ───────────────────────────────────────────────────

    def run_test(self):
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Reset environment once (creates target marker, etc.)
        self.env.reset()

        heights = np.arange(args.h_min, args.h_max + 1e-9, args.h_step)
        capture_sizes = args.capture_sizes
        target_frac = args.target_fraction
        r = args.target_radius

        # ── Theoretical predictions ──
        d_theory = theoretical_distance(
            target_frac, r, self.focal, self.film_w, self.eff_h)
        theory_h = np.linspace(max(r + 0.01, args.h_min), args.h_max, 300)
        theory_f = [theoretical_fraction(
            h, r, self.focal, self.film_w, self.eff_h) for h in theory_h]

        # Also compute for other aspect ratios (reference)
        d_native = theoretical_distance(
            target_frac, r, self.focal, self.film_w, self.film_h)
        d_16_9 = theoretical_distance(
            target_frac, r, self.focal, self.film_w,
            effective_film_height(self.film_w, self.film_h, 1920, 1080))
        d_square = theoretical_distance(
            target_frac, r, self.focal, self.film_w,
            effective_film_height(self.film_w, self.film_h, 512, 512))

        print(f"\n{'='*65}")
        print(f"  ALTITUDE CALIBRATION TEST")
        print(f"{'='*65}")
        print(f"  Sphere radius:     {r} m")
        print(f"  Target fraction:   {target_frac*100:.0f} %")
        print(f"  Camera:            f={self.focal}mm, "
              f"film={self.film_w}×{self.film_h}mm")
        print(f"  Buffer:            {self.buf_w}×{self.buf_h}")
        print(f"  Effective film h:  {self.eff_h:.2f} mm")
        print(f"  Heights:           {args.h_min:.2f} → {args.h_max:.2f} m "
              f"(step {args.h_step:.2f}, n={len(heights)})")
        print(f"  Capture sizes:     {capture_sizes}")
        print(f"\n  Theoretical distances (pinhole):")
        print(f"    This buffer ({self.buf_w}×{self.buf_h}):"
              f"  {d_theory:.4f} m")
        print(f"    Film native  (3:2):         {d_native:.4f} m")
        print(f"    Training     (16:9):         {d_16_9:.4f} m")
        print(f"    Square       (1:1):          {d_square:.4f} m")
        print(f"{'='*65}\n")

        # ── Sweep ────────────────────────────────────────────────────
        results = {sz: [] for sz in capture_sizes}
        sample_heights = set()

        # Mark interesting heights for sample images
        for h_sample in [0.75, 1.0, d_theory, 1.5, 2.0, 2.5]:
            closest = heights[np.argmin(np.abs(heights - h_sample))]
            sample_heights.add(closest)

        sample_imgs = {}

        for i, h in enumerate(heights):
            self._force_hover(h)

            line_parts = [f"  h = {h:5.2f} m"]
            for sz in capture_sizes:
                frac, img, mask = self._detect_fraction(sz)
                results[sz].append(frac)
                line_parts.append(f"{sz}px: {frac*100:5.1f}%")

            print("  |  ".join(line_parts))

            # Save sample images at selected heights
            if h in sample_heights:
                frac_lo, img_lo, mask_lo = self._detect_fraction(32)
                frac_hi, img_hi, mask_hi = self._detect_fraction(256)
                sample_imgs[h] = (img_lo, mask_lo, frac_lo,
                                  img_hi, mask_hi, frac_hi)

        # ── Find empirical crossing ──────────────────────────────────
        calibrated = {}
        for sz in capture_sizes:
            fracs = np.array(results[sz])
            for j in range(len(fracs) - 1):
                if fracs[j] >= target_frac and fracs[j + 1] < target_frac:
                    # Linear interpolation
                    f0, f1 = fracs[j], fracs[j + 1]
                    h0, h1 = heights[j], heights[j + 1]
                    d_interp = h0 + (target_frac - f0) / (f1 - f0) * (h1 - h0)
                    calibrated[sz] = d_interp
                    break
            else:
                calibrated[sz] = None

        # ── Report ───────────────────────────────────────────────────
        print(f"\n{'='*65}")
        print(f"  RESULTS")
        print(f"{'='*65}")
        print(f"  Theoretical (pinhole, this buffer): {d_theory:.4f} m")
        for sz in capture_sizes:
            d = calibrated.get(sz)
            if d is not None:
                err_pct = abs(d - d_theory) / d_theory * 100
                print(f"  Empirical  ({sz:3d}×{sz:3d} px):  "
                      f"         {d:.4f} m  "
                      f"(Δ = {d - d_theory:+.4f} m, {err_pct:.1f} %)")
            else:
                print(f"  Empirical  ({sz:3d}×{sz:3d} px):  "
                      f"         NOT FOUND in [{args.h_min}, {args.h_max}] m")

        # Recommendation: use the smallest capture size (matches reward)
        reward_sz = min(capture_sizes)
        rec = calibrated.get(reward_sz, d_theory)
        print(f"\n  ► RECOMMENDED HOVER HEIGHT: {rec:.3f} m")
        print(f"    (measured at {reward_sz}×{reward_sz} px — same as reward)")
        print(f"{'='*65}")

        # ── Save outputs ─────────────────────────────────────────────
        self._save_plot(heights, results, theory_h, theory_f,
                        target_frac, d_theory, calibrated, out_dir)
        self._save_samples(sample_imgs, out_dir)
        self._save_result_txt(d_theory, d_native, d_16_9, d_square,
                              calibrated, rec, out_dir)

        print(f"\n  All output saved to {out_dir}/")

    # ── Plot ─────────────────────────────────────────────────────────

    def _save_plot(self, heights, results, th_h, th_f, target_frac,
                   theory_d, calibrated, out_dir):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib unavailable — plot skipped)")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Theory curve
        ax.plot(th_h, [f * 100 for f in th_f], 'k--', lw=1.5,
                label=f'Pinhole theory → {theory_d:.3f} m')

        # Empirical curves
        colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                   'tab:purple']
        for idx, sz in enumerate(sorted(results.keys())):
            fracs_pct = [f * 100 for f in results[sz]]
            c = colours[idx % len(colours)]
            ax.plot(heights, fracs_pct, 'o-', ms=3, lw=1.2, color=c,
                    label=f'{sz}×{sz} px')
            d = calibrated.get(sz)
            if d is not None:
                ax.axvline(d, color=c, ls=':', lw=0.9, alpha=0.7)
                ax.annotate(f'{d:.3f} m', xy=(d, target_frac * 100),
                            xytext=(d + 0.15, target_frac * 100 + 8),
                            fontsize=8, color=c,
                            arrowprops=dict(arrowstyle='->', color=c,
                                            lw=0.8))

        # Target line
        ax.axhline(target_frac * 100, color='red', ls='-', lw=2, alpha=0.35,
                   label=f'Target = {target_frac * 100:.0f} %')

        ax.set_xlabel('Vertical distance drone → sphere (m)')
        ax.set_ylabel('Detected pixel fraction (%)')
        ax.set_title(
            f'Altitude Calibration — magenta sphere '
            f'(r = {args.target_radius} m, '
            f'buffer {self.buf_w}×{self.buf_h})')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(heights[0] - 0.1, heights[-1] + 0.1)
        y_max = max(f * 100 for v in results.values() for f in v)
        ax.set_ylim(0, min(105, y_max + 10))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(out_dir / 'altitude_vs_fraction.png'), dpi=150)
        plt.close()
        print(f"  Plot saved: altitude_vs_fraction.png")

    # ── Sample images ────────────────────────────────────────────────

    def _save_samples(self, sample_imgs, out_dir):
        if not sample_imgs:
            return
        UP = 8
        for h in sorted(sample_imgs.keys()):
            img32, mask32, frac32, img256, mask256, frac256 = sample_imgs[h]
            if img32 is None or img256 is None:
                continue

            # Upscale 32×32 for visibility
            vis32 = cv2.resize(img32, (32 * UP, 32 * UP),
                               interpolation=cv2.INTER_NEAREST)
            mvis32 = cv2.resize(mask32, (32 * UP, 32 * UP),
                                interpolation=cv2.INTER_NEAREST)
            mvis32_rgb = cv2.cvtColor(mvis32, cv2.COLOR_GRAY2BGR)

            # 256px panels
            mask256_rgb = cv2.cvtColor(mask256, cv2.COLOR_GRAY2BGR)

            # Normalise to same height
            tgt = 256
            panels = [
                cv2.resize(vis32, (tgt, tgt)),
                cv2.resize(mvis32_rgb, (tgt, tgt)),
                cv2.resize(img256, (tgt, tgt)),
                cv2.resize(mask256_rgb, (tgt, tgt)),
            ]
            combined = np.hstack(panels)

            # Label
            cv2.putText(combined,
                        f"h={h:.2f}m  32px:{frac32*100:.1f}%  "
                        f"256px:{frac256*100:.1f}%",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1)

            fname = f"sample_{h:.2f}m.png"
            cv2.imwrite(str(out_dir / fname),
                        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        print(f"  {len(sample_imgs)} sample images saved")

    # ── Text summary ─────────────────────────────────────────────────

    def _save_result_txt(self, d_theory, d_native, d_16_9, d_square,
                         calibrated, recommended, out_dir):
        lines = [
            "ALTITUDE CALIBRATION RESULT",
            "=" * 50,
            f"Date:                 {__import__('datetime').date.today()}",
            f"Sphere radius:        {args.target_radius} m",
            f"Target fraction:      {args.target_fraction * 100:.0f} %",
            f"Camera focal length:  {self.focal} mm",
            f"Camera film size:     {self.film_w}×{self.film_h} mm",
            f"Buffer:               {self.buf_w}×{self.buf_h}",
            f"Effective film h:     {self.eff_h:.2f} mm",
            "",
            "THEORETICAL (pinhole model):",
            f"  This buffer:        {d_theory:.4f} m",
            f"  Film native (3:2):  {d_native:.4f} m",
            f"  Training   (16:9):  {d_16_9:.4f} m",
            f"  Square     (1:1):   {d_square:.4f} m",
            "",
            "EMPIRICAL:",
        ]
        for sz, d in sorted(calibrated.items()):
            if d is not None:
                lines.append(f"  {sz}×{sz} px:           {d:.4f} m")
            else:
                lines.append(f"  {sz}×{sz} px:           NOT FOUND")
        lines += [
            "",
            f"RECOMMENDED HOVER HEIGHT: {recommended:.3f} m",
            f"  (measured at {min(args.capture_sizes)}×{min(args.capture_sizes)} px"
            f" — same resolution as reward pipeline)",
            "",
            "ASPECT RATIO NOTES:",
            "  The effective vertical FOV depends on the buffer aspect ratio.",
            "  Panda3D adjusts VFOV to match: eff_h = film_w / buffer_aspect.",
            "  A 16:9 buffer narrows the VFOV → sphere appears bigger →",
            "  larger distance needed for 25 %.",
            "  Ensure the training window matches the calibration window.",
        ]

        txt = "\n".join(lines)
        (out_dir / 'calibration_result.txt').write_text(txt, encoding='utf-8')
        print(f"  Result saved: calibration_result.txt")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = AltitudeCalibrationApp()
    app.run_test()
    app.userExit()
    sys.exit(0)
