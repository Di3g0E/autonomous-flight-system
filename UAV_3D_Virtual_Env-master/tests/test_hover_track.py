#!/usr/bin/env python
"""
Evaluate the trained hover-tracking model with optional spiral search.

Runs N episodes where the drone must keep the magenta sphere centred
in the downward-facing camera.  When the sphere disappears for K
consecutive steps, the pre-trained spiral model takes over until the
sphere is re-acquired, then control blends back to the tracking policy.

Outputs
-------
  experiments/hover_track/videos/episode_*_quad_view.mp4  (quad-view 2x2)
  experiments/hover_track/plots/episode_*_altitude.png    (per-episode altitude)
  experiments/hover_track/plots/all_altitudes.png         (combined altitude)
  experiments/hover_track/telemetry.csv                   (per-step data)
  experiments/hover_track/summary.json                    (aggregate metrics)

Usage:
    python tests/test_hover_track.py
    python tests/test_hover_track.py --target-mode moving --target-speed 0.1
    python tests/test_hover_track.py --no-spiral   (disable spiral fallback)
"""

import argparse, csv, json, os, sys
import numpy as np
import cv2
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from panda3d.core import Filename, loadPrcFile, loadPrcFileData

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from direct.showbase.ShowBase import ShowBase
from stable_baselines3 import SAC

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.agents.spiral_search_controller import SpiralSearchController


# ── Video constants ───────────────────────────────────────────────────
PANEL_SIZE = 480
LABEL_H = 40
SEP = 3


def parse_args():
    p = argparse.ArgumentParser(
        description="Test hover-tracking model with spiral fallback")
    p.add_argument('--model-path', type=str,
                   default='./models/hover_track/best_model.zip')
    p.add_argument('--spiral-model', type=str,
                   default='./models/spiral_follow/best_model.zip')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/hover_track')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=2000,
                   help="Max steps per episode (2000 = 20s)")
    p.add_argument('--hover-height', type=float, default=1.394)
    p.add_argument('--target-mode', type=str, default='fixed',
                   choices=['fixed', 'moving'],
                   help="Target behaviour during test")
    p.add_argument('--target-speed', type=float, default=0.0,
                   help="Speed for moving target (m/s)")
    p.add_argument('--lemniscate-scale', type=float, default=2.5,
                   help="Half-width for lemniscate trajectory")
    p.add_argument('--no-spiral', action='store_true',
                   help="Disable spiral fallback (pure RL)")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--panel-size', type=int, default=PANEL_SIZE)
    return p.parse_args()


class HoverTrackTestApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Aerial camera for main window
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -8, 14)
        self.cam.lookAt(0, 0, 5)

        # FPV camera — pointing DOWN
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        # External camera for recording — fixed, far away to always
        # frame both the drone and the target sphere
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.cam.setPos(0, -18, 25)
        self.ext_camera.cam.lookAt(0, 0, 2)
        self.ext_camera.buffer.setActive(1)

        # Environment
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
            target_mode=args.target_mode,
            target_speed=args.target_speed,
            target_radius=0.25,
            filming_mode=True,
            enable_collisions=False,
            n=args.max_steps,
            t_step=0.01,
            direct_control=1,
            lemniscate_scale=args.lemniscate_scale,
            centroid_obs=True,
            camera_down=True,
            hover_height=args.hover_height,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.2,
            init_vel_range=0.1,
            init_ang_range=0.05,
        )

        # Load tracking model
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found: {args.model_path}")
            sys.exit(1)
        self.model = SAC.load(args.model_path, env=None)
        print(f"Tracking model: {args.model_path}")

        # Spiral search controller
        self.spiral_ctrl = None
        if not args.no_spiral and os.path.exists(args.spiral_model):
            self.spiral_ctrl = SpiralSearchController(
                spiral_model_path=args.spiral_model,
                hover_height=args.hover_height,
            )
            print(f"Spiral model:   {args.spiral_model}")
        elif not args.no_spiral:
            print(f"WARNING: Spiral model not found: {args.spiral_model}")
            print("         Running without spiral fallback.")

        # Output dirs
        self.out_dir = Path(args.output_dir)
        self.video_dir = self.out_dir / 'videos'
        self.plot_dir = self.out_dir / 'plots'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Quad-view helpers (same style as quad_view.png reference)
    # ──────────────────────────────────────────────────────────────────

    def _make_panel(self, img_bgr, title, title_color=(0, 255, 255)):
        PS = self.args.panel_size
        resized = cv2.resize(img_bgr, (PS, PS))
        panel = np.zeros((PS + LABEL_H, PS, 3), dtype=np.uint8)
        panel[LABEL_H:, :] = resized
        cv2.putText(panel, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 2)
        return panel

    def _build_detection_frame(self, img_32_rgb):
        PS = self.args.panel_size
        h, w = img_32_rgb.shape[:2]
        UP = PS // w

        img_bgr = cv2.cvtColor(img_32_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (140, 100, 100), (170, 255, 255))

        pixel_count = int(np.sum(mask > 0))
        frac = pixel_count / (h * w)

        annotated = cv2.resize(img_bgr, (PS, PS),
                               interpolation=cv2.INTER_NEAREST)
        mask_up = cv2.resize(mask, (PS, PS),
                             interpolation=cv2.INTER_NEAREST)

        overlay = np.zeros_like(annotated)
        overlay[mask_up > 0] = (0, 255, 0)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.5, 0)

        cx_img, cy_img = PS // 2, PS // 2
        cv2.line(annotated, (cx_img - 20, cy_img),
                 (cx_img + 20, cy_img), (255, 255, 255), 1)
        cv2.line(annotated, (cx_img, cy_img - 20),
                 (cx_img, cy_img + 20), (255, 255, 255), 1)

        visible = pixel_count > 2
        if visible:
            ys, xs = np.where(mask > 0)
            cent_x = float(np.mean(xs)) * UP
            cent_y = float(np.mean(ys)) * UP
            cv2.circle(annotated, (int(cent_x), int(cent_y)), 8,
                       (0, 0, 255), -1)
            cv2.circle(annotated, (int(cent_x), int(cent_y)), 8,
                       (255, 255, 255), 2)
            x_min, x_max = int(np.min(xs)) * UP, int(np.max(xs) + 1) * UP
            y_min, y_max = int(np.min(ys)) * UP, int(np.max(ys) + 1) * UP
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max),
                          (0, 255, 255), 2)
            cx_n = (np.mean(xs) - w / 2) / (w / 2)
            cy_n = (np.mean(ys) - h / 2) / (h / 2)
            info_text = f"cx={cx_n:+.2f} cy={cy_n:+.2f} frac={frac:.3f}"
        else:
            info_text = "NOT DETECTED"

        cv2.putText(annotated, info_text, (8, PS - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        return annotated



    # ──────────────────────────────────────────────────────────────────
    # Altitude plot generation
    # ──────────────────────────────────────────────────────────────────

    def _plot_altitude(self, times, drone_alts, target_alts, ep_num, path):
        """Generate a single-episode altitude plot using matplotlib."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, drone_alts, label='Drone altitude', color='#2196F3',
                linewidth=1.5)
        ax.plot(times, target_alts, label='Target altitude', color='#E91E63',
                linewidth=1.5, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title(f'Altitude — Episode {ep_num}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    def _plot_all_altitudes(self, all_data, path):
        """Combined altitude plot with all episodes."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap_d = plt.cm.Blues
        cmap_t = plt.cm.Reds
        n = len(all_data)
        for i, (ep, times, d_alts, t_alts) in enumerate(all_data):
            c_d = cmap_d(0.4 + 0.6 * i / max(n - 1, 1))
            c_t = cmap_t(0.4 + 0.6 * i / max(n - 1, 1))
            ax.plot(times, d_alts, color=c_d, linewidth=1.2, alpha=0.8,
                    label=f'Drone Ep{ep}')
            ax.plot(times, t_alts, color=c_t, linewidth=1.2, alpha=0.8,
                    linestyle='--', label=f'Target Ep{ep}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Altitude — All Episodes')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────

    def run_tests(self):
        args = self.args
        np.random.seed(args.seed)
        PS = args.panel_size

        print(f"\n{'='*60}")
        print(f"  Hover-Track Evaluation  (quad-view + altitude plots)")
        print(f"  Target mode: {args.target_mode}  "
              f"Speed: {args.target_speed}  "
              f"Height: {args.hover_height}m")
        print(f"  Episodes: {args.episodes}  "
              f"Max steps: {args.max_steps}")
        print(f"  Spiral: {'ON' if self.spiral_ctrl else 'OFF'}")
        print(f"{'='*60}\n")

        telemetry_path = self.out_dir / 'telemetry.csv'
        results = []
        all_altitude_data = []  # (ep, times, drone_alts, target_alts)

        with open(telemetry_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'episode', 'step', 'drone_x', 'drone_y', 'drone_z',
                'target_x', 'target_y', 'target_z',
                'centroid_x', 'centroid_y', 'fraction', 'visible',
                'r_stability', 'r_centering', 'r_scale',
                'reward', 'action_mag', 'controller_state',
            ])

            for ep in range(args.episodes):
                print(f"  Episode {ep+1}/{args.episodes}...")
                obs, info = self.env.reset()
                if self.spiral_ctrl:
                    self.spiral_ctrl.reset()

                # ── Setup quad-view video writer ──
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                quad_w = 2 * PS + SEP
                quad_h = 2 * (PS + LABEL_H) + SEP + 50
                vpath = self.video_dir / f"episode_{ep+1:03d}_quad_view.mp4"
                w_quad = cv2.VideoWriter(
                    str(vpath), fourcc, args.fps, (quad_w, quad_h))

                frame_interval = max(1, 100 // args.fps)
                step = 0
                ep_rewards = []
                ep_centering = []
                ep_action_mags = []
                ep_times = []
                ep_drone_alts = []
                ep_target_alts = []
                end_reason = "max_steps"
                cumul_reward = 0.0

                while step < args.max_steps:
                    # Determine action
                    vt = info.get('visual_tracking', {})
                    target_visible = vt.get('target_visible', False)
                    ctrl_state = 'track'

                    if self.spiral_ctrl:
                        action = self.spiral_ctrl.get_action(
                            obs, target_visible, self.model, self.env)
                        ctrl_state = self.spiral_ctrl.current_state
                    else:
                        action, _ = self.model.predict(
                            obs, deterministic=True)

                    obs, reward, terminated, truncated, info = \
                        self.env.step(action)
                    step += 1
                    cumul_reward += reward
                    self.taskMgr.step()

                    # Extract data
                    vt = info.get('visual_tracking', {})
                    drone_pos = self.env.base_env.state[0:5:2]
                    target_pos = self.env.target_pos
                    action_mag = float(np.mean(np.abs(action)))

                    ep_rewards.append(reward)
                    ep_action_mags.append(action_mag)
                    ep_times.append(step * 0.01)
                    ep_drone_alts.append(float(drone_pos[2]))
                    ep_target_alts.append(float(target_pos[2]))

                    if (vt.get('target_visible', False)
                            and 'centering_dist' in vt):
                        ep_centering.append(vt['centering_dist'])

                    # Telemetry row
                    writer.writerow([
                        ep + 1, step,
                        round(drone_pos[0], 3), round(drone_pos[1], 3),
                        round(drone_pos[2], 3),
                        round(target_pos[0], 3), round(target_pos[1], 3),
                        round(target_pos[2], 3),
                        round(obs[13], 3), round(obs[14], 3),
                        round(obs[15], 4), round(obs[16], 0),
                        round(vt.get('r_stability', 0), 3),
                        round(vt.get('r_centering', 0), 3),
                        round(vt.get('r_scale', 0), 3),
                        round(reward, 3), round(action_mag, 3),
                        ctrl_state,
                    ])

                    # ── Quad-view frame ──
                    if step % frame_interval == 0:
                        fpv_32 = self.env._last_high_freq_image
                        if fpv_32 is not None:

                            ok_fpv, fpv_rgba = \
                                self.fpv_camera.get_image()
                            ok_ext, ext_rgba = \
                                self.ext_camera.get_image()

                            if ok_fpv and ok_ext:
                                # Panel 1: Raw Camera
                                raw_bgr = cv2.cvtColor(
                                    fpv_rgba, cv2.COLOR_RGBA2BGR)
                                raw_rs = cv2.resize(raw_bgr, (PS, PS))

                                # Panel 2: RL Input 32x32
                                rl_bgr = cv2.resize(
                                    cv2.cvtColor(fpv_32,
                                                 cv2.COLOR_RGB2BGR),
                                    (PS, PS),
                                    interpolation=cv2.INTER_NEAREST)

                                # Panel 3: HSV Detection
                                det = self._build_detection_frame(fpv_32)

                                # Panel 4: External View
                                ext_bgr = cv2.cvtColor(
                                    ext_rgba, cv2.COLOR_RGBA2BGR)
                                ext_rs = cv2.resize(ext_bgr, (PS, PS))

                                # Assemble quad
                                p1 = self._make_panel(
                                    raw_rs, "1. Raw Camera")
                                p2 = self._make_panel(
                                    rl_bgr, "2. RL Input (32x32)",
                                    (255, 0, 255))
                                p3 = self._make_panel(
                                    det, "3. HSV Detection + Centroid",
                                    (0, 255, 0))
                                p4 = self._make_panel(
                                    ext_rs, "4. External View")

                                ph = PS + LABEL_H
                                sv = np.full((ph, SEP, 3), 255,
                                             dtype=np.uint8)
                                sh = np.full((SEP, 2*PS+SEP, 3), 255,
                                             dtype=np.uint8)

                                top = np.hstack([p1, sv, p2])
                                bot = np.hstack([p3, sv, p4])
                                qf = np.vstack([top, sh, bot])

                                # Metrics band
                                band = np.zeros(
                                    (50, qf.shape[1], 3), np.uint8)
                                qf = np.vstack([qf, band])

                                vis = vt.get('target_visible', False)
                                vis_t = "YES" if vis else "NO"
                                vis_c = (0,255,0) if vis else (0,0,255)
                                fnt = cv2.FONT_HERSHEY_SIMPLEX
                                y0 = qf.shape[0] - 50

                                l1 = (f"Ep {ep+1}/{args.episodes}  |  "
                                      f"Step {step}/{args.max_steps}  "
                                      f"|  ctrl={ctrl_state}")
                                cv2.putText(qf, l1, (10, y0+20),
                                            fnt, 0.5, (0,255,255), 1)

                                cd = vt.get('centering_dist', -1)
                                fr = vt.get('target_fraction', 0)
                                l2 = (f"R={cumul_reward:.0f}  |  "
                                      f"vis={vis_t}  cent={cd:.3f}"
                                      f"  frac={fr:.3f}  |  "
                                      f"alt_d={drone_pos[2]:.2f}  "
                                      f"alt_t={target_pos[2]:.2f}")
                                cv2.putText(qf, l2, (10, y0+42),
                                            fnt, 0.45, (200,200,200), 1)
                                cv2.circle(
                                    qf, (qf.shape[1]-25, y0+15),
                                    10, vis_c, -1)

                                w_quad.write(qf)

                    if terminated:
                        end_reason = "out_of_bounds"
                        break
                    if truncated:
                        end_reason = "truncated"
                        break

                w_quad.release()

                # ── Per-episode altitude plot ──
                alt_path = self.plot_dir / f"episode_{ep+1:03d}_altitude.png"
                self._plot_altitude(ep_times, ep_drone_alts,
                                    ep_target_alts, ep + 1, alt_path)
                all_altitude_data.append(
                    (ep + 1, ep_times, ep_drone_alts, ep_target_alts))

                mean_cent = (float(np.mean(ep_centering))
                             if ep_centering else -1)
                mean_act = float(np.mean(ep_action_mags))
                total_reward = float(np.sum(ep_rewards))

                results.append({
                    'episode': ep + 1,
                    'steps': step,
                    'end_reason': end_reason,
                    'total_reward': round(total_reward, 1),
                    'mean_centering_dist': round(mean_cent, 3),
                    'mean_action_mag': round(mean_act, 3),
                    'visibility_pct': round(
                        100 * sum(1 for r in ep_rewards if r > 0)
                        / max(step, 1), 1),
                })

                flag = " [!ACT]" if mean_act > 0.3 else ""
                print(f"    {end_reason:15s} step={step:4d}  "
                      f"R={total_reward:7.1f}  cent={mean_cent:.3f}  "
                      f"|a|={mean_act:.3f}{flag}")

        # ── Combined altitude plot ──
        combined_path = self.plot_dir / "all_altitudes.png"
        self._plot_all_altitudes(all_altitude_data, combined_path)

        # Summary
        summary = {
            'config': {
                'hover_height': args.hover_height,
                'target_mode': args.target_mode,
                'target_speed': args.target_speed,
                'max_steps': args.max_steps,
                'model': args.model_path,
                'spiral': args.spiral_model if self.spiral_ctrl else None,
            },
            'mean_reward': round(float(np.mean(
                [r['total_reward'] for r in results])), 1),
            'mean_steps': round(float(np.mean(
                [r['steps'] for r in results])), 1),
            'mean_centering_dist': round(float(np.mean(
                [r['mean_centering_dist'] for r in results
                 if r['mean_centering_dist'] >= 0])), 3),
            'mean_action_mag': round(float(np.mean(
                [r['mean_action_mag'] for r in results])), 3),
            'episodes': results,
        }
        with open(self.out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Done! Results in {self.out_dir}/")
        print(f"  Videos:    {self.video_dir}/")
        print(f"  Plots:     {self.plot_dir}/")
        print(f"  Telemetry: {telemetry_path}")
        print(f"  Summary:   {self.out_dir / 'summary.json'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 320 240')
        loadPrcFileData('', 'undecorated true')
    app = HoverTrackTestApp(args)
    app.run_tests()
    app.userExit()
    sys.exit(0)