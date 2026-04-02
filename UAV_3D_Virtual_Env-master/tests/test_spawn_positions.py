#!/usr/bin/env python
"""
Visual test: 10 drone–target spawn positions.

For each of 10 environment resets the target is placed at exactly 2 m from
the drone at the same height.  A perspective aerial camera records ~3 s per
initialisation and a summary matplotlib plot is saved.

Usage:
    python tests/test_spawn_positions.py
    python tests/test_spawn_positions.py --n-inits 10 --hold-seconds 3
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from panda3d.core import Filename, loadPrcFile, loadPrcFileData
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Spawn-position visual test")
    p.add_argument('--n-inits', type=int, default=10,
                   help="Number of initialisations to record")
    p.add_argument('--hold-seconds', type=float, default=3.0,
                   help="Seconds to hold each initialisation on camera")
    p.add_argument('--target-distance', type=float, default=2.0,
                   help="Exact distance from drone to target (metres)")
    p.add_argument('--output-dir', type=str,
                   default='./experiments/spawn_test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-display', action='store_true',
                   help="Reduce the Panda3D window to a small size")
    return p.parse_args()


# ── App ───────────────────────────────────────────────────────────────
class SpawnTestApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        # Override main window: fixed aerial perspective
        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)

        # FPV camera (policy input – not recorded, but env needs it)
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0.3, -0.05)
        self.fpv_camera.cam.setHpr(0, 0, 0)
        self.fpv_camera.buffer.setActive(1)

        # Bird's-eye camera for recording
        self.bird_camera = opencv_camera(self, 'bird_cam', 1)
        self.bird_camera.cam.reparentTo(self.render)
        self.bird_camera.buffer.setActive(1)

        # Environment (fixed target mode, no model needed)
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
            target_radius=0.25,
            filming_mode=True,
        )
        self.env._bird_camera = self.bird_camera

    # ---------------------------------------------------------------- #
    def _place_target_at_distance(self, distance):
        """Move the target to exactly *distance* metres from the drone,
        at the same height, at a random angle."""
        drone_pos = self.env.base_env.state[0:5:2]  # [x, y, z]
        angle = np.random.uniform(0, 2 * np.pi)
        self.env.target_pos = np.array([
            drone_pos[0] + distance * np.cos(angle),
            drone_pos[1] + distance * np.sin(angle),
            drone_pos[2],  # same height
        ])
        self.env._update_target_marker_pos()
        return drone_pos.copy(), self.env.target_pos.copy(), angle

    # ---------------------------------------------------------------- #
    def _point_cameras_at(self, midpoint):
        """Aim main window + bird cam to look at *midpoint* from above."""
        mx, my, mz = float(midpoint[0]), float(midpoint[1]), float(midpoint[2]) + 5
        cam_pos = (mx, my - 10, mz + 12)

        self.cam.setPos(*cam_pos)
        self.cam.lookAt(mx, my, mz)

        self.bird_camera.cam.setPos(*cam_pos)
        self.bird_camera.cam.lookAt(mx, my, mz)

    # ---------------------------------------------------------------- #
    def run_test(self):
        np.random.seed(self.args.seed)
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n_inits = self.args.n_inits
        hold_frames = int(self.args.hold_seconds * 30)  # 30 fps
        target_dist = self.args.target_distance

        panel_w, panel_h = 800, 600
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = str(out_dir / 'spawn_positions.mp4')
        writer = cv2.VideoWriter(video_path, fourcc, 30, (panel_w, panel_h))

        print(f"\n{'='*60}")
        print(f"  Spawn-position test")
        print(f"  Initialisations: {n_inits}")
        print(f"  Target distance: {target_dist:.1f} m")
        print(f"  Hold per init:   {self.args.hold_seconds:.1f} s ({hold_frames} frames)")
        print(f"{'='*60}\n")

        records = []  # (drone_pos, target_pos, angle, dist)

        for init_idx in range(n_inits):
            # Reset environment (randomises drone position + orientation)
            obs, info = self.env.reset()

            # Override target → exactly target_dist metres, same height
            drone_pos, target_pos, angle = self._place_target_at_distance(target_dist)
            actual_dist = np.linalg.norm(target_pos - drone_pos)
            records.append((drone_pos, target_pos, angle, actual_dist))

            # Camera centred on midpoint of drone–target pair
            midpoint = (drone_pos + target_pos) / 2.0
            self._point_cameras_at(midpoint)

            angle_deg = np.degrees(angle)
            print(f"  Init {init_idx+1:2d}/{n_inits}  "
                  f"drone=({drone_pos[0]:+.2f}, {drone_pos[1]:+.2f}, {drone_pos[2]:+.2f})  "
                  f"target=({target_pos[0]:+.2f}, {target_pos[1]:+.2f})  "
                  f"dist={actual_dist:.2f}m  angle={angle_deg:.0f}°")

            # Record hold_frames of static aerial view
            for frame_idx in range(hold_frames):
                # Advance Panda3D renderer (no physics step – static scene)
                self.graphicsEngine.renderFrame()
                self.taskMgr.step()

                ok, bird_rgba = self.bird_camera.get_image()
                if ok and bird_rgba is not None:
                    panel = cv2.resize(bird_rgba[:, :, :3], (panel_w, panel_h))
                else:
                    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

                # ── Overlay ──
                # Title bar
                cv2.rectangle(panel, (0, 0), (panel_w, 50), (0, 0, 0), -1)
                cv2.putText(panel,
                            f"Init {init_idx+1}/{n_inits}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 255), 2)
                cv2.putText(panel,
                            f"t = {frame_idx/30:.1f}s / {self.args.hold_seconds:.1f}s",
                            (panel_w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 200, 200), 1)

                # Info box (bottom-left)
                box_y = panel_h - 130
                cv2.rectangle(panel, (0, box_y), (380, panel_h), (0, 0, 0), -1)

                lines = [
                    f"Drone:   ({drone_pos[0]:+.2f}, {drone_pos[1]:+.2f}, {drone_pos[2]:+.2f})",
                    f"Target:  ({target_pos[0]:+.2f}, {target_pos[1]:+.2f}, {target_pos[2]:+.2f})",
                    f"Distance: {actual_dist:.2f} m",
                    f"Angle:    {angle_deg:.0f} deg",
                ]
                ly = box_y + 25
                for ln in lines:
                    cv2.putText(panel, ln, (10, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (0, 255, 0), 1)
                    ly += 25

                # Legend dots
                cv2.circle(panel, (panel_w - 130, box_y + 25), 8, (0, 255, 0), -1)
                cv2.putText(panel, "Drone", (panel_w - 115, box_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
                cv2.circle(panel, (panel_w - 130, box_y + 55), 8, (255, 0, 255), -1)
                cv2.putText(panel, "Target", (panel_w - 115, box_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 255), 1)

                writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"\nVideo saved: {video_path}")

        # ── Summary plot ──
        self._save_summary_plot(records, out_dir, target_dist)

        print(f"Plot saved:  {out_dir / 'spawn_summary.png'}")
        print(f"\nDone!")

    # ---------------------------------------------------------------- #
    def _save_summary_plot(self, records, out_dir, target_dist):
        """Top-down 2D plot of all drone-target pairs."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib not available — skipping plot)")
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        for i, (drone_pos, target_pos, angle, dist) in enumerate(records):
            dx, dy = drone_pos[0], drone_pos[1]
            tx, ty = target_pos[0], target_pos[1]

            # Line from drone to target
            ax.plot([dx, tx], [dy, ty], 'k-', lw=0.8, alpha=0.4)

            # Drone (green)
            ax.plot(dx, dy, 'o', color='limegreen', ms=10, zorder=5,
                    markeredgecolor='darkgreen', markeredgewidth=1.0)

            # Target (magenta)
            ax.plot(tx, ty, 's', color='magenta', ms=9, zorder=5,
                    markeredgecolor='darkmagenta', markeredgewidth=1.0)

            # Label
            mid_x = (dx + tx) / 2
            mid_y = (dy + ty) / 2
            ax.annotate(f"{i+1}", (mid_x, mid_y),
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='gray', alpha=0.8))

        # Environment bounds
        rect = plt.Rectangle((-5, -5), 10, 10, fill=False,
                              edgecolor='red', linestyle='--', lw=1.2,
                              label='Env bounds (±5 m)')
        ax.add_patch(rect)

        # Formatting
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Spawn Positions — {len(records)} initialisations '
                     f'(target at {target_dist:.1f} m)')

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                   markersize=10, markeredgecolor='darkgreen', label='Drone'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='magenta',
                   markersize=9, markeredgecolor='darkmagenta', label='Target'),
            Line2D([0], [0], color='k', lw=0.8, alpha=0.4,
                   label=f'd = {target_dist:.1f} m'),
            rect,
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(str(out_dir / 'spawn_summary.png'), dpi=150)
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    if args.no_display:
        loadPrcFileData('', 'win-size 320 240')
    app = SpawnTestApp(args)
    app.run_test()
    app.userExit()
    sys.exit(0)
