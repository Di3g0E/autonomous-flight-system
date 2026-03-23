#!/usr/bin/env python
"""
Visual test of the lemniscate (∞) target trajectory.

Only the 3D scene and the orange sphere are created — no drone, no RL env.
Records a video from a perspective aerial camera to verify the figure-8
movement in the horizontal plane.

Usage:
    python tests/test_lemniscate_trajectory.py
    python tests/test_lemniscate_trajectory.py --steps 2000 --speed 0.3 --scale 3.0
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

from panda3d.core import Filename, loadPrcFile, Material, LColor
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from src.simulation.world_setup import world_setup
from src.vision.img_2_cv import opencv_camera


# ── Lemniscate math ────────────────────────────────────────────────────
def lemniscate_point(t, scale):
    """Return (x, y) on a Bernoulli lemniscate of half-width *scale*."""
    denom = 1.0 + np.sin(t) ** 2
    x = scale * np.cos(t) / denom
    y = scale * np.sin(t) * np.cos(t) / denom
    return x, y


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Lemniscate trajectory test")
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--speed', type=float, default=0.25,
                   help="Angular-speed multiplier")
    p.add_argument('--scale', type=float, default=2.5,
                   help="Half-width of the ∞ in metres (full width = 2×scale)")
    p.add_argument('--output-dir', type=str,
                   default='./experiments/lemniscate_test')
    return p.parse_args()


# ── App ────────────────────────────────────────────────────────────────
class LemniscateTestApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        # Load city scene (no drone)
        print("Loading 3D world (no drone)...")
        world_setup(self, self.render, mydir)

        # Create orange target sphere
        self.sphere = self.loader.loadModel("models/misc/sphere")
        self.sphere.reparentTo(self.render)
        self.sphere.setScale(0.5)
        mat = Material()
        mat.setEmission(LColor(1.0, 0.0, 1.0, 1.0))
        mat.setDiffuse(LColor(0.0, 0.0, 0.0, 1.0))
        mat.setAmbient(LColor(0.0, 0.0, 0.0, 1.0))
        self.sphere.setMaterial(mat)
        self.sphere.setColor(1.0, 0.0, 1.0, 1.0)
        self.sphere.setLightOff()

        # Aerial perspective camera (angled so depth is visible)
        self.bird = opencv_camera(self, 'bird_cam', 1)
        self.bird.cam.reparentTo(self.render)
        self.bird.cam.setPos(0, -10, 14)
        self.bird.cam.lookAt(0, 0, 5)
        self.bird.buffer.setActive(1)

    # ----------------------------------------------------------------- #
    def run_test(self):
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        scale = self.args.scale
        speed = self.args.speed
        steps = self.args.steps
        dt = 0.01
        fixed_z = 5.0  # fixed height (matches the city's visual offset)

        panel_w, panel_h = 800, 600
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = str(out_dir / 'lemniscate_bird.mp4')
        writer = cv2.VideoWriter(video_path, fourcc, fps,
                                 (panel_w, panel_h))

        print(f"\nLemniscate  |  scale={scale}m  width={2*scale:.1f}m  "
              f"speed={speed}  steps={steps}")
        print("Recording...\n")

        trail = []
        t_accum = 0.0

        for step in range(steps):
            t_accum += dt
            t = t_accum * speed * 2.0
            x, y = lemniscate_point(t, scale)

            # Move sphere in Panda3D scene
            self.sphere.setPos(float(x), float(y), fixed_z)
            trail.append((x, y))

            # Advance Panda3D renderer
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()

            # Capture frame
            ok, rgba = self.bird.get_image()
            if ok and rgba is not None:
                panel = cv2.resize(rgba[:, :, :3], (panel_w, panel_h))
            else:
                panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

            # Overlay
            cv2.putText(panel, "Aerial Perspective", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            lines = [
                f"Step: {step+1}/{steps}",
                f"Scale: {scale:.1f}m   Speed: {speed}",
                f"Pos: ({x:+.2f}, {y:+.2f})",
            ]
            ly = 60
            for ln in lines:
                cv2.putText(panel, ln, (10, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                ly += 22

            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            if (step + 1) % 500 == 0:
                print(f"  step {step+1}  pos=({x:+.2f}, {y:+.2f})")

        writer.release()
        self._save_plot(np.array(trail), out_dir, scale, speed)

        print(f"\nDone!")
        print(f"  Video:      {video_path}")
        print(f"  Trajectory: {out_dir / 'lemniscate_trajectory.png'}")

    # ----------------------------------------------------------------- #
    def _save_plot(self, trail, out_dir, scale, speed):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(trail[:, 0], trail[:, 1], 'r-', lw=1.5, alpha=0.7,
                    label='Target path')
            ax.plot(trail[0, 0], trail[0, 1], 'go', ms=10, label='Start')
            ax.plot(trail[-1, 0], trail[-1, 1], 'bs', ms=10, label='End')
            lim = scale + 1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Lemniscate  |  scale={scale:.1f}m  speed={speed}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(str(out_dir / 'lemniscate_trajectory.png'), dpi=150)
            plt.close()
        except ImportError:
            print("  (matplotlib not available — skipping plot)")


if __name__ == "__main__":
    args = parse_args()
    app = LemniscateTestApp(args)
    app.run_test()