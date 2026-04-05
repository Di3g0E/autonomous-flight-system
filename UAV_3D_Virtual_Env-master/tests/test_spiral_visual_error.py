#!/usr/bin/env python
"""
Spiral visual-error test: video + plots showing how accurately the drone
follows the Archimedes spiral reference trajectory.

Outputs (in experiments/spiral_visual_error/):

  VIDEO
  ─────
  spiral_follow_3d.mp4
      External 3D camera recording where:
        - Green line   = reference spiral (drawn at the start)
        - Cyan trail   = actual drone path (grows in real time)
        - Red line     = instantaneous error vector (ref → drone)
        - HUD overlay  = time, error, phase stats

  PLOTS  (6 PNG figures)
  ──────
  spiral_error_vectors.png      Reference vs actual + error arrows
  spiral_coloured_by_error.png  Trajectory colour-coded by error magnitude
  error_vs_time.png             Position & altitude error vs time
  error_vs_angle.png            Error vs spiral angle (theta)
  error_histogram.png           Error distribution
  error_summary.png             Dashboard with all metrics

Usage:
    python tests/test_spiral_visual_error.py
    python tests/test_spiral_visual_error.py --max-steps 5000 --fps 30
    python tests/test_spiral_visual_error.py --episodes 3
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

import torch  # noqa: F401 (before Panda3D to avoid DLL conflicts)
from panda3d.core import Filename, loadPrcFile, LineSegs, Vec4

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from direct.showbase.ShowBase import ShowBase
from stable_baselines3 import PPO

from src.simulation.world_setup import world_setup, quad_setup
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.envs.spiral_follow_env import SpiralFollowEnv


# -- Constants -------------------------------------------------------------
Z_VIS_OFFSET = 5          # Panda3D visualisation z-offset
PANEL_SIZE = 640           # default video resolution
FPS = 30                   # default video frame-rate


# -- CLI -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise spiral-follow error (video + plots)")
    p.add_argument('--model-path', type=str,
                   default='./models/spiral_follow/best_model.zip')
    p.add_argument('--output-dir', type=str,
                   default='./experiments/spiral_visual_error')
    p.add_argument('--max-steps', type=int, default=3000,
                   help="Steps per episode (3000 = 30 s)")
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--fps', type=int, default=FPS)
    p.add_argument('--panel-size', type=int, default=PANEL_SIZE)
    p.add_argument('--omega', type=float, default=1.8)
    p.add_argument('--r-growth', type=float, default=0.12)
    p.add_argument('--hover-height', type=float, default=1.39)
    p.add_argument('--vision-radius', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# -- 3D helpers (LineSegs) -------------------------------------------------

def draw_reference_spiral(render_node, cx, cy, hover_height,
                          omega, r_growth, max_tilt, dt, n_points):
    """Pre-compute and draw the full reference spiral as a green line."""
    ls = LineSegs()
    ls.setColor(Vec4(1.0, 1.0, 0.0, 0.95))   # yellow — high contrast vs blue sky
    ls.setThickness(3.0)

    theta = 0.0
    z = hover_height + Z_VIS_OFFSET
    for i in range(n_points):
        t = i * dt
        r = r_growth * t + 0.05
        a_budget = 0.70 * 9.82 * math.sin(max_tilt)
        w = min(omega, math.sqrt(a_budget / max(r, 0.05)))
        theta += w * dt
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        if i == 0:
            ls.moveTo(x, y, z)
        else:
            ls.drawTo(x, y, z)

    node = ls.create()
    return render_node.attachNewNode(node)


def build_trail_node(render_node, positions, colour, thickness=2.5):
    """Create a LineSegs node from a list of (x, y, z) positions."""
    if len(positions) < 2:
        return None
    ls = LineSegs()
    ls.setColor(colour)
    ls.setThickness(thickness)
    ls.moveTo(*positions[0])
    for pos in positions[1:]:
        ls.drawTo(*pos)
    node = ls.create()
    return render_node.attachNewNode(node)


def build_error_line(render_node, p_ref, p_drone):
    """White line from reference point to drone (error vector)."""
    ls = LineSegs()
    ls.setColor(Vec4(1.0, 1.0, 1.0, 0.95))   # white — neutral, distinct
    ls.setThickness(2.5)
    ls.moveTo(*p_ref)
    ls.drawTo(*p_drone)
    node = ls.create()
    return render_node.attachNewNode(node)


# -- HUD overlay (OpenCV on captured frame) --------------------------------

def draw_hud(frame, step, total_steps, pos_err, alt_err, vision_r,
             mean_err, theta_deg):
    """Draw a translucent info panel on the video frame."""
    h, w = frame.shape[:2]
    time_s = step * 0.01

    # Semi-transparent black bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    err_colour = (0, 255, 0) if pos_err <= vision_r else (0, 0, 255)

    cv2.putText(frame, f"t = {time_s:.1f} s", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"error = {pos_err:.4f} m", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, err_colour, 2)
    cv2.putText(frame, f"alt err = {alt_err:.3f} m", (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(frame, f"media = {mean_err:.4f} m", (w - 230, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"theta = {theta_deg:.0f} deg", (w - 230, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Progress bar at the bottom
    bar_y = h - 6
    progress = step / max(total_steps, 1)
    cv2.rectangle(frame, (0, bar_y), (int(w * progress), h), (0, 200, 255), -1)

    # Legend (bottom-left) — BGR colours matching the 3D lines
    ly = h - 28
    cv2.line(frame, (12, ly), (32, ly), (0, 255, 255), 2)       # yellow
    cv2.putText(frame, "Referencia", (38, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.line(frame, (140, ly), (160, ly), (0, 115, 255), 2)     # orange
    cv2.putText(frame, "Dron (real)", (166, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 115, 255), 1)
    cv2.line(frame, (280, ly), (300, ly), (255, 255, 255), 2)   # white
    cv2.putText(frame, "Error", (306, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


# -- Matplotlib plots (same as before) ------------------------------------

def generate_plots(episodes_data, out_dir, args):
    """Generate all error-visualisation figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.gridspec as gridspec

    out_dir = Path(out_dir)
    vr = args.vision_radius

    all_pos_err = np.concatenate([ep['pos_error'] for ep in episodes_data])
    all_alt_err = np.concatenate([ep['alt_error'] for ep in episodes_data])

    # ── 1. Error vectors ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    for ep in episodes_data:
        ax.plot(ep['ref_x'], ep['ref_y'], 'g-', lw=1.2, alpha=0.5,
                label='Referencia' if ep is episodes_data[0] else None)
        ax.plot(ep['drone_x'], ep['drone_y'], 'b-', lw=1.0, alpha=0.7,
                label='Trayectoria real' if ep is episodes_data[0] else None)
        N = max(1, len(ep['ref_x']) // 60)
        for i in range(0, len(ep['ref_x']), N):
            dx = ep['drone_x'][i] - ep['ref_x'][i]
            dy = ep['drone_y'][i] - ep['ref_y'][i]
            err = math.sqrt(dx**2 + dy**2)
            colour = 'orange' if err < vr else 'red'
            ax.annotate('', xy=(ep['drone_x'][i], ep['drone_y'][i]),
                        xytext=(ep['ref_x'][i], ep['ref_y'][i]),
                        arrowprops=dict(arrowstyle='->', color=colour,
                                        lw=0.8, alpha=0.6))
    ax.set_xlabel('X (m)');  ax.set_ylabel('Y (m)')
    ax.set_title('Espiral: referencia vs real (flechas = error)')
    ax.set_aspect('equal'); ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / 'spiral_error_vectors.png', dpi=150); plt.close(fig)
    print(f"  [1/6] spiral_error_vectors.png")

    # ── 2. Trajectory coloured by error ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    for ep in episodes_data:
        ax.plot(ep['ref_x'], ep['ref_y'], color='#cccccc', lw=1.5,
                alpha=0.6, zorder=1)
        points = np.column_stack([ep['drone_x'], ep['drone_y']])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        err_mid = (ep['pos_error'][:-1] + ep['pos_error'][1:]) / 2
        norm = Normalize(vmin=0, vmax=max(vr, np.percentile(err_mid, 95)))
        lc = LineCollection(segments, cmap='RdYlGn_r', norm=norm,
                            linewidths=2.0, zorder=2)
        lc.set_array(err_mid)
        ax.add_collection(lc)
        ax.plot(ep['drone_x'][0], ep['drone_y'][0], 'go', ms=8, zorder=3,
                label='Inicio' if ep is episodes_data[0] else None)
        ax.plot(ep['drone_x'][-1], ep['drone_y'][-1], 'rs', ms=8, zorder=3,
                label='Fin' if ep is episodes_data[0] else None)
    cbar = fig.colorbar(lc, ax=ax, shrink=0.7)
    cbar.set_label('Error de posicion (m)')
    ax.set_xlabel('X (m)');  ax.set_ylabel('Y (m)')
    ax.set_title('Trayectoria coloreada por error de seguimiento')
    ax.set_aspect('equal'); ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3); ax.autoscale_view(); fig.tight_layout()
    fig.savefig(out_dir / 'spiral_coloured_by_error.png', dpi=150)
    plt.close(fig); print(f"  [2/6] spiral_coloured_by_error.png")

    # ── 3. Error vs time ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ep in episodes_data:
        axes[0].plot(ep['time'], ep['pos_error'], lw=0.8, alpha=0.8)
        axes[1].plot(ep['time'], ep['alt_error'], lw=0.8, alpha=0.8)
    axes[0].axhline(vr, color='r', ls='--', lw=1, alpha=0.6,
                     label=f'Radio de vision ({vr} m)')
    axes[0].axhline(np.mean(all_pos_err), color='k', ls=':', lw=1, alpha=0.6,
                     label=f'Media = {np.mean(all_pos_err):.3f} m')
    axes[0].set_ylabel('Error posicion (m)')
    axes[0].set_title('Error de seguimiento a lo largo del tiempo')
    axes[0].legend(loc='upper right'); axes[0].grid(True, alpha=0.3)
    axes[1].axhline(np.mean(all_alt_err), color='k', ls=':', lw=1, alpha=0.6,
                     label=f'Media = {np.mean(all_alt_err):.3f} m')
    axes[1].set_xlabel('Tiempo (s)'); axes[1].set_ylabel('Error altitud (m)')
    axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'error_vs_time.png', dpi=150); plt.close(fig)
    print(f"  [3/6] error_vs_time.png")

    # ── 4. Error vs spiral angle ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for ep in episodes_data:
        theta_deg = np.degrees(ep['theta'])
        ax.plot(theta_deg, ep['pos_error'], lw=0.8, alpha=0.8)
        for ft in np.arange(360, theta_deg[-1], 360):
            ax.axvline(ft, color='grey', ls=':', lw=0.5, alpha=0.3)
    ax.axhline(vr, color='r', ls='--', lw=1, alpha=0.6,
               label=f'Radio de vision ({vr} m)')
    ax.set_xlabel('Angulo de la espiral (grados)')
    ax.set_ylabel('Error posicion (m)')
    ax.set_title('Error vs angulo de la espiral')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / 'error_vs_angle.png', dpi=150); plt.close(fig)
    print(f"  [4/6] error_vs_angle.png")

    # ── 5. Error histogram ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_pos_err, bins=60, color='steelblue', edgecolor='white',
            alpha=0.8, density=True)
    ax.axvline(np.mean(all_pos_err), color='k', ls='-', lw=1.5,
               label=f'Media = {np.mean(all_pos_err):.4f} m')
    ax.axvline(np.median(all_pos_err), color='orange', ls='--', lw=1.5,
               label=f'Mediana = {np.median(all_pos_err):.4f} m')
    ax.axvline(vr, color='r', ls='--', lw=1.5,
               label=f'Radio de vision = {vr} m')
    ax.set_xlabel('Error de posicion (m)'); ax.set_ylabel('Densidad')
    ax.set_title('Distribucion del error de posicion')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / 'error_histogram.png', dpi=150); plt.close(fig)
    print(f"  [5/6] error_histogram.png")

    # ── 6. Dashboard summary ─────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    for ep in episodes_data:
        ax1.plot(ep['ref_x'], ep['ref_y'], 'g-', lw=1, alpha=0.4)
        ax1.plot(ep['drone_x'], ep['drone_y'], 'b-', lw=0.8, alpha=0.7)
    ax1.set_title('Trayectoria XY', fontsize=10)
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)', fontsize=8); ax1.set_ylabel('Y (m)', fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    for ep in episodes_data:
        ax2.plot(ep['ref_x'], ep['ref_y'], color='#ddd', lw=1.2, zorder=1)
        pts = np.column_stack([ep['drone_x'], ep['drone_y']])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        err_m = (ep['pos_error'][:-1] + ep['pos_error'][1:]) / 2
        norm2 = Normalize(vmin=0, vmax=max(vr, np.percentile(err_m, 95)))
        lc2 = LineCollection(segs, cmap='RdYlGn_r', norm=norm2, lw=1.5,
                             zorder=2)
        lc2.set_array(err_m)
        ax2.add_collection(lc2)
    ax2.set_title('Color = error', fontsize=10)
    ax2.set_aspect('equal'); ax2.autoscale_view(); ax2.grid(True, alpha=0.3)
    fig.colorbar(lc2, ax=ax2, shrink=0.7, label='Error (m)')

    ax3 = fig.add_subplot(gs[0, 2]); ax3.axis('off')
    within_vr = 100.0 * np.sum(all_pos_err <= vr) / len(all_pos_err)
    stats_text = (
        f"Episodios:        {len(episodes_data)}\n"
        f"Pasos/episodio:   {args.max_steps}\n"
        f"Duracion:         {args.max_steps * 0.01:.0f} s\n\n"
        f"ERROR POSICION\n"
        f"  Media:          {np.mean(all_pos_err):.4f} m\n"
        f"  Mediana:        {np.median(all_pos_err):.4f} m\n"
        f"  Std:            {np.std(all_pos_err):.4f} m\n"
        f"  Max:            {np.max(all_pos_err):.4f} m\n"
        f"  P95:            {np.percentile(all_pos_err, 95):.4f} m\n"
        f"  Dentro vision:  {within_vr:.1f}%\n\n"
        f"ERROR ALTITUD\n"
        f"  Media:          {np.mean(all_alt_err):.4f} m\n"
        f"  Std:            {np.std(all_alt_err):.4f} m\n"
        f"  Max:            {np.max(all_alt_err):.4f} m\n\n"
        f"PARAMETROS\n"
        f"  omega:          {args.omega} rad/s\n"
        f"  r_growth:       {args.r_growth} m/s\n"
        f"  hover_height:   {args.hover_height} m\n"
        f"  vision_radius:  {vr} m"
    )
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('Estadisticas', fontsize=10)

    ax4 = fig.add_subplot(gs[1, 0])
    for ep in episodes_data:
        ax4.plot(ep['time'], ep['pos_error'], lw=0.7, alpha=0.8)
    ax4.axhline(vr, color='r', ls='--', lw=1, alpha=0.5)
    ax4.set_xlabel('Tiempo (s)', fontsize=8)
    ax4.set_ylabel('Error posicion (m)', fontsize=8)
    ax4.set_title('Error vs tiempo', fontsize=10); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    for ep in episodes_data:
        ax5.plot(np.degrees(ep['theta']), ep['pos_error'], lw=0.7, alpha=0.8)
    ax5.axhline(vr, color='r', ls='--', lw=1, alpha=0.5)
    ax5.set_xlabel('Angulo (deg)', fontsize=8)
    ax5.set_ylabel('Error (m)', fontsize=8)
    ax5.set_title('Error vs angulo espiral', fontsize=10)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(all_pos_err, bins=50, color='steelblue', edgecolor='white',
             alpha=0.8, density=True)
    ax6.axvline(np.mean(all_pos_err), color='k', ls='-', lw=1.2)
    ax6.axvline(vr, color='r', ls='--', lw=1.2)
    ax6.set_xlabel('Error (m)', fontsize=8)
    ax6.set_ylabel('Densidad', fontsize=8)
    ax6.set_title('Histograma del error', fontsize=10); ax6.grid(True, alpha=0.3)

    fig.suptitle('Analisis de error de seguimiento de espiral', fontsize=13,
                 fontweight='bold')
    fig.savefig(out_dir / 'error_summary.png', dpi=150); plt.close(fig)
    print(f"  [6/6] error_summary.png")


# -- Panda3D Application --------------------------------------------------

class SpiralVisualErrorApp(ShowBase):
    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        self.mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone...")
        world_setup(self, self.render, self.mydir)
        quad_setup(self, self.render, self.mydir)

        # External camera for video recording
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.buffer.setActive(1)

        self.base_env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=False,
            use_depth=False,
            use_target=False,
            enable_collisions=False,
            n=args.max_steps + 50,
            t_step=0.01,
            direct_control=1,
            filming_mode=True,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.3,
            init_vel_range=0.15,
            init_ang_range=0.05,
        )

        self.spiral_env = SpiralFollowEnv(
            self.base_env,
            omega=args.omega,
            r_growth=args.r_growth,
            hover_height=args.hover_height,
            vision_radius=args.vision_radius,
        )
        self.spiral_env.omega_scale = 1.0

        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found at {args.model_path}")
            sys.exit(1)
        self.model = PPO.load(args.model_path, env=None)
        print(f"Model loaded: {args.model_path}")

        self.out_dir = Path(args.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Scene-graph nodes for dynamic lines (reused each frame)
        self._trail_node = None
        self._error_node = None

    # -- Dynamic 3D line helpers -------------------------------------------

    def _update_trail(self, trail_positions):
        """Rebuild the cyan drone trail in the 3D scene."""
        if self._trail_node is not None:
            self._trail_node.removeNode()
            self._trail_node = None
        if len(trail_positions) >= 2:
            self._trail_node = build_trail_node(
                self.render, trail_positions,
                Vec4(1.0, 0.45, 0.0, 0.95),   # orange — warm, distinct from yellow
                thickness=3.0)

    def _update_error_line(self, ref_xyz, drone_xyz):
        """Rebuild the red error vector line in the 3D scene."""
        if self._error_node is not None:
            self._error_node.removeNode()
            self._error_node = None
        self._error_node = build_error_line(self.render, ref_xyz, drone_xyz)

    # -- Camera positioning ------------------------------------------------

    def _position_camera(self, center_x, center_y, hover_height, max_radius):
        """Place external camera high enough to see the whole spiral."""
        view_r = max(max_radius, 1.0) + 1.5
        cam_dist = view_r * 2.8
        z_look = hover_height + Z_VIS_OFFSET
        self.ext_camera.cam.setPos(
            center_x - cam_dist * 0.45,
            center_y - cam_dist * 0.45,
            z_look + cam_dist * 0.65,
        )
        self.ext_camera.cam.lookAt(
            float(center_x), float(center_y), float(z_look - 0.5))

    # -- Main evaluation ---------------------------------------------------

    def run_evaluation(self):
        args = self.args
        np.random.seed(args.seed)
        PS = args.panel_size
        fps = args.fps

        arm_spacing = args.r_growth * 2 * np.pi / args.omega
        print(f"\n{'='*60}")
        print(f"  SPIRAL VISUAL ERROR ANALYSIS")
        print(f"  omega={args.omega}  r_growth={args.r_growth}  "
              f"arm_spacing={arm_spacing:.3f} m")
        print(f"  Episodes: {args.episodes}  Max steps: {args.max_steps}")
        print(f"  Video: {PS}x{PS} @ {fps} fps")
        print(f"{'='*60}\n")

        episodes_data = []

        for ep in range(args.episodes):
            print(f"  Episode {ep+1}/{args.episodes}...")
            obs, info = self.spiral_env.reset()

            center_x = self.spiral_env._center_x
            center_y = self.spiral_env._center_y

            # Draw the full reference spiral (green) in the 3D scene
            max_r = args.r_growth * args.max_steps * 0.01 + 0.05
            ref_spiral_node = draw_reference_spiral(
                self.render, center_x, center_y,
                args.hover_height, args.omega, args.r_growth,
                0.25, 0.01, args.max_steps)

            # Position camera to frame the whole spiral
            self._position_camera(center_x, center_y,
                                  args.hover_height, max_r)

            # Video writer
            video_path = self.out_dir / f'spiral_follow_3d.mp4'
            if args.episodes > 1:
                video_path = self.out_dir / f'spiral_follow_3d_ep{ep+1}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (PS, PS))

            # Warm up Panda3D rendering buffers
            neutral = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            for _ in range(10):
                obs, _, _, _, info = self.spiral_env.step(neutral)
                self.graphicsEngine.renderFrame()
                self.taskMgr.step()
            # Reset after warm-up
            obs, info = self.spiral_env.reset()
            # Re-centre spiral (reset changes centre)
            center_x = self.spiral_env._center_x
            center_y = self.spiral_env._center_y

            # Data collection lists
            drone_x_list, drone_y_list, drone_z_list = [], [], []
            ref_x_list, ref_y_list = [], []
            pos_err_list, alt_err_list = [], []
            theta_list, time_list = [], []
            trail_positions = []  # (x, y, z) in Panda3D coords

            frame_interval = max(1, 100 // fps)
            total_steps = args.max_steps
            frames_written = 0
            cum_pos_err = 0.0

            for step in range(total_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = \
                    self.spiral_env.step(action)
                self.taskMgr.step()

                sp = info.get('spiral', {})
                state = self.base_env.base_env.state
                dx = float(state[0])
                dy = float(state[2])
                dz = float(state[4])
                rx = sp.get('x_ref', 0.0)
                ry = sp.get('y_ref', 0.0)
                pe = sp.get('pos_error', 0.0)
                ae = sp.get('alt_error', 0.0)
                th = float(self.spiral_env._theta_accum)

                drone_x_list.append(dx)
                drone_y_list.append(dy)
                drone_z_list.append(dz)
                ref_x_list.append(rx)
                ref_y_list.append(ry)
                pos_err_list.append(pe)
                alt_err_list.append(ae)
                theta_list.append(th)
                time_list.append((step + 1) * 0.01)
                cum_pos_err += pe

                # 3D positions (with Panda3D z-offset)
                z_vis = args.hover_height + Z_VIS_OFFSET
                trail_positions.append((dx, dy, z_vis))

                # Write video frame at the right interval
                if step % frame_interval == 0:
                    # Update 3D trail and error line
                    self._update_trail(trail_positions)
                    ref_3d = (rx, ry, z_vis)
                    drone_3d = (dx, dy, z_vis)
                    self._update_error_line(ref_3d, drone_3d)

                    # Render and capture
                    self.graphicsEngine.renderFrame()
                    success, ext_rgba = self.ext_camera.get_image()
                    if success:
                        frame = cv2.cvtColor(ext_rgba, cv2.COLOR_RGBA2BGR)
                        frame = cv2.resize(frame, (PS, PS))
                        mean_err = cum_pos_err / (step + 1)
                        draw_hud(frame, step, total_steps, pe, ae,
                                 args.vision_radius, mean_err,
                                 math.degrees(th))
                        writer.write(frame)
                        frames_written += 1

                if terminated or truncated:
                    reason = "terminated" if terminated else "truncated"
                    print(f"    Episode ended ({reason}) at step {step+1}")
                    break

            writer.release()

            # Clean up 3D nodes
            ref_spiral_node.removeNode()
            if self._trail_node is not None:
                self._trail_node.removeNode()
                self._trail_node = None
            if self._error_node is not None:
                self._error_node.removeNode()
                self._error_node = None

            episodes_data.append({
                'drone_x': np.array(drone_x_list),
                'drone_y': np.array(drone_y_list),
                'drone_z': np.array(drone_z_list),
                'ref_x': np.array(ref_x_list),
                'ref_y': np.array(ref_y_list),
                'pos_error': np.array(pos_err_list),
                'alt_error': np.array(alt_err_list),
                'theta': np.array(theta_list),
                'time': np.array(time_list),
            })

            mean_err = np.mean(pos_err_list)
            max_err = np.max(pos_err_list)
            within = 100.0 * np.sum(
                np.array(pos_err_list) <= args.vision_radius
            ) / len(pos_err_list)
            print(f"    pos_err: mean={mean_err:.4f} m  "
                  f"max={max_err:.4f} m  within_vision={within:.1f}%")
            print(f"    Video: {video_path}  ({frames_written} frames)")

        # Generate matplotlib plots
        print(f"\n  Generating plots...")
        generate_plots(episodes_data, self.out_dir, args)

        # Summary
        all_err = np.concatenate([ep['pos_error'] for ep in episodes_data])
        all_alt = np.concatenate([ep['alt_error'] for ep in episodes_data])
        within_total = 100.0 * np.sum(
            all_err <= args.vision_radius) / len(all_err)

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"  Position error:  {np.mean(all_err):.4f} +/- "
              f"{np.std(all_err):.4f} m  (max {np.max(all_err):.4f})")
        print(f"  Altitude error:  {np.mean(all_alt):.4f} +/- "
              f"{np.std(all_alt):.4f} m")
        print(f"  Within vision:   {within_total:.1f}%")
        print(f"  P95 error:       {np.percentile(all_err, 95):.4f} m")
        print(f"\n  Output: {self.out_dir}/")
        print(f"    spiral_follow_3d.mp4      - 3D video with trails")
        print(f"    spiral_error_vectors.png   - Error arrows plot")
        print(f"    spiral_coloured_by_error.png")
        print(f"    error_vs_time.png")
        print(f"    error_vs_angle.png")
        print(f"    error_histogram.png")
        print(f"    error_summary.png          - Dashboard")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    app = SpiralVisualErrorApp(args)
    app.run_evaluation()
    app.userExit()
    sys.exit(0)
