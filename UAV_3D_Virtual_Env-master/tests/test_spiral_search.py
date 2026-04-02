#!/usr/bin/env python
"""
Spiral search controller: empirical test and parameter tuning.

Tests the SpiralSearchController — a deterministic Archimedes spiral
that activates when the target is lost for K consecutive steps.  The
controller uses PD-stabilised motor commands.

Design constraint for tight spiral
-----------------------------------
For 360 deg coverage before expanding, the radial expansion per turn
must be smaller than the camera FOV radius:

    Delta_r / turn  <=  FOV_radius  (~0.56 m at h=1.39 m)

With yaw rate w ~ 2.5 rad/s  (1 turn per ~2.5 s):
    Delta_r = g * pitch_avg * T_turn^2 / 2
    pitch_max ~ 0.012 rad  gives  Delta_r ~ 0.25 m / turn  (OK)
    pitch_max ~ 0.080 rad  gives  Delta_r ~ 1.60 m / turn  (too wide!)

Output -> experiments/spiral_search/

Usage
-----
    python tests/test_spiral_search.py
    python tests/test_spiral_search.py --hover-height 1.39 --max-steps 1500
"""

import argparse
import math
import os
import sys
import time
import traceback
import numpy as np
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from panda3d.core import Filename, loadPrcFile, loadPrcFileData


# -- CLI -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Spiral search controller test")
    p.add_argument('--hover-height', type=float, default=1.39,
                   help="Hover altitude (m). Default from calibration test.")
    p.add_argument('--target-radius', type=float, default=0.25)
    p.add_argument('--max-steps', type=int, default=2500,
                   help="Max steps per trial (default: 2500 = 25 s)")
    p.add_argument('--K', type=int, default=20,
                   help="Steps without detection before spiral activates")
    p.add_argument('--damping-steps', type=int, default=15,
                   help="Handoff blending duration (steps)")
    p.add_argument('--output-dir', type=str,
                   default='./experiments/spiral_search')
    p.add_argument('--no-display', action='store_true',
                   help="Shrink Panda3D window for speed")
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


args = parse_args()

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

if args.no_display:
    loadPrcFileData('', 'win-size 64 64')

from direct.showbase.ShowBase import ShowBase
from src.simulation.world_setup import world_setup, quad_setup
from src.simulation.camera_control import camera_control
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv


# ======================================================================
#  Spiral Search Controller
# ======================================================================

class SpiralSearchController:
    """Archimedes spiral via trajectory tracking with feedforward.

    Motor layout (+ config, direct_control=1):
        0=left  1=front  2=right  3=back

    Computes the desired position on a parametric Archimedes spiral
    as a function of time:
        r(t) = r_growth * t + r0
        theta(t) = omega * t
        x_des = r * cos(theta),  y_des = r * sin(theta)

    Uses a PD position controller + acceleration feedforward
    (centripetal + Coriolis) so that the steady-state tracking
    error is zero.  Previous approach (velocity P + centripetal FF)
    failed because at steady orbit v_err -> 0, leaving no centripetal
    force.  Trajectory tracking solves this structurally.

    Parameters
    ----------
    omega_orbit : float   Orbital angular rate (rad/s).
    r_growth    : float   Radial expansion speed (m/s).
    Kv          : float   Velocity damping gain (PD velocity part).
    yaw_delta   : float   Open-loop yaw for camera scanning.
    max_tilt    : float   Safety clamp on roll/pitch (rad).
    """

    # PD gains
    Kp_z   = 0.50
    Kd_z   = 0.30
    Kp_att = 1.00
    Kd_att = 0.15
    Kp_xy  = 1.50     # position tracking gain

    def __init__(self, hover_height, K=20, yaw_delta=0.02,
                 omega_orbit=1.5, r_growth=0.15, Kv=1.50,
                 max_tilt=0.25,
                 max_steps=2500, damping_steps=15):
        self.hover_height = hover_height
        self.K = K
        self.yaw_delta = yaw_delta
        self.omega_orbit = omega_orbit
        self.r_growth = r_growth
        self.Kv = Kv
        self.max_tilt = max_tilt
        self.max_steps = max_steps
        self.damping_steps = damping_steps
        self.reset()

    def reset(self):
        self.blind_count = 0
        self.active = False
        self.spiral_step = 0
        self.handoff_active = False
        self.handoff_step = 0
        self._last_spiral_action = np.zeros(4, dtype=np.float32)
        self.total_spiral_steps = 0
        self._theta_accum = 0.0

    def get_action(self, state, ang, target_visible, rl_action=None):
        hover = np.zeros(4, dtype=np.float32)

        if target_visible:
            if self.active:
                self.active = False
                self.handoff_active = True
                self.handoff_step = 0
            self.blind_count = 0
        else:
            self.blind_count += 1
            if (self.blind_count >= self.K
                    and not self.active
                    and not self.handoff_active):
                self.active = True
                self.spiral_step = 0

        if self.handoff_active:
            alpha = min(1.0, self.handoff_step / max(self.damping_steps, 1))
            target_act = rl_action if rl_action is not None else hover
            blended = (1 - alpha) * self._last_spiral_action + alpha * target_act
            self.handoff_step += 1
            if alpha >= 1.0:
                self.handoff_active = False
            return np.clip(blended, -1, 1).astype(np.float32), 'handoff'

        if self.active:
            action = self._spiral_action(state, ang)
            self._last_spiral_action = action.copy()
            self.spiral_step += 1
            self.total_spiral_steps += 1
            if self.spiral_step >= self.max_steps:
                self.active = False
            return action, 'spiral'

        return (rl_action if rl_action is not None else hover), 'rl'

    def _spiral_action(self, state, ang):
        roll_ang, pitch_ang, yaw = ang[0], ang[1], ang[2]
        x, y = state[0], state[2]
        vx, vy = state[1], state[3]
        z, vz = state[4], state[5]
        wx, wy = state[10], state[11]

        t = self.spiral_step * 0.01   # elapsed time

        # ── Altitude hold ──
        alt = self.Kp_z * (self.hover_height - z) - self.Kd_z * vz

        # ── Desired spiral trajectory (Archimedes spiral) ──
        # Adaptive omega: cap so centripetal accel stays within
        # 70% of max tilt budget (reserve 30% for PD corrections).
        # a_c = w² * r <= 0.7 * g * sin(max_tilt)
        # => w <= sqrt(0.7 * g * sin(max_tilt) / r)
        dr = self.r_growth
        r_des = dr * t + 0.05          # radius grows linearly

        a_budget = 0.70 * 9.82 * math.sin(self.max_tilt)  # ~1.70 m/s²
        w_max = math.sqrt(a_budget / max(r_des, 0.05))
        w = min(self.omega_orbit, w_max)

        # Accumulate angle with variable omega (not w*t)
        if not hasattr(self, '_theta_accum'):
            self._theta_accum = 0.0
        self._theta_accum += w * 0.01  # dt = 0.01s
        theta_des = self._theta_accum
        cos_d = math.cos(theta_des)
        sin_d = math.sin(theta_des)

        # Desired position
        x_des = r_des * cos_d
        y_des = r_des * sin_d

        # Desired velocity (first derivative of position)
        # dr/dt = dr, dθ/dt = w (variable)
        vx_des = dr * cos_d - r_des * w * sin_d
        vy_des = dr * sin_d + r_des * w * cos_d

        # Desired acceleration (second derivative = feedforward)
        # With variable w (but dw/dt ≈ 0 over one step):
        # d²x/dt² = -2·dr·w·sin(θ) - r·w²·cos(θ)
        # d²y/dt² =  2·dr·w·cos(θ) - r·w²·sin(θ)
        ax_ff = -2 * dr * w * sin_d - r_des * w * w * cos_d
        ay_ff =  2 * dr * w * cos_d - r_des * w * w * sin_d

        # ── Trajectory tracking: feedforward + PD ──
        ax_des = ax_ff + self.Kp_xy * (x_des - x) + self.Kv * (vx_des - vx)
        ay_des = ay_ff + self.Kp_xy * (y_des - y) + self.Kv * (vy_des - vy)

        # ── Convert inertial accel to body-frame tilt ──
        cos_h = math.cos(yaw)
        sin_h = math.sin(yaw)
        a_body_right =  cos_h * ax_des + sin_h * ay_des
        a_body_fwd   = -sin_h * ax_des + cos_h * ay_des

        desired_pitch =  a_body_right / 9.82   # pitch = cos(ψ)·ax + sin(ψ)·ay
        desired_roll  = -a_body_fwd / 9.82   # roll  = sin(ψ)·ax - cos(ψ)·ay

        desired_roll  = np.clip(desired_roll,  -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)

        # ── PD stabilisation toward desired tilt ──
        roll_corr  =  self.Kp_att * (roll_ang - desired_roll)  + self.Kd_att * wx
        pitch_corr = -self.Kp_att * (pitch_ang - desired_pitch) - self.Kd_att * wy

        # ── Yaw ──
        yaw_cmd = self.yaw_delta

        # ── Motor allocation (+ config) ──
        action = np.array([
            alt + roll_corr - yaw_cmd,
            alt + pitch_corr + yaw_cmd,
            alt - roll_corr - yaw_cmd,
            alt - pitch_corr + yaw_cmd,
        ], dtype=np.float32)

        return np.clip(action, -1, 1)


# ======================================================================
#  Test Application
# ======================================================================

class SpiralTestApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        print("Loading 3D world and drone ...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)
        camera_control(self, self.render)

        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)

        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.05)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='fixed',
            target_range=5.0,
            target_speed=0.0,
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            enable_collisions=False,
            n=4000,
            t_step=0.01,
            direct_control=1,
            target_radius=args.target_radius,
            filming_mode=True,
            use_new_reward=True,
            search_timeout_steps=4000,
        )

    # -- Reset -------------------------------------------------------------

    def _reset_state(self, drone_z, sphere_x, sphere_y):
        """Full env.reset() with det_state, then override target position."""
        init_state = np.zeros(13, dtype=np.float64)
        init_state[4] = drone_z
        init_state[6] = 1.0

        self.env.reset(options={'det_state': init_state})

        self.env.target_pos = np.array([sphere_x, sphere_y, 0.0])
        self.env._update_target_marker_pos()
        self.env._target_ever_seen = False
        self.env._target_visible_last_step = False

        for _ in range(5):
            self.graphicsEngine.renderFrame()
            self.taskMgr.step()
        self.env._capture_camera_images(force_capture=True)

    # -- Single run --------------------------------------------------------

    def _run_spiral(self, controller, sphere_x, sphere_y,
                    max_steps, record_trajectory=False):
        hover_z = controller.hover_height
        self._reset_state(hover_z, sphere_x, sphere_y)
        controller.reset()

        traj = [] if record_trajectory else None
        detected = False
        steps_to_detect = None
        terminated = False
        oob = False
        target_visible = False

        for step in range(max_steps):
            state = self.env.base_env.state
            ang = self.env.base_env.ang

            action, mode = controller.get_action(
                state, ang, target_visible)

            if record_trajectory:
                traj.append((float(state[0]), float(state[2]),
                             float(state[4]), float(ang[2]),
                             mode, target_visible))

            try:
                obs, reward, term, trunc, info = self.env.step(action)
            except Exception as exc:
                print(f"      [!] Exception at step {step}: {exc}")
                traceback.print_exc()
                terminated = True
                oob = True
                break

            vis_info = info.get('visual_tracking', {})
            target_visible = vis_info.get('target_visible', False)

            if target_visible and steps_to_detect is None:
                steps_to_detect = step + 1
                detected = True

            if term:
                terminated = True
                oob = True
                break

        final_state = self.env.base_env.state
        final_xy = np.array([float(final_state[0]), float(final_state[2])])
        displacement = float(np.linalg.norm(final_xy))

        return {
            'detected': detected,
            'steps_to_detect': steps_to_detect,
            'terminated': terminated,
            'oob': oob,
            'final_pos': (float(final_state[0]),
                          float(final_state[2]),
                          float(final_state[4])),
            'final_yaw_rate': float(final_state[12]),
            'displacement': displacement,
            'trajectory': traj,
        }

    # ==================================================================
    #  Test phases
    # ==================================================================

    def run_test(self):
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(args.seed)

        hover_h = args.hover_height
        max_steps = args.max_steps

        fov_half_w = hover_h * math.tan(math.atan(18 / 45))

        print(f"\n{'='*65}")
        print(f"  SPIRAL SEARCH TEST")
        print(f"{'='*65}")
        print(f"  Hover height:    {hover_h:.2f} m")
        print(f"  FOV half-width:  {fov_half_w:.3f} m")
        print(f"  Sphere radius:   {args.target_radius} m")
        print(f"  K (blind gate):  {args.K} steps ({args.K*0.01:.2f} s)")
        print(f"  Max steps:       {max_steps} ({max_steps*0.01:.1f} s)")
        print(f"  Damping steps:   {args.damping_steps}")

        # -- Phase 1: Parameter sweep --------------------------------------
        # Test at TWO opposite angles (0 deg and 180 deg) to validate
        # 360 deg coverage.  A combo passes only if BOTH are detected.
        print(f"\n{'~'*65}")
        print("  Phase 1 -- Parameter sweep (delta x pitch_max)")
        print(f"  Sweep at 1.0 m, angles 0 deg + 180 deg (worst pair)")
        print(f"{'~'*65}")

        sweep_dist = 1.0
        sweep_angles = [0, 180]  # opposite directions

        omegas = [1.2, 1.5, 1.8]
        r_growths = [0.12, 0.15, 0.18]

        sweep_results = {}  # (omega, rg) -> max_steps across angles (or None)

        for omega in omegas:
            for rg in r_growths:
                worst_steps = 0
                all_found = True
                details = []

                for ang_deg in sweep_angles:
                    ang_rad = math.radians(ang_deg)
                    sx = sweep_dist * math.cos(ang_rad)
                    sy = sweep_dist * math.sin(ang_rad)
                    ctrl = SpiralSearchController(
                        hover_height=hover_h,
                        K=args.K,
                        omega_orbit=omega,
                        r_growth=rg,
                        max_steps=max_steps,
                        damping_steps=args.damping_steps,
                    )
                    res = self._run_spiral(ctrl, sx, sy, max_steps)
                    steps = res['steps_to_detect']
                    if steps is not None:
                        worst_steps = max(worst_steps, steps)
                        details.append(f"{ang_deg}deg:{steps}")
                    else:
                        all_found = False
                        details.append(f"{ang_deg}deg:FAIL")
                        flag = "[OOB]" if res['oob'] else ""
                        details[-1] += flag

                if all_found:
                    sweep_results[(omega, rg)] = worst_steps
                else:
                    sweep_results[(omega, rg)] = None

                status = (f"worst={worst_steps} steps ({worst_steps*0.01:.2f}s)"
                          if all_found else "INCOMPLETE")
                print(f"    w={omega:.1f}  rg={rg:.2f}  "
                      f"-> {status}  [{', '.join(details)}]")

        # Find best: combo where ALL angles found, with lowest worst-case
        valid = {k: v for k, v in sweep_results.items() if v is not None}
        if valid:
            best_key = min(valid, key=valid.get)
            best_omega, best_rg = best_key
            best_steps = valid[best_key]
            print(f"\n  >> Best: omega={best_omega:.1f}, "
                  f"r_growth={best_rg:.2f} "
                  f"-> worst case {best_steps} steps ({best_steps*0.01:.2f}s)")
        else:
            best_omega, best_rg = omegas[1], r_growths[1]
            print(f"\n  >> No full coverage -- fallback: "
                  f"omega={best_omega}, rg={best_rg}")

        # -- Phase 2: Position robustness ----------------------------------
        print(f"\n{'~'*65}")
        print(f"  Phase 2 -- Position robustness "
              f"(w={best_omega:.1f}, rg={best_rg:.2f})")
        print(f"{'~'*65}")

        angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        distances = [0.5, 1.0, 1.5, 2.0, 2.5]

        pos_results = {}

        for dist in distances:
            for ang_deg in angles_deg:
                ang_rad = math.radians(ang_deg)
                sx = dist * math.cos(ang_rad)
                sy = dist * math.sin(ang_rad)
                ctrl = SpiralSearchController(
                    hover_height=hover_h,
                    K=args.K,
                    omega_orbit=best_omega,
                    r_growth=best_rg,
                    max_steps=max_steps,
                    damping_steps=args.damping_steps,
                )
                res = self._run_spiral(ctrl, sx, sy, max_steps)
                steps = res['steps_to_detect']
                pos_results[(ang_deg, dist)] = steps
                status = (f"{steps:4d}" if steps is not None else "FAIL")
                flag = " [OOB]" if res['oob'] else ""
                print(f"    dist={dist:.1f}m  angle={ang_deg:3d}deg  "
                      f"-> {status}{flag}")

        total = len(pos_results)
        found = sum(1 for v in pos_results.values() if v is not None)
        print(f"\n  Detection rate: {found}/{total} "
              f"({found/total*100:.0f}%)")
        if found > 0:
            steps_list = [v for v in pos_results.values() if v is not None]
            print(f"  Mean steps:  {np.mean(steps_list):.0f} "
                  f"({np.mean(steps_list)*0.01:.2f}s)")
            print(f"  Max  steps:  {max(steps_list)}")

            # Per-distance summary
            for dist in distances:
                dist_steps = [pos_results[(a, dist)]
                              for a in angles_deg
                              if pos_results[(a, dist)] is not None]
                rate = len(dist_steps)
                if rate > 0:
                    print(f"    {dist:.1f}m: {rate}/8 angles, "
                          f"mean={np.mean(dist_steps):.0f} steps")
                else:
                    print(f"    {dist:.1f}m: 0/8 angles")

        # -- Phase 3: Handoff quality + trajectory -------------------------
        print(f"\n{'~'*65}")
        print(f"  Phase 3 -- Handoff + trajectory")
        print(f"{'~'*65}")

        # Find a (angle, distance) that worked in Phase 2 at dist >= 1.0m
        handoff_target = None
        for dist in [1.0, 1.5, 2.0]:
            for ang_deg in angles_deg:
                if pos_results.get((ang_deg, dist)) is not None:
                    handoff_target = (ang_deg, dist)
                    break
            if handoff_target:
                break
        if handoff_target is None:
            handoff_target = (180, 1.0)  # fallback

        h_ang, h_dist = handoff_target
        h_rad = math.radians(h_ang)
        h_sx = h_dist * math.cos(h_rad)
        h_sy = h_dist * math.sin(h_rad)

        print(f"  Sphere at {h_dist:.1f} m, {h_ang} deg")

        ctrl = SpiralSearchController(
            hover_height=hover_h,
            K=args.K,
            omega_orbit=best_omega,
            r_growth=best_rg,
            max_steps=max_steps,
            damping_steps=args.damping_steps,
        )
        handoff_run_steps = max_steps + 300
        handoff_res = self._run_spiral(
            ctrl, h_sx, h_sy, handoff_run_steps, record_trajectory=True)

        if handoff_res['detected']:
            det_step = handoff_res['steps_to_detect']
            traj = handoff_res['trajectory']

            post_det = traj[det_step:]
            if len(post_det) > 1:
                handoff_end = None
                for i, (x, y, z, yaw, mode, vis) in enumerate(post_det):
                    if mode == 'rl':
                        handoff_end = i
                        break

                yaw_angles = [t[3] for t in traj]
                yaw_rates = np.diff(yaw_angles) / 0.01
                yaw_at_det = (yaw_rates[det_step]
                              if det_step < len(yaw_rates) else 0.0)

                z_at_det = post_det[0][2]
                z_end_idx = min(args.damping_steps + 20, len(post_det) - 1)
                z_after = post_det[z_end_idx][2]
                z_drift = abs(z_after - z_at_det)

                print(f"  Detection at step {det_step} "
                      f"({det_step*0.01:.2f}s)")
                print(f"  Yaw rate at detection: "
                      f"{yaw_at_det:.2f} rad/s "
                      f"({math.degrees(yaw_at_det):.1f} deg/s)")
                print(f"  Altitude drift during handoff: "
                      f"{z_drift:.4f} m")
                if handoff_end is not None:
                    yaw_idx = det_step + handoff_end
                    yaw_after = (yaw_rates[yaw_idx]
                                 if yaw_idx < len(yaw_rates) else 0.0)
                    print(f"  Yaw rate after handoff: "
                          f"{yaw_after:.2f} rad/s")
                    print(f"  Handoff duration: {handoff_end} steps "
                          f"({handoff_end*0.01:.2f}s)")
        else:
            print("  Target never detected -- cannot test handoff")
            print(f"  (displacement: {handoff_res['displacement']:.2f} m)")

        # -- Save outputs --------------------------------------------------
        print(f"\n{'~'*65}")
        print(f"  Saving outputs ...")
        print(f"{'~'*65}")

        self._save_sweep_heatmap(omegas, r_growths, sweep_results,
                                 out_dir)
        self._save_position_polar(angles_deg, distances, pos_results,
                                  max_steps, out_dir)
        if handoff_res.get('trajectory'):
            self._save_trajectory(handoff_res, hover_h,
                                  h_sx, h_sy, out_dir)
        self._save_summary(best_omega, best_rg, sweep_results,
                           pos_results, handoff_res,
                           handoff_target, out_dir)

        print(f"\n  All output saved to {out_dir}/")
        print(f"{'='*65}")

    # ==================================================================
    #  Visualisation
    # ==================================================================

    def _save_sweep_heatmap(self, omegas, r_growths, results, out_dir):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        grid = np.full((len(r_growths), len(omegas)), np.nan)
        for j, rg in enumerate(r_growths):
            for i, w in enumerate(omegas):
                val = results.get((w, rg))
                if val is not None:
                    grid[j, i] = val * 0.01

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(grid, aspect='auto', origin='lower',
                        cmap='RdYlGn_r',
                        extent=[omegas[0] - 0.2, omegas[-1] + 0.2,
                                r_growths[0] - 0.02,
                                r_growths[-1] + 0.02])
        ax.set_xlabel('Omega orbit (rad/s)')
        ax.set_ylabel('r_growth (m/s)')
        ax.set_title('Worst-case time to detection (s)\n'
                     'sphere at 1.0 m, both 0 deg + 180 deg must pass')

        for j, rg in enumerate(r_growths):
            for i, w in enumerate(omegas):
                val = results.get((w, rg))
                if val is not None:
                    ax.text(w, rg, f"{val*0.01:.1f}s",
                            ha='center', va='center', fontsize=10,
                            color='white' if val * 0.01 > 5 else 'black')
                else:
                    ax.text(w, rg, "X", ha='center', va='center',
                            fontsize=14, color='red', fontweight='bold')

        plt.colorbar(im, ax=ax, label='Time (s)')
        plt.tight_layout()
        plt.savefig(str(out_dir / 'param_sweep_heatmap.png'), dpi=150)
        plt.close()
        print(f"  Saved: param_sweep_heatmap.png")

    def _save_position_polar(self, angles, distances, results,
                             max_steps, out_dir):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                               figsize=(8, 8))

        for dist in distances:
            for ang_deg in angles:
                ang_rad = math.radians(ang_deg)
                steps = results.get((ang_deg, dist))
                if steps is not None:
                    color_val = min(1.0, steps / max_steps)
                    color = plt.cm.RdYlGn_r(color_val)
                    ax.scatter(ang_rad, dist, c=[color], s=120,
                               edgecolors='black', linewidths=0.5,
                               zorder=3)
                    ax.annotate(f"{steps*0.01:.1f}s",
                                (ang_rad, dist), fontsize=7,
                                ha='center', va='bottom')
                else:
                    ax.scatter(ang_rad, dist, c='red', s=120,
                               marker='x', linewidths=2, zorder=3)

        ax.set_title('Detection time by sphere position\n'
                     '(seconds, red X = not found)', pad=20)
        ax.set_rlabel_position(45)
        ax.set_ylabel('Distance (m)', labelpad=30)
        plt.tight_layout()
        plt.savefig(str(out_dir / 'position_polar.png'), dpi=150)
        plt.close()
        print(f"  Saved: position_polar.png")

    def _save_trajectory(self, res, hover_h, sphere_x, sphere_y, out_dir):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        traj = res['trajectory']
        if len(traj) < 2:
            print("  (trajectory too short to plot)")
            return

        xs = [t[0] for t in traj]
        ys = [t[1] for t in traj]
        zs = [t[2] for t in traj]
        modes = [t[4] for t in traj]
        det_step = res['steps_to_detect']

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # -- Top-down --
        ax = axes[0]
        for i in range(len(xs) - 1):
            color = {'spiral': 'tab:orange', 'handoff': 'tab:purple',
                     'rl': 'tab:green'}.get(modes[i], 'gray')
            ax.plot(xs[i:i+2], ys[i:i+2], '-', color=color, lw=1.0)

        ax.plot(xs[0], ys[0], 'go', ms=8, label='Start')
        if det_step is not None and det_step < len(xs):
            ax.plot(xs[det_step], ys[det_step], 'b^', ms=10,
                    label=f'Detection (step {det_step})')
        ax.plot(sphere_x, sphere_y, 'ms', ms=12, label='Sphere')

        fov_r = hover_h * math.tan(math.atan(18 / 45))
        circle = plt.Circle((xs[0], ys[0]), fov_r, fill=False,
                             ls='--', color='gray', lw=0.8,
                             label=f'FOV r={fov_r:.2f}m')
        ax.add_patch(circle)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Top-down trajectory')
        ax.set_aspect('equal')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # -- Altitude --
        ax = axes[1]
        ts = np.arange(len(zs)) * 0.01
        ax.plot(ts, zs, 'b-', lw=1.2)
        ax.axhline(hover_h, color='red', ls='--', lw=1, alpha=0.5,
                   label=f'Target z={hover_h:.2f}m')
        if det_step is not None:
            ax.axvline(det_step * 0.01, color='green', ls=':',
                       label='Detection')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude z (m)')
        ax.set_title('Altitude hold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # -- Yaw --
        ax = axes[2]
        yaws = [math.degrees(t[3]) for t in traj]
        ax.plot(ts, yaws, 'r-', lw=1.0)
        if det_step is not None:
            ax.axvline(det_step * 0.01, color='green', ls=':',
                       label='Detection')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw (deg)')
        ax.set_title('Yaw rotation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        sphere_dist = math.sqrt(sphere_x**2 + sphere_y**2)
        sphere_ang = math.degrees(math.atan2(sphere_y, sphere_x))
        plt.suptitle(f'Spiral search -- sphere at {sphere_dist:.1f} m, '
                     f'{sphere_ang:.0f} deg', fontsize=12)
        plt.tight_layout()
        plt.savefig(str(out_dir / 'trajectory.png'), dpi=150)
        plt.close()
        print(f"  Saved: trajectory.png")

    def _save_summary(self, best_omega, best_rg, sweep_results,
                      pos_results, handoff_res, handoff_target, out_dir):
        lines = [
            "SPIRAL SEARCH TEST -- SUMMARY",
            "=" * 50,
            f"Hover height:       {args.hover_height:.2f} m",
            f"Sphere radius:      {args.target_radius} m",
            f"K (blind gate):     {args.K} steps ({args.K*0.01:.2f}s)",
            f"Max steps:          {args.max_steps} ({args.max_steps*0.01:.1f}s)",
            f"Damping steps:      {args.damping_steps}",
            "",
            "BEST PARAMETERS:",
            f"  omega_orbit:      {best_omega:.1f} rad/s",
            f"  r_growth:         {best_rg:.2f} m/s",
            f"  yaw_delta:        0.02 (slow FOV rotation)",
            f"  Kv:               1.50 (velocity damping in PD)",
            f"  Kp_xy:            {SpiralSearchController.Kp_xy} (position tracking)",
            f"  max_tilt:         0.25 rad (14.3 deg)",
            "",
            "PD GAINS:",
            f"  Kp_z={SpiralSearchController.Kp_z}  "
            f"Kd_z={SpiralSearchController.Kd_z}",
            f"  Kp_att={SpiralSearchController.Kp_att}  "
            f"Kd_att={SpiralSearchController.Kd_att}",
            "",
            "PARAMETER SWEEP (1.0 m, 0 deg + 180 deg):",
        ]

        for (omega, rg), steps in sorted(sweep_results.items()):
            s = (f"worst={steps} steps ({steps*0.01:.2f}s)"
                 if steps is not None else "INCOMPLETE")
            lines.append(f"  w={omega:.1f}  rg={rg:.2f}  -> {s}")

        lines += ["", "POSITION ROBUSTNESS:"]
        total = len(pos_results)
        found = sum(1 for v in pos_results.values() if v is not None)
        lines.append(f"  Detection rate: {found}/{total} "
                     f"({found/total*100:.0f}%)")

        for (ang, dist), steps in sorted(pos_results.items()):
            s = f"{steps} steps" if steps is not None else "FAIL"
            lines.append(f"  angle={ang:3d} deg  dist={dist:.1f}m  -> {s}")

        lines += ["", f"HANDOFF (sphere at {handoff_target}):"]
        if handoff_res['detected']:
            lines.append(f"  Detection step: {handoff_res['steps_to_detect']}")
            lines.append(f"  Displacement: {handoff_res['displacement']:.2f} m")
        else:
            lines.append("  Target never detected")
            lines.append(f"  Displacement: {handoff_res['displacement']:.2f} m")

        lines += [
            "",
            "DESIGN NOTES:",
            "  Trajectory tracking with feedforward + PD position control.",
            "  Desired position: x=r(t)*cos(wt), y=r(t)*sin(wt)",
            "  where r(t)=r_growth*t+0.05 (Archimedes spiral).",
            "  Feedforward = d²x/dt²: includes centripetal (-r*w²) and",
            "  Coriolis (-2*dr*w) terms automatically.",
            "  PD on (pos_err, vel_err) corrects tracking drift.",
            "  CRITICAL FIX: desired_pitch = (cos(ψ)·ax + sin(ψ)·ay)/g",
            "  and desired_roll = (sin(ψ)·ax - cos(ψ)·ay)/g. Previous",
            "  versions had roll/pitch SWAPPED, causing 90 deg direction error.",
        ]

        (out_dir / 'spiral_summary.txt').write_text(
            "\n".join(lines), encoding='utf-8')
        print(f"  Saved: spiral_summary.txt")


# ======================================================================
if __name__ == "__main__":
    t0 = time.time()
    app = SpiralTestApp()
    app.run_test()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    app.userExit()
    sys.exit(0)
