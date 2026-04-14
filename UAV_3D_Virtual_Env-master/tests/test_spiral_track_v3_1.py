#!/usr/bin/env python
"""
Test: Espiral + Tracking con modelos v3.1 y target móvil.

Prueba la integración completa del pipeline simplificado:
    TRACK → (pierde target N steps) → SEARCH (espiral) → (encuentra target) → TRACK

El target sigue una lemniscata de Bernoulli (∞) mientras el dron
alterna entre seguimiento visual (SAC v3.1) y búsqueda en espiral.

Puntos críticos probados:
  1. Transición SEARCH→TRACK sin HANDOFF — ¿cuánto tarda en recuperar?
  2. Target móvil durante espiral     — ¿puede escapar del radio de búsqueda?
  3. Sensibilidad de K (paciencia)    — k=10 vs k=30 vs k=60 steps
  4. Velocidad del target             — slow/medium/fast
  5. Offset inicial                   — easy/medium/hard

Modelos evaluados por defecto: 150k, 400k, best_model (≈500k)

Escenarios por defecto:
  slow_easy    — speed=0.10, offset=0.3 m
  medium       — speed=0.22, offset=0.6 m
  fast_hard    — speed=0.35, offset=1.0 m
  recovery     — speed=0.20, offset=2.0 m (dron lejos, empieza en espiral)
  k_sensitive  — speed=0.20, offset=0.5 m × K=10/30/60

Usage:
    python tests/test_spiral_track_v3_1.py
    python tests/test_spiral_track_v3_1.py --episodes 10 --duration 40
    python tests/test_spiral_track_v3_1.py --no-video --scenarios slow_easy fast_hard
    python tests/test_spiral_track_v3_1.py --k-values 10 30 60
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: F401
from panda3d.core import Filename, loadPrcFile, loadPrcFileData
from direct.showbase.ShowBase import ShowBase

loadPrcFile(Filename.fromOsSpecific(
    os.path.join(project_root, 'config', 'conf.prc')))

from stable_baselines3 import SAC, PPO

from src.simulation.world_setup import world_setup, quad_setup
from src.vision.img_2_cv import opencv_camera
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.utils.episode_recorder import EpisodeRecorder


# ══════════════════════════════════════════════════════════════════════
# Configuración de modelos y escenarios
# ══════════════════════════════════════════════════════════════════════

HOVER_HEIGHT = 1.394
VISION_RADIUS = 0.5
SPIRAL_OMEGA = 1.8
SPIRAL_R_GROWTH = 0.12

# Modelos v3.1 disponibles (ruta, etiqueta)
MODEL_DEFS = {
    '150k':  './models/hover_track_v3_1/checkpoints/model_150000_steps.zip',
    '400k':  './models/hover_track_v3_1/checkpoints/model_400000_steps.zip',
    'best':  './models/hover_track_v3_1/best_model.zip',
    '250k':  './models/hover_track_v3_1/checkpoints/model_250000_steps.zip',
    '350k':  './models/hover_track_v3_1/checkpoints/model_350000_steps.zip',
    '500k':  './models/hover_track_v3_1/checkpoints/model_500000_steps.zip',
}

# Escenarios base (se añaden variantes de K dinámicamente)
BASE_SCENARIOS = {
    'slow_easy': {
        'target_speed': 0.10,
        'init_offset': 0.3,
        'init_vel': 0.10,
        'description': 'Target lento, inicio fácil (≈v4 Phase A)',
    },
    'medium': {
        'target_speed': 0.22,
        'init_offset': 0.6,
        'init_vel': 0.20,
        'description': 'Target velocidad media (≈v4 Phase B)',
    },
    'fast_hard': {
        'target_speed': 0.35,
        'init_offset': 1.0,
        'init_vel': 0.35,
        'description': 'Target rápido, inicio difícil (≈v4 Phase C)',
    },
    'recovery': {
        'target_speed': 0.20,
        'init_offset': 2.5,
        'init_vel': 0.30,
        'description': 'Recuperación: dron lejos, empieza en espiral',
    },
}


# ══════════════════════════════════════════════════════════════════════
# FSM de 2 estados: TRACK / SEARCH
# ══════════════════════════════════════════════════════════════════════

class SpiralTrackFSM:
    """Finite State Machine de 2 estados para el pipeline spiral→track.

    TRACK: el SAC v3.1 controla los motores.
           Si no ve el target durante k_invisible steps consecutivos
           → transición a SEARCH.

    SEARCH: el modelo de espiral controla los motores.
            Cuando el target vuelve a ser visible → transición inmediata
            a TRACK (sin HANDOFF). El v3.1 hard tier entrena esta
            recuperación directa.

    Parámetros críticos
    -------------------
    k_invisible : int
        Steps consecutivos sin ver el target antes de lanzar espiral.
        Demasiado bajo → espirales frecuentes por pérdidas momentáneas.
        Demasiado alto → el dron tarda en reaccionar (pierde al target).
    """

    TRACK  = 'TRACK'
    SEARCH = 'SEARCH'

    def __init__(self, spiral_model, k_invisible=20,
                 omega=SPIRAL_OMEGA, r_growth=SPIRAL_R_GROWTH,
                 hover_height=HOVER_HEIGHT, vision_radius=VISION_RADIUS):
        self.spiral_model = spiral_model
        self.k_invisible  = k_invisible
        self.omega        = omega
        self.r_growth     = r_growth
        self.hover_height = hover_height
        self.vision_radius = vision_radius

        self._state = self.TRACK
        self._invisible_count = 0
        self._spiral_step = 0
        self._theta = 0.0
        self._cx = 0.0
        self._cy = 0.0
        self._ref_x = self._ref_y = 0.0
        self._ref_vx = self._ref_vy = 0.0

        # Métricas de transición
        self.n_spiral_activations = 0
        self._steps_since_handoff = 0
        self._post_handoff_dists  = []   # centering dist en 1ras 5s tras handoff

    def reset(self, drone_x, drone_y):
        self._state = self.TRACK
        self._invisible_count = 0
        self.n_spiral_activations = 0
        self._steps_since_handoff = 0
        self._post_handoff_dists  = []
        self._reset_spiral(drone_x, drone_y)

    # ── Spiral interno ──────────────────────────────────────────────

    def _reset_spiral(self, cx, cy):
        self._cx = cx
        self._cy = cy
        self._spiral_step = 0
        self._theta = 0.0
        self._ref_x = cx + 0.05
        self._ref_y = cy
        self._ref_vx = 0.0
        self._ref_vy = 0.0

    def _advance_spiral(self, dt=0.01):
        self._spiral_step += 1
        t = self._spiral_step * dt
        r = self.r_growth * t + 0.05
        a_budget = 0.70 * 9.82 * math.sin(0.25)
        w = min(self.omega, math.sqrt(a_budget / max(r, 0.05)))
        self._theta += w * dt
        ct, st = math.cos(self._theta), math.sin(self._theta)
        self._ref_x  = self._cx + r * ct
        self._ref_y  = self._cy + r * st
        self._ref_vx = self.r_growth * ct - r * w * st
        self._ref_vy = self.r_growth * st + r * w * ct

    def _build_spiral_obs(self, state13):
        """Construye observación 18-D para el modelo de espiral."""
        dx = (self._ref_x - state13[0]) / self.vision_radius
        dy = (self._ref_y - state13[2]) / self.vision_radius
        dz = (self.hover_height - state13[4]) / self.hover_height
        v_mag = math.sqrt(self._ref_vx**2 + self._ref_vy**2) + 1e-6
        vx_n  = self._ref_vx / v_mag
        vy_n  = self._ref_vy / v_mag
        ref = np.array([dx, dy, dz, vx_n, vy_n], dtype=np.float32)
        return np.concatenate([state13.astype(np.float32), ref])

    # ── Interfaz principal ──────────────────────────────────────────

    def get_action(self, obs19, target_visible, sac_model,
                   state13, centering_dist, dt=0.01):
        """Devuelve la acción según el estado actual de la FSM.

        Parámetros
        ----------
        obs19         : np.ndarray (19,) — observación del entorno para SAC
        target_visible: bool             — ¿el target es visible ahora?
        sac_model     : SAC              — modelo v3.1 activo
        state13       : np.ndarray (13,) — estado físico del dron
        centering_dist: float            — distancia al centro (para métricas post-handoff)
        dt            : float
        """
        prev_state = self._state

        # ── Transiciones ──────────────────────────────────────────
        if self._state == self.TRACK:
            if target_visible:
                self._invisible_count = 0
            else:
                self._invisible_count += 1
                if self._invisible_count >= self.k_invisible:
                    self._state = self.SEARCH
                    self._reset_spiral(float(state13[0]), float(state13[2]))
                    self.n_spiral_activations += 1
                    self._invisible_count = 0

        elif self._state == self.SEARCH:
            if target_visible:
                # Encontró el target → regresa a TRACK sin HANDOFF
                self._state = self.TRACK
                self._invisible_count = 0
                self._steps_since_handoff = 0

        # ── Métricas post-handoff ──────────────────────────────────
        if self._state == self.TRACK and prev_state == self.SEARCH:
            pass  # acaba de entrar en TRACK
        if self._state == self.TRACK and self._steps_since_handoff < 500:
            self._steps_since_handoff += 1
            if target_visible:
                self._post_handoff_dists.append(centering_dist)

        # ── Acción ────────────────────────────────────────────────
        if self._state == self.SEARCH:
            self._advance_spiral(dt)
            spiral_obs = self._build_spiral_obs(state13)
            action, _ = self.spiral_model.predict(
                spiral_obs, deterministic=True)
            return action

        # TRACK
        action, _ = sac_model.predict(obs19, deterministic=True)
        return action

    @property
    def current_state(self):
        return self._state

    def post_handoff_mean_centering(self, window=200):
        """Distancia media al centro en las primeras 2s tras cada handoff."""
        recent = self._post_handoff_dists[:window]
        return float(np.mean(recent)) if recent else float('nan')


# ══════════════════════════════════════════════════════════════════════
# Utilidades de carga
# ══════════════════════════════════════════════════════════════════════

def load_sac_model(label_or_path):
    path = MODEL_DEFS.get(label_or_path, label_or_path)
    p = Path(path)
    if not p.exists():
        print(f"  [WARN] Modelo no encontrado: {p}")
        return None, path
    model = SAC.load(str(p), env=None)
    return model, str(p)


def load_spiral_model(path='./models/spiral_follow/best_model.zip'):
    p = Path(path)
    if not p.exists():
        print(f"  [ERROR] Modelo espiral no encontrado: {p}")
        sys.exit(1)
    return PPO.load(str(p), env=None)


# ══════════════════════════════════════════════════════════════════════
# Ejecución de un episodio
# ══════════════════════════════════════════════════════════════════════

def run_episode(env, app, fsm, sac_model, scenario, duration_s,
                record=False, recorder=None, ext_camera=None):
    """Ejecuta un episodio completo con la FSM.

    Devuelve un diccionario de métricas.
    """
    dt = env.base_env.t_step
    total_steps = int(duration_s / dt)

    # ── Reset del entorno ──────────────────────────────────────────
    env.target_speed = scenario['target_speed']
    env.init_vel_range = scenario['init_vel']
    env.stabilization_only = False

    obs, info = env.reset()

    # Reposicionar dron encima del target + offset inicial
    state = env.base_env.state.copy()
    off = scenario['init_offset']
    angle = np.random.uniform(0, 2 * np.pi)
    state[0] = env.target_pos[0] + off * math.cos(angle)
    state[2] = env.target_pos[1] + off * math.sin(angle)
    state[4] = env.target_pos[2] + env.hover_height
    vr = scenario['init_vel']
    state[1] = np.random.uniform(-vr, vr)
    state[3] = np.random.uniform(-vr, vr)
    state[5] = np.random.uniform(-0.05, 0.05)
    env.base_env.state = state
    env._update_visualization()
    app.graphicsEngine.renderFrame()
    env._capture_camera_images(force_capture=True)
    obs = env._build_observation(state.astype(np.float32))

    drone_x, drone_y = float(state[0]), float(state[2])
    fsm.reset(drone_x, drone_y)

    # ── Acumuladores ──────────────────────────────────────────────
    rewards          = []
    track_steps      = 0   # steps en TRACK con target visible
    search_steps     = 0   # steps en SEARCH
    visible_steps    = 0
    centering_dists  = []
    fractions        = []
    action_jerks     = []
    prev_action      = None
    terminated_early = False

    # Para análisis de transiciones
    transitions      = []   # (step, 'TRACK'|'SEARCH')
    prev_fsm_state   = fsm.current_state

    if record and recorder:
        recorder.start_episode(0)

    for step in range(total_steps):
        state13       = env.base_env.state.astype(np.float32)
        vt            = info.get('visual_tracking', {})
        target_visible = vt.get('target_visible', False)
        centering_dist = vt.get('centering_dist', 1.0) if target_visible else 1.0

        action = fsm.get_action(
            obs, target_visible, sac_model, state13, centering_dist, dt)

        obs, reward, terminated, truncated, info = env.step(action)
        app.graphicsEngine.renderFrame()

        rewards.append(reward)

        # Métricas de estado FSM
        cur_fsm = fsm.current_state
        if cur_fsm != prev_fsm_state:
            transitions.append((step, cur_fsm))
        prev_fsm_state = cur_fsm

        vt = info.get('visual_tracking', {})
        target_visible = vt.get('target_visible', False)

        if cur_fsm == fsm.TRACK and target_visible:
            track_steps += 1
            cd = vt.get('centering_dist', np.nan)
            centering_dists.append(cd)
            fractions.append(vt.get('target_fraction', 0.0))
        if cur_fsm == fsm.SEARCH:
            search_steps += 1
        if target_visible:
            visible_steps += 1

        if prev_action is not None:
            action_jerks.append(
                float(np.mean(np.abs(action - prev_action))))
        prev_action = action.copy()

        if record and recorder:
            if step % 10 == 0:   # ~10 FPS a dt=0.01
                fpv = env._last_high_freq_image
                bird = None
                if ext_camera is not None:
                    ok, rgba = ext_camera.get_image()
                    if ok:
                        bird = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
                overlay = {
                    'FSM': cur_fsm,
                    'Visible': target_visible,
                    'Spirals': fsm.n_spiral_activations,
                    'Step': step,
                    'Reward': round(float(reward), 1),
                    'visual_tracking': vt,
                }
                recorder.capture_frame(fpv, bird, overlay)

        if terminated or truncated:
            terminated_early = True
            break

    if record and recorder:
        recorder.end_episode()

    _m = lambda lst: float(np.mean(lst)) if lst else float('nan')

    return {
        'steps':              len(rewards),
        'terminated_early':   terminated_early,
        'total_reward':       sum(rewards),
        'track_pct':          100 * track_steps  / max(len(rewards), 1),
        'search_pct':         100 * search_steps / max(len(rewards), 1),
        'visibility_pct':     100 * visible_steps / max(len(rewards), 1),
        'n_spiral_activations': fsm.n_spiral_activations,
        'mean_centering_dist': _m(centering_dists),
        'mean_fraction':       _m(fractions),
        'mean_action_jerk':    _m(action_jerks),
        'post_handoff_cent':   fsm.post_handoff_mean_centering(),
        'n_transitions':       len(transitions),
    }


# ══════════════════════════════════════════════════════════════════════
# Ejecución de un escenario (N episodios)
# ══════════════════════════════════════════════════════════════════════

def run_scenario(env, app, fsm, sac_model, model_label,
                 scenario_name, scenario, args, output_dir,
                 recorder=None, ext_camera=None):
    """Corre N episodios y devuelve estadísticas agregadas."""
    ep_results = []
    did_record = False

    for ep in range(args.episodes):
        record_this = (not did_record and not args.no_video
                       and recorder is not None)
        res = run_episode(
            env, app, fsm, sac_model, scenario,
            duration_s=args.duration,
            record=record_this,
            recorder=recorder,
            ext_camera=ext_camera,
        )
        if record_this:
            did_record = True
        res['episode'] = ep + 1
        res['model'] = model_label
        res['scenario'] = scenario_name
        ep_results.append(res)

        tag = 'CRASH' if res['terminated_early'] else 'OK'
        print(f"    Ep {ep+1:2d}  R={res['total_reward']:8.0f}  "
              f"track={res['track_pct']:5.1f}%  "
              f"search={res['search_pct']:5.1f}%  "
              f"spirals={res['n_spiral_activations']}  "
              f"cent={res['mean_centering_dist']:.3f}  "
              f"p_cent={res['post_handoff_cent']:.3f}  [{tag}]")

    return ep_results


# ══════════════════════════════════════════════════════════════════════
# Impresión de tabla de resultados
# ══════════════════════════════════════════════════════════════════════

def print_results_table(all_results):
    """Imprime tabla comparativa de todos los resultados."""
    from collections import defaultdict

    # Agrupar por (modelo, escenario)
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r['model'], r['scenario'])].append(r)

    print(f"\n{'='*120}")
    print("  RESULTADOS: Spiral + Tracking v3.1 — Target Móvil (Lemniscata)")
    print(f"{'='*120}")

    header = (f"  {'Model':>8} | {'Scenario':<16} | "
              f"{'Surv%':>5} | {'R_mean':>7} | "
              f"{'Track%':>6} | {'Srch%':>5} | {'Vis%':>5} | "
              f"{'Spirals':>7} | {'Cent':>5} | {'PostH':>5} | {'Jerk':>5}")
    print(header)
    print(f"  {'-'*114}")

    # Ordenar por escenario, luego modelo
    for key in sorted(grouped.keys(), key=lambda k: (k[1], k[0])):
        model_label, scenario_name = key
        eps = grouped[key]
        n = len(eps)
        survived = [e for e in eps if not e['terminated_early']]
        surv_pct = 100 * len(survived) / n

        def _m(field):
            vals = [e[field] for e in eps if not math.isnan(e[field])]
            return float(np.mean(vals)) if vals else float('nan')

        print(f"  {model_label:>8} | {scenario_name:<16} | "
              f"{surv_pct:>5.1f} | {_m('total_reward'):>7.0f} | "
              f"{_m('track_pct'):>6.1f} | {_m('search_pct'):>5.1f} | "
              f"{_m('visibility_pct'):>5.1f} | "
              f"{_m('n_spiral_activations'):>7.1f} | "
              f"{_m('mean_centering_dist'):>5.3f} | "
              f"{_m('post_handoff_cent'):>5.3f} | "
              f"{_m('mean_action_jerk'):>5.3f}")

    print(f"\n  Columnas: Surv%=supervivencia  Track%=% steps tracking"
          f"  Srch%=% steps espiral  Spirals=activaciones/ep"
          f"  Cent=centrado tracking  PostH=centrado post-handoff")
    print(f"{'='*120}\n")


# ══════════════════════════════════════════════════════════════════════
# Generación de plots
# ══════════════════════════════════════════════════════════════════════

def generate_plots(all_results, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no encontrado — omitiendo plots)")
        return

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r['model'], r['scenario'])].append(r)

    scenarios = sorted(set(r['scenario'] for r in all_results))
    models    = sorted(set(r['model']    for r in all_results))
    colors    = {'150k': '#4CAF50', '400k': '#2196F3',
                 'best': '#FF5722', '250k': '#9C27B0',
                 '350k': '#FF9800', '500k': '#795548'}

    def _agg(eps, field):
        vals = [e[field] for e in eps
                if not (isinstance(e[field], float) and math.isnan(e[field]))]
        return float(np.mean(vals)) if vals else 0.0

    # ── Fig 1: métricas principales por escenario y modelo ──
    metrics = [
        ('total_reward',        'Reward total'),
        ('track_pct',           'Track % (steps)'),
        ('n_spiral_activations','Activaciones espiral'),
        ('post_handoff_cent',   'Centrado post-handoff'),
        ('visibility_pct',      'Visibilidad %'),
        ('mean_action_jerk',    'Action Jerk'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(models), 1)

    for ax, (key, title) in zip(axes.flat, metrics):
        for i, model in enumerate(models):
            vals = [_agg(grouped[(model, sc)], key) for sc in scenarios]
            ax.bar(x + i * width - 0.4 + width/2, vals,
                   width=width * 0.9,
                   label=model,
                   color=colors.get(model, '#607D8B'),
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Spiral + Tracking v3.1 — Target Móvil\nComparación de modelos y escenarios',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(str(output_dir / 'comparison_main.png'), dpi=150)
    plt.close(fig)

    # ── Fig 2: supervivencia por modelo y escenario ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        surv = []
        for sc in scenarios:
            eps = grouped[(model, sc)]
            surv.append(100 * sum(1 for e in eps if not e['terminated_early'])
                        / max(len(eps), 1))
        ax.plot(scenarios, surv, marker='o', label=model,
                color=colors.get(model, '#607D8B'), linewidth=2)
    ax.set_ylabel('Supervivencia (%)')
    ax.set_title('Tasa de Supervivencia por Modelo y Escenario')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(str(output_dir / 'survival_rates.png'), dpi=150)
    plt.close(fig)

    print(f"  Plots guardados en {output_dir}/")


# ══════════════════════════════════════════════════════════════════════
# App principal (Panda3D)
# ══════════════════════════════════════════════════════════════════════

class SpiralTrackTestApp(ShowBase):

    def __init__(self, args):
        ShowBase.__init__(self)
        self.args = args
        mydir = Filename.fromOsSpecific(
            os.path.abspath(project_root)).getFullpath()

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Cargando escena 3D...")
        world_setup(self, self.render, mydir)
        quad_setup(self, self.render, mydir)

        self.taskMgr.remove('Camera Movement')
        self.cam.reparentTo(self.render)
        self.cam.setPos(0, -10, 15)
        self.cam.lookAt(0, 0, 4)

        # Cámara FPV (mirando hacia abajo)
        self.fpv_camera = opencv_camera(self, 'fpv_cam', 1)
        self.fpv_camera.cam.reparentTo(self.quad_model)
        self.fpv_camera.cam.setPos(0, 0, -0.1)
        self.fpv_camera.cam.setHpr(0, -90, 0)
        self.fpv_camera.buffer.setActive(1)

        # Cámara externa (vista de pájaro / aérea)
        self.ext_camera = opencv_camera(self, 'ext_cam', 1)
        self.ext_camera.cam.reparentTo(self.render)
        self.ext_camera.cam.setPos(0, -8, 14)
        self.ext_camera.cam.lookAt(0, 0, 4)
        self.ext_camera.buffer.setActive(1)

        # Entorno: target móvil, reward v3.1, cámara abajo
        print("Creando entorno (target móvil, lemniscata, reward v3.1)...")
        self.env = Panda3DQuadrotorEnv(
            panda3d_app=self,
            quad_model=self.quad_model,
            render_node=self.render,
            use_camera=True,
            use_depth=False,
            use_target=True,
            target_mode='moving',          # ← lemniscata en movimiento
            target_speed=0.10,             # sobreescrito por escenario
            lemniscate_scale=2.0,
            target_radius=0.25,
            filming_mode=True,
            enable_collisions=False,
            n=int(args.duration / 0.01) + 100,
            t_step=0.01,
            direct_control=1,
            centroid_obs=True,
            camera_down=True,
            hover_height=HOVER_HEIGHT,
            use_new_reward=True,
            constrained_init=True,
            init_pos_range=0.3,
            init_vel_range=0.10,
            init_ang_range=0.05,
            reward_version='v3.1',
            camera_high_freq_obj=self.fpv_camera,
            camera_high_freq_size=(32, 32),
            camera_low_freq_size=(32, 32),
            exclude_low_freq_camera=True,
        )
        # Permitir que el target empiece cerca del dron
        self.env.min_start_distance = 0.0

        # Cargar modelo de espiral
        print(f"Cargando modelo de espiral: {args.spiral_model}")
        self.spiral_model = load_spiral_model(args.spiral_model)

        # Cargar modelos SAC v3.1
        self.sac_models = {}
        for label in args.models:
            m, path = load_sac_model(label)
            if m is not None:
                self.sac_models[label] = m
                print(f"  ✓ {label}: {Path(path).name}")
            else:
                print(f"  ✗ {label}: no encontrado")

        if not self.sac_models:
            print("ERROR: Ningún modelo SAC disponible.")
            sys.exit(1)

        # Construir lista de escenarios activos
        self.scenarios = self._build_scenarios(args)

        for _ in range(5):
            self.graphicsEngine.renderFrame()

    def _build_scenarios(self, args):
        """Construye el dict de escenarios según los args."""
        active = {}

        # Escenarios base seleccionados
        for name in args.scenarios:
            if name in BASE_SCENARIOS:
                active[name] = BASE_SCENARIOS[name].copy()
                active[name]['k_invisible'] = args.k_invisible

        # Variantes de K si se solicitan
        if len(args.k_values) > 1:
            base_scenario = BASE_SCENARIOS.get('medium',
                            list(BASE_SCENARIOS.values())[0])
            for k in args.k_values:
                key = f"medium_k{k}"
                active[key] = base_scenario.copy()
                active[key]['k_invisible'] = k
                active[key]['description'] = (
                    f"Medium target, k_invisible={k}")

        return active

    def run_all_scenarios(self):
        all_results = []
        all_episode_rows = []

        total_runs = len(self.sac_models) * len(self.scenarios)
        run_idx = 0

        for model_label, sac_model in self.sac_models.items():
            for scenario_name, scenario in self.scenarios.items():
                run_idx += 1
                k = scenario.get('k_invisible', self.args.k_invisible)
                fsm = SpiralTrackFSM(
                    spiral_model=self.spiral_model,
                    k_invisible=k,
                    omega=SPIRAL_OMEGA,
                    r_growth=SPIRAL_R_GROWTH,
                    hover_height=HOVER_HEIGHT,
                    vision_radius=VISION_RADIUS,
                )

                print(f"\n[{run_idx}/{total_runs}] "
                      f"Modelo={model_label}  Escenario={scenario_name}  "
                      f"(k={k}, speed={scenario['target_speed']}, "
                      f"off={scenario['init_offset']}m)")
                print(f"  {scenario.get('description', '')}")

                # Grabador de vídeo para este (modelo, escenario)
                recorder = None
                if not self.args.no_video:
                    vid_dir = (self.output_dir / 'videos'
                               / f"{model_label}_{scenario_name}")
                    recorder = EpisodeRecorder(
                        output_dir=str(vid_dir),
                        fps=10,
                        resolution=(480, 360),
                    )

                # Posicionar cámara externa según offset del escenario
                cam_dist = max(6.0, scenario['init_offset'] * 2.5)
                self.ext_camera.cam.setPos(
                    -cam_dist * 0.5, -cam_dist * 0.8, cam_dist * 0.7)
                self.ext_camera.cam.lookAt(0, 0, 2)

                episode_results = run_scenario(
                    env=self.env,
                    app=self,
                    fsm=fsm,
                    sac_model=sac_model,
                    model_label=model_label,
                    scenario_name=scenario_name,
                    scenario=scenario,
                    args=self.args,
                    output_dir=self.output_dir,
                    recorder=recorder,
                    ext_camera=(self.ext_camera
                                if not self.args.no_video else None),
                )

                all_results.extend(episode_results)
                all_episode_rows.extend(episode_results)

                if recorder is not None:
                    try:
                        tl = recorder.compile_timelapse(
                            f"{model_label}_{scenario_name}.mp4",
                            max_frames_per_ep=300)
                        if tl:
                            print(f"  Vídeo: {tl}")
                    except Exception as e:
                        print(f"  [WARN] No se pudo compilar vídeo: {e}")

        # ── Guardar CSV ──
        if all_episode_rows:
            csv_path = self.output_dir / 'results.csv'
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=all_episode_rows[0].keys())
                w.writeheader()
                w.writerows(all_episode_rows)
            print(f"\n  CSV: {csv_path}")

        # ── Guardar JSON ──
        json_path = self.output_dir / 'summary.json'
        summary = {}
        for r in all_results:
            key = f"{r['model']}__{r['scenario']}"
            if key not in summary:
                summary[key] = []
            summary[key].append({k: v for k, v in r.items()
                                  if k not in ('model', 'scenario')})
        with open(json_path, 'w') as f:
            json.dump({
                'args': vars(self.args),
                'scenarios': self.scenarios,
                'results': summary,
            }, f, indent=2)
        print(f"  JSON: {json_path}")

        # ── Tabla resumen ──
        print_results_table(all_results)

        # ── Plots ──
        if not self.args.no_plots:
            generate_plots(all_results, self.output_dir)

        self.env.close()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Test: Espiral + Tracking v3.1 con target móvil")

    # Modelos
    p.add_argument('--models', nargs='+',
                   default=['150k', '400k', 'best'],
                   help="Etiquetas o rutas de modelos SAC v3.1 "
                        "(opciones: 150k 250k 350k 400k 500k best)")

    # Escenarios
    p.add_argument('--scenarios', nargs='+',
                   default=['slow_easy', 'medium', 'fast_hard', 'recovery'],
                   help="Escenarios a ejecutar "
                        "(opciones: slow_easy medium fast_hard recovery)")

    # FSM
    p.add_argument('--k-invisible', type=int, default=20,
                   help="Steps sin ver target antes de activar espiral "
                        "(default: 20 = 0.2s)")
    p.add_argument('--k-values', nargs='+', type=int, default=[],
                   help="Probar variantes de k sobre escenario 'medium' "
                        "(ej: --k-values 10 30 60)")

    # Episodios
    p.add_argument('--episodes', type=int, default=5,
                   help="Episodios por combinación modelo×escenario")
    p.add_argument('--duration', type=float, default=30.0,
                   help="Duración de cada episodio en segundos")

    # Espiral
    p.add_argument('--spiral-model', type=str,
                   default='./models/spiral_follow/best_model.zip')
    p.add_argument('--omega', type=float, default=SPIRAL_OMEGA)
    p.add_argument('--r-growth', type=float, default=SPIRAL_R_GROWTH)

    # Salida
    p.add_argument('--output-dir', type=str,
                   default='./experiments/spiral_track_v3_1')
    p.add_argument('--no-video', action='store_true',
                   help="Desactivar grabación de vídeo")
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--no-display', action='store_true',
                   help="Ventana mínima (sin interfaz visual)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.no_display:
        loadPrcFileData('', 'win-size 64 64')
        loadPrcFileData('', 'undecorated true')

    print("\n" + "=" * 65)
    print("  TEST: Spiral + Tracking v3.1 — Target Móvil (Lemniscata)")
    print("=" * 65)
    print(f"  Modelos:    {args.models}")
    print(f"  Escenarios: {args.scenarios}")
    print(f"  K_invisible:{args.k_invisible}  "
          f"K_variantes:{args.k_values}")
    print(f"  Episodios:  {args.episodes} × {args.duration}s")
    print(f"  Output:     {args.output_dir}")
    print("=" * 65 + "\n")

    app = SpiralTrackTestApp(args)
    try:
        app.run_all_scenarios()
    except (KeyboardInterrupt, SystemExit):
        print("\nTest interrumpido.")
        app.env.close()


if __name__ == "__main__":
    main()