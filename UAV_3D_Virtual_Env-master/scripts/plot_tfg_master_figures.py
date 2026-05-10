#!/usr/bin/env python
"""
Genera las figuras y tablas maestras del TFG.

Produce los siguientes ficheros en `../memoria/figuras/` (relativo al raíz
del proyecto UAV_3D_Virtual_Env-master) en formato PDF (vectorial, ideal
para LaTeX) y PNG (150 dpi para previsualización):

  figura_progreso_estatico.{pdf,png}
      Curvas de aprendizaje sobre target estático: v8.1 (baseline, techo
      ~196 pasos) vs v9.1.1 (auditoría aplicada, llega a 3000 pasos).
      Demuestra que la sinergia gamma=0.995 + VecNormalize rompe el
      techo estructural.

  figura_progreso_v10.{pdf,png}
      Curvas de aprendizaje hacia el objetivo del TFG (target en
      movimiento): v9.1.1 (referencia estática) vs v10.4-pilot
      (validación de infraestructura) vs v10.4 (final con currículo).
      Líneas verticales señalan las transiciones del currículo
      target_speed: 0 → 0.05 → 0.10 → 0.15 m/s.

  figura_ablations_v9.{pdf,png}
      Bar chart de la sinergia gamma × VecNormalize: 4 configuraciones
      (baseline v8.1, ablate_gamma, ablate_normalize, full) con su
      best_mean_steps. Cuantifica la sinergia (×14.31).

  figura_robustez_v10.{pdf,png}
      Bar chart de robustez del modelo final v10.4 sobre seeds nuevos
      (test offline) en 4 velocidades del target. Eje izquierdo: survival
      rate (%). Eje derecho: mean_steps.

  tabla_ablations.tex     LaTeX-ready, listo para \\input{}
  tabla_robustez.tex      LaTeX-ready, listo para \\input{}
  tabla_versiones.tex     LaTeX-ready, cronología de versiones del TFG

Uso:
    python scripts/plot_tfg_master_figures.py

Es idempotente: cada ejecución sobrescribe los outputs.
"""

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 stdout so → and other unicode chars print on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Estilo unificado para todas las figuras
# ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Paleta consistente entre figuras
COLORS = {
    'v8_1':       '#888888',  # gris (baseline)
    'v9_1_1':     '#1f77b4',  # azul (target estático cerrado)
    'v10_4_pilot':'#2ca02c',  # verde (validación)
    'v10_4':      '#d62728',  # rojo (final TFG)
    'ablate_g':   '#ff7f0e',  # naranja
    'ablate_n':   '#9467bd',  # violeta
}


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Figuras maestras del TFG")
    p.add_argument('--models-dir', type=str, default='./models')
    p.add_argument('--output-dir', type=str,
                   default='../memoria/figuras',
                   help="Dónde guardar las figuras y tablas. Por defecto "
                        "../memoria/figuras (proyecto LaTeX hermano).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Helpers de carga
# ──────────────────────────────────────────────────────────────────────

def load_eval_log(model_dir):
    p = Path(model_dir) / 'eval_log.json'
    if not p.exists():
        print(f"  WARNING: {p} no existe")
        return None
    with open(p) as f:
        data = json.load(f)
    if not data:
        return None
    return {
        'timesteps': np.array([d['timestep'] for d in data]),
        'mean_steps': np.array([d['mean_steps'] for d in data]),
        'visibility': np.array([d['visibility'] for d in data]),
        'jerk': np.array([d['jerk'] for d in data]),
    }


def load_test_results(model_dir, name):
    p = Path(model_dir) / f'{name}.json'
    if not p.exists():
        print(f"  WARNING: {p} no existe")
        return None
    with open(p) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# Figura 1 — progreso sobre target estático (v8.1 vs v9.1.1)
# ──────────────────────────────────────────────────────────────────────

def plot_progreso_estatico(models_dir, output_dir):
    print("\n[Figura 1] progreso_estatico — v8.1 vs v9.1.1")
    v81 = load_eval_log(Path(models_dir) / 'hover_track_v8_1')
    v911 = load_eval_log(Path(models_dir) / 'hover_track_v9_1_1')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    if v81 is not None:
        ax.plot(v81['timesteps'] / 1000, v81['mean_steps'],
                color=COLORS['v8_1'], marker='s', markersize=4,
                linewidth=1.5, label='v8.1 (baseline)')
    if v911 is not None:
        ax.plot(v911['timesteps'] / 1000, v911['mean_steps'],
                color=COLORS['v9_1_1'], marker='o', markersize=4,
                linewidth=1.5, label='v9.1.1 (auditoría aplicada)')
    ax.axhline(196, color=COLORS['v8_1'], linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(3000, color='black', linestyle='--', linewidth=0.8,
               alpha=0.4, label='Techo del entorno (3000)')
    ax.set_xlabel('Pasos de entrenamiento (×1000)')
    ax.set_ylabel('Pasos promedio por episodio (eval)')
    ax.set_title('Duración media del episodio en evaluación')
    ax.legend(loc='center right')
    ax.set_ylim(0, 3300)

    ax = axes[1]
    if v81 is not None:
        ax.plot(v81['timesteps'] / 1000, 100 * v81['visibility'],
                color=COLORS['v8_1'], marker='s', markersize=4,
                linewidth=1.5, label='v8.1')
    if v911 is not None:
        ax.plot(v911['timesteps'] / 1000, 100 * v911['visibility'],
                color=COLORS['v9_1_1'], marker='o', markersize=4,
                linewidth=1.5, label='v9.1.1')
    ax.set_xlabel('Pasos de entrenamiento (×1000)')
    ax.set_ylabel('Visibilidad del target (%)')
    ax.set_title('Porcentaje de pasos con target visible')
    ax.legend(loc='lower right')
    ax.set_ylim(50, 105)

    fig.suptitle('Progreso sobre target estático: efecto de la auditoría '
                 '(γ=0.995, VecNormalize)', fontsize=12, y=1.02)
    fig.tight_layout()

    out = Path(output_dir) / 'figura_progreso_estatico'
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png')
    plt.close(fig)
    print(f"  → {out}.pdf, {out}.png")


# ──────────────────────────────────────────────────────────────────────
# Figura 2 — progreso v10 con currículo target_speed
# ──────────────────────────────────────────────────────────────────────

def plot_progreso_v10(models_dir, output_dir):
    print("\n[Figura 2] progreso_v10 — v9.1.1 vs v10.4-pilot vs v10.4")
    v911 = load_eval_log(Path(models_dir) / 'hover_track_v9_1_1')
    pilot = load_eval_log(Path(models_dir) / 'hover_track_v10_4_pilot')
    v104 = load_eval_log(Path(models_dir) / 'hover_track_v10_4')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel izquierdo: mean_steps
    ax = axes[0]
    if v911 is not None:
        ax.plot(v911['timesteps'] / 1000, v911['mean_steps'],
                color=COLORS['v9_1_1'], marker='o', markersize=3,
                linewidth=1.2, alpha=0.5,
                label='v9.1.1 (referencia estática)')
    if pilot is not None:
        ax.plot(pilot['timesteps'] / 1000, pilot['mean_steps'],
                color=COLORS['v10_4_pilot'], marker='^', markersize=4,
                linewidth=1.5, label='v10.4-pilot (validación)')
    if v104 is not None:
        ax.plot(v104['timesteps'] / 1000, v104['mean_steps'],
                color=COLORS['v10_4'], marker='D', markersize=4,
                linewidth=1.7, label='v10.4 (final TFG)')

    # Líneas del currículo
    for ts, lbl, off in [(100, 'v=0.05', 50), (130, 'v=0.10', 1500),
                          (160, 'v=0.15', 50)]:
        ax.axvline(ts, color='black', linestyle=':', linewidth=0.7, alpha=0.4)
        ax.text(ts + 1, 3100, lbl, fontsize=8, color='dimgray',
                rotation=0, va='top')
    ax.axhline(3000, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Pasos de entrenamiento (×1000)')
    ax.set_ylabel('Pasos promedio por episodio (eval)')
    ax.set_title('Convergencia con currículo de target_speed')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 3300)

    # Panel derecho: visibility
    ax = axes[1]
    if v911 is not None:
        ax.plot(v911['timesteps'] / 1000, 100 * v911['visibility'],
                color=COLORS['v9_1_1'], marker='o', markersize=3,
                linewidth=1.2, alpha=0.5, label='v9.1.1')
    if pilot is not None:
        ax.plot(pilot['timesteps'] / 1000, 100 * pilot['visibility'],
                color=COLORS['v10_4_pilot'], marker='^', markersize=4,
                linewidth=1.5, label='v10.4-pilot')
    if v104 is not None:
        ax.plot(v104['timesteps'] / 1000, 100 * v104['visibility'],
                color=COLORS['v10_4'], marker='D', markersize=4,
                linewidth=1.7, label='v10.4')
    for ts in [100, 130, 160]:
        ax.axvline(ts, color='black', linestyle=':', linewidth=0.7, alpha=0.4)
    ax.set_xlabel('Pasos de entrenamiento (×1000)')
    ax.set_ylabel('Visibilidad del target (%)')
    ax.set_title('Visibilidad durante el currículo')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(40, 105)

    fig.suptitle('Aprendizaje sobre target en movimiento: pilot → final + currículo',
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = Path(output_dir) / 'figura_progreso_v10'
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png')
    plt.close(fig)
    print(f"  → {out}.pdf, {out}.png")


# ──────────────────────────────────────────────────────────────────────
# Figura 3 — bar chart de ablations v9.1.1
# ──────────────────────────────────────────────────────────────────────

def plot_ablations(models_dir, output_dir):
    print("\n[Figura 3] ablations_v9 — sinergia gamma × VecNormalize")

    def best_steps(model_dir):
        log = load_eval_log(Path(models_dir) / model_dir)
        return float(np.max(log['mean_steps'])) if log else 0.0

    runs = [
        ('v8.1\nbaseline\n(γ=0.99, VN=OFF)',
         best_steps('hover_track_v8_1'), COLORS['v8_1']),
        ('ablate_gamma\n(γ=0.99, VN=ON)',
         best_steps('hover_track_v9_1_1_ablate_gamma'), COLORS['ablate_g']),
        ('ablate_normalize\n(γ=0.995, VN=OFF)',
         best_steps('hover_track_v9_1_1_ablate_normalize'), COLORS['ablate_n']),
        ('v9.1.1 full\n(γ=0.995, VN=ON)',
         best_steps('hover_track_v9_1_1'), COLORS['v9_1_1']),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [r[0] for r in runs]
    values = [r[1] for r in runs]
    colors = [r[2] for r in runs]
    xs = np.arange(len(runs))
    bars = ax.bar(xs, values, color=colors, edgecolor='black', linewidth=0.8)

    # Etiquetas con valor numérico encima de cada barra
    for x, v, b in zip(xs, values, bars):
        ax.text(x, v + 50, f'{v:.0f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Anotación de la mejora relativa
    base = values[0] if values[0] > 0 else 1
    for x, v in zip(xs[1:], values[1:]):
        improvement = (v - base) / base * 100
        ax.text(x, v / 2, f'+{improvement:.0f}%',
                ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Pasos promedio máximo en eval')
    ax.set_title('Ablaciones v9.1.1: sinergia γ × VecNormalize')
    ax.axhline(3000, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.text(3.4, 3030, 'Techo (3000)', fontsize=8, color='dimgray', va='bottom')
    ax.set_ylim(0, 3300)
    ax.grid(axis='x', alpha=0)

    fig.tight_layout()
    out = Path(output_dir) / 'figura_ablations_v9'
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png')
    plt.close(fig)
    print(f"  → {out}.pdf, {out}.png")


# ──────────────────────────────────────────────────────────────────────
# Figura 4 — robustez v10.4 a 4 velocidades (test offline)
# ──────────────────────────────────────────────────────────────────────

def plot_robustez_v10(models_dir, output_dir):
    print("\n[Figura 4] robustez_v10 — test offline a 4 velocidades")
    v10_dir = Path(models_dir) / 'hover_track_v10_4'

    speed_files = [
        (0.0, 'test_results_speed_0_00'),
        (0.05, 'test_results_speed_0_05'),
        (0.10, 'test_results_speed_0_10'),
        (0.15, 'test_results_speed_0_15'),
    ]

    speeds, surv, mean_steps, mean_vis, mean_jerk = [], [], [], [], []
    for sp, fname in speed_files:
        data = load_test_results(v10_dir, fname)
        if data is None:
            continue
        speeds.append(sp)
        surv.append(100 * data.get('survival_rate', 0))
        mean_steps.append(data.get('mean_steps', 0))
        mean_vis.append(100 * data.get('mean_visibility', 0))
        mean_jerk.append(data.get('mean_jerk', 0))

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    xs = np.arange(len(speeds))
    width = 0.35

    # Barras: survival rate (eje izq.)
    bars_s = ax1.bar(xs - width / 2, surv, width,
                     color='#2ca02c', edgecolor='black', linewidth=0.8,
                     label='Tasa de supervivencia (%)')
    ax1.set_xlabel('target_speed (m/s)')
    ax1.set_ylabel('Tasa de supervivencia (%)', color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.set_ylim(0, 110)

    # Barras: mean_steps (eje der.)
    ax2 = ax1.twinx()
    bars_m = ax2.bar(xs + width / 2, mean_steps, width,
                     color='#1f77b4', edgecolor='black', linewidth=0.8,
                     label='Pasos medios (eval offline)')
    ax2.set_ylabel('Pasos medios por episodio (de 3000 max)',
                   color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.axhline(3000, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax2.set_ylim(0, 3300)

    # Etiquetas numéricas
    for x, s in zip(xs, surv):
        ax1.text(x - width / 2, s + 2, f'{s:.0f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    for x, m in zip(xs, mean_steps):
        ax2.text(x + width / 2, m + 50, f'{m:.0f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(xs)
    ax1.set_xticklabels([f'{s:.2f}' for s in speeds])
    ax1.set_title('Robustez del modelo final v10.4 (10 episodios eval offline, '
                  'seeds 2000-2009)')

    # Leyenda combinada
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)
    ax1.grid(axis='x', alpha=0)

    fig.tight_layout()
    out = Path(output_dir) / 'figura_robustez_v10'
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png')
    plt.close(fig)
    print(f"  → {out}.pdf, {out}.png")


# ──────────────────────────────────────────────────────────────────────
# Tablas LaTeX
# ──────────────────────────────────────────────────────────────────────

def write_tabla_ablations(models_dir, output_dir):
    print("\n[Tabla 1] tabla_ablations.tex")

    def best_eval(model_dir):
        log = load_eval_log(Path(models_dir) / model_dir)
        if not log:
            return None
        idx = int(np.argmax(log['mean_steps']))
        return {
            'best_step': int(log['timesteps'][idx]),
            'best_mean_steps': float(log['mean_steps'][idx]),
            'vis': float(log['visibility'][idx]),
            'jerk': float(log['jerk'][idx]),
        }

    rows = [
        ('v8.1', 0.99,   'OFF', best_eval('hover_track_v8_1')),
        ('ablate\\_gamma',     0.99,   'ON',  best_eval('hover_track_v9_1_1_ablate_gamma')),
        ('ablate\\_normalize', 0.995,  'OFF', best_eval('hover_track_v9_1_1_ablate_normalize')),
        ('v9.1.1 (full)',      0.995,  'ON',  best_eval('hover_track_v9_1_1')),
    ]

    base = rows[0][3]['best_mean_steps'] if rows[0][3] else 1.0

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Ablación de los hiperparámetros estructurales sobre el '
        r'rendimiento de la política. La sinergia $\gamma$~$\times$~VecNormalize '
        r'es no lineal: ningún cambio individual rompe el techo de v8.1; '
        r'su combinación lo multiplica por 14.}',
        r'\label{tab:ablations_v9}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Versión & $\gamma$ & VecNorm. & Mejor paso & Pasos eval & Visibilidad & '
        r'Mejora \\',
        r'\midrule',
    ]
    for name, gamma, vn, ev in rows:
        if ev is None:
            continue
        improvement = (ev['best_mean_steps'] - base) / base * 100
        lines.append(
            f'{name} & {gamma} & {vn} & '
            f'{ev["best_step"] // 1000}k & '
            f'{ev["best_mean_steps"]:.0f} & '
            f'{ev["vis"] * 100:.1f}\\% & '
            f'{"--" if name == "v8.1" else f"$+{improvement:.0f}\\%$"} '
            r'\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    out = Path(output_dir) / 'tabla_ablations.tex'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  → {out}")


def write_tabla_robustez(models_dir, output_dir):
    print("\n[Tabla 2] tabla_robustez.tex")
    v10_dir = Path(models_dir) / 'hover_track_v10_4'
    speed_files = [
        (0.00, 'test_results_speed_0_00'),
        (0.05, 'test_results_speed_0_05'),
        (0.10, 'test_results_speed_0_10'),
        (0.15, 'test_results_speed_0_15'),
    ]

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Robustez del modelo final v10.4 sobre el problema de '
        r'\emph{tracking} de un target en movimiento. Cada fila reporta '
        r'10 episodios de evaluación \emph{offline} con 10 \emph{seeds} '
        r'no vistos durante el entrenamiento (2000–2009). El modelo '
        r'mantiene un comportamiento robusto hasta $\mathit{target\_speed}=0{,}10$ '
        r'm/s; falla por descontrol vertical (\emph{altitude\_low}) a '
        r'$0{,}15$ m/s. Esto define el régimen operativo demostrado.}',
        r'\label{tab:robustez_v10}',
        r'\begin{tabular}{cccccc}',
        r'\toprule',
        r'$\mathit{target\_speed}$ & Supervivencia & Pasos medios & Visibilidad & '
        r'\emph{Jerk} & Recompensa media \\',
        r'(m/s) & (\%) & (de 3000) & (\%) & (-) & (-) \\',
        r'\midrule',
    ]

    for sp, fname in speed_files:
        data = load_test_results(v10_dir, fname)
        if data is None:
            continue
        lines.append(
            f'{sp:.2f} & '
            f'{data.get("survival_rate", 0) * 100:.0f}\\% & '
            f'{data.get("mean_steps", 0):.0f} '
            f'$\\pm$ {data.get("std_steps", 0):.0f} & '
            f'{data.get("mean_visibility", 0) * 100:.1f}\\% & '
            f'{data.get("mean_jerk", 0):.3f} & '
            f'{data.get("mean_reward", 0):.0f} '
            r'\\'
        )

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    out = Path(output_dir) / 'tabla_robustez.tex'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  → {out}")


def write_tabla_versiones(models_dir, output_dir):
    print("\n[Tabla 3] tabla_versiones.tex")

    def best_steps(model_dir):
        log = load_eval_log(Path(models_dir) / model_dir)
        return float(np.max(log['mean_steps'])) if log else 0.0

    base = best_steps('hover_track_v8_1') or 1.0

    rows = [
        ('v8.1',        'baseline reward minimal',
         best_steps('hover_track_v8_1'), 'estático'),
        ('v9.1.1',      'auditoría: $\\gamma$=0.995, VecNormalize',
         best_steps('hover_track_v9_1_1'), 'estático'),
        ('v10.4-pilot', 'replicar v9.1.1 con script v10',
         best_steps('hover_track_v10_4_pilot'), 'estático'),
        ('v10.4',       'currículo $\\mathit{target\\_speed}$ 0$\\to$0.15',
         best_steps('hover_track_v10_4'), 'movimiento'),
    ]

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Cronología de las versiones que conforman la trayectoria '
        r'experimental del TFG. La columna de mejora relativa muestra el '
        r'avance sobre la línea base v8.1.}',
        r'\label{tab:versiones}',
        r'\begin{tabular}{llccc}',
        r'\toprule',
        r'Versión & Cambio principal & Mejor paso & Régimen & Mejora \\',
        r'\midrule',
    ]
    for name, change, best, regime in rows:
        improvement = (best - base) / base * 100
        lines.append(
            f'{name} & {change} & {best:.0f} & {regime} & '
            f'{"--" if name == "v8.1" else f"$+{improvement:.0f}\\%$"} '
            r'\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    out = Path(output_dir) / 'tabla_versiones.tex'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  → {out}")


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Generación de figuras y tablas maestras del TFG")
    print(f"  Salida: {output_dir.resolve()}")
    print("=" * 70)

    plot_progreso_estatico(args.models_dir, output_dir)
    plot_progreso_v10(args.models_dir, output_dir)
    plot_ablations(args.models_dir, output_dir)
    plot_robustez_v10(args.models_dir, output_dir)
    write_tabla_ablations(args.models_dir, output_dir)
    write_tabla_robustez(args.models_dir, output_dir)
    write_tabla_versiones(args.models_dir, output_dir)

    print("\n" + "=" * 70)
    print("  Generación completa.")
    print("=" * 70)


if __name__ == "__main__":
    main()
