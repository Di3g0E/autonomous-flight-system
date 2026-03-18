#!/usr/bin/env python
"""
Generate training analysis plots from per-episode CSV data.

Reads training_log.csv produced by train_goal_controller.py and generates
publication-ready figures for the TFG documentation.

Usage:
    python scripts/generate_training_plots.py --csv models/goal_controller/training_log.csv
    python scripts/generate_training_plots.py --csv models/goal_controller/training_log.csv --output figures/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

WINDOW = 100  # rolling average window


def rolling(series, window=WINDOW):
    """Compute rolling mean, filling NaN at the start with expanding mean."""
    return series.rolling(window, min_periods=1).mean()


# ── Plot 1: Reward ───────────────────────────────────────────────────

def plot_reward(df, output_dir):
    fig, ax = plt.subplots()
    episodes = df['episode']

    ax.plot(episodes, df['reward'], alpha=0.25, color='steelblue', linewidth=0.5)
    ax.plot(episodes, rolling(df['reward']), color='steelblue', linewidth=2,
            label=f'Media móvil ({WINDOW} ep.)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa total')
    ax.set_title('Evolución de la recompensa durante el entrenamiento')
    ax.legend()

    path = output_dir / 'plot_reward.png'
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")
    return path


# ── Plot 2: Distance ─────────────────────────────────────────────────

def plot_distance(df, output_dir, filming_distance=None):
    fig, ax = plt.subplots()
    episodes = df['episode']

    # Mean distance with std band
    mean_d = rolling(df['mean_distance'])
    std_d = rolling(df['std_distance'])
    ax.fill_between(episodes, mean_d - std_d, mean_d + std_d,
                    alpha=0.15, color='steelblue')
    ax.plot(episodes, df['mean_distance'], alpha=0.2, color='steelblue', linewidth=0.5)
    ax.plot(episodes, mean_d, color='steelblue', linewidth=2,
            label=f'Distancia media ({WINDOW} ep.)')

    # Min distance
    ax.plot(episodes, rolling(df['min_distance']), color='crimson',
            linewidth=1.2, linestyle='--', label='Distancia mínima')

    # Reference line
    if filming_distance is not None:
        ax.axhline(filming_distance, color='green', linestyle='-.',
                   linewidth=1.5, label=f'Distancia ideal ({filming_distance}m)')

    ax.set_xlabel('Episodio')
    ax.set_ylabel('Distancia al objetivo (m)')
    ax.set_title('Distancia de seguimiento durante el entrenamiento')
    ax.legend()

    path = output_dir / 'plot_distance.png'
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")
    return path


# ── Plot 3: Visual Quality ──────────────────────────────────────────

def plot_visual_quality(df, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    episodes = df['episode']

    # Centering (normalized to 0-100%)
    centering_pct = df['mean_centering'] / 3.0 * 100
    ax1.plot(episodes, centering_pct, alpha=0.2, color='darkorange', linewidth=0.5)
    ax1.plot(episodes, rolling(centering_pct), color='darkorange', linewidth=2,
             label=f'Centrado ({WINDOW} ep.)')

    # Scale (normalized to 0-100%)
    scale_pct = df['mean_scale'] / 2.0 * 100
    ax1.plot(episodes, scale_pct, alpha=0.2, color='mediumseagreen', linewidth=0.5)
    ax1.plot(episodes, rolling(scale_pct), color='mediumseagreen', linewidth=2,
             label=f'Escala ({WINDOW} ep.)')

    ax1.set_ylabel('Calidad (%)')
    ax1.set_title('Calidad visual del seguimiento')
    ax1.set_ylim(-5, 105)
    ax1.legend()

    # Target fraction
    frac_pct = df['mean_target_fraction'] * 100
    ax2.plot(episodes, frac_pct, alpha=0.2, color='purple', linewidth=0.5)
    ax2.plot(episodes, rolling(frac_pct), color='purple', linewidth=2,
             label=f'Fracción de imagen ({WINDOW} ep.)')
    ax2.axhline(8.0, color='green', linestyle='-.', linewidth=1.2,
                label='Ideal (8%)')
    ax2.axhline(20.0, color='red', linestyle='--', linewidth=1.0,
                label='Límite proximidad (20%)')

    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Fracción de imagen (%)')
    ax2.legend()

    path = output_dir / 'plot_visual_quality.png'
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")
    return path


# ── Plot 4: Safety & Visibility ─────────────────────────────────────

def plot_safety(df, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    episodes = df['episode']

    # Visibility %
    ax1.plot(episodes, df['visibility_pct'], alpha=0.2, color='teal', linewidth=0.5)
    ax1.plot(episodes, rolling(df['visibility_pct']), color='teal', linewidth=2,
             label=f'Visibilidad ({WINDOW} ep.)')
    ax1.axhline(100, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel('Visibilidad (%)')
    ax1.set_title('Seguridad y visibilidad durante el entrenamiento')
    ax1.set_ylim(-5, 105)
    ax1.legend()

    # Proximity violations
    ax2.plot(episodes, df['proximity_violations'], alpha=0.2, color='crimson', linewidth=0.5)
    ax2.plot(episodes, rolling(df['proximity_violations']), color='crimson', linewidth=2,
             label=f'Violaciones proximidad ({WINDOW} ep.)')
    ax2.axhline(0, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Violaciones por episodio')
    ax2.legend()

    path = output_dir / 'plot_safety.png'
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")
    return path


# ── Main ─────────────────────────────────────────────────────────────

def generate_all_plots(csv_path, output_dir, filming_distance=None):
    """Generate all training plots from a CSV file.

    Args:
        csv_path: Path to training_log.csv
        output_dir: Directory to save PNG files
        filming_distance: Optional reference distance for plot_distance

    Returns:
        List of generated file paths
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"  Warning: CSV not found at {csv_path}, skipping plot generation.")
        return []

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        print(f"  Warning: CSV has {len(df)} rows, need at least 2 for plots.")
        return []

    print(f"  Generating plots from {len(df)} episodes...")
    paths = [
        plot_reward(df, output_dir),
        plot_distance(df, output_dir, filming_distance=filming_distance),
        plot_visual_quality(df, output_dir),
        plot_safety(df, output_dir),
    ]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate TFG training plots")
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to training_log.csv')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as CSV)')
    parser.add_argument('--filming-distance', type=float, default=None,
                        help='Reference filming distance for distance plot')
    args = parser.parse_args()

    output_dir = args.output or str(Path(args.csv).parent)
    paths = generate_all_plots(args.csv, output_dir,
                               filming_distance=args.filming_distance)
    if paths:
        print(f"\n  {len(paths)} plots generated in {output_dir}")


if __name__ == '__main__':
    main()
