#!/usr/bin/env python
"""
Comparative plots for the v9.1.1 main run + ablations.

Reads eval_log.json from each run and produces a 2x2 figure with:
  - mean_steps vs training step  (the headline metric)
  - mean_visibility vs training step
  - mean_jerk vs training step
  - bar chart of peak performance per run

The figure is saved as both PNG (150 dpi for TFG) and PDF (vector).

Default directory layout (current project state on 2026-05-05):
  models/hover_track_v9_1_1/                   ← main run with corrected
                                                 callback that captures
                                                 the @90k peak (steps=3000)
  models/hover_track_v9_1_1_ablate_gamma/      ← gamma=0.99 ablation
  models/hover_track_v9_1_1_ablate_normalize/  ← --no-vec-normalize ablation

Optional historical context (use --include-* flags):
  models/hover_track_v8_1/   baseline (~196 steps ceiling)
  models/hover_track_v9/     target_entropy=-2 collapse (~220 steps)
  models/hover_track_v9_1/   first run that hit @90k peak but lost it

Usage:
    python scripts/plot_v9_2_comparison.py
    python scripts/plot_v9_2_comparison.py --include-baseline --include-history
    python scripts/plot_v9_2_comparison.py --include-all
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="v9.1.1 ablation comparison plots")
    p.add_argument('--main-dir', type=str,
                   default='./models/hover_track_v9_1_1',
                   help="Main run with corrected callback (defaults to "
                        "v9.1.1 in current project naming).")
    p.add_argument('--ablate-gamma-dir', type=str,
                   default='./models/hover_track_v9_1_1_ablate_gamma')
    p.add_argument('--ablate-normalize-dir', type=str,
                   default='./models/hover_track_v9_1_1_ablate_normalize')
    p.add_argument('--include-baseline', action='store_true',
                   help="Add v8.1 baseline (the ~196 steps ceiling).")
    p.add_argument('--include-history', action='store_true',
                   help="Add v9 (target_entropy=-2 collapse) and v9.1 "
                        "(first run that hit peak but lost it).")
    p.add_argument('--include-all', action='store_true',
                   help="Shortcut for --include-baseline --include-history.")
    p.add_argument('--output-dir', type=str,
                   default='./experiments/v9_1_1_comparison')
    p.add_argument('--max-steps', type=int, default=3000,
                   help="Y-axis cap for steps panel (= max_ep_steps).")
    return p.parse_args()


def load_curve(eval_log_path):
    """Load eval_log.json and return arrays (timesteps, steps, vis, jerk)."""
    path = Path(eval_log_path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not data:
        return None
    ts = np.array([e['timestep'] for e in data])
    steps = np.array([e['mean_steps'] for e in data])
    vis = np.array([e['visibility'] for e in data])
    jerk = np.array([e['jerk'] for e in data])
    return ts, steps, vis, jerk


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the convenience --include-all flag.
    if args.include_all:
        args.include_baseline = True
        args.include_history = True

    # Each tuple: (label, color, eval_log_path, linestyle, linewidth)
    # Main run + ablations: drawn solid/dashed in saturated colors.
    # Historical runs: drawn dotted in muted colors so the comparison
    # focuses on v9.1.1 vs its ablations.
    runs = [
        ('v9.1.1 (full: gamma=0.995 + VecNormalize + target_entropy=-1)',
         '#1f77b4',
         Path(args.main_dir) / 'eval_log.json',
         '-', 2.2),
        ('Ablation: gamma=0.99 (no horizon extension)',
         '#ff7f0e',
         Path(args.ablate_gamma_dir) / 'eval_log.json',
         '--', 1.8),
        ('Ablation: --no-vec-normalize (raw 19-D obs to MLP)',
         '#2ca02c',
         Path(args.ablate_normalize_dir) / 'eval_log.json',
         '--', 1.8),
    ]
    if args.include_history:
        runs.append((
            'v9 (target_entropy=-2, collapsed to high-altitude attractor)',
            '#d62728',
            Path('./models/hover_track_v9/eval_log.json'),
            ':', 1.4))
        runs.append((
            'v9.1 (same hyperparams as v9.1.1, lost peak by callback bug)',
            '#9467bd',
            Path('./models/hover_track_v9_1/eval_log.json'),
            ':', 1.4))
    if args.include_baseline:
        runs.append((
            'v8.1 baseline (original ~196-step ceiling)',
            '#7f7f7f',
            Path('./models/hover_track_v8_1/eval_log.json'),
            ':', 1.4))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_steps, ax_vis = axes[0]
    ax_jerk, ax_summary = axes[1]

    summary_rows = []

    for label, color, path, ls, lw in runs:
        curve = load_curve(path)
        if curve is None:
            print(f"  [skip] {label}: no eval_log at {path}")
            continue
        ts, steps, vis, jerk = curve
        ax_steps.plot(ts, steps, label=label, color=color,
                      linestyle=ls, linewidth=lw, marker='o', markersize=4)
        ax_vis.plot(ts, vis * 100, label=label, color=color,
                    linestyle=ls, linewidth=lw, marker='o', markersize=4)
        ax_jerk.plot(ts, jerk, label=label, color=color,
                     linestyle=ls, linewidth=lw, marker='o', markersize=4)
        summary_rows.append({
            'label': label,
            'color': color,
            'best_steps': float(steps.max()),
            'best_steps_at': int(ts[steps.argmax()]),
            'final_steps': float(steps[-1]),
            'best_vis': float(vis.max()),
            'final_vis': float(vis[-1]),
        })

    ax_steps.set_xlabel('Training step')
    ax_steps.set_ylabel('Mean episode length (steps)')
    ax_steps.set_title('Episode length over training\n'
                       f'(survival = {args.max_steps} steps)')
    ax_steps.axhline(args.max_steps, color='gray',
                     linestyle=':', alpha=0.6,
                     label=f'max_ep_steps={args.max_steps}')
    ax_steps.set_ylim(0, args.max_steps * 1.05)
    ax_steps.grid(alpha=0.3)
    ax_steps.legend(fontsize=8, loc='best')

    ax_vis.set_xlabel('Training step')
    ax_vis.set_ylabel('Mean visibility (%)')
    ax_vis.set_title('Visual tracking visibility over training')
    ax_vis.set_ylim(0, 105)
    ax_vis.grid(alpha=0.3)
    ax_vis.legend(fontsize=8, loc='best')

    ax_jerk.set_xlabel('Training step')
    ax_jerk.set_ylabel('Mean action jerk')
    ax_jerk.set_title('Control smoothness over training\n'
                      '(lower is smoother)')
    ax_jerk.grid(alpha=0.3)
    ax_jerk.legend(fontsize=8, loc='best')

    # Summary panel: bar chart of best_steps per run
    ax_summary.axis('off')
    if summary_rows:
        labels = [r['label'].split(' (')[0].split(': ')[-1]
                  for r in summary_rows]
        best = [r['best_steps'] for r in summary_rows]
        colors = [r['color'] for r in summary_rows]
        ax_summary.set_axis_on()
        bars = ax_summary.bar(labels, best, color=colors)
        ax_summary.set_ylabel('Best mean_steps achieved')
        ax_summary.set_title('Peak performance per run')
        ax_summary.axhline(args.max_steps, color='gray',
                           linestyle=':', alpha=0.6)
        ax_summary.set_ylim(0, args.max_steps * 1.05)
        ax_summary.tick_params(axis='x', rotation=20, labelsize=8)
        ax_summary.grid(alpha=0.3, axis='y')
        for bar, val, row in zip(bars, best, summary_rows):
            ax_summary.text(bar.get_x() + bar.get_width() / 2,
                            val + 50,
                            f"{int(val)}\n@{row['best_steps_at']:,}",
                            ha='center', fontsize=8)

    fig.suptitle('v9.2 vs ablations — eval metrics over training',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    png_path = out_dir / 'v9_2_comparison.png'
    pdf_path = out_dir / 'v9_2_comparison.pdf'
    fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
    fig.savefig(str(pdf_path), bbox_inches='tight')

    # Also dump a JSON summary that's easy to cite in the TFG
    json_summary = []
    for r in summary_rows:
        json_summary.append({
            'run': r['label'],
            'best_steps': r['best_steps'],
            'best_steps_at_timestep': r['best_steps_at'],
            'final_steps': r['final_steps'],
            'best_visibility': r['best_vis'],
            'final_visibility': r['final_vis'],
        })
    with open(out_dir / 'comparison_summary.json', 'w') as f:
        json.dump(json_summary, f, indent=2)

    print('\n' + '=' * 70)
    print('  COMPARISON RESULTS')
    print('=' * 70)
    for r in summary_rows:
        print(f"  {r['label']}")
        print(f"    best steps:  {r['best_steps']:.0f}  "
              f"@ step {r['best_steps_at']:,}")
        print(f"    final steps: {r['final_steps']:.0f}")
        print(f"    best vis:    {r['best_vis']:.2%}")
        print()
    print(f"  Saved figure: {png_path}")
    print(f"  Saved figure: {pdf_path}")
    print(f"  Saved summary: {out_dir / 'comparison_summary.json'}")
    print('=' * 70)


if __name__ == '__main__':
    main()
