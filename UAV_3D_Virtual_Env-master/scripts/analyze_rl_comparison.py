#!/usr/bin/env python
"""
Analyze and plot RL comparison results.

Generates publication-quality plots comparing baseline (state-only)
vs depth-augmented PPO controllers for the TFG report.

Usage:
    python scripts/analyze_rl_comparison.py \
        --results-dir ./experiments/rl_comparison \
        --output-dir ./results/rl_analysis
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def load_metrics(experiment_dir):
    """Load training metrics from a single experiment."""
    metrics_file = experiment_dir / 'logs' / 'training_metrics.json'
    summary_file = experiment_dir / 'training_summary.json'
    
    metrics = []
    summary = {}
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    
    return metrics, summary


def plot_learning_curves(baseline_metrics, depth_metrics, output_dir):
    """Plot reward learning curves for both agents."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RL Controller Comparison: State-Only vs State + Depth', 
                 fontsize=14, fontweight='bold')
    
    # Extract data
    def extract(metrics, key):
        return [m['timestep'] for m in metrics], [m[key] for m in metrics]
    
    # --- Mean Reward ---
    ax = axes[0, 0]
    if baseline_metrics:
        ts, vals = extract(baseline_metrics, 'mean_reward')
        ax.plot(ts, vals, label='Baseline (state)', alpha=0.8, color='#2196F3')
    if depth_metrics:
        ts, vals = extract(depth_metrics, 'mean_reward')
        ax.plot(ts, vals, label='Depth (state+depth)', alpha=0.8, color='#FF5722')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Reward (100 ep)')
    ax.set_title('Convergence Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Episode Length ---
    ax = axes[0, 1]
    if baseline_metrics:
        ts, vals = extract(baseline_metrics, 'mean_length')
        ax.plot(ts, vals, label='Baseline', alpha=0.8, color='#2196F3')
    if depth_metrics:
        ts, vals = extract(depth_metrics, 'mean_length')
        ax.plot(ts, vals, label='Depth', alpha=0.8, color='#FF5722')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Episode Length')
    ax.set_title('Episode Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Collision Rate ---
    ax = axes[1, 0]
    if baseline_metrics:
        ts, vals = extract(baseline_metrics, 'collision_rate')
        ax.plot(ts, vals, label='Baseline', alpha=0.8, color='#2196F3')
    if depth_metrics:
        ts, vals = extract(depth_metrics, 'collision_rate')
        ax.plot(ts, vals, label='Depth', alpha=0.8, color='#FF5722')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Collision Rate')
    ax.set_title('Collision Avoidance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Solve Rate ---
    ax = axes[1, 1]
    if baseline_metrics:
        ts, vals = extract(baseline_metrics, 'solve_rate')
        ax.plot(ts, vals, label='Baseline', alpha=0.8, color='#2196F3')
    if depth_metrics:
        ts, vals = extract(depth_metrics, 'solve_rate')
        ax.plot(ts, vals, label='Depth', alpha=0.8, color='#FF5722')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Solve Rate')
    ax.set_title('Task Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ learning_curves.png")


def plot_comparison_bars(baseline_summary, depth_summary, output_dir):
    """Create bar chart comparing final metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Final Performance Comparison', fontsize=14, fontweight='bold')
    
    labels = ['Baseline\n(State-Only)', 'Depth\n(State+Depth)']
    colors = ['#2196F3', '#FF5722']
    
    # Training time
    ax = axes[0]
    times = [
        baseline_summary.get('training_time_seconds', 0),
        depth_summary.get('training_time_seconds', 0)
    ]
    bars = ax.bar(labels, times, color=colors, alpha=0.85, edgecolor='white')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Parameters
    ax = axes[1]
    params = [
        baseline_summary.get('policy_params', 0),
        depth_summary.get('policy_params', 0)
    ]
    bars = ax.bar(labels, params, color=colors, alpha=0.85, edgecolor='white')
    ax.set_ylabel('Parameters')
    ax.set_title('Policy Size')
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Collision rate
    ax = axes[2]
    crash = [
        baseline_summary.get('final_collision_rate', 0),
        depth_summary.get('final_collision_rate', 0)
    ]
    bars = ax.bar(labels, crash, color=colors, alpha=0.85, edgecolor='white')
    ax.set_ylabel('Rate')
    ax.set_title('Final Collision Rate')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, crash):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_bars.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ comparison_bars.png")


def generate_latex_table(baseline_summary, depth_summary, output_dir):
    """Generate a LaTeX table for the TFG report."""
    b = baseline_summary
    d = depth_summary
    
    latex = r"""\begin{table}[h]
\centering
\caption{Comparación de controladores RL: Estado vs Estado + Profundidad}
\label{tab:rl_comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Métrica} & \textbf{Baseline (Estado)} & \textbf{Depth (Estado+Prof.)} \\
\midrule
Observación & Estado 13D & Estado 13D + Depth 64×64 \\
"""
    latex += f"Parámetros de la política & {b.get('policy_params', 0):,} & {d.get('policy_params', 0):,} \\\\\n"
    latex += f"Tiempo de entrenamiento (s) & {b.get('training_time_seconds', 0):.0f} & {d.get('training_time_seconds', 0):.0f} \\\\\n"
    latex += f"Episodios totales & {b.get('final_episodes', 0):,} & {d.get('final_episodes', 0):,} \\\\\n"
    latex += f"Tasa de colisión & {b.get('final_collision_rate', 0):.3f} & {d.get('final_collision_rate', 0):.3f} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'comparison_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"  ✓ comparison_table.tex")


def main():
    parser = argparse.ArgumentParser(description="Analyze RL comparison results")
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory with comparison results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: results-dir/analysis)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Analyzing RL Comparison Results")
    print(f"{'='*70}")
    print(f"Results: {results_dir}")
    print(f"Output: {output_dir}\n")
    
    # Load data
    baseline_metrics, baseline_summary = load_metrics(results_dir / 'baseline')
    depth_metrics, depth_summary = load_metrics(results_dir / 'depth')
    
    print(f"Baseline: {len(baseline_metrics)} data points")
    print(f"Depth: {len(depth_metrics)} data points\n")
    
    # Generate plots
    print("Generating visualizations:")
    plot_learning_curves(baseline_metrics, depth_metrics, output_dir)
    plot_comparison_bars(baseline_summary, depth_summary, output_dir)
    generate_latex_table(baseline_summary, depth_summary, output_dir)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
