import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

def generate_report(metrics_path, output_dir):
    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found.")
        return

    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    history = data.get('history', [])
    if not history:
        print("No history found in metrics file.")
        return

    # Extract columns
    timesteps = [h.get('timestep', 0) for h in history]
    rewards = [h.get('reward', 0) for h in history]
    distances = [h.get('distance_to_target', np.nan) for h in history]
    centering = [h.get('centering_reward', 0) for h in history]
    scale = [h.get('scale_reward', 0) for h in history]
    visible = [1.0 if h.get('target_visible', False) else 0.0 for h in history]

    # Save CSV manually to avoid pandas dependency
    csv_path = output_dir / 'training_log.csv'
    with open(csv_path, 'w') as f:
        f.write("timestep,reward,distance,centering,scale,visible\n")
        for i in range(len(timesteps)):
            f.write(f"{timesteps[i]},{rewards[i]},{distances[i]},{centering[i]},{scale[i]},{visible[i]}\n")
    print(f"Saved {csv_path}")

    # Plotting
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Resumen de Entrenamiento - Seguimiento Visual (TFG)', fontsize=16)

    # Reward
    axes[0, 0].plot(timesteps, rewards, label='Reward', alpha=0.7)
    axes[0, 0].set_title('Recompensa por Paso')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].grid(True)

    # Distance
    valid_dist = [d for d in distances if not np.isnan(d)]
    if valid_dist:
        valid_ts = [timesteps[i] for i, d in enumerate(distances) if not np.isnan(d)]
        axes[0, 1].plot(valid_ts, valid_dist, color='orange', label='Distance')
        axes[0, 1].set_title('Distancia al Objetivo (m)')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].grid(True)

    # Visual Tracking Metrics
    axes[1, 0].plot(timesteps, centering, color='green', label='Centering', alpha=0.6)
    axes[1, 0].plot(timesteps, scale, color='blue', label='Scale', alpha=0.6)
    axes[1, 0].set_title('Calidad de Seguimiento Visual')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Target Visibility (Smoothed)
    if len(visible) > 100:
        win = 100
        viz_smooth = np.convolve(visible, np.ones(win)/win, mode='valid')
        ts_smooth = timesteps[win-1:]
        axes[1, 1].plot(ts_smooth, viz_smooth, color='red', label='Visibility Rate')
    else:
        axes[1, 1].plot(timesteps, visible, color='red', label='Visibility')
        
    axes[1, 1].set_title('Tasa de Visibilidad (Media móvil)')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = output_dir / 'training_plots.png'
    plt.savefig(plot_path)
    print(f"Saved {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, required=True, help="Path to training_metrics.json")
    parser.add_argument('--output', type=str, required=True, help="Directory to save CSV and PNG")
    args = parser.parse_args()
    generate_report(args.metrics, args.output)
