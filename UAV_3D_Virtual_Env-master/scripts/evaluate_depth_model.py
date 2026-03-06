#!/usr/bin/env python
"""
Evaluate trained depth prediction model and visualize predictions.

Usage:
    python scripts/evaluate_depth_model.py \
        --model-path ./models/depth_v1/best_model.pth \
        --dataset-dir ./data/depth_dataset \
        --output-dir ./results/depth_eval
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.depth_unet import get_model
from src.dataset.depth_visualization import apply_colormap_to_depth

# Import dataset from training script
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from train_depth_model import DepthDataset, compute_depth_metrics


def load_model(model_path, model_type='lightweight', device='cpu'):
    """Load trained model from checkpoint."""
    model = get_model(model_type)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset."""
    all_metrics = {
        'rmse': [],
        'abs_rel': [],
        'delta1': [],
        'delta2': [],
        'delta3': []
    }
    
    with torch.no_grad():
        for rgb, depth in tqdm(dataloader, desc="Evaluating"):
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            # Predict
            pred = model(rgb)
            
            # Compute metrics
            metrics = compute_depth_metrics(pred, depth)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {key: np.mean(vals) for key, vals in all_metrics.items()}
    std_metrics = {f'{key}_std': np.std(vals) for key, vals in all_metrics.items()}
    
    return {**avg_metrics, **std_metrics}


def visualize_predictions(model, dataset, num_samples, output_dir, device):
    """Visualize predictions vs ground truth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        rgb, depth_gt = dataset[idx]
        
        # Predict
        with torch.no_grad():
            rgb_batch = rgb.unsqueeze(0).to(device)
            depth_pred = model(rgb_batch)
            depth_pred = depth_pred.squeeze(0).cpu()
        
        # Convert to numpy
        rgb_np = rgb.permute(1, 2, 0).numpy()  # (H, W, 3)
        depth_gt_np = depth_gt.squeeze(0).numpy()  # (H, W)
        depth_pred_np = depth_pred.squeeze(0).numpy()  # (H, W)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB
        axes[0].imshow(rgb_np)
        axes[0].set_title('RGB Input')
        axes[0].axis('off')
        
        # Ground truth
        gt_colored = apply_colormap_to_depth(depth_gt_np, colormap='turbo')
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth Depth')
        axes[1].axis('off')
        
        # Prediction
        pred_colored = apply_colormap_to_depth(depth_pred_np, colormap='turbo')
        axes[2].imshow(pred_colored)
        axes[2].set_title('Predicted Depth')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'prediction_{i+1:03d}.png', dpi=150)
        plt.close()
        
        print(f"  Saved: prediction_{i+1:03d}.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate depth prediction model")
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='./results/depth_eval',
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate')
    parser.add_argument('--model-type', type=str, default='lightweight',
                        choices=['lightweight', 'standard'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--camera', type=str, default='high_freq',
                        choices=['high_freq', 'low_freq'],
                        help='Which camera data to use')
    parser.add_argument('--num-viz', type=int, default=10,
                        help='Number of predictions to visualize')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"Evaluating Monocular Depth Prediction Model")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, args.model_type, device)
    print(f"  Model parameters: {model.count_parameters():,}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = DepthDataset(args.dataset_dir, split=args.split, camera=args.camera)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  Samples: {len(dataset)}\n")
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, dataloader, device)
    
    print(f"\n{'='*80}")
    print(f"Evaluation Results ({args.split} set)")
    print(f"{'='*80}")
    print(f"RMSE:    {metrics['rmse']:.4f} ± {metrics['rmse_std']:.4f}")
    print(f"AbsRel:  {metrics['abs_rel']:.4f} ± {metrics['abs_rel_std']:.4f}")
    print(f"δ < 1.25:   {metrics['delta1']:.3f} ± {metrics['delta1_std']:.3f}")
    print(f"δ < 1.25²:  {metrics['delta2']:.3f} ± {metrics['delta2_std']:.3f}")
    print(f"δ < 1.25³:  {metrics['delta3']:.3f} ± {metrics['delta3_std']:.3f}")
    print(f"{'='*80}\n")
    
    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")
    
    # Visualize predictions
    print(f"\nVisualizing {args.num_viz} predictions...")
    visualize_predictions(model, dataset, args.num_viz, output_dir, device)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
