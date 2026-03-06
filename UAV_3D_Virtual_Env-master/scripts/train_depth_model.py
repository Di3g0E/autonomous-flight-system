#!/usr/bin/env python
"""
Train monocular depth prediction model.

This script trains a U-Net model on collected RGB-Depth dataset.
Optimized for personal computers with limited GPU memory.

Usage:
    python scripts/train_depth_model.py \
        --dataset-dir ./data/depth_dataset \
        --output-dir ./models/depth_v1 \
        --epochs 50 \
        --batch-size 16
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.depth_unet import get_model


class DepthDataset(Dataset):
    """Dataset for RGB-Depth pairs from HDF5 files."""
    
    def __init__(self, dataset_dir, split='train', camera='high_freq', normalize_rgb=True):
        self.dataset_dir = Path(dataset_dir) / split
        self.camera = camera
        self.normalize_rgb = normalize_rgb
        
        # Collect all H5 files
        self.h5_files = sorted(list(self.dataset_dir.glob("*.h5")))
        
        if len(self.h5_files) == 0:
            raise ValueError(f"No HDF5 files found in {self.dataset_dir}")
        
        # Build index: (file_idx, sample_idx)
        self.index = []
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, 'r') as f:
                num_samples = f.attrs['num_samples']
                for sample_idx in range(num_samples):
                    self.index.append((file_idx, sample_idx))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.index[idx]
        h5_file = self.h5_files[file_idx]
        
        with h5py.File(h5_file, 'r') as f:
            rgb = f['rgb'][sample_idx]  # (H, W, 3) uint8
            depth = f['depth'][sample_idx]  # (H, W, 1) float32
        
        # Convert to torch tensors
        rgb = torch.from_numpy(rgb).float()
        depth = torch.from_numpy(depth).float()
        
        # Normalize RGB to [0, 1] if requested
        if self.normalize_rgb:
            rgb = rgb / 255.0
        
        # Transpose to (C, H, W) for PyTorch
        rgb = rgb.permute(2, 0, 1)      # (3, H, W)
        depth = depth.permute(2, 0, 1)  # (1, H, W)
        
        return rgb, depth


def compute_depth_metrics(pred, target, mask=None):
    """
    Compute standard depth estimation metrics.
    
    Args:
        pred: Predicted depth (B, 1, H, W)
        target: Ground truth depth (B, 1, H, W)
        mask: Valid pixel mask (B, 1, H, W), optional
    
    Returns:
        metrics: Dictionary with RMSE, AbsRel, δ1, δ2, δ3
    """
    if mask is None:
        mask = torch.ones_like(target).bool()
    
    # Flatten and filter valid pixels
    pred = pred[mask]
    target = target[mask]
    
    # Avoid division by zero
    eps = 1e-6
    target = torch.clamp(target, min=eps)
    pred = torch.clamp(pred, min=eps)
    
    # RMSE (Root Mean Squared Error)
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    
    # AbsRel (Absolute Relative Error)
    abs_rel = torch.mean(torch.abs(pred - target) / target)
    
    # Threshold accuracy (δ < 1.25, 1.25^2, 1.25^3)
    ratio = torch.max(pred / target, target / pred)
    delta1 = torch.mean((ratio < 1.25).float())
    delta2 = torch.mean((ratio < 1.25 ** 2).float())
    delta3 = torch.mean((ratio < 1.25 ** 3).float())
    
    return {
        'rmse': rmse.item(),
        'abs_rel': abs_rel.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item()
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for rgb, depth in pbar:
        rgb = rgb.to(device)
        depth = depth.to(device)
        
        # Forward
        optimizer.zero_grad()
        pred = model(rgb)
        
        # Loss
        loss = criterion(pred, depth)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_metrics = {
        'rmse': [],
        'abs_rel': [],
        'delta1': [],
        'delta2': [],
        'delta3': []
    }
    
    with torch.no_grad():
        for rgb, depth in tqdm(dataloader, desc="Validating"):
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            # Forward
            pred = model(rgb)
            
            # Loss
            loss = criterion(pred, depth)
            total_loss += loss.item()
            
            # Metrics
            metrics = compute_depth_metrics(pred, depth)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {key: np.mean(vals) for key, vals in all_metrics.items()}
    avg_metrics['loss'] = total_loss / len(dataloader)
    
    return avg_metrics


def plot_training_curves(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # RMSE
    axes[0, 1].plot(history['val_rmse'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Validation RMSE')
    axes[0, 1].grid(True)
    
    # AbsRel
    axes[1, 0].plot(history['val_abs_rel'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AbsRel')
    axes[1, 0].set_title('Validation Absolute Relative Error')
    axes[1, 0].grid(True)
    
    # Delta accuracy
    axes[1, 1].plot(history['val_delta1'], label='δ < 1.25')
    axes[1, 1].plot(history['val_delta2'], label='δ < 1.25²')
    axes[1, 1].plot(history['val_delta3'], label='δ < 1.25³')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Threshold Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train monocular depth prediction model")
    
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='./models/depth_v1',
                        help='Output directory for model and logs')
    parser.add_argument('--model-type', type=str, default='lightweight',
                        choices=['lightweight', 'standard'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--camera', type=str, default='high_freq',
                        choices=['high_freq', 'low_freq'],
                        help='Which camera data to use')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"Training Monocular Depth Prediction Model")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Model: {args.model_type}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*80}\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DepthDataset(args.dataset_dir, split='train', camera=args.camera)
    val_dataset = DepthDataset(args.dataset_dir, split='val', camera=args.camera)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = get_model(args.model_type)
    model = model.to(device)
    print(f"  Parameters: {model.count_parameters():,}\n")
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # MAE loss (better for depth than MSE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_abs_rel': [],
        'val_delta1': [],
        'val_delta2': [],
        'val_delta3': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("Starting training...\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_abs_rel'].append(val_metrics['abs_rel'])
        history['val_delta1'].append(val_metrics['delta1'])
        history['val_delta2'].append(val_metrics['delta2'])
        history['val_delta3'].append(val_metrics['delta3'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"Val AbsRel: {val_metrics['abs_rel']:.4f}")
        print(f"Val δ1: {val_metrics['delta1']:.3f}")
        print()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or val_metrics['loss'] < best_val_loss:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'history': history
            }
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = output_dir / 'best_model.pth'
                print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
            else:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                print(f"  ✓ Checkpoint saved")
            
            torch.save(checkpoint, checkpoint_path)
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'history': history
    }
    torch.save(final_checkpoint, output_dir / 'final_model.pth')
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
