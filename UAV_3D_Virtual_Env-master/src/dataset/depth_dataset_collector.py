"""
Dataset collection utilities for RGB-Depth pairs.

This module provides tools for collecting and saving paired RGB-Depth data
from the simulation environment for training monocular depth prediction models.
"""

import numpy as np
import h5py
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import cv2


class DepthDatasetCollector:
    """
    Utility class for collecting and saving RGB-Depth paired data.
    
    This collector saves data in HDF5 format for efficient storage and loading.
    Organizes dataset into train/val/test splits automatically.
    """
    
    def __init__(
        self,
        save_dir: str,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        max_samples_per_file: int = 1000,
        camera_type: str = "high_freq"
    ):
        """
        Initialize the dataset collector.
        
        Args:
            save_dir: Root directory to save dataset
            validation_split: Fraction of data for validation (0.0-1.0)
            test_split: Fraction of data for test (0.0-1.0)
            max_samples_per_file: Maximum samples per HDF5 file (for manageable file sizes)
            camera_type: Which camera to collect from ("high_freq" or "low_freq")
        """
        self.save_dir = Path(save_dir)
        self.validation_split = validation_split
        self.test_split = test_split
        self.max_samples_per_file = max_samples_per_file
        self.camera_type = camera_type
        
        # Create directory structure
        self.train_dir = self.save_dir / "train"
        self.val_dir = self.save_dir / "val"
        self.test_dir = self.save_dir / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Sample buffers
        self.rgb_samples = []
        self.depth_samples = []
        self.metadata_samples = []
        
        # Statistics
        self.total_samples = 0
        self.train_samples = 0
        self.val_samples = 0
        self.test_samples = 0
        
        # File counters
        self.train_file_count = 0
        self.val_file_count = 0
        self.test_file_count = 0
    
    def add_sample(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """
        Add a single RGB-Depth pair to the collector.
        
        Args:
            rgb: RGB image, shape (H, W, 3), dtype uint8
            depth: Depth map, shape (H, W, 1), dtype float32
            metadata: Optional metadata dict (e.g., episode_id, step, drone_state)
        """
        # Validate shapes
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"RGB must be (H,W,3), got {rgb.shape}"
        assert depth.ndim == 3 and depth.shape[2] == 1, f"Depth must be (H,W,1), got {depth.shape}"
        assert rgb.shape[:2] == depth.shape[:2], f"RGB and depth must have same (H,W), got RGB {rgb.shape[:2]} vs Depth {depth.shape[:2]}"
        
        # Add to buffers
        self.rgb_samples.append(rgb)
        self.depth_samples.append(depth)
        self.metadata_samples.append(metadata or {})
        
        # Auto-save if buffer is full
        if len(self.rgb_samples) >= self.max_samples_per_file:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Save buffered samples to disk and clear buffer."""
        if len(self.rgb_samples) == 0:
            return
        
        num_samples = len(self.rgb_samples)
        
        # Determine splits
        val_count = int(num_samples * self.validation_split)
        test_count = int(num_samples * self.test_split)
        train_count = num_samples - val_count - test_count
        
        # Create indices array and shuffle
        indices = np.random.permutation(num_samples)
        
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count+val_count]
        test_indices = indices[train_count+val_count:]
        
        # Save each split
        if len(train_indices) > 0:
            self._save_split(train_indices, self.train_dir, self.train_file_count)
            self.train_file_count += 1
            self.train_samples += len(train_indices)
        
        if len(val_indices) > 0:
            self._save_split(val_indices, self.val_dir, self.val_file_count)
            self.val_file_count += 1
            self.val_samples += len(val_indices)
        
        if len(test_indices) > 0:
            self._save_split(test_indices, self.test_dir, self.test_file_count)
            self.test_file_count += 1
            self.test_samples += len(test_indices)
        
        self.total_samples += num_samples
        
        # Clear buffers
        self.rgb_samples = []
        self.depth_samples = []
        self.metadata_samples = []
    
    def _save_split(self, indices: np.ndarray, save_dir: Path, file_index: int):
        """Save a subset of samples to HDF5 file."""
        filename = save_dir / f"data_{file_index:04d}.h5"
        
        with h5py.File(filename, 'w') as f:
            # Stack samples
            rgb_batch = np.stack([self.rgb_samples[i] for i in indices], axis=0)
            depth_batch = np.stack([self.depth_samples[i] for i in indices], axis=0)
            
            # Save datasets
            f.create_dataset('rgb', data=rgb_batch, compression='gzip', compression_opts=4)
            f.create_dataset('depth', data=depth_batch, compression='gzip', compression_opts=4)
            
            # Save metadata as JSON strings
            metadata_list = [self.metadata_samples[i] for i in indices]
            metadata_json = json.dumps(metadata_list)
            f.attrs['metadata'] = metadata_json
            f.attrs['num_samples'] = len(indices)
            f.attrs['camera_type'] = self.camera_type
    
    def finalize(self):
        """Flush remaining samples and save dataset summary."""
        # Save remaining samples
        self._flush_buffer()
        
        # Create dataset summary
        summary = {
            "total_samples": self.total_samples,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "camera_type": self.camera_type,
            "splits": {
                "train": self.validation_split,
                "val": self.test_split,
                "test": 1.0 - self.validation_split - self.test_split
            },
            "created_at": datetime.now().isoformat()
        }
        
        # Save summary as JSON
        summary_path = self.save_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Dataset collection complete!")
        print(f"{'='*60}")
        print(f"Total samples: {self.total_samples}")
        print(f"  Train: {self.train_samples} ({self.train_samples/max(self.total_samples,1)*100:.1f}%)")
        print(f"  Val:   {self.val_samples} ({self.val_samples/max(self.total_samples,1)*100:.1f}%)")
        print(f"  Test:  {self.test_samples} ({self.test_samples/max(self.total_samples,1)*100:.1f}%)")
        print(f"Saved to: {self.save_dir}")
        print(f"{'='*60}\n")
        
        return summary
    
    def get_statistics(self) -> Dict:
        """Return current collection statistics."""
        return {
            "total_collected": self.total_samples + len(self.rgb_samples),
            "buffered": len(self.rgb_samples),
            "saved": {
                "train": self.train_samples,
                "val": self.val_samples,
                "test": self.test_samples
            }
        }


def load_dataset_sample(h5_path: str, index: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a single sample from an HDF5 dataset file.
    
    Args:
        h5_path: Path to HDF5 file
        index: Sample index within the file
    
    Returns:
        (rgb, depth, metadata): RGB image, depth map, and metadata dict
    """
    with h5py.File(h5_path, 'r') as f:
        rgb = f['rgb'][index]
        depth = f['depth'][index]
        
        metadata_json = f.attrs.get('metadata', '[]')
        metadata_list = json.loads(metadata_json)
        metadata = metadata_list[index] if index < len(metadata_list) else {}
    
    return rgb, depth, metadata


if __name__ == "__main__":
    # Example usage
    print("DepthDatasetCollector utility")
    print("Import this module to use the dataset collector")
