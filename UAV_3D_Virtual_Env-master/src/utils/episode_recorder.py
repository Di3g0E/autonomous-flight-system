"""
Episode recorder for generating training progress videos.

Captures frames from Panda3D's cameras during RL episodes and compiles
them into MP4 videos. Supports recording both FPV (drone camera) and
bird's-eye (main camera) views side by side.

Usage:
    recorder = EpisodeRecorder(output_dir="./recordings", fps=30)
    recorder.start_episode(episode_num=100)
    # In step loop:
    recorder.capture_frame(fpv_image, bird_image, info_overlay={...})
    recorder.end_episode()
    # At the end:
    recorder.compile_timelapse()
"""

import cv2
import numpy as np
from pathlib import Path


class EpisodeRecorder:
    """
    Records RL training episodes as video files.
    
    Features:
    - Captures FPV + bird's-eye view side by side
    - Overlays training metrics (episode, reward, distance, etc.)
    - Records periodic episodes during training for timelapse
    - Compiles all recordings into a single progress video
    """
    
    def __init__(self, output_dir, fps=30, resolution=(640, 360)):
        """
        Args:
            output_dir: Directory to save video files
            fps: Frames per second for output video
            resolution: Resolution of each camera panel (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.panel_w, self.panel_h = resolution
        
        # State
        self.is_recording = False
        self.current_writer = None
        self.current_episode = 0
        self.frame_count = 0
        self.episode_files = []
    
    def start_episode(self, episode_num):
        """Start recording a new episode."""
        self.current_episode = episode_num
        self.frame_count = 0
        self.is_recording = True
        
        # Frame size: FPV + bird's-eye side by side
        frame_w = self.panel_w * 2
        frame_h = self.panel_h
        
        video_path = self.output_dir / f"episode_{episode_num:06d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.current_writer = cv2.VideoWriter(
            str(video_path), fourcc, self.fps, (frame_w, frame_h)
        )
        self.episode_files.append(video_path)
    
    def capture_frame(self, fpv_image=None, bird_image=None, info=None):
        """
        Capture one frame with optional info overlay.
        
        Args:
            fpv_image: RGB image from FPV camera (np.array, any size)
            bird_image: RGB image from bird's-eye camera (np.array, any size)
            info: Dict with overlay info (episode, reward, distance, etc.)
        """
        if not self.is_recording or self.current_writer is None:
            return
        
        # Resize/prepare FPV panel
        if fpv_image is not None:
            fpv_panel = cv2.resize(fpv_image, (self.panel_w, self.panel_h))
        else:
            fpv_panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        
        # Resize/prepare bird's-eye panel
        if bird_image is not None:
            bird_panel = cv2.resize(bird_image, (self.panel_w, self.panel_h))
        else:
            bird_panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        
        # Add labels
        cv2.putText(fpv_panel, "FPV Camera", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bird_panel, "Bird's Eye", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add info overlay
        if info:
            y_offset = 50
            for key, value in info.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.2f}"
                else:
                    text = f"{key}: {value}"
                cv2.putText(bird_panel, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                y_offset += 20
        
        # Combine side by side
        frame = np.hstack([fpv_panel, bird_panel])
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.current_writer.write(frame_bgr)
        self.frame_count += 1
    
    def end_episode(self):
        """Stop recording the current episode."""
        if self.current_writer is not None:
            self.current_writer.release()
            self.current_writer = None
        self.is_recording = False
    
    def compile_timelapse(self, output_name="training_timelapse.mp4", max_frames_per_ep=100):
        """
        Compile all recorded episodes into a single timelapse video.
        
        Args:
            output_name: Output filename
            max_frames_per_ep: Max frames to include per episode (to keep video manageable)
        """
        output_path = self.output_dir / output_name
        
        if not self.episode_files:
            print("No episodes recorded. Nothing to compile.")
            return None
        
        # Get frame size from first video
        cap = cv2.VideoCapture(str(self.episode_files[0]))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))
        
        for ep_file in self.episode_files:
            if not ep_file.exists():
                continue
            
            cap = cv2.VideoCapture(str(ep_file))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames if episode is too long
            if total_frames > max_frames_per_ep:
                step = total_frames // max_frames_per_ep
            else:
                step = 1
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    writer.write(frame)
                frame_idx += 1
            
            cap.release()
        
        writer.release()
        print(f"Timelapse saved to {output_path} ({len(self.episode_files)} episodes)")
        return output_path
    
    def cleanup_episodes(self):
        """Delete individual episode files (after compiling timelapse)."""
        for f in self.episode_files:
            if f.exists():
                f.unlink()
        self.episode_files.clear()
