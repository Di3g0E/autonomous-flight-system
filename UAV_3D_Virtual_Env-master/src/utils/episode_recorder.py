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
        Capture one frame with filming-mode overlays.

        Args:
            fpv_image: RGB image from FPV camera (np.array, any size)
            bird_image: RGB image from bird's-eye camera (np.array, any size)
            info: Dict with full step info including 'visual_tracking' and 'target' sub-dicts,
                  plus top-level keys like 'Chunk', 'Step', 'Timestep', 'Reward', 'Distance'.
        """
        if not self.is_recording or self.current_writer is None:
            return

        info = info or {}
        vt = info.get('visual_tracking', {})
        target_info = info.get('target', {})

        # ── FPV Panel (left) — "what the drone sees" ──
        if fpv_image is not None:
            fpv_panel = cv2.resize(fpv_image, (self.panel_w, self.panel_h))
        else:
            fpv_panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)

        # Label
        cv2.putText(fpv_panel, "FPV Camera", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Center crosshair (green)
        ch_x, ch_y = self.panel_w // 2, self.panel_h // 2
        ch_size = 12
        cv2.line(fpv_panel, (ch_x - ch_size, ch_y), (ch_x + ch_size, ch_y), (0, 255, 0), 1)
        cv2.line(fpv_panel, (ch_x, ch_y - ch_size), (ch_x, ch_y + ch_size), (0, 255, 0), 1)

        # Target centroid (red dot)
        if vt.get('target_visible', False) and 'target_center' in vt:
            tc = vt['target_center']
            if fpv_image is not None:
                src_h, src_w = fpv_image.shape[:2]
                px = int(tc[0] / src_w * self.panel_w)
                py = int(tc[1] / src_h * self.panel_h)
                cv2.circle(fpv_panel, (px, py), 6, (255, 0, 0), -1)
                cv2.circle(fpv_panel, (px, py), 6, (255, 255, 255), 1)

        # ── FPV filming status bar (bottom) ──
        frac = vt.get('target_fraction', 0.0)
        centering = vt.get('centering_reward', 0.0)
        visible = vt.get('target_visible', False)

        if not visible:
            status_text = "TARGET LOST"
            status_color = (200, 200, 200)
        elif frac > 0.20:
            status_text = "TOO CLOSE!"
            status_color = (255, 50, 50)
        elif frac < 0.04:
            status_text = "TOO FAR"
            status_color = (255, 255, 50)
        else:
            status_text = "FILMING OK"
            status_color = (50, 255, 50)

        # Dark bar background
        bar_y = self.panel_h - 45
        cv2.rectangle(fpv_panel, (0, bar_y), (self.panel_w, self.panel_h),
                      (0, 0, 0), -1)

        cv2.putText(fpv_panel, status_text, (10, self.panel_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        if visible:
            detail = f"Target: {frac*100:.1f}% | Center: {centering:.1f}/3.0"
            cv2.putText(fpv_panel, detail, (10, self.panel_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # ── Bird's-eye Panel (right) — "external view" ──
        if bird_image is not None:
            bird_panel = cv2.resize(bird_image, (self.panel_w, self.panel_h))
        else:
            bird_panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)

        cv2.putText(bird_panel, "Bird's Eye", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # ── Bird's-eye structured overlay ──
        y = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        green = (0, 255, 0)
        white = (220, 220, 220)
        small = 0.40

        # Section: Training
        cv2.putText(bird_panel, "-- Training --", (10, y), font, 0.45, white, 1)
        y += 20
        for key in ('Chunk', 'Step', 'Timestep'):
            if key in info:
                cv2.putText(bird_panel, f"{key}: {info[key]}", (10, y), font, small, green, 1)
                y += 18

        # Section: Filming Quality
        y += 6
        cv2.putText(bird_panel, "-- Filming Quality --", (10, y), font, 0.45, white, 1)
        y += 20

        reward_val = info.get('Reward', 0.0)
        if isinstance(reward_val, (int, float)):
            cv2.putText(bird_panel, f"Reward: {reward_val:+.1f}", (10, y), font, small, green, 1)
        else:
            cv2.putText(bird_panel, f"Reward: {reward_val}", (10, y), font, small, green, 1)
        y += 18

        dist_val = target_info.get('distance_to_target', info.get('Distance', 0.0))
        cv2.putText(bird_panel, f"Distance: {dist_val:.2f}m", (10, y), font, small, green, 1)
        y += 18

        cv2.putText(bird_panel, f"Centering: {centering:.1f}/3.0", (10, y), font, small, green, 1)
        y += 18

        scale_val = vt.get('scale_reward', 0.0)
        cv2.putText(bird_panel, f"Scale: {scale_val:.1f}/2.0", (10, y), font, small, green, 1)
        y += 18

        vis_text = "YES" if visible else "NO"
        vis_color = (0, 255, 0) if visible else (255, 80, 80)
        cv2.putText(bird_panel, f"Visible: {vis_text}", (10, y), font, small, vis_color, 1)

        # ── Combine and write ──
        frame = np.hstack([fpv_panel, bird_panel])
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
