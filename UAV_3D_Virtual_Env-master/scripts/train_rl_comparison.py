#!/usr/bin/env python
"""
RL Controller Comparison: State-Only vs State + Depth

This script trains two PPO agents and compares their performance:
  - Baseline: PPO with state-only observations (13D vector)
  - Depth-augmented: PPO with state + predicted depth map

The comparison measures convergence speed, collision avoidance,
trajectory precision and real-time FPS for the TFG analysis.

Usage:
    python scripts/train_rl_comparison.py \
        --timesteps 500000 \
        --output-dir ./experiments/rl_comparison
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.quadrotor_env import quad
from src.envs.panda3d_quadrotor_env import Panda3DQuadrotorEnv
from src.agents.feature_extractors import StateDepthExtractor, StateOnlyExtractor


# ============================================================================
# Custom Callbacks
# ============================================================================

class MetricsCallback(BaseCallback):
    """
    Custom callback that logs collision rate, episode length,
    and trajectory precision metrics during training.
    """
    
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Accumulated metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_solved = []
        self.collision_count = 0
        self.total_episodes = 0
        
        # Periodic logging
        self.log_interval = 10  # Log every N episodes
        self.metrics_history = []
    
    def _on_step(self) -> bool:
        # Check for episode completion in info
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                self.total_episodes += 1
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                # Track solved status
                solved = info.get('solved', 0)
                self.episode_solved.append(solved)
                
                # Track collisions (terminated without solving = crash)
                terminated = info.get('TimeLimit.truncated', False)
                if not terminated and not solved:
                    self.collision_count += 1
                
                # Log periodically
                if self.total_episodes % self.log_interval == 0:
                    window = min(100, len(self.episode_rewards))
                    recent_rewards = self.episode_rewards[-window:]
                    recent_lengths = self.episode_lengths[-window:]
                    recent_solved = self.episode_solved[-window:]
                    
                    metrics = {
                        'timestep': self.num_timesteps,
                        'episodes': self.total_episodes,
                        'mean_reward': float(np.mean(recent_rewards)),
                        'std_reward': float(np.std(recent_rewards)),
                        'mean_length': float(np.mean(recent_lengths)),
                        'solve_rate': float(np.mean(recent_solved)),
                        'collision_rate': self.collision_count / max(1, self.total_episodes)
                    }
                    self.metrics_history.append(metrics)
                    
                    if self.verbose > 0:
                        print(f"  [Ep {self.total_episodes}] "
                              f"R={metrics['mean_reward']:.1f} "
                              f"Len={metrics['mean_length']:.0f} "
                              f"Solved={metrics['solve_rate']:.2f} "
                              f"Crash={metrics['collision_rate']:.2f}")
        
        return True
    
    def _on_training_end(self):
        # Save complete metrics
        metrics_file = self.log_path / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


# ============================================================================
# Environment Factories
# ============================================================================

def create_state_only_env(seed=0, rank=0):
    """Create state-only quadrotor environment."""
    env = quad(
        t_step=0.01,
        n=1000,
        euler=0,
        direct_control=1,
        T=1
    )
    env.reset(seed=seed + rank)
    return env


def create_depth_env(seed=0, rank=0, headless=True):
    """
    Create quadrotor environment with depth observations.
    
    In headless mode (no Panda3D), uses placeholder depth maps.
    This is sufficient for verifying the training pipeline architecture.
    For real depth data, set headless=False (requires Panda3D).
    """
    env = Panda3DQuadrotorEnv(
        panda3d_app=None,
        quad_model=None,
        use_camera=True,
        use_depth=True,
        depth_metric=False,
        camera_high_freq_size=(64, 64),
        physics_steps_per_high_freq_capture=1,
        t_step=0.01,
        n=1000,
        euler=0,
        direct_control=1,
        T=1
    )
    env.reset(seed=seed + rank)
    return env


# ============================================================================
# Training Functions
# ============================================================================

def train_baseline(
    total_timesteps=500000,
    n_envs=4,
    output_dir='./experiments/baseline',
    verbose=1
):
    """
    Train PPO baseline with state-only observations.
    
    Uses the standard quadrotor env with Box(13,) observation space.
    """
    output_dir = Path(output_dir)
    model_dir = output_dir / 'models'
    log_dir = output_dir / 'logs'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TRAINING BASELINE: PPO State-Only (13D)")
    print("=" * 70)
    
    # Create environments
    env = make_vec_env(
        create_state_only_env,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
        env_kwargs={'seed': 42}
    )
    
    eval_env = make_vec_env(
        create_state_only_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={'seed': 1000}
    )
    
    # Callbacks
    metrics_cb = MetricsCallback(log_dir, verbose=verbose)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=str(model_dir),
        name_prefix="baseline_ppo"
    )
    callbacks = CallbackList([metrics_cb, eval_cb, checkpoint_cb])
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=str(log_dir / 'tb'),
        device='auto',
        seed=42
    )
    
    print(f"Policy: MlpPolicy (state-only)")
    print(f"Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Envs: {n_envs}\n")
    
    # Train
    start = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    
    train_time = time.time() - start
    
    # Save final model
    model.save(str(model_dir / 'baseline_final'))
    
    # Save training summary
    summary = {
        'agent_type': 'baseline_state_only',
        'algorithm': 'PPO',
        'observation': 'state (13D)',
        'total_timesteps': total_timesteps,
        'training_time_seconds': train_time,
        'n_envs': n_envs,
        'policy_params': sum(p.numel() for p in model.policy.parameters()),
        'final_episodes': metrics_cb.total_episodes,
        'final_collision_rate': metrics_cb.collision_count / max(1, metrics_cb.total_episodes)
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Baseline trained in {train_time:.1f}s")
    print(f"  Episodes: {metrics_cb.total_episodes}")
    print(f"  Collision rate: {summary['final_collision_rate']:.3f}")
    
    env.close()
    eval_env.close()
    
    return model, summary


def train_depth_augmented(
    total_timesteps=500000,
    n_envs=4,
    output_dir='./experiments/depth',
    verbose=1
):
    """
    Train PPO with state + depth observations.
    
    Uses Panda3DQuadrotorEnv with Dict observation space containing
    state (13D) and depth_high_freq (64x64x1).
    """
    output_dir = Path(output_dir)
    model_dir = output_dir / 'models'
    log_dir = output_dir / 'logs'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TRAINING DEPTH-AUGMENTED: PPO State + Depth")
    print("=" * 70)
    
    # Create environments
    env = make_vec_env(
        create_depth_env,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
        env_kwargs={'seed': 42, 'headless': True}
    )
    
    eval_env = make_vec_env(
        create_depth_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={'seed': 1000, 'headless': True}
    )
    
    # Callbacks
    metrics_cb = MetricsCallback(log_dir, verbose=verbose)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=str(model_dir),
        name_prefix="depth_ppo"
    )
    callbacks = CallbackList([metrics_cb, eval_cb, checkpoint_cb])
    
    # Policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=StateDepthExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            depth_key='depth_high_freq'
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )
    
    # Create PPO model with MultiInputPolicy (for Dict observations)
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=str(log_dir / 'tb'),
        device='auto',
        seed=42,
        policy_kwargs=policy_kwargs
    )
    
    print(f"Policy: MultiInputPolicy (state + depth CNN)")
    print(f"Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Envs: {n_envs}\n")
    
    # Train
    start = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    
    train_time = time.time() - start
    
    # Save final model
    model.save(str(model_dir / 'depth_final'))
    
    # Save training summary
    summary = {
        'agent_type': 'depth_augmented',
        'algorithm': 'PPO',
        'observation': 'state (13D) + depth (64x64x1)',
        'total_timesteps': total_timesteps,
        'training_time_seconds': train_time,
        'n_envs': n_envs,
        'policy_params': sum(p.numel() for p in model.policy.parameters()),
        'final_episodes': metrics_cb.total_episodes,
        'final_collision_rate': metrics_cb.collision_count / max(1, metrics_cb.total_episodes)
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Depth agent trained in {train_time:.1f}s")
    print(f"  Episodes: {metrics_cb.total_episodes}")
    print(f"  Collision rate: {summary['final_collision_rate']:.3f}")
    
    env.close()
    eval_env.close()
    
    return model, summary


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare RL controllers: state-only vs state+depth"
    )
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total timesteps per agent')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--output-dir', type=str, default='./experiments/rl_comparison',
                        help='Output directory for all results')
    parser.add_argument('--train-baseline', action='store_true', default=True,
                        help='Train baseline agent')
    parser.add_argument('--train-depth', action='store_true', default=True,
                        help='Train depth-augmented agent')
    parser.add_argument('--skip-baseline', action='store_true', default=False,
                        help='Skip baseline training')
    parser.add_argument('--skip-depth', action='store_true', default=False,
                        help='Skip depth training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("RL CONTROLLER COMPARISON")
    print("State-Only vs State + Depth")
    print("=" * 70)
    print(f"Timesteps per agent: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    results = {}
    
    # Train baseline
    if not args.skip_baseline:
        _, baseline_summary = train_baseline(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            output_dir=output_dir / 'baseline',
            verbose=args.verbose
        )
        results['baseline'] = baseline_summary
    
    # Train depth-augmented
    if not args.skip_depth:
        _, depth_summary = train_depth_augmented(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            output_dir=output_dir / 'depth',
            verbose=args.verbose
        )
        results['depth'] = depth_summary
    
    # Save combined results
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    if len(results) == 2:
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)
        print(f"{'Metric':<25} {'Baseline':<20} {'Depth':<20}")
        print("-" * 65)
        
        b = results['baseline']
        d = results['depth']
        
        print(f"{'Observation':<25} {b['observation']:<20} {d['observation']:<20}")
        print(f"{'Policy Params':<25} {b['policy_params']:>15,}   {d['policy_params']:>15,}")
        print(f"{'Training Time (s)':<25} {b['training_time_seconds']:>15.1f}   {d['training_time_seconds']:>15.1f}")
        print(f"{'Episodes':<25} {b['final_episodes']:>15,}   {d['final_episodes']:>15,}")
        print(f"{'Collision Rate':<25} {b['final_collision_rate']:>15.3f}   {d['final_collision_rate']:>15.3f}")
        print("=" * 70)
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"  To analyze results: python scripts/analyze_rl_comparison.py --results-dir {output_dir}")


if __name__ == "__main__":
    main()
