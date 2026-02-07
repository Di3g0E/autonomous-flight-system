"""
Training Script: Stable-Baselines3 PPO for Quadrotor Hover Control

This script trains a PPO agent using Stable-Baselines3 to learn hover control
for the quadrotor. It replaces the manual controller with a standard RL approach.

The goal is to verify that the drone can learn to maintain stable hover position
using modern RL algorithms.
"""

import os
import numpy as np
from datetime import datetime

# Stable-Baselines3 imports
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    StopTrainingOnRewardThreshold,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor

# Custom environment
from src.envs.quadrotor_env import quad


def create_env(rank=0, seed=0):
    """
    Create a single quadrotor environment.
    
    Args:
        rank: Environment rank (for parallel envs)
        seed: Random seed
    
    Returns:
        Function that creates the environment
    """
    def _init():
        env = quad(
            t_step=0.01,      # 10ms integration step
            n=1000,           # Max 1000 steps per episode (10 seconds)
            euler=0,          # Use quaternions
            direct_control=1, # Direct motor control
            T=1               # 1 warm-up step
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo_hover(
    total_timesteps=500000,
    n_envs=4,
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
    save_dir="./models/sb3_ppo",
    log_dir="./logs/sb3_ppo",
    eval_freq=10000,
    eval_episodes=10,
    save_freq=50000,
    verbose=1
):
    """
    Train a PPO agent for quadrotor hover control.
    
    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        save_dir: Directory to save models
        log_dir: Directory for logs
        eval_freq: Evaluation frequency
        eval_episodes: Number of evaluation episodes
        save_freq: Model save frequency
        verbose: Verbosity level
    
    Returns:
        Trained PPO model
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 70)
    print("TRAINING PPO AGENT FOR QUADROTOR HOVER CONTROL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per update: {n_epochs}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE Lambda: {gae_lambda}")
    print(f"  Clip range: {clip_range}")
    print(f"  Entropy coefficient: {ent_coef}")
    print(f"\nSave directory: {save_dir}")
    print(f"Log directory: {log_dir}")
    print("=" * 70 + "\n")
    
    # Create vectorized environment
    print("Creating training environments...")
    env = make_vec_env(
        create_env,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,  # Use DummyVecEnv for simplicity
        env_kwargs={'seed': 0}
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        create_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={'seed': 1000}
    )
    
    # Create callbacks
    print("Setting up callbacks...")
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Adjust for parallel envs
        save_path=save_dir,
        name_prefix="ppo_quadrotor",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq // n_envs,  # Adjust for parallel envs
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback])
    
    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        tensorboard_log=log_dir,
        device='auto'  # Automatically use GPU if available
    )
    
    # Print model architecture
    print("\nModel architecture:")
    print(model.policy)
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_quadrotor_final")
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"\nTraining time: {training_time}")
    print(f"Final model saved to: {final_model_path}.zip")
    print("=" * 70 + "\n")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


def train_sac_hover(
    total_timesteps=500000,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto',
    save_dir="./models/sb3_sac",
    log_dir="./logs/sb3_sac",
    eval_freq=10000,
    eval_episodes=10,
    save_freq=50000,
    verbose=1
):
    """
    Train a SAC agent for quadrotor hover control.
    
    SAC is an off-policy algorithm that can be more sample-efficient than PPO.
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 70)
    print("TRAINING SAC AGENT FOR QUADROTOR HOVER CONTROL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Buffer size: {buffer_size:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gamma: {gamma}")
    print(f"  Tau: {tau}")
    print("=" * 70 + "\n")
    
    # Create environment
    print("Creating training environment...")
    env = create_env(seed=0)()
    env = Monitor(env, log_dir)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_env(seed=1000)()
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="sac_quadrotor"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1
    )
    
    callback = CallbackList([checkpoint_callback, eval_callback])
    
    # Create SAC model
    print("\nCreating SAC model...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
        verbose=verbose,
        tensorboard_log=log_dir,
        device='auto'
    )
    
    print("\nModel architecture:")
    print(model.policy)
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    # Save final model
    final_model_path = os.path.join(save_dir, "sac_quadrotor_final")
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"\nTraining time: {training_time}")
    print(f"Final model saved to: {final_model_path}.zip")
    print("=" * 70 + "\n")
    
    env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agent for quadrotor hover control")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "sac"],
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments (PPO only)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for logs")
    
    args = parser.parse_args()
    
    # Set default directories if not specified
    if args.save_dir is None:
        args.save_dir = f"./models/sb3_{args.algorithm}"
    if args.log_dir is None:
        args.log_dir = f"./logs/sb3_{args.algorithm}"
    
    # Train based on algorithm choice
    if args.algorithm == "ppo":
        model = train_ppo_hover(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
    elif args.algorithm == "sac":
        model = train_sac_hover(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
    
    print("\nTraining completed successfully!")
    print(f"To visualize training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")
