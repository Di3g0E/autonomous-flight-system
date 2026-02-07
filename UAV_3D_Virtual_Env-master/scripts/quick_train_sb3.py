"""
Quick Training Script: Fast verification that SB3 training works

This script runs a quick training session to verify that the drone can learn
basic hover control. It's designed for fast iteration and testing.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from src.envs.quadrotor_env import quad


def quick_train_and_test():
    """
    Quick training session for verification purposes.
    
    This trains a PPO agent for a short time and evaluates it to verify
    that the system is working correctly.
    """
    
    print("=" * 70)
    print("QUICK TRAINING & VERIFICATION")
    print("=" * 70)
    print("\nThis is a quick training run to verify that:")
    print("  1. The environment works with Stable-Baselines3")
    print("  2. The agent can learn basic hover control")
    print("  3. Training and evaluation pipelines work correctly")
    print("\nNote: This is NOT a full training run!")
    print("=" * 70 + "\n")
    
    # Configuration for quick training
    TOTAL_TIMESTEPS = 50000  # Short training for quick verification
    N_ENVS = 4
    EVAL_EPISODES = 5
    
    print(f"Configuration:")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Evaluation episodes: {EVAL_EPISODES}")
    print("\n" + "-" * 70 + "\n")
    
    # Create environment factory
    def make_env(rank=0):
        def _init():
            env = quad(
                t_step=0.01,
                n=500,  # Shorter episodes for quick training
                direct_control=1,
                T=1
            )
            env.reset(seed=rank)
            return env
        return _init
    
    # Create vectorized training environment
    print("Creating training environments...")
    train_env = make_vec_env(
        make_env,
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_env(rank=1000)()
    
    # Evaluate random policy (baseline)
    print("\n" + "=" * 70)
    print("BASELINE: Random Policy")
    print("=" * 70)
    
    random_rewards = []
    for ep in range(EVAL_EPISODES):
        obs, info = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = eval_env.action_space.sample()
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        random_rewards.append(episode_reward)
        print(f"  Episode {ep+1}: Reward = {episode_reward:.2f}")
    
    baseline_mean = np.mean(random_rewards)
    baseline_std = np.std(random_rewards)
    
    print(f"\nRandom policy performance: {baseline_mean:.2f} ± {baseline_std:.2f}")
    print("=" * 70 + "\n")
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device='auto'
    )
    
    print("\nModel architecture:")
    print(model.policy)
    
    # Train model
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70 + "\n")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,
        progress_bar=True
    )
    
    print("\nTraining completed!")
    
    # Evaluate trained model
    print("\n" + "=" * 70)
    print("EVALUATION: Trained PPO Policy")
    print("=" * 70)
    
    trained_rewards = []
    trained_lengths = []
    success_count = 0
    
    for ep in range(EVAL_EPISODES):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        trained_rewards.append(episode_reward)
        trained_lengths.append(episode_length)
        
        if info.get('solved', 0) == 1:
            success_count += 1
        
        final_pos = obs[0:6:2]
        final_dist = np.linalg.norm(final_pos)
        
        print(f"  Episode {ep+1}: "
              f"Reward = {episode_reward:7.2f}, "
              f"Length = {episode_length:4d}, "
              f"Final dist = {final_dist:.3f}m, "
              f"Solved = {info.get('solved', 0)==1}")
    
    trained_mean = np.mean(trained_rewards)
    trained_std = np.std(trained_rewards)
    success_rate = success_count / EVAL_EPISODES
    
    print(f"\nTrained policy performance: {trained_mean:.2f} ± {trained_std:.2f}")
    print(f"Success rate: {success_rate*100:.1f}% ({success_count}/{EVAL_EPISODES})")
    print("=" * 70 + "\n")
    
    # Compare results
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nRandom policy:  {baseline_mean:7.2f} ± {baseline_std:.2f}")
    print(f"Trained policy: {trained_mean:7.2f} ± {trained_std:.2f}")
    
    improvement = trained_mean - baseline_mean
    improvement_pct = (improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
    
    print(f"\nImprovement: {improvement:+.2f} ({improvement_pct:+.1f}%)")
    
    if improvement > 0:
        print("\n✓ SUCCESS: The agent learned to improve performance!")
    else:
        print("\n✗ WARNING: The agent did not improve. Try longer training.")
    
    print("=" * 70 + "\n")
    
    # Save model
    save_dir = "./models/quick_test"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "ppo_quick_test")
    model.save(model_path)
    
    print(f"Model saved to: {model_path}.zip")
    print("\nTo continue training, run:")
    print(f"  python train_sb3.py --timesteps 500000")
    print("\nTo evaluate this model, run:")
    print(f"  python evaluate_sb3.py {model_path}.zip --episodes 20")
    print("\n" + "=" * 70 + "\n")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return {
        'baseline_mean': baseline_mean,
        'trained_mean': trained_mean,
        'improvement': improvement,
        'success_rate': success_rate,
        'model_path': model_path
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STABLE-BASELINES3 INTEGRATION VERIFICATION")
    print("=" * 70 + "\n")
    
    results = quick_train_and_test()
    
    print("Verification complete!")
    
    if results['improvement'] > 0:
        print("\n✓ The system is working correctly!")
        print("  The agent successfully learned to improve hover control.")
    else:
        print("\n⚠ The agent needs more training time.")
        print("  Run the full training script for better results.")
