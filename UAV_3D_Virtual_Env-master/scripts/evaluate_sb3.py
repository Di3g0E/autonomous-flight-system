"""
Evaluation Script: Test trained SB3 models

This script loads a trained Stable-Baselines3 model and evaluates its performance
on the quadrotor hover task.
"""

import os
import numpy as np
from stable_baselines3 import PPO, SAC
from src.envs.quadrotor_env import quad


def evaluate_model(
    model_path,
    n_episodes=10,
    render=False,
    deterministic=True,
    save_trajectory=False
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the saved model (.zip file)
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Use deterministic actions
        save_trajectory: Save trajectory data
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print("=" * 70)
    print("EVALUATING TRAINED MODEL")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print("=" * 70 + "\n")
    
    # Determine algorithm from path
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
        algorithm = "PPO"
    elif 'sac' in model_path.lower():
        model = SAC.load(model_path)
        algorithm = "SAC"
    else:
        raise ValueError("Cannot determine algorithm from model path")
    
    print(f"Loaded {algorithm} model")
    
    # Create environment
    env = quad(t_step=0.01, n=1000, direct_control=1, T=1)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    trajectories = []
    
    for episode in range(n_episodes):
        observation, info = env.reset(seed=episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        if save_trajectory:
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': []
            }
        
        while not done:
            # Get action from model
            action, _states = model.predict(observation, deterministic=deterministic)
            
            if save_trajectory:
                trajectory['observations'].append(observation.copy())
                trajectory['actions'].append(action.copy())
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if save_trajectory:
                trajectory['rewards'].append(reward)
            
            if render:
                env.render()
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get('solved', 0) == 1:
            success_count += 1
        
        if save_trajectory:
            trajectories.append(trajectory)
        
        # Print episode summary
        final_pos = observation[0:6:2]
        final_dist = np.linalg.norm(final_pos)
        
        print(f"Episode {episode+1}/{n_episodes}: "
              f"Reward={episode_reward:7.2f}, "
              f"Length={episode_length:4d}, "
              f"Final dist={final_dist:.3f}m, "
              f"Solved={info.get('solved', 0)==1}")
    
    env.close()
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    success_rate = success_count / n_episodes
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nAlgorithm: {algorithm}")
    print(f"Episodes: {n_episodes}")
    print(f"\nReward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Episode length: {mean_length:.1f} ± {std_length:.1f}")
    print(f"Success rate: {success_rate*100:.1f}% ({success_count}/{n_episodes})")
    print("=" * 70 + "\n")
    
    results = {
        'algorithm': algorithm,
        'n_episodes': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'success_rate': success_rate,
        'success_count': success_count
    }
    
    if save_trajectory:
        results['trajectories'] = trajectories
    
    return results


def compare_models(model_paths, n_episodes=10):
    """
    Compare multiple trained models.
    
    Args:
        model_paths: List of paths to models
        n_episodes: Number of episodes per model
    
    Returns:
        Dictionary with comparison results
    """
    
    print("=" * 70)
    print("COMPARING MULTIPLE MODELS")
    print("=" * 70 + "\n")
    
    all_results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.zip', '')
        print(f"\nEvaluating: {model_name}")
        print("-" * 70)
        
        results = evaluate_model(
            model_path,
            n_episodes=n_episodes,
            deterministic=True
        )
        
        all_results[model_name] = results
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Mean Reward':<15} {'Success Rate':<15}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<30} "
              f"{results['mean_reward']:>7.2f} ± {results['std_reward']:<5.2f} "
              f"{results['success_rate']*100:>6.1f}%")
    
    print("=" * 70 + "\n")
    
    return all_results


def visualize_episode(model_path, seed=42):
    """
    Visualize a single episode with detailed information.
    
    Args:
        model_path: Path to the saved model
        seed: Random seed for reproducibility
    """
    
    print("=" * 70)
    print("VISUALIZING EPISODE")
    print("=" * 70 + "\n")
    
    # Load model
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    elif 'sac' in model_path.lower():
        model = SAC.load(model_path)
    else:
        raise ValueError("Cannot determine algorithm from model path")
    
    # Create environment
    env = quad(t_step=0.01, n=1000, direct_control=1, T=1)
    
    observation, info = env.reset(seed=seed)
    done = False
    step_count = 0
    episode_reward = 0
    
    print(f"Initial position: {observation[0:6:2]}")
    print(f"Initial velocity: {observation[1:6:2]}")
    print("\nStep-by-step execution:")
    print("-" * 70)
    
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        step_count += 1
        done = terminated or truncated
        
        # Print every 50 steps
        if step_count % 50 == 0 or done:
            position = observation[0:6:2]
            velocity = observation[1:6:2]
            dist = np.linalg.norm(position)
            
            print(f"Step {step_count:4d}: "
                  f"Pos=({position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}), "
                  f"Dist={dist:6.3f}m, "
                  f"Reward={reward:7.2f}")
    
    print("-" * 70)
    print(f"\nEpisode finished:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Solved: {info.get('solved', 0) == 1}")
    print(f"  Final distance: {np.linalg.norm(observation[0:6:2]):.3f}m")
    print("=" * 70 + "\n")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained SB3 models")
    parser.add_argument("model_path", type=str,
                       help="Path to trained model (.zip file)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic actions")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize a single episode with details")
    parser.add_argument("--compare", nargs='+', type=str,
                       help="Compare multiple models")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        compare_models(args.compare, n_episodes=args.episodes)
    elif args.visualize:
        # Visualize single episode
        visualize_episode(args.model_path)
    else:
        # Evaluate single model
        evaluate_model(
            args.model_path,
            n_episodes=args.episodes,
            deterministic=args.deterministic
        )
