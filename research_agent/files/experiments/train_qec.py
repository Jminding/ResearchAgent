"""
Training Pipeline for RL-based Surface Code QEC Decoder

Trains PPO agents across multiple code distances and error rates.
Implements curriculum learning, early stopping, and comprehensive logging.

Author: Research Agent
Date: 2024-12-22
"""

import numpy as np
import torch
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle

# Import local modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from surface_code_qec import SurfaceCodeSimulator, QECEnvironment, NoiseModel
from ppo_agent import PPOAgent
from mwpm_decoder import MWPMDecoder, SimpleLookupDecoder


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    else:
        return obj


class TrainingConfig:
    """Configuration for training."""

    def __init__(self,
                 code_distances: List[int] = [3, 5, 7],
                 error_rates: List[float] = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15],
                 noise_model: str = "depolarizing",
                 gamma_bias: float = 0.0,
                 n_episodes: int = 50000,
                 T_max: int = 50,
                 history_window: int = 3,
                 eval_interval: int = 1000,
                 n_eval_episodes: int = 500,
                 hidden_dims: List[int] = [64, 64],
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 ppo_gamma: float = 0.99,
                 ppo_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 K_epochs: int = 10,
                 batch_size: int = 64,
                 buffer_size: int = 2048,
                 entropy_coef: float = 0.01,
                 save_dir: str = None):

        self.code_distances = code_distances
        self.error_rates = error_rates
        self.noise_model = noise_model
        self.gamma_bias = gamma_bias
        self.n_episodes = n_episodes
        self.T_max = T_max
        self.history_window = history_window
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.hidden_dims = hidden_dims
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.ppo_gamma = ppo_gamma
        self.ppo_lambda = ppo_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.entropy_coef = entropy_coef

        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = f"/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results/qec_rl_{timestamp}"
        else:
            self.save_dir = save_dir

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


class TrainingMetrics:
    """Track training metrics."""

    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.logical_error_rates = []
        self.eval_results = []
        self.training_losses = defaultdict(list)
        self.best_P_L = float('inf')
        self.best_episode = 0

    def add_episode(self, reward: float, length: int, logical_error: bool):
        self.episode_rewards.append(float(reward))
        self.episode_lengths.append(int(length))
        self.logical_error_rates.append(int(logical_error))

    def add_eval(self, episode: int, P_L: float, metrics: Dict):
        # Convert all values to native Python types
        eval_entry = {
            'episode': int(episode),
            'P_L': float(P_L),
        }
        for k, v in metrics.items():
            if isinstance(v, (np.float32, np.float64)):
                eval_entry[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                eval_entry[k] = int(v)
            else:
                eval_entry[k] = v

        self.eval_results.append(eval_entry)
        if P_L < self.best_P_L:
            self.best_P_L = float(P_L)
            self.best_episode = int(episode)

    def add_loss(self, actor_loss: float, critic_loss: float, entropy: float):
        self.training_losses['actor'].append(float(actor_loss))
        self.training_losses['critic'].append(float(critic_loss))
        self.training_losses['entropy'].append(float(entropy))

    def get_recent_stats(self, window: int = 100) -> Dict:
        if len(self.episode_rewards) < window:
            window = len(self.episode_rewards)
        if window == 0:
            return {'avg_reward': 0, 'avg_length': 0, 'error_rate': 0}

        return {
            'avg_reward': float(np.mean(self.episode_rewards[-window:])),
            'avg_length': float(np.mean(self.episode_lengths[-window:])),
            'error_rate': float(np.mean(self.logical_error_rates[-window:]))
        }

    def to_dict(self) -> Dict:
        return {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_lengths': [int(x) for x in self.episode_lengths],
            'logical_error_rates': [int(x) for x in self.logical_error_rates],
            'eval_results': self.eval_results,
            'training_losses': {k: [float(x) for x in v] for k, v in self.training_losses.items()},
            'best_P_L': float(self.best_P_L) if self.best_P_L != float('inf') else None,
            'best_episode': int(self.best_episode)
        }


def evaluate_agent(agent: PPOAgent, env: QECEnvironment,
                   n_episodes: int = 500, deterministic: bool = True) -> Tuple[float, Dict]:
    """
    Evaluate agent performance.

    Args:
        agent: PPO agent to evaluate
        env: QEC environment
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy

    Returns:
        (logical_error_rate, metrics_dict)
    """
    n_logical_errors = 0
    total_steps = 0
    total_reward = 0
    episode_lengths = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_steps = 0
        ep_reward = 0

        while not done:
            action, _, _ = agent.select_action(state, deterministic=deterministic)
            state, reward, done, info = env.step(action)
            ep_steps += 1
            ep_reward += reward

        if info.get('logical_error', False):
            n_logical_errors += 1

        total_steps += ep_steps
        total_reward += ep_reward
        episode_lengths.append(ep_steps)

    P_L = n_logical_errors / n_episodes
    metrics = {
        'n_episodes': int(n_episodes),
        'n_logical_errors': int(n_logical_errors),
        'avg_episode_length': float(np.mean(episode_lengths)),
        'avg_reward': float(total_reward / n_episodes),
        'survival_rate': float(1 - P_L)
    }

    return float(P_L), metrics


def evaluate_mwpm(env: QECEnvironment, n_episodes: int = 500) -> Tuple[float, Dict]:
    """
    Evaluate MWPM decoder performance.

    Args:
        env: QEC environment
        n_episodes: Number of evaluation episodes

    Returns:
        (logical_error_rate, metrics_dict)
    """
    decoder = SimpleLookupDecoder(distance=env.sim.d) if env.sim.d == 3 else MWPMDecoder(distance=env.sim.d, p=env.p)

    n_logical_errors = 0
    total_steps = 0
    episode_lengths = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_steps = 0

        while not done:
            # Get current syndrome (last window)
            syndrome = env.syndrome_history[-1]
            action = decoder.decode_to_action(syndrome.astype(np.int8))
            state, reward, done, info = env.step(action)
            ep_steps += 1

        if info.get('logical_error', False):
            n_logical_errors += 1

        total_steps += ep_steps
        episode_lengths.append(ep_steps)

    P_L = n_logical_errors / n_episodes
    metrics = {
        'n_episodes': int(n_episodes),
        'n_logical_errors': int(n_logical_errors),
        'avg_episode_length': float(np.mean(episode_lengths)),
        'survival_rate': float(1 - P_L)
    }

    return float(P_L), metrics


def train_single_agent(config: TrainingConfig, d: int, p: float,
                       verbose: bool = True) -> Tuple[PPOAgent, TrainingMetrics]:
    """
    Train a single agent for given distance and error rate.

    Args:
        config: Training configuration
        d: Code distance
        p: Physical error rate
        verbose: Print progress

    Returns:
        (trained_agent, training_metrics)
    """
    # Create environment
    noise_model = NoiseModel.DEPOLARIZING if config.noise_model == "depolarizing" else NoiseModel.BIASED
    env = QECEnvironment(
        distance=d,
        p=p,
        noise_model=noise_model,
        gamma=config.gamma_bias,
        T_max=config.T_max,
        history_window=config.history_window
    )

    # Create agent
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.ppo_gamma,
        lambda_gae=config.ppo_lambda,
        eps_clip=config.eps_clip,
        K_epochs=config.K_epochs,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        entropy_coef=config.entropy_coef
    )

    metrics = TrainingMetrics()

    # Training loop
    start_time = time.time()

    for episode in range(1, config.n_episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, log_prob, reward, done, value)

            state = next_state
            episode_reward += reward
            steps += 1

            # Update when buffer is full
            if agent.buffer.is_full():
                update_stats = agent.update()
                metrics.add_loss(
                    update_stats['actor_loss'],
                    update_stats['critic_loss'],
                    update_stats['entropy']
                )

        metrics.add_episode(episode_reward, steps, info.get('logical_error', False))

        # Periodic evaluation
        if episode % config.eval_interval == 0:
            P_L, eval_metrics = evaluate_agent(agent, env, config.n_eval_episodes)
            metrics.add_eval(episode, P_L, eval_metrics)

            if verbose:
                recent = metrics.get_recent_stats(100)
                elapsed = time.time() - start_time
                print(f"  Episode {episode:5d} | P_L: {P_L:.4f} | "
                      f"Avg Reward: {recent['avg_reward']:.3f} | "
                      f"Avg Length: {recent['avg_length']:.1f} | "
                      f"Time: {elapsed:.1f}s")

    return agent, metrics


def train_all_agents(config: TrainingConfig, verbose: bool = True) -> Dict:
    """
    Train agents for all combinations of distance and error rate.

    Args:
        config: Training configuration
        verbose: Print progress

    Returns:
        Results dictionary
    """
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(os.path.join(config.save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(config.save_dir, 'metrics'), exist_ok=True)

    # Save config
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    results = {
        'config': config.to_dict(),
        'training_results': {},
        'rl_P_L': {},
        'mwpm_P_L': {},
        'rl_metrics': {},
        'mwpm_metrics': {}
    }

    total_combinations = len(config.code_distances) * len(config.error_rates)
    current = 0

    for d in config.code_distances:
        for p in config.error_rates:
            current += 1
            if verbose:
                print(f"\n[{current}/{total_combinations}] Training d={d}, p={p:.2f}")
                print("-" * 60)

            # Train RL agent
            agent, metrics = train_single_agent(config, d, p, verbose=verbose)

            # Save agent
            agent_path = os.path.join(config.save_dir, 'models', f'agent_d{d}_p{p:.3f}')
            agent.save(agent_path)

            # Save metrics
            metrics_path = os.path.join(config.save_dir, 'metrics', f'metrics_d{d}_p{p:.3f}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics.to_dict(), f)

            # Final evaluation
            noise_model = NoiseModel.DEPOLARIZING if config.noise_model == "depolarizing" else NoiseModel.BIASED
            env = QECEnvironment(
                distance=d, p=p, noise_model=noise_model,
                gamma=config.gamma_bias, T_max=config.T_max,
                history_window=config.history_window
            )

            final_P_L, final_metrics = evaluate_agent(agent, env, config.n_eval_episodes * 2)
            mwpm_P_L, mwpm_metrics = evaluate_mwpm(env, config.n_eval_episodes * 2)

            results['rl_P_L'][(d, p)] = final_P_L
            results['mwpm_P_L'][(d, p)] = mwpm_P_L
            results['rl_metrics'][(d, p)] = final_metrics
            results['mwpm_metrics'][(d, p)] = mwpm_metrics

            if verbose:
                print(f"  Final RL P_L: {final_P_L:.4f} | MWPM P_L: {mwpm_P_L:.4f}")

            # Store training results
            results['training_results'][(d, p)] = {
                'best_P_L': float(metrics.best_P_L) if metrics.best_P_L != float('inf') else None,
                'best_episode': int(metrics.best_episode),
                'final_P_L': float(final_P_L),
                'mwpm_P_L': float(mwpm_P_L)
            }

    # Convert tuple keys to strings for JSON serialization
    results_json = {
        'config': results['config'],
        'training_results': {f"d{k[0]}_p{k[1]:.3f}": v for k, v in results['training_results'].items()},
        'rl_P_L': {f"d{k[0]}_p{k[1]:.3f}": float(v) for k, v in results['rl_P_L'].items()},
        'mwpm_P_L': {f"d{k[0]}_p{k[1]:.3f}": float(v) for k, v in results['mwpm_P_L'].items()},
        'rl_metrics': {f"d{k[0]}_p{k[1]:.3f}": convert_to_serializable(v) for k, v in results['rl_metrics'].items()},
        'mwpm_metrics': {f"d{k[0]}_p{k[1]:.3f}": convert_to_serializable(v) for k, v in results['mwpm_metrics'].items()}
    }

    # Save results
    with open(os.path.join(config.save_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save raw results with pickle (preserves tuple keys)
    with open(os.path.join(config.save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    if verbose:
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Results saved to: {config.save_dir}")

    return results


def quick_train(verbose: bool = True) -> Dict:
    """
    Quick training with reduced parameters for testing.
    """
    config = TrainingConfig(
        code_distances=[3],
        error_rates=[0.03, 0.05, 0.07],
        n_episodes=5000,
        T_max=30,
        eval_interval=500,
        n_eval_episodes=200,
        hidden_dims=[32, 32],
        buffer_size=1024
    )

    return train_all_agents(config, verbose=verbose)


def full_train(verbose: bool = True) -> Dict:
    """
    Full training with production parameters.
    """
    config = TrainingConfig(
        code_distances=[3, 5, 7],
        error_rates=[0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15],
        n_episodes=50000,
        T_max=50,
        eval_interval=2000,
        n_eval_episodes=1000,
        hidden_dims=[64, 64],
        buffer_size=2048
    )

    return train_all_agents(config, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL decoder for surface code QEC")
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'test'],
                        help='Training mode')
    parser.add_argument('--distances', type=int, nargs='+', default=None,
                        help='Code distances to train')
    parser.add_argument('--error-rates', type=float, nargs='+', default=None,
                        help='Physical error rates')
    parser.add_argument('--n-episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    if args.mode == 'test':
        # Minimal test run
        print("Running minimal test...")
        config = TrainingConfig(
            code_distances=[3],
            error_rates=[0.05],
            n_episodes=500,
            T_max=20,
            eval_interval=100,
            n_eval_episodes=50,
            hidden_dims=[16, 16],
            buffer_size=256
        )
        results = train_all_agents(config, verbose=not args.quiet)

    elif args.mode == 'quick':
        results = quick_train(verbose=not args.quiet)

    elif args.mode == 'full':
        results = full_train(verbose=not args.quiet)

    print("\nTraining completed!")
