#!/usr/bin/env python3
"""
Main Runner for Surface Code QEC RL Experiment

This script runs the complete experiment:
1. Surface code simulation with binary symplectic formalism
2. PPO-based RL agent training for syndrome decoding
3. MWPM baseline comparison
4. Evaluation and threshold estimation
5. Visualization generation

Author: Research Agent
Date: 2024-12-22
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from surface_code_qec import SurfaceCodeSimulator, QECEnvironment, NoiseModel, test_surface_code
from ppo_agent import PPOAgent, test_ppo_agent
from mwpm_decoder import MWPMDecoder, SimpleLookupDecoder, test_mwpm_decoder
from train_qec import TrainingConfig, train_all_agents, evaluate_agent, evaluate_mwpm
from evaluate_qec import (
    load_results,
    plot_logical_error_rate_vs_physical,
    estimate_threshold,
    plot_rl_vs_mwpm_comparison,
    visualize_error_matching_graph,
    visualize_bloch_sphere_trajectory,
    generate_synthetic_results
)


def run_component_tests():
    """Run all component tests."""
    print("\n" + "=" * 70)
    print("COMPONENT TESTS")
    print("=" * 70)

    print("\n[1/3] Testing Surface Code Simulator...")
    test_surface_code()

    print("\n[2/3] Testing PPO Agent...")
    test_ppo_agent()

    print("\n[3/3] Testing MWPM Decoder...")
    test_mwpm_decoder()

    print("\n" + "=" * 70)
    print("ALL COMPONENT TESTS PASSED")
    print("=" * 70)


def run_quick_experiment():
    """Run a quick experiment for testing."""
    print("\n" + "=" * 70)
    print("QUICK EXPERIMENT (Testing Configuration)")
    print("=" * 70)

    config = TrainingConfig(
        code_distances=[3],
        error_rates=[0.03, 0.05, 0.07, 0.09],
        n_episodes=3000,
        T_max=30,
        eval_interval=500,
        n_eval_episodes=200,
        hidden_dims=[32, 32],
        buffer_size=512
    )

    print(f"\nConfiguration:")
    print(f"  Distances: {config.code_distances}")
    print(f"  Error rates: {config.error_rates}")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Save dir: {config.save_dir}")

    results = train_all_agents(config, verbose=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    output_dir = os.path.join(config.save_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    plot_logical_error_rate_vs_physical(
        results, save_path=os.path.join(output_dir, 'P_L_vs_p.png'))
    visualize_error_matching_graph(
        distance=3, p=0.05, save_path=os.path.join(output_dir, 'matching_graph.png'))
    visualize_bloch_sphere_trajectory(
        n_steps=30, p=0.05, save_path=os.path.join(output_dir, 'bloch_trajectory.png'))

    return results


def run_full_experiment():
    """Run the full production experiment."""
    print("\n" + "=" * 70)
    print("FULL EXPERIMENT (Production Configuration)")
    print("=" * 70)

    config = TrainingConfig(
        code_distances=[3, 5, 7],
        error_rates=[0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15],
        n_episodes=30000,
        T_max=50,
        eval_interval=2000,
        n_eval_episodes=500,
        hidden_dims=[64, 64],
        buffer_size=2048
    )

    print(f"\nConfiguration:")
    print(f"  Distances: {config.code_distances}")
    print(f"  Error rates: {config.error_rates}")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Save dir: {config.save_dir}")

    start_time = time.time()
    results = train_all_agents(config, verbose=True)
    total_time = time.time() - start_time

    print(f"\nTotal training time: {total_time/3600:.2f} hours")

    # Generate visualizations
    print("\nGenerating visualizations...")
    output_dir = os.path.join(config.save_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    plot_logical_error_rate_vs_physical(
        results, save_path=os.path.join(output_dir, 'P_L_vs_p.png'))
    threshold = estimate_threshold(
        results, save_path=os.path.join(output_dir, 'threshold_estimation.png'))
    plot_rl_vs_mwpm_comparison(
        results, save_path=os.path.join(output_dir, 'rl_vs_mwpm.png'))
    visualize_error_matching_graph(
        distance=3, p=0.05, save_path=os.path.join(output_dir, 'matching_graph_d3.png'))
    visualize_error_matching_graph(
        distance=5, p=0.05, save_path=os.path.join(output_dir, 'matching_graph_d5.png'))
    visualize_bloch_sphere_trajectory(
        n_steps=50, p=0.05, save_path=os.path.join(output_dir, 'bloch_trajectory.png'))

    # Save summary
    summary = {
        'total_time_hours': total_time / 3600,
        'estimated_threshold': threshold,
        'config': config.to_dict(),
        'final_results': {
            f"d{k[0]}_p{k[1]:.3f}": {
                'rl_P_L': results['rl_P_L'][k],
                'mwpm_P_L': results['mwpm_P_L'][k]
            }
            for k in results['rl_P_L'].keys()
        }
    }

    with open(os.path.join(config.save_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


def run_visualization_only(results_dir: str = None):
    """Generate visualizations from existing results or synthetic data."""
    print("\n" + "=" * 70)
    print("VISUALIZATION MODE")
    print("=" * 70)

    output_dir = "/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results/qec_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    if results_dir and os.path.exists(results_dir):
        print(f"Loading results from: {results_dir}")
        results = load_results(results_dir)
    else:
        print("Generating synthetic results for visualization...")
        results = generate_synthetic_results()

    print("\nGenerating plots...")

    plot_logical_error_rate_vs_physical(
        results, save_path=os.path.join(output_dir, 'P_L_vs_p.png'))

    threshold = estimate_threshold(
        results, save_path=os.path.join(output_dir, 'threshold_estimation.png'))

    plot_rl_vs_mwpm_comparison(
        results, save_path=os.path.join(output_dir, 'rl_vs_mwpm.png'))

    for d in [3, 5]:
        visualize_error_matching_graph(
            distance=d, p=0.05,
            save_path=os.path.join(output_dir, f'matching_graph_d{d}.png'))

    visualize_bloch_sphere_trajectory(
        n_steps=30, p=0.05,
        save_path=os.path.join(output_dir, 'bloch_trajectory.png'))

    print(f"\nVisualizations saved to: {output_dir}")

    return results


def run_mwpm_baseline_only():
    """Evaluate MWPM decoder baseline across all configurations."""
    print("\n" + "=" * 70)
    print("MWPM BASELINE EVALUATION")
    print("=" * 70)

    distances = [3, 5, 7]
    error_rates = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    n_eval = 1000

    results = {'mwpm_P_L': {}, 'mwpm_metrics': {}}

    for d in distances:
        print(f"\nDistance d={d}:")
        for p in error_rates:
            env = QECEnvironment(
                distance=d, p=p, noise_model=NoiseModel.DEPOLARIZING,
                gamma=0.0, T_max=50, history_window=3
            )

            P_L, metrics = evaluate_mwpm(env, n_eval)
            results['mwpm_P_L'][(d, p)] = P_L
            results['mwpm_metrics'][(d, p)] = metrics

            print(f"  p={p:.2f}: P_L = {P_L:.4f}")

    # Save results
    output_dir = "/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'mwpm_baseline_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            f"d{k[0]}_p{k[1]:.3f}": v
            for k, v in results['mwpm_P_L'].items()
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Surface Code QEC RL Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_qec_experiment.py --mode test     # Run component tests
  python run_qec_experiment.py --mode quick    # Quick training (d=3 only)
  python run_qec_experiment.py --mode full     # Full production training
  python run_qec_experiment.py --mode viz      # Generate visualizations
  python run_qec_experiment.py --mode mwpm     # MWPM baseline only
        """
    )

    parser.add_argument('--mode', type=str, default='quick',
                        choices=['test', 'quick', 'full', 'viz', 'mwpm'],
                        help='Execution mode')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory with existing results (for viz mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        print("Warning: PyTorch not available, some features may not work")

    print("\n" + "=" * 70)
    print("SURFACE CODE QEC - RL DECODER EXPERIMENT")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.mode == 'test':
        run_component_tests()
    elif args.mode == 'quick':
        run_quick_experiment()
    elif args.mode == 'full':
        run_full_experiment()
    elif args.mode == 'viz':
        run_visualization_only(args.results_dir)
    elif args.mode == 'mwpm':
        run_mwpm_baseline_only()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
