#!/usr/bin/env python3
"""
Run all QEC experiments for peer review revision.
This script runs experiments in a single process to ensure proper result collection.
"""

import sys
import os

# Add experiment directory to path
sys.path.insert(0, os.path.dirname(__file__))

from qec_simulation import (
    ResultsTable,
    run_extended_training_experiment,
    run_baseline_comparison,
    run_reward_shaping_ablation,
    run_gnn_depth_ablation,
    run_zero_shot_generalization,
    run_mwpm_validation,
    run_learning_curve_analysis
)

def main():
    output_dir = "/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize master results table
    results_table = ResultsTable(project_name="QEC_RL_Scaling_Revision")

    print("=" * 70)
    print("QEC RL Decoder Experiments - Full Peer Review Revision Suite")
    print("=" * 70)
    print()

    # =========================================================================
    # EXPERIMENT 1: Extended Training at d=15
    # Addresses: Undertraining hypothesis
    # =========================================================================
    print("\n[1/7] Extended Training Experiments at d=15")
    print("-" * 50)
    run_extended_training_experiment(
        results_table,
        distances=[15],
        episodes_list=[200, 500, 1000, 2000, 5000],
        seeds=list(range(1, 11)),  # 10 seeds for statistical confidence
        physical_error_rate=0.005,
        eval_samples=1000
    )

    # =========================================================================
    # EXPERIMENT 2: RL vs MWPM Baseline Comparison
    # Addresses: Performance comparison across code distances
    # =========================================================================
    print("\n[2/7] RL vs MWPM Comparison across Code Distances")
    print("-" * 50)
    run_baseline_comparison(
        results_table,
        distances=[3, 5, 7, 9, 11, 13, 15],
        seeds=list(range(1, 6)),  # 5 seeds
        physical_error_rate=0.005,
        training_episodes=2000,
        eval_samples=1000
    )

    # =========================================================================
    # EXPERIMENT 3: Reward Shaping Ablation
    # Addresses: Reviewer request for ablation study
    # =========================================================================
    print("\n[3/7] Reward Shaping Ablation Study")
    print("-" * 50)
    run_reward_shaping_ablation(
        results_table,
        distances=[7, 15],
        reward_types=["sparse", "dense_syndrome", "dense_distance", "shaped_curriculum"],
        seeds=list(range(1, 6)),
        physical_error_rate=0.005,
        training_episodes=2000,
        eval_samples=1000
    )

    # =========================================================================
    # EXPERIMENT 4: GNN Depth Ablation
    # Addresses: Receptive field / architecture ablation
    # =========================================================================
    print("\n[4/7] GNN Architecture Ablation Study")
    print("-" * 50)
    run_gnn_depth_ablation(
        results_table,
        distances=[7, 15],
        layer_configs=[(2, 64), (4, 64), (6, 64), (8, 64), (4, 128), (6, 128)],
        seeds=list(range(1, 4)),  # 3 seeds
        physical_error_rate=0.005,
        training_episodes=2000,
        eval_samples=1000
    )

    # =========================================================================
    # EXPERIMENT 5: Zero-Shot Generalization Retest
    # Addresses: d=7->d=15 transfer with extended training
    # =========================================================================
    print("\n[5/7] Zero-Shot Generalization (d=7 -> d=15)")
    print("-" * 50)
    run_zero_shot_generalization(
        results_table,
        train_distance=7,
        test_distance=15,
        episodes_list=[200, 1000, 2000, 5000],
        seeds=list(range(1, 6)),
        physical_error_rate=0.005,
        eval_samples=1000
    )

    # =========================================================================
    # EXPERIMENT 6: MWPM Validation Against Benchmarks
    # Addresses: Reviewer concern about baseline validity
    # =========================================================================
    print("\n[6/7] MWPM Benchmark Validation")
    print("-" * 50)
    run_mwpm_validation(
        results_table,
        distances=[3, 5, 7, 9, 11, 13, 15],
        error_rates=[0.001, 0.003, 0.005, 0.007, 0.01],
        num_samples=10000
    )

    # =========================================================================
    # EXPERIMENT 7: Learning Curve Analysis at d=15
    # Addresses: Training saturation / convergence analysis
    # =========================================================================
    print("\n[7/7] Learning Curve Analysis at d=15")
    print("-" * 50)
    run_learning_curve_analysis(
        results_table,
        distance=15,
        max_episodes=5000,
        checkpoint_interval=250,
        seeds=list(range(1, 4)),  # 3 seeds
        physical_error_rate=0.005,
        eval_samples=500
    )

    # =========================================================================
    # Save All Results
    # =========================================================================
    json_path = os.path.join(output_dir, "extended_results_table.json")
    csv_path = os.path.join(output_dir, "extended_results_table.csv")

    results_table.to_json(json_path)
    results_table.to_csv(csv_path)

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 70)
    print(f"Total experiments run: {len(results_table.results)}")
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("=" * 70)

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 50)

    # Count by experiment type
    from collections import Counter
    config_types = Counter()
    for r in results_table.results:
        if "extended_" in r.config_name:
            config_types["Extended Training"] += 1
        elif "comparison_" in r.config_name:
            config_types["RL vs MWPM"] += 1
        elif "reward_" in r.config_name:
            config_types["Reward Ablation"] += 1
        elif "gnn_" in r.config_name:
            config_types["GNN Ablation"] += 1
        elif "zeroshot_" in r.config_name:
            config_types["Zero-Shot"] += 1
        elif "mwpm_validation" in r.config_name:
            config_types["MWPM Validation"] += 1
        elif "learning_curve" in r.config_name:
            config_types["Learning Curves"] += 1

    for exp_type, count in config_types.items():
        print(f"  {exp_type}: {count} experiments")

    # Error count
    error_count = sum(1 for r in results_table.results if r.error)
    print(f"\n  Total errors: {error_count}")

    return results_table


if __name__ == "__main__":
    main()
