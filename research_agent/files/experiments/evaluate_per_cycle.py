"""
Per-Cycle Logical Error Rate Evaluation for Surface Code QEC

This script evaluates the logical error rate on a per-cycle basis,
which is the standard metric for comparing QEC decoders.

Author: Research Agent
Date: 2024-12-22
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from surface_code_qec import SurfaceCodeSimulator, NoiseModel
from mwpm_decoder import MWPMDecoder, SimpleLookupDecoder


def evaluate_per_cycle_error_rate(distance: int, p: float, n_trials: int = 10000,
                                   decoder_type: str = 'mwpm') -> float:
    """
    Evaluate per-cycle logical error rate.

    In each trial:
    1. Start with no errors
    2. Apply one round of noise
    3. Extract syndrome
    4. Apply decoder correction
    5. Check if logical error occurred

    Args:
        distance: Code distance
        p: Physical error rate per qubit per cycle
        n_trials: Number of trials
        decoder_type: 'mwpm' or 'lookup'

    Returns:
        Logical error rate (probability of logical error per cycle)
    """
    sim = SurfaceCodeSimulator(distance=distance)

    if decoder_type == 'lookup' and distance == 3:
        decoder = SimpleLookupDecoder(distance=distance)
    else:
        decoder = MWPMDecoder(distance=distance, p=p)

    n_logical_errors = 0

    for _ in range(n_trials):
        # Reset simulator
        sim.reset()

        # Apply noise
        sim.apply_noise(p, NoiseModel.DEPOLARIZING)

        # Extract syndrome
        syndrome = sim.extract_syndrome()

        # Get correction from decoder
        correction = decoder.decode(syndrome)

        # Apply correction
        sim.apply_correction(correction)

        # Check for logical error
        if sim.has_logical_error():
            n_logical_errors += 1

    return n_logical_errors / n_trials


def evaluate_no_decoder(distance: int, p: float, n_trials: int = 10000) -> float:
    """
    Evaluate error rate with no decoder (just to see uncorrected error rate).
    """
    sim = SurfaceCodeSimulator(distance=distance)
    n_logical_errors = 0

    for _ in range(n_trials):
        sim.reset()
        sim.apply_noise(p, NoiseModel.DEPOLARIZING)
        if sim.has_logical_error():
            n_logical_errors += 1

    return n_logical_errors / n_trials


def run_threshold_analysis():
    """
    Run full threshold analysis for surface codes.
    """
    distances = [3, 5, 7]
    # Use very low error rates to see proper scaling
    error_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    n_trials = 5000

    results = {
        'distances': distances,
        'error_rates': error_rates,
        'mwpm_P_L': {},
        'no_decoder_P_L': {}
    }

    print("=" * 70)
    print("SURFACE CODE THRESHOLD ANALYSIS (Per-Cycle Logical Error Rate)")
    print("=" * 70)

    for d in distances:
        print(f"\nDistance d = {d}:")
        print("-" * 50)
        print(f"{'p':>8}  {'No Decoder':>12}  {'MWPM':>12}  {'Improvement':>12}")
        print("-" * 50)

        for p in error_rates:
            # No decoder
            P_L_none = evaluate_no_decoder(d, p, n_trials)

            # MWPM decoder
            P_L_mwpm = evaluate_per_cycle_error_rate(d, p, n_trials, decoder_type='mwpm')

            results['no_decoder_P_L'][(d, p)] = P_L_none
            results['mwpm_P_L'][(d, p)] = P_L_mwpm

            improvement = (P_L_none - P_L_mwpm) / P_L_none * 100 if P_L_none > 0 else 0

            print(f"{p:>8.3f}  {P_L_none:>12.4f}  {P_L_mwpm:>12.4f}  {improvement:>11.1f}%")

    return results


def plot_threshold_curves(results: Dict, save_path: str = None):
    """
    Plot threshold curves from per-cycle evaluation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    distances = results['distances']
    error_rates = results['error_rates']

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(distances)))

    # Plot 1: P_L vs p (linear)
    ax1 = axes[0]
    for i, d in enumerate(distances):
        p_vals = []
        P_L_vals = []

        for p in error_rates:
            if (d, p) in results['mwpm_P_L']:
                p_vals.append(p)
                P_L_vals.append(results['mwpm_P_L'][(d, p)])

        ax1.plot(p_vals, P_L_vals, 'o-', color=colors[i],
                 label=f'd={d}', linewidth=2, markersize=8)

    ax1.set_xlabel('Physical Error Rate p', fontsize=12)
    ax1.set_ylabel('Logical Error Rate P_L', fontsize=12)
    ax1.set_title('Per-Cycle Logical Error Rate (MWPM Decoder)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: P_L vs p (log scale)
    ax2 = axes[1]
    for i, d in enumerate(distances):
        p_vals = []
        P_L_vals = []

        for p in error_rates:
            if (d, p) in results['mwpm_P_L']:
                P_L = results['mwpm_P_L'][(d, p)]
                if P_L > 0:
                    p_vals.append(p)
                    P_L_vals.append(P_L)

        if p_vals:
            ax2.loglog(p_vals, P_L_vals, 'o-', color=colors[i],
                       label=f'd={d}', linewidth=2, markersize=8)

    # Add reference lines
    p_range = np.array(error_rates)
    ax2.loglog(p_range, p_range, 'k:', alpha=0.3, label='P_L = p')
    ax2.loglog(p_range, p_range**2, 'k--', alpha=0.3, label='P_L = p^2')

    ax2.set_xlabel('Physical Error Rate p', fontsize=12)
    ax2.set_ylabel('Logical Error Rate P_L', fontsize=12)
    ax2.set_title('Per-Cycle Logical Error Rate (Log-Log)', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def main():
    """Main function."""
    output_dir = "/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results"
    os.makedirs(output_dir, exist_ok=True)

    # Run threshold analysis
    results = run_threshold_analysis()

    # Save results
    results_json = {
        'distances': results['distances'],
        'error_rates': results['error_rates'],
        'mwpm_P_L': {f"d{k[0]}_p{k[1]:.4f}": float(v) for k, v in results['mwpm_P_L'].items()},
        'no_decoder_P_L': {f"d{k[0]}_p{k[1]:.4f}": float(v) for k, v in results['no_decoder_P_L'].items()}
    }

    with open(os.path.join(output_dir, 'per_cycle_threshold_analysis.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Plot results
    plot_threshold_curves(results, save_path=os.path.join(output_dir, 'per_cycle_threshold.png'))

    # Estimate threshold
    print("\n" + "=" * 70)
    print("THRESHOLD ESTIMATION")
    print("=" * 70)

    distances = results['distances']
    error_rates = results['error_rates']

    # Find crossing points
    for i in range(len(distances) - 1):
        d1, d2 = distances[i], distances[i + 1]
        print(f"\nLooking for crossing between d={d1} and d={d2}:")

        for j in range(len(error_rates) - 1):
            p1, p2 = error_rates[j], error_rates[j + 1]

            if (d1, p1) in results['mwpm_P_L'] and (d2, p1) in results['mwpm_P_L']:
                if (d1, p2) in results['mwpm_P_L'] and (d2, p2) in results['mwpm_P_L']:
                    P_L_d1_p1 = results['mwpm_P_L'][(d1, p1)]
                    P_L_d2_p1 = results['mwpm_P_L'][(d2, p1)]
                    P_L_d1_p2 = results['mwpm_P_L'][(d1, p2)]
                    P_L_d2_p2 = results['mwpm_P_L'][(d2, p2)]

                    diff1 = P_L_d2_p1 - P_L_d1_p1
                    diff2 = P_L_d2_p2 - P_L_d1_p2

                    if diff1 * diff2 < 0:
                        # Crossing found
                        p_cross = p1 + (p2 - p1) * abs(diff1) / (abs(diff1) + abs(diff2))
                        print(f"  Crossing found at p approximately {p_cross:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
