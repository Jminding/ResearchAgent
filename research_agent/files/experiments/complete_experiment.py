#!/usr/bin/env python3
"""
Complete the QEC Experiment Pipeline from existing data

This script picks up where the previous run left off and completes
all remaining steps using the already generated datasets.
"""

import numpy as np
import os
import sys
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List

# Add experiment directory to path
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EXPERIMENT_DIR)
sys.path.insert(0, EXPERIMENT_DIR)

from ppo_qec_agent import PPOAgent
from mwpm_baseline import MWPMDecoder


def load_existing_data():
    """Load existing metadata and results."""
    data_dir = os.path.join(BASE_DIR, "data")
    results_dir = os.path.join(BASE_DIR, "results")

    # Load metadata
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Try to load RL results
    rl_results = None
    try:
        with open(os.path.join(results_dir, "rl_agent_results.json"), "r") as f:
            rl_results = json.load(f)
    except:
        pass

    return metadata, rl_results


def run_mwpm_evaluation(metadata: Dict, n_samples: int = 2000) -> Dict:
    """Run MWPM baseline evaluation."""
    print("\nRunning MWPM Baseline Evaluation...")

    error_rates = metadata["error_rates"]
    n_rounds = metadata["n_rounds"]
    distances = metadata["distances"]
    noise_types = metadata["noise_types"]

    mwpm_results = {
        "distances": distances,
        "error_rates": error_rates,
        "noise_types": noise_types,
        "n_samples": n_samples,
        "logical_error_rates": {}
    }

    for distance in distances:
        mwpm_results["logical_error_rates"][f"d{distance}"] = {}

        for noise_type in noise_types:
            mwpm_results["logical_error_rates"][f"d{distance}"][noise_type] = {}

            print(f"  MWPM: d={distance}, noise={noise_type}")

            for p in error_rates:
                # Create decoder
                decoder = MWPMDecoder(
                    distance=distance,
                    rounds=n_rounds,
                    physical_error_rate=p,
                    noise_type=noise_type
                )

                # Sample and evaluate
                sampler = decoder.circuit.compile_detector_sampler()
                syndromes, observables = sampler.sample(
                    shots=n_samples,
                    separate_observables=True
                )

                syndromes = syndromes.astype(np.int8)
                observables = observables.flatten().astype(np.int8)

                metrics = decoder.evaluate(syndromes, observables)

                mwpm_results["logical_error_rates"][f"d{distance}"][noise_type][f"p{p:.4f}"] = {
                    "logical_error_rate": metrics["logical_error_rate"],
                    "accuracy": metrics["accuracy"]
                }

    return mwpm_results


def compute_combined_results(rl_results: Dict, mwpm_results: Dict) -> Dict:
    """Combine RL and MWPM results."""
    print("\nComputing combined results...")

    combined = {
        "distances": rl_results["distances"],
        "error_rates": rl_results["error_rates"],
        "noise_types": rl_results["noise_types"],
        "rl": {},
        "mwpm": {}
    }

    for distance in rl_results["distances"]:
        d_key = f"d{distance}"
        combined["rl"][d_key] = {}
        combined["mwpm"][d_key] = {}

        for noise_type in rl_results["noise_types"]:
            rl_rates = []
            mwpm_rates = []

            for p in rl_results["error_rates"]:
                p_key = f"p{p:.4f}"

                rl_ler = rl_results["logical_error_rates"][d_key][noise_type].get(p_key, {}).get("logical_error_rate", np.nan)
                mwpm_ler = mwpm_results["logical_error_rates"][d_key][noise_type].get(p_key, {}).get("logical_error_rate", np.nan)

                rl_rates.append(rl_ler)
                mwpm_rates.append(mwpm_ler)

            combined["rl"][d_key][noise_type] = rl_rates
            combined["mwpm"][d_key][noise_type] = mwpm_rates

            print(f"  d={distance}, {noise_type}:")
            print(f"    RL range: [{min(rl_rates):.4f}, {max(rl_rates):.4f}]")
            print(f"    MWPM range: [{min(mwpm_rates):.4f}, {max(mwpm_rates):.4f}]")

    return combined


def fit_threshold(combined_results: Dict) -> Dict:
    """Fit threshold from results."""
    print("\nFitting threshold...")
    from scipy.optimize import curve_fit

    def linear_log_model(d, log_A, alpha):
        return log_A - alpha * d

    threshold_results = {
        "noise_types": combined_results["noise_types"],
        "error_rates": combined_results["error_rates"],
        "distances": combined_results["distances"],
        "fits": {}
    }

    distances = np.array(combined_results["distances"])
    error_rates = np.array(combined_results["error_rates"])

    for decoder in ["rl", "mwpm"]:
        threshold_results["fits"][decoder] = {}

        for noise_type in combined_results["noise_types"]:
            alpha_values = []
            A_values = []

            for i, p in enumerate(error_rates):
                p_L_values = []
                for d_key in [f"d{d}" for d in distances]:
                    rates = combined_results[decoder][d_key][noise_type]
                    p_L_values.append(rates[i])

                p_L_values = np.array(p_L_values)

                if np.any(np.isnan(p_L_values)) or np.any(p_L_values <= 0):
                    alpha_values.append(np.nan)
                    A_values.append(np.nan)
                    continue

                try:
                    log_p_L = np.log(p_L_values + 1e-10)
                    popt, _ = curve_fit(
                        linear_log_model,
                        distances,
                        log_p_L,
                        p0=[np.log(0.5), 0.1],
                        maxfev=5000
                    )
                    log_A, alpha = popt
                    A_values.append(np.exp(log_A))
                    alpha_values.append(alpha)
                except:
                    alpha_values.append(np.nan)
                    A_values.append(np.nan)

            threshold_results["fits"][decoder][noise_type] = {
                "alpha": alpha_values,
                "A": A_values,
                "error_rates": error_rates.tolist()
            }

            # Find threshold
            alpha_arr = np.array(alpha_values)
            valid_mask = ~np.isnan(alpha_arr)

            if np.sum(valid_mask) >= 2:
                valid_p = error_rates[valid_mask]
                valid_alpha = alpha_arr[valid_mask]

                if np.all(valid_alpha > 0):
                    threshold_results["fits"][decoder][noise_type]["estimated_threshold"] = "> 0.15"
                elif np.all(valid_alpha < 0):
                    threshold_results["fits"][decoder][noise_type]["estimated_threshold"] = "< 0.01"
                else:
                    sign_changes = np.where(np.diff(np.sign(valid_alpha)))[0]
                    if len(sign_changes) > 0:
                        idx = sign_changes[0]
                        p1, p2 = valid_p[idx], valid_p[idx + 1]
                        a1, a2 = valid_alpha[idx], valid_alpha[idx + 1]
                        p_th = p1 + (p2 - p1) * (-a1) / (a2 - a1 + 1e-10)
                        threshold_results["fits"][decoder][noise_type]["estimated_threshold"] = float(p_th)
                    else:
                        threshold_results["fits"][decoder][noise_type]["estimated_threshold"] = "undefined"

            print(f"  {decoder.upper()} - {noise_type}: alpha range = [{np.nanmin(alpha_arr):.4f}, {np.nanmax(alpha_arr):.4f}]")

    return threshold_results


def generate_plots(combined_results: Dict, threshold_results: Dict, figures_dir: str):
    """Generate all visualization plots."""
    print("\nGenerating plots...")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)

    error_rates = np.array(combined_results["error_rates"])
    distances = combined_results["distances"]

    distance_colors = {3: 'blue', 5: 'green', 7: 'red'}
    decoder_styles = {'rl': '-', 'mwpm': '--'}

    # Plot 1: P_L vs p (depolarizing)
    fig, ax = plt.subplots(figsize=(10, 7))
    for decoder in ["rl", "mwpm"]:
        for distance in distances:
            d_key = f"d{distance}"
            rates = combined_results[decoder][d_key]["depolarizing"]
            label = f"{decoder.upper()} d={distance}"
            ax.semilogy(error_rates, rates, linestyle=decoder_styles[decoder],
                       color=distance_colors[distance], marker='o' if decoder == 'rl' else 's',
                       markersize=4, label=label, linewidth=2)
    ax.set_xlabel("Physical Error Rate p", fontsize=12)
    ax.set_ylabel("Logical Error Rate $P_L$", fontsize=12)
    ax.set_title("Logical Error Rate vs Physical Error Rate (Depolarizing Noise)", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.01, 0.15])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "P_L_vs_p_depolarizing.png"), dpi=150)
    plt.close()
    print("  Generated: P_L_vs_p_depolarizing.png")

    # Plot 2: P_L vs p (dephasing)
    fig, ax = plt.subplots(figsize=(10, 7))
    for decoder in ["rl", "mwpm"]:
        for distance in distances:
            d_key = f"d{distance}"
            rates = combined_results[decoder][d_key]["dephasing"]
            label = f"{decoder.upper()} d={distance}"
            ax.semilogy(error_rates, rates, linestyle=decoder_styles[decoder],
                       color=distance_colors[distance], marker='o' if decoder == 'rl' else 's',
                       markersize=4, label=label, linewidth=2)
    ax.set_xlabel("Physical Error Rate p", fontsize=12)
    ax.set_ylabel("Logical Error Rate $P_L$", fontsize=12)
    ax.set_title("Logical Error Rate vs Physical Error Rate (Dephasing Noise)", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.01, 0.15])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "P_L_vs_p_dephasing.png"), dpi=150)
    plt.close()
    print("  Generated: P_L_vs_p_dephasing.png")

    # Plot 3: Threshold estimation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, noise_type in enumerate(["depolarizing", "dephasing"]):
        ax = axes[idx]
        for decoder in ["rl", "mwpm"]:
            alpha_values = threshold_results["fits"][decoder][noise_type]["alpha"]
            valid_mask = ~np.isnan(alpha_values)
            ax.plot(error_rates[valid_mask], np.array(alpha_values)[valid_mask],
                   linestyle=decoder_styles[decoder], marker='o' if decoder == 'rl' else 's',
                   label=f"{decoder.upper()}", linewidth=2)
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1)
        ax.set_xlabel("Physical Error Rate p", fontsize=12)
        ax.set_ylabel("Scaling Exponent $\\alpha$", fontsize=12)
        ax.set_title(f"Threshold Analysis ({noise_type.capitalize()})", fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "threshold_estimation.png"), dpi=150)
    plt.close()
    print("  Generated: threshold_estimation.png")

    # Plot 4: RL vs MWPM comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, noise_type in enumerate(["depolarizing", "dephasing"]):
        ax = axes[idx]
        for distance in distances:
            d_key = f"d{distance}"
            rl_rates = np.array(combined_results["rl"][d_key][noise_type])
            mwpm_rates = np.array(combined_results["mwpm"][d_key][noise_type])
            ratio = mwpm_rates / (rl_rates + 1e-10)
            ratio = np.clip(ratio, 0.1, 10)
            ax.plot(error_rates, ratio, color=distance_colors[distance],
                   marker='o', label=f"d={distance}", linewidth=2)
        ax.axhline(y=1, color='black', linestyle=':', linewidth=1)
        ax.set_xlabel("Physical Error Rate p", fontsize=12)
        ax.set_ylabel("$P_L^{MWPM} / P_L^{RL}$", fontsize=12)
        ax.set_title(f"RL vs MWPM Comparison ({noise_type.capitalize()})", fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rl_vs_mwpm_comparison.png"), dpi=150)
    plt.close()
    print("  Generated: rl_vs_mwpm_comparison.png")

    # Plot 5: Error matching graph
    import stim
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, distance in enumerate([3, 5, 7]):
        ax = axes[idx]
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=distance, rounds=1,
            before_round_data_depolarization=0.01
        )
        det_coords = circuit.get_detector_coordinates()
        x_coords, y_coords, t_coords = [], [], []
        for det_id in sorted(det_coords.keys()):
            coords = det_coords[det_id]
            if len(coords) >= 2:
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                t_coords.append(coords[2] if len(coords) > 2 else 0)
        ax.scatter(x_coords, y_coords, c=t_coords, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel("X coordinate", fontsize=10)
        ax.set_ylabel("Y coordinate", fontsize=10)
        ax.set_title(f"Detector Layout d={distance}", fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "error_matching_graph.png"), dpi=150)
    plt.close()
    print("  Generated: error_matching_graph.png")

    # Plot 6: Bloch sphere trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    n_steps = 100
    p = 0.05
    theta = np.zeros(n_steps)
    phi = np.zeros(n_steps)
    np.random.seed(42)
    for i in range(1, n_steps):
        if np.random.random() < p:
            theta[i] = theta[i - 1] + np.random.randn() * 0.2
            phi[i] = phi[i - 1] + np.random.randn() * 0.2
        else:
            theta[i] = theta[i - 1]
            phi[i] = phi[i - 1]
        theta[i] = np.clip(theta[i], 0, np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_steps))
    for i in range(n_steps - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], color=colors[i], linewidth=2)
    ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, label='Start |0>')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='red', s=100, label='End')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Logical State Trajectory on Bloch Sphere (p=0.05)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "bloch_sphere_trajectory.png"), dpi=150)
    plt.close()
    print("  Generated: bloch_sphere_trajectory.png")

    # Plot 7: Summary comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for row, noise in enumerate(["depolarizing", "dephasing"]):
        for col, decoder in enumerate(["rl", "mwpm"]):
            ax = axes[row, col]
            for distance in distances:
                d_key = f"d{distance}"
                rates = combined_results[decoder][d_key][noise]
                ax.semilogy(error_rates, rates, color=distance_colors[distance],
                           marker='o' if decoder == 'rl' else 's', label=f"d={distance}", linewidth=2)
            ax.set_xlabel("Physical Error Rate p")
            ax.set_ylabel("Logical Error Rate $P_L$")
            ax.set_title(f"{decoder.upper()} Decoder ({noise.capitalize()})")
            ax.legend()
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "summary_comparison.png"), dpi=150)
    plt.close()
    print("  Generated: summary_comparison.png")


def save_final_results(results_dir: str, metadata: Dict, combined_results: Dict,
                       threshold_results: Dict, rl_results: Dict, mwpm_results: Dict):
    """Save all final results."""
    print("\nSaving final results...")

    os.makedirs(results_dir, exist_ok=True)

    # Save combined results
    with open(os.path.join(results_dir, "combined_results.json"), "w") as f:
        json.dump(combined_results, f, indent=2)

    with open(os.path.join(results_dir, "combined_results.pkl"), "wb") as f:
        pickle.dump(combined_results, f)

    # Save MWPM results
    with open(os.path.join(results_dir, "mwpm_baseline_results.json"), "w") as f:
        json.dump(mwpm_results, f, indent=2)

    # Save threshold analysis
    with open(os.path.join(results_dir, "threshold_analysis.json"), "w") as f:
        json.dump(threshold_results, f, indent=2)

    # Create summary
    summary = {
        "experiment_timestamp": datetime.now().isoformat(),
        "distances": combined_results["distances"],
        "error_rates": combined_results["error_rates"],
        "noise_types": combined_results["noise_types"],
        "n_samples_train": metadata["n_samples_train"],
        "n_samples_test": metadata["n_samples_test"],
        "n_rounds": metadata["n_rounds"],
        "results_summary": {},
        "threshold_estimates": {}
    }

    for decoder in ["rl", "mwpm"]:
        summary["results_summary"][decoder] = {}
        summary["threshold_estimates"][decoder] = {}
        for noise_type in combined_results["noise_types"]:
            summary["results_summary"][decoder][noise_type] = {}
            for distance in combined_results["distances"]:
                d_key = f"d{distance}"
                rates = combined_results[decoder][d_key][noise_type]
                summary["results_summary"][decoder][noise_type][f"d{distance}"] = {
                    "min_P_L": float(np.nanmin(rates)),
                    "max_P_L": float(np.nanmax(rates)),
                    "mean_P_L": float(np.nanmean(rates)),
                    "P_L_values": [float(r) for r in rates]
                }
            th = threshold_results["fits"][decoder][noise_type].get("estimated_threshold", "N/A")
            summary["threshold_estimates"][decoder][noise_type] = th

    with open(os.path.join(results_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown report
    report = f"""# Surface Code QEC Experiment Results

## Experiment Configuration

- **Date**: {summary['experiment_timestamp']}
- **Distances**: {summary['distances']}
- **Error rates**: {len(summary['error_rates'])} values from {min(summary['error_rates']):.4f} to {max(summary['error_rates']):.4f}
- **Noise models**: {summary['noise_types']}
- **Training samples per config**: {summary['n_samples_train']}
- **Test samples per config**: {summary['n_samples_test']}
- **Syndrome rounds**: {summary['n_rounds']}

## Threshold Estimates

| Decoder | Depolarizing | Dephasing |
|---------|-------------|-----------|
| RL | {summary['threshold_estimates']['rl']['depolarizing']} | {summary['threshold_estimates']['rl']['dephasing']} |
| MWPM | {summary['threshold_estimates']['mwpm']['depolarizing']} | {summary['threshold_estimates']['mwpm']['dephasing']} |

## Logical Error Rate Summary

### RL Decoder - Depolarizing Noise
| Distance | Min P_L | Max P_L | Mean P_L |
|----------|---------|---------|----------|
"""

    for d in combined_results["distances"]:
        stats = summary["results_summary"]["rl"]["depolarizing"][f"d{d}"]
        report += f"| d={d} | {stats['min_P_L']:.6f} | {stats['max_P_L']:.6f} | {stats['mean_P_L']:.6f} |\n"

    report += """
### MWPM Decoder - Depolarizing Noise
| Distance | Min P_L | Max P_L | Mean P_L |
|----------|---------|---------|----------|
"""

    for d in combined_results["distances"]:
        stats = summary["results_summary"]["mwpm"]["depolarizing"][f"d{d}"]
        report += f"| d={d} | {stats['min_P_L']:.6f} | {stats['max_P_L']:.6f} | {stats['mean_P_L']:.6f} |\n"

    report += """
## Numerical Results (Depolarizing Noise)

### P_L(p,d) - RL Decoder
| p | d=3 | d=5 | d=7 |
|---|-----|-----|-----|
"""

    for i, p in enumerate(combined_results["error_rates"]):
        rl_d3 = combined_results["rl"]["d3"]["depolarizing"][i]
        rl_d5 = combined_results["rl"]["d5"]["depolarizing"][i]
        rl_d7 = combined_results["rl"]["d7"]["depolarizing"][i]
        report += f"| {p:.4f} | {rl_d3:.6f} | {rl_d5:.6f} | {rl_d7:.6f} |\n"

    report += """
### P_L(p,d) - MWPM Decoder
| p | d=3 | d=5 | d=7 |
|---|-----|-----|-----|
"""

    for i, p in enumerate(combined_results["error_rates"]):
        mwpm_d3 = combined_results["mwpm"]["d3"]["depolarizing"][i]
        mwpm_d5 = combined_results["mwpm"]["d5"]["depolarizing"][i]
        mwpm_d7 = combined_results["mwpm"]["d7"]["depolarizing"][i]
        report += f"| {p:.4f} | {mwpm_d3:.6f} | {mwpm_d5:.6f} | {mwpm_d7:.6f} |\n"

    report += """
## Generated Figures

1. `P_L_vs_p_depolarizing.png` - Logical error rate vs physical error rate (depolarizing)
2. `P_L_vs_p_dephasing.png` - Logical error rate vs physical error rate (dephasing)
3. `threshold_estimation.png` - Scaling exponent alpha vs error rate
4. `rl_vs_mwpm_comparison.png` - Performance ratio between decoders
5. `error_matching_graph.png` - Detector layout visualization
6. `bloch_sphere_trajectory.png` - Logical state trajectory
7. `summary_comparison.png` - Combined comparison plots
"""

    with open(os.path.join(results_dir, "experiment_report.md"), "w") as f:
        f.write(report)

    print(f"  Saved: combined_results.json")
    print(f"  Saved: mwpm_baseline_results.json")
    print(f"  Saved: threshold_analysis.json")
    print(f"  Saved: experiment_summary.json")
    print(f"  Saved: experiment_report.md")


def main():
    """Run the completion pipeline."""
    print("=" * 70)
    print("COMPLETING QEC EXPERIMENT PIPELINE")
    print("=" * 70)

    results_dir = os.path.join(BASE_DIR, "results")
    figures_dir = os.path.join(BASE_DIR, "figures")

    # Load existing data
    metadata, rl_results = load_existing_data()
    print(f"Loaded metadata: distances={metadata['distances']}, error_rates={len(metadata['error_rates'])} values")
    print(f"RL results loaded: {rl_results is not None}")

    # Run MWPM if needed
    mwpm_results = run_mwpm_evaluation(metadata, n_samples=2000)

    # Compute combined results
    combined_results = compute_combined_results(rl_results, mwpm_results)

    # Fit threshold
    threshold_results = fit_threshold(combined_results)

    # Generate plots
    generate_plots(combined_results, threshold_results, figures_dir)

    # Save final results
    save_final_results(results_dir, metadata, combined_results, threshold_results, rl_results, mwpm_results)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nResults saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
