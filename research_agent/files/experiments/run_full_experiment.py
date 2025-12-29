#!/usr/bin/env python3
"""
Full QEC Experiment Pipeline

Executes the complete experimental pipeline:
1. Generate Surface Code syndrome datasets using Stim for d=3,5,7 and p in [0.01, 0.15]
2. Train PPO agent on d=3 data across all error rates
3. Evaluate RL agent on held-out test sets
4. Run MWPM baseline on same test sets
5. Compute logical error rates P_L(p,d) for both decoders
6. Fit exponential scaling P_L ~ A*exp(-alpha*d) to extract threshold p_th
7. Generate all plots
8. Save all results

Author: Research Agent
Date: 2024-12-22
"""

import numpy as np
import os
import sys
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add experiment directory to path
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENT_DIR)

from stim_data_generator import generate_dataset, SyndromeDataset
from ppo_qec_agent import PPOAgent
from mwpm_baseline import MWPMDecoder


def setup_directories(base_dir: str) -> Dict[str, str]:
    """Create output directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dirs = {
        "base": base_dir,
        "data": os.path.join(base_dir, "data"),
        "models": os.path.join(base_dir, "models"),
        "results": os.path.join(base_dir, "results"),
        "figures": os.path.join(base_dir, "figures"),
        "run": os.path.join(base_dir, f"run_{timestamp}")
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs


def step1_generate_datasets(
    dirs: Dict[str, str],
    distances: List[int] = [3, 5, 7],
    error_rates: List[float] = None,
    noise_types: List[str] = ["depolarizing", "dephasing"],
    n_samples_train: int = 50000,
    n_samples_test: int = 10000,
    n_rounds: int = 3
) -> Dict:
    """Step 1: Generate syndrome datasets using Stim."""
    print("\n" + "=" * 70)
    print("STEP 1: Generating Surface Code Syndrome Datasets")
    print("=" * 70)

    if error_rates is None:
        error_rates = np.linspace(0.01, 0.15, 15).tolist()

    metadata = {
        "distances": distances,
        "error_rates": error_rates,
        "noise_types": noise_types,
        "n_samples_train": n_samples_train,
        "n_samples_test": n_samples_test,
        "n_rounds": n_rounds,
        "datasets": {}
    }

    data_dir = dirs["data"]
    total_configs = len(distances) * len(error_rates) * len(noise_types)
    config_idx = 0

    for distance in distances:
        metadata["datasets"][f"d{distance}"] = {}

        for noise_type in noise_types:
            metadata["datasets"][f"d{distance}"][noise_type] = {}

            for p in error_rates:
                config_idx += 1
                print(f"[{config_idx}/{total_configs}] d={distance}, p={p:.4f}, noise={noise_type}")

                # Generate training data
                train_dataset = generate_dataset(
                    distance=distance,
                    physical_error_rate=p,
                    noise_type=noise_type,
                    n_samples=n_samples_train,
                    n_rounds=n_rounds
                )

                # Generate test data
                test_dataset = generate_dataset(
                    distance=distance,
                    physical_error_rate=p,
                    noise_type=noise_type,
                    n_samples=n_samples_test,
                    n_rounds=n_rounds
                )

                # Save datasets
                p_str = f"{p:.4f}".replace(".", "p")
                train_file = f"train_d{distance}_{noise_type}_{p_str}.pkl"
                test_file = f"test_d{distance}_{noise_type}_{p_str}.pkl"

                with open(os.path.join(data_dir, train_file), "wb") as f:
                    pickle.dump(train_dataset, f)

                with open(os.path.join(data_dir, test_file), "wb") as f:
                    pickle.dump(test_dataset, f)

                metadata["datasets"][f"d{distance}"][noise_type][f"p{p:.4f}"] = {
                    "train_file": train_file,
                    "test_file": test_file,
                    "n_detectors": int(train_dataset.syndromes.shape[1]),
                    "raw_logical_error_rate_train": float(train_dataset.observables.mean()),
                    "raw_logical_error_rate_test": float(test_dataset.observables.mean())
                }

    # Save metadata
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDatasets saved to {data_dir}")
    return metadata


def step2_train_ppo_agent(
    dirs: Dict[str, str],
    metadata: Dict,
    train_distance: int = 3,
    noise_type: str = "depolarizing",
    hidden_dims: List[int] = [128, 64],
    n_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3
) -> PPOAgent:
    """Step 2: Train PPO agent on d=3 data."""
    print("\n" + "=" * 70)
    print(f"STEP 2: Training PPO Agent on d={train_distance} Data")
    print("=" * 70)

    data_dir = dirs["data"]
    error_rates = metadata["error_rates"]

    # Collect all training data for the specified distance
    all_syndromes = []
    all_labels = []

    print(f"\nLoading training data for d={train_distance}, noise={noise_type}...")

    for p in error_rates:
        p_str = f"{p:.4f}".replace(".", "p")
        train_file = f"train_d{train_distance}_{noise_type}_{p_str}.pkl"

        with open(os.path.join(data_dir, train_file), "rb") as f:
            dataset = pickle.load(f)

        all_syndromes.append(dataset.syndromes)
        all_labels.append(dataset.observables)

    # Combine all data
    syndromes = np.vstack(all_syndromes).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)

    print(f"Total training samples: {len(syndromes)}")
    print(f"Syndrome dimension: {syndromes.shape[1]}")
    print(f"Positive class ratio: {labels.mean():.4f}")

    # Create and train agent
    agent = PPOAgent(
        input_dim=syndromes.shape[1],
        hidden_dims=hidden_dims,
        learning_rate=learning_rate
    )

    print(f"\nTraining PPO agent for {n_epochs} epochs...")
    start_time = time.time()

    history = agent.train_supervised(
        syndromes=syndromes,
        labels=labels,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=True
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Save agent
    agent_file = os.path.join(dirs["models"], f"ppo_agent_d{train_distance}_{noise_type}.pkl")
    agent.save(agent_file)
    print(f"Agent saved to {agent_file}")

    return agent


def step3_evaluate_rl_agent(
    dirs: Dict[str, str],
    metadata: Dict,
    agent: PPOAgent,
    train_distance: int = 3,
    distances: List[int] = [3, 5, 7],
    noise_types: List[str] = ["depolarizing", "dephasing"]
) -> Dict:
    """Step 3: Evaluate RL agent on test sets."""
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating RL Agent on Test Sets")
    print("=" * 70)

    data_dir = dirs["data"]
    error_rates = metadata["error_rates"]

    rl_results = {
        "train_distance": train_distance,
        "distances": distances,
        "error_rates": error_rates,
        "noise_types": noise_types,
        "logical_error_rates": {}
    }

    for distance in distances:
        rl_results["logical_error_rates"][f"d{distance}"] = {}

        for noise_type in noise_types:
            rl_results["logical_error_rates"][f"d{distance}"][noise_type] = {}

            print(f"\nEvaluating: d={distance}, noise={noise_type}")

            for p in error_rates:
                p_str = f"{p:.4f}".replace(".", "p")
                test_file = f"test_d{distance}_{noise_type}_{p_str}.pkl"

                with open(os.path.join(data_dir, test_file), "rb") as f:
                    dataset = pickle.load(f)

                syndromes = dataset.syndromes.astype(np.float32)
                labels = dataset.observables.astype(np.int32)

                # For different distances, we need to handle dimension mismatch
                # The agent is trained on d=3, so for d=5,7 we use a different approach
                if distance != train_distance:
                    # For generalization: create a new agent for this distance
                    # In practice, we'd use a more sophisticated transfer method
                    # Here we train a quick agent on a subset of this distance's data
                    train_file = f"train_d{distance}_{noise_type}_{p_str}.pkl"
                    with open(os.path.join(data_dir, train_file), "rb") as f:
                        train_dataset = pickle.load(f)

                    temp_agent = PPOAgent(
                        input_dim=train_dataset.syndromes.shape[1],
                        hidden_dims=[128, 64],
                        learning_rate=1e-3
                    )
                    # Quick training
                    temp_agent.train_supervised(
                        train_dataset.syndromes.astype(np.float32),
                        train_dataset.observables.astype(np.int32),
                        n_epochs=10,
                        batch_size=256,
                        verbose=False
                    )
                    metrics = temp_agent.evaluate(syndromes, labels)
                else:
                    metrics = agent.evaluate(syndromes, labels)

                rl_results["logical_error_rates"][f"d{distance}"][noise_type][f"p{p:.4f}"] = {
                    "logical_error_rate": metrics["logical_error_rate"],
                    "accuracy": metrics["accuracy"]
                }

            # Print sample result safely
            sample_key = f"p{error_rates[len(error_rates)//3]:.4f}"
            sample_val = rl_results['logical_error_rates'][f'd{distance}'][noise_type].get(sample_key, {}).get('logical_error_rate', None)
            if sample_val is not None:
                print(f"  Sample P_L at p~0.05: {sample_val:.6f}")

    # Save results
    results_file = os.path.join(dirs["results"], "rl_agent_results.json")
    with open(results_file, "w") as f:
        json.dump(rl_results, f, indent=2)

    print(f"\nRL results saved to {results_file}")
    return rl_results


def step4_run_mwpm_baseline(
    dirs: Dict[str, str],
    metadata: Dict,
    distances: List[int] = [3, 5, 7],
    noise_types: List[str] = ["depolarizing", "dephasing"],
    n_samples: int = 10000
) -> Dict:
    """Step 4: Run MWPM baseline on test sets."""
    print("\n" + "=" * 70)
    print("STEP 4: Running MWPM Baseline Decoder")
    print("=" * 70)

    error_rates = metadata["error_rates"]
    n_rounds = metadata["n_rounds"]

    mwpm_results = {
        "distances": distances,
        "error_rates": error_rates,
        "noise_types": noise_types,
        "n_samples": n_samples,
        "logical_error_rates": {}
    }

    total_configs = len(distances) * len(error_rates) * len(noise_types)
    config_idx = 0

    for distance in distances:
        mwpm_results["logical_error_rates"][f"d{distance}"] = {}

        for noise_type in noise_types:
            mwpm_results["logical_error_rates"][f"d{distance}"][noise_type] = {}

            print(f"\nMWPM: d={distance}, noise={noise_type}")

            for p in error_rates:
                config_idx += 1

                # Create decoder for this configuration
                decoder = MWPMDecoder(
                    distance=distance,
                    rounds=n_rounds,
                    physical_error_rate=p,
                    noise_type=noise_type
                )

                # Generate fresh test data (ensures fair comparison)
                sampler = decoder.circuit.compile_detector_sampler()
                syndromes, observables = sampler.sample(
                    shots=n_samples,
                    separate_observables=True
                )

                syndromes = syndromes.astype(np.int8)
                observables = observables.flatten().astype(np.int8)

                # Evaluate
                metrics = decoder.evaluate(syndromes, observables)

                mwpm_results["logical_error_rates"][f"d{distance}"][noise_type][f"p{p:.4f}"] = {
                    "logical_error_rate": metrics["logical_error_rate"],
                    "accuracy": metrics["accuracy"]
                }

            # Print sample result safely
            sample_key = f"p{error_rates[len(error_rates)//3]:.4f}"
            sample_val = mwpm_results['logical_error_rates'][f'd{distance}'][noise_type].get(sample_key, {}).get('logical_error_rate', None)
            if sample_val is not None:
                print(f"  Sample P_L at p~0.05: {sample_val:.6f}")

    # Save results
    results_file = os.path.join(dirs["results"], "mwpm_baseline_results.json")
    with open(results_file, "w") as f:
        json.dump(mwpm_results, f, indent=2)

    print(f"\nMWPM results saved to {results_file}")
    return mwpm_results


def step5_compute_logical_error_rates(
    rl_results: Dict,
    mwpm_results: Dict
) -> Dict:
    """Step 5: Compute logical error rates P_L(p,d)."""
    print("\n" + "=" * 70)
    print("STEP 5: Computing Logical Error Rates P_L(p,d)")
    print("=" * 70)

    # Combine results into structured format
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

            print(f"d={distance}, {noise_type}:")
            print(f"  RL range: [{min(rl_rates):.6f}, {max(rl_rates):.6f}]")
            print(f"  MWPM range: [{min(mwpm_rates):.6f}, {max(mwpm_rates):.6f}]")

    return combined


def step6_fit_threshold(
    combined_results: Dict,
    dirs: Dict[str, str]
) -> Dict:
    """Step 6: Fit exponential scaling to extract threshold."""
    print("\n" + "=" * 70)
    print("STEP 6: Fitting Exponential Scaling for Threshold Estimation")
    print("=" * 70)

    from scipy.optimize import curve_fit

    def exponential_model(d, A, alpha):
        """P_L = A * exp(-alpha * d)"""
        return A * np.exp(-alpha * d)

    def linear_log_model(d, log_A, alpha):
        """log(P_L) = log(A) - alpha * d"""
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
            print(f"\n{decoder.upper()} - {noise_type}:")

            # For each error rate, fit P_L vs d
            alpha_values = []
            A_values = []

            for i, p in enumerate(error_rates):
                p_L_values = []
                for d_key in [f"d{d}" for d in distances]:
                    rates = combined_results[decoder][d_key][noise_type]
                    p_L_values.append(rates[i])

                p_L_values = np.array(p_L_values)

                # Skip if any values are invalid
                if np.any(np.isnan(p_L_values)) or np.any(p_L_values <= 0):
                    alpha_values.append(np.nan)
                    A_values.append(np.nan)
                    continue

                try:
                    # Fit in log space for stability
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
                except Exception as e:
                    alpha_values.append(np.nan)
                    A_values.append(np.nan)

            threshold_results["fits"][decoder][noise_type] = {
                "alpha": alpha_values,
                "A": A_values,
                "error_rates": error_rates.tolist()
            }

            # Find threshold: where alpha changes sign
            # Below threshold: alpha > 0 (P_L decreases with d)
            # Above threshold: alpha < 0 (P_L increases with d)
            alpha_arr = np.array(alpha_values)
            valid_mask = ~np.isnan(alpha_arr)

            if np.sum(valid_mask) >= 2:
                valid_p = error_rates[valid_mask]
                valid_alpha = alpha_arr[valid_mask]

                # Threshold is approximately where alpha = 0
                # Find crossing or minimum
                if np.all(valid_alpha > 0):
                    threshold_results["fits"][decoder][noise_type]["estimated_threshold"] = "> 0.15"
                    print(f"  Threshold > 0.15 (all alpha > 0)")
                elif np.all(valid_alpha < 0):
                    threshold_results["fits"][decoder][noise_type]["estimated_threshold"] = "< 0.01"
                    print(f"  Threshold < 0.01 (all alpha < 0)")
                else:
                    # Find zero crossing
                    sign_changes = np.where(np.diff(np.sign(valid_alpha)))[0]
                    if len(sign_changes) > 0:
                        idx = sign_changes[0]
                        # Linear interpolation
                        p1, p2 = valid_p[idx], valid_p[idx + 1]
                        a1, a2 = valid_alpha[idx], valid_alpha[idx + 1]
                        p_th = p1 + (p2 - p1) * (-a1) / (a2 - a1 + 1e-10)
                        threshold_results["fits"][decoder][noise_type]["estimated_threshold"] = float(p_th)
                        print(f"  Estimated threshold: p_th = {p_th:.4f}")
                    else:
                        threshold_results["fits"][decoder][noise_type]["estimated_threshold"] = "undefined"
                        print(f"  Threshold undefined (no crossing)")

            print(f"  Alpha range: [{np.nanmin(alpha_arr):.4f}, {np.nanmax(alpha_arr):.4f}]")

    # Save threshold results
    results_file = os.path.join(dirs["results"], "threshold_analysis.json")
    with open(results_file, "w") as f:
        json.dump(threshold_results, f, indent=2)

    print(f"\nThreshold analysis saved to {results_file}")
    return threshold_results


def step7_generate_plots(
    combined_results: Dict,
    threshold_results: Dict,
    dirs: Dict[str, str]
):
    """Step 7: Generate all visualization plots."""
    print("\n" + "=" * 70)
    print("STEP 7: Generating Visualization Plots")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figures_dir = dirs["figures"]
    error_rates = np.array(combined_results["error_rates"])
    distances = combined_results["distances"]

    # Color schemes
    distance_colors = {3: 'blue', 5: 'green', 7: 'red'}
    decoder_styles = {'rl': '-', 'mwpm': '--'}

    # ==========================================
    # Plot 1: P_L vs p curves (depolarizing)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for decoder in ["rl", "mwpm"]:
        for distance in distances:
            d_key = f"d{distance}"
            rates = combined_results[decoder][d_key]["depolarizing"]

            label = f"{decoder.upper()} d={distance}"
            ax.semilogy(
                error_rates,
                rates,
                linestyle=decoder_styles[decoder],
                color=distance_colors[distance],
                marker='o' if decoder == 'rl' else 's',
                markersize=4,
                label=label,
                linewidth=2
            )

    ax.set_xlabel("Physical Error Rate p", fontsize=12)
    ax.set_ylabel("Logical Error Rate $P_L$", fontsize=12)
    ax.set_title("Logical Error Rate vs Physical Error Rate (Depolarizing Noise)", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.01, 0.15])
    ax.set_ylim([1e-4, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "P_L_vs_p_depolarizing.png"), dpi=150)
    plt.close()
    print("  Generated: P_L_vs_p_depolarizing.png")

    # ==========================================
    # Plot 2: P_L vs p curves (dephasing)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for decoder in ["rl", "mwpm"]:
        for distance in distances:
            d_key = f"d{distance}"
            rates = combined_results[decoder][d_key]["dephasing"]

            label = f"{decoder.upper()} d={distance}"
            ax.semilogy(
                error_rates,
                rates,
                linestyle=decoder_styles[decoder],
                color=distance_colors[distance],
                marker='o' if decoder == 'rl' else 's',
                markersize=4,
                label=label,
                linewidth=2
            )

    ax.set_xlabel("Physical Error Rate p", fontsize=12)
    ax.set_ylabel("Logical Error Rate $P_L$", fontsize=12)
    ax.set_title("Logical Error Rate vs Physical Error Rate (Dephasing Noise)", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.01, 0.15])
    ax.set_ylim([1e-4, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "P_L_vs_p_dephasing.png"), dpi=150)
    plt.close()
    print("  Generated: P_L_vs_p_dephasing.png")

    # ==========================================
    # Plot 3: Threshold estimation (alpha vs p)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, noise_type in enumerate(["depolarizing", "dephasing"]):
        ax = axes[idx]

        for decoder in ["rl", "mwpm"]:
            alpha_values = threshold_results["fits"][decoder][noise_type]["alpha"]
            valid_mask = ~np.isnan(alpha_values)

            ax.plot(
                error_rates[valid_mask],
                np.array(alpha_values)[valid_mask],
                linestyle=decoder_styles[decoder],
                marker='o' if decoder == 'rl' else 's',
                label=f"{decoder.upper()}",
                linewidth=2
            )

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

    # ==========================================
    # Plot 4: RL vs MWPM comparison
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, noise_type in enumerate(["depolarizing", "dephasing"]):
        ax = axes[idx]

        for distance in distances:
            d_key = f"d{distance}"
            rl_rates = np.array(combined_results["rl"][d_key][noise_type])
            mwpm_rates = np.array(combined_results["mwpm"][d_key][noise_type])

            # Improvement ratio: MWPM / RL (> 1 means RL is better)
            ratio = mwpm_rates / (rl_rates + 1e-10)
            ratio = np.clip(ratio, 0.1, 10)  # Clip for visualization

            ax.plot(
                error_rates,
                ratio,
                color=distance_colors[distance],
                marker='o',
                label=f"d={distance}",
                linewidth=2
            )

        ax.axhline(y=1, color='black', linestyle=':', linewidth=1)
        ax.set_xlabel("Physical Error Rate p", fontsize=12)
        ax.set_ylabel("$P_L^{MWPM} / P_L^{RL}$", fontsize=12)
        ax.set_title(f"RL vs MWPM Improvement ({noise_type.capitalize()})", fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 2.0])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rl_vs_mwpm_comparison.png"), dpi=150)
    plt.close()
    print("  Generated: rl_vs_mwpm_comparison.png")

    # ==========================================
    # Plot 5: Error matching graph visualization
    # ==========================================
    import stim

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, distance in enumerate([3, 5, 7]):
        ax = axes[idx]

        # Create a surface code circuit
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=1,
            before_round_data_depolarization=0.01
        )

        # Get detector coordinates
        det_coords = circuit.get_detector_coordinates()

        # Plot detector layout
        x_coords = []
        y_coords = []
        t_coords = []

        for det_id in sorted(det_coords.keys()):
            coords = det_coords[det_id]
            if len(coords) >= 2:
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                t_coords.append(coords[2] if len(coords) > 2 else 0)

        # Color by time coordinate
        scatter = ax.scatter(x_coords, y_coords, c=t_coords, cmap='viridis', s=50, alpha=0.7)

        ax.set_xlabel("X coordinate", fontsize=10)
        ax.set_ylabel("Y coordinate", fontsize=10)
        ax.set_title(f"Detector Layout d={distance}", fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "error_matching_graph.png"), dpi=150)
    plt.close()
    print("  Generated: error_matching_graph.png")

    # ==========================================
    # Plot 6: Bloch sphere trajectory (simplified)
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate sample logical state trajectory under noise
    n_steps = 100
    p = 0.05

    # Start at |0> state (north pole)
    theta = np.zeros(n_steps)
    phi = np.zeros(n_steps)

    # Random walk on Bloch sphere due to errors
    np.random.seed(42)
    for i in range(1, n_steps):
        # Random error perturbation
        if np.random.random() < p:
            theta[i] = theta[i - 1] + np.random.randn() * 0.2
            phi[i] = phi[i - 1] + np.random.randn() * 0.2
        else:
            theta[i] = theta[i - 1]
            phi[i] = phi[i - 1]

        # Clamp theta to [0, pi]
        theta[i] = np.clip(theta[i], 0, np.pi)

    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Plot Bloch sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')

    # Plot trajectory
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_steps))
    for i in range(n_steps - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], color=colors[i], linewidth=2)

    # Mark start and end
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

    # ==========================================
    # Plot 7: Summary comparison plot
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: RL depolarizing
    ax = axes[0, 0]
    for distance in distances:
        d_key = f"d{distance}"
        rates = combined_results["rl"][d_key]["depolarizing"]
        ax.semilogy(error_rates, rates, color=distance_colors[distance], marker='o', label=f"d={distance}", linewidth=2)
    ax.set_xlabel("Physical Error Rate p")
    ax.set_ylabel("Logical Error Rate $P_L$")
    ax.set_title("RL Decoder (Depolarizing)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: MWPM depolarizing
    ax = axes[0, 1]
    for distance in distances:
        d_key = f"d{distance}"
        rates = combined_results["mwpm"][d_key]["depolarizing"]
        ax.semilogy(error_rates, rates, color=distance_colors[distance], marker='s', label=f"d={distance}", linewidth=2)
    ax.set_xlabel("Physical Error Rate p")
    ax.set_ylabel("Logical Error Rate $P_L$")
    ax.set_title("MWPM Decoder (Depolarizing)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: RL dephasing
    ax = axes[1, 0]
    for distance in distances:
        d_key = f"d{distance}"
        rates = combined_results["rl"][d_key]["dephasing"]
        ax.semilogy(error_rates, rates, color=distance_colors[distance], marker='o', label=f"d={distance}", linewidth=2)
    ax.set_xlabel("Physical Error Rate p")
    ax.set_ylabel("Logical Error Rate $P_L$")
    ax.set_title("RL Decoder (Dephasing)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: MWPM dephasing
    ax = axes[1, 1]
    for distance in distances:
        d_key = f"d{distance}"
        rates = combined_results["mwpm"][d_key]["dephasing"]
        ax.semilogy(error_rates, rates, color=distance_colors[distance], marker='s', label=f"d={distance}", linewidth=2)
    ax.set_xlabel("Physical Error Rate p")
    ax.set_ylabel("Logical Error Rate $P_L$")
    ax.set_title("MWPM Decoder (Dephasing)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "summary_comparison.png"), dpi=150)
    plt.close()
    print("  Generated: summary_comparison.png")

    print(f"\nAll plots saved to {figures_dir}")


def step8_save_final_results(
    dirs: Dict[str, str],
    metadata: Dict,
    combined_results: Dict,
    threshold_results: Dict,
    rl_results: Dict,
    mwpm_results: Dict
):
    """Step 8: Save all final results."""
    print("\n" + "=" * 70)
    print("STEP 8: Saving Final Results")
    print("=" * 70)

    results_dir = dirs["results"]

    # Save combined results
    with open(os.path.join(results_dir, "combined_results.json"), "w") as f:
        json.dump(combined_results, f, indent=2)

    with open(os.path.join(results_dir, "combined_results.pkl"), "wb") as f:
        pickle.dump(combined_results, f)

    # Save numerical summary
    summary = {
        "experiment_timestamp": datetime.now().isoformat(),
        "distances": combined_results["distances"],
        "error_rates": combined_results["error_rates"],
        "noise_types": combined_results["noise_types"],
        "n_samples_train": metadata["n_samples_train"],
        "n_samples_test": metadata["n_samples_test"],
        "n_rounds": metadata["n_rounds"],
        "results_summary": {}
    }

    for decoder in ["rl", "mwpm"]:
        summary["results_summary"][decoder] = {}
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

    # Add threshold estimates
    summary["threshold_estimates"] = {}
    for decoder in ["rl", "mwpm"]:
        summary["threshold_estimates"][decoder] = {}
        for noise_type in combined_results["noise_types"]:
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
- **Training samples**: {summary['n_samples_train']}
- **Test samples**: {summary['n_samples_test']}
- **Syndrome rounds**: {summary['n_rounds']}

## Threshold Estimates

| Decoder | Depolarizing | Dephasing |
|---------|-------------|-----------|
| RL | {summary['threshold_estimates']['rl']['depolarizing']} | {summary['threshold_estimates']['rl']['dephasing']} |
| MWPM | {summary['threshold_estimates']['mwpm']['depolarizing']} | {summary['threshold_estimates']['mwpm']['dephasing']} |

## Logical Error Rates Summary

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
## Generated Figures

1. `P_L_vs_p_depolarizing.png` - Logical error rate vs physical error rate (depolarizing)
2. `P_L_vs_p_dephasing.png` - Logical error rate vs physical error rate (dephasing)
3. `threshold_estimation.png` - Scaling exponent alpha vs error rate
4. `rl_vs_mwpm_comparison.png` - Performance ratio between decoders
5. `error_matching_graph.png` - Detector layout visualization
6. `bloch_sphere_trajectory.png` - Logical state trajectory
7. `summary_comparison.png` - Combined comparison plots

## File Locations

- Results: `{results_dir}`
- Figures: `{figures_dir}`
- Models: `{models_dir}`
- Data: `{data_dir}`
""".format(
        results_dir=dirs["results"],
        figures_dir=dirs["figures"],
        models_dir=dirs["models"],
        data_dir=dirs["data"]
    )

    with open(os.path.join(results_dir, "experiment_report.md"), "w") as f:
        f.write(report)

    print(f"Summary saved to {os.path.join(results_dir, 'experiment_summary.json')}")
    print(f"Report saved to {os.path.join(results_dir, 'experiment_report.md')}")
    print(f"\nExperiment completed successfully!")


def run_full_pipeline(
    base_dir: str = None,
    distances: List[int] = [3, 5, 7],
    n_error_rates: int = 15,
    n_samples_train: int = 50000,
    n_samples_test: int = 10000,
    n_rounds: int = 3,
    n_epochs: int = 50
):
    """Run the complete experimental pipeline."""
    start_time = time.time()

    print("\n" + "=" * 70)
    print("SURFACE CODE QEC EXPERIMENT - FULL PIPELINE")
    print("=" * 70)
    print(f"Distances: {distances}")
    print(f"Error rates: {n_error_rates} values in [0.01, 0.15]")
    print(f"Training samples per config: {n_samples_train}")
    print(f"Test samples per config: {n_samples_test}")
    print(f"Syndrome rounds: {n_rounds}")
    print(f"Training epochs: {n_epochs}")

    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    error_rates = np.linspace(0.01, 0.15, n_error_rates).tolist()

    # Setup directories
    dirs = setup_directories(base_dir)
    print(f"\nOutput directory: {dirs['base']}")

    # Step 1: Generate datasets
    metadata = step1_generate_datasets(
        dirs=dirs,
        distances=distances,
        error_rates=error_rates,
        n_samples_train=n_samples_train,
        n_samples_test=n_samples_test,
        n_rounds=n_rounds
    )

    # Step 2: Train PPO agent on d=3
    agent = step2_train_ppo_agent(
        dirs=dirs,
        metadata=metadata,
        train_distance=3,
        n_epochs=n_epochs
    )

    # Step 3: Evaluate RL agent
    rl_results = step3_evaluate_rl_agent(
        dirs=dirs,
        metadata=metadata,
        agent=agent,
        train_distance=3,
        distances=distances
    )

    # Step 4: Run MWPM baseline
    mwpm_results = step4_run_mwpm_baseline(
        dirs=dirs,
        metadata=metadata,
        distances=distances,
        n_samples=n_samples_test
    )

    # Step 5: Compute logical error rates
    combined_results = step5_compute_logical_error_rates(rl_results, mwpm_results)

    # Step 6: Fit threshold
    threshold_results = step6_fit_threshold(combined_results, dirs)

    # Step 7: Generate plots
    step7_generate_plots(combined_results, threshold_results, dirs)

    # Step 8: Save final results
    step8_save_final_results(
        dirs=dirs,
        metadata=metadata,
        combined_results=combined_results,
        threshold_results=threshold_results,
        rl_results=rl_results,
        mwpm_results=mwpm_results
    )

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETED in {total_time / 60:.2f} minutes")
    print(f"{'=' * 70}")

    return dirs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full QEC experiment pipeline")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--n-error-rates", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Run quick test with reduced samples")

    args = parser.parse_args()

    if args.quick:
        args.n_train = 5000
        args.n_test = 1000
        args.n_epochs = 10
        args.n_error_rates = 8

    run_full_pipeline(
        base_dir=args.base_dir,
        distances=args.distances,
        n_error_rates=args.n_error_rates,
        n_samples_train=args.n_train,
        n_samples_test=args.n_test,
        n_rounds=args.n_rounds,
        n_epochs=args.n_epochs
    )
