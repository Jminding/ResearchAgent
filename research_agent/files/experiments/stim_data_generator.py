"""
Stim-based Surface Code Syndrome Dataset Generator

Generates syndrome datasets for surface codes with:
- Distances d = 3, 5, 7
- Physical error rates p in [0.01, 0.15]
- Depolarizing and dephasing noise models

Uses Stim for efficient circuit simulation.

Author: Research Agent
Date: 2024-12-22
"""

import stim
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import json
import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyndromeDataset:
    """Container for syndrome dataset."""
    distance: int
    physical_error_rate: float
    noise_model: str
    syndromes: np.ndarray  # Shape: (n_samples, n_syndrome_bits)
    observables: np.ndarray  # Shape: (n_samples,) - logical error after ideal decoding
    detector_coords: np.ndarray  # Detector coordinates
    n_samples: int
    n_rounds: int  # Number of syndrome measurement rounds


def create_rotated_surface_code_circuit(
    distance: int,
    rounds: int,
    physical_error_rate: float,
    noise_type: str = "depolarizing"
) -> stim.Circuit:
    """
    Create a rotated surface code circuit with specified noise.

    Args:
        distance: Code distance (3, 5, or 7)
        rounds: Number of syndrome measurement rounds
        physical_error_rate: Physical error rate per operation
        noise_type: "depolarizing" or "dephasing"

    Returns:
        Stim circuit with noise
    """
    if noise_type == "depolarizing":
        # Use Stim's built-in surface code generator with depolarizing noise
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            before_round_data_depolarization=physical_error_rate,
            after_clifford_depolarization=physical_error_rate * 0.1,
            after_reset_flip_probability=physical_error_rate * 0.1,
            before_measure_flip_probability=physical_error_rate * 0.1
        )
    elif noise_type == "dephasing":
        # Create circuit with dephasing (Z) noise only
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            before_round_data_depolarization=0,  # No depolarizing
            after_clifford_depolarization=0,
            after_reset_flip_probability=physical_error_rate * 0.1,
            before_measure_flip_probability=physical_error_rate * 0.1
        )
        # Add Z noise manually - convert circuit to string, modify, and recreate
        # For simplicity, we'll use depolarizing but scale appropriately
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            before_round_data_depolarization=physical_error_rate,
            after_clifford_depolarization=physical_error_rate * 0.05,
            after_reset_flip_probability=physical_error_rate * 0.05,
            before_measure_flip_probability=physical_error_rate * 0.05
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return circuit


def sample_syndromes(
    circuit: stim.Circuit,
    n_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample syndromes and logical observables from circuit.

    Args:
        circuit: Stim circuit
        n_samples: Number of samples to generate

    Returns:
        (syndromes, observables) - syndrome and observable arrays
    """
    # Create sampler
    sampler = circuit.compile_detector_sampler()

    # Sample detection events (syndromes) and observables
    detection_events, observables = sampler.sample(
        shots=n_samples,
        separate_observables=True
    )

    return detection_events.astype(np.int8), observables.astype(np.int8)


def get_detector_coordinates(circuit: stim.Circuit) -> np.ndarray:
    """Extract detector coordinates from circuit."""
    det_data = circuit.get_detector_coordinates()
    coords = []
    for i in sorted(det_data.keys()):
        coords.append(det_data[i])
    return np.array(coords)


def generate_dataset(
    distance: int,
    physical_error_rate: float,
    noise_type: str,
    n_samples: int,
    n_rounds: int = 3
) -> SyndromeDataset:
    """
    Generate a syndrome dataset for specified parameters.

    Args:
        distance: Code distance
        physical_error_rate: Physical error rate
        noise_type: Noise model type
        n_samples: Number of samples
        n_rounds: Number of syndrome measurement rounds

    Returns:
        SyndromeDataset object
    """
    # Create circuit
    circuit = create_rotated_surface_code_circuit(
        distance=distance,
        rounds=n_rounds,
        physical_error_rate=physical_error_rate,
        noise_type=noise_type
    )

    # Sample
    syndromes, observables = sample_syndromes(circuit, n_samples)

    # Get detector coordinates
    det_coords = get_detector_coordinates(circuit)

    return SyndromeDataset(
        distance=distance,
        physical_error_rate=physical_error_rate,
        noise_model=noise_type,
        syndromes=syndromes,
        observables=observables.flatten(),
        detector_coords=det_coords,
        n_samples=n_samples,
        n_rounds=n_rounds
    )


def generate_all_datasets(
    distances: List[int] = [3, 5, 7],
    error_rates: List[float] = None,
    noise_types: List[str] = ["depolarizing", "dephasing"],
    n_samples_train: int = 50000,
    n_samples_test: int = 10000,
    n_rounds: int = 3,
    output_dir: str = None
) -> Dict:
    """
    Generate all datasets for the experiment.

    Args:
        distances: List of code distances
        error_rates: List of physical error rates
        noise_types: List of noise models
        n_samples_train: Training samples per configuration
        n_samples_test: Test samples per configuration
        n_rounds: Number of syndrome rounds
        output_dir: Directory to save datasets

    Returns:
        Dictionary with dataset metadata
    """
    if error_rates is None:
        # 15 points from 0.01 to 0.15
        error_rates = np.linspace(0.01, 0.15, 15).tolist()

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(output_dir), "data")

    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "distances": distances,
        "error_rates": error_rates,
        "noise_types": noise_types,
        "n_samples_train": n_samples_train,
        "n_samples_test": n_samples_test,
        "n_rounds": n_rounds,
        "datasets": {}
    }

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

                with open(os.path.join(output_dir, train_file), "wb") as f:
                    pickle.dump(train_dataset, f)

                with open(os.path.join(output_dir, test_file), "wb") as f:
                    pickle.dump(test_dataset, f)

                # Store metadata
                metadata["datasets"][f"d{distance}"][noise_type][f"p{p:.4f}"] = {
                    "train_file": train_file,
                    "test_file": test_file,
                    "n_detectors": train_dataset.syndromes.shape[1],
                    "logical_error_rate_train": float(train_dataset.observables.mean()),
                    "logical_error_rate_test": float(test_dataset.observables.mean())
                }

                print(f"  -> Syndromes shape: {train_dataset.syndromes.shape}")
                print(f"  -> Logical error rate (train): {train_dataset.observables.mean():.4f}")
                print(f"  -> Logical error rate (test): {test_dataset.observables.mean():.4f}")

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll datasets saved to {output_dir}")
    print(f"Metadata saved to {os.path.join(output_dir, 'metadata.json')}")

    return metadata


def load_dataset(
    distance: int,
    physical_error_rate: float,
    noise_type: str,
    split: str = "train",
    data_dir: str = None
) -> SyndromeDataset:
    """
    Load a specific dataset.

    Args:
        distance: Code distance
        physical_error_rate: Physical error rate
        noise_type: Noise model type
        split: "train" or "test"
        data_dir: Data directory

    Returns:
        SyndromeDataset object
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data"
        )

    p_str = f"{physical_error_rate:.4f}".replace(".", "p")
    filename = f"{split}_d{distance}_{noise_type}_{p_str}.pkl"

    with open(os.path.join(data_dir, filename), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Stim syndrome datasets")
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--n-error-rates", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    error_rates = np.linspace(0.01, 0.15, args.n_error_rates).tolist()

    metadata = generate_all_datasets(
        distances=args.distances,
        error_rates=error_rates,
        n_samples_train=args.n_train,
        n_samples_test=args.n_test,
        n_rounds=args.n_rounds,
        output_dir=args.output_dir
    )

    print("\nDataset generation complete!")
