"""
MWPM Baseline Decoder using PyMatching

Implements Minimum Weight Perfect Matching decoder as baseline
for comparison with RL-based decoder.

Author: Research Agent
Date: 2024-12-22
"""

import numpy as np
import stim
import pymatching
from typing import Dict, List, Tuple, Optional
import os
import pickle


class MWPMDecoder:
    """
    MWPM Decoder using PyMatching for surface codes.

    Uses the detector error model from Stim to construct
    the matching graph automatically.
    """

    def __init__(
        self,
        distance: int,
        rounds: int = 3,
        physical_error_rate: float = 0.01,
        noise_type: str = "depolarizing"
    ):
        """
        Initialize MWPM decoder.

        Args:
            distance: Code distance
            rounds: Number of syndrome measurement rounds
            physical_error_rate: Physical error rate (for building DEM)
            noise_type: Noise model type
        """
        self.distance = distance
        self.rounds = rounds
        self.physical_error_rate = physical_error_rate
        self.noise_type = noise_type

        # Build matching from detector error model
        self._build_matching()

    def _build_matching(self):
        """Build PyMatching decoder from Stim circuit."""
        # Create representative circuit for building DEM
        if self.noise_type == "depolarizing":
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=self.distance,
                rounds=self.rounds,
                before_round_data_depolarization=self.physical_error_rate,
                after_clifford_depolarization=self.physical_error_rate * 0.1,
                after_reset_flip_probability=self.physical_error_rate * 0.1,
                before_measure_flip_probability=self.physical_error_rate * 0.1
            )
        else:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=self.distance,
                rounds=self.rounds,
                before_round_data_depolarization=self.physical_error_rate,
                after_clifford_depolarization=self.physical_error_rate * 0.05,
                after_reset_flip_probability=self.physical_error_rate * 0.05,
                before_measure_flip_probability=self.physical_error_rate * 0.05
            )

        # Get detector error model
        dem = circuit.detector_error_model(decompose_errors=True)

        # Create PyMatching decoder
        self.matching = pymatching.Matching.from_detector_error_model(dem)

        # Store circuit for reference
        self.circuit = circuit

    def decode(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Decode syndromes to predict logical observable.

        Args:
            syndromes: Shape (n_samples, n_detectors) or (n_detectors,)

        Returns:
            Predicted logical observable flips, shape (n_samples,) or scalar
        """
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)

        predictions = []
        for syndrome in syndromes:
            # PyMatching expects boolean array
            syndrome_bool = syndrome.astype(bool)
            prediction = self.matching.decode(syndrome_bool)
            predictions.append(prediction[0] if len(prediction) > 0 else 0)

        return np.array(predictions, dtype=np.int8)

    def evaluate(
        self,
        syndromes: np.ndarray,
        true_observables: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate decoder on test data.

        Args:
            syndromes: Test syndromes
            true_observables: True logical observables

        Returns:
            Evaluation metrics
        """
        predictions = self.decode(syndromes)

        # Logical error rate: XOR of prediction and true observable
        # If they differ, decoding failed
        logical_errors = predictions != true_observables
        logical_error_rate = np.mean(logical_errors)

        # Accuracy (correct decoding)
        accuracy = 1 - logical_error_rate

        return {
            "accuracy": float(accuracy),
            "logical_error_rate": float(logical_error_rate),
            "n_samples": len(syndromes),
            "n_errors": int(np.sum(logical_errors))
        }


def run_mwpm_baseline(
    distances: List[int] = [3, 5, 7],
    error_rates: List[float] = None,
    noise_types: List[str] = ["depolarizing", "dephasing"],
    n_samples: int = 10000,
    n_rounds: int = 3,
    output_dir: str = None
) -> Dict:
    """
    Run MWPM baseline evaluation.

    Args:
        distances: Code distances to evaluate
        error_rates: Physical error rates
        noise_types: Noise models
        n_samples: Samples per configuration
        n_rounds: Syndrome measurement rounds
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    if error_rates is None:
        error_rates = np.linspace(0.01, 0.15, 15).tolist()

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "distances": distances,
        "error_rates": error_rates,
        "noise_types": noise_types,
        "n_samples": n_samples,
        "n_rounds": n_rounds,
        "logical_error_rates": {}
    }

    total_configs = len(distances) * len(error_rates) * len(noise_types)
    config_idx = 0

    for distance in distances:
        results["logical_error_rates"][f"d{distance}"] = {}

        for noise_type in noise_types:
            results["logical_error_rates"][f"d{distance}"][noise_type] = {}

            for p in error_rates:
                config_idx += 1
                print(f"[{config_idx}/{total_configs}] MWPM: d={distance}, p={p:.4f}, noise={noise_type}")

                # Create decoder
                decoder = MWPMDecoder(
                    distance=distance,
                    rounds=n_rounds,
                    physical_error_rate=p,
                    noise_type=noise_type
                )

                # Generate fresh test data
                sampler = decoder.circuit.compile_detector_sampler()
                syndromes, observables = sampler.sample(
                    shots=n_samples,
                    separate_observables=True
                )

                syndromes = syndromes.astype(np.int8)
                observables = observables.flatten().astype(np.int8)

                # Evaluate
                metrics = decoder.evaluate(syndromes, observables)

                results["logical_error_rates"][f"d{distance}"][noise_type][f"p{p:.4f}"] = {
                    "logical_error_rate": metrics["logical_error_rate"],
                    "accuracy": metrics["accuracy"],
                    "n_samples": metrics["n_samples"],
                    "n_errors": metrics["n_errors"]
                }

                print(f"  -> Logical error rate: {metrics['logical_error_rate']:.6f}")

    # Save results
    results_file = os.path.join(output_dir, "mwpm_baseline_results.json")
    import json
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MWPM baseline decoder")
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--n-error-rates", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    error_rates = np.linspace(0.01, 0.15, args.n_error_rates).tolist()

    results = run_mwpm_baseline(
        distances=args.distances,
        error_rates=error_rates,
        n_samples=args.n_samples,
        n_rounds=args.n_rounds,
        output_dir=args.output_dir
    )

    print("\nMWPM baseline evaluation complete!")
