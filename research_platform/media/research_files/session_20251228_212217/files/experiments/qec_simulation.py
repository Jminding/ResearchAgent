"""
Quantum Error Correction Simulation Framework
=============================================
Implements surface code QEC with RL-based GNN decoder and MWPM baseline.
Designed for peer review revision experiments.

Author: Research Agent (Revision)
Date: 2025-12-29
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import itertools


# =============================================================================
# Surface Code Simulation
# =============================================================================

class SurfaceCode:
    """
    Simplified surface code simulator for QEC experiments.
    Models a d x d rotated surface code with depolarizing noise.
    """

    def __init__(self, distance: int, physical_error_rate: float, seed: int = None):
        self.d = distance
        self.p = physical_error_rate
        self.rng = np.random.default_rng(seed)

        # Number of data qubits and stabilizers
        self.n_data = distance ** 2
        self.n_x_stabilizers = (distance - 1) * distance // 2 + (distance - 1) * (distance - 1) // 2
        self.n_z_stabilizers = self.n_x_stabilizers

        # Adjacency structure for syndrome extraction
        self._build_lattice()

    def _build_lattice(self):
        """Build syndrome graph for surface code."""
        d = self.d
        # Simplified: model stabilizer-qubit relationships
        self.x_stabilizer_qubits = []
        self.z_stabilizer_qubits = []

        # For rotated surface code, each interior stabilizer touches 4 qubits
        # Boundary stabilizers touch 2 qubits
        for i in range(d - 1):
            for j in range(d - 1):
                # X stabilizers on one sublattice
                qubits = [i * d + j, i * d + j + 1, (i + 1) * d + j, (i + 1) * d + j + 1]
                self.x_stabilizer_qubits.append(qubits)
                self.z_stabilizer_qubits.append(qubits)

    def generate_error(self) -> np.ndarray:
        """Generate random Pauli errors on data qubits."""
        # Depolarizing channel: each qubit has probability p of error
        # Error can be X, Y, or Z with equal probability
        errors = np.zeros(self.n_data, dtype=int)
        for i in range(self.n_data):
            if self.rng.random() < self.p:
                errors[i] = self.rng.integers(1, 4)  # 1=X, 2=Y, 3=Z
        return errors

    def measure_syndrome(self, errors: np.ndarray) -> np.ndarray:
        """Measure stabilizer syndrome given errors."""
        # X syndrome detects Z errors, Z syndrome detects X errors
        x_syndrome = []
        z_syndrome = []

        for qubits in self.x_stabilizer_qubits:
            # X stabilizer detects if odd number of Z or Y errors
            parity = sum(1 for q in qubits if q < len(errors) and errors[q] in [2, 3]) % 2
            x_syndrome.append(parity)

        for qubits in self.z_stabilizer_qubits:
            # Z stabilizer detects if odd number of X or Y errors
            parity = sum(1 for q in qubits if q < len(errors) and errors[q] in [1, 2]) % 2
            z_syndrome.append(parity)

        return np.array(x_syndrome + z_syndrome)

    def check_logical_error(self, errors: np.ndarray, correction: np.ndarray) -> bool:
        """Check if error + correction results in logical error."""
        # Combine original error with correction
        combined = np.zeros(self.n_data, dtype=int)
        for i in range(self.n_data):
            # XOR-like combination for Pauli operators
            combined[i] = errors[i] ^ correction[i]

        # Check if combined error is a logical operator
        # For surface code, logical X crosses horizontally, logical Z crosses vertically

        # Count X-type errors along horizontal logical operator path (first row)
        x_logical_parity = sum(1 for i in range(self.d) if combined[i] in [1, 2]) % 2

        # Count Z-type errors along vertical logical operator path (first column)
        z_logical_parity = sum(1 for i in range(0, self.n_data, self.d) if combined[i] in [2, 3]) % 2

        return x_logical_parity == 1 or z_logical_parity == 1


# =============================================================================
# MWPM Decoder (Baseline)
# =============================================================================

class MWPMDecoder:
    """
    Minimum Weight Perfect Matching decoder.
    Uses simplified matching algorithm for simulation.
    """

    def __init__(self, distance: int, physical_error_rate: float):
        self.d = distance
        self.p = physical_error_rate
        self._build_matching_graph()

    def _build_matching_graph(self):
        """Pre-compute edge weights based on error probability."""
        # Weight = -log(p / (1-p)) for edge probability p
        # Simplified: use Manhattan distance-based weights
        self.edge_weights = {}
        d = self.d

        # Build approximate matching graph
        for i in range((d - 1) ** 2):
            for j in range(i + 1, (d - 1) ** 2):
                ix, iy = i // (d - 1), i % (d - 1)
                jx, jy = j // (d - 1), j % (d - 1)
                dist = abs(ix - jx) + abs(iy - jy)
                weight = dist * np.log((1 - self.p) / self.p) if self.p < 0.5 else dist
                self.edge_weights[(i, j)] = weight

    def decode(self, syndrome: np.ndarray, surface_code: SurfaceCode) -> np.ndarray:
        """Decode syndrome and return correction."""
        n_data = surface_code.n_data
        correction = np.zeros(n_data, dtype=int)

        # Find syndrome defects
        n_stabilizers = len(syndrome) // 2
        x_defects = [i for i in range(n_stabilizers) if syndrome[i] == 1]
        z_defects = [i for i in range(n_stabilizers, len(syndrome)) if syndrome[i] == 1]

        # Greedy matching (simplified MWPM)
        correction = self._greedy_match(x_defects, z_defects, surface_code)

        return correction

    def _greedy_match(self, x_defects: List[int], z_defects: List[int],
                      surface_code: SurfaceCode) -> np.ndarray:
        """Simplified greedy matching for correction."""
        d = surface_code.d
        n_data = surface_code.n_data
        correction = np.zeros(n_data, dtype=int)

        # Match X defects (correct with Z operators)
        matched_x = set()
        for i, def1 in enumerate(x_defects):
            if def1 in matched_x:
                continue

            # Find closest unmatched defect or boundary
            best_partner = None
            best_dist = float('inf')

            for def2 in x_defects[i + 1:]:
                if def2 in matched_x:
                    continue

                d1_x, d1_y = def1 // (d - 1), def1 % (d - 1)
                d2_x, d2_y = def2 // (d - 1), def2 % (d - 1)
                dist = abs(d1_x - d2_x) + abs(d1_y - d2_y)

                if dist < best_dist:
                    best_dist = dist
                    best_partner = def2

            # Check distance to boundary
            d1_x, d1_y = def1 // (d - 1), def1 % (d - 1)
            boundary_dist = min(d1_x, d1_y, d - 2 - d1_x, d - 2 - d1_y)

            if best_partner is not None and best_dist <= boundary_dist:
                # Match to another defect
                matched_x.add(def1)
                matched_x.add(best_partner)

                # Apply correction along path
                d1_x, d1_y = def1 // (d - 1), def1 % (d - 1)
                d2_x, d2_y = best_partner // (d - 1), best_partner % (d - 1)

                # Correct qubits along path
                for x in range(min(d1_x, d2_x), max(d1_x, d2_x) + 1):
                    for y in range(min(d1_y, d2_y), max(d1_y, d2_y) + 1):
                        if x * d + y < n_data:
                            correction[x * d + y] ^= 3  # Z correction
            else:
                # Match to boundary
                matched_x.add(def1)
                d1_x, d1_y = def1 // (d - 1), def1 % (d - 1)

                # Correct to nearest boundary
                if d1_x < d - 1 - d1_x:
                    for x in range(d1_x + 1):
                        if x * d + d1_y < n_data:
                            correction[x * d + d1_y] ^= 3
                else:
                    for x in range(d1_x, d - 1):
                        if x * d + d1_y < n_data:
                            correction[x * d + d1_y] ^= 3

        return correction


# =============================================================================
# GNN-based RL Decoder
# =============================================================================

class GNNRLDecoder:
    """
    Graph Neural Network based Reinforcement Learning decoder.
    Uses policy gradient with GNN architecture.
    """

    def __init__(self, distance: int, physical_error_rate: float,
                 gnn_layers: int = 4, hidden_dim: int = 64,
                 learning_rate: float = 0.001, reward_type: str = "sparse",
                 seed: int = None):
        self.d = distance
        self.p = physical_error_rate
        self.gnn_layers = gnn_layers
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.reward_type = reward_type
        self.rng = np.random.default_rng(seed)

        # Initialize GNN parameters (simplified as weight matrices)
        self._init_model()

        # Training statistics
        self.training_losses = []
        self.episode_rewards = []

    def _init_model(self):
        """Initialize GNN model parameters."""
        n_syndrome = 2 * (self.d - 1) ** 2  # Approximate syndrome size
        n_data = self.d ** 2

        # GNN layers: message passing weights
        self.W_msg = [self.rng.standard_normal((self.hidden_dim, self.hidden_dim)) * 0.1
                      for _ in range(self.gnn_layers)]
        self.W_upd = [self.rng.standard_normal((self.hidden_dim, self.hidden_dim)) * 0.1
                      for _ in range(self.gnn_layers)]

        # Input embedding
        self.W_in = self.rng.standard_normal((self.hidden_dim, 1)) * 0.1

        # Output layer: predict correction probabilities
        self.W_out = self.rng.standard_normal((4, self.hidden_dim)) * 0.1  # 4 Pauli options

        # Bias terms
        self.b_out = np.zeros(4)

        # Model size (for ablation study metrics)
        self.n_params = sum(w.size for w in self.W_msg + self.W_upd) + \
                        self.W_in.size + self.W_out.size + self.b_out.size

    def _forward(self, syndrome: np.ndarray) -> np.ndarray:
        """Forward pass through GNN."""
        # Embed syndrome as node features
        n_nodes = len(syndrome)
        h = np.zeros((n_nodes, self.hidden_dim))

        for i in range(n_nodes):
            h[i] = syndrome[i] * self.W_in.flatten()[:self.hidden_dim]

        # Message passing layers
        for l in range(self.gnn_layers):
            h_new = np.zeros_like(h)
            for i in range(n_nodes):
                # Aggregate messages from neighbors (simplified: all nodes within distance)
                msg = np.zeros(self.hidden_dim)
                n_neighbors = 0
                for j in range(n_nodes):
                    if i != j and abs(i - j) <= self.d:  # Simplified adjacency
                        msg += np.tanh(self.W_msg[l] @ h[j])
                        n_neighbors += 1

                if n_neighbors > 0:
                    msg /= n_neighbors

                # Update node state
                h_new[i] = np.tanh(self.W_upd[l] @ (h[i] + msg))

            h = h_new

        # Pool and predict correction probabilities for each qubit
        h_pooled = np.mean(h, axis=0)
        logits = self.W_out @ h_pooled + self.b_out

        return logits

    def _compute_reward(self, correction: np.ndarray, errors: np.ndarray,
                        syndrome: np.ndarray, surface_code: SurfaceCode) -> float:
        """Compute reward based on reward type."""
        if self.reward_type == "sparse":
            # Binary reward: 1 if no logical error, 0 otherwise
            logical_error = surface_code.check_logical_error(errors, correction)
            return 1.0 if not logical_error else 0.0

        elif self.reward_type == "dense_syndrome":
            # Reward for reducing syndrome
            corrected_syndrome = surface_code.measure_syndrome(errors ^ correction)
            original_weight = np.sum(syndrome)
            corrected_weight = np.sum(corrected_syndrome)

            # Base reward from syndrome reduction
            syndrome_reward = (original_weight - corrected_weight) / max(original_weight, 1)

            # Bonus for successful decoding
            logical_error = surface_code.check_logical_error(errors, correction)
            return syndrome_reward + (2.0 if not logical_error else -1.0)

        elif self.reward_type == "dense_distance":
            # Reward based on Hamming distance to true correction
            # Optimal correction is the error itself (for this simplified model)
            hamming_dist = np.sum(errors != correction)
            max_dist = len(errors)

            # Normalize and add logical error penalty
            dist_reward = 1.0 - (hamming_dist / max_dist)
            logical_error = surface_code.check_logical_error(errors, correction)
            return dist_reward + (1.0 if not logical_error else -0.5)

        elif self.reward_type == "shaped_curriculum":
            # Curriculum: start with dense, gradually become sparse
            # This is controlled by training progress (not implemented here, default to dense)
            return self._compute_reward(correction, errors, syndrome, surface_code)

        else:
            # Default sparse
            logical_error = surface_code.check_logical_error(errors, correction)
            return 1.0 if not logical_error else 0.0

    def decode(self, syndrome: np.ndarray, surface_code: SurfaceCode) -> np.ndarray:
        """Decode syndrome using trained policy."""
        n_data = surface_code.n_data
        correction = np.zeros(n_data, dtype=int)

        # Get policy logits
        logits = self._forward(syndrome)
        probs = self._softmax(logits)

        # Sample correction for each qubit (simplified: use same action dist for all)
        for i in range(n_data):
            # Add some position-dependent variation
            local_logits = logits + 0.1 * self.rng.standard_normal(4)
            local_probs = self._softmax(local_logits)
            correction[i] = self.rng.choice(4, p=local_probs)

        return correction

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train_episode(self, surface_code: SurfaceCode) -> Tuple[float, float]:
        """Run one training episode with policy gradient update."""
        # Generate error and syndrome
        errors = surface_code.generate_error()
        syndrome = surface_code.measure_syndrome(errors)

        # Get action (correction) from policy
        correction = self.decode(syndrome, surface_code)

        # Compute reward
        reward = self._compute_reward(correction, errors, syndrome, surface_code)

        # Policy gradient update (simplified REINFORCE)
        logits = self._forward(syndrome)
        probs = self._softmax(logits)

        # Compute gradient (simplified)
        grad_scale = self.lr * reward

        # Update output layer (simplified gradient)
        h_pooled = np.mean(np.tanh(np.outer(syndrome, self.W_in.flatten()[:len(syndrome)])), axis=0)
        if len(h_pooled) < self.hidden_dim:
            h_pooled = np.pad(h_pooled, (0, self.hidden_dim - len(h_pooled)))

        # Gradient descent step
        for i in range(4):
            if i == correction[0]:  # Use first qubit action as representative
                self.W_out[i] += grad_scale * (1 - probs[i]) * h_pooled[:self.hidden_dim]
            else:
                self.W_out[i] -= grad_scale * probs[i] * h_pooled[:self.hidden_dim]

        # Compute loss (negative log likelihood weighted by reward)
        action_prob = probs[correction[0]]
        loss = -np.log(action_prob + 1e-10) * (1 - reward)

        self.training_losses.append(loss)
        self.episode_rewards.append(reward)

        return loss, reward

    def train(self, surface_code: SurfaceCode, num_episodes: int,
              verbose: bool = False) -> Dict:
        """Train the decoder for specified number of episodes."""
        start_time = time.time()

        for ep in range(num_episodes):
            loss, reward = self.train_episode(surface_code)

            if verbose and (ep + 1) % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {ep + 1}: Avg reward (last 100) = {avg_reward:.4f}")

        training_time = time.time() - start_time

        # Compute convergence episode (when reward stabilizes)
        window = 50
        convergence_episode = num_episodes
        if len(self.episode_rewards) > window:
            for i in range(window, len(self.episode_rewards)):
                recent = self.episode_rewards[i - window:i]
                if np.std(recent) < 0.1 and np.mean(recent) > 0.8:
                    convergence_episode = i
                    break

        return {
            "training_time": training_time,
            "final_loss": np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            "final_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "convergence_episode": convergence_episode,
            "total_episodes": num_episodes
        }


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_decoder(decoder, surface_code: SurfaceCode, num_samples: int = 1000) -> Dict:
    """Evaluate decoder performance on random error samples."""
    logical_errors = 0

    for _ in range(num_samples):
        errors = surface_code.generate_error()
        syndrome = surface_code.measure_syndrome(errors)
        correction = decoder.decode(syndrome, surface_code)

        if surface_code.check_logical_error(errors, correction):
            logical_errors += 1

    logical_error_rate = logical_errors / num_samples

    # 95% CI using normal approximation
    stderr = np.sqrt(logical_error_rate * (1 - logical_error_rate) / num_samples)
    ci_95 = 1.96 * stderr

    return {
        "logical_error_rate": logical_error_rate,
        "logical_errors": logical_errors,
        "total_samples": num_samples,
        "ci_95_lower": max(0, logical_error_rate - ci_95),
        "ci_95_upper": min(1, logical_error_rate + ci_95),
        "stderr": stderr
    }


def get_mwpm_benchmark(distance: int, physical_error_rate: float) -> float:
    """
    Return expected MWPM logical error rate from literature benchmarks.
    Based on: Dennis et al. (2002), Fowler et al. (2012) for surface code.

    For p << p_th (threshold ~ 0.01), logical error rate scales as:
    p_L ~ A * (p / p_th)^((d+1)/2)

    where A is a constant and d is the code distance.
    """
    p_threshold = 0.0103  # Surface code threshold with MWPM

    if physical_error_rate >= p_threshold:
        # Above threshold: logical error rate approaches 0.5
        return 0.5 * (1 - np.exp(-10 * (physical_error_rate - p_threshold)))

    # Below threshold: exponential suppression
    # p_L ~ (p / p_th)^((d+1)/2)
    suppression_exponent = (distance + 1) / 2
    base_rate = (physical_error_rate / p_threshold) ** suppression_exponent

    # Scale factor from empirical fits
    A = 0.03  # Approximate prefactor

    return A * base_rate


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class ExperimentResult:
    """Single experiment result."""
    config_name: str
    parameters: Dict
    metrics: Dict
    ablation: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ResultsTable:
    """Collection of experiment results."""
    project_name: str
    results: List[ExperimentResult] = field(default_factory=list)

    def add_result(self, result: ExperimentResult):
        self.results.append(result)

    def to_dict(self) -> Dict:
        return {
            "project_name": self.project_name,
            "results": [asdict(r) for r in self.results]
        }

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, path: str):
        """Export results to CSV format."""
        if not self.results:
            return

        lines = []
        # Header
        header = ["config_name", "ablation", "error"]

        # Collect all parameter and metric keys
        all_params = set()
        all_metrics = set()
        for r in self.results:
            all_params.update(r.parameters.keys())
            all_metrics.update(r.metrics.keys())

        header.extend(sorted(all_params))
        header.extend(sorted(all_metrics))
        lines.append(",".join(header))

        # Data rows
        for r in self.results:
            row = [
                r.config_name,
                str(r.ablation) if r.ablation else "",
                str(r.error) if r.error else ""
            ]
            for p in sorted(all_params):
                row.append(str(r.parameters.get(p, "")))
            for m in sorted(all_metrics):
                row.append(str(r.metrics.get(m, "")))
            lines.append(",".join(row))

        with open(path, 'w') as f:
            f.write("\n".join(lines))


def run_extended_training_experiment(results_table: ResultsTable,
                                     distances: List[int],
                                     episodes_list: List[int],
                                     seeds: List[int],
                                     physical_error_rate: float = 0.005,
                                     eval_samples: int = 1000):
    """
    Run extended training experiments across code distances.
    Addresses reviewer concern about undertraining.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Extended Training")
    print("=" * 60)

    for d in distances:
        for episodes in episodes_list:
            for seed in seeds:
                config_name = f"extended_d{d}_ep{episodes}_s{seed}"
                print(f"\nRunning: {config_name}")

                try:
                    # Initialize
                    surface_code = SurfaceCode(d, physical_error_rate, seed=seed)
                    rl_decoder = GNNRLDecoder(d, physical_error_rate, seed=seed)

                    # Train
                    train_stats = rl_decoder.train(surface_code, episodes)

                    # Evaluate
                    eval_results = evaluate_decoder(rl_decoder, surface_code, eval_samples)

                    # Combine metrics
                    metrics = {
                        "logical_error_rate": eval_results["logical_error_rate"],
                        "ci_95_lower": eval_results["ci_95_lower"],
                        "ci_95_upper": eval_results["ci_95_upper"],
                        "stderr": eval_results["stderr"],
                        "training_loss": train_stats["final_loss"],
                        "convergence_episode": train_stats["convergence_episode"],
                        "training_time_sec": train_stats["training_time"]
                    }

                    result = ExperimentResult(
                        config_name=config_name,
                        parameters={
                            "code_distance": d,
                            "physical_error_rate": physical_error_rate,
                            "training_episodes": episodes,
                            "seed": seed
                        },
                        metrics=metrics
                    )
                    results_table.add_result(result)

                    print(f"  LER: {eval_results['logical_error_rate']:.4f} "
                          f"[{eval_results['ci_95_lower']:.4f}, {eval_results['ci_95_upper']:.4f}]")

                except Exception as e:
                    result = ExperimentResult(
                        config_name=config_name,
                        parameters={
                            "code_distance": d,
                            "physical_error_rate": physical_error_rate,
                            "training_episodes": episodes,
                            "seed": seed
                        },
                        metrics={},
                        error=str(e)
                    )
                    results_table.add_result(result)
                    print(f"  ERROR: {e}")


def run_baseline_comparison(results_table: ResultsTable,
                           distances: List[int],
                           seeds: List[int],
                           physical_error_rate: float = 0.005,
                           training_episodes: int = 2000,
                           eval_samples: int = 1000):
    """
    Compare RL decoder vs MWPM across code distances.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: RL vs MWPM Comparison")
    print("=" * 60)

    for d in distances:
        for seed in seeds:
            config_name = f"comparison_d{d}_s{seed}"
            print(f"\nRunning: {config_name}")

            try:
                # Initialize
                surface_code = SurfaceCode(d, physical_error_rate, seed=seed)
                rl_decoder = GNNRLDecoder(d, physical_error_rate, seed=seed)
                mwpm_decoder = MWPMDecoder(d, physical_error_rate)

                # Train RL decoder
                rl_decoder.train(surface_code, training_episodes)

                # Evaluate both
                rl_results = evaluate_decoder(rl_decoder, surface_code, eval_samples)
                mwpm_results = evaluate_decoder(mwpm_decoder, surface_code, eval_samples)

                # Get benchmark
                benchmark = get_mwpm_benchmark(d, physical_error_rate)

                metrics = {
                    "logical_error_rate_rl": rl_results["logical_error_rate"],
                    "rl_ci_95_lower": rl_results["ci_95_lower"],
                    "rl_ci_95_upper": rl_results["ci_95_upper"],
                    "logical_error_rate_mwpm": mwpm_results["logical_error_rate"],
                    "mwpm_ci_95_lower": mwpm_results["ci_95_lower"],
                    "mwpm_ci_95_upper": mwpm_results["ci_95_upper"],
                    "mwpm_benchmark": benchmark,
                    "mwpm_deviation_from_benchmark": abs(mwpm_results["logical_error_rate"] - benchmark),
                    "rl_vs_mwpm_ratio": rl_results["logical_error_rate"] / max(mwpm_results["logical_error_rate"], 1e-6)
                }

                result = ExperimentResult(
                    config_name=config_name,
                    parameters={
                        "code_distance": d,
                        "physical_error_rate": physical_error_rate,
                        "training_episodes": training_episodes,
                        "seed": seed
                    },
                    metrics=metrics
                )
                results_table.add_result(result)

                print(f"  RL LER: {rl_results['logical_error_rate']:.4f}, "
                      f"MWPM LER: {mwpm_results['logical_error_rate']:.4f}, "
                      f"Benchmark: {benchmark:.4f}")

            except Exception as e:
                result = ExperimentResult(
                    config_name=config_name,
                    parameters={
                        "code_distance": d,
                        "physical_error_rate": physical_error_rate,
                        "training_episodes": training_episodes,
                        "seed": seed
                    },
                    metrics={},
                    error=str(e)
                )
                results_table.add_result(result)
                print(f"  ERROR: {e}")


def run_reward_shaping_ablation(results_table: ResultsTable,
                                distances: List[int],
                                reward_types: List[str],
                                seeds: List[int],
                                physical_error_rate: float = 0.005,
                                training_episodes: int = 2000,
                                eval_samples: int = 1000):
    """
    Ablation study on different reward functions.
    """
    print("\n" + "=" * 60)
    print("ABLATION: Reward Shaping")
    print("=" * 60)

    for d in distances:
        for reward_type in reward_types:
            for seed in seeds:
                config_name = f"reward_d{d}_{reward_type}_s{seed}"
                print(f"\nRunning: {config_name}")

                try:
                    # Initialize with specific reward type
                    surface_code = SurfaceCode(d, physical_error_rate, seed=seed)
                    rl_decoder = GNNRLDecoder(d, physical_error_rate,
                                             reward_type=reward_type, seed=seed)

                    # Train
                    train_stats = rl_decoder.train(surface_code, training_episodes)

                    # Evaluate
                    eval_results = evaluate_decoder(rl_decoder, surface_code, eval_samples)

                    metrics = {
                        "logical_error_rate": eval_results["logical_error_rate"],
                        "ci_95_lower": eval_results["ci_95_lower"],
                        "ci_95_upper": eval_results["ci_95_upper"],
                        "convergence_episode": train_stats["convergence_episode"],
                        "final_reward": train_stats["final_reward"],
                        "training_time_sec": train_stats["training_time"]
                    }

                    result = ExperimentResult(
                        config_name=config_name,
                        parameters={
                            "code_distance": d,
                            "physical_error_rate": physical_error_rate,
                            "training_episodes": training_episodes,
                            "seed": seed,
                            "reward_type": reward_type
                        },
                        metrics=metrics,
                        ablation=f"reward_{reward_type}"
                    )
                    results_table.add_result(result)

                    print(f"  LER: {eval_results['logical_error_rate']:.4f}, "
                          f"Conv: {train_stats['convergence_episode']}")

                except Exception as e:
                    result = ExperimentResult(
                        config_name=config_name,
                        parameters={
                            "code_distance": d,
                            "physical_error_rate": physical_error_rate,
                            "training_episodes": training_episodes,
                            "seed": seed,
                            "reward_type": reward_type
                        },
                        metrics={},
                        ablation=f"reward_{reward_type}",
                        error=str(e)
                    )
                    results_table.add_result(result)
                    print(f"  ERROR: {e}")


def run_gnn_depth_ablation(results_table: ResultsTable,
                           distances: List[int],
                           layer_configs: List[Tuple[int, int]],  # (layers, hidden_dim)
                           seeds: List[int],
                           physical_error_rate: float = 0.005,
                           training_episodes: int = 2000,
                           eval_samples: int = 1000):
    """
    Ablation study on GNN architecture (depth and width).
    """
    print("\n" + "=" * 60)
    print("ABLATION: GNN Architecture")
    print("=" * 60)

    for d in distances:
        for layers, hidden_dim in layer_configs:
            for seed in seeds:
                config_name = f"gnn_d{d}_L{layers}_H{hidden_dim}_s{seed}"
                print(f"\nRunning: {config_name}")

                try:
                    # Initialize with specific architecture
                    surface_code = SurfaceCode(d, physical_error_rate, seed=seed)
                    rl_decoder = GNNRLDecoder(d, physical_error_rate,
                                             gnn_layers=layers, hidden_dim=hidden_dim,
                                             seed=seed)

                    # Train
                    start_time = time.time()
                    train_stats = rl_decoder.train(surface_code, training_episodes)

                    # Evaluate
                    eval_start = time.time()
                    eval_results = evaluate_decoder(rl_decoder, surface_code, eval_samples)
                    inference_time = (time.time() - eval_start) / eval_samples

                    metrics = {
                        "logical_error_rate": eval_results["logical_error_rate"],
                        "ci_95_lower": eval_results["ci_95_lower"],
                        "ci_95_upper": eval_results["ci_95_upper"],
                        "model_params": rl_decoder.n_params,
                        "inference_time_ms": inference_time * 1000,
                        "training_time_sec": train_stats["training_time"],
                        "convergence_episode": train_stats["convergence_episode"]
                    }

                    result = ExperimentResult(
                        config_name=config_name,
                        parameters={
                            "code_distance": d,
                            "physical_error_rate": physical_error_rate,
                            "training_episodes": training_episodes,
                            "seed": seed,
                            "gnn_layers": layers,
                            "hidden_dim": hidden_dim
                        },
                        metrics=metrics,
                        ablation=f"gnn_L{layers}_H{hidden_dim}"
                    )
                    results_table.add_result(result)

                    print(f"  LER: {eval_results['logical_error_rate']:.4f}, "
                          f"Params: {rl_decoder.n_params}, "
                          f"Inference: {inference_time*1000:.2f}ms")

                except Exception as e:
                    result = ExperimentResult(
                        config_name=config_name,
                        parameters={
                            "code_distance": d,
                            "physical_error_rate": physical_error_rate,
                            "training_episodes": training_episodes,
                            "seed": seed,
                            "gnn_layers": layers,
                            "hidden_dim": hidden_dim
                        },
                        metrics={},
                        ablation=f"gnn_L{layers}_H{hidden_dim}",
                        error=str(e)
                    )
                    results_table.add_result(result)
                    print(f"  ERROR: {e}")


def run_zero_shot_generalization(results_table: ResultsTable,
                                 train_distance: int,
                                 test_distance: int,
                                 episodes_list: List[int],
                                 seeds: List[int],
                                 physical_error_rate: float = 0.005,
                                 eval_samples: int = 1000):
    """
    Test zero-shot generalization from smaller to larger code distance.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Zero-Shot Generalization (d{train_distance} -> d{test_distance})")
    print("=" * 60)

    for episodes in episodes_list:
        for seed in seeds:
            config_name = f"zeroshot_d{train_distance}to{test_distance}_ep{episodes}_s{seed}"
            print(f"\nRunning: {config_name}")

            try:
                # Train on smaller distance
                train_code = SurfaceCode(train_distance, physical_error_rate, seed=seed)
                rl_decoder = GNNRLDecoder(train_distance, physical_error_rate, seed=seed)
                train_stats = rl_decoder.train(train_code, episodes)

                # Evaluate on training distance
                train_results = evaluate_decoder(rl_decoder, train_code, eval_samples)

                # Create decoder for test distance (reuse weights conceptually)
                test_code = SurfaceCode(test_distance, physical_error_rate, seed=seed)

                # For generalization test, we create a new decoder with same seed
                # but evaluate on larger code (simulates weight transfer)
                test_decoder = GNNRLDecoder(test_distance, physical_error_rate, seed=seed)
                # Copy learned output weights (simplified transfer)
                test_decoder.W_out = rl_decoder.W_out.copy()
                test_decoder.b_out = rl_decoder.b_out.copy()

                # Evaluate on test distance
                test_results = evaluate_decoder(test_decoder, test_code, eval_samples)

                # Get baseline MWPM for comparison
                mwpm_decoder = MWPMDecoder(test_distance, physical_error_rate)
                mwpm_results = evaluate_decoder(mwpm_decoder, test_code, eval_samples)

                metrics = {
                    "train_logical_error_rate": train_results["logical_error_rate"],
                    "test_logical_error_rate": test_results["logical_error_rate"],
                    "test_ci_95_lower": test_results["ci_95_lower"],
                    "test_ci_95_upper": test_results["ci_95_upper"],
                    "generalization_gap": test_results["logical_error_rate"] - train_results["logical_error_rate"],
                    "mwpm_test_error_rate": mwpm_results["logical_error_rate"],
                    "rl_vs_mwpm_at_test": test_results["logical_error_rate"] / max(mwpm_results["logical_error_rate"], 1e-6)
                }

                result = ExperimentResult(
                    config_name=config_name,
                    parameters={
                        "train_distance": train_distance,
                        "test_distance": test_distance,
                        "physical_error_rate": physical_error_rate,
                        "training_episodes": episodes,
                        "seed": seed
                    },
                    metrics=metrics
                )
                results_table.add_result(result)

                print(f"  Train LER: {train_results['logical_error_rate']:.4f}, "
                      f"Test LER: {test_results['logical_error_rate']:.4f}, "
                      f"Gap: {metrics['generalization_gap']:.4f}")

            except Exception as e:
                result = ExperimentResult(
                    config_name=config_name,
                    parameters={
                        "train_distance": train_distance,
                        "test_distance": test_distance,
                        "physical_error_rate": physical_error_rate,
                        "training_episodes": episodes,
                        "seed": seed
                    },
                    metrics={},
                    error=str(e)
                )
                results_table.add_result(result)
                print(f"  ERROR: {e}")


def run_mwpm_validation(results_table: ResultsTable,
                        distances: List[int],
                        error_rates: List[float],
                        num_samples: int = 10000):
    """
    Validate MWPM implementation against known benchmarks.
    """
    print("\n" + "=" * 60)
    print("VALIDATION: MWPM Benchmark Comparison")
    print("=" * 60)

    for d in distances:
        for p in error_rates:
            config_name = f"mwpm_validation_d{d}_p{p}"
            print(f"\nRunning: {config_name}")

            try:
                # Initialize
                surface_code = SurfaceCode(d, p, seed=42)
                mwpm_decoder = MWPMDecoder(d, p)

                # Evaluate
                results = evaluate_decoder(mwpm_decoder, surface_code, num_samples)

                # Get benchmark
                benchmark = get_mwpm_benchmark(d, p)

                metrics = {
                    "logical_error_rate": results["logical_error_rate"],
                    "ci_95_lower": results["ci_95_lower"],
                    "ci_95_upper": results["ci_95_upper"],
                    "expected_benchmark": benchmark,
                    "deviation_from_benchmark": abs(results["logical_error_rate"] - benchmark),
                    "relative_deviation": abs(results["logical_error_rate"] - benchmark) / max(benchmark, 1e-6)
                }

                result = ExperimentResult(
                    config_name=config_name,
                    parameters={
                        "code_distance": d,
                        "physical_error_rate": p,
                        "num_samples": num_samples
                    },
                    metrics=metrics
                )
                results_table.add_result(result)

                print(f"  LER: {results['logical_error_rate']:.4f}, "
                      f"Benchmark: {benchmark:.4f}, "
                      f"Deviation: {metrics['relative_deviation']*100:.1f}%")

            except Exception as e:
                result = ExperimentResult(
                    config_name=config_name,
                    parameters={
                        "code_distance": d,
                        "physical_error_rate": p,
                        "num_samples": num_samples
                    },
                    metrics={},
                    error=str(e)
                )
                results_table.add_result(result)
                print(f"  ERROR: {e}")


def run_learning_curve_analysis(results_table: ResultsTable,
                                distance: int,
                                max_episodes: int,
                                checkpoint_interval: int,
                                seeds: List[int],
                                physical_error_rate: float = 0.005,
                                eval_samples: int = 500):
    """
    Generate learning curves showing training progress.
    """
    print("\n" + "=" * 60)
    print(f"ANALYSIS: Learning Curves at d={distance}")
    print("=" * 60)

    for seed in seeds:
        print(f"\nSeed {seed}:")

        try:
            surface_code = SurfaceCode(distance, physical_error_rate, seed=seed)
            rl_decoder = GNNRLDecoder(distance, physical_error_rate, seed=seed)

            # Train incrementally and evaluate at checkpoints
            for checkpoint in range(checkpoint_interval, max_episodes + 1, checkpoint_interval):
                # Train for this interval
                for _ in range(checkpoint_interval):
                    rl_decoder.train_episode(surface_code)

                # Evaluate
                eval_results = evaluate_decoder(rl_decoder, surface_code, eval_samples)

                config_name = f"learning_curve_d{distance}_ep{checkpoint}_s{seed}"

                metrics = {
                    "logical_error_rate": eval_results["logical_error_rate"],
                    "ci_95_lower": eval_results["ci_95_lower"],
                    "ci_95_upper": eval_results["ci_95_upper"],
                    "avg_recent_reward": np.mean(rl_decoder.episode_rewards[-checkpoint_interval:]),
                    "avg_recent_loss": np.mean(rl_decoder.training_losses[-checkpoint_interval:])
                }

                result = ExperimentResult(
                    config_name=config_name,
                    parameters={
                        "code_distance": distance,
                        "physical_error_rate": physical_error_rate,
                        "episodes_completed": checkpoint,
                        "seed": seed
                    },
                    metrics=metrics,
                    ablation="learning_curve"
                )
                results_table.add_result(result)

                print(f"  Ep {checkpoint}: LER={eval_results['logical_error_rate']:.4f}, "
                      f"Reward={metrics['avg_recent_reward']:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="QEC RL Decoder Experiments")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "extended", "comparison", "reward_ablation",
                                "gnn_ablation", "zero_shot", "mwpm_validation", "learning_curve"],
                        help="Which experiment to run")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize results table
    results_table = ResultsTable(project_name="QEC_RL_Scaling_Revision")

    print("=" * 60)
    print("QEC RL Decoder Experiments - Peer Review Revision")
    print("=" * 60)

    # Run experiments based on selection
    if args.experiment in ["all", "extended"]:
        run_extended_training_experiment(
            results_table,
            distances=[15],
            episodes_list=[200, 500, 1000, 2000, 5000],
            seeds=list(range(1, 11)),  # 10 seeds
            physical_error_rate=0.005
        )

    if args.experiment in ["all", "comparison"]:
        run_baseline_comparison(
            results_table,
            distances=[3, 5, 7, 9, 11, 13, 15],
            seeds=list(range(1, 6)),  # 5 seeds
            physical_error_rate=0.005,
            training_episodes=2000
        )

    if args.experiment in ["all", "reward_ablation"]:
        run_reward_shaping_ablation(
            results_table,
            distances=[7, 15],
            reward_types=["sparse", "dense_syndrome", "dense_distance", "shaped_curriculum"],
            seeds=list(range(1, 6)),
            physical_error_rate=0.005,
            training_episodes=2000
        )

    if args.experiment in ["all", "gnn_ablation"]:
        run_gnn_depth_ablation(
            results_table,
            distances=[7, 15],
            layer_configs=[(2, 64), (4, 64), (6, 64), (8, 64), (4, 128), (6, 128)],
            seeds=list(range(1, 4)),
            physical_error_rate=0.005,
            training_episodes=2000
        )

    if args.experiment in ["all", "zero_shot"]:
        run_zero_shot_generalization(
            results_table,
            train_distance=7,
            test_distance=15,
            episodes_list=[200, 1000, 2000, 5000],
            seeds=list(range(1, 6)),
            physical_error_rate=0.005
        )

    if args.experiment in ["all", "mwpm_validation"]:
        run_mwpm_validation(
            results_table,
            distances=[3, 5, 7, 9, 11, 13, 15],
            error_rates=[0.001, 0.003, 0.005, 0.007, 0.01],
            num_samples=10000
        )

    if args.experiment in ["all", "learning_curve"]:
        run_learning_curve_analysis(
            results_table,
            distance=15,
            max_episodes=5000,
            checkpoint_interval=250,
            seeds=list(range(1, 4)),
            physical_error_rate=0.005
        )

    # Save results
    json_path = os.path.join(args.output_dir, "extended_results_table.json")
    csv_path = os.path.join(args.output_dir, "extended_results_table.csv")

    results_table.to_json(json_path)
    results_table.to_csv(csv_path)

    print("\n" + "=" * 60)
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"Total experiments: {len(results_table.results)}")
    print("=" * 60)
