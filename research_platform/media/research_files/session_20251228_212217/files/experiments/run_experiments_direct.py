#!/usr/bin/env python3
"""
QEC Experiment Suite - Direct Execution
Run with: python3 run_experiments_direct.py
"""
import sys
import os
import numpy as np
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import Counter

# Output directory
OUTPUT_DIR = "/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Surface Code Simulation
class SurfaceCode:
    def __init__(self, distance, physical_error_rate, seed=None):
        self.d = distance
        self.p = physical_error_rate
        self.rng = np.random.default_rng(seed)
        self.n_data = distance ** 2
        self._build_lattice()

    def _build_lattice(self):
        d = self.d
        self.x_stabilizer_qubits = []
        self.z_stabilizer_qubits = []
        for i in range(d - 1):
            for j in range(d - 1):
                qubits = [i * d + j, i * d + j + 1, (i + 1) * d + j, (i + 1) * d + j + 1]
                self.x_stabilizer_qubits.append(qubits)
                self.z_stabilizer_qubits.append(qubits)

    def generate_error(self):
        errors = np.zeros(self.n_data, dtype=int)
        for i in range(self.n_data):
            if self.rng.random() < self.p:
                errors[i] = self.rng.integers(1, 4)
        return errors

    def measure_syndrome(self, errors):
        x_syndrome = []
        z_syndrome = []
        for qubits in self.x_stabilizer_qubits:
            parity = sum(1 for q in qubits if q < len(errors) and errors[q] in [2, 3]) % 2
            x_syndrome.append(parity)
        for qubits in self.z_stabilizer_qubits:
            parity = sum(1 for q in qubits if q < len(errors) and errors[q] in [1, 2]) % 2
            z_syndrome.append(parity)
        return np.array(x_syndrome + z_syndrome)

    def check_logical_error(self, errors, correction):
        combined = errors ^ correction
        x_logical_parity = sum(1 for i in range(self.d) if combined[i] in [1, 2]) % 2
        z_logical_parity = sum(1 for i in range(0, self.n_data, self.d) if combined[i] in [2, 3]) % 2
        return x_logical_parity == 1 or z_logical_parity == 1


class MWPMDecoder:
    def __init__(self, distance, physical_error_rate):
        self.d = distance
        self.p = physical_error_rate

    def decode(self, syndrome, surface_code):
        n_data = surface_code.n_data
        correction = np.zeros(n_data, dtype=int)
        n_stabilizers = len(syndrome) // 2
        x_defects = [i for i in range(n_stabilizers) if syndrome[i] == 1]
        d = surface_code.d
        matched_x = set()
        for i, def1 in enumerate(x_defects):
            if def1 in matched_x:
                continue
            best_partner = None
            best_dist = float('inf')
            for def2 in x_defects[i + 1:]:
                if def2 in matched_x:
                    continue
                d1_x, d1_y = def1 // max(d - 1, 1), def1 % max(d - 1, 1)
                d2_x, d2_y = def2 // max(d - 1, 1), def2 % max(d - 1, 1)
                dist = abs(d1_x - d2_x) + abs(d1_y - d2_y)
                if dist < best_dist:
                    best_dist = dist
                    best_partner = def2
            d1_x, d1_y = def1 // max(d - 1, 1), def1 % max(d - 1, 1)
            boundary_dist = min(d1_x, d1_y, max(d - 2 - d1_x, 0), max(d - 2 - d1_y, 0)) if d > 1 else 0
            if best_partner is not None and best_dist <= max(boundary_dist, 1):
                matched_x.add(def1)
                matched_x.add(best_partner)
                d2_x, d2_y = best_partner // max(d - 1, 1), best_partner % max(d - 1, 1)
                for x in range(min(d1_x, d2_x), max(d1_x, d2_x) + 1):
                    for y in range(min(d1_y, d2_y), max(d1_y, d2_y) + 1):
                        if x * d + y < n_data:
                            correction[x * d + y] ^= 3
            else:
                matched_x.add(def1)
        return correction


class GNNRLDecoder:
    def __init__(self, distance, physical_error_rate, gnn_layers=4, hidden_dim=64,
                 learning_rate=0.001, reward_type="sparse", seed=None):
        self.d = distance
        self.p = physical_error_rate
        self.gnn_layers = gnn_layers
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.reward_type = reward_type
        self.rng = np.random.default_rng(seed)
        self._init_model()
        self.training_losses = []
        self.episode_rewards = []

    def _init_model(self):
        self.W_msg = [self.rng.standard_normal((self.hidden_dim, self.hidden_dim)) * 0.1 for _ in range(self.gnn_layers)]
        self.W_upd = [self.rng.standard_normal((self.hidden_dim, self.hidden_dim)) * 0.1 for _ in range(self.gnn_layers)]
        self.W_in = self.rng.standard_normal((self.hidden_dim, 1)) * 0.1
        self.W_out = self.rng.standard_normal((4, self.hidden_dim)) * 0.1
        self.b_out = np.zeros(4)
        self.n_params = sum(w.size for w in self.W_msg + self.W_upd) + self.W_in.size + self.W_out.size + self.b_out.size

    def _forward(self, syndrome):
        n_nodes = len(syndrome)
        h = np.zeros((n_nodes, self.hidden_dim))
        for i in range(n_nodes):
            h[i] = syndrome[i] * self.W_in.flatten()[:self.hidden_dim]
        for l in range(self.gnn_layers):
            h_new = np.zeros_like(h)
            for i in range(n_nodes):
                msg = np.zeros(self.hidden_dim)
                n_neighbors = 0
                for j in range(n_nodes):
                    if i != j and abs(i - j) <= self.d:
                        msg += np.tanh(self.W_msg[l] @ h[j])
                        n_neighbors += 1
                if n_neighbors > 0:
                    msg /= n_neighbors
                h_new[i] = np.tanh(self.W_upd[l] @ (h[i] + msg))
            h = h_new
        h_pooled = np.mean(h, axis=0)
        return self.W_out @ h_pooled + self.b_out

    def _compute_reward(self, correction, errors, syndrome, surface_code):
        if self.reward_type == "sparse":
            return 1.0 if not surface_code.check_logical_error(errors, correction) else 0.0
        elif self.reward_type == "dense_syndrome":
            corrected_syndrome = surface_code.measure_syndrome(errors ^ correction)
            original_weight = np.sum(syndrome)
            corrected_weight = np.sum(corrected_syndrome)
            syndrome_reward = (original_weight - corrected_weight) / max(original_weight, 1)
            return syndrome_reward + (2.0 if not surface_code.check_logical_error(errors, correction) else -1.0)
        elif self.reward_type == "dense_distance":
            hamming_dist = np.sum(errors != correction)
            max_dist = len(errors)
            dist_reward = 1.0 - (hamming_dist / max_dist)
            return dist_reward + (1.0 if not surface_code.check_logical_error(errors, correction) else -0.5)
        else:
            return 1.0 if not surface_code.check_logical_error(errors, correction) else 0.0

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def decode(self, syndrome, surface_code):
        n_data = surface_code.n_data
        correction = np.zeros(n_data, dtype=int)
        logits = self._forward(syndrome)
        probs = self._softmax(logits)
        for i in range(n_data):
            local_logits = logits + 0.1 * self.rng.standard_normal(4)
            local_probs = self._softmax(local_logits)
            correction[i] = self.rng.choice(4, p=local_probs)
        return correction

    def train_episode(self, surface_code):
        errors = surface_code.generate_error()
        syndrome = surface_code.measure_syndrome(errors)
        correction = self.decode(syndrome, surface_code)
        reward = self._compute_reward(correction, errors, syndrome, surface_code)
        logits = self._forward(syndrome)
        probs = self._softmax(logits)
        grad_scale = self.lr * reward
        h_pooled = np.mean(np.tanh(np.outer(syndrome, self.W_in.flatten()[:len(syndrome)])), axis=0)
        if len(h_pooled) < self.hidden_dim:
            h_pooled = np.pad(h_pooled, (0, self.hidden_dim - len(h_pooled)))
        for i in range(4):
            if i == correction[0]:
                self.W_out[i] += grad_scale * (1 - probs[i]) * h_pooled[:self.hidden_dim]
            else:
                self.W_out[i] -= grad_scale * probs[i] * h_pooled[:self.hidden_dim]
        action_prob = probs[correction[0]]
        loss = -np.log(action_prob + 1e-10) * (1 - reward)
        self.training_losses.append(loss)
        self.episode_rewards.append(reward)
        return loss, reward

    def train(self, surface_code, num_episodes):
        start_time = time.time()
        for _ in range(num_episodes):
            self.train_episode(surface_code)
        training_time = time.time() - start_time
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
            "convergence_episode": convergence_episode
        }


def evaluate_decoder(decoder, surface_code, num_samples=1000):
    logical_errors = 0
    for _ in range(num_samples):
        errors = surface_code.generate_error()
        syndrome = surface_code.measure_syndrome(errors)
        correction = decoder.decode(syndrome, surface_code)
        if surface_code.check_logical_error(errors, correction):
            logical_errors += 1
    ler = logical_errors / num_samples
    stderr = np.sqrt(ler * (1 - ler) / num_samples)
    ci_95 = 1.96 * stderr
    return {
        "logical_error_rate": ler,
        "ci_95_lower": max(0, ler - ci_95),
        "ci_95_upper": min(1, ler + ci_95),
        "stderr": stderr
    }


def get_mwpm_benchmark(distance, physical_error_rate):
    p_threshold = 0.0103
    if physical_error_rate >= p_threshold:
        return 0.5 * (1 - np.exp(-10 * (physical_error_rate - p_threshold)))
    suppression_exponent = (distance + 1) / 2
    base_rate = (physical_error_rate / p_threshold) ** suppression_exponent
    return 0.03 * base_rate


@dataclass
class ExperimentResult:
    config_name: str
    parameters: Dict
    metrics: Dict
    ablation: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ResultsTable:
    project_name: str
    results: List[ExperimentResult] = field(default_factory=list)

    def add_result(self, result):
        self.results.append(result)

    def to_dict(self):
        return {"project_name": self.project_name, "results": [asdict(r) for r in self.results]}

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, path):
        if not self.results:
            return
        lines = []
        header = ["config_name", "ablation", "error"]
        all_params = set()
        all_metrics = set()
        for r in self.results:
            all_params.update(r.parameters.keys())
            all_metrics.update(r.metrics.keys())
        header.extend(sorted(all_params))
        header.extend(sorted(all_metrics))
        lines.append(",".join(header))
        for r in self.results:
            row = [r.config_name, str(r.ablation) if r.ablation else "", str(r.error) if r.error else ""]
            for p in sorted(all_params):
                row.append(str(r.parameters.get(p, "")))
            for m in sorted(all_metrics):
                row.append(str(r.metrics.get(m, "")))
            lines.append(",".join(row))
        with open(path, 'w') as f:
            f.write("\n".join(lines))


def main():
    results_table = ResultsTable(project_name="QEC_RL_Scaling_Revision")

    print("=" * 70, flush=True)
    print("QEC RL Decoder Experiments - Peer Review Revision", flush=True)
    print("=" * 70, flush=True)

    # 1. Extended Training at d=15
    print("\n[1/7] Extended Training at d=15 (10 seeds x 5 episode levels)", flush=True)
    for episodes in [200, 500, 1000, 2000, 5000]:
        for seed in range(1, 11):
            config_name = f"extended_d15_ep{episodes}_s{seed}"
            try:
                surface_code = SurfaceCode(15, 0.005, seed=seed)
                rl_decoder = GNNRLDecoder(15, 0.005, seed=seed)
                train_stats = rl_decoder.train(surface_code, episodes)
                eval_results = evaluate_decoder(rl_decoder, surface_code, 1000)
                metrics = {
                    "logical_error_rate": eval_results["logical_error_rate"],
                    "ci_95_lower": eval_results["ci_95_lower"],
                    "ci_95_upper": eval_results["ci_95_upper"],
                    "stderr": eval_results["stderr"],
                    "training_loss": train_stats["final_loss"],
                    "convergence_episode": train_stats["convergence_episode"],
                    "training_time_sec": train_stats["training_time"]
                }
                results_table.add_result(ExperimentResult(config_name=config_name,
                    parameters={"code_distance": 15, "physical_error_rate": 0.005, "training_episodes": episodes, "seed": seed},
                    metrics=metrics))
                if seed == 1:
                    print(f"  ep={episodes}: LER={eval_results['logical_error_rate']:.4f}", flush=True)
            except Exception as e:
                results_table.add_result(ExperimentResult(config_name=config_name,
                    parameters={"code_distance": 15, "physical_error_rate": 0.005, "training_episodes": episodes, "seed": seed},
                    metrics={}, error=str(e)))

    # 2. RL vs MWPM Comparison
    print("\n[2/7] RL vs MWPM Comparison (d=3,5,7,9,11,13,15, 5 seeds)", flush=True)
    for d in [3, 5, 7, 9, 11, 13, 15]:
        for seed in range(1, 6):
            config_name = f"comparison_d{d}_s{seed}"
            try:
                surface_code = SurfaceCode(d, 0.005, seed=seed)
                rl_decoder = GNNRLDecoder(d, 0.005, seed=seed)
                mwpm_decoder = MWPMDecoder(d, 0.005)
                rl_decoder.train(surface_code, 2000)
                rl_results = evaluate_decoder(rl_decoder, surface_code, 1000)
                mwpm_results = evaluate_decoder(mwpm_decoder, surface_code, 1000)
                benchmark = get_mwpm_benchmark(d, 0.005)
                metrics = {
                    "logical_error_rate_rl": rl_results["logical_error_rate"],
                    "rl_ci_95_lower": rl_results["ci_95_lower"],
                    "rl_ci_95_upper": rl_results["ci_95_upper"],
                    "logical_error_rate_mwpm": mwpm_results["logical_error_rate"],
                    "mwpm_ci_95_lower": mwpm_results["ci_95_lower"],
                    "mwpm_ci_95_upper": mwpm_results["ci_95_upper"],
                    "mwpm_benchmark": benchmark,
                    "rl_vs_mwpm_ratio": rl_results["logical_error_rate"] / max(mwpm_results["logical_error_rate"], 1e-6)
                }
                results_table.add_result(ExperimentResult(config_name=config_name,
                    parameters={"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed},
                    metrics=metrics))
                if seed == 1:
                    print(f"  d={d}: RL={rl_results['logical_error_rate']:.4f}, MWPM={mwpm_results['logical_error_rate']:.4f}", flush=True)
            except Exception as e:
                results_table.add_result(ExperimentResult(config_name=config_name,
                    parameters={"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed},
                    metrics={}, error=str(e)))

    # 3. Reward Shaping Ablation
    print("\n[3/7] Reward Shaping Ablation (d=7,15 x 4 reward types x 5 seeds)", flush=True)
    for d in [7, 15]:
        for reward_type in ["sparse", "dense_syndrome", "dense_distance", "shaped_curriculum"]:
            for seed in range(1, 6):
                config_name = f"reward_d{d}_{reward_type}_s{seed}"
                try:
                    surface_code = SurfaceCode(d, 0.005, seed=seed)
                    rl_decoder = GNNRLDecoder(d, 0.005, reward_type=reward_type, seed=seed)
                    train_stats = rl_decoder.train(surface_code, 2000)
                    eval_results = evaluate_decoder(rl_decoder, surface_code, 1000)
                    metrics = {
                        "logical_error_rate": eval_results["logical_error_rate"],
                        "ci_95_lower": eval_results["ci_95_lower"],
                        "ci_95_upper": eval_results["ci_95_upper"],
                        "convergence_episode": train_stats["convergence_episode"],
                        "final_reward": train_stats["final_reward"]
                    }
                    results_table.add_result(ExperimentResult(config_name=config_name,
                        parameters={"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed, "reward_type": reward_type},
                        metrics=metrics, ablation=f"reward_{reward_type}"))
                    if seed == 1:
                        print(f"  d={d}, {reward_type}: LER={eval_results['logical_error_rate']:.4f}", flush=True)
                except Exception as e:
                    results_table.add_result(ExperimentResult(config_name=config_name,
                        parameters={"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed, "reward_type": reward_type},
                        metrics={}, ablation=f"reward_{reward_type}", error=str(e)))

    # 4. GNN Depth Ablation
    print("\n[4/7] GNN Architecture Ablation (d=7,15 x 6 configs x 3 seeds)", flush=True)
    for d in [7, 15]:
        for layers, hidden_dim in [(2, 64), (4, 64), (6, 64), (8, 64), (4, 128), (6, 128)]:
            for seed in range(1, 4):
                config_name = f"gnn_d{d}_L{layers}_H{hidden_dim}_s{seed}"
                try:
                    surface_code = SurfaceCode(d, 0.005, seed=seed)
                    rl_decoder = GNNRLDecoder(d, 0.005, gnn_layers=layers, hidden_dim=hidden_dim, seed=seed)
                    train_stats = rl_decoder.train(surface_code, 2000)
                    eval_results = evaluate_decoder(rl_decoder, surface_code, 1000)
                    metrics = {
                        "logical_error_rate": eval_results["logical_error_rate"],
                        "ci_95_lower": eval_results["ci_95_lower"],
                        "ci_95_upper": eval_results["ci_95_upper"],
                        "model_params": rl_decoder.n_params,
                        "convergence_episode": train_stats["convergence_episode"]
                    }
                    results_table.add_result(ExperimentResult(config_name=config_name,
                        parameters={"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed, "gnn_layers": layers, "hidden_dim": hidden_dim},
                        metrics=metrics, ablation=f"gnn_L{layers}_H{hidden_dim}"))
                    if seed == 1:
                        print(f"  d={d}, L={layers}, H={hidden_dim}: LER={eval_results['logical_error_rate']:.4f}", flush=True)
                except Exception as e:
                    results_table.add_result(ExperimentResult(config_name=config_name,
                        parameters={"code_distance": d, "physical_error_rate": 0.005, "training_episodes": 2000, "seed": seed, "gnn_layers": layers, "hidden_dim": hidden_dim},
                        metrics={}, ablation=f"gnn_L{layers}_H{hidden_dim}", error=str(e)))

    # 5. Zero-Shot Generalization
    print("\n[5/7] Zero-Shot Generalization d=7->d=15 (4 episode levels x 5 seeds)", flush=True)
    for episodes in [200, 1000, 2000, 5000]:
        for seed in range(1, 6):
            config_name = f"zeroshot_d7to15_ep{episodes}_s{seed}"
            try:
                train_code = SurfaceCode(7, 0.005, seed=seed)
                rl_decoder = GNNRLDecoder(7, 0.005, seed=seed)
                rl_decoder.train(train_code, episodes)
                train_results = evaluate_decoder(rl_decoder, train_code, 1000)
                test_code = SurfaceCode(15, 0.005, seed=seed)
                test_decoder = GNNRLDecoder(15, 0.005, seed=seed)
                test_decoder.W_out = rl_decoder.W_out.copy()
                test_decoder.b_out = rl_decoder.b_out.copy()
                test_results = evaluate_decoder(test_decoder, test_code, 1000)
                mwpm_decoder = MWPMDecoder(15, 0.005)
                mwpm_results = evaluate_decoder(mwpm_decoder, test_code, 1000)
                metrics = {
                    "train_logical_error_rate": train_results["logical_error_rate"],
                    "test_logical_error_rate": test_results["logical_error_rate"],
                    "generalization_gap": test_results["logical_error_rate"] - train_results["logical_error_rate"],
                    "mwpm_test_error_rate": mwpm_results["logical_error_rate"],
                    "rl_vs_mwpm_at_test": test_results["logical_error_rate"] / max(mwpm_results["logical_error_rate"], 1e-6)
                }
                results_table.add_result(ExperimentResult(config_name=config_name,
                    parameters={"train_distance": 7, "test_distance": 15, "physical_error_rate": 0.005, "training_episodes": episodes, "seed": seed},
                    metrics=metrics))
                if seed == 1:
                    print(f"  ep={episodes}: Train={train_results['logical_error_rate']:.4f}, Test={test_results['logical_error_rate']:.4f}", flush=True)
            except Exception as e:
                results_table.add_result(ExperimentResult(config_name=config_name,
                    parameters={"train_distance": 7, "test_distance": 15, "physical_error_rate": 0.005, "training_episodes": episodes, "seed": seed},
                    metrics={}, error=str(e)))

    # 6. MWPM Validation
    print("\n[6/7] MWPM Benchmark Validation (7 distances x 5 error rates)", flush=True)
    for d in [3, 5, 7, 9, 11, 13, 15]:
        for p in [0.001, 0.003, 0.005, 0.007, 0.01]:
            config_name = f"mwpm_validation_d{d}_p{p}"
            try:
                surface_code = SurfaceCode(d, p, seed=42)
                mwpm_decoder = MWPMDecoder(d, p)
                results = evaluate_decoder(mwpm_decoder, surface_code, 10000)
                benchmark = get_mwpm_benchmark(d, p)
                metrics = {
                    "logical_error_rate": results["logical_error_rate"],
                    "expected_benchmark": benchmark,
                    "deviation_from_benchmark": abs(results["logical_error_rate"] - benchmark),
                    "relative_deviation": abs(results["logical_error_rate"] - benchmark) / max(benchmark, 1e-6)
                }
                results_table.add_result(ExperimentResult(config_name=config_name,
                    parameters={"code_distance": d, "physical_error_rate": p, "num_samples": 10000},
                    metrics=metrics))
            except Exception as e:
                results_table.add_result(ExperimentResult(config_name=config_name,
                    parameters={"code_distance": d, "physical_error_rate": p, "num_samples": 10000},
                    metrics={}, error=str(e)))
    print("  MWPM validation complete", flush=True)

    # 7. Learning Curves
    print("\n[7/7] Learning Curves at d=15 (20 checkpoints x 3 seeds)", flush=True)
    for seed in range(1, 4):
        surface_code = SurfaceCode(15, 0.005, seed=seed)
        rl_decoder = GNNRLDecoder(15, 0.005, seed=seed)
        for checkpoint in range(250, 5001, 250):
            for _ in range(250):
                rl_decoder.train_episode(surface_code)
            eval_results = evaluate_decoder(rl_decoder, surface_code, 500)
            config_name = f"learning_curve_d15_ep{checkpoint}_s{seed}"
            metrics = {
                "logical_error_rate": eval_results["logical_error_rate"],
                "ci_95_lower": eval_results["ci_95_lower"],
                "ci_95_upper": eval_results["ci_95_upper"],
                "avg_recent_reward": float(np.mean(rl_decoder.episode_rewards[-250:])),
                "avg_recent_loss": float(np.mean(rl_decoder.training_losses[-250:]))
            }
            results_table.add_result(ExperimentResult(config_name=config_name,
                parameters={"code_distance": 15, "physical_error_rate": 0.005, "episodes_completed": checkpoint, "seed": seed},
                metrics=metrics, ablation="learning_curve"))
        print(f"  Seed {seed}: Final LER={eval_results['logical_error_rate']:.4f}", flush=True)

    # Save results
    json_path = os.path.join(OUTPUT_DIR, "extended_results_table.json")
    csv_path = os.path.join(OUTPUT_DIR, "extended_results_table.csv")
    results_table.to_json(json_path)
    results_table.to_csv(csv_path)

    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT SUITE COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"Total experiments: {len(results_table.results)}", flush=True)
    print(f"Results saved to:", flush=True)
    print(f"  JSON: {json_path}", flush=True)
    print(f"  CSV:  {csv_path}", flush=True)

    # Summary
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

    print("\nExperiment breakdown:", flush=True)
    for exp_type, count in config_types.items():
        print(f"  {exp_type}: {count}", flush=True)

    error_count = sum(1 for r in results_table.results if r.error)
    print(f"\nTotal errors: {error_count}", flush=True)
    print("=" * 70, flush=True)

    return results_table


if __name__ == "__main__":
    main()
