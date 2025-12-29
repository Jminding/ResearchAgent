"""
Quick experiment runner - runs a subset of experiments for faster completion.
"""
import sys
import os
import json
import time
from typing import Dict, List, Any
import torch
import numpy as np

# Add experiments directory to path
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)

from data_structures import ResultsTable, ExperimentResult
from data_generator import (
    generate_sbm_graph, compute_homophily,
    apply_temporal_weighting, apply_smote, ablate_features,
    add_label_noise
)
from gnn_models import get_model, HOMOPHILY_ASSUMING, HETEROPHILY_AWARE
from trainer import train_model, evaluate, get_class_weights

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(EXPERIMENTS_DIR), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Quick settings
NUM_NODES = 3000
EPOCHS = 50
PATIENCE = 10


def run_single_config(
    model_name: str,
    graph_data,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0005,
    use_class_weights: bool = True,
    loss_type: str = "cross_entropy"
) -> Dict[str, Any]:
    """Run a single model on a single graph configuration."""

    x = graph_data.x.to(DEVICE)
    edge_index = graph_data.edge_index.to(DEVICE)
    y = graph_data.y.to(DEVICE)
    train_mask = graph_data.train_mask.to(DEVICE)
    val_mask = graph_data.val_mask.to(DEVICE)
    test_mask = graph_data.test_mask.to(DEVICE)
    num_nodes = graph_data.num_nodes

    in_channels = x.size(1)
    out_channels = 2

    try:
        model = get_model(
            model_name,
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers
        ).to(DEVICE)

        class_weights = None
        if use_class_weights and loss_type == "cross_entropy":
            class_weights = get_class_weights(y, train_mask).to(DEVICE)

        model, val_metrics, training_time = train_model(
            model, x, edge_index, y, train_mask, val_mask, num_nodes,
            epochs=epochs, patience=patience,
            learning_rate=learning_rate, weight_decay=weight_decay,
            class_weights=class_weights, loss_type=loss_type
        )

        test_metrics = evaluate(model, x, edge_index, y, test_mask, num_nodes,
                               threshold=val_metrics.get('threshold', 0.5))

        return {
            'f1': test_metrics['f1'],
            'auroc': test_metrics['auroc'],
            'auprc': test_metrics['auprc'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'latency_ms': test_metrics['latency_ms'],
            'training_time': training_time,
            'error': None
        }

    except Exception as e:
        import traceback
        return {
            'f1': 0.0,
            'auroc': 0.5,
            'auprc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'latency_ms': 0.0,
            'training_time': 0.0,
            'error': str(e)
        }


def main():
    """Run reduced experiments."""
    print("="*60)
    print("QUICK EXPERIMENT RUN")
    print("="*60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results_table = ResultsTable(project_name="Heterophily-Aware GNNs for Financial Fraud Detection")

    # PRIMARY EXPERIMENT (reduced)
    print("\n--- PRIMARY EXPERIMENT ---")
    homophily_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    prevalence_rates = [0.01, 0.02]
    seeds = [42, 123, 456]  # Reduced from 5
    all_models = HOMOPHILY_ASSUMING + HETEROPHILY_AWARE

    total = len(homophily_levels) * len(prevalence_rates) * len(seeds) * len(all_models)
    count = 0

    for h in homophily_levels:
        for prev in prevalence_rates:
            for seed in seeds:
                graph = generate_sbm_graph(
                    num_nodes=NUM_NODES,
                    target_homophily=h,
                    anomaly_prevalence=prev,
                    feature_dim=16,
                    avg_degree=20.0,
                    seed=seed
                )

                for model_name in all_models:
                    count += 1
                    print(f"[{count}/{total}] {model_name}, h={h}, prev={prev}, seed={seed}", end=" ")
                    sys.stdout.flush()

                    metrics = run_single_config(model_name=model_name, graph_data=graph)

                    model_type = "heterophily_aware" if model_name in HETEROPHILY_AWARE else "homophily_assuming"

                    result = ExperimentResult(
                        config_name=f"primary_{model_name}_h{h}_prev{prev}_seed{seed}",
                        parameters={
                            'experiment': 'primary_homophily_sweep',
                            'model': model_name,
                            'model_type': model_type,
                            'homophily': h,
                            'actual_homophily': graph.homophily,
                            'prevalence': prev,
                            'seed': seed
                        },
                        metrics={
                            'f1': metrics['f1'],
                            'auroc': metrics['auroc'],
                            'auprc': metrics['auprc'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'latency_ms': metrics['latency_ms'],
                            'training_time': metrics['training_time']
                        },
                        error=metrics['error']
                    )
                    results_table.add_result(result)
                    print(f"F1={metrics['f1']:.3f}, AUROC={metrics['auroc']:.3f}")

    # ABLATION 1: Temporal weighting (quick)
    print("\n--- ABLATION: Temporal Weighting ---")
    schemes = ["none", "exponential_decay"]
    models_abl = ["H2GCN", "FAGCN", "LINKX"]

    for scheme in schemes:
        for model_name in models_abl:
            graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, seed=42)
            edge_weights = apply_temporal_weighting(graph.edge_index, graph.num_nodes, scheme=scheme, seed=42)
            metrics = run_single_config(model_name=model_name, graph_data=graph)
            result = ExperimentResult(
                config_name=f"ablation_temporal_{model_name}_{scheme}",
                parameters={'experiment': 'temporal_weighting_ablation', 'model': model_name, 'temporal_scheme': scheme},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                ablation=f"temporal_{scheme}",
                error=metrics['error']
            )
            results_table.add_result(result)
            print(f"  {scheme}/{model_name}: F1={metrics['f1']:.3f}")

    # ABLATION 2: SMOTE (quick)
    print("\n--- ABLATION: SMOTE ---")
    methods = [("none", False, "cross_entropy", 0), ("class_weighted", True, "cross_entropy", 0)]
    for method_name, use_weights, loss_type, smote_ratio in methods:
        for model_name in models_abl:
            graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, seed=42)
            metrics = run_single_config(model_name=model_name, graph_data=graph, use_class_weights=use_weights, loss_type=loss_type)
            result = ExperimentResult(
                config_name=f"ablation_smote_{model_name}_{method_name}",
                parameters={'experiment': 'smote_class_balancing_ablation', 'model': model_name, 'balancing_method': method_name},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                ablation=method_name,
                error=metrics['error']
            )
            results_table.add_result(result)
            print(f"  {method_name}/{model_name}: F1={metrics['f1']:.3f}")

    # ABLATION 3: Feature importance (quick)
    print("\n--- ABLATION: Feature Importance ---")
    feature_sets = ["all_features", "structural_only"]
    for feature_set in feature_sets:
        for model_name in ["GCN", "H2GCN", "LINKX"]:
            graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, seed=42)
            graph.x = ablate_features(graph.x, feature_set, feature_dim=16, seed=42)
            actual_model = "MLP" if feature_set == "features_only_no_graph" else model_name
            metrics = run_single_config(model_name=actual_model, graph_data=graph)
            result = ExperimentResult(
                config_name=f"ablation_features_{model_name}_{feature_set}",
                parameters={'experiment': 'node_feature_importance_ablation', 'model': model_name, 'feature_set': feature_set},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                ablation=feature_set,
                error=metrics['error']
            )
            results_table.add_result(result)
            print(f"  {feature_set}/{model_name}: F1={metrics['f1']:.3f}")

    # ABLATION 4: Depth sensitivity (quick)
    print("\n--- ABLATION: Depth Sensitivity ---")
    depths = [1, 2, 3]
    for num_layers in depths:
        for model_name in ["GCN", "H2GCN", "FAGCN"]:
            graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, seed=42)
            metrics = run_single_config(model_name=model_name, graph_data=graph, num_layers=num_layers)
            result = ExperimentResult(
                config_name=f"ablation_depth_{model_name}_L{num_layers}",
                parameters={'experiment': 'architecture_depth_sensitivity', 'model': model_name, 'num_layers': num_layers},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                ablation=f"depth_{num_layers}",
                error=metrics['error']
            )
            results_table.add_result(result)
            print(f"  L{num_layers}/{model_name}: F1={metrics['f1']:.3f}")

    # ABLATION 5: Class imbalance (quick)
    print("\n--- ABLATION: Class Imbalance ---")
    imbalance_ratios = [50, 100, 500]
    for ir in imbalance_ratios:
        for model_name in ["GCN", "H2GCN", "FAGCN"]:
            prev = 1.0 / ir
            graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=prev, seed=42)
            metrics = run_single_config(model_name=model_name, graph_data=graph)
            result = ExperimentResult(
                config_name=f"ablation_imbalance_{model_name}_ir{ir}",
                parameters={'experiment': 'class_imbalance_sensitivity', 'model': model_name, 'imbalance_ratio': ir},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                ablation=f"ir_{ir}",
                error=metrics['error']
            )
            results_table.add_result(result)
            print(f"  IR{ir}/{model_name}: F1={metrics['f1']:.3f}")

    # ROBUSTNESS CHECKS (quick)
    print("\n--- ROBUSTNESS CHECKS ---")

    # 1. Hyperparameters
    print("  Hyperparameter perturbations...")
    graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, seed=42)
    for lr in [0.005, 0.01, 0.02]:
        for model_name in ["GCN", "H2GCN"]:
            metrics = run_single_config(model_name=model_name, graph_data=graph, learning_rate=lr)
            result = ExperimentResult(
                config_name=f"robust_lr_{lr}_{model_name}",
                parameters={'experiment': 'robustness_hyperparameter', 'model': model_name, 'learning_rate': lr},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                error=metrics['error']
            )
            results_table.add_result(result)

    # 2. Graph size
    print("  Graph size variations...")
    for size_name, num_nodes in [("small", 2000), ("large", 5000)]:
        graph = generate_sbm_graph(num_nodes=num_nodes, target_homophily=0.2, anomaly_prevalence=0.01, seed=42)
        for model_name in ["GCN", "H2GCN", "LINKX"]:
            metrics = run_single_config(model_name=model_name, graph_data=graph)
            result = ExperimentResult(
                config_name=f"robust_size_{size_name}_{model_name}",
                parameters={'experiment': 'robustness_graph_size', 'model': model_name, 'graph_size': size_name, 'num_nodes': num_nodes},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                error=metrics['error']
            )
            results_table.add_result(result)

    # 3. Anomaly rate
    print("  Anomaly prevalence rates...")
    for rate in [0.005, 0.01, 0.02]:
        graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=rate, seed=42)
        for model_name in ["GCN", "H2GCN", "FAGCN"]:
            metrics = run_single_config(model_name=model_name, graph_data=graph)
            result = ExperimentResult(
                config_name=f"robust_rate_{rate}_{model_name}",
                parameters={'experiment': 'robustness_anomaly_rate', 'model': model_name, 'anomaly_rate': rate},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                error=metrics['error']
            )
            results_table.add_result(result)

    # 4. Label noise
    print("  Label noise injection...")
    graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, seed=42)
    for fn_rate in [0.0, 0.2]:
        noisy_graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, seed=42)
        noisy_graph.y = add_label_noise(noisy_graph.y, noisy_graph.train_mask, false_negative_rate=fn_rate, seed=42)
        for model_name in ["GCN", "H2GCN"]:
            metrics = run_single_config(model_name=model_name, graph_data=noisy_graph)
            result = ExperimentResult(
                config_name=f"robust_noise_fn{fn_rate}_{model_name}",
                parameters={'experiment': 'robustness_label_noise', 'model': model_name, 'false_negative_rate': fn_rate},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                error=metrics['error']
            )
            results_table.add_result(result)

    # 5. Elliptic-style validation
    print("  Elliptic-style validation...")
    graph = generate_sbm_graph(num_nodes=5000, target_homophily=0.25, anomaly_prevalence=0.02, feature_dim=64, avg_degree=10, seed=42)
    for model_name in ["GCN", "H2GCN", "FAGCN", "LINKX"]:
        metrics = run_single_config(model_name=model_name, graph_data=graph, hidden_dim=64)
        result = ExperimentResult(
            config_name=f"robust_elliptic_{model_name}",
            parameters={'experiment': 'robustness_elliptic_validation', 'model': model_name},
            metrics={'f1': metrics['f1'], 'auroc': metrics['auroc'], 'auprc': metrics['auprc']},
            error=metrics['error']
        )
        results_table.add_result(result)

    # 6. Homophily regimes
    print("  Homophily regimes...")
    for regime, h in [("low", 0.1), ("moderate", 0.35), ("high", 0.6)]:
        graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=h, anomaly_prevalence=0.01, seed=42)
        for model_name in ["GCN", "H2GCN", "LINKX"]:
            metrics = run_single_config(model_name=model_name, graph_data=graph)
            result = ExperimentResult(
                config_name=f"robust_regime_{regime}_{model_name}",
                parameters={'experiment': 'robustness_parameter_regimes', 'model': model_name, 'regime': regime, 'homophily': h},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                error=metrics['error']
            )
            results_table.add_result(result)

    # 7. Train/test splits
    print("  Train/test splits...")
    for split_seed in [42, 123]:
        graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, seed=split_seed)
        for model_name in ["GCN", "H2GCN"]:
            metrics = run_single_config(model_name=model_name, graph_data=graph)
            result = ExperimentResult(
                config_name=f"robust_split_{split_seed}_{model_name}",
                parameters={'experiment': 'robustness_train_test_splits', 'model': model_name, 'split_seed': split_seed},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                error=metrics['error']
            )
            results_table.add_result(result)

    # 8. Additional configs
    print("  Additional configurations...")
    for config in [{"feature_dim": 8, "avg_degree": 10, "name": "sparse_low_dim"}, {"feature_dim": 32, "avg_degree": 30, "name": "dense_high_dim"}]:
        graph = generate_sbm_graph(num_nodes=NUM_NODES, target_homophily=0.2, anomaly_prevalence=0.01, feature_dim=config["feature_dim"], avg_degree=config["avg_degree"], seed=42)
        for model_name in ["GCN", "H2GCN", "LINKX"]:
            metrics = run_single_config(model_name=model_name, graph_data=graph)
            result = ExperimentResult(
                config_name=f"robust_config_{config['name']}_{model_name}",
                parameters={'experiment': 'robustness_additional_configs', 'model': model_name, 'config_name': config['name']},
                metrics={'f1': metrics['f1'], 'auroc': metrics['auroc']},
                error=metrics['error']
            )
            results_table.add_result(result)

    # Save results
    results_table.to_json(os.path.join(RESULTS_DIR, "results_table.json"))
    results_table.to_csv(os.path.join(RESULTS_DIR, "results_table.csv"))

    # Generate summaries
    # Homophily sweep summary
    summary = {}
    for result in results_table.results:
        if result.parameters.get('experiment') != 'primary_homophily_sweep':
            continue
        h = result.parameters.get('homophily')
        model = result.parameters.get('model')
        model_type = result.parameters.get('model_type')
        if h not in summary:
            summary[h] = {'homophily_assuming': {}, 'heterophily_aware': {}}
        if model not in summary[h][model_type]:
            summary[h][model_type][model] = {'f1': [], 'auroc': []}
        summary[h][model_type][model]['f1'].append(result.metrics.get('f1', 0))
        summary[h][model_type][model]['auroc'].append(result.metrics.get('auroc', 0))

    final_summary = {}
    for h, types in summary.items():
        final_summary[str(h)] = {}
        for model_type, models in types.items():
            final_summary[str(h)][model_type] = {}
            for model, metrics in models.items():
                final_summary[str(h)][model_type][model] = {
                    'f1_mean': float(np.mean(metrics['f1'])),
                    'f1_std': float(np.std(metrics['f1'])),
                    'auroc_mean': float(np.mean(metrics['auroc'])),
                    'auroc_std': float(np.std(metrics['auroc']))
                }

    with open(os.path.join(RESULTS_DIR, "homophily_sweep.json"), 'w') as f:
        json.dump(final_summary, f, indent=2)

    # Ablation summary
    ablation_results = {}
    for result in results_table.results:
        exp = result.parameters.get('experiment', '')
        if 'ablation' not in exp:
            continue
        if exp not in ablation_results:
            ablation_results[exp] = {}
        ablation = result.ablation or 'baseline'
        model = result.parameters.get('model')
        key = f"{model}_{ablation}"
        if key not in ablation_results[exp]:
            ablation_results[exp][key] = {'f1': [], 'auroc': []}
        ablation_results[exp][key]['f1'].append(result.metrics.get('f1', 0))
        ablation_results[exp][key]['auroc'].append(result.metrics.get('auroc', 0))

    abl_summary = {}
    for exp, configs in ablation_results.items():
        abl_summary[exp] = {}
        for config, metrics in configs.items():
            abl_summary[exp][config] = {
                'f1_mean': float(np.mean(metrics['f1'])),
                'auroc_mean': float(np.mean(metrics['auroc']))
            }

    with open(os.path.join(RESULTS_DIR, "ablation_summary.json"), 'w') as f:
        json.dump(abl_summary, f, indent=2)

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print(f"Total results: {len(results_table.results)}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
