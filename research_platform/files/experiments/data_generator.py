"""
Synthetic graph generation using Stochastic Block Model (SBM) with controlled homophily.
"""
import torch
import numpy as np
import networkx as nx
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class GraphData:
    """Container for graph data."""
    x: torch.Tensor           # Node features [N, F]
    edge_index: torch.Tensor  # Edge list [2, E]
    y: torch.Tensor           # Node labels [N]
    train_mask: torch.Tensor  # Training mask [N]
    val_mask: torch.Tensor    # Validation mask [N]
    test_mask: torch.Tensor   # Test mask [N]
    num_nodes: int
    num_edges: int
    homophily: float          # Actual computed homophily


def compute_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute edge homophily ratio: fraction of edges connecting same-class nodes.
    h = |{(u,v) in E : y_u = y_v}| / |E|
    """
    src, dst = edge_index
    same_class = (y[src] == y[dst]).float()
    return same_class.mean().item()


def generate_sbm_graph(
    num_nodes: int = 10000,
    target_homophily: float = 0.3,
    anomaly_prevalence: float = 0.01,
    feature_dim: int = 16,
    avg_degree: float = 20.0,
    feature_signal_strength: float = 0.15,  # Reduced from 0.5 to make task harder
    seed: int = 42
) -> GraphData:
    """
    Generate a Stochastic Block Model graph with controlled homophily for fraud detection.

    Args:
        num_nodes: Total number of nodes
        target_homophily: Target edge homophily ratio h in [0, 1]
        anomaly_prevalence: Fraction of nodes that are anomalous/fraud (class 1)
        feature_dim: Node feature dimension
        avg_degree: Average node degree
        feature_signal_strength: How separable features are (std units) - lower = harder
        seed: Random seed for reproducibility

    Returns:
        GraphData object with features, edges, labels, and masks
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Compute number of nodes per class
    num_fraud = max(1, int(num_nodes * anomaly_prevalence))
    num_normal = num_nodes - num_fraud

    # Create labels
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[:num_fraud] = 1  # First num_fraud nodes are fraudulent

    n1, n2 = num_fraud, num_normal
    n = num_nodes

    # Total expected edges = avg_degree * n / 2
    total_edges = avg_degree * n / 2

    # Fraction of potential intra-class edges
    intra_potential = (n1 * (n1 - 1) + n2 * (n2 - 1)) / 2
    inter_potential = n1 * n2

    if target_homophily < 0.5:
        # Heterophilic: more inter-class edges
        ratio = target_homophily / (1 - target_homophily) if target_homophily < 1 else 1000

        if intra_potential > 0:
            p_ratio = ratio * inter_potential / intra_potential
        else:
            p_ratio = ratio

        denom = p_ratio * intra_potential + inter_potential
        if denom > 0:
            p_out = total_edges / denom
        else:
            p_out = 0.01
        p_in = p_ratio * p_out

    else:
        # Homophilic: more intra-class edges
        ratio = (1 - target_homophily) / target_homophily if target_homophily > 0 else 0.001

        if inter_potential > 0:
            p_ratio = ratio * intra_potential / inter_potential
        else:
            p_ratio = 0.01

        denom = intra_potential + p_ratio * inter_potential
        if denom > 0:
            p_in = total_edges / denom
        else:
            p_in = 0.01
        p_out = p_ratio * p_in

    # Clamp probabilities
    p_in = min(max(p_in, 0.001), 0.99)
    p_out = min(max(p_out, 0.001), 0.99)

    # Create SBM graph using NetworkX
    sizes = [num_fraud, num_normal]
    probs = [[p_in, p_out], [p_out, p_in]]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    # Convert to edge index
    edges = list(G.edges())
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        # Fallback: create random edges
        edge_index = torch.randint(0, num_nodes, (2, int(avg_degree * num_nodes)))

    # Compute actual homophily
    actual_homophily = compute_homophily(edge_index, y)

    # Generate node features with weak class signal
    # All nodes: base features ~ N(0, 1)
    x = torch.randn(num_nodes, feature_dim)

    # Add subtle signal to fraud nodes in a subset of features
    # Use fewer dimensions and smaller shift for more challenging detection
    signal_dims = max(1, feature_dim // 4)  # Only 25% of features have signal
    x[:num_fraud, :signal_dims] += feature_signal_strength

    # Add correlated noise to make pure feature-based detection harder
    noise_scale = 0.3
    x += noise_scale * torch.randn_like(x)

    # Shuffle nodes to mix classes
    perm = torch.randperm(num_nodes)
    x = x[perm]
    y = y[perm]

    # Re-sort edge_index according to permutation
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(num_nodes)
    edge_index = inv_perm[edge_index]

    # Create train/val/test masks (70/15/15 split)
    # Use stratified sampling to ensure fraud nodes in all splits
    fraud_idx = (y == 1).nonzero().squeeze()
    normal_idx = (y == 0).nonzero().squeeze()

    if fraud_idx.dim() == 0:
        fraud_idx = fraud_idx.unsqueeze(0)
    if normal_idx.dim() == 0:
        normal_idx = normal_idx.unsqueeze(0)

    # Shuffle
    fraud_perm = torch.randperm(len(fraud_idx))
    normal_perm = torch.randperm(len(normal_idx))

    n_fraud_train = int(0.7 * len(fraud_idx))
    n_fraud_val = int(0.15 * len(fraud_idx))
    n_normal_train = int(0.7 * len(normal_idx))
    n_normal_val = int(0.15 * len(normal_idx))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Assign fraud nodes
    train_mask[fraud_idx[fraud_perm[:n_fraud_train]]] = True
    val_mask[fraud_idx[fraud_perm[n_fraud_train:n_fraud_train + n_fraud_val]]] = True
    test_mask[fraud_idx[fraud_perm[n_fraud_train + n_fraud_val:]]] = True

    # Assign normal nodes
    train_mask[normal_idx[normal_perm[:n_normal_train]]] = True
    val_mask[normal_idx[normal_perm[n_normal_train:n_normal_train + n_normal_val]]] = True
    test_mask[normal_idx[normal_perm[n_normal_train + n_normal_val:]]] = True

    return GraphData(
        x=x.float(),
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_edges=edge_index.size(1) // 2,
        homophily=actual_homophily
    )


def generate_experiment_graphs(
    homophily_levels: list = [0.1, 0.2, 0.3, 0.4, 0.5],
    prevalence_rates: list = [0.01, 0.02],
    seeds: list = [42, 123, 456, 789, 1000],
    num_nodes: int = 10000,
    feature_dim: int = 16,
    avg_degree: float = 20.0
) -> Dict[str, GraphData]:
    """
    Generate all graphs needed for the primary experiment.

    Returns dict with keys like "h0.1_prev0.01_seed42"
    """
    graphs = {}

    for h in homophily_levels:
        for prev in prevalence_rates:
            for seed in seeds:
                key = f"h{h}_prev{prev}_seed{seed}"
                graphs[key] = generate_sbm_graph(
                    num_nodes=num_nodes,
                    target_homophily=h,
                    anomaly_prevalence=prev,
                    feature_dim=feature_dim,
                    avg_degree=avg_degree,
                    seed=seed
                )

    return graphs


def apply_temporal_weighting(
    edge_index: torch.Tensor,
    num_nodes: int,
    scheme: str = "exponential_decay",
    decay_rate: float = 0.95,
    seed: int = 42
) -> torch.Tensor:
    """
    Apply temporal edge weighting to simulate transaction recency.

    Args:
        edge_index: Original edges [2, E]
        num_nodes: Number of nodes
        scheme: Weighting scheme ("none", "exponential_decay", "inverse_time", "recency_rank")
        decay_rate: Decay parameter for exponential scheme
        seed: Random seed

    Returns:
        Edge weights [E]
    """
    torch.manual_seed(seed)
    num_edges = edge_index.size(1)

    if scheme == "none":
        return torch.ones(num_edges)

    # Simulate timestamps (random order for synthetic data)
    timestamps = torch.rand(num_edges)  # Uniform [0, 1] as relative time

    if scheme == "exponential_decay":
        # Recent edges (t close to 1) get higher weight
        weights = decay_rate ** ((1 - timestamps) * 10)

    elif scheme == "inverse_time":
        # Weight = 1 / (1 + time_delta)
        time_delta = 1 - timestamps
        weights = 1.0 / (1 + time_delta * 10)

    elif scheme == "recency_rank":
        # Rank-based weighting
        ranks = timestamps.argsort().argsort().float()
        weights = (ranks + 1) / num_edges

    else:
        weights = torch.ones(num_edges)

    # Normalize to mean 1
    weights = weights / weights.mean()

    return weights


def apply_smote(
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    target_ratio: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply SMOTE-like oversampling to minority class (fraud).

    Args:
        x: Node features [N, F]
        y: Node labels [N]
        train_mask: Training mask
        target_ratio: Target ratio (100 = upsample to match majority)

    Returns:
        Augmented (x, y, train_mask)
    """
    train_idx = train_mask.nonzero().squeeze()
    train_y = y[train_idx]
    train_x = x[train_idx]

    minority_idx = (train_y == 1).nonzero().squeeze()
    majority_idx = (train_y == 0).nonzero().squeeze()

    if minority_idx.dim() == 0:
        minority_idx = minority_idx.unsqueeze(0)
    if majority_idx.dim() == 0:
        majority_idx = majority_idx.unsqueeze(0)

    n_minority = minority_idx.size(0)
    n_majority = majority_idx.size(0)

    if n_minority == 0 or n_majority == 0:
        return x, y, train_mask

    # Target number of minority samples
    if target_ratio == 100:
        n_target = n_majority
    else:
        n_target = max(n_minority, int(n_majority * target_ratio / 100))

    n_synthetic = n_target - n_minority

    if n_synthetic <= 0:
        return x, y, train_mask

    # Generate synthetic samples via interpolation
    synthetic_features = []
    minority_features = train_x[minority_idx]

    for _ in range(n_synthetic):
        # Pick two random minority samples
        idx1, idx2 = torch.randint(0, n_minority, (2,))
        # Interpolate
        alpha = torch.rand(1).item()
        synthetic = alpha * minority_features[idx1] + (1 - alpha) * minority_features[idx2]
        synthetic_features.append(synthetic)

    synthetic_x = torch.stack(synthetic_features)
    synthetic_y = torch.ones(n_synthetic, dtype=torch.long)
    synthetic_mask = torch.ones(n_synthetic, dtype=torch.bool)

    # Concatenate
    new_x = torch.cat([x, synthetic_x], dim=0)
    new_y = torch.cat([y, synthetic_y], dim=0)
    new_train_mask = torch.cat([train_mask, synthetic_mask], dim=0)

    return new_x, new_y, new_train_mask


def ablate_features(
    x: torch.Tensor,
    feature_set: str = "all_features",
    feature_dim: int = 16,
    seed: int = 42
) -> torch.Tensor:
    """
    Ablate node features for feature importance study.

    Args:
        x: Original features [N, F]
        feature_set: Which features to keep
        feature_dim: Original feature dimension

    Returns:
        Ablated features
    """
    torch.manual_seed(seed)

    if feature_set == "all_features":
        return x

    elif feature_set == "no_behavioral":
        # Zero out first quarter of features
        mask = x.clone()
        mask[:, :feature_dim // 4] = 0
        return mask

    elif feature_set == "no_velocity":
        # Zero out second quarter
        mask = x.clone()
        mask[:, feature_dim // 4:feature_dim // 2] = 0
        return mask

    elif feature_set == "no_temporal":
        # Zero out third quarter
        mask = x.clone()
        mask[:, feature_dim // 2:3 * feature_dim // 4] = 0
        return mask

    elif feature_set == "structural_only":
        # Use only random features (no information)
        return torch.randn_like(x) * 0.01

    elif feature_set == "features_only_no_graph":
        # Keep features as-is (graph will be ignored by MLP)
        return x

    return x


def add_label_noise(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    false_negative_rate: float = 0.0,
    false_positive_rate: float = 0.0,
    seed: int = 42
) -> torch.Tensor:
    """
    Add label noise to training labels.

    Args:
        y: True labels [N]
        train_mask: Training mask
        false_negative_rate: Rate of flipping 1 -> 0
        false_positive_rate: Rate of flipping 0 -> 1
        seed: Random seed

    Returns:
        Noisy labels
    """
    torch.manual_seed(seed)

    noisy_y = y.clone()
    train_idx = train_mask.nonzero().squeeze()

    for idx in train_idx:
        if y[idx] == 1 and torch.rand(1).item() < false_negative_rate:
            noisy_y[idx] = 0
        elif y[idx] == 0 and torch.rand(1).item() < false_positive_rate:
            noisy_y[idx] = 1

    return noisy_y


def generate_graph_size_variants(
    target_homophily: float = 0.2,
    anomaly_prevalence: float = 0.01,
    seed: int = 42
) -> Dict[str, GraphData]:
    """
    Generate graphs of different sizes for robustness testing.
    """
    sizes = {
        "small": 5000,
        "medium": 10000,
        "large": 50000
    }

    graphs = {}
    for name, num_nodes in sizes.items():
        # Adjust avg_degree for larger graphs
        avg_degree = 20 if num_nodes <= 10000 else 15

        graphs[name] = generate_sbm_graph(
            num_nodes=num_nodes,
            target_homophily=target_homophily,
            anomaly_prevalence=anomaly_prevalence,
            avg_degree=avg_degree,
            seed=seed
        )

    return graphs
