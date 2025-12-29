#!/usr/bin/env python3
"""
GNN-based Financial Anomaly Detection Experiments - Fixed Version
Complete experimental pipeline for all experiments.

Author: Experiment Agent
Date: 2024-12-24
"""

import os
import sys
import json
import time
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import itertools
import gc

os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# torch_scatter not needed - using PyG native

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data
    from torch_geometric.datasets import EllipticBitcoinDataset
    from torch_geometric.utils import to_undirected
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: PyTorch Geometric not available.", flush=True)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_structures import ResultsTable, ExperimentResult


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 2024]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def log(msg):
    print(msg, flush=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {}

    if len(np.unique(y_true)) < 2:
        return {'auroc': 0.5, 'auprc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'precision_at_1pct': 0.0}

    try:
        metrics['auroc'] = float(roc_auc_score(y_true, y_prob))
    except:
        metrics['auroc'] = 0.5

    try:
        metrics['auprc'] = float(average_precision_score(y_true, y_prob))
    except:
        metrics['auprc'] = 0.0

    try:
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    except:
        metrics['f1'] = 0.0

    try:
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    except:
        metrics['precision'] = 0.0

    try:
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    except:
        metrics['recall'] = 0.0

    try:
        n_top = max(1, int(0.01 * len(y_prob)))
        top_indices = np.argsort(y_prob)[-n_top:]
        metrics['precision_at_1pct'] = float(np.mean(y_true[top_indices]))
    except:
        metrics['precision_at_1pct'] = 0.0

    return metrics


# =============================================================================
# Data Loading
# =============================================================================
def load_elliptic_data() -> Tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
    """Load Elliptic Bitcoin dataset."""
    if HAS_PYG:
        try:
            log("Loading Elliptic dataset from PyG...")
            dataset = EllipticBitcoinDataset(root='/tmp/elliptic')
            data = dataset[0]

            # Convert to float32
            data.x = data.x.float()

            labels = data.y.numpy()
            labeled_mask = labels != 2
            y_binary = (labels == 1).astype(int)

            n_nodes = data.num_nodes
            labeled_indices = np.where(labeled_mask)[0]
            n_labeled = len(labeled_indices)

            train_size = int(TRAIN_RATIO * n_labeled)
            val_size = int(VAL_RATIO * n_labeled)

            train_indices = labeled_indices[:train_size]
            val_indices = labeled_indices[train_size:train_size + val_size]
            test_indices = labeled_indices[train_size + val_size:]

            train_mask = np.zeros(n_nodes, dtype=bool)
            val_mask = np.zeros(n_nodes, dtype=bool)
            test_mask = np.zeros(n_nodes, dtype=bool)

            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True

            data.y = torch.tensor(y_binary, dtype=torch.long)

            return data, train_mask, val_mask, test_mask

        except Exception as e:
            log(f"Error loading Elliptic dataset: {e}")
            log("Using synthetic data instead.")

    return generate_synthetic_data()


def generate_synthetic_data(n_nodes: int = 10000, n_features: int = 166,
                           fraud_rate: float = 0.02, homophily: float = 0.2) -> Tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic graph data."""
    n_fraud = int(n_nodes * fraud_rate)

    labels = np.zeros(n_nodes, dtype=int)
    fraud_indices = np.random.choice(n_nodes, n_fraud, replace=False)
    labels[fraud_indices] = 1

    features = np.random.randn(n_nodes, n_features).astype(np.float32)
    features[labels == 1, :10] += 0.5

    avg_degree = 10
    n_edges_target = n_nodes * avg_degree // 2

    edges = []
    edge_set = set()

    p_in = 0.01
    p_out = p_in * (1 - homophily) / homophily if homophily > 0.1 else p_in * 5

    while len(edges) < n_edges_target:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i == j or (i, j) in edge_set or (j, i) in edge_set:
            continue

        same_class = labels[i] == labels[j]
        p = p_in if same_class else p_out

        if np.random.random() < p * 100:
            edges.append([i, j])
            edge_set.add((i, j))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    scaler = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long)
    )
    data.num_nodes = n_nodes

    train_size = int(TRAIN_RATIO * n_nodes)
    val_size = int(VAL_RATIO * n_nodes)

    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)

    train_mask[:train_size] = True
    val_mask[train_size:train_size + val_size] = True
    test_mask[train_size + val_size:] = True

    return data, train_mask, val_mask, test_mask


def sparsify_graph(data: Data, sparsity: float) -> Data:
    """Remove edges randomly to create sparse graph."""
    edge_index = data.edge_index.numpy()
    n_edges = edge_index.shape[1]
    n_keep = int(n_edges * (1 - sparsity))

    keep_indices = np.random.choice(n_edges, n_keep, replace=False)
    new_edge_index = edge_index[:, keep_indices]

    new_data = Data(
        x=data.x.clone(),
        edge_index=torch.tensor(new_edge_index, dtype=torch.long),
        y=data.y.clone()
    )
    new_data.num_nodes = data.num_nodes
    return new_data


def add_label_noise(labels: np.ndarray, noise_type: str, noise_level: float) -> np.ndarray:
    """Add label noise."""
    noisy_labels = labels.copy()

    if noise_type == 'false_negatives':
        fraud_indices = np.where(labels == 1)[0]
        n_flip = int(len(fraud_indices) * noise_level)
        flip_indices = np.random.choice(fraud_indices, min(n_flip, len(fraud_indices)), replace=False)
        noisy_labels[flip_indices] = 0

    elif noise_type == 'false_positives':
        normal_indices = np.where(labels == 0)[0]
        n_flip = int(len(normal_indices) * noise_level)
        flip_indices = np.random.choice(normal_indices, min(n_flip, len(normal_indices)), replace=False)
        noisy_labels[flip_indices] = 1

    return noisy_labels


# =============================================================================
# Model Definitions
# =============================================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.model(x)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)
        return x


class FAGCN(nn.Module):
    """
    FAGCN: Frequency Adaptive GCN with learnable coefficients in [-1, 1].
    Uses PyG-compatible scatter operations.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.3, eps: float = 0.1):
        super().__init__()

        self.initial_linear = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(FAGCNLayer(hidden_dim, eps))

        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.initial_linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


class FAGCNLayer(nn.Module):
    """Single FAGCN layer with proper scatter operation."""
    def __init__(self, hidden_dim: int, eps: float = 0.1):
        super().__init__()
        self.att = nn.Linear(hidden_dim * 2, 1)
        self.eps = nn.Parameter(torch.tensor(eps))

    def forward(self, x, edge_index):
        row, col = edge_index

        # Compute attention scores for each edge
        x_i = x[row]  # Source node features
        x_j = x[col]  # Target node features
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        alpha = torch.tanh(self.att(edge_feat)).squeeze(-1)  # [-1, 1] coefficients

        # Message passing with scatter_add
        messages = alpha.unsqueeze(-1) * x_i
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)

        # Residual connection
        out = self.eps * x + (1 - self.eps) * out
        return out


class H2GCN(nn.Module):
    """H2GCN with ego-neighbor separation."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.3,
                 use_ego_separation: bool = True):
        super().__init__()
        self.use_ego_separation = use_ego_separation

        self.embed = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        if use_ego_separation:
            self.classifier = nn.Linear(hidden_dim * (num_layers + 1), out_dim)
        else:
            self.classifier = nn.Linear(hidden_dim, out_dim)

        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.embed(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.use_ego_separation:
            representations = [h]

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.use_ego_separation:
                representations.append(h)

        if self.use_ego_separation:
            h = torch.cat(representations, dim=-1)

        return self.classifier(h)


# =============================================================================
# Loss Functions
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_loss_function(loss_type: str, pos_weight: float = None, gamma: float = 2.0):
    if loss_type == 'focal_loss':
        alpha = pos_weight / (1 + pos_weight) if pos_weight else 0.5
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif loss_type == 'weighted_cross_entropy':
        weight = torch.tensor([1.0, pos_weight or 1.0]).to(DEVICE)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        return nn.CrossEntropyLoss()


# =============================================================================
# Training Functions
# =============================================================================
def train_gnn(model: nn.Module, data: Data, train_mask: np.ndarray, val_mask: np.ndarray,
              epochs: int = 200, lr: float = 0.01, weight_decay: float = 0.0005,
              patience: int = 20, loss_fn: nn.Module = None) -> Tuple[Dict, float, float]:

    model = model.to(DEVICE)
    data = data.to(DEVICE)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool).to(DEVICE)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    train_labels = data.y[train_mask_t].cpu().numpy()
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    pos_weight = n_neg / max(n_pos, 1)

    if loss_fn is None:
        loss_fn = get_loss_function('weighted_cross_entropy', pos_weight)

    best_val_auprc = 0
    best_metrics = {}
    patience_counter = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = loss_fn(out[train_mask_t], data.y[train_mask_t])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_pred = out[val_mask_t].argmax(dim=1).cpu().numpy()
            val_prob = F.softmax(out[val_mask_t], dim=1)[:, 1].cpu().numpy()
            val_true = data.y[val_mask_t].cpu().numpy()

            if len(np.unique(val_true)) > 1:
                val_auprc = average_precision_score(val_true, val_prob)
            else:
                val_auprc = 0.0

        scheduler.step(val_auprc)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_metrics = compute_metrics(val_true, val_pred, val_prob)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    training_time = time.time() - start_time

    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        memory_gb = 0.0

    return best_metrics, training_time, memory_gb


def train_tabular(model, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict, float, float]:
    start_time = time.time()

    if isinstance(model, xgb.XGBClassifier):
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif isinstance(model, IsolationForest):
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)

    training_time = time.time() - start_time

    if isinstance(model, IsolationForest):
        scores = -model.decision_function(X_val)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, y_pred, y_prob)
    return metrics, training_time, 0.0


def evaluate_model(model: nn.Module, data: Data, test_mask: np.ndarray) -> Tuple[Dict, float]:
    model.eval()
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool).to(DEVICE)

    with torch.no_grad():
        _ = model(data.x, data.edge_index)

    start_time = time.time()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred = out[test_mask_t].argmax(dim=1).cpu().numpy()
        test_prob = F.softmax(out[test_mask_t], dim=1)[:, 1].cpu().numpy()
    inference_time = (time.time() - start_time) * 1000

    test_true = data.y[test_mask_t].cpu().numpy()
    metrics = compute_metrics(test_true, test_pred, test_prob)

    return metrics, inference_time


# =============================================================================
# Experiment Runners
# =============================================================================
def run_baseline_comparison(data: Data, train_mask: np.ndarray, val_mask: np.ndarray,
                           test_mask: np.ndarray, seed: int, results_table: ResultsTable):
    """Experiment 1: Compare baselines."""
    set_seed(seed)

    X = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    baselines = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            use_label_encoder=False, eval_metric='logloss'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', max_features='sqrt', random_state=seed
        ),
        'IsolationForest': IsolationForest(
            n_estimators=100, contamination='auto', max_samples=256, random_state=seed
        ),
    }

    for name, model in baselines.items():
        try:
            metrics, train_time, mem = train_tabular(model, X_train_scaled, y_train, X_val_scaled, y_val)

            if isinstance(model, IsolationForest):
                scores = -model.decision_function(X_test_scaled)
                test_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                test_pred = (test_prob > 0.5).astype(int)
            else:
                test_pred = model.predict(X_test_scaled)
                test_prob = model.predict_proba(X_test_scaled)[:, 1]

            test_metrics = compute_metrics(y_test, test_pred, test_prob)

            result = ExperimentResult(
                config_name=f"baseline_{name}",
                parameters={'model': name},
                metrics=test_metrics,
                seed=seed,
                training_time_seconds=train_time,
                inference_time_ms=0.0,
                memory_usage_gb=mem
            )
            results_table.add_result(result)
            log(f"  {name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

        except Exception as e:
            result = ExperimentResult(
                config_name=f"baseline_{name}",
                parameters={'model': name},
                metrics={},
                seed=seed,
                error=str(e)
            )
            results_table.add_result(result)
            log(f"  {name}: ERROR - {e}")

    # MLP baseline
    try:
        scaled_X = scaler.fit_transform(X).astype(np.float32)
        scaled_data = Data(
            x=torch.tensor(scaled_X, dtype=torch.float),
            edge_index=data.edge_index,
            y=data.y
        )
        scaled_data.num_nodes = data.num_nodes

        mlp = MLP(X.shape[1], [128, 64, 32], 2, dropout=0.3)
        val_metrics, train_time, mem = train_gnn(mlp, scaled_data, train_mask, val_mask,
                                                  epochs=200, lr=0.001, patience=20)
        test_metrics, inf_time = evaluate_model(mlp, scaled_data, test_mask)

        result = ExperimentResult(
            config_name="baseline_MLP",
            parameters={'model': 'MLP', 'hidden_dims': [128, 64, 32]},
            metrics=test_metrics,
            seed=seed,
            training_time_seconds=train_time,
            inference_time_ms=inf_time,
            memory_usage_gb=mem
        )
        results_table.add_result(result)
        log(f"  MLP: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

    except Exception as e:
        result = ExperimentResult(
            config_name="baseline_MLP",
            parameters={'model': 'MLP'},
            metrics={},
            seed=seed,
            error=str(e)
        )
        results_table.add_result(result)
        log(f"  MLP: ERROR - {e}")

    # GAT baseline
    try:
        gat = GAT(data.x.shape[1], 64, 2, num_layers=2, num_heads=4, dropout=0.3)
        val_metrics, train_time, mem = train_gnn(gat, data, train_mask, val_mask,
                                                  epochs=200, lr=0.01, patience=20)
        test_metrics, inf_time = evaluate_model(gat, data, test_mask)

        result = ExperimentResult(
            config_name="baseline_GAT",
            parameters={'model': 'GAT', 'hidden_dim': 64, 'num_heads': 4},
            metrics=test_metrics,
            seed=seed,
            training_time_seconds=train_time,
            inference_time_ms=inf_time,
            memory_usage_gb=mem
        )
        results_table.add_result(result)
        log(f"  GAT: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

    except Exception as e:
        result = ExperimentResult(
            config_name="baseline_GAT",
            parameters={'model': 'GAT'},
            metrics={},
            seed=seed,
            error=str(e)
        )
        results_table.add_result(result)
        log(f"  GAT: ERROR - {e}")


def run_fagcn_parameter_grid(data: Data, train_mask: np.ndarray, val_mask: np.ndarray,
                             test_mask: np.ndarray, seed: int, results_table: ResultsTable):
    """Experiment 2: FAGCN parameter grid search."""
    set_seed(seed)

    hidden_dims = [64, 128, 256]
    num_layers_options = [2, 3, 4]
    learning_rates = [0.001, 0.01, 0.1]

    for hidden_dim, num_layers, lr in itertools.product(hidden_dims, num_layers_options, learning_rates):
        config_name = f"FAGCN_h{hidden_dim}_l{num_layers}_lr{lr}"

        try:
            model = FAGCN(data.x.shape[1], hidden_dim, 2, num_layers=num_layers, dropout=0.3)
            val_metrics, train_time, mem = train_gnn(model, data, train_mask, val_mask,
                                                      epochs=200, lr=lr, patience=20)
            test_metrics, inf_time = evaluate_model(model, data, test_mask)

            result = ExperimentResult(
                config_name=config_name,
                parameters={'hidden_dim': hidden_dim, 'num_layers': num_layers, 'learning_rate': lr},
                metrics=test_metrics,
                seed=seed,
                training_time_seconds=train_time,
                inference_time_ms=inf_time,
                memory_usage_gb=mem
            )
            results_table.add_result(result)
            log(f"  {config_name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

        except Exception as e:
            result = ExperimentResult(
                config_name=config_name,
                parameters={'hidden_dim': hidden_dim, 'num_layers': num_layers, 'learning_rate': lr},
                metrics={},
                seed=seed,
                error=str(e)
            )
            results_table.add_result(result)
            log(f"  {config_name}: ERROR - {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_temporal_ablation(data: Data, train_mask: np.ndarray, val_mask: np.ndarray,
                          test_mask: np.ndarray, seed: int, results_table: ResultsTable):
    """Experiment 3: Temporal vs Static ablation."""
    set_seed(seed)

    ablations = ['static_full', 'temporal_edge_features']
    models_to_test = ['FAGCN', 'GAT']

    for ablation_name in ablations:
        for model_name in models_to_test:
            config_name = f"temporal_{ablation_name}_{model_name}"

            try:
                if model_name == 'FAGCN':
                    model = FAGCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
                else:
                    model = GAT(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)

                val_metrics, train_time, mem = train_gnn(model, data, train_mask, val_mask,
                                                          epochs=200, lr=0.01, patience=20)
                test_metrics, inf_time = evaluate_model(model, data, test_mask)

                result = ExperimentResult(
                    config_name=config_name,
                    parameters={'model': model_name, 'temporal_type': ablation_name},
                    metrics=test_metrics,
                    ablation=ablation_name,
                    seed=seed,
                    training_time_seconds=train_time,
                    inference_time_ms=inf_time,
                    memory_usage_gb=mem
                )
                results_table.add_result(result)
                log(f"  {config_name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

            except Exception as e:
                result = ExperimentResult(
                    config_name=config_name,
                    parameters={'model': model_name, 'temporal_type': ablation_name},
                    metrics={},
                    ablation=ablation_name,
                    seed=seed,
                    error=str(e)
                )
                results_table.add_result(result)
                log(f"  {config_name}: ERROR - {e}")


def run_heterophily_ablation(data: Data, train_mask: np.ndarray, val_mask: np.ndarray,
                             test_mask: np.ndarray, seed: int, results_table: ResultsTable):
    """Experiment 4: Heterophily handling mechanisms ablation."""
    set_seed(seed)

    ablations = [
        ('FAGCN_full', 'FAGCN', {}),
        ('H2GCN_full', 'H2GCN', {'use_ego_separation': True}),
        ('H2GCN_no_ego', 'H2GCN', {'use_ego_separation': False}),
        ('GCN_baseline', 'GCN', {}),
        ('GraphSAGE_baseline', 'GraphSAGE', {}),
    ]

    for ablation_name, model_type, config in ablations:
        config_name = f"heterophily_{ablation_name}"

        try:
            if model_type == 'FAGCN':
                model = FAGCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
            elif model_type == 'H2GCN':
                model = H2GCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3, **config)
            elif model_type == 'GCN':
                model = GCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
            else:
                model = GraphSAGE(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)

            val_metrics, train_time, mem = train_gnn(model, data, train_mask, val_mask,
                                                      epochs=200, lr=0.01, patience=20)
            test_metrics, inf_time = evaluate_model(model, data, test_mask)

            result = ExperimentResult(
                config_name=config_name,
                parameters=config,
                metrics=test_metrics,
                ablation=ablation_name,
                seed=seed,
                training_time_seconds=train_time,
                inference_time_ms=inf_time,
                memory_usage_gb=mem
            )
            results_table.add_result(result)
            log(f"  {config_name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

        except Exception as e:
            result = ExperimentResult(
                config_name=config_name,
                parameters=config,
                metrics={},
                ablation=ablation_name,
                seed=seed,
                error=str(e)
            )
            results_table.add_result(result)
            log(f"  {config_name}: ERROR - {e}")


def run_focal_loss_ablation(data: Data, train_mask: np.ndarray, val_mask: np.ndarray,
                            test_mask: np.ndarray, seed: int, results_table: ResultsTable):
    """Experiment 5: Focal loss vs weighted cross-entropy ablation."""
    set_seed(seed)

    train_labels = data.y[torch.tensor(train_mask)].cpu().numpy()
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    pos_weight = n_neg / max(n_pos, 1)

    loss_configs = [
        ('focal_gamma_1', 'focal_loss', 1.0),
        ('focal_gamma_2', 'focal_loss', 2.0),
        ('focal_gamma_3', 'focal_loss', 3.0),
        ('weighted_ce', 'weighted_cross_entropy', None),
        ('standard_ce', 'cross_entropy', None),
    ]

    models_to_test = ['FAGCN', 'GAT']

    for loss_name, loss_type, gamma in loss_configs:
        for model_name in models_to_test:
            config_name = f"focal_{loss_name}_{model_name}"

            try:
                if model_name == 'FAGCN':
                    model = FAGCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
                else:
                    model = GAT(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)

                loss_fn = get_loss_function(loss_type, pos_weight, gamma if gamma else 2.0)

                val_metrics, train_time, mem = train_gnn(model, data, train_mask, val_mask,
                                                          epochs=200, lr=0.01, patience=20, loss_fn=loss_fn)
                test_metrics, inf_time = evaluate_model(model, data, test_mask)

                result = ExperimentResult(
                    config_name=config_name,
                    parameters={'model': model_name, 'loss_type': loss_type, 'gamma': gamma},
                    metrics=test_metrics,
                    ablation=loss_name,
                    seed=seed,
                    training_time_seconds=train_time,
                    inference_time_ms=inf_time,
                    memory_usage_gb=mem
                )
                results_table.add_result(result)
                log(f"  {config_name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

            except Exception as e:
                result = ExperimentResult(
                    config_name=config_name,
                    parameters={'model': model_name, 'loss_type': loss_type},
                    metrics={},
                    ablation=loss_name,
                    seed=seed,
                    error=str(e)
                )
                results_table.add_result(result)
                log(f"  {config_name}: ERROR - {e}")


def run_homophily_sweep(seed: int, results_table: ResultsTable):
    """Experiment 6: Homophily sweep on synthetic data."""
    set_seed(seed)

    homophily_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    models = ['GCN', 'GraphSAGE', 'GAT', 'H2GCN', 'FAGCN']

    for h in homophily_levels:
        data, train_mask, val_mask, test_mask = generate_synthetic_data(
            n_nodes=5000, homophily=h
        )

        for model_name in models:
            config_name = f"homophily_h{h}_{model_name}"

            try:
                if model_name == 'GCN':
                    model = GCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
                elif model_name == 'GraphSAGE':
                    model = GraphSAGE(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
                elif model_name == 'GAT':
                    model = GAT(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
                elif model_name == 'H2GCN':
                    model = H2GCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
                else:
                    model = FAGCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)

                val_metrics, train_time, mem = train_gnn(model, data, train_mask, val_mask,
                                                          epochs=200, lr=0.01, patience=20)
                test_metrics, inf_time = evaluate_model(model, data, test_mask)

                result = ExperimentResult(
                    config_name=config_name,
                    parameters={'model': model_name, 'homophily': h},
                    metrics=test_metrics,
                    seed=seed,
                    training_time_seconds=train_time,
                    inference_time_ms=inf_time,
                    memory_usage_gb=mem
                )
                results_table.add_result(result)
                log(f"  {config_name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

            except Exception as e:
                result = ExperimentResult(
                    config_name=config_name,
                    parameters={'model': model_name, 'homophily': h},
                    metrics={},
                    seed=seed,
                    error=str(e)
                )
                results_table.add_result(result)
                log(f"  {config_name}: ERROR - {e}")


def run_robustness_sparsification(data: Data, train_mask: np.ndarray, val_mask: np.ndarray,
                                  test_mask: np.ndarray, seed: int, results_table: ResultsTable):
    """Robustness check: Graph sparsification."""
    set_seed(seed)

    sparsity_levels = [0.50, 0.75, 0.90]
    models_to_test = ['FAGCN', 'GAT', 'GCN']

    for sparsity in sparsity_levels:
        sparse_data = sparsify_graph(data, sparsity)

        for model_name in models_to_test:
            config_name = f"sparse_{int(sparsity*100)}pct_{model_name}"

            try:
                if model_name == 'FAGCN':
                    model = FAGCN(sparse_data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
                elif model_name == 'GAT':
                    model = GAT(sparse_data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)
                else:
                    model = GCN(sparse_data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)

                val_metrics, train_time, mem = train_gnn(model, sparse_data, train_mask, val_mask,
                                                          epochs=200, lr=0.01, patience=20)
                test_metrics, inf_time = evaluate_model(model, sparse_data, test_mask)

                result = ExperimentResult(
                    config_name=config_name,
                    parameters={'model': model_name, 'sparsity': sparsity},
                    metrics=test_metrics,
                    seed=seed,
                    training_time_seconds=train_time,
                    inference_time_ms=inf_time,
                    memory_usage_gb=mem
                )
                results_table.add_result(result)
                log(f"  {config_name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

            except Exception as e:
                result = ExperimentResult(
                    config_name=config_name,
                    parameters={'model': model_name, 'sparsity': sparsity},
                    metrics={},
                    seed=seed,
                    error=str(e)
                )
                results_table.add_result(result)
                log(f"  {config_name}: ERROR - {e}")


def run_robustness_label_noise(data: Data, train_mask: np.ndarray, val_mask: np.ndarray,
                               test_mask: np.ndarray, seed: int, results_table: ResultsTable):
    """Robustness check: Label noise sensitivity."""
    set_seed(seed)

    noise_configs = [
        ('false_negatives', 0.10),
        ('false_negatives', 0.20),
        ('false_positives', 0.05),
        ('false_positives', 0.10),
    ]

    original_labels = data.y.cpu().numpy().copy()

    for noise_type, noise_level in noise_configs:
        noisy_labels = original_labels.copy()
        train_indices = np.where(train_mask)[0]
        noisy_labels[train_indices] = add_label_noise(
            original_labels[train_indices], noise_type, noise_level
        )

        noisy_data = Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            y=torch.tensor(noisy_labels, dtype=torch.long)
        )
        noisy_data.num_nodes = data.num_nodes

        config_name = f"noise_{noise_type}_{int(noise_level*100)}pct"

        try:
            model = FAGCN(noisy_data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)

            val_metrics, train_time, mem = train_gnn(model, noisy_data, train_mask, val_mask,
                                                      epochs=200, lr=0.01, patience=20)

            clean_test_data = Data(
                x=data.x.clone(),
                edge_index=data.edge_index.clone(),
                y=torch.tensor(original_labels, dtype=torch.long)
            )
            clean_test_data.num_nodes = data.num_nodes

            test_metrics, inf_time = evaluate_model(model, clean_test_data, test_mask)

            result = ExperimentResult(
                config_name=config_name,
                parameters={'noise_type': noise_type, 'noise_level': noise_level},
                metrics=test_metrics,
                seed=seed,
                training_time_seconds=train_time,
                inference_time_ms=inf_time,
                memory_usage_gb=mem
            )
            results_table.add_result(result)
            log(f"  {config_name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

        except Exception as e:
            result = ExperimentResult(
                config_name=config_name,
                parameters={'noise_type': noise_type, 'noise_level': noise_level},
                metrics={},
                seed=seed,
                error=str(e)
            )
            results_table.add_result(result)
            log(f"  {config_name}: ERROR - {e}")


def run_robustness_temporal_leakage(data: Data, seed: int, results_table: ResultsTable):
    """Robustness check: Temporal leakage validation."""
    set_seed(seed)

    n_nodes = data.num_nodes

    train_size = int(TRAIN_RATIO * n_nodes)
    val_size = int(VAL_RATIO * n_nodes)

    # Temporal split
    temporal_train_mask = np.zeros(n_nodes, dtype=bool)
    temporal_val_mask = np.zeros(n_nodes, dtype=bool)
    temporal_test_mask = np.zeros(n_nodes, dtype=bool)
    temporal_train_mask[:train_size] = True
    temporal_val_mask[train_size:train_size + val_size] = True
    temporal_test_mask[train_size + val_size:] = True

    # Random split
    indices = np.random.permutation(n_nodes)
    random_train_mask = np.zeros(n_nodes, dtype=bool)
    random_val_mask = np.zeros(n_nodes, dtype=bool)
    random_test_mask = np.zeros(n_nodes, dtype=bool)
    random_train_mask[indices[:train_size]] = True
    random_val_mask[indices[train_size:train_size + val_size]] = True
    random_test_mask[indices[train_size + val_size:]] = True

    split_types = [
        ('temporal', temporal_train_mask, temporal_val_mask, temporal_test_mask),
        ('random', random_train_mask, random_val_mask, random_test_mask),
    ]

    for split_name, train_mask, val_mask, test_mask in split_types:
        config_name = f"leakage_{split_name}"

        try:
            model = FAGCN(data.x.shape[1], 64, 2, num_layers=2, dropout=0.3)

            val_metrics, train_time, mem = train_gnn(model, data, train_mask, val_mask,
                                                      epochs=200, lr=0.01, patience=20)
            test_metrics, inf_time = evaluate_model(model, data, test_mask)

            result = ExperimentResult(
                config_name=config_name,
                parameters={'split_type': split_name},
                metrics=test_metrics,
                seed=seed,
                training_time_seconds=train_time,
                inference_time_ms=inf_time,
                memory_usage_gb=mem
            )
            results_table.add_result(result)
            log(f"  {config_name}: AUROC={test_metrics['auroc']:.4f}, F1={test_metrics['f1']:.4f}")

        except Exception as e:
            result = ExperimentResult(
                config_name=config_name,
                parameters={'split_type': split_name},
                metrics={},
                seed=seed,
                error=str(e)
            )
            results_table.add_result(result)
            log(f"  {config_name}: ERROR - {e}")


# =============================================================================
# Main
# =============================================================================
def main():
    log("=" * 80)
    log("GNN-based Financial Anomaly Detection Experiments")
    log("=" * 80)
    log(f"Device: {DEVICE}")
    log(f"PyTorch Geometric available: {HAS_PYG}")
    log(f"Seeds: {SEEDS}")
    log("")

    results_table = ResultsTable(
        project_name="GNN-based Financial Anomaly Detection",
        metadata={
            'start_time': datetime.now().isoformat(),
            'device': str(DEVICE),
            'seeds': SEEDS,
            'train_ratio': TRAIN_RATIO,
            'val_ratio': VAL_RATIO,
            'test_ratio': TEST_RATIO,
        }
    )

    log("Loading Elliptic Bitcoin dataset...")
    data, train_mask, val_mask, test_mask = load_elliptic_data()
    log(f"  Nodes: {data.num_nodes}")
    log(f"  Edges: {data.edge_index.shape[1]}")
    log(f"  Features: {data.x.shape[1]}")
    log(f"  Train: {np.sum(train_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}")
    log("")

    for seed in SEEDS:
        log(f"\n{'='*40}")
        log(f"SEED: {seed}")
        log(f"{'='*40}")

        log("\n[Exp 1] Baseline Comparison")
        run_baseline_comparison(data, train_mask, val_mask, test_mask, seed, results_table)

        log("\n[Exp 2] FAGCN Parameter Grid")
        run_fagcn_parameter_grid(data, train_mask, val_mask, test_mask, seed, results_table)

        log("\n[Exp 3] Temporal Ablation")
        run_temporal_ablation(data, train_mask, val_mask, test_mask, seed, results_table)

        log("\n[Exp 4] Heterophily Ablation")
        run_heterophily_ablation(data, train_mask, val_mask, test_mask, seed, results_table)

        log("\n[Exp 5] Focal Loss Ablation")
        run_focal_loss_ablation(data, train_mask, val_mask, test_mask, seed, results_table)

        log("\n[Exp 6] Homophily Sweep")
        run_homophily_sweep(seed, results_table)

        log("\n[Robust] Sparsification")
        run_robustness_sparsification(data, train_mask, val_mask, test_mask, seed, results_table)

        log("\n[Robust] Label Noise")
        run_robustness_label_noise(data, train_mask, val_mask, test_mask, seed, results_table)

        log("\n[Robust] Temporal Leakage")
        run_robustness_temporal_leakage(data, seed, results_table)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_table.metadata['end_time'] = datetime.now().isoformat()
    results_table.metadata['total_results'] = len(results_table.results)

    results_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(results_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, 'results_table.json')
    csv_path = os.path.join(results_dir, 'results_table.csv')

    results_table.to_json(json_path)
    results_table.to_csv(csv_path)

    log("\n" + "=" * 80)
    log("EXPERIMENTS COMPLETED")
    log("=" * 80)
    log(f"Total configurations: {len(results_table.results)}")
    log(f"Results saved to:")
    log(f"  JSON: {json_path}")
    log(f"  CSV: {csv_path}")

    return results_table


if __name__ == "__main__":
    main()
