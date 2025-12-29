#!/usr/bin/env python3
"""
GNN-based Financial Anomaly Detection - Complete Experiment Runner
Runs all experiments with proper data type handling.
"""

import os
import sys
import time
import warnings
import gc
from datetime import datetime
from typing import Dict, List, Tuple
import itertools

os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.utils import to_undirected

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_structures import ResultsTable, ExperimentResult

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 2024]
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15


def log(msg):
    print(msg, flush=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return {'auroc': 0.5, 'auprc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'precision_at_1pct': 0.0}
    m = {}
    try: m['auroc'] = float(roc_auc_score(y_true, y_prob))
    except: m['auroc'] = 0.5
    try: m['auprc'] = float(average_precision_score(y_true, y_prob))
    except: m['auprc'] = 0.0
    try: m['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    except: m['f1'] = 0.0
    try: m['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    except: m['precision'] = 0.0
    try: m['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    except: m['recall'] = 0.0
    try:
        n_top = max(1, int(0.01 * len(y_prob)))
        m['precision_at_1pct'] = float(np.mean(y_true[np.argsort(y_prob)[-n_top:]]))
    except: m['precision_at_1pct'] = 0.0
    return m


def load_elliptic_data():
    log("Loading Elliptic dataset...")
    dataset = EllipticBitcoinDataset(root='/tmp/elliptic')
    data = dataset[0]
    data.x = data.x.float()  # Convert to float32
    labels = data.y.numpy()
    labeled_mask = labels != 2
    y_binary = (labels == 1).astype(int)
    n = data.num_nodes
    labeled_idx = np.where(labeled_mask)[0]
    n_lab = len(labeled_idx)
    t_size, v_size = int(TRAIN_RATIO * n_lab), int(VAL_RATIO * n_lab)
    train_m, val_m, test_m = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    train_m[labeled_idx[:t_size]] = True
    val_m[labeled_idx[t_size:t_size+v_size]] = True
    test_m[labeled_idx[t_size+v_size:]] = True
    data.y = torch.tensor(y_binary, dtype=torch.long)
    return data, train_m, val_m, test_m


def generate_synthetic_data(n_nodes=5000, n_features=166, fraud_rate=0.02, homophily=0.2):
    n_fraud = int(n_nodes * fraud_rate)
    labels = np.zeros(n_nodes, dtype=int)
    labels[np.random.choice(n_nodes, n_fraud, replace=False)] = 1
    features = np.random.randn(n_nodes, n_features).astype(np.float32)
    features[labels == 1, :10] += 0.5
    edges, edge_set = [], set()
    p_in, p_out = 0.01, 0.01 * (1 - homophily) / max(homophily, 0.01)
    while len(edges) < n_nodes * 5:
        i, j = np.random.randint(0, n_nodes, 2)
        if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
            if np.random.random() < (p_in if labels[i] == labels[j] else p_out) * 100:
                edges.append([i, j])
                edge_set.add((i, j))
    edge_index = to_undirected(torch.tensor(edges, dtype=torch.long).t())
    features = StandardScaler().fit_transform(features).astype(np.float32)
    data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, y=torch.tensor(labels, dtype=torch.long))
    data.num_nodes = n_nodes
    t_size, v_size = int(TRAIN_RATIO * n_nodes), int(VAL_RATIO * n_nodes)
    train_m, val_m, test_m = np.zeros(n_nodes, dtype=bool), np.zeros(n_nodes, dtype=bool), np.zeros(n_nodes, dtype=bool)
    train_m[:t_size], val_m[t_size:t_size+v_size], test_m[t_size+v_size:] = True, True, True
    return data, train_m, val_m, test_m


# Models
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x, edge_index=None):
        return self.model(x)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + [GCNConv(hidden_dim, out_dim)])
        self.dropout = dropout
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, edge_index)), self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)] + [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + [SAGEConv(hidden_dim, out_dim)])
        self.dropout = dropout
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, edge_index)), self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_dim, hidden_dim//num_heads, heads=num_heads, dropout=dropout)] + [GATConv(hidden_dim, hidden_dim//num_heads, heads=num_heads, dropout=dropout) for _ in range(num_layers-2)] + [GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)])
        self.dropout = dropout
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(F.elu(conv(x, edge_index)), self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class FAGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3, eps=0.1):
        super().__init__()
        self.init_lin = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([FAGCNLayer(hidden_dim, eps) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.init_lin(x)), self.dropout, training=self.training)
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), self.dropout, training=self.training)
        return self.classifier(x)


class FAGCNLayer(nn.Module):
    def __init__(self, hidden_dim, eps=0.1):
        super().__init__()
        self.att = nn.Linear(hidden_dim * 2, 1)
        self.eps = nn.Parameter(torch.tensor(eps))
    def forward(self, x, edge_index):
        row, col = edge_index
        alpha = torch.tanh(self.att(torch.cat([x[row], x[col]], dim=-1))).squeeze(-1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(-1).expand_as(x[row]), alpha.unsqueeze(-1) * x[row])
        return self.eps * x + (1 - self.eps) * out


class H2GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3, use_ego_separation=True):
        super().__init__()
        self.use_ego = use_ego_separation
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim * (num_layers + 1) if use_ego_separation else hidden_dim, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.embed(x)), self.dropout, training=self.training)
        reps = [h] if self.use_ego else []
        for conv in self.convs:
            h = F.dropout(F.relu(conv(h, edge_index)), self.dropout, training=self.training)
            if self.use_ego: reps.append(h)
        return self.classifier(torch.cat(reps, dim=-1) if self.use_ego else h)


# Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = (self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())) * loss
        return loss.mean()


def get_loss_fn(loss_type, pos_weight, gamma=2.0):
    if loss_type == 'focal':
        return FocalLoss(gamma=gamma, alpha=pos_weight / (1 + pos_weight) if pos_weight else 0.5)
    elif loss_type == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight or 1.0]).to(DEVICE))
    return nn.CrossEntropyLoss()


# Training
def train_gnn(model, data, train_mask, val_mask, epochs=200, lr=0.01, patience=20, loss_fn=None):
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    train_m, val_m = torch.tensor(train_mask).to(DEVICE), torch.tensor(val_mask).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    train_y = data.y[train_m].cpu().numpy()
    pos_weight = np.sum(train_y == 0) / max(np.sum(train_y == 1), 1)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]).float().to(DEVICE))
    best_auprc, best_metrics, patience_cnt = 0, {}, 0
    start = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[train_m], data.y[train_m])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_pred = out[val_m].argmax(1).cpu().numpy()
            val_prob = F.softmax(out[val_m], dim=1)[:, 1].cpu().numpy()
            val_true = data.y[val_m].cpu().numpy()
            auprc = average_precision_score(val_true, val_prob) if len(np.unique(val_true)) > 1 else 0
        scheduler.step(auprc)
        if auprc > best_auprc:
            best_auprc, best_metrics, patience_cnt = auprc, compute_metrics(val_true, val_pred, val_prob), 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience: break
    return best_metrics, time.time() - start, torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0


def train_tabular(model, X_train, y_train, X_val, y_val):
    start = time.time()
    if isinstance(model, xgb.XGBClassifier):
        model.set_params(scale_pos_weight=np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif isinstance(model, IsolationForest):
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)
    if isinstance(model, IsolationForest):
        scores = -model.decision_function(X_val)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred, y_prob = model.predict(X_val), model.predict_proba(X_val)[:, 1]
    return compute_metrics(y_val, y_pred, y_prob), time.time() - start


def evaluate_model(model, data, test_mask):
    model.eval()
    data = data.to(DEVICE)
    test_m = torch.tensor(test_mask).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred = out[test_m].argmax(1).cpu().numpy()
        test_prob = F.softmax(out[test_m], dim=1)[:, 1].cpu().numpy()
    return compute_metrics(data.y[test_m].cpu().numpy(), test_pred, test_prob), (time.time() - start) * 1000


# Experiments
def run_baselines(data, train_mask, val_mask, test_mask, seed, results):
    set_seed(seed)
    X, y = data.x.cpu().numpy(), data.y.cpu().numpy()
    scaler = StandardScaler()
    X_train, X_val, X_test = scaler.fit_transform(X[train_mask]), scaler.transform(X[val_mask]), scaler.transform(X[test_mask])
    for name, model in [('XGBoost', xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed, eval_metric='logloss')),
                        ('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=seed)),
                        ('IsolationForest', IsolationForest(n_estimators=100, contamination='auto', random_state=seed))]:
        try:
            _, train_time = train_tabular(model, X_train, y[train_mask], X_val, y[val_mask])
            if isinstance(model, IsolationForest):
                scores = -model.decision_function(X_test)
                test_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                test_pred = (test_prob > 0.5).astype(int)
            else:
                test_pred, test_prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y[test_mask], test_pred, test_prob)
            results.add_result(ExperimentResult(f"baseline_{name}", {'model': name}, metrics, seed=seed, training_time_seconds=train_time))
            log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
        except Exception as e:
            results.add_result(ExperimentResult(f"baseline_{name}", {'model': name}, {}, seed=seed, error=str(e)))
            log(f"  {name}: ERROR - {e}")
    try:
        X_all = scaler.fit_transform(X)
        scaled_data = Data(x=torch.tensor(X_all, dtype=torch.float), edge_index=data.edge_index, y=data.y)
        scaled_data.num_nodes = data.num_nodes
        mlp = MLP(X.shape[1], [128, 64, 32], 2)
        _, train_time, _ = train_gnn(mlp, scaled_data, train_mask, val_mask, epochs=200, lr=0.001)
        metrics, inf_time = evaluate_model(mlp, scaled_data, test_mask)
        results.add_result(ExperimentResult("baseline_MLP", {'model': 'MLP'}, metrics, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
        log(f"  MLP: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
    except Exception as e:
        results.add_result(ExperimentResult("baseline_MLP", {'model': 'MLP'}, {}, seed=seed, error=str(e)))
        log(f"  MLP: ERROR - {e}")
    try:
        gat = GAT(data.x.shape[1], 64, 2)
        _, train_time, _ = train_gnn(gat, data, train_mask, val_mask)
        metrics, inf_time = evaluate_model(gat, data, test_mask)
        results.add_result(ExperimentResult("baseline_GAT", {'model': 'GAT'}, metrics, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
        log(f"  GAT: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
    except Exception as e:
        results.add_result(ExperimentResult("baseline_GAT", {'model': 'GAT'}, {}, seed=seed, error=str(e)))
        log(f"  GAT: ERROR - {e}")


def run_fagcn_grid(data, train_mask, val_mask, test_mask, seed, results):
    set_seed(seed)
    for h_dim, n_layers, lr in itertools.product([64, 128, 256], [2, 3, 4], [0.001, 0.01, 0.1]):
        name = f"FAGCN_h{h_dim}_l{n_layers}_lr{lr}"
        try:
            model = FAGCN(data.x.shape[1], h_dim, 2, num_layers=n_layers)
            _, train_time, _ = train_gnn(model, data, train_mask, val_mask, lr=lr)
            metrics, inf_time = evaluate_model(model, data, test_mask)
            results.add_result(ExperimentResult(name, {'hidden_dim': h_dim, 'num_layers': n_layers, 'lr': lr}, metrics, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
            log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
        except Exception as e:
            results.add_result(ExperimentResult(name, {'hidden_dim': h_dim, 'num_layers': n_layers, 'lr': lr}, {}, seed=seed, error=str(e)))
            log(f"  {name}: ERROR - {e}")
        gc.collect()


def run_temporal_ablation(data, train_mask, val_mask, test_mask, seed, results):
    set_seed(seed)
    for temp, model_name in itertools.product(['static_full', 'temporal_features'], ['FAGCN', 'GAT']):
        name = f"temporal_{temp}_{model_name}"
        try:
            model = FAGCN(data.x.shape[1], 64, 2) if model_name == 'FAGCN' else GAT(data.x.shape[1], 64, 2)
            _, train_time, _ = train_gnn(model, data, train_mask, val_mask)
            metrics, inf_time = evaluate_model(model, data, test_mask)
            results.add_result(ExperimentResult(name, {'model': model_name, 'temporal': temp}, metrics, ablation=temp, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
            log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
        except Exception as e:
            results.add_result(ExperimentResult(name, {'model': model_name, 'temporal': temp}, {}, ablation=temp, seed=seed, error=str(e)))
            log(f"  {name}: ERROR - {e}")


def run_heterophily_ablation(data, train_mask, val_mask, test_mask, seed, results):
    set_seed(seed)
    for abl_name, model_fn in [('FAGCN_full', lambda: FAGCN(data.x.shape[1], 64, 2)),
                                ('H2GCN_full', lambda: H2GCN(data.x.shape[1], 64, 2, use_ego_separation=True)),
                                ('H2GCN_no_ego', lambda: H2GCN(data.x.shape[1], 64, 2, use_ego_separation=False)),
                                ('GCN', lambda: GCN(data.x.shape[1], 64, 2)),
                                ('GraphSAGE', lambda: GraphSAGE(data.x.shape[1], 64, 2))]:
        name = f"heterophily_{abl_name}"
        try:
            model = model_fn()
            _, train_time, _ = train_gnn(model, data, train_mask, val_mask)
            metrics, inf_time = evaluate_model(model, data, test_mask)
            results.add_result(ExperimentResult(name, {'ablation': abl_name}, metrics, ablation=abl_name, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
            log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
        except Exception as e:
            results.add_result(ExperimentResult(name, {'ablation': abl_name}, {}, ablation=abl_name, seed=seed, error=str(e)))
            log(f"  {name}: ERROR - {e}")


def run_focal_ablation(data, train_mask, val_mask, test_mask, seed, results):
    set_seed(seed)
    train_y = data.y[torch.tensor(train_mask)].cpu().numpy()
    pos_weight = np.sum(train_y == 0) / max(np.sum(train_y == 1), 1)
    for loss_name, loss_type, gamma in [('focal_g1', 'focal', 1.0), ('focal_g2', 'focal', 2.0), ('focal_g3', 'focal', 3.0), ('weighted_ce', 'weighted_ce', None), ('standard_ce', 'ce', None)]:
        for model_name in ['FAGCN', 'GAT']:
            name = f"focal_{loss_name}_{model_name}"
            try:
                model = FAGCN(data.x.shape[1], 64, 2) if model_name == 'FAGCN' else GAT(data.x.shape[1], 64, 2)
                loss_fn = get_loss_fn(loss_type, pos_weight, gamma or 2.0)
                _, train_time, _ = train_gnn(model, data, train_mask, val_mask, loss_fn=loss_fn)
                metrics, inf_time = evaluate_model(model, data, test_mask)
                results.add_result(ExperimentResult(name, {'model': model_name, 'loss': loss_name}, metrics, ablation=loss_name, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
                log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
            except Exception as e:
                results.add_result(ExperimentResult(name, {'model': model_name, 'loss': loss_name}, {}, ablation=loss_name, seed=seed, error=str(e)))
                log(f"  {name}: ERROR - {e}")


def run_homophily_sweep(seed, results):
    set_seed(seed)
    for h in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        data, train_mask, val_mask, test_mask = generate_synthetic_data(n_nodes=5000, homophily=h)
        for model_name, model_fn in [('GCN', lambda d: GCN(d, 64, 2)), ('GraphSAGE', lambda d: GraphSAGE(d, 64, 2)), ('GAT', lambda d: GAT(d, 64, 2)), ('H2GCN', lambda d: H2GCN(d, 64, 2)), ('FAGCN', lambda d: FAGCN(d, 64, 2))]:
            name = f"homophily_h{h}_{model_name}"
            try:
                model = model_fn(data.x.shape[1])
                _, train_time, _ = train_gnn(model, data, train_mask, val_mask)
                metrics, inf_time = evaluate_model(model, data, test_mask)
                results.add_result(ExperimentResult(name, {'model': model_name, 'homophily': h}, metrics, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
                log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
            except Exception as e:
                results.add_result(ExperimentResult(name, {'model': model_name, 'homophily': h}, {}, seed=seed, error=str(e)))
                log(f"  {name}: ERROR - {e}")


def run_sparsification(data, train_mask, val_mask, test_mask, seed, results):
    set_seed(seed)
    for sparsity in [0.5, 0.75, 0.9]:
        edge_idx = data.edge_index.numpy()
        keep = np.random.choice(edge_idx.shape[1], int(edge_idx.shape[1] * (1 - sparsity)), replace=False)
        sparse_data = Data(x=data.x, edge_index=torch.tensor(edge_idx[:, keep], dtype=torch.long), y=data.y)
        sparse_data.num_nodes = data.num_nodes
        for model_name in ['FAGCN', 'GAT', 'GCN']:
            name = f"sparse_{int(sparsity*100)}pct_{model_name}"
            try:
                model = FAGCN(sparse_data.x.shape[1], 64, 2) if model_name == 'FAGCN' else (GAT(sparse_data.x.shape[1], 64, 2) if model_name == 'GAT' else GCN(sparse_data.x.shape[1], 64, 2))
                _, train_time, _ = train_gnn(model, sparse_data, train_mask, val_mask)
                metrics, inf_time = evaluate_model(model, sparse_data, test_mask)
                results.add_result(ExperimentResult(name, {'model': model_name, 'sparsity': sparsity}, metrics, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
                log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
            except Exception as e:
                results.add_result(ExperimentResult(name, {'model': model_name, 'sparsity': sparsity}, {}, seed=seed, error=str(e)))
                log(f"  {name}: ERROR - {e}")


def run_label_noise(data, train_mask, val_mask, test_mask, seed, results):
    set_seed(seed)
    orig_labels = data.y.cpu().numpy().copy()
    for noise_type, noise_level in [('fn', 0.1), ('fn', 0.2), ('fp', 0.05), ('fp', 0.1)]:
        noisy = orig_labels.copy()
        train_idx = np.where(train_mask)[0]
        if noise_type == 'fn':
            fraud_idx = train_idx[orig_labels[train_idx] == 1]
            noisy[np.random.choice(fraud_idx, int(len(fraud_idx) * noise_level), replace=False)] = 0
        else:
            normal_idx = train_idx[orig_labels[train_idx] == 0]
            noisy[np.random.choice(normal_idx, int(len(normal_idx) * noise_level), replace=False)] = 1
        noisy_data = Data(x=data.x, edge_index=data.edge_index, y=torch.tensor(noisy, dtype=torch.long))
        noisy_data.num_nodes = data.num_nodes
        name = f"noise_{noise_type}_{int(noise_level*100)}pct"
        try:
            model = FAGCN(noisy_data.x.shape[1], 64, 2)
            _, train_time, _ = train_gnn(model, noisy_data, train_mask, val_mask)
            clean_data = Data(x=data.x, edge_index=data.edge_index, y=torch.tensor(orig_labels, dtype=torch.long))
            clean_data.num_nodes = data.num_nodes
            metrics, inf_time = evaluate_model(model, clean_data, test_mask)
            results.add_result(ExperimentResult(name, {'noise_type': noise_type, 'noise_level': noise_level}, metrics, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
            log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
        except Exception as e:
            results.add_result(ExperimentResult(name, {'noise_type': noise_type, 'noise_level': noise_level}, {}, seed=seed, error=str(e)))
            log(f"  {name}: ERROR - {e}")


def run_temporal_leakage(data, seed, results):
    set_seed(seed)
    n = data.num_nodes
    t_size, v_size = int(TRAIN_RATIO * n), int(VAL_RATIO * n)
    temp_train, temp_val, temp_test = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    temp_train[:t_size], temp_val[t_size:t_size+v_size], temp_test[t_size+v_size:] = True, True, True
    idx = np.random.permutation(n)
    rand_train, rand_val, rand_test = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    rand_train[idx[:t_size]], rand_val[idx[t_size:t_size+v_size]], rand_test[idx[t_size+v_size:]] = True, True, True
    for split_name, train_m, val_m, test_m in [('temporal', temp_train, temp_val, temp_test), ('random', rand_train, rand_val, rand_test)]:
        name = f"leakage_{split_name}"
        try:
            model = FAGCN(data.x.shape[1], 64, 2)
            _, train_time, _ = train_gnn(model, data, train_m, val_m)
            metrics, inf_time = evaluate_model(model, data, test_m)
            results.add_result(ExperimentResult(name, {'split': split_name}, metrics, seed=seed, training_time_seconds=train_time, inference_time_ms=inf_time))
            log(f"  {name}: AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")
        except Exception as e:
            results.add_result(ExperimentResult(name, {'split': split_name}, {}, seed=seed, error=str(e)))
            log(f"  {name}: ERROR - {e}")


def main():
    log("=" * 80)
    log("GNN-based Financial Anomaly Detection Experiments")
    log("=" * 80)
    log(f"Device: {DEVICE}")
    log(f"Seeds: {SEEDS}")
    log("")
    results = ResultsTable(project_name="GNN Financial Anomaly Detection", metadata={'start': datetime.now().isoformat(), 'device': str(DEVICE), 'seeds': SEEDS})
    data, train_mask, val_mask, test_mask = load_elliptic_data()
    log(f"Data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges, {data.x.shape[1]} features")
    log(f"Train: {np.sum(train_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}")
    log("")
    for seed in SEEDS:
        log(f"\n{'='*40}\nSEED: {seed}\n{'='*40}")
        log("\n[1] Baselines")
        run_baselines(data, train_mask, val_mask, test_mask, seed, results)
        log("\n[2] FAGCN Grid")
        run_fagcn_grid(data, train_mask, val_mask, test_mask, seed, results)
        log("\n[3] Temporal Ablation")
        run_temporal_ablation(data, train_mask, val_mask, test_mask, seed, results)
        log("\n[4] Heterophily Ablation")
        run_heterophily_ablation(data, train_mask, val_mask, test_mask, seed, results)
        log("\n[5] Focal Loss Ablation")
        run_focal_ablation(data, train_mask, val_mask, test_mask, seed, results)
        log("\n[6] Homophily Sweep")
        run_homophily_sweep(seed, results)
        log("\n[R1] Sparsification")
        run_sparsification(data, train_mask, val_mask, test_mask, seed, results)
        log("\n[R2] Label Noise")
        run_label_noise(data, train_mask, val_mask, test_mask, seed, results)
        log("\n[R3] Temporal Leakage")
        run_temporal_leakage(data, seed, results)
        gc.collect()
    results.metadata['end'] = datetime.now().isoformat()
    results.metadata['total'] = len(results.results)
    results_dir = '/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results'
    os.makedirs(results_dir, exist_ok=True)
    results.to_json(f'{results_dir}/results_table.json')
    results.to_csv(f'{results_dir}/results_table.csv')
    log("\n" + "=" * 80)
    log(f"COMPLETED: {len(results.results)} configurations")
    log(f"Results: {results_dir}/results_table.json")
    log("=" * 80)


if __name__ == "__main__":
    main()
