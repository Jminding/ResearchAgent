"""
Training and evaluation utilities for GNN experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from typing import Dict, Optional, Tuple
import time


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


def get_class_weights(y: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """Compute inverse class frequency weights."""
    train_y = y[train_mask]
    n_pos = (train_y == 1).sum().float()
    n_neg = (train_y == 0).sum().float()

    if n_pos == 0:
        n_pos = 1
    if n_neg == 0:
        n_neg = 1

    # Use sqrt of ratio to avoid too extreme weights
    w_pos = torch.sqrt(n_neg / n_pos)
    w_neg = 1.0

    weights = torch.tensor([w_neg, w_pos])
    return weights


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def train_epoch(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    num_nodes: int,
    class_weights: Optional[torch.Tensor] = None,
    loss_type: str = "cross_entropy"
) -> float:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    out = model(x, edge_index, num_nodes)

    if loss_type == "focal_loss":
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(out[train_mask], y[train_mask])
    elif class_weights is not None:
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=class_weights)
    else:
        loss = F.cross_entropy(out[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    num_nodes: int,
    threshold: float = None
) -> Dict[str, float]:
    """Evaluate model and compute metrics."""
    model.eval()

    start_time = time.time()
    out = model(x, edge_index, num_nodes)
    inference_time = (time.time() - start_time) * 1000  # ms

    probs = F.softmax(out, dim=1)

    y_true = y[mask].cpu().numpy()
    y_score = probs[mask, 1].cpu().numpy()

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'auroc': 0.5,
            'auprc': 0.0,
            'latency_ms': inference_time,
            'threshold': 0.5
        }

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_score),
        'auprc': average_precision_score(y_true, y_score),
        'latency_ms': inference_time,
        'threshold': threshold
    }

    return metrics


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    num_nodes: int,
    epochs: int = 200,
    patience: int = 20,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0005,
    class_weights: Optional[torch.Tensor] = None,
    loss_type: str = "cross_entropy",
    verbose: bool = False
) -> Tuple[nn.Module, Dict[str, float], float]:
    """
    Train model with early stopping based on validation AUROC.

    Returns:
        Trained model, best validation metrics, training time
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_auroc = 0.0
    best_metrics = {}
    best_state = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        loss = train_epoch(
            model, x, edge_index, y, train_mask, optimizer, num_nodes,
            class_weights=class_weights, loss_type=loss_type
        )

        val_metrics = evaluate(model, x, edge_index, y, val_mask, num_nodes)

        # Use AUROC for early stopping (more stable than F1 for imbalanced data)
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_metrics = val_metrics
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}, val_auroc={val_metrics['auroc']:.4f}, val_f1={val_metrics['f1']:.4f}")

    training_time = time.time() - start_time

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics, training_time


def run_single_experiment(
    model_name: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_nodes: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 200,
    patience: int = 20,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0005,
    use_class_weights: bool = True,
    loss_type: str = "cross_entropy",
    device: str = "cpu"
) -> Dict:
    """
    Run a single experiment configuration.

    Returns dict with model name, test metrics, training time, etc.
    """
    from gnn_models import get_model

    # Move data to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Initialize model
    in_channels = x.size(1)
    out_channels = 2  # Binary classification

    model = get_model(
        model_name,
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=out_channels,
        num_layers=num_layers
    ).to(device)

    # Get class weights if needed
    class_weights = None
    if use_class_weights and loss_type == "cross_entropy":
        class_weights = get_class_weights(y, train_mask).to(device)

    # Train
    model, val_metrics, training_time = train_model(
        model, x, edge_index, y, train_mask, val_mask, num_nodes,
        epochs=epochs, patience=patience,
        learning_rate=learning_rate, weight_decay=weight_decay,
        class_weights=class_weights, loss_type=loss_type
    )

    # Test using threshold from validation
    test_metrics = evaluate(model, x, edge_index, y, test_mask, num_nodes,
                           threshold=val_metrics.get('threshold', 0.5))

    return {
        'model': model_name,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time
    }
