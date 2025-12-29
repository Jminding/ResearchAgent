"""
TOFM Training Module: Training Procedure with Early Stopping

This module implements the training procedure as specified in framework.md Section 8.3
Key features:
- AdamW optimizer with weight decay
- Cosine annealing with warmup
- Class-weighted cross-entropy loss
- Auxiliary loss for microstructure regularization
- Early stopping based on validation loss
- Comprehensive logging
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class CosineAnnealingWithWarmup:
    """
    Learning rate scheduler with linear warmup and cosine annealing.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def compute_class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.

    class_weights = max(class_counts) / class_counts
    Normalized so sum = n_classes
    """
    class_counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1)  # Avoid division by zero

    # Inverse frequency weighting
    weights = np.max(class_counts) / class_counts

    # Normalize
    weights = weights / weights.sum() * n_classes

    return torch.tensor(weights, dtype=torch.float32)


def compute_financial_metrics(predictions: np.ndarray, labels: np.ndarray,
                              mid_prices: np.ndarray, horizon: int = 10) -> Dict:
    """
    Compute financial performance metrics.

    Args:
        predictions: Model predictions (0=down, 1=stable, 2=up)
        labels: True labels
        mid_prices: Mid-price series
        horizon: Prediction horizon

    Returns:
        Dictionary of financial metrics
    """
    # Convert predictions to signals (-1, 0, +1)
    signals = predictions.astype(np.float32) - 1

    # Compute returns
    if len(mid_prices) > horizon:
        returns = (mid_prices[horizon:] - mid_prices[:-horizon]) / mid_prices[:-horizon]
        # Align with predictions
        n = min(len(signals), len(returns))
        signals = signals[:n]
        returns = returns[:n]
    else:
        returns = np.zeros_like(signals)

    # Strategy returns
    strategy_returns = signals * returns

    # Cumulative PnL
    cumulative_pnl = np.cumsum(strategy_returns)

    # Sharpe ratio (annualized, assuming ~6.5 hours of trading = 23400 seconds)
    # With tick data, assume ~10 ticks per second on average
    ticks_per_day = 23400 * 10
    annualization_factor = np.sqrt(252 * ticks_per_day / horizon)

    if np.std(strategy_returns) > 1e-10:
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * annualization_factor
    else:
        sharpe_ratio = 0.0

    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

    # Hit rate
    hit_rate = np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0.0

    # Total return
    total_return = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0.0

    return {
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'hit_rate': float(hit_rate),
        'total_return': float(total_return),
        'cumulative_pnl': cumulative_pnl.tolist() if len(cumulative_pnl) <= 1000 else cumulative_pnl[::len(cumulative_pnl)//1000].tolist()
    }


class TrainingLogger:
    """Logger for training metrics."""
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f"{experiment_name}_training_log.txt")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.json")

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log: {experiment_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("="*60 + "\n\n")

    def log(self, message: str, print_msg: bool = True):
        """Log message to file and optionally print."""
        if print_msg:
            print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  val_accuracy: float, val_f1: float, lr: float, epoch_time: float):
        """Log epoch metrics."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        self.history['val_f1'].append(val_f1)
        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)

        message = (f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
                  f"Val F1: {val_f1:.4f} | LR: {lr:.6f} | Time: {epoch_time:.1f}s")
        self.log(message)

    def save(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_tofm(model: nn.Module,
               data_dict: Dict,
               config: Dict,
               device: torch.device,
               logger: TrainingLogger,
               checkpoint_dir: str) -> Tuple[nn.Module, Dict]:
    """
    Train TOFM model following framework.md Section 8.3

    Args:
        model: TOFM model instance
        data_dict: Dictionary with X_train, y_train, X_val, y_val, etc.
        config: Training configuration
        device: torch device
        logger: TrainingLogger instance
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Trained model and training history
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Extract config
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 256)
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    patience = config.get('patience', 10)
    alpha_aux = config.get('alpha_aux', 0.1)
    warmup_steps = config.get('warmup_steps', 1000)
    gradient_clip = config.get('gradient_clip', 1.0)

    logger.log(f"\nTraining Configuration:")
    logger.log(f"  Epochs: {epochs}")
    logger.log(f"  Batch size: {batch_size}")
    logger.log(f"  Learning rate: {lr}")
    logger.log(f"  Weight decay: {weight_decay}")
    logger.log(f"  Patience: {patience}")
    logger.log(f"  Alpha aux: {alpha_aux}")
    logger.log(f"  Warmup steps: {warmup_steps}")
    logger.log(f"  Gradient clip: {gradient_clip}")
    logger.log("")

    # Create datasets
    X_train = torch.tensor(data_dict['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data_dict['y_train'], dtype=torch.long)
    X_val = torch.tensor(data_dict['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data_dict['y_val'], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights
    class_weights = compute_class_weights(data_dict['y_train']).to(device)
    logger.log(f"Class weights: {class_weights.cpu().numpy()}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Loss functions
    criterion_main = nn.CrossEntropyLoss(weight=class_weights)
    criterion_aux = nn.MSELoss()

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    model.to(device)

    logger.log("\n" + "="*60)
    logger.log("Starting Training")
    logger.log("="*60 + "\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            logits, aux_pred, _ = model(X_batch)

            # Main classification loss
            loss_main = criterion_main(logits, y_batch)

            # Auxiliary loss (predict OFI and lambda from last timestep)
            ofi_true = X_batch[:, -1, 0]  # OFI
            lambda_true = X_batch[:, -1, 5]  # Kyle's lambda
            aux_target = torch.stack([ofi_true, lambda_true], dim=1)
            loss_aux = criterion_aux(aux_pred, aux_target)

            # Total loss
            loss_total = loss_main + alpha_aux * loss_aux

            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss_total.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _, _ = model(X_batch)
                loss = criterion_main(logits, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        val_accuracy = np.mean(all_preds == all_labels)

        # Macro F1 score
        f1_scores = []
        for c in range(3):
            tp = np.sum((all_preds == c) & (all_labels == c))
            fp = np.sum((all_preds == c) & (all_labels != c))
            fn = np.sum((all_preds != c) & (all_labels == c))
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)
        val_f1 = np.mean(f1_scores)

        epoch_time = time.time() - epoch_start

        # Log epoch metrics
        logger.log_epoch(epoch, avg_train_loss, avg_val_loss,
                        val_accuracy, val_f1, scheduler.get_lr(), epoch_time)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.log(f"\nEarly stopping triggered at epoch {epoch}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.log(f"\nLoaded best model with val_loss: {best_val_loss:.4f}")

    logger.log("\n" + "="*60)
    logger.log("Training Complete")
    logger.log("="*60)

    logger.save()

    return model, logger.history


def train_with_walk_forward(model_class, model_kwargs: Dict,
                            data_dict: Dict,
                            config: Dict,
                            device: torch.device,
                            results_dir: str,
                            n_splits: int = 5) -> List[Dict]:
    """
    Walk-forward validation training.

    Splits data into n_splits folds temporally and trains on expanding window.

    Args:
        model_class: Model class to instantiate
        model_kwargs: Model constructor arguments
        data_dict: Full data dictionary
        config: Training configuration
        device: torch device
        results_dir: Directory to save results
        n_splits: Number of walk-forward splits

    Returns:
        List of results for each split
    """
    X_all = np.concatenate([data_dict['X_train'], data_dict['X_val'], data_dict['X_test']], axis=0)
    y_all = np.concatenate([data_dict['y_train'], data_dict['y_val'], data_dict['y_test']], axis=0)

    n_total = len(X_all)
    split_size = n_total // (n_splits + 1)

    results = []

    for fold in range(n_splits):
        print(f"\n{'='*60}")
        print(f"Walk-Forward Fold {fold + 1}/{n_splits}")
        print(f"{'='*60}")

        # Training set: all data up to current fold
        train_end = (fold + 1) * split_size
        val_start = train_end
        val_end = train_end + split_size // 2
        test_start = val_end
        test_end = min((fold + 2) * split_size, n_total)

        fold_data = {
            'X_train': X_all[:train_end],
            'y_train': y_all[:train_end],
            'X_val': X_all[val_start:val_end],
            'y_val': y_all[val_start:val_end],
            'X_test': X_all[test_start:test_end],
            'y_test': y_all[test_start:test_end]
        }

        print(f"Train: 0-{train_end}, Val: {val_start}-{val_end}, Test: {test_start}-{test_end}")

        # Create model
        model = model_class(**model_kwargs)

        # Create logger
        logger = TrainingLogger(results_dir, f"walk_forward_fold_{fold+1}")

        # Train
        checkpoint_dir = os.path.join(results_dir, f"checkpoints_fold_{fold+1}")
        model, history = train_tofm(model, fold_data, config, device, logger, checkpoint_dir)

        # Evaluate on test fold
        model.eval()
        X_test = torch.tensor(fold_data['X_test'], dtype=torch.float32).to(device)
        y_test = fold_data['y_test']

        with torch.no_grad():
            logits, _, _ = model(X_test)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        test_accuracy = np.mean(preds == y_test)

        results.append({
            'fold': fold + 1,
            'train_samples': train_end,
            'test_samples': test_end - test_start,
            'test_accuracy': test_accuracy,
            'best_val_loss': min(history['val_loss']),
            'final_val_accuracy': history['val_accuracy'][-1]
        })

        print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.4f}")

    return results


if __name__ == "__main__":
    # Test training utilities
    print("Testing training utilities...")

    # Test class weights
    y = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    weights = compute_class_weights(y)
    print(f"Class weights: {weights}")

    # Test scheduler
    model = nn.Linear(10, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps=100, total_steps=1000)

    lrs = []
    for _ in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_lr())

    print(f"LR at step 50 (warmup): {lrs[49]:.6f}")
    print(f"LR at step 100 (end warmup): {lrs[99]:.6f}")
    print(f"LR at step 500 (middle): {lrs[499]:.6f}")
    print(f"LR at step 999 (end): {lrs[998]:.6f}")
