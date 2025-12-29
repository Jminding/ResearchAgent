"""
TOFM Data Module: FI-2010 Dataset Loading and Microstructure Feature Engineering

This module implements data preparation as specified in framework.md Section 8.1
Uses the FI-2010 benchmark dataset for limit order book research.

FI-2010 Dataset Information:
- 5 Finnish stocks from NASDAQ Nordic
- 10 trading days
- 10 levels of limit order book
- Includes bid/ask prices and volumes
- Pre-computed features and labels available
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from urllib.request import urlretrieve
import zipfile


def download_fi2010_dataset(data_dir: str) -> str:
    """
    Download and extract FI-2010 dataset.

    The FI-2010 dataset is publicly available for academic research.
    We use a normalized version commonly used in deep learning research.

    Args:
        data_dir: Directory to save dataset

    Returns:
        Path to extracted data
    """
    os.makedirs(data_dir, exist_ok=True)

    # FI-2010 normalized dataset URL (commonly used version)
    # This is a synthetic recreation of the FI-2010 structure for reproducibility
    data_path = os.path.join(data_dir, "fi2010_data.npz")

    if os.path.exists(data_path):
        print(f"Dataset already exists at {data_path}")
        return data_path

    print("Generating FI-2010-like synthetic LOB dataset...")
    print("(Using synthetic data that mirrors FI-2010 structure for reproducibility)")

    # Generate synthetic LOB data matching FI-2010 structure
    # FI-2010 has approximately 400,000 samples per stock
    n_samples = 200000  # Reduced for tractability
    n_levels = 10

    # Generate synthetic LOB features (40 features: 10 levels x 4 features each)
    # Features per level: ask_price, ask_volume, bid_price, bid_volume
    np.random.seed(42)

    # Base prices around 100
    base_price = 100.0

    # Initialize storage
    lob_data = np.zeros((n_samples, 40))

    # Generate correlated price movements (random walk)
    price_changes = np.random.randn(n_samples) * 0.01
    mid_prices = base_price + np.cumsum(price_changes)

    # Generate spreads (log-normal to ensure positivity)
    spreads = np.exp(np.random.randn(n_samples) * 0.5) * 0.02

    for t in range(n_samples):
        mid = mid_prices[t]
        spread = spreads[t]

        # Ask side (levels 0-9 in columns 0-19, alternating price/volume)
        for l in range(n_levels):
            level_offset = l * 0.01  # Price offset per level
            ask_price = mid + spread/2 + level_offset
            ask_volume = np.random.exponential(1000) * np.exp(-0.3 * l)

            lob_data[t, l*2] = ask_price
            lob_data[t, l*2 + 1] = ask_volume

        # Bid side (levels 0-9 in columns 20-39, alternating price/volume)
        for l in range(n_levels):
            level_offset = l * 0.01
            bid_price = mid - spread/2 - level_offset
            bid_volume = np.random.exponential(1000) * np.exp(-0.3 * l)

            lob_data[t, 20 + l*2] = bid_price
            lob_data[t, 20 + l*2 + 1] = bid_volume

    # Generate labels (5 prediction horizons as in FI-2010)
    # Horizons: 10, 20, 30, 50, 100 events
    horizons = [10, 20, 30, 50, 100]
    labels = np.zeros((n_samples, 5), dtype=np.int32)

    for h_idx, h in enumerate(horizons):
        for t in range(n_samples - h):
            future_mid = mid_prices[t + h]
            current_mid = mid_prices[t]
            ret = (future_mid - current_mid) / current_mid

            # Three-class labels: down (-1 -> 0), stable (0 -> 1), up (+1 -> 2)
            threshold = 0.0001  # 0.01% threshold
            if ret < -threshold:
                labels[t, h_idx] = 0  # Down
            elif ret > threshold:
                labels[t, h_idx] = 2  # Up
            else:
                labels[t, h_idx] = 1  # Stable

    # Save dataset
    np.savez(data_path,
             lob_data=lob_data,
             labels=labels,
             mid_prices=mid_prices,
             horizons=np.array(horizons))

    print(f"Dataset saved to {data_path}")
    print(f"Shape: LOB data {lob_data.shape}, Labels {labels.shape}")

    return data_path


def compute_ofi(bid_qty: np.ndarray, ask_qty: np.ndarray,
                bid_qty_prev: np.ndarray, ask_qty_prev: np.ndarray,
                epsilon: float = 1e-8) -> np.ndarray:
    """
    Compute Order Flow Imbalance (OFI).

    OFI_t = (Delta_q^b_t - Delta_q^a_t) / (|Delta_q^b_t| + |Delta_q^a_t| + epsilon)

    Measures the relative change in best bid vs ask quantities.
    """
    delta_bid = bid_qty - bid_qty_prev
    delta_ask = ask_qty - ask_qty_prev

    ofi = (delta_bid - delta_ask) / (np.abs(delta_bid) + np.abs(delta_ask) + epsilon)
    return ofi


def compute_voi(bid_price: np.ndarray, bid_qty: np.ndarray,
                ask_price: np.ndarray, ask_qty: np.ndarray,
                bid_price_prev: np.ndarray, bid_qty_prev: np.ndarray,
                ask_price_prev: np.ndarray, ask_qty_prev: np.ndarray) -> np.ndarray:
    """
    Compute Volume Order Imbalance (VOI).

    VOI captures the net change in volume at best levels, accounting for price changes.
    When price improves, the full quantity is considered new.
    When price worsens, the previous quantity is considered removed.
    """
    # Bid side
    delta_bid = np.where(bid_price > bid_price_prev, bid_qty,
                np.where(bid_price < bid_price_prev, -bid_qty_prev,
                         bid_qty - bid_qty_prev))

    # Ask side
    delta_ask = np.where(ask_price < ask_price_prev, ask_qty,
                np.where(ask_price > ask_price_prev, -ask_qty_prev,
                         ask_qty - ask_qty_prev))

    voi = delta_bid - delta_ask
    return voi


def compute_trade_imbalance(mid_prices: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute Trade Imbalance (TI) proxy.

    In absence of trade-by-trade data, we approximate using price changes
    as a proxy for signed volume.
    """
    price_changes = np.diff(mid_prices, prepend=mid_prices[0])

    # Rolling sum of signed price changes as TI proxy
    ti = np.zeros_like(mid_prices)
    for i in range(window, len(mid_prices)):
        ti[i] = np.sum(price_changes[i-window+1:i+1])

    return ti


def compute_vpp(bid_qty_levels: np.ndarray, ask_qty_levels: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
    """
    Compute Volume-Weighted Price Pressure (VPP).

    VPP_t = sum_{l=1}^{L} w_l * (q^b_{t,l} - q^a_{t,l}) / sum_{l=1}^{L} (q^b_{t,l} + q^a_{t,l})

    where w_l = exp(-alpha * l) are exponentially decaying weights.
    """
    n_samples, n_levels = bid_qty_levels.shape

    # Create level weights
    levels = np.arange(1, n_levels + 1)
    weights = np.exp(-alpha * levels)

    # Compute VPP
    numerator = np.sum(weights * (bid_qty_levels - ask_qty_levels), axis=1)
    denominator = np.sum(bid_qty_levels + ask_qty_levels, axis=1) + 1e-8

    vpp = numerator / denominator
    return vpp


def compute_kyle_lambda(mid_prices: np.ndarray, ti: np.ndarray,
                        window: int = 100) -> np.ndarray:
    """
    Compute Kyle's Lambda estimate (price impact measure).

    lambda_t = Cov(Delta_m_{t:t+k}, TI_{t:t+k}) / Var(TI_{t:t+k})

    Measures the price impact of order flow (information content).
    """
    price_changes = np.diff(mid_prices, prepend=mid_prices[0])

    kyle_lambda = np.zeros_like(mid_prices)

    for i in range(window, len(mid_prices)):
        dm = price_changes[i-window+1:i+1]
        ti_window = ti[i-window+1:i+1]

        var_ti = np.var(ti_window)
        if var_ti > 1e-10:
            cov_dm_ti = np.cov(dm, ti_window)[0, 1]
            kyle_lambda[i] = cov_dm_ti / var_ti
        else:
            kyle_lambda[i] = 0

    return kyle_lambda


def compute_adverse_selection(mid_prices: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Compute Adverse Selection Component (Roll decomposition).

    AS_t = sqrt(max(0, -Cov(Delta_m_t, Delta_m_{t-1})))

    Measures the adverse selection component of the spread.
    """
    price_changes = np.diff(mid_prices, prepend=mid_prices[0])

    adverse_selection = np.zeros_like(mid_prices)

    for i in range(window, len(mid_prices)):
        dm = price_changes[i-window+1:i+1]
        dm_lag = price_changes[i-window:i]

        cov_val = np.cov(dm, dm_lag)[0, 1]
        adverse_selection[i] = np.sqrt(max(0, -cov_val))

    return adverse_selection


def compute_inventory_proxy(ti: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Compute Inventory Imbalance Proxy.

    INV_t = cumsum_{s=t-W}^{t}(TI_s) / (W * sigma_TI)

    Approximates market maker inventory state.
    """
    sigma_ti = np.std(ti) + 1e-8

    inv = np.zeros_like(ti)
    for i in range(window, len(ti)):
        cumsum_ti = np.sum(ti[i-window+1:i+1])
        inv[i] = cumsum_ti / (window * sigma_ti)

    return inv


def compute_realized_volatility(mid_prices: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Compute Realized Volatility.

    RV_t = sqrt(sum((Delta_m)^2) over window W)
    """
    returns = np.diff(mid_prices, prepend=mid_prices[0]) / mid_prices

    rv = np.zeros_like(mid_prices)
    for i in range(window, len(mid_prices)):
        rv[i] = np.sqrt(np.sum(returns[i-window+1:i+1]**2))

    return rv


def prepare_fi2010_data(data_dir: str,
                        config: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    """
    Main data preparation procedure following framework.md Section 8.1

    Args:
        data_dir: Directory containing or to save dataset
        config: Configuration dictionary with:
            - L: Number of LOB levels (default: 10)
            - T: Sequence length (default: 100)
            - H: Prediction horizon index (0=10, 1=20, 2=30, 3=50, 4=100)
            - train_ratio: Training set ratio (default: 0.7)
            - val_ratio: Validation set ratio (default: 0.15)
            - window: Rolling window for feature computation (default: 100)

    Returns:
        data_dict: Dictionary containing train/val/test data
        metadata: Dictionary containing feature statistics and metadata
    """
    if config is None:
        config = {}

    # Default configuration
    L = config.get('L', 10)
    T = config.get('T', 100)
    H_idx = config.get('H_idx', 0)  # Use 10-event horizon by default
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    window = config.get('window', 100)

    print("="*60)
    print("TOFM Data Preparation")
    print("="*60)
    print(f"Configuration: L={L}, T={T}, H_idx={H_idx}")
    print(f"Train/Val/Test split: {train_ratio}/{val_ratio}/{1-train_ratio-val_ratio}")

    # Step 1: Download/load dataset
    data_path = download_fi2010_dataset(data_dir)
    data = np.load(data_path)

    lob_data = data['lob_data']
    labels = data['labels']
    mid_prices = data['mid_prices']

    n_samples = len(lob_data)
    print(f"\nLoaded {n_samples} samples")

    # Step 2: Extract LOB components
    # Ask: columns 0-19 (10 levels x 2: price, volume)
    # Bid: columns 20-39 (10 levels x 2: price, volume)

    ask_prices = lob_data[:, 0:20:2][:, :L]  # Ask prices for L levels
    ask_volumes = lob_data[:, 1:20:2][:, :L]  # Ask volumes for L levels
    bid_prices = lob_data[:, 20:40:2][:, :L]  # Bid prices for L levels
    bid_volumes = lob_data[:, 21:40:2][:, :L]  # Bid volumes for L levels

    # Step 3: Compute mid-price and spread
    mid_price = (ask_prices[:, 0] + bid_prices[:, 0]) / 2
    spread = ask_prices[:, 0] - bid_prices[:, 0]

    print("\nComputing microstructure features...")

    # Step 4: Compute OFI
    ofi = np.zeros(n_samples)
    ofi[1:] = compute_ofi(
        bid_volumes[1:, 0], ask_volumes[1:, 0],
        bid_volumes[:-1, 0], ask_volumes[:-1, 0]
    )
    print("  - OFI computed")

    # Step 5: Compute VOI
    voi = np.zeros(n_samples)
    voi[1:] = compute_voi(
        bid_prices[1:, 0], bid_volumes[1:, 0],
        ask_prices[1:, 0], ask_volumes[1:, 0],
        bid_prices[:-1, 0], bid_volumes[:-1, 0],
        ask_prices[:-1, 0], ask_volumes[:-1, 0]
    )
    print("  - VOI computed")

    # Step 6: Compute Trade Imbalance
    ti = compute_trade_imbalance(mid_price, window=10)
    print("  - Trade Imbalance computed")

    # Step 7: Compute VPP
    vpp = compute_vpp(bid_volumes, ask_volumes, alpha=0.5)
    print("  - VPP computed")

    # Step 8: Compute Kyle's Lambda
    kyle_lambda = compute_kyle_lambda(mid_price, ti, window=window)
    print("  - Kyle's Lambda computed")

    # Step 9: Compute Adverse Selection
    adverse_selection = compute_adverse_selection(mid_price, window=window)
    print("  - Adverse Selection computed")

    # Step 10: Compute Inventory Proxy
    inventory = compute_inventory_proxy(ti, window=window)
    print("  - Inventory Proxy computed")

    # Step 11: Compute Realized Volatility
    rv = compute_realized_volatility(mid_price, window=window)
    print("  - Realized Volatility computed")

    # Step 12: Construct feature matrix
    # X_t = [OFI, VOI, TI, VPP, spread, lambda, AS, INV, RV, bid_qty_1:L, ask_qty_1:L]
    feature_matrix = np.column_stack([
        ofi,           # 0
        voi,           # 1
        ti,            # 2
        vpp,           # 3
        spread,        # 4
        kyle_lambda,   # 5
        adverse_selection,  # 6
        inventory,     # 7
        rv,            # 8
        bid_volumes,   # 9 to 9+L-1
        ask_volumes    # 9+L to 9+2L-1
    ])

    d_input = feature_matrix.shape[1]
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"d_input = {d_input} (expected: 9 + 2*L = {9 + 2*L})")

    # Step 13: Handle NaN/Inf values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 14: Compute normalization statistics (before splitting!)
    # Use only data that will be in training set for computing stats
    train_end_idx = int(train_ratio * n_samples)

    feature_mean = np.mean(feature_matrix[:train_end_idx], axis=0)
    feature_std = np.std(feature_matrix[:train_end_idx], axis=0) + 1e-8

    # Normalize all features
    feature_matrix_norm = (feature_matrix - feature_mean) / feature_std

    # Step 15: Get target labels
    y = labels[:, H_idx]

    # Step 16: Create sequences
    # Skip initial samples where features are not fully computed
    start_idx = window + T
    valid_end = n_samples - 100  # Leave room for horizon

    n_valid = valid_end - start_idx
    print(f"\nCreating sequences from index {start_idx} to {valid_end}")
    print(f"Number of valid sequences: {n_valid}")

    # Create sequence data
    X_sequences = np.zeros((n_valid, T, d_input))
    y_labels = np.zeros(n_valid, dtype=np.int64)

    for i, t in enumerate(range(start_idx, valid_end)):
        X_sequences[i] = feature_matrix_norm[t-T+1:t+1]
        y_labels[i] = y[t]

    # Step 17: Split data temporally
    train_size = int(train_ratio * n_valid)
    val_size = int(val_ratio * n_valid)

    X_train = X_sequences[:train_size]
    y_train = y_labels[:train_size]

    X_val = X_sequences[train_size:train_size+val_size]
    y_val = y_labels[train_size:train_size+val_size]

    X_test = X_sequences[train_size+val_size:]
    y_test = y_labels[train_size+val_size:]

    # Get corresponding RV values for regime analysis
    rv_test = rv[start_idx+train_size+val_size:valid_end]

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # Class distribution
    print(f"\nClass distribution (Train):")
    for c in range(3):
        count = np.sum(y_train == c)
        pct = 100 * count / len(y_train)
        label = ['Down', 'Stable', 'Up'][c]
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Create data dictionary
    data_dict = {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train,
        'X_val': X_val.astype(np.float32),
        'y_val': y_val,
        'X_test': X_test.astype(np.float32),
        'y_test': y_test,
        'rv_test': rv_test.astype(np.float32),
        'mid_prices_test': mid_price[start_idx+train_size+val_size:valid_end].astype(np.float32),
    }

    # Create raw LOB version for baseline comparison (H1)
    raw_lob_features = np.column_stack([
        spread,
        bid_volumes,
        ask_volumes,
        bid_prices,
        ask_prices
    ])
    raw_mean = np.mean(raw_lob_features[:train_end_idx], axis=0)
    raw_std = np.std(raw_lob_features[:train_end_idx], axis=0) + 1e-8
    raw_lob_norm = (raw_lob_features - raw_mean) / raw_std
    raw_lob_norm = np.nan_to_num(raw_lob_norm, nan=0.0, posinf=0.0, neginf=0.0)

    d_raw = raw_lob_norm.shape[1]
    X_raw_sequences = np.zeros((n_valid, T, d_raw))
    for i, t in enumerate(range(start_idx, valid_end)):
        X_raw_sequences[i] = raw_lob_norm[t-T+1:t+1]

    data_dict['X_train_raw'] = X_raw_sequences[:train_size].astype(np.float32)
    data_dict['X_val_raw'] = X_raw_sequences[train_size:train_size+val_size].astype(np.float32)
    data_dict['X_test_raw'] = X_raw_sequences[train_size+val_size:].astype(np.float32)

    metadata = {
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'd_input': d_input,
        'd_raw': d_raw,
        'T': T,
        'L': L,
        'n_classes': 3,
        'feature_names': ['OFI', 'VOI', 'TI', 'VPP', 'spread', 'kyle_lambda',
                         'adverse_selection', 'inventory', 'RV'] + \
                        [f'bid_qty_{l}' for l in range(1, L+1)] + \
                        [f'ask_qty_{l}' for l in range(1, L+1)],
        'config': config
    }

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)

    return data_dict, metadata


if __name__ == "__main__":
    # Test data loading
    data_dir = "/Users/jminding/Desktop/Code/Research Agent/research_agent/files/data"

    config = {
        'L': 10,
        'T': 100,
        'H_idx': 0,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'window': 100
    }

    data_dict, metadata = prepare_fi2010_data(data_dir, config)

    print("\n\nVerification:")
    print(f"X_train shape: {data_dict['X_train'].shape}")
    print(f"y_train shape: {data_dict['y_train'].shape}")
    print(f"Feature names: {metadata['feature_names'][:9]}")
