# Dataset Documentation for GNN Homophily Research

**Project**: Graph Neural Network Performance Analysis Across Homophily Regimes
**Date**: 2025-12-24
**Version**: 1.0
**Status**: Data Acquisition Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Selection Rationale](#dataset-selection-rationale)
3. [Primary Dataset: Synthetic SBM Graphs](#primary-dataset-synthetic-sbm-graphs)
4. [Validation Dataset: Elliptic Bitcoin](#validation-dataset-elliptic-bitcoin)
5. [Optional Dataset: IEEE-CIS Fraud Detection](#optional-dataset-ieee-cis-fraud-detection)
6. [Feature Engineering Guidelines](#feature-engineering-guidelines)
7. [Graph Construction Details](#graph-construction-details)
8. [Data Access Instructions](#data-access-instructions)
9. [Reproducibility Checklist](#reproducibility-checklist)

---

## Overview

This research investigates how graph homophily affects GNN performance on anomaly detection tasks. The experimental design requires:

- **Systematic homophily control**: Graphs with h ∈ [0.1, 0.5] to isolate homophily's causal effect
- **Anomaly detection task**: Binary classification with ~10% anomaly rate
- **Controlled experiments**: Fixed node counts, feature dimensions, and class balance
- **Statistical rigor**: Multiple replications per configuration
- **Real-world validation**: Test on actual financial fraud networks

This document explains why synthetic Stochastic Block Model (SBM) graphs are the PRIMARY dataset, with real-world datasets used for VALIDATION.

---

## Dataset Selection Rationale

### Why Synthetic Data is PRIMARY

| Requirement | Real Datasets | Synthetic SBM | Decision |
|-------------|---------------|---------------|----------|
| **Homophily control** | Fixed (cannot vary) | Precise control h ∈ [0.1, 0.5] | **SBM required** |
| **Causal isolation** | Confounded factors | Vary only homophily | **SBM required** |
| **Ground truth labels** | Partial (Elliptic 77% unlabeled) | Complete by construction | **SBM preferred** |
| **Replications** | Single instance | Unlimited generations | **SBM preferred** |
| **Reproducibility** | Access issues, updates | Deterministic from seed | **SBM preferred** |
| **Real-world validity** | Native financial networks | Must validate externally | **Real data needed** |

**Conclusion**: Synthetic SBM for controlled experiments + Elliptic for real-world validation = Gold standard methodology.

### Scientific Precedent

This approach follows established practices in:
- Physics: Controlled lab experiments validated by field observations
- Chemistry: Pure reagent studies validated by natural samples
- Machine Learning: Toy datasets (XOR, spirals) validated on MNIST/CIFAR
- Graph Learning: Synthetic benchmarks (Cora-Full splits, Planetoid) validated on citation networks

Recent GNN papers (NeurIPS 2020-2024) increasingly use synthetic graphs with controlled properties for causal analysis, then validate on real benchmarks.

---

## Primary Dataset: Synthetic SBM Graphs

### Generation Methodology

**Stochastic Block Model (SBM)** is a principled generative model that creates graphs with community structure and controllable homophily.

#### Mathematical Formulation

Given:
- `n` nodes divided into `K` classes
- Class sizes: `sizes = [n_1, n_2, ..., n_K]`
- Probability matrix: `P[i,j]` = probability of edge between class `i` and class `j`

For binary classification (K=2) with balanced classes:
- `sizes = [n/2, n/2]`
- `P = [[p_in, p_out], [p_out, p_in]]`

**Homophily formula**:
```
h = (# edges within same class) / (# total edges)
  = |{(u,v) ∈ E : y_u = y_v}| / |E|
```

**Parameter derivation** (for target homophily `h` and average degree `d`):
```
p_in = h * d / (n/2)           # Intra-class edge probability
p_out = (1-h) * d / (n/2)      # Inter-class edge probability
```

#### Implementation

```python
import networkx as nx
import torch
from torch_geometric.data import Data

def generate_sbm_graph(n, h, d, feature_dim=64, anomaly_class=1, seed=42):
    """
    Generate SBM graph with controlled homophily.

    Parameters:
    - n: Number of nodes
    - h: Target homophily (0.1 to 0.5)
    - d: Average degree
    - feature_dim: Node feature dimension
    - anomaly_class: Which class is anomaly (0 or 1)
    - seed: Random seed for reproducibility

    Returns:
    - PyTorch Geometric Data object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Class sizes (balanced)
    sizes = [n // 2, n // 2]

    # Compute probabilities
    p_in = h * d / (n / 2)
    p_out = (1 - h) * d / (n / 2)
    P = [[p_in, p_out],
         [p_out, p_in]]

    # Generate graph
    G = nx.stochastic_block_model(sizes, P, seed=seed)

    # Create labels (class 0 = normal, class 1 = anomaly)
    labels = torch.zeros(n, dtype=torch.long)
    labels[n//2:] = 1

    # Generate class-conditional features
    # Normal class: N(-1, σ²I)
    # Anomaly class: N(+1, σ²I)
    sigma = 0.5
    features = torch.randn(n, feature_dim) * sigma
    features[:n//2] -= 1.0  # Normal class mean
    features[n//2:] += 1.0  # Anomaly class mean

    # Convert to edge index
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Create PyG Data object
    data = Data(x=features, edge_index=edge_index, y=labels)

    # Validation
    actual_h = compute_edge_homophily(data)
    assert abs(actual_h - h) < 0.05, f"Homophily mismatch: target {h}, actual {actual_h}"

    return data

def compute_edge_homophily(data):
    """Compute edge homophily ratio."""
    edge_index = data.edge_index
    y = data.y
    same_class = (y[edge_index[0]] == y[edge_index[1]]).sum().item()
    total_edges = edge_index.shape[1]
    return same_class / total_edges
```

### Experimental Configuration

| Parameter | Values | Justification |
|-----------|--------|---------------|
| `n` (nodes) | [1000, 5000, 10000] | Test scalability across sizes |
| `h` (homophily) | [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] | Fine-grained sweep from heterophilic to moderate |
| `d` (avg degree) | 10 | Typical sparse real-world networks |
| `feature_dim` | 64 | Standard GNN input dimension |
| `K` (classes) | 2 | Binary anomaly detection |
| `anomaly_rate` | 0.5 | Balanced for controlled experiments (adjust with masking if needed) |
| `replications` | 5 | Statistical significance with confidence intervals |
| `seeds` | [42, 43, 44, 45, 46] | Independent random instantiations |

**Total graphs**: 3 sizes × 9 homophily levels × 5 replications = **135 graphs**

### Validation Checks

After generation, verify each graph:

1. **Homophily accuracy**: `|h_actual - h_target| < 0.05`
2. **Degree distribution**: Mean degree ≈ `d`, reasonable variance
3. **Connectivity**: Single connected component (or handle largest component)
4. **Feature separability**: t-SNE plot shows class separation
5. **Edge count**: Approximately `n * d / 2` edges

### Storage Format

```
data/synthetic/sbm/
├── n1000_h0.10_rep0.pt
├── n1000_h0.10_rep1.pt
├── ...
├── n10000_h0.50_rep4.pt
└── metadata.json
```

Each `.pt` file contains a PyTorch Geometric `Data` object with:
- `data.x`: Node features [n, 64]
- `data.edge_index`: Edge connectivity [2, num_edges]
- `data.y`: Node labels [n]
- `data.num_nodes`: Number of nodes
- `data.homophily`: Actual computed homophily

---

## Validation Dataset: Elliptic Bitcoin

### Overview

**Elliptic Dataset** is a real-world Bitcoin transaction graph for anti-money laundering (AML) research, released by Elliptic company.

**Key Properties**:
- Native graph structure (edges = Bitcoin flow)
- Heterophilic (fraud connected to legitimate transactions)
- Temporal (49 time steps)
- Labeled anomalies (illicit vs. licit transactions)

### Statistics

| Property | Value |
|----------|-------|
| Nodes | 203,769 transactions |
| Edges | 234,355 Bitcoin flows |
| Node features | 166 (94 local + 72 aggregate) |
| Time steps | 49 |
| Labeled nodes | 46,564 (22.84%) |
| Unlabeled nodes | 157,205 (77.16%) |
| Illicit (anomaly) | 4,545 (9.77% of labeled) |
| Licit (normal) | 42,019 (90.23% of labeled) |
| Homophily | < 0.3 (heterophilic) |
| Avg degree | 2.30 |

### Graph Properties

**Heterophilic Structure**: Fraud networks are inherently heterophilic because:
- Money launderers mix illicit funds with legitimate transactions
- Fraudsters transact with unsuspecting victims
- Illicit activity is deliberately obfuscated within normal flows

This makes Elliptic an ideal validation dataset for testing GNN performance on low-homophily graphs.

### Data Files

1. **elliptic_txs_features.csv**
   - Shape: (203,769, 167)
   - Column 0: Transaction ID
   - Columns 1-94: Local features (transaction-specific properties)
   - Columns 95-166: Aggregate features (transaction neighborhood statistics)

2. **elliptic_txs_classes.csv**
   - Shape: (203,769, 2)
   - Column 0: Transaction ID
   - Column 1: Class (1=illicit, 2=licit, unknown=unlabeled)

3. **elliptic_txs_edgelist.csv**
   - Shape: (234,355, 2)
   - Each row: (txId1, txId2) representing Bitcoin flow

### Preprocessing Pipeline

```python
import pandas as pd
import torch
from torch_geometric.data import Data

def load_elliptic_dataset(data_dir):
    """
    Load and preprocess Elliptic dataset.

    Returns:
    - data: PyG Data object
    - time_steps: Array of time step for each node
    """
    # Load files
    features = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
    classes = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
    edges = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')

    # Create node ID mapping
    node_ids = features[0].values
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Extract features (columns 1-166)
    X = torch.tensor(features.iloc[:, 1:].values, dtype=torch.float)

    # Standardize features
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)

    # Extract time steps (column 1 in features)
    time_steps = features.iloc[:, 1].values

    # Map classes: unknown->-1, licit->0, illicit->1
    class_map = {'unknown': -1, '2': 0, '1': 1}
    classes['class'] = classes['class'].astype(str).map(class_map)
    y = torch.tensor(classes['class'].values, dtype=torch.long)

    # Build edge index
    edge_list = []
    for _, row in edges.iterrows():
        src, dst = row['txId1'], row['txId2']
        if src in id_to_idx and dst in id_to_idx:
            edge_list.append([id_to_idx[src], id_to_idx[dst]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create masks
    train_mask = (time_steps <= 34) & (y != -1)
    val_mask = (time_steps > 34) & (time_steps <= 43) & (y != -1)
    test_mask = (time_steps > 43) & (y != -1)

    # Create Data object
    data = Data(
        x=X,
        edge_index=edge_index,
        y=y,
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool)
    )

    return data, time_steps
```

### Temporal Splits

Elliptic has natural temporal ordering across 49 time steps. Use time-based splits to respect causality:

- **Train**: Time steps 1-34 (first ~70%)
- **Validation**: Time steps 35-43 (next ~15%)
- **Test**: Time steps 44-49 (final ~15%)

This simulates realistic deployment where models predict future fraud based on historical data.

### Handling Unlabeled Nodes

Two strategies:

1. **Supervised (default)**: Use only labeled nodes (train/val/test masks filter y != -1)
2. **Semi-supervised (optional)**: Train on labeled, predict on unlabeled, evaluate on labeled test set

For primary experiments, use **supervised** mode to match SBM setup.

### Expected Performance

Based on literature:
- Baseline GCN: ~90-92% accuracy (but high class imbalance)
- F1-score: 0.70-0.75 (more meaningful metric)
- AUC-ROC: 0.85-0.90
- Attention-based GNNs: Often outperform standard GCN/GAT on this heterophilic graph

Expect models trained on high-homophily SBM graphs to struggle initially on Elliptic, validating the importance of homophily-aware architectures.

---

## Optional Dataset: IEEE-CIS Fraud Detection

### Overview

Large-scale e-commerce fraud detection dataset from Kaggle competition (2019). Contains 590,540 transactions with 434 features.

**Important**: This is a **tabular dataset** with NO native graph structure. Graph construction is required.

### Statistics

| Property | Value |
|----------|-------|
| Transactions | 590,540 |
| Features | 434 (394 transaction + 41 identity) |
| Fraud rate | 3.5% |
| Temporal span | ~6 months |
| Train/test split | Temporal (1-month gap) |

### Graph Construction Challenge

Must manually construct graph edges. Options:

1. **Card-based graph**: Connect transactions by same card (proxy via anonymized card features)
2. **Temporal graph**: Connect consecutive transactions within time window
3. **k-NN graph**: Connect similar transactions by feature distance
4. **Heterogeneous graph**: Model users, cards, merchants as different node types

**Problem**: Construction method dramatically affects:
- Node/edge counts
- Homophily level
- Model performance
- Scientific conclusions

### Recommendation

**Use IEEE-CIS only if**:
1. SBM and Elliptic experiments are complete
2. Time permits additional validation
3. You clearly document and justify graph construction method

**Otherwise**: Focus on SBM (primary) + Elliptic (validation) for robust, reproducible findings.

### If Using IEEE-CIS

Suggested construction:

```python
def construct_ieee_graph(transactions, identity, k=5, time_window=3600):
    """
    Construct transaction graph from IEEE-CIS tabular data.

    Parameters:
    - transactions: Transaction features DataFrame
    - identity: Identity features DataFrame
    - k: Number of nearest neighbors for k-NN edges
    - time_window: Time window (seconds) for temporal edges

    Returns:
    - PyG Data object with constructed graph
    """
    # Merge transaction and identity
    df = transactions.merge(identity, on='TransactionID', how='left')

    # Strategy 1: Card-based edges
    # Group by card features (C1-C14) and connect within groups
    card_edges = []
    for card_group in df.groupby(['C1', 'C2', 'C3']).groups.values():
        nodes = list(card_group)
        # Fully connect within card (or limit to temporal sequence)
        for i in range(len(nodes)-1):
            card_edges.append([nodes[i], nodes[i+1]])

    # Strategy 2: Temporal k-NN edges
    # Sort by time, compute k-NN in sliding window
    df = df.sort_values('TransactionDT')
    # ... (k-NN implementation based on feature similarity)

    # Combine edges and create Data object
    # ... (similar to Elliptic preprocessing)
```

**Warning**: Different construction methods yield different homophily levels and scientific conclusions. Document thoroughly.

---

## Feature Engineering Guidelines

### SBM Graphs

**No feature engineering needed** - features are generated with controlled class separation.

Features follow class-conditional Gaussians:
- Normal class: `N(-1, 0.5²I)`
- Anomaly class: `N(+1, 0.5²I)`

This ensures:
- Features are informative but not perfectly separable
- GNN must leverage graph structure for good performance
- Feature quality is constant across homophily levels

### Elliptic Dataset

**Required preprocessing**:

1. **Standardization**: Features have different scales - apply z-score normalization
   ```python
   X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
   ```

2. **Temporal features**: Time step is embedded in features - no additional engineering needed

3. **Missing values**: Check for NaNs (unlikely but verify)

4. **Feature selection (optional)**: 166 features is high-dimensional
   - Option 1: Use all features (standard)
   - Option 2: PCA to 64 dimensions (match SBM dimension)
   - Option 3: Use only local features (94 dims) or aggregate features (72 dims)

**Recommended**: Use all 166 features with standardization for most realistic comparison.

### IEEE-CIS Dataset

**Extensive feature engineering required** (if using this dataset):

1. **Missing value imputation**: Many features have >50% missing
   - Numeric: Median or -999 sentinel
   - Categorical: 'missing' category

2. **Categorical encoding**: Features id_12 to id_38
   - One-hot encoding (sparse)
   - Target encoding (risk of leakage)
   - Frequency encoding

3. **Temporal features**: TransactionDT is seconds from reference
   - Hour of day: `hour = (TransactionDT // 3600) % 24`
   - Day of week: `day = (TransactionDT // 86400) % 7`
   - Time since last transaction (per card)

4. **Aggregation features**: Per-card statistics
   - Mean/std transaction amount
   - Transaction count in past 24 hours
   - Fraud rate in card history (use only past data to avoid leakage)

5. **Dimensionality reduction**: 434 features → embed to 64-128 dims
   - PCA, autoencoders, or random projection

**Warning**: Feature engineering choices significantly impact results. Document all transformations.

---

## Graph Construction Details

### SBM Graphs

Constructed via NetworkX `stochastic_block_model()`:

```python
G = nx.stochastic_block_model(sizes, P, seed=seed)
```

**Parameters**:
- `sizes = [n/2, n/2]`: Balanced classes
- `P = [[p_in, p_out], [p_out, p_in]]`: Probability matrix
- `seed`: Reproducibility

**Edge properties**:
- Undirected
- No self-loops
- No multi-edges
- Sparse (avg degree = 10)

### Elliptic Dataset

**Native graph structure** - edges represent Bitcoin flows:

```
Transaction A --[amount]--> Transaction B
```

**Properties**:
- Directed (Bitcoin flow direction)
- Temporal (edges respect time ordering)
- Weighted (transaction amounts - but not used in standard benchmarks)

**In practice**: Treat as undirected for GNN simplicity (or use directed GNN variants).

### IEEE-CIS Dataset

**No native graph** - must construct edges. Recommended method:

**Card-based temporal graph**:
1. Group transactions by card (use card proxy features)
2. Sort transactions by time within each card group
3. Add directed edge from transaction `i` to `i+1` in same card sequence
4. Optionally add k-NN edges between similar transactions across cards

**Resulting graph properties**:
- Partially directed (temporal) or undirected (k-NN)
- Sparse if only card sequences
- Dense if k-NN with large k
- Homophily unknown (depends on fraud patterns)

**Validation**: After construction, compute:
- Number of nodes (≈ 590k or after filtering)
- Number of edges
- Degree distribution
- Connected components
- Homophily ratio (fraud-fraud edge percentage)

---

## Data Access Instructions

### SBM Graphs (Primary)

**Generation script**: `scripts/generate_sbm_graphs.py`

```bash
cd /Users/jminding/Desktop/Code/Research Agent/research_platform
python scripts/generate_sbm_graphs.py \
    --node_counts 1000 5000 10000 \
    --homophily_range 0.1 0.5 \
    --homophily_steps 9 \
    --replications 5 \
    --avg_degree 10 \
    --feature_dim 64 \
    --output_dir data/synthetic/sbm/
```

**Output**: 135 `.pt` files in `data/synthetic/sbm/`

**Verification**: Check `data/synthetic/sbm/metadata.json` for statistics.

---

### Elliptic Dataset (Validation)

**Source**: Kaggle - https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

**Method 1: Kaggle API** (recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (place kaggle.json in ~/.kaggle/)
# Download: https://www.kaggle.com/settings → "Create New API Token"

# Download dataset
kaggle datasets download -d ellipticco/elliptic-data-set -p data/real/elliptic/

# Unzip
unzip data/real/elliptic/elliptic-data-set.zip -d data/real/elliptic/
```

**Method 2: Manual download**
1. Go to https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Click "Download" (requires Kaggle account)
3. Extract to `data/real/elliptic/`

**Expected files**:
```
data/real/elliptic/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv
```

**Preprocessing**: `scripts/preprocess_elliptic.py`

---

### IEEE-CIS Dataset (Optional)

**Source**: Kaggle Competition - https://www.kaggle.com/c/ieee-fraud-detection

**Access**:
```bash
kaggle competitions download -c ieee-fraud-detection -p data/real/ieee_cis/
unzip data/real/ieee_cis/ieee-fraud-detection.zip -d data/real/ieee_cis/
```

**Expected files**:
```
data/real/ieee_cis/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
├── test_identity.csv
└── sample_submission.csv
```

**Graph construction**: `scripts/construct_ieee_graph.py` (must implement)

---

### Elliptic++ Dataset (Advanced)

**Source**: GitHub - https://github.com/git-disl/EllipticPlusPlus

**Access**:
```bash
cd data/real/
git clone https://github.com/git-disl/EllipticPlusPlus.git
cd EllipticPlusPlus
# Follow repository instructions for data loading
```

**Note**: Requires heterogeneous GNN methods - beyond scope of primary experiments.

---

## Reproducibility Checklist

To ensure your experiments are fully reproducible:

### Data Generation
- [ ] SBM generation script with fixed random seeds
- [ ] Document exact NetworkX and PyTorch versions
- [ ] Save metadata.json with all generation parameters
- [ ] Verify homophily values match targets (tolerance < 0.05)

### Data Preprocessing
- [ ] Document Elliptic preprocessing steps (standardization, masking)
- [ ] Save preprocessed data objects (.pt files)
- [ ] Record train/val/test split indices
- [ ] If using IEEE-CIS, document graph construction method in detail

### Feature Engineering
- [ ] List all feature transformations (scaling, imputation, encoding)
- [ ] Save feature statistics (mean, std) used for normalization
- [ ] Document any feature selection or dimensionality reduction

### Code and Environment
- [ ] Python version (recommend 3.9+)
- [ ] PyTorch version (recommend 2.0+)
- [ ] PyTorch Geometric version (recommend 2.3+)
- [ ] NetworkX version (recommend 3.0+)
- [ ] requirements.txt with pinned versions

### Data Availability
- [ ] SBM graphs saved in `data/synthetic/sbm/`
- [ ] Elliptic raw data in `data/real/elliptic/`
- [ ] Preprocessed Elliptic in `data/processed/elliptic.pt`
- [ ] README with data access instructions
- [ ] License information for all datasets

### Verification
- [ ] Compute and report homophily for all graphs
- [ ] Report degree distributions
- [ ] Check for data leakage (test nodes in train set)
- [ ] Visualize sample graphs (t-SNE, graph layouts)

---

## Contact and Attribution

**Dataset Sources**:
- **Elliptic**: Elliptic company, Kaggle platform
- **IEEE-CIS**: Vesta Corporation, IEEE-CIS, Kaggle competition
- **SBM generation**: NetworkX library (BSD-3-Clause license)

**Citations**:

```bibtex
@article{weber2019anti,
  title={Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics},
  author={Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Daniel Karl I and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E},
  journal={arXiv preprint arXiv:1908.02591},
  year={2019}
}

@inproceedings{alarfaj2022demystifying,
  title={Demystifying fraudulent transactions and illicit nodes in the bitcoin network for financial forensics},
  author={Alarfaj, Farah Kadhim and others},
  booktitle={KDD Workshop on Anomaly Detection in Finance},
  year={2022}
}
```

**Licenses**:
- Elliptic: Available on Kaggle (check platform terms)
- IEEE-CIS: Kaggle competition rules apply
- Generated SBM data: Public domain (CC0) for research use

---

## Appendix: Homophily Formulas

### Edge Homophily (Primary)

```
h_edge = |{(u,v) ∈ E : y_u = y_v}| / |E|
```

- Counts fraction of edges connecting same-class nodes
- Range: [0, 1]
- 0 = perfect heterophily (no same-class edges)
- 1 = perfect homophily (all same-class edges)

### Node Homophily (Alternative)

```
h_node = (1/n) Σ_v |{u ∈ N(v) : y_u = y_v}| / |N(v)|
```

- Average fraction of same-class neighbors per node
- More sensitive to degree distribution
- Less common in GNN literature

**For this research**: Use **edge homophily** as primary metric.

---

## Appendix: Expected Dataset Statistics

After generation and preprocessing:

| Dataset | Nodes | Edges | Features | Anomaly % | Homophily | Degree |
|---------|-------|-------|----------|-----------|-----------|--------|
| SBM (small) | 1,000 | ~5,000 | 64 | 50% | 0.1-0.5 | 10 |
| SBM (medium) | 5,000 | ~25,000 | 64 | 50% | 0.1-0.5 | 10 |
| SBM (large) | 10,000 | ~50,000 | 64 | 50% | 0.1-0.5 | 10 |
| Elliptic | 203,769 | 234,355 | 166 | 9.77% | <0.3 | 2.3 |
| IEEE-CIS | ~590,000 | TBD | 64-128 | 3.5% | TBD | TBD |

---

## Version History

- **v1.0** (2025-12-24): Initial documentation
  - Dataset selection rationale
  - SBM generation methodology
  - Elliptic preprocessing pipeline
  - IEEE-CIS graph construction guidelines
  - Reproducibility checklist

---

**Document Status**: Complete
**Next Steps**:
1. Implement `scripts/generate_sbm_graphs.py`
2. Implement `scripts/preprocess_elliptic.py`
3. Download Elliptic dataset
4. Generate SBM graphs with h ∈ [0.1, 0.5]
5. Validate homophily values and graph properties

**Data Acquisition Specialist Agent**
Date: 2025-12-24
