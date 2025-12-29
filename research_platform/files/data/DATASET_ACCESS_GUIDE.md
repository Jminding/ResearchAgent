# Quick Dataset Access Guide

**Last Updated**: 2025-12-24

---

## Synthetic SBM Graphs (PRIMARY DATASET)

**Generation Script**:
```bash
cd /Users/jminding/Desktop/Code/Research\ Agent/research_platform
python scripts/generate_sbm_graphs.py
```

**Output Location**: `data/synthetic/sbm/*.pt`

**Expected Files**: 135 graph files (3 sizes × 9 homophily levels × 5 replications)

**No download needed** - generated locally

---

## Elliptic Bitcoin Dataset (VALIDATION)

### Quick Download (Kaggle API)

**Prerequisites**:
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
# Get API token from: https://www.kaggle.com/settings
```

**Download Commands**:
```bash
# Create directory
mkdir -p data/real/elliptic

# Download dataset
kaggle datasets download -d ellipticco/elliptic-data-set -p data/real/elliptic/

# Unzip
cd data/real/elliptic
unzip elliptic-data-set.zip
cd ../../..

# Verify files
ls -lh data/real/elliptic/
# Should see:
# - elliptic_txs_features.csv
# - elliptic_txs_classes.csv
# - elliptic_txs_edgelist.csv
```

**Preprocessing**:
```bash
python scripts/preprocess_elliptic.py
```

**Output**: `data/processed/elliptic.pt`

### Manual Download (No API)

1. Go to: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Click "Download" (requires Kaggle login)
3. Extract zip to `data/real/elliptic/`

---

## IEEE-CIS Fraud Detection (OPTIONAL)

### Quick Download (Kaggle API)

```bash
# Create directory
mkdir -p data/real/ieee_cis

# Download competition data
kaggle competitions download -c ieee-fraud-detection -p data/real/ieee_cis/

# Unzip
cd data/real/ieee_cis
unzip ieee-fraud-detection.zip
cd ../../..

# Verify files
ls -lh data/real/ieee_cis/
# Should see:
# - train_transaction.csv
# - train_identity.csv
# - test_transaction.csv
# - test_identity.csv
# - sample_submission.csv
```

**Graph Construction** (requires custom implementation):
```bash
python scripts/construct_ieee_graph.py
```

**Note**: This dataset is OPTIONAL. Focus on SBM + Elliptic first.

---

## Elliptic++ Extended (ADVANCED)

**GitHub Clone**:
```bash
cd data/real/
git clone https://github.com/git-disl/EllipticPlusPlus.git
cd EllipticPlusPlus/
# Follow repository instructions
```

**Note**: Requires heterogeneous GNN methods - beyond primary experiment scope.

---

## Complete Setup Script

Run this to set up all data directories and download public datasets:

```bash
#!/bin/bash
# File: scripts/setup_data.sh

set -e  # Exit on error

echo "=== Setting up data directories ==="
mkdir -p data/synthetic/sbm
mkdir -p data/real/elliptic
mkdir -p data/real/ieee_cis
mkdir -p data/processed

echo "=== Downloading Elliptic dataset ==="
if command -v kaggle &> /dev/null; then
    kaggle datasets download -d ellipticco/elliptic-data-set -p data/real/elliptic/
    cd data/real/elliptic && unzip -q elliptic-data-set.zip && cd ../../..
    echo "Elliptic downloaded successfully"
else
    echo "Kaggle CLI not found. Install with: pip install kaggle"
    echo "Manual download: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
fi

echo "=== Generating SBM graphs ==="
if [ -f scripts/generate_sbm_graphs.py ]; then
    python scripts/generate_sbm_graphs.py
    echo "SBM graphs generated successfully"
else
    echo "Warning: scripts/generate_sbm_graphs.py not found. Create this script first."
fi

echo "=== Preprocessing Elliptic ==="
if [ -f scripts/preprocess_elliptic.py ]; then
    python scripts/preprocess_elliptic.py
    echo "Elliptic preprocessed successfully"
else
    echo "Warning: scripts/preprocess_elliptic.py not found. Create this script first."
fi

echo "=== Data setup complete ==="
echo "Check data/ directories for downloaded and generated files"
```

**Run setup**:
```bash
chmod +x scripts/setup_data.sh
./scripts/setup_data.sh
```

---

## Dataset URLs (Direct Links)

| Dataset | URL |
|---------|-----|
| **Elliptic Bitcoin** | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set |
| **Elliptic Paper** | https://arxiv.org/abs/1908.02591 |
| **Elliptic++ GitHub** | https://github.com/git-disl/EllipticPlusPlus |
| **IEEE-CIS Competition** | https://www.kaggle.com/c/ieee-fraud-detection |
| **IEEE-CIS Data Page** | https://www.kaggle.com/c/ieee-fraud-detection/data |
| **NetworkX SBM Docs** | https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.stochastic_block_model.html |
| **Kaggle API Setup** | https://www.kaggle.com/docs/api |

---

## Verification After Download

### Check Elliptic Files

```bash
# File sizes
ls -lh data/real/elliptic/

# Expected:
# elliptic_txs_features.csv   (~140 MB)
# elliptic_txs_classes.csv    (~3 MB)
# elliptic_txs_edgelist.csv   (~5 MB)

# Row counts
wc -l data/real/elliptic/*.csv
# Expected:
# 203,770 elliptic_txs_features.csv (203,769 + header)
# 203,770 elliptic_txs_classes.csv
# 234,356 elliptic_txs_edgelist.csv (234,355 + header)
```

### Check SBM Generation

```bash
# Count generated files
ls data/synthetic/sbm/*.pt | wc -l
# Expected: 135

# Check metadata
cat data/synthetic/sbm/metadata.json | python -m json.tool

# Load and verify one graph
python -c "
import torch
data = torch.load('data/synthetic/sbm/n1000_h0.10_rep0.pt')
print(f'Nodes: {data.num_nodes}')
print(f'Edges: {data.edge_index.shape[1]}')
print(f'Features: {data.x.shape}')
print(f'Labels: {data.y.unique()}')
print(f'Homophily: {data.homophily:.3f}')
"
```

### Check IEEE-CIS Files (if downloaded)

```bash
ls -lh data/real/ieee_cis/

# Expected files:
# train_transaction.csv  (~600 MB)
# train_identity.csv     (~30 MB)
# test_transaction.csv   (~500 MB)
# test_identity.csv      (~30 MB)

# Row counts
wc -l data/real/ieee_cis/train_*.csv
# train_transaction.csv: 590,541 rows (590,540 + header)
# train_identity.csv: 144,234 rows (144,233 + header)
```

---

## Troubleshooting

### Kaggle API Authentication Failed

**Error**: `401 - Unauthorized`

**Solution**:
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place in `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Dataset Already Downloaded

**Error**: `File already exists`

**Solution**:
```bash
# Force redownload
kaggle datasets download -d ellipticco/elliptic-data-set -p data/real/elliptic/ --force

# Or delete and redownload
rm -rf data/real/elliptic/*
kaggle datasets download -d ellipticco/elliptic-data-set -p data/real/elliptic/
```

### SBM Generation Too Slow

**Problem**: Generating 135 graphs takes too long

**Solution**: Parallelize generation
```python
# In generate_sbm_graphs.py, use multiprocessing
from multiprocessing import Pool

def generate_single_graph(config):
    n, h, rep = config
    # ... generation code ...
    return filepath

configs = [(n, h, rep) for n in [1000, 5000, 10000]
                        for h in np.arange(0.1, 0.55, 0.05)
                        for rep in range(5)]

with Pool(processes=8) as pool:
    pool.map(generate_single_graph, configs)
```

### Out of Memory (Large Graphs)

**Problem**: 10k node graphs crash

**Solution**: Generate smaller batches or reduce largest size
```python
# Option 1: Generate in batches
for size in [1000, 5000, 10000]:
    for h in homophily_range:
        for rep in range(5):
            generate_and_save()  # Generate one at a time

# Option 2: Reduce max size
node_counts = [1000, 5000]  # Skip 10000
```

---

## Expected Disk Space

| Dataset | Size | Location |
|---------|------|----------|
| Elliptic (raw) | ~150 MB | `data/real/elliptic/` |
| Elliptic (processed) | ~200 MB | `data/processed/elliptic.pt` |
| IEEE-CIS (raw) | ~1.5 GB | `data/real/ieee_cis/` |
| IEEE-CIS (processed) | ~500 MB | `data/processed/ieee_cis.pt` |
| SBM graphs (135 files) | ~50-100 MB | `data/synthetic/sbm/` |
| **Total** | ~2-3 GB | `data/` |

Make sure you have at least **5 GB free** to be safe.

---

## Data Loading in Python

### Load SBM Graph

```python
import torch

# Load single graph
data = torch.load('data/synthetic/sbm/n1000_h0.10_rep0.pt')

# Access properties
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.edge_index.shape[1]}")
print(f"Features: {data.x.shape}")
print(f"Labels: {data.y.shape}")
print(f"Homophily: {data.homophily}")
```

### Load Elliptic Dataset

```python
import torch

# Load preprocessed Elliptic
data = torch.load('data/processed/elliptic.pt')

# Access splits
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Train set statistics
train_labels = data.y[train_mask]
print(f"Train nodes: {train_mask.sum()}")
print(f"Train anomalies: {(train_labels == 1).sum()}")
```

### Load All SBM Graphs for Experiment

```python
import torch
from pathlib import Path

# Load all graphs of specific size
sbm_dir = Path('data/synthetic/sbm')
n = 1000
h_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

graphs_by_h = {h: [] for h in h_values}

for graph_file in sbm_dir.glob(f'n{n}_h*.pt'):
    data = torch.load(graph_file)
    h = data.homophily
    # Round to nearest 0.05 to handle float precision
    h_key = round(h * 20) / 20
    graphs_by_h[h_key].append(data)

# Now graphs_by_h[0.1] contains 5 replications of h=0.1 graphs
print(f"Loaded {sum(len(g) for g in graphs_by_h.values())} graphs")
```

---

## Next Steps After Data Acquisition

1. **Verify all datasets**:
   ```bash
   python scripts/verify_datasets.py
   ```

2. **Generate summary statistics**:
   ```bash
   python scripts/dataset_statistics.py > files/data/dataset_stats.txt
   ```

3. **Visualize sample graphs**:
   ```bash
   python scripts/visualize_datasets.py
   ```

4. **Ready for modeling**: Proceed to model implementation phase

---

## Contact & Support

**Issues with Kaggle datasets**: Check Kaggle dataset discussion forums
**Issues with SBM generation**: Check NetworkX GitHub issues
**Issues with this project**: Contact research team

---

**Document Status**: Complete
**Last Verified**: 2025-12-24
**Maintenance**: Update if dataset URLs or access methods change
