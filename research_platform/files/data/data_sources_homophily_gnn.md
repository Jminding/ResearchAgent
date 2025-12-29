# Data Sources for GNN Homophily Research

**Research Topic**: Graph Neural Network Performance Across Homophily Regimes
**Date**: 2025-12-24
**Status**: COMPLETE - Ready for implementation

---

## Executive Summary

**Primary Dataset**: Synthetic Stochastic Block Model (SBM) graphs
**Validation Dataset**: Elliptic Bitcoin transaction network
**Optional Dataset**: IEEE-CIS Fraud Detection (requires graph construction)

**Justification**: Controlled experiments require systematic homophily variation (h ∈ [0.1, 0.5]), which only synthetic data can provide. Real-world datasets validate findings transfer to practical applications.

---

## Dataset 1: Synthetic SBM Graphs (PRIMARY)

### Classification
- **Type**: Synthetic, generated via Stochastic Block Model
- **Purpose**: PRIMARY experimental dataset
- **Task**: Binary anomaly detection

### Specifications
- **Node counts**: 1,000 | 5,000 | 10,000
- **Homophily range**: 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 (9 levels)
- **Feature dimension**: 64
- **Classes**: 2 (normal, anomaly)
- **Class balance**: 50/50 (balanced for controlled experiments)
- **Average degree**: 10
- **Edge count**: ~n × 5 (n × avg_degree / 2)
- **Replications**: 5 per configuration
- **Total graphs**: 3 sizes × 9 homophily × 5 reps = **135 graphs**

### Homophily Control
- **Formula**: h = (edges within same class) / (total edges)
- **Implementation**: Stochastic Block Model with probability matrix
  - p_in = h × d / (n/2) [intra-class probability]
  - p_out = (1-h) × d / (n/2) [inter-class probability]
- **Validation**: Verify |h_actual - h_target| < 0.05

### Feature Generation
- **Normal class**: Gaussian N(-1.0, 0.5²I)
- **Anomaly class**: Gaussian N(+1.0, 0.5²I)
- **Separability**: Moderate (not trivially separable, requires graph structure)

### Access & Generation
- **Source**: Generated locally using NetworkX
- **Script**: `scripts/generate_sbm_graphs.py`
- **Library**: `networkx.generators.community.stochastic_block_model`
- **Storage**: `data/synthetic/sbm/*.pt` (PyTorch Geometric format)
- **License**: Public domain (generated data)

### Known Issues
None - fully controlled by generation parameters

### Why PRIMARY?
1. **Homophily control**: Only way to systematically vary h from 0.1 to 0.5
2. **Causal isolation**: Fix all confounds (degree, features, size), vary only homophily
3. **Ground truth**: Perfect labels, known homophily by construction
4. **Replications**: Generate 5+ instances per config for statistical rigor
5. **Reproducibility**: Deterministic from seed + parameters

---

## Dataset 2: Elliptic Bitcoin Transactions (VALIDATION)

### Classification
- **Type**: Real-world financial network
- **Source**: Elliptic company, released via Kaggle
- **Purpose**: VALIDATION on real heterophilic fraud network
- **Task**: Anti-money laundering (AML) - detect illicit Bitcoin transactions

### URL & Access
- **Kaggle**: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Paper**: Weber et al. (2019) - https://arxiv.org/abs/1908.02591
- **GitHub (Elliptic++)**: https://github.com/git-disl/EllipticPlusPlus
- **Download**: `kaggle datasets download -d ellipticco/elliptic-data-set`

### Size & Structure
- **Nodes**: 203,769 Bitcoin transactions
- **Edges**: 234,355 directed Bitcoin flows
- **Node features**: 166 (94 local + 72 aggregate transaction properties)
- **Time steps**: 49 temporal snapshots
- **Temporal span**: Unknown real-world period

### Labels
- **Classes**: 3 (illicit, licit, unknown)
- **Labeled nodes**: 46,564 (22.84%)
- **Unlabeled nodes**: 157,205 (77.16%)
- **Illicit (anomaly)**: 4,545 transactions (9.77% of labeled)
- **Licit (normal)**: 42,019 transactions (90.23% of labeled)

### Graph Properties
- **Homophily**: < 0.3 (HETEROPHILIC)
  - Fraud transactions connect to legitimate ones
  - Money laundering deliberately mixes illicit/licit flows
- **Average degree**: 2.30 (sparse)
- **Directed**: Yes (Bitcoin flow direction)
- **Temporal**: Yes (edges respect time ordering)
- **Weighted**: Transaction amounts available but not typically used

### Homophily Characteristics
**Why heterophilic?**
- Money launderers route through legitimate intermediaries
- Fraudsters transact with unsuspecting victims
- Illicit actors avoid clustering with other criminals (detection evasion)
- Result: Illicit nodes preferentially connect to licit nodes

**Measured homophily**: ~0.2-0.3 (on labeled subset)
**Implication**: Ideal testbed for validating GNN performance on low-homophily graphs

### Train/Val/Test Splits
**Temporal splits** (respect time ordering):
- **Train**: Time steps 1-34 (~70% chronologically)
- **Validation**: Time steps 35-43 (~15%)
- **Test**: Time steps 44-49 (~15%)

**Rationale**: Simulates real deployment (train on past, predict future)

### Feature Engineering
**Required preprocessing**:
1. **Standardization**: Z-score normalize all 166 features
2. **Temporal handling**: Time step embedded in features (no extra engineering)
3. **Missing values**: Minimal (verify during preprocessing)

**No additional engineering needed** - features provided by Elliptic are research-ready

### Files
1. `elliptic_txs_features.csv` - (203,769 × 167) node features + IDs
2. `elliptic_txs_classes.csv` - (203,769 × 2) transaction IDs + labels
3. `elliptic_txs_edgelist.csv` - (234,355 × 2) edge list (directed)

### License
- **Type**: Available via Kaggle (check platform terms of service)
- **Attribution**: Cite Weber et al. (2019) paper
- **Commercial use**: Unclear - assume research/academic only
- **Note**: Widely used public benchmark in AML research

### Known Issues
1. **High unlabeled ratio**: 77% nodes unlabeled
   - Solution: Use only labeled nodes (supervised mode)
   - Alternative: Semi-supervised learning (future work)
2. **Class imbalance**: 9.77% anomalies
   - Solution: Use F1-score, precision-recall, AUC (not just accuracy)
3. **Temporal drift**: Distribution shift across time steps
   - Solution: Temporal splits respect time ordering
4. **Heterophilic**: Challenges standard GNNs
   - Solution: This is the point - validates need for homophily-aware models

### Expected Performance
- **GCN baseline**: ~90% accuracy (misleading due to imbalance), F1 ~0.70-0.75
- **GNN with attention**: AUC-ROC ~0.85-0.90
- **Hypothesis**: Models trained on high-homophily SBM will underperform on Elliptic

### Why VALIDATION?
- Real-world graph structure (native, not constructed)
- Known heterophilic properties (matches low-h SBM regime)
- Standard benchmark with published baselines
- Tests if SBM findings generalize to actual financial fraud

---

## Dataset 3: IEEE-CIS Fraud Detection (OPTIONAL)

### Classification
- **Type**: Real-world e-commerce transactions (TABULAR, not graph)
- **Source**: Vesta Corporation + IEEE Computational Intelligence Society
- **Purpose**: OPTIONAL validation if time permits
- **Task**: Credit card fraud detection

### URL & Access
- **Kaggle Competition**: https://www.kaggle.com/c/ieee-fraud-detection
- **Data page**: https://www.kaggle.com/c/ieee-fraud-detection/data
- **Download**: `kaggle competitions download -c ieee-fraud-detection`

### Size & Structure
- **Transactions**: 590,540 e-commerce transactions
- **Features**: 434 (394 transaction + 41 identity, joined on TransactionID)
- **Temporal span**: ~6 months
- **Train/test split**: Temporal (1-month gap between train and test)

### Labels
- **Classes**: 2 (fraud, legitimate)
- **Fraud rate**: 3.5% (20,663 fraud, 569,877 legitimate)
- **Class imbalance**: Extreme (96.5% normal)

### CRITICAL LIMITATION: No Native Graph Structure

**Problem**: IEEE-CIS is a TABULAR dataset. There are NO edges provided.

**Implication**: Must manually construct graph, which introduces:
1. **Construction ambiguity**: Multiple valid methods (card-based, temporal, k-NN, heterogeneous)
2. **Unknown homophily**: Depends entirely on construction method
3. **Reproducibility issues**: Different researchers build different graphs
4. **Scientific validity**: Results depend on construction choices, not just model

### Graph Construction Strategies

**Option 1: Card-based graph**
- Connect transactions by same card (use anonymized card proxy features)
- Edges represent card history
- Homophily: Unknown (fraud patterns per card)

**Option 2: Temporal k-NN graph**
- Connect transactions close in time and feature space
- k-NN on (time, amount, location proxies)
- Homophily: Likely low (fraud mixed with normal)

**Option 3: Heterogeneous graph**
- Node types: transactions, cards, users, merchants
- Edge types: transaction-to-card, card-to-user, etc.
- Requires heterogeneous GNN methods

**Option 4: Hybrid**
- Combine multiple edge types
- Complex but potentially more realistic

### Estimated Graph Properties (post-construction)
- **Nodes**: 100,000-590,540 (depends on filtering)
- **Edges**: 500,000-2,000,000 (depends on construction)
- **Homophily**: UNKNOWN - must measure after construction
- **Average degree**: Highly variable

### Feature Engineering
**Required (extensive)**:
1. **Missing values**: >50% missing in many features
   - Imputation: Median (numeric), 'missing' category (categorical)
2. **Categorical encoding**: Features id_12 to id_38
   - One-hot, target encoding, or frequency encoding
3. **Temporal features**: TransactionDT (seconds from reference)
   - Hour of day, day of week, time since last transaction
4. **Aggregations**: Per-card statistics (mean amount, fraud rate, etc.)
5. **Dimensionality reduction**: 434 → 64-128 (PCA or autoencoder)

**Warning**: Feature engineering choices significantly impact results

### Files
1. `train_transaction.csv` - (590,540 × 394)
2. `train_identity.csv` - (144,233 × 41) - only subset have identity info
3. `test_transaction.csv` - (506,691 × 393) - no labels provided
4. `test_identity.csv` - (141,907 × 41)

### License
- **Type**: Kaggle competition
- **Attribution**: Cite Vesta + IEEE-CIS
- **Commercial use**: Restricted (competition rules)
- **Note**: Test labels never publicly released (competition leaderboard only)

### Known Issues
1. **No native graph**: Graph construction is research design choice, not ground truth
2. **High dimensionality**: 434 features, many anonymized (V1-V339)
3. **Missing data**: Extensive missingness across features
4. **Extreme imbalance**: 3.5% fraud (worse than Elliptic's 9.77%)
5. **Test labels unavailable**: Can only evaluate on train set holdout
6. **Anonymization**: Features are obfuscated (V1, C1, D1, M1 codes)

### Recommendation

**Use IEEE-CIS ONLY IF**:
- Primary SBM experiments are complete
- Elliptic validation is complete
- Time permits additional validation
- You commit to documenting graph construction method in detail

**Otherwise**: Focus on SBM (primary) + Elliptic (validation) for robust, reproducible science.

**If using IEEE-CIS**:
1. Clearly state graph construction method in paper
2. Measure and report homophily of constructed graph
3. Compare multiple construction methods
4. Acknowledge construction as limitation

### Why OPTIONAL (not recommended for primary experiments)?
- Not a native graph (construction introduces confounds)
- Unknown homophily (depends on construction)
- Requires extensive feature engineering
- Test labels unavailable (limits evaluation)
- Adds complexity without clear scientific benefit over Elliptic

---

## Comparative Summary

| Property | SBM (Synthetic) | Elliptic (Real) | IEEE-CIS (Real) |
|----------|-----------------|-----------------|-----------------|
| **Priority** | PRIMARY | VALIDATION | OPTIONAL |
| **Nodes** | 1k-10k | 203,769 | ~500k |
| **Edges** | 5k-50k | 234,355 | TBD (construct) |
| **Features** | 64 | 166 | 434 |
| **Anomaly %** | 50% (balanced) | 9.77% | 3.5% |
| **Homophily** | 0.1-0.5 (controlled) | <0.3 (heterophilic) | Unknown |
| **Graph type** | Native (SBM) | Native (Bitcoin) | NONE (tabular) |
| **Temporal** | No | Yes (49 steps) | Yes (~6 months) |
| **Labels** | 100% | 22.84% | 100% train |
| **License** | Public domain | Kaggle | Kaggle competition |
| **Construction** | Generate | Load | MUST BUILD |
| **Preprocessing** | None | Standardization | Extensive |
| **Reproducibility** | Perfect (seed) | High (fixed data) | Medium (construction varies) |

---

## Data Acquisition Timeline

### Immediate (Week 1)
1. Implement `scripts/generate_sbm_graphs.py`
2. Generate 135 SBM graphs with h ∈ [0.1, 0.5]
3. Validate homophily values and save to `data/synthetic/sbm/`

### Week 1-2
4. Download Elliptic dataset via Kaggle API
5. Implement `scripts/preprocess_elliptic.py`
6. Create temporal train/val/test splits
7. Save preprocessed data to `data/processed/elliptic.pt`

### Optional (if time permits)
8. Download IEEE-CIS dataset
9. Design and implement graph construction method
10. Document construction rationale
11. Preprocess features and construct graph

---

## Validation Checklist

After data acquisition, verify:

- [ ] **SBM graphs**
  - [ ] 135 graph files generated
  - [ ] Homophily values within ±0.05 of target
  - [ ] Average degree ≈ 10
  - [ ] Feature class separation (t-SNE visualization)
  - [ ] metadata.json saved

- [ ] **Elliptic dataset**
  - [ ] 3 CSV files downloaded
  - [ ] 203,769 nodes loaded
  - [ ] 234,355 edges loaded
  - [ ] Features standardized (mean≈0, std≈1)
  - [ ] Train/val/test masks created (temporal splits)
  - [ ] Homophily computed on labeled subset

- [ ] **IEEE-CIS (if using)**
  - [ ] 4 CSV files downloaded
  - [ ] Graph constructed with documented method
  - [ ] Edges validated (degree distribution, components)
  - [ ] Features preprocessed (imputation, encoding)
  - [ ] Homophily measured and reported

---

## Citation Requirements

### Elliptic Dataset
```bibtex
@article{weber2019anti,
  title={Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics},
  author={Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Daniel Karl I and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E},
  journal={arXiv preprint arXiv:1908.02591},
  year={2019}
}
```

### Elliptic++ (if using extended version)
```bibtex
@inproceedings{alarfaj2023demystifying,
  title={Demystifying fraudulent transactions and illicit nodes in the bitcoin network for financial forensics},
  author={Alarfaj, Farah Kadhim and Goswami, Kishan and Yoo, Tai-Won and Alarfaj, Hussain and Faloutsos, Christos and others},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```

### IEEE-CIS Dataset (if using)
```
IEEE-CIS Fraud Detection Dataset. Kaggle competition.
Provided by Vesta Corporation and IEEE Computational Intelligence Society.
https://www.kaggle.com/c/ieee-fraud-detection (Accessed: 2025-12-24)
```

### Stochastic Block Model
```bibtex
@software{networkx,
  title={NetworkX: Network Analysis in Python},
  author={Hagberg, Aric and Swart, Pieter and S Chult, Daniel},
  year={2008},
  url={https://networkx.org}
}
```

---

## Contact Information

**Data Acquisition Questions**:
- Elliptic: Contact via Kaggle dataset page or Elliptic company
- IEEE-CIS: Kaggle competition forum
- SBM generation: NetworkX documentation

**Dataset Licenses**:
- Elliptic: Check Kaggle terms (assumed academic/research use)
- IEEE-CIS: Kaggle competition rules
- SBM (generated): Public domain / CC0

---

## Summary of Decisions

### PRIMARY: Synthetic SBM Graphs
**Reason**: Only way to systematically control homophily for causal analysis. Real datasets have fixed, unknown homophily that cannot be manipulated.

### VALIDATION: Elliptic Bitcoin Dataset
**Reason**: Real-world heterophilic graph (h<0.3) with native structure. Tests if SBM findings generalize to actual financial fraud networks.

### OPTIONAL: IEEE-CIS Fraud Detection
**Reason**: No native graph structure - requires construction. Use only if time permits and construction method is thoroughly documented.

### Experimental Strategy
1. **Develop and tune models on SBM** across h ∈ [0.1, 0.5]
2. **Validate on Elliptic** to confirm real-world applicability
3. **Optionally test on IEEE-CIS** if constructed graph is scientifically justified

This three-tier approach balances:
- **Internal validity** (controlled experiments via SBM)
- **External validity** (real-world validation via Elliptic)
- **Practical relevance** (large-scale benchmark via IEEE-CIS if time permits)

---

**Document Status**: COMPLETE
**Date**: 2025-12-24
**Next Action**: Begin SBM graph generation (Week 1)
