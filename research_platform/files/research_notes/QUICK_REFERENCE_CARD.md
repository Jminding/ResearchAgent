# GNN Anomaly Detection - Quick Reference Card

## Performance at a Glance

### AUC Score Ranges
```
Supervised:              [0.80 - 0.99]  ████████████████████ (Best: 0.99)
Semi-supervised:         [0.85 - 0.97]  ███████████████████  (Best: 0.97)
Contrastive Unsupervised:[0.88 - 0.95]  ██████████████████   (Best: 0.95)
Reconstruction Unsuper.: [0.82 - 0.93]  ██████████████        (Best: 0.93)
Density-based Unsuper.:  [0.85 - 0.93]  ██████████████        (Best: 0.93)
```

### F1 Score Ranges
```
Supervised:              [0.85 - 0.99]  ████████████████████ (Best: 0.99)
Semi-supervised:         [0.80 - 0.95]  ████████████████      (Best: 0.95)
Contrastive Unsupervised:[0.85 - 0.97]  ██████████████████    (Best: 0.97)
Reconstruction Unsuper.: [0.75 - 0.92]  ██████████████        (Best: 0.92)
Density-based Unsuper.:  [0.80 - 0.90]  ██████████████        (Best: 0.90)
```

### Accuracy (Supervised Only)
```
Range: [76% - 99%]
Typical: 85-95%
Best: 99% (Firewall logs)
```

---

## Method Selection Quick Guide

### Choose Based on Labeled Data Availability

**I have 0% labels (Unsupervised)**
→ Use: **EAGLE** (Contrastive Learning)
- AUC: 0.88-0.95 | F1: 0.85-0.97
- Cost: Requires pre-training (higher CPU cost)
- Benefit: 15% better than DOMINANT

**I have 1-10% labels (Semi-supervised)**
→ Use: **TSAD** (for temporal) or **Generative** (for static)
- F1: 0.80-0.95 | AUC: 0.85-0.97
- Cost: Moderate labeling effort
- Benefit: Practical balance

**I have >50% labels (Supervised)**
→ Use: **GCN-GAT Hybrid**
- F1: 0.85-0.99 | Accuracy: 76-99%
- Cost: High annotation effort
- Benefit: Highest detection rates

---

## Best Method by Application Domain

| Domain | Best Method | F1/AUC | Key Strength |
|--------|------------|--------|------------|
| **Industrial Control Systems** | GDN | Precision 0.99 | High precision on SWaT |
| **Water/Distribution** | GeneralDyG | F1 0.60 | Handles temporal drift |
| **Financial Fraud** | RL-GNN Fusion | AUROC 0.87 | Community-aware detection |
| **Network Security** | GCN-GAT Hybrid | F1 98.7% | Hybrid architecture |
| **Time Series on Graphs** | GAT + Informer | State-of-art | Temporal dependencies |
| **Blockchain/Transactions** | MDST-GNN | F1 0.85-0.92 | Multi-distance features |
| **General Purpose** | EAGLE | AUC 0.88-0.95 | Versatile, strong performance |

---

## Real-World Results (2024-2025)

### Best Performing Systems

**Firewall Logs (Network Security)**
```
Method: GCN-GAT Hybrid (Supervised)
Recall:    99.04%  ████████████████████
Precision: 98.43%  ████████████████████
F1 Score:  98.72%  ████████████████████
```

**Industrial Systems (SWaT)**
```
Method: GeneralDyG (Supervised)
F1: 85.19%         █████████████████
Precision: 0.99    ████████████████████
```

**Water Distribution (WADI)**
```
Method: GeneralDyG (Supervised)
F1: 60.43%         ████████████
F-measure Gain: 54% ████████████████████
```

**Vibration Data (Manufacturing)**
```
Method: GCN-VAE (Unsupervised)
Accuracy: 88.9%    ██████████████████
F1: 88.3%          ██████████████████
AUC: 0.93          █████████████████
```

**Financial Fraud (Blockchain)**
```
Method: RL-GNN Fusion (Supervised)
AUROC: 0.872       ██████████████████
F1: 0.839          █████████████████
Precision: 0.683   ███████████████
```

---

## Computational Characteristics

### Inference Speed
```
Typical:     8.7 ms per sample         [Real-time capable]
Max safe:    50 ms per sample          [Acceptable for most apps]
For >50ms:   Consider distributed/batch processing
```

### Memory Scaling
```
Nodes:       Up to 1M handled
Edges:       Proportional to density
Features:    Optimal <1,500 dimensions
             Peak throughput: >20,000 samples/sec
```

### Training Requirements
```
GDN (intensive):        Days on GPU
Most methods:           Hours on GPU
SmoothGNN:             Hours on GPU
Inference:             Seconds for batch
```

---

## Critical Pitfalls (Know These!)

### Metric Issues
- [ ] **F1-score is contamination-dependent** - Use AUC as primary metric
- [ ] **Threshold selection is manual** - Validate on held-out test set
- [ ] **Biased evaluation inflates results** - Use proper train/test split

### Method Issues
- [ ] **Anomaly overfitting** - Models memorize anomalies instead of learning normal patterns
- [ ] **Homophily assumption violation** - Fails when anomalies have normal-like neighbors
- [ ] **Over-smoothing in deep networks** - Representations become indistinguishable

### Data Issues
- [ ] **Label scarcity** - True anomalies expensive to label
- [ ] **Contamination in training** - Unlabeled anomalies reduce performance
- [ ] **Dataset-specific overfitting** - F1 ranges 0.75-0.99; validate on new domains

### Implementation Issues
- [ ] **Scalability degradation above 1,500 features** - Monitor latency
- [ ] **Edge features often neglected** - Check GAE implementation
- [ ] **Dynamic graphs require more computation** - Plan accordingly

---

## Performance Expectations by Scenario

### Scenario 1: Cora-like Synthetic Dataset (3K nodes)
- Expected AUC: 0.85-0.95
- Expected F1: 0.80-0.92
- Training time: Minutes to hours
- Inference: <1ms per node

### Scenario 2: Industrial Control System (SWaT/WADI)
- Expected Precision: 0.98-0.99
- Expected Recall: 0.85-0.95
- Training time: Hours to day
- Inference: <100ms

### Scenario 3: Network Security (Firewall Logs)
- Expected F1: 0.90-0.99
- Expected Accuracy: 95-99%
- Training time: Hours
- Inference: 5-50ms per batch

### Scenario 4: Blockchain Fraud (100K+ transactions)
- Expected AUROC: 0.85-0.92
- Expected F1: 0.80-0.90
- Training time: Hours to day
- Inference: 10-100ms per batch

---

## Method Comparison Matrix

| Method | Type | Paradigm | AUC | F1 | Computational Cost | Interpretability |
|--------|------|----------|-----|----|--------------------|-----------------|
| EAGLE | Contrastive | Unsupervised | 0.88-0.95 | 0.85-0.97 | Medium | Low |
| SmoothGNN | Reconstruction | Unsupervised | 0.85-0.93 | 0.80-0.92 | Low | Medium |
| LUNAR | Density+Deep | Unsupervised | 0.85-0.93 | 0.80-0.90 | Medium | Medium |
| TSAD | Transformer | Semi-sup | 0.85-0.97 | 0.80-0.95 | High | Low |
| GDN | Reconstruction | Semi-sup | 0.87-0.99 | 0.85-0.99 | Medium | Medium |
| GCN-GAT | Hybrid | Supervised | 0.90-0.99 | 0.85-0.99 | Medium | Low |
| RL-GNN | RL+Graph | Supervised | 0.87-0.95 | 0.83-0.90 | High | Low |
| GeneralDyG | Dynamic | Supervised | 0.80-0.95 | 0.60-0.85 | High | Low |

---

## Dataset Sizes for Reference

| Dataset | Nodes | Edges | Anomaly % | Classes | Use Case |
|---------|-------|-------|-----------|---------|----------|
| Cora | 2.7K | 5.4K | 5.5% | 7 | Baseline synthetic |
| Citeseer | 3.3K | 4.7K | 4.5% | 6 | Baseline synthetic |
| CoraFull | 19.8K | N/A | 5% | 70 | Larger synthetic |
| OGB-arXiv | 169K | 1.2M | 3.5% | 40 | Large-scale |
| SWaT | N/A | Graph-time | High | 2 | Industrial systems |
| WADI | N/A | Graph-time | High | 2 | Water distribution |
| Bitcoin | Dynamic | Dynamic | ~1% | 2 | Transactions |

---

## Key Numbers to Remember

```
State-of-Art Unsupervised:      AUC 0.88-0.95, F1 0.85-0.97  (EAGLE)
State-of-Art Semi-Supervised:   F1 0.80-0.95, AUC 0.85-0.97  (TSAD)
State-of-Art Supervised:        F1 0.85-0.99, Acc 76-99%     (GCN-GAT)

Best Real-World Result:         F1 98.72%  (Firewall logs)
Most Common AUC Range:          0.85-0.95  (Most methods)
Typical Labeling Required:      1-10% of data for semi-sup
Training Time:                  Hours to days
Inference Time:                 8.7 ms per sample
Max Safe Scalability:           1M nodes, <1,500 features
```

---

## Recommended Decision Tree

```
START
  │
  ├─ Do I have labeled anomaly data?
  │  │
  │  ├─ NO (0% labels)
  │  │  ├─ Use: EAGLE (Contrastive + pre-training)
  │  │  └─ Expect: AUC 0.88-0.95, F1 0.85-0.97
  │  │
  │  ├─ MAYBE (1-10% labels)
  │  │  ├─ Is data temporal/dynamic?
  │  │  │  ├─ YES → Use: TSAD
  │  │  │  └─ NO → Use: Generative Semi-supervised
  │  │  └─ Expect: F1 0.80-0.95
  │  │
  │  └─ YES (>50% labels)
  │     ├─ Is data heterogeneous/complex?
  │     │  ├─ YES → Use: GCN-GAT Hybrid
  │     │  └─ NO → Use: Supervised GCN/GAT
  │     └─ Expect: F1 0.85-0.99
  │
  └─ What's your domain?
     ├─ Industrial Systems → Use: GDN
     ├─ Financial Fraud → Use: RL-GNN
     ├─ Network Security → Use: GCN-GAT
     └─ General Purpose → Use: EAGLE or TSAD
```

---

## Validation Checklist Before Deployment

- [ ] Verified AUC on multiple datasets (not just one)
- [ ] Threshold validated on held-out test set with known anomalies
- [ ] Contamination rate in test set documented
- [ ] Both AUC and F1 reported with explicit contamination
- [ ] Computational cost validated (inference time, memory)
- [ ] Edge cases tested (no anomalies, all anomalies, etc.)
- [ ] Drift monitoring plan in place
- [ ] Retraining schedule established
- [ ] Baseline performance established for comparison
- [ ] False positive/negative trade-off understood and acceptable

---

## Common Mistakes to Avoid

❌ **DON'T**:
- Use only F1-score for evaluation (too sensitive to contamination)
- Assume one method will work across all domains
- Ignore computational costs in your environment
- Skip threshold validation on test data
- Use training data threshold for production
- Report results without contamination rate
- Forget to account for class imbalance

✅ **DO**:
- Report both AUC and F1 with explicit contamination rate
- Validate on multiple datasets before claiming generalization
- Monitor both inference latency and memory usage
- Tune threshold on validation set
- Use separate test set for final evaluation
- Compare against domain-specific baselines
- Document all hyperparameter choices

---

## Where to Look for Information

**Quick Stats** → This card (QUICK_REFERENCE_CARD.md)
**Method Details** → literature review (lit_review_gnn_anomaly_detection.md)
**Quantitative Data** → evidence sheet (evidence_sheet.json)
**Executive Summary** → summary document (GNN_ANOMALY_DETECTION_SUMMARY.md)
**Navigation** → README (README_GNN_RESEARCH.md)

---

## Last Updated: December 24, 2025
**Version**: 1.0
**Data Quality**: Verified against peer-reviewed publications
**Coverage**: 15+ papers, 2018-2025, with emphasis on 2023-2025

