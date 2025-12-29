# Graph Neural Network Anomaly Detection Research Materials

## Document Index and Navigation Guide

This research package contains a comprehensive literature review and evidence synthesis on **Anomaly Detection Techniques using Graph Neural Networks**, covering unsupervised, semi-supervised, and supervised approaches with reconstruction-based, distance-based, and density-based methodologies.

---

## Core Research Documents

### 1. **lit_review_gnn_anomaly_detection.md** (Primary Literature Review)
**Location**: `/files/research_notes/lit_review_gnn_anomaly_detection.md`

**Content**:
- Comprehensive overview of the research area
- Chronological development from 2018-2025
- Detailed methodology overview for all major approaches:
  - Reconstruction-based methods (DOMINANT, SmoothGNN, ADA-GAD, G3AD)
  - Distance-based methods (MDST-GNN, Graph Deviation Network)
  - Density-based methods (LUNAR, FRAUDAR)
  - Contrastive learning methods (EAGLE, ANEMONE, DE-GAD)
- Learning paradigm analysis (unsupervised, semi-supervised, supervised)
- GNN backbone architectures (GCN, GAT, GIN, hybrids)
- Dataset benchmarks and evaluation protocols
- Computational complexity analysis
- Identified research gaps and open problems
- State-of-the-art summary (2025)
- Quality assessment and recommendations for future research

**Use Case**: For writing literature review sections of papers; understanding methodological developments and trends

**Key Metrics Covered**:
- AUC ranges: [0.72, 0.99]
- F1-score ranges: [0.75, 0.99]
- Precision/Recall ranges: [0.87, 0.99]
- Accuracy ranges: [0.76, 0.99]

---

### 2. **evidence_sheet.json** (Quantitative Evidence Database)
**Location**: `/files/research_notes/evidence_sheet.json`

**Content Structure**:
```json
{
  "metric_ranges": {
    "auc_unsupervised": [0.82, 0.95],
    "auc_semi_supervised": [0.85, 0.97],
    "auc_supervised": [0.80, 0.99],
    "f1_unsupervised": [0.75, 0.92],
    "f1_semi_supervised": [0.80, 0.95],
    "f1_supervised": [0.85, 0.99],
    ...
  },
  "method_performance_benchmarks": {
    "dominant_gae": {...},
    "eagle_contrastive": {...},
    "smoothgnn": {...},
    ...
  },
  "known_pitfalls": [...],
  "key_references": [...]
}
```

**Key Features**:
- 18 metric ranges with min/max values
- Performance benchmarks for 16 major methods
- 20 identified pitfalls with explanations
- 15 key references with findings and impact
- Typical sample sizes for 10+ datasets
- Computational characteristics
- Future research priorities

**Use Case**: For experimental design, setting realistic performance thresholds, identifying methodological pitfalls, quick-reference performance lookup

**Data Quality**: Cross-verified from peer-reviewed papers, with sources traceable to published works

---

### 3. **GNN_ANOMALY_DETECTION_SUMMARY.md** (Executive Summary)
**Location**: `/files/research_notes/GNN_ANOMALY_DETECTION_SUMMARY.md`

**Content**:
- Executive summary of key findings
- Performance summary by learning paradigm (unsupervised/semi-supervised/supervised)
- Methodology performance breakdown with comparison tables
- Quantitative evidence summary with visual metric ranges
- Real-world benchmark results
- Critical known pitfalls (12 major categories)
- Dataset benchmarks overview
- State-of-the-art method comparison (2025)
- Identified research gaps (14 major categories)
- Recommendations for practitioners
- Future research directions (near/medium/long-term)
- Key takeaways
- Quick reference tables

**Use Case**: For quick reference, executive presentations, method selection decision trees, deployment recommendations

**Format**: Markdown with extensive tables and visual organization

---

## Quick Reference Tables

### Performance by Learning Paradigm

| Paradigm | AUC Range | F1 Range | Best Method | Use Case |
|----------|-----------|----------|-------------|----------|
| **Unsupervised** | [0.82, 0.95] | [0.75, 0.92] | EAGLE | No labeled data |
| **Semi-supervised** | [0.85, 0.97] | [0.80, 0.95] | TSAD | 1-10% labeled |
| **Supervised** | [0.80, 0.99] | [0.85, 0.99] | GCN-GAT | Full labeling |

### Method Selection Guide

**No Labels (Unsupervised)**:
- Best: **EAGLE** (Contrastive + pre-training) → AUC 0.88-0.95, F1 0.85-0.97
- Alternative: **SmoothGNN** (Reconstruction) → AUC 0.85-0.93
- For subgraphs: **LUNAR** (Density-based) → AUC 0.85-0.93

**Few Labels (Semi-supervised, 1-10%)**:
- Temporal data: **TSAD** (Transformer-based) → F1 >0.80
- Static graphs: **Generative semi-supervised** → F1 0.80-0.95
- Industrial systems: **GDN** (Time-series GNN) → Precision 0.98+

**Full Labels (Supervised)**:
- Hybrid networks: **GCN-GAT** → F1 0.9872 (Firewall logs)
- Fraud detection: **RL-GNN Fusion** → AUROC 0.872
- Dynamic networks: **GeneralDyG** → F1 0.60-0.85

---

## Research Papers Covered (Selected Highlights)

### Foundational Works
- DOMINANT (2019): First GAE for anomaly detection
- Tang et al. (2022, ICML): Critical analysis of GNN limitations
- GDN (2021, AAAI): Graph Deviation Network for time series

### Recent State-of-the-Art (2024-2025)
- **EAGLE** (2025): 15% improvement via contrastive learning + pre-training
- **TSAD** (2024): Transformer-based semi-supervised dynamic graphs
- **GeneralDyG** (2024): Generalizable dynamic graph approach
- **GCN-GAT Hybrid** (2025): 98.72% F1 on firewall logs
- **RL-GNN Fusion** (2025): 0.872 AUROC on financial fraud
- **Deep Graph Anomaly Detection Survey** (2025, TKDE): Comprehensive taxonomy

### Methodological Contributions
- LUNAR (2021): GNN + LOF hybrid
- ANEMONE (2022): Multi-scale contrastive learning
- ADA-GAD (2024): Anomaly-denoised autoencoders
- DE-GAD (2025): Diffusion-enhanced multi-view
- SmoothGNN (2024): Smoothing-aware regularization

---

## Key Quantitative Findings

### Performance Hierarchy
```
Supervised (Full labels):     F1 0.85-0.99, Accuracy 76-99%  [BEST]
Semi-supervised (1-10%):      F1 0.80-0.95, AUC 0.85-0.97
Contrastive Unsupervised:     F1 0.85-0.97, AUC 0.88-0.95    [NEW LEADER]
Reconstruction Unsupervised:  F1 0.75-0.92, AUC 0.82-0.93
Density-based Unsupervised:   F1 0.80-0.90, AUC 0.85-0.93
```

### Real-World Performance Examples

**Industrial Control Systems (SWaT/WADI)**:
- GeneralDyG: F1 0.8519 (SWaT), F1 0.6043 (WADI)
- GDN: Precision 0.99 (SWaT), 0.98 (WADI)

**Network Security (Firewall Logs)**:
- GCN-GAT: Recall 99.04%, Precision 98.43%, F1 98.72%

**Financial Fraud (Blockchain)**:
- RL-GNN: AUROC 0.872, F1 0.839, Average Precision 0.683

**Time Series on Graphs**:
- GCN-VAE: Accuracy 88.9%, Precision 89.1%, Recall 87.6%, AUC 0.93

### Computational Characteristics
- **Inference**: 8.7 ms per flow (real-time capable)
- **Scalability**: Optimal at 1,500 feature dimensions; peak throughput >20,000 samples/sec
- **Max Nodes Handled**: Up to 1M nodes
- **Training**: Hours to days on GPU (method-dependent)

---

## Critical Pitfalls to Avoid

### Metric-Related
1. **F1-Score Bias**: Sensitive to contamination rate; use AUC as primary metric
2. **Threshold Assumption**: Manual thresholding required per dataset
3. **Evaluation Protocol**: Biased protocols can inflate reported scores

### Methodological
4. **Anomaly Overfitting**: Reconstruction models memorize anomalies
5. **Homophily Violation**: Anomalies with similar neighbors evade detection
6. **Over-Smoothing**: Deep networks suffer representation collapse
7. **Label Scarcity**: True anomalies expensive to label
8. **Train-Test Contamination**: Improper data separation

### Computational
9. **Scalability Degradation**: Latency increases above 1,500 dimensions
10. **Memory Explosion**: Exponential growth with adjacency matrices

### Domain-Specific
11. **Edge Feature Neglect**: GAEs often ignore edge characteristics
12. **Graph Structure Assumptions**: Fails on dynamic/partially observed graphs

---

## How to Use This Research Package

### For Literature Review Writing
1. **Start with**: `lit_review_gnn_anomaly_detection.md`
2. **Extract sections**: Copy relevant methodology sections and benchmarks
3. **Validate numbers**: Cross-check with `evidence_sheet.json`
4. **Cite references**: Use key_references from evidence sheet

### For Experimental Design
1. **Read**: `GNN_ANOMALY_DETECTION_SUMMARY.md` (State of the Art section)
2. **Check**: `evidence_sheet.json` (metric_ranges and known_pitfalls)
3. **Select method**: Use method selection guide
4. **Set thresholds**: Use realistic ranges from benchmarks

### For Method Selection
1. **Input**: Your available labeled data percentage
2. **Look up**: Method selection guide in summary document
3. **Verify**: Performance ranges in evidence_sheet.json
4. **Plan**: Realistic expectations based on similar datasets

### For Quick Reference
1. **Use**: `GNN_ANOMALY_DETECTION_SUMMARY.md` performance tables
2. **Check**: Key takeaways section
3. **Verify**: Quick reference metrics

### For Deep Understanding
1. **Start**: Chronological development in `lit_review_gnn_anomaly_detection.md`
2. **Understand**: Each methodology section
3. **Learn**: Dataset benchmark characteristics
4. **Apply**: Recommendations section for practitioners

---

## Citation Information

When using these research materials, cite as:

**Literature Review**:
```
Comprehensive Literature Review on Anomaly Detection using Graph Neural Networks
Research conducted: December 2025
Sources: 15+ peer-reviewed papers and recent surveys (2024-2025)
Papers analyzed from: ICML, IJCAI, NeurIPS, AAAI, TKDE, WSDM, WWW, and preprints
```

**Evidence Sheet**:
```
Quantitative Evidence Database for GNN Anomaly Detection
Compiled from: Published benchmark results and experimental findings
Quality: Cross-verified from multiple sources
Coverage: 16 major methods, 18 metric ranges, 20+ identified pitfalls
```

---

## Key Statistics

**Research Synthesis**:
- Papers analyzed: 15+ peer-reviewed articles
- Surveys reviewed: 3 comprehensive surveys (2021, 2025)
- Time period: 2018-2025 (focus: 2023-2025)
- Methods analyzed: 16 major approaches
- Datasets covered: 10+ benchmark and real-world datasets
- Metrics tracked: 18 performance metric ranges
- Pitfalls identified: 20+ critical limitations

**Coverage**:
- Unsupervised methods: 5+ detailed
- Semi-supervised methods: 4+ detailed
- Supervised methods: 4+ detailed
- Reconstruction-based: 4+ methods
- Distance-based: 2+ methods
- Density-based: 2+ methods
- Contrastive learning: 3+ methods

---

## Version Information

**Research Package Version**: 1.0
**Date Compiled**: December 24, 2025
**Last Updated**: December 24, 2025
**Status**: Complete - Ready for publication and research use

---

## Notes for Future Research

The field of GNN-based anomaly detection is rapidly evolving. Key emerging areas include:

1. **Contrastive Learning** has become dominant (2024-2025)
2. **Transformer Integration** showing promise for temporal dependencies
3. **Semi-supervised** approaches gaining practical importance
4. **Dynamic Graphs** becoming increasingly relevant
5. **Interpretability** recognized as critical gap

Researchers should monitor these areas for new developments and updated benchmarks.

---

## File Structure

```
files/research_notes/
├── lit_review_gnn_anomaly_detection.md    (10,000+ words, comprehensive literature review)
├── evidence_sheet.json                     (Structured quantitative data)
├── GNN_ANOMALY_DETECTION_SUMMARY.md        (Executive summary with tables)
└── README_GNN_RESEARCH.md                  (This file - navigation guide)
```

---

## Contact & Support

For questions about research interpretation or methodology, refer to:
1. The detailed explanations in `lit_review_gnn_anomaly_detection.md`
2. The methodology sections explaining each approach type
3. The known_pitfalls section in `evidence_sheet.json`
4. The recommendations for practitioners in the summary document

---

**End of Navigation Guide**

All materials are ready for academic publication, research paper writing, experimental design, and practical implementation guidance.
