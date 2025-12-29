# Graph Anomaly Detection Literature Review - Complete Package

## Overview

This package contains a comprehensive literature review of graph anomaly detection (GAD) techniques applied to graphs, with detailed analysis of outlier scoring, reconstruction error, contrastive learning, and graph autoencoders. The review synthesizes over 40 peer-reviewed papers and provides quantitative metrics for detection accuracy across benchmark datasets.

## Files Included

### 1. **lit_review_graph_anomaly_detection.md** (Main Literature Review)
Comprehensive literature review covering:
- Historical development (2018-2025)
- Chronological summary of major developments
- Detailed table of prior work with methods and results
- Identified gaps and open problems
- State-of-the-art summary by task type
- Methodological assumptions and limitations
- Quantitative findings with metric ranges

**Key Sections**:
- Overview of the research area
- Chronological developments from early reconstruction methods through recent spectral approaches
- Summary table: 20+ methods with venues, tasks, and benchmark results
- 10 major open research problems identified
- SOTA methods: GAD-NR (WSDM 2024), GADAM (ICLR 2024), UniGAD (NeurIPS 2024)
- Known limitations and failure modes for each approach

### 2. **evidence_sheet_gad.json** (Quantitative Evidence)
Structured JSON file containing:
- **Metric Ranges**: AUC, AUPRC, precision, recall, F1 across graph types
  - Citation networks: AUC 0.70-0.92
  - Large-scale networks: AUC 0.55-0.65
  - Social networks: AUC 0.65-0.90
  - Fraud detection: AUC 0.70-0.95

- **Typical Sample Sizes**: Dataset specifications
  - Cora: 2,708 nodes, 5,429 edges
  - CiteSeer: 3,327 nodes
  - OGBn-Arxiv: 169,343 nodes
  - Fraud datasets: 130k-350k nodes
  - GADBench: 10 datasets, up to 6M nodes

- **Known Pitfalls**: 20 documented pitfalls
  - Reconstruction error insufficiency (critical finding)
  - Sparse graph degradation
  - Local inconsistency deception (2025)
  - Message passing signal suppression
  - Data contamination
  - Hyperparameter sensitivity
  - And 14 more

- **Key References**: 20 seminal papers with findings
  - Tang 2022 (ICML): Fundamental criticism of reconstruction methods
  - Roy 2024 (WSDM): GAD-NR achieving 30% AUC improvement
  - GADAM 2024 (ICLR): Adaptive message passing framework
  - And 17 more critical papers

### 3. **gad_methods_comparison.md** (Detailed Method Analysis)
In-depth technical comparison of 6 method categories:

1. **Reconstruction Error Methods**
   - DOMINANT, DONE, GAE, GDAE
   - Problem: Normal neighborhoods sometimes harder to reconstruct
   - Performance: AUC 0.65-0.85

2. **Outlier Scoring Methods**
   - Local inconsistency mining (LIM)
   - MLP-based LIM (GADAM)
   - Spectral-based scoring
   - Performance: Varies by approach

3. **Contrastive Learning Methods**
   - ANEMONE (multi-scale)
   - EAGLE (heterogeneous graphs)
   - Problem: Local consistency deception (2025)
   - Performance: AUC 0.75-0.90

4. **Graph Autoencoder Architectures**
   - Standard GAE
   - GRASPED (spectral enhancement)
   - ADA-GAD (denoising autoencoders)
   - Enhanced GAE with subgraph information

5. **Spectral Methods**
   - Foundational theory (eigenvalue shifts)
   - Dynamic wavelets
   - SPS-GAD (for heterophilic graphs)
   - Performance: AUC 0.70-0.88

6. **Adaptive Message Passing**
   - GADAM framework
   - Conflict-free LIM + adaptive MP
   - Performance: AUC 0.82-0.92

Plus detailed benchmarks and selection guide

### 4. **gad_datasets_protocols.md** (Datasets and Experimental Standards)
Complete reference for benchmark datasets and experimental protocols:

**Datasets Covered**:
- Citation networks: Cora, CiteSeer, Pubmed, OGBn-Arxiv
- Social networks: BlogCatalog, Reddit, ACM
- Fraud detection: YelpChi, Amazon
- Other: Books network, GADBench compilation

**Standard Evaluation Protocol**:
- Data splits: 40% training, 20% validation, 40% test
- Metrics: AUROC, AUPRC, Recall@K
- Anomaly injection: Structural, contextual, combined
- Cross-validation: 10-fold with fixed random seeds

**Known Issues**:
- Data contamination effects
- Class imbalance sensitivity
- Feature sparsity problems
- Injection artifacts
- Hyperparameter tuning leakage
- Threshold optimization pitfalls

**Reproducibility Checklist**: 12-point checklist for reproducible research

## Key Quantitative Findings

### Performance by Method Type

| Method Type | AUC Range | Strengths | Weaknesses |
|-------------|-----------|----------|-----------|
| Reconstruction | 0.65-0.85 | Simple, fast | Insufficient signal |
| Contrastive | 0.75-0.90 | Multi-scale capture | Interfering edges |
| Spectral | 0.70-0.88 | Theoretically grounded | Computational cost |
| Adaptive MP | 0.82-0.92 | SOTA performance | Complex architecture |
| Multi-Level Unified | 0.78-0.91 | All anomaly types | Newer, less eval |
| Hybrid | 0.85-0.93 | Most robust | Slowest, many params |

### Performance by Graph Type

| Graph Type | AUC Range | Best Methods | Challenges |
|-----------|-----------|-------------|-----------|
| Citation Networks | 0.70-0.92 | GAD-NR, GADAM | Sparse features |
| Large-Scale Networks | 0.55-0.65 | Contrastive, Spectral | Scalability |
| Social Networks | 0.65-0.90 | ANEMONE, GADAM | Heterophily |
| Fraud Detection | 0.70-0.95 | Contrastive methods | Organic anomalies |
| Dynamic Graphs | Variable | STGNN, Memory-based | Temporal drift |
| Heterophilic | Variable | SPS-GAD, Spectral | Homophily violations |

### Real-World Performance

- **Real-time GNN**: 96.8% accuracy, 1.45s latency per 50k packets
- **CGTS (CAN)**: 99.0% accuracy, 99.4% precision, 99.3% F1
- **Streaming Detection**: O(1) time and memory per edge

## Critical Findings

### 1. Reconstruction Error Limitation (Tang et al., 2022)
- Normal neighborhoods can be harder to reconstruct than anomalous ones
- Reconstruction loss alone is insufficient for anomaly detection
- Solution: Combine with neighborhood contrast or spectral methods

### 2. Message Passing Paradox (GADAM, 2024)
- GNN message passing suppresses local anomaly signals
- Conflicts with local inconsistency mining assumptions
- Solution: Adaptive message passing with MLP-based LIM

### 3. Local Consistency Deception (2025)
- Interfering edges invalidate contrastive learning assumptions
- Low similarity to neighbors doesn't always indicate anomaly
- Solution: Clean-view perspective, edge filtering

### 4. Sparse Graph Degradation
- Methods degrade 15-30% on sparse graphs (Cora, CiteSeer, OGBn-Arxiv)
- Citation networks: ~98-99% feature sparsity
- Contrastive methods more robust than reconstruction

### 5. Hyperparameter Sensitivity
- Detection performance highly dependent on:
  - Self-supervised learning strategy selection
  - Hyperparameter tuning (learning rate, dimensions, layers)
  - Combination weights in multi-method approaches
- Current practice: Arbitrary or dataset-specific selection

## Research Gaps Identified

1. **Reconstruction Methods**: Need principled fix or replacement
2. **Sparse Graphs**: Limited methods work well on sparse feature graphs
3. **Edge/Subgraph Anomalies**: Under-explored (UniGAD 2024 first unified approach)
4. **Dynamic Graphs**: Temporal aspects less developed
5. **Heterophilic Graphs**: Most methods assume homophily
6. **Interpretability**: Black-box models lack explainability (GRAM 2023 emerging)
7. **Data Contamination**: Unlabeled anomalies in training poison methods
8. **Transfer Learning**: Cross-domain generalization not addressed
9. **Adversarial Robustness**: Limited work on adversarial anomalies
10. **Few-Shot Learning**: Limited methods for anomaly detection with few labels

## State-of-the-Art Methods

### Node-Level (Static Graphs)
1. **GAD-NR (WSDM 2024)**: 87.55±2.56 AUC (Cora), 30% improvement
2. **GADAM (ICLR 2024)**: 90±2 AUC (Cora), adaptive MP approach
3. **ANEMONE (AAAI 2023)**: 89±2 AUC (Cora), multi-scale contrastive

### Multi-Level (Nodes, Edges, Subgraphs)
- **UniGAD (NeurIPS 2024)**: First unified framework, spectral sampling

### Heterophilic Graphs
- **SPS-GAD (2025)**: Spectral-spatial for heterophily

### Dynamic Graphs
- **Memory-Enhanced (2024)**: Preserves temporal normality patterns
- **Real-Time GNN (2025)**: 96.8% accuracy, 1.45s latency

### Interpretable
- **GRAM (2023)**: Gradient attention maps for explainability

## Benchmark Datasets Summary

| Dataset | Nodes | Edges | Type | AUC Range |
|---------|-------|-------|------|-----------|
| Cora | 2,708 | 5,429 | Citation | 0.78-0.92 |
| CiteSeer | 3,327 | 4,732 | Citation | 0.70-0.88 |
| Pubmed | 19,717 | 44,338 | Citation | 0.73-0.90 |
| OGBn-Arxiv | 169,343 | 1.17M | Citation | 0.55-0.65 |
| BlogCatalog | 10,312 | 333,983 | Social | 0.65-0.90 |
| YelpChi | ~130k | - | Fraud | 0.70-0.95 |
| Amazon | ~350k | - | Fraud | 0.70-0.95 |
| Reddit | ~5k | - | Social | Organic |
| ACM | Variable | - | Collab | Organic |
| **GADBench** | **up to 6M** | **- | Multi | Varies |

## Standard Evaluation Metrics

- **AUROC (Area Under ROC Curve)**: Threshold-independent, standard metric
- **AUPRC (Area Under PR Curve)**: Better for imbalanced data
- **Precision, Recall, F1**: For specific operating points
- **Recall@K**: Ranking-based metric
- **Computational Metrics**: Training time, inference latency, memory

## How to Use This Package

### For Literature Review Writing
1. Start with `lit_review_graph_anomaly_detection.md`
2. Extract citations and method descriptions
3. Use quantitative results from `evidence_sheet_gad.json`
4. Add specific method details from `gad_methods_comparison.md`

### For Method Selection
1. Review `gad_methods_comparison.md` performance table
2. Check applicability to your graph type
3. Verify available implementations (PyGOD library)
4. Consider computational constraints

### For Experimental Design
1. Consult `gad_datasets_protocols.md` for benchmark standards
2. Follow evaluation protocol section
3. Use reproducibility checklist
4. Document hyperparameter choices

### For Understanding Limitations
1. Check `evidence_sheet_gad.json` known pitfalls
2. Review "Identified Gaps" in main review
3. Understand failure modes in methods comparison
4. Plan mitigation strategies

## Tools and Libraries

### Recommended Python Libraries
- **PyGOD**: Graph Outlier Detection library
  - Website: https://github.com/pygod-team/pygod
  - Implements 10+ methods
  - Built on PyTorch Geometric
  - Ready-to-use baselines

- **PyTorch Geometric (PyG)**: GNN implementation
- **NetworkX**: Graph utilities

## Recommendations for Future Work

1. **Theoretical Foundation**: Develop principled approach to anomaly scoring beyond reconstruction
2. **Sparse Graphs**: Design methods specifically for sparse feature graphs
3. **Edge Detection**: Extend node-level methods to edges systematically
4. **Real-Time Systems**: Optimize for streaming/online detection
5. **Heterophily**: General methods working on both homophilic and heterophilic graphs
6. **Interpretability**: Integrate explainability into model design, not post-hoc
7. **Contamination Robustness**: Build methods resilient to unlabeled anomalies in training
8. **Transfer Learning**: Develop domain adaptation for graph anomaly detection
9. **Benchmark Expansion**: More diverse datasets (bipartite, hypergraphs, knowledge graphs)
10. **Human-in-the-Loop**: Interactive anomaly detection with human feedback

## Citation Information

When referencing this literature review package, cite:

**APA Format**:
Literature Review Agent (2025). Graph Anomaly Detection: Comprehensive Literature Review and Evidence Sheet. Research Platform.

**BibTeX**:
```bibtex
@misc{lit_review_gad2025,
  title={Graph Anomaly Detection: Comprehensive Literature Review and Evidence Sheet},
  author={Literature Review Agent},
  year={2025},
  organization={Research Platform}
}
```

## Contact and Updates

- **Last Updated**: 2025-12-24
- **Papers Reviewed**: 40+ peer-reviewed papers
- **Coverage**: Static graphs, dynamic graphs, heterophilic networks, fraud detection, network intrusion detection
- **Quality Standard**: Comprehensive coverage of recent SOTA (2022-2025) with historical context (2018-2022)

---

## Document Organization

```
files/research_notes/
├── lit_review_graph_anomaly_detection.md (Main review, 50+ pages)
├── evidence_sheet_gad.json (Quantitative metrics, 20 key references)
├── gad_methods_comparison.md (6 method categories, detailed comparison)
├── gad_datasets_protocols.md (4 dataset sections, evaluation standards)
└── README_gad_literature_review.md (This file)
```

## Quick Reference

**For Quick Facts**: Check README_gad_literature_review.md (this file)
**For Complete Details**: Check lit_review_graph_anomaly_detection.md
**For Metrics**: Check evidence_sheet_gad.json
**For Method Implementation**: Check gad_methods_comparison.md
**For Benchmarking**: Check gad_datasets_protocols.md

---

**Ready to use in formal research papers, experimental design, and method selection.**
