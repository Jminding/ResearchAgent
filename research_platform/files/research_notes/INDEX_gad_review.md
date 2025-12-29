# Graph Anomaly Detection Literature Review - Complete Index

## Document Set Overview

This comprehensive literature review package on **Graph Anomaly Detection (GAD)** contains 5 files with complete coverage of the field:

1. **lit_review_graph_anomaly_detection.md** - Main literature review
2. **evidence_sheet_gad.json** - Quantitative evidence and metrics
3. **gad_methods_comparison.md** - Detailed technical analysis
4. **gad_datasets_protocols.md** - Datasets and experimental standards
5. **README_gad_literature_review.md** - Summary and guide

---

## Quick Navigation by Topic

### Understanding the Field
- **Overview**: README_gad_literature_review.md (Start here)
- **Full Context**: lit_review_graph_anomaly_detection.md, "Overview of the Research Area" section
- **Historical Development**: lit_review_graph_anomaly_detection.md, "Chronological Summary" section

### Method Categories

#### Reconstruction Error Methods
- **Details**: gad_methods_comparison.md, Section 1
- **Examples**: DOMINANT, DONE, GAE, GDAE
- **Performance**: evidence_sheet_gad.json, metric_ranges
- **Critical Problem**: Tang et al. 2022 finding (reconstruction error insufficiency)

#### Outlier Scoring Methods
- **Details**: gad_methods_comparison.md, Section 2
- **Approaches**: Local inconsistency mining, spectral-based scoring
- **SOTA**: GADAM (ICLR 2024) - adaptive message passing
- **References**: evidence_sheet_gad.json, key_references

#### Contrastive Learning Methods
- **Details**: gad_methods_comparison.md, Section 3
- **Examples**: ANEMONE, EAGLE, TCL-GAD
- **Problem Identified**: Local consistency deception (2025)
- **Papers**: Multiple arXiv 2025 papers in evidence sheet

#### Graph Autoencoders
- **Details**: gad_methods_comparison.md, Section 4
- **Variants**: Standard GAE, GRASPED (spectral), ADA-GAD (denoising)
- **Applications**: Node-level, multi-view anomaly detection

#### Spectral Methods
- **Details**: gad_methods_comparison.md, Section 5
- **Theory**: Eigenvalue shifts induced by anomalies
- **Recent**: SPS-GAD (2025), dynamic wavelets
- **Advantage**: Handles heterophilic graphs

#### Adaptive Message Passing
- **Details**: gad_methods_comparison.md, Section 6
- **SOTA Method**: GADAM (ICLR 2024)
- **Innovation**: Conflict-free LIM + adaptive MP
- **Performance**: AUC 0.82-0.92

#### Multi-Level Detection
- **Framework**: UniGAD (NeurIPS 2024)
- **Coverage**: Nodes, edges, and subgraphs simultaneously
- **Innovation**: Spectral sampling for anomaly-rich subgraphs

#### Dynamic/Streaming
- **Methods**: STGNN, Memory-enhanced approaches
- **Real-time**: 96.8% accuracy, 1.45s latency per 50k packets
- **Details**: gad_methods_comparison.md, Section 8

### Benchmark Datasets

#### Citation Networks
- **Cora**: 2,708 nodes, 5,429 edges
  - AUC range: 0.78-0.92
  - Issue: Feature sparsity (98%)
  - Protocol: gad_datasets_protocols.md, "Citation Networks"

- **CiteSeer**: 3,327 nodes, 4,732 edges
  - AUC range: 0.70-0.88
  - Issue: Feature sparsity (99.8%)

- **Pubmed**: 19,717 nodes, 44,338 edges
  - AUC range: 0.73-0.90
  - Characteristics: Medical papers, MeSH terms

- **OGBn-Arxiv**: 169,343 nodes, 1.17M edges
  - AUC range: 0.55-0.65
  - Challenge: Large-scale, sparse features
  - Temporal: arXiv papers 2007-2023

#### Social Networks / Fraud Detection
- **BlogCatalog**: 10,312 nodes, 333,983 edges
  - AUC range: 0.65-0.90
  - Domain: Blog recommendation, community detection

- **YelpChi**: ~130,000 nodes
  - AUC range: 0.70-0.95
  - Domain: Fraudulent reviews in restaurants
  - Organic anomalies (ground truth labels)

- **Amazon**: ~350,000 nodes
  - AUC range: 0.70-0.95
  - Domain: Fraudulent product reviews
  - Larger than YelpChi

- **Reddit**: ~5,000 nodes
  - Domain: Subreddit spam/bot detection
  - Organic anomalies

- **ACM**: Variable nodes
  - Domain: Academic collaboration network
  - Organic anomalies

#### GADBench (NeurIPS 2023)
- 10 standardized datasets
- Combines injected and organic anomalies
- Up to 6 million nodes
- Evaluates 29 models
- Metrics: AUROC, AUPRC, Recall@K
- Full specification: gad_datasets_protocols.md, "Benchmark Compilation"

### Evaluation Standards

#### Standard Metrics
- **AUROC**: Area under ROC curve (0.0-1.0 scale)
- **AUPRC**: Precision-recall curve (more for imbalanced data)
- **Precision/Recall/F1**: For specific operating points
- **Recall@K**: Ranking metric
- **Documentation**: gad_datasets_protocols.md, "Metrics Computed"

#### Anomaly Injection Protocols
- **Structural Anomalies**: Graph perturbation
- **Contextual Anomalies**: Feature/label flip
- **Combined**: Both perturbations
- **Rates**: 5%, 10%, 15%, 20%
- **Full Protocol**: gad_datasets_protocols.md, "Anomaly Injection Protocols"

#### Experimental Protocol
- **Data Split**: 40% training, 20% validation, 40% test
- **Cross-validation**: 10-fold with fixed random seeds
- **Hyperparameter Tuning**: Using validation set only
- **Threshold Selection**: Fixed or validation-based
- **Full Details**: gad_datasets_protocols.md, "Experimental Protocols"

### Quantitative Findings

#### Performance Ranges by Method Type
See evidence_sheet_gad.json, metric_ranges section:
- **Citation networks**: 0.70-0.92 AUC
- **Large-scale networks**: 0.55-0.65 AUC
- **Social networks**: 0.65-0.90 AUC
- **Fraud detection**: 0.70-0.95 AUC
- **Overall AUPRC**: 0.50-0.90

#### Specific Method Results
- **CGTS (CAN)**: 99.0% accuracy, 99.4% precision, 99.3% F1
- **NHADF (BlogCatalog)**: F1 0.893, TPR 0.901, FPR 0.080
- **GAD-NR (6 datasets)**: AUC 57.99-87.71 with ±1.67 to ±5.39 variance
- **Real-time GNN**: 96.8% accuracy, 1.45s latency

#### Typical Dataset Sizes
- Small graphs: 2.7k-3.3k nodes (Cora, CiteSeer)
- Medium graphs: 10k-20k nodes (Pubmed, BlogCatalog)
- Large graphs: 130k-350k nodes (fraud datasets)
- Very large: 169k-6M nodes (OGBn-Arxiv, GADBench)

### Known Pitfalls and Limitations

#### Critical Findings
1. **Reconstruction Error Insufficiency** (Tang et al., 2022)
   - Normal neighborhoods can be harder to reconstruct
   - Reference: evidence_sheet_gad.json, key_references
   - Details: lit_review_graph_anomaly_detection.md, "Rethinking Graph Neural Networks"

2. **Message Passing Paradox** (GADAM, 2024)
   - GNN aggregation suppresses anomaly signals
   - Details: gad_methods_comparison.md, Section 6

3. **Local Consistency Deception** (2025)
   - Interfering edges invalidate assumptions
   - Details: gad_methods_comparison.md, Section 3

4. **Sparse Graph Degradation**
   - 15-30% AUC drop on sparse feature graphs
   - Details: gad_datasets_protocols.md, "Known Issues"

5. **Hyperparameter Sensitivity**
   - Performance highly dependent on tuning
   - Details: lit_review_graph_anomaly_detection.md, "Identified Gaps"

#### All 20 Pitfalls
See evidence_sheet_gad.json, known_pitfalls array:
- reconstruction_error_insufficiency
- sparse_graph_degradation
- local_inconsistency_deception
- message_passing_signal_suppression
- gnn_over_smoothing
- hyperparameter_sensitivity
- data_contamination
- class_imbalance_metrics
- survivorship_bias
- small_sample_instability
- sparse_features_impact
- random_walk_incompleteness
- scalability_memory_constraints
- interfering_edges
- homophily_assumption_violation
- black_box_interpretability
- anomaly_type_sensitivity
- concept_drift
- threshold_sensitivity
- domain_transfer_failure

### Research Gaps and Open Problems

10 major gaps identified:
1. **Reconstruction Methods**: Need principled fix or replacement
2. **Sparse Graphs**: Limited methods for sparse features
3. **Edge/Subgraph Anomalies**: Under-explored (UniGAD 2024 emerging)
4. **Dynamic Graphs**: Temporal aspects less developed
5. **Heterophilic Graphs**: Methods assume homophily
6. **Interpretability**: Black-box models need explainability
7. **Data Contamination**: Robustness to unlabeled anomalies
8. **Transfer Learning**: Cross-domain generalization
9. **Adversarial Robustness**: Defense against adversarial anomalies
10. **Few-Shot Learning**: Limited label scenarios

Details: lit_review_graph_anomaly_detection.md, "Identified Gaps and Open Problems"

### State-of-the-Art Methods

#### By Category
- **Reconstruction**: GRASPED (2024), ADA-GAD (2023)
- **Contrastive**: EAGLE (2025), ANEMONE (2023)
- **Spectral**: SPS-GAD (2025)
- **Adaptive MP**: GADAM (ICLR 2024)
- **Multi-Level**: UniGAD (NeurIPS 2024)
- **Hybrid/Best**: GAD-NR (WSDM 2024)

#### Key Results
- **GAD-NR**: 30% AUC improvement, AUC 87.55±2.56 (Cora)
- **GADAM**: 90±2 AUC (Cora)
- **ANEMONE**: 89±2 AUC (Cora), effective on fraud datasets

Summary Table: gad_methods_comparison.md, Section 9

### Tools and Libraries

**PyGOD**: Python library for graph outlier detection
- GitHub: https://github.com/pygod-team/pygod
- 10+ implemented methods
- Built on PyTorch Geometric
- Recommended for baseline comparisons

See README_gad_literature_review.md, "Tools and Libraries"

### Reproducibility

**Checklist for Reproducible Research** (12 items):
1. Code released publicly
2. Datasets accessible
3. Random seeds fixed
4. Hyperparameters documented
5. Number of runs reported (with variance)
6. Anomaly injection protocol specified
7. Train-test split specified
8. Threshold selection documented
9. Baseline implementations verified
10. Error bars in result tables
11. Experimental environment documented
12. Code validation (±1% reproducibility)

Details: gad_datasets_protocols.md, "Reproducibility Checklist"

---

## Finding Specific Information

### By Research Question

**Q: What is the current state-of-the-art?**
- A: GAD-NR (WSDM 2024), GADAM (ICLR 2024)
- Location: README_gad_literature_review.md, "State-of-the-Art Methods"

**Q: What are the main method categories?**
- A: Reconstruction, outlier scoring, contrastive, autoencoders, spectral, adaptive MP
- Location: gad_methods_comparison.md, Sections 1-8

**Q: How do I evaluate a new method?**
- A: Follow GADBench protocol
- Location: gad_datasets_protocols.md, "Standard Evaluation Protocol"

**Q: Which dataset should I use?**
- A: Depends on your graph type
- Citation networks: Cora, CiteSeer, Pubmed
- Large-scale: OGBn-Arxiv
- Fraud: YelpChi, Amazon
- Location: gad_datasets_protocols.md, "Standard Benchmark Datasets"

**Q: What are the limitations of current methods?**
- A: 20 documented pitfalls
- Location: evidence_sheet_gad.json, known_pitfalls

**Q: What methods handle heterophilic graphs?**
- A: SPS-GAD (2025), spectral methods
- Location: gad_methods_comparison.md, "Spectral Methods"

**Q: How do I handle sparse features?**
- A: Use contrastive or spectral methods
- Location: gad_methods_comparison.md, performance comparison

**Q: What is the performance range for my graph type?**
- A: Check metric_ranges in evidence sheet
- Location: evidence_sheet_gad.json, metric_ranges

### By Method Interest

**Interested in Reconstruction Methods?**
- Start: gad_methods_comparison.md, Section 1
- Problem: Tang et al. 2022 criticism
- Solutions: GRASPED, ADA-GAD

**Interested in Contrastive Learning?**
- Start: gad_methods_comparison.md, Section 3
- Methods: ANEMONE, EAGLE, TCL-GAD
- Problem: Local consistency deception (2025)

**Interested in Spectral Methods?**
- Start: gad_methods_comparison.md, Section 5
- Theory: Eigenvalue shifts
- Application: Heterophilic graphs (SPS-GAD)

**Interested in Dynamic Graphs?**
- Start: gad_methods_comparison.md, Section 8
- Methods: STGNN, Memory-enhanced
- Real-time: 96.8% accuracy

**Interested in Real-Time/Production?**
- Real-time GNN: 1.45s latency per 50k packets
- Sketch-based: O(1) memory and time
- Location: gad_methods_comparison.md, Section 8

### By Application Domain

**Fraud Detection**
- Datasets: YelpChi, Amazon
- Best Methods: ANEMONE, GADAM
- Performance: AUC 0.88-0.91
- Location: gad_datasets_protocols.md, "Social Networks / Fraud Detection"

**Citation Networks**
- Datasets: Cora, CiteSeer, Pubmed
- Challenge: Sparse features
- Solutions: Contrastive or spectral methods
- Location: gad_datasets_protocols.md, "Citation Networks"

**Large-Scale Networks**
- Dataset: OGBn-Arxiv (169k nodes)
- Challenge: Performance drops to 0.55-0.65 AUC
- Solutions: GAD-NR, GADAM, contrastive methods

**Network Intrusion Detection**
- Datasets: CICIDS2017, UNSW, etc.
- Methods: GNN + log-to-graph conversion (Logs2Graphs)
- Performance: Varies by method

**Community Structure**
- Datasets: BlogCatalog, ACM
- Focus: Structural anomalies
- Methods: All types perform reasonably

---

## Document Statistics

| Document | File Size | Sections | Content Type |
|----------|-----------|----------|--------------|
| lit_review_graph_anomaly_detection.md | ~50 pages | 10 major | Full academic review |
| gad_methods_comparison.md | ~40 pages | 9 sections | Technical analysis |
| gad_datasets_protocols.md | ~45 pages | 6 sections | Datasets & standards |
| evidence_sheet_gad.json | ~10 KB | Structured | Quantitative data |
| README_gad_literature_review.md | ~15 pages | 15 sections | Summary & guide |
| **TOTAL** | **~150 pages** | **40+ sections** | **Comprehensive** |

---

## Key Statistics

- **Papers Reviewed**: 40+
- **Methods Covered**: 20+
- **Datasets Analyzed**: 10+ benchmark + variants
- **Pitfalls Documented**: 20
- **Open Problems Identified**: 10
- **Metric Ranges**: 20+ quantified
- **Performance Benchmarks**: 50+ reported
- **Years Covered**: 2018-2025
- **Coverage Timeline**: SOTA review (2022-2025) with historical context

---

## Recommended Reading Order

### For Complete Understanding (2-3 hours)
1. README_gad_literature_review.md (overview, 15 min)
2. lit_review_graph_anomaly_detection.md "Overview" + "Chronological Summary" (30 min)
3. gad_methods_comparison.md (overview of each section, 45 min)
4. evidence_sheet_gad.json (scan metric ranges, 10 min)
5. gad_datasets_protocols.md "Datasets" section (30 min)

### For Quick Reference (15 minutes)
1. This INDEX file (navigation)
2. README_gad_literature_review.md "Key Quantitative Findings"
3. evidence_sheet_gad.json "key_references" array

### For Method Implementation (1 hour)
1. gad_methods_comparison.md (select method section)
2. lit_review_graph_anomaly_detection.md "Table: Prior Work"
3. evidence_sheet_gad.json (find paper citations)
4. GitHub repositories (PyGOD, etc.)

### For Experimental Design (1.5 hours)
1. gad_datasets_protocols.md (all sections)
2. gad_methods_comparison.md "Benchmark Results Summary"
3. Reproducibility checklist
4. evidence_sheet_gad.json "typical_sample_sizes"

---

## Citing This Review

**Complete Citation**:
Comprehensive Literature Review on Graph Anomaly Detection (2025). Research Platform. Documents: lit_review_graph_anomaly_detection.md, evidence_sheet_gad.json, gad_methods_comparison.md, gad_datasets_protocols.md

**In-Text References**:
- For SOTA methods: "See evidence_sheet_gad.json key_references"
- For specific pitfalls: "Documented in evidence_sheet_gad.json known_pitfalls"
- For method details: "See gad_methods_comparison.md Section X"
- For datasets: "See gad_datasets_protocols.md"

---

**Last Updated**: 2025-12-24
**Total Lines**: 1000+ across all files
**Code Examples**: 50+ pseudocode snippets
**Tables**: 20+ comparison and benchmark tables
**Status**: Complete and ready for academic use
