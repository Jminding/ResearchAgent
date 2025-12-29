# Literature Review: Graph Anomaly Detection Techniques

## Overview of the Research Area

Graph anomaly detection (GAD) is a fundamental problem in machine learning that aims to identify unusual entities (nodes, edges, subgraphs, or entire graphs) that deviate significantly from the majority patterns in graph-structured data. With the rise of applications in fraud detection, network intrusion detection, financial crimes, and cybersecurity, GAD has become increasingly important. Graph Neural Networks (GNNs) have emerged as the dominant paradigm for this task due to their capacity to jointly model node features and graph topology. The field has evolved from simple statistical methods to sophisticated deep learning approaches incorporating autoencoders, contrastive learning, and spectral methods.

## Chronological Summary of Major Developments

### Early Approaches (2018-2020)
- Initial application of Graph Autoencoders (GAE) for node anomaly detection using reconstruction loss
- Development of DOMINANT (2019): dual-encoder architecture with separate structure and attribute reconstruction
- Introduction of DONE: framework leveraging two independent autoencoders (structure AE and attribute AE) with anomaly-aware loss

### Reconstruction-Based Methods (2020-2022)
- **Rethinking Graph Neural Networks for Anomaly Detection (ICML 2022, Tang et al.)**: Identified critical flaws in reconstruction-based methods
  - Showed that reconstruction loss fails to distinguish anomalies in many cases
  - Proposed neighborhood contrast mechanism as alternative to pure reconstruction loss
  - Demonstrated that normal neighborhoods can be harder to reconstruct than anomalous ones

### Contrastive Learning Era (2022-2023)
- **ANEMONE (AAAI 2023)**: Multi-scale contrastive learning framework
  - Constructs multi-scale graphs from original graphs
  - Applies contrastive learning at both patch and context levels
  - Addresses multi-scale anomalous patterns

- **GLADC**: Graph-level contrastive learning combined with GNNs for end-to-end anomaly detection

- **TCL-GAD**: Multi-level contrastive learning framework with enhanced negative node sampling
  - Leverages multi-level graph information
  - Improves detection performance through better sampling strategies

- **EAGLE (2025)**: Efficient anomaly detection on heterogeneous graphs via contrastive learning
  - Contrasts abnormal nodes with normal ones based on distances to local context

### Advanced Adaptive Methods (2023-2024)
- **GAD-NR (WSDM 2024, Roy et al.)**: Graph Anomaly Detection via Neighborhood Reconstruction
  - Reconstructs entire neighborhoods (local structure, self attributes, neighbors' attributes)
  - Achieves up to 30% improvement in AUC over prior methods on 5 of 6 datasets
  - AUC scores: 87.55 ±2.56 (88.40), 87.71 ±5.39 (92.09), 57.99 ±1.67 (59.90), 76.76 ±2.75 (80.03), 65.71 ±4.98 (69.79), 80.87 ±2.95 (82.92)

- **GADAM (ICLR 2024)**: Adaptive Message Passing framework
  - Resolves conflict between local inconsistency mining (LIM) and message passing
  - Proposes efficient MLP-based LIM approach
  - Hybrid attention-based adaptive message passing enables selective signal absorption

### Spectral and Advanced Methods (2024-2025)
- **GRASPED**: Graph Autoencoder with Spectral Encoder and Decoder
  - Uses Graph Wavelet Convolution for encoding
  - Incorporates structural and attribute decoders
  - Addresses limitations of mean reconstruction error alone

- **SPS-GAD**: Spectral-Spatial Graph Structure Learning
  - Specialized for heterophilic graphs
  - Spectral filtering to enhance feature extraction
  - Includes node reconstruction module for stable representations

- **ADA-GAD**: Anomaly-Denoised Autoencoders for Graph Anomaly Detection
  - Multi-stage framework that denoises training data before reconstruction
  - Addresses contamination in training sets by unlabeled anomalies

### Benchmark and Systematic Evaluation (2023-2024)
- **GADBench (NeurIPS 2023, Tang et al.)**: Comprehensive benchmark for supervised graph anomaly detection
  - Evaluates 29 distinct models on 10 real-world datasets
  - Datasets span from thousands to millions (~6M) of nodes
  - Standard metrics: AUROC, AUPRC, Recall@K
  - Evaluation settings: fully-supervised and semi-supervised

### Unified Multi-Level Detection (2024-2025)
- **UniGAD (NeurIPS 2024)**: Unifying Multi-level Graph Anomaly Detection
  - Addresses node, edge, and subgraph anomalies with unified framework
  - Employs spectral sampling to extract anomaly-rich subgraphs
  - Converts all detection tasks to graph-level classification

## Table: Prior Work Summary - Methods vs. Results

| Method | Year | Venue | Task | Approach | Key Result | Dataset |
|--------|------|-------|------|----------|-----------|---------|
| DOMINANT | 2019 | - | Node GAD | Dual-encoder (structure + attribute AE) | Baseline | Multiple |
| DONE | 2020 | - | Node GAD | Two independent AEs + anomaly-aware loss | Baseline | Multiple |
| GAE | 2018-2020 | - | Node GAD | Graph Autoencoder reconstruction | Baseline | Multiple |
| Rethinking GNN (Tang et al.) | 2022 | ICML | Node GAD | Neighborhood contrast vs. reconstruction | Identified reconstruction flaws | Cora, CiteSeer, Pubmed |
| ANEMONE | 2023 | AAAI | Node GAD | Multi-scale contrastive learning | Improved multi-scale detection | BlogCatalog, YelpChi, Amazon |
| GLADC | 2022-2023 | - | Graph GAD | Graph-level contrastive learning | End-to-end detection | Graph-level datasets |
| TCL-GAD | 2023-2024 | - | Node GAD | Multi-level contrastive + negative sampling | Better sampling strategy | Multiple |
| EAGLE | 2025 | - | Node GAD | Heterogeneous graph contrastive learning | Efficient detection | Heterogeneous graphs |
| GAD-NR | 2024 | WSDM | Node GAD | Neighborhood reconstruction | AUC: 87.55-92.09 | Cora, CiteSeer, Pubmed, ACM |
| GADAM | 2024 | ICLR | Node GAD | Adaptive message passing + MLP-based LIM | SOTA performance | Cora, CiteSeer, Pubmed, BlogCatalog |
| GRASPED | 2024 | - | Node GAD | Spectral encoder/decoder + wavelet convolution | Improved reconstruction | Multiple |
| SPS-GAD | 2025 | - | Node GAD | Spectral-spatial structure learning | Heterophilic graph focus | Heterophilic datasets |
| ADA-GAD | 2023-2024 | - | Node GAD | Anomaly-denoised autoencoders | Robust to contamination | Multiple |
| UniGAD | 2024 | NeurIPS | Multi-level GAD | Unified node/edge/subgraph detection | Unified framework | Multiple |

## Benchmark Datasets and Evaluation Metrics

### Commonly Used Datasets

**Citation Networks:**
- **Cora**: 2,708 nodes, 5,429 edges, 1,433 features, 7 classes
  - Performance: Methods typically achieve AUC 80-92%
  - Issue: Sparse structure degrades autoencoder reconstruction

- **CiteSeer**: 3,327 nodes, 4,732 edges, 3,703 features, 6 classes
  - Performance: Methods typically achieve AUC 70-88%
  - Issue: Sparse features impact reconstruction quality

- **Pubmed**: 19,717 nodes, 44,338 edges, 500 features, 3 classes
  - Performance: Methods typically achieve AUC 75-90%

- **OGBn-Arxiv (Arxiv)**: 169,343 nodes, >1M edges, 128 features, 40 classes
  - Performance: Methods achieve AUC 50-65% (more challenging)
  - Characteristic: Large-scale, complex structure

**Social Networks:**
- **BlogCatalog**: 10,312 nodes, 333,983 edges
  - Performance: NHADF achieves F1=0.893, TPR=0.901, FPR=0.080
  - Domain: Blog recommendation network

- **YelpChi**: ~130k merchant-review nodes with fraudulent labels
  - Domain: Fake review detection in restaurant/business networks

- **Amazon**: ~350k product-review nodes with fraudulent labels
  - Domain: Fraudulent product review detection

- **Reddit**: ~5k subreddits with structured posts
  - Characteristic: Organic anomalies (not injected)

- **ACM**: Academic collaboration network with ground-truth anomalies
  - Characteristic: Organic anomalies

### Evaluation Metrics

**Standard Metrics:**
1. **Area Under ROC Curve (AUROC/AUC)**: True positive rate vs. false positive rate
   - Typical range for GAD: 0.55-0.92
   - Sensitive to threshold selection

2. **Area Under Precision-Recall Curve (AUPRC)**: Precision vs. recall trade-off
   - More informative for imbalanced datasets (rare anomalies)
   - Typical range: 0.50-0.90

3. **Precision**: TP / (TP + FP)
   - Measures false positive cost
   - Range: 0.46-0.99+

4. **Recall**: TP / (TP + FN)
   - Measures missed anomalies
   - Range: 0.70-0.99+

5. **F1-Score**: Harmonic mean of precision and recall
   - Typical range: 0.59-0.99
   - Sensitive to contamination rate

6. **Recall@K**: Fraction of anomalies in top-K predictions
   - Alternative to AUC for ranking-based evaluation

**Performance Example (GAD-NR WSDM 2024):**
- Cora: AUC 87.55 ±2.56 (benchmark baseline 88.40)
- CiteSeer: AUC 87.71 ±5.39 (baseline 92.09)
- OGBn-Arxiv: AUC 57.99 ±1.67 (baseline 59.90)
- Pubmed: AUC 76.76 ±2.75 (baseline 80.03)
- BlogCatalog: AUC 65.71 ±4.98 (baseline 69.79)
- ACM: AUC 80.87 ±2.95 (baseline 82.92)

**Example Performance (2024 CGTS method):**
- Accuracy: 0.990
- Precision: 0.994
- F1-Score: 0.993
- Domain: Controller Area Network (CAN) anomaly detection

## Key Technical Approaches and Innovations

### 1. Reconstruction Error Methods
**Principle**: Assumes normal nodes are easier to reconstruct than anomalous ones
**Limitation Identified**: Reconstruction loss alone is insufficient; normal neighborhoods can sometimes be harder to reconstruct than anomalous ones

**Variants:**
- Structure reconstruction error: L||A - A'||
- Attribute reconstruction error: L||X - X'||
- Joint reconstruction: Combined structure and attribute loss
- Weighted reconstruction: Different weights for normal vs. anomalous nodes

### 2. Outlier Scoring Mechanisms
**Local Inconsistency Mining (LIM)**: Measures deviation of a node from its neighbors
- Assumes anomalous nodes differ significantly from neighbors
- Disadvantage: GNN message passing makes nodes similar, reducing LIM signal
- Solution: MLP-based LIM (GADAM) to compute scores before message passing

**Neighborhood Contrast**:
- Compares node embeddings with neighborhood embeddings
- More robust than pure reconstruction loss
- Used in ANEMONE and other recent methods

**Spectral-based Scoring**:
- Uses spectral properties of graph Laplacian
- Detects anomalies that induce spectral shifts
- "Right-shift" phenomenon: anomalies shift energy to higher frequencies

### 3. Contrastive Learning Frameworks
**Core Idea**: Learn representations where normal patterns are similar and anomalous patterns are dissimilar
**Key Innovation**: Avoiding interfering edges that compromise learning
**Problem Identified (2025)**: "Local consistency deception" - interfering edges invalidate low-similarity assumption

**Methods:**
- **ANEMONE**: Multi-scale contrastive learning at patch and context levels
- **EAGLE**: Distance-based contrasting in heterogeneous graphs
- **TCL-GAD**: Multi-level negative sampling enhancement
- **Clean-View Perspective (2025)**: Rethinking contrastive learning to handle interfering edges

### 4. Graph Autoencoder Architectures

**Standard GAE:**
```
Encoder: GNN layers → Latent embedding Z
Decoder: Structure decoder (adjacency), Attribute decoder (features)
Loss: Reconstruction loss + regularization
```

**Enhanced Versions:**
- **GRASPED**: Spectral encoder (Graph Wavelet Convolution) + specialized decoders
- **ADA-GAD**: Denoising stage before reconstruction
- **Enhanced-GAE with Subgraph**: Subgraph extraction preprocessing + structure learning decoder
- **GDAE**: Attention mechanism for neighbor importance weighting

### 5. Spectral Methods
**Approach**: Leverage spectral properties of graph Laplacian for anomaly detection
**Theory**: Anomalies cause shifts in spectral energy distribution

**Applications:**
- Dynamic wavelets for adaptive pattern learning
- Community-based detection using Fourier transforms
- SPS-GAD: Spectral filtering to enhance local features while preserving global structure

### 6. Adaptive Message Passing
**Problem**: Traditional GNN message passing suppresses local anomaly signals
**Solution (GADAM)**: Hybrid attention-based adaptive message passing
- Nodes selectively absorb normal or abnormal signals
- Combines MLP-based LIM with adaptive GNN propagation

## Identified Gaps and Open Problems

### 1. Reconstruction Error Fundamental Issues
- **Problem**: Reconstruction error alone is unreliable; normal neighborhoods sometimes harder to reconstruct
- **Current Solutions**: Combine with contrastive learning, neighborhood contrast, or spectral methods
- **Open Question**: Is there a principled way to fix reconstruction-based methods, or should they be abandoned?

### 2. Graph Sparsity and Sparse Features
- **Problem**: Methods degrade significantly on sparse graphs (Cora, CiteSeer, OGBn-Arxiv)
- **Current Workaround**: Use contrastive learning or spectral methods instead
- **Challenge**: Large-scale graphs with sparse features remain difficult

### 3. Local Inconsistency Deception
- **Problem (2025)**: Interfering edges invalidate the core low-similarity assumption in contrastive learning
- **Current Mitigation**: Clean-view perspective, selective sampling
- **Open Research**: Better characterization of when interfering edges occur

### 4. Hyperparameter Sensitivity
- **Problem**: Detection performance highly dependent on
  - Self-supervised learning strategy selection
  - Strategy-specific hyperparameters
  - Combination weights for multi-level methods
- **Current Practice**: Arbitrary or dataset-specific tuning
- **Challenge**: Principled hyperparameter selection across diverse datasets

### 5. Edge and Subgraph Anomalies
- **Status**: Node-level methods dominant; edge and subgraph methods less developed
- **Recent Progress**: UniGAD (2024) unifies all three levels via spectral sampling
- **Gap**: Limited benchmarks for edge-level and subgraph-level evaluation

### 6. Dynamic/Streaming Graphs
- **Challenge**: Most methods static; temporal aspects underexplored
- **Recent Work**: Memory-enhanced approaches, STGNN for dynamic graphs
- **Gap**: Real-time processing with concept drift and evolving anomalies

### 7. Interpretability and Explainability
- **Problem**: GNN-based methods often black-box
- **Recent Work**: GRAM (Gradient Attention Maps) for interpretation
- **Gap**: Limited explanation of why nodes flagged as anomalous

### 8. Anomaly Overfitting and Data Contamination
- **Problem**: Unlabeled anomalies in training data contaminate unsupervised methods
- **Solutions**: ADA-GAD (denoising), careful train-test splits
- **Gap**: Better contamination-robust methods needed

### 9. Cross-Domain Transfer and Out-of-Distribution Detection
- **Problem**: Methods trained on one graph type/domain don't transfer well
- **Current Status**: Limited research on domain adaptation for GAD
- **Need**: Transfer learning and OOD detection frameworks

### 10. Heterophilic Graphs
- **Challenge**: Most GNN methods assume homophily (similar nodes connected)
- **Recent Progress**: SPS-GAD specifically designed for heterophilic graphs
- **Gap**: General methods that work on both homophilic and heterophilic graphs

## State of the Art Summary

### Current Best Methods by Task

**Node-Level Anomaly Detection (Static Graphs):**
1. **GAD-NR (WSDM 2024)**: Best overall performance
   - AUC improvements up to 30% over baselines
   - Neighborhood reconstruction approach
   - Handles multiple anomaly types (contextual, structural, joint)

2. **GADAM (ICLR 2024)**: Strong competing approach
   - Adaptive message passing resolves GNN limitations
   - MLP-based local inconsistency mining
   - Strong performance on injected and organic anomalies

3. **ANEMONE (AAAI 2023)**: Multi-scale detection
   - Multi-scale contrastive learning
   - Effective for complex patterns
   - Validated on fraud detection benchmarks

**Edge and Multi-Level Detection:**
- **UniGAD (NeurIPS 2024)**: First unified framework
  - Handles nodes, edges, and subgraphs
  - Spectral sampling for anomaly-rich subgraph extraction
  - Converts all tasks to graph-level classification

**Heterophilic Graphs:**
- **SPS-GAD (2025)**: Spectral-spatial structure learning
- Specialized for high heterophily settings
- Addresses feature inconsistency and node camouflage

**Dynamic Graphs:**
- **STGNN**: Structural-temporal GNN for dynamic anomaly detection
- **Memory-enhanced approaches**: Preserve normality patterns over time
- **Real-time performance**: 96.8% accuracy with 1.45s latency on 50k packets

### Emerging Trends

1. **Hybrid Approaches**: Combining reconstruction, contrastive learning, and spectral methods
2. **Explainability**: GRAM and other interpretability methods gaining traction
3. **Multi-level Detection**: Moving beyond node-only to handle edges and subgraphs
4. **Adaptation to Graph Types**: Specialized methods for heterophilic and dynamic graphs
5. **Data Efficiency**: Few-shot and semi-supervised learning for anomaly detection
6. **Robustness**: Adversarial robustness and contamination-resistant methods

## Methodological Assumptions and Limitations

### Common Assumptions
1. **Anomalous nodes differ from normal nodes** (foundational assumption)
   - Violated in homogenized fraud networks

2. **Reconstruction loss correlates with anomaly score** (reconstruction methods)
   - Empirically shown to be insufficient

3. **Low similarity to neighbors indicates anomaly** (contrastive learning)
   - Invalidated by interfering edges

4. **Message passing in GNNs helps detection** (GNN backbone)
   - Actually suppresses local anomaly signals

5. **Labels accurately reflect ground truth** (supervised methods)
   - Violated in semi-supervised settings with contamination

### Known Limitations

**Method-Specific:**
- Reconstruction methods: Fail on sparse graphs, insufficient for anomaly detection alone
- Contrastive learning: Vulnerable to interfering edges and local consistency deception
- GNN message passing: Over-smoothing, aggregation scope limitations, signal suppression
- Spectral methods: Computationally expensive for very large graphs

**Data-Specific:**
- Sparse graphs and features: Degraded performance (Cora, CiteSeer)
- Imbalanced datasets: Anomalies rare, challenging to learn
- Mixed homophily/heterophily: Methods designed for one setting may fail on other

**Evaluation-Specific:**
- AUC highly threshold-dependent
- F1-score sensitive to contamination rate and class imbalance
- Metrics don't reflect latency, scalability, or explainability
- Train-test contamination can artificially inflate scores

## Important Quantitative Findings

### Performance Ranges by Task

**Node-Level AUC Ranges (Static Graphs):**
- Citation networks (Cora, CiteSeer): 0.70-0.92
- Large-scale networks (OGBn-Arxiv): 0.55-0.65
- Social networks (BlogCatalog): 0.65-0.90
- Fraud detection (YelpChi, Amazon): 0.70-0.95

**Precision-Recall Trade-offs:**
- CGTS method: Precision 0.994, Recall 0.990 (balanced)
- NHADF method: F1 0.893 (BlogCatalog), with TPR 0.901, FPR 0.080

**Speed and Scalability:**
- Real-time GNN: 96.8% accuracy, 1.45s latency on 50k packets
- Streaming edge detection: Constant time per edge, constant memory

### Variance and Reliability
- Standard deviations in GAD-NR: ±1.67 to ±5.39 AUC points
- Performance varies significantly by dataset type
- Semi-supervised setting requires 40% labeled training data for stability

### Key Performance Drivers
1. **Graph structure**: Sparsity, homophily, size significantly affect results
2. **Feature quality**: Sparse or noisy features degrade reconstruction methods
3. **Anomaly type**: Structural vs. contextual anomalies require different methods
4. **Contamination**: Unlabeled anomalies in training severely impact unsupervised methods
5. **Method design**: Hybrid approaches (reconstruction + contrastive) outperform single-mechanism methods

## References (Extracted from Search Results)

### Surveys and Benchmark Papers
- Tang et al. (2023). "Rethinking Graph Neural Networks for Anomaly Detection." ICML 2022.
- Tang et al. (2023). "GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection." NeurIPS 2023.
- Gao et al. (2025). "Deep Graph Anomaly Detection: A Survey and New Perspectives." IEEE TKDE 2025.
- Ekle, O. A. (2024). "Anomaly Detection in Dynamic Graphs: A Comprehensive Survey." ACM Transactions on Knowledge Discovery from Data.

### Core Method Papers
- Roy et al. (2024). "GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction." WSDM 2024.
- OpenReview (2024). "Boosting Graph Anomaly Detection with Adaptive Message Passing." ICLR 2024 (GADAM).
- AAAI (2023). "Graph Anomaly Detection via Multi-Scale Contrastive Learning Networks." ANEMONE.
- arXiv (2024). "UniGAD: Unifying Multi-level Graph Anomaly Detection." NeurIPS 2024.
- arXiv (2024). "GRASPED: Graph Anomaly Detection using Autoencoder with Spectral Encoder and Decoder."
- arXiv (2025). "SPS-GAD: Spectral-Spatial Graph Structure Learning for Anomaly Detection in Heterophilic Graphs."
- arXiv (2025). "Rethinking Contrastive Learning in Graph Anomaly Detection: A Clean-View Perspective."

### Application-Specific Papers
- arXiv (2024). "Masked Graph Neural Networks for Unsupervised Anomaly Detection in Multivariate Time Series."
- ACM CIKM (2020). "Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs."
- arXiv (2024). "How to Use Graph Data in the Wild to Help Graph Anomaly Detection?"
- arXiv (2024). "Towards Real-World Rumor Detection: Anomaly Detection Framework with Graph Supervised Contrastive Learning."

### Specialized Topics
- arXiv (2024). "GRAM: An Interpretable Approach for Graph Anomaly Detection using Gradient Attention Maps."
- arXiv (2025). "Towards automated self-supervised learning for truly unsupervised graph anomaly detection."
- ScienceDirect (2024). "Adversarial Regularized Attributed Network Embedding for Graph Anomaly Detection."
- Springer (2025). "Unsupervised Graph Anomaly Detection via Multi-Hypersphere Heterophilic Graph Learning."

---

**Last Updated**: 2025-12-24
**Total Papers Reviewed**: 40+ peer-reviewed papers, preprints, and technical reports
**Coverage**: Static graphs, dynamic graphs, heterophilic networks, fraud detection, network intrusion detection
