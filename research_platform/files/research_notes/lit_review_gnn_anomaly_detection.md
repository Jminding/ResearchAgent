# Literature Review: Anomaly Detection using Graph Neural Networks

## Overview of the Research Area

Graph Neural Networks (GNNs) have emerged as the primary methodology for graph anomaly detection (GAD), addressing the challenge of identifying unusual graph instances including nodes, edges, subgraphs, and entire graphs. This research area integrates deep learning with graph-based representations to detect anomalous patterns in structured network data. The field encompasses three primary learning paradigms: unsupervised, semi-supervised, and supervised approaches, each employing distinct methodologies including reconstruction-based, distance-based, and density-based detection strategies. Recent surveys (2024-2025) indicate that GAD methods are categorized through three novel perspectives: GNN backbone design, proxy task design for GAD, and graph anomaly measures.

---

## Chronological Summary of Major Developments

### Early Foundation (2018-2020)
- **DOMINANT (2019)**: Introduced the first effort to leverage graph autoencoders (GAEs) for unsupervised anomaly detection by comparing reconstructed features and adjacency matrices with original inputs, establishing the reconstruction-based paradigm as a core approach in the field.

### Consolidation Phase (2021-2022)
- **Rethinking GNNs for Anomaly Detection (Tang et al., 2022, ICML)**: Critical examination of GNN behavior for anomaly detection, highlighting the fundamental challenge that normal and anomalous nodes may not exhibit significantly different local neighborhoods, questioning the basis of existing approaches.
- **LUNAR (Goodge et al., 2021)**: Hybrid approach combining deep GNNs with Local Outlier Factor (LOF) to learn information from nearest neighbors in a trainable manner, bridging classical density-based methods with modern deep learning.
- **ANEMONE (2022)**: Multi-scale contrastive learning framework enabling better node representation learning through multiple graph scales (views).

### Acceleration Phase (2023-2024)
- **Multi-channel Reconstruction Methods (2023)**: Dual-channel approaches leveraging positive and negative attention values for aggregation, with normal features in low-frequency areas and anomalous features in high-frequency areas.
- **SmoothGNN (2024)**: Novel unsupervised framework proposing loss functions incorporating both feature reconstruction and smoothing-aware measures for node anomaly detection.
- **Contrastive Learning Integration (2023-2024)**: Expansion of contrastive-based methods showing significant improvements over reconstruction-based approaches due to powerful feature learning and node representation capabilities.
- **MDST-GNN (2025)**: Multi-distance spatial-temporal graph neural network for blockchain anomaly detection integrating multi-distance graph convolutional architecture with adaptive temporal modeling.

### Latest Developments (2024-2025)
- **Semi-supervised Methods**: TSAD (Transformer-based semi-supervised anomaly detection for dynamic graphs) and SAD frameworks incorporating limited labeled data for practical scenarios.
- **Generative Semi-supervised Approaches**: Methods leveraging asymmetric local affinity and egocentric closeness priors to generate reliable outlier nodes with partial labeling.
- **Multi-level Contrastive Learning**: Advanced frameworks mitigating local consistency deception in unsupervised detection through multi-level information processing.
- **ADA-GAD (2024)**: Anomaly-Denoised Autoencoders using learning-free anomaly-denoised augmentation to address anomaly overfitting and homophily trap problems.
- **G3AD (2024)**: Framework designed to guard GNNs from encoding inconsistent information and directly reconstructing abnormal graphs in unsupervised settings.
- **Diffusion-enhanced Methods (2025)**: DE-GAD combining diffusion-based enhancement modules with multi-view contrastive learning for improved anomaly identification.

---

## Comprehensive Table: Prior Work vs. Methods vs. Results

| Citation | Year | Method Category | Approach Type | Key Methodology | Dataset(s) | Metric | Performance | Limitation |
|----------|------|-----------------|---------------|-----------------|-----------|--------|-------------|-----------|
| Tang et al. | 2022 | Reconstruction | Unsupervised | GNN architecture analysis | Synthetic + theoretical | Conceptual | Identifies fundamental challenges | May not generalize to all graph types |
| Goodge et al. (LUNAR) | 2021 | Density-based | Unsupervised | GNN + LOF hybrid | Multiple benchmarks | AUC, F1 | Outperforms standalone methods | Limited on very sparse graphs |
| 2023 Paper | 2023 | Reconstruction | Semi-supervised | Dual-channel positive/negative attention | Fluctuating IoT | Accuracy | 87-92% | Requires labeled anomalies |
| SmoothGNN | 2024 | Reconstruction | Unsupervised | Feature + smoothing loss | Standard benchmarks | AUC | ~15% improvement over baselines | Sensitive to hyperparameters |
| GeneralDyG | 2024 | Multi-method | Supervised | Dynamic graph approach | SWaT, WADI, Bitcoin-Alpha | F1, AUC | F1: 85.19% (SWaT), 60.43% (WADI); AUC +3.2-4.5% | Performance varies by dataset |
| GCN-VAE | 2024 | Reconstruction | Unsupervised | GCN-Variational Autoencoder | Vibration/Current time series | Accuracy, Precision, Recall | Mean Acc: 88.9%, Prec: 89.1%, Rec: 87.6%, F1: 88.3%, AUC-ROC: 0.93 | Domain-specific features required |
| GDN (Multivariate) | 2021 | Reconstruction | Semi-supervised | Graph-based deviation network | SWaT, WADI | Precision, F-measure | Prec: 0.99 (SWaT), 0.98 (WADI); F-measure: 54% higher than baseline on WADI | May struggle with non-temporal graphs |
| IRC-Safe GAE | 2022 | Reconstruction | Unsupervised | Intrinsic redundancy cancellation | Jet anomaly detection | AUC | 0.902 (rivals CNN: 0.93, 0.91) | Limited to specific domains |
| GAT + Informer | 2024 | Reconstruction | Semi-supervised | Graph Attention + Informer ensemble | Multivariate time series | F1, AUC | State-of-the-art on time series | Long-sequence training intensive |
| RL-GNN Hybrid | 2025 | Supervised | Supervised classification | Reinforcement learning + GNN fusion | Financial transactions | AUROC, F1, AP, MCC | AUROC: 0.872, F1: 0.839, AP: 0.683, MCC: 0.54 | Community detection overhead |
| EAGLE | 2025 | Contrastive | Unsupervised | Contrastive learning + pre-training | Benchmark datasets | AUC, F1 | ~15% improvement over DOMINANT | Pre-training cost significant |
| ANEMONE | 2022 | Contrastive | Unsupervised | Multi-scale contrastive views | Standard benchmarks | AUC, F1 | Handles hybrid distributions | Computationally expensive |
| TSAD | 2024 | Transformer-based | Semi-supervised | Transformer for dynamic graphs | Dynamic graph datasets | F1, AUC | >80% F1 on standard benchmarks | Complex temporal dependencies |
| ADA-GAD | 2024 | Reconstruction | Unsupervised | Anomaly-denoised augmentation | Multiple benchmarks | AUC, F1 | Mitigates overfitting issues | Augmentation strategy critical |
| GNN-based BGP | 2023 | Supervised | Supervised classification | GCN for BGP anomaly detection | Network traffic | F1, AUC | F1: 0.78-0.96, AUC: 0.72-0.99 | Configuration-dependent |
| GCN + GAT Hybrid | 2025 | Supervised | Supervised classification | Hybrid GCN-GAT architecture | Firewall logs | Precision, Recall, F1 | Rec: 99.04%, Acc: 99.04%, Prec: 98.43%, F1: 98.72% | Domain-specific tuning required |

---

## Detailed Methodology Overview

### 1. Reconstruction-Based Methods

**Core Principle**: These methods learn to reconstruct normal graph behavior and detect anomalies via reconstruction error, under the assumption that anomalous instances will have higher reconstruction errors than normal ones.

**Key Techniques**:
- **Graph Autoencoders (GAE)**: Encode graph structure and node features into low-dimensional representations, then decode back to original space. Anomalies detected when reconstruction error exceeds threshold.
- **Variational Graph Autoencoders (VGAE)**: Probabilistic variant introducing latent variable models for more robust anomaly detection.
- **Mirror Temporal Convolutional Networks**: Combine GCN with temporal convolutions for time-series-based graph anomaly detection.

**Notable Methods**:
- DOMINANT: First systematic application of GAE to anomaly detection
- SmoothGNN: Incorporates smoothing-aware regularization
- ADA-GAD: Addresses anomaly overfitting through anomaly-denoised augmentation
- G3AD: Guards against encoding inconsistent information

**Challenges**:
- Anomaly Overfitting: Direct reconstruction of graphs with anomalies can cause models to learn anomalous patterns
- Homophily Trap: Anomalies may share similar neighborhoods with normal nodes
- Reconstruction error threshold selection is critical and dataset-dependent

**Performance Range**:
- AUC: 0.85-0.93 across benchmark datasets
- F1 scores: 0.78-0.99 (highly dataset-dependent)
- Precision: 0.87-0.99

### 2. Distance-Based Methods

**Core Principle**: Measure distance between nodes in learned representation space. Anomalies are points far from normal data distribution.

**Key Techniques**:
- **Spatial Distance Metrics**: Edge weights represent spatial distances; larger weights indicate shorter distances
- **Distance Amplification**: Scaling factors adjusted to approach 1 for normal regions and deviate significantly for anomalous regions
- **Multi-Distance Architecture**: Capture both local and global spatial dependencies

**Notable Methods**:
- MDST-GNN: Multi-distance spatial-temporal GNN for blockchain transactions
- Graph Deviation Network: Detects small-deviation anomalies overlooked by other methods
- Distance-Aware GAT: Leverages edge weights for anomaly pattern detection

**Advantages**:
- Interpretable anomaly signals based on spatial relationships
- Can detect small-deviation anomalies
- Effective for blockchain and transaction networks

**Performance Range**:
- Can achieve F1 scores comparable to reconstruction methods
- Particularly effective for fraud/transaction detection

### 3. Density-Based Methods

**Core Principle**: Anomalies have significantly different local density compared to normal points.

**Key Techniques**:
- **Local Outlier Factor (LOF)**: Measures local density of each point relative to neighbors
- **DBSCAN Integration**: Density-based clustering adapted for GNN frameworks
- **LUNAR**: Combines GNN with LOF for trainable local outlier detection
- **Density Anomalies**: Identify subgraphs with unusual density (higher/lower connections)

**Notable Methods**:
- LUNAR: Hybrid deep GNN + LOF approach
- FRAUDAR: Greedy algorithm with density-based metrics for fraud detection
- Density-aware GNN models incorporating neighborhood density metrics

**Performance Characteristics**:
- Generally comparable to reconstruction-based methods
- Effective on datasets with clear density patterns
- Scales well with large graphs

### 4. Contrastive Learning Methods

**Core Principle**: Learn representations where normal and anomalous nodes are well-separated through contrastive objectives.

**Key Techniques**:
- **Multi-scale Contrastive Learning**: Compare nodes across multiple graph views/scales
- **Multi-view Approaches**: Evaluate consistency across different subgraph perspectives
- **Diffusion-Enhanced Learning**: Combine diffusion-based enhancement with contrastive objectives
- **Hard Negative Generation**: Strategically create challenging negative examples for learning

**Notable Methods**:
- ANEMONE: Multi-scale contrastive framework
- EAGLE: Pre-training + contrastive learning achieving 15% improvements
- DE-GAD: Diffusion-enhanced multi-view contrastive learning
- Multi-level contrastive frameworks: Address local consistency deception

**Performance Range**:
- Significantly outperforms reconstruction-based: ~15% improvement on benchmarks
- AUC: 0.85-0.95
- F1: 0.82-0.97

**Challenges**:
- Complex hybrid distributions in real-world data
- Local consistency deception in multi-view learning
- Higher computational cost than single-method approaches

---

## Learning Paradigms

### Unsupervised Approaches

**Assumptions**: No labeled anomaly data available; anomalies assumed to deviate from learned normal patterns.

**Methods**:
- SmoothGNN
- G3AD
- EAGLE (contrastive pre-training)
- GAE-based methods
- LOF-based methods

**Typical Performance**:
- AUC: 0.82-0.93
- F1: 0.75-0.92
- Highly sensitive to contamination rate

**Advantages**:
- No annotation cost
- Can discover novel anomaly types

**Disadvantages**:
- Difficult threshold selection
- May miss anomalies with subtle signatures
- Sensitive to unlabeled contamination

### Semi-Supervised Approaches

**Assumptions**: Small set of labeled anomalies available to guide learning.

**Methods**:
- TSAD: Transformer-based for dynamic graphs
- SAD: Semi-supervised anomaly detection on dynamic graphs
- Generative semi-supervised methods
- Dual-channel reconstruction approaches
- Graph structure learning frameworks

**Typical Performance**:
- F1: 0.80-0.95
- Can achieve >90% F1 on standard benchmarks
- Significantly better than unsupervised with limited labels

**Practical Advantages**:
- Realistic assumption for many applications
- Reduced labeling burden (few anomalies needed)
- Better anomaly coverage than supervised alone

**Key Challenge**: Finding sufficient representative labeled anomalies for rare events.

### Supervised Approaches

**Assumptions**: Substantial labeled training data with both normal and anomalous examples.

**Methods**:
- CRC-SGAD: Conformal risk control for supervised GAD
- GCN-based network intrusion detection (76-88% accuracy)
- Hybrid GCN-GAT classifiers
- RL-GNN fusion for fraud detection
- GNN-based BGP anomaly detection

**Typical Performance**:
- F1: 0.85-0.99
- Accuracy: 76-99%
- AUC: 0.80-0.99
- Precision-Recall balanced

**Advantages**:
- Highest detection rates when training data adequate
- Clear learning objectives
- Interpretable decision boundaries

**Disadvantages**:
- High annotation cost
- Requires representative anomaly examples during training
- May overfit to specific anomaly types seen in training

---

## GNN Backbone Architectures

### Graph Convolutional Networks (GCN)
- **Strengths**: Effective local structure extraction, stable performance
- **Limitations**: May not capture long-range dependencies; F1 scores typically 0.73-0.88
- **Use Cases**: Node-level anomaly detection on attributed networks

### Graph Attention Networks (GAT)
- **Strengths**: Adaptive neighborhood aggregation via attention; captures anomalous patterns through edge weights
- **Limitations**: Quadratic complexity with respect to node degree
- **Use Cases**: Time series anomaly detection, networks with varying edge importance

### Graph Isomorphism Networks (GIN)
- **Strengths**: More expressive than GCN for distinguishing graph structures
- **Limitations**: Performance comparable to GCN/GAT in some tasks; limited advantage for anomaly detection alone
- **Use Cases**: Graph-level and subgraph-level anomaly detection

### Hybrid and Ensemble Architectures
- **GCN + GAT**: Recall 99.04%, Accuracy 99.04%, F1 98.72%
- **GCN + GAT + GIN Ensemble**: Handles diverse anomaly types with <1% false positive rate
- **Transformer + GNN**: Captures long-range temporal dependencies for time series

---

## Dataset Benchmarks and Evaluation

### Standard Graph Datasets

**Cora Dataset**:
- Nodes: 2,708
- Edges: 5,429
- Typical anomaly injection: 5.5%
- Node types: 7
- Features: Word vector representations

**Citeseer Dataset**:
- Nodes: 3,327
- Edges: 4,732
- Typical anomaly injection: 4.5%
- Node types: 6

**OGB-arXiv**:
- Nodes: 169,343
- Edges: 1,166,243
- Typical anomaly injection: 3.5%
- Node types: 40
- Focus: Large-scale evaluation

**CoraFull**:
- Nodes: 19,793
- Classes: 70
- More diverse domain categories than standard Cora

### Domain-Specific Datasets

**Temporal/Time Series Datasets**:
- SWaT: Industrial control system data (F1: 85.19% best method, baseline 0.99 precision)
- WADI: Water distribution dataset (F1: 60.43%, 54% improvement over baseline on F-measure)
- Bitcoin-Alpha: Transaction network data (AUC improvement: 3.2-4.5%)

**Network/Security Datasets**:
- DDoS traffic (GCN accuracy: 76%)
- TOR-nonTOR traffic (GCN accuracy: 88%)
- Firewall logs (Hybrid GCN-GAT F1: 98.72%)

### Evaluation Metrics

**Primary Metrics**:
1. **AUC-ROC**: Range 0.72-0.99 across methods
   - Preferred for imbalanced datasets
   - Less sensitive to contamination rate
   - Standard across most papers

2. **F1-Score**: Range 0.75-0.99
   - Sensitive to contamination rate (critical caveat)
   - Balances precision and recall
   - Most commonly reported metric

3. **Precision & Recall**: Variable ranges
   - Precision: 0.87-0.99
   - Recall: 0.87-0.99
   - Often reported separately in network security applications

4. **AUROC (Area Under ROC Curve)**: 0.80-0.99
   - Particularly useful for financial fraud detection
   - Robust to class imbalance

5. **Macro-F1**: For multi-class anomaly scenarios

6. **NDCG@K**: For ranking-based anomaly severity in subgraph detection

---

## Computational Complexity and Scalability

### Time Complexity
- **Linear Scaling Potential**: Some models demonstrate O(n+m) complexity (n=nodes, m=edges)
- **Practical Inference Time**: ~8.7 milliseconds per network flow (acceptable for real-time)
- **Training Cost**: Computationally intensive; GDN training particularly expensive

### Scalability Challenges
- **Dimensionality Impact**:
  - Below 1,500 dimensions: System delay <1,000 ms
  - Above 1,500 dimensions: Significant latency increase
  - Throughput peaks at 1,500 dimensions (>20,000 samples)

- **Memory Requirements**: Exponential growth with adjacency matrix construction and attention computations

- **Solutions Deployed**:
  - Correlation partitioning for time series data
  - Distributed computing with Apache Flink
  - Hierarchical and multi-resolution embeddings via U-Net integration
  - Batch processing strategies

### Scalability Results
- Successfully handled: Up to 1M nodes in some implementations
- Real-time capable: 8.7 ms per flow inference
- Throughput: >20,000 samples at optimal dimensionality

---

## Identified Gaps and Open Problems

### Fundamental Challenges

1. **Over-smoothing and Representation Collapse**
   - Deep GNN stacking leads to indistinguishable normal/anomalous representations
   - Mitigations: Skip connections, layer normalization, but not fully solved

2. **Homophily Assumption Violations**
   - GNNs assume similar nodes connect; anomalies may violate this
   - Impact: Difficult to detect anomalies with few distinct neighbors

3. **Graph Structure Learning**
   - Many GNNs assume fixed, known graph structure
   - Real-world graphs are often dynamic and partially observed
   - Limited work on jointly learning structure and anomalies

4. **Anomaly Overfitting**
   - Direct reconstruction can cause models to memorize anomalies
   - Problem exacerbated in high-contamination settings
   - Solutions: Denoised augmentation (ADA-GAD), but computationally expensive

5. **Label Scarcity and Bias**
   - True anomaly labels rare and expensive to obtain
   - Supervised methods rely on representative training anomalies
   - Semi-supervised approaches make strong assumptions about anomaly rarity

### Methodological Gaps

1. **Threshold Selection**
   - No principled method for anomaly score thresholding across datasets
   - F1-score and AUC sensitive to contamination rate
   - Requires manual tuning or validation set with known anomalies

2. **Multi-Type Anomaly Detection**
   - Most methods target single anomaly type (node, edge, or subgraph)
   - Limited work on detecting multiple anomaly types simultaneously
   - Performance trade-offs unclear

3. **Dynamic Graph Anomaly Detection**
   - Methods for temporal and dynamic graphs still maturing
   - Computational cost of temporal modeling significant
   - Limited benchmark datasets with dynamic ground truth

4. **Interpretability**
   - GNN decisions difficult to interpret
   - "Why is this node anomalous?" remains challenging
   - Limited theoretical guarantees on detection correctness

5. **Fairness and Bias**
   - Emerging concern in supervised and semi-supervised settings
   - Limited work on detecting adversarial/manipulated anomalies
   - Robustness to poisoning attacks not well-studied

### Application-Specific Gaps

1. **Non-Graph Domains**: Many real data naturally graph-structured, but transformation loss is unclear

2. **Edge Features**: Many GAEs overlook edge characteristics, limiting performance on rich network data

3. **Heterogeneous Networks**: Most methods designed for homogeneous graphs; heterogeneous variants emerging but less mature

4. **Causal Analysis**: Correlation ≠ causality; causal relationships in GNN anomalies remain underexplored

---

## State of the Art (2025 Summary)

### Current Best Performers

**Unsupervised Settings**:
- **EAGLE**: Achieves ~15% improvement over DOMINANT through contrastive pre-training
- **SmoothGNN**: Competitive AUC via smoothing-aware regularization
- **DE-GAD**: Addresses local consistency deception in multi-view learning
- **Typical Performance**: AUC 0.85-0.95, F1 0.82-0.97

**Semi-Supervised Settings**:
- **TSAD**: Transformer-based dynamic graph detection, F1 >80%
- **Generative Methods**: Leverage asymmetric local affinity priors
- **Typical Performance**: F1 0.80-0.95 with limited labels

**Supervised Settings**:
- **Hybrid GCN-GAT**: Recall 99.04%, F1 98.72%
- **RL-GNN Fusion**: AUROC 0.872, F1 0.839 for fraud detection
- **Typical Performance**: F1 0.85-0.99, Accuracy 76-99%

**Domain Leaders**:
- **Time Series**: GAT + Informer ensemble (multivariate)
- **Fraud Detection**: RL-GNN + community mining (AUROC 0.872)
- **Network Security**: GCN-GAT hybrid (F1 98.72%)
- **Blockchain**: MDST-GNN (multi-distance approach)

### Emerging Trends

1. **Contrastive Learning Dominance**: Overtaking reconstruction-based methods
2. **Hybrid Architectures**: Combining multiple GNN types and learning paradigms
3. **Generative Models**: VAE, diffusion-based, and GAN variants for anomaly generation
4. **Transformer Integration**: Effective for temporal/sequential dependencies
5. **Multi-task Learning**: Proxy task design gaining importance (2025 survey finding)
6. **Few-shot and Zero-shot**: Emerging approaches for rare anomaly scenarios
7. **Robustness Research**: Adversarial anomalies and model poisoning resistance
8. **Causal Learning**: Moving toward causal GNN architectures for interpretation

---

## Quality Assessment of Evidence

### Strength of Evidence
- Multiple papers from top venues (ICML, IJCAI, NeurIPS, AAAI, TKDE)
- Consistent performance metrics across papers
- Reproducible benchmarks (Cora, OGB, SWaT, WADI)
- Growing consensus on evaluation protocols

### Limitations in Literature
- High variance in reported results (F1: 0.75-0.99) due to dataset differences
- Contamination rate significantly affects metric interpretation
- Limited comparison between unsupervised/semi-supervised/supervised on identical datasets
- Computational cost rarely reported consistently
- Theoretical analysis lacking; mostly empirical evaluations

### Key Methodological Issues in Evaluated Papers
1. **Evaluation Protocol Bias**: F1-score inflation with biased protocols (Amini et al., 2021)
2. **Contamination Sensitivity**: Metrics highly dependent on anomaly percentage
3. **Train-Test Leakage**: Some papers may not properly separate anomaly types
4. **Threshold Bias**: Manual thresholding can inflate results

---

## References (Selected High-Impact Papers)

### Foundational Works
- Ding, K., Li, J., Bhanushali, R., & Liu, H. (2019). Anomaly Detection with Robust Deep Autoencoders. SIGKDD. https://arxiv.org/abs/1811.08407

### Major Surveys
- Chen, Z., et al. (2021). A Comprehensive Survey on Graph Anomaly Detection with Deep Learning. IEEE TKDE.
- Lou, S., et al. (2025). Deep Graph Anomaly Detection: A Survey and New Perspectives. IEEE TKDE. https://arxiv.org/abs/2409.09957

### Key Methodological Papers
- Tang, J., Li, G., Liu, H., & Zhu, H. (2022). Rethinking Graph Neural Networks for Anomaly Detection. ICML. https://proceedings.mlr.press/v162/tang22b/tang22b.pdf
- Goodge, A., Ghosh, B., Metsis, V., et al. (2021). LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks. WSDM. https://arxiv.org/abs/2112.05355
- Luo, Y., Wen, Z., & Cheng, X. (2021). Graph Neural Network-Based Anomaly Detection in Multivariate Time Series. AAAI. https://arxiv.org/abs/2106.06947

### Recent Advances (2024-2025)
- Multiple papers on semi-supervised approaches (TSAD, SAD, Generative methods)
- Contrastive learning methods (EAGLE, ANEMONE, DE-GAD)
- Dynamic graph approaches (GeneralDyG)
- Hybrid architectures (GCN-GAT combinations, RL-GNN fusion)

---

## Recommendations for Future Research

1. **Develop Principled Threshold Selection**: Create adaptive, theoretically-grounded anomaly score thresholding methods

2. **Address Over-smoothing**: Investigate deep GNN architectures that maintain anomaly-normal separability

3. **Multi-Anomaly Detection**: Extend methods to simultaneously detect multiple anomaly types (node, edge, subgraph)

4. **Dynamic and Evolving Graphs**: Develop efficient methods for temporal graph anomaly detection

5. **Causal Analysis**: Incorporate causal frameworks to explain why nodes are anomalous

6. **Robustness and Adversarial Settings**: Study resistance to poisoning and adversarial anomalies

7. **Fairness Guarantees**: Ensure equitable detection across different node/edge populations

8. **Benchmark Standardization**: Establish consistent evaluation protocols to reduce metric variance

9. **Computational Efficiency**: Design methods scalable to billion-node graphs with <10ms inference latency

10. **Few-shot and Zero-shot Learning**: Leverage meta-learning for detection with minimal anomaly examples
