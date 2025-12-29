# Graph Anomaly Detection Methods: Detailed Comparison

## 1. Reconstruction Error Methods

### Overview
Base assumption: Normal nodes are easier to reconstruct than anomalous nodes from the graph structure and features.

### Key Methods

#### DOMINANT (2019)
- **Architecture**: Dual-encoder with separate structure and attribute autoencoders
- **Structure Loss**: L||A - A'||_F^2 (Frobenius norm)
- **Attribute Loss**: L||X - X'||_F^2
- **Anomaly Score**: max(structure_error, attribute_error) or weighted combination
- **Limitations**: Fails when normal neighborhoods are harder to reconstruct
- **Performance**: Baseline method; superseded by newer approaches

#### DONE (2020)
- **Architecture**: Two independent autoencoders + anomaly-aware loss
- **Key Innovation**: Anomaly-aware loss function to quantify anomaly scores
- **Structure AE**: Reconstructs adjacency matrix A
- **Attribute AE**: Reconstructs node features X
- **Score Function**: Combined reconstruction errors with weighting
- **Issues**: Still relies on reconstruction error as primary signal

#### Graph Autoencoder (GAE) - General Framework
```
Encoder: GNN_enc(X, A) -> Z (latent embedding)
Decoder_struct: GNN_dec(Z) -> A' (structure)
Decoder_attr: MLP(Z) -> X' (attributes)
Loss: L_struct + λ·L_attr + regularization
Anomaly score: ||A - A'|| + α·||X - X'||
```

#### GDAE (Graph Deep Autoencoder)
- **Key Feature**: Attention mechanism for neighbor importance weighting
- **Encoder**: GNN with attention to node features and neighbors
- **Decoder**: Separate structure and attribute reconstruction
- **Application Domain**: Multi-attributed networks
- **Advantages**: Better handling of heterogeneous node features

### Identified Problems (Tang et al., 2022)

**Critical Finding**: Reconstruction loss is insufficient and can be misleading
- Normal nodes in homogeneous neighborhoods may have HIGH reconstruction error
- Anomalous nodes camouflaged in similar neighborhoods may have LOW reconstruction error
- Root cause: Autoencoder learns local patterns too well, making both normal and anomalous nodes reconstructible

**Solution Direction**: Combine with neighborhood contrast or spectral methods

---

## 2. Outlier Scoring Methods

### Local Inconsistency Mining (LIM)

**Core Principle**: Anomalies deviate significantly from their neighborhoods

**Mathematical Formulation**:
```
anomaly_score(v) = divergence(embedding(v), embeddings(neighbors(v)))
```

**Distance Metrics**:
- Euclidean distance: ||z_v - mean(z_neighbors)||
- Cosine distance: 1 - similarity(z_v, z_neighbors)
- KL divergence: KL(p_v || p_neighbors)

#### MLP-Based LIM (GADAM, 2024)

**Innovation**: Compute local inconsistency BEFORE message passing

**Architecture**:
```
Input: Raw node features X, adjacency A
LIM_score(v) = MLP(X_v, X_neighbors) - before GNN layers
anomaly_score(v) = LIM_score(v) + adaptive_message_passing(v)
```

**Advantages**:
- Preserves local anomaly signals
- Avoids suppression by GNN message passing
- Conflict-free combination with global perspective

#### Spectral-Based Outlier Scoring

**Theory**: Anomalies induce spectral shifts in graph Laplacian

**Method**:
1. Compute graph Laplacian L = D - A
2. Compute eigenvalues λ and eigenvectors of L
3. Transform node into spectral domain
4. Measure "right-shift" of spectral energy
5. Nodes causing energy shift to higher frequencies → anomalies

**Advantages**:
- Theoretically grounded in graph signal processing
- Captures global structural patterns
- Less sensitive to local noise

---

## 3. Contrastive Learning Methods

### Core Idea
Learn representations where normal patterns are similar and anomalous patterns are dissimilar.

### ANEMONE (AAAI 2023)

**Multi-Scale Contrastive Learning**:

```
Input: Graph G = (V, E, X)
Step 1: Multi-scale construction
  G_patch = local neighborhoods (k-hop)
  G_context = larger subgraphs (k+δ-hop)

Step 2: Multi-view representation
  z_patch = GNN(G_patch)
  z_context = GNN(G_context)

Step 3: Contrastive loss
  L_contrastive = -log(exp(sim(z_patch, z_context)/τ) / Σ exp(sim/τ))

Step 4: Anomaly scoring
  score(v) = -sim(z_patch(v), z_context(v))
```

**Key Insight**: Multi-scale enables capturing anomalies at different pattern scales

**Performance**: Effective on fraud detection datasets (YelpChi, Amazon, BlogCatalog)

### EAGLE (2025)

**Heterogeneous Graph Contrastive Learning**:

**Method**:
```
For each node v:
  D_v = distance to local context (neighbors)
  Positive pairs: normal nodes with high D_v similarity
  Negative pairs: anomalous nodes with different D_v

Loss: Triplet or NT-Xent based on distance distribution
```

**Efficiency**: Optimized for large heterogeneous graphs

### Problem Identified (2025): Local Consistency Deception

**Issue**: Interfering edges invalidate the low-similarity-means-anomaly assumption

**Example**:
- Normal node N connected to multiple normal nodes N1, N2, N3
- Interfering edge connects N to anomalous node A
- Similarity(N, neighbors) becomes ambiguous
- Contrastive loss direction becomes unclear

**Solutions**:
- Clean-view perspective: Filter interfering edges
- Selective sampling: Oversample clean edges
- Robust loss: Handle edge uncertainty

---

## 4. Graph Autoencoder Architectures

### Standard Architecture

```
Encoder Block:
  X, A -> GNN_layers -> Latent_Z
  Z dimension: typically 64-256

Decoder Block (Structure):
  Z -> FC -> A' (adjacency reconstruction)
  Loss: binary cross-entropy on edge predictions

Decoder Block (Attributes):
  Z -> MLP_layers -> X' (feature reconstruction)
  Loss: MSE or cross-entropy depending on feature type

Total Loss:
  L = α·L_structure + β·L_attr + λ·L_reg
```

### GRASPED (2024): Spectral Enhancement

**Key Innovation**: Spectral encoder and decoder

**Encoder**:
- Graph Wavelet Convolution (GWC)
- Captures both low and high-frequency components
- Better preservation of anomalous high-frequency signals

**Decoders**:
- Structural decoder: Predicts adjacency
- Attribute decoder: Reconstructs features
- Joint optimization prevents reconstruction error trade-off

**Advantage**: Addresses limitation that mean reconstruction error can mislead

### ADA-GAD (2023): Denoising Autoencoders

**Two-Stage Approach**:

```
Stage 1: Data Denoising
  Input: Contaminated training set with unlabeled anomalies
  Method: Iterative cleaning, remove likely anomalies
  Output: Clean training set

Stage 2: Autoencoder Training
  Input: Cleaned training data
  Architecture: Standard GAE
  Anomaly detection: Reconstruction error on test set
```

**Motivation**: Unlabeled anomalies in training poison unsupervised methods

### Enhanced GAE with Subgraph (2025)

**Preprocessing Stage**:
1. Extract subgraphs around each node (k-hop neighborhood)
2. Compute local structural features (clustering coefficient, degree, etc.)
3. Aggregate node-subgraph relationships

**Encoder**:
- Input: Node features + subgraph embeddings
- GNN with attention to subgraph importance
- Output: Rich latent representation

**Decoders**:
- Structure learning decoder: Learns graph structure relationships
- Attribute decoder: Reconstructs features
- Multi-view reconstruction

**Benefit**: Leverages local topology for better representations

---

## 5. Spectral Methods

### Foundational Theory

**Spectral Property of Anomalies**:
- Graph Laplacian L = D - A (diagonal degree matrix - adjacency)
- Eigenvalue decomposition: L = UΛU^T
- Anomalies cause shifts in eigenvalue distribution

**Observations**:
- Normal graphs: Spectral energy concentrated at lower frequencies
- Anomalous nodes: Induce "right-shift" phenomenon (energy shifts to higher frequencies)

### Dynamic Wavelets Approach

**Method**:
```
Step 1: Define trainable wavelets ψ_k on graph
Step 2: For each node v:
  w_k(v) = [ψ_k * signal](v)  (wavelet transform)

Step 3: Anomaly features
  anomaly_features(v) = [w_1(v), w_2(v), ..., w_K(v)]

Step 4: Classification
  anomaly_score(v) = classifier(anomaly_features(v))
```

**Advantage**: Adapts wavelets to data rather than fixed wavelets

### SPS-GAD (2025): Spectral-Spatial for Heterophilic Graphs

**Problem**: Standard methods assume homophily (similar nodes connected)
**Solution**: Spectral filtering + spatial reconstruction

```
Step 1: Spectral Filtering
  - Apply low-pass and high-pass filters in spectral domain
  - Preserve global structure, filter local noise

Step 2: Node Reconstruction Module
  - Extract stable intermediate representations
  - Mitigate feature inconsistencies from node camouflage

Step 3: Combined Scoring
  score = spectral_energy_shift + reconstruction_error
```

**Key Innovation**: Two-step process handles heterophily by not assuming similarity

---

## 6. Adaptive Message Passing

### GADAM (ICLR 2024)

**Problem**: Message passing in GNNs suppresses local anomaly signals
- GNNs make connected nodes similar (aggregation)
- This conflicts with LIM principle (anomalies differ from neighbors)

**Solution**: Conflict-free local inconsistency mining + adaptive MP

```
Architecture:
├─ Local Inconsistency Mining (MLP-based)
│  Input: Raw features X, neighborhood X_neighbors
│  Output: local_scores (before GNN layers)
│
├─ Adaptive Message Passing
│  Hybrid attention mechanism
│  For each node v:
│    attention(v, u) = f(local_score(v), local_score(u))
│    If both normal: high attention (aggregate)
│    If both anomalous: high attention (preserve signal)
│    If mixed: low attention (avoid suppression)
│
└─ Final Score
   anomaly_score(v) = local_scores(v) + combine(MP_features(v))
```

**Key Components**:

1. **MLP-Based LIM**:
```
local_score(v) = MLP([X_v; mean(X_neighbors); std(X_neighbors)])
```

2. **Attention Gates**:
```
attention_weight(v,u) = sigmoid(W * [local_score(v); local_score(u)])
```

3. **Selective Aggregation**:
```
aggregated(v) = Σ_u attention(v,u) * feature(u)
```

**Result**: Combines best of both worlds - local inconsistency + global perspective

---

## 7. Multi-Level Detection (Nodes, Edges, Subgraphs)

### UniGAD (NeurIPS 2024)

**Goal**: Unified framework for all anomaly types

**Key Insight**: Convert node/edge/subgraph detection to graph-level classification

**Method**:

```
Input: Graph G with potential anomalies at multiple levels

Step 1: Spectral Sampling (Anomaly-Rich Subgraph Extraction)
  - Sample subgraphs using spectral importance
  - Prioritize subgraphs containing anomalies
  - For nodes: extract ego-graphs around suspected nodes
  - For edges: extract subgraphs containing edge endpoints
  - For subgraphs: sample using spectral energy metrics

Step 2: Graph-Level Classification
  - Embed each subgraph using GNN
  - Classify as anomalous or normal subgraph
  - Aggregate scores back to original level

Step 3: Multi-Level Output
  - Node anomaly: score from node-centered subgraphs
  - Edge anomaly: score from edge-centered subgraphs
  - Subgraph anomaly: direct graph-level classification
```

**Advantages**:
- Single model for all anomaly types
- Principled framework using spectral theory
- Handles complex multi-level anomalies

---

## 8. Dynamic/Streaming Anomaly Detection

### STGNN (2020)

**Architecture for Temporal Graphs**:

```
Input: Dynamic graph snapshots G_t1, G_t2, ..., G_tn

Structural Component:
  - GCN or GAT for each snapshot
  - Captures spatial structure at time t

Temporal Component:
  - GRU over sequence of structural embeddings
  - Learns temporal patterns

Combined:
  h_v^t = GRU(GCN(G_t1), GCN(G_t2), ..., GCN(G_t))

Anomaly Detection:
  - Reconstruction error in temporal embeddings
  - Deviation from learned temporal patterns
```

**Edge Anomaly Detection**:
```
For each edge (u, v, t):
  - Predict embedding based on history
  - Measure reconstruction error
  - Threshold to detect anomalous edges
```

### Memory-Enhanced Dynamic Detection (2024)

**Key Idea**: Preserve normality patterns over time

```
Components:
1. Memory Module
   - Store temporal patterns of normal behavior
   - Update memory as new normal patterns emerge

2. Memory Reader
   - Retrieve relevant normal patterns for current timestamp
   - Weighted retrieval based on recency

3. Graph Autoencoder
   - Reconstruct current graph based on memory
   - Anomalies: difficult to reconstruct given normal patterns

4. Training Loss
   L = reconstruction_loss + memory_consistency + temporal_smoothness
```

**Advantage**: Naturally handles concept drift and evolving anomalies

### Streaming Detection (Sketch-Based, 2023)

**Constraint**: Constant time O(1) and memory O(1) per edge

**Method**:
```
Sketch Structure:
  - Maintain compact summary of normal patterns
  - Count sketches for subgraph frequencies
  - Min-hash for anomaly signatures

Online Processing:
  For each edge (u, v, t):
    1. Query sketch: is (u, v) normal?
    2. Compute local anomaly score O(1)
    3. Update sketch with new edge O(1)
    4. Output anomaly decision

Memory: Sketch size independent of graph size
```

---

## Performance Comparison by Method Type

### Reconstruction Methods
- **Strengths**: Simple, interpretable, fast
- **Weaknesses**: Insufficient signal, fails on sparse graphs
- **AUC Range**: 0.65-0.85
- **Failure Case**: Normal neighborhoods harder to reconstruct

### Contrastive Methods
- **Strengths**: Captures semantic patterns, multi-scale capable
- **Weaknesses**: Sensitive to interfering edges, hyperparameter tuning
- **AUC Range**: 0.75-0.90
- **Failure Case**: Local consistency deception on heterophilic graphs

### Spectral Methods
- **Strengths**: Theoretically grounded, handles heterophily, global awareness
- **Weaknesses**: Computationally expensive, eigendecomposition overhead
- **AUC Range**: 0.70-0.88
- **Failure Case**: High computational cost on very large graphs (>1M nodes)

### Adaptive Message Passing
- **Strengths**: Resolves GNN limitations, combines local and global
- **Weaknesses**: Complex architecture, hyperparameter dependent
- **AUC Range**: 0.82-0.92
- **Best Use Case**: Medium to large graphs with mixed anomaly types

### Multi-Level Unified
- **Strengths**: Single model for all tasks, principled approach
- **Weaknesses**: Newer method, less extensive evaluation
- **AUC Range**: 0.78-0.91
- **Best Use Case**: Applications requiring simultaneous node/edge/subgraph detection

### Hybrid (Reconstruction + Contrastive + Spectral)
- **Strengths**: Combines best of all approaches, most robust
- **Weaknesses**: Complex, multiple hyperparameters, slower
- **AUC Range**: 0.85-0.93
- **Current SOTA**: GAD-NR (neighborhood reconstruction + refinement)

---

## Benchmark Results Summary

### Citation Networks (Cora, CiteSeer, Pubmed)

| Method | Cora AUC | CiteSeer AUC | Pubmed AUC |
|--------|----------|--------------|-----------|
| DOMINANT | ~0.80 | ~0.78 | ~0.75 |
| DONE | ~0.81 | ~0.79 | ~0.76 |
| GAE | ~0.78 | ~0.76 | ~0.73 |
| Tang et al. (Neighborhood Contrast) | ~0.88 | ~0.85 | ~0.82 |
| ANEMONE | ~0.89 | ~0.86 | ~0.83 |
| GAD-NR | **0.8755** | **0.8771** | **0.7676** |
| GADAM | ~0.90 | ~0.88 | ~0.84 |

### Large-Scale Networks (OGBn-Arxiv)

| Method | AUC |
|--------|-----|
| DOMINANT | ~0.58 |
| GAE | ~0.56 |
| ANEMONE | ~0.61 |
| GAD-NR | **0.5799** |
| GADAM | ~0.62 |

**Observation**: All methods struggle on large sparse graphs; performance ~15-20% lower than citation networks

### Fraud Detection (YelpChi, Amazon)

| Method | YelpChi AUC | Amazon AUC |
|--------|-------------|-----------|
| DOMINANT | ~0.80 | ~0.78 |
| ANEMONE | ~0.90 | ~0.89 |
| GAD-NR | ~0.88 | ~0.87 |
| GADAM | ~0.91 | ~0.90 |

**Observation**: Contrastive and adaptive methods outperform reconstruction-only approaches significantly

### Real-Time/Streaming

| Method | Accuracy | Latency (ms) | Throughput |
|--------|----------|--------------|-----------|
| STGNN | 95.2% | ~150ms | 6.7k packets/s |
| Real-Time GNN + XAI | **96.8%** | **1450ms** (for 50k packets) | 50k packets |
| Sketch-Based | 92.1% | <1ms per edge | Unlimited (constant) |

**Note**: Real-time GNN latency is for 50k packets total, ~0.029ms per packet

---

## Selection Guide for Practitioners

### Choose Reconstruction Methods If:
- Simple implementation required
- Interpretability critical
- Computational resources limited
- Graphs are small and dense

### Choose Contrastive Methods If:
- Multi-scale anomalies present
- Fraud/social network domain
- Some hyperparameter tuning acceptable
- Homophilic graphs

### Choose Spectral Methods If:
- Heterophilic graphs (dissimilar nodes connected)
- Theoretical foundation important
- Can afford eigendecomposition cost
- Need to handle global patterns

### Choose Adaptive Message Passing If:
- Want best empirical performance
- Can handle complexity
- Need to suppress message passing artifacts
- Mixed normal and anomalous neighborhoods

### Choose Multi-Level Unified If:
- Need simultaneous node/edge/subgraph detection
- Single model preferred
- Can standardize evaluation metrics

---

**Last Updated**: 2025-12-24
**Methods Covered**: 20+ major approaches and variants
**Evaluation Basis**: 40+ peer-reviewed papers
