# Technical Summary: GNN Architecture Comparison

## Quick Reference Table

| Aspect | GCN | GraphSAGE | GAT | GIN |
|--------|-----|-----------|-----|-----|
| **Publication** | ICLR 2017 | NeurIPS 2017 | ICLR 2018 | ICLR 2019 |
| **Learning Type** | Transductive | Inductive | Transductive | Graph Classification |
| **Aggregation** | Sum (normalized) | Mean/LSTM/Pooling | Attention (multi-head) | Sum (MLP) |
| **Time per Layer** | O(\|E\|F) | O(S^L·L·F²) | O(\|E\|F'²) | O(\|V\|F²) |
| **Parameters** | ~120K | ~200K | ~280K | ~400K |
| **Cora Accuracy** | 81.5% | 86.3%* | 83.3% | - |
| **Key Strength** | Scalability, efficiency | Large graphs, induction | Heterophily, adaptive | Expressiveness guarantee |
| **Key Weakness** | Homophily assumption | Sampling overhead | Attention cost | Small datasets |

*Inductive setting (different from standard transductive benchmark)

## Mathematical Formulation Comparison

### GCN: Spectral-Inspired Aggregation
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

where: Ã = A + I (self-loops)
       D̃ = degree(Ã)
       σ = ReLU (or similar activation)
```

**Key Property**: Approximates spectral convolution via first-order Chebyshev polynomial
**Computational Advantage**: Sparse matrix multiplication O(|E|)
**Limitation**: Equal weighting of neighbors

---

### GraphSAGE: Sampling-Based Aggregation
```
h_v^(l) = σ(W^(l) [h_v^(l-1) || f_agg({h_u^(l-1) : u ∈ S(v)})])

where: S(v) = randomly sampled |S| neighbors
       f_agg ∈ {MEAN, LSTM, POOL}
       || = concatenation
```

**Key Property**: Generalization to unseen nodes via learned aggregation function
**Sampling Variants**:
- MEAN: f_agg(Z) = (1/|Z|) Σ z ∈ Z
- LSTM: f_agg(Z) = LSTM(z₁, z₂, ..., z_{|Z|})
- POOL: f_agg(Z) = max({σ(W·z + b) : z ∈ Z})

**Computational Advantage**: Fixed mini-batch size regardless of |V|
**Practical Advantage**: Handles dynamic/temporal graphs

---

### GAT: Attention-Based Aggregation
```
α_ij = exp(LeakyReLU(a^T[W·h_i || W·h_j])) / Σ_k∈N(i) exp(...)

h_v^(l) = || (k=1 to K) σ(Σ_u∈N(v) α_vu^(k) W^(k) h_u^(l-1))

where: || = concatenation over K attention heads
       α_vu^(k) = attention coefficient for edge (v,u) in head k
       a^(k) = learnable attention vector per head
```

**Key Property**: Adaptive, learned neighbor importance weights
**Multi-Head Benefit**: Captures different relationship types simultaneously
**Architecture Details** (Veličković et al.):
- Hidden layers: K=4 heads × 256 features = 1024 total
- Output layer: K=6 heads, averaged (not concatenated)

**Computational Cost**: O(|E|·F'²) due to attention computation
**Advantage for Heterophily**: Can learn to downweight dissimilar neighbors

---

### GIN: Maximally Expressive Aggregation
```
h_v^(l) = MLP^(l)((1+ε)·h_v^(l-1) + Σ_u∈N(v) h_u^(l-1))

where: ε = learnable scalar (or fixed small value)
       MLP^(l) = fully-connected layers (≥2 layers recommended)
       Σ = summation (injective on multisets)
```

**Theoretical Guarantee**: Equivalent to Weisfeiler-Lehman test (most expressive MPNN)
**Injective Aggregation**: Summation never collapses different multisets to same value
**Universal Approximation**: MLP ensures injectivity on full domain via overparameterization

**Proof Sketch**: For distinct multisets {v₁,...,vₙ} and {u₁,...,uₘ}, if Σvᵢ = Σuⱼ, then either n=m and sets are identical (by linear algebra). MLP is injective on ℝ^F.

---

## Unified Message Passing Framework

All four architectures fit the MPNN formulation:

```
m_v^(l) = ⊕_{u∈N(v)} M^(l)(h_u^(l-1), e_uv)  [Message Function]

h_v^(l) = U^(l)(h_v^(l-1), m_v^(l))            [Update Function]

where: ⊕ = aggregation (sum, mean, max, attention, etc.)
       M^(l) = learnable message function
       U^(l) = learnable update function
       e_uv = optional edge features/attributes
```

**Architecture Mapping**:

| Architecture | M^(l) | ⊕ | U^(l) |
|--------------|-------|---|-------|
| GCN | h_u·W | Normalized sum | σ |
| GraphSAGE | h_u | Mean/LSTM/Pool | σ([·\|\|·]) |
| GAT | h_u·W | Attention | σ(concat) |
| GIN | h_u·W | Sum | MLP |

---

## Complexity Analysis

### Forward Pass Time Complexity

**GCN (L layers, no sampling)**
```
T = L × (|E|·F + |V|·F²)

Bottleneck: Sparse matrix @ dense matrix (O(|E|F))
           + Dense matrix @ dense matrix (O(|V|F²))

For sparse graphs (|E| ~ |V|): T ~ O(L·|V|·F²)
For dense graphs (|E| ~ |V|²): T ~ O(L·|V|²·F)
```

**GraphSAGE (L layers, S samples per layer)**
```
T = L × S^L × |V| × (sample_gather + aggregation + MLP)
  ~ O(L·S^L·|V|·F²)

Example: L=2, S=25, |V|=1M → 1.25M operations (vs 1T for full GCN)
Reduction factor: ~10^6 for large graphs
```

**GAT (L layers, multi-head attention)**
```
T = L × |E| × F' × F' + L × |V| × F'²
  ~ O(L·|E|·F'²)

For K heads with F' = F/K:
T_total ~ O(L·|E|·F²)  [same as GCN but with higher constant]

Practical overhead: 4× for typical parameters
```

**GIN (L layers, MLP aggregation)**
```
T = L × (|E|·F² + |V|·F² + |V|·mlp_cost)
  ~ O(L·|V|·F² + mlp_overhead)

MLP in aggregation adds 2-3× overhead vs simple sum
```

### Memory Complexity

**Activation Memory** (gradient computation requires caching)
```
Storage = |V| × F × L + |E| × (batch_factor)
        ~ O(|V|·F·L) for full batch
        ~ O(batch_size·F·L) for mini-batch

Example: 1M nodes, F=64, L=3 → 192M floats = 768 MB
         vs 10K batch, same params → 7.68 MB (99.8% reduction)
```

**Optimizer State** (Adam stores 1st and 2nd moments)
```
Memory_optimizer = 2 × num_parameters × 4 bytes
                 = 2 × (|V|·F·L + other_params)

Example: 200K parameters → 1.6 MB per optimizer state
```

---

## Receptive Field and Depth Analysis

### Why 2-3 Layers is Typical

**Receptive Field Growth**
```
Layer 1: Each node sees its 1-hop neighborhood (~average_degree neighbors)
Layer 2: Each node sees 2-hop neighborhood (~degree² nodes)
Layer 3: Each node sees 3-hop neighborhood (~degree³ nodes)

For social graphs (avg_degree ~ 50):
  L=1: ~50 neighbors
  L=2: ~2,500 neighbors
  L=3: ~125,000 neighbors

For large graphs with 1M nodes, L=3 causes "neighborhood explosion"
```

**Over-Smoothing Effect** (empirical)
```
Accuracy vs Depth (Cora dataset):
  1 layer:  ~75%  (underfitting, missing context)
  2 layers: ~81%  (optimal, good accuracy)
  3 layers: ~80%  (negligible degradation)
  4 layers: ~78%  (significant degradation)
  5 layers: ~70%  (severe over-smoothing)

Similarity of node representations:
  Layer 1: ~0.2 average cosine similarity
  Layer 2: ~0.4 average cosine similarity
  Layer 3: ~0.6 average cosine similarity
  Layer 4: ~0.85 average cosine similarity (high redundancy)
```

**Root Cause**: Laplacian smoothing analogy
- Aggregation resembles heat diffusion on graph
- Information from all nodes gradually spreads and converges
- Loss of node distinctiveness (all nodes → similar embedding space point)

### Practical Mitigation Strategies

1. **Skip Connections** (ResNet-style)
```
h_v^(l) = h_v^(l-1) + σ(message_passing_update^(l))
```
Preserves original node features through depth, partially preventing convergence.

2. **Batch Normalization**
```
h^(l) = BatchNorm(W·aggregate(h) + b)
```
Standardizes intermediate activations, reduces feature alignment.

3. **Layer Normalization** (preferred for graphs)
```
h^(l) = γ·(h_normalized) + β
```
Independent normalization per sample, less sensitive to batch composition.

4. **Decoupled Receptive Field**
```
Use spectral filters (Chebyshev) for distant receptive field
Use single/few layers for refinement
Orthogonalizes depth from receptive field size
```

---

## Aggregation Function Expressiveness

### Mathematical Property: Injectivity

An aggregation function ⊕ is **injective on multisets** if:
```
⊕({x₁, x₂, ..., xₙ}) = ⊕({y₁, y₂, ..., yₘ})  ⟹  {x} = {y} (as multisets)
```

**Consequence for GNNs**:
- Injective aggregation → Different node neighborhoods → Different node embeddings
- Non-injective aggregation → May confuse different neighborhoods

### Empirical Ranking

1. **Sum Aggregation** (Injective)
```
⊕_sum({x₁, x₂, ..., xₙ}) = Σ xᵢ
```
✓ Injective (linear combinations are unique)
✓ Most expressive (equivalent to WL-test)
✓ Efficient (O(n) time)
✗ Can saturate with large neighborhoods (numerical issues)

2. **Mean/Max Aggregation** (Non-injective)
```
⊕_mean({x₁, x₂, ..., xₙ}) = (1/n) Σ xᵢ
⊕_max({x₁, x₂, ..., xₙ}) = max_i xᵢ
```
✓ Efficient (O(n) time)
✓ Stable (bounded output range)
✗ Lose injectivity information
✗ Lose information about neighborhood size (mean loses scale, max loses diversity)

3. **Attention Aggregation** (Injective if weights are learned)
```
⊕_attn({x₁, x₂, ..., xₙ}) = Σ αᵢ·xᵢ   where αᵢ ∈ (0,1), Σ αᵢ = 1
```
✓ Injective (learned weights can be diverse)
✓ Adaptive (learns neighbor importance)
✗ High computational cost O(n²F')
✗ Attention patterns may be noisy/unstable

---

## Benchmark Performance Summary

### Citation Networks (Transductive, Semi-Supervised)

**Dataset: Cora (2,708 nodes, 5,429 edges)**

| Method | Accuracy | Year | Notes |
|--------|----------|------|-------|
| GCN baseline | 81.5% | 2017 | Original GCN paper |
| GAT | 83.3% | 2018 | Multi-head attention |
| AAGCN | 83.3% | 2024 | Adaptive architecture |
| Theoretical max* | ~85% | - | Citation network saturation |

*Citation networks are relatively simple; 85-90% appears to be practical ceiling

**Dataset: CiteSeer (3,327 nodes, 4,732 edges)**

| Method | Accuracy | Year | Notes |
|--------|----------|------|-------|
| GCN baseline | 70.3% | 2017 | Original GCN paper |
| GAT | 72.5% | 2018 | Better on this dataset |
| NTK-GCN | 74.0% ± 1.5% | 2023 | Neural tangent kernel approach |

**Dataset: PubMed (19,717 nodes, 44,338 edges)**

| Method | Accuracy | Year | Notes |
|--------|----------|------|-------|
| GCN baseline | 79.0% | 2017 | Original GCN paper |
| GAT | 79.0% | 2018 | Matches GCN |
| AAGCN | 80.4% | 2024 | Adaptive learning |
| NTK-GCN | 88.8% ± 0.5% | 2023 | Highest reported |

### Graph Classification Benchmarks (GIN, Xu et al. 2019)

| Dataset | GIN | Baseline | Size |
|---------|-----|----------|------|
| PROTEINS | 74.2% | 71.0% | Bioinformatics |
| MUTAG | 89.4% | 85.6% | Molecular graphs |
| COLLAB | 80.2% | 73.8% | Collaboration networks |
| REDDIT-BINARY | 92.5% | 85.4% | Social networks |
| NCI1 | 82.6% | 78.4% | Chemical compounds |

**Key Insight**: GIN consistently outperforms baselines, validating theoretical expressiveness advantage

### Large-Scale Benchmarks (Open Graph Benchmark)

| Dataset | Size | GCN | GraphSAGE | GAT | Task |
|---------|------|-----|-----------|-----|------|
| ogbn-arxiv | 169K nodes | 71.7% | - | ~73% | Node classification |
| ogbn-products | 2.45M nodes | - | 82.5% | ~80% | Node classification |
| ogbn-papers100M | 111M nodes | - | ~70%* | - | Node classification |

*Requires sampling; full-batch training infeasible in memory/time

---

## Key Lessons from Literature

### 1. Depth is Not Always Better
```
Empirical Pattern (most datasets):
  - Increasing depth from 1 to 2 layers: +5-10% accuracy
  - Increasing depth from 2 to 3 layers: ±0-2% (marginal)
  - Increasing depth from 3 to 4+ layers: -2-5% (degradation)

Implication: Use 2 layers as default, with skip connections if deeper
```

### 2. Sampling Preserves Accuracy
```
Mini-batch with neighbor sampling (S=10-25):
  - Accuracy retention: 95-98% of full-batch
  - Training time reduction: 100-1000×
  - Memory reduction: 100-1000×

Critical for production systems on large graphs
```

### 3. Citation Networks Saturate
```
Benchmark progression:
  Cora:  81.5% (2017) → 83.3% (2024)  = +1.8% in 7 years
  CiteSeer: 70.3% (2017) → 72.5% (2024) = +2.2% in 7 years
  PubMed: 79.0% (2017) → 80.4% (2024) = +1.4% in 7 years

Problem: Citation networks only ~1K training labels, <20K total nodes
Solution: Move to large-scale benchmarks (OGB) for meaningful progress
```

### 4. Architecture Choice Matters Less Than Depth/Sampling
```
Accuracy spread at optimal depth:
  GCN vs GAT vs GraphSAGE: ±2-3%

Accuracy spread from depth choices:
  L=2 vs L=3: ±0-5%
  L=2 vs L=4: ±5-10%

Implications:
  - Pick simplest/most efficient architecture (GCN)
  - Tune depth carefully (2 layers typical)
  - Focus effort on sampling strategy for large graphs
```

### 5. Homophily Assumption is Critical
```
Graph Type          | Standard GNN | With Attention | Over-smoothing
Homophilic         | 80-85%      | 82-87%        | Moderate
Heterophilic       | 60-70%      | 70-80%        | Severe

Adaptive aggregation (GAT) crucial for non-homophilic graphs
```

---

## Practical Recommendations

### For Small Graphs (<10K nodes, <100K edges)
1. **Architecture**: GCN (simplest) or GAT (if heterophilic)
2. **Depth**: 2 layers (can afford to try 3)
3. **Batch**: Full-batch training feasible
4. **Feature engineering**: Important; good features >10% improvement

### For Medium Graphs (10K-100K nodes)
1. **Architecture**: GraphSAGE with sampling
2. **Sampling**: S=10-15 per layer
3. **Depth**: 2 layers (over-smoothing manageable)
4. **Batch size**: 1K-5K nodes per mini-batch

### For Large Graphs (>1M nodes)
1. **Architecture**: GraphSAGE or simplified GCN variants
2. **Sampling**: S=5-10 per layer (critical)
3. **Mini-batch training**: Essential
4. **Depth**: Typically 2 layers (deeper leads to neighborhood explosion)

### Hyperparameter Defaults
```
Learning rate:      0.001 - 0.01
Dropout:           0.5
Weight decay L2:    0.0001 - 0.001
Hidden dim:        64 - 256
Optimization:      Adam (preferred) or SGD
Scheduler:         ReduceLROnPlateau (validate-based)
Early stopping:    10-20 epochs patience
```

---

## Open Research Questions

1. **Theoretical**: Formal characterization of when depth > 2 helps despite over-smoothing
2. **Practical**: Scalable attention mechanisms for dense graphs
3. **Representation**: How to incorporate global graph properties (diameter, clustering coefficient)
4. **Transfer**: Pre-training objectives for graphs (self-supervised learning)
5. **Robustness**: Adversarial attacks and distribution shift for GNNs
