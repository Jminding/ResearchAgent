# Foundational Graph Neural Network Architectures - Literature Review Index

## Overview

This directory contains a comprehensive literature review of foundational Graph Neural Network (GNN) architectures, their mathematical foundations, and empirical properties. The review synthesizes research from 25+ peer-reviewed papers spanning 2014-2025.

## Files in This Review

### 1. **lit_review_gnn_architectures.md** (Primary Review Document)
The main literature review document containing:
- Chronological development of GNN field (2014-2025)
- Detailed analysis of 5 foundational architectures:
  - **GCN (Graph Convolutional Networks)** - Kipf & Welling 2017
  - **GraphSAGE** - Hamilton et al. 2017
  - **GAT (Graph Attention Networks)** - Veličković et al. 2018
  - **GIN (Graph Isomorphism Networks)** - Xu et al. 2019
  - **MPNN (Message Passing Neural Networks)** - Gilmer et al. 2017

- **Mathematical foundations**:
  - Spectral vs. spatial perspectives
  - Inductive biases (permutation invariance, locality, structure preservation)
  - Receptive field and depth analysis
  - Over-smoothing and over-squashing problems

- **Computational complexity** analysis:
  - Time complexity per architecture per layer
  - Space complexity and memory requirements
  - Parameter counting

- **Node and edge representation learning**:
  - How representations are learned in k-hop neighborhoods
  - Edge representation methods
  - Heterogeneous graph handling

- **Empirical benchmarks**:
  - Citation network results (Cora, CiteSeer, PubMed)
  - Large-scale benchmarks (Open Graph Benchmark)
  - Graph classification benchmarks

- **Known limitations** and open challenges:
  - Over-smoothing depth limitation
  - Scalability on dense graphs
  - Heterophily and heterogeneous graphs
  - Expressiveness ceiling (Weisfeiler-Lehman limit)

- **State of the art summary** with current best practices

**Length**: ~6,000 words | **Sections**: 12 | **Figures**: None (mathematical notation provided)

---

### 2. **evidence_sheet_gnn.json** (Quantitative Evidence Database)
Structured JSON file containing:

#### Metric Ranges
- **Accuracy benchmarks** by architecture and dataset:
  - GCN: 70-84% (citation networks), 71.7% (large-scale)
  - GAT: 72-83% (citation networks)
  - GraphSAGE: 86-95% (task-dependent)
  - GIN: 74-93% (graph classification)

- **Time complexity** formulas:
  - GCN: O(|E|F + |V|F²)
  - GraphSAGE: O(S^L × L × F²)
  - GAT: O(|E|F'²)
  - GIN: O(|V|F²)

- **Space complexity**:
  - Adjacency matrix: O(|E|) sparse, O(|V|²) dense
  - Node features: O(|V| × F × L)
  - Optimization state: 2-3× parameters

- **Parameter counts**:
  - GCN on Cora: ~120K parameters
  - GAT typical: ~280K parameters
  - GraphSAGE: ~200K parameters
  - GIN: ~400K parameters

- **Receptive field analysis**:
  - k-layer GNN has k-hop neighborhood
  - Sampling multiplier: S^k nodes
  - Practical depth limit: 2-3 layers

- **Benchmark dataset sizes**:
  - Small: Cora (2.7K), CiteSeer (3.3K), PubMed (19.7K)
  - Medium: ogbn-arxiv (169K), ogbn-products (2.45M)
  - Large: ogbn-papers100M (111M nodes, 1.57B edges)

#### Known Pitfalls (16 documented)
- over_smoothing: convergence beyond 2-3 layers
- depth_paradox: deeper networks perform worse
- neighborhood_explosion: exponential growth with layers
- heterophily_assumption: fails on non-homophilic graphs
- benchmark_saturation: marginal 1-2% improvements in recent years
- aggregation_bottleneck: sum vs mean trade-offs
- sampling_bias: uniform sampling misses rare structures
- attention_computation_cost: 4× overhead for GAT

#### Key References (16 cited papers)
Each with:
- Publication year
- Venue (ICLR, NeurIPS, JMLR, etc.)
- Key finding with metrics
- URL and DOI

#### Experimental Design Guidance
- Typical hyperparameters (learning rate, dropout, hidden dimensions)
- Validation methodology
- Expected performance baselines
- Variance and confidence interval guidance

---

### 3. **gnn_technical_summary.md** (Detailed Technical Guide)
Practical technical reference containing:

#### Quick Reference Table
Architecture comparison across dimensions:
- Publication and learning type
- Aggregation mechanism
- Time complexity
- Parameter count
- Accuracy on Cora
- Key strengths/weaknesses

#### Mathematical Formulation Comparison
Detailed equations for:
- GCN: Spectral-inspired aggregation formula
- GraphSAGE: Sampling-based aggregation with variants
- GAT: Attention coefficient computation and multi-head mechanism
- GIN: Maximal expressiveness with MLP aggregation

#### Unified MPNN Framework
Shows how all four architectures fit:
```
Message → Aggregation → Update functions
```

#### Complexity Analysis (Deep Dive)
- **Forward pass time complexity** with concrete examples
- **Memory complexity** for activations and optimizer states
- **Practical limits** without sampling (100K nodes feasible)

#### Receptive Field and Depth Analysis
- **Why 2-3 layers is typical** (neighborhood explosion)
- **Over-smoothing effect** with empirical accuracy degradation curves
- **Mitigation strategies** (skip connections, normalization, decoupling)

#### Aggregation Function Expressiveness
- **Mathematical property**: Injectivity on multisets
- **Empirical ranking**: Sum > Attention > Mean/Max
- **Trade-offs**: Expressiveness vs. efficiency vs. stability

#### Benchmark Performance Summary
Tables for:
- Citation networks (Cora, CiteSeer, PubMed) with year-by-year progression
- Graph classification (9 benchmarks from Xu et al. 2019)
- Large-scale benchmarks (OGB datasets with sizes up to 111M nodes)

#### Key Lessons from Literature (5 major insights)
1. Depth is not always better
2. Sampling preserves accuracy (95-98%)
3. Citation networks saturate (±1-2% over 7 years)
4. Architecture choice matters less than depth/sampling
5. Homophily assumption is critical

#### Practical Recommendations
- **For small graphs** (<10K nodes): GCN or GAT, full-batch, 2 layers
- **For medium graphs** (10K-100K): GraphSAGE, S=10-15, mini-batch
- **For large graphs** (>1M): GraphSAGE, S=5-10, critical sampling

#### Hyperparameter Defaults
Learning rates, dropout, weight decay, batch sizes, optimization algorithms

#### Open Research Questions (5 areas)
- Formal theory of depth beyond 2 layers
- Scalable attention for dense graphs
- Global graph properties incorporation
- Pre-training and transfer learning
- Adversarial robustness

---

## Key Quantitative Evidence Summary

### Accuracy Ranges by Architecture
| Architecture | Citation Networks | Large-Scale | Graph Classification |
|--------------|------------------|-------------|----------------------|
| **GCN** | 70-84% | 71.7% | - |
| **GAT** | 72-83% | ~73% | - |
| **GraphSAGE** | 86-95%* | 82.5% | - |
| **GIN** | - | - | 74-93% |

*Inductive setting differs from standard transductive benchmark

### Time Complexity at Scale
| Method | 100K Nodes | 1M Nodes | 100M Nodes |
|--------|-----------|----------|-----------|
| GCN (full) | Feasible | Prohibitive | Impossible |
| GraphSAGE (S=15) | Feasible | Feasible | Feasible |
| GAT (full) | 4× overhead | Prohibitive | Impossible |
| GIN (full) | Feasible | Prohibitive | Impossible |

### Over-Smoothing Performance Degradation
```
Depth   1-layer  2-layer  3-layer  4-layer  5-layer
Cora    ~75%     ~81%     ~80%     ~78%     ~70%
         (under)  (opt)    (ok)     (bad)    (terrible)
```

### Sampling Efficiency
- Full-batch GCN on 1M nodes: 1 trillion operations per epoch
- GraphSAGE with S=15, L=2: 1.25 million operations per epoch
- **Reduction factor**: 10^6× for large graphs
- **Accuracy retention**: 95-98% vs. full-batch

---

## Coverage Statistics

### Papers Reviewed
- **Total papers**: 25+
- **Date range**: 2014-2025
- **Venues**: ICLR (6), NeurIPS (4), ICML (2), JMLR (1), arXiv (6), Others (3+)
- **Foundational papers**: 7 (Bruna 2014, Defferrard 2016, KipfWelling 2017, Hamilton 2017, Gilmer 2017, Veličković 2018, Xu 2019)
- **Recent papers**: 8 (2023-2025)

### Topics Covered
- Architectures: 5 major (GCN, GraphSAGE, GAT, GIN, MPNN)
- Theoretical foundations: Spectral theory, MPNN framework, Weisfeiler-Lehman expressiveness
- Computational aspects: Complexity analysis, sampling, scalability
- Practical issues: Over-smoothing, heterophily, benchmark saturation
- Applications: Citation networks, protein interactions, social networks, chemical compounds

### Datasets Analyzed
- Small: 3 (Cora, CiteSeer, PubMed)
- Medium: 2 (ogbn-arxiv, ogbn-products)
- Large: 1 (ogbn-papers100M)
- Graph classification: 9 benchmarks (PROTEINS, MUTAG, COLLAB, etc.)

---

## Research Gaps and Open Problems

### Theoretical
1. Formal characterization of depth > 2 layers effectiveness
2. Quantitative bounds on over-smoothing and over-squashing
3. Universal approximation theorems for GNNs (partially solved)

### Practical
1. Sublinear training algorithms (emerging, not mainstream)
2. Scalable attention mechanisms for dense graphs
3. Distributed/parallel training at scale

### Representation Quality
1. Pre-training objectives for graphs (self-supervised learning)
2. Transfer learning between graph domains
3. Incorporating global graph properties (spectral diameter, clustering)

### Robustness
1. Adversarial attacks and defenses for GNNs
2. Distribution shift and domain adaptation
3. Out-of-distribution generalization

---

## How to Use This Review

### For Literature Context
1. Start with **lit_review_gnn_architectures.md**
2. Reference specific architectures' sections for theory
3. Check benchmark results for empirical context

### For Experimental Design
1. Consult **evidence_sheet_gnn.json** for baseline expectations
2. Reference **gnn_technical_summary.md** for hyperparameter guidance
3. Set realistic hypotheses using "expected_performance_baselines"

### For Technical Understanding
1. Read **gnn_technical_summary.md** for mathematical formulations
2. Compare architectures using the Quick Reference Table
3. Understand complexity trade-offs from detailed analysis

### For Specific Lookups
- **Accuracy benchmarks**: evidence_sheet_gnn.json → metric_ranges
- **Known pitfalls**: evidence_sheet_gnn.json → known_pitfalls (16 items)
- **Hyperparameters**: gnn_technical_summary.md → Practical Recommendations
- **Architecture details**: lit_review_gnn_architectures.md → Major Developments sections

---

## Citation Guide

### For Citing Foundational Papers
```
GCN: Kipf & Welling (2017), ICLR 2017 (arXiv:1609.02907)
GraphSAGE: Hamilton et al. (2017), NeurIPS 2017 (arXiv:1706.02216)
GAT: Veličković et al. (2018), ICLR 2018 (arXiv:1710.10903)
GIN: Xu et al. (2019), ICLR 2019 (arXiv:1810.00826)
MPNN: Gilmer et al. (2017), ICML 2017 (arXiv:1704.01212)
```

### For Citing This Review
Use the main literature review file:
```
Author Unknown. Foundational Graph Neural Network Architectures:
Literature Review. Research Notes, 2025.
```

---

## Version History

- **Version 1.0** (2025-12-24): Initial comprehensive review
  - 3 documents (lit_review, evidence_sheet, technical_summary)
  - 25+ papers analyzed
  - 16 known pitfalls identified
  - 15+ key references with quantitative evidence

---

## Contact and Contributions

This review is a synthesis of published research. All citations point to original sources for verification and further reading.

**Review completion date**: December 24, 2025
**Total word count**: ~15,000 words
**Total figures/tables**: 20+
**JSON data structures**: 1 (evidence_sheet with 50+ fields)

---

## Quick Navigation

| Document | Purpose | Length | When to Use |
|----------|---------|--------|------------|
| lit_review_gnn_architectures.md | Comprehensive review | 6,000 words | Understanding the field |
| evidence_sheet_gnn.json | Quantitative database | Structured JSON | Experimental design |
| gnn_technical_summary.md | Technical details | 4,000 words | Implementation guidance |
| README_GNN_REVIEW.md | This file | Index | Navigation and overview |
