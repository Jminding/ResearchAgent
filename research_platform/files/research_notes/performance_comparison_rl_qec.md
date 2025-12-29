# Performance Comparison: RL-Based vs Classical Quantum Error Correction Decoders

## Executive Summary

This document compares performance metrics across hybrid RL + neural network approaches versus classical decoding methods for quantum error correction. Data compiled from 12+ peer-reviewed sources (2019-2024).

---

## 1. Error Suppression Rates

### AlphaQubit v2 (Google DeepMind, 2024) - Current SOTA

**Method:** Transformer-based recurrent neural network with hybrid supervised + RL adaptation

| Comparison | Error Reduction | Hardware | Code Distance | Notes |
|-----------|-----------------|----------|----------------|-------|
| vs. Correlated Matching | 30% lower | Google Sycamore | d=3-11 | Primary classical baseline |
| vs. Tensor Network | 6% lower | Google Sycamore | d=3-11 | Highest accuracy but slow |
| Suppression by rejection | 20× factor | Google Sycamore | d=11 | Rejects only 0.2% of experiments |
| Logical error rate | Near optimal | Google Sycamore | d=11 | Approaches theoretical limits |

**Quantitative:** By rejecting 0.2% of 25-round experiments at distance 11, AlphaQubit reduces error rate by factor of ~20.

### GNN-Based Temporal Decoders (2023-2024)

| Decoder Type | Logical Error Reduction | Threshold Improvement | Code Type | Source |
|--------------|------------------------|----------------------|-----------|--------|
| Temporal GNN (GraphQEC) | 94.6% | N/A | Surface, Toric, QLDPC | Bny et al. 2023 |
| Generic GNN decoder | 25% vs MWPM | 19.12% higher | Multiple | Leuzzi et al. 2023 |
| GNN-based (low-bias noise) | Variable | 19.12% | Surface codes | Multiple sources |

**Validation:** GNN results tested on real Google Quantum AI experimental data; 25% reduction verified against MWPM.

### Deep Q-Learning Decoders (2019-2020)

| Noise Model | Code Type | Distance | vs. MWPM | Notes |
|-------------|-----------|----------|----------|-------|
| Uncorrelated | Toric | d ≤ 7 | Asymptotically equivalent | Andreasson et al. 2019 |
| Depolarizing | Toric | d ≤ 9 | Outperforms | Fosel et al. 2020 |
| Uncorrelated | Toric | d=3,5,7 | Near-optimal | Self-trained, few hours |

**Key Finding:** Deep Q-learning **exploits error correlations** that MWPM assumes are independent, yielding superior performance under realistic noise.

---

## 2. Decoding Latency / Wall-Clock Time

### Real-Time Performance (2024)

| Decoder | Latency | Distance | Platform | Hardware | Notes |
|---------|---------|----------|----------|----------|-------|
| AlphaQubit v2 | <1 μs/cycle | d=11 | Commercial AI accelerators | GPU/TPU | Real-time, production-ready |
| FPGA-integrated | <1 μs/round | d=3-5 | Superconducting processor | FPGA | Directly integrated with qubits |
| Scalable ANN | Independent of d | d>1000 | Theory | CPU/GPU | Theoretical result |

### Latency Impact on Quantum Algorithms

From Sundaresan et al. (2024):

| Decoding Latency | Algorithm | Runtime | Impact |
|-----------------|-----------|---------|--------|
| 10 μs | RSA-2048 factorization | ~8 hours | Baseline |
| 100 μs | RSA-2048 factorization | ~48 hours | 6× slowdown |

**Critical Requirement:** Latency must be <1 μs to prevent syndrome data backlog and maintain fast logical clock rates.

### Why This Matters

- **Physical Qubits Decohere:** Each microsecond of decoding time allows additional errors to accumulate
- **Syndrome Backlog:** If decoder is slower than error detection cycle, queue builds exponentially
- **RL Advantage:** Neural decoders avoid expensive matching computations (O(n³) complexity)

---

## 3. Computational Resources for Training

### GPU Requirements by Approach

| Method | Network Size | Code Distance | GPUs | Duration | Reference |
|--------|--------------|----------------|------|----------|-----------|
| NVIDIA Neural Decoder | Large | d=3 | 42 × H100 | 1 hour | NVIDIA/QuEra 2024 |
| Scalable ANN | Very Large | d>1000 | Multiple (10+) | Weeks | Varsamopoulos 2021 |
| Deep Q-Learning | Moderate | d≤9 | Standard HW | Few hours | Andreasson 2019 |
| RL-Enhanced Greedy | Small | d≤11 | Standard HW | Hours-days | Xiang et al. 2024 |

### Training Data Requirements

| Approach | Error Instances | Code Distance | Notes |
|----------|-----------------|----------------|-------|
| Scalable ANN | 50M-100M | d>1000 | Synthetic, supervised learning |
| Deep Q-Learning | 10K-100K | d≤9 | RL-based, self-trained |
| AlphaQubit | Synthetic (d=3-5) + limited real | d=11 | Hybrid: offline synthetic + real adaptation |
| RL-Enhanced Greedy | Variable | d≤11 | Leverages existing greedy decoder |

**Cost Scaling:** Training cost grows **exponentially with code distance** for synthetic sampling approaches due to syndrome space explosion.

---

## 4. Generalization and Robustness

### AlphaQubit Generalization (Google, 2024)

| Training Scenario | Test Scenario | Performance | Notes |
|------------------|--------------|-------------|-------|
| 25-round experiments | 100,000-round experiments | Maintained accuracy | 4× multiplication in sequence length |
| Synthetic Sycamore data | Real Sycamore data | Good transfer | Limited real data budget used |

**Implication:** Generalization beyond training data is stronger for neural decoders than traditional methods.

### Adversarial Robustness (Critical Vulnerability)

From Schaffner et al. (2024) and Arnon et al. (2024):

| Decoder Type | Attack Type | Impact | Severity |
|--------------|------------|--------|----------|
| DeepQ (basic RL) | Adversarial syndrome modification | 5 OOM reduction in logical qubit lifetime | Critical |
| GNN decoder (undefended) | Minimal syndrome changes | Misclassification | High |
| GNN decoder (adversarially trained) | Same adversarial attacks | Significantly improved robustness | Mitigated |

**Mitigation:** Iterative adversarial training loop (RL-based vulnerability discovery + retraining) enhances robustness substantially.

---

## 5. Code Distance Scalability

### Demonstrated Code Distances

| Decoder Type | Max Distance | Qubits | Performance | Notes |
|--------------|--------------|--------|-------------|-------|
| Scalable ANN | >1000 | 4M+ | Maintained latency | Largest ML demonstration |
| AlphaQubit | 11 | 241 | Real-time <1 μs | Tested on hardware |
| Deep Q-Learning | 9 | ~100 | Near-optimal | Synthetic data |
| RL-Enhanced Greedy | 11+ | Scalable | Hybrid efficiency | Low computational cost |

**Scalability Challenge:** Training time and memory grow exponentially; practical limit currently d≤11 for real-time implementations.

---

## 6. Threshold Performance (Error Rates)

### Threshold Definitions

Error correction codes work only below a critical physical error rate (threshold). Above this, errors compound exponentially.

### RL vs Classical Thresholds

| Decoder | Code Type | Threshold | vs. Theory | Notes |
|---------|-----------|-----------|-----------|-------|
| MWPM (classical) | Surface | ~0.01 | Near-optimal | Baseline |
| GNN | Surface | ~0.01 (19% higher) | Near-optimal | Leuzzi et al. 2023 |
| Deep Q-Learning | Toric | Code-distance dependent | 0.0058 (d=3) | Andreasson et al. 2019 |

**Result:** RL and GNN decoders achieve comparable or superior thresholds without hand-tuned error probabilities.

---

## 7. Comparative Table: All Methods

| Criterion | MWPM | Tensor Network | Deep Q-Learning | GNN | AlphaQubit | RL+Greedy |
|-----------|------|-----------------|-----------------|-----|-----------|-----------|
| **Latency** | Slow (O(n³)) | Very slow | <1 μs (learned) | <1 μs | <1 μs | ~1 μs |
| **Error Rate** | Baseline | -6% vs ALPHA | Comparable/better | -25% | -30% | Near-optimal |
| **Threshold** | ~0.01 | Similar | Adaptive | +19% | Near-optimal | Similar |
| **Training Cost** | None | High (classical) | Hours | Medium | 42 H100 × 1h (d=3) | Low-medium |
| **Generalization** | Fixed | Fixed | Limited | Limited | Good (4×) | Depends on baseline |
| **Adversarial Robust** | N/A | Untested | Vulnerable (5 OOM) | Improvable | Untested | Likely similar |
| **Scalability** | d < 20 | d < 15 | d ≤ 9 | d ≤ 15+ | d ≤ 11 | d ≤ 20+ |
| **Hardware Integration** | Off-chip | Off-chip | GPU/FPGA | GPU/FPGA | GPU/FPGA | Flexible |

**Legend:**
- MWPM: Minimum Weight Perfect Matching (classical baseline)
- Tensor Network: Classical numerical method (high accuracy, slow)
- Deep Q-Learning: Basic RL approach (fast, vulnerable)
- GNN: Graph Neural Networks (scalable, robust-improves with training)
- AlphaQubit: Google's transformer-based (SOTA, production)
- RL+Greedy: Hybrid approach (efficiency, near-optimal)

---

## 8. Critical Findings

### 1. Error Suppression Rates (Winner: AlphaQubit)

**Result:** 30% improvement over best classical method (correlated matching).

**Achievable via:**
- Transformer attention mechanisms modeling long-range syndrome correlations
- Hybrid training combining synthetic data + real hardware adaptation
- Selective rejection of high-uncertainty cases (0.2% overhead for 20× suppression)

### 2. Real-Time Feasibility (Winner: Neural Decoders)

**Result:** <1 microsecond decoding compatible with qubit error correction cycles.

**Reason:**
- Avoids expensive matching graph computation
- Direct feed-forward neural computation highly parallelizable
- Proven on commercial AI accelerators

### 3. Adversarial Vulnerability (Critical Issue)

**Result:** Undefended RL decoders vulnerable to adversarial syndrome attacks (5 OOM qubit lifetime reduction).

**Mitigation:**
- Mandatory adversarial training loop during deployment
- RL-based automated vulnerability discovery proven effective
- Production systems must include robustness verification

### 4. Training Cost vs. Performance Tradeoff

**High-Resource Approaches:**
- Supervised ANN (50M+ instances): Best raw performance, no adaptivity
- Full RL training (42 H100 × 1h for d=3): Production-quality for specific code

**Low-Resource Approaches:**
- RL-enhanced greedy (hours, standard HW): Near-optimal, low cost, hybrid benefits
- Deep Q-Learning (few hours): Foundational RL result, but limited scalability

### 5. Generalization (Advantage: Neural Decoders)

**Result:** AlphaQubit generalizes 4× beyond training data (trained on 25 rounds, succeeds on 100k rounds).

**Classical methods:** Fixed algorithms, no generalization capability.

---

## 9. Practical Recommendations

### For High-Performance Requirements

**Choose: AlphaQubit-style (transformer + hybrid training)**
- Best error suppression (30% improvement)
- Real-time latency (<1 μs)
- Hardware-validated approach
- Cost: 42 H100 GPUs × 1 hour minimum per training

### For Efficient, Low-Cost Deployment

**Choose: RL-Enhanced Greedy Hybrid**
- Near-optimal performance
- Low computational cost
- Works with existing infrastructure
- Flexible hardware requirements

### For Code Exploration / New Codes

**Choose: GNN-Based Decoder**
- Topology-agnostic
- Scalable to new code geometries
- 25% error improvement verified
- Moderate training cost

### For Adversarial Robustness

**Mandatory:** Integrate adversarial training loop
- Use RL agent to discover vulnerabilities
- Retrain decoder on adversarial examples
- Validate robustness before production deployment

---

## 10. Open Questions and Future Work

1. **Can RL decoders be formally verified?** Current evidence suggests no formal guarantees exist.

2. **What is the sample complexity bound?** Theoretical understanding of how many error instances are needed is open.

3. **Transfer learning across codes/distances?** Could reduce training cost dramatically if solved.

4. **Optimal reward structures?** Current RL reward design lacks standardization.

5. **Subcritical latency?** Can neural decoders achieve <100 nanoseconds for future applications?

---

## References

All citations are included in the comprehensive literature review file (`lit_review_rl_qec_hybrid.md`) and evidence sheet JSON (`evidence_sheet_qec.json`).

Key sources:
- Google DeepMind et al. (2024): AlphaQubit in Nature
- Leuzzi et al. (2023): GNN decoder performance benchmarks
- Schaffner et al. (2024): Adversarial robustness of neural decoders
- Sundaresan et al. (2024): Real-time QEC demonstration
- Xiang et al. (2024): RL-enhanced greedy hybrid approach
