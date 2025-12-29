# Index: Complete Literature Review on RL for Quantum Error Decoding

**Review Compilation Date:** December 2025
**Total Documents:** 4
**Total Citations:** 26+ peer-reviewed papers, preprints
**Time Span:** 2016–2025 (emphasis on 2019–2025)

---

## Document Structure and Navigation

### Main Literature Review
**File:** `lit_review_rl_quantum_error_decoding.md`

**Contents:**
1. **Overview of Research Area** - Problem formulation, why QEC is hard, why RL fits
2. **Chronological Summary** - Major developments from 2019–2025
3. **Detailed Method Summary** - DQN, policy gradient, actor-critic, neural decoders
4. **Reward Structures** - Binary, dense, multi-objective, Knill-Laflamme
5. **Datasets and Experimental Setups** - Simulated vs. real processor data
6. **Identified Gaps** - Open problems and future directions
7. **State of the Art Summary** - Current leaders (AlphaQubit, GraphQEC, Relay-BP)
8. **Summary Table** - Methods vs. Datasets vs. Results (comparative benchmarks)

**Best for:** Initial orientation, understanding broad landscape, state-of-the-art status

---

### Detailed References and Extraction
**File:** `rl_qec_detailed_references.md`

**Contents:**
- **26 papers with full extraction:** Authors, year, venue, DOI/URL
- **Problem statement** for each paper
- **Methodology** and approach details
- **Dataset characteristics** and experimental setup
- **Quantitative results** - Thresholds, accuracies, error rates
- **Stated limitations** for each approach
- **Summary tables** - Thresholds, accuracies, complexity

**Sections:**
1. Deep Q-Learning Foundations (Papers 1–2)
2. Policy Gradient and Code Optimization (Papers 3–6)
3. Graph Neural Networks (Papers 9–10)
4. Convolutional Neural Networks (Paper 11)
5. Belief Propagation and Message-Passing (Papers 12–15)
6. Scalable Neural Decoders (Paper 16)
7. Hybrid Quantum-Classical (Papers 17–18)
8. Classical Baselines: PyMatching (Papers 24–25)
9. Comprehensive Reviews (Paper 26)

**Best for:** Detailed extraction, quantitative benchmarks, finding specific papers, comprehensive reference list

---

### Technical Details: Datasets, Rewards, Training
**File:** `rl_qec_technical_details.md`

**Contents:**
1. **Datasets and Simulation Environments**
   - Toric code, surface code, heavy hexagonal, XZZX, LDPC
   - Dataset sizes and characteristics
   - Real quantum processor data (Google Sycamore, IBM)
   - Dataset generation procedures (supervised and RL)

2. **Reward Structures and Design**
   - Sparse binary reward with HER
   - Dense syndrome magnitude reward
   - Multi-objective reward (code discovery)
   - Reward from error detection events
   - Knill-Laflamme conditions as reward

3. **Training Protocols and Hyperparameters**
   - DQN training (toric code example)
   - PPO training (code discovery)
   - Supervised learning: Transformer pre-training and fine-tuning
   - GNN training with graph construction

4. **Inference and Deployment**
   - Speed comparison across decoders
   - Real-time latency requirements
   - Edge device deployment challenges

5. **Validation and Benchmarking**
   - Logical error rate (LER) computation
   - Threshold calculation methodology
   - Classical baseline comparisons

6. **Reproducibility and Open-Source Resources**
   - Simulation frameworks (Stim, Cirq, QuTiP, etc.)
   - Public decoder implementations
   - Repository links

7. **Common Pitfalls and Lessons Learned**
   - Training challenges (sparse rewards, overfitting, noise mismatch)
   - Evaluation mistakes to avoid

**Best for:** Implementing decoders, understanding hyperparameter choices, training details, reproducibility

---

## Quick Reference: Key Findings

### Benchmark Results (Error Correction Thresholds)

| Decoder | Code | Threshold | Improvement |
|---------|------|-----------|-------------|
| **MWPM** (classical) | Surface | 0.010 | Baseline |
| **ML (general)** | Surface | **0.0245** | **2.4× higher** |
| **CNN** | Heavy Hex (d=9) | 0.0065 | Near-optimal |
| **GNN (GraphQEC)** | XZZX | MWPM +19–20% | Better on biased noise |
| **AlphaQubit** | Surface (real hardware) | -30% errors vs. matching | Best on Sycamore |
| **Mamba** | Surface (real-time) | **0.0104** | Higher than transformer |
| **QGAN+Transformer** | Rotated surface | 7.5% vs. 65% MWPM | Major improvement |

### Computational Complexity

| Decoder | Complexity | Real-Time? |
|---------|-----------|-----------|
| MWPM | O(n^2.5) to O(n log n) | Marginal |
| CNN | O(n log n) | Possible |
| GNN | O(n) | Partial |
| Transformer | O(n²) attention | No |
| Mamba | O(n) | **Promising** |
| Belief Propagation | O(n·iterations) | **Yes** |

### Dataset Requirements

| Approach | Training Data Scale | Real Data Needed? |
|----------|---|---|
| **DQN (Toric)** | ~1M syndromes | No |
| **PPO (Code discovery)** | ~10M trajectories | No |
| **CNN** | ~2M labeled pairs | No |
| **GNN** | ~5M labeled pairs | No |
| **Transformer (AlphaQubit)** | **Hundreds of millions** | Thousands (fine-tune) |
| **Classical BP** | 0 (algorithm) | N/A |

---

## Recommended Reading Order

### For Practitioners (Implementing a Decoder)
1. `rl_qec_technical_details.md` - Training protocols, hyperparameters
2. `lit_review_rl_quantum_error_decoding.md` - Method overview
3. `rl_qec_detailed_references.md` - Find specific papers for your code family

### For Researchers (Literature Review)
1. `lit_review_rl_quantum_error_decoding.md` - Full landscape, gaps, open problems
2. `rl_qec_detailed_references.md` - 26 detailed paper extractions
3. `rl_qec_technical_details.md` - Implementation insights

### For Quick Lookup
1. **Summary tables** in `lit_review_rl_quantum_error_decoding.md` (Section 8)
2. **Benchmark tables** in `rl_qec_detailed_references.md` (Final section)
3. **INDEX file** (this document) for navigation

---

## Key Topics by Location

### Reinforcement Learning Algorithms
- **DQN with HER:** Main review (Section 3A), References (Papers 1–2), Technical (Section 3.1)
- **Policy Gradient (PPO):** Main review (Section 3A.2), References (Paper 3), Technical (Section 3.2)
- **Actor-Critic:** References (Papers 5–6, 17)
- **Policy Reuse:** References (Paper 5)

### Neural Network Decoders
- **Graph Neural Networks:** Main review (Section 3B.4), References (Papers 9–10), Technical (Section 3.4)
- **Transformers (AlphaQubit):** Main review (Section 3B.5), References (Papers 7–8), Technical (Section 3.3)
- **CNNs:** Main review (Section 3B.6), References (Paper 11)
- **Message-Passing:** Main review (Section 3B.7), References (Papers 12–15)

### Code Families
- **Toric Code:** References (Papers 1–2, 4)
- **Surface Code:** References (Papers 3, 7–10), Main review (throughout)
- **Heavy Hexagonal:** References (Papers 5, 11), Main review (Section 3B.6)
- **XZZX, LDPC:** References (Papers 9–10, 14)

### Datasets
- **Simulated:** Main review (Section 5), Technical (Section 1)
- **Real Processor (Sycamore):** References (Paper 7), Technical (Section 1.3)
- **Dataset Generation:** Technical (Section 1.4)

### Reward Structures
- **Sparse Binary:** Main review (Section 4), Technical (Section 2.1)
- **Dense Penalty:** Technical (Section 2.2)
- **Multi-Objective:** Technical (Section 2.3)
- **HER Details:** Technical (Section 2.1)

### Benchmarks and Comparisons
- **Threshold Comparisons:** Main review (Section 3C), References (Section 8)
- **Accuracy Metrics:** Main review (Section 3C), References (Section 8)
- **Computational Complexity:** Main review (Section 3C), Technical (Section 4.1)

### Training and Implementation
- **Hyperparameters:** Technical (Section 3, all algorithms)
- **Training Curves:** Technical (Section 3.1)
- **Inference Speed:** Technical (Section 4.1)
- **Validation Methods:** Technical (Section 5)
- **Open-Source Resources:** Technical (Section 6)

---

## Open Problems and Future Directions

**Listed in:** Main review, Section 6

### Critical Challenges (2025)
1. **Real-time latency:** DecoderGate too slow for microsecond requirements
2. **Data efficiency:** AlphaQubit needs hundreds of millions of training examples
3. **Generalization:** Most decoders trained on specific codes don't transfer
4. **Quantum advantage:** Quantum-classical hybrid decoders not yet proven beneficial
5. **Robustness:** Failure on out-of-distribution syndrome patterns

### Emerging Research Directions
1. Streaming/online decoders
2. Meta-learning for few-shot code adaptation
3. Formal verification of learned decoders
4. Quantum advantage in hybrid circuits
5. Adaptive syndrome extraction via RL

---

## Quick Stats

- **Total Papers Reviewed:** 26
- **RL-Specific Papers:** 10
- **Neural Network Decoder Papers:** 8
- **Classical Baseline Papers:** 3
- **Hybrid/Review Papers:** 5
- **Time Span:** 2016–2025 (9 years)
- **Primary Venues:** Nature, Physical Review, arXiv, Quantum, IEEE, ACM, IOP
- **Geographic Distribution:** US (Google, MIT, Yale), Europe (Switzerland, Sweden), Asia
- **Open-Source Implementations:** PyMatching, Fusion Blossom, GNN repos, DQN decoders

---

## Citation Statistics by Method

| Method | Papers | Years | Status |
|--------|--------|-------|--------|
| **Deep Q-Learning** | 3 | 2019–2020 | Established |
| **Policy Gradient (PPO)** | 4 | 2019–2025 | Active |
| **GNN Decoders** | 3 | 2023–2025 | Emerging |
| **Transformer (AlphaQubit)** | 2 | 2023–2024 | Mature |
| **CNN** | 2 | 2023–2024 | Mature |
| **Belief Propagation** | 4 | 2016–2025 | Classical baseline |
| **Hybrid Quantum-Classical** | 2 | 2024–2025 | Early-stage |
| **Code Discovery** | 2 | 2019–2024 | Active |

---

## How to Use This Review in Your Paper

### Section 2: Related Work
- **Main source:** `lit_review_rl_quantum_error_decoding.md` (Section 2–4)
- **Direct quotes:** Available for all 26 papers in `rl_qec_detailed_references.md`
- **Chronological narrative:** Use Section 2 of main review

### Section 3: Methods and Datasets
- **Own decoder:** Describe based on `rl_qec_detailed_references.md` extraction template
- **Comparison:** Use benchmark tables from main review and references
- **Hyperparameter justification:** Reference `rl_qec_technical_details.md`

### Section 4: Experiments and Results
- **Baseline selection:** Justify using benchmark comparisons (main review Section 3C)
- **Metrics:** Use LER definition from `rl_qec_technical_details.md` (Section 5)
- **Statistical testing:** Consult papers cited for significance test methodology

### Section 5: Discussion and Conclusion
- **Open problems:** Use Section 6 of main review
- **Future work:** Emerging directions in main review Section 7
- **Position in landscape:** Use state-of-the-art summary (main review Section 7)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial compilation; 26 papers; 3 supporting documents |

---

## Contact and Contribution

This review synthesizes 2019–2025 literature on RL and neural network approaches to quantum error correction. It is organized for direct incorporation into research papers and serves as a comprehensive reference for both practitioners and researchers.

**Document Organization:**
- Main review: Narrative + tables + gap analysis
- References: Detailed extraction (problem, method, results, limitations)
- Technical details: Implementation, datasets, training, validation
- This index: Navigation and quick reference

**For updates or corrections:** Refer to original venues and papers cited; this review reflects state as of December 2025.

---

**End of Index**

*All documents are citation-ready for academic use. Quantitative results, thresholds, and benchmark comparisons are extracted directly from original publications.*
