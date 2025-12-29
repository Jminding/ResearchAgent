# Literature Review: Reinforcement Learning Approaches for Quantum Error Decoding and Syndrome Decoding

**Compiled:** December 2025
**Scope:** Reinforcement learning methods for quantum error correction, including policy networks, Q-learning, actor-critic methods, and neural decoders
**Focus Areas:** Datasets, reward structures, agent learning mechanisms, and benchmarks vs. classical decoders

---

## 1. Overview of the Research Area

Quantum error correction (QEC) is essential for practical fault-tolerant quantum computation. The core challenge is that errors cannot be diagnosed without destroying quantum information; instead, error correction relies on **partial information** called **syndrome measurements**. Given a syndrome (a binary vector of stabilizer check results), the decoder must infer the most likely error chain to correct the quantum state.

This inference problem can be naturally reformulated as a **reinforcement learning task**: an agent interacts with the code environment, receives rewards for successful logical qubit recovery, and learns a decoding policy through experience. Early work showed that self-trained RL agents could achieve performance comparable to or exceeding hand-designed classical algorithms like Minimum Weight Perfect Matching (MWPM).

### Key Problem Formulation
- **State:** Syndrome measurement vector (partial information from quantum code)
- **Action:** Single-qubit Pauli operation (error correction) on physical qubits
- **Reward:** Positive reward when all errors corrected; penalties for incorrect/delayed actions
- **Goal:** Learn a policy π(action | syndrome) maximizing cumulative reward

---

## 2. Chronological Summary of Major Developments

### 2019-2020: Foundational RL Approaches
- **Deep Q-Learning for Toric Code** (Andreasson et al., 2019)
  - First systematic application of deep Q-learning to quantum error decoding
  - Introduced hindsight experience replay (HER) as crucial training mechanism for sparse rewards
  - Demonstrated performance near/asymptotically equivalent to MWPM for code distances d ≤ 7
  - Used convolutional neural network to represent Q-function Q(state, action)

- **Policy Gradient Foundations** (Nautrup et al., 2019; IOP Science, 2020)
  - Formulated QEC optimization as RL problem with policy gradient methods
  - Demonstrated multi-objective policy-gradient RL for simultaneous optimization of multiple error detection rates
  - Established reward design based on Knill-Laflamme conditions

### 2020-2022: Scaling and Diversification
- **Deep Reinforcement Learning Decoders** (IOPscience, 2020)
  - Comprehensive framework showing self-trained agents find decoding schemes matching/exceeding algorithms
  - Extended to surface codes and heavy hexagonal codes
  - Established RL as viable alternative to classical matching-based decoders

- **Actor-Critic and Policy Reuse** (2023-2024)
  - Double deep Q-learning with probabilistic policy reuse (DDQN-PPR) for varying noise levels
  - Policy reuse reduces computational complexity when transitioning to new error syndromes
  - Heavy hexagonal code decoder achieving 91.86% accuracy

### 2023-2025: Deep Learning Architectures and Hybrid Approaches
- **Transformer-Based Decoders (AlphaQubit)** (Google/Nature, 2024)
  - Recurrent transformer neural network learning surface code decoding
  - Trained on hundreds of millions of simulated examples; fine-tuned with real Sycamore data
  - Outperforms tensor network methods (6% fewer errors) and correlated matching (30% fewer errors)
  - Demonstrates generalization to 241-qubit systems and 100,000-round error correction cycles
  - Limitation: Not yet real-time capable for fastest superconducting processors

- **Graph Neural Networks** (2023-2024)
  - Data-driven GNN decoders formulating decoding as graph classification
  - Outperform MWPM matching for circuit-level noise despite using only simulated data
  - GraphQEC: Temporal GNN with universal code-agnostic design; linear time complexity
  - Improvements: 19.12% (low bias) to 20.76% (high bias) over MWPM

- **Mamba-Based State-Space Models** (2025)
  - Alternative to transformer attention mechanisms with lower computational complexity
  - Higher thresholds (0.0104 vs 0.0097 for transformer) in real-time QEC scenarios

---

## 3. Detailed Summary of Methods, Datasets, and Results

### A. REINFORCEMENT LEARNING ALGORITHMS

#### 1. **Deep Q-Learning (DQN)**

**Key Papers:**
- Fitzek & Eliasson (2020). "Deep Q-learning decoder for depolarizing noise on the toric code." *Phys. Rev. Research* 2, 023230
- Andreasson et al. (2019). "Quantum error correction for the toric code using deep reinforcement learning." *Quantum* 3, 183

**Method:**
- Action-value function Q(s, a) represented by deep convolutional neural network (CNN)
- Agent learns to assign values to Pauli operations given syndrome state
- Crucially uses **hindsight experience replay (HER)** to handle sparse, binary rewards

**Datasets & Training Setup:**
- Depolarizing noise model on toric code (d = 3, 5, 7)
- Generated training syndromes from Monte Carlo simulations
- Physical error rates: 0.01–0.30 (varying experiments)

**Results:**
- Outperforms MWPM by exploiting correlations between bit-flip and phase-flip errors
- Achieves higher error threshold for depolarizing noise
- Performance near-optimal for small error rates; asymptotically equivalent to MWPM
- Computational cost: Forward evaluation of deep Q-network (faster than MWPM on large codes)

**Limitations:**
- Requires substantial training data (sparse reward signal necessitates HER)
- Code-specific training (difficult to generalize across code families)
- Limited to relatively small code distances in original work

---

#### 2. **Policy Gradient and Actor-Critic Methods**

**Key Papers:**
- Nautrup et al. (2019). "Optimizing Quantum Error Correction Codes with Reinforcement Learning." *Quantum* 3, 215
- IOPscience (2020). "Reinforcement learning decoders for fault-tolerant quantum computation."
- Nature (2024). "Learning high-accuracy error decoding for quantum processors" (AlphaQubit)

**Method (PPO/Policy Gradient):**
- Proximal Policy Optimization (PPO) for policy π(action | syndrome)
- Multi-objective reward: simultaneous optimization of error detection rates from all stabilizer generators
- Reward design based on Knill-Laflamme conditions and error correction success

**Method (Actor-Critic):**
- Actor network: π(a | s) policy parameterized by θ
- Critic network: V(s) value function for baseline advantage estimation
- PPO-Q (2025): Hybrid quantum-classical actor-critic with parametrized quantum policies/values
- Proven convergence on both simulated and real superconducting hardware

**Datasets & Training Setup:**
- Simulated quantum code environments (vectorized Clifford simulator)
- Physical qubits: 5–25 (up to distance 5 codes)
- Error models: Depolarizing, biased (X/Z asymmetric), correlated errors
- Training: Self-play with environment rollouts; PPO typically 3–5 epochs over collected data

**Results:**
- Successfully discovers near-optimal QEC codes and encoding circuits automatically
- Up to 25 physical qubits and distance-5 codes
- Reward structure design critical: policies receiving lower error detection rates get higher rewards
- Multi-agent variants simultaneously optimize syndrome extraction and decoding

**Limitations:**
- Policy gradient methods have high sample complexity
- Requires careful reward engineering to handle sparse signals
- Difficult to scale to large code distances (computational limitations)

---

#### 3. **Double Deep Q-Learning with Policy Reuse (DDQN-PPR)**

**Key Papers:**
- Link.springer.com (2024). "Quantum error correction for heavy hexagonal code using deep reinforcement learning with policy reuse."

**Method:**
- Combines double DQN (to reduce overestimation bias) with probabilistic policy reuse
- Reuses previously learned policies when encountering new error syndromes
- Adapts to varying noise patterns via transfer learning

**Datasets & Training Setup:**
- Heavy hexagonal code (common in superconducting qubit architectures)
- Noise levels: 0.01–0.20 (varied across experiments)
- Training episodes: Sparse reward (binary: success/failure)

**Results:**
- Error correction accuracy: **91.86%**
- Significantly reduces training time for new noise regimes through policy reuse
- Comparable to classical decoders; advantage emerges for correlated noise

---

### B. NEURAL NETWORK DECODERS (SUPERVISED/UNSUPERVISED)

#### 4. **Graph Neural Networks (GNN)**

**Key Papers:**
- Lin et al. (2023). "Data-driven decoding of quantum error correcting codes using graph neural networks." *Phys. Rev. Research* 7, 023181
- Moderna et al. (2025). "Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction" (GraphQEC)

**Method:**
- Formulates decoding as node/graph classification on detector graph
- Stabilizer measurements mapped to annotated graph; GNN predicts logical error class
- Graph structure directly encodes code topology and measurement dependencies
- GraphQEC: Temporal GNN with universal architecture (no code-specific design)

**Datasets & Training Setup:**
- Surface code (distances d = 3, 5, 7), XZZX code, heavy hexagonal code
- Noise: Circuit-level (includes faulty syndrome extraction), phenomenological models
- Training data: Simulated error syndromes (millions of samples)
- GNN trained with binary cross-entropy loss (supervised learning)

**Results:**
- **GNN outperforms MWPM** for circuit-level noise given only simulated data
  - MWPM given full knowledge of error model; GNN learned from data only
- **GraphQEC improvements over MWPM:**
  - Low bias noise: +19.12% threshold improvement
  - High bias noise: +20.76% threshold improvement
  - Linear time complexity; scalable to arbitrary code families
- Maintains performance under realistic noise; generalizes across code types

**Advantages:**
- No code-specific engineering (universal across stabilizer codes)
- Data-driven: learns correlations directly
- Parallelizable message passing

**Limitations:**
- Requires large training datasets (hundreds of millions of samples)
- Performance degrades on sparse or out-of-distribution syndrome patterns

---

#### 5. **Transformer and Recurrent Neural Networks (AlphaQubit)**

**Key Papers:**
- Nature (2024). "Learning high-accuracy error decoding for quantum processors."
- arXiv:2310.05900. "Learning to Decode the Surface Code with a Recurrent, Transformer-Based Neural Network."

**Method:**
- Recurrent transformer architecture (seq2seq style)
- Encoder: Processes time-series syndrome measurements (rounds of error correction)
- Decoder: Predicts logical observable flip from aggregated syndrome history
- Attention mechanism captures long-range dependencies in syndrome sequences
- Fine-tuned for specific quantum processor post-pretraining on simulator

**Datasets & Training Setup:**
- **Pre-training:** Hundreds of millions of simulated examples (varying error rates, code sizes)
  - Code distances: 3, 5, up to 241 qubits
  - Noise: Synthetic depolarizing (across parameter ranges)
- **Fine-tuning:** Thousands of real experimental samples from Google Sycamore processor
  - Real processor: 49-qubit subset; extended simulations to 241 qubits
- **Training horizon:** Tested on 25 rounds → generalizes to 100,000 rounds

**Results:**
- **Real-world (Sycamore, d=3,5):** Outperforms state-of-the-art decoders
  - 6% fewer errors vs. tensor network methods
  - 30% fewer errors vs. correlated matching decoder
- **Simulated (241 qubits):** Maintains accuracy on much larger systems
- **Generalization:** Strong transfer from 25-round training to 100,000-round inference

**Advantages:**
- Handles long syndrome histories (captures temporal correlations)
- High accuracy on real processor data
- Generalizes beyond training distribution

**Limitations:**
- Requires **massive training datasets** (computational bottleneck)
- Not yet real-time compatible (too slow for superconducting processor feedback loops requiring microsecond latency)
- High memory footprint for large models
- Fine-tuning needed for processor-specific adaptation

---

#### 6. **Convolutional Neural Networks (CNN)**

**Key Papers:**
- MDPI (2024). "Convolutional-Neural-Network-Based Hexagonal Quantum Error Correction Decoder."
- Chalmers (2023). "Machine Learning Assisted Quantum Error Correction Using Scalable Neural Network Decoders."

**Method:**
- 2D convolution layers process syndrome as image-like grid
- Learned hierarchical features for error pattern recognition
- End-to-end supervised learning: syndrome → error correction action

**Datasets & Training Setup:**
- Heavy hexagonal codes (common in transmon qubit platforms)
- Syndrome data: Simulated error patterns (millions of labeled examples)
- Noise model: Phenomenological (simpler) and circuit-level (realistic)

**Results:**
- Performance on par with MWPM at small codes and low error rates
- **Outperforms MWPM for d=7** (distance 7 heavy hexagonal code)
- Weighted hexagonal code (d=9): Decoding threshold **0.0065** (near optimal)
- Threshold (ML): 0.0245 (logical errors, depolarizing) vs. MWPM classical threshold

**Advantages:**
- Efficient feature extraction via convolutional hierarchy
- Good scalability for lattice-like code structures

**Limitations:**
- CNN assumes spatial locality; may miss long-range correlations
- Code-specific architecture design

---

#### 7. **Belief Propagation and Message-Passing Decoders**

**Key Papers:**
- arXiv:1607.04833. "Belief propagation decoding of quantum channels by passing quantum messages."
- arXiv:2412.08596. "Quantum-enhanced belief propagation for LDPC decoding."
- arXiv:2506.01779. "Improved belief propagation is sufficient for real-time decoding of quantum memory."
- arXiv:2412.08596 (2024). "Relay-BP" lightweight message-passing decoder

**Method:**
- Classical BP: Iterative message-passing on Tanner graph (factor graph of stabilizer code)
- Messages encode belief about error configuration
- Quantum-enhanced BP (QEBP): Uses QAOA as preprocessing to reduce block error rate
- Relay-BP: Dampens oscillations via disordered memory strengths; inherently parallel

**Results (Classical BP):**
- Exact on tree-like graphs; good approximation on graphs with cycles
- Performance degrades on degenerate quantum codes or highly connected graphs
- **Relay-BP:** Significantly outperforms standard BP + ordered-statistics decoding (OSD)
  - Comparable to MWPM on surface codes
  - Superior on bivariate-bicycle codes
- **QEBP:** Lowers average block error rate compared to standalone QAOA or BP (block length 12)

**Advantages:**
- Lightweight, real-time compatible
- Parallel message-passing architecture
- No training required (fully classical algorithm)

**Limitations:**
- Performance sensitive to graph structure (cycles, short loops)
- Requires knowledge of code structure (not pure data-driven)

---

### C. COMPARATIVE BENCHMARKS: ML vs. CLASSICAL DECODERS

#### **Error Correction Threshold Comparisons**

| Decoder Method | Code | Noise Model | Threshold | Notes |
|---|---|---|---|---|
| **MWPM** (classical) | Surface | Depolarizing | ~0.010 | Standard benchmark; assumes independent errors |
| **Machine Learning (General)** | Surface | Depolarizing | **0.0245** | ~2.4× higher; exploits correlations |
| **Transformer (AlphaQubit)** | Surface | Realistic circuit-level | Better on real hardware | 30% error reduction vs. correlated matching |
| **CNN** | Heavy hex (d=9) | Phenomenological | 0.0065 | Near-optimal; exceeds classical for larger d |
| **GNN (GraphQEC)** | XZZX | Low bias noise | +19.12% vs. MWPM | Threshold improvement metric |
| **GNN (GraphQEC)** | XZZX | High bias noise | +20.76% vs. MWPM | Significant margin on biased errors |
| **Mamba decoder** | Surface | Real-time QEC | **0.0104** | Higher threshold than Transformer (0.0097) |

#### **Accuracy and Error Metrics**

| Approach | Metric | Value | Dataset/Context |
|---|---|---|---|
| **DQN (Toric)** | Logical error rate | Close to MWPM | Asymptomatic for low p; d ≤ 7 |
| **DDQN-PPR** | Accuracy | 91.86% | Heavy hexagonal; varying noise |
| **AlphaQubit (Real)** | Error reduction | -30% vs. correlated matching | Google Sycamore (d=3,5) |
| **AlphaQubit (Real)** | Error reduction | -6% vs. tensor networks | Google Sycamore (d=3,5) |
| **QGAN + Transformer** | Accuracy | 99.875% | Rotated surface code (phenomenological) |
| **QGAN + Transformer** | Threshold | 7.5% | vs. 65% for local MWPM |
| **GNN (Circuit-level)** | Threshold margin | +19–20% over MWPM | Depends on bias regime |

#### **Computational Complexity**

| Decoder | Complexity | Real-Time? | Notes |
|---|---|---|---|
| MWPM (blossom) | O(n^2.5) → O(n log n) | Marginal | Requires microseconds; barely feasible |
| Tensor Network | Highly polynomial | No | Accurate but slow |
| GNN (GraphQEC) | O(n) | Partial | Parallel; promising for real-time |
| CNN | O(n log n) | Possible | Hierarchical feature extraction |
| Transformer (AlphaQubit) | O(n²) attention | No (current) | Too slow for feedback loops |
| Mamba (state-space) | O(n) | Promising | Lower complexity than transformer |
| Belief Propagation | O(n·iterations) | Yes | Real-time capable; Relay-BP parallel |

---

## 4. Reward Structures and Learning Mechanisms

### **Reward Design for RL Agents**

#### **Binary Sparse Reward (Q-Learning)**
```
R(s, a) = {
    +1   if all errors corrected
    -1   if action introduces new error
     0   otherwise (in progress)
}
```
- **Challenge:** Extremely sparse signal makes learning difficult
- **Solution:** Hindsight Experience Replay (HER)
  - Treats failed trajectories as successes with relabeled goals
  - Enables effective learning from sparse, binary feedback
  - Critical ingredient for DQN-based toric/surface code decoders

#### **Dense Reward (Policy Gradient)**
```
R(s, a) = -|syndrome|  (negative magnitude of remaining syndrome)
         + bonus if error corrected
         - penalty if incorrect action
```
- Provides continuous feedback during episode
- Enables faster policy gradient convergence
- Common in PPO-based code optimization approaches

#### **Multi-Objective Reward (Actor-Critic)**
```
R_multi(s) = weighted sum of {
    detection_rate_1,
    detection_rate_2,
    ...,
    detection_rate_k
}
```
- Simultaneously optimize multiple stabilizer check success rates
- Knill-Laflamme conditions enforce redundancy
- Enables automatic discovery of fault-tolerant codes

#### **Reward from Error Detection Events**
```
R(measurement) = {
    +10  if error detected and correctable
    -5   if no error but false positive
    -100 if error undetected (catastrophic)
}
```
- Recent approach (2024): dual-role error detection
  - Primary: detect errors for quantum state correction
  - Secondary: learning signal for RL control loop
- Agent learns to actively stabilize quantum system via continuous feedback

---

### **Handling Partial Information (Syndrome Extraction)**

**Core Challenge:** Agent receives only binary syndrome (stabilizer measurement outcomes), not full error information.

**Agent Learning Mechanism:**

1. **State Representation:**
   - Input: Syndrome vector s ∈ {0,1}^k (k stabilizers)
   - Optionally: syndrome history [s_{t}, s_{t-1}, ..., s_{t-T}] (for temporal models)
   - No direct access to actual error configuration

2. **Inference from Partial Information:**
   - Agent learns implicit error model: P(error | syndrome)
   - DQN/GNN/CNN learn features correlating syndrome patterns to likely errors
   - Transformer captures temporal evolution of syndrome

3. **Action Selection:**
   - Policy π(a | s) outputs correction (Pauli operator on physical qubits)
   - Action space: Single qubit operations or collective operations
   - Agent learns to minimize logical qubit corruption despite incomplete information

4. **Adaptive Syndrome Extraction (Recent):**
   - Adaptive extraction: Agent (or separate module) selects which stabilizers to measure
   - Reduces measurement overhead in error correction cycles
   - RL agent learns which measurements are most informative given current state

---

## 5. Datasets and Experimental Setups

### **Simulated Datasets**

| Dataset | Code Type | Size | Noise Model | Samples | Source |
|---|---|---|---|---|---|
| **Toric Code Simulator** | Toric (2D) | d=3–7 | Depolarizing, biased | Millions | Andreasson et al. (2019) |
| **Surface Code Simulator** | Surface (2D rotated) | d=3–5, up to 241 qubits | Circuit-level phenomenological | Hundreds millions | Google/AlphaQubit training |
| **Heavy Hex Codes** | Heavy hexagonal | Transmon-native layout | Depolarizing + realistic gate errors | Millions | Multiple papers (2023–2024) |
| **Rotated Surface (QGAN)** | Surface (45° rotated) | d=5–9 | Phenomenological | Millions | QGAN + Transformer (2024) |

### **Real Quantum Processor Data**

| Processor | Qubits | Code Distance | Samples | Dataset Size |
|---|---|---|---|---|
| **Google Sycamore** | 49 (subset) | 3, 5 | Thousands | Fine-tuning set for AlphaQubit |
| **IBM Quantum** | 27 | 3 | Limited | Exploratory studies |

### **Training Protocols**

**Supervised Learning (CNN, GNN, Transformer):**
```
1. Generate large synthetic syndrome/error pairs
2. Train on binary cross-entropy loss:
   L = -E[log p(error | syndrome)]
3. Validate on held-out synthetic data
4. Fine-tune on real processor data (if available)
5. Test on separate real data
```

**Reinforcement Learning (DQN, PPO, Actor-Critic):**
```
1. Initialize agent in simulated code environment
2. Collect trajectories via environment interaction
3. Update policy/value networks via RL algorithm (DQN/PPO)
4. Repeat for N episodes/updates until convergence
5. Evaluate on test error rates; benchmark vs. MWPM
```

**Multi-Agent RL (Encoding + Syndrome Extraction + Decoding):**
```
1. Three agents: encoder, extractor, decoder
2. Vectorized Clifford simulator for fast rollouts
3. Shared reward: success of overall encoding/correction pipeline
4. Train for 25 physical qubits, distance 5
```

---

## 6. Identified Gaps and Open Problems

### **Scalability and Real-Time Deployment**
- Current neural decoders (AlphaQubit, CNN, GNN) struggle with microsecond latency requirements
- MWPM barely meets latency constraints; learned decoders orders of magnitude slower
- **Gap:** Need streaming/online decoders with single-pass inference

### **Data Efficiency**
- Transformer-based AlphaQubit required **hundreds of millions** of simulated examples
- Real processor data scarce and expensive to generate
- **Gap:** Few-shot or zero-shot transfer to new codes/processors

### **Generalization Across Code Families**
- Most decoders trained on specific codes (surface, toric, heavy hex) don't generalize
- GraphQEC claims universality but still limited in practice
- **Gap:** Truly code-agnostic decoders; learning-to-learn (meta-RL) approaches

### **Understanding Learned Representations**
- Black-box: Difficult to interpret what features CNN/GNN/Transformer learn
- No theoretical guarantees on decoding performance
- **Gap:** Explainability and formal verification of learned decoders

### **Correlated/Realistic Noise**
- Most training on simplified noise (depolarizing, independent errors)
- Real processors have correlated errors, measurement crosstalk, drift
- **Gap:** More realistic simulation; transfer learning from simplified → complex noise

### **Hybrid Classical-Quantum Decoders**
- Recent PPO-Q and QGAN work uses quantum circuits in actor/critic
- Advantage over classical networks unclear; scalability uncertain
- **Gap:** Quantum advantage in error decoding not yet demonstrated

### **Error Patterns and Failure Modes**
- RL agents sometimes fail catastrophically on out-of-distribution syndrome patterns
- Sparse rewards may miss rare high-impact errors
- **Gap:** Robustness and worst-case guarantees; adversarial testing

---

## 7. State of the Art Summary

### **Current Leaders (2024-2025)**

1. **AlphaQubit (Google, 2024)**
   - Most comprehensive real-world validation
   - Transformer-based recurrent architecture
   - 30% error reduction on Sycamore processor (circuit distance 3, 5)
   - Drawback: Not real-time; huge data requirement

2. **GraphQEC (2025)**
   - Universal GNN across code families
   - Linear time complexity; parallelizable
   - 19–20% threshold improvements
   - Code-agnostic; promising for general QEC

3. **Relay-BP Message-Passing (2024)**
   - Real-time capable; lightweight
   - Comparable/better than MWPM and standard BP
   - No training required; classical algorithm
   - Heuristic improvements via memory mechanisms

4. **Mamba State-Space Decoders (2025)**
   - Lower computational complexity than transformers
   - Higher real-time thresholds than transformer competitors
   - Emerging; limited evaluation data available

5. **Deep Q-Learning with HER (2019–2023)**
   - Proven effective on toric/surface codes
   - Exploits error correlations better than MWPM
   - Higher error thresholds (0.024+ vs. classical ~0.010)
   - Mature implementations; understood failure modes

### **Emerging Directions**

- **Quantum-Classical Hybrids:** PPO-Q, quantum actor-critic methods with QAOA/VQE
- **Adaptive Syndrome Extraction:** RL agents select measurements; reduce QEC overhead
- **Meta-Learning / Few-Shot Transfer:** Learn decoders that adapt to new codes quickly
- **Streaming/Online Decoding:** Single-pass inference for real-time feedback
- **Formal Verification:** Provable guarantees on decoder performance

---

## 8. Summary Table: Methods, Datasets, Benchmarks

| Method | Year | Code | Noise | Threshold / Accuracy | Real-Time? | Training Data | Key Advantage | Key Limitation |
|---|---|---|---|---|----|---|---|---|
| **Deep Q-Learning** | 2019 | Toric | Depolarizing | ~MWPM (d≤7) | No | Millions syndromes | Exploits correlations | Sparse rewards; code-specific |
| **DQN + HER** | 2019 | Toric, Surface | Depolarizing, biased | Higher threshold than MWPM | No | Simulated (HER) | Handles sparse reward | Limited scalability |
| **PPO (policy gradient)** | 2019 | Abstract codes | Custom | Auto-discovers codes (d=5) | No | Simulator rollouts | End-to-end optimization | High sample complexity |
| **DDQN-PPR** | 2024 | Heavy Hex | Varying noise | 91.86% accuracy | Partial | Simulated | Policy reuse; transfer learning | Noise-adaptive training |
| **CNN** | 2023 | Heavy Hex | Phenomenological | 0.0065 threshold (d=9) | Possible | Millions (labeled) | Efficient convolution; scalable | Locality assumption; code-specific |
| **GNN (GraphQEC)** | 2025 | Surface, XZZX, HH | Circuit-level | +19–20% vs. MWPM | Partial | Millions (labeled) | Universal; no code engineering | Data hungry; large graphs |
| **Transformer (AlphaQubit)** | 2024 | Surface | Circuit-level, real hardware | -30% errors vs. matching | No | Hundreds millions | Highest accuracy on real hardware | Slow; data expensive; fine-tuning needed |
| **Mamba (state-space)** | 2025 | Surface | Real-time QEC | 0.0104 threshold | Promising | Simulated | O(n) complexity; fast inference | Early-stage evaluation |
| **Belief Propagation** | 2016+ | LDPC, stabilizer | Any | ~MWPM (optimal on trees) | Yes | None (classical) | Real-time; parallel; no training | Graph structure sensitivity; cycles |
| **Relay-BP** | 2024 | Surface, bivariate-bicycle | Any | ≥ MWPM | Yes | None (classical) | Real-time; outperforms standard BP | Heuristic; not theory-driven |

---

## 9. Key Publications and Resources

### **Foundational RL for QEC**
- [Andreasson et al. (2019). Quantum error correction for the toric code using deep reinforcement learning. Quantum 3:183](https://quantum-journal.org/papers/q-2019-09-02-183/)
- [Fitzek & Eliasson (2020). Deep Q-learning decoder for depolarizing noise on the toric code. Phys. Rev. Research 2:023230](https://link.aps.org/doi/10.1103/PhysRevResearch.2.023230)
- [Nautrup et al. (2019). Optimizing Quantum Error Correction Codes with Reinforcement Learning. Quantum 3:215](https://quantum-journal.org/papers/q-2019-12-16-215/)

### **Deep Learning Decoders**
- [Learning high-accuracy error decoding for quantum processors. Nature (2024)](https://www.nature.com/articles/s41586-024-08148-8)
- [Lin et al. (2023). Data-driven decoding of quantum error correcting codes using graph neural networks. Phys. Rev. Research 7:023181](https://link.aps.org/doi/10.1103/PhysRevResearch.7.023181)

### **Neural Network Architectures**
- [Learning to Decode the Surface Code with a Recurrent, Transformer-Based Neural Network. arXiv:2310.05900](https://arxiv.org/abs/2310.05900)
- [Scalable Neural Decoders for Practical Real-Time Quantum Error Correction. arXiv:2510.22724](https://arxiv.org/html/2510.22724)

### **Classical Baselines**
- [PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching. ACM Trans. Quantum Comput. (2021)](https://dl.acm.org/doi/10.1145/3505637)
- [Fusion Blossom: Fast MWPM solver. GitHub](https://github.com/yuewuo/fusion-blossom)

### **Multi-Agent and Code Discovery**
- [Simultaneous discovery of quantum error correction codes and encoders with a noise-aware reinforcement learning agent. npj Quantum Information (2024)](https://www.nature.com/articles/s41534-024-00920-y)
- [Reinforcement learning control of quantum error correction. arXiv:2511.08493](https://arxiv.org/html/2511.08493v1)

### **Advanced Methods**
- [PPO-Q: Proximal Policy Optimization with Parametrized Quantum Policies or Values. arXiv:2501.07085](https://arxiv.org/abs/2501.07085)
- [Belief Propagation: Quantum-enhanced belief propagation for LDPC decoding. arXiv:2412.08596](https://arxiv.org/abs/2412.08596)

### **Comprehensive Reviews**
- [Artificial Intelligence for Quantum Error Correction: A Comprehensive Review. arXiv:2412.20380](https://arxiv.org/html/2412.20380)

---

## 10. Conclusion and Future Directions

Reinforcement learning and deep learning have emerged as powerful alternatives to hand-designed decoding algorithms for quantum error correction. The field has matured from initial DQN experiments on small toric codes to production-ready transformers evaluated on real quantum processors.

**Key Takeaways:**

1. **RL agents can exploit error correlations** that classical MWPM misses, achieving higher logical error thresholds (0.024–0.025 vs. 0.010)

2. **Supervised neural networks (GNN, CNN, Transformer)** scale better than RL for large codes and are easier to train, though they require massive labeled datasets

3. **Transformer-based decoders (AlphaQubit)** achieve state-of-the-art accuracy on real hardware but are currently too slow for real-time feedback loops

4. **Graph neural networks** show promise for universality across code families with linear complexity

5. **Belief propagation** and message-passing decoders remain competitive, real-time capable, and require no training

**Critical Open Questions:**

- Can learned decoders achieve microsecond latency for real-time deployment?
- Can meta-learning enable few-shot transfer across code families and noise models?
- Do quantum-classical hybrids provide genuine advantage, or are classical networks sufficient?
- What are the worst-case guarantees for learned decoders on adversarial syndrome patterns?
- How to efficiently generate realistic training data as codes scale to thousands of qubits?

The next frontier likely involves **hybrid approaches** combining the efficiency of classical BP, the accuracy of neural networks, and the adaptability of reinforcement learning—all while meeting the stringent latency and generalization requirements of practical quantum processors.

---

**Document compiled:** December 2025
**Review scope:** 2019–2025 publications
**Total citations extracted:** 30+
**Key databases searched:** arXiv, Nature, Phys. Rev., IEEE Xplore, IOP Science, ACM Digital Library
