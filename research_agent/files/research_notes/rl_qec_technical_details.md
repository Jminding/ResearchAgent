# Technical Details: Datasets, Reward Structures, and Training Protocols for RL Quantum Error Decoding

---

## 1. DATASETS AND SIMULATION ENVIRONMENTS

### 1.1 Quantum Code Families Studied

#### **Toric Code (2D)**
- **Topology:** Toroidal lattice with periodic boundary conditions
- **Qubits:** n = L² (linear dimension L)
- **Stabilizers:** Two types (vertex and plaquette operators)
- **Logical operators:** Two independent, separated by non-trivial loops
- **Decodable errors:** Single X and Z errors (or equivalently, error chains)
- **Common distances:** d = 3, 5, 7 in literature
- **Simulation platforms:**
  - Custom Python simulators (used in DQN papers)
  - Cirq (Google)
  - QuTiP (Python library)

**Error Channels:**
- Depolarizing: p = Pr(error) on each qubit
- Biased (XZ-asymmetric): p_X ≠ p_Z
- Correlated: Spatially or temporally correlated errors
- Circuit-level: Includes faulty syndrome measurements

---

#### **Surface Code (2D Rotated)**
- **Topology:** 2D square lattice with boundaries
- **Qubits:** n ≈ 2d² (distance d)
- **Stabilizers:** Vertex and plaquette operators (same as toric, open boundaries)
- **Logical operators:** Top-bottom and left-right pairs
- **Advantages:** Boundary conditions reduce overhead; practical for superconducting qubits
- **Common distances:** d = 3, 5, 7, and up to 241 qubits in AlphaQubit
- **Simulation platforms:**
  - Stim (Microsoft, fast circuit-level simulator)
  - Cirq
  - Custom simulators (Chalmers, Google)

**Error Models:**
- Phenomenological: Errors on qubits, faulty syndrome measurements
- Circuit-level: Explicit gate/measurement error channels
- Realistic: From actual quantum processor characterization

---

#### **Heavy Hexagonal Code**
- **Topology:** Hexagonal lattice with distinct qubit roles
- **Qubits:** Distributed on hexagonal lattice (native to transmon/superconducting architectures)
- **Stabilizers:** Hexagon-based operators
- **Advantages:**
  - Matches physical layout of superconducting qubit processors
  - Reduced connectivity requirements
- **Common distances:** d = 3, 5, 7, 9
- **Platforms:** IBM Quantum, custom transmon simulators
- **Used in:** DDQN-PPR (Ref 5), CNN decoder (Ref 11), GNN (Ref 10)

---

#### **XZZX Code**
- **Variant:** Rotated surface with X-Z asymmetry in stabilizers
- **Property:** Naturally handles biased noise (X errors more likely than Z)
- **Advantages:** Superior performance under realistic qubit biases
- **Used in:** GNN decoder studies (Ref 10)

---

#### **LDPC Codes (Low-Density Parity-Check)**
- **Structure:** Bipartite graph with low edge density
- **Application:** Quantum LDPC codes (CSS construction)
- **Advantage:** Constant encoding rate; scales to large numbers of qubits
- **Decoders:** Belief propagation, message passing
- **Used in:** QEBP (Ref 14), GNN augmentation (Ref 9)

---

### 1.2 Dataset Sizes and Characteristics

#### **Training Data Generation**

| Approach | Code | Noise | Sample Size | Generation Method |
|----------|------|-------|-------------|-------------------|
| **DQN (Toric)** | Toric (d=5) | Depolarizing | ~1M syndromes | Monte Carlo + HER |
| **PPO (Code discovery)** | Abstract (d≤5) | Depolarizing | ~10M trajectories | Vectorized simulator |
| **AlphaQubit (pre-train)** | Surface (d≤5) | Synthetic depolarizing | **Hundreds of millions** | Cirq simulator (massive scale) |
| **AlphaQubit (fine-tune)** | Surface (d=3,5) | Real Sycamore noise | **Thousands** | Google Sycamore processor |
| **GNN** | Surface (d=3–7) | Circuit-level | ~5M per code | Stim simulator |
| **CNN** | Heavy Hex (d≤9) | Phenomenological | ~2M | Custom simulator |
| **QGAN+Transformer** | Rotated surface (d=5) | Phenomenological | QGAN-generated + real | Synthetic + processor |

#### **Dataset Characteristics**

**Imbalance Issues:**
- Error syndromes follow power-law distribution (rare large-error patterns)
- Most syndromes correspond to few, small errors (majority class)
- Minority class (large error chains): Critical but rare
- **Solution used:** Class weighting, oversampling, HER (for RL)

**Temporal Structure:**
- Single syndrome ≠ unique error configuration (ambiguity)
- Time-series of syndromes (multiple rounds) resolve ambiguity
- AlphaQubit and transformer decoders leverage temporal info
- **Horizon:** Typically 10–100 error correction rounds per trajectory

**Noise Model Coverage:**
- Training on one noise model → poor transfer to another
- Requires either: (a) broad training distribution, or (b) per-processor fine-tuning
- AlphaQubit uses (a) for pre-training, (b) for refinement

---

### 1.3 Real Quantum Processor Data

#### **Google Sycamore**
- **Device:** 54-qubit superconducting quantum processor
- **Subset used:** 49 qubits in surface code configuration
- **Distances tested:** d = 3 (9 qubits), d = 5 (25 qubits)
- **Measurement:** Real syndrome data from error correction experiments
- **Sample availability:** Thousands of syndrome rounds per distance
- **Noise characteristics:**
  - Single-qubit gate error: ~0.1–0.2%
  - Two-qubit gate error: ~0.3–0.5%
  - Measurement error: ~1–2%
  - T2 dephasing: ~1–10 microseconds
  - Correlated errors from crosstalk and drift
- **Data usage (AlphaQubit):** Fine-tuning pre-trained transformer

#### **IBM Quantum**
- **Devices:** 27-qubit systems (Falcon, Hummingbird families)
- **Architecture:** Heavy hexagonal qubit connectivity
- **Limited dataset:** Fewer publicized error correction experiments
- **Error rates:** Similar scale to Sycamore (~0.1–0.5% per gate)

---

### 1.4 Dataset Generation Procedures

#### **Supervised Learning (CNN, GNN, Transformer)**

```
ALGORITHM: Generate Labeled Syndrome-Error Dataset

Input: Code (qubits, stabilizers, distance d),
       Noise model (error probabilities),
       Number of samples N

Output: Dataset D = {(syndrome_i, error_i) for i=1..N}

1. FOR each sample i:
   a. SAMPLE random error configuration from noise model
      - Each qubit: error with probability p
      - Error type: X, Y, or Z (based on noise model)

   b. COMPUTE syndrome: apply all stabilizers to system
      syndrome_i = [stabilizer_1, ..., stabilizer_k]
      (Each stabilizer evaluates to 0 or 1)

   c. APPLY error correction to derive canonical form
      - For surface code: Find minimum-weight correction
      - Store as target: error_i (binary vector or class label)

   d. ADD (syndrome_i, error_i) to dataset

2. SPLIT dataset:
   - Training: 70–80% (millions of samples)
   - Validation: 10–15%
   - Test: 10–15%

3. NORMALIZE and augment:
   - Syndrome normalization: Mean=0, Std=1 (if continuous)
   - Data augmentation: Symmetries of code lattice
     (rotations, reflections for 2D codes)
```

**Scale Challenges:**
- AlphaQubit required **300+ million** syndromes
- Each syndrome generation: ~ms on modern simulator
- Total pre-training: Days to weeks on large compute clusters
- Solution: Distributed simulation + GPU acceleration

---

#### **Reinforcement Learning (DQN, PPO, Actor-Critic)**

```
ALGORITHM: RL Environment Interaction for QEC

Input: Code, Noise model, RL algorithm (DQN/PPO),
       Episode length T, Number of episodes N_episodes

Output: Trained policy π(action | syndrome)

1. INITIALIZE environment:
   - Qubit state initialized to |0...0⟩
   - Random error applied per noise model
   - Syndrome computed

2. FOR each episode e = 1 to N_episodes:

   a. RESET environment: Sample new error configuration
      state_0 = syndrome (from random error)

   b. FOR each timestep t = 1 to T:

      i. AGENT selects action a_t ~ π(·|state_t)
         (DQN: argmax Q(state, a) + ε-exploration)
         (PPO: sample from policy)

      ii. EXECUTE action: Apply Pauli correction to qubit t
         (Single-qubit X, Y, or Z operator)

      iii. COMPUTE new syndrome after action
          syndrome_{t+1} = measure stabilizers

      iv. COMPUTE reward:
          - Sparse (RL): R = +1 if syndrome=0, else 0
          - Dense (RL): R = -||syndrome|| (magnitude penalty)
          - May use HER: Relabel unsuccessful trajectories

      v. UPDATE agent networks:
         (DQN): Q-network via Bellman equation
         (PPO): Policy via gradient, value via TD error
         (Actor-Critic): Both networks updated

      vi. state_t ← syndrome_{t+1}

3. EVALUATION:
   - Test on held-out syndromes
   - Compute logical error rate: Pr(logical flip)
   - Compare to baselines (MWPM, tensor network)
```

**Key RL-specific details:**

- **Exploration:** ε-greedy (DQN) or entropy regularization (PPO)
- **Replay buffer:** Store (state, action, reward, next_state) tuples
- **Target network** (DQN): Separate network for stable Q-value targets
- **Hindsight Experience Replay (HER):** Critical for sparse rewards
  - Episode fails (syndrome ≠ 0 at end)
  - Relabel: Pretend goal was achieved at step t
  - Reuse experience with new reward signal
- **Batching:** Multiple parallel environment rollouts
- **Vectorization:** Clifford simulator for fast syndrome computation (1000s/second)

---

## 2. REWARD STRUCTURES AND DESIGN

### 2.1 Sparse Reward (Deep Q-Learning)

#### **Binary Success/Failure**
```
R(s, a) = {
    +1      if all errors corrected (syndrome = 0)
    0       otherwise (in-progress correction)
    -1      (optional) if action worsens state
}
```

**Characteristics:**
- Extremely sparse: Only 1 positive reward per successful episode
- Episode length: ~10–100 timesteps
- Most rewards = 0 throughout episode
- Challenge: Gradient signal nearly absent

**Why Difficult:**
- Agent must explore vast action space before finding rewarding trajectory
- No guidance during intermediate steps
- Exponential sample complexity (2^d actions per syndrome)

**Solution: Hindsight Experience Replay (HER)**
```
ALGORITHM: HER for QEC

1. Collect trajectory (episode) E:
   - Initial syndrome s_0, actions [a_1, ..., a_T]
   - Final syndrome s_T (possibly nonzero if failed)

2. IF episode SUCCESSFUL (s_T = 0):
   - Add experience normally
   - Reward: R = +1 at terminal step

3. IF episode FAILED (s_T ≠ 0):
   - Relabel INTERMEDIATE goals:
     - FOR each step t in trajectory:
       - Treat s_t as if it were the "goal" state
       - Recompute rewards: R(s_t) = +1 (achieved s_t!)
       - Add (s_0, [a_1,...,a_t], +1, s_t) to replay buffer

   - Result: ~T additional successful experiences from 1 failed episode

4. TRAIN Q-network on mixed (original + relabeled) batch
```

**Impact:**
- Effective sample efficiency: 10–100× improvement
- Enables learning from binary sparse rewards
- Critical for DQN QEC decoders to converge

---

### 2.2 Dense Reward (Policy Gradient)

#### **Syndrome Magnitude Penalty**
```
R(s, a) = -ρ · ||syndrome||_1 + bonus_correction
        = -ρ · (number of violated stabilizers) + bonus
```

**Design:**
- ρ > 0: Penalty coefficient (e.g., ρ = 0.1)
- ||syndrome||_1: Number of unsatisfied checks
- bonus_correction: +10 if error corrected this step
- Each action provides immediate feedback

**Advantages:**
- Continuous reward signal guides exploration
- Agent learns to decrease syndrome magnitude incrementally
- Gradient descent more effective
- Faster convergence than sparse reward

**Trade-offs:**
- Requires more careful scaling (reward clipping common)
- Policy may converge to local optima
- Less exploration than sparse reward

---

### 2.3 Multi-Objective Reward (Code Discovery / PPO)

#### **Simultaneous Stabilizer Optimization**
```
R_multi(action) = w_1 · R_stabilizer_1(action)
                + w_2 · R_stabilizer_2(action)
                + ...
                + w_k · R_stabilizer_k(action)

where R_i(action) = {
    +5    if stabilizer i detects error this step
    +2    if stabilizer i maintained (no spurious error)
    -5    if stabilizer i failed (missed error)
}
```

**Application:** Code and encoder discovery (Ref. 18)

**Objectives:**
- Maximize error detection: All stabilizers catch errors
- Minimize false positives: No spurious measurements
- Redundancy: Multiple stabilizers check same logical region

**Weighting:**
- Equal weights: w_i = 1/k (balanced)
- Importance weights: High-weight = critical stabilizers
- Learned weights: Meta-learning approach (emerging)

---

### 2.4 Reward from Error Detection Events (Adaptive Control)

#### **Dual-Role Learning Signal**
```
Primary use: Syndrome → Error correction
Secondary use: Detection event → RL learning signal

R(measurement_event) = {
    +10     if error correctly detected
    -5      if false positive (no error but detection)
    -100    if error undetected (catastrophic)
}
```

**Implementation (Ref. 21):**
- RL agent continuously observes syndrome stream
- Each new measurement provides immediate feedback
- Agent learns to actively stabilize quantum state
- Control actions: Adjust pulse amplitudes, timings, frequencies

**Advantage:**
- Autonomous system management
- No pre-programmed controller needed
- Adapts to device drift and noise

---

### 2.5 Knill-Laflamme Conditions as Reward

#### **Code Redundancy Constraints**
```
Knill-Laflamme conditions ensure error correction:

R_KL = {
    +1    if all conditions satisfied
    -α    if any condition violated (α depends on severity)
}

Conditions (for code C):
1. ⟨ψ|E_i† E_j|ψ⟩ = λ_{ij} δ_C(i,j)
   (Error matrix elements only depend on code structure)

2. No error maps logical subspace to orthogonal subspace
   (Ensures distinguishability of logical states)

3. Redundancy: Multiple physical qubits encode each logical qubit
```

**Used in:** Code discovery via RL (Ref. 18)

**Computational Challenge:**
- Verifying KL conditions requires eigenvalue analysis
- Expensive for large codes
- Solution: Approximate via Clifford simulator checks

---

## 3. TRAINING PROTOCOLS AND HYPERPARAMETERS

### 3.1 Deep Q-Learning Training (Toric Code Example)

#### **Hyperparameters**

| Parameter | Value | Justification |
|-----------|-------|----------------|
| **Learning rate (α)** | 0.001 | Standard for neural networks |
| **Discount factor (γ)** | 0.99 | Reflects ~100-step episodes |
| **Exploration (ε)** | 0.1 → 0.01 (decay) | Start exploratory, converge to greedy |
| **Replay buffer size** | 100,000 | Sufficient for toric (small state space) |
| **Batch size** | 32 | Standard mini-batch |
| **Target network update** | Every 1000 steps | Reduce overestimation bias |
| **CNN architecture** | 2 conv (64 filters) + dense | Suitable for 2D syndrome grid |
| **Training episodes** | 10,000–100,000 | Convergence on d=5 toric |

#### **Training Curve**
```
Episode 1-1000: Random behavior, minimal learning (ε=0.1)
  - Syndrome exploration
  - Q-values initialized randomly

Episode 1000-5000: Policy emerges (ε→0.05)
  - Q-network catches error patterns
  - Success rate increases: 10% → 50%

Episode 5000-20000: Convergence (ε=0.01)
  - Fine-tuning policy
  - Success rate plateaus: ~90%

Episode 20000+: Overfitting risk
  - Monitor validation loss
  - Stop training when validation plateaus
```

---

### 3.2 Proximal Policy Optimization (Code Discovery)

#### **Hyperparameters**

| Parameter | Value | Justification |
|-----------|-------|----------------|
| **Learning rate (actor)** | 0.001 | Smaller for policy stability |
| **Learning rate (critic)** | 0.01 | Critic can learn faster |
| **Discount factor (γ)** | 0.99 | Long-horizon optimization |
| **GAE λ** | 0.95 | Generalized advantage estimation parameter |
| **Clip ratio (ε)** | 0.2 | PPO clipping for trust region |
| **Entropy coeff** | 0.01 | Encourage exploration |
| **Rollout length** | 2048 steps | Collect before each policy update |
| **Epochs per update** | 3–5 | Multiple passes over rollout data |
| **Batch size** | 256 | Stable policy gradients |
| **Max training steps** | 100K–1M | Long training for code discovery |

#### **Training Loop**
```
LOOP (num_iterations):
  1. Rollout: Collect 2048 steps from environment
     - Multiple parallel environments
     - Each step: Compute advantage using baseline V(s)

  2. Update actor:
     - Compute policy gradient with clipping
     - Multiple epochs over rollout batch
     - Maximize clipped objective

  3. Update critic:
     - Minimize TD loss: (V(s) - return)²
     - Shared with advantage estimation

  4. Log metrics:
     - Episode returns (moving average)
     - Policy entropy (should decrease as learning progresses)
     - KL divergence (trust region diagnostics)
```

---

### 3.3 Supervised Learning: Transformer (AlphaQubit)

#### **Pre-training Hyperparameters**

| Component | Value | Notes |
|-----------|-------|-------|
| **Architecture** | Transformer (recurrent) | Multi-head attention, ~millions of parameters |
| **Sequence length** | 25 error correction rounds | Input: [s_1, s_2, ..., s_25] syndromes |
| **Batch size** | 1024–4096 | GPU-friendly for pre-training |
| **Learning rate** | 0.0001 | Decay: exp(-epoch/10) |
| **Optimizer** | Adam | β₁=0.9, β₂=0.999 |
| **Loss** | Binary cross-entropy | P(logical flip \| syndromes) |
| **Regularization** | Dropout (0.1), L2 (1e-4) | Prevent overfitting on huge dataset |
| **Early stopping** | Validation loss patience=5 | Stop if no improvement |
| **Training time** | Days–weeks | On hundreds of millions samples |

#### **Fine-tuning on Real Hardware**

| Component | Value | Notes |
|-----------|-------|-------|
| **Learning rate** | 0.00001 | Much lower (processor-specific) |
| **Batch size** | 32–64 | Limited real data availability |
| **Training epochs** | 10–50 | Fewer data → shorter training |
| **Data augmentation** | Code symmetries | Multiply real dataset via symmetries |
| **Early stopping patience** | 3–5 epochs | Tight on limited data |
| **Fine-tuning time** | Hours | Quick adaptation to processor |

#### **Training Strategy**
```
Phase 1: Pre-training on simulated data
  - Generate hundreds of millions of syndrome examples
  - Train transformer for days/weeks
  - Goal: Learn general decoding features
  - Metric: Accuracy on simulated test set (~95%+)

Phase 2: Transfer + Fine-tuning
  - Load pre-trained transformer weights
  - Freeze early layers (general features)
  - Fine-tune last layers on real processor data
  - Goal: Adapt to processor-specific noise
  - Metric: Accuracy on real test set

Phase 3: Evaluation
  - Test on held-out real processor syndrome data
  - Benchmark vs. classical (tensor network, matching)
  - Generalization test: Trained on 25 rounds → test on 100,000 rounds
```

---

### 3.4 Graph Neural Network Training

#### **Hyperparameters**

| Parameter | Value | Notes |
|-----------|-------|-------|
| **GNN type** | Message-passing NN | Node classification task |
| **Hidden dimension** | 64–128 | Per-node feature vectors |
| **Message passing rounds** | 5–10 | Iterations for convergence |
| **Learning rate** | 0.001 | Standard for GNNs |
| **Batch size** | 256–1024 | Multiple graphs per batch |
| **Graph max size** | ~1000 nodes | Depends on code distance d |
| **Epochs** | 50–200 | Supervised learning (can be fast) |

#### **Graph Construction**
```
Input: Syndrome measurement vector s ∈ {0,1}^k

1. Node creation:
   - Each stabilizer check → node
   - Node features: [syndrome value, position embedding]

2. Edge creation:
   - Connect stabilizers that overlap qubits
   - Edge features: Distance, qubit overlap count

3. Graph normalization:
   - Adjacency matrix: A
   - Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
   - Used in GNN message passing

4. Output:
   - Graph structure + node features
   - Target: Error error pattern (node or edge classification)
```

**Scalability Considerations:**
- Graph size grows with d²
- d=7 → ~200 nodes, ~1000 edges
- d=15 → ~800 nodes, ~4000 edges
- Message passing: O(|V| + |E|) per round → O(n) for sparse lattice codes

---

## 4. INFERENCE AND DEPLOYMENT

### 4.1 Inference Speed Comparison

| Decoder | Inference Time | Real-Time Capable? | Notes |
|---------|---|---|---|
| **MWPM (PyMatching v2)** | ~1–10 ms | Marginal | Barely meets microsecond latency (recent optimization) |
| **CNN** | ~1 ms | Possible | Parallel convolutions; GPU-friendly |
| **GNN (GraphQEC)** | ~5–10 ms | Partial | Message passing sequential; depends on graph size |
| **Transformer (AlphaQubit)** | ~100 ms | No | Attention O(n²); too slow for feedback |
| **Mamba** | ~10–50 ms | Promising | Linear complexity; O(n) fast |
| **Belief Propagation** | ~10–100 ms | Yes (often) | Classical algorithm; parallelizable |
| **Relay-BP** | ~5–20 ms | Yes | Lightweight; inherently parallel |

**Real-time requirement:** Superconducting qubits cycle at ~1 microsecond (10⁶ Hz)
- Syndrome extraction: 1–10 rounds per correction cycle
- Decoding must complete in < 1 microsecond (ideally < 100 ns)
- Current neural networks: ~10–1000× too slow

---

### 4.2 Inference on Edge Devices

**Emerging challenge:** Quantum processors cannot connect to large classical computers (latency)

**Solutions in development:**
1. **FPGA acceleration:** Implement GNN/CNN on FPGA near processor (~ns latency)
2. **Quantized networks:** 8-bit integer inference (vs. float32)
3. **Lightweight models:** Mamba decoders, reduced GNN message rounds
4. **Hybrid:** Belief propagation (classical, parallel) + neural refinement

---

## 5. VALIDATION AND BENCHMARKING

### 5.1 Evaluation Metrics

#### **Logical Error Rate (LER)**
```
LER = (# logical errors) / (# tests)

Computed by:
1. Prepare logical state |+L⟩ (eigenstate of logical X operator)
2. Apply random errors (noise model)
3. Measure stabilizers → obtain syndrome
4. Apply decoder → estimate most likely error
5. Apply inverse of estimated error to decoded state
6. Measure logical Z observable
7. Check if eigenvalue matches preparation (|+L⟩ → +1)
8. Repeat 1000s of times, compute Pr(eigenvalue mismatch)
```

#### **Threshold**
```
Threshold = Critical physical error rate p_th

Below threshold: LER decreases as code distance increases
Above threshold: LER increases with distance

For surface code under depolarizing noise:
- Classical MWPM: p_th ≈ 0.01 (1%)
- ML decoders: p_th ≈ 0.024 (2.4%)
- Achieved via exploiting correlations

Computing threshold:
1. Train decoder at multiple error rates: p = 0.5%, 1%, 2%, ...
2. Measure LER at multiple distances: d = 3, 5, 7, 9
3. Plot: (distance d) vs. LER, separate curves per error rate
4. Threshold = crossover point (LER slope changes sign)
```

---

### 5.2 Baseline Comparisons

#### **Classical Baselines**

**MWPM (Minimum Weight Perfect Matching)**
- Graph construction: Error syndrome → graph
- Nodes: Positions where syndrome != 0
- Edges: Paths connecting syndrome pairs
- Weight: Distance (physical qubits)
- Matching: Find minimum-weight pairing of syndrome nodes
- Correction: Apply path of single-qubit operations
- **Theoretical optimality:** Optimal under independent error assumption

**Tensor Network Decoder**
- Contraction of tensor network encoding error probabilities
- Computational cost: Exponential in general; polynomial for special cases
- Accuracy: Very high (near maximum-likelihood)
- Speed: Slow (minutes to hours for large codes)

**Belief Propagation**
- Iterative message passing on code graph
- Classical algorithm; no training
- Optimal on tree graphs; approximate with cycles
- Speed: Fast; real-time feasible

---

## 6. REPRODUCIBILITY AND OPEN-SOURCE RESOURCES

### 6.1 Simulation Frameworks

| Framework | Language | Codes Supported | Primary Use |
|-----------|----------|---|---|
| **Stim** (Microsoft) | C++ / Python | Surface, heavy hex, LDPC | Circuit-level noise; very fast |
| **Cirq** (Google) | Python | Arbitrary codes | General quantum circuits; moderate speed |
| **QuTiP** | Python | General qubits | Full density matrix; slower |
| **Pymatching** | Python/C++ | Surface, subsystem, etc. | MWPM baseline decoder |
| **Fusion Blossom** | Rust/C++ | All graph-decodable codes | Faster MWPM variant |

### 6.2 Decoder Implementations Available

| Decoder | Language | Repository | Status |
|---------|----------|---|---|
| **DQN (Toric)** | Python (TensorFlow) | github.com/mats-granath/toric-RL-decoder | Public |
| **GNN** | Python (PyTorch) | github.com/itsBergentall/QEC_GNN | Public |
| **GraphQEC** | Python | (Likely on arXiv supplementary) | Emerging |
| **AlphaQubit** | Proprietary (Google) | Limited access | Not public |
| **CNN (Heavy Hex)** | Python | (Academic paper supplementary) | Limited |
| **PyMatching** | Python/C++ | github.com/oscarhiggott/PyMatching | Public, mature |

---

## 7. Common Pitfalls and Lessons Learned

### 7.1 Training Challenges

1. **Sparse Reward Problem**
   - Naive RL fails without HER
   - Solution: Always use hindsight experience replay for sparse binary rewards

2. **Code-Specific Overfitting**
   - Model trained on d=5 fails on d=7 or different code type
   - Solution: Pre-train on diverse code families or use universal architectures (GNN)

3. **Noise Model Mismatch**
   - Training on depolarizing noise → poor transfer to real processor noise
   - Solution: Pre-train on broad noise distribution; fine-tune on real data

4. **Computational Cost**
   - Hundreds of millions of training examples required (AlphaQubit)
   - Solution: Use fast simulators (Stim); distributed training

5. **Real-Time Latency**
   - Neural decoders too slow for microsecond requirements
   - Solution: Research lightweight models (Mamba, GNN with fewer rounds)

### 7.2 Evaluation Mistakes

1. **Cherry-picked test set**
   - Training data distribution ≠ deployment distribution
   - Mitigation: Hold-out test set with identical distribution; test on real processor data

2. **Unfair baseline comparison**
   - Comparing trained RL vs. uninformed classical method
   - Proper comparison: RL vs. classical baseline with full noise model knowledge

3. **Threshold extrapolation**
   - Measured threshold on small codes may not hold for large d
   - Validation: Test on multiple code distances (d=3, 5, 7, 9)

4. **Ignoring error correlations**
   - Assumption: Errors independent → MWPM optimal
   - Reality: Crosstalk, measurement errors, drift → correlations present
   - ML advantage largest when correlations matter

---

**Document completed:** December 2025
