# Theoretical Framework: Reinforcement Learning for Quantum Error Correction

## Abstract

This document formalizes the application of reinforcement learning (RL) to quantum error correction (QEC), establishing a rigorous Markov Decision Process (MDP) formulation for both syndrome decoding and circuit optimization problems. We define complete state and action spaces, reward structures, and provide theoretical predictions for sample complexity, convergence rates, and generalization bounds.

---

## 1. Core Problem Formulation

### 1.1 Markov Decision Process for Syndrome Decoding

We formalize syndrome decoding as a finite-horizon MDP:

**Definition 1.1 (Syndrome Decoding MDP):**
The tuple M = (S, A, P, R, gamma, H) where:

- S: State space (syndrome histories and decoder memory)
- A: Action space (recovery operations)
- P: S x A x S -> [0,1] transition probability function
- R: S x A -> R reward function
- gamma in [0,1]: Discount factor
- H: Decoding horizon (number of syndrome measurement rounds)

### 1.2 Alternative Formulation: Circuit Optimization MDP

For circuit synthesis and optimization:

**Definition 1.2 (Circuit Optimization MDP):**
The tuple M_c = (S_c, A_c, P_c, R_c, gamma, T) where:

- S_c: Partial circuit configurations
- A_c: Gate insertion/modification actions
- P_c: Deterministic transitions (circuit construction is deterministic)
- R_c: Circuit quality metrics (depth, gate count, fidelity)
- T: Maximum circuit depth

---

## 2. State Space Formalization

### 2.1 Syndrome Decoding State Space

For a distance-d surface code with n = d^2 data qubits and m = d^2 - 1 syndrome qubits:

**Definition 2.1 (Syndrome State):**
```
s_t = (sigma_t, sigma_{t-1}, ..., sigma_{t-k+1}, h_t)
```

Where:
- sigma_t in {0,1}^m: Binary syndrome vector at time t
- k: Syndrome history window length
- h_t in R^d_h: Hidden state encoding decoder memory

**Formal State Space:**
```
S = {0,1}^{m x k} x R^{d_h}
```

**State Dimension:**
```
dim(S) = m * k + d_h = (d^2 - 1) * k + d_h
```

For d = 15, k = 5, d_h = 256:
```
dim(S) = 224 * 5 + 256 = 1,376
```

### 2.2 Detector Error Model State Representation

**Definition 2.2 (Detector State):**
```
s_t^det = (D_t, G)
```

Where:
- D_t in {0,1}^{n_det}: Detection event vector
- G = (V, E): Detector graph structure (static)
- n_det = O(d^2 * T_syndrome): Number of detectors

### 2.3 Circuit Optimization State Space

**Definition 2.3 (Circuit State):**
```
s_c = (C, Q, T_remaining)
```

Where:
- C = [g_1, g_2, ..., g_l]: Ordered list of gates applied
- Q in C^{2^n x 2^n}: Current quantum state (or stabilizer tableau for Clifford circuits)
- T_remaining: Remaining time budget

---

## 3. Action Space Formalization

### 3.1 Syndrome Decoding Action Space

**Definition 3.1 (Pauli Recovery Actions):**
```
A = {I, X, Y, Z}^n
```

For practical implementation with sparse corrections:

**Definition 3.2 (Sparse Recovery Actions):**
```
A_sparse = {(i, P) : i in {1,...,n}, P in {I, X, Y, Z}} U {NULL}
```

Where NULL indicates no correction.

**Cardinality:** |A_sparse| = 4n + 1

For distance-15 code: |A_sparse| = 4(225) + 1 = 901

### 3.2 Hierarchical Action Space (Recommended)

**Definition 3.3 (Two-Level Action Hierarchy):**

Level 1 (Cluster Selection):
```
A_1 = {cluster_1, ..., cluster_K, NULL}
```

Level 2 (Local Correction):
```
A_2(cluster_k) = {corrections within cluster_k}
```

### 3.3 Circuit Optimization Action Space

**Definition 3.4 (Gate Actions):**
```
A_gate = {(g, q_target, q_control) : g in G_universal, q_* in Q_available}
```

Where G_universal = {H, T, T_dag, CNOT, CZ, S, S_dag, ...}

---

## 4. Reward Structure

### 4.1 Syndrome Decoding Reward

**Definition 4.1 (Logical Error Reward):**
```
R_logical(s, a, s') =
    +1    if decoding succeeds (no logical error)
    -1    if logical error occurs
    0     for intermediate steps
```

**Definition 4.2 (Shaped Reward for Dense Feedback):**
```
R_shaped(s, a, s') = R_logical + lambda_1 * R_syndrome + lambda_2 * R_efficiency
```

Where:
```
R_syndrome = -|sigma_{t+1}|_1 / m          (syndrome weight reduction)
R_efficiency = -c(a)                        (correction cost penalty)
```

Hyperparameters: lambda_1, lambda_2 in [0, 0.1]

### 4.2 Episodic vs. Continuing Formulation

**Episodic (Recommended for Training):**
- Episode terminates after H syndrome rounds
- Final reward based on logical measurement outcome

**Continuing (Runtime Deployment):**
- Infinite horizon with gamma < 1
- Rewards based on syndrome evolution

### 4.3 Circuit Optimization Reward

**Definition 4.3 (Circuit Quality Reward):**
```
R_circuit(C) = -w_1 * depth(C) - w_2 * gate_count(C) + w_3 * fidelity(C)
```

Where fidelity is estimated via:
```
F(C) = |<psi_target|C|psi_0>|^2
```

---

## 5. Transition Dynamics

### 5.1 Noise Model Assumptions

**Assumption 5.1 (Depolarizing Channel):**
For physical error rate p, each qubit independently experiences:
```
E_depol(rho) = (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)
```

**Assumption 5.2 (Circuit-Level Noise):**
```
P(sigma_{t+1} | sigma_t, a_t, E_t) = P_noise(E_t) * P_syndrome(sigma_{t+1} | sigma_t, a_t, E_t)
```

Where E_t represents the physical error configuration.

**Assumption 5.3 (Measurement Noise):**
Each syndrome measurement has flip probability p_m:
```
P(sigma_measured = 1 | sigma_true = 0) = p_m
P(sigma_measured = 0 | sigma_true = 1) = p_m
```

### 5.2 Hardware Constraint Assumptions

**Assumption 5.4 (Qubit Connectivity):**
For surface codes, only nearest-neighbor interactions on 2D lattice.

**Assumption 5.5 (Gate Fidelity):**
Two-qubit gate fidelity: F_2Q >= 0.99
Single-qubit gate fidelity: F_1Q >= 0.999
Measurement fidelity: F_M >= 0.99

**Assumption 5.6 (Coherence Time Constraint):**
Total circuit depth D must satisfy:
```
D * t_gate << T_1, T_2
```

---

## 6. Formal Hypothesis

### 6.1 Primary Hypothesis

**Hypothesis H1 (RL Decoder Superiority):**
```
There exists an RL policy pi* such that for surface codes with distance d >= 15
and physical error rates p in [0.001, 0.01]:

    L_RL(pi*, d, p) <= (1 - delta) * L_MWPM(d, p)

where:
    - L_RL: Logical error rate of RL decoder
    - L_MWPM: Logical error rate of Minimum Weight Perfect Matching
    - delta >= 0.20 (at least 20% improvement)
```

### 6.2 Secondary Hypotheses

**Hypothesis H2 (Scalability):**
```
The trained policy pi_d generalizes such that:

    E_{d' in [d, d+4]}[L_RL(pi_d, d', p)] <= 1.1 * L_optimal(d', p)
```

**Hypothesis H3 (Sample Efficiency):**
```
The sample complexity to achieve delta-improvement scales as:

    N_samples = O(d^alpha) where alpha <= 3
```

### 6.3 Falsification Criteria

The hypothesis is FALSIFIED if:
1. RL decoder fails to exceed MWPM by >= 10% on any tested distance d >= 15
2. Training requires > 10^9 episodes for d = 15
3. Generalization gap exceeds 50% when transferring between distances

The hypothesis is CONFIRMED if:
1. RL decoder exceeds MWPM by >= 20% on d in {15, 17, 19, 21}
2. Training converges within 10^7 episodes for d = 15
3. Zero-shot transfer achieves >= 80% of fine-tuned performance

---

## 7. Theoretical Predictions

### 7.1 Sample Complexity Bounds

**Theorem 7.1 (Sample Complexity Upper Bound):**
For an epsilon-optimal policy under the syndrome decoding MDP with state dimension |S| and action dimension |A|:

```
N_samples = O(|S| * |A| * H^2 / epsilon^2 * log(1/delta))
```

For our parametric neural network approximation:
```
N_samples = O(d_theta * H^2 / epsilon^2 * log(1/delta))
```

Where d_theta is the number of network parameters.

**Prediction 7.1:**
For d = 15 with d_theta ~ 10^6, H = 15, epsilon = 0.05:
```
N_samples ~ O(10^8) episodes
```

### 7.2 Convergence Rate Analysis

**Theorem 7.2 (Policy Gradient Convergence):**
Under standard assumptions (bounded gradients, Lipschitz continuity), PPO converges at rate:

```
||nabla J(theta_t)|| <= O(1/sqrt(T))
```

Where T is the number of gradient updates.

**Prediction 7.2:**
Expected convergence to 95% of final performance:
```
T_95 ~ 10^5 gradient updates ~ 10^7 environment steps
```

### 7.3 Generalization Bounds

**Theorem 7.3 (PAC-Bayes Generalization):**
With probability >= 1 - delta over training data:

```
E_test[L(pi)] <= E_train[L(pi)] + sqrt((KL(pi || pi_0) + log(2*sqrt(n)/delta)) / (2n))
```

**Prediction 7.3:**
Generalization gap for d' = d + 2:
```
Delta_gen <= 0.15 * L_train  (15% degradation expected)
```

### 7.4 Threshold Improvement Prediction

**Prediction 7.4 (Error Threshold):**
Classical MWPM threshold: p_th^MWPM ~ 0.0103

RL decoder predicted threshold:
```
p_th^RL >= 1.15 * p_th^MWPM ~ 0.0118
```

Confidence: 70% based on analogous ML decoder results.

---

## 8. Mathematical Formalism for Key Components

### 8.1 Policy Network Architecture

**Definition 8.1 (Graph Neural Network Policy):**
```
pi_theta(a | s) = softmax(MLP(GNN(G_syndrome, X_features)))
```

Where:
```
GNN: R^{n_nodes x d_in} -> R^{n_nodes x d_out}
GNN(X, G) = sigma(D^{-1/2} A D^{-1/2} X W)  (simplified GCN)
```

### 8.2 Value Function Approximation

**Definition 8.2 (Critic Network):**
```
V_phi(s) = MLP(Aggregate(GNN(G_syndrome, X_features)))
```

Aggregate: Mean or attention-weighted pooling over graph nodes.

### 8.3 Advantage Estimation

**Definition 8.3 (Generalized Advantage Estimation):**
```
A_t^GAE = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}

delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

Recommended: lambda = 0.95, gamma = 0.99

---

## 9. Baseline Comparisons

### 9.1 Classical Decoders

| Decoder | Complexity | Threshold |
|---------|------------|-----------|
| MWPM | O(n^3) | ~1.03% |
| Union-Find | O(n * alpha(n)) | ~0.95% |
| Neural BP | O(n) | ~1.05% |

### 9.2 Improvement Targets

**Target 9.1:** Exceed MWPM by >= 20% in logical error rate
**Target 9.2:** Maintain O(n) or O(n log n) inference complexity
**Target 9.3:** Generalize across distances without retraining

---

## 10. Key Equations Summary

### Bellman Optimality Equation:
```
V*(s) = max_a [R(s,a) + gamma * sum_{s'} P(s'|s,a) * V*(s')]
```

### Policy Gradient:
```
nabla_theta J(theta) = E_{tau ~ pi_theta}[sum_t nabla_theta log(pi_theta(a_t|s_t)) * A_t]
```

### PPO Clipped Objective:
```
L^CLIP(theta) = E_t[min(r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t)]
```

### Logical Error Rate:
```
p_L = 1 - sum_{E: dec(sigma(E)) = E mod S} P(E)
```

Where S is the stabilizer group.

---

## 11. Experimental Design Requirements

### 11.1 Independent Variables
- Code distance: d in {5, 7, 9, 11, 13, 15, 17, 19, 21}
- Physical error rate: p in {0.001, 0.003, 0.005, 0.007, 0.01}
- Noise model: {depolarizing, biased, correlated}

### 11.2 Dependent Variables
- Logical error rate: p_L
- Decoding latency: t_decode
- Training sample complexity: N_samples

### 11.3 Control Variables
- Number of syndrome rounds: H = d
- Network architecture: Fixed across distances
- Training hyperparameters: Fixed per experiment set

### 11.4 Evaluation Protocol
1. Train on distance d_train in {5, 7, 9}
2. Evaluate on d_test in {5, 7, 9, 11, 13, 15}
3. Report mean +/- std over 10 random seeds
4. Statistical significance: p < 0.01 via paired t-test

---

## 12. Conclusion

This framework provides a complete mathematical specification for applying reinforcement learning to quantum error correction. The MDP formulation enables rigorous analysis of sample complexity, convergence, and generalization. The falsifiable hypothesis with clear success criteria enables objective evaluation of the research outcomes.

**Key Deliverables for Implementation:**
1. State representation: Syndrome history + GNN embedding
2. Action space: Sparse Pauli corrections
3. Reward: Logical error rate (episodic)
4. Algorithm: PPO with GAE
5. Baseline: MWPM decoder

---

*Document Version: 1.0*
*Framework Type: Markov Decision Process + Deep Reinforcement Learning*
*Target Application: Surface Code Syndrome Decoding*
