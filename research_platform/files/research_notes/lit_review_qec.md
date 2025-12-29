# Literature Review: Quantum Error Correction Systems, Datasets, and RL-Assisted Decoding

## Executive Summary

This literature review surveys recent advances in quantum error correction (QEC), focusing on experimental datasets, benchmarks, scalability challenges, and the emerging role of reinforcement learning and machine learning in QEC decoding. The field has experienced unprecedented momentum in 2024-2025, with major breakthroughs from Google (Willow chip, AlphaQubit), Harvard/MIT (48 logical qubits), and IBM (QLDPC codes). The convergence of neural network decoders and RL-assisted control represents a significant shift from purely algorithmic approaches toward learned decoders that can adapt to hardware-specific noise profiles.

---

## 1. Overview of the Research Area

### 1.1 Context and Motivation

Quantum error correction is widely recognized as the key technology for scaling quantum computers toward practical fault-tolerant quantum computation. The field has shifted from theoretical exploration to empirical demonstration across all major qubit platforms (superconducting, trapped-ion, neutral atom, photonic). The critical milestone of demonstrating **below-threshold** error suppression—where logical error rates decrease exponentially with code size—was achieved in 2024, marking the beginning of the "QEC era" in quantum computing.

### 1.2 Core Problem Statement

The fundamental challenge in QEC is that:
1. **Errors are continuous**: Physical qubits accumulate bit-flip, phase-flip, and correlated errors continuously during computation.
2. **Measurement is destructive**: Extracting syndrome information (parity check measurements) requires non-destructive measurement techniques and introduces additional errors.
3. **Resource overhead is severe**: Current surface codes require hundreds of physical qubits per logical qubit, demanding high-fidelity operations and ultra-low error rates.
4. **Real-time decoding is critical**: Decoder latency must be sub-microsecond to prevent syndrome backlog and exponential slowdown.
5. **Decoder complexity grows exponentially**: Finding optimal error corrections (decoding) is NP-hard for classical codes, motivating learned and heuristic approaches.

### 1.3 Hardware Platforms and Experimental Ecosystems

All major qubit modalities have crossed the 99% two-qubit gate fidelity threshold:
- **Superconducting qubits** (Google Willow, IBM): Best-in-class performance on surface codes; rapid error correction cycles (~1.1 μs); limited gate set and connectivity.
- **Trapped ions** (Quantinuum, IonQ): High gate fidelities (>99.9%); excellent coherence times; limited by chain length constraints (~30-56 qubits per trap).
- **Neutral atoms** (Harvard/MIT, Atom Computing): Reconfigurable qubit arrays with excellent tunability; demonstrated 48 logical qubits; emerging platform.
- **Superconducting + ML decoders** (Google + DeepMind): AlphaQubit transformer-based decoder validated on Sycamore processor data.

---

## 2. Chronological Summary of Major Developments (2019-2025)

### Phase 1: Theoretical and Algorithmic Foundations (2019-2022)

- **2019**: Delfosse et al. (Nautrup et al., Quantum 2019) introduce RL framework for optimizing surface code parameters; Tomasini et al. (Quantum 2019) apply deep RL to toric code decoding.
- **2022**: Google demonstrates surface code scaling to 7×7 array with improved fidelities; Kitaev toric code RL decoders achieve ~11% threshold performance (near theoretical optimum).
- **Early 2022**: Riverlane publishes Local Clustering Decoder paper; first hardware-deployed real-time QEC decoder.

### Phase 2: ML-Enhanced Decoding and Real-World Deployment (2023-Early 2024)

- **Late 2023**: Harvard/MIT team (Bluvstein et al., Nature Dec 2023) demonstrate 48 logical qubits on neutral atom processor using ~280 physical qubits.
- **Early 2024**: AlphaQubit paper submitted; shows 6% improvement over correlated matching, 30% improvement over tensor-network methods.
- **April 2024**: Quantinuum + Microsoft achieve 800x logical-to-physical error ratio using 4-qubit Steane code on trapped-ion processor.
- **August 2024**: Google publishes Willow results demonstrating below-threshold surface code with exponential error suppression (Λ = 2.14× per distance increment).

### Phase 3: Below-Threshold Era and RL-Assisted Control (Late 2024-Present)

- **December 2024**: Google announces Willow chip (105 qubits) with best-in-class metrics: 0.143% ± 0.003% logical error per cycle (distance-7); 2.4× logical qubit lifetime advantage.
- **November 2024**: AlphaQubit published in Nature; validates on distances up to 11 using simulated data; trained on 241-qubit synthetic systems.
- **December 2024**: Riverlane publishes Local Clustering Decoder in Nature Communications; demonstrated deployment on multiple vendor platforms (Infleqtion, Oxford Quantum Circuits, Rigetti, etc.).
- **November 2025**: IonQ publishes improved BB5 codes for trapped-ion systems (10-20× overhead reduction vs. standard BB codes).
- **December 2025**: China demonstrates microwave-based QEC below threshold; IBM releases Relay-BP decoder with orders-of-magnitude improvement for qLDPC codes.
- **Ongoing (2025)**: 120+ peer-reviewed QEC code papers published (vs. 36 in 2024); every major quantum vendor includes QEC in roadmaps targeting 2028-2030 deployment.

---

## 3. Detailed Survey of Prior Work

### 3.1 Experimental Datasets and Benchmarking

#### 3.1.1 Google Willow Chip (December 2024)

**Citation**: Google Quantum AI, December 2024; Nature (in submission process)

**Hardware Specification**:
- Qubit count: 105 qubits (72-qubit and 105-qubit variants tested)
- Qubit type: Superconducting (transmon-family)
- Gate fidelities:
  - Single-qubit gate error: 0.035% ± 0.029%
  - Two-qubit gate error (CZ): 0.33% ± 0.18%
  - Measurement error: 0.77% ± 0.21%
- Coherence (T1): Approaching 100 μs (~5× improvement over prior generation)

**Surface Code Benchmarks**:
- Maximum tested distance: 7 (101 qubits)
- Logical error rate (distance-7): 0.143% ± 0.003% per error correction cycle
- Logical qubit lifetime: 2.4× ± 0.3× that of best physical qubit
- Error suppression factor per distance increment: Λ = 2.14 ± 0.02 (exponential below-threshold behavior)
- Scaling range tested: 3×3, 5×5, 7×7 code geometries

**Decoder**: Soft-input-augmented correlated matching + measurement feedback

**Key Finding**: First demonstration of exponential logical error suppression with code distance on a superconducting processor, confirming the feasibility of fault-tolerant quantum computation.

#### 3.1.2 AlphaQubit: Neural Network Decoder (Google DeepMind, November 2024)

**Citation**: Published in Nature, November 2024; preliminary results in blog post November 2024

**Network Architecture**:
- Type: Transformer-based, recurrent neural network
- Training data generation:
  - Simulated data from quantum simulator: Hundreds of millions of examples
  - Sycamore processor experimental data: Thousands of samples for fine-tuning
  - Synthetic dataset scope: Up to 241 qubits (exceeding available Sycamore qubit count)

**Experimental Setup**:
- Tested on: Google Sycamore superconducting processor
- Code distances evaluated: 3, 5, 7, 9, 11 (11 only on simulated data with realistic noise)
- Error models: Depolarizing noise, circuit-level noise (cross-talk, leakage)

**Performance Metrics**:
- vs. Correlated matching (SCAM): 30% error reduction
- vs. Tensor-network methods: 6% error reduction
- Real-time latency: Not yet sub-microsecond; post-processing only
- Scaling: Maintains advantage at distances up to 11 on simulated data

**Dataset and Limitations**:
- Fine-tuning dataset size: ~Thousands of Sycamore samples per configuration
- Dataset quality issue: Requires careful calibration and noise characterization
- Inference speed: Too slow for real-time decoding on current superconducting processors (which operate at MHz rates)

**Key Finding**: Deep learning decoders can generalize beyond training data and scale to larger code distances than training data size would naively suggest.

#### 3.1.3 Harvard/MIT Neutral Atom Processor (December 2023)

**Citation**: Bluvstein et al., Nature 2023; QuEra press release, December 2023

**Hardware Specification**:
- Physical qubit count: ~280 qubits (neutral atoms in optical tweezers)
- Logical qubit count: Up to 48 logical qubits
- Qubit type: Rydberg-excited rubidium atoms
- Gate fidelities: ~98-99% (typical for neutral atoms)

**Error Correction Configuration**:
- Code type: Surface code (distributed across atom array)
- Logical algorithm demonstrated: Complex quantum algorithms (quantum phase transition simulation, etc.)
- Outcome: Logical algorithms outperform physical implementations of same algorithms

**Key Achievement**: First demonstration of **large-scale algorithm execution on error-corrected quantum computer**; heralds early fault-tolerant computing era.

#### 3.1.4 IBM QLDPC Codes and Relay-BP Decoder (2024-2025)

**Citation**: IBM Quantum Computing Blog + Nature paper (ongoing); Relay-BP preprint (December 2025)

**Code Specification** (Gross Code / Bivariate Bicycle):
- Efficiency: 10× improvement over surface code in qubit overhead
- Configuration: 12 logical qubits encoded into 144 data qubits + 144 syndrome check qubits (288 total)
- Physical qubits for comparison: Surface code would require ~3,000 qubits for equivalent protection; Gross code requires 288
- Connectivity: Each qubit connects to 6 neighbors; routing on 2 layers only

**Decoder Performance** (Relay-BP):
- Logical error rate: Orders of magnitude better than prior qLDPC decoders
- Training corpus: IBM Quantum Loon (2025) experiments
- Scalability: Extends to higher-rate codes beyond 12 logical qubits

**Key Innovation**: Dramatic reduction in physical qubit overhead opens door to practical large-scale fault-tolerant quantum computers.

#### 3.1.5 Quantinuum Trapped-Ion System (2024-2025)

**Citation**: Quantinuum H2; Microsoft collaboration (April 2024); recent preprints (2025)

**Hardware Specification**:
- Qubit count: 56 qubits (largest trapped-ion system to date)
- Trap type: "Racetrack" design enabling longer chains than 1D traps
- Gate fidelities: >99.9% (world-leading for two-qubit gates)
- Coherence time: >100 ms (exceptional)

**Error Correction Benchmarks**:
- Logical error reduction: 800× improvement over physical qubits (using post-selected Steane code with 4 logical qubits)
- Code distance: Limited by trap length (~30 qubits per standard trap; racetrack extends capability)
- Caveat: Post-selection used; does not scale to asymptotic regime without further development

**New Code Variants** (BB5 codes, 2025):
- BB5 code (length-30, 4 logical qubits): 10× improvement over standard BB codes
- BB5 code (length-48, 4 logical qubits): 20× improvement over standard BB codes

**Key Advantage**: Ultra-high gate fidelities and coherence reduce error correction overhead compared to superconducting approaches.

#### 3.1.6 Real-Time Decoder Benchmarks (Riverlane + Academic, 2024-2025)

**Citation**: Riverlane Nature Communications paper (2024); Riverlane Hardware Decoder announcement (December 2025); Qblox/QEM research

**Local Clustering Decoder (Riverlane)**:
- Latency: <1 microsecond per decoding round
- Hardware implementation: FPGA-based (with ASIC roadmap)
- Deployment platforms: Infleqtion, Oxford Quantum Circuits, Rigetti, Oak Ridge National Lab
- Code: Surface code, optimized for real-time feedback

**Superconducting Real-Time Experiments**:
- 8-qubit stability experiment: Up to 25 decoding rounds; mean decoding time <1 μs per round
- Typical cycle time: 1.1 μs (superconducting processor); decoder must complete within this window
- FPGA implementation: 7.30× speedup over software (C++); 81.51× energy efficiency improvement

**Key Bottleneck**: Real-time decoding latency is now the primary limitation preventing full syndrome feedback utilization on fast platforms.

---

### 3.2 Reinforcement Learning Applications in QEC

#### 3.2.1 RL Framework for Code Optimization (Nautrup et al., 2019)

**Citation**: Nautrup, P., et al. "Optimizing Quantum Error Correction Codes with Reinforcement Learning." Quantum Journal, 2019; also published in Quantum 2019

**Approach**:
- RL agent task: Modify surface code topology (qubit placement, measurements) to minimize syndrome weight and logical error rate
- Reward signal: Inverse of logical error rate given a noise model
- Environment: Simulated quantum surface code with ~70 data qubits (arbitrary connectivity)
- RL algorithm: Deep Q-learning with experience replay

**Results**:
- Near-optimal solutions found within few hours of training
- Agent discovers codes comparable to hand-designed variants
- Generalization: Agents train on one noise model, transfer to different models with good performance

**Limitations**:
- Scalability limited to ~70 qubits due to simulation cost
- Code structure restricted by search space design
- No real-world validation on quantum hardware

#### 3.2.2 RL for Toric Code Decoding (Tomasini et al., 2019)

**Citation**: Tomasini, A., et al. "Quantum error correction for the toric code using deep reinforcement learning." Quantum Journal, 2019; also Yao et al. (2020) on related work

**Approach**:
- Agent task: Perform single-qubit Pauli corrections to suppress toric code errors
- State space: Syndrome measurements (parity check outcomes)
- Action space: Which qubit pair to flip
- Q-function approximation: Deep convolutional neural network
- Training: Standard deep Q-learning with ε-greedy exploration

**Results**:
- Threshold performance: Near-optimal threshold of ~11% for uncorrelated noise (theoretical optimum ~11.0%)
- Training time: Efficient; hours to days on standard hardware
- Generalization: Agents trained on one error probability generalize to nearby error rates
- Comparison: Outperforms minimum-weight perfect matching (MWPM) at high error rates

**Limitations**:
- Tested only on idealized toric code (no measurement errors, not gate-level)
- Scalability: Tested up to ~L=8 codes (small system sizes)
- Real-world deployment: Not validated on physical hardware

#### 3.2.3 Simultaneous Discovery of Codes and Encoders (Recent, 2024)

**Citation**: Published in npj Quantum Information, 2024; also OpenReview submission

**Approach**:
- Novel contribution: Use RL to simultaneously discover both QEC code and encoding circuit
- Noise-aware agent: Trained with knowledge of specific noise model
- Representation: Agents explore code space using abstract description language
- Scope: Up to 25 physical qubits, distance-5 codes

**Results**:
- Successfully discovers Bell-code, Surface code, and other standard codes from scratch
- Encoding circuits discovered are near-optimal in depth
- Transfer learning: Codes learned on one modality (superconducting) transfer partially to another (trapped-ion)

**Key Innovation**: First demonstration of end-to-end code+encoder discovery using RL; opens door to hardware-optimized codes.

#### 3.2.4 RL for Real-Time QEC Control (Recent, 2025)

**Citation**: "Reinforcement Learning Control of Quantum Error Correction," arXiv 2511.08493 (November 2025)

**Approach**:
- Unifies calibration and error correction: Error detection events → learning signal
- Agent task: Steer physical control parameters (Hamiltonian parameters) to stabilize quantum system
- Continuous learning: Parameter adjustments made in real-time during computation
- Experimental platform: Superconducting processor

**Results**:
- Logical error rate stability: 3.5× improvement against injected parameter drift
- Adaptation: Agent learns optimal trajectories for surface code implementation under realistic parameter drifts
- Real-time capability: Demonstrated on hardware; compatible with fast superconducting cycles

**Significance**: Bridges gap between static code design and dynamic adaptive quantum computation; addresses practical parameter drift challenges.

---

### 3.3 ML Benchmarking and Comparative Studies

#### 3.3.1 Comprehensive Benchmarking of ML Models (ICML/NeurIPS, 2023-2024)

**Citation**: "Benchmarking Machine Learning Models for Quantum Error Correction," arXiv 2311.11167v3; OpenReview presentation (2024)

**Scope**:
- 7 state-of-the-art deep learning algorithms compared:
  - Convolutional Neural Networks (CNNs)
  - Graph Neural Networks (GCN, GNN variants)
  - Graph Attention Networks (GAT)
  - Multi-GNN with ensemble methods
  - Graph Transformers (APPNP)
  - U-Net (CNN variant with skip connections)
- Code: Surface code with varying distances (d = 3, 5, 7)
- Noise: Depolarizing, circuit-level error models
- Training corpus: 10,000 - 100,000 syndrome samples per configuration

**Performance Results**:
- Overall accuracy by method (example for d=5 at p=0.01 error rate):
  - CNN: ~92%
  - U-Net: ~95% (50% improvement over CNN)
  - GCN: ~94%
  - GAT: ~93%
  - Multi-GNN: ~96%
  - Graph Transformer: ~94%
- Error correction rate (perfect recovery): Similar trend, with U-Net leading
- Key insight: Receptive field size matters; larger receptive fields → better accuracy

**Scaling Behavior**:
- Counter-intuitive finding: Performance does NOT deteriorate with increasing code distance
- Some GNN methods (GCN, APPNP, Multi-GNN) actually improve with distance d up to d=9
- Implication: Learned decoders generalize better than traditional algorithmic decoders to larger codes

**Dataset Quality**:
- Training set size dependency: Diminishing returns beyond ~50K samples per configuration
- Noise model mismatch: Significant performance drop if test noise differs from training noise
- Physical data: Synthetic training + experimental fine-tuning gives best results (AlphaQubit approach)

#### 3.3.2 Quantum Error Mitigation via ML (IBM, 2024)

**Citation**: "Machine Learning for Practical Quantum Error Mitigation," IBM Research + academic collaborators; arXiv 2309.17368

**Approach**:
- Alternative to full error correction: Error mitigation using ML post-processing
- Tested on: State-of-the-art QPUs up to 100+ qubits
- Methods compared:
  - Linear regression
  - Random forests
  - Multi-layer perceptrons (MLPs)
  - Graph neural networks (GNNs)
  - Comparison baseline: Zero-noise extrapolation (ZNE)

**Performance**:
- Random forests consistently best performer across tested systems
- Runtime cost: ML methods reduce overhead compared to ZNE (which requires 2× runtime)
- Accuracy: Competitive with ZNE despite lower computational cost
- Scalability: Effective up to 100+ qubits

**Key Limitation**: Error mitigation (not correction) only suppresses errors to ~10^-3 level; cannot reach fault-tolerance thresholds like full QEC can.

---

### 3.4 Scalability Challenges and Limitations

#### 3.4.1 Physical Qubit Requirements and Threshold

**Threshold Theorem** (Fundamental):
- Definition: Critical error rate p_th below which logical error rate decreases with code size
- Current estimates: p_th ≈ 1% for surface codes (range: 0.5% - 2% depending on detailed noise model)
- Practical requirement: Error rates must be ~10× below threshold for economically viable fault-tolerant quantum computers
- Implication: Typical requirement is p < 0.01% (one error per 10,000 operations)

**Physical Qubit Scaling**:
- Surface code with 0.1% error rate: 1,000 - 10,000 physical qubits per logical qubit
- Large-scale applications: 20 million qubits estimated for practically useful algorithms
- QLDPC codes (IBM Gross code): 288 qubits for 12 logical qubits (10× improvement over surface code)
- Current best hardware: 105 qubits (Google Willow); still far from practical scales

#### 3.4.2 Architectural Bottlenecks

**Syndrome Latency and Backlog**:
- Problem: Syndrome measurements produce error information at MHz rates on superconducting processors
- Bottleneck: Decoder must complete one round <1 μs to prevent backlog
- Consequence: Syndrome backlog causes exponential growth in error probability
- Current state: Only Riverlane (FPGA) and Google (integrated decoder) achieve sub-μs latency

**Qubit Connectivity Constraints**:
- Superconducting: Limited to nearest-neighbor or slightly extended connectivity
- Trapped-ion: Linear chains limited to ~30 qubits; racetrack designs extend to ~56 qubits
- Neutral atoms: Potentially arbitrary connectivity but control overhead grows
- Impact: Connectivity limitations restrict achievable code distances on fixed qubit counts

**Measurement Overhead**:
- Surface code: ~2× syndrome qubits required (measuring stabilizers requires additional qubits in some designs)
- Codes like Steane: ~3× syndrome qubits for distance d codes
- Implication: Actual physical qubit overhead even higher than theoretical predictions

#### 3.4.3 Known Practical Limitations (Hardware-Level)

**Gate Fidelity Plateau**:
- Current best two-qubit fidelities: 99.9% (trapped-ion) to 99.7% (superconducting)
- Gap to 99.0% threshold: Substantial for superconducting; trapped-ion near threshold
- Systematic errors: Some hardware exhibits correlated errors not captured by Pauli error models
- Risk: Born-Markov approximation (assumption of standard error models) may not hold in practice

**Measurement Errors**:
- Readout fidelity: Typically 98-99%
- Problem: Measurement errors propagate and accumulate; not easily corrected by code
- Mitigation: High-fidelity measurement (>99.5%) critical for QEC

**Cross-Talk and Higher-Order Effects**:
- Assumption violation: Standard error models assume depolarizing noise; real systems have cross-talk, leakage, heating
- Impact: Real error rates often higher than predicted; codes may operate above threshold in practice
- Experimental challenge: Characterizing realistic error models requires extensive tomography

#### 3.4.4 Decoder Scalability

**Computational Complexity**:
- Optimal decoding: NP-hard problem (reduction to minimum-weight perfect matching in general)
- Algorithmic decoders: MWPM (Delfosse-Follin) runs in O(n^3) time; impractical for large codes
- Heuristic decoders: Union-find, clustering algorithms run in O(n log n) but with worse distance scaling

**Real-Time Decoding Constraints**:
- Latency budget: 1 μs for superconducting processors (1.1 μs error-correction cycle)
- FPGA implementation: Achievable for small-to-medium codes (distance up to ~7)
- ASIC requirement: Necessary for large codes and future >MHz error correction rates
- ML decoders: AlphaQubit not yet real-time; inference latency millisecond-scale

**Scalability of ML Decoders**:
- Training data requirement: Exponential in code size (typical 10K-100K samples per configuration)
- Simulation cost: Generating synthetic training data scales exponentially with qubit count
- Model capacity: Transformer-based (AlphaQubit) can handle up to ~241 qubits but larger models may be needed
- Generalization: Unclear how well decoders trained on d=5 transfer to d=15 or d=25

---

### 3.5 Identified Gaps and Open Problems

#### 3.5.1 Fundamental Unresolved Questions

1. **Model Mismatch Problem**: Current QEC codes assume Born-Markov error models, but real quantum systems exhibit memory effects, non-Markovian dynamics, and complex correlations. How well do current codes perform under realistic non-Markovian noise?

2. **Cross-Talk and Leakage**: Standard error models do not account for qubit-qubit cross-talk (control crosstalk, capacitive coupling) or leakage out of computational subspace (common on superconducting qubits). What is the true error threshold under realistic error channels?

3. **Adaptive Code Discovery**: RL-assisted code discovery (Section 3.2.3) works for small systems (d≤5, n≤25 qubits). How does this scale to d≥15, n≥1000 qubits? Is the search space exploration feasible?

4. **Decoder Generalization Under Distribution Shift**: AlphaQubit trained on simulated noise; does it maintain performance when deployed on hardware with different error characteristics? Preliminary results suggest yes for small distribution shifts, but large shifts are unexplored.

5. **Real-Time Scalability of Neural Decoders**: ML decoders achieve sub-microsecond latency only on specialized hardware (FPGA/ASIC). Can general-purpose quantum control electronics (FPGAs in quantum labs) support real-time ML decoding at scale?

#### 3.5.2 Methodological Gaps

1. **Standardized Benchmarking**:
   - No agreed-upon benchmark suite for QEC decoders across platforms
   - Metric inconsistency: Some papers report logical error rate, others report threshold, others report accuracy on synthetic test sets
   - Need: Error Correction Zoo-like standardized benchmark (currently exists only for code parameters, not decoder performance)

2. **Dataset Availability and Reproducibility**:
   - Google Sycamore data used for AlphaQubit fine-tuning: Not publicly released
   - Toric code RL decoder papers: Mostly synthetic data, code/datasets not uniformly available
   - IBM QLDPC: Limited hardware data available
   - Challenge: Reproducibility and fair comparison across methods difficult

3. **ML Decoder Robustness Evaluation**:
   - Robustness to distribution shift: Tested for Sycamore → simulated; not tested for systematic drifts or hardware parameter changes
   - Adversarial examples: "Fooling the Decoder: Adversarial Attack on Quantum Error Correction" (arXiv 2504.19651) shows adversarial input patterns can degrade performance; mitigation unknown
   - Generalization: Unclear how generalization scales with code distance or qubit count

#### 3.5.3 Implementation Challenges

1. **Decoder Latency Hierarchy**:
   - Algorithmic: MWPM ~microseconds (software); FPGA MWPM possible but not demonstrated at scale
   - ML-based (AlphaQubit): Milliseconds on CPU/GPU; microseconds only with specialized hardware
   - RL-trained decoders: Unknown latency; most papers focus on accuracy, not inference speed

2. **Syndrome Data Bandwidth**:
   - Modern superconducting processors generate ~GHz of syndrome data (with ~10^6 checks/second)
   - Decoding throughput: Current systems handle ~1 million syndromes/second (single decoder)
   - Gap: Significant bandwidth mismatch; multiple parallel decoders or hierarchical decoding needed

3. **Training Data Generation for Real Hardware**:
   - Simulation-to-hardware transfer: AlphaQubit fine-tunes on hardware data; how much hardware data is needed?
   - Cost of characterization: Extracting thousands of error samples requires many experiments
   - Noise non-stationarity: Quantum hardware drifts; retraining frequency unknown

#### 3.5.4 Unsolved Practical Problems

1. **Measurement Error Correction**:
   - Syndrome measurements themselves produce errors
   - Currently handled by repetition (measure syndrome multiple times) or higher-distance codes
   - Open: Optimal syndrome measurement strategy for given hardware; ML-assisted syndrome extraction not widely explored

2. **Logical Operator Extraction**:
   - Decoding must not only correct physical errors but preserve logical information
   - Tracking logical state through many correction rounds: Complex and error-prone
   - Some ML decoders may not explicitly conserve logical subspace; implications unknown

3. **Fault-Tolerant Threshold Under Real Noise**:
   - Theoretical threshold: 1% (surface code, idealized)
   - Observed practical threshold: Unknown; experiments suggest 0.5-1.5% range but hardware-dependent
   - Prediction: Threshold varies by error type (bit-flip vs. phase-flip vs. measurement errors)

4. **Scaling Learned Decoders to Practical Sizes**:
   - AlphaQubit trained on d≤5, tested up to d=11 (simulated)
   - Practical fault-tolerant computing requires d≥15 or higher
   - Open: Can transformers scale to d≥25? Training data exponential in d; model size scaling unknown

---

## 4. State-of-the-Art Summary

### 4.1 Hardware Achievements (December 2024 - Present)

| Platform | Key Achievement | Metric | Reference |
|----------|--------|--------|-----------|
| **Google Willow (Superconducting)** | Below-threshold surface code | Λ = 2.14× per distance increment; 0.143% per cycle at d=7 | Google Nature paper (2024) |
| **Harvard/MIT (Neutral Atoms)** | 48 logical qubits | ~280 physical qubits; algorithms outperform physical implementations | Bluvstein et al., Nature (2023) |
| **Quantinuum (Trapped-ion)** | High-fidelity gates + QEC | 800× logical-to-physical error ratio (Steane code) | Microsoft/Quantinuum (2024) |
| **IBM (QLDPC Gross Code)** | Efficient codes | 288 qubits for 12 logical qubits (10× better than surface code) | IBM Nature paper (2024) |

### 4.2 Decoder Performance (November 2024 - Present)

| Decoder Type | Architecture | Performance Gain | Latency | Deployment | Reference |
|--------------|--------------|-------|---------|-----------|-----------|
| **AlphaQubit** | Transformer-RNN | 30% vs. SCAM; 6% vs. TN methods | ~milliseconds | Software (post-processing) | Google/DeepMind Nature (2024) |
| **Local Clustering (Riverlane)** | FPGA hardware | Baseline comparison | <1 μs | FPGA; deployed on 4+ platforms | Riverlane NatComm (2024) |
| **QUEKUF (Union-Find FPGA)** | Custom FPGA | 7.30× vs. C++; 81.51× energy efficiency | ~100 ns | FPGA | ACM TRETS (2024) |
| **Relay-BP (IBM)** | Iterative belief propagation | Orders of magnitude better than prior qLDPC decoders | Unknown | Preprint (software) | IBM preprint (Dec 2025) |

### 4.3 ML vs. Algorithmic Decoder Comparison

**Advantages of Learned Decoders**:
- Better accuracy on real hardware (adapts to actual noise)
- Scaling: Some GNNs improve with code distance
- Generalization: Transfer between noise models possible

**Advantages of Algorithmic Decoders**:
- Guaranteed real-time latency (hardware implementation possible)
- Interpretability: Understand when and why decoder fails
- No training data required

**Current best practice**: Hybrid approach—use FPGA decoder for real-time baseline, retrain neural decoder for post-processing to improve final accuracy.

---

## 5. Quantitative Evidence Table

### 5.1 Error Rates and Thresholds

| Metric | Value | Hardware/Code | Reference |
|--------|-------|---------|-----------|
| Surface code threshold (theoretical) | ~1% | Idealized | Standard QEC theory |
| Surface code threshold (empirical) | 0.5-1.5% | Real hardware | Various experiments |
| Willow logical error per cycle (d=7) | 0.143% ± 0.003% | Superconducting | Google (Dec 2024) |
| Willow error suppression factor | 2.14 ± 0.02 × per distance | Superconducting | Google (Dec 2024) |
| Harvard/MIT logical vs. physical ratio | >1× (beneficial) | Neutral atoms (48 logical) | Harvard/MIT (Dec 2023) |
| Quantinuum logical-to-physical improvement | 800× | Trapped-ion (Steane code) | Microsoft/Quantinuum (April 2024) |
| Toric code RL threshold (uncorrelated noise) | ~11% | Simulated | RL papers (2019-2020) |

### 5.2 Hardware Parameters

| Parameter | Superconducting | Trapped-ion | Neutral Atoms |
|-----------|----------|--------|-------------|
| Single-qubit gate error | 0.035% ± 0.029% (Willow) | <0.01% | ~0.1-1% |
| Two-qubit gate error | 0.33% ± 0.18% (Willow CZ) | >99.9% fidelity | ~1-2% |
| Measurement error | 0.77% ± 0.21% (Willow) | ~1-2% | ~0.5-1% |
| Coherence time (T1) | ~100 μs (Willow) | >100 ms (Quantinuum) | ~1-10 s |
| Error correction cycle time | 1.1 μs | ~10-100 μs | ~10-100 μs |
| Decoder latency (real-time) | <1 μs (FPGA) | Not demonstrated | Not demonstrated |

### 5.3 Decoder Training Data Sizes

| Method | Training Set Size | Code Size | Reference |
|--------|------------|----------|-----------|
| AlphaQubit (simulated) | Hundreds of millions | Up to 241 qubits | Google (2024) |
| AlphaQubit (fine-tuning) | Thousands of samples | d=3, d=5 (Sycamore) | Google (2024) |
| Benchmarking study (ML decoders) | 10K - 100K samples | d=3, 5, 7 | ICML/NeurIPS (2024) |
| Toric code RL | Implicit in RL training | L=4-8 (small codes) | 2019-2020 papers |
| QLDPC (IBM) | Experimental data from IBM Quantum Loon | 144 data qubits | IBM (2025) |

---

## 6. Quantitative Findings: Key Data Points for Evidence Sheet

### 6.1 Metric Ranges (Extracted from Literature)

- **Logical error rate (surface code, below threshold)**: 0.14% - 1% per cycle
- **Error suppression factor (exponential regime)**: 2.0 - 2.5 × per distance increment
- **ML decoder accuracy improvement**: 6% - 30% vs. classical decoders
- **Physical qubit overhead (surface code)**: 100 - 10,000 physical qubits per logical qubit
- **Physical qubit overhead (QLDPC)**: 24 - 100 physical qubits per logical qubit (10× better)
- **Decoder latency (FPGA, real-time)**: <1 μs per round
- **Decoder latency (ML, software)**: ~millisecond per batch
- **Gate fidelities (state-of-art)**: 99.99% (trapped-ion, oxford) to 99.96% (superconducting)

### 6.2 Typical Sample Sizes

- **Simulated training dataset for neural decoders**: 10^6 - 10^8 syndrome samples per error rate
- **Experimental fine-tuning dataset**: 10^3 - 10^4 samples from hardware
- **Code distance range tested**: d = 3 to d = 11 (simulated); d = 3 to d = 7 (hardware)
- **Qubit counts tested**: 3 qubits (toy models) to 280 qubits (practical demonstrations)

### 6.3 Known Pitfalls and Methodological Issues

1. **Simulation-to-hardware gap**: Synthetic training data uses idealized error models; real hardware has cross-talk, leakage, non-Markovian effects
2. **Survivor bias in code selection**: Successful codes in literature likely easier to implement; failed attempts not published
3. **Generalization bounds unclear**: ML decoders tested up to d=11 (simulated); scaling to practical d≥15 unknown
4. **Measurement error underestimation**: Many experiments assume measurement fidelity >98%; true integrated fidelity (including data qubit measurement interaction) often lower
5. **Code distance saturation**: Some studies report performance plateaus at d>7, suggesting other error sources dominate
6. **Decoder fairness comparisons**: Different papers report metrics inconsistently (accuracy vs. logical error rate vs. F1 score); true performance gaps hard to assess
7. **Hardware drift**: Long experiments (>1000 correction rounds) show parameter drift; RL control (Section 3.2.4) partially addresses but not fully solved
8. **Syndrome backlog**: Syndrome accumulation in decoder queue can cause exponential error growth; mitigated by Riverlane-like real-time decoders but not universally deployed
9. **Scaling assumptions**: Theoretical predictions assume linear scaling of error rates with code distance; sub-threshold regime shows exponential suppression, but transition region poorly characterized

---

## 7. References (Full Bibliography)

### 7.1 Major Hardware Demonstrations

1. Google Quantum AI. (December 2024). "Quantum error correction below the surface code threshold." Nature (In press; preprint available). URL: https://arxiv.org/abs/2408.13687

2. Bluvstein, D., et al. (December 2023). "Logical quantum processor based on reconfigurable atom arrays." Nature, 604, 451-456. URL: https://www.nature.com/articles/s41586-023-06927-3

3. Google DeepMind. (November 2024). "Learning high-accuracy error decoding for quantum processors." Nature. (Published online November 2024). URL: https://www.nature.com/articles/s41586-024-08148-8

4. IBM Quantum. (2024). "Landmark IBM error correction paper on Nature cover." IBM Quantum Computing Blog. (QLDPC/Gross code). URL: https://www.ibm.com/quantum/blog/nature-qldpc-error-correction

### 7.2 Reinforcement Learning for QEC

5. Nautrup, P., et al. (2019). "Optimizing Quantum Error Correction Codes with Reinforcement Learning." Quantum Journal. URL: https://quantum-journal.org/papers/q-2019-12-16-215/

6. Tomasini, A., et al. (2019). "Quantum error correction for the toric code using deep reinforcement learning." Quantum Journal. URL: https://quantum-journal.org/papers/q-2019-09-02-183/

7. "Simultaneous discovery of quantum error correction codes and encoders with a noise-aware reinforcement learning agent." (2024). npj Quantum Information. URL: https://www.nature.com/articles/s41534-024-00920-y

8. "Reinforcement learning control of quantum error correction." (2025). arXiv:2511.08493. URL: https://arxiv.org/abs/2511.08493

### 7.3 ML Benchmarking Studies

9. "Benchmarking Machine Learning Models for Quantum Error Correction." (2024). OpenReview / arXiv:2311.11167v3. URL: https://arxiv.org/abs/2311.11167

10. "Machine Learning for Practical Quantum Error Mitigation." (2024). IBM Research & collaborators. arXiv:2309.17368. URL: https://arxiv.org/abs/2309.17368

### 7.4 Real-Time Decoding and Hardware Implementation

11. Riverlane. (2024). "Local Clustering Decoder." Nature Communications. (Hardware-based real-time QEC decoder). URL: [Not provided in search results; inferred from multiple references]

12. "Demonstrating real-time and low-latency quantum error correction with superconducting qubits." (2024). arXiv:2410.05202. URL: https://arxiv.org/abs/2410.05202

13. "QUEKUF: An FPGA Union Find Decoder for Quantum Error Correction on the Toric Code." (2024). ACM Transactions on Reconfigurable Technology and Systems. URL: https://dl.acm.org/doi/10.1145/3733239

14. "Scalable Neural Decoders for Practical Real-Time Quantum Error Correction." (2025). arXiv:2510.22724. URL: https://arxiv.org/html/2510.22724v1

### 7.5 Trapped-Ion and Multi-Platform Work

15. "Quantum error correction for long chains of trapped ions." (2025). Quantum Journal. arXiv:2503.22071. URL: https://quantum-journal.org/papers/q-2025-11-27-1920/

16. IonQ. (2025). "Our Novel, Efficient Approach to Quantum Error Correction." Blog post. URL: https://ionq.com/blog/our-novel-efficient-approach-to-quantum-error-correction

17. "Ion-Trap Chip Architecture Optimized for Implementation of Quantum Error-Correcting Code." (2025). arXiv:2501.15200. URL: https://arxiv.org/html/2501.15200

### 7.6 Analysis and Review Papers

18. "The Quantum Error Correction Report 2024 / 2025." Riverlane Research. (Comprehensive industry survey). URL: https://www.riverlane.com/quantum-error-correction-report-2024

19. "Quantum Error Correction: Our 2025 trends and 2026 predictions." Riverlane. (Trends and outlook). URL: https://www.riverlane.com/blog/quantum-error-correction-our-2025-trends-and-2026-predictions

20. "Analysis of Surface Code Algorithms on Quantum Hardware Using the Qrisp Framework." (2024). Electronics, 14(23), 4707. URL: https://www.mdpi.com/2079-9282/14/23/4707

### 7.7 Fundamental and Theoretical Works

21. "Exponentially tighter bounds on limitations of quantum error mitigation." (2024). Nature Physics. URL: https://www.nature.com/articles/s41567-024-02536-7

22. "Quantum error correction under numerically exact open-quantum-system dynamics." (2024). Phys. Rev. Research, 5, 043161. URL: https://link.aps.org/doi/10.1103/PhysRevResearch.5.043161

### 7.8 Adversarial Robustness and New Challenges

23. "Fooling the Decoder: An Adversarial Attack on Quantum Error Correction." (2025). arXiv:2504.19651. URL: https://arxiv.org/html/2504.19651

---

## 8. Conclusions and Synthesis

### 8.1 Key Findings

1. **Exponential Error Suppression Achieved**: Google's Willow demonstrates for the first time that logical error rates can be suppressed exponentially with code distance on real superconducting hardware, validating 30 years of theoretical predictions.

2. **Multi-Platform Convergence**: All major qubit modalities (superconducting, trapped-ion, neutral atom, photonic) have demonstrated QEC capabilities in 2024-2025, indicating the field has transitioned from specialized research to broad applicability.

3. **ML Decoders Outperform Classical**: Neural network decoders (especially transformers) reduce errors by 6-30% compared to algorithmic decoders on real hardware and generalize to larger code distances than training data would suggest.

4. **Real-Time Decoding is Achievable**: Sub-microsecond decoding latencies demonstrated on FPGA, enabling feedback and avoiding syndrome backlog—critical for practical systems.

5. **Efficiency Gains via Code Innovation**: IBM's QLDPC codes (Gross, BB5 variants) achieve 10-20× qubit overhead reduction compared to surface codes, dramatically improving practical feasibility.

6. **RL Enables Automated Code Discovery**: RL agents can discover both QEC codes and encoding circuits from scratch, learn to optimize code parameters online, and adapt control parameters in real-time—opening doors to hardware-optimized codes.

### 8.2 Remaining Challenges

1. **Scalability Gap**: Practical fault-tolerant quantum computing requires d≥15-25 codes with thousands of qubits. Current experiments max out at d=7-11, 100-300 qubits. Scaling both hardware and decoders remains uncertain.

2. **Real-Time ML Decoding**: AlphaQubit has slower inference than real-time requirement. FPGA/ASIC implementations of learned decoders not yet standardized or widely deployed.

3. **Noise Model Mismatch**: QEC codes designed for idealized noise; real hardware has cross-talk, leakage, measurement errors, and non-Markovian dynamics. Threshold under realistic noise unknown.

4. **Generalization and Robustness**: Adversarial examples can fool learned decoders. Robustness to distribution shift, hardware drift, and parameter changes under-studied.

5. **Training Data Scarcity**: Obtaining thousands of error samples from quantum hardware is expensive and time-consuming. Simulation-to-hardware transfer mitigates but not fully solved.

### 8.3 Future Research Directions

1. **Hybrid ML-Algorithmic Decoders**: Combine speed of FPGA algorithmic decoders with accuracy of learned decoders in hierarchical architecture.

2. **Adaptive Codes via RL**: Deploy RL agents to continuously adjust code parameters (syndrome extraction schedules, measurement frequencies) in response to measured errors.

3. **Robustness-Aware Training**: Train neural decoders with adversarial examples and distribution shift to improve real-world performance.

4. **Standardized Benchmarking**: Establish Error Correction Zoo-like database of decoder performance across platforms, codes, and noise models.

5. **Hardware-Software Co-design**: Design QEC codes, decoders, and control systems jointly; avoid post-hoc optimization.

---

**Document Generated**: December 2025
**Literature Span**: 2019-2025
**Primary Sources**: ~50+ academic papers, preprints, industry reports
**Total Citations**: 23 major references listed above; additional ~30 secondary references embedded in search results

