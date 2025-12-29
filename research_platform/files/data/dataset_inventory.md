# Quantum Error Correction Dataset Inventory

**Date:** 2025-12-28
**Prepared by:** Data Acquisition Specialist
**Purpose:** Comprehensive inventory of available QEC datasets for surface code ML decoder research

---

## Executive Summary

After extensive investigation, **no public real-world surface code syndrome datasets exist for distances d≥11 that meet the experiment requirements**. Most published QEC research relies on synthetic data generated using circuit-level noise simulators, primarily Google's Stim. Real hardware data from Google Willow, IBM Quantum, and Quantinuum is limited to distances d≤7 and is not publicly available as downloadable datasets.

**Recommendation:** Use synthetic data generation via Stim simulator (Google's open-source tool) combined with realistic noise models calibrated to match hardware characteristics.

---

## 1. Real Hardware Datasets

### 1.1 Google Sycamore Quantum Supremacy Data
- **Source:** Google Quantum AI / Nature (2019)
- **URL:** https://www.nature.com/articles/s41586-019-1666-5
- **Data Repository:** Dryad (publicly accessible)
- **Description:** Experimental data from quantum supremacy demonstration
- **Specifications:**
  - Processor: 53 qubits (Sycamore)
  - Task: Random circuit sampling, not surface code QEC
  - Format: Raw measurement data
- **Limitations:**
  - Not surface code syndrome data
  - Limited to specific benchmarking circuits
  - Does not include syndrome measurements for QEC codes
- **License:** Creative Commons (via Nature publication)
- **Relevance Score:** 2/10 (architectural reference only)

### 1.2 Google Willow - Surface Code Below Threshold
- **Source:** Google Quantum AI (December 2024)
- **URL:** https://blog.google/technology/research/google-willow-quantum-chip/
- **Nature Paper:** "Quantum error correction below the surface code threshold"
- **Specifications:**
  - Processor: 105 qubits (Willow)
  - Code distances tested: d=3, d=5, d=7
  - Achievement: Exponential error suppression with scaling
- **Data Availability:** NOT PUBLICLY RELEASED AS DATASET
  - Only aggregate results in Nature paper
  - No downloadable syndrome measurement data
  - No public GitHub repository with raw data
- **Limitations:**
  - Maximum distance d=7 (insufficient for d≥11 requirement)
  - Proprietary hardware access required
  - Data only available in aggregated form in published paper
- **Relevance Score:** 6/10 (validates approach but data unavailable)

### 1.3 IBM Quantum Experience - Surface Code Implementations
- **Source:** IBM Quantum
- **URL:** https://quantum.cloud.ibm.com/
- **GitHub:** https://github.com/uuudown/QASMBench (benchmark suite)
- **Specifications:**
  - Heavy-hexagonal lattice architecture
  - Rotated surface code implementations
  - Magic state injection protocols tested
  - Code distance: primarily d=3, d=5
- **Data Availability:** LIMITED
  - Access requires IBM Quantum account
  - No pre-collected surface code syndrome dataset
  - Users must run experiments on live hardware or simulators
  - QASMBench provides benchmark circuits but not syndrome data
- **Limitations:**
  - No datasets for d≥11
  - Hardware queue times variable
  - Noise characteristics change over time
- **Relevance Score:** 5/10 (platform accessible but no ready datasets)

### 1.4 Quantinuum H2 - 4D Surface Code
- **Source:** Quantinuum (2024)
- **URL:** https://www.quantinuum.com/
- **GitHub:** https://github.com/CQCL/quantinuum-hardware-h2-benchmark
- **arXiv:** https://arxiv.org/html/2408.08865 (4D surface code experiments)
- **Specifications:**
  - Processor: H2 trapped-ion quantum computer (56 qubits)
  - Implementations: 2D and 4D surface codes
  - First hardware demonstration of single-shot QEC
  - Code distances: d=3, d=5
- **Data Availability:** PARTIAL
  - H2 benchmark repository contains general performance data
  - 4D surface code paper includes experimental results
  - Raw syndrome measurement data NOT publicly released
- **Limitations:**
  - Maximum distance tested d=5
  - Proprietary hardware
  - Dataset focuses on benchmarking metrics, not training data
- **Relevance Score:** 5/10 (innovative but limited availability)

---

## 2. Research Paper Datasets

### 2.1 AlphaQubit (Google DeepMind + Google Quantum AI)
- **Source:** Nature (November 2024)
- **URL:** https://www.nature.com/articles/s41586-024-08148-8
- **PMC (Open Access):** https://pmc.ncbi.nlm.nih.gov/articles/PMC11602728/
- **Specifications:**
  - Architecture: Recurrent transformer neural network
  - Codes tested: Surface code, repetition code
  - Distances: d=3, d=5 (real hardware), up to d=11 (simulation)
  - Training data: 100 million+ synthetic samples + limited experimental data
  - Hardware: Google Sycamore processor experimental data
- **Data Generation Method:**
  - Pretraining: Stim-generated synthetic data with circuit-level noise
  - Finetuning: Limited experimental samples from Sycamore
  - Detector error models calibrated using cross-entropy benchmarking
- **Data Availability:** NOT DIRECTLY RELEASED
  - Paper has CC-BY 4.0 license (code/data sharing encouraged)
  - No explicit GitHub repository linked in search results
  - Supplementary materials available through Nature (subscription may be required)
  - Data availability statement in paper should be checked
- **Relevance Score:** 9/10 (highly relevant, methodology described, but data not packaged)

### 2.2 GNN-Based QEC Decoders
- **Source:** Multiple papers (2023-2025)
- **Key Papers:**
  - Lange et al. "Data-driven decoding of quantum error correcting codes using graph neural networks" (arXiv:2307.01241, Phys. Rev. Research 2025)
  - "Benchmarking Machine Learning Models for Quantum Error Correction" (arXiv:2311.11167)
- **URLs:**
  - https://arxiv.org/abs/2307.01241
  - https://arxiv.org/html/2311.11167v2
  - https://github.com/gongaa/Feedback-GNN (post-processing QLDPC codes)
- **Specifications:**
  - Dataset sizes: Up to 100 million synthetic samples
  - Surface code and repetition code
  - Code distances: d=3, d=5, d=7, d=9, d=11
  - Noise models: Circuit-level noise with depolarizing channels
- **Data Generation Method:**
  - Stim simulator with detector error models
  - Pauli noise models calibrated to hardware
  - Circuit-level noise including cross-talk and leakage
- **Data Availability:** PARTIAL
  - Code repositories available for some projects
  - Datasets typically generated on-the-fly during training
  - No centralized dataset repository identified
  - Researchers expected to regenerate data using provided scripts
- **Relevance Score:** 8/10 (methodologies available, datasets regenerable)

---

## 3. Synthetic Data Generation Tools (AVAILABLE)

### 3.1 Stim - Google's Stabilizer Circuit Simulator
- **Source:** Google Quantum AI
- **GitHub:** https://github.com/quantumlib/Stim
- **Documentation:** https://quantum-journal.org/papers/q-2021-07-06-497/
- **PyPI:** `pip install stim`
- **Specifications:**
  - Performance: Distance d=100 surface code (20K qubits, 8M gates) analyzed in 15 seconds
  - Sampling rate: 1 kHz for full circuit shots
  - Vectorization: 256-bit AVX instructions
  - Pauli string operations: 100 billion terms/second
- **Features:**
  - Pre-defined surface code circuits
  - Circuit-level noise models
  - Detector error model (DEM) generation
  - Integration with PyMatching decoder
  - Syndrome sampling at scale
- **Data Generation Capability:**
  - Generate syndrome data for any distance d (tested up to d=100)
  - Flexible noise models (depolarizing, coherent, circuit-level)
  - Realistic detector error models matching hardware
  - Export syndrome + logical observable data
- **Tutorials Available:**
  - https://nordiquest.net/application-library/training-material/qas2024/notebooks/surface_code_threshold.html
  - https://textbook.riverlane.com/en/latest/notebooks/ch5-decoding-surfcodes/simulating-surface-codes-stim.html
  - Coursera course: https://www.coursera.org/learn/quantum-error-correction
- **License:** Open source (Apache 2.0)
- **Relevance Score:** 10/10 (RECOMMENDED PRIMARY TOOL)

### 3.2 PyMatching - Minimum Weight Perfect Matching Decoder
- **Source:** Oscar Higgott & Craig Gidney
- **GitHub:** https://github.com/oscarhiggott/PyMatching
- **Documentation:** https://pymatching.readthedocs.io/
- **PyPI:** `pip install pymatching`
- **Purpose:** Decoder for validating generated syndrome data
- **Integration:** Works seamlessly with Stim DetectorErrorModel
- **Features:**
  - Constructs matching graphs from Stim DEMs
  - Decodes syndrome measurements to error predictions
  - Benchmark baseline for ML decoder comparison
- **License:** Open source
- **Relevance Score:** 10/10 (essential companion to Stim)

### 3.3 Qiskit Aer - IBM's Quantum Simulator
- **Source:** IBM Quantum
- **GitHub:** https://github.com/Qiskit/qiskit-aer
- **Documentation:** https://qiskit.github.io/qiskit-aer/
- **PyPI:** `pip install qiskit-aer`
- **Specifications:**
  - Noise models: Device-calibrated, custom
  - Simulator backends: Statevector, density matrix, stabilizer
  - Noise channels: Depolarizing, amplitude damping, phase damping, etc.
- **QEC Capability:**
  - Can simulate surface codes with custom circuits
  - NoiseModel class for realistic noise
  - Syndrome extraction via stabilizer measurements
- **Limitations:**
  - Slower than Stim for large-scale stabilizer simulation
  - More general-purpose (not QEC-specialized)
  - Limited to smaller distances for noisy simulation
- **Use Case:** Alternative for cross-validation, different noise models
- **License:** Open source (Apache 2.0)
- **Relevance Score:** 7/10 (useful supplementary tool)

### 3.4 Custom PauliSum Simulator
- **Approach:** Lightweight Pauli-frame simulation
- **Advantages:**
  - Simple implementation
  - Fast for specific noise models
  - Full control over physics
- **Disadvantages:**
  - Requires implementation from scratch
  - Less validated than Stim
  - No ecosystem of tools
- **Use Case:** Research into novel noise models or code structures
- **Relevance Score:** 4/10 (only if specialized needs)

---

## 4. Public Dataset Repositories Searched (No QEC Datasets Found)

### 4.1 Kaggle
- **URL:** https://www.kaggle.com/datasets
- **Search Results:** "Quantum Dataset" exists but not QEC-specific
- **Finding:** No surface code syndrome datasets available
- **Relevance:** General quantum ML datasets exist (QDataSet project)

### 4.2 UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/ml
- **Search Results:** No quantum error correction datasets
- **Finding:** UCI focuses on classical ML benchmarks

### 4.3 GitHub Topics
- **Search:** "quantum-error-correction" topic
- **Finding:** Code repositories and tools, not datasets
- **Notable Repos:**
  - MQT QECC (Munich Quantum Toolkit)
  - Various decoder implementations
  - No centralized dataset repositories

### 4.4 ArXiv Datasets
- **Finding:** Papers describe methodology but datasets not archived
- **Pattern:** QEC research relies on regenerable synthetic data

---

## 5. Dataset Gaps and Justification for Synthetic Data

### 5.1 Why No Real Datasets for d≥11?

1. **Hardware Limitations:**
   - Current quantum hardware: 50-1000 qubits
   - Surface code overhead: d² qubits per logical qubit
   - Distance d=11 requires 121 physical qubits
   - Distance d=21 requires 441 physical qubits
   - Full experiments with multiple syndrome rounds difficult

2. **Noise Variability:**
   - Hardware noise drifts over time
   - Pre-collected datasets become stale
   - Real-time calibration preferred for experiments

3. **Access Restrictions:**
   - Cutting-edge hardware proprietary (Google, IBM, Quantinuum)
   - Limited public access to large-scale experiments
   - Computational cost of data collection high

4. **Research Practice:**
   - QEC field standard: generate synthetic data locally
   - Simulators (especially Stim) are extremely fast and accurate
   - Synthetic data enables controlled experiments
   - Hardware validation done with small-scale experiments

### 5.2 Simulation-Reality Gap Considerations

**Known Differences:**
- Synthetic data assumes perfect knowledge of noise model
- Real hardware has correlated errors, cross-talk, leakage
- Timing and control pulse imperfections in real systems
- Measurement assignment errors more complex

**Mitigation Strategies:**
1. Use circuit-level noise (not just Pauli channels)
2. Incorporate cross-talk and leakage in noise models
3. Calibrate noise parameters to match published hardware data
4. Fine-tune models on small real hardware samples (AlphaQubit approach)
5. Validate on available d=3, d=5 real data before scaling

**Field Consensus:**
- Stim-generated data with realistic noise models is accepted methodology
- AlphaQubit and leading GNN decoder papers use this approach
- Hardware validation done at smaller scales

---

## 6. Recommended Data Strategy

### Phase 1: Synthetic Data Generation (Distances d=3 to d=21)
- **Tool:** Stim simulator
- **Noise Model:** Circuit-level depolarizing noise + measurement error
- **Calibration:** Match published hardware error rates (Willow: ~10⁻³)
- **Sample Sizes:**
  - Training: 10-100 million samples per distance
  - Validation: 1 million samples per distance
  - Test: 1 million samples per distance
- **Distances:** d ∈ {3, 5, 7, 9, 11, 13, 15, 17, 19, 21}
- **Output Format:**
  - Syndrome measurements (binary arrays)
  - Logical observable outcomes (labels)
  - Detector error models (for graph construction)

### Phase 2: Cross-Validation with Available Real Data
- **Sources:**
  - Google Willow published results (d=3, 5, 7) - aggregate comparison
  - IBM Quantum Experience experiments (small scale)
  - Literature benchmarks
- **Validation Metrics:**
  - Logical error rate vs published hardware results
  - Threshold estimation comparison
  - Decoder accuracy on reported syndromes

### Phase 3: Documentation and Reproducibility
- **Version Control:** Git track all generation scripts
- **Random Seeds:** Fixed for reproducibility
- **Metadata:** Document noise parameters, sample counts, code parameters
- **Storage:** HDF5 or NPZ format for efficient I/O

---

## 7. Summary Table

| Dataset / Tool | Type | Distances | Availability | Size | Relevance | Status |
|---------------|------|-----------|--------------|------|-----------|--------|
| Google Willow Data | Real | d=3,5,7 | Not Released | N/A | High | Unavailable |
| IBM Quantum Exp | Real | d=3,5 | Account Req | Variable | Medium | Limited Access |
| Quantinuum H2 | Real | d=3,5 | Not Released | N/A | Medium | Unavailable |
| AlphaQubit Data | Synthetic+Real | d≤11 | Not Released | 100M+ | Very High | Methodology Only |
| GNN Decoder Data | Synthetic | d≤11 | Regenerable | 100M+ | High | Scripts Available |
| Stim Generator | Synthetic | Any d | Open Source | Unlimited | Critical | AVAILABLE |
| PyMatching | Decoder | Any d | Open Source | N/A | Critical | AVAILABLE |
| Qiskit Aer | Synthetic | d≤9 | Open Source | Unlimited | Medium | AVAILABLE |

---

## 8. Conclusion

**Primary Finding:** No suitable real-world datasets exist for surface codes with d≥11.

**Recommended Approach:**
Synthetic data generation using Stim simulator with circuit-level noise models calibrated to match published hardware error rates. This is the standard approach used by Google DeepMind (AlphaQubit), leading GNN decoder research, and the broader QEC community.

**Justification:**
1. Stim is the gold standard tool (developed by Google Quantum AI)
2. Can generate unlimited training data at all required distances
3. Methodology validated by top-tier research (Nature publications)
4. Hardware validation feasible at smaller scales (d=3, 5, 7)
5. Simulation-reality gap manageable with realistic noise models

**Next Steps:**
1. Implement Stim-based data generation pipeline (see synthetic_data_plan.md)
2. Validate noise model calibration against literature
3. Generate initial datasets for d=3, 5, 7 and compare to published benchmarks
4. Scale to larger distances d=11, 15, 21

---

## References

See data_sources.json for complete URL list and access details.

**Key Publications:**
- Stim simulator: Quantum 5, 497 (2021)
- AlphaQubit: Nature 636, 798-803 (2024)
- Google Willow: Nature (2024)
- GNN decoders: Phys. Rev. Research 7, 023181 (2025)

**Last Updated:** 2025-12-28
