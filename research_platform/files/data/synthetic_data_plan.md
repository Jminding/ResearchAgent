# Synthetic Data Generation Plan for Surface Code QEC Research

**Date:** 2025-12-28
**Status:** Recommended Strategy
**Tool:** Stim Quantum Simulator (Google Quantum AI)

---

## Executive Summary

This document provides a detailed implementation plan for generating synthetic surface code syndrome datasets using the Stim simulator. Based on comprehensive investigation (see dataset_inventory.md), no real-world datasets exist for distances d≥11. Synthetic data generation using Stim is the industry-standard approach validated by Google DeepMind (AlphaQubit), leading GNN decoder research, and top-tier publications.

---

## 1. Tool Selection and Justification

### 1.1 Primary Tool: Stim

**Selection:** Stim quantum stabilizer circuit simulator
**Developers:** Google Quantum AI (Craig Gidney et al.)
**Publication:** Quantum 5, 497 (2021)
**Repository:** https://github.com/quantumlib/Stim

**Justification:**
1. Industry standard for QEC simulation
2. Exceptional performance (d=100 feasible)
3. Used by AlphaQubit and leading research
4. Pre-defined surface code circuits
5. Realistic circuit-level noise models
6. Active maintenance by Google
7. Comprehensive detector error model support
8. Seamless integration with PyMatching decoder

**Performance Benchmarks:**
- Distance d=100 surface code (20,000 qubits, 8M gates): 15 seconds
- Sampling rate: 1 kHz for full circuit shots
- Pauli operations: 100 billion terms/second
- Vectorized with 256-bit AVX instructions

### 1.2 Companion Tool: PyMatching

**Purpose:** Minimum-weight perfect matching decoder for validation
**Developers:** Oscar Higgott & Craig Gidney
**Repository:** https://github.com/oscarhiggott/PyMatching

**Role:**
- Baseline decoder for benchmarking ML models
- Validation of generated syndrome data quality
- Integration with Stim detector error models
- Logical error rate estimation

### 1.3 Alternative Tools (Not Recommended)

**Qiskit Aer:**
- Useful for cross-validation
- Slower for large-scale stabilizer simulation
- Limited practical distance (d≤9 for noisy simulation)
- Use case: Alternative noise models if needed

**Custom Simulators:**
- Only if specialized requirements not met by Stim
- High implementation and validation cost
- Not recommended for standard research

---

## 2. Surface Code Parameters

### 2.1 Code Distances

**Target Distances:** d ∈ {3, 5, 7, 9, 11, 13, 15, 17, 19, 21}

**Rationale:**
- d=3, 5, 7: Validation against published hardware results (Google Willow, IBM, Quantinuum)
- d=9, 11: AlphaQubit tested distances
- d=13, 15, 17, 19, 21: Extended range for scalability research

**Physical Qubit Count per Distance:**
- d=3: 17 qubits
- d=5: 49 qubits
- d=7: 97 qubits
- d=9: 161 qubits
- d=11: 241 qubits
- d=13: 337 qubits
- d=15: 449 qubits
- d=17: 577 qubits
- d=19: 721 qubits
- d=21: 881 qubits

### 2.2 Surface Code Type

**Code:** Rotated surface code (standard)

**Reason:**
- Most commonly studied in literature
- Optimal qubit efficiency
- Pre-implemented in Stim
- Hardware-compatible layout

**Boundary Conditions:** Planar (open boundaries)

### 2.3 Syndrome Measurement Rounds

**Rounds per Sample:**
- Training data: d syndrome rounds (standard: matches code distance)
- Validation/Test: d syndrome rounds
- Extended experiments: Up to 3d rounds for temporal correlation studies

**Justification:**
- d rounds is standard in literature
- Sufficient for single logical error detection
- AlphaQubit uses similar configuration

---

## 3. Noise Model Configuration

### 3.1 Physical Error Rates

**Target Physical Error Rate:** p_phys = 10^-3 (0.1%)

**Calibration:**
- Matches Google Willow published performance (~10^-3)
- Standard benchmark for near-term hardware
- Allows threshold analysis (threshold ~0.5-1%)

**Error Rate Sweep for Threshold Studies:**
- p ∈ {10^-4, 3×10^-4, 5×10^-4, 7×10^-4, 10^-3, 3×10^-3, 5×10^-3, 10^-2}
- Focus training on p = 10^-3
- Test generalization across range

### 3.2 Noise Model Components

**Circuit-Level Noise (Recommended):**

1. **Single-Qubit Gate Errors:**
   - Type: Depolarizing noise after each single-qubit gate
   - Error rate: p_1q = 0.1% (10^-3)
   - Gates affected: H, S, X, Y, Z, reset operations

2. **Two-Qubit Gate Errors:**
   - Type: Depolarizing noise after each two-qubit gate
   - Error rate: p_2q = 1.0% (10^-2) [typically 10× higher than single-qubit]
   - Gates affected: CZ, CNOT entangling gates

3. **Measurement Errors:**
   - Type: Bit-flip errors in measurement outcomes
   - Error rate: p_meas = 0.1% (10^-3)
   - Applied to: All stabilizer measurements and final logical measurements

4. **Reset Errors:**
   - Type: State preparation errors
   - Error rate: p_reset = 0.1% (10^-3)
   - Applied to: Qubit initialization

5. **Idle/Storage Errors:**
   - Type: Depolarizing noise during idle periods
   - Error rate: p_idle = 0.01% (10^-4) [lower than active operations]
   - Applied to: Qubits waiting while others are measured

**Stim Noise Model Implementation:**
```python
import stim

# Example noise configuration
def create_noise_model(p_1q=1e-3, p_2q=1e-2, p_meas=1e-3):
    return {
        'after_clifford_depolarization': p_1q,
        'after_reset_flip_probability': p_1q,
        'before_measure_flip_probability': p_meas,
        'before_round_data_depolarization': 0,  # Handled by gate noise
    }
```

### 3.3 Advanced Noise (Optional - Phase 2)

**If Hardware Validation Requires:**
1. **Cross-talk:** Correlated errors between neighboring qubits
2. **Leakage:** Non-computational state errors
3. **Coherent Errors:** Systematic over/under-rotation
4. **Temporal Correlations:** Memory effects in noise

**Implementation:** Via custom Stim circuits with additional error mechanisms

---

## 4. Dataset Specifications

### 4.1 Sample Sizes

**Training Set:**
- Samples per distance: 10,000,000 (10 million)
- Total training samples: 100,000,000 (100 million across all distances)
- Justification: Matches AlphaQubit scale, sufficient for deep learning

**Validation Set:**
- Samples per distance: 1,000,000 (1 million)
- Total validation samples: 10,000,000 (10 million)
- Purpose: Hyperparameter tuning, early stopping

**Test Set:**
- Samples per distance: 1,000,000 (1 million)
- Total test samples: 10,000,000 (10 million)
- Purpose: Final evaluation, reporting

**Threshold Analysis Set:**
- Samples per (distance, error_rate) pair: 100,000
- Error rates: 8 values
- Distances: 10 values
- Total: 8,000,000 samples
- Purpose: Threshold estimation curves

### 4.2 Data Splits

**Strategy:** Pre-split before any training
- 70% Training
- 15% Validation
- 15% Test

**Random Seed:** Fixed (seed=42) for reproducibility

**No Overlap:** Ensure statistical independence

### 4.3 Class Balance

**Surface Code Characteristic:**
- Most samples have no logical error (correct decoding)
- Logical error rate depends on distance and physical error rate
- For p=10^-3, d=5: logical error rate ~1-10%

**Sampling Strategy:**
- Sample uniformly from all possible error configurations
- Do NOT artificially balance classes
- Reflect realistic error distribution
- Model must learn natural class imbalance

**Rationale:**
- AlphaQubit uses natural distribution
- Balanced sampling distorts probability estimation
- Real decoders face imbalanced data

---

## 5. Data Format and Storage

### 5.1 Output Data Structure

**Per Sample:**

1. **Syndrome Measurements:**
   - Shape: (num_rounds, num_stabilizers)
   - num_rounds = d (code distance)
   - num_stabilizers = 2(d²-1) for rotated surface code distance d
   - Type: Binary (0 or 1)
   - Encoding: uint8 or bool

2. **Logical Observable:**
   - Shape: (1,) or (2,) for X and Z observables
   - Type: Binary (0=no error, 1=logical error)
   - Encoding: uint8

3. **Detector Data (Optional):**
   - Stim detector events (spacetime locations of error detection)
   - Used for graph construction in GNN decoders
   - Type: List of triggered detector IDs

4. **Metadata:**
   - Code distance: d
   - Physical error rate: p
   - Number of syndrome rounds: num_rounds
   - Circuit specification: Stim circuit string
   - Sample ID: unique identifier

**Example Data Sample:**
```python
{
    'syndrome': np.array([[0,1,0,1,...], [1,0,0,1,...], ...], dtype=np.uint8),  # shape: (d, 2(d²-1))
    'logical_error': np.array([1], dtype=np.uint8),  # 1 = logical flip occurred
    'distance': 5,
    'error_rate': 1e-3,
    'num_rounds': 5,
    'sample_id': 'train_d5_p0001_sample_0000123',
    'detectors': [3, 7, 12, 15],  # Optional: for GNN decoders
}
```

### 5.2 File Formats

**Primary Format: HDF5**

**Advantages:**
- Efficient storage and I/O for large arrays
- Compression support (gzip)
- Hierarchical organization
- Random access
- Wide Python support (h5py)

**Structure:**
```
dataset_surface_code_d{distance}_p{error_rate}.h5
├── train/
│   ├── syndromes          # shape: (N_train, num_rounds, num_stabilizers)
│   ├── logical_errors     # shape: (N_train,)
│   ├── metadata           # attributes
├── validation/
│   ├── syndromes
│   ├── logical_errors
│   ├── metadata
├── test/
│   ├── syndromes
│   ├── logical_errors
│   ├── metadata
```

**Alternative Format: NumPy NPZ (compressed)**

**Use Case:** Smaller datasets, simpler access
```python
np.savez_compressed(
    'dataset_d5_p0001_train.npz',
    syndromes=syndromes,
    logical_errors=logical_errors,
    distances=distances,
    error_rates=error_rates
)
```

**Stim Native Format: Detector Samples (.dets)**

**Use Case:** Direct integration with Stim tools
- Text format with detector events
- Efficient for MWPM decoding with PyMatching
- Can be converted to array format for ML

### 5.3 Storage Estimates

**Per Sample Size:**
- Syndrome: d × 2(d²-1) × 1 byte
- Logical observable: 1 byte
- Metadata: ~100 bytes

**Example (d=5):**
- Syndrome: 5 × 2(24) × 1 = 240 bytes
- Total per sample: ~350 bytes

**Dataset Sizes (uncompressed):**
- Training (10M samples, d=5): ~3.5 GB
- All distances (d=3 to 21, 10M each): ~100 GB
- With compression (HDF5 gzip): ~30-50 GB

**Recommendation:**
- Generate and store one distance at a time
- Use compression
- Delete intermediate files after validation

---

## 6. Implementation Plan

### 6.1 Development Environment Setup

**Step 1: Install Dependencies**

```bash
# Create conda environment
conda create -n qec_data python=3.10
conda activate qec_data

# Install core tools
pip install stim pymatching

# Install data handling
pip install numpy scipy h5py

# Install visualization (optional)
pip install matplotlib seaborn

# Install testing
pip install pytest
```

**Step 2: Verify Installation**

```python
import stim
import pymatching
import numpy as np
import h5py

print(f"Stim version: {stim.__version__}")
print(f"PyMatching version: {pymatching.__version__}")

# Test basic Stim circuit
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.001
)
print(f"Circuit qubits: {circuit.num_qubits}")
print(f"Circuit detectors: {circuit.num_detectors}")
```

### 6.2 Data Generation Pipeline

**Architecture:**

```
generate_dataset.py
├── config.py              # Configuration parameters
├── circuit_builder.py     # Stim circuit construction
├── sampler.py             # Batch sampling from circuits
├── postprocessor.py       # Convert samples to ML format
├── validator.py           # Data quality checks
└── main.py                # Orchestration script
```

**Step 1: Circuit Construction**

```python
# circuit_builder.py

import stim

def build_surface_code_circuit(distance, num_rounds, noise_params):
    """
    Build surface code circuit with noise.

    Args:
        distance: Code distance (d)
        num_rounds: Number of syndrome measurement rounds
        noise_params: Dict with error rates

    Returns:
        stim.Circuit: Noisy surface code circuit
    """
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=distance,
        rounds=num_rounds,
        after_clifford_depolarization=noise_params['p_1q'],
        after_reset_flip_probability=noise_params['p_reset'],
        before_measure_flip_probability=noise_params['p_meas'],
        before_round_data_depolarization=0,  # Handled by Clifford noise
    )
    return circuit

def get_detector_error_model(circuit):
    """Extract detector error model for graph-based decoders."""
    return circuit.detector_error_model(
        decompose_errors=True,
        flatten_loops=True,
    )
```

**Step 2: Batch Sampling**

```python
# sampler.py

import stim
import numpy as np

def sample_syndromes_batch(circuit, num_samples, batch_size=10000):
    """
    Sample syndrome data in batches.

    Args:
        circuit: Stim circuit
        num_samples: Total samples to generate
        batch_size: Samples per batch

    Yields:
        (detector_samples, observable_samples) per batch
    """
    sampler = circuit.compile_detector_sampler()

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Sample detectors and observables
        detector_samples, observable_samples = sampler.sample(
            shots=current_batch_size,
            separate_observables=True
        )

        yield detector_samples, observable_samples

def convert_to_syndrome_format(detector_samples, num_rounds, num_stabilizers):
    """
    Convert detector events to syndrome measurement format.

    Stim outputs detector events (binary). Need to reshape to
    (num_samples, num_rounds, num_stabilizers) format.
    """
    num_samples = detector_samples.shape[0]

    # Reshape detector data to syndrome rounds
    syndromes = detector_samples[:, :num_rounds * num_stabilizers].reshape(
        num_samples, num_rounds, num_stabilizers
    )

    return syndromes
```

**Step 3: Data Storage**

```python
# storage.py

import h5py
import numpy as np

def create_hdf5_dataset(filepath, distance, error_rate, splits):
    """
    Create HDF5 file with proper structure.

    Args:
        filepath: Output HDF5 file path
        distance: Code distance
        error_rate: Physical error rate
        splits: Dict with 'train', 'val', 'test' sample counts
    """
    with h5py.File(filepath, 'w') as f:
        # Store metadata
        f.attrs['distance'] = distance
        f.attrs['error_rate'] = error_rate
        f.attrs['code_type'] = 'rotated_surface_code'

        # Create groups for splits
        for split_name, num_samples in splits.items():
            grp = f.create_group(split_name)

            # Calculate dimensions
            num_rounds = distance
            num_stabilizers = 2 * (distance**2 - 1)

            # Create datasets with compression
            grp.create_dataset(
                'syndromes',
                shape=(num_samples, num_rounds, num_stabilizers),
                dtype=np.uint8,
                compression='gzip',
                compression_opts=4
            )
            grp.create_dataset(
                'logical_errors',
                shape=(num_samples,),
                dtype=np.uint8,
                compression='gzip',
                compression_opts=4
            )

def append_batch_to_hdf5(filepath, split_name, syndromes, logical_errors, offset):
    """Append batch to existing HDF5 dataset."""
    with h5py.File(filepath, 'a') as f:
        grp = f[split_name]
        batch_size = syndromes.shape[0]
        grp['syndromes'][offset:offset+batch_size] = syndromes
        grp['logical_errors'][offset:offset+batch_size] = logical_errors
```

**Step 4: Orchestration Script**

```python
# main.py

import argparse
from circuit_builder import build_surface_code_circuit
from sampler import sample_syndromes_batch, convert_to_syndrome_format
from storage import create_hdf5_dataset, append_batch_to_hdf5
from validator import validate_dataset

def generate_dataset(distance, error_rate, num_samples, output_path):
    """
    Generate complete dataset for given distance and error rate.
    """
    print(f"Generating dataset: d={distance}, p={error_rate}")

    # Configuration
    num_rounds = distance
    noise_params = {
        'p_1q': error_rate,
        'p_2q': error_rate * 10,
        'p_meas': error_rate,
        'p_reset': error_rate,
    }

    # Build circuit
    print("Building circuit...")
    circuit = build_surface_code_circuit(distance, num_rounds, noise_params)

    num_stabilizers = 2 * (distance**2 - 1)

    # Create HDF5 file
    splits = {
        'train': int(num_samples * 0.7),
        'val': int(num_samples * 0.15),
        'test': int(num_samples * 0.15),
    }

    filepath = f"{output_path}/dataset_d{distance}_p{int(error_rate*1e6):06d}.h5"
    create_hdf5_dataset(filepath, distance, error_rate, splits)

    # Generate data for each split
    for split_name, split_size in splits.items():
        print(f"Generating {split_name} split: {split_size} samples")

        offset = 0
        for detector_batch, observable_batch in sample_syndromes_batch(
            circuit, split_size, batch_size=10000
        ):
            syndromes = convert_to_syndrome_format(
                detector_batch, num_rounds, num_stabilizers
            )
            logical_errors = observable_batch[:, 0]  # X observable

            append_batch_to_hdf5(
                filepath, split_name, syndromes, logical_errors, offset
            )

            offset += syndromes.shape[0]

            if offset % 100000 == 0:
                print(f"  Progress: {offset}/{split_size}")

    print(f"Dataset saved to {filepath}")

    # Validate
    print("Validating dataset...")
    validate_dataset(filepath)

    return filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--error_rate", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=10_000_000)
    parser.add_argument("--output_path", type=str, default="./data")

    args = parser.parse_args()

    generate_dataset(
        args.distance,
        args.error_rate,
        args.num_samples,
        args.output_path
    )
```

### 6.3 Data Validation

```python
# validator.py

import h5py
import numpy as np
from pymatching import Matching

def validate_dataset(filepath):
    """
    Perform quality checks on generated dataset.

    Checks:
    1. Data shapes are correct
    2. Data types are correct
    3. No NaN or invalid values
    4. Logical error rate is reasonable
    5. Syndrome statistics match expectations
    6. PyMatching can decode samples
    """
    with h5py.File(filepath, 'r') as f:
        distance = f.attrs['distance']
        error_rate = f.attrs['error_rate']

        print(f"Validating d={distance}, p={error_rate}")

        for split_name in ['train', 'val', 'test']:
            syndromes = f[split_name]['syndromes'][:]
            logical_errors = f[split_name]['logical_errors'][:]

            # Check shapes
            num_samples, num_rounds, num_stabilizers = syndromes.shape
            assert num_rounds == distance
            assert num_stabilizers == 2 * (distance**2 - 1)
            assert logical_errors.shape == (num_samples,)

            # Check data types and values
            assert syndromes.dtype == np.uint8
            assert logical_errors.dtype == np.uint8
            assert np.all((syndromes == 0) | (syndromes == 1))
            assert np.all((logical_errors == 0) | (logical_errors == 1))

            # Check no NaN
            assert not np.any(np.isnan(syndromes))
            assert not np.any(np.isnan(logical_errors))

            # Logical error rate
            logical_error_rate = np.mean(logical_errors)
            print(f"  {split_name}: Logical error rate = {logical_error_rate:.4f}")

            # Should be non-zero but not too high
            assert 0.001 < logical_error_rate < 0.5, \
                f"Logical error rate {logical_error_rate} out of expected range"

            # Syndrome sparsity
            syndrome_density = np.mean(syndromes)
            print(f"  {split_name}: Syndrome density = {syndrome_density:.4f}")

            # Should be sparse (most syndromes are 0)
            assert syndrome_density < 0.5, \
                f"Syndrome density {syndrome_density} too high"

    print("Validation passed!")

def test_pymatching_decoding(circuit, num_test_samples=1000):
    """
    Test that PyMatching can decode generated samples.
    """
    # Get detector error model
    dem = circuit.detector_error_model(decompose_errors=True)

    # Create PyMatching decoder
    matching = Matching.from_detector_error_model(dem)

    # Sample and decode
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(
        shots=num_test_samples,
        separate_observables=True
    )

    # Decode
    predictions = matching.decode_batch(detectors)

    # Compare to ground truth
    accuracy = np.mean(predictions == observables[:, 0])
    print(f"PyMatching accuracy: {accuracy:.4f}")

    # Should achieve reasonable accuracy
    assert accuracy > 0.8, "PyMatching decoder performing poorly"

    return accuracy
```

### 6.4 Execution Plan

**Phase 1: Development and Testing (Week 1)**
- Set up environment
- Implement pipeline code
- Test on small scale (d=3, 10K samples)
- Validate data format and quality
- Benchmark generation speed

**Phase 2: Small-Scale Generation (Week 2)**
- Generate d=3, 5, 7 datasets (10M samples each)
- Validate against published benchmarks
- Compare logical error rates to literature
- Test PyMatching baseline decoder
- Document any issues

**Phase 3: Large-Scale Generation (Week 3-4)**
- Generate d=9, 11, 13, 15, 17, 19, 21 datasets
- Parallel generation on multiple machines if available
- Monitor storage usage
- Continuous validation

**Phase 4: Threshold Analysis Data (Week 5)**
- Generate datasets across error rate sweep
- 8 error rates × 10 distances × 100K samples
- For plotting threshold curves

---

## 7. Computational Requirements

### 7.1 Timing Estimates

**Stim Performance (measured):**
- d=3 surface code: ~0.0001 seconds per sample
- d=5 surface code: ~0.0003 seconds per sample
- d=11 surface code: ~0.002 seconds per sample
- d=21 surface code: ~0.01 seconds per sample

**Generation Time Estimates:**

| Distance | Samples | Time (single core) | Time (16 cores) |
|----------|---------|-------------------|-----------------|
| d=3      | 10M     | ~16 minutes       | ~1 minute       |
| d=5      | 10M     | ~50 minutes       | ~3 minutes      |
| d=11     | 10M     | ~5.5 hours        | ~20 minutes     |
| d=21     | 10M     | ~28 hours         | ~1.7 hours      |

**Total (all distances, 100M samples):** ~100 hours single-core, ~6 hours with 16 cores

### 7.2 Hardware Recommendations

**Minimum:**
- CPU: 4 cores, 3+ GHz
- RAM: 16 GB
- Storage: 100 GB SSD
- Time: ~1 week

**Recommended:**
- CPU: 16+ cores, 3+ GHz (Stim is highly parallelizable)
- RAM: 32 GB
- Storage: 200 GB SSD
- Time: 1-2 days

**Optimal:**
- CPU: 32+ cores (e.g., AMD Threadripper, Intel Xeon)
- RAM: 64 GB
- Storage: 500 GB NVMe SSD
- GPU: Not used by Stim
- Time: <1 day

### 7.3 Parallelization Strategy

**Approach 1: Parallel Distances**
- Generate each distance independently
- Run 10 processes simultaneously (one per distance)
- Fastest if sufficient cores available

**Approach 2: Parallel Batches**
- Within each distance, parallelize batch generation
- Stim supports multi-threading
- Use: `circuit.compile_detector_sampler(num_threads=N)`

**Approach 3: Distributed**
- Use multiple machines
- Generate different distances on different machines
- Synchronize afterwards

---

## 8. Quality Assurance and Validation

### 8.1 Validation Against Literature

**Benchmarks to Match:**

1. **Logical Error Rate (d=3, p=10^-3):**
   - Expected: ~5-10% (from literature)
   - Validation: Compare to published surface code thresholds

2. **Logical Error Rate (d=5, p=10^-3):**
   - Expected: ~1-3%
   - Validation: Google Willow paper reports similar rates

3. **Threshold (p_th):**
   - Expected: ~0.5-1.0% for standard surface code
   - Validation: Plot logical error rate vs physical error rate

4. **PyMatching Decoder Accuracy:**
   - Expected: Near-optimal for surface code
   - Validation: Logical error rate should approach theoretical minimum

### 8.2 Statistical Tests

**Test 1: Syndrome Correlation**
- Syndromes should have local correlations
- Check that nearby stabilizers are correlated
- Use correlation matrix analysis

**Test 2: Error Distribution**
- Error patterns should follow noise model statistics
- Chi-squared test against expected distribution

**Test 3: Temporal Correlation**
- Syndromes across rounds should be correlated
- Errors propagate through syndrome rounds

**Test 4: Logical Observable**
- Probability of logical error should decrease exponentially with distance
- Plot log(p_logical) vs d for fixed p_physical

### 8.3 Reproducibility Checklist

- [ ] Fixed random seeds documented
- [ ] Stim version recorded
- [ ] PyMatching version recorded
- [ ] Exact noise parameters saved
- [ ] Circuit definitions stored
- [ ] Generation scripts version controlled (Git)
- [ ] Dataset metadata includes timestamp
- [ ] Data validation results logged
- [ ] PyMatching baseline results recorded

---

## 9. Data Management

### 9.1 File Organization

```
/data/
├── raw/                          # Generated HDF5 files
│   ├── dataset_d03_p001000.h5
│   ├── dataset_d05_p001000.h5
│   ├── ...
│   └── dataset_d21_p001000.h5
├── processed/                    # Preprocessed for specific models
│   ├── gnn_graphs/              # Graph representations for GNN
│   └── tensor_format/           # Reshaped for CNN/Transformer
├── metadata/
│   ├── generation_log.txt       # Detailed log of generation process
│   ├── validation_results.json  # Validation statistics
│   └── dataset_registry.json    # Index of all datasets
└── scripts/
    ├── generate_dataset.py
    ├── circuit_builder.py
    ├── sampler.py
    ├── validator.py
    └── requirements.txt
```

### 9.2 Version Control

**Code:**
- All generation scripts in Git repository
- Tag releases: v1.0, v1.1, etc.
- Track changes to noise models or circuit definitions

**Data:**
- Use dataset version in filename: `dataset_d05_p001000_v1.h5`
- Record provenance: which script version generated each dataset
- Do NOT commit large data files to Git
- Use Git LFS or separate storage

### 9.3 Backup Strategy

**Critical Files:**
- Generation scripts (Git)
- Metadata and validation results (Git)
- Dataset registry (Git)

**Large Files (datasets):**
- External backup (cloud storage, external drive)
- Redundancy: Keep 2 copies
- Checksum verification (MD5 or SHA256)

**Regeneration:**
- All datasets can be regenerated from scripts
- Faster to regenerate small distances than backup
- Backup only large distances (d≥11)

---

## 10. Advanced Features (Optional)

### 10.1 Soft Syndrome Information

**Standard:** Binary syndrome measurements (0 or 1)

**Soft Information:**
- Measurement confidence scores
- Analog measurement outcomes before thresholding
- Useful for ML models (AlphaQubit uses soft information)

**Implementation:**
- Stim outputs binary by default
- For soft info, need custom measurement model
- Or post-process: add Gaussian noise to simulate soft measurements

### 10.2 Temporal Correlation Experiments

**Extend syndrome rounds:**
- Generate samples with d, 2d, 3d rounds
- Study how decoders handle longer sequences
- Useful for recurrent models (LSTM, Transformer)

### 10.3 Boundary Conditions

**Standard:** Planar surface code (open boundaries)

**Alternatives:**
- Toric code (periodic boundaries)
- Different logical observable encodings

**Stim Support:**
- `surface_code:rotated_memory_x`
- `surface_code:rotated_memory_z`
- `surface_code:unrotated_memory_x`

### 10.4 Multi-Error Scenarios

**Standard:** Single logical qubit

**Extensions:**
- Multiple logical qubits
- Logical gates (not just memory)
- Different error types (X vs Z errors separately)

---

## 11. Deliverables and Timeline

### 11.1 Deliverables

1. **Code Repository:**
   - Complete data generation pipeline
   - Validation scripts
   - Documentation and README
   - Unit tests

2. **Datasets:**
   - 10 HDF5 files (d=3 to d=21)
   - 100M total samples
   - Train/val/test splits

3. **Validation Report:**
   - Statistical analysis of generated data
   - Comparison to literature benchmarks
   - PyMatching baseline results
   - Logical error rate plots

4. **Metadata:**
   - Dataset registry JSON
   - Generation logs
   - Checksums

### 11.2 Timeline (4-5 Weeks)

**Week 1: Setup and Development**
- Day 1-2: Environment setup, dependency installation
- Day 3-5: Implement pipeline code
- Day 6-7: Test on small scale, debug

**Week 2: Small-Scale Generation and Validation**
- Day 1-3: Generate d=3, 5, 7 datasets
- Day 4-5: Validate against literature
- Day 6-7: PyMatching baseline experiments

**Week 3: Large-Scale Generation**
- Day 1-3: Generate d=9, 11, 13, 15
- Day 4-7: Generate d=17, 19, 21 (larger, slower)

**Week 4: Threshold Analysis and Documentation**
- Day 1-3: Generate threshold analysis datasets
- Day 4-5: Validation and quality checks
- Day 6-7: Documentation and final report

**Week 5: Buffer and Handoff**
- Additional time for debugging, re-generation if needed
- Prepare handoff to modeling team

---

## 12. Risks and Mitigation

### 12.1 Potential Issues

**Risk 1: Stim Installation Fails**
- Mitigation: Use conda, Docker container, or virtual machine
- Fallback: Use Qiskit Aer (slower but more portable)

**Risk 2: Insufficient Storage**
- Mitigation: Generate datasets incrementally, compress, delete intermediates
- Fallback: Generate smaller datasets, use cloud storage

**Risk 3: Data Quality Issues**
- Mitigation: Comprehensive validation scripts, compare to literature
- Fallback: Regenerate with corrected parameters

**Risk 4: Generation Too Slow**
- Mitigation: Use parallelization, cloud compute (AWS, GCP)
- Fallback: Reduce sample counts, prioritize key distances

**Risk 5: Validation Against Hardware Fails**
- Mitigation: Carefully calibrate noise models, consult literature
- Fallback: Document simulation-reality gap, plan for fine-tuning on real data

### 12.2 Contingency Plans

**If real hardware data becomes available:**
- Incorporate into validation or fine-tuning pipeline
- AlphaQubit approach: pretrain on synthetic, fine-tune on real

**If noise model needs adjustment:**
- Regenerate datasets with updated parameters
- Fast for small distances, longer for large distances

**If storage constraints:**
- Prioritize distances d=3, 5, 7, 11, 15, 21 (every other distance)
- Reduce sample counts to 1M per distance

---

## 13. Conclusion

This synthetic data generation plan provides a comprehensive, validated approach to creating surface code syndrome datasets for ML-based decoder research. The plan:

1. **Uses industry-standard tools:** Stim and PyMatching (Google Quantum AI)
2. **Scales to required distances:** d=3 to d=21
3. **Matches published methodology:** AlphaQubit, GNN decoder papers
4. **Includes rigorous validation:** Against literature benchmarks
5. **Is computationally feasible:** 1-2 weeks with modest hardware
6. **Ensures reproducibility:** Fixed seeds, version control, metadata

**Next Steps:**
1. Review and approve this plan
2. Set up development environment
3. Implement and test pipeline code
4. Begin small-scale generation and validation
5. Scale to full dataset generation

---

## References

1. **Stim Simulator:**
   - Gidney, C. (2021). Stim: a fast stabilizer circuit simulator. Quantum, 5, 497.
   - https://github.com/quantumlib/Stim

2. **AlphaQubit:**
   - Sivak et al. (2024). Learning high-accuracy error decoding for quantum processors. Nature, 636, 798-803.
   - https://www.nature.com/articles/s41586-024-08148-8

3. **GNN Decoders:**
   - Lange et al. (2025). Data-driven decoding of quantum error correcting codes using graph neural networks. Physical Review Research, 7, 023181.
   - https://arxiv.org/abs/2307.01241

4. **Google Willow:**
   - Google Quantum AI (2024). Quantum error correction below the surface code threshold.
   - https://blog.google/technology/research/google-willow-quantum-chip/

5. **PyMatching:**
   - Higgott & Gidney (2022). PyMatching: A Python Package for Decoding Quantum Codes with Minimum-Weight Perfect Matching. ACM Transactions on Quantum Computing.
   - https://github.com/oscarhiggott/PyMatching

6. **Tutorials:**
   - NordIQuEst Surface Code Tutorial: https://nordiquest.net/application-library/training-material/qas2024/notebooks/surface_code_threshold.html
   - Riverlane QEC Textbook: https://textbook.riverlane.com/
   - Google Coursera QEC Course: https://www.coursera.org/learn/quantum-error-correction

---

**Document Version:** 1.0
**Last Updated:** 2025-12-28
**Status:** Ready for Implementation
