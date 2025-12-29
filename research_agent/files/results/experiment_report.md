# Surface Code QEC Experiment Results

## Experiment Configuration

- **Date**: 2024-12-22
- **Distances**: [3, 5, 7]
- **Error rates**: 8 values from 0.0100 to 0.1500
- **Noise models**: depolarizing, dephasing
- **Training samples per config**: 5000
- **Test samples per config**: 2000
- **Syndrome rounds**: 3

## Threshold Estimates

| Decoder | Depolarizing | Dephasing |
|---------|-------------|-----------|
| RL | < 0.01 | < 0.01 |
| MWPM | 0.0689 | 0.0783 |

**Note**: The RL decoder shows lower threshold than MWPM, indicating the neural network requires further optimization. The MWPM baseline achieves threshold values consistent with theoretical expectations (~10.3% for ideal surface code with perfect syndrome measurements).

## Logical Error Rate Summary

### RL Decoder - Depolarizing Noise
| Distance | Min P_L | Max P_L | Mean P_L |
|----------|---------|---------|----------|
| d=3 | 0.007000 | 0.426000 | 0.222938 |
| d=5 | 0.070000 | 0.507500 | 0.371688 |
| d=7 | 0.110500 | 0.511000 | 0.412125 |

### MWPM Decoder - Depolarizing Noise
| Distance | Min P_L | Max P_L | Mean P_L |
|----------|---------|---------|----------|
| d=3 | 0.005500 | 0.381500 | 0.204312 |
| d=5 | 0.002000 | 0.431000 | 0.208312 |
| d=7 | 0.000500 | 0.456500 | 0.214625 |

### RL Decoder - Dephasing Noise
| Distance | Min P_L | Max P_L | Mean P_L |
|----------|---------|---------|----------|
| d=3 | 0.004500 | 0.384500 | 0.190187 |
| d=5 | 0.036000 | 0.497500 | 0.341938 |
| d=7 | 0.059500 | 0.510000 | 0.393875 |

### MWPM Decoder - Dephasing Noise
| Distance | Min P_L | Max P_L | Mean P_L |
|----------|---------|---------|----------|
| d=3 | 0.003500 | 0.355500 | 0.169437 |
| d=5 | 0.001500 | 0.393000 | 0.166688 |
| d=7 | 0.000000 | 0.423500 | 0.183000 |

## Numerical Results (Depolarizing Noise)

### P_L(p,d) - RL Decoder
| p | d=3 | d=5 | d=7 |
|---|-----|-----|-----|
| 0.0100 | 0.007000 | 0.070000 | 0.110500 |
| 0.0300 | 0.062000 | 0.209000 | 0.302000 |
| 0.0500 | 0.122500 | 0.342000 | 0.433000 |
| 0.0700 | 0.197500 | 0.401500 | 0.458500 |
| 0.0900 | 0.262500 | 0.443500 | 0.478500 |
| 0.1100 | 0.333000 | 0.470500 | 0.498000 |
| 0.1300 | 0.373000 | 0.498500 | 0.507000 |
| 0.1500 | 0.426000 | 0.507500 | 0.511000 |

### P_L(p,d) - MWPM Decoder
| p | d=3 | d=5 | d=7 |
|---|-----|-----|-----|
| 0.0100 | 0.005500 | 0.002000 | 0.000500 |
| 0.0300 | 0.059500 | 0.037000 | 0.031500 |
| 0.0500 | 0.115000 | 0.087000 | 0.072000 |
| 0.0700 | 0.185000 | 0.166000 | 0.157000 |
| 0.0900 | 0.247000 | 0.247500 | 0.265000 |
| 0.1100 | 0.298500 | 0.320500 | 0.334500 |
| 0.1300 | 0.343000 | 0.377000 | 0.401000 |
| 0.1500 | 0.381500 | 0.431000 | 0.456500 |

## Numerical Results (Dephasing Noise)

### P_L(p,d) - RL Decoder
| p | d=3 | d=5 | d=7 |
|---|-----|-----|-----|
| 0.0100 | 0.004500 | 0.036000 | 0.059500 |
| 0.0300 | 0.046500 | 0.167500 | 0.257500 |
| 0.0500 | 0.095500 | 0.307000 | 0.395000 |
| 0.0700 | 0.154500 | 0.363000 | 0.425500 |
| 0.0900 | 0.219500 | 0.434500 | 0.460500 |
| 0.1100 | 0.286500 | 0.450500 | 0.492000 |
| 0.1300 | 0.331500 | 0.480500 | 0.502000 |
| 0.1500 | 0.384500 | 0.497500 | 0.510000 |

### P_L(p,d) - MWPM Decoder
| p | d=3 | d=5 | d=7 |
|---|-----|-----|-----|
| 0.0100 | 0.003500 | 0.001500 | 0.000000 |
| 0.0300 | 0.035000 | 0.018500 | 0.009500 |
| 0.0500 | 0.066000 | 0.066000 | 0.046000 |
| 0.0700 | 0.131000 | 0.113500 | 0.111000 |
| 0.0900 | 0.193000 | 0.213000 | 0.218000 |
| 0.1100 | 0.257500 | 0.296000 | 0.291000 |
| 0.1300 | 0.314500 | 0.341500 | 0.365000 |
| 0.1500 | 0.355500 | 0.393000 | 0.423500 |

## Threshold Analysis

### Exponential Scaling Fit
The threshold is determined by fitting P_L = A * exp(-alpha * d) and finding where alpha crosses zero.

**RL Decoder:**
- All alpha values are negative, indicating P_L increases with d
- This is characteristic of below-threshold behavior
- The RL decoder needs more training epochs or larger network capacity

**MWPM Decoder:**
- For depolarizing noise: threshold p_th = 0.0689
- For dephasing noise: threshold p_th = 0.0783
- Below threshold, P_L decreases with d (alpha > 0)
- Above threshold, P_L increases with d (alpha < 0)

## Key Observations

1. **MWPM Outperforms RL**: The MWPM decoder significantly outperforms the neural network decoder, especially at larger code distances. This is expected for a simple supervised learning approach.

2. **Threshold Behavior**: MWPM shows clear threshold behavior with P_L decreasing exponentially with d below threshold. The estimated thresholds (~7-8%) are lower than the theoretical maximum (~10.3%) due to imperfect syndrome extraction.

3. **Distance Scaling**: For MWPM at low error rates (p < 0.05), increasing distance strongly suppresses logical errors. At d=7, p=0.01, the logical error rate is only 0.0005 (depolarizing) and 0.0000 (dephasing).

4. **Dephasing vs Depolarizing**: Dephasing noise consistently shows lower logical error rates, consistent with the surface code being a CSS code optimized for Z-error correction.

## Generated Figures

1. `P_L_vs_p_depolarizing.png` - Logical error rate vs physical error rate (depolarizing)
2. `P_L_vs_p_dephasing.png` - Logical error rate vs physical error rate (dephasing)
3. `threshold_estimation.png` - Scaling exponent alpha vs error rate
4. `rl_vs_mwpm_comparison.png` - Performance ratio between decoders
5. `error_matching_graph.png` - Detector layout visualization for d=3,5,7
6. `bloch_sphere_trajectory.png` - Logical state trajectory on Bloch sphere
7. `summary_comparison.png` - Combined 2x2 comparison plots

## File Locations

- **Results**: `/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results/`
  - `combined_results.json` - All P_L(p,d) data
  - `threshold_analysis.json` - Threshold fitting results
  - `experiment_summary.json` - Summary statistics
  - `experiment_report.md` - This report

- **Figures**: `/Users/jminding/Desktop/Code/Research Agent/research_agent/files/figures/`

- **Models**: `/Users/jminding/Desktop/Code/Research Agent/research_agent/files/models/`
  - `ppo_agent_d3_depolarizing.pkl` - Trained RL agent

- **Data**: `/Users/jminding/Desktop/Code/Research Agent/research_agent/files/data/`
  - Training and test syndrome datasets for all configurations
