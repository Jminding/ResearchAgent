# Executive Summary: QEC RL Decoder Analysis

**Date**: 2025-12-28
**Analyst**: Research Analyst Agent
**Total Experiments**: 162 configurations
**Status**: PRIMARY HYPOTHESIS REJECTED

---

## Bottom Line

**The RL-based quantum error correction decoder does NOT achieve the required >=20% improvement over MWPM at code distance d>=15.**

At d=15, the RL decoder performs **6.7% WORSE** than the classical MWPM baseline, with logical error rates of 51.5% vs 48.2%.

---

## Key Findings

### 1. Strong Small-Distance Performance
- **d=3**: RL achieves 57.5% improvement (L_RL=4.5% vs L_MWPM=10.9%)
- **d=5**: RL achieves 44.8% improvement (L_RL=17.5% vs L_MWPM=31.7%)
- **d=7**: RL achieves 30.3% improvement (L_RL=26.5% vs L_MWPM=38.0%)

**Verdict**: RL substantially outperforms MWPM at small distances.

### 2. Catastrophic Scaling Failure
- **d=11**: RL achieves only 16.9% improvement (below 20% threshold)
- **d=15**: RL achieves **-6.7% improvement** (RL WORSE than baseline!)
  - L_RL = 51.5% (seed 0: 47%, seed 1: 56%)
  - L_MWPM = 48.2% (seed 0: 46.6%, seed 1: 49.7%)

**Verdict**: RL fails to scale beyond d=7. Performance degrades monotonically with increasing distance.

### 3. Statistical Significance
- **Overall p-value**: 0.05 (not significant at alpha=0.01 level)
- **Mean improvement**: 26.2% across all distances (misleading - driven by d=3-7)
- **95% Confidence Interval**: [-8%, +60%] (includes zero and negative values)
- **Effect size (Cohen's d)**: 0.62 (medium)

**Verdict**: Not statistically significant with n=2 seeds. Requires n>=10 for adequate statistical power.

### 4. No Quantum Error Suppression Observed
Both RL and MWPM show **error amplification** rather than suppression:
- Error rates **increase** from d=3 to d=15 (should decrease exponentially)
- Suppression factors Lambda < 1 at most distances (wrong direction)
- Both decoders plateau at 40-50% error rates

**Interpretation**: System is operating **above error threshold**. Neither decoder is successfully performing quantum error correction at p=0.005.

### 5. Generalization Failure
- **Hypothesis**: Zero-shot transfer gap <15%
- **Observed**: d=7 to d=15 shows 94% degradation
- **Verdict**: REJECTED - RL does not learn transferable QEC principles

### 6. Critical Discrepancy from Expected Outcomes
- **Expected L_RL at d=15**: 0.0008-0.0015 (from experiment_plan.json)
- **Observed L_RL at d=15**: 0.515
- **Discrepancy factor**: **343x worse** than expected!

**Interpretation**: This 2-3 order of magnitude discrepancy suggests either:
1. Severe undertraining (200 vs 5-10M planned episodes), OR
2. Implementation errors in MWPM/syndrome generation, OR
3. Fundamental issues with the RL approach

---

## Root Cause Analysis

### Most Likely: Insufficient Training (75% confidence)
- Only **200 training episodes** used vs **5-10 million planned**
- RL shows clear learning at d=3-7 (proves concept works)
- Monotonic degradation with distance is signature of undertraining
- Sample complexity scales superlinearly with d

### Plausible Alternative Causes:
1. **Sparse reward signal** (50% confidence): Credit assignment problem in long syndrome chains
2. **Shallow GNN architecture** (45% confidence): Cannot propagate information across d=15 graph diameter
3. **Overfitting to specific distances** (40% confidence): No size-invariant representations learned
4. **Above-threshold operation** (25% confidence): p=0.005 may be too high for effective RL learning

---

## Recommendations

### Immediate Actions (Execute Now)
1. **Extend training to 1000-5000 steps** at d=15
   - Monitor learning curves for continued improvement
   - Target: L_RL < 0.38 (achieves 20% improvement)
   - Cost: 5-25x baseline (~4 hours compute)
   - **Priority: CRITICAL**

2. **Validate MWPM implementation**
   - Compare against PyMatching library reference
   - Verify Stim syndrome generation is correct
   - Check for implementation bugs causing anomalous 48% error rates
   - **Priority: CRITICAL**

3. **Increase statistical rigor**
   - Repeat d=15 experiments with 10 seeds (currently only 2)
   - Compute proper confidence intervals and p-values
   - **Priority: HIGH**

### Short-Term Actions (If Extended Training Fails)
4. **Test reward shaping** (dense vs sparse rewards)
5. **Test deeper architectures** (12-layer GNN or Graph Transformer)
6. **Implement curriculum learning** (progressive distance training)
7. **Conduct threshold analysis** (sweep p from 0.001 to 0.02)

### Long-Term Actions
8. If all follow-ups fail: Consider **hybrid RL+MWPM** or **alternative RL algorithms**
9. If follow-ups succeed: **Scale to d=21** and validate on **real quantum hardware** (Google Willow, IBM Quantum)
10. **Publish results** (positive or negative) in high-impact venue (Nature Physics, PRL, PRX Quantum)

---

## Decision Tree

```
START: Primary hypothesis rejected
    |
    v
Execute H1: Extend training to 1000-5000 steps at d=15
    |
    +--> L_RL < 0.38? --> YES --> SUCCESS: Extend to 5-10M, validate on d=21, publish
    |
    +--> NO --> Execute H2 (reward shaping) and H3 (architecture) in parallel
              |
              +--> Either succeeds? --> YES --> Iterate and scale up
              |
              +--> NO --> Execute H4 (curriculum learning)
                       |
                       +--> Succeeds? --> YES --> Validate and scale
                       |
                       +--> NO --> Execute H5 (threshold analysis + validation)
                                |
                                +--> Find bugs or lower threshold works? --> YES --> Fix and restart
                                |
                                +--> NO --> CONCLUSION: RL does not scale to d>=15
                                        --> Consider hybrid approaches or publish negative result
```

**Estimated success probability with full follow-up plan**: 60%

---

## Scientific Impact

### If Follow-Ups Succeed:
- **First demonstration** of RL outperforming classical decoders at practical distances (d>=15)
- Enables deployment on near-term quantum devices
- **High-impact publication** (Nature Physics, Physical Review X)
- Opens new research direction for ML-based QEC

### If Follow-Ups Fail:
- **Important negative result**: RL does not scale to practical QEC distances
- Identifies fundamental limitations of current RL+GNN approaches
- Motivates alternative architectures (hybrid, model-based RL, etc.)
- **Valuable publication** (Physical Review Letters - negative results have scientific value!)

---

## Cost and Timeline

### Immediate Follow-Ups (H1)
- **Compute cost**: ~$50
- **Time**: 1-2 days
- **Personnel**: 1 researcher

### Full Diagnostic Plan (H1-H5)
- **Compute cost**: ~$250
- **Time**: 8 days
- **Personnel**: 1-2 researchers

### Full-Scale Validation (if H1 succeeds)
- **Compute cost**: ~$12,000 (full experiment plan)
- **Time**: 3-6 weeks
- **Personnel**: 2-3 researchers + access to real quantum hardware (optional)

---

## Conclusion

**The primary hypothesis "RL achieves >=20% improvement at d>=15" is REJECTED based on current evidence.**

However, the strong performance at d=3-7 combined with the clear undertraining (200 vs 5-10M episodes) suggests the failure is likely **not fundamental** but rather due to **insufficient training**.

**Recommendation: Execute follow-up plan H1 (extended training) before declaring the approach infeasible.**

**Confidence in this recommendation: 75%**

---

## Files Generated

All analysis files are saved in `/files/results/`:

1. **analysis_summary.json** - Complete statistical analysis with all metrics
2. **comparison_rl_vs_mwpm.json** - Detailed RL vs MWPM comparison with p-values and CIs
3. **generalization_curves.json** - Distance-dependent performance and suppression analysis
4. **followup_plan.json** - 5 diagnostic hypotheses with experimental designs
5. **analysis_qec_comprehensive.md** - Full technical analysis (30+ pages)
6. **executive_summary.md** - This document

**Status**: Ready for review and execution approval

---

**Prepared by**: Research Analyst Agent
**Review required**: Lead Scientist / PI
**Next action**: Approve and execute followup_plan.json OR archive project

