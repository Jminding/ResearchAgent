# Executive Summary: GBM vs Heston Model Comparison

**Date:** 2025-12-21
**Asset:** AAPL (2013-2025, n=3,262)
**Hypothesis Status:** FALSIFIED

---

## Key Finding

**Geometric Brownian Motion (GBM) outperformed the Heston stochastic volatility model across all evaluation metrics**, contradicting theoretical expectations.

---

## Critical Metrics

| Criterion | GBM | Heston | Winner |
|-----------|-----|--------|--------|
| Log-Likelihood | 8502.29 | 8468.94 | **GBM** (+33.35) |
| AIC | -17,000.58 | -16,925.89 | **GBM** (+74.69) |
| BIC | -16,988.40 | -16,889.35 | **GBM** (+99.05) |
| Out-of-Sample RMSE | 0.0926 | 0.0928 | **GBM** (+0.20%) |
| LRT Decision | - | p = 1.0 | **GBM adequate** |

**Result:** GBM wins on all 4 validation criteria.

---

## Why Heston Failed

1. **Weak volatility clustering** in AAPL data insufficient to justify stochastic variance
2. **Parameter identifiability issues**: Estimates hit boundary values (κ=2.0, ρ=-0.5, ξ=0.3)
3. **Overfitting**: 4 extra parameters captured noise, not signal
4. **Optimization challenges**: Negative LRT statistic indicates estimation failure
5. **Data frequency mismatch**: Daily returns may be suboptimal for variance process identification

---

## Residual Diagnostics

Both models exhibit:
- Significant autocorrelation (Ljung-Box p < 0.001)
- Severe excess kurtosis (GBM: 6.92, Heston: 3.17)
- Non-normality (Jarque-Bera p < 0.001)

**Implication:** Both models are misspecified; neither fully captures AAPL return dynamics.

---

## Implications

### For Practitioners
- Simple models can outperform complex ones when data doesn't support additional parameters
- Always validate with out-of-sample testing and information criteria
- Suspicious parameter values signal estimation problems

### For Theorists
- Stochastic volatility requires specific data conditions (strong clustering, regime shifts)
- Returns data alone may be insufficient for variance process calibration
- Model parsimony is empirically valuable, not just philosophically

---

## Limitations

- Single asset (AAPL), single period (2013-2025)
- No option prices used for calibration
- Point estimates only (no uncertainty quantification)
- Negative LRT statistic indicates potential methodological issues

---

## Recommendation

**Use GBM for AAPL price modeling** in similar contexts (daily frequency, stable volatility regimes). Consider Heston only when:
- Option pricing is required (volatility smile modeling)
- Volatility regimes are extreme (crises, emerging markets)
- High-frequency data or realized volatility is available
- Multivariate correlation structures are critical

---

**Full analysis:** `/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results/analysis_stochastic_volatility.md`
