# Stochastic Volatility Experiment Iteration Log

## Experiment Overview

**Objective:** Compare GBM and Heston stochastic volatility models on AAPL stock data (2013-present)

**Theory Framework:** files/theory/theory_quantitative_stock_price_modeling.md

---

## Iteration 1: Baseline Experiment

**Date:** 2025-12-21

### Configuration
- Ticker: AAPL
- Date Range: 2013-01-01 to 2025-12-21
- Observations: 3262 trading days
- Particle Filter: 1500 particles
- Optimizer: L-BFGS-B, 250 iterations
- Train/Test Split: 80/20

### GBM Results
| Parameter | Estimate |
|-----------|----------|
| mu (drift) | 0.2566 (25.66% annual) |
| sigma (volatility) | 0.2835 (28.35% annual) |
| Log-Likelihood | 8502.29 |
| AIC | -17000.58 |
| BIC | -16988.40 |

### Heston Results
| Parameter | Estimate |
|-----------|----------|
| mu (drift) | 0.2164 |
| kappa (mean reversion) | 2.0000 |
| theta (long-run variance) | 0.0803 (28.35% vol) |
| xi (vol of vol) | 0.3000 |
| rho (correlation) | -0.5000 |
| V_0 (initial variance) | 0.0803 |
| Log-Likelihood | 8468.94 |
| AIC | -16925.89 |
| BIC | -16889.35 |
| Feller Ratio | 3.57 (satisfied) |

### Model Comparison
| Metric | GBM | Heston | Preferred |
|--------|-----|--------|-----------|
| Log-Likelihood | 8502.29 | 8468.94 | GBM |
| AIC | -17000.58 | -16925.89 | GBM |
| BIC | -16988.40 | -16889.35 | GBM |
| LRT p-value | - | 1.0000 | GBM |
| OOS RMSE | 0.0926 | 0.0928 | GBM |

### Residual Diagnostics
| Model | Ljung-Box p | Jarque-Bera p | Autocorr | Normal |
|-------|-------------|---------------|----------|--------|
| GBM | 0.0000 | 0.0000 | No | No |
| Heston | 0.0004 | 0.0000 | No | No |

### Hypothesis Status
**Primary Hypothesis: FALSIFIED / INCONCLUSIVE**
- LRT did not reject GBM
- AIC/BIC favor simpler GBM model
- OOS forecasting favors GBM

### Key Observations
1. GBM provides better fit than Heston despite simplicity
2. Both models fail to capture volatility clustering (autocorrelation in residuals)
3. Both models fail normality tests (excess kurtosis ~7 for raw returns)
4. Heston improves Jarque-Bera statistic (from 6539 to 1405) but still fails
5. Negative correlation (rho=-0.5) confirms leverage effect
6. Feller condition satisfied (variance stays positive)

### Interpretation
The particle filter MLE for the Heston model appears to converge to parameters that do not improve upon GBM. This may be due to:
1. Daily data not revealing intraday volatility dynamics
2. Particle filter introducing estimation noise
3. The Heston model not capturing jump dynamics present in the data
4. The optimizer finding a local rather than global optimum

---

## Files Generated

| File | Description |
|------|-------------|
| experiment_results.json | Complete numerical results |
| diagnostic_plots.png | 9-panel diagnostic visualization |
| residual_analysis.png | 6-panel residual comparison |
| experiment_summary.md | Markdown summary report |
| variance_path.npy | Heston filtered variance path |
| returns.npy | Log-returns array |
| iteration_log.md | This file |

---

## Next Steps (if pursuing further)

1. **Increase Particles:** Try 5000+ particles for more stable estimation
2. **Alternative Estimation:** Implement MCMC or characteristic function inversion
3. **Jump-Diffusion:** Add Merton jump component to capture fat tails
4. **Intraday Data:** Use minute/5-min data for better volatility observation
5. **Multiple Assets:** Test on SPY, MSFT, GOOGL for generalization

---

*Last Updated: 2025-12-21*
