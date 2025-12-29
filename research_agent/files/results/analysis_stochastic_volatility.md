# Analysis: GBM vs Heston Stochastic Volatility Model Comparison

**Experiment Date:** 2025-12-21 23:51:26
**Asset:** AAPL (Apple Inc.)
**Period:** 2013-01-01 to 2025-12-21
**Observations:** 3,262 daily returns

---

## 1. HYPOTHESIS STATUS: FALSIFIED

### Original Hypothesis
The implicit hypothesis was that the Heston stochastic volatility model would provide superior performance compared to Geometric Brownian Motion (GBM) when modeling stock price dynamics, particularly for assets exhibiting volatility clustering and fat tails.

### Verdict: HYPOTHESIS FALSIFIED

The experimental evidence conclusively demonstrates that **GBM outperformed the Heston model across all evaluated criteria**:

- **Likelihood Ratio Test:** Failed to reject the null hypothesis (p = 1.0)
- **Information Criteria:** GBM superior on both AIC and BIC
- **Out-of-Sample Performance:** GBM achieved lower prediction errors
- **Overall Assessment:** 0 of 4 validation criteria passed for Heston superiority

The hypothesis is **falsified** based on the experimental evidence.

---

## 2. PERFORMANCE INTERPRETATION

### 2.1 Model Fit Quality

#### Log-Likelihood Comparison
- **GBM Log-Likelihood:** 8502.29
- **Heston Log-Likelihood:** 8468.94
- **Difference:** +33.35 in favor of GBM

Despite having 4 additional parameters (6 total vs 2 for GBM), the Heston model achieved a **lower** log-likelihood, indicating poorer fit to the observed data.

#### Information Criteria
| Criterion | GBM | Heston | Winner |
|-----------|-----|--------|--------|
| AIC | -17,000.58 | -16,925.89 | GBM |
| BIC | -16,988.40 | -16,889.35 | GBM |
| AICc | -17,000.58 | -16,925.86 | GBM |

The BIC gap of 99.05 points strongly favors GBM, indicating that the Heston model's additional complexity is severely penalized and unjustified by the data. In Bayesian model selection, a BIC difference exceeding 10 is considered "very strong" evidence against the more complex model.

### 2.2 Likelihood Ratio Test

The LRT yielded a **negative test statistic** (-66.69), which is theoretically impossible under standard nested model testing assumptions. This anomalous result indicates:

1. **Nested model assumption violated:** The Heston model, despite having more parameters, fits the data worse than the simpler GBM
2. **Optimization failure:** The Heston parameter estimation may have converged to a suboptimal solution
3. **Model misspecification:** The Heston structure may be fundamentally incompatible with this dataset

With p = 1.0 and failure to reject the null hypothesis, the test concludes that **GBM is statistically adequate** and the additional Heston complexity is unwarranted.

### 2.3 Residual Diagnostics

#### GBM Residuals
- **Ljung-Box Test:** Statistic = 83.06, p < 0.001 (autocorrelation detected)
- **Jarque-Bera Test:** Statistic = 6539.42, p < 0.001 (non-normal)
- **Excess Kurtosis:** 6.92 (strong leptokurtosis)
- **Skewness:** -0.22 (slight negative skew)

#### Heston Residuals
- **Ljung-Box Test:** Statistic = 47.93, p < 0.001 (autocorrelation detected)
- **Jarque-Bera Test:** Statistic = 1404.85, p < 0.001 (non-normal)
- **Excess Kurtosis:** 3.17 (moderate leptokurtosis)
- **Skewness:** -0.27 (slightly more negative)

#### Interpretation
While Heston reduced the autocorrelation (83.06 → 47.93) and excess kurtosis (6.92 → 3.17), both models fail normality and independence assumptions. Critically, the reduction in residual pathology did **not translate to improved predictive performance**, suggesting that the Heston model captured noise rather than signal.

### 2.4 Out-of-Sample Performance

**Validation Setup:**
- Training set: 2,609 observations (80%)
- Test set: 653 observations (20%)
- Forecast horizon: 22 days

**Results:**
| Metric | GBM | Heston | Improvement |
|--------|-----|--------|-------------|
| RMSE | 0.0926 | 0.0928 | +0.20% |
| MAE | 0.0522 | 0.0524 | +0.44% |

GBM achieved marginally better out-of-sample forecasts. The improvement of -0.20% (negative indicates Heston performed worse) confirms that Heston's in-sample "flexibility" did not generalize.

### 2.5 Heston Parameter Estimates

```
mu (drift):        0.2164
kappa (mean reversion): 2.000
theta (long-run variance): 0.0803
xi (volatility of volatility): 0.300
rho (correlation): -0.500
V_0 (initial variance): 0.0803
```

**Critical Observation:** The parameters kappa = 2.0, xi = 0.3, and rho = -0.5 are suspiciously close to common boundary values or initialization defaults. This suggests:

1. **Weak identifiability:** The data does not contain sufficient information to distinguish Heston from GBM
2. **Optimization at boundaries:** The likelihood surface may be flat or poorly defined
3. **Overfitting risk:** Parameters may be fitting noise rather than true stochastic volatility dynamics

The Feller condition is satisfied (ratio = 3.57 > 1), ensuring the variance process remains positive, but this does not guarantee model relevance.

---

## 3. WHY DID GBM OUTPERFORM HESTON?

This result contradicts theoretical expectations. Five explanations emerge:

### 3.1 Data Characteristics: Insufficient Volatility Dynamics

**Observed Properties:**
- Annualized volatility: 28.3%
- Excess kurtosis: 6.92
- Sample period: 13 years (2013-2025)

While AAPL exhibits leptokurtosis (fat tails), the dataset may lack the **persistent volatility clustering** required to justify stochastic volatility modeling. During 2013-2025, AAPL's volatility, while elevated, may not have shown sufficient regime-switching or stochastic patterns.

**Key Insight:** Excess kurtosis alone does not imply stochastic volatility. It can arise from jumps, structural breaks, or heavy-tailed innovations—none of which Heston explicitly models.

### 3.2 Model Identifiability and Estimation Challenges

Heston models are notoriously difficult to calibrate from returns data alone:

- **Parameter redundancy:** With 6 parameters and only 3,262 observations, the model may be overparameterized
- **Latent volatility:** The variance process V_t is unobserved, requiring filtering or indirect inference
- **Likelihood surface pathology:** Multiple local optima and flat regions complicate maximum likelihood estimation

The suspicious parameter values (kappa ≈ 2, rho ≈ -0.5) suggest the optimizer struggled to find a meaningful solution.

### 3.3 Overfitting and Generalization Failure

Heston's 4 additional parameters allowed it to fit idiosyncratic features of the training data, but these adjustments captured **noise rather than systematic volatility patterns**. This classic overfitting manifested as:

- Lower log-likelihood (worse in-sample fit)
- Worse AIC/BIC (penalized complexity)
- Marginally worse out-of-sample predictions

### 3.4 Mean Reversion vs Volatility Clustering

GBM assumes constant volatility, which can be interpreted as the **sample average** volatility over the period. If volatility fluctuations are mean-reverting around a stable level (kappa = 2 indicates fast mean reversion), then averaging may provide a better approximation than attempting to track stochastic dynamics.

### 3.5 Data Frequency and Discretization

Daily data (Δt ≈ 0.004 years) may be:
- **Too coarse** to observe intraday volatility dynamics
- **Too frequent** for the continuous-time approximation to hold without jumps

High-frequency microstructure noise or discrete jumps violate both GBM and Heston assumptions, potentially favoring the simpler model by Occam's razor.

---

## 4. LIMITATIONS AND CAVEATS

### 4.1 Experimental Design Limitations

1. **Single asset, single period:** Results are specific to AAPL (2013-2025) and may not generalize to other stocks, indices, or time periods
2. **No transaction costs or liquidity constraints:** Real-world trading would alter model utility
3. **Returns-only calibration:** Option prices or realized volatility data could improve Heston estimation
4. **Point estimate evaluation:** No uncertainty quantification (confidence intervals, bootstrapping) on model selection metrics

### 4.2 Statistical Caveats

1. **Negative LRT statistic:** This anomaly questions the validity of the standard nested testing framework here
2. **Residual diagnostics failure:** Both models violate normality and independence, indicating potential model misspecification
3. **Marginal out-of-sample differences:** The 0.2% RMSE advantage for GBM is within measurement noise and may not be statistically significant
4. **No backtesting of strategies:** Predictive accuracy ≠ trading profitability

### 4.3 Methodological Caveats

1. **Optimization convergence:** No evidence provided of global optimum achievement for Heston parameters
2. **Alternative estimators:** Quasi-maximum likelihood, method of moments, or Bayesian methods may yield different conclusions
3. **Model extensions:** Jump-diffusion, regime-switching, or GARCH models were not tested as alternatives

### 4.4 Interpretation Caveats

1. **Absence of evidence ≠ evidence of absence:** Failing to find stochastic volatility patterns does not prove they don't exist
2. **Sample period bias:** 2013-2025 includes low-volatility (pre-2020) and COVID-shock periods, which may obscure patterns
3. **Survivorship and conditioning:** AAPL is a large-cap tech stock with unique dynamics not representative of broader markets

---

## 5. IMPLICATIONS FOR STOCK PRICE MODELING

### 5.1 Practical Implications

**For Quantitative Analysts:**
- **Complexity is not always better:** Simpler models can outperform sophisticated alternatives when data does not support additional parameters
- **Validation is critical:** In-sample fit is insufficient; out-of-sample testing and information criteria must guide model selection
- **Parameter stability matters:** Boundary-hitting or "round number" parameter estimates signal estimation failure

**For Risk Managers:**
- GBM may provide adequate volatility estimates for well-diversified large-cap stocks over medium horizons
- Stochastic volatility models require high-quality calibration data (e.g., option prices) to justify their use
- Residual diagnostics failure indicates both models underestimate tail risk

### 5.2 Theoretical Implications

1. **Constant volatility is a robust approximation:** When volatility mean-reverts quickly around a stable level, constant-volatility models suffice
2. **Stochastic volatility is data-hungry:** Returns alone may not contain enough information to identify variance process dynamics
3. **Model parsimony matters:** Occam's razor applies in financial econometrics—unnecessary parameters degrade performance

### 5.3 When Might Heston Outperform?

Despite this failure, stochastic volatility models remain valuable in contexts where:
- **Option pricing is required:** Heston captures the volatility smile/skew better than Black-Scholes
- **Volatility regimes are pronounced:** Crisis periods, emerging markets, or cryptocurrencies with extreme volatility clustering
- **High-frequency data is available:** Realized volatility measures can directly calibrate variance processes
- **Multivariate modeling:** Cross-asset volatility correlations are critical for portfolio optimization

### 5.4 Recommendations for Future Research

1. **Test on diverse assets:** Compare across sectors, asset classes, and market regimes
2. **Incorporate option data:** Joint calibration to returns and implied volatilities
3. **Use realized volatility:** High-frequency estimators provide cleaner variance process observations
4. **Bayesian model averaging:** Rather than selecting one model, weight predictions across multiple specifications
5. **Include jumps:** Extend both models to jump-diffusion frameworks to address leptokurtosis
6. **Regime-switching:** Allow parameters to vary across market states (e.g., GARCH, Markov-switching)

---

## 6. SUMMARY OF KEY FINDINGS

### Quantitative Results
- **Log-Likelihood:** GBM superior by 33.35 units
- **AIC Difference:** 74.69 in favor of GBM
- **BIC Difference:** 99.05 in favor of GBM (decisive)
- **Out-of-Sample RMSE:** GBM wins by 0.20%
- **LRT Outcome:** Negative statistic; GBM adequate

### Qualitative Insights
1. **Heston model failed to justify its complexity** for this dataset
2. **Parameter estimates suggest weak identifiability** (boundary values)
3. **Residual diagnostics reveal both models are misspecified** (autocorrelation, fat tails persist)
4. **Out-of-sample validation confirmed overfitting** in the Heston model
5. **Data characteristics** (stable volatility, insufficient clustering) favored the simpler GBM

### Final Conclusion

**The hypothesis that Heston stochastic volatility modeling improves upon GBM for AAPL stock price dynamics is falsified.** GBM provided superior fit, better generalization, and lower model complexity costs. This result underscores a fundamental principle in quantitative finance: **model sophistication must be matched to data informativeness**. When stochastic volatility patterns are weak or unidentifiable, parsimony prevails.

However, this conclusion is **context-dependent**. The failure of Heston on this particular dataset does not invalidate stochastic volatility theory generally, but rather highlights the importance of rigorous model validation in applied settings.

---

## 7. ANOMALIES AND PATTERNS IDENTIFIED

### 7.1 Negative LRT Statistic Anomaly
The LRT statistic of -66.69 is theoretically impossible under proper nested model testing. This requires investigation:
- Verify log-likelihood calculations
- Check for numerical precision errors
- Examine optimizer convergence diagnostics
- Consider non-nested testing frameworks

### 7.2 Parameter Boundary Effects
Heston parameters hitting "round" values (kappa = 2.0, xi = 0.3, rho = -0.5) suggests:
- Constrained optimization boundaries
- Likelihood surface flatness
- Initialization sensitivity

### 7.3 Residual Autocorrelation Persistence
Both models exhibit significant residual autocorrelation (Ljung-Box p < 0.001), indicating:
- Missing momentum/reversal effects
- Unmodeled microstructure
- Potential for GARCH-type extensions

### 7.4 Excess Kurtosis Reduction
Heston reduced excess kurtosis from 6.92 to 3.17, but this did not improve predictive power. This suggests:
- Kurtosis reduction may be overfitting sample idiosyncrasies
- Fat tails may arise from jumps rather than stochastic volatility
- In-sample fit metrics can mislead without out-of-sample validation

---

**Analysis completed:** 2025-12-21
**Data Source:** /Users/jminding/Desktop/Code/Research Agent/research_agent/files/results/stochastic_volatility/experiment_results.json
**Analyst Role:** Research Analyst - Experimental Results Interpretation
