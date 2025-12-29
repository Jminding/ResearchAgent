# Quick Reference Guide: Stock Price Model Validation

**Purpose:** Rapid lookup for formulas, test procedures, and decision rules for validating stock price prediction models.

---

## 1. Residual Diagnostic Tests

### 1.1 Ljung-Box Q-Test (Autocorrelation)

**Null Hypothesis:** H₀: ρ₁ = ρ₂ = ... = ρₕ = 0 (no autocorrelation at lags 1 through H)

**Test Statistic:**
```
Q_LB = T(T+2) * Σ(ρ²ₖ / (T-k))  for k = 1 to H
```

**Distribution:** χ²(H) under H₀

**Interpretation:**
- p-value > 0.05: Fail to reject H₀; residuals likely white noise.
- p-value < 0.05: Reject H₀; significant autocorrelation detected.

**Common Choices:** H = 10, 20, 40 lags (depending on frequency).

**R/Python Implementation:**
```R
# R: forecast::checkresiduals() or stats::Box.test()
Box.test(residuals, type = "Ljung-Box", lag = 20)

# Python: statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(residuals, lags=20)
```

---

### 1.2 ARCH LM Test (Conditional Heteroskedasticity)

**Null Hypothesis:** H₀: No ARCH effects (constant conditional variance)

**Procedure:**
1. Estimate primary model; extract residuals eₜ.
2. Regress e²ₜ on constants and lagged squared residuals: e²ₜ = α₀ + α₁e²ₜ₋₁ + ... + αq e²ₜ₋q + εₜ
3. Compute R² from auxiliary regression.

**Test Statistic:**
```
LM = T * R²
```

**Distribution:** χ²(q) under H₀ (q = number of lags)

**Decision Rule:**
- LM < χ²_q,α: Fail to reject H₀; no significant ARCH effects.
- LM > χ²_q,α: Reject H₀; GARCH-type model may be needed.

**Common Choices:** q = 1, 5, 10 lags.

**Python Implementation:**
```python
from statsmodels.stats.diagnostic import het_arch
lm_stat, p_value, f_stat, f_p = het_arch(residuals, nlags=10)
```

---

### 1.3 Jarque-Bera Test (Normality)

**Null Hypothesis:** H₀: Skewness = 0 and Excess Kurtosis = 0 (normal distribution)

**Skewness:** S = m₃ / σ³
**Excess Kurtosis:** K = m₄ / σ⁴ - 3

**Test Statistic:**
```
JB = (n/6) * [S² + (K²/4)]
```

**Distribution:** χ²(2) under H₀

**Interpretation:**
- p-value > 0.05: Normal distribution reasonable.
- p-value < 0.05: Reject normality; consider Student-t or skewed distributions.

**Important:** In financial applications, rejection is **common and expected** due to fat tails.

**Python Implementation:**
```python
from scipy import stats
jb_stat, p_value = stats.jarque_bera(residuals)

# Or using statsmodels
from statsmodels.stats.diagnostic import jarque_bera
jb_stat, p_value, skewness, kurtosis = jarque_bera(residuals)
```

---

### 1.4 Augmented Dickey-Fuller (ADF) Test (Unit Root)

**Null Hypothesis:** H₀: Series has unit root (non-stationary)

**Test Statistic:** t-statistic on lagged differenced series.

**Interpretation:**
- p-value < 0.05: Reject H₀; series is stationary (suitable for ARIMA).
- p-value > 0.05: Fail to reject H₀; series is non-stationary (difference or transform).

**Python Implementation:**
```python
from statsmodels.tsa.stattools import adfuller
adf_stat, p_value, n_lags, nobs, crit_vals, ic_best = adfuller(series)
```

---

## 2. Performance Metrics

### 2.1 Error Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|-----------------|
| **MAE** | (1/n)Σ\|eₜ\| | [0, ∞) | Avg absolute error; robust to outliers |
| **RMSE** | √((1/n)Σe²ₜ) | [0, ∞) | Emphasizes large errors; in original units |
| **MSE** | (1/n)Σe²ₜ | [0, ∞) | Mean squared error; penalizes large deviations |
| **MAPE** | (1/n)Σ\|\|eₜ/yₜ\|\|×100 | [0, ∞) | % error; scale-independent; undefined for y=0 |
| **MASE** | MAE / MAE_naive | [0, ∞) | Scaled by naive forecast; interpretable |

**Selection Guide:**
- **MAE:** When outliers not problematic; want interpretable units.
- **RMSE:** When large errors very costly; want metric in original units.
- **MAPE:** When comparing across datasets; avoid near-zero actuals.
- **MSE:** When very large errors disproportionately penalized (financial applications).

---

### 2.2 Directional Accuracy Metrics

**Directional Accuracy (DA):**
```
DA = (# correct direction predictions / total predictions) × 100%
```

**Example:** If model predicts direction correctly 65/100 times, DA = 65%.

**Benchmark:** Random direction = 50%; models should exceed this significantly.

---

### 2.3 Volatility-Specific Metrics

**QLIKE (Quasi-Likelihood):**
```
QLIKE = (1/n) Σ [log(σ̂²ₜ) + (rₜ² / σ̂²ₜ)]
```

where σ̂²ₜ = model's forecasted volatility, r²ₜ = squared returns.

**Interpretation:** Lower QLIKE indicates better distributional fit of volatility forecast.

---

## 3. Statistical Tests for Model Comparison

### 3.1 Diebold-Mariano Test (Two Models)

**Null Hypothesis:** H₀: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t)

(Loss functions can be any form: absolute error, squared error, etc.)

**Test Statistic:**
```
DM = (d̄) / √(2πf₀(0) / T)
```

where d̄ = average loss differential, f₀(0) = spectral density at frequency 0.

**Distribution:** N(0,1) asymptotically

**Decision Rule:**
- |DM| < z₀.₀₂₅ (≈ 1.96): Fail to reject; forecasts equally accurate.
- |DM| > z₀.₀₂₅: Reject H₀; forecasts significantly different.

**Python Implementation:**
```python
from statsmodels.tsa.stattools import dm_test
dm_stat, p_value = dm_test(actual, pred1, pred2, alternative='two-sided')
```

---

### 3.2 Model Confidence Set (3+ Models)

**Concept:** Identifies set of models that contains the best model with given confidence level (e.g., 90%).

**Output:** Set MCS ⊆ {Model 1, Model 2, ..., Model M}

**Interpretation:**
- Models in MCS have statistically equal predictive ability.
- Models outside MCS can be eliminated.

**Algorithm Outline:**
1. Compute loss differential between each pair.
2. Apply equivalence test (e.g., t-test).
3. Remove model with worst performance (if significantly different).
4. Repeat until no model can be eliminated.

**Python Package:**
```python
# Install: pip install arch
from arch.bootstrap import MCS
mcs = MCS(loss_diffs, size=0.1)  # size=0.1 for 90% confidence
```

---

## 4. VaR Backtesting Tests

### 4.1 Kupiec's Proportion of Failures (POF) Test

**Null Hypothesis:** H₀: p = p₀, where p₀ = VaR confidence level (e.g., 0.01 for 99% VaR)

**Procedure:**
1. Estimate VaR at confidence level p₀ (e.g., VaR_0.99 = 5th percentile loss).
2. Count exceptions: X = # of times loss > VaR.
3. Compute likelihood ratio statistic.

**Test Statistic:**
```
LR_uc = 2 * [log(L(p̂)) - log(L(p₀))]
      = 2 * [X*log(p̂/p₀) + (N-X)*log((1-p̂)/(1-p₀))]

where p̂ = X/N (empirical exception rate)
```

**Distribution:** χ²(1) under H₀

**Decision Rule:**
- LR_uc < χ²₁,α: Fail to reject; model adequately predicts VaR.
- LR_uc > χ²₁,α: Reject; model underestimates risk.

**Critical Values (α = 0.05):** χ²₁,₀.₀₅ ≈ 3.841

**Example:**
- Test window: 250 days; VaR confidence: 99% (p₀ = 0.01)
- Expected exceptions: 250 × 0.01 = 2.5 (2-3 exceptions typical)
- Observed exceptions: 5 → LR_uc = 2[5*log(0.02/0.01) + 245*log(0.98/0.99)] ≈ 1.88 < 3.841 ✓

---

### 4.2 Basel Traffic Light Framework (ad hoc, regulatory)

**Based on Binomial Distribution:** X ~ Binomial(N=250, p)

**Zones for 95% VaR (p = 0.05), p₀ = 0.05:**

| Zone | # Exceptions | P(X ≤ k) | Action |
|------|---|---|---|
| Green | 0-8 | ≤95% | Model acceptable; no action |
| Yellow | 9-11 | 95%-99.99% | Under review; capital multiplier increased |
| Red | ≥12 | >99.99% | Model likely inadequate; rejection likely |

**For 99% VaR (p = 0.01):**

| Zone | # Exceptions | Interpretation |
|------|---|---|
| Green | 0-4 | Model acceptable |
| Yellow | 5-9 | Under review |
| Red | ≥10 | Rejected |

**Note:** Basel framework is **not a formal hypothesis test**; zones are ad hoc but regulatory standard.

---

### 4.3 Christoffersen's Independence Test

**Extension:** Tests whether VaR exceptions are **independent** (not clustered).

**Components:**
- **Unconditional Coverage (UC) Test:** Kupiec's POF test (LR_uc ~ χ²(1))
- **Independence Test:** LR_ind ~ χ²(1), testing that exceptions don't cluster

**Joint Test:**
```
LR_cc = LR_uc + LR_ind ~ χ²(2)
```

**Interpretation:**
- If LR_uc pass but LR_ind fail: Model misses regime changes (volatility clustering).
- If both pass: Model is adequate on both frequency and timing of exceptions.

---

## 5. GARCH Model Diagnostics Checklist

### Step 1: Specification Selection
- [ ] Plot returns; visually inspect for volatility clustering.
- [ ] Ljung-Box test on returns (should not reject).
- [ ] ARCH LM test on returns (should reject if GARCH needed).
- [ ] Select tentative GARCH(p,q) order (often GARCH(1,1) sufficient).

### Step 2: Parameter Estimation
- [ ] Estimate GARCH(p,q) via maximum likelihood.
- [ ] Record AIC, BIC values.
- [ ] Test alternative orders; select lowest AIC/BIC.

### Step 3: Residual Diagnostics
- [ ] Extract standardized residuals: ẑₜ = eₜ / σ̂ₜ
- [ ] Ljung-Box on ẑₜ (should not reject; p > 0.05).
- [ ] ARCH LM on ẑ²ₜ (should not reject; p > 0.05).
- [ ] Jarque-Bera on ẑₜ (likely to reject; note non-normality).
- [ ] Q-Q plot; inspect tail fit.

### Step 4: Misspecification Tests (optional)
- [ ] Sign-bias test: ẑₜ = α₀ + α₁*S⁻ₜ₋₁ + εₜ, where S⁻ = 1 if rₜ₋₁ < 0
  - Rejects if negative shocks increase volatility (leverage effect).
- [ ] Size-bias test: similar, using magnitude of past shocks.

### Step 5: Forecast Evaluation
- [ ] Compute volatility forecasts h-steps ahead.
- [ ] Compare to realized volatility (from high-frequency data, if available).
- [ ] RMSE, MAE, or QLIKE metric.
- [ ] Diebold-Mariano test vs. competing models.

---

## 6. Walk-Forward Backtesting Procedure

**Gold Standard for Out-of-Sample Validation**

### Pseudo-Code

```
window_size = 1000  # in-sample training window
test_period = 250   # out-of-sample test period
step_size = 50      # shift window by this amount each iteration

results = []
for i in range(0, len(data) - window_size - test_period, step_size):
    train_start = i
    train_end = i + window_size
    test_start = i + window_size
    test_end = test_start + test_period

    # Fit model on training data
    model.fit(data[train_start:train_end])

    # Forecast on test data
    forecasts = model.predict(data[test_start:test_end])
    actuals = data[test_start:test_end]

    # Compute metrics
    mae = mean_absolute_error(actuals, forecasts)
    rmse = sqrt(mean_squared_error(actuals, forecasts))

    results.append({'mae': mae, 'rmse': rmse, 'window': i})

# Summary statistics
avg_mae = mean([r['mae'] for r in results])
std_mae = std([r['mae'] for r in results])
```

### Python Implementation (Pandas/sklearn)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def walk_forward_backtest(data, model_fn, window=1000, test_period=250, step=50):
    results = []

    for i in range(0, len(data) - window - test_period, step):
        X_train = data[i:i+window]
        X_test = data[i+window:i+window+test_period]

        # Fit and predict
        model = model_fn()
        model.fit(X_train)
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(X_test, y_pred)
        rmse = np.sqrt(mean_squared_error(X_test, y_pred))

        results.append({'mae': mae, 'rmse': rmse, 'period': i})

    return pd.DataFrame(results)
```

---

## 7. Decision Tree: Which Test to Use?

```
Question: What are you testing?

├─ "Residuals are white noise?"
│  ├─ Ljung-Box test (H = 10-40 lags)
│  └─ If p-value > 0.05 → Good
│
├─ "No conditional heteroskedasticity?"
│  ├─ ARCH LM test (q = 5-10 lags)
│  └─ If p-value > 0.05 → Good
│
├─ "Returns are normally distributed?"
│  ├─ Jarque-Bera test
│  ├─ Shapiro-Wilk test
│  └─ Note: Rejection common in finance (not a problem)
│
├─ "VaR model is adequate?"
│  ├─ Kupiec POF test (LR_uc ~ χ²(1))
│  ├─ Christoffersen test (LR_cc ~ χ²(2))
│  └─ Basel traffic light (ad hoc)
│
├─ "Forecast 1 better than Forecast 2?"
│  ├─ Diebold-Mariano test (DM ~ N(0,1))
│  └─ If |DM| > 1.96 → Significant difference
│
└─ "Which model among 3+ is best?"
   ├─ Model Confidence Set (Hansen 2011)
   └─ Returns set of models with equal predictive ability
```

---

## 8. Common Pitfalls and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Look-ahead Bias** | Unrealistically high backtest results | Use time-series split; no random shuffling |
| **Overfitting** | Great in-sample, poor out-of-sample | Walk-forward validation; cross-validation |
| **Non-stationarity** | Model breaks down in new regime | Test for unit root (ADF); difference if needed |
| **Ignored Costs** | Sharpe Ratio inflated by ~15-20% | Include bid-ask spread (0.01-0.1%), slippage (0.02-0.05%) |
| **Non-normality** | Tail risk underestimated | Use Student-t or skewed-t; stress-test extreme scenarios |
| **Autocorrelation** | Ljung-Box fails despite good diagnostics | Increase lags; consider ARMA or GARCH term |
| **Parameter Instability** | Model performance deteriorates over time | Implement rolling window re-estimation; test for breaks |

---

## 9. Code Examples

### Python: GARCH Diagnostics (using arch package)

```python
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, jarque_bera
import numpy as np

# Fit GARCH(1,1)
model = arch_model(returns, vol='Garch', p=1, q=1)
res = model.fit(disp='off')

# Extract standardized residuals
std_resid = res.std_resid

# Diagnostic tests
ljung_box = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
jb_stat, jb_pval, skew, kurt = jarque_bera(std_resid)

print(f"Ljung-Box p-values:\n{ljung_box}")
print(f"Jarque-Bera: stat={jb_stat:.4f}, p={jb_pval:.4f}")
print(f"Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}")

# Summary
print(res.summary())
```

### R: Ljung-Box and ARCH Tests

```R
library(forecast)
library(FinTS)

# Ljung-Box test
Box.test(residuals, type="Ljung-Box", lag=20)

# ARCH LM test
ArchTest(residuals, lags=10)

# Diagnostic plots
checkresiduals(model, lag.max=20)
```

### Python: Diebold-Mariano Test

```python
from statsmodels.tsa.stattools import dm_test

# actual: true values
# pred1, pred2: two forecasts
dm_stat, p_value = dm_test(actual, pred1, pred2, alternative='two-sided')

print(f"DM Statistic: {dm_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Forecasts are significantly different.")
else:
    print("No significant difference between forecasts.")
```

---

## 10. Summary Table: Test Selection by Purpose

| Purpose | Test(s) | Null H₀ | Distribution | Threshold |
|---------|---------|---------|---|---|
| Check autocorrelation | Ljung-Box | ρ=0 | χ²(H) | p > 0.05 |
| Check ARCH effects | ARCH LM | No ARCH | χ²(q) | p > 0.05 |
| Check normality | Jarque-Bera | Normal dist. | χ²(2) | p > 0.05* |
| Compare 2 forecasts | Diebold-Mariano | Equal accuracy | N(0,1) | \|DM\| < 1.96 |
| Compare 3+ models | MCS (Hansen) | Equal ability | See MCS algo | Within set |
| Validate VaR | Kupiec POF | p = p₀ | χ²(1) | LR < 3.84 |
| VaR + timing | Christoffersen | UC + Indep. | χ²(2) | LR < 5.99 |

*Note: In financial applications, normality rejection is common and expected.

---

**Last Updated:** December 21, 2025
