# Key Papers and Applications: Stock Price Model Validation

**Purpose:** Detailed annotations of seminal papers and practical application examples for model testing and validation.

---

## Part 1: Seminal Papers with Annotations

### 1. Ljung & Box (1978) - Portmanteau Test

**Full Citation:** Ljung, G. M., & Box, G. E. (1978). "On a measure of lack of fit in time series models." *Biometrika*, 65(2), 297-303.

**Abstract Summary:** Proposes the Ljung-Box Q statistic as an improvement over Box-Pierce test for testing autocorrelation in residuals. Better small-sample properties than original Box-Pierce test.

**Key Contribution:**
- Test statistic: Q_LB = T(T+2) Σ(ρ²ₖ/(T-k)) asymptotically χ²(H)
- Avoids testing each lag individually (multiple comparison problem).
- Widely adopted in practice for routine residual diagnostics.

**Practical Impact:** Standard diagnostic for all ARIMA/GARCH software packages.

**When to Apply:**
- After fitting any time series model
- On raw residuals or standardized residuals
- Typical lags: H = 10, 20, 40 (depending on frequency)

**Interpretation in Financial Context:**
- If Ljung-Box rejects on raw residuals → Possible model misspecification
- If Ljung-Box passes but ARCH LM rejects on squared residuals → GARCH-type model needed
- If both pass → Model captures mean and conditional variance adequately

---

### 2. Engle (1982) - ARCH Models and LM Test

**Full Citation:** Engle, R. F. (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance of UK inflation." *Econometrica*, 50(4), 987-1007.

**Abstract Summary:** Introduces ARCH (Autoregressive Conditional Heteroskedasticity) models to capture time-varying volatility in economic data, particularly inflation. Seminal paper revolutionizing modeling of financial volatility.

**Key Contributions:**
1. **ARCH Model:** σ²ₜ = α₀ + α₁e²ₜ₋₁ + ... + αqe²ₜ₋q
2. **LM Test for ARCH:** Regress e²ₜ on lags of e²ₜ; test joint significance via LM ~ χ²(q)
3. **Maximum Likelihood Estimation:** Provides numerical methods for estimation

**Practical Impact:** Foundation for all modern volatility models (GARCH, EGARCH, etc.); ARCH test is now standard diagnostic.

**When to Apply:**
- Stock returns typically exhibit strong ARCH effects
- Test to determine if conditional volatility modeling needed
- Use as specification test for GARCH adequacy

**Financial Application Example:**
- Daily S&P 500 returns often show:
  - Ljung-Box: p-value > 0.05 (white noise in mean)
  - ARCH LM (q=10): p-value < 0.01 (strong ARCH effects present)
  - → Conclusion: GARCH(1,1) or higher-order model needed

---

### 3. Jarque & Bera (1987) - Normality Test

**Full Citation:** Jarque, C. M., & Bera, A. K. (1987). "A test for normality of observations and regression residuals." *International Statistical Review*, 55(2), 163-172.

**Abstract Summary:** Develops test for normality based on skewness and kurtosis, simple to compute from sample moments.

**Key Contribution:**
- JB = (n/6)[S² + (K²/4)] ~ χ²(2) under normality
- S = skewness (m₃/σ³), K = excess kurtosis (m₄/σ⁴ - 3)
- Easy to implement; detects both skewness and tail fatness

**Critical for Finance:**
- Stock returns exhibit negative skewness (crash risk) and excess kurtosis (fat tails)
- Jarque-Bera almost always rejects for financial returns
- This is NOT a flaw; it's a stylized fact
- If Jarque-Bera fails to reject → Suspect data quality or market inefficiency

**Expected Findings in Practice:**
- S&P 500 daily returns: Skewness ≈ -0.8 to -1.2 (negative), Kurtosis ≈ 10-15
- Jarque-Bera test: p-value ≈ 0.000*** (highly significant)
- Implication: Standard normal assumption underestimates tail risk

**What to Do When Normality Fails:**
1. Use Student-t distribution (heavier tails)
2. Use skewed-t distribution (captures negative skew)
3. Implement semi-parametric methods (don't assume distribution)
4. Stress-test VaR models with empirical quantiles

---

### 4. Kupiec (1995) - VaR Backtesting

**Full Citation:** Kupiec, P. H. (1995). "Techniques for verifying the accuracy of risk measurement models." Working Paper, Federal Reserve Bank of Chicago.

**Abstract Summary:** Proposes formal statistical test (POF test) for validating Value-at-Risk models. Foundation for regulatory VaR backtesting frameworks.

**Key Contributions:**
1. **Proportion of Failures (POF) Test:** LR_uc = 2[X*log(p̂/p₀) + (N-X)*log((1-p̂)/(1-p₀))] ~ χ²(1)
2. **Hypothesis Test Framework:** Tests whether observed VaR exceptions match expected frequency
3. **Regulatory Adoption:** Basel Committee adopts this test as standard for bank internal models

**Test Mechanics Example:**
```
Suppose: 99% VaR (p₀ = 0.01), test window N = 250 days
Expected exceptions: 250 × 0.01 = 2.5 exceptions

Scenario 1: Observed 2 exceptions
  p̂ = 2/250 = 0.008
  LR = 2[2*log(0.008/0.01) + 248*log(0.992/0.99)]
     ≈ 0.19 < 3.84 → PASS (model adequate)

Scenario 2: Observed 8 exceptions
  p̂ = 8/250 = 0.032
  LR = 2[8*log(0.032/0.01) + 242*log(0.968/0.99)]
     ≈ 15.2 > 3.84 → FAIL (model underestimates risk)
```

**Importance in Practice:**
- Regulatory requirement for banks' internal risk models
- Simple, transparent, and powerful test
- Prevents banks from systematically underestimating tail risk
- Forces banks to hold adequate capital buffers

---

### 5. Diebold & Mariano (1995) - Forecast Comparison

**Full Citation:** Diebold, F. X., & Mariano, R. S. (1995). "Comparing predictive accuracy." *Journal of Business & Economic Statistics*, 13(3), 253-263.

**Abstract Summary:** Develops asymptotic test for comparing accuracy of two competing forecasts. Allows loss functions to be asymmetric and errors to be serially correlated and non-normal.

**Key Innovations:**
1. **General Loss Function:** No requirement for symmetric or quadratic loss; can use any function
2. **Serial Correlation Allowed:** Previous tests assumed iid errors; DM allows correlation
3. **Non-Normal Errors:** Valid even if forecast errors are non-normal (common in finance)
4. **Asymptotically Normal:** DM ~ N(0,1) under H₀

**Test Procedure:**
```
Step 1: Compute loss for each forecast
  L₁ₜ = f(e₁ₜ)  [loss from forecast 1]
  L₂ₜ = f(e₂ₜ)  [loss from forecast 2]

Step 2: Compute loss differential
  dₜ = L₁ₜ - L₂ₜ

Step 3: Test if E[dₜ] = 0
  DM = d̄ / √(2πf̂₀(0)/T) ~ N(0,1)

  where d̄ = (1/T)Σdₜ
        f̂₀(0) = spectral density estimate at frequency 0

Step 4: Decision
  |DM| > 1.96 at α=0.05 → Forecasts significantly different
```

**Real-World Example:**
```
Compare LSTM stock price forecast vs. GARCH volatility forecast

LSTM Forecast: yₗₛₜₘ,ₜ, error = rₜ - yₗₛₜₘ,ₜ
GARCH Forecast: yᵍₐᵣcₕ,ₜ, error = rₜ - yᵍₐᵣcₕ,ₜ

Loss function: L(e) = |e| (absolute error)
Loss differential: dₜ = |e₁ₜ| - |e₂ₜ|

Sample: 250 out-of-sample predictions
d̄ = 0.015 (LSTM on average 0.015 worse)
Std error = 0.008

DM = 0.015 / 0.008 = 1.875 < 1.96 → NO significant difference at 5% level
```

**When to Use:**
- Comparing two competing models (ARIMA vs. GARCH; LSTM vs. Transformer)
- Want formal statistical test of forecast accuracy
- Have multiple loss functions to consider
- Forecast errors exhibit autocorrelation (common in finance)

**Advantages over Alternatives:**
- Simpler than MCS for just two models
- Allows any loss function
- Robust to non-normality
- Accounts for serial correlation

---

### 6. Hansen & Lunde (2003) & Hansen (2011) - Model Confidence Sets

**Citation 1:** Hansen, P. R., & Lunde, A. (2003). "A comparison of volatility models: Does anything beat a GARCH(1,1)?" Working Paper, Aarhus University.

**Citation 2:** Hansen, P. R., Lunde, A., & Nason, J. M. (2011). "The model confidence set." *Econometrica*, 79(2), 453-497.

**Abstract Summary:** Proposes MCS as a method for identifying a set of models containing the true best model with given confidence level. Extends confidence interval concept to model selection.

**Key Innovation:**
- MCS = {M: P(M ∈ MCS | data) ≥ 1-α}
- Contains all models with equal predictive ability at level α
- Avoids multiple testing problem inherent in pairwise comparisons

**Example Application:**

```
Volatility Models Compared:
1. GARCH(1,1)
2. GARCH(1,2)
3. GJR-GARCH(1,1)
4. EGARCH(1,1)
5. HAR-RV (Heterogeneous Autoregression)

Loss metric: Quasi-Likelihood (QLIKE)
Test period: 2000 observations

Results (90% confidence, α=0.10):
Model Confidence Set = {GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1)}

Interpretation:
- These three models have statistically equal volatility forecasting ability
- GARCH(1,2) and HAR-RV can be eliminated; they're significantly worse
- Practitioners can choose GARCH(1,1) (simplest) without loss
```

**Advantages:**
- Handles multiple model comparisons correctly
- Identifies "best" set, not just single best model
- Useful when many models are nearly equivalent
- Respects uncertainty in model selection

**Computational Note:**
- MCS uses sequential elimination algorithm
- Computationally intensive for very large model sets (100+)
- Usually applied to 5-50 models

---

### 7. Engle & Ng (1993) - GARCH Asymmetry Tests

**Full Citation:** Engle, R. F., & Ng, V. K. (1993). "Measuring and testing the impact of news on volatility." *Journal of Finance*, 48(5), 1749-1778.

**Abstract Summary:** Proposes tests for detecting asymmetric volatility responses (leverage effect) in GARCH models. Important for capturing stylized fact that negative shocks increase volatility more than positive shocks.

**Key Tests:**

**1. Sign-Bias Test**
```
Auxiliary regression: ẑₜ = α₀ + α₁S⁻ₜ₋₁ + εₜ
where S⁻ₜ₋₁ = 1 if rₜ₋₁ < 0, else 0
       ẑₜ = standardized residuals from GARCH

H₀: α₁ = 0 (no sign bias)
Test statistic: t = α̂₁ / SE(α̂₁) ~ N(0,1)

Interpretation: Reject → Negative shocks increase volatility
```

**2. Size-Bias Test**
```
Auxiliary regression: ẑₜ = α₀ + α₁S⁻ₜ₋₁|rₜ₋₁| + εₜ
Tests whether magnitude of past negative shocks matters

H₀: α₁ = 0 (no size bias)
Interpretation: Reject → Large negative shocks matter more
```

**Practical Example (S&P 500):**
```
Fit GARCH(1,1) to daily returns

Ljung-Box on standardized residuals: p = 0.45 (PASS)
ARCH LM on squared residuals: p = 0.38 (PASS)
Jarque-Bera: p < 0.01 (REJECT - fat tails, expected)

Sign-Bias Test:
  t-stat = 2.15, p-value = 0.032 → REJECT
  Interpretation: Negative shocks increase volatility significantly
  → Standard GARCH(1,1) may underestimate volatility after crashes
  → Consider GJR-GARCH or EGARCH instead
```

**When to Apply:**
- After fitting standard GARCH model
- Before finalizing model for risk forecasting
- If testing for leverage effect (stock market property)

---

### 8. Nyberg et al. (2024) - Conditional Score Residuals

**Full Citation:** Nyberg, H., et al. (2024). "Conditional Score Residuals and Diagnostic Analysis of Serial Dependence in Time Series Models." *Journal of Time Series Analysis*, Online first.

**Abstract Summary:** Provides unified framework for residual diagnostics encompassing ARMA, GARCH, and nonlinear models. Extends classical residual analysis to modern complex specifications.

**Key Contribution:**
- **Unified Framework:** Encompasses ARMA residuals, squared residuals (ARCH), Pearson residuals, etc.
- **Nonlinear Detection:** Can detect nonlinear patterns missed by standard tests
- **Multiple Diagnosis:** Detects serial dependence, volatility clustering, parameter instability

**Advanced Methods Discussed:**
1. **Kernel-based Tests:** Non-parametric tests avoiding distribution assumptions
2. **Neural Network Analysis:** Use fitted neural networks to detect patterns in residuals
3. **CUSUM Tests:** Cumulative sum tests for parameter stability

**Typical Application in Modern Pipeline:**
```
1. Fit baseline model (GARCH, LSTM, etc.)
2. Extract residuals
3. Apply standard tests (Ljung-Box, ARCH LM)
4. If any test marginal:
   - Apply conditional score residuals framework
   - Use kernel-based or NN-based methods
   - Identify specific type of misspecification
5. Refine model based on findings
```

**Advantage Over Classical Tests:**
- More flexible; doesn't require strict distributional assumptions
- Detects subtle nonlinearities
- Provides guidance on model improvement direction

---

## Part 2: Practical Application Examples

### Example 1: Full Diagnostic Pipeline for GARCH Model

**Objective:** Validate GARCH(1,1) model of S&P 500 daily returns.

**Data:** Daily log-returns, 2000 observations (≈8 years)

**Step 1: Visual Inspection**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('sp500_daily.csv', index_col='Date', parse_dates=True)
returns = data['Close'].pct_change().dropna()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(returns)
axes[0].set_title('S&P 500 Daily Returns')
axes[1].hist(returns, bins=100, density=True)
axes[1].set_title('Distribution of Returns')
plt.show()

# Summary statistics
print(f"Mean: {returns.mean():.6f}")
print(f"Std Dev: {returns.std():.6f}")
print(f"Skewness: {returns.skew():.4f}")
print(f"Kurtosis: {returns.kurtosis():.4f}")
```

**Expected Output:**
```
Mean: 0.000352
Std Dev: 0.012345
Skewness: -0.850  (negative; crash risk)
Kurtosis: 8.230   (excess; fat tails)
```

**Step 2: Preliminary Tests on Raw Returns**
```python
from statsmodels.stats.diagnostic import acorr_ljungbox, jarque_bera
from statsmodels.stats.diagnostic import het_arch

# Ljung-Box test (should fail to reject for returns)
lb_result = acorr_ljungbox(returns, lags=[10, 20], return_df=True)
print("Ljung-Box on Returns:")
print(lb_result)
# Expected: p-values > 0.05 (returns are white noise)

# Jarque-Bera
jb_stat, jb_pval, skew, kurt = jarque_bera(returns)
print(f"\nJarque-Bera: stat={jb_stat:.2f}, p={jb_pval:.4f}")
# Expected: p < 0.01 (non-normal; expected)

# ARCH LM test (should reject; indicates GARCH needed)
lm_stat, p_value, f_stat, f_pval = het_arch(returns, nlags=10)
print(f"\nARCH LM Test: stat={lm_stat:.4f}, p={p_value:.4f}")
# Expected: p < 0.05 (strong ARCH effects present)
```

**Expected Output:**
```
Ljung-Box on Returns:
    lb_stat      lb_pvalue
10   8.34         0.586  (PASS)
20  14.92         0.789  (PASS)

Jarque-Bera: stat=1254.23, p=0.0000  (REJECT - expected)

ARCH LM Test: stat=87.34, p=0.0000  (REJECT - GARCH needed)
```

**Step 3: Fit GARCH(1,1)**
```python
from arch import arch_model

# Fit GARCH(1,1)
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant')
results = model.fit(disp='off')

print(results.summary())

# Extract standardized residuals
std_resid = results.std_resid
```

**Expected Output:**
```
                         Constant Mean - GARCH Model Results
==============================================================================
Dep. Variable:     Close    R-squared:                       0.000
Mean Model:  Constant Mean   Adj. R-squared:                  -0.001
Vol Model:          GARCH    Log-Likelihood:               5234.18
Date:              12/21/25 AIC:                          -10460.36
Time:                09:15 BIC:                          -10441.60
Number of obs:          2000
Df Residuals:           2000   Df Model:                        3
==============================================================================
                        coef    std err          t      P>|t|   [0.025  0.975]
──────────────────────────────────────────────────────────────────────────────
μ (mean)              0.0003   0.0001       4.120      0.000    0.0002   0.0005
ω (constant vol)      0.0000   0.0000       3.102      0.002    0.0000   0.0000
α₁ (arch)             0.1234   0.0234       5.271      0.000    0.0776   0.1691
β₁ (garch)            0.8521   0.0145      58.69       0.000    0.8237   0.8805
==============================================================================
```

**Step 4: Diagnostic Tests on Standardized Residuals**
```python
from statsmodels.stats.diagnostic import acorr_ljungbox, jarque_bera

# Ljung-Box on standardized residuals (should PASS)
lb_std = acorr_ljungbox(std_resid, lags=[10, 20], return_df=True)
print("Ljung-Box on Standardized Residuals:")
print(lb_std)
# Expected: p-values > 0.05 (no autocorrelation)

# ARCH LM on squared standardized residuals (should PASS)
lm_stat, p_value, _, _ = het_arch(std_resid, nlags=10)
print(f"\nARCH LM on Squared Std Residuals: p={p_value:.4f}")
# Expected: p > 0.05 (no remaining ARCH effects)

# Jarque-Bera on standardized residuals
jb_stat, jb_pval, skew, kurt = jarque_bera(std_resid)
print(f"\nJarque-Bera on Std Residuals: p={jb_pval:.4f}")
# Expected: Still reject (fatter tails than normal; expected)
```

**Expected Output:**
```
Ljung-Box on Standardized Residuals:
    lb_stat      lb_pvalue
10   6.23         0.715  (PASS)
20  15.78         0.773  (PASS)

ARCH LM on Squared Std Residuals: p=0.487  (PASS)

Jarque-Bera on Std Residuals: p=0.0002  (REJECT - still non-normal)
```

**Step 5: Interpretation and Conclusion**
```python
# Summary
print("Model Adequacy Summary:")
print("=" * 50)
print("✓ Ljung-Box (std resid): PASS → No autocorrelation")
print("✓ ARCH LM (squared): PASS → Volatility captured")
print("✗ Jarque-Bera: REJECT → Non-normal (fat tails)")
print("\nConclusion: GARCH(1,1) is adequate for mean")
print("and conditional variance, but consider Student-t")
print("distribution for better tail modeling.")
```

---

### Example 2: Backtesting VaR Model with Walk-Forward Analysis

**Objective:** Validate 99% VaR model using walk-forward procedure and Basel framework.

**Setup:**
```python
import numpy as np
from scipy import stats

def compute_var(returns, confidence=0.99):
    """Compute VaR at given confidence level."""
    return np.percentile(returns, (1 - confidence) * 100)

def backtest_var_walkforward(returns, confidence=0.99, train_window=500, test_period=250):
    """Walk-forward VaR backtesting."""

    results = []

    for i in range(0, len(returns) - train_window - test_period, 50):
        # Training window
        train_start = i
        train_end = i + train_window
        train_returns = returns[train_start:train_end]

        # Test window
        test_start = i + train_window
        test_end = test_start + test_period
        test_returns = returns[test_start:test_end]

        # Estimate VaR on training data
        var_estimate = compute_var(train_returns, confidence)

        # Count exceptions (violations) in test period
        exceptions = (test_returns < var_estimate).sum()
        exception_rate = exceptions / test_period

        results.append({
            'period': i // 50,
            'var': var_estimate,
            'exceptions': exceptions,
            'exception_rate': exception_rate,
            'test_start': test_start,
            'test_end': test_end
        })

    return pd.DataFrame(results)

# Run backtest
backtest_df = backtest_var_walkforward(returns.values, confidence=0.99,
                                       train_window=500, test_period=250)

print(backtest_df)
```

**Expected Output:**
```
   period         var  exceptions  exception_rate  test_start  test_end
0       0  -0.026543           2             0.008         500       750
1       1  -0.024891           1             0.004         550       800
2       2  -0.025123           3             0.012         600       850
...
```

**Statistical Testing:**
```python
from scipy.stats import chi2

def kupiec_pof_test(exceptions, test_periods, confidence=0.99):
    """Kupiec's Proportion of Failures Test."""

    n = test_periods
    x = exceptions
    p0 = 1 - confidence  # Expected exception rate (0.01 for 99% VaR)

    # Empirical exception rate
    p_hat = x / n

    # Likelihood Ratio
    if p_hat == 0 or p_hat == 1:
        lr = 0  # Avoid log(0)
    else:
        lr = 2 * (x * np.log(p_hat / p0) + (n - x) * np.log((1 - p_hat) / (1 - p0)))

    # Critical value (χ²(1))
    critical_value = chi2.ppf(0.95, df=1)  # 3.841 for α=0.05
    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        'observed_exceptions': x,
        'expected_exceptions': n * p0,
        'exception_rate': p_hat,
        'lr_statistic': lr,
        'critical_value': critical_value,
        'p_value': p_value,
        'pass_test': lr < critical_value
    }

# Aggregate results
total_exceptions = backtest_df['exceptions'].sum()
total_test_periods = len(backtest_df) * 250

# Run test
test_result = kupiec_pof_test(total_exceptions, total_test_periods, confidence=0.99)

print("\nKupiec POF Test Results:")
print("=" * 50)
print(f"Total Test Periods: {total_test_periods}")
print(f"Observed Exceptions: {test_result['observed_exceptions']}")
print(f"Expected Exceptions: {test_result['expected_exceptions']:.1f}")
print(f"Exception Rate: {test_result['exception_rate']:.4f}")
print(f"LR Statistic: {test_result['lr_statistic']:.4f}")
print(f"Critical Value: {test_result['critical_value']:.4f}")
print(f"P-Value: {test_result['p_value']:.4f}")
print(f"Result: {'PASS' if test_result['pass_test'] else 'FAIL'}")
```

**Expected Output:**
```
Kupiec POF Test Results:
==================================================
Total Test Periods: 3000
Observed Exceptions: 28
Expected Exceptions: 30.0
Exception Rate: 0.0093
LR Statistic: 0.1342
Critical Value: 3.8415
P-Value: 0.7143
Result: PASS
```

**Basel Traffic Light Classification:**
```python
def basel_traffic_light(exceptions, confidence=0.99):
    """Classify exception count to Basel zones."""

    # For 250-day windows and 99% VaR
    zones = {
        'green': {'range': (0, 4), 'description': 'Model acceptable'},
        'yellow': {'range': (5, 9), 'description': 'Under review'},
        'red': {'range': (10, float('inf')), 'description': 'Likely inadequate'}
    }

    for zone, info in zones.items():
        if info['range'][0] <= exceptions <= info['range'][1]:
            return zone, info['description']

    return None, 'Invalid'

# Classify observed exceptions
zone, description = basel_traffic_light(total_exceptions)

print(f"\nBasel Traffic Light Classification:")
print("=" * 50)
print(f"Observed Exceptions (aggregate): {total_exceptions}")
print(f"Zone: {zone.upper()} → {description}")
```

---

### Example 3: Diebold-Mariano Test for Deep Learning vs. Traditional Model

**Objective:** Compare forecast accuracy of LSTM vs. ARIMA model.

```python
from statsmodels.tsa.stattools import dm_test
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assume we have:
# actual: true returns (out-of-sample, length=250)
# lstm_pred: LSTM predictions
# arima_pred: ARIMA predictions

# Method 1: Direct DM test with default quadratic loss
dm_stat, p_value = dm_test(actual, lstm_pred, arima_pred, alternative='two-sided')

print("Diebold-Mariano Test: LSTM vs ARIMA")
print("=" * 50)
print(f"DM Statistic: {dm_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"Interpretation: ", end="")
if p_value < 0.05:
    if dm_stat > 0:
        print("ARIMA significantly better (loss favors ARIMA)")
    else:
        print("LSTM significantly better (loss favors LSTM)")
else:
    print("No significant difference in forecast accuracy")

# Method 2: Using custom loss function (absolute error)
ae_lstm = np.abs(actual - lstm_pred)
ae_arima = np.abs(actual - arima_pred)
loss_diff = ae_lstm - ae_arima

dm_stat_manual = loss_diff.mean() / np.sqrt(2 * np.pi * loss_diff.var() / len(loss_diff))
p_value_manual = 2 * (1 - stats.norm.cdf(np.abs(dm_stat_manual)))

print(f"\nUsing Absolute Error Loss:")
print(f"DM Statistic: {dm_stat_manual:.4f}")
print(f"P-Value: {p_value_manual:.4f}")

# Additional metrics
mae_lstm = mean_absolute_error(actual, lstm_pred)
mae_arima = mean_absolute_error(actual, arima_pred)
rmse_lstm = np.sqrt(mean_squared_error(actual, lstm_pred))
rmse_arima = np.sqrt(mean_squared_error(actual, arima_pred))

print(f"\nPerformance Summary:")
print(f"Model           MAE        RMSE")
print(f"LSTM           {mae_lstm:.6f}  {rmse_lstm:.6f}")
print(f"ARIMA          {mae_arima:.6f}  {rmse_arima:.6f}")
```

---

### Example 4: Model Confidence Set for Multiple Volatility Models

**Objective:** Compare 5 volatility models and identify best set.

```python
from arch import arch_model
from statsmodels.tsa.stattools import dm_test
import pandas as pd

# Fit multiple models
models_fit = {
    'GARCH(1,1)': arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off'),
    'GARCH(1,2)': arch_model(returns, vol='Garch', p=1, q=2).fit(disp='off'),
    'GJR-GARCH(1,1)': arch_model(returns, vol='GARCH', p=1, q=1).fit(disp='off'),
    'EGARCH(1,1)': arch_model(returns, vol='EG', p=1, q=1).fit(disp='off'),
}

# For each model, compute QLIKE on test set (out-of-sample)
# QLIKE = (1/n) * Σ[log(σ̂²_t) + (r²_t / σ̂²_t)]

test_returns = returns[-250:]  # Last 250 obs

qlike_scores = {}
for name, model in models_fit.items():
    # Get fitted conditional variance
    # (In practice, use rolling forecast)
    sigma2 = model.conditional_volatility ** 2

    # Compute QLIKE (truncate to test period)
    qlike = np.mean(np.log(sigma2[-250:]) + (test_returns.values ** 2) / sigma2[-250:].values)
    qlike_scores[name] = qlike

print("QLIKE Loss (lower is better):")
for name, qlike in sorted(qlike_scores.items(), key=lambda x: x[1]):
    print(f"{name}: {qlike:.6f}")

# Pairwise DM tests (simplified; compute loss differentials)
loss_diffs = {}
for model1 in models_fit.keys():
    for model2 in models_fit.keys():
        if model1 < model2:  # Avoid duplicates
            sigma1 = models_fit[model1].conditional_volatility[-250:].values ** 2
            sigma2 = models_fit[model2].conditional_volatility[-250:].values ** 2

            qlike1 = np.log(sigma1) + (test_returns.values ** 2) / sigma1
            qlike2 = np.log(sigma2) + (test_returns.values ** 2) / sigma2
            loss_diff = qlike1 - qlike2

            loss_diffs[(model1, model2)] = loss_diff

# Simple MCS logic: models with median loss_diff ~ 0 are in MCS
mcs_set = set(models_fit.keys())
threshold = np.std(list(loss_diffs.values())) * 1.96 / np.sqrt(250)

print(f"\nModel Confidence Set (90% confidence):")
print(f"Loss difference threshold: ±{threshold:.6f}")
print("Models in MCS:")

# (Simplified; actual MCS more complex)
for name in sorted(models_fit.keys()):
    print(f"  - {name}")
```

---

## Part 3: Common Pitfalls and Solutions

### Pitfall 1: Look-Ahead Bias

**Problem:** Using future information in model training.

**Example:**
```python
# WRONG: Normalizing entire dataset before train/test split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns.values.reshape(-1, 1))  # Entire dataset!

train_data = returns_scaled[:1000]
test_data = returns_scaled[1000:1250]
# Scaler fitted on future data → inflated performance
```

**Solution:**
```python
# RIGHT: Fit scaler on training data only
train_returns = returns[:1000]
test_returns = returns[1000:1250]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_returns.values.reshape(-1, 1))
test_scaled = scaler.transform(test_returns.values.reshape(-1, 1))
# Scaler fitted on training data only
```

---

### Pitfall 2: Ignoring Transaction Costs

**Problem:** Backtests ignore bid-ask spreads, slippage, commissions.

**Example:**
```python
# WRONG: Perfect execution assumption
strategy_pnl = (returns * positions).sum()
sharpe_ratio = strategy_pnl.mean() / strategy_pnl.std() * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")  # Likely inflated
```

**Solution:**
```python
# RIGHT: Include transaction costs
bid_ask_spread = 0.001  # 0.1% per round-trip
slippage = 0.0005      # 0.05% per trade
commission = 0.0005    # 0.05% per trade

transaction_cost = bid_ask_spread + slippage + commission  # 0.2%

# Adjust returns
net_returns = returns - transaction_cost
strategy_pnl = (net_returns * positions).sum()
sharpe_ratio = strategy_pnl.mean() / strategy_pnl.std() * np.sqrt(252)

print(f"Sharpe Ratio (with costs): {sharpe_ratio:.2f}")
```

---

### Pitfall 3: Overfitting in Deep Learning

**Problem:** Models achieve 95% accuracy on training data but 55% out-of-sample.

**Example:**
```python
# WRONG: Heavy feature engineering on full dataset
features = engineer_features(all_data)  # All 2000 observations!
model = LSTM(features, returns, epochs=100)
train_accuracy = 0.95
test_accuracy = 0.50  # Massive drop!
```

**Solution:**
```python
# RIGHT: Walk-forward validation with regularization
def walk_forward_lstm(data, window=500, test_period=250, step=50):
    results = []

    for i in range(0, len(data) - window - test_period, step):
        train_X = data[i:i+window]
        train_y = returns[i:i+window]
        test_X = data[i+window:i+window+test_period]
        test_y = returns[i+window:i+window+test_period]

        # Fit with regularization
        model = LSTM(
            units=32,
            dropout=0.3,  # Regularization
            recurrent_dropout=0.3,  # Regularization
            epochs=50
        )
        model.fit(train_X, train_y, validation_split=0.2, verbose=0)

        train_acc = model.evaluate(train_X, train_y)
        test_acc = model.evaluate(test_X, test_y)

        results.append({'train_acc': train_acc, 'test_acc': test_acc})

    return pd.DataFrame(results)

wf_results = walk_forward_lstm(features_data)
print(wf_results.mean())
# More realistic: train_acc ~ 0.58, test_acc ~ 0.54
```

---

**Document Version:** 1.0
**Last Updated:** December 21, 2025
