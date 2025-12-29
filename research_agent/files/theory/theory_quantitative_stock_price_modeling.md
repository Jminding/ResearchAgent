# Theoretical Framework: Quantitative Stock Price Modeling via Stochastic Differential Equations

## 1. Problem Formalization

### 1.1 Objective

Develop a tractable stochastic differential equation (SDE) model for stock price dynamics that:
1. Captures empirical stylized facts (volatility clustering, fat tails, mean reversion)
2. Admits closed-form or semi-closed-form solutions for pricing and estimation
3. Provides testable predictions against observed market data

### 1.2 Notation and Variables

| Symbol | Description | Domain |
|--------|-------------|--------|
| S(t) | Stock price at time t | R+ |
| r | Risk-free interest rate | R |
| sigma(t) | Instantaneous volatility at time t | R+ |
| V(t) | Instantaneous variance V(t) = sigma(t)^2 | R+ |
| W_S(t) | Wiener process driving stock price | R |
| W_V(t) | Wiener process driving variance | R |
| rho | Correlation between W_S and W_V | [-1, 1] |
| mu | Drift rate (expected return) | R |
| kappa | Mean reversion speed for variance | R+ |
| theta | Long-run variance level | R+ |
| xi | Volatility of volatility | R+ |
| T | Terminal time / maturity | R+ |
| Delta_t | Discrete time step | R+ |
| N | Number of observations | N |

---

## 2. Model Specification

### 2.1 Base Model: Geometric Brownian Motion (GBM)

The classical reference model:

```
dS(t) = mu * S(t) * dt + sigma * S(t) * dW_S(t)
```

**Solution:**
```
S(t) = S(0) * exp((mu - sigma^2/2) * t + sigma * W_S(t))
```

**Limitations:** Constant volatility assumption contradicts empirical evidence.

### 2.2 Primary Model: Heston Stochastic Volatility Model

We adopt the Heston (1993) model as our primary framework:

```
dS(t) = mu * S(t) * dt + sqrt(V(t)) * S(t) * dW_S(t)
dV(t) = kappa * (theta - V(t)) * dt + xi * sqrt(V(t)) * dW_V(t)
```

with correlation structure:
```
dW_S(t) * dW_V(t) = rho * dt
```

**Parameter Vector:**
```
Theta = (mu, kappa, theta, xi, rho, V_0)
```

### 2.3 Feller Condition

For the variance process to remain strictly positive:

```
2 * kappa * theta >= xi^2
```

This constraint must be enforced during estimation.

### 2.4 Log-Price Dynamics

Define X(t) = log(S(t)). By Ito's lemma:

```
dX(t) = (mu - V(t)/2) * dt + sqrt(V(t)) * dW_S(t)
```

---

## 3. Assumptions

### 3.1 Statistical Assumptions

**A1 (Markovian Structure):** The joint process (S(t), V(t)) is Markovian.

**A2 (Stationarity of Variance):** The variance process V(t) is ergodic and stationary with:
```
E[V(infinity)] = theta
Var[V(infinity)] = (xi^2 * theta) / (2 * kappa)
```

**A3 (Observation Model):** Discrete observations are obtained at regular intervals:
```
{S(t_i) : t_i = i * Delta_t, i = 0, 1, ..., N}
```

**A4 (No Jumps):** Price paths are continuous (no jump discontinuities).

**A5 (Constant Parameters):** All parameters in Theta are constant over the estimation window.

### 3.2 Market Assumptions

**A6 (No Arbitrage):** The market is arbitrage-free.

**A7 (Liquidity):** The asset is sufficiently liquid that observed prices reflect fair value.

**A8 (No Transaction Costs):** Frictionless trading for theoretical development.

---

## 4. Hypothesis

### 4.1 Primary Hypothesis (H1)

**Statement:**
"The Heston stochastic volatility model provides a statistically superior fit to empirical stock return distributions compared to the geometric Brownian motion model, as measured by log-likelihood ratio and information criteria, when applied to high-frequency or daily return data."

**Formal Expression:**
```
H0: L_GBM >= L_Heston (GBM is adequate)
H1: L_Heston > L_GBM + chi^2_{0.95,3}/2 (Heston is significantly better)
```

where L denotes log-likelihood and the threshold accounts for 3 additional parameters.

### 4.2 Secondary Hypothesis (H2)

**Statement:**
"Estimated model parameters satisfy the Feller condition (2*kappa*theta >= xi^2) for liquid, large-cap equities but may violate it for small-cap or illiquid securities."

### 4.3 Falsification Criteria

The primary hypothesis is **falsified** if:
1. The likelihood ratio test statistic LRT = 2*(L_Heston - L_GBM) < chi^2_{0.95,3} = 7.815
2. The Akaike Information Criterion favors GBM: AIC_Heston > AIC_GBM
3. Model residuals exhibit significant autocorrelation (Ljung-Box p < 0.05)

The primary hypothesis is **confirmed** if:
1. LRT > 7.815 with p-value < 0.05
2. AIC_Heston < AIC_GBM AND BIC_Heston < BIC_GBM
3. Standardized residuals pass normality tests (Jarque-Bera p > 0.05)
4. Volatility forecasts from Heston outperform GBM in out-of-sample RMSE

---

## 5. Estimation Methodology

### 5.1 Log-Returns Transformation

Define discrete log-returns:
```
r_i = log(S(t_i)) - log(S(t_{i-1})) = X(t_i) - X(t_{i-1})
```

### 5.2 Transition Density (Exact)

The Heston model admits a characteristic function in closed form:

```
phi(u; X_0, V_0, t) = E[exp(i*u*X(t)) | X(0)=X_0, V(0)=V_0]
                    = exp(C(u,t) + D(u,t)*V_0 + i*u*X_0)
```

where:
```
C(u,t) = mu*i*u*t + (kappa*theta/xi^2) * [(kappa - rho*xi*i*u + d)*t - 2*log((1-g*exp(d*t))/(1-g))]

D(u,t) = ((kappa - rho*xi*i*u + d)/xi^2) * (1 - exp(d*t))/(1 - g*exp(d*t))

d = sqrt((rho*xi*i*u - kappa)^2 + xi^2*(i*u + u^2))

g = (kappa - rho*xi*i*u + d) / (kappa - rho*xi*i*u - d)
```

### 5.3 Approximate Likelihood via FFT

The transition density p(X_t | X_0, V_0) is obtained by Fourier inversion:

```
p(x; X_0, V_0, t) = (1/(2*pi)) * integral_{-infinity}^{infinity} exp(-i*u*x) * phi(u) du
```

Discretized via FFT:
```
p_j approx (1/N_FFT) * sum_{k=0}^{N_FFT-1} exp(-2*pi*i*j*k/N_FFT) * phi(u_k) * w_k
```

where w_k are integration weights (Simpson's rule).

### 5.4 Particle Filter for Latent Variance

Since V(t) is unobserved, we employ a particle filter:

**State-Space Representation:**
```
State equation:    V_{t+1} = V_t + kappa*(theta - V_t)*Delta_t + xi*sqrt(V_t)*sqrt(Delta_t)*epsilon_V
Observation equation: r_t = (mu - V_t/2)*Delta_t + sqrt(V_t)*sqrt(Delta_t)*epsilon_S
```

with correlation Corr(epsilon_S, epsilon_V) = rho.

---

## 6. Pseudocode: Maximum Likelihood Estimation

### 6.1 Data Preparation

```
PROCEDURE PrepareData(raw_prices, frequency)
INPUT:
  - raw_prices: array of N+1 closing prices [P_0, P_1, ..., P_N]
  - frequency: string in {"daily", "weekly", "intraday"}

OUTPUT:
  - returns: array of N log-returns
  - Delta_t: time step in years

STEPS:
1. Set Delta_t based on frequency:
   - IF frequency == "daily" THEN Delta_t = 1/252
   - IF frequency == "weekly" THEN Delta_t = 1/52
   - IF frequency == "intraday" THEN Delta_t = 1/(252*78)  // 5-min bars

2. FOR i = 1 TO N:
     returns[i] = log(raw_prices[i]) - log(raw_prices[i-1])

3. Remove any NaN or Inf values from returns

4. RETURN (returns, Delta_t)
```

### 6.2 GBM Parameter Estimation (Closed-Form MLE)

```
PROCEDURE EstimateGBM(returns, Delta_t)
INPUT:
  - returns: array of N log-returns
  - Delta_t: time step

OUTPUT:
  - mu_hat: drift estimate
  - sigma_hat: volatility estimate
  - log_likelihood: log-likelihood value

STEPS:
1. Compute sample statistics:
   mean_r = (1/N) * sum(returns)
   var_r = (1/(N-1)) * sum((returns - mean_r)^2)

2. MLE estimates:
   sigma_hat = sqrt(var_r / Delta_t)
   mu_hat = mean_r / Delta_t + (sigma_hat^2) / 2

3. Compute log-likelihood:
   log_likelihood = 0
   FOR i = 1 TO N:
     z_i = (returns[i] - (mu_hat - sigma_hat^2/2)*Delta_t) / (sigma_hat*sqrt(Delta_t))
     log_likelihood = log_likelihood + log(NormalPDF(z_i)) - log(sigma_hat*sqrt(Delta_t))

4. RETURN (mu_hat, sigma_hat, log_likelihood)
```

### 6.3 Heston Parameter Estimation (Particle Filter MLE)

```
PROCEDURE EstimateHeston(returns, Delta_t, M_particles, max_iter, tolerance)
INPUT:
  - returns: array of N log-returns
  - Delta_t: time step
  - M_particles: number of particles (default 1000)
  - max_iter: maximum optimization iterations (default 500)
  - tolerance: convergence tolerance (default 1e-6)

OUTPUT:
  - Theta_hat: estimated parameter vector (mu, kappa, theta, xi, rho, V_0)
  - log_likelihood: maximized log-likelihood
  - variance_path: filtered variance estimates

STEPS:
1. Initialize parameter bounds:
   mu_bounds = [-0.5, 0.5]
   kappa_bounds = [0.1, 10.0]
   theta_bounds = [0.001, 1.0]
   xi_bounds = [0.01, 2.0]
   rho_bounds = [-0.99, 0.99]
   V0_bounds = [0.001, 1.0]

2. Set initial parameter guess:
   sigma_sample = std(returns) / sqrt(Delta_t)
   Theta_init = (mean(returns)/Delta_t, 2.0, sigma_sample^2, 0.3, -0.5, sigma_sample^2)

3. Define negative log-likelihood function:
   FUNCTION NegLogLik(Theta):
     (mu, kappa, theta, xi, rho, V_0) = Theta

     // Check Feller condition (soft constraint via penalty)
     IF 2*kappa*theta < xi^2 THEN
       penalty = 1e6 * (xi^2 - 2*kappa*theta)
     ELSE
       penalty = 0

     // Initialize particles
     V_particles = array of M_particles, all initialized to V_0
     weights = array of M_particles, all initialized to 1/M_particles
     log_lik = 0

     FOR t = 1 TO N:
       // Propagate particles (Euler-Maruyama)
       FOR m = 1 TO M_particles:
         epsilon_V = sample from Normal(0, 1)
         V_particles[m] = V_particles[m] + kappa*(theta - V_particles[m])*Delta_t
                          + xi*sqrt(max(V_particles[m], 0))*sqrt(Delta_t)*epsilon_V
         V_particles[m] = max(V_particles[m], 1e-8)  // Ensure positivity

       // Compute observation likelihoods
       FOR m = 1 TO M_particles:
         mean_r = (mu - V_particles[m]/2) * Delta_t
         std_r = sqrt(V_particles[m] * Delta_t)
         weights[m] = NormalPDF((returns[t] - mean_r) / std_r) / std_r

       // Normalize weights and compute marginal likelihood contribution
       sum_weights = sum(weights)
       IF sum_weights < 1e-300 THEN
         RETURN infinity  // Numerical failure
       log_lik = log_lik + log(sum_weights / M_particles)
       weights = weights / sum_weights

       // Resample if effective sample size too low
       ESS = 1 / sum(weights^2)
       IF ESS < M_particles / 2 THEN
         indices = systematic_resample(weights)
         V_particles = V_particles[indices]
         weights = array of M_particles, all set to 1/M_particles

     RETURN -log_lik + penalty

4. Optimize using constrained optimizer:
   result = minimize(NegLogLik, Theta_init,
                     method="L-BFGS-B",
                     bounds=[mu_bounds, kappa_bounds, theta_bounds, xi_bounds, rho_bounds, V0_bounds],
                     maxiter=max_iter, tol=tolerance)

5. Extract optimal parameters:
   Theta_hat = result.x
   log_likelihood = -result.fun

6. Run final particle filter pass to extract variance path:
   variance_path = ParticleFilterSmooth(returns, Theta_hat, M_particles*2)

7. RETURN (Theta_hat, log_likelihood, variance_path)
```

### 6.4 Systematic Resampling

```
PROCEDURE SystematicResample(weights)
INPUT:
  - weights: normalized particle weights (sum = 1)

OUTPUT:
  - indices: resampled particle indices

STEPS:
1. M = length(weights)
2. cumsum = cumulative_sum(weights)
3. u_0 = sample from Uniform(0, 1/M)
4. indices = empty array

5. j = 0
   FOR i = 0 TO M-1:
     u_i = u_0 + i/M
     WHILE cumsum[j] < u_i:
       j = j + 1
     append j to indices

6. RETURN indices
```

---

## 7. Pseudocode: Alternative Calibration via Method of Moments

### 7.1 Generalized Method of Moments (GMM)

```
PROCEDURE CalibrateHestonGMM(returns, Delta_t, lags)
INPUT:
  - returns: array of N log-returns
  - Delta_t: time step
  - lags: number of autocorrelation lags (default 10)

OUTPUT:
  - Theta_hat: estimated parameters
  - J_statistic: GMM overidentification test statistic

STEPS:
1. Compute sample moments:
   m1 = mean(returns)                          // First moment
   m2 = mean(returns^2)                        // Second moment (variance proxy)
   m3 = mean(returns^3)                        // Skewness proxy
   m4 = mean(returns^4)                        // Kurtosis proxy

   // Squared return autocorrelations (volatility clustering)
   FOR k = 1 TO lags:
     acf_sq[k] = autocorrelation(returns^2, lag=k)

2. Define theoretical moment functions for Heston:
   FUNCTION TheoreticalMoments(Theta):
     (mu, kappa, theta, xi, rho, V_0) = Theta

     // Under stationarity:
     E_V = theta
     Var_V = (xi^2 * theta) / (2 * kappa)

     // Unconditional return moments:
     E_r = (mu - theta/2) * Delta_t
     E_r2 = theta * Delta_t + E_r^2
     E_r3 = 3*rho*xi*theta*Delta_t^2 / kappa + ...  // Higher-order terms
     E_r4 = 3*theta^2*Delta_t^2 + 6*Var_V*Delta_t^2 + ...

     // Squared return autocorrelation:
     FOR k = 1 TO lags:
       acf_sq_theo[k] = exp(-kappa * k * Delta_t) * Var_V / (Var_V + ...)

     RETURN [E_r, E_r2, E_r3, E_r4, acf_sq_theo]

3. Define moment condition errors:
   FUNCTION MomentErrors(Theta):
     theo = TheoreticalMoments(Theta)
     errors = [m1 - theo[0], m2 - theo[1], m3 - theo[2], m4 - theo[3]]
     FOR k = 1 TO lags:
       append (acf_sq[k] - theo.acf_sq_theo[k]) to errors
     RETURN errors

4. Estimate optimal weighting matrix (HAC):
   W = NeweyWestCovarianceMatrix(MomentErrors(Theta_init))

5. GMM objective:
   FUNCTION GMMObjective(Theta):
     g = MomentErrors(Theta)
     RETURN g' * inv(W) * g

6. Two-step GMM:
   // Step 1: Identity weighting
   Theta_step1 = minimize(sum(MomentErrors(Theta)^2), Theta_init)

   // Step 2: Optimal weighting
   W = NeweyWestCovarianceMatrix(MomentErrors(Theta_step1))
   Theta_hat = minimize(GMMObjective, Theta_step1)

7. J-statistic for overidentification:
   g_hat = MomentErrors(Theta_hat)
   J = N * g_hat' * inv(W) * g_hat
   df = num_moments - num_parameters
   p_value = 1 - ChiSquareCDF(J, df)

8. RETURN (Theta_hat, J, p_value)
```

---

## 8. Pseudocode: Model Validation

### 8.1 Likelihood Ratio Test

```
PROCEDURE LikelihoodRatioTest(L_GBM, L_Heston, df)
INPUT:
  - L_GBM: log-likelihood of GBM model
  - L_Heston: log-likelihood of Heston model
  - df: degrees of freedom difference (default 3)

OUTPUT:
  - LRT: test statistic
  - p_value: p-value
  - reject_null: boolean (TRUE if Heston significantly better)

STEPS:
1. LRT = 2 * (L_Heston - L_GBM)
2. p_value = 1 - ChiSquareCDF(LRT, df)
3. reject_null = (p_value < 0.05)
4. RETURN (LRT, p_value, reject_null)
```

### 8.2 Information Criteria

```
PROCEDURE ComputeInformationCriteria(log_likelihood, num_params, N)
INPUT:
  - log_likelihood: maximized log-likelihood
  - num_params: number of model parameters
  - N: number of observations

OUTPUT:
  - AIC: Akaike Information Criterion
  - BIC: Bayesian Information Criterion
  - AICc: corrected AIC for small samples

STEPS:
1. AIC = -2 * log_likelihood + 2 * num_params
2. BIC = -2 * log_likelihood + num_params * log(N)
3. AICc = AIC + (2 * num_params * (num_params + 1)) / (N - num_params - 1)
4. RETURN (AIC, BIC, AICc)
```

### 8.3 Residual Diagnostics

```
PROCEDURE ResidualDiagnostics(returns, variance_path, Theta, Delta_t)
INPUT:
  - returns: observed log-returns
  - variance_path: filtered variance estimates
  - Theta: estimated parameters
  - Delta_t: time step

OUTPUT:
  - standardized_residuals: array
  - ljung_box_stat: Ljung-Box test statistic
  - ljung_box_pval: p-value
  - jarque_bera_stat: normality test statistic
  - jarque_bera_pval: p-value
  - diagnostics_pass: boolean

STEPS:
1. Extract parameters:
   (mu, kappa, theta, xi, rho, V_0) = Theta

2. Compute standardized residuals:
   FOR t = 1 TO N:
     expected_return = (mu - variance_path[t]/2) * Delta_t
     std_dev = sqrt(variance_path[t] * Delta_t)
     standardized_residuals[t] = (returns[t] - expected_return) / std_dev

3. Ljung-Box test for autocorrelation:
   K = min(20, N/5)  // Number of lags
   acf = autocorrelation(standardized_residuals, lags=K)
   ljung_box_stat = N * (N + 2) * sum_{k=1}^{K} (acf[k]^2 / (N - k))
   ljung_box_pval = 1 - ChiSquareCDF(ljung_box_stat, K)

4. Jarque-Bera normality test:
   skew = (1/N) * sum(standardized_residuals^3)
   kurt = (1/N) * sum(standardized_residuals^4)
   jarque_bera_stat = (N/6) * (skew^2 + (kurt - 3)^2 / 4)
   jarque_bera_pval = 1 - ChiSquareCDF(jarque_bera_stat, 2)

5. Overall diagnostic assessment:
   diagnostics_pass = (ljung_box_pval > 0.05) AND (jarque_bera_pval > 0.05)

6. RETURN (standardized_residuals, ljung_box_stat, ljung_box_pval,
           jarque_bera_stat, jarque_bera_pval, diagnostics_pass)
```

### 8.4 Out-of-Sample Validation

```
PROCEDURE OutOfSampleValidation(returns, Delta_t, train_ratio, forecast_horizon)
INPUT:
  - returns: full return series
  - Delta_t: time step
  - train_ratio: proportion for training (default 0.8)
  - forecast_horizon: days ahead to forecast variance (default 22)

OUTPUT:
  - rmse_gbm: RMSE of GBM variance forecasts
  - rmse_heston: RMSE of Heston variance forecasts
  - heston_wins: boolean (TRUE if Heston has lower RMSE)

STEPS:
1. Split data:
   N_train = floor(train_ratio * N)
   returns_train = returns[1:N_train]
   returns_test = returns[N_train+1:N]

2. Estimate models on training data:
   (mu_gbm, sigma_gbm, _) = EstimateGBM(returns_train, Delta_t)
   (Theta_heston, _, var_path_train) = EstimateHeston(returns_train, Delta_t)

3. Compute realized variance in test set (rolling window):
   FOR t = 1 TO length(returns_test) - forecast_horizon:
     realized_var[t] = mean(returns_test[t:t+forecast_horizon-1]^2) / Delta_t

4. Generate variance forecasts:
   // GBM: constant variance
   var_forecast_gbm = array of length(realized_var), all set to sigma_gbm^2

   // Heston: mean-reverting forecast
   (mu, kappa, theta, xi, rho, V_0) = Theta_heston
   V_current = var_path_train[end]
   FOR t = 1 TO length(realized_var):
     // E[V(t+h) | V(t)] = theta + (V(t) - theta) * exp(-kappa * h)
     h = t * Delta_t
     var_forecast_heston[t] = theta + (V_current - theta) * exp(-kappa * h)

5. Compute RMSE:
   rmse_gbm = sqrt(mean((realized_var - var_forecast_gbm)^2))
   rmse_heston = sqrt(mean((realized_var - var_forecast_heston)^2))

6. heston_wins = (rmse_heston < rmse_gbm)

7. RETURN (rmse_gbm, rmse_heston, heston_wins)
```

---

## 9. Complete Experimental Pipeline

```
PROCEDURE RunCompleteExperiment(ticker, start_date, end_date, data_source)
INPUT:
  - ticker: stock ticker symbol (e.g., "AAPL")
  - start_date: start of data period (e.g., "2015-01-01")
  - end_date: end of data period (e.g., "2024-12-01")
  - data_source: API or file path for price data

OUTPUT:
  - results: dictionary containing all estimation and validation results
  - figures: list of diagnostic plots

STEPS:
1. DATA ACQUISITION:
   raw_prices = fetch_price_data(ticker, start_date, end_date, data_source)
   (returns, Delta_t) = PrepareData(raw_prices, "daily")
   N = length(returns)

   PRINT "Loaded {N} observations for {ticker}"

2. PRELIMINARY ANALYSIS:
   // Summary statistics
   mean_ret = mean(returns)
   std_ret = std(returns)
   skewness = compute_skewness(returns)
   kurtosis = compute_kurtosis(returns)

   PRINT "Sample mean: {mean_ret}, std: {std_ret}, skew: {skewness}, kurt: {kurtosis}"

   // Test for ARCH effects (prerequisite for stochastic volatility)
   arch_test_stat = LagrangeMultiplierARCHTest(returns, lags=5)
   IF arch_test_stat.pvalue > 0.05 THEN
     WARN "No significant ARCH effects; stochastic volatility may not be necessary"

3. GBM ESTIMATION:
   (mu_gbm, sigma_gbm, L_gbm) = EstimateGBM(returns, Delta_t)
   (AIC_gbm, BIC_gbm, _) = ComputeInformationCriteria(L_gbm, 2, N)

   PRINT "GBM: mu={mu_gbm:.4f}, sigma={sigma_gbm:.4f}, LogLik={L_gbm:.2f}"

4. HESTON ESTIMATION:
   (Theta_heston, L_heston, var_path) = EstimateHeston(returns, Delta_t, M_particles=2000)
   (mu_h, kappa_h, theta_h, xi_h, rho_h, V0_h) = Theta_heston
   (AIC_heston, BIC_heston, _) = ComputeInformationCriteria(L_heston, 6, N)

   PRINT "Heston: mu={mu_h:.4f}, kappa={kappa_h:.4f}, theta={theta_h:.4f}"
   PRINT "        xi={xi_h:.4f}, rho={rho_h:.4f}, V0={V0_h:.4f}"
   PRINT "        LogLik={L_heston:.2f}"

5. FELLER CONDITION CHECK:
   feller_ratio = 2 * kappa_h * theta_h / (xi_h^2)
   feller_satisfied = (feller_ratio >= 1)
   PRINT "Feller ratio: {feller_ratio:.4f} (satisfied: {feller_satisfied})"

6. MODEL COMPARISON:
   (LRT, p_value, reject_H0) = LikelihoodRatioTest(L_gbm, L_heston, df=4)
   PRINT "Likelihood Ratio Test: LRT={LRT:.4f}, p-value={p_value:.6f}"

   IF reject_H0 THEN
     PRINT "RESULT: Heston model significantly outperforms GBM (p < 0.05)"
   ELSE
     PRINT "RESULT: Insufficient evidence to prefer Heston over GBM"

   PRINT "AIC - GBM: {AIC_gbm:.2f}, Heston: {AIC_heston:.2f}"
   PRINT "BIC - GBM: {BIC_gbm:.2f}, Heston: {BIC_heston:.2f}"

7. RESIDUAL DIAGNOSTICS:
   (resid, lb_stat, lb_pval, jb_stat, jb_pval, diag_pass) =
       ResidualDiagnostics(returns, var_path, Theta_heston, Delta_t)

   PRINT "Ljung-Box test: stat={lb_stat:.4f}, p-value={lb_pval:.4f}"
   PRINT "Jarque-Bera test: stat={jb_stat:.4f}, p-value={jb_pval:.4f}"
   PRINT "Residual diagnostics passed: {diag_pass}"

8. OUT-OF-SAMPLE VALIDATION:
   (rmse_gbm, rmse_heston, heston_wins) =
       OutOfSampleValidation(returns, Delta_t, train_ratio=0.8, forecast_horizon=22)

   PRINT "Out-of-sample variance forecast RMSE:"
   PRINT "  GBM: {rmse_gbm:.6f}"
   PRINT "  Heston: {rmse_heston:.6f}"
   PRINT "  Heston wins: {heston_wins}"

9. COMPILE RESULTS:
   results = {
     "ticker": ticker,
     "n_observations": N,
     "gbm_params": {"mu": mu_gbm, "sigma": sigma_gbm},
     "gbm_loglik": L_gbm,
     "gbm_aic": AIC_gbm,
     "gbm_bic": BIC_gbm,
     "heston_params": {"mu": mu_h, "kappa": kappa_h, "theta": theta_h,
                       "xi": xi_h, "rho": rho_h, "V0": V0_h},
     "heston_loglik": L_heston,
     "heston_aic": AIC_heston,
     "heston_bic": BIC_heston,
     "feller_ratio": feller_ratio,
     "feller_satisfied": feller_satisfied,
     "lrt_statistic": LRT,
     "lrt_pvalue": p_value,
     "heston_preferred": reject_H0 AND (AIC_heston < AIC_gbm),
     "ljung_box_pvalue": lb_pval,
     "jarque_bera_pvalue": jb_pval,
     "residuals_ok": diag_pass,
     "oos_rmse_gbm": rmse_gbm,
     "oos_rmse_heston": rmse_heston,
     "oos_heston_wins": heston_wins
   }

10. HYPOTHESIS EVALUATION:
    hypothesis_confirmed = reject_H0 AND (AIC_heston < AIC_gbm) AND heston_wins

    IF hypothesis_confirmed THEN
      PRINT "PRIMARY HYPOTHESIS CONFIRMED: Heston provides superior fit"
    ELSE
      PRINT "PRIMARY HYPOTHESIS FALSIFIED or INCONCLUSIVE"
      IF NOT reject_H0 THEN
        PRINT "  - LRT did not reject GBM"
      IF AIC_heston >= AIC_gbm THEN
        PRINT "  - AIC does not favor Heston"
      IF NOT heston_wins THEN
        PRINT "  - Out-of-sample forecasts favor GBM"

11. RETURN (results, figures)
```

---

## 10. Data Requirements

| Requirement | Specification |
|-------------|---------------|
| **Data type** | Daily adjusted closing prices |
| **Minimum observations** | N >= 500 (approximately 2 years of trading days) |
| **Recommended observations** | N >= 2520 (10 years) for robust estimation |
| **Frequency** | Daily preferred; weekly acceptable; intraday requires microstructure adjustments |
| **Data quality** | No missing values; adjusted for splits and dividends |
| **Assets** | Individual stocks (preferably liquid, large-cap for benchmark testing) |

---

## 11. Parameter Specifications

### 11.1 Estimation Parameters

| Parameter | Default Value | Acceptable Range |
|-----------|---------------|------------------|
| M_particles | 2000 | [500, 10000] |
| max_iter | 500 | [100, 2000] |
| tolerance | 1e-6 | [1e-8, 1e-4] |
| lags (GMM) | 10 | [5, 30] |
| train_ratio | 0.8 | [0.6, 0.9] |
| forecast_horizon | 22 | [5, 63] (1 week to 1 quarter) |

### 11.2 Statistical Thresholds

| Test | Threshold | Interpretation |
|------|-----------|----------------|
| LRT p-value | 0.05 | Reject GBM if p < 0.05 |
| Ljung-Box p-value | 0.05 | Residual autocorrelation if p < 0.05 |
| Jarque-Bera p-value | 0.05 | Non-normality if p < 0.05 |
| Feller ratio | 1.0 | Variance stays positive if ratio >= 1 |

---

## 12. Expected Outcomes

### 12.1 If Hypothesis is Confirmed

- LRT statistic > 7.815 (chi-square critical value at df=4, alpha=0.05)
- AIC_Heston < AIC_GBM by at least 10 points
- Estimated rho < 0 (leverage effect)
- Residuals pass both Ljung-Box and Jarque-Bera tests
- Out-of-sample RMSE reduction of at least 10%

### 12.2 If Hypothesis is Falsified

- LRT statistic < 7.815
- AIC_GBM <= AIC_Heston
- Residuals exhibit significant autocorrelation or non-normality
- Out-of-sample performance favors GBM

### 12.3 Model Limitations to Report

1. Particle filter estimation introduces Monte Carlo error
2. Feller condition violations require careful numerical handling
3. Model assumes no jumps (may miss extreme events)
4. Constant parameters may not capture regime changes
5. Correlation estimate rho is often imprecise

---

## 13. Extensions for Future Work

1. **Jump-Diffusion Extension:** Add Poisson jumps to capture extreme moves
2. **Regime-Switching:** Allow parameters to vary across market regimes
3. **Multifactor Models:** Add second volatility factor for term structure
4. **Bayesian Estimation:** Replace MLE with MCMC for uncertainty quantification
5. **High-Frequency Adaptation:** Modify for intraday data with microstructure noise

---

## 14. Summary

This theoretical framework provides:

1. **Mathematical Model:** Heston stochastic volatility SDE with explicit dynamics
2. **Testable Hypothesis:** Heston outperforms GBM in likelihood and forecasting
3. **Estimation Procedures:** Particle filter MLE and GMM pseudocode
4. **Validation Suite:** LRT, information criteria, residual diagnostics, out-of-sample tests
5. **Falsification Criteria:** Clear conditions under which the hypothesis is rejected

The Experimentalist can implement this framework verbatim using any scientific computing language (Python/NumPy, R, Julia, MATLAB) following the numbered pseudocode steps.
