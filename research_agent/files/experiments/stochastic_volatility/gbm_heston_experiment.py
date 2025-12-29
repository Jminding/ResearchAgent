"""
GBM and Heston Stochastic Volatility Model Implementation
=========================================================

This module implements:
1. Geometric Brownian Motion (GBM) parameter estimation via MLE
2. Heston Stochastic Volatility model estimation via Particle Filter MLE
3. Complete validation suite:
   - Likelihood Ratio Test
   - Information Criteria (AIC/BIC)
   - Residual Diagnostics (Ljung-Box, Jarque-Bera)
   - Out-of-Sample Variance Forecasting

Based on the theoretical framework from:
files/theory/theory_quantitative_stock_price_modeling.md

Author: Experimentalist Agent
Date: 2025-12-21
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from datetime import datetime
import json
import warnings
import os

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

def prepare_data(raw_prices, frequency='daily'):
    """
    Prepare log-returns from raw prices.

    Parameters:
    -----------
    raw_prices : array-like
        Array of closing prices
    frequency : str
        'daily', 'weekly', or 'intraday'

    Returns:
    --------
    returns : np.array
        Log-returns
    Delta_t : float
        Time step in years
    """
    if frequency == 'daily':
        Delta_t = 1/252
    elif frequency == 'weekly':
        Delta_t = 1/52
    elif frequency == 'intraday':
        Delta_t = 1/(252*78)
    else:
        Delta_t = 1/252

    prices = np.array(raw_prices)
    log_prices = np.log(prices)
    returns = np.diff(log_prices)

    # Remove NaN and Inf
    mask = np.isfinite(returns)
    returns = returns[mask]

    return returns, Delta_t


# ==============================================================================
# GBM MODEL ESTIMATION (Closed-Form MLE)
# ==============================================================================

def estimate_gbm(returns, Delta_t):
    """
    Estimate GBM parameters using closed-form MLE.

    Model: dS(t) = mu * S(t) * dt + sigma * S(t) * dW(t)

    Parameters:
    -----------
    returns : np.array
        Log-returns
    Delta_t : float
        Time step

    Returns:
    --------
    params : dict
        {'mu': mu_hat, 'sigma': sigma_hat}
    log_likelihood : float
        Maximized log-likelihood
    residuals : np.array
        Standardized residuals
    """
    N = len(returns)

    # Sample statistics
    mean_r = np.mean(returns)
    var_r = np.var(returns, ddof=1)

    # MLE estimates
    sigma_hat = np.sqrt(var_r / Delta_t)
    mu_hat = mean_r / Delta_t + (sigma_hat**2) / 2

    # Compute log-likelihood
    log_likelihood = 0.0
    standardized_residuals = np.zeros(N)

    for i in range(N):
        expected_return = (mu_hat - sigma_hat**2 / 2) * Delta_t
        std_dev = sigma_hat * np.sqrt(Delta_t)
        z_i = (returns[i] - expected_return) / std_dev
        standardized_residuals[i] = z_i
        log_likelihood += stats.norm.logpdf(z_i) - np.log(std_dev)

    params = {
        'mu': mu_hat,
        'sigma': sigma_hat
    }

    return params, log_likelihood, standardized_residuals


# ==============================================================================
# HESTON MODEL ESTIMATION (Particle Filter MLE)
# ==============================================================================

def systematic_resample(weights):
    """
    Systematic resampling for particle filter.

    Parameters:
    -----------
    weights : np.array
        Normalized particle weights

    Returns:
    --------
    indices : np.array
        Resampled particle indices
    """
    M = len(weights)
    cumsum = np.cumsum(weights)
    u_0 = np.random.uniform(0, 1/M)

    indices = np.zeros(M, dtype=int)
    j = 0
    for i in range(M):
        u_i = u_0 + i/M
        while cumsum[j] < u_i and j < M - 1:
            j += 1
        indices[i] = j

    return indices


def particle_filter_likelihood(params, returns, Delta_t, M_particles=1000):
    """
    Compute log-likelihood using particle filter for Heston model.

    Model:
    dS(t) = mu * S(t) * dt + sqrt(V(t)) * S(t) * dW_S(t)
    dV(t) = kappa * (theta - V(t)) * dt + xi * sqrt(V(t)) * dW_V(t)

    Parameters:
    -----------
    params : tuple
        (mu, kappa, theta, xi, rho, V_0)
    returns : np.array
        Log-returns
    Delta_t : float
        Time step
    M_particles : int
        Number of particles

    Returns:
    --------
    neg_log_lik : float
        Negative log-likelihood (for minimization)
    """
    mu, kappa, theta, xi, rho, V_0 = params
    N = len(returns)

    # Feller condition penalty (soft constraint)
    feller_ratio = 2 * kappa * theta / (xi**2 + 1e-10)
    if feller_ratio < 1:
        penalty = 1e6 * (1 - feller_ratio)
    else:
        penalty = 0

    # Parameter validity checks
    if kappa <= 0 or theta <= 0 or xi <= 0 or V_0 <= 0:
        return 1e10
    if np.abs(rho) >= 1:
        return 1e10

    # Initialize particles
    V_particles = np.ones(M_particles) * V_0
    weights = np.ones(M_particles) / M_particles
    log_lik = 0.0

    sqrt_dt = np.sqrt(Delta_t)
    sqrt_1_minus_rho2 = np.sqrt(1 - rho**2)

    for t in range(N):
        # Propagate particles (Euler-Maruyama discretization)
        epsilon_V = np.random.randn(M_particles)
        epsilon_S = rho * epsilon_V + sqrt_1_minus_rho2 * np.random.randn(M_particles)

        # Variance dynamics
        V_particles_sqrt = np.sqrt(np.maximum(V_particles, 1e-8))
        V_particles = V_particles + kappa * (theta - V_particles) * Delta_t \
                      + xi * V_particles_sqrt * sqrt_dt * epsilon_V
        V_particles = np.maximum(V_particles, 1e-8)  # Ensure positivity

        # Compute observation likelihoods
        mean_r = (mu - V_particles / 2) * Delta_t
        std_r = np.sqrt(V_particles * Delta_t)

        # Avoid numerical issues
        std_r = np.maximum(std_r, 1e-10)

        # Likelihood weights
        z_scores = (returns[t] - mean_r) / std_r
        weights = np.exp(-0.5 * z_scores**2) / (std_r * np.sqrt(2 * np.pi))

        # Marginal likelihood contribution
        sum_weights = np.sum(weights)
        if sum_weights < 1e-300:
            return 1e10  # Numerical failure

        log_lik += np.log(sum_weights / M_particles)

        # Normalize weights
        weights = weights / sum_weights

        # Resample if effective sample size is too low
        ESS = 1.0 / np.sum(weights**2)
        if ESS < M_particles / 2:
            indices = systematic_resample(weights)
            V_particles = V_particles[indices]
            weights = np.ones(M_particles) / M_particles

    return -log_lik + penalty


def estimate_heston(returns, Delta_t, M_particles=1000, max_iter=300, verbose=True):
    """
    Estimate Heston model parameters using particle filter MLE.

    Parameters:
    -----------
    returns : np.array
        Log-returns
    Delta_t : float
        Time step
    M_particles : int
        Number of particles
    max_iter : int
        Maximum optimization iterations
    verbose : bool
        Print progress

    Returns:
    --------
    params : dict
        Estimated parameters
    log_likelihood : float
        Maximized log-likelihood
    variance_path : np.array
        Filtered variance estimates
    """
    if verbose:
        print("Estimating Heston model parameters...")

    # Initial parameter guess based on sample statistics
    sigma_sample = np.std(returns) / np.sqrt(Delta_t)

    # Initial guess: (mu, kappa, theta, xi, rho, V_0)
    x0 = np.array([
        np.mean(returns) / Delta_t,  # mu
        2.0,                           # kappa
        sigma_sample**2,               # theta
        0.3,                           # xi
        -0.5,                          # rho
        sigma_sample**2                # V_0
    ])

    # Parameter bounds
    bounds = [
        (-0.5, 0.5),      # mu
        (0.1, 15.0),      # kappa
        (0.001, 1.0),     # theta
        (0.01, 2.0),      # xi
        (-0.99, 0.99),    # rho
        (0.001, 1.0)      # V_0
    ]

    # Optimize using L-BFGS-B
    def objective(params):
        return particle_filter_likelihood(params, returns, Delta_t, M_particles)

    result = minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'disp': verbose}
    )

    # Extract optimal parameters
    mu, kappa, theta, xi, rho, V_0 = result.x
    log_likelihood = -result.fun

    params = {
        'mu': mu,
        'kappa': kappa,
        'theta': theta,
        'xi': xi,
        'rho': rho,
        'V_0': V_0
    }

    if verbose:
        print(f"Optimization converged: {result.success}")
        print(f"Log-likelihood: {log_likelihood:.4f}")

    # Run final pass to extract variance path
    variance_path = run_particle_filter_smoother(returns, params, Delta_t, M_particles * 2)

    return params, log_likelihood, variance_path


def run_particle_filter_smoother(returns, params, Delta_t, M_particles=2000):
    """
    Run particle filter to extract filtered variance path.

    Parameters:
    -----------
    returns : np.array
        Log-returns
    params : dict
        Heston parameters
    Delta_t : float
        Time step
    M_particles : int
        Number of particles

    Returns:
    --------
    variance_path : np.array
        Weighted mean variance at each time step
    """
    mu = params['mu']
    kappa = params['kappa']
    theta = params['theta']
    xi = params['xi']
    rho = params['rho']
    V_0 = params['V_0']

    N = len(returns)
    variance_path = np.zeros(N)

    V_particles = np.ones(M_particles) * V_0
    weights = np.ones(M_particles) / M_particles

    sqrt_dt = np.sqrt(Delta_t)
    sqrt_1_minus_rho2 = np.sqrt(1 - rho**2)

    for t in range(N):
        # Propagate particles
        epsilon_V = np.random.randn(M_particles)

        V_particles_sqrt = np.sqrt(np.maximum(V_particles, 1e-8))
        V_particles = V_particles + kappa * (theta - V_particles) * Delta_t \
                      + xi * V_particles_sqrt * sqrt_dt * epsilon_V
        V_particles = np.maximum(V_particles, 1e-8)

        # Compute observation likelihoods
        mean_r = (mu - V_particles / 2) * Delta_t
        std_r = np.sqrt(np.maximum(V_particles * Delta_t, 1e-10))

        z_scores = (returns[t] - mean_r) / std_r
        weights = np.exp(-0.5 * z_scores**2) / (std_r * np.sqrt(2 * np.pi))

        sum_weights = np.sum(weights)
        if sum_weights > 1e-300:
            weights = weights / sum_weights
        else:
            weights = np.ones(M_particles) / M_particles

        # Store weighted mean variance
        variance_path[t] = np.sum(weights * V_particles)

        # Resample
        ESS = 1.0 / np.sum(weights**2)
        if ESS < M_particles / 2:
            indices = systematic_resample(weights)
            V_particles = V_particles[indices]
            weights = np.ones(M_particles) / M_particles

    return variance_path


def compute_heston_residuals(returns, variance_path, params, Delta_t):
    """
    Compute standardized residuals for Heston model.

    Parameters:
    -----------
    returns : np.array
        Log-returns
    variance_path : np.array
        Filtered variance path
    params : dict
        Heston parameters
    Delta_t : float
        Time step

    Returns:
    --------
    residuals : np.array
        Standardized residuals
    """
    mu = params['mu']
    N = len(returns)
    residuals = np.zeros(N)

    for t in range(N):
        expected_return = (mu - variance_path[t] / 2) * Delta_t
        std_dev = np.sqrt(variance_path[t] * Delta_t)
        if std_dev < 1e-10:
            std_dev = 1e-10
        residuals[t] = (returns[t] - expected_return) / std_dev

    return residuals


# ==============================================================================
# MODEL VALIDATION
# ==============================================================================

def likelihood_ratio_test(L_GBM, L_Heston, df=4):
    """
    Perform Likelihood Ratio Test comparing GBM to Heston.

    H0: GBM is adequate
    H1: Heston is significantly better

    Parameters:
    -----------
    L_GBM : float
        Log-likelihood of GBM
    L_Heston : float
        Log-likelihood of Heston
    df : int
        Degrees of freedom difference (Heston has 4 more params)

    Returns:
    --------
    dict : Test results
    """
    LRT = 2 * (L_Heston - L_GBM)
    p_value = 1 - stats.chi2.cdf(LRT, df)
    reject_null = p_value < 0.05

    # Critical value at alpha=0.05
    critical_value = stats.chi2.ppf(0.95, df)

    return {
        'LRT_statistic': LRT,
        'p_value': p_value,
        'critical_value': critical_value,
        'df': df,
        'reject_null': reject_null,
        'interpretation': 'Heston significantly better' if reject_null else 'GBM adequate'
    }


def compute_information_criteria(log_likelihood, num_params, N):
    """
    Compute AIC, BIC, and AICc.

    Parameters:
    -----------
    log_likelihood : float
        Maximized log-likelihood
    num_params : int
        Number of model parameters
    N : int
        Number of observations

    Returns:
    --------
    dict : Information criteria
    """
    AIC = -2 * log_likelihood + 2 * num_params
    BIC = -2 * log_likelihood + num_params * np.log(N)

    # Corrected AIC for small samples
    if N - num_params - 1 > 0:
        AICc = AIC + (2 * num_params * (num_params + 1)) / (N - num_params - 1)
    else:
        AICc = AIC

    return {
        'AIC': AIC,
        'BIC': BIC,
        'AICc': AICc
    }


def ljung_box_test(residuals, lags=20):
    """
    Perform Ljung-Box test for autocorrelation in residuals.

    H0: No autocorrelation
    H1: Significant autocorrelation exists

    Parameters:
    -----------
    residuals : np.array
        Standardized residuals
    lags : int
        Number of lags to test

    Returns:
    --------
    dict : Test results
    """
    N = len(residuals)
    K = min(lags, N // 5)  # Limit lags

    # Compute autocorrelations
    acf = np.zeros(K)
    for k in range(1, K + 1):
        acf[k-1] = np.corrcoef(residuals[k:], residuals[:-k])[0, 1]

    # Ljung-Box statistic
    Q = N * (N + 2) * np.sum(acf**2 / (N - np.arange(1, K+1)))
    p_value = 1 - stats.chi2.cdf(Q, K)

    return {
        'statistic': Q,
        'p_value': p_value,
        'lags': K,
        'no_autocorrelation': p_value > 0.05,
        'interpretation': 'No significant autocorrelation' if p_value > 0.05 else 'Autocorrelation detected'
    }


def jarque_bera_test(residuals):
    """
    Perform Jarque-Bera test for normality.

    H0: Residuals are normally distributed
    H1: Residuals are not normal

    Parameters:
    -----------
    residuals : np.array
        Standardized residuals

    Returns:
    --------
    dict : Test results
    """
    N = len(residuals)

    # Skewness and kurtosis
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)  # Excess kurtosis

    # Jarque-Bera statistic
    JB = (N / 6) * (skew**2 + (kurt**2) / 4)
    p_value = 1 - stats.chi2.cdf(JB, 2)

    return {
        'statistic': JB,
        'p_value': p_value,
        'skewness': skew,
        'excess_kurtosis': kurt,
        'is_normal': p_value > 0.05,
        'interpretation': 'Normally distributed' if p_value > 0.05 else 'Non-normal distribution'
    }


def residual_diagnostics(returns, variance_path, params, Delta_t, model_name='Heston'):
    """
    Complete residual diagnostics suite.

    Parameters:
    -----------
    returns : np.array
        Log-returns
    variance_path : np.array
        Filtered variance (or constant for GBM)
    params : dict
        Model parameters
    Delta_t : float
        Time step
    model_name : str
        'GBM' or 'Heston'

    Returns:
    --------
    dict : Diagnostic results
    """
    if model_name == 'GBM':
        mu = params['mu']
        sigma = params['sigma']
        N = len(returns)
        residuals = np.zeros(N)
        for t in range(N):
            expected_return = (mu - sigma**2 / 2) * Delta_t
            std_dev = sigma * np.sqrt(Delta_t)
            residuals[t] = (returns[t] - expected_return) / std_dev
    else:
        residuals = compute_heston_residuals(returns, variance_path, params, Delta_t)

    lb_result = ljung_box_test(residuals)
    jb_result = jarque_bera_test(residuals)

    diagnostics_pass = lb_result['no_autocorrelation'] and jb_result['is_normal']

    return {
        'model': model_name,
        'residuals': residuals,
        'ljung_box': lb_result,
        'jarque_bera': jb_result,
        'diagnostics_pass': diagnostics_pass
    }


def out_of_sample_validation(returns, Delta_t, train_ratio=0.8, forecast_horizon=22):
    """
    Out-of-sample variance forecasting comparison.

    Parameters:
    -----------
    returns : np.array
        Full return series
    Delta_t : float
        Time step
    train_ratio : float
        Proportion for training
    forecast_horizon : int
        Days ahead to forecast

    Returns:
    --------
    dict : Out-of-sample results
    """
    N = len(returns)
    N_train = int(train_ratio * N)

    returns_train = returns[:N_train]
    returns_test = returns[N_train:]

    print("\n--- Out-of-Sample Validation ---")
    print(f"Training: {N_train} observations")
    print(f"Testing: {len(returns_test)} observations")

    # Estimate models on training data
    gbm_params, gbm_ll, gbm_resid = estimate_gbm(returns_train, Delta_t)
    heston_params, heston_ll, var_path_train = estimate_heston(
        returns_train, Delta_t, M_particles=500, max_iter=150, verbose=False
    )

    # Compute realized variance in test set (rolling window)
    N_test = len(returns_test)
    n_forecasts = max(1, N_test - forecast_horizon)

    realized_var = np.zeros(n_forecasts)
    for t in range(n_forecasts):
        window = returns_test[t:t+forecast_horizon]
        realized_var[t] = np.var(window) / Delta_t

    # GBM variance forecast: constant
    var_forecast_gbm = np.ones(n_forecasts) * gbm_params['sigma']**2

    # Heston variance forecast: mean-reverting
    kappa = heston_params['kappa']
    theta = heston_params['theta']
    V_current = var_path_train[-1] if len(var_path_train) > 0 else theta

    var_forecast_heston = np.zeros(n_forecasts)
    for t in range(n_forecasts):
        h = (t + 1) * forecast_horizon * Delta_t
        var_forecast_heston[t] = theta + (V_current - theta) * np.exp(-kappa * h)

    # Compute RMSE
    rmse_gbm = np.sqrt(np.mean((realized_var - var_forecast_gbm)**2))
    rmse_heston = np.sqrt(np.mean((realized_var - var_forecast_heston)**2))

    # Compute MAE
    mae_gbm = np.mean(np.abs(realized_var - var_forecast_gbm))
    mae_heston = np.mean(np.abs(realized_var - var_forecast_heston))

    heston_wins = rmse_heston < rmse_gbm

    print(f"GBM RMSE: {rmse_gbm:.6f}")
    print(f"Heston RMSE: {rmse_heston:.6f}")
    print(f"Heston wins: {heston_wins}")

    return {
        'n_train': N_train,
        'n_test': len(returns_test),
        'forecast_horizon': forecast_horizon,
        'rmse_gbm': rmse_gbm,
        'rmse_heston': rmse_heston,
        'mae_gbm': mae_gbm,
        'mae_heston': mae_heston,
        'heston_wins': heston_wins,
        'improvement_pct': (rmse_gbm - rmse_heston) / rmse_gbm * 100 if rmse_gbm > 0 else 0,
        'realized_var': realized_var,
        'var_forecast_gbm': var_forecast_gbm,
        'var_forecast_heston': var_forecast_heston
    }


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_diagnostics(results, output_dir):
    """
    Generate diagnostic plots.

    Parameters:
    -----------
    results : dict
        Complete experiment results
    output_dir : str
        Output directory for plots
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 1. Price History
    ax = axes[0, 0]
    dates = pd.to_datetime(results['dates'])
    ax.plot(dates, results['prices'], 'b-', linewidth=0.5)
    ax.set_title('AAPL Stock Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, alpha=0.3)

    # 2. Returns Distribution
    ax = axes[0, 1]
    returns = results['returns']
    ax.hist(returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, stats.norm.pdf(x, np.mean(returns), np.std(returns)), 'r-', lw=2, label='Normal fit')
    ax.set_title('Log-Returns Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Filtered Variance (Heston)
    ax = axes[0, 2]
    if 'variance_path' in results and results['variance_path'] is not None:
        ax.plot(dates[1:], results['variance_path'], 'g-', linewidth=0.5)
        ax.axhline(y=results['heston_params']['theta'], color='r', linestyle='--',
                   label=f"theta={results['heston_params']['theta']:.4f}")
    ax.set_title('Heston Filtered Variance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. GBM Residuals Q-Q Plot
    ax = axes[1, 0]
    gbm_resid = results['gbm_diagnostics']['residuals']
    stats.probplot(gbm_resid, dist="norm", plot=ax)
    ax.set_title('GBM Residuals Q-Q Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Heston Residuals Q-Q Plot
    ax = axes[1, 1]
    heston_resid = results['heston_diagnostics']['residuals']
    stats.probplot(heston_resid, dist="norm", plot=ax)
    ax.set_title('Heston Residuals Q-Q Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 6. ACF of Squared Returns (Volatility Clustering)
    ax = axes[1, 2]
    squared_returns = returns**2
    N = len(squared_returns)
    lags = min(40, N // 4)
    acf_vals = [np.corrcoef(squared_returns[k:], squared_returns[:-k])[0, 1] for k in range(1, lags+1)]
    ax.bar(range(1, lags+1), acf_vals, color='steelblue', alpha=0.7)
    conf_int = 1.96 / np.sqrt(N)
    ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
    ax.set_title('ACF of Squared Returns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True, alpha=0.3)

    # 7. Model Comparison: Information Criteria
    ax = axes[2, 0]
    criteria = ['AIC', 'BIC', 'AICc']
    gbm_ic = results['gbm_ic']
    heston_ic = results['heston_ic']
    x_pos = np.arange(len(criteria))
    width = 0.35
    ax.bar(x_pos - width/2, [gbm_ic['AIC'], gbm_ic['BIC'], gbm_ic['AICc']],
           width, label='GBM', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, [heston_ic['AIC'], heston_ic['BIC'], heston_ic['AICc']],
           width, label='Heston', color='green', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(criteria)
    ax.set_title('Information Criteria Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 8. Out-of-Sample Variance Forecasts
    ax = axes[2, 1]
    oos = results['oos_validation']
    ax.plot(oos['realized_var'], 'b-', label='Realized Variance', alpha=0.7)
    ax.plot(oos['var_forecast_gbm'], 'r--', label=f"GBM (RMSE={oos['rmse_gbm']:.4f})")
    ax.plot(oos['var_forecast_heston'], 'g--', label=f"Heston (RMSE={oos['rmse_heston']:.4f})")
    ax.set_title('Out-of-Sample Variance Forecasts', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Variance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 9. Summary Statistics Table
    ax = axes[2, 2]
    ax.axis('off')
    table_data = [
        ['Metric', 'GBM', 'Heston'],
        ['Log-Likelihood', f"{results['gbm_loglik']:.2f}", f"{results['heston_loglik']:.2f}"],
        ['AIC', f"{gbm_ic['AIC']:.2f}", f"{heston_ic['AIC']:.2f}"],
        ['BIC', f"{gbm_ic['BIC']:.2f}", f"{heston_ic['BIC']:.2f}"],
        ['LRT p-value', '-', f"{results['lrt']['p_value']:.6f}"],
        ['Ljung-Box p', f"{results['gbm_diagnostics']['ljung_box']['p_value']:.4f}",
         f"{results['heston_diagnostics']['ljung_box']['p_value']:.4f}"],
        ['Jarque-Bera p', f"{results['gbm_diagnostics']['jarque_bera']['p_value']:.4f}",
         f"{results['heston_diagnostics']['jarque_bera']['p_value']:.4f}"],
        ['OOS RMSE', f"{oos['rmse_gbm']:.6f}", f"{oos['rmse_heston']:.6f}"]
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.3, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title('Model Comparison Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'diagnostic_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plots saved to: {plot_path}")

    return plot_path


def plot_residual_analysis(gbm_resid, heston_resid, output_dir):
    """
    Detailed residual analysis plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # GBM Residuals Histogram
    ax = axes[0, 0]
    ax.hist(gbm_resid, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='N(0,1)')
    ax.set_title('GBM Standardized Residuals', fontsize=11, fontweight='bold')
    ax.set_xlabel('Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heston Residuals Histogram
    ax = axes[0, 1]
    ax.hist(heston_resid, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='N(0,1)')
    ax.set_title('Heston Standardized Residuals', fontsize=11, fontweight='bold')
    ax.set_xlabel('Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual Time Series Comparison
    ax = axes[0, 2]
    ax.plot(gbm_resid, 'b-', alpha=0.5, linewidth=0.5, label='GBM')
    ax.plot(heston_resid, 'g-', alpha=0.5, linewidth=0.5, label='Heston')
    ax.set_title('Residual Time Series', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ACF of GBM Residuals
    ax = axes[1, 0]
    lags = min(30, len(gbm_resid) // 10)
    acf_gbm = [np.corrcoef(gbm_resid[k:], gbm_resid[:-k])[0, 1] for k in range(1, lags+1)]
    ax.bar(range(1, lags+1), acf_gbm, color='blue', alpha=0.7)
    conf = 1.96 / np.sqrt(len(gbm_resid))
    ax.axhline(y=conf, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf, color='r', linestyle='--', alpha=0.5)
    ax.set_title('ACF of GBM Residuals', fontsize=11, fontweight='bold')
    ax.set_xlabel('Lag')
    ax.grid(True, alpha=0.3)

    # ACF of Heston Residuals
    ax = axes[1, 1]
    acf_heston = [np.corrcoef(heston_resid[k:], heston_resid[:-k])[0, 1] for k in range(1, lags+1)]
    ax.bar(range(1, lags+1), acf_heston, color='green', alpha=0.7)
    ax.axhline(y=conf, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf, color='r', linestyle='--', alpha=0.5)
    ax.set_title('ACF of Heston Residuals', fontsize=11, fontweight='bold')
    ax.set_xlabel('Lag')
    ax.grid(True, alpha=0.3)

    # ACF of Squared Residuals
    ax = axes[1, 2]
    acf_sq_gbm = [np.corrcoef(gbm_resid[k:]**2, gbm_resid[:-k]**2)[0, 1] for k in range(1, lags+1)]
    acf_sq_heston = [np.corrcoef(heston_resid[k:]**2, heston_resid[:-k]**2)[0, 1] for k in range(1, lags+1)]
    x_pos = np.arange(1, lags+1)
    width = 0.4
    ax.bar(x_pos - width/2, acf_sq_gbm, width, label='GBM', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, acf_sq_heston, width, label='Heston', color='green', alpha=0.7)
    ax.axhline(y=conf, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf, color='r', linestyle='--', alpha=0.5)
    ax.set_title('ACF of Squared Residuals', fontsize=11, fontweight='bold')
    ax.set_xlabel('Lag')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'residual_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Residual analysis plots saved to: {plot_path}")

    return plot_path


# ==============================================================================
# MAIN EXPERIMENT PIPELINE
# ==============================================================================

def run_complete_experiment(ticker='AAPL', start_date='2013-01-01', end_date=None,
                            output_dir=None):
    """
    Run the complete GBM vs Heston experiment.

    Parameters:
    -----------
    ticker : str
        Stock ticker
    start_date : str
        Start date
    end_date : str
        End date (default: today)
    output_dir : str
        Output directory for results

    Returns:
    --------
    results : dict
        Complete experiment results
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if output_dir is None:
        output_dir = '/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results/stochastic_volatility'

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("STOCHASTIC VOLATILITY MODEL COMPARISON EXPERIMENT")
    print("GBM vs. Heston Model")
    print("=" * 70)
    print(f"\nTicker: {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Output Directory: {output_dir}\n")

    # ==========================================================================
    # 1. DATA ACQUISITION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 50)

    print(f"Downloading {ticker} data from Yahoo Finance...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data retrieved for {ticker}")

    prices = df['Close'].values
    dates = df.index.tolist()

    print(f"Downloaded {len(prices)} observations")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Prepare returns
    returns, Delta_t = prepare_data(prices, 'daily')
    N = len(returns)

    print(f"Computed {N} log-returns")
    print(f"Delta_t = {Delta_t:.6f} (1/252 years)")

    # ==========================================================================
    # 2. PRELIMINARY ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: PRELIMINARY ANALYSIS")
    print("=" * 50)

    # Summary statistics
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis

    print(f"\nSample Statistics:")
    print(f"  Mean return (daily): {mean_ret:.6f} ({mean_ret*252*100:.2f}% annualized)")
    print(f"  Std dev (daily): {std_ret:.6f} ({std_ret*np.sqrt(252)*100:.2f}% annualized)")
    print(f"  Skewness: {skewness:.4f}")
    print(f"  Excess Kurtosis: {kurtosis:.4f}")

    # ARCH-LM test for heteroskedasticity
    squared_returns = returns**2
    acf_sq = np.corrcoef(squared_returns[1:], squared_returns[:-1])[0, 1]
    print(f"\n  ACF(1) of squared returns: {acf_sq:.4f}")
    print(f"  (Significant ACF suggests volatility clustering)")

    # ==========================================================================
    # 3. GBM ESTIMATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: GBM ESTIMATION")
    print("=" * 50)

    gbm_params, gbm_loglik, gbm_residuals = estimate_gbm(returns, Delta_t)
    gbm_ic = compute_information_criteria(gbm_loglik, 2, N)

    print(f"\nGBM Parameters:")
    print(f"  mu (drift): {gbm_params['mu']:.6f} ({gbm_params['mu']*100:.2f}% annual)")
    print(f"  sigma (volatility): {gbm_params['sigma']:.6f} ({gbm_params['sigma']*100:.2f}% annual)")
    print(f"\nGBM Log-Likelihood: {gbm_loglik:.4f}")
    print(f"GBM AIC: {gbm_ic['AIC']:.4f}")
    print(f"GBM BIC: {gbm_ic['BIC']:.4f}")

    # ==========================================================================
    # 4. HESTON ESTIMATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: HESTON ESTIMATION")
    print("=" * 50)

    heston_params, heston_loglik, variance_path = estimate_heston(
        returns, Delta_t, M_particles=1000, max_iter=200, verbose=True
    )
    heston_ic = compute_information_criteria(heston_loglik, 6, N)

    print(f"\nHeston Parameters:")
    print(f"  mu (drift): {heston_params['mu']:.6f}")
    print(f"  kappa (mean reversion): {heston_params['kappa']:.6f}")
    print(f"  theta (long-run variance): {heston_params['theta']:.6f} ({np.sqrt(heston_params['theta'])*100:.2f}% vol)")
    print(f"  xi (vol of vol): {heston_params['xi']:.6f}")
    print(f"  rho (correlation): {heston_params['rho']:.6f}")
    print(f"  V_0 (initial variance): {heston_params['V_0']:.6f}")

    # Feller condition check
    feller_ratio = 2 * heston_params['kappa'] * heston_params['theta'] / (heston_params['xi']**2)
    feller_satisfied = feller_ratio >= 1
    print(f"\nFeller Condition (2*kappa*theta >= xi^2):")
    print(f"  Feller ratio: {feller_ratio:.4f}")
    print(f"  Satisfied: {feller_satisfied}")

    print(f"\nHeston Log-Likelihood: {heston_loglik:.4f}")
    print(f"Heston AIC: {heston_ic['AIC']:.4f}")
    print(f"Heston BIC: {heston_ic['BIC']:.4f}")

    # ==========================================================================
    # 5. MODEL COMPARISON (LIKELIHOOD RATIO TEST)
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 5: MODEL COMPARISON - LIKELIHOOD RATIO TEST")
    print("=" * 50)

    lrt_result = likelihood_ratio_test(gbm_loglik, heston_loglik, df=4)

    print(f"\nLikelihood Ratio Test:")
    print(f"  LRT Statistic: {lrt_result['LRT_statistic']:.4f}")
    print(f"  Critical Value (alpha=0.05, df={lrt_result['df']}): {lrt_result['critical_value']:.4f}")
    print(f"  p-value: {lrt_result['p_value']:.6f}")
    print(f"  Reject H0 (GBM adequate): {lrt_result['reject_null']}")
    print(f"  Interpretation: {lrt_result['interpretation']}")

    # ==========================================================================
    # 6. INFORMATION CRITERIA COMPARISON
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 6: INFORMATION CRITERIA COMPARISON")
    print("=" * 50)

    print(f"\n{'Criterion':<10} {'GBM':>12} {'Heston':>12} {'Preferred':>12}")
    print("-" * 48)

    aic_preferred = 'Heston' if heston_ic['AIC'] < gbm_ic['AIC'] else 'GBM'
    bic_preferred = 'Heston' if heston_ic['BIC'] < gbm_ic['BIC'] else 'GBM'
    aicc_preferred = 'Heston' if heston_ic['AICc'] < gbm_ic['AICc'] else 'GBM'

    print(f"{'AIC':<10} {gbm_ic['AIC']:>12.2f} {heston_ic['AIC']:>12.2f} {aic_preferred:>12}")
    print(f"{'BIC':<10} {gbm_ic['BIC']:>12.2f} {heston_ic['BIC']:>12.2f} {bic_preferred:>12}")
    print(f"{'AICc':<10} {gbm_ic['AICc']:>12.2f} {heston_ic['AICc']:>12.2f} {aicc_preferred:>12}")

    # ==========================================================================
    # 7. RESIDUAL DIAGNOSTICS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 7: RESIDUAL DIAGNOSTICS")
    print("=" * 50)

    gbm_diagnostics = residual_diagnostics(returns, None, gbm_params, Delta_t, 'GBM')
    heston_diagnostics = residual_diagnostics(returns, variance_path, heston_params, Delta_t, 'Heston')

    print("\n--- GBM Residual Diagnostics ---")
    print(f"  Ljung-Box Test:")
    print(f"    Statistic: {gbm_diagnostics['ljung_box']['statistic']:.4f}")
    print(f"    p-value: {gbm_diagnostics['ljung_box']['p_value']:.6f}")
    print(f"    {gbm_diagnostics['ljung_box']['interpretation']}")
    print(f"  Jarque-Bera Test:")
    print(f"    Statistic: {gbm_diagnostics['jarque_bera']['statistic']:.4f}")
    print(f"    p-value: {gbm_diagnostics['jarque_bera']['p_value']:.6f}")
    print(f"    Skewness: {gbm_diagnostics['jarque_bera']['skewness']:.4f}")
    print(f"    Excess Kurtosis: {gbm_diagnostics['jarque_bera']['excess_kurtosis']:.4f}")
    print(f"    {gbm_diagnostics['jarque_bera']['interpretation']}")

    print("\n--- Heston Residual Diagnostics ---")
    print(f"  Ljung-Box Test:")
    print(f"    Statistic: {heston_diagnostics['ljung_box']['statistic']:.4f}")
    print(f"    p-value: {heston_diagnostics['ljung_box']['p_value']:.6f}")
    print(f"    {heston_diagnostics['ljung_box']['interpretation']}")
    print(f"  Jarque-Bera Test:")
    print(f"    Statistic: {heston_diagnostics['jarque_bera']['statistic']:.4f}")
    print(f"    p-value: {heston_diagnostics['jarque_bera']['p_value']:.6f}")
    print(f"    Skewness: {heston_diagnostics['jarque_bera']['skewness']:.4f}")
    print(f"    Excess Kurtosis: {heston_diagnostics['jarque_bera']['excess_kurtosis']:.4f}")
    print(f"    {heston_diagnostics['jarque_bera']['interpretation']}")

    # ==========================================================================
    # 8. OUT-OF-SAMPLE VALIDATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 8: OUT-OF-SAMPLE VARIANCE FORECASTING")
    print("=" * 50)

    oos_validation = out_of_sample_validation(returns, Delta_t, train_ratio=0.8, forecast_horizon=22)

    print(f"\nVariance Forecast Improvement: {oos_validation['improvement_pct']:.2f}%")

    # ==========================================================================
    # 9. HYPOTHESIS EVALUATION
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 9: HYPOTHESIS EVALUATION")
    print("=" * 50)

    # Primary Hypothesis: Heston provides superior fit
    lrt_passed = lrt_result['reject_null']
    aic_passed = heston_ic['AIC'] < gbm_ic['AIC']
    bic_passed = heston_ic['BIC'] < gbm_ic['BIC']
    oos_passed = oos_validation['heston_wins']

    hypothesis_confirmed = lrt_passed and aic_passed and oos_passed

    print(f"\nPrimary Hypothesis: 'Heston provides statistically superior fit to GBM'")
    print(f"\nConfirmation Criteria:")
    print(f"  1. LRT rejects GBM (p < 0.05): {lrt_passed} (p = {lrt_result['p_value']:.6f})")
    print(f"  2. AIC(Heston) < AIC(GBM): {aic_passed} ({heston_ic['AIC']:.2f} vs {gbm_ic['AIC']:.2f})")
    print(f"  3. BIC(Heston) < BIC(GBM): {bic_passed} ({heston_ic['BIC']:.2f} vs {gbm_ic['BIC']:.2f})")
    print(f"  4. OOS Heston RMSE < GBM RMSE: {oos_passed} ({oos_validation['rmse_heston']:.6f} vs {oos_validation['rmse_gbm']:.6f})")

    print(f"\n{'='*50}")
    if hypothesis_confirmed:
        print("PRIMARY HYPOTHESIS: CONFIRMED")
        print("Heston stochastic volatility model provides a statistically")
        print("superior fit compared to Geometric Brownian Motion.")
    else:
        print("PRIMARY HYPOTHESIS: PARTIALLY SUPPORTED / INCONCLUSIVE")
        if not lrt_passed:
            print("  - LRT did not reject GBM at alpha=0.05")
        if not aic_passed:
            print("  - AIC does not favor Heston")
        if not bic_passed:
            print("  - BIC does not favor Heston")
        if not oos_passed:
            print("  - Out-of-sample forecasts favor GBM")
    print("=" * 50)

    # ==========================================================================
    # 10. COMPILE AND SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 50)
    print("STEP 10: SAVING RESULTS")
    print("=" * 50)

    # Compile all results
    results = {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'n_observations': N,
        'Delta_t': Delta_t,

        # Data
        'prices': prices.tolist(),
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'returns': returns.tolist(),

        # Sample statistics
        'sample_stats': {
            'mean_return': float(mean_ret),
            'std_return': float(std_ret),
            'skewness': float(skewness),
            'excess_kurtosis': float(kurtosis),
            'annualized_mean': float(mean_ret * 252),
            'annualized_volatility': float(std_ret * np.sqrt(252))
        },

        # GBM results
        'gbm_params': {k: float(v) for k, v in gbm_params.items()},
        'gbm_loglik': float(gbm_loglik),
        'gbm_ic': {k: float(v) for k, v in gbm_ic.items()},
        'gbm_diagnostics': {
            'ljung_box': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in gbm_diagnostics['ljung_box'].items()},
            'jarque_bera': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in gbm_diagnostics['jarque_bera'].items()}
        },

        # Heston results
        'heston_params': {k: float(v) for k, v in heston_params.items()},
        'heston_loglik': float(heston_loglik),
        'heston_ic': {k: float(v) for k, v in heston_ic.items()},
        'feller_ratio': float(feller_ratio),
        'feller_satisfied': feller_satisfied,
        'variance_path': variance_path.tolist(),
        'heston_diagnostics': {
            'ljung_box': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in heston_diagnostics['ljung_box'].items()},
            'jarque_bera': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in heston_diagnostics['jarque_bera'].items()}
        },

        # Model comparison
        'lrt': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in lrt_result.items()},

        # Out-of-sample validation
        'oos_validation': {
            'n_train': int(oos_validation['n_train']),
            'n_test': int(oos_validation['n_test']),
            'forecast_horizon': int(oos_validation['forecast_horizon']),
            'rmse_gbm': float(oos_validation['rmse_gbm']),
            'rmse_heston': float(oos_validation['rmse_heston']),
            'mae_gbm': float(oos_validation['mae_gbm']),
            'mae_heston': float(oos_validation['mae_heston']),
            'heston_wins': bool(oos_validation['heston_wins']),
            'improvement_pct': float(oos_validation['improvement_pct']),
            'realized_var': oos_validation['realized_var'].tolist(),
            'var_forecast_gbm': oos_validation['var_forecast_gbm'].tolist(),
            'var_forecast_heston': oos_validation['var_forecast_heston'].tolist()
        },

        # Hypothesis evaluation
        'hypothesis_evaluation': {
            'lrt_passed': bool(lrt_passed),
            'aic_passed': bool(aic_passed),
            'bic_passed': bool(bic_passed),
            'oos_passed': bool(oos_passed),
            'hypothesis_confirmed': bool(hypothesis_confirmed)
        },

        # Timestamp
        'experiment_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save JSON results
    json_path = os.path.join(output_dir, 'experiment_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Generate diagnostic plots
    plot_data = {
        'prices': prices,
        'dates': dates,
        'returns': returns,
        'variance_path': variance_path,
        'gbm_params': gbm_params,
        'heston_params': heston_params,
        'gbm_loglik': gbm_loglik,
        'heston_loglik': heston_loglik,
        'gbm_ic': gbm_ic,
        'heston_ic': heston_ic,
        'gbm_diagnostics': gbm_diagnostics,
        'heston_diagnostics': heston_diagnostics,
        'lrt': lrt_result,
        'oos_validation': oos_validation
    }

    plot_path = plot_diagnostics(plot_data, output_dir)
    resid_path = plot_residual_analysis(
        gbm_diagnostics['residuals'],
        heston_diagnostics['residuals'],
        output_dir
    )

    # Generate summary report
    generate_summary_report(results, output_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {json_path}")
    print(f"  - {plot_path}")
    print(f"  - {resid_path}")
    print(f"  - {os.path.join(output_dir, 'experiment_summary.md')}")

    return results


def generate_summary_report(results, output_dir):
    """
    Generate markdown summary report.
    """
    report = f"""# Stochastic Volatility Model Comparison: GBM vs Heston

## Experiment Summary

**Ticker:** {results['ticker']}
**Date Range:** {results['start_date']} to {results['end_date']}
**Observations:** {results['n_observations']}
**Experiment Date:** {results['experiment_timestamp']}

---

## Sample Statistics

| Statistic | Value |
|-----------|-------|
| Mean Return (daily) | {results['sample_stats']['mean_return']:.6f} |
| Std Dev (daily) | {results['sample_stats']['std_return']:.6f} |
| Annualized Mean | {results['sample_stats']['annualized_mean']*100:.2f}% |
| Annualized Volatility | {results['sample_stats']['annualized_volatility']*100:.2f}% |
| Skewness | {results['sample_stats']['skewness']:.4f} |
| Excess Kurtosis | {results['sample_stats']['excess_kurtosis']:.4f} |

---

## GBM Parameter Estimates

| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| mu (drift) | {results['gbm_params']['mu']:.6f} | {results['gbm_params']['mu']*100:.2f}% annual |
| sigma (volatility) | {results['gbm_params']['sigma']:.6f} | {results['gbm_params']['sigma']*100:.2f}% annual |

**Log-Likelihood:** {results['gbm_loglik']:.4f}

---

## Heston Parameter Estimates

| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| mu (drift) | {results['heston_params']['mu']:.6f} | Expected return |
| kappa (mean reversion) | {results['heston_params']['kappa']:.6f} | Speed of reversion |
| theta (long-run variance) | {results['heston_params']['theta']:.6f} | {np.sqrt(results['heston_params']['theta'])*100:.2f}% vol |
| xi (vol of vol) | {results['heston_params']['xi']:.6f} | Volatility volatility |
| rho (correlation) | {results['heston_params']['rho']:.6f} | Leverage effect |
| V_0 (initial variance) | {results['heston_params']['V_0']:.6f} | Starting variance |

**Log-Likelihood:** {results['heston_loglik']:.4f}

### Feller Condition
- **Ratio:** {results['feller_ratio']:.4f}
- **Satisfied:** {results['feller_satisfied']}

---

## Likelihood Ratio Test

| Metric | Value |
|--------|-------|
| LRT Statistic | {results['lrt']['LRT_statistic']:.4f} |
| Critical Value (df=4, alpha=0.05) | {results['lrt']['critical_value']:.4f} |
| p-value | {results['lrt']['p_value']:.6f} |
| Reject H0 (GBM adequate) | {results['lrt']['reject_null']} |

**Interpretation:** {results['lrt']['interpretation']}

---

## Information Criteria Comparison

| Criterion | GBM | Heston | Preferred |
|-----------|-----|--------|-----------|
| AIC | {results['gbm_ic']['AIC']:.2f} | {results['heston_ic']['AIC']:.2f} | {'Heston' if results['heston_ic']['AIC'] < results['gbm_ic']['AIC'] else 'GBM'} |
| BIC | {results['gbm_ic']['BIC']:.2f} | {results['heston_ic']['BIC']:.2f} | {'Heston' if results['heston_ic']['BIC'] < results['gbm_ic']['BIC'] else 'GBM'} |
| AICc | {results['gbm_ic']['AICc']:.2f} | {results['heston_ic']['AICc']:.2f} | {'Heston' if results['heston_ic']['AICc'] < results['gbm_ic']['AICc'] else 'GBM'} |

---

## Residual Diagnostics

### GBM

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Ljung-Box | {results['gbm_diagnostics']['ljung_box']['statistic']:.4f} | {results['gbm_diagnostics']['ljung_box']['p_value']:.6f} | {results['gbm_diagnostics']['ljung_box']['interpretation']} |
| Jarque-Bera | {results['gbm_diagnostics']['jarque_bera']['statistic']:.4f} | {results['gbm_diagnostics']['jarque_bera']['p_value']:.6f} | {results['gbm_diagnostics']['jarque_bera']['interpretation']} |

### Heston

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Ljung-Box | {results['heston_diagnostics']['ljung_box']['statistic']:.4f} | {results['heston_diagnostics']['ljung_box']['p_value']:.6f} | {results['heston_diagnostics']['ljung_box']['interpretation']} |
| Jarque-Bera | {results['heston_diagnostics']['jarque_bera']['statistic']:.4f} | {results['heston_diagnostics']['jarque_bera']['p_value']:.6f} | {results['heston_diagnostics']['jarque_bera']['interpretation']} |

---

## Out-of-Sample Variance Forecasting

| Metric | GBM | Heston |
|--------|-----|--------|
| RMSE | {results['oos_validation']['rmse_gbm']:.6f} | {results['oos_validation']['rmse_heston']:.6f} |
| MAE | {results['oos_validation']['mae_gbm']:.6f} | {results['oos_validation']['mae_heston']:.6f} |

**Heston Improvement:** {results['oos_validation']['improvement_pct']:.2f}%
**Winner:** {'Heston' if results['oos_validation']['heston_wins'] else 'GBM'}

---

## Hypothesis Evaluation

**Primary Hypothesis:** "The Heston stochastic volatility model provides a statistically superior fit to empirical stock return distributions compared to the geometric Brownian motion model."

### Confirmation Criteria

| Criterion | Met? |
|-----------|------|
| LRT rejects GBM (p < 0.05) | {'Yes' if results['hypothesis_evaluation']['lrt_passed'] else 'No'} |
| AIC(Heston) < AIC(GBM) | {'Yes' if results['hypothesis_evaluation']['aic_passed'] else 'No'} |
| BIC(Heston) < BIC(GBM) | {'Yes' if results['hypothesis_evaluation']['bic_passed'] else 'No'} |
| OOS Heston RMSE < GBM RMSE | {'Yes' if results['hypothesis_evaluation']['oos_passed'] else 'No'} |

### Conclusion

**Hypothesis Status:** {'CONFIRMED' if results['hypothesis_evaluation']['hypothesis_confirmed'] else 'PARTIALLY SUPPORTED / INCONCLUSIVE'}

---

## Files Generated

1. `experiment_results.json` - Complete numerical results
2. `diagnostic_plots.png` - Visual diagnostics
3. `residual_analysis.png` - Residual analysis plots
4. `experiment_summary.md` - This summary report

---

*Generated by Stochastic Volatility Experiment Pipeline*
*Theory Framework: files/theory/theory_quantitative_stock_price_modeling.md*
"""

    report_path = os.path.join(output_dir, 'experiment_summary.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Summary report saved to: {report_path}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    # Run the complete experiment
    results = run_complete_experiment(
        ticker='AAPL',
        start_date='2013-01-01',
        end_date=None,  # Use current date
        output_dir='/Users/jminding/Desktop/Code/Research Agent/research_agent/files/results/stochastic_volatility'
    )
