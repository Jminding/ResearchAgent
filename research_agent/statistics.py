"""
Statistical analysis tools for rigorous hypothesis testing.

Provides bootstrap confidence intervals, Diebold-Mariano tests,
and other statistical methods for comparing experimental results.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        BootstrapResult with point estimate, CI, and standard error

    Example:
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
        >>> sharpe = lambda x: np.mean(x) / np.std(x)
        >>> result = bootstrap_ci(returns, statistic=sharpe)
        >>> print(f"Sharpe: {result.point_estimate:.3f}, CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(data)
    bootstrap_statistics = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_statistics[i] = statistic(sample)

    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_statistics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2))

    point_estimate = statistic(data)
    std_error = np.std(bootstrap_statistics)

    return BootstrapResult(
        point_estimate=float(point_estimate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        std_error=float(std_error),
        n_bootstrap=n_bootstrap
    )


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (same frequency as returns)

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns)


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> BootstrapResult:
    """
    Bootstrap confidence interval for Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed

    Returns:
        BootstrapResult for Sharpe ratio
    """
    def sharpe_fn(x):
        return sharpe_ratio(x, risk_free_rate)

    return bootstrap_ci(returns, statistic=sharpe_fn, n_bootstrap=n_bootstrap,
                       confidence_level=confidence_level, random_seed=random_seed)


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    loss_fn: str = "squared"
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests the null hypothesis that two forecasts have equal accuracy.

    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        h: Forecast horizon (for HAC correction)
        loss_fn: Loss function ("squared" or "absolute")

    Returns:
        Tuple of (test_statistic, p_value)

    Reference:
        Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy.
        Journal of Business & Economic Statistics, 13(3), 253-263.

    Example:
        >>> errors_lstm = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
        >>> errors_hybrid = np.array([0.08, -0.15, 0.12, -0.08, 0.03])
        >>> dm_stat, p_val = diebold_mariano_test(errors_lstm, errors_hybrid)
        >>> if p_val < 0.05:
        >>>     print("Hybrid significantly more accurate")
    """
    if len(errors1) != len(errors2):
        raise ValueError("Error arrays must have same length")

    # Compute loss differential
    if loss_fn == "squared":
        loss_diff = errors1**2 - errors2**2
    elif loss_fn == "absolute":
        loss_diff = np.abs(errors1) - np.abs(errors2)
    else:
        raise ValueError("loss_fn must be 'squared' or 'absolute'")

    # Mean loss differential
    d_bar = np.mean(loss_diff)

    # Variance with HAC correction (Newey-West)
    n = len(loss_diff)
    gamma_0 = np.var(loss_diff, ddof=1)

    # Autocorrelations up to lag h-1
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.mean((loss_diff[k:] - d_bar) * (loss_diff[:-k] - d_bar))
        gamma_sum += 2 * gamma_k

    variance = (gamma_0 + gamma_sum) / n

    # DM statistic
    if variance <= 0:
        # Degenerate case
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(variance)

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return float(dm_stat), float(p_value)


def t_test_independent(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Independent samples t-test.

    Args:
        sample1: First sample
        sample2: Second sample
        alternative: "two-sided", "less", or "greater"

    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_val = stats.ttest_ind(sample1, sample2, alternative=alternative)
    return float(t_stat), float(p_val)


def paired_t_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Paired samples t-test.

    Args:
        sample1: First sample (paired observations)
        sample2: Second sample (paired observations)
        alternative: "two-sided", "less", or "greater"

    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_val = stats.ttest_rel(sample1, sample2, alternative=alternative)
    return float(t_stat), float(p_val)


def calibration_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics for probabilistic predictions.

    Args:
        predictions: Predicted probabilities [0, 1]
        outcomes: Binary outcomes {0, 1}
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with:
        - brier_score: Brier score
        - calibration_error: Mean absolute calibration error
        - bins: Bin edges
        - bin_accuracies: Observed frequency in each bin
        - bin_predictions: Mean predicted probability in each bin
        - bin_counts: Number of samples in each bin
    """
    # Brier score
    brier_score = np.mean((predictions - outcomes) ** 2)

    # Binned calibration
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = []
    bin_predictions = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(outcomes[mask]))
            bin_predictions.append(np.mean(predictions[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(np.nan)
            bin_predictions.append(np.nan)
            bin_counts.append(0)

    # Calibration error (mean absolute error between predicted and observed)
    valid_bins = [i for i in range(n_bins) if bin_counts[i] > 0]
    if len(valid_bins) > 0:
        calibration_error = np.mean([
            np.abs(bin_predictions[i] - bin_accuracies[i])
            for i in valid_bins
        ])
    else:
        calibration_error = np.nan

    return {
        'brier_score': float(brier_score),
        'calibration_error': float(calibration_error),
        'bins': bins.tolist(),
        'bin_accuracies': bin_accuracies,
        'bin_predictions': bin_predictions,
        'bin_counts': bin_counts
    }


def effect_size_cohens_d(
    sample1: np.ndarray,
    sample2: np.ndarray
) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        sample1: First sample
        sample2: Second sample

    Returns:
        Cohen's d (standardized mean difference)
    """
    mean_diff = np.mean(sample1) - np.mean(sample2)
    pooled_std = np.sqrt((np.var(sample1, ddof=1) + np.var(sample2, ddof=1)) / 2)

    if pooled_std == 0:
        return 0.0

    return mean_diff / pooled_std


def bootstrap_difference(
    data1: np.ndarray,
    data2: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> BootstrapResult:
    """
    Bootstrap confidence interval for difference between two statistics.

    Args:
        data1: First sample
        data2: Second sample
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed

    Returns:
        BootstrapResult for the difference stat(data1) - stat(data2)

    Example:
        >>> returns_a = np.array([0.01, 0.02, -0.01])
        >>> returns_b = np.array([0.005, 0.015, -0.005])
        >>> result = bootstrap_difference(returns_a, returns_b, statistic=np.mean)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n1 = len(data1)
    n2 = len(data2)
    bootstrap_diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=n1, replace=True)
        sample2 = np.random.choice(data2, size=n2, replace=True)
        bootstrap_diffs[i] = statistic(sample1) - statistic(sample2)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    point_estimate = statistic(data1) - statistic(data2)
    std_error = np.std(bootstrap_diffs)

    return BootstrapResult(
        point_estimate=float(point_estimate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        std_error=float(std_error),
        n_bootstrap=n_bootstrap
    )


def reliability_diagram_data(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10
) -> Dict[str, List]:
    """
    Prepare data for reliability diagram (calibration plot).

    Args:
        predictions: Predicted probabilities
        outcomes: Binary outcomes
        n_bins: Number of bins

    Returns:
        Dictionary with bin_predictions and bin_accuracies for plotting
    """
    metrics = calibration_metrics(predictions, outcomes, n_bins)

    # Filter out empty bins
    valid_indices = [i for i, count in enumerate(metrics['bin_counts']) if count > 0]

    return {
        'bin_predictions': [metrics['bin_predictions'][i] for i in valid_indices],
        'bin_accuracies': [metrics['bin_accuracies'][i] for i in valid_indices],
        'bin_counts': [metrics['bin_counts'][i] for i in valid_indices]
    }


def compute_comparison_summary(
    data1: np.ndarray,
    data2: np.ndarray,
    metric_name: str,
    comparison_name: str,
    test_method: str = "bootstrap",
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive comparison summary between two samples.

    Convenience function that combines multiple statistical tests.

    Args:
        data1: First sample (e.g., metric values from model A)
        data2: Second sample (e.g., metric values from model B)
        metric_name: Name of the metric (e.g., "Sharpe", "RMSE")
        comparison_name: Name of comparison (e.g., "quarterly_vs_weekly")
        test_method: "bootstrap", "t_test", or "dm_test" (for forecast errors)
        confidence_level: Confidence level for CI
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed

    Returns:
        Dictionary suitable for creating AnalysisSummary dataclass
    """
    result = {}

    if test_method == "bootstrap":
        boot_result = bootstrap_difference(
            data1, data2, statistic=np.mean,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed
        )
        result = {
            'comparison': comparison_name,
            'metric': metric_name,
            'estimate_diff': boot_result.point_estimate,
            'ci_95': (boot_result.ci_lower, boot_result.ci_upper),
            'test_method': 'bootstrap',
            'sample_size': len(data1)
        }

        # Check if CI excludes zero
        if boot_result.ci_lower > 0:
            result['conclusion'] = f"{comparison_name.split('_vs_')[0]} significantly better at {int(confidence_level*100)}% level."
            result['p_value'] = (1 - confidence_level) / 2  # Approximate
        elif boot_result.ci_upper < 0:
            result['conclusion'] = f"{comparison_name.split('_vs_')[-1]} significantly better at {int(confidence_level*100)}% level."
            result['p_value'] = (1 - confidence_level) / 2
        else:
            result['conclusion'] = "No significant difference detected."
            result['p_value'] = None

    elif test_method == "t_test":
        t_stat, p_val = t_test_independent(data1, data2)
        mean_diff = np.mean(data1) - np.mean(data2)

        # Compute CI from t-distribution
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        n1, n2 = len(data1), len(data2)
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
        t_crit = stats.t.ppf(1 - (1 - confidence_level)/2, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        result = {
            'comparison': comparison_name,
            'metric': metric_name,
            'estimate_diff': float(mean_diff),
            'ci_95': (float(ci_lower), float(ci_upper)),
            'p_value': p_val,
            'test_statistic': t_stat,
            'test_method': 't_test',
            'sample_size': len(data1)
        }

        alpha = 1 - confidence_level
        if p_val < alpha:
            if mean_diff > 0:
                result['conclusion'] = f"{comparison_name.split('_vs_')[0]} significantly better at {int(confidence_level*100)}% level (p={p_val:.4f})."
            else:
                result['conclusion'] = f"{comparison_name.split('_vs_')[-1]} significantly better at {int(confidence_level*100)}% level (p={p_val:.4f})."
        else:
            result['conclusion'] = f"No significant difference detected (p={p_val:.4f})."

    elif test_method == "dm_test":
        # Diebold-Mariano for forecast errors
        dm_stat, p_val = diebold_mariano_test(data1, data2)
        mean_diff = np.mean(data1**2) - np.mean(data2**2)  # MSE difference

        result = {
            'comparison': comparison_name,
            'metric': metric_name,
            'estimate_diff': float(mean_diff),
            'ci_95': (np.nan, np.nan),  # DM doesn't provide CI directly
            'p_value': p_val,
            'test_statistic': dm_stat,
            'test_method': 'diebold_mariano',
            'sample_size': len(data1)
        }

        alpha = 1 - confidence_level
        if p_val < alpha:
            if dm_stat > 0:
                result['conclusion'] = f"{comparison_name.split('_vs_')[-1]} significantly more accurate at {int(confidence_level*100)}% level (DM p={p_val:.4f})."
            else:
                result['conclusion'] = f"{comparison_name.split('_vs_')[0]} significantly more accurate at {int(confidence_level*100)}% level (DM p={p_val:.4f})."
        else:
            result['conclusion'] = f"No significant difference in forecast accuracy detected (p={p_val:.4f})."

    else:
        raise ValueError(f"Unknown test_method: {test_method}")

    # Add effect size
    cohens_d = effect_size_cohens_d(data1, data2)
    result['additional_metrics'] = {'cohens_d': float(cohens_d)}

    return result
