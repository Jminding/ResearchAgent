"""
Example: Statistical Analysis with Bootstrap CIs and Comparisons

This script demonstrates how to use the statistics module and data structures
to perform rigorous statistical analysis of experimental results.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'research_agent'))

from data_structures import (
    ResultsTable, ExperimentResult, AnalysisSummary
)
from statistics import (
    bootstrap_ci, bootstrap_sharpe_ci, compute_comparison_summary,
    diebold_mariano_test, calibration_metrics
)


def example_1_bootstrap_confidence_intervals():
    """Example 1: Computing bootstrap confidence intervals for returns."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Bootstrap Confidence Intervals")
    print("="*60)

    # Simulated monthly returns
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.03, 100)

    # Bootstrap CI for mean return
    result = bootstrap_ci(
        data=returns,
        statistic=np.mean,
        n_bootstrap=10000,
        confidence_level=0.95
    )

    print(f"\nMean Return:")
    print(f"  Point Estimate: {result.point_estimate:.4f}")
    print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"  Std Error: {result.std_error:.4f}")

    # Bootstrap CI for Sharpe ratio
    sharpe_result = bootstrap_sharpe_ci(
        returns=returns,
        risk_free_rate=0.0,
        n_bootstrap=10000
    )

    print(f"\nSharpe Ratio:")
    print(f"  Point Estimate: {sharpe_result.point_estimate:.4f}")
    print(f"  95% CI: [{sharpe_result.ci_lower:.4f}, {sharpe_result.ci_upper:.4f}]")


def example_2_comparison_analysis():
    """Example 2: Comparing two strategies with statistical tests."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Strategy Comparison with Bootstrap")
    print("="*60)

    # Simulated returns for two strategies
    np.random.seed(42)
    strategy_a_returns = np.random.normal(0.012, 0.03, 50)
    strategy_b_returns = np.random.normal(0.008, 0.03, 50)

    # Compute Sharpe ratios
    sharpe_a = np.mean(strategy_a_returns) / np.std(strategy_a_returns)
    sharpe_b = np.mean(strategy_b_returns) / np.std(strategy_b_returns)

    print(f"\nStrategy A Sharpe: {sharpe_a:.4f}")
    print(f"Strategy B Sharpe: {sharpe_b:.4f}")

    # Statistical comparison using bootstrap
    summary_dict = compute_comparison_summary(
        data1=strategy_a_returns,
        data2=strategy_b_returns,
        metric_name="Returns",
        comparison_name="strategy_a_vs_strategy_b",
        test_method="bootstrap",
        confidence_level=0.95,
        n_bootstrap=10000
    )

    analysis = AnalysisSummary(**summary_dict)

    print(f"\nStatistical Analysis:")
    print(f"  Difference: {analysis.estimate_diff:.4f}")
    print(f"  95% CI: [{analysis.ci_95[0]:.4f}, {analysis.ci_95[1]:.4f}]")
    print(f"  Conclusion: {analysis.conclusion}")
    print(f"  Cohen's d: {analysis.additional_metrics.get('cohens_d', 'N/A'):.4f}")


def example_3_diebold_mariano_test():
    """Example 3: Comparing forecast accuracy with Diebold-Mariano test."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Diebold-Mariano Test for Forecast Comparison")
    print("="*60)

    # Simulated forecast errors for two models
    np.random.seed(42)
    lstm_errors = np.random.normal(0, 0.05, 100)
    hybrid_errors = np.random.normal(0, 0.04, 100)  # Slightly more accurate

    # Compute MSE
    mse_lstm = np.mean(lstm_errors**2)
    mse_hybrid = np.mean(hybrid_errors**2)

    print(f"\nLSTM MSE: {mse_lstm:.6f}")
    print(f"Hybrid MSE: {mse_hybrid:.6f}")

    # Diebold-Mariano test
    dm_stat, p_val = diebold_mariano_test(
        errors1=lstm_errors,
        errors2=hybrid_errors,
        h=1,
        loss_fn="squared"
    )

    print(f"\nDiebold-Mariano Test:")
    print(f"  Test Statistic: {dm_stat:.4f}")
    print(f"  P-value: {p_val:.4f}")

    if p_val < 0.05:
        if dm_stat > 0:
            print("  Result: Hybrid significantly more accurate (p < 0.05)")
        else:
            print("  Result: LSTM significantly more accurate (p < 0.05)")
    else:
        print("  Result: No significant difference in forecast accuracy")


def example_4_calibration_analysis():
    """Example 4: Analyzing probabilistic prediction calibration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Calibration Metrics for Probabilistic Predictions")
    print("="*60)

    # Simulated predictions and outcomes
    np.random.seed(42)
    predictions = np.random.beta(2, 5, 1000)  # Predicted probabilities
    outcomes = (np.random.random(1000) < predictions).astype(int)  # Binary outcomes

    # Compute calibration metrics
    metrics = calibration_metrics(
        predictions=predictions,
        outcomes=outcomes,
        n_bins=10
    )

    print(f"\nCalibration Metrics:")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Calibration Error: {metrics['calibration_error']:.4f}")

    print(f"\n  Calibration by Bin:")
    for i, (pred, acc, count) in enumerate(zip(
        metrics['bin_predictions'],
        metrics['bin_accuracies'],
        metrics['bin_counts']
    )):
        if count > 0:
            print(f"    Bin {i+1}: Predicted={pred:.3f}, Observed={acc:.3f}, N={count}")


def example_5_results_table_workflow():
    """Example 5: Creating and analyzing a ResultsTable."""
    print("\n" + "="*60)
    print("EXAMPLE 5: ResultsTable Workflow")
    print("="*60)

    # Create a results table
    results_table = ResultsTable(project_name="Momentum Strategy Study")

    # Simulate experimental results for different configurations
    np.random.seed(42)
    frequencies = ["weekly", "monthly", "quarterly"]
    transaction_costs = [5, 10, 20]

    print("\nSimulating experiments...")
    for freq in frequencies:
        for tc in transaction_costs:
            # Simulate performance (quarterly + lower tc = better)
            base_sharpe = 0.3
            if freq == "quarterly":
                base_sharpe += 0.1
            elif freq == "monthly":
                base_sharpe += 0.05

            tc_penalty = tc * 0.01
            sharpe = base_sharpe - tc_penalty + np.random.normal(0, 0.05)

            result = ExperimentResult(
                config_name=f"momentum_{freq}_tc{tc}",
                parameters={
                    "frequency": freq,
                    "transaction_cost_bps": tc
                },
                metrics={
                    "sharpe": max(0.1, sharpe),  # Keep reasonable
                    "annual_return": sharpe * 0.15,
                    "volatility": 0.15
                }
            )
            results_table.add_result(result)

    print(f"Created {len(results_table.results)} experimental results")

    # Analyze: Compare quarterly vs monthly
    quarterly_sharpe = [
        r.metrics['sharpe'] for r in results_table.results
        if r.parameters['frequency'] == 'quarterly'
    ]
    monthly_sharpe = [
        r.metrics['sharpe'] for r in results_table.results
        if r.parameters['frequency'] == 'monthly'
    ]

    print(f"\nQuarterly Sharpe: {np.mean(quarterly_sharpe):.4f} ± {np.std(quarterly_sharpe):.4f}")
    print(f"Monthly Sharpe: {np.mean(monthly_sharpe):.4f} ± {np.std(monthly_sharpe):.4f}")

    # Statistical comparison
    summary_dict = compute_comparison_summary(
        data1=np.array(quarterly_sharpe),
        data2=np.array(monthly_sharpe),
        metric_name="Sharpe",
        comparison_name="quarterly_vs_monthly",
        test_method="t_test",
        confidence_level=0.95
    )

    analysis = AnalysisSummary(**summary_dict)
    print(f"\nComparison Result:")
    print(f"  {analysis.conclusion}")
    print(f"  Difference: {analysis.estimate_diff:.4f}")
    print(f"  95% CI: [{analysis.ci_95[0]:.4f}, {analysis.ci_95[1]:.4f}]")
    print(f"  P-value: {analysis.p_value:.4f}")

    # Save results (optional - commented out to avoid creating files)
    # results_table.to_json('results_table.json')
    # results_table.to_csv('results_table.csv')
    # analysis.to_json('comparison_quarterly_vs_monthly.json')


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS EXAMPLES")
    print("Demonstrating the enhanced research agent capabilities")
    print("="*60)

    example_1_bootstrap_confidence_intervals()
    example_2_comparison_analysis()
    example_3_diebold_mariano_test()
    example_4_calibration_analysis()
    example_5_results_table_workflow()

    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. Use bootstrap_ci() for confidence intervals on any statistic")
    print("  2. Use compute_comparison_summary() for standardized comparisons")
    print("  3. Use diebold_mariano_test() for forecast accuracy comparisons")
    print("  4. Use calibration_metrics() for probabilistic predictions")
    print("  5. ResultsTable + AnalysisSummary provide structured workflow")
    print("\nSee ENHANCED_FEATURES.md for full documentation.")
    print()


if __name__ == "__main__":
    main()
