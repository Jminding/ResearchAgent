"""
Example: Creating and Using Data Structures

This script demonstrates how to programmatically create and manipulate
the enhanced research agent data structures.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'research_agent'))

from data_structures import (
    EvidenceSheet, Reference, ResearchDomain,
    ExperimentPlan, ExperimentConfig, RobustnessChecklist,
    DataSelectionGuidelines, ExperimentStatus,
    ResultsTable, ExperimentResult,
    AnalysisSummary, FollowUpPlan, FollowUpHypothesis
)


def example_1_evidence_sheet():
    """Example 1: Creating an EvidenceSheet from literature review."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Creating an EvidenceSheet")
    print("="*60)

    # Create references
    references = [
        Reference(
            shortname="JT1993",
            year=1993,
            finding="Momentum strategy achieves Sharpe ratio ~0.4 in US equities",
            doi="10.1111/j.1540-6261.1993.tb05003.x"
        ),
        Reference(
            shortname="HS2016",
            year=2016,
            finding="LOB features improve in-sample but degrade 20-50% out-of-sample",
            url="https://example.com/paper"
        ),
        Reference(
            shortname="AH2018",
            year=2018,
            finding="Transaction costs of 5-10 bps typical for institutional traders"
        )
    ]

    # Create evidence sheet
    evidence = EvidenceSheet(
        metric_ranges={
            "large_cap_momentum_sharpe": [0.35, 0.42],
            "lob_oos_degradation": [0.2, 0.5],
            "transaction_cost_impact": [-0.05, -0.15]
        },
        typical_sample_sizes={
            "momentum_universe_size": "> 1000 stocks",
            "lob_training_period": "1-3 years",
            "backtest_period": "10+ years"
        },
        known_pitfalls=[
            "survivorship_bias",
            "small_sample_instability",
            "transaction_cost_underestimation",
            "overfitting_to_recent_regimes"
        ],
        key_references=references,
        domain=ResearchDomain.FINANCE,
        notes="Focus on US large-cap equities. International markets may differ."
    )

    print("\nEvidence Sheet Created:")
    print(f"  Domain: {evidence.domain.value}")
    print(f"  Metric Ranges: {len(evidence.metric_ranges)} metrics")
    print(f"  Known Pitfalls: {len(evidence.known_pitfalls)} pitfalls")
    print(f"  References: {len(evidence.key_references)} papers")

    print("\n  Key Metrics:")
    for metric, range_vals in evidence.metric_ranges.items():
        print(f"    {metric}: [{range_vals[0]}, {range_vals[1]}]")

    print("\n  Known Pitfalls:")
    for pitfall in evidence.known_pitfalls:
        print(f"    - {pitfall}")

    # Save to JSON (commented to avoid file creation)
    # evidence.to_json(Path('evidence_sheet.json'))

    return evidence


def example_2_experiment_plan(evidence: EvidenceSheet):
    """Example 2: Creating an ExperimentPlan with grids and ablations."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Creating an ExperimentPlan")
    print("="*60)

    # Create experiment config with parameter grid
    momentum_exp = ExperimentConfig(
        name="momentum_rebalancing",
        description="Test rebalancing frequency and transaction cost sensitivity",
        parameters={
            "frequency": ["weekly", "monthly", "quarterly"],
            "transaction_cost_bps": [5, 10, 20]
        },
        ablations=[],  # No ablations for this experiment
        status=ExperimentStatus.PENDING
    )

    # Create another experiment with ablations
    hybrid_exp = ExperimentConfig(
        name="hybrid_model_study",
        description="Compare full hybrid model against ablated versions",
        parameters={},
        ablations=[
            "full_hybrid",
            "hybrid_no_constraints",
            "hybrid_no_microstructure",
            "pure_deep_model",
            "pure_classical_model"
        ],
        status=ExperimentStatus.PENDING
    )

    # Create robustness checklist
    robustness = RobustnessChecklist(
        hyperparameter_perturbations=[
            "learning_rate_±25%",
            "batch_size_±50%",
            "lookback_window_±25%"
        ],
        additional_datasets=[
            "small_cap_universe",
            "international_markets"
        ],
        parameter_regimes=[],
        required_checks=5,
        notes="Test robustness across market conditions and hyperparameters"
    )

    # Create data selection guidelines
    data_guidelines = DataSelectionGuidelines(
        prefer_real_data=True,
        real_data_sources=["CRSP", "Compustat", "TAQ"],
        synthetic_data_justification="Use synthetic LOB features only when real LOB data unavailable",
        synthetic_data_generation_method="Hawkes process calibrated to real data moments",
        known_synthetic_biases=["May not capture full microstructure complexity"],
        data_labeling={
            "prices": "real",
            "volumes": "real",
            "fundamentals": "real",
            "lob_features": "synthetic"
        }
    )

    # Create complete experiment plan
    plan = ExperimentPlan(
        project_name="Enhanced Momentum Strategy",
        experiments=[momentum_exp, hybrid_exp],
        robustness_checklist=robustness,
        data_guidelines=data_guidelines,
        hypotheses=[
            "Quarterly rebalancing improves Sharpe ratio by at least 0.1 vs monthly",
            "Hybrid model outperforms pure deep learning by at least 10% RMSE reduction",
            "Constraints reduce overfitting, improving OOS performance by 15-25%"
        ],
        expected_outcomes={
            "sharpe_range": evidence.metric_ranges["large_cap_momentum_sharpe"],
            "transaction_cost_impact": evidence.metric_ranges["transaction_cost_impact"],
            "oos_degradation_bound": evidence.metric_ranges["lob_oos_degradation"][1]
        },
        mode="discovery"
    )

    print("\nExperiment Plan Created:")
    print(f"  Project: {plan.project_name}")
    print(f"  Mode: {plan.mode}")
    print(f"  Experiments: {len(plan.experiments)}")

    for exp in plan.experiments:
        print(f"\n  Experiment: {exp.name}")
        print(f"    Description: {exp.description}")
        if exp.parameters:
            grid_size = exp.get_grid_size()
            print(f"    Grid Size: {grid_size} configurations")
            print(f"    Parameters:")
            for param, values in exp.parameters.items():
                print(f"      - {param}: {values}")
        if exp.ablations:
            print(f"    Ablations: {len(exp.ablations)} variants")
            for ablation in exp.ablations:
                print(f"      - {ablation}")

    print(f"\n  Hypotheses:")
    for i, hyp in enumerate(plan.hypotheses, 1):
        print(f"    {i}. {hyp}")

    print(f"\n  Expected Outcomes (from evidence):")
    for metric, value in plan.expected_outcomes.items():
        print(f"    {metric}: {value}")

    # Save to JSON (commented to avoid file creation)
    # plan.to_json(Path('experiment_plan.json'))

    return plan


def example_3_results_table():
    """Example 3: Creating a ResultsTable with experimental results."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Creating a ResultsTable")
    print("="*60)

    # Create results table
    results_table = ResultsTable(project_name="Momentum Strategy Study")

    # Add some example results
    results = [
        ExperimentResult(
            config_name="momentum_quarterly_tc5",
            parameters={"frequency": "quarterly", "transaction_cost_bps": 5},
            metrics={"sharpe": 0.45, "max_drawdown": -0.12, "annual_return": 0.067}
        ),
        ExperimentResult(
            config_name="momentum_monthly_tc5",
            parameters={"frequency": "monthly", "transaction_cost_bps": 5},
            metrics={"sharpe": 0.32, "max_drawdown": -0.15, "annual_return": 0.048}
        ),
        ExperimentResult(
            config_name="momentum_quarterly_tc10",
            parameters={"frequency": "quarterly", "transaction_cost_bps": 10},
            metrics={"sharpe": 0.38, "max_drawdown": -0.13, "annual_return": 0.057}
        ),
        ExperimentResult(
            config_name="hybrid_model",
            parameters={},
            metrics={"rmse": 0.045, "mae": 0.032, "r2": 0.75},
            ablation="full_hybrid"
        ),
        ExperimentResult(
            config_name="hybrid_model",
            parameters={},
            metrics={"rmse": 0.052, "mae": 0.038, "r2": 0.68},
            ablation="hybrid_no_constraints"
        )
    ]

    for result in results:
        results_table.add_result(result)

    print(f"\nResults Table Created:")
    print(f"  Project: {results_table.project_name}")
    print(f"  Total Results: {len(results_table.results)}")

    print("\n  Sample Results:")
    for result in results_table.results[:3]:
        print(f"\n    Config: {result.config_name}")
        print(f"    Parameters: {result.parameters}")
        print(f"    Metrics: {result.metrics}")
        if result.ablation:
            print(f"    Ablation: {result.ablation}")

    # Save to JSON and CSV (commented to avoid file creation)
    # results_table.to_json(Path('results_table.json'))
    # results_table.to_csv(Path('results_table.csv'))

    return results_table


def example_4_analysis_summary():
    """Example 4: Creating an AnalysisSummary for a comparison."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Creating an AnalysisSummary")
    print("="*60)

    # Create analysis summary
    analysis = AnalysisSummary(
        comparison="quarterly_vs_monthly",
        metric="Sharpe",
        estimate_diff=0.13,
        ci_95=(0.04, 0.22),
        p_value=0.01,
        test_statistic=2.45,
        test_method="bootstrap",
        conclusion="Quarterly significantly better at 95% level (p=0.01).",
        sample_size=100,
        additional_metrics={"cohens_d": 0.65}
    )

    print("\nAnalysis Summary Created:")
    print(f"  Comparison: {analysis.comparison}")
    print(f"  Metric: {analysis.metric}")
    print(f"  Difference: {analysis.estimate_diff:.3f}")
    print(f"  95% CI: [{analysis.ci_95[0]:.3f}, {analysis.ci_95[1]:.3f}]")
    print(f"  P-value: {analysis.p_value:.4f}")
    print(f"  Test Method: {analysis.test_method}")
    print(f"  Conclusion: {analysis.conclusion}")
    print(f"  Effect Size (Cohen's d): {analysis.additional_metrics['cohens_d']:.3f}")

    # Save to JSON (commented to avoid file creation)
    # analysis.to_json(Path('comparison_quarterly_vs_monthly.json'))

    return analysis


def example_5_followup_plan():
    """Example 5: Creating a FollowUpPlan when hypothesis fails."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Creating a FollowUpPlan")
    print("="*60)

    # Create follow-up hypotheses
    hypotheses = [
        FollowUpHypothesis(
            hypothesis="Constraints too strong, preventing model from fitting data",
            diagnostic_experiment="Relax constraint strength by 50% and re-run",
            expected_outcome="If correct, relaxed constraints should improve fit to training data",
            priority=1
        ),
        FollowUpHypothesis(
            hypothesis="Feature conflict between classical and neural inputs",
            diagnostic_experiment="Remove Heston inputs, keep only neural features",
            expected_outcome="If correct, removing Heston features should improve performance",
            priority=2
        ),
        FollowUpHypothesis(
            hypothesis="Loss function misaligned with evaluation metric",
            diagnostic_experiment="Change loss to directly optimize Sharpe ratio",
            expected_outcome="If correct, aligned loss should improve Sharpe metric",
            priority=2
        )
    ]

    # Create follow-up plan
    followup_plan = FollowUpPlan(
        trigger="Hybrid model worse than pure LSTM (expected +10% improvement, observed -5%)",
        hypotheses=hypotheses,
        selected_followup="Relax constraint strength by 50% and re-run",
        mode="discovery"
    )

    print("\nFollow-Up Plan Created:")
    print(f"  Trigger: {followup_plan.trigger}")
    print(f"  Mode: {followup_plan.mode}")
    print(f"  Number of Hypotheses: {len(followup_plan.hypotheses)}")

    print("\n  Diagnostic Hypotheses:")
    for i, hyp in enumerate(followup_plan.hypotheses, 1):
        print(f"\n    {i}. {hyp.hypothesis}")
        print(f"       Priority: {hyp.priority}")
        print(f"       Experiment: {hyp.diagnostic_experiment}")
        print(f"       Expected: {hyp.expected_outcome}")

    print(f"\n  Selected Follow-Up: {followup_plan.selected_followup}")

    # Save to JSON (commented to avoid file creation)
    # followup_plan.to_json(Path('followup_plan.json'))

    return followup_plan


def example_6_loading_from_json():
    """Example 6: Loading data structures from JSON (demonstration)."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Loading from JSON Files")
    print("="*60)

    print("\nTo load data structures from JSON files:")
    print("\n  # Load evidence sheet")
    print("  evidence = EvidenceSheet.from_json('files/research_notes/evidence_sheet.json')")
    print("\n  # Load experiment plan")
    print("  plan = ExperimentPlan.from_json('files/theory/experiment_plan.json')")
    print("\n  # Load results table")
    print("  results = ResultsTable.from_json('files/results/results_table.json')")
    print("\n  # Load analysis summary")
    print("  analysis = AnalysisSummary.from_json('files/results/comparison_X_vs_Y.json')")
    print("\n  # Load follow-up plan")
    print("  followup = FollowUpPlan.from_json('files/results/followup_plan.json')")

    print("\nAll data structures support:")
    print("  - .to_json(path) - Save to JSON file")
    print("  - .from_json(path) - Load from JSON file")
    print("  - .to_dict() - Convert to dictionary")
    print("  - .from_dict(dict) - Create from dictionary")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("DATA STRUCTURES EXAMPLES")
    print("Demonstrating the enhanced research agent data structures")
    print("="*60)

    # Run examples in sequence
    evidence = example_1_evidence_sheet()
    plan = example_2_experiment_plan(evidence)
    results = example_3_results_table()
    analysis = example_4_analysis_summary()
    followup = example_5_followup_plan()
    example_6_loading_from_json()

    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. EvidenceSheet captures quantitative evidence from literature")
    print("  2. ExperimentPlan defines parameter grids, ablations, robustness")
    print("  3. ResultsTable stores all experimental results in structured format")
    print("  4. AnalysisSummary provides statistical backing for claims")
    print("  5. FollowUpPlan proposes diagnostics when hypotheses fail")
    print("\nAll structures serialize to/from JSON for agent communication.")
    print("See ENHANCED_FEATURES.md for full documentation.")
    print()


if __name__ == "__main__":
    main()
