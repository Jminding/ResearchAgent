#!/usr/bin/env python3
"""
Comprehensive revision analysis for QEC peer review response.
Generates all statistical comparisons, hypothesis tests, and follow-up plans.
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict
import sys

def bootstrap_ci(data, confidence=0.95, n_bootstrap=10000):
    """Bootstrap confidence interval for mean."""
    data = np.array(data)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return lower, upper

def compute_comparison_summary(data_a, data_b, metric_name, comparison_name):
    """Compute full statistical comparison between two datasets."""
    data_a = np.array(data_a)
    data_b = np.array(data_b)

    # Basic statistics
    mean_a, mean_b = data_a.mean(), data_b.mean()
    std_a, std_b = data_a.std(ddof=1) if len(data_a) > 1 else 0, data_b.std(ddof=1) if len(data_b) > 1 else 0

    # Difference and CI
    diff = mean_a - mean_b

    # T-test
    if len(data_a) > 1 and len(data_b) > 1:
        t_stat, p_value = stats.ttest_ind(data_a, data_b)

        # Effect size
        pooled_std = np.sqrt((data_a.var() + data_b.var()) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0

        # CI for difference using t-distribution
        se_diff = np.sqrt(std_a**2/len(data_a) + std_b**2/len(data_b))
        df = len(data_a) + len(data_b) - 2
        t_crit = stats.t.ppf(0.975, df)
        ci_95 = [diff - t_crit * se_diff, diff + t_crit * se_diff]
    else:
        t_stat, p_value = np.nan, np.nan
        cohens_d = np.nan
        ci_95 = [diff, diff]

    # Determine conclusion
    if p_value < 0.05:
        if diff > 0:
            conclusion = f"{comparison_name}: {metric_name} significantly higher in A vs B (p={p_value:.4f})"
        else:
            conclusion = f"{comparison_name}: {metric_name} significantly lower in A vs B (p={p_value:.4f})"
    else:
        conclusion = f"{comparison_name}: No significant difference in {metric_name} (p={p_value:.4f})"

    return {
        "comparison": comparison_name,
        "metric": metric_name,
        "group_a_mean": float(mean_a),
        "group_a_std": float(std_a),
        "group_a_n": int(len(data_a)),
        "group_b_mean": float(mean_b),
        "group_b_std": float(std_b),
        "group_b_n": int(len(data_b)),
        "estimate_diff": float(diff),
        "ci_95": [float(ci_95[0]), float(ci_95[1])],
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
        "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
        "test_method": "two_sample_t_test",
        "conclusion": conclusion
    }

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("QUANTUM ERROR CORRECTION - REVISION ANALYSIS")
print("="*80)

with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/extended_results_table.json', 'r') as f:
    data = json.load(f)

results = data['results']
print(f"\nLoaded {len(results)} experimental results")

# Load experiment plan
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/theory/experiment_plan.json', 'r') as f:
    exp_plan = json.load(f)

original_d15_ler = exp_plan['original_results_summary']['d15_rl_logical_error_rate']
original_d15_mwpm = exp_plan['original_results_summary']['d15_mwpm_logical_error_rate']

print(f"\nOriginal results (from experiment plan):")
print(f"  d=15 RL LER: {original_d15_ler:.4f}")
print(f"  d=15 MWPM LER: {original_d15_mwpm:.4f}")
print(f"  Training episodes: {exp_plan['original_results_summary']['original_training_episodes']}")
print(f"  Seeds: {exp_plan['original_results_summary']['original_seeds']}")

# ============================================================================
# ORGANIZE DATA
# ============================================================================
# Group experiments by type
d15_extended = []
baseline_comparison = []
reward_shaping = []
gnn_architecture = []
zero_shot = []
mwpm_validation = []

for r in results:
    config = r['config_name']
    params = r['parameters']

    if params.get('code_distance') == 15:
        if 'd15' in config and 'ep' in config:
            d15_extended.append(r)
        elif config.startswith('comparison'):
            baseline_comparison.append(r)
        elif config.startswith('reward'):
            reward_shaping.append(r)
        elif config.startswith('gnn'):
            gnn_architecture.append(r)

    if config.startswith('d7') and 'zeroshot' in config:
        zero_shot.append(r)

    if config.startswith('mwpm'):
        mwpm_validation.append(r)

print(f"\nExperiment counts:")
print(f"  d=15 extended training: {len(d15_extended)}")
print(f"  Baseline comparison (d=15): {len([r for r in baseline_comparison if r['parameters']['code_distance']==15])}")
print(f"  Reward shaping (d=15): {len(reward_shaping)}")
print(f"  GNN architecture (d=15): {len(gnn_architecture)}")
print(f"  Zero-shot generalization: {len(zero_shot)}")
print(f"  MWPM validation: {len(mwpm_validation)}")

# ============================================================================
# 1. EXTENDED TRAINING ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. EXTENDED TRAINING AT d=15 - TESTING UNDERTRAINING HYPOTHESIS")
print("="*80)

# Group by training episodes
d15_by_episodes = defaultdict(list)
for r in d15_extended:
    episodes = r['parameters']['training_episodes']
    ler = r['metrics']['logical_error_rate']
    d15_by_episodes[episodes].append(ler)

print(f"\nLearning curve at d=15:")
print(f"{'Episodes':<12} {'Mean LER':<12} {'Std':<12} {'n':<6} {'95% CI':<30}")
print("-" * 80)

episodes_list = sorted(d15_by_episodes.keys())
for ep in episodes_list:
    lers = np.array(d15_by_episodes[ep])
    mean_ler = lers.mean()
    std_ler = lers.std(ddof=1) if len(lers) > 1 else 0
    n = len(lers)

    if n > 1:
        ci_95 = stats.t.interval(0.95, n-1, loc=mean_ler, scale=std_ler/np.sqrt(n))
    else:
        ci_95 = (mean_ler, mean_ler)

    print(f"{ep:<12} {mean_ler:<12.4f} {std_ler:<12.4f} {n:<6} [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

# Statistical test: 200 vs 5000 episodes
comparison_200_5000 = None
if 200 in d15_by_episodes and 5000 in d15_by_episodes:
    comparison_200_5000 = compute_comparison_summary(
        d15_by_episodes[200],
        d15_by_episodes[5000],
        "logical_error_rate",
        "200ep_vs_5000ep"
    )

    print(f"\nStatistical comparison: 200 vs 5000 episodes")
    print(f"  200 ep: {comparison_200_5000['group_a_mean']:.4f} ± {comparison_200_5000['group_a_std']:.4f} (n={comparison_200_5000['group_a_n']})")
    print(f"  5000 ep: {comparison_200_5000['group_b_mean']:.4f} ± {comparison_200_5000['group_b_std']:.4f} (n={comparison_200_5000['group_b_n']})")
    print(f"  Difference: {comparison_200_5000['estimate_diff']:.4f}")
    print(f"  95% CI: [{comparison_200_5000['ci_95'][0]:.4f}, {comparison_200_5000['ci_95'][1]:.4f}]")
    print(f"  p-value: {comparison_200_5000['p_value']:.4f}")
    print(f"  Cohen's d: {comparison_200_5000['cohens_d']:.3f}")
    print(f"  Conclusion: {comparison_200_5000['conclusion']}")

# Linear trend analysis
if len(episodes_list) >= 3:
    x = np.log10(episodes_list)
    y = np.array([np.mean(d15_by_episodes[ep]) for ep in episodes_list])
    slope, intercept, r_value, p_value_trend, stderr = stats.linregress(x, y)

    print(f"\nLearning curve trend (log10(episodes) vs LER):")
    print(f"  Slope: {slope:.6f} ({'improvement' if slope < 0 else 'degradation'})")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value_trend:.4f}")
    print(f"  Interpretation: {'SIGNIFICANT' if p_value_trend < 0.05 else 'NOT SIGNIFICANT'} trend")

# ============================================================================
# 2. UNDERTRAINING HYPOTHESIS VERDICT
# ============================================================================
print("\n" + "="*80)
print("2. UNDERTRAINING HYPOTHESIS VERDICT")
print("="*80)

if comparison_200_5000:
    improvement_pct = (comparison_200_5000['estimate_diff'] / comparison_200_5000['group_a_mean']) * 100
    p_val = comparison_200_5000['p_value']

    print(f"\nOriginal Hypothesis:")
    print(f"  'Insufficient training (200 episodes) limits RL performance at d=15'")

    print(f"\nEvidence from extended experiments:")
    print(f"  - Training increased from 200 to 5000 episodes (25x)")
    print(f"  - LER change: {comparison_200_5000['estimate_diff']:.4f} ({improvement_pct:.1f}%)")
    print(f"  - Statistical significance: p = {p_val:.4f}")
    print(f"  - Effect size: Cohen's d = {comparison_200_5000['cohens_d']:.3f}")

    # Determine verdict
    significant = p_val < 0.05
    meaningful = abs(improvement_pct) > 5  # 5% threshold for meaningful change

    if significant and meaningful and improvement_pct > 0:
        verdict = "PARTIALLY CONFIRMED"
        explanation = f"Extended training shows statistically significant improvement ({improvement_pct:.1f}%, p={p_val:.4f}), but the magnitude is modest. Undertraining contributes to poor performance but is not the primary limiting factor."
    elif not significant:
        verdict = "REJECTED"
        explanation = f"Extended training (25x more episodes) produces statistically insignificant improvement (p={p_val:.4f}). The performance gap is NOT primarily due to undertraining."
    else:
        verdict = "INCONCLUSIVE"
        explanation = f"Results show {improvement_pct:.1f}% change but statistical evidence is mixed."

    print(f"\n{'='*80}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*80}")
    print(f"{explanation}")

    undertraining_analysis = {
        "original_hypothesis": "Insufficient training (200 episodes) limits RL performance at d=15",
        "verdict": verdict,
        "evidence": {
            "training_increase": "25x (200 -> 5000 episodes)",
            "ler_change": f"{comparison_200_5000['estimate_diff']:.4f}",
            "percent_improvement": f"{improvement_pct:.1f}%",
            "p_value": f"{p_val:.4f}",
            "cohens_d": f"{comparison_200_5000['cohens_d']:.3f}",
            "statistically_significant": significant,
            "practically_meaningful": meaningful
        },
        "explanation": explanation
    }

# ============================================================================
# 3. BASELINE COMPARISON (RL vs MWPM)
# ============================================================================
print("\n" + "="*80)
print("3. BASELINE COMPARISON: RL vs MWPM (2000 episodes)")
print("="*80)

baseline_d15 = [r for r in baseline_comparison if r['parameters']['code_distance'] == 15]
if baseline_d15:
    rl_lers_d15 = [r['metrics']['logical_error_rate_rl'] for r in baseline_d15]
    mwpm_lers_d15 = [r['metrics']['logical_error_rate_mwpm'] for r in baseline_d15]
    ratios_d15 = [r['metrics']['rl_vs_mwpm_ratio'] for r in baseline_d15]

    print(f"\nAt d=15 (with 2000 episodes training):")
    print(f"  RL LER:   {np.mean(rl_lers_d15):.4f} ± {np.std(rl_lers_d15):.4f} (n={len(rl_lers_d15)})")
    print(f"  MWPM LER: {np.mean(mwpm_lers_d15):.4f} ± {np.std(mwpm_lers_d15):.4f} (n={len(mwpm_lers_d15)})")
    print(f"  Ratio:    {np.mean(ratios_d15):.2f}x ± {np.std(ratios_d15):.2f}x")

    comparison_rl_mwpm = compute_comparison_summary(
        rl_lers_d15,
        mwpm_lers_d15,
        "logical_error_rate",
        "RL_vs_MWPM_d15"
    )

    print(f"\nStatistical comparison:")
    print(f"  Difference (RL - MWPM): {comparison_rl_mwpm['estimate_diff']:.4f}")
    print(f"  95% CI: [{comparison_rl_mwpm['ci_95'][0]:.4f}, {comparison_rl_mwpm['ci_95'][1]:.4f}]")
    print(f"  p-value: {comparison_rl_mwpm['p_value']:.4f}")
    print(f"  Conclusion: {comparison_rl_mwpm['conclusion']}")

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING ANALYSIS RESULTS")
print("="*80)

# Save comparison JSONs
if comparison_200_5000:
    outfile = '/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/comparison_200ep_vs_5000ep.json'
    with open(outfile, 'w') as f:
        json.dump(comparison_200_5000, f, indent=2)
    print(f"Saved: comparison_200ep_vs_5000ep.json")

if baseline_d15:
    outfile = '/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/comparison_rl_vs_mwpm_d15.json'
    with open(outfile, 'w') as f:
        json.dump(comparison_rl_mwpm, f, indent=2)
    print(f"Saved: comparison_rl_vs_mwpm_d15.json")

# Save comprehensive revision analysis
revision_analysis = {
    "project": "QEC_RL_Scaling_Revision",
    "analysis_date": "2025-12-29",
    "total_experiments": len(results),
    "undertraining_hypothesis": undertraining_analysis,
    "key_comparisons": {
        "200ep_vs_5000ep_d15": comparison_200_5000,
        "rl_vs_mwpm_d15": comparison_rl_mwpm if baseline_d15 else None
    },
    "learning_curve_d15": {
        "episodes": episodes_list,
        "mean_lers": [np.mean(d15_by_episodes[ep]) for ep in episodes_list],
        "std_lers": [np.std(d15_by_episodes[ep]) for ep in episodes_list],
        "n_seeds": [len(d15_by_episodes[ep]) for ep in episodes_list]
    }
}

outfile = '/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/revision_analysis.json'
with open(outfile, 'w') as f:
    json.dump(revision_analysis, f, indent=2)
print(f"Saved: revision_analysis.json")

# ============================================================================
# GENERATE FOLLOW-UP PLAN (if undertraining hypothesis rejected)
# ============================================================================
if undertraining_analysis['verdict'] == 'REJECTED':
    print("\n" + "="*80)
    print("GENERATING FOLLOW-UP PLAN")
    print("="*80)

    followup_plan = {
        "trigger": f"Undertraining hypothesis REJECTED: Extended training (25x) shows no significant improvement (p={comparison_200_5000['p_value']:.4f})",
        "original_hypothesis_failed": "Insufficient training (200 episodes) limits RL performance at d=15",
        "mode": "demo",
        "proposed_hypotheses": [
            {
                "hypothesis": "Insufficient model capacity: GNN architecture cannot represent complex d=15 decoding policy",
                "rationale": "If the model lacks capacity, more training cannot improve performance regardless of episode count.",
                "diagnostic_experiment": "Increase GNN depth to 8-12 layers and hidden dimensions to 256-512, retrain at d=15",
                "expected_outcome": "If correct, larger model should significantly reduce LER even with same training budget",
                "required_comparisons": ["4L_128H vs 12L_512H at d=15 with 2000 episodes"],
                "priority": 1
            },
            {
                "hypothesis": "Inadequate reward signal: Sparse logical error reward provides insufficient learning signal for d=15 complexity",
                "rationale": "Large code distances have exponentially more error configurations, sparse reward may be too delayed.",
                "diagnostic_experiment": "Compare dense reward shaping (syndrome-based intermediate rewards) vs sparse at d=15",
                "expected_outcome": "If correct, dense reward should improve learning curve convergence and final LER",
                "required_comparisons": ["sparse vs dense_syndrome reward at d=15"],
                "priority": 1
            },
            {
                "hypothesis": "Fundamental algorithm limitation: GNN-based RL may be inherently unsuited for surface code decoding at scale",
                "rationale": "Surface code decoding may require global optimization (like MWPM) that local GNN message passing cannot achieve.",
                "diagnostic_experiment": "Analyze trained GNN decision boundaries and compare to MWPM optimal matching structure",
                "expected_outcome": "If correct, GNN decisions will show systematic deviations from optimal matching even on simple error patterns",
                "required_comparisons": ["Qualitative analysis of GNN vs MWPM matching decisions"],
                "priority": 2
            }
        ],
        "recommended_next_steps": [
            "Run model capacity ablation (priority 1) to rule out architecture limitations",
            "Run reward shaping ablation (priority 1) to test if learning signal is the issue",
            "If both fail, conduct qualitative analysis to understand fundamental mismatch"
        ]
    }

    outfile = '/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/followup_plan_revision.json'
    with open(outfile, 'w') as f:
        json.dump(followup_plan, f, indent=2)
    print(f"Saved: followup_plan_revision.json")

    print(f"\nProposed follow-up hypotheses:")
    for i, h in enumerate(followup_plan['proposed_hypotheses'], 1):
        print(f"{i}. {h['hypothesis']}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - files/results/comparison_200ep_vs_5000ep.json")
print("  - files/results/comparison_rl_vs_mwpm_d15.json")
print("  - files/results/revision_analysis.json")
if undertraining_analysis['verdict'] == 'REJECTED':
    print("  - files/results/followup_plan_revision.json")
