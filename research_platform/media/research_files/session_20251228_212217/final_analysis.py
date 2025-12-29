#!/usr/bin/env python3
"""
Complete statistical analysis of QEC revision experiments.
Generates all required comparisons with confidence intervals and p-values.
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict

# Load data
print("Loading extended results...")
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/extended_results_table.json', 'r') as f:
    data = json.load(f)

results = data['results']
print(f"Loaded {len(results)} experimental results\n")

# Load experiment plan
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/theory/experiment_plan.json', 'r') as f:
    exp_plan = json.load(f)

original_d15_ler = exp_plan['original_results_summary']['d15_rl_logical_error_rate']
original_d15_mwpm = exp_plan['original_results_summary']['d15_mwpm_logical_error_rate']

print(f"Original results (from experiment plan):")
print(f"  d=15 RL LER: {original_d15_ler:.4f}")
print(f"  d=15 MWPM LER: {original_d15_mwpm:.4f}")
print(f"  Training episodes: {exp_plan['original_results_summary']['original_training_episodes']}")
print(f"  Seeds: {exp_plan['original_results_summary']['original_seeds']}\n")

def compute_comparison_summary(data_a, data_b, metric_name, comparison_name, group_a_name="A", group_b_name="B"):
    """Compute full statistical comparison between two datasets."""
    data_a = np.array(data_a)
    data_b = np.array(data_b)

    mean_a, mean_b = data_a.mean(), data_b.mean()
    std_a, std_b = data_a.std(ddof=1) if len(data_a) > 1 else 0, data_b.std(ddof=1) if len(data_b) > 1 else 0
    diff = mean_a - mean_b

    if len(data_a) > 1 and len(data_b) > 1:
        t_stat, p_value = stats.ttest_ind(data_a, data_b)
        pooled_std = np.sqrt((data_a.var() + data_b.var()) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        se_diff = np.sqrt(std_a**2/len(data_a) + std_b**2/len(data_b))
        df = len(data_a) + len(data_b) - 2
        t_crit = stats.t.ppf(0.975, df)
        ci_95 = [diff - t_crit * se_diff, diff + t_crit * se_diff]
    else:
        t_stat, p_value = np.nan, np.nan
        cohens_d = np.nan
        ci_95 = [diff, diff]

    if p_value < 0.05:
        if diff > 0:
            conclusion = f"{group_a_name} significantly higher than {group_b_name} (p={p_value:.4f})"
        else:
            conclusion = f"{group_a_name} significantly lower than {group_b_name} (p={p_value:.4f})"
    else:
        conclusion = f"No significant difference between {group_a_name} and {group_b_name} (p={p_value:.4f})"

    return {
        "comparison": comparison_name,
        "metric": metric_name,
        "group_a_name": group_a_name,
        "group_a_mean": float(mean_a),
        "group_a_std": float(std_a),
        "group_a_n": int(len(data_a)),
        "group_b_name": group_b_name,
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
# 1. EXTENDED TRAINING ANALYSIS AT d=15
# ============================================================================
print("="*80)
print("1. EXTENDED TRAINING AT d=15 - TESTING UNDERTRAINING HYPOTHESIS")
print("="*80)

# Extract learning curve experiments
learning_curve_d15 = [r for r in results if r['config_name'].startswith('learning_curve_d15')]

# Group by episodes
d15_by_episodes = defaultdict(list)
for r in learning_curve_d15:
    episodes = r['parameters']['episodes_completed']
    ler = r['metrics']['logical_error_rate']
    d15_by_episodes[episodes].append(ler)

# Also include comparison_d15 results (at 2000 episodes)
comparison_d15 = [r for r in results if r['config_name'].startswith('comparison_d15')]
for r in comparison_d15:
    ler = r['metrics']['logical_error_rate_rl']
    d15_by_episodes[2000].append(ler)

# Add original 200-episode results as reference point
d15_by_episodes[200] = [original_d15_ler]  # Only 1 data point from original

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

# Statistical test: Compare earliest multi-seed vs latest multi-seed
# Find first and last episodes with n>=3
valid_episodes = [ep for ep in episodes_list if len(d15_by_episodes[ep]) >= 3]

if len(valid_episodes) >= 2:
    first_ep = min(valid_episodes)
    last_ep = max(valid_episodes)

    comparison_first_last = compute_comparison_summary(
        d15_by_episodes[first_ep],
        d15_by_episodes[last_ep],
        "logical_error_rate",
        f"{first_ep}ep_vs_{last_ep}ep",
        f"{first_ep} episodes",
        f"{last_ep} episodes"
    )

    print(f"\nStatistical comparison: {first_ep} vs {last_ep} episodes")
    print(f"  {first_ep} ep: {comparison_first_last['group_a_mean']:.4f} ± {comparison_first_last['group_a_std']:.4f} (n={comparison_first_last['group_a_n']})")
    print(f"  {last_ep} ep: {comparison_first_last['group_b_mean']:.4f} ± {comparison_first_last['group_b_std']:.4f} (n={comparison_first_last['group_b_n']})")
    print(f"  Difference: {comparison_first_last['estimate_diff']:.4f}")
    print(f"  95% CI: [{comparison_first_last['ci_95'][0]:.4f}, {comparison_first_last['ci_95'][1]:.4f}]")
    print(f"  p-value: {comparison_first_last['p_value']:.4f}")
    print(f"  Cohen's d: {comparison_first_last['cohens_d']:.3f}")
    print(f"  Conclusion: {comparison_first_last['conclusion']}")

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

if len(valid_episodes) >= 2:
    improvement_pct = (comparison_first_last['estimate_diff'] / comparison_first_last['group_a_mean']) * 100
    p_val = comparison_first_last['p_value']
    training_increase = last_ep / first_ep

    print(f"\nOriginal Hypothesis:")
    print(f"  'Insufficient training (200 episodes) limits RL performance at d=15'")

    print(f"\nEvidence from extended experiments:")
    print(f"  - Training increased from {first_ep} to {last_ep} episodes ({training_increase:.0f}x)")
    print(f"  - LER change: {comparison_first_last['estimate_diff']:.4f} ({improvement_pct:.1f}%)")
    print(f"  - Statistical significance: p = {p_val:.4f}")
    print(f"  - Effect size: Cohen's d = {comparison_first_last['cohens_d']:.3f}")

    # Determine verdict
    significant = p_val < 0.05
    meaningful = abs(improvement_pct) > 5

    if not significant:
        verdict = "REJECTED"
        explanation = f"Extended training ({training_increase:.0f}x more episodes) produces statistically insignificant improvement (p={p_val:.4f}). The performance gap is NOT primarily due to undertraining."
    elif significant and meaningful and improvement_pct > 0:
        verdict = "PARTIALLY CONFIRMED"
        explanation = f"Extended training shows statistically significant improvement ({improvement_pct:.1f}%, p={p_val:.4f}), but the magnitude is modest. Undertraining contributes to poor performance but is not the primary limiting factor."
    else:
        verdict = "INCONCLUSIVE"
        explanation = f"Results show {improvement_pct:.1f}% change but evidence is mixed."

    print(f"\n{'='*80}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*80}")
    print(f"{explanation}")

    undertraining_analysis = {
        "original_hypothesis": "Insufficient training (200 episodes) limits RL performance at d=15",
        "verdict": verdict,
        "evidence": {
            "training_increase": f"{training_increase:.0f}x ({first_ep} -> {last_ep} episodes)",
            "ler_change": f"{comparison_first_last['estimate_diff']:.4f}",
            "percent_improvement": f"{improvement_pct:.1f}%",
            "p_value": f"{p_val:.4f}",
            "cohens_d": f"{comparison_first_last['cohens_d']:.3f}",
            "statistically_significant": significant,
            "practically_meaningful": meaningful
        },
        "explanation": explanation
    }

# ============================================================================
# 3. BASELINE COMPARISON (RL vs MWPM at d=15)
# ============================================================================
print("\n" + "="*80)
print("3. BASELINE COMPARISON: RL vs MWPM at d=15 (2000 episodes)")
print("="*80)

rl_lers_d15 = [r['metrics']['logical_error_rate_rl'] for r in comparison_d15]
mwpm_lers_d15 = [r['metrics']['logical_error_rate_mwpm'] for r in comparison_d15]
ratios_d15 = [r['metrics']['rl_vs_mwpm_ratio'] for r in comparison_d15]

print(f"\nAt d=15 (with 2000 episodes training):")
print(f"  RL LER:   {np.mean(rl_lers_d15):.4f} ± {np.std(rl_lers_d15):.4f} (n={len(rl_lers_d15)})")
print(f"  MWPM LER: {np.mean(mwpm_lers_d15):.4f} ± {np.std(mwpm_lers_d15):.4f} (n={len(mwpm_lers_d15)})")
print(f"  Ratio:    {np.mean(ratios_d15):.2f}x ± {np.std(ratios_d15):.2f}x")

comparison_rl_mwpm = compute_comparison_summary(
    rl_lers_d15,
    mwpm_lers_d15,
    "logical_error_rate",
    "RL_vs_MWPM_d15",
    "RL",
    "MWPM"
)

print(f"\nStatistical comparison:")
print(f"  Difference (RL - MWPM): {comparison_rl_mwpm['estimate_diff']:.4f}")
print(f"  95% CI: [{comparison_rl_mwpm['ci_95'][0]:.4f}, {comparison_rl_mwpm['ci_95'][1]:.4f}]")
print(f"  p-value: {comparison_rl_mwpm['p_value']:.4f}")
print(f"  Cohen's d: {comparison_rl_mwpm['cohens_d']:.3f}")
print(f"  Conclusion: {comparison_rl_mwpm['conclusion']}")

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING ANALYSIS RESULTS")
print("="*80)

# Save comparison JSONs
if len(valid_episodes) >= 2:
    outfile = '/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/comparison_first_vs_last_episodes.json'
    with open(outfile, 'w') as f:
        json.dump(comparison_first_last, f, indent=2)
    print(f"Saved: comparison_first_vs_last_episodes.json")

outfile = '/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/comparison_rl_vs_mwpm_d15.json'
with open(outfile, 'w') as f:
    json.dump(comparison_rl_mwpm, f, indent=2)
print(f"Saved: comparison_rl_vs_mwpm_d15.json")

# Save comprehensive revision analysis
revision_analysis = {
    "project": "QEC_RL_Scaling_Revision",
    "analysis_date": "2025-12-29",
    "total_experiments": len(results),
    "undertraining_hypothesis": undertraining_analysis if len(valid_episodes) >= 2 else None,
    "key_comparisons": {
        "first_vs_last_episodes_d15": comparison_first_last if len(valid_episodes) >= 2 else None,
        "rl_vs_mwpm_d15": comparison_rl_mwpm
    },
    "learning_curve_d15": {
        "episodes": episodes_list,
        "mean_lers": [np.mean(d15_by_episodes[ep]) for ep in episodes_list],
        "std_lers": [np.std(d15_by_episodes[ep]) for ep in episodes_list],
        "n_seeds": [len(d15_by_episodes[ep]) for ep in episodes_list]
    },
    "summary": {
        "original_d15_ler": original_d15_ler,
        "original_mwpm_ler": original_d15_mwpm,
        "extended_d15_ler_2000ep": np.mean(rl_lers_d15),
        "extended_mwpm_ler": np.mean(mwpm_lers_d15),
        "rl_vs_mwpm_ratio": np.mean(ratios_d15)
    }
}

outfile = '/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/revision_analysis.json'
with open(outfile, 'w') as f:
    json.dump(revision_analysis, f, indent=2)
print(f"Saved: revision_analysis.json")

# ============================================================================
# GENERATE FOLLOW-UP PLAN (if undertraining hypothesis rejected)
# ============================================================================
if len(valid_episodes) >= 2 and undertraining_analysis['verdict'] == 'REJECTED':
    print("\n" + "="*80)
    print("GENERATING FOLLOW-UP PLAN")
    print("="*80)

    followup_plan = {
        "trigger": f"Undertraining hypothesis REJECTED: Extended training ({training_increase:.0f}x) shows no significant improvement (p={p_val:.4f})",
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
print("  - files/results/comparison_first_vs_last_episodes.json")
print("  - files/results/comparison_rl_vs_mwpm_d15.json")
print("  - files/results/revision_analysis.json")
if len(valid_episodes) >= 2 and undertraining_analysis['verdict'] == 'REJECTED':
    print("  - files/results/followup_plan_revision.json")
