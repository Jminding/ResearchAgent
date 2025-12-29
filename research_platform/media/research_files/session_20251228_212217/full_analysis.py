#!/usr/bin/env python3
"""
Complete statistical analysis of QEC revision experiments.
This script generates all required comparisons with confidence intervals and p-values.
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

# Organize by experiment type
experiments = defaultdict(list)
for r in results:
    config_name = r['config_name']
    # Identify experiment type
    if config_name.startswith('d'):
        # d15_200ep_s1 format
        if '200ep' in config_name or '500ep' in config_name or '1000ep' in config_name or '2000ep' in config_name or '5000ep' in config_name:
            exp_type = 'extended_training'
        elif 'zeroshot' in config_name:
            exp_type = 'zero_shot'
        else:
            exp_type = 'unknown_d'
    elif config_name.startswith('comparison'):
        exp_type = 'baseline_comparison'
    elif config_name.startswith('reward'):
        exp_type = 'reward_shaping'
    elif config_name.startswith('gnn'):
        exp_type = 'gnn_architecture'
    elif config_name.startswith('mwpm'):
        exp_type = 'mwpm_validation'
    else:
        exp_type = 'other'

    experiments[exp_type].append(r)

print("Experiment types found:")
for exp_type, exps in experiments.items():
    print(f"  {exp_type}: {len(exps)}")

# ============================================================================
# 1. ORIGINAL vs EXTENDED d=15 COMPARISON
# ============================================================================
print("\n" + "="*80)
print("1. ORIGINAL vs EXTENDED d=15 COMPARISON")
print("="*80)

original_d15_ler = 0.312  # From experiment plan

# Extract d=15 results by training episodes
d15_by_episodes = defaultdict(list)
for r in experiments['extended_training']:
    if r['parameters']['code_distance'] == 15:
        episodes = r['parameters']['training_episodes']
        ler = r['metrics']['logical_error_rate']
        d15_by_episodes[episodes].append(ler)

print("\nLearning curve at d=15:")
print(f"{'Episodes':<12} {'Mean LER':<12} {'Std':<12} {'n':<6} {'95% CI':<25}")
print("-" * 75)

episodes_sorted = sorted(d15_by_episodes.keys())
for ep in episodes_sorted:
    lers = np.array(d15_by_episodes[ep])
    mean_ler = lers.mean()
    std_ler = lers.std(ddof=1) if len(lers) > 1 else 0
    n = len(lers)

    # Compute 95% CI using t-distribution
    if n > 1:
        ci_95 = stats.t.interval(0.95, n-1, loc=mean_ler, scale=std_ler/np.sqrt(n))
    else:
        ci_95 = (mean_ler, mean_ler)

    print(f"{ep:<12} {mean_ler:<12.4f} {std_ler:<12.4f} {n:<6} [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

# Test for improvement: 200 vs 5000 episodes
if 200 in d15_by_episodes and 5000 in d15_by_episodes:
    lers_200 = np.array(d15_by_episodes[200])
    lers_5000 = np.array(d15_by_episodes[5000])

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(lers_200, lers_5000)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((lers_200.var() + lers_5000.var()) / 2)
    cohens_d = (lers_200.mean() - lers_5000.mean()) / pooled_std if pooled_std > 0 else 0

    print(f"\nStatistical test: 200 vs 5000 episodes")
    print(f"  200 ep: {lers_200.mean():.4f} ± {lers_200.std():.4f} (n={len(lers_200)})")
    print(f"  5000 ep: {lers_5000.mean():.4f} ± {lers_5000.std():.4f} (n={len(lers_5000)})")
    print(f"  Difference: {lers_200.mean() - lers_5000.mean():.4f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Interpretation: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at 95% level")

# Linear regression on learning curve
if len(episodes_sorted) >= 3:
    x = np.log10(episodes_sorted)
    y = np.array([np.mean(d15_by_episodes[ep]) for ep in episodes_sorted])
    slope, intercept, r_value, p_value_reg, stderr = stats.linregress(x, y)

    print(f"\nLearning curve trend (linear regression on log10(episodes)):")
    print(f"  Slope: {slope:.4f} (negative = improvement with training)")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value_reg:.4f}")
    print(f"  Interpretation: {'SIGNIFICANT' if p_value_reg < 0.05 else 'NOT SIGNIFICANT'} trend")

# ============================================================================
# 2. UNDERTRAINING HYPOTHESIS VERDICT
# ============================================================================
print("\n" + "="*80)
print("2. UNDERTRAINING HYPOTHESIS VERDICT")
print("="*80)

if 200 in d15_by_episodes and 5000 in d15_by_episodes:
    improvement_pct = (lers_200.mean() - lers_5000.mean()) / lers_200.mean() * 100

    print(f"\nOriginal hypothesis: 'Insufficient training episodes (200) limits RL performance at d=15'")
    print(f"\nEvidence:")
    print(f"  - Increased training from 200 to 5000 episodes (25x increase)")
    print(f"  - LER improvement: {improvement_pct:.1f}%")
    print(f"  - Statistical significance: p={p_value:.4f}")
    print(f"  - Effect size: Cohen's d = {cohens_d:.3f}")

    # Decision criteria
    significant_improvement = (p_value < 0.05) and (improvement_pct > 10)

    if significant_improvement:
        verdict = "PARTIALLY CONFIRMED"
        explanation = "Extended training shows statistically significant but modest improvement. Original hypothesis underestimated the degree of undertraining."
    else:
        verdict = "REJECTED"
        explanation = f"Extended training (25x more episodes) produces {'statistically insignificant' if p_value >= 0.05 else 'negligible'} improvement ({improvement_pct:.1f}%). The performance gap is not primarily due to undertraining."

    print(f"\nVERDICT: {verdict}")
    print(f"EXPLANATION: {explanation}")

# ============================================================================
# 3. REWARD SHAPING ABLATION
# ============================================================================
print("\n" + "="*80)
print("3. REWARD SHAPING ABLATION")
print("="*80)

# Group by reward type and code distance
reward_results = defaultdict(lambda: defaultdict(list))
for r in experiments['reward_shaping']:
    reward_type = r['parameters']['reward_type']
    code_dist = r['parameters']['code_distance']
    ler = r['metrics']['logical_error_rate']
    reward_results[code_dist][reward_type].append(ler)

for dist in sorted(reward_results.keys()):
    print(f"\nCode distance d={dist}:")
    print(f"{'Reward Type':<20} {'Mean LER':<12} {'Std':<12} {'n':<6} {'95% CI':<25}")
    print("-" * 80)

    reward_types = sorted(reward_results[dist].keys())
    for rt in reward_types:
        lers = np.array(reward_results[dist][rt])
        mean_ler = lers.mean()
        std_ler = lers.std(ddof=1) if len(lers) > 1 else 0
        n = len(lers)

        if n > 1:
            ci_95 = stats.t.interval(0.95, n-1, loc=mean_ler, scale=std_ler/np.sqrt(n))
        else:
            ci_95 = (mean_ler, mean_ler)

        print(f"{rt:<20} {mean_ler:<12.4f} {std_ler:<12.4f} {n:<6} [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

    # Compare best vs worst
    if len(reward_types) >= 2:
        best_rt = min(reward_types, key=lambda rt: np.mean(reward_results[dist][rt]))
        worst_rt = max(reward_types, key=lambda rt: np.mean(reward_results[dist][rt]))

        best_lers = np.array(reward_results[dist][best_rt])
        worst_lers = np.array(reward_results[dist][worst_rt])

        t_stat, p_value = stats.ttest_ind(best_lers, worst_lers)
        pooled_std = np.sqrt((best_lers.var() + worst_lers.var()) / 2)
        cohens_d = (worst_lers.mean() - best_lers.mean()) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Best: {best_rt} (LER = {best_lers.mean():.4f})")
        print(f"  Worst: {worst_rt} (LER = {worst_lers.mean():.4f})")
        print(f"  Difference: {worst_lers.mean() - best_lers.mean():.4f} (p={p_value:.4f}, d={cohens_d:.3f})")

# ============================================================================
# 4. GNN ARCHITECTURE ABLATION
# ============================================================================
print("\n" + "="*80)
print("4. GNN ARCHITECTURE ABLATION")
print("="*80)

# Group by architecture and code distance
gnn_results = defaultdict(lambda: defaultdict(list))
for r in experiments['gnn_architecture']:
    layers = r['parameters']['gnn_layers']
    hidden = r['parameters']['hidden_dim']
    arch = f"{layers}L_{hidden}H"
    code_dist = r['parameters']['code_distance']
    ler = r['metrics']['logical_error_rate']
    gnn_results[code_dist][arch].append(ler)

for dist in sorted(gnn_results.keys()):
    print(f"\nCode distance d={dist}:")
    print(f"{'Architecture':<20} {'Mean LER':<12} {'Std':<12} {'n':<6} {'95% CI':<25}")
    print("-" * 80)

    archs = sorted(gnn_results[dist].keys())
    for arch in archs:
        lers = np.array(gnn_results[dist][arch])
        mean_ler = lers.mean()
        std_ler = lers.std(ddof=1) if len(lers) > 1 else 0
        n = len(lers)

        if n > 1:
            ci_95 = stats.t.interval(0.95, n-1, loc=mean_ler, scale=std_ler/np.sqrt(n))
        else:
            ci_95 = (mean_ler, mean_ler)

        print(f"{arch:<20} {mean_ler:<12.4f} {std_ler:<12.4f} {n:<6} [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

# ============================================================================
# 5. ZERO-SHOT GENERALIZATION
# ============================================================================
print("\n" + "="*80)
print("5. ZERO-SHOT GENERALIZATION (d=7 → d=15)")
print("="*80)

# Group by training episodes
zeroshot_by_episodes = defaultdict(lambda: {'train': [], 'test': [], 'gap': []})
for r in experiments['zero_shot']:
    episodes = r['parameters']['training_episodes']
    zeroshot_by_episodes[episodes]['train'].append(r['metrics']['logical_error_rate_train_dist'])
    zeroshot_by_episodes[episodes]['test'].append(r['metrics']['logical_error_rate_test_dist'])
    zeroshot_by_episodes[episodes]['gap'].append(r['metrics']['generalization_gap'])

print(f"\n{'Episodes':<12} {'Train@d=7':<15} {'Test@d=15':<15} {'Gap':<15}")
print("-" * 60)

for ep in sorted(zeroshot_by_episodes.keys()):
    train_lers = np.array(zeroshot_by_episodes[ep]['train'])
    test_lers = np.array(zeroshot_by_episodes[ep]['test'])
    gaps = np.array(zeroshot_by_episodes[ep]['gap'])

    print(f"{ep:<12} {train_lers.mean():<6.4f}±{train_lers.std():<7.4f} {test_lers.mean():<6.4f}±{test_lers.std():<7.4f} {gaps.mean():<6.4f}±{gaps.std():<7.4f}")

# ============================================================================
# 6. BASELINE COMPARISON (RL vs MWPM)
# ============================================================================
print("\n" + "="*80)
print("6. BASELINE COMPARISON (RL vs MWPM, 2000 episodes)")
print("="*80)

# Group by code distance
baseline_by_dist = defaultdict(lambda: {'rl': [], 'mwpm': [], 'ratio': []})
for r in experiments['baseline_comparison']:
    dist = r['parameters']['code_distance']
    baseline_by_dist[dist]['rl'].append(r['metrics']['logical_error_rate_rl'])
    baseline_by_dist[dist]['mwpm'].append(r['metrics']['logical_error_rate_mwpm'])
    baseline_by_dist[dist]['ratio'].append(r['metrics']['rl_vs_mwpm_ratio'])

print(f"\n{'d':<6} {'RL LER':<18} {'MWPM LER':<18} {'Ratio (RL/MWPM)':<18}")
print("-" * 65)

for dist in sorted(baseline_by_dist.keys()):
    rl_lers = np.array(baseline_by_dist[dist]['rl'])
    mwpm_lers = np.array(baseline_by_dist[dist]['mwpm'])
    ratios = np.array(baseline_by_dist[dist]['ratio'])

    print(f"{dist:<6} {rl_lers.mean():<6.4f}±{rl_lers.std():<10.4f} {mwpm_lers.mean():<6.4f}±{mwpm_lers.std():<10.4f} {ratios.mean():<6.2f}±{ratios.std():<10.2f}")

# ============================================================================
# 7. MWPM VALIDATION
# ============================================================================
print("\n" + "="*80)
print("7. MWPM VALIDATION vs BENCHMARKS")
print("="*80)

# Group by code distance and physical error rate
mwpm_by_config = defaultdict(lambda: {'observed': [], 'expected': [], 'deviation': []})
for r in experiments['mwpm_validation']:
    dist = r['parameters']['code_distance']
    p_err = r['parameters']['physical_error_rate']
    key = (dist, p_err)
    mwpm_by_config[key]['observed'].append(r['metrics']['logical_error_rate'])
    mwpm_by_config[key]['expected'].append(r['metrics']['expected_benchmark_rate'])
    mwpm_by_config[key]['deviation'].append(r['metrics']['deviation_from_benchmark'])

# Focus on p=0.005
print(f"\nAt physical error rate p=0.005:")
print(f"{'d':<6} {'Observed':<15} {'Expected':<15} {'Deviation':<15}")
print("-" * 55)

for (dist, p_err), data in sorted(mwpm_by_config.items()):
    if p_err == 0.005:
        obs = np.array(data['observed'])
        exp = np.array(data['expected'])
        dev = np.array(data['deviation'])
        print(f"{dist:<6} {obs.mean():<15.6f} {exp.mean():<15.6f} {dev.mean():<+15.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
