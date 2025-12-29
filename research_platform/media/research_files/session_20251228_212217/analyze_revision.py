import json
import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.append('/Users/jminding/Desktop/Code/Research Agent/research_platform')

# Load extended results
print("Loading extended results table...")
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/media/research_files/session_20251228_212217/files/results/extended_results_table.json', 'r') as f:
    extended_data = json.load(f)

print(f"Total experiments: {len(extended_data['results'])}")

# Convert to DataFrame for easier analysis
results = []
for r in extended_data['results']:
    row = {
        'experiment_name': r['experiment_name'],
        'code_distance': r['parameters'].get('code_distance'),
        'physical_error_rate': r['parameters'].get('physical_error_rate'),
        'training_episodes': r['parameters'].get('training_episodes'),
        'seed': r['parameters'].get('seed'),
        'reward_type': r['parameters'].get('reward_type'),
        'gnn_layers': r['parameters'].get('gnn_layers'),
        'hidden_dim': r['parameters'].get('hidden_dim'),
        'train_distance': r['parameters'].get('train_distance'),
        'test_distance': r['parameters'].get('test_distance'),
    }
    row.update(r['metrics'])
    results.append(row)

df = pd.DataFrame(results)

print("\n=== EXPERIMENT SUMMARY ===")
print(f"Experiments by type:")
print(df['experiment_name'].value_counts())
print(f"\nTotal rows: {len(df)}")

# 1. ORIGINAL vs EXTENDED d=15 COMPARISON
print("\n=== 1. ORIGINAL vs EXTENDED d=15 COMPARISON ===")
original_d15_ler = 0.312  # From experiment plan
original_seeds = 2
original_episodes = 200

# Extended d=15 results with 200 episodes (apples-to-apples)
extended_d15_200ep = df[(df['experiment_name'] == 'extended_training_d15') &
                         (df['code_distance'] == 15) &
                         (df['training_episodes'] == 200)]

if len(extended_d15_200ep) > 0:
    extended_lers_200 = extended_d15_200ep['logical_error_rate'].values
    print(f"\nOriginal d=15 (200 ep, 2 seeds): LER = {original_d15_ler:.4f}")
    print(f"Extended d=15 (200 ep, {len(extended_lers_200)} seeds): LER = {extended_lers_200.mean():.4f} ± {extended_lers_200.std():.4f}")
    print(f"  Range: [{extended_lers_200.min():.4f}, {extended_lers_200.max():.4f}]")

# Extended d=15 with maximum training (5000 episodes)
extended_d15_5000ep = df[(df['experiment_name'] == 'extended_training_d15') &
                          (df['code_distance'] == 15) &
                          (df['training_episodes'] == 5000)]

if len(extended_d15_5000ep) > 0:
    extended_lers_5000 = extended_d15_5000ep['logical_error_rate'].values
    print(f"Extended d=15 (5000 ep, {len(extended_lers_5000)} seeds): LER = {extended_lers_5000.mean():.4f} ± {extended_lers_5000.std():.4f}")
    print(f"  Range: [{extended_lers_5000.min():.4f}, {extended_lers_5000.max():.4f}]")

    if len(extended_lers_200) > 0:
        improvement = (extended_lers_200.mean() - extended_lers_5000.mean()) / extended_lers_200.mean() * 100
        print(f"  Improvement from 200→5000 episodes: {improvement:.1f}%")

# 2. TEST UNDERTRAINING HYPOTHESIS - Learning Curves
print("\n=== 2. UNDERTRAINING HYPOTHESIS - LEARNING CURVES ===")
extended_d15_all = df[(df['experiment_name'] == 'extended_training_d15') &
                       (df['code_distance'] == 15)]

if len(extended_d15_all) > 0:
    # Group by training episodes
    curve_data = extended_d15_all.groupby('training_episodes').agg({
        'logical_error_rate': ['mean', 'std', 'count']
    }).reset_index()
    curve_data.columns = ['training_episodes', 'ler_mean', 'ler_std', 'count']
    curve_data = curve_data.sort_values('training_episodes')

    print("\nLearning Curve at d=15:")
    print(curve_data.to_string(index=False))

    # Test for convergence using linear regression on log(episodes) vs LER
    from scipy.stats import linregress
    x = np.log10(curve_data['training_episodes'].values)
    y = curve_data['ler_mean'].values
    slope, intercept, r_value, p_value, stderr = linregress(x, y)

    print(f"\nLinear regression: LER vs log10(episodes)")
    print(f"  Slope: {slope:.4f} (negative = improvement with training)")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Interpretation: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} improvement with training")

    # Compare first vs last
    first_episodes = curve_data.iloc[0]
    last_episodes = curve_data.iloc[-1]
    improvement_pct = (first_episodes['ler_mean'] - last_episodes['ler_mean']) / first_episodes['ler_mean'] * 100
    print(f"\nOverall improvement: {first_episodes['training_episodes']:.0f}→{last_episodes['training_episodes']:.0f} episodes")
    print(f"  LER: {first_episodes['ler_mean']:.4f} → {last_episodes['ler_mean']:.4f} ({improvement_pct:.1f}% reduction)")

# 3. ABLATION STUDIES
print("\n=== 3. REWARD SHAPING ABLATION ===")
reward_d15 = df[(df['experiment_name'] == 'reward_shaping_ablation') &
                 (df['code_distance'] == 15)]

if len(reward_d15) > 0:
    reward_summary = reward_d15.groupby('reward_type').agg({
        'logical_error_rate': ['mean', 'std', 'count']
    }).reset_index()
    reward_summary.columns = ['reward_type', 'ler_mean', 'ler_std', 'count']
    reward_summary = reward_summary.sort_values('ler_mean')
    print("\nReward types at d=15:")
    print(reward_summary.to_string(index=False))

    best_reward = reward_summary.iloc[0]
    print(f"\nBest reward type: {best_reward['reward_type']} (LER = {best_reward['ler_mean']:.4f})")

print("\n=== 4. GNN ARCHITECTURE ABLATION ===")
gnn_d15 = df[(df['experiment_name'] == 'gnn_depth_ablation') &
              (df['code_distance'] == 15)]

if len(gnn_d15) > 0:
    gnn_summary = gnn_d15.groupby(['gnn_layers', 'hidden_dim']).agg({
        'logical_error_rate': ['mean', 'std', 'count']
    }).reset_index()
    gnn_summary.columns = ['gnn_layers', 'hidden_dim', 'ler_mean', 'ler_std', 'count']
    gnn_summary = gnn_summary.sort_values('ler_mean')
    print("\nGNN architectures at d=15:")
    print(gnn_summary.to_string(index=False))

    best_arch = gnn_summary.iloc[0]
    print(f"\nBest architecture: {best_arch['gnn_layers']} layers, {best_arch['hidden_dim']} hidden (LER = {best_arch['ler_mean']:.4f})")

# 5. ZERO-SHOT GENERALIZATION
print("\n=== 5. ZERO-SHOT GENERALIZATION (d=7→d=15) ===")
zero_shot = df[df['experiment_name'] == 'zero_shot_generalization']

if len(zero_shot) > 0:
    gen_summary = zero_shot.groupby('training_episodes').agg({
        'logical_error_rate_train_dist': ['mean', 'std'],
        'logical_error_rate_test_dist': ['mean', 'std'],
        'generalization_gap': ['mean', 'std']
    }).reset_index()
    gen_summary.columns = ['training_episodes', 'train_mean', 'train_std',
                           'test_mean', 'test_std', 'gap_mean', 'gap_std']
    gen_summary = gen_summary.sort_values('training_episodes')
    print("\nGeneralization by training budget:")
    print(gen_summary.to_string(index=False))

# 6. MWPM VALIDATION
print("\n=== 6. MWPM BASELINE VALIDATION ===")
mwpm_val = df[df['experiment_name'] == 'mwpm_validation']

if len(mwpm_val) > 0:
    # Focus on p=0.005 for comparison
    mwpm_005 = mwpm_val[mwpm_val['physical_error_rate'] == 0.005]
    if len(mwpm_005) > 0:
        mwpm_summary = mwpm_005.groupby('code_distance').agg({
            'logical_error_rate': ['mean', 'std'],
            'expected_benchmark_rate': ['mean'],
            'deviation_from_benchmark': ['mean']
        }).reset_index()
        mwpm_summary.columns = ['code_distance', 'observed', 'std', 'expected', 'deviation']
        print(f"\nMWPM validation at p=0.005:")
        print(mwpm_summary.to_string(index=False))

# 7. BASELINE COMPARISON (RL vs MWPM with extended training)
print("\n=== 7. BASELINE COMPARISON (Extended Training) ===")
baseline_comp = df[df['experiment_name'] == 'baseline_comparison']

if len(baseline_comp) > 0:
    baseline_summary = baseline_comp.groupby('code_distance').agg({
        'logical_error_rate_rl': ['mean', 'std'],
        'logical_error_rate_mwpm': ['mean', 'std'],
        'rl_vs_mwpm_ratio': ['mean', 'std']
    }).reset_index()
    baseline_summary.columns = ['code_distance', 'rl_mean', 'rl_std',
                                 'mwpm_mean', 'mwpm_std', 'ratio_mean', 'ratio_std']
    print("\nRL vs MWPM comparison (2000 episodes):")
    print(baseline_summary.to_string(index=False))

    d15_comp = baseline_summary[baseline_summary['code_distance'] == 15]
    if len(d15_comp) > 0:
        print(f"\nAt d=15:")
        print(f"  RL: {d15_comp['rl_mean'].values[0]:.4f} ± {d15_comp['rl_std'].values[0]:.4f}")
        print(f"  MWPM: {d15_comp['mwpm_mean'].values[0]:.4f} ± {d15_comp['mwpm_std'].values[0]:.4f}")
        print(f"  Ratio (RL/MWPM): {d15_comp['ratio_mean'].values[0]:.2f}x")

print("\n=== ANALYSIS COMPLETE ===")
print("\nKey files will be generated:")
print("  - revision_analysis.json")
print("  - comparison_*.json (statistical tests)")
print("  - followup_plan_revision.json (if hypothesis rejected)")
