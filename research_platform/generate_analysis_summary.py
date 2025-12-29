#!/usr/bin/env python3
"""
Generate comprehensive analysis directly from results
"""

import json
import math
from collections import defaultdict

def mean(data):
    return sum(data) / len(data) if data else 0

def std(data):
    if len(data) < 2:
        return 0
    m = mean(data)
    return math.sqrt(sum((x-m)**2 for x in data) / (len(data)-1))

def median(data):
    s = sorted(data)
    n = len(s)
    if n == 0:
        return 0
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2

# Load data
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results/results_table.json') as f:
    data = json.load(f)

experiments = [e for e in data['results'] if e.get('error') is None]

# Separate RL and MWPM
rl_exps = [e for e in experiments if 'RL' in e['algorithm']]
mwpm_exps = [e for e in experiments if e['algorithm'] == 'MWPM']

print(f"Total experiments: {len(experiments)}")
print(f"RL experiments: {len(rl_exps)}")
print(f"MWPM experiments: {len(mwpm_exps)}")

# Match pairs
mwpm_lookup = {}
for e in mwpm_exps:
    key = (e['distance'], e['noise_model'], e['additional_metrics']['physical_error_rate'], e['seed'])
    mwpm_lookup[key] = e

pairs = []
for rl in rl_exps:
    key = (rl['distance'], rl['noise_model'], rl['additional_metrics']['physical_error_rate'], rl['seed'])
    if key in mwpm_lookup:
        mwpm = mwpm_lookup[key]
        L_rl = rl['logical_error_rate']
        L_mwpm = mwpm['logical_error_rate']
        improvement = (L_mwpm - L_rl) / L_mwpm if L_mwpm > 0 else 0
        pairs.append({
            'd': rl['distance'],
            'noise': rl['noise_model'],
            'p': rl['additional_metrics']['physical_error_rate'],
            'seed': rl['seed'],
            'L_rl': L_rl,
            'L_mwpm': L_mwpm,
            'improvement': improvement
        })

print(f"Matched pairs: {len(pairs)}")

if not pairs:
    print("ERROR: No matched pairs!")
    exit(1)

# Primary hypothesis test
L_rl = [p['L_rl'] for p in pairs]
L_mwpm = [p['L_mwpm'] for p in pairs]
improvements = [p['improvement'] for p in pairs]

mean_improvement = mean(improvements)
std_improvement = std(improvements)
median_improvement = median(improvements)

# Paired t-test approximation
diff = [L_mwpm[i] - L_rl[i] for i in range(len(pairs))]
n = len(diff)
mean_diff = mean(diff)
std_diff = std(diff)
t_stat = mean_diff / (std_diff / math.sqrt(n)) if std_diff > 0 else 0

# Rough p-value
if abs(t_stat) > 3:
    p_value = 0.001
elif abs(t_stat) > 2.5:
    p_value = 0.01
elif abs(t_stat) > 2:
    p_value = 0.05
else:
    p_value = 0.1

# CI
se = std_improvement / math.sqrt(n)
ci_lower = mean_improvement - 2.0 * se
ci_upper = mean_improvement + 2.0 * se

# Results
print("\n" + "="*80)
print("PRIMARY HYPOTHESIS TEST: RL >= 20% improvement over MWPM")
print("="*80)
print(f"Sample size: {n} paired comparisons")
print(f"Mean improvement: {mean_improvement*100:.2f}%")
print(f"Median improvement: {median_improvement*100:.2f}%")
print(f"Std deviation: {std_improvement*100:.2f}%")
print(f"95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value (approx): {p_value:.4f}")
print(f"Significant (p<0.01): {p_value < 0.01}")
print(f"HYPOTHESIS SUPPORTED: {(mean_improvement >= 0.20) and (p_value < 0.01)}")

# By distance
print("\n" + "="*80)
print("ANALYSIS BY DISTANCE")
print("="*80)

for algo in ['RL_GNN', 'MWPM']:
    print(f"\n{algo}:")
    exps = [e for e in experiments if e['algorithm'] == algo]

    dist_groups = defaultdict(list)
    for e in exps:
        dist_groups[e['distance']].append(e['logical_error_rate'])

    for d in sorted(dist_groups.keys()):
        m = mean(dist_groups[d])
        s = std(dist_groups[d])
        print(f"  d={d}: {m:.4f} ± {s:.4f}")

# By noise model
print("\n" + "="*80)
print("NOISE MODEL TRANSFER")
print("="*80)

noise_groups = defaultdict(list)
for p in pairs:
    noise_groups[p['noise']].append(p['improvement'])

for noise in sorted(noise_groups.keys()):
    improvements = noise_groups[noise]
    m = mean(improvements)
    s = std(improvements)
    print(f"{noise}: {m*100:.2f}% ± {s*100:.2f}% (n={len(improvements)})")

# Save results
results = {
    'primary_hypothesis': {
        'overall': {
            'comparison': 'RL_vs_MWPM',
            'n_pairs': n,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_improvement_ratio': float(mean_improvement),
            'mean_improvement_percent': float(mean_improvement * 100),
            'std_improvement': float(std_improvement),
            'ci_95_ratio': [float(ci_lower), float(ci_upper)],
            'ci_95_percent': [float(ci_lower * 100), float(ci_upper * 100)],
            'mean_L_RL': float(mean(L_rl)),
            'mean_L_MWPM': float(mean(L_mwpm)),
            'significant': p_value < 0.01,
            'hypothesis_supported': (mean_improvement >= 0.20) and (p_value < 0.01)
        }
    }
}

with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results/analysis_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results/comparison_rl_vs_mwpm.json', 'w') as f:
    json.dump(results['primary_hypothesis']['overall'], f, indent=2)

# Follow-up if needed
if not results['primary_hypothesis']['overall']['hypothesis_supported']:
    followup = {
        'trigger': f"Improvement {mean_improvement*100:.1f}% < 20% (p={p_value:.4f})",
        'failed': True,
        'observed_percent': float(mean_improvement * 100),
        'target_percent': 20.0,
        'p_value': float(p_value),
        'hypotheses': [
            {
                'hypothesis': 'Insufficient training (200 steps)',
                'experiment': 'Extend to 1000+ steps',
                'expected': 'Improvement >= 10%',
                'priority': 1
            },
            {
                'hypothesis': 'Suboptimal reward function',
                'experiment': 'Test shaped vs sparse rewards',
                'expected': 'Lower error rates',
                'priority': 1
            },
            {
                'hypothesis': 'Architecture limitations',
                'experiment': 'Test deeper GNN or Transformer',
                'expected': 'Improvement >= 15%',
                'priority': 2
            }
        ],
        'mode': 'discovery'
    }

    with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results/followup_plan.json', 'w') as f:
        json.dump(followup, f, indent=2)

    print("\n⚠ Follow-up plan saved")

print("\n✓ Analysis files saved")
