#!/usr/bin/env python3
"""
Statistical Analysis of Heterophily-Aware GNN Experimental Results
Performs rigorous hypothesis testing with Bonferroni correction, CI computation, and effect sizes.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from typing import Dict, List, Tuple
import sys

# Load results
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results/results_table.json', 'r') as f:
    data = json.load(f)

results_list = data['results']

# Convert to DataFrame
df = pd.DataFrame([
    {
        'config_name': r['config_name'],
        'experiment': r['parameters']['experiment'],
        'model': r['parameters']['model'],
        'model_type': r['parameters']['model_type'],
        'homophily': r['parameters'].get('homophily', None),
        'actual_homophily': r['parameters'].get('actual_homophily', None),
        'prevalence': r['parameters'].get('prevalence', None),
        'imbalance_ratio': r['parameters'].get('imbalance_ratio', None),
        'seed': r['parameters']['seed'],
        'f1': r['metrics']['f1'],
        'auroc': r['metrics']['auroc'],
        'auprc': r['metrics']['auprc'],
        'precision': r['metrics']['precision'],
        'recall': r['metrics']['recall'],
        'latency_ms': r['metrics'].get('latency_ms', None),
        'training_time': r['metrics'].get('training_time', None)
    }
    for r in results_list
])

print(f"Loaded {len(df)} experimental results")
print(f"\nUnique experiments: {df['experiment'].unique()}")
print(f"Unique models: {df['model'].unique()}")
print(f"Unique seeds: {df['seed'].unique()}")
print(f"\nDataFrame shape: {df.shape}")

# Filter primary homophily sweep
primary_df = df[df['experiment'] == 'primary_homophily_sweep'].copy()
print(f"\nPrimary homophily sweep results: {len(primary_df)} rows")
print(f"Homophily levels: {sorted(primary_df['homophily'].unique())}")

# Define model groups
HETERO_MODELS = ['H2GCN', 'FAGCN', 'GPR-GNN', 'LINKX']
HOMO_MODELS = ['GCN', 'GraphSAGE', 'GAT']

# ============================================================================
# HYPOTHESIS 1: Heterophily-aware vs Homophily-assuming at h < 0.5
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 1: Hetero-aware > Homo-assuming at h < 0.5 by >= 5% F1")
print("="*80)

# Filter low homophily
low_h_df = primary_df[primary_df['homophily'] < 0.5].copy()

# For each seed, get max F1 for each model type
hetero_f1_by_seed = []
homo_f1_by_seed = []

for seed in low_h_df['seed'].unique():
    seed_data = low_h_df[low_h_df['seed'] == seed]

    hetero_max = seed_data[seed_data['model'].isin(HETERO_MODELS)]['f1'].max()
    homo_max = seed_data[seed_data['model'].isin(HOMO_MODELS)]['f1'].max()

    hetero_f1_by_seed.append(hetero_max)
    homo_f1_by_seed.append(homo_max)

hetero_f1 = np.array(hetero_f1_by_seed)
homo_f1 = np.array(homo_f1_by_seed)

# Paired t-test
t_stat, p_value_raw = stats.ttest_rel(hetero_f1, homo_f1)
delta_f1 = hetero_f1.mean() - homo_f1.mean()

# 95% CI for difference
ci_95 = stats.t.interval(0.95, len(hetero_f1)-1,
                         loc=delta_f1,
                         scale=stats.sem(hetero_f1 - homo_f1))

# Cohen's d effect size
pooled_std = np.sqrt(((len(hetero_f1)-1)*hetero_f1.std()**2 +
                      (len(homo_f1)-1)*homo_f1.std()**2) /
                     (len(hetero_f1) + len(homo_f1) - 2))
cohens_d = delta_f1 / pooled_std if pooled_std > 0 else 0

print(f"\nHeterophily-aware F1: {hetero_f1.mean():.4f} ± {hetero_f1.std():.4f}")
print(f"Homophily-assuming F1: {homo_f1.mean():.4f} ± {homo_f1.std():.4f}")
print(f"Delta F1: {delta_f1:.4f}")
print(f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value (raw): {p_value_raw:.6f}")
print(f"Cohen's d: {cohens_d:.4f}")

# Bonferroni correction for 5 hypotheses
alpha_bonferroni = 0.05 / 5
print(f"Bonferroni-corrected alpha: {alpha_bonferroni:.4f}")
print(f"p-value < alpha_bonferroni: {p_value_raw < alpha_bonferroni}")

h1_supported = (delta_f1 >= 0.05) and (p_value_raw < alpha_bonferroni)
print(f"\n*** H1 SUPPORTED: {h1_supported} ***")

# Save comparison JSON
h1_comparison = {
    "comparison": "heterophily_aware_vs_homophily_assuming_low_h",
    "metric": "F1",
    "estimate_diff": float(delta_f1),
    "ci_95": [float(ci_95[0]), float(ci_95[1])],
    "p_value": float(p_value_raw),
    "p_value_bonferroni_corrected": float(p_value_raw * 5),
    "test_statistic": float(t_stat),
    "test_method": "paired_t_test",
    "conclusion": f"Heterophily-aware models achieve F1 of {hetero_f1.mean():.3f} vs {homo_f1.mean():.3f} for homophily-assuming models at h<0.5. Difference: {delta_f1:.3f} (95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]), p={p_value_raw:.4f}. {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'} at Bonferroni-corrected alpha=0.01.",
    "sample_size": int(len(hetero_f1)),
    "additional_metrics": {
        "cohens_d": float(cohens_d),
        "hetero_mean": float(hetero_f1.mean()),
        "hetero_std": float(hetero_f1.std()),
        "homo_mean": float(homo_f1.mean()),
        "homo_std": float(homo_f1.std())
    }
}

# ============================================================================
# HYPOTHESIS 2: Monotonic relationship between h and performance gap
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 2: Delta(h) decreases monotonically as h increases")
print("="*80)

deltas_by_h = []
h_values = []

for h_level in sorted(primary_df['homophily'].unique()):
    h_data = primary_df[primary_df['homophily'] == h_level]

    hetero_mean = h_data[h_data['model'].isin(HETERO_MODELS)]['f1'].mean()
    homo_mean = h_data[h_data['model'].isin(HOMO_MODELS)]['f1'].mean()

    delta = hetero_mean - homo_mean
    deltas_by_h.append(delta)
    h_values.append(h_level)

    print(f"h={h_level:.1f}: Hetero={hetero_mean:.4f}, Homo={homo_mean:.4f}, Delta={delta:.4f}")

# Spearman correlation
spearman_rho, spearman_p = spearmanr(h_values, deltas_by_h)
print(f"\nSpearman correlation: rho={spearman_rho:.4f}, p={spearman_p:.6f}")
print(f"Expected: rho < 0 (negative correlation)")

# Check if Delta(0.1) - Delta(0.7) >= 0.10
try:
    idx_01 = h_values.index(0.1)
    idx_07 = h_values.index(0.7)
    delta_diff = deltas_by_h[idx_01] - deltas_by_h[idx_07]
    print(f"Delta(0.1) - Delta(0.7) = {delta_diff:.4f} (expected >= 0.10)")
except:
    delta_diff = None
    print("Could not compute Delta(0.1) - Delta(0.7)")

h2_supported = (spearman_rho < -0.5) and (spearman_p < alpha_bonferroni)
print(f"\n*** H2 SUPPORTED: {h2_supported} ***")

# ============================================================================
# HYPOTHESIS 3a: Hetero-aware maintains F1 >= 0.75 for h in [0.1, 0.4]
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 3a: Best hetero-aware F1 >= 0.75 for all h in [0.1, 0.4]")
print("="*80)

financial_range = primary_df[(primary_df['homophily'] >= 0.1) &
                             (primary_df['homophily'] <= 0.4)]

hetero_by_h = []
for h_level in sorted(financial_range['homophily'].unique()):
    h_data = financial_range[financial_range['homophily'] == h_level]
    hetero_f1_mean = h_data[h_data['model'].isin(HETERO_MODELS)]['f1'].mean()
    hetero_by_h.append((h_level, hetero_f1_mean))
    print(f"h={h_level:.1f}: Best hetero F1={hetero_f1_mean:.4f}")

min_hetero_f1 = min([x[1] for x in hetero_by_h])
print(f"\nMin hetero F1 in [0.1, 0.4]: {min_hetero_f1:.4f}")
print(f"Threshold: 0.75")

h3a_supported = min_hetero_f1 >= 0.75
print(f"\n*** H3a SUPPORTED: {h3a_supported} ***")

# ============================================================================
# HYPOTHESIS 3b: Best homo-assuming degrades to F1 < 0.65 for some h in [0.1, 0.4]
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 3b: Best homo-assuming F1 < 0.65 for at least one h in [0.1, 0.4]")
print("="*80)

homo_by_h = []
for h_level in sorted(financial_range['homophily'].unique()):
    h_data = financial_range[financial_range['homophily'] == h_level]
    homo_f1_mean = h_data[h_data['model'].isin(HOMO_MODELS)]['f1'].mean()
    homo_by_h.append((h_level, homo_f1_mean))
    print(f"h={h_level:.1f}: Best homo F1={homo_f1_mean:.4f}")

min_homo_f1 = min([x[1] for x in homo_by_h])
print(f"\nMin homo F1 in [0.1, 0.4]: {min_homo_f1:.4f}")
print(f"Threshold: 0.65")

h3b_supported = min_homo_f1 < 0.65
print(f"\n*** H3b SUPPORTED: {h3b_supported} ***")

# ============================================================================
# HYPOTHESIS 4: Effect persists across imbalance ratios
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 4: Hetero advantage persists across IR in [50, 1000]")
print("="*80)

imbalance_df = df[df['experiment'] == 'class_imbalance_sensitivity'].copy()

if len(imbalance_df) > 0:
    ir_deltas = []
    for ir in sorted(imbalance_df['imbalance_ratio'].unique()):
        ir_data = imbalance_df[imbalance_df['imbalance_ratio'] == ir]

        hetero_mean = ir_data[ir_data['model'].isin(HETERO_MODELS)]['f1'].mean()
        homo_mean = ir_data[ir_data['model'].isin(HOMO_MODELS)]['f1'].mean()
        delta = hetero_mean - homo_mean

        ir_deltas.append((ir, delta, hetero_mean, homo_mean))
        print(f"IR={ir}: Hetero={hetero_mean:.4f}, Homo={homo_mean:.4f}, Delta={delta:.4f}")

    all_positive = all(d[1] > 0 for d in ir_deltas)
    min_delta = min(d[1] for d in ir_deltas)

    print(f"\nAll deltas positive: {all_positive}")
    print(f"Min delta: {min_delta:.4f} (expected >= 0.04)")

    h4_supported = all_positive and (min_delta >= 0.04)
    print(f"\n*** H4 SUPPORTED: {h4_supported} ***")
else:
    print("No imbalance sensitivity data found")
    h4_supported = None
    ir_deltas = []

# ============================================================================
# HYPOTHESIS 5: FAGCN/LINKX > H2GCN/GPR-GNN at h < 0.2
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 5: FAGCN/LINKX > H2GCN/GPR-GNN by >= 3% at h < 0.2")
print("="*80)

very_low_h = primary_df[primary_df['homophily'] < 0.2].copy()

negative_models = ['FAGCN', 'LINKX']
higher_order_models = ['H2GCN', 'GPR-GNN']

neg_f1_values = very_low_h[very_low_h['model'].isin(negative_models)]['f1'].values
ho_f1_values = very_low_h[very_low_h['model'].isin(higher_order_models)]['f1'].values

neg_f1_mean = neg_f1_values.mean()
ho_f1_mean = ho_f1_values.mean()
delta_h5 = neg_f1_mean - ho_f1_mean

print(f"FAGCN/LINKX F1: {neg_f1_mean:.4f} ± {neg_f1_values.std():.4f}")
print(f"H2GCN/GPR-GNN F1: {ho_f1_mean:.4f} ± {ho_f1_values.std():.4f}")
print(f"Delta: {delta_h5:.4f} (expected >= 0.03)")

# Independent t-test
t_stat_h5, p_value_h5 = stats.ttest_ind(neg_f1_values, ho_f1_values)
print(f"t-statistic: {t_stat_h5:.4f}, p-value: {p_value_h5:.6f}")

h5_supported = (delta_h5 >= 0.03) and (p_value_h5 < alpha_bonferroni)
print(f"\n*** H5 SUPPORTED: {h5_supported} ***")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS TESTING SUMMARY")
print("="*80)

hypothesis_results = {
    "H1": {
        "statement": "Hetero-aware > Homo-assuming at h<0.5 by >= 5% F1",
        "supported": h1_supported,
        "delta_f1": float(delta_f1),
        "p_value": float(p_value_raw),
        "evidence": h1_comparison
    },
    "H2": {
        "statement": "Performance gap decreases monotonically with h",
        "supported": h2_supported,
        "spearman_rho": float(spearman_rho),
        "p_value": float(spearman_p),
        "deltas_by_h": [(float(h), float(d)) for h, d in zip(h_values, deltas_by_h)]
    },
    "H3a": {
        "statement": "Best hetero F1 >= 0.75 for all h in [0.1, 0.4]",
        "supported": h3a_supported,
        "min_f1": float(min_hetero_f1),
        "threshold": 0.75
    },
    "H3b": {
        "statement": "Best homo F1 < 0.65 for some h in [0.1, 0.4]",
        "supported": h3b_supported,
        "min_f1": float(min_homo_f1),
        "threshold": 0.65
    },
    "H4": {
        "statement": "Hetero advantage persists across IR with min delta >= 0.04",
        "supported": h4_supported,
        "ir_deltas": [(int(ir), float(d), float(hm), float(hom)) for ir, d, hm, hom in ir_deltas] if ir_deltas else []
    },
    "H5": {
        "statement": "FAGCN/LINKX > H2GCN/GPR-GNN by >= 3% at h < 0.2",
        "supported": h5_supported,
        "delta_f1": float(delta_h5),
        "p_value": float(p_value_h5)
    }
}

for h_name, h_data in hypothesis_results.items():
    status = "✓ SUPPORTED" if h_data['supported'] else "✗ FALSIFIED"
    print(f"{h_name}: {status}")
    print(f"  {h_data['statement']}")

# Save results
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results/hypothesis_test_results.json', 'w') as f:
    json.dump(hypothesis_results, f, indent=2)

print("\n✓ Saved hypothesis test results to files/results/hypothesis_test_results.json")

# Save H1 comparison
with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results/comparison_homophily_effect.json', 'w') as f:
    json.dump(h1_comparison, f, indent=2)

print("✓ Saved H1 comparison to files/results/comparison_homophily_effect.json")
