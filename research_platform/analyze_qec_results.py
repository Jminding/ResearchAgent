#!/usr/bin/env python3
"""
Comprehensive statistical analysis of 162 QEC experiment results.
Tests primary hypothesis: "RL achieves ≥20% improvement over MWPM baseline"
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import sys
from pathlib import Path

def load_results():
    """Load experiment results and metadata"""
    with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results/results_table.json', 'r') as f:
        results = json.load(f)

    with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/theory/experiment_plan.json', 'r') as f:
        plan = json.load(f)

    with open('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/research_notes/evidence_sheet.json', 'r') as f:
        evidence = json.load(f)

    return results, plan, evidence

def parse_results_to_dataframe(results):
    """Convert results JSON to pandas DataFrame for analysis"""
    records = []

    results_list = results['results']

    for result in results_list:
        if result.get('error') is not None:
            continue  # Skip failed experiments

        record = {
            'experiment_id': result['experiment_id'],
            'distance': result['distance'],
            'algorithm': result['algorithm'],
            'noise_model': result['noise_model'],
            'training_episodes': result['training_episodes'],
            'logical_error_rate': result['logical_error_rate'],
            'logical_error_rate_std': result['logical_error_rate_std'],
            'improvement_ratio': result.get('improvement_ratio', 0),
            'generalization_gap': result.get('generalization_gap', None),
            'wall_clock_time': result['wall_clock_time'],
            'seed': result['seed'],
            'physical_error_rate': result['additional_metrics'].get('physical_error_rate', None)
        }

        records.append(record)

    df = pd.DataFrame(records)
    return df

def compute_improvement_ratio_paired(df):
    """
    Compute improvement_ratio = (L_MWPM - L_RL) / L_MWPM for all configs
    Match RL and MWPM results by distance, noise_model, physical_error_rate, seed
    """
    # Separate RL and MWPM results
    df_rl = df[df['algorithm'].str.contains('RL|PPO|GNN|CNN', na=False)].copy()
    df_mwpm = df[df['algorithm'] == 'MWPM'].copy()

    print(f"  RL experiments: {len(df_rl)}")
    print(f"  MWPM experiments: {len(df_mwpm)}")

    # Merge on matching conditions
    merge_keys = ['distance', 'physical_error_rate', 'noise_model', 'seed']

    df_merged = pd.merge(
        df_rl, df_mwpm,
        on=merge_keys,
        suffixes=('_rl', '_mwpm'),
        how='inner'
    )

    print(f"  Matched pairs: {len(df_merged)}")

    if len(df_merged) == 0:
        return df_merged

    # Compute improvement ratio
    df_merged['improvement_ratio_computed'] = (
        (df_merged['logical_error_rate_mwpm'] - df_merged['logical_error_rate_rl']) /
        df_merged['logical_error_rate_mwpm']
    )

    df_merged['absolute_improvement'] = (
        df_merged['logical_error_rate_mwpm'] - df_merged['logical_error_rate_rl']
    )

    # Compute relative improvement
    df_merged['relative_improvement_percent'] = df_merged['improvement_ratio_computed'] * 100

    return df_merged

def paired_t_test_rl_vs_mwpm(df_merged, alpha=0.01):
    """
    Perform paired t-test: RL vs MWPM across seeds
    """
    if len(df_merged) == 0:
        return {'error': 'No matched pairs found'}

    results = {}

    # Overall paired t-test
    l_rl = df_merged['logical_error_rate_rl'].values
    l_mwpm = df_merged['logical_error_rate_mwpm'].values

    # Test if MWPM > RL (alternative: MWPM error rates are greater)
    t_stat, p_value = stats.ttest_rel(l_mwpm, l_rl, alternative='greater')

    # Effect size (Cohen's d for paired samples)
    diff = l_mwpm - l_rl
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    # 95% confidence interval on mean improvement ratio
    improvement_ratios = df_merged['improvement_ratio_computed'].values
    mean_improvement = np.mean(improvement_ratios)
    se_improvement = stats.sem(improvement_ratios)
    ci_95 = stats.t.interval(0.95, len(improvement_ratios)-1,
                             loc=mean_improvement, scale=se_improvement)

    # Bootstrap CI for robustness
    n_bootstrap = 1000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(improvement_ratios), len(improvement_ratios), replace=True)
        bootstrap_means.append(np.mean(improvement_ratios[sample_idx]))

    bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5])

    # Power analysis (post-hoc)
    # Effect size for power calculation
    from scipy.stats import power
    # Using simplified power calculation
    n_pairs = len(df_merged)

    results['overall'] = {
        'comparison': 'RL_vs_MWPM',
        'metric': 'logical_error_rate',
        'n_pairs': int(n_pairs),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'mean_improvement_ratio': float(mean_improvement),
        'mean_improvement_percent': float(mean_improvement * 100),
        'ci_95_improvement_ratio': [float(ci_95[0]), float(ci_95[1])],
        'ci_95_improvement_percent': [float(ci_95[0]*100), float(ci_95[1]*100)],
        'bootstrap_ci_95': [float(bootstrap_ci[0]), float(bootstrap_ci[1])],
        'mean_L_RL': float(np.mean(l_rl)),
        'mean_L_MWPM': float(np.mean(l_mwpm)),
        'median_L_RL': float(np.median(l_rl)),
        'median_L_MWPM': float(np.median(l_mwpm)),
        'std_L_RL': float(np.std(l_rl)),
        'std_L_MWPM': float(np.std(l_mwpm)),
        'significant': bool(p_value < alpha),
        'hypothesis_supported': bool((mean_improvement >= 0.20) and (p_value < alpha)),
        'test_method': 'paired_t_test',
        'confidence_level': 0.95,
        'significance_level': alpha
    }

    return results

def analyze_by_distance(df_merged, df_all):
    """
    Analyze logical error rates by code distance d={3,5,7,11,15}
    Fit exponential suppression model: L(d) = A * exp(-alpha * d)
    """
    results = {}

    for algorithm in ['RL', 'MWPM']:
        if algorithm == 'RL':
            df_algo = df_all[df_all['algorithm'].str.contains('RL|PPO|GNN', na=False)]
        else:
            df_algo = df_all[df_all['algorithm'] == algorithm]

        # Group by distance
        grouped = df_algo.groupby('distance')['logical_error_rate'].agg(['mean', 'std', 'count', 'median'])

        distances = grouped.index.values
        error_rates = grouped['mean'].values
        error_stds = grouped['std'].values

        # Fit exponential model: L(d) = A * exp(-alpha * d)
        try:
            def exp_model(d, A, alpha):
                return A * np.exp(-alpha * d)

            # Initial guess
            popt, pcov = curve_fit(exp_model, distances, error_rates,
                                   p0=[0.1, 0.1], maxfev=10000)

            A_fit, alpha_fit = popt
            perr = np.sqrt(np.diag(pcov))

            # Compute R-squared
            residuals = error_rates - exp_model(distances, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((error_rates - np.mean(error_rates))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Compute suppression factors Lambda = L_d / L_{d+2}
            suppression_factors = []
            for i in range(len(distances) - 1):
                lambda_d = error_rates[i] / error_rates[i+1]
                suppression_factors.append({
                    'd': int(distances[i]),
                    'd_next': int(distances[i+1]),
                    'lambda': float(lambda_d),
                    'L_d': float(error_rates[i]),
                    'L_d_next': float(error_rates[i+1])
                })

            # Extrapolate to d=21
            if 21 not in distances:
                L_21_pred = exp_model(21, A_fit, alpha_fit)

                # Confidence band using covariance matrix
                # Simple approximation
                L_21_std = np.sqrt(perr[0]**2 * np.exp(-2*alpha_fit*21) +
                                   (A_fit * 21 * np.exp(-alpha_fit*21) * perr[1])**2)

                extrapolation = {
                    'd': 21,
                    'L_pred': float(L_21_pred),
                    'L_std_approx': float(L_21_std),
                    'ci_95_lower': float(L_21_pred - 1.96*L_21_std),
                    'ci_95_upper': float(L_21_pred + 1.96*L_21_std)
                }
            else:
                extrapolation = None

            results[algorithm] = {
                'distances': distances.tolist(),
                'error_rates': error_rates.tolist(),
                'error_stds': error_stds.tolist(),
                'error_medians': grouped['median'].values.tolist(),
                'fit_A': float(A_fit),
                'fit_alpha': float(alpha_fit),
                'fit_A_stderr': float(perr[0]),
                'fit_alpha_stderr': float(perr[1]),
                'r_squared': float(r_squared),
                'suppression_factors': suppression_factors,
                'extrapolation_d21': extrapolation
            }

        except Exception as e:
            print(f"  Warning: Failed to fit exponential model for {algorithm}: {e}")
            results[algorithm] = {
                'distances': distances.tolist(),
                'error_rates': error_rates.tolist(),
                'error_stds': error_stds.tolist(),
                'fit_failed': True,
                'error': str(e)
            }

    # Compare suppression rates
    if 'RL' in results and 'MWPM' in results:
        if 'fit_alpha' in results['RL'] and 'fit_alpha' in results['MWPM']:
            alpha_improvement = results['RL']['fit_alpha'] - results['MWPM']['fit_alpha']

            # Test if difference is significant using error bars
            alpha_rl_se = results['RL']['fit_alpha_stderr']
            alpha_mwpm_se = results['MWPM']['fit_alpha_stderr']
            se_diff = np.sqrt(alpha_rl_se**2 + alpha_mwpm_se**2)
            z_score = alpha_improvement / se_diff
            p_value_diff = 2 * (1 - stats.norm.cdf(abs(z_score)))

            results['comparison'] = {
                'alpha_RL': float(results['RL']['fit_alpha']),
                'alpha_MWPM': float(results['MWPM']['fit_alpha']),
                'alpha_improvement': float(alpha_improvement),
                'alpha_improvement_se': float(se_diff),
                'z_score': float(z_score),
                'p_value': float(p_value_diff),
                'significant': bool(p_value_diff < 0.05),
                'better_suppression': 'RL' if results['RL']['fit_alpha'] > results['MWPM']['fit_alpha'] else 'MWPM',
                'interpretation': f"RL suppression rate is {'significantly' if p_value_diff < 0.05 else 'not significantly'} " +
                                f"{'better' if alpha_improvement > 0 else 'worse'} than MWPM"
            }

    return results

def architecture_comparison(df):
    """
    Compare architectures: GNN vs CNN vs MLP (if present)
    One-way ANOVA and post-hoc pairwise comparisons
    """
    # Extract architecture from algorithm name
    df['architecture'] = df['algorithm'].apply(lambda x:
        'GNN' if 'GNN' in str(x) else
        'CNN' if 'CNN' in str(x) else
        'MLP' if 'MLP' in str(x) else
        'MWPM' if x == 'MWPM' else
        'RL_generic')

    df_rl = df[df['architecture'] != 'MWPM'].copy()

    architectures = df_rl['architecture'].unique()
    architectures = [a for a in architectures if a != 'MWPM']

    if len(architectures) < 2:
        return {
            'error': 'Insufficient architectures for comparison',
            'available_architectures': architectures.tolist()
        }

    # Group by architecture
    groups = []
    group_names = []
    for arch in architectures:
        group_data = df_rl[df_rl['architecture'] == arch]['logical_error_rate'].values
        if len(group_data) > 0:
            groups.append(group_data)
            group_names.append(arch)

    if len(groups) < 2:
        return {'error': 'Insufficient data for architecture comparison'}

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    results = {
        'architectures': group_names,
        'anova': {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'test_method': 'one_way_anova'
        },
        'performance': {}
    }

    # Performance by architecture
    for arch in group_names:
        arch_data = df_rl[df_rl['architecture'] == arch]['logical_error_rate'].values
        results['performance'][arch] = {
            'mean': float(np.mean(arch_data)),
            'std': float(np.std(arch_data)),
            'median': float(np.median(arch_data)),
            'min': float(np.min(arch_data)),
            'max': float(np.max(arch_data)),
            'n': int(len(arch_data))
        }

    # Pairwise comparisons (Tukey HSD alternative: t-tests with Bonferroni correction)
    pairwise = []
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    for i, arch1 in enumerate(group_names):
        for j, arch2 in enumerate(group_names):
            if i < j:
                group1 = df_rl[df_rl['architecture'] == arch1]['logical_error_rate'].values
                group2 = df_rl[df_rl['architecture'] == arch2]['logical_error_rate'].values

                t_stat, p_val = stats.ttest_ind(group1, group2)

                # Cohen's d
                pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
                cohens_d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)

                pairwise.append({
                    'arch1': arch1,
                    'arch2': arch2,
                    'mean_diff': float(np.mean(group1) - np.mean(group2)),
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'p_value_bonferroni': float(min(p_val * n_comparisons, 1.0)),
                    'cohens_d': float(cohens_d),
                    'significant': bool(p_val < bonferroni_alpha),
                    'better': arch1 if np.mean(group1) < np.mean(group2) else arch2
                })

    results['pairwise_comparisons'] = pairwise
    results['bonferroni_alpha'] = float(bonferroni_alpha)

    # Identify best architecture
    best_arch = min(results['performance'].items(), key=lambda x: x[1]['mean'])
    results['best_architecture'] = {
        'name': best_arch[0],
        'mean_error_rate': float(best_arch[1]['mean']),
        'std_error_rate': float(best_arch[1]['std'])
    }

    return results

def noise_model_transfer_analysis(df_merged):
    """
    Analyze transfer performance across noise models
    """
    results = {}

    # Group by noise model
    noise_models = df_merged['noise_model'].unique()

    for noise in noise_models:
        df_noise = df_merged[df_merged['noise_model'] == noise]

        if len(df_noise) > 0:
            improvement_ratios = df_noise['improvement_ratio_computed'].values

            results[noise] = {
                'mean_improvement_ratio': float(np.mean(improvement_ratios)),
                'std_improvement_ratio': float(np.std(improvement_ratios)),
                'median_improvement_ratio': float(np.median(improvement_ratios)),
                'mean_improvement_percent': float(np.mean(improvement_ratios) * 100),
                'n_configs': int(len(df_noise))
            }

    # Compare phenomenological vs circuit_level
    if 'phenomenological' in results and 'circuit_level' in results:
        phenom_improvement = results['phenomenological']['mean_improvement_ratio']
        circuit_improvement = results['circuit_level']['mean_improvement_ratio']

        transfer_loss = (phenom_improvement - circuit_improvement) / phenom_improvement

        results['transfer_analysis'] = {
            'phenomenological_improvement': float(phenom_improvement),
            'circuit_level_improvement': float(circuit_improvement),
            'transfer_loss_ratio': float(transfer_loss),
            'transfer_loss_percent': float(transfer_loss * 100),
            'transfer_hypothesis_supported': bool(abs(transfer_loss) < 0.15),
            'interpretation': f"Transfer loss: {transfer_loss*100:.1f}% ({'within' if abs(transfer_loss) < 0.15 else 'exceeds'} 15% threshold)"
        }

    # Compare phenomenological vs biased
    if 'phenomenological' in results and 'biased' in results:
        phenom_improvement = results['phenomenological']['mean_improvement_ratio']
        biased_improvement = results['biased']['mean_improvement_ratio']

        transfer_loss = (phenom_improvement - biased_improvement) / phenom_improvement

        results['transfer_to_biased'] = {
            'phenomenological_improvement': float(phenom_improvement),
            'biased_improvement': float(biased_improvement),
            'transfer_loss_ratio': float(transfer_loss),
            'transfer_loss_percent': float(transfer_loss * 100),
            'transfer_hypothesis_supported': bool(abs(transfer_loss) < 0.15)
        }

    return results

def cross_distance_generalization(df):
    """
    Analyze cross-distance generalization gap
    """
    results = {}

    # Look for generalization_gap in data
    df_gen = df[df['generalization_gap'].notna()].copy()

    if len(df_gen) > 0:
        grouped = df_gen.groupby('distance')['generalization_gap'].agg(['mean', 'std', 'count'])

        results['by_distance'] = {
            'distances': grouped.index.tolist(),
            'mean_gap': grouped['mean'].tolist(),
            'std_gap': grouped['std'].tolist(),
            'count': grouped['count'].tolist()
        }

        # Overall statistics
        results['overall'] = {
            'mean_generalization_gap': float(df_gen['generalization_gap'].mean()),
            'std_generalization_gap': float(df_gen['generalization_gap'].std()),
            'median_generalization_gap': float(df_gen['generalization_gap'].median()),
            'max_gap': float(df_gen['generalization_gap'].max()),
            'hypothesis_supported': bool(df_gen['generalization_gap'].mean() < 0.15),
            'interpretation': f"Mean gap: {df_gen['generalization_gap'].mean()*100:.1f}% " +
                            f"({'within' if df_gen['generalization_gap'].mean() < 0.15 else 'exceeds'} 15% threshold)"
        }
    else:
        results['note'] = 'No cross-distance generalization data available in results'

    return results

def robustness_under_noise_variation(df_merged, df_all):
    """
    Analyze performance across physical error rates
    Fit curves L(p) and identify thresholds
    """
    results = {}

    error_rates = sorted(df_all['physical_error_rate'].unique())

    for algorithm in ['RL', 'MWPM']:
        if algorithm == 'RL':
            df_algo = df_all[df_all['algorithm'].str.contains('RL|PPO|GNN', na=False)]
        else:
            df_algo = df_all[df_all['algorithm'] == algorithm]

        performance = []
        for p in error_rates:
            df_p = df_algo[df_algo['physical_error_rate'] == p]
            if len(df_p) > 0:
                mean_L = df_p['logical_error_rate'].mean()
                std_L = df_p['logical_error_rate'].std()
                median_L = df_p['logical_error_rate'].median()

                performance.append({
                    'physical_error_rate': float(p),
                    'mean_logical_error_rate': float(mean_L),
                    'std_logical_error_rate': float(std_L),
                    'median_logical_error_rate': float(median_L)
                })

        results[algorithm] = performance

        # Fit logistic/sigmoid curve
        # L(p) = L_max / (1 + exp(-k*(p - p_mid)))
        if len(performance) >= 3:
            try:
                p_vals = np.array([x['physical_error_rate'] for x in performance])
                L_vals = np.array([x['mean_logical_error_rate'] for x in performance])

                def sigmoid(p, L_max, k, p_mid):
                    return L_max / (1 + np.exp(-k * (p - p_mid)))

                popt, pcov = curve_fit(sigmoid, p_vals, L_vals,
                                       p0=[np.max(L_vals), 100, 0.005],
                                       maxfev=10000)

                L_max_fit, k_fit, p_mid_fit = popt

                results[f'{algorithm}_fit'] = {
                    'L_max': float(L_max_fit),
                    'k': float(k_fit),
                    'p_mid': float(p_mid_fit),
                    'model': 'sigmoid',
                    'interpretation': f"Midpoint (inflection) at p={p_mid_fit:.5f}"
                }

            except Exception as e:
                print(f"  Warning: Failed to fit sigmoid for {algorithm}: {e}")

    return results

def identify_trends_and_anomalies(df):
    """
    Identify unexpected patterns, failures, or anomalies
    """
    results = {
        'total_configs': int(len(df)),
        'algorithms': df['algorithm'].value_counts().to_dict(),
        'noise_models': df['noise_model'].value_counts().to_dict(),
        'distances': df['distance'].value_counts().to_dict()
    }

    # Check for outliers using IQR method
    Q1 = df['logical_error_rate'].quantile(0.25)
    Q3 = df['logical_error_rate'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    outliers = df[df['logical_error_rate'] > outlier_threshold].copy()

    results['outliers'] = {
        'count': int(len(outliers)),
        'threshold': float(outlier_threshold),
        'Q1': float(Q1),
        'Q3': float(Q3),
        'IQR': float(IQR)
    }

    if len(outliers) > 0:
        results['outliers']['examples'] = outliers[
            ['experiment_id', 'distance', 'algorithm', 'physical_error_rate',
             'logical_error_rate']
        ].head(10).to_dict('records')

    # Check for very low error rates (suspiciously good)
    very_low = df[df['logical_error_rate'] < 0.0001]
    if len(very_low) > 0:
        results['suspiciously_low_errors'] = {
            'count': int(len(very_low)),
            'examples': very_low[
                ['experiment_id', 'distance', 'algorithm', 'logical_error_rate']
            ].head(5).to_dict('records')
        }

    # Performance summary statistics
    results['performance_summary'] = {
        'overall_mean': float(df['logical_error_rate'].mean()),
        'overall_std': float(df['logical_error_rate'].std()),
        'overall_median': float(df['logical_error_rate'].median()),
        'min': float(df['logical_error_rate'].min()),
        'max': float(df['logical_error_rate'].max())
    }

    # Check if results match expected ranges from experiment plan
    # Expected: [0.0008, 0.0015] for RL at d=15, p=0.005
    # Expected: [0.0012, 0.002] for MWPM at d=15, p=0.005

    df_d15_p005 = df[(df['distance'] == 15) &
                      (abs(df['physical_error_rate'] - 0.005) < 0.0001)]

    if len(df_d15_p005) > 0:
        results['d15_p005_comparison'] = {}

        df_rl_d15 = df_d15_p005[df_d15_p005['algorithm'].str.contains('RL|PPO|GNN', na=False)]
        df_mwpm_d15 = df_d15_p005[df_d15_p005['algorithm'] == 'MWPM']

        if len(df_rl_d15) > 0:
            mean_rl = df_rl_d15['logical_error_rate'].mean()
            results['d15_p005_comparison']['RL'] = {
                'mean': float(mean_rl),
                'expected_range': [0.0008, 0.0015],
                'within_expected': bool(0.0008 <= mean_rl <= 0.0015)
            }

        if len(df_mwpm_d15) > 0:
            mean_mwpm = df_mwpm_d15['logical_error_rate'].mean()
            results['d15_p005_comparison']['MWPM'] = {
                'mean': float(mean_mwpm),
                'expected_range': [0.0012, 0.002],
                'within_expected': bool(0.0012 <= mean_mwpm <= 0.002)
            }

    return results

def main():
    print("="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF QEC EXPERIMENTS")
    print("="*80)

    print("\n[1/8] Loading results...")
    results_data, plan, evidence = load_results()

    print(f"  Metadata: {results_data['metadata']}")

    print("\n[2/8] Parsing results to DataFrame...")
    df = parse_results_to_dataframe(results_data)

    print(f"  Total experiments: {len(df)}")
    print(f"  Algorithms: {df['algorithm'].unique()}")
    print(f"  Distances: {sorted(df['distance'].unique())}")
    print(f"  Noise models: {df['noise_model'].unique()}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")

    # Analysis outputs
    analysis_results = {
        'metadata': {
            'total_configs': int(len(df)),
            'analysis_date': '2025-12-28',
            'experiment_metadata': results_data['metadata']
        }
    }

    # 1. Test primary hypothesis
    print("\n[3/8] TESTING PRIMARY HYPOTHESIS: RL >= 20% improvement over MWPM")
    print("-"*80)
    df_merged = compute_improvement_ratio_paired(df)

    if len(df_merged) > 0:
        hypothesis_test = paired_t_test_rl_vs_mwpm(df_merged)
        analysis_results['primary_hypothesis'] = hypothesis_test

        print(f"  Mean improvement: {hypothesis_test['overall']['mean_improvement_percent']:.2f}%")
        print(f"  95% CI: [{hypothesis_test['overall']['ci_95_improvement_percent'][0]:.2f}%, "
              f"{hypothesis_test['overall']['ci_95_improvement_percent'][1]:.2f}%]")
        print(f"  Bootstrap CI: [{hypothesis_test['overall']['bootstrap_ci_95'][0]:.2f}, "
              f"{hypothesis_test['overall']['bootstrap_ci_95'][1]:.2f}]")
        print(f"  p-value: {hypothesis_test['overall']['p_value']:.6f} (alpha=0.01)")
        print(f"  Cohen's d: {hypothesis_test['overall']['cohens_d']:.3f}")
        print(f"  Sample size: {hypothesis_test['overall']['n_pairs']} pairs")
        print(f"  Significant: {hypothesis_test['overall']['significant']}")
        print(f"  HYPOTHESIS SUPPORTED: {hypothesis_test['overall']['hypothesis_supported']}")
    else:
        print("  ERROR: No matched RL-MWPM pairs found!")
        analysis_results['primary_hypothesis'] = {'error': 'No matched pairs'}

    # 2. Analyze by distance
    print("\n[4/8] ANALYZING BY DISTANCE: Exponential suppression model")
    print("-"*80)
    distance_analysis = analyze_by_distance(df_merged if len(df_merged) > 0 else df, df)
    analysis_results['distance_analysis'] = distance_analysis

    if 'RL' in distance_analysis and 'fit_alpha' in distance_analysis['RL']:
        print(f"  RL suppression rate (alpha): {distance_analysis['RL']['fit_alpha']:.4f} ± "
              f"{distance_analysis['RL']['fit_alpha_stderr']:.4f}")
        print(f"  R²: {distance_analysis['RL']['r_squared']:.4f}")
    if 'MWPM' in distance_analysis and 'fit_alpha' in distance_analysis['MWPM']:
        print(f"  MWPM suppression rate (alpha): {distance_analysis['MWPM']['fit_alpha']:.4f} ± "
              f"{distance_analysis['MWPM']['fit_alpha_stderr']:.4f}")
        print(f"  R²: {distance_analysis['MWPM']['r_squared']:.4f}")

    if 'comparison' in distance_analysis:
        comp = distance_analysis['comparison']
        print(f"  Comparison: {comp['interpretation']}")
        print(f"  p-value: {comp['p_value']:.4f}")

    # 3. Architecture comparison
    print("\n[5/8] ARCHITECTURE COMPARISON: GNN vs CNN vs MLP")
    print("-"*80)
    arch_analysis = architecture_comparison(df)
    analysis_results['architecture_comparison'] = arch_analysis

    if 'anova' in arch_analysis:
        print(f"  Architectures tested: {arch_analysis['architectures']}")
        print(f"  ANOVA F-statistic: {arch_analysis['anova']['f_statistic']:.3f}")
        print(f"  p-value: {arch_analysis['anova']['p_value']:.6f}")
        print(f"  Significant: {arch_analysis['anova']['significant']}")

        if 'best_architecture' in arch_analysis:
            print(f"  Best architecture: {arch_analysis['best_architecture']['name']}")
            print(f"  Mean error rate: {arch_analysis['best_architecture']['mean_error_rate']:.6f}")

        if 'pairwise_comparisons' in arch_analysis:
            print(f"  Pairwise comparisons:")
            for pw in arch_analysis['pairwise_comparisons']:
                print(f"    {pw['arch1']} vs {pw['arch2']}: "
                      f"p={pw['p_value']:.4f}, d={pw['cohens_d']:.3f}, "
                      f"better={pw['better']}")

    # 4. Noise model transfer
    print("\n[6/8] NOISE MODEL TRANSFER ANALYSIS")
    print("-"*80)
    if len(df_merged) > 0:
        transfer_analysis = noise_model_transfer_analysis(df_merged)
        analysis_results['noise_transfer'] = transfer_analysis

        for noise_model, metrics in transfer_analysis.items():
            if noise_model not in ['transfer_analysis', 'transfer_to_biased']:
                print(f"  {noise_model}: {metrics['mean_improvement_percent']:.2f}% improvement")

        if 'transfer_analysis' in transfer_analysis:
            ta = transfer_analysis['transfer_analysis']
            print(f"  {ta['interpretation']}")

    # 5. Cross-distance generalization
    print("\n[7/8] CROSS-DISTANCE GENERALIZATION")
    print("-"*80)
    generalization_analysis = cross_distance_generalization(df)
    analysis_results['generalization'] = generalization_analysis

    if 'overall' in generalization_analysis:
        print(f"  {generalization_analysis['overall']['interpretation']}")
    else:
        print(f"  {generalization_analysis.get('note', 'No data available')}")

    # 6. Robustness under noise variation
    print("\n[8/8] ROBUSTNESS UNDER NOISE VARIATION")
    print("-"*80)
    noise_robustness = robustness_under_noise_variation(df_merged if len(df_merged) > 0 else df, df)
    analysis_results['noise_robustness'] = noise_robustness

    for algo in ['RL', 'MWPM']:
        if algo in noise_robustness:
            print(f"  {algo}: {len(noise_robustness[algo])} error rates tested")

    # 7. Trend identification
    print("\n[9/9] IDENTIFYING TRENDS AND ANOMALIES")
    print("-"*80)
    trends = identify_trends_and_anomalies(df)
    analysis_results['trends'] = trends

    print(f"  Total configs: {trends['total_configs']}")
    print(f"  Outliers detected: {trends['outliers']['count']}")
    if 'd15_p005_comparison' in trends:
        print(f"  d=15, p=0.005 comparison:")
        for algo, data in trends['d15_p005_comparison'].items():
            print(f"    {algo}: mean={data['mean']:.6f}, "
                  f"expected={data['expected_range']}, "
                  f"within={data['within_expected']}")

    # Save comprehensive analysis
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = Path('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results')

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print("✓ Saved: files/results/analysis_summary.json")

    # Save RL vs MWPM comparison
    if 'primary_hypothesis' in analysis_results and 'overall' in analysis_results['primary_hypothesis']:
        comparison_data = analysis_results['primary_hypothesis']['overall']

        with open(output_dir / 'comparison_rl_vs_mwpm.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print("✓ Saved: files/results/comparison_rl_vs_mwpm.json")

    # Save architecture ranking
    if 'architecture_comparison' in analysis_results and 'performance' in analysis_results['architecture_comparison']:
        arch_ranking = {
            'ranking': sorted(
                [(k, v) for k, v in analysis_results['architecture_comparison']['performance'].items()],
                key=lambda x: x[1]['mean']
            ),
            'anova': analysis_results['architecture_comparison'].get('anova', {}),
            'pairwise': analysis_results['architecture_comparison'].get('pairwise_comparisons', []),
            'best_architecture': analysis_results['architecture_comparison'].get('best_architecture', {})
        }

        with open(output_dir / 'architecture_ranking.json', 'w') as f:
            json.dump(arch_ranking, f, indent=2)
        print("✓ Saved: files/results/architecture_ranking.json")

    # Save generalization curves
    if 'distance_analysis' in analysis_results:
        with open(output_dir / 'generalization_curves.json', 'w') as f:
            json.dump(analysis_results['distance_analysis'], f, indent=2)
        print("✓ Saved: files/results/generalization_curves.json")

    # Check if follow-up needed
    print("\n" + "="*80)
    print("HYPOTHESIS EVALUATION")
    print("="*80)

    if 'primary_hypothesis' in analysis_results and 'overall' in analysis_results['primary_hypothesis']:
        hypothesis_supported = analysis_results['primary_hypothesis']['overall']['hypothesis_supported']
        mean_improvement = analysis_results['primary_hypothesis']['overall']['mean_improvement_percent']
        p_value = analysis_results['primary_hypothesis']['overall']['p_value']

        if not hypothesis_supported:
            print(f"\n⚠ PRIMARY HYPOTHESIS NOT SUPPORTED")
            print(f"  Observed improvement: {mean_improvement:.2f}% (target: 20%)")
            print(f"  p-value: {p_value:.6f} (alpha: 0.01)")
            print(f"\n  Creating follow-up plan...")

            followup_plan = {
                'trigger': f"RL improvement {mean_improvement:.1f}% < 20% threshold (p={p_value:.6f})",
                'primary_hypothesis_failed': True,
                'observed_improvement_percent': float(mean_improvement),
                'target_improvement_percent': 20.0,
                'p_value': float(p_value),
                'hypotheses': [
                    {
                        'hypothesis': 'Insufficient training episodes for convergence at d>=15',
                        'diagnostic_experiment': 'Extend training from 200 to 1000 steps for d=15',
                        'expected_outcome': 'If correct, extended training improves L_RL by >=10%',
                        'priority': 1,
                        'rationale': 'Training was limited to 200 steps in this simulation; may need more episodes'
                    },
                    {
                        'hypothesis': 'Reward function not properly shaped for logical error minimization',
                        'diagnostic_experiment': 'Test dense reward with syndrome penalty vs pure logical error reward',
                        'expected_outcome': 'If correct, shaped reward converges faster and achieves lower L_RL',
                        'priority': 1,
                        'rationale': 'Current reward may not provide sufficient learning signal'
                    },
                    {
                        'hypothesis': 'GNN architecture insufficient for capturing long-range error correlations',
                        'diagnostic_experiment': 'Test deeper GNN (8-12 layers) or Transformer with attention',
                        'expected_outcome': 'If correct, deeper architecture improves performance by >=15%',
                        'priority': 2,
                        'rationale': 'Surface code errors may have long-range dependencies requiring deeper models'
                    }
                ],
                'selected_followup': None,
                'mode': 'discovery',
                'next_steps': [
                    'Select highest priority hypothesis',
                    'Design minimal diagnostic experiment',
                    'Run follow-up with same statistical rigor',
                    'Iterate until hypothesis supported or definitively falsified'
                ]
            }

            with open(output_dir / 'followup_plan.json', 'w') as f:
                json.dump(followup_plan, f, indent=2)
            print("✓ Saved: files/results/followup_plan.json")

        else:
            print(f"\n✓ PRIMARY HYPOTHESIS SUPPORTED")
            print(f"  Observed improvement: {mean_improvement:.2f}% >= 20% target")
            print(f"  p-value: {p_value:.6f} < 0.01")
            print(f"  Conclusion: RL decoder achieves statistically significant improvement over MWPM")
            print(f"\n  No follow-up experiments needed.")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - files/results/analysis_summary.json")
    print("  - files/results/comparison_rl_vs_mwpm.json")
    print("  - files/results/architecture_ranking.json")
    print("  - files/results/generalization_curves.json")
    if not hypothesis_supported:
        print("  - files/results/followup_plan.json")

if __name__ == '__main__':
    main()
