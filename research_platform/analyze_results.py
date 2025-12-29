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

# Add research_agent to path for statistics module
sys.path.append('/Users/jminding/Desktop/Code/Research Agent/research_platform')

def load_results():
    """Load experiment results and metadata"""
    with open('files/results/results_table.json', 'r') as f:
        results = json.load(f)

    with open('files/theory/experiment_plan.json', 'r') as f:
        plan = json.load(f)

    with open('files/research_notes/evidence_sheet.json', 'r') as f:
        evidence = json.load(f)

    return results, plan, evidence

def parse_results_to_dataframe(results):
    """Convert results JSON to pandas DataFrame for analysis"""
    records = []

    if isinstance(results, dict) and 'results' in results:
        results_list = results['results']
    elif isinstance(results, list):
        results_list = results
    else:
        results_list = [results]

    for result in results_list:
        record = {}

        # Extract configuration
        if 'config' in result:
            config = result['config']
            record['code_distance'] = config.get('code_distance', None)
            record['decoder_type'] = config.get('decoder_type', None)
            record['physical_error_rate'] = config.get('physical_error_rate', None)
            record['noise_model'] = config.get('noise_model', None)
            record['rl_algorithm'] = config.get('rl_algorithm', None)
            record['network_architecture'] = config.get('network_architecture', None)
            record['training_episodes'] = config.get('training_episodes', None)
            record['seed'] = config.get('seed', None)

        # Extract metrics
        if 'metrics' in result:
            metrics = result['metrics']
            record['logical_error_rate'] = metrics.get('logical_error_rate', None)
            record['inference_time_ms'] = metrics.get('inference_time_ms', None)
            record['training_time_hours'] = metrics.get('training_time_hours', None)

        # Status
        record['status'] = result.get('status', 'unknown')

        records.append(record)

    df = pd.DataFrame(records)
    return df

def compute_improvement_ratio(df):
    """
    Compute improvement_ratio = (L_MWPM - L_RL) / L_MWPM for all configs
    """
    # Filter successful runs only
    df_success = df[df['status'] == 'completed'].copy()

    # Separate RL and MWPM results
    df_rl = df_success[df_success['decoder_type'] == 'RL'].copy()
    df_mwpm = df_success[df_success['decoder_type'] == 'MWPM'].copy()

    # Merge on matching conditions
    merge_keys = ['code_distance', 'physical_error_rate', 'noise_model', 'seed']

    df_merged = pd.merge(
        df_rl, df_mwpm,
        on=merge_keys,
        suffixes=('_rl', '_mwpm')
    )

    # Compute improvement ratio
    df_merged['improvement_ratio'] = (
        (df_merged['logical_error_rate_mwpm'] - df_merged['logical_error_rate_rl']) /
        df_merged['logical_error_rate_mwpm']
    )

    df_merged['absolute_improvement'] = (
        df_merged['logical_error_rate_mwpm'] - df_merged['logical_error_rate_rl']
    )

    return df_merged

def paired_t_test_rl_vs_mwpm(df_merged, alpha=0.01):
    """
    Perform paired t-test: RL vs MWPM across 10 seeds
    """
    results = {}

    # Overall paired t-test
    l_rl = df_merged['logical_error_rate_rl'].values
    l_mwpm = df_merged['logical_error_rate_mwpm'].values

    t_stat, p_value = stats.ttest_rel(l_mwpm, l_rl, alternative='greater')

    # Effect size (Cohen's d for paired samples)
    diff = l_mwpm - l_rl
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    # 95% confidence interval on mean improvement ratio
    improvement_ratios = df_merged['improvement_ratio'].values
    mean_improvement = np.mean(improvement_ratios)
    se_improvement = stats.sem(improvement_ratios)
    ci_95 = stats.t.interval(0.95, len(improvement_ratios)-1,
                             loc=mean_improvement, scale=se_improvement)

    results['overall'] = {
        'comparison': 'RL_vs_MWPM',
        'metric': 'logical_error_rate',
        'n_pairs': len(df_merged),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'mean_improvement_ratio': float(mean_improvement),
        'ci_95_improvement_ratio': [float(ci_95[0]), float(ci_95[1])],
        'mean_L_RL': float(np.mean(l_rl)),
        'mean_L_MWPM': float(np.mean(l_mwpm)),
        'std_L_RL': float(np.std(l_rl)),
        'std_L_MWPM': float(np.std(l_mwpm)),
        'significant': p_value < alpha,
        'hypothesis_supported': (mean_improvement >= 0.20) and (p_value < alpha)
    }

    return results

def analyze_by_distance(df_merged):
    """
    Analyze logical error rates by code distance d={3,5,7,11,15}
    Fit exponential suppression model: L(d) = A * exp(-alpha * d)
    """
    results = {}

    for decoder_type in ['RL', 'MWPM']:
        col_name = f'logical_error_rate_{decoder_type.lower()}'

        # Group by distance
        grouped = df_merged.groupby('code_distance')[col_name].agg(['mean', 'std', 'count'])

        distances = grouped.index.values
        error_rates = grouped['mean'].values

        # Fit exponential model
        try:
            def exp_model(d, A, alpha):
                return A * np.exp(-alpha * d)

            popt, pcov = curve_fit(exp_model, distances, error_rates,
                                   p0=[0.1, 0.1], maxfev=10000)

            A_fit, alpha_fit = popt
            perr = np.sqrt(np.diag(pcov))

            # Compute suppression factors Lambda = L_d / L_{d+2}
            suppression_factors = []
            for i in range(len(distances) - 1):
                if i+1 < len(distances):
                    lambda_d = error_rates[i] / error_rates[i+1]
                    suppression_factors.append({
                        'd': int(distances[i]),
                        'd_next': int(distances[i+1]),
                        'lambda': float(lambda_d)
                    })

            results[decoder_type] = {
                'distances': distances.tolist(),
                'error_rates': error_rates.tolist(),
                'error_stds': grouped['std'].values.tolist(),
                'fit_A': float(A_fit),
                'fit_alpha': float(alpha_fit),
                'fit_A_stderr': float(perr[0]),
                'fit_alpha_stderr': float(perr[1]),
                'suppression_factors': suppression_factors
            }
        except Exception as e:
            print(f"Warning: Failed to fit exponential model for {decoder_type}: {e}")
            results[decoder_type] = {
                'distances': distances.tolist(),
                'error_rates': error_rates.tolist(),
                'error_stds': grouped['std'].values.tolist(),
                'fit_failed': True
            }

    # Compare suppression rates
    if 'RL' in results and 'MWPM' in results:
        if 'fit_alpha' in results['RL'] and 'fit_alpha' in results['MWPM']:
            results['comparison'] = {
                'alpha_RL': results['RL']['fit_alpha'],
                'alpha_MWPM': results['MWPM']['fit_alpha'],
                'alpha_improvement': results['RL']['fit_alpha'] - results['MWPM']['fit_alpha'],
                'better_suppression': 'RL' if results['RL']['fit_alpha'] > results['MWPM']['fit_alpha'] else 'MWPM'
            }

    return results

def architecture_comparison(df):
    """
    Compare architectures: GNN vs CNN vs Transformer
    One-way ANOVA and post-hoc pairwise comparisons
    """
    df_success = df[(df['status'] == 'completed') & (df['decoder_type'] == 'RL')].copy()

    # Filter to architecture comparison
    architectures = df_success['network_architecture'].unique()
    architectures = [a for a in architectures if a is not None and a != '']

    if len(architectures) < 2:
        return {'error': 'Insufficient architectures for comparison'}

    # Group by architecture
    groups = []
    for arch in architectures:
        group_data = df_success[df_success['network_architecture'] == arch]['logical_error_rate'].values
        groups.append(group_data)

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    results = {
        'architectures': architectures.tolist(),
        'anova': {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        },
        'performance': {}
    }

    # Performance by architecture
    for arch in architectures:
        arch_data = df_success[df_success['network_architecture'] == arch]['logical_error_rate'].values
        results['performance'][arch] = {
            'mean': float(np.mean(arch_data)),
            'std': float(np.std(arch_data)),
            'median': float(np.median(arch_data)),
            'n': int(len(arch_data))
        }

    # Pairwise comparisons
    pairwise = []
    for i, arch1 in enumerate(architectures):
        for j, arch2 in enumerate(architectures):
            if i < j:
                group1 = df_success[df_success['network_architecture'] == arch1]['logical_error_rate'].values
                group2 = df_success[df_success['network_architecture'] == arch2]['logical_error_rate'].values

                t_stat, p_val = stats.ttest_ind(group1, group2)

                # Cohen's d
                cohens_d = (np.mean(group1) - np.mean(group2)) / np.sqrt(
                    (np.std(group1)**2 + np.std(group2)**2) / 2
                )

                pairwise.append({
                    'arch1': arch1,
                    'arch2': arch2,
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'cohens_d': float(cohens_d),
                    'significant': p_val < 0.05
                })

    results['pairwise_comparisons'] = pairwise

    # Identify best architecture
    best_arch = min(results['performance'].items(), key=lambda x: x[1]['mean'])
    results['best_architecture'] = {
        'name': best_arch[0],
        'mean_error_rate': best_arch[1]['mean']
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

        mean_improvement = df_noise['improvement_ratio'].mean()

        results[noise] = {
            'mean_improvement_ratio': float(mean_improvement),
            'std_improvement_ratio': float(df_noise['improvement_ratio'].std()),
            'n_configs': int(len(df_noise))
        }

    # Compare phenomenological vs circuit_level
    if 'phenomenological' in results and 'circuit_level' in results:
        transfer_loss = 1.0 - (results['circuit_level']['mean_improvement_ratio'] /
                               results['phenomenological']['mean_improvement_ratio'])

        results['transfer_analysis'] = {
            'phenomenological_to_circuit_level_loss': float(transfer_loss),
            'transfer_hypothesis_supported': abs(transfer_loss) < 0.15
        }

    return results

def cross_distance_generalization(df):
    """
    Analyze cross-distance generalization
    """
    df_success = df[(df['status'] == 'completed') & (df['decoder_type'] == 'RL')].copy()

    # This requires train_distance and test_distance info
    # For now, compute generalization within same distance across seeds

    results = {
        'note': 'Cross-distance generalization requires explicit train/test distance labels in data'
    }

    return results

def robustness_under_noise_variation(df_merged):
    """
    Analyze performance across physical error rates
    """
    results = {}

    error_rates = sorted(df_merged['physical_error_rate'].unique())

    for decoder_type in ['RL', 'MWPM']:
        col_name = f'logical_error_rate_{decoder_type.lower()}'

        performance = []
        for p in error_rates:
            df_p = df_merged[df_merged['physical_error_rate'] == p]
            mean_L = df_p[col_name].mean()
            std_L = df_p[col_name].std()

            performance.append({
                'physical_error_rate': float(p),
                'mean_logical_error_rate': float(mean_L),
                'std_logical_error_rate': float(std_L)
            })

        results[decoder_type] = performance

    return results

def identify_trends_and_anomalies(df):
    """
    Identify unexpected patterns, failures, or anomalies
    """
    results = {
        'total_configs': len(df),
        'completed': int((df['status'] == 'completed').sum()),
        'failed': int((df['status'] == 'failed').sum()),
        'pending': int((df['status'] == 'pending').sum())
    }

    # Check for outliers
    df_success = df[df['status'] == 'completed']

    if len(df_success) > 0:
        # Identify configs with anomalously high error rates
        q95 = df_success['logical_error_rate'].quantile(0.95)
        outliers = df_success[df_success['logical_error_rate'] > q95]

        results['outliers'] = {
            'count': int(len(outliers)),
            'threshold': float(q95),
            'configs': outliers[['code_distance', 'decoder_type', 'physical_error_rate',
                                'logical_error_rate']].to_dict('records')
        }

    return results

def main():
    print("Loading results...")
    results_data, plan, evidence = load_results()

    print("Parsing results to DataFrame...")
    df = parse_results_to_dataframe(results_data)

    print(f"Loaded {len(df)} experiment configurations")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Decoder types: {df['decoder_type'].unique()}")
    print(f"Status counts:\n{df['status'].value_counts()}")

    # Filter to completed runs
    df_success = df[df['status'] == 'completed']
    print(f"\nCompleted runs: {len(df_success)}")

    # Analysis outputs
    analysis_results = {
        'metadata': {
            'total_configs': int(len(df)),
            'completed_configs': int(len(df_success)),
            'analysis_date': '2025-12-28'
        }
    }

    # 1. Test primary hypothesis
    print("\n=== Testing Primary Hypothesis ===")
    df_merged = compute_improvement_ratio(df)
    print(f"Matched RL-MWPM pairs: {len(df_merged)}")

    if len(df_merged) > 0:
        hypothesis_test = paired_t_test_rl_vs_mwpm(df_merged)
        analysis_results['primary_hypothesis'] = hypothesis_test

        print(f"Mean improvement ratio: {hypothesis_test['overall']['mean_improvement_ratio']:.3f}")
        print(f"95% CI: [{hypothesis_test['overall']['ci_95_improvement_ratio'][0]:.3f}, "
              f"{hypothesis_test['overall']['ci_95_improvement_ratio'][1]:.3f}]")
        print(f"p-value: {hypothesis_test['overall']['p_value']:.6f}")
        print(f"Cohen's d: {hypothesis_test['overall']['cohens_d']:.3f}")
        print(f"Hypothesis supported: {hypothesis_test['overall']['hypothesis_supported']}")

    # 2. Analyze by distance
    print("\n=== Analyzing by Distance ===")
    if len(df_merged) > 0:
        distance_analysis = analyze_by_distance(df_merged)
        analysis_results['distance_analysis'] = distance_analysis

        if 'RL' in distance_analysis and 'fit_alpha' in distance_analysis['RL']:
            print(f"RL suppression rate (alpha): {distance_analysis['RL']['fit_alpha']:.4f} ± "
                  f"{distance_analysis['RL']['fit_alpha_stderr']:.4f}")
        if 'MWPM' in distance_analysis and 'fit_alpha' in distance_analysis['MWPM']:
            print(f"MWPM suppression rate (alpha): {distance_analysis['MWPM']['fit_alpha']:.4f} ± "
                  f"{distance_analysis['MWPM']['fit_alpha_stderr']:.4f}")

    # 3. Architecture comparison
    print("\n=== Architecture Comparison ===")
    arch_analysis = architecture_comparison(df)
    analysis_results['architecture_comparison'] = arch_analysis

    if 'anova' in arch_analysis:
        print(f"ANOVA F-statistic: {arch_analysis['anova']['f_statistic']:.3f}, "
              f"p-value: {arch_analysis['anova']['p_value']:.6f}")
        if 'best_architecture' in arch_analysis:
            print(f"Best architecture: {arch_analysis['best_architecture']['name']}")

    # 4. Noise model transfer
    print("\n=== Noise Model Transfer Analysis ===")
    if len(df_merged) > 0:
        transfer_analysis = noise_model_transfer_analysis(df_merged)
        analysis_results['noise_transfer'] = transfer_analysis
        print(f"Transfer analysis: {json.dumps(transfer_analysis, indent=2)}")

    # 5. Cross-distance generalization
    print("\n=== Cross-Distance Generalization ===")
    generalization_analysis = cross_distance_generalization(df)
    analysis_results['generalization'] = generalization_analysis

    # 6. Robustness under noise variation
    print("\n=== Robustness Under Noise Variation ===")
    if len(df_merged) > 0:
        noise_robustness = robustness_under_noise_variation(df_merged)
        analysis_results['noise_robustness'] = noise_robustness

    # 7. Trend identification
    print("\n=== Identifying Trends and Anomalies ===")
    trends = identify_trends_and_anomalies(df)
    analysis_results['trends'] = trends
    print(f"Trends: {json.dumps(trends, indent=2)}")

    # Save comprehensive analysis
    print("\n=== Saving Results ===")

    with open('files/results/analysis_summary.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print("Saved: files/results/analysis_summary.json")

    # Save RL vs MWPM comparison
    if 'primary_hypothesis' in analysis_results:
        comparison_data = {
            'comparison': 'RL_vs_MWPM',
            'metric': 'logical_error_rate',
            **analysis_results['primary_hypothesis']['overall']
        }

        with open('files/results/comparison_rl_vs_mwpm.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print("Saved: files/results/comparison_rl_vs_mwpm.json")

    # Save architecture ranking
    if 'architecture_comparison' in analysis_results and 'performance' in analysis_results['architecture_comparison']:
        arch_ranking = {
            'ranking': sorted(
                analysis_results['architecture_comparison']['performance'].items(),
                key=lambda x: x[1]['mean']
            ),
            'anova': analysis_results['architecture_comparison'].get('anova', {}),
            'pairwise': analysis_results['architecture_comparison'].get('pairwise_comparisons', [])
        }

        with open('files/results/architecture_ranking.json', 'w') as f:
            json.dump(arch_ranking, f, indent=2)
        print("Saved: files/results/architecture_ranking.json")

    # Save generalization curves
    if 'distance_analysis' in analysis_results:
        with open('files/results/generalization_curves.json', 'w') as f:
            json.dump(analysis_results['distance_analysis'], f, indent=2)
        print("Saved: files/results/generalization_curves.json")

    # Check if follow-up needed
    if 'primary_hypothesis' in analysis_results:
        hypothesis_supported = analysis_results['primary_hypothesis']['overall']['hypothesis_supported']

        if not hypothesis_supported:
            print("\n=== Primary Hypothesis NOT Supported - Creating Follow-up Plan ===")

            mean_improvement = analysis_results['primary_hypothesis']['overall']['mean_improvement_ratio']
            p_value = analysis_results['primary_hypothesis']['overall']['p_value']

            followup_plan = {
                'trigger': f"RL improvement {mean_improvement*100:.1f}% < 20% threshold (p={p_value:.4f})",
                'hypotheses': [
                    {
                        'hypothesis': 'Insufficient training episodes for convergence at d>=15',
                        'diagnostic_experiment': 'Extend training from 1e7 to 5e7 episodes for d=15',
                        'expected_outcome': 'If correct, extended training improves L_RL by >=10%',
                        'priority': 1
                    },
                    {
                        'hypothesis': 'Reward function not properly shaped for logical error minimization',
                        'diagnostic_experiment': 'Test dense reward with syndrome penalty vs pure logical error reward',
                        'expected_outcome': 'If correct, shaped reward converges faster and achieves lower L_RL',
                        'priority': 1
                    },
                    {
                        'hypothesis': 'GNN architecture insufficient for capturing long-range error correlations',
                        'diagnostic_experiment': 'Test deeper GNN (8-12 layers) or Transformer with attention',
                        'expected_outcome': 'If correct, deeper architecture improves performance by >=15%',
                        'priority': 2
                    }
                ],
                'selected_followup': None,
                'mode': 'discovery'
            }

            with open('files/results/followup_plan.json', 'w') as f:
                json.dump(followup_plan, f, indent=2)
            print("Saved: files/results/followup_plan.json")
        else:
            print("\n=== Primary Hypothesis SUPPORTED ===")
            print("No follow-up experiments needed.")

    print("\n=== Analysis Complete ===")

if __name__ == '__main__':
    main()
