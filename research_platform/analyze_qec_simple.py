#!/usr/bin/env python3
"""
Comprehensive statistical analysis of 162 QEC experiment results.
Tests primary hypothesis: "RL achieves ≥20% improvement over MWPM baseline"
Uses only standard library for maximum compatibility.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

def mean(data):
    """Calculate mean"""
    if len(data) == 0:
        return 0.0
    return sum(data) / len(data)

def variance(data, ddof=1):
    """Calculate variance"""
    if len(data) <= ddof:
        return 0.0
    m = mean(data)
    return sum((x - m)**2 for x in data) / (len(data) - ddof)

def std(data, ddof=1):
    """Calculate standard deviation"""
    return math.sqrt(variance(data, ddof))

def median(data):
    """Calculate median"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n % 2 == 0:
        return (sorted_data[n//2-1] + sorted_data[n//2]) / 2
    return sorted_data[n//2]

def percentile(data, p):
    """Calculate percentile"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1

def t_test_paired(x, y):
    """Paired t-test: test if mean(x) > mean(y)"""
    if len(x) != len(y) or len(x) == 0:
        return None, None

    diff = [x[i] - y[i] for i in range(len(x))]
    n = len(diff)
    mean_diff = mean(diff)
    std_diff = std(diff, ddof=1)

    if std_diff == 0:
        return None, None

    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Approximate p-value using normal distribution for large n
    # For small n, this is an approximation
    df = n - 1
    # Simple approximation of p-value
    if n > 30:
        # Use normal approximation
        import math
        p_value = 0.5 * math.erfc(t_stat / math.sqrt(2))
    else:
        # For small n, use rough approximation
        p_value = 0.01 if abs(t_stat) > 2.5 else 0.05

    return t_stat, p_value

def cohens_d_paired(x, y):
    """Cohen's d for paired samples"""
    diff = [x[i] - y[i] for i in range(len(x))]
    return mean(diff) / (std(diff) + 1e-10)

def bootstrap_ci(data, n_bootstrap=1000, alpha=0.95):
    """Bootstrap confidence interval"""
    import random
    random.seed(42)

    bootstrap_means = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n-1)] for _ in range(n)]
        bootstrap_means.append(mean(sample))

    lower = percentile(bootstrap_means, (1 - alpha) / 2 * 100)
    upper = percentile(bootstrap_means, (1 + alpha) / 2 * 100)

    return lower, upper

def load_results():
    """Load experiment results and metadata"""
    base_path = Path('/Users/jminding/Desktop/Code/Research Agent/research_platform')

    with open(base_path / 'files/results/results_table.json', 'r') as f:
        results = json.load(f)

    with open(base_path / 'files/theory/experiment_plan.json', 'r') as f:
        plan = json.load(f)

    with open(base_path / 'files/research_notes/evidence_sheet.json', 'r') as f:
        evidence = json.load(f)

    return results, plan, evidence

def parse_results(results):
    """Convert results JSON to structured format"""
    experiments = []

    results_list = results['results']

    for result in results_list:
        if result.get('error') is not None:
            continue  # Skip failed experiments

        exp = {
            'id': result['experiment_id'],
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

        experiments.append(exp)

    return experiments

def match_rl_mwpm_pairs(experiments):
    """Match RL and MWPM results by distance, noise, error rate, seed"""
    # Separate RL and MWPM
    rl_exps = [e for e in experiments if 'RL' in e['algorithm'] or 'PPO' in e['algorithm'] or 'GNN' in e['algorithm']]
    mwpm_exps = [e for e in experiments if e['algorithm'] == 'MWPM']

    print(f"  RL experiments: {len(rl_exps)}")
    print(f"  MWPM experiments: {len(mwpm_exps)}")

    # Create lookup for MWPM
    mwpm_lookup = {}
    for e in mwpm_exps:
        key = (e['distance'], e['noise_model'], e['physical_error_rate'], e['seed'])
        mwpm_lookup[key] = e

    # Match RL to MWPM
    pairs = []
    for rl_exp in rl_exps:
        key = (rl_exp['distance'], rl_exp['noise_model'], rl_exp['physical_error_rate'], rl_exp['seed'])

        if key in mwpm_lookup:
            mwpm_exp = mwpm_lookup[key]

            L_rl = rl_exp['logical_error_rate']
            L_mwpm = mwpm_exp['logical_error_rate']

            improvement_ratio = (L_mwpm - L_rl) / L_mwpm if L_mwpm > 0 else 0

            pair = {
                'distance': rl_exp['distance'],
                'noise_model': rl_exp['noise_model'],
                'physical_error_rate': rl_exp['physical_error_rate'],
                'seed': rl_exp['seed'],
                'L_rl': L_rl,
                'L_mwpm': L_mwpm,
                'improvement_ratio': improvement_ratio,
                'rl_algorithm': rl_exp['algorithm']
            }

            pairs.append(pair)

    print(f"  Matched pairs: {len(pairs)}")

    return pairs

def test_primary_hypothesis(pairs, alpha=0.01):
    """Test if RL achieves >= 20% improvement over MWPM"""
    if len(pairs) == 0:
        return {'error': 'No matched pairs'}

    L_rl = [p['L_rl'] for p in pairs]
    L_mwpm = [p['L_mwpm'] for p in pairs]
    improvement_ratios = [p['improvement_ratio'] for p in pairs]

    # Paired t-test
    t_stat, p_value = t_test_paired(L_mwpm, L_rl)

    # Effect size
    cohens_d_val = cohens_d_paired(L_mwpm, L_rl)

    # Mean and CI
    mean_improvement = mean(improvement_ratios)
    std_improvement = std(improvement_ratios)
    n = len(improvement_ratios)

    # 95% CI using t-distribution approximation
    t_critical = 1.96  # Approximate for large n
    if n < 30:
        t_critical = 2.5  # Rough approximation for smaller n

    se = std_improvement / math.sqrt(n)
    ci_lower = mean_improvement - t_critical * se
    ci_upper = mean_improvement + t_critical * se

    # Bootstrap CI
    boot_lower, boot_upper = bootstrap_ci(improvement_ratios)

    results = {
        'comparison': 'RL_vs_MWPM',
        'metric': 'logical_error_rate',
        'n_pairs': n,
        't_statistic': t_stat if t_stat is not None else 0,
        'p_value': p_value if p_value is not None else 1.0,
        'cohens_d': cohens_d_val,
        'mean_improvement_ratio': mean_improvement,
        'mean_improvement_percent': mean_improvement * 100,
        'std_improvement_ratio': std_improvement,
        'ci_95_improvement_ratio': [ci_lower, ci_upper],
        'ci_95_improvement_percent': [ci_lower * 100, ci_upper * 100],
        'bootstrap_ci_95': [boot_lower, boot_upper],
        'mean_L_RL': mean(L_rl),
        'mean_L_MWPM': mean(L_mwpm),
        'median_L_RL': median(L_rl),
        'median_L_MWPM': median(L_mwpm),
        'std_L_RL': std(L_rl),
        'std_L_MWPM': std(L_mwpm),
        'significant': p_value < alpha if p_value is not None else False,
        'hypothesis_supported': (mean_improvement >= 0.20) and (p_value < alpha if p_value is not None else False),
        'test_method': 'paired_t_test',
        'confidence_level': 0.95,
        'significance_level': alpha
    }

    return results

def analyze_by_distance(pairs, experiments):
    """Analyze performance by distance"""
    results = {}

    # Group by algorithm and distance
    for algo_name in ['RL', 'MWPM']:
        if algo_name == 'RL':
            exps = [e for e in experiments if 'RL' in e['algorithm'] or 'PPO' in e['algorithm'] or 'GNN' in e['algorithm']]
        else:
            exps = [e for e in experiments if e['algorithm'] == algo_name]

        # Group by distance
        distance_groups = defaultdict(list)
        for e in exps:
            distance_groups[e['distance']].append(e['logical_error_rate'])

        distances = sorted(distance_groups.keys())
        error_rates = [mean(distance_groups[d]) for d in distances]
        error_stds = [std(distance_groups[d]) for d in distances]
        error_medians = [median(distance_groups[d]) for d in distances]

        # Fit exponential model: L(d) = A * exp(-alpha * d)
        # Using simple log-linear regression
        try:
            # Take log: log(L) = log(A) - alpha * d
            log_L = [math.log(L) if L > 0 else -10 for L in error_rates]

            # Simple linear regression
            n = len(distances)
            sum_d = sum(distances)
            sum_log_L = sum(log_L)
            sum_d2 = sum(d**2 for d in distances)
            sum_d_log_L = sum(distances[i] * log_L[i] for i in range(n))

            alpha_fit = (n * sum_d_log_L - sum_d * sum_log_L) / (n * sum_d2 - sum_d**2)
            log_A_fit = (sum_log_L - alpha_fit * sum_d) / n
            A_fit = math.exp(log_A_fit)

            # Note: alpha should be negative for exponential decay, so we negate
            alpha_fit = -alpha_fit

            # Compute suppression factors
            suppression_factors = []
            for i in range(len(distances) - 1):
                if error_rates[i+1] > 0:
                    lambda_d = error_rates[i] / error_rates[i+1]
                    suppression_factors.append({
                        'd': distances[i],
                        'd_next': distances[i+1],
                        'lambda': lambda_d,
                        'L_d': error_rates[i],
                        'L_d_next': error_rates[i+1]
                    })

            # Extrapolate to d=21
            L_21_pred = A_fit * math.exp(-alpha_fit * 21)

            results[algo_name] = {
                'distances': distances,
                'error_rates': error_rates,
                'error_stds': error_stds,
                'error_medians': error_medians,
                'fit_A': A_fit,
                'fit_alpha': alpha_fit,
                'suppression_factors': suppression_factors,
                'extrapolation_d21': {
                    'd': 21,
                    'L_pred': L_21_pred
                }
            }

        except Exception as e:
            print(f"  Warning: Failed to fit for {algo_name}: {e}")
            results[algo_name] = {
                'distances': distances,
                'error_rates': error_rates,
                'error_stds': error_stds,
                'fit_failed': True
            }

    # Compare suppression rates
    if 'RL' in results and 'MWPM' in results:
        if 'fit_alpha' in results['RL'] and 'fit_alpha' in results['MWPM']:
            alpha_improvement = results['RL']['fit_alpha'] - results['MWPM']['fit_alpha']

            results['comparison'] = {
                'alpha_RL': results['RL']['fit_alpha'],
                'alpha_MWPM': results['MWPM']['fit_alpha'],
                'alpha_improvement': alpha_improvement,
                'better_suppression': 'RL' if alpha_improvement > 0 else 'MWPM',
                'interpretation': f"RL suppression rate is {'better' if alpha_improvement > 0 else 'worse'} than MWPM"
            }

    return results

def architecture_comparison(experiments):
    """Compare different architectures"""
    # Extract architecture from algorithm name
    arch_groups = defaultdict(list)

    for e in experiments:
        if 'GNN' in e['algorithm']:
            arch = 'GNN'
        elif 'CNN' in e['algorithm']:
            arch = 'CNN'
        elif 'MLP' in e['algorithm']:
            arch = 'MLP'
        elif e['algorithm'] == 'MWPM':
            continue
        else:
            arch = 'RL_generic'

        arch_groups[arch].append(e['logical_error_rate'])

    if len(arch_groups) < 2:
        return {
            'error': 'Insufficient architectures',
            'available': list(arch_groups.keys())
        }

    # Performance by architecture
    performance = {}
    for arch, data in arch_groups.items():
        performance[arch] = {
            'mean': mean(data),
            'std': std(data),
            'median': median(data),
            'min': min(data),
            'max': max(data),
            'n': len(data)
        }

    # Find best
    best_arch = min(performance.items(), key=lambda x: x[1]['mean'])

    results = {
        'architectures': list(arch_groups.keys()),
        'performance': performance,
        'best_architecture': {
            'name': best_arch[0],
            'mean_error_rate': best_arch[1]['mean'],
            'std_error_rate': best_arch[1]['std']
        }
    }

    return results

def noise_model_transfer(pairs):
    """Analyze transfer across noise models"""
    noise_groups = defaultdict(list)

    for p in pairs:
        noise_groups[p['noise_model']].append(p['improvement_ratio'])

    results = {}
    for noise, improvements in noise_groups.items():
        results[noise] = {
            'mean_improvement_ratio': mean(improvements),
            'std_improvement_ratio': std(improvements),
            'median_improvement_ratio': median(improvements),
            'mean_improvement_percent': mean(improvements) * 100,
            'n_configs': len(improvements)
        }

    # Compare phenomenological vs others
    if 'phenomenological' in results:
        phenom_improvement = results['phenomenological']['mean_improvement_ratio']

        for noise in ['circuit_level', 'biased']:
            if noise in results:
                other_improvement = results[noise]['mean_improvement_ratio']
                transfer_loss = (phenom_improvement - other_improvement) / phenom_improvement if phenom_improvement != 0 else 0

                results[f'transfer_to_{noise}'] = {
                    'phenomenological_improvement': phenom_improvement,
                    'other_improvement': other_improvement,
                    'transfer_loss_ratio': transfer_loss,
                    'transfer_loss_percent': transfer_loss * 100,
                    'hypothesis_supported': abs(transfer_loss) < 0.15
                }

    return results

def robustness_analysis(experiments):
    """Analyze robustness across physical error rates"""
    results = {}

    for algo_name in ['RL', 'MWPM']:
        if algo_name == 'RL':
            exps = [e for e in experiments if 'RL' in e['algorithm'] or 'PPO' in e['algorithm'] or 'GNN' in e['algorithm']]
        else:
            exps = [e for e in experiments if e['algorithm'] == algo_name]

        # Group by physical error rate
        p_groups = defaultdict(list)
        for e in exps:
            if e['physical_error_rate'] is not None:
                p_groups[e['physical_error_rate']].append(e['logical_error_rate'])

        performance = []
        for p in sorted(p_groups.keys()):
            performance.append({
                'physical_error_rate': p,
                'mean_logical_error_rate': mean(p_groups[p]),
                'std_logical_error_rate': std(p_groups[p]),
                'median_logical_error_rate': median(p_groups[p])
            })

        results[algo_name] = performance

    return results

def identify_anomalies(experiments):
    """Identify outliers and anomalies"""
    error_rates = [e['logical_error_rate'] for e in experiments]

    Q1 = percentile(error_rates, 25)
    Q3 = percentile(error_rates, 75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    outliers = [e for e in experiments if e['logical_error_rate'] > outlier_threshold]

    results = {
        'total_configs': len(experiments),
        'outliers': {
            'count': len(outliers),
            'threshold': outlier_threshold,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        },
        'performance_summary': {
            'mean': mean(error_rates),
            'std': std(error_rates),
            'median': median(error_rates),
            'min': min(error_rates),
            'max': max(error_rates)
        }
    }

    # Check expected ranges at d=15, p=0.005
    d15_p005 = [e for e in experiments
                if e['distance'] == 15 and abs(e.get('physical_error_rate', 0) - 0.005) < 0.0001]

    if d15_p005:
        rl_d15 = [e for e in d15_p005 if 'RL' in e['algorithm'] or 'PPO' in e['algorithm'] or 'GNN' in e['algorithm']]
        mwpm_d15 = [e for e in d15_p005 if e['algorithm'] == 'MWPM']

        results['d15_p005_comparison'] = {}

        if rl_d15:
            mean_rl = mean([e['logical_error_rate'] for e in rl_d15])
            results['d15_p005_comparison']['RL'] = {
                'mean': mean_rl,
                'expected_range': [0.0008, 0.0015],
                'within_expected': 0.0008 <= mean_rl <= 0.0015
            }

        if mwpm_d15:
            mean_mwpm = mean([e['logical_error_rate'] for e in mwpm_d15])
            results['d15_p005_comparison']['MWPM'] = {
                'mean': mean_mwpm,
                'expected_range': [0.0012, 0.002],
                'within_expected': 0.0012 <= mean_mwpm <= 0.002
            }

    return results

def main():
    print("="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF QEC EXPERIMENTS")
    print("="*80)

    print("\n[1/7] Loading results...")
    results_data, plan, evidence = load_results()
    print(f"  Metadata: {results_data['metadata']}")

    print("\n[2/7] Parsing results...")
    experiments = parse_results(results_data)
    print(f"  Total experiments: {len(experiments)}")

    # Count by algorithm
    algo_counts = defaultdict(int)
    for e in experiments:
        algo_counts[e['algorithm']] += 1
    print(f"  Algorithms: {dict(algo_counts)}")

    # Analysis outputs
    analysis_results = {
        'metadata': {
            'total_configs': len(experiments),
            'analysis_date': '2025-12-28',
            'experiment_metadata': results_data['metadata']
        }
    }

    # 1. Test primary hypothesis
    print("\n[3/7] TESTING PRIMARY HYPOTHESIS: RL >= 20% improvement over MWPM")
    print("-"*80)
    pairs = match_rl_mwpm_pairs(experiments)

    if len(pairs) > 0:
        hypothesis_test = test_primary_hypothesis(pairs)
        analysis_results['primary_hypothesis'] = {'overall': hypothesis_test}

        print(f"  Mean improvement: {hypothesis_test['mean_improvement_percent']:.2f}%")
        print(f"  95% CI: [{hypothesis_test['ci_95_improvement_percent'][0]:.2f}%, "
              f"{hypothesis_test['ci_95_improvement_percent'][1]:.2f}%]")
        print(f"  p-value: {hypothesis_test['p_value']:.6f} (alpha=0.01)")
        print(f"  Cohen's d: {hypothesis_test['cohens_d']:.3f}")
        print(f"  Sample size: {hypothesis_test['n_pairs']} pairs")
        print(f"  Significant: {hypothesis_test['significant']}")
        print(f"  HYPOTHESIS SUPPORTED: {hypothesis_test['hypothesis_supported']}")
    else:
        print("  ERROR: No matched pairs!")
        analysis_results['primary_hypothesis'] = {'error': 'No matched pairs'}

    # 2. Analyze by distance
    print("\n[4/7] ANALYZING BY DISTANCE: Exponential suppression")
    print("-"*80)
    distance_analysis = analyze_by_distance(pairs, experiments)
    analysis_results['distance_analysis'] = distance_analysis

    if 'RL' in distance_analysis and 'fit_alpha' in distance_analysis['RL']:
        print(f"  RL suppression rate (alpha): {distance_analysis['RL']['fit_alpha']:.4f}")
    if 'MWPM' in distance_analysis and 'fit_alpha' in distance_analysis['MWPM']:
        print(f"  MWPM suppression rate (alpha): {distance_analysis['MWPM']['fit_alpha']:.4f}")

    if 'comparison' in distance_analysis:
        print(f"  {distance_analysis['comparison']['interpretation']}")

    # 3. Architecture comparison
    print("\n[5/7] ARCHITECTURE COMPARISON")
    print("-"*80)
    arch_analysis = architecture_comparison(experiments)
    analysis_results['architecture_comparison'] = arch_analysis

    if 'best_architecture' in arch_analysis:
        print(f"  Best: {arch_analysis['best_architecture']['name']} "
              f"(mean: {arch_analysis['best_architecture']['mean_error_rate']:.6f})")

    # 4. Noise model transfer
    print("\n[6/7] NOISE MODEL TRANSFER")
    print("-"*80)
    if len(pairs) > 0:
        transfer_analysis = noise_model_transfer(pairs)
        analysis_results['noise_transfer'] = transfer_analysis

        for noise_model, metrics in transfer_analysis.items():
            if not noise_model.startswith('transfer_to_'):
                print(f"  {noise_model}: {metrics['mean_improvement_percent']:.2f}%")

    # 5. Robustness
    print("\n[7/7] ROBUSTNESS UNDER NOISE VARIATION")
    print("-"*80)
    robustness = robustness_analysis(experiments)
    analysis_results['noise_robustness'] = robustness

    for algo in ['RL', 'MWPM']:
        if algo in robustness:
            print(f"  {algo}: {len(robustness[algo])} error rates tested")

    # 6. Anomalies
    print("\n[8/8] IDENTIFYING ANOMALIES")
    print("-"*80)
    anomalies = identify_anomalies(experiments)
    analysis_results['trends'] = anomalies

    print(f"  Total configs: {anomalies['total_configs']}")
    print(f"  Outliers: {anomalies['outliers']['count']}")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = Path('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results')

    # Save comprehensive analysis
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print("✓ Saved: files/results/analysis_summary.json")

    # Save RL vs MWPM comparison
    if 'primary_hypothesis' in analysis_results and 'overall' in analysis_results['primary_hypothesis']:
        with open(output_dir / 'comparison_rl_vs_mwpm.json', 'w') as f:
            json.dump(analysis_results['primary_hypothesis']['overall'], f, indent=2)
        print("✓ Saved: files/results/comparison_rl_vs_mwpm.json")

    # Save architecture ranking
    if 'architecture_comparison' in analysis_results:
        with open(output_dir / 'architecture_ranking.json', 'w') as f:
            json.dump(analysis_results['architecture_comparison'], f, indent=2)
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
        hyp = analysis_results['primary_hypothesis']['overall']
        hypothesis_supported = hyp.get('hypothesis_supported', False)
        mean_improvement = hyp.get('mean_improvement_percent', 0)
        p_value = hyp.get('p_value', 1.0)

        if not hypothesis_supported:
            print(f"\n⚠ PRIMARY HYPOTHESIS NOT SUPPORTED")
            print(f"  Observed: {mean_improvement:.2f}% (target: 20%)")
            print(f"  p-value: {p_value:.6f}")

            followup_plan = {
                'trigger': f"RL improvement {mean_improvement:.1f}% < 20% threshold (p={p_value:.6f})",
                'primary_hypothesis_failed': True,
                'observed_improvement_percent': mean_improvement,
                'target_improvement_percent': 20.0,
                'p_value': p_value,
                'hypotheses': [
                    {
                        'hypothesis': 'Insufficient training episodes for convergence',
                        'diagnostic_experiment': 'Extend training from 200 to 1000+ steps',
                        'expected_outcome': 'Extended training improves L_RL by >= 10%',
                        'priority': 1,
                        'rationale': 'Limited to 200 steps in simulation'
                    },
                    {
                        'hypothesis': 'Reward function not optimally shaped',
                        'diagnostic_experiment': 'Test dense vs sparse reward variants',
                        'expected_outcome': 'Shaped reward achieves lower L_RL',
                        'priority': 1,
                        'rationale': 'Reward may lack sufficient learning signal'
                    },
                    {
                        'hypothesis': 'Architecture insufficient for long-range correlations',
                        'diagnostic_experiment': 'Test deeper GNN or Transformer',
                        'expected_outcome': 'Deeper model improves by >= 15%',
                        'priority': 2,
                        'rationale': 'Surface codes have long-range dependencies'
                    }
                ],
                'selected_followup': None,
                'mode': 'discovery',
                'next_steps': [
                    'Select highest priority hypothesis',
                    'Design minimal diagnostic experiment',
                    'Run with same statistical rigor',
                    'Iterate until supported or falsified'
                ]
            }

            with open(output_dir / 'followup_plan.json', 'w') as f:
                json.dump(followup_plan, f, indent=2)
            print("✓ Saved: files/results/followup_plan.json")

        else:
            print(f"\n✓ PRIMARY HYPOTHESIS SUPPORTED")
            print(f"  Observed: {mean_improvement:.2f}% >= 20% target")
            print(f"  p-value: {p_value:.6f} < 0.01")
            print(f"  Conclusion: RL achieves significant improvement")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
