#!/usr/bin/env python3
"""
Comprehensive statistical analysis of 162 QEC experiment results.
Tests primary hypothesis: "RL achieves >=20% improvement over MWPM baseline"
"""

import json
import math
from pathlib import Path
from collections import defaultdict

def mean(data):
    return sum(data) / len(data) if len(data) > 0 else 0.0

def std(data, ddof=1):
    if len(data) <= ddof:
        return 0.0
    m = mean(data)
    return math.sqrt(sum((x - m)**2 for x in data) / (len(data) - ddof))

def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n % 2 == 0:
        return (sorted_data[n//2-1] + sorted_data[n//2]) / 2
    return sorted_data[n//2]

def percentile(data, p):
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
    """Paired t-test"""
    if len(x) != len(y) or len(x) == 0:
        return None, 1.0

    diff = [x[i] - y[i] for i in range(len(x))]
    n = len(diff)
    mean_diff = mean(diff)
    std_diff = std(diff, ddof=1)

    if std_diff == 0:
        return 0.0, 1.0

    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Rough p-value approximation
    if n > 30:
        p_value = 0.5 * math.erfc(t_stat / math.sqrt(2))
    else:
        p_value = 0.001 if abs(t_stat) > 3 else (0.01 if abs(t_stat) > 2.5 else 0.05)

    return t_stat, p_value

def cohens_d(x, y):
    """Cohen's d effect size"""
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

# Main analysis functions

def load_and_parse():
    """Load all data files"""
    base = Path('/Users/jminding/Desktop/Code/Research Agent/research_platform')

    with open(base / 'files/results/results_table.json') as f:
        results = json.load(f)

    with open(base / 'files/theory/experiment_plan.json') as f:
        plan = json.load(f)

    with open(base / 'files/research_notes/evidence_sheet.json') as f:
        evidence = json.load(f)

    # Parse experiments
    experiments = []
    for result in results['results']:
        if result.get('error') is not None:
            continue

        experiments.append({
            'id': result['experiment_id'],
            'distance': result['distance'],
            'algorithm': result['algorithm'],
            'noise_model': result['noise_model'],
            'training_episodes': result['training_episodes'],
            'L': result['logical_error_rate'],
            'L_std': result['logical_error_rate_std'],
            'improvement_ratio': result.get('improvement_ratio', 0),
            'gen_gap': result.get('generalization_gap', None),
            'time': result['wall_clock_time'],
            'seed': result['seed'],
            'p': result['additional_metrics'].get('physical_error_rate', None)
        })

    return experiments, results['metadata'], plan, evidence

def match_pairs(experiments):
    """Match RL_GNN with MWPM by distance, noise, p, seed"""
    rl_exps = [e for e in experiments if 'RL' in e['algorithm']]
    mwpm_exps = [e for e in experiments if e['algorithm'] == 'MWPM']

    # Create MWPM lookup
    mwpm_lookup = {}
    for e in mwpm_exps:
        key = (e['distance'], e['noise_model'], e['p'], e['seed'])
        mwpm_lookup[key] = e

    # Match pairs
    pairs = []
    for rl in rl_exps:
        key = (rl['distance'], rl['noise_model'], rl['p'], rl['seed'])
        if key in mwpm_lookup:
            mwpm = mwpm_lookup[key]
            L_rl = rl['L']
            L_mwpm = mwpm['L']
            improvement = (L_mwpm - L_rl) / L_mwpm if L_mwpm > 0 else 0

            pairs.append({
                'distance': rl['distance'],
                'noise': rl['noise_model'],
                'p': rl['p'],
                'seed': rl['seed'],
                'L_rl': L_rl,
                'L_mwpm': L_mwpm,
                'improvement': improvement
            })

    return pairs

def test_hypothesis(pairs):
    """Test primary hypothesis: RL >= 20% improvement"""
    L_rl = [p['L_rl'] for p in pairs]
    L_mwpm = [p['L_mwpm'] for p in pairs]
    improvements = [p['improvement'] for p in pairs]

    # Paired t-test
    t_stat, p_value = t_test_paired(L_mwpm, L_rl)

    # Effect size
    d = cohens_d(L_mwpm, L_rl)

    # CI
    m = mean(improvements)
    s = std(improvements)
    n = len(improvements)
    se = s / math.sqrt(n)
    t_crit = 2.0 if n > 30 else 2.5
    ci_lower = m - t_crit * se
    ci_upper = m + t_crit * se

    # Bootstrap CI
    boot_lower, boot_upper = bootstrap_ci(improvements)

    result = {
        'comparison': 'RL_vs_MWPM',
        'metric': 'logical_error_rate',
        'n_pairs': n,
        't_statistic': float(t_stat) if t_stat is not None else 0,
        'p_value': float(p_value),
        'cohens_d': float(d),
        'mean_improvement_ratio': float(m),
        'mean_improvement_percent': float(m * 100),
        'std_improvement': float(s),
        'ci_95_ratio': [float(ci_lower), float(ci_upper)],
        'ci_95_percent': [float(ci_lower * 100), float(ci_upper * 100)],
        'bootstrap_ci': [float(boot_lower), float(boot_upper)],
        'mean_L_RL': float(mean(L_rl)),
        'mean_L_MWPM': float(mean(L_mwpm)),
        'median_L_RL': float(median(L_rl)),
        'median_L_MWPM': float(median(L_mwpm)),
        'significant': p_value < 0.01,
        'hypothesis_supported': (m >= 0.20) and (p_value < 0.01),
        'test_method': 'paired_t_test',
        'alpha': 0.01
    }

    return result

def analyze_distance(experiments):
    """Distance-dependent analysis"""
    results = {}

    for algo in ['RL_GNN', 'MWPM']:
        exps = [e for e in experiments if e['algorithm'] == algo]

        # Group by distance
        distance_groups = defaultdict(list)
        for e in exps:
            distance_groups[e['distance']].append(e['L'])

        distances = sorted(distance_groups.keys())
        L_means = [mean(distance_groups[d]) for d in distances]
        L_stds = [std(distance_groups[d]) for d in distances]

        # Fit exponential: L(d) = A * exp(-alpha * d)
        # log(L) = log(A) - alpha * d
        try:
            log_L = [math.log(L) if L > 0 else -10 for L in L_means]
            n = len(distances)
            sum_d = sum(distances)
            sum_log_L = sum(log_L)
            sum_d2 = sum(d**2 for d in distances)
            sum_d_log_L = sum(distances[i] * log_L[i] for i in range(n))

            alpha = (n * sum_d_log_L - sum_d * sum_log_L) / (n * sum_d2 - sum_d**2)
            log_A = (sum_log_L - alpha * sum_d) / n
            A = math.exp(log_A)
            alpha = -alpha  # Negate for decay

            # Suppression factors
            sup_factors = []
            for i in range(len(distances) - 1):
                if L_means[i+1] > 0:
                    lam = L_means[i] / L_means[i+1]
                    sup_factors.append({
                        'd': int(distances[i]),
                        'd_next': int(distances[i+1]),
                        'lambda': float(lam)
                    })

            # Extrapolate to d=21
            L_21 = A * math.exp(-alpha * 21)

            results[algo] = {
                'distances': [int(d) for d in distances],
                'error_rates': [float(L) for L in L_means],
                'error_stds': [float(s) for s in L_stds],
                'fit_A': float(A),
                'fit_alpha': float(alpha),
                'suppression_factors': sup_factors,
                'extrapolation_d21': {'d': 21, 'L_pred': float(L_21)}
            }
        except Exception as e:
            results[algo] = {
                'distances': [int(d) for d in distances],
                'error_rates': [float(L) for L in L_means],
                'fit_failed': True,
                'error': str(e)
            }

    # Compare
    if 'RL_GNN' in results and 'MWPM' in results:
        if 'fit_alpha' in results['RL_GNN'] and 'fit_alpha' in results['MWPM']:
            alpha_rl = results['RL_GNN']['fit_alpha']
            alpha_mwpm = results['MWPM']['fit_alpha']
            results['comparison'] = {
                'alpha_RL': float(alpha_rl),
                'alpha_MWPM': float(alpha_mwpm),
                'alpha_improvement': float(alpha_rl - alpha_mwpm),
                'better': 'RL' if alpha_rl > alpha_mwpm else 'MWPM'
            }

    return results

def noise_transfer(pairs):
    """Noise model transfer analysis"""
    noise_groups = defaultdict(list)
    for p in pairs:
        noise_groups[p['noise']].append(p['improvement'])

    results = {}
    for noise, improvements in noise_groups.items():
        results[noise] = {
            'mean_improvement': float(mean(improvements)),
            'std_improvement': float(std(improvements)),
            'mean_percent': float(mean(improvements) * 100),
            'n': len(improvements)
        }

    # Transfer loss
    if 'phenomenological' in results:
        phenom = results['phenomenological']['mean_improvement']
        for noise in ['circuit_level', 'biased']:
            if noise in results:
                other = results[noise]['mean_improvement']
                loss = (phenom - other) / phenom if phenom != 0 else 0
                results[f'transfer_to_{noise}'] = {
                    'loss_ratio': float(loss),
                    'loss_percent': float(loss * 100),
                    'hypothesis_supported': abs(loss) < 0.15
                }

    return results

def robustness_p(experiments):
    """Robustness across physical error rates"""
    results = {}

    for algo in ['RL_GNN', 'MWPM']:
        exps = [e for e in experiments if e['algorithm'] == algo]

        p_groups = defaultdict(list)
        for e in exps:
            if e['p'] is not None:
                p_groups[e['p']].append(e['L'])

        perf = []
        for p in sorted(p_groups.keys()):
            perf.append({
                'p': float(p),
                'mean_L': float(mean(p_groups[p])),
                'std_L': float(std(p_groups[p])),
                'median_L': float(median(p_groups[p]))
            })

        results[algo] = perf

    return results

def anomalies(experiments):
    """Identify anomalies"""
    error_rates = [e['L'] for e in experiments]

    Q1 = percentile(error_rates, 25)
    Q3 = percentile(error_rates, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR

    outliers = [e for e in experiments if e['L'] > threshold]

    results = {
        'total': len(experiments),
        'outliers': {
            'count': len(outliers),
            'threshold': float(threshold)
        },
        'summary': {
            'mean': float(mean(error_rates)),
            'median': float(median(error_rates)),
            'std': float(std(error_rates)),
            'min': float(min(error_rates)),
            'max': float(max(error_rates))
        }
    }

    # Check d=15, p=0.005 expectations
    d15 = [e for e in experiments
           if e['distance'] == 15 and abs(e['p'] - 0.005) < 0.0001]

    if d15:
        rl_d15 = [e for e in d15 if 'RL' in e['algorithm']]
        mwpm_d15 = [e for e in d15 if e['algorithm'] == 'MWPM']

        results['d15_p005'] = {}

        if rl_d15:
            m = mean([e['L'] for e in rl_d15])
            results['d15_p005']['RL'] = {
                'mean': float(m),
                'expected': [0.0008, 0.0015],
                'within_expected': 0.0008 <= m <= 0.0015
            }

        if mwpm_d15:
            m = mean([e['L'] for e in mwpm_d15])
            results['d15_p005']['MWPM'] = {
                'mean': float(m),
                'expected': [0.0012, 0.002],
                'within_expected': 0.0012 <= m <= 0.002
            }

    return results

def main():
    print("="*80)
    print("QEC EXPERIMENT STATISTICAL ANALYSIS")
    print("="*80)

    # Load data
    print("\n[1/7] Loading data...")
    experiments, metadata, plan, evidence = load_and_parse()
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Training steps: {metadata['training_steps']}")
    print(f"  Seeds: {metadata['num_seeds']}")

    # Count algorithms
    algo_counts = defaultdict(int)
    for e in experiments:
        algo_counts[e['algorithm']] += 1
    print(f"  Algorithms: {dict(algo_counts)}")

    # Analysis results
    analysis = {
        'metadata': {
            'total_experiments': len(experiments),
            'date': '2025-12-28',
            'experiment_metadata': metadata
        }
    }

    # 1. Primary hypothesis
    print("\n[2/7] TESTING PRIMARY HYPOTHESIS")
    print("-"*80)
    pairs = match_pairs(experiments)
    print(f"  Matched pairs: {len(pairs)}")

    if pairs:
        hyp_test = test_hypothesis(pairs)
        analysis['primary_hypothesis'] = {'overall': hyp_test}

        print(f"  Mean improvement: {hyp_test['mean_improvement_percent']:.2f}%")
        print(f"  95% CI: [{hyp_test['ci_95_percent'][0]:.2f}%, {hyp_test['ci_95_percent'][1]:.2f}%]")
        print(f"  p-value: {hyp_test['p_value']:.6f}")
        print(f"  Cohen's d: {hyp_test['cohens_d']:.3f}")
        print(f"  Significant (p<0.01): {hyp_test['significant']}")
        print(f"  HYPOTHESIS SUPPORTED: {hyp_test['hypothesis_supported']}")
    else:
        print("  ERROR: No pairs found!")
        analysis['primary_hypothesis'] = {'error': 'No pairs'}

    # 2. Distance analysis
    print("\n[3/7] DISTANCE ANALYSIS")
    print("-"*80)
    dist_analysis = analyze_distance(experiments)
    analysis['distance_analysis'] = dist_analysis

    for algo in ['RL_GNN', 'MWPM']:
        if algo in dist_analysis and 'fit_alpha' in dist_analysis[algo]:
            print(f"  {algo} suppression rate: {dist_analysis[algo]['fit_alpha']:.4f}")

    if 'comparison' in dist_analysis:
        print(f"  Better suppression: {dist_analysis['comparison']['better']}")

    # 3. Noise transfer
    print("\n[4/7] NOISE TRANSFER")
    print("-"*80)
    if pairs:
        transfer = noise_transfer(pairs)
        analysis['noise_transfer'] = transfer

        for noise in ['phenomenological', 'circuit_level', 'biased']:
            if noise in transfer:
                print(f"  {noise}: {transfer[noise]['mean_percent']:.2f}%")

    # 4. Robustness
    print("\n[5/7] ROBUSTNESS UNDER NOISE VARIATION")
    print("-"*80)
    robustness = robustness_p(experiments)
    analysis['robustness'] = robustness

    for algo in ['RL_GNN', 'MWPM']:
        if algo in robustness:
            print(f"  {algo}: {len(robustness[algo])} error rates tested")

    # 5. Anomalies
    print("\n[6/7] ANOMALY DETECTION")
    print("-"*80)
    anom = anomalies(experiments)
    analysis['anomalies'] = anom

    print(f"  Outliers: {anom['outliers']['count']}")
    if 'd15_p005' in anom:
        print(f"  d=15, p=0.005 results:")
        for algo in ['RL', 'MWPM']:
            if algo in anom['d15_p005']:
                data = anom['d15_p005'][algo]
                print(f"    {algo}: {data['mean']:.6f} (expected: {data['expected']}, within: {data['within_expected']})")

    # Save results
    print("\n[7/7] SAVING RESULTS")
    print("-"*80)

    base = Path('/Users/jminding/Desktop/Code/Research Agent/research_platform/files/results')

    with open(base / 'analysis_summary.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("  ✓ analysis_summary.json")

    if 'primary_hypothesis' in analysis and 'overall' in analysis['primary_hypothesis']:
        with open(base / 'comparison_rl_vs_mwpm.json', 'w') as f:
            json.dump(analysis['primary_hypothesis']['overall'], f, indent=2)
        print("  ✓ comparison_rl_vs_mwpm.json")

    if 'distance_analysis' in analysis:
        with open(base / 'generalization_curves.json', 'w') as f:
            json.dump(analysis['distance_analysis'], f, indent=2)
        print("  ✓ generalization_curves.json")

    # Follow-up plan if needed
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if 'primary_hypothesis' in analysis and 'overall' in analysis['primary_hypothesis']:
        hyp = analysis['primary_hypothesis']['overall']
        supported = hyp.get('hypothesis_supported', False)
        improvement = hyp.get('mean_improvement_percent', 0)
        p_val = hyp.get('p_value', 1.0)

        if not supported:
            print(f"\n⚠ HYPOTHESIS NOT SUPPORTED")
            print(f"  Improvement: {improvement:.2f}% (target: 20%)")
            print(f"  p-value: {p_val:.6f}")

            followup = {
                'trigger': f"Improvement {improvement:.1f}% < 20% (p={p_val:.6f})",
                'failed': True,
                'observed_percent': float(improvement),
                'target_percent': 20.0,
                'p_value': float(p_val),
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

            with open(base / 'followup_plan.json', 'w') as f:
                json.dump(followup, f, indent=2)
            print("  ✓ followup_plan.json")

        else:
            print(f"\n✓ HYPOTHESIS SUPPORTED")
            print(f"  Improvement: {improvement:.2f}% >= 20%")
            print(f"  p-value: {p_val:.6f} < 0.01")
            print(f"  Conclusion: RL significantly outperforms MWPM")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
