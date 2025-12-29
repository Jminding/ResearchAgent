# Formal Hypotheses and Experimental Validation Strategy
## GNNs for Financial Anomaly Detection

---

## 1. Central Research Question

**RQ:** Under what conditions do heterophily-aware Graph Neural Networks outperform homophily-assuming GNNs for financial anomaly detection, and what is the magnitude of improvement?

---

## 2. Formal Hypotheses

### Primary Hypothesis (H1)

**Statement:**
Heterophily-aware GNN architectures (H2GCN, FAGCN, GPR-GNN, LINKX) significantly outperform homophily-assuming GNN architectures (GCN, GraphSAGE, GAT) on financial transaction graphs when the edge homophily ratio h_edge < 0.5.

**Formal Definition:**
Let:
- M_hetero = {H2GCN, FAGCN, GPR-GNN, LINKX}
- M_homo = {GCN, GraphSAGE, GAT}
- F1(m, G) = F1-score of model m on graph G
- h(G) = edge homophily ratio of G

**H1:** For financial graph G with h(G) < 0.5:
```
max_{m in M_hetero} F1(m, G) > max_{m in M_homo} F1(m, G) + delta
```
where delta >= 0.05 (5 percentage points).

**Quantitative Prediction:**
- On graphs with h(G) in [0.1, 0.3]: Delta F1 >= 0.08
- On graphs with h(G) in [0.3, 0.5]: Delta F1 >= 0.05
- On graphs with h(G) > 0.7: Delta F1 < 0.02 (no significant difference)

---

### Secondary Hypothesis (H2)

**Statement:**
The performance gap between heterophily-aware and homophily-assuming GNNs increases monotonically as the homophily ratio decreases.

**Formal Definition:**
Let Delta(h) = max_{m in M_hetero} F1(m, G_h) - max_{m in M_homo} F1(m, G_h)
where G_h is a graph with homophily ratio h.

**H2:**
```
d(Delta)/dh < 0 for h in [0.1, 0.7]
```

**Quantitative Prediction:**
```
Delta(0.1) - Delta(0.7) >= 0.10
```

---

### Tertiary Hypothesis (H3)

**Statement:**
Heterophily-aware GNNs maintain robust performance (F1 > 0.75) across the typical financial fraud homophily range [0.1, 0.4], while homophily-assuming GNNs degrade below acceptable thresholds.

**Formal Definition:**
Let tau = 0.75 be the minimum acceptable F1 threshold.

**H3a (Heterophily-aware robustness):**
```
For all h in [0.1, 0.4]: max_{m in M_hetero} F1(m, G_h) >= tau
```

**H3b (Homophily-assuming degradation):**
```
There exists h* in [0.1, 0.4] such that: max_{m in M_homo} F1(m, G_{h*}) < tau - 0.10
```

**Quantitative Prediction:**
- Best heterophily-aware model: F1 >= 0.80 across all h in [0.1, 0.4]
- Best homophily-assuming model: F1 < 0.70 when h < 0.25

---

### Hypothesis on Class Imbalance Interaction (H4)

**Statement:**
The superiority of heterophily-aware GNNs persists across class imbalance ratios typical of financial fraud (IR in [50, 1000]), but the absolute performance of all models decreases with increasing imbalance.

**Formal Definition:**
Let IR = n_normal / n_fraud be the imbalance ratio.

**H4:**
```
For all IR in {50, 100, 500, 1000}:
  max_{m in M_hetero} F1(m, G, IR) > max_{m in M_homo} F1(m, G, IR)
```

AND

```
F1(m, G, IR=50) > F1(m, G, IR=1000) for all models m
```

**Quantitative Prediction:**
- Performance degradation from IR=50 to IR=1000: 0.05 <= Delta F1 <= 0.15
- Heterophily-aware advantage persists: Delta_{hetero-homo} >= 0.04 at all IR levels

---

### Hypothesis on Optimal Architecture (H5)

**Statement:**
Among heterophily-aware architectures, methods that explicitly model negative correlations (FAGCN) or decouple structure and features (LINKX) outperform methods that only use higher-order neighborhoods (H2GCN) on financial graphs with very low homophily (h < 0.2).

**Formal Definition:**
Let M_negative = {FAGCN, LINKX} and M_higher_order = {H2GCN, GPR-GNN}.

**H5:** For G with h(G) < 0.2:
```
max_{m in M_negative} F1(m, G) > max_{m in M_higher_order} F1(m, G) + 0.03
```

---

## 3. Falsification Criteria

### Criteria for Rejecting H1

H1 is **falsified** if ANY of the following hold:
1. On 3+ datasets with h < 0.5, best homophily-assuming model matches or exceeds best heterophily-aware model (Delta F1 < 0.02)
2. The difference is not statistically significant (p > 0.05 via paired t-test across seeds)
3. Effect is not reproducible across 5 random seeds

### Criteria for Rejecting H2

H2 is **falsified** if:
1. Spearman correlation between h and Delta is positive (rho > 0)
2. The relationship is non-monotonic with reversals > 0.03 F1

### Criteria for Rejecting H3

H3a is **falsified** if: Best heterophily-aware model achieves F1 < 0.75 on any h in [0.1, 0.4]
H3b is **falsified** if: All homophily-assuming models achieve F1 > 0.65 across all h in [0.1, 0.4]

### Criteria for Rejecting H4

H4 is **falsified** if:
1. At any IR level, homophily-assuming models outperform heterophily-aware models
2. Performance does not degrade with increasing IR (counter to expectation)

### Criteria for Rejecting H5

H5 is **falsified** if:
1. H2GCN or GPR-GNN matches or exceeds FAGCN and LINKX at h < 0.2
2. The difference is not statistically significant

---

## 4. Confirmation Thresholds

### Strong Confirmation

Hypothesis is **strongly confirmed** if:
- Effect size exceeds predicted threshold by 50%+
- p-value < 0.01
- Effect replicates across all tested datasets
- Effect holds for 95% confidence intervals

### Moderate Confirmation

Hypothesis is **moderately confirmed** if:
- Effect size meets predicted threshold
- p-value < 0.05
- Effect replicates on majority (>75%) of datasets

### Weak Confirmation

Hypothesis is **weakly confirmed** if:
- Effect is in predicted direction but below threshold
- p-value < 0.10
- Effect shows dataset-specific variation

---

## 5. Experimental Validation Pseudocode

### 5.1 Main Experimental Pipeline

```
ALGORITHM: GNN_Financial_Anomaly_Detection_Experiment

INPUT:
  - datasets: List of financial graph datasets
  - homophily_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  - imbalance_ratios: [50, 100, 200, 500, 1000]
  - models_hetero: [H2GCN, FAGCN, GPR-GNN, LINKX]
  - models_homo: [GCN, GraphSAGE, GAT]
  - n_seeds: 5
  - n_folds: 5

OUTPUT:
  - results: DataFrame with columns [dataset, model, homophily, IR, seed, fold, F1, AUPRC, AUROC, P@1%]
  - statistical_tests: Dictionary of hypothesis test results

PROCEDURE:

1. INITIALIZE results_list = []

2. FOR EACH dataset in datasets:

   2.1. LOAD graph G = (V, E, X, Y) from dataset

   2.2. COMPUTE empirical_homophily = calculate_edge_homophily(G, Y)

   2.3. PRINT "Dataset: {dataset}, Nodes: {|V|}, Edges: {|E|}, Homophily: {empirical_homophily}"

3. FOR EACH target_homophily in homophily_levels:

   3.1. IF using synthetic data:
        G_modified = generate_SBM_graph(n_nodes=10000,
                                         n_fraud=200,
                                         target_homophily=target_homophily)
   3.2. ELSE IF using real data with modification:
        G_modified = rewire_to_homophily(G, target_homophily, max_rewires=1000)

   3.3. VERIFY actual_homophily = calculate_edge_homophily(G_modified, Y)
        ASSERT |actual_homophily - target_homophily| < 0.05

4. FOR EACH target_IR in imbalance_ratios:

   4.1. G_imbalanced = subsample_to_imbalance(G_modified, target_IR)

   4.2. VERIFY actual_IR = count_normal(G_imbalanced) / count_fraud(G_imbalanced)
        ASSERT |actual_IR - target_IR| / target_IR < 0.1

5. FOR EACH seed in range(n_seeds):

   5.1. SET random_seed(seed)

   5.2. splits = create_temporal_splits(G_imbalanced, n_folds, test_ratio=0.2)

   5.3. FOR EACH (train_idx, val_idx, test_idx) in splits:

        5.3.1. X_train, Y_train = X[train_idx], Y[train_idx]
        5.3.2. X_val, Y_val = X[val_idx], Y[val_idx]
        5.3.3. X_test, Y_test = X[test_idx], Y[test_idx]

        5.3.4. A_train = extract_subgraph_adjacency(A, train_idx)
        5.3.5. A_full = A  // For transductive setting

6. FOR EACH model_name in (models_hetero + models_homo):

   6.1. model = initialize_model(model_name,
                                  input_dim=d,
                                  hidden_dim=64,
                                  output_dim=2,
                                  num_layers=2)

   6.2. optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

   6.3. loss_fn = weighted_cross_entropy(weight_positive=IR)

   6.4. // Training loop
        best_val_f1 = 0
        patience_counter = 0

        FOR epoch in range(200):

            6.4.1. model.train()
            6.4.2. optimizer.zero_grad()

            6.4.3. logits = model(X, A_full)
            6.4.4. loss = loss_fn(logits[train_idx], Y_train)

            6.4.5. loss.backward()
            6.4.6. optimizer.step()

            6.4.7. // Validation
                   model.eval()
                   val_logits = model(X, A_full)[val_idx]
                   val_preds = argmax(val_logits, dim=1)
                   val_f1 = f1_score(Y_val, val_preds, pos_label=1)

            6.4.8. IF val_f1 > best_val_f1:
                       best_val_f1 = val_f1
                       save_checkpoint(model, "best_model.pt")
                       patience_counter = 0
                   ELSE:
                       patience_counter += 1

            6.4.9. IF patience_counter >= 20:
                       BREAK  // Early stopping

7. // Evaluation
   7.1. load_checkpoint(model, "best_model.pt")
   7.2. model.eval()

   7.3. test_logits = model(X, A_full)[test_idx]
   7.4. test_probs = softmax(test_logits)[:, 1]
   7.5. test_preds = (test_probs > 0.5).int()

   7.6. // Compute metrics
        f1 = f1_score(Y_test, test_preds, pos_label=1)
        precision = precision_score(Y_test, test_preds, pos_label=1)
        recall = recall_score(Y_test, test_preds, pos_label=1)
        auprc = average_precision_score(Y_test, test_probs)
        auroc = roc_auc_score(Y_test, test_probs)
        p_at_1 = precision_at_k(Y_test, test_probs, k=0.01)

   7.7. results_list.append({
            'dataset': dataset,
            'model': model_name,
            'model_type': 'hetero' if model_name in models_hetero else 'homo',
            'homophily': target_homophily,
            'imbalance_ratio': target_IR,
            'seed': seed,
            'fold': fold_idx,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auprc': auprc,
            'auroc': auroc,
            'p_at_1': p_at_1
        })

8. // Aggregate results
   results_df = DataFrame(results_list)

   8.1. grouped = results_df.groupby(['model', 'model_type', 'homophily', 'imbalance_ratio'])
   8.2. summary = grouped.agg({
            'f1': ['mean', 'std'],
            'auprc': ['mean', 'std'],
            'auroc': ['mean', 'std']
        })

9. RETURN results_df, summary

END ALGORITHM
```

### 5.2 Homophily Calculation Subroutine

```
FUNCTION calculate_edge_homophily(G, Y):

    INPUT:
      - G: Graph with edge list E
      - Y: Node labels

    OUTPUT:
      - h: Edge homophily ratio in [0, 1]

    PROCEDURE:
    1. same_label_edges = 0
    2. total_edges = |E|

    3. FOR EACH (i, j) in E:
           IF Y[i] == Y[j]:
               same_label_edges += 1

    4. h = same_label_edges / total_edges

    5. RETURN h

END FUNCTION
```

### 5.3 Synthetic Graph Generation Subroutine

```
FUNCTION generate_SBM_graph(n_nodes, n_fraud, target_homophily):

    INPUT:
      - n_nodes: Total number of nodes
      - n_fraud: Number of fraud nodes (class 1)
      - target_homophily: Desired edge homophily ratio

    OUTPUT:
      - G: NetworkX graph
      - X: Node feature matrix
      - Y: Node labels

    PROCEDURE:
    1. n_normal = n_nodes - n_fraud

    2. // Compute block probability matrix for 2-block SBM
       // h = (p_in * n_fraud^2 + p_in * n_normal^2) / (total_edges)
       // For simplicity, use:

       avg_degree = 20
       total_edges = n_nodes * avg_degree / 2

       // Solve for p_in and p_out given target_homophily
       // Let p_in = probability of edge within same class
       // Let p_out = probability of edge between classes

       // Approximate: h = (p_in * (n_f^2 + n_n^2)) / (p_in*(n_f^2+n_n^2) + p_out*2*n_f*n_n)
       // Rearranging to get p_out/p_in ratio from h

       r = ((1 - target_homophily) / target_homophily) * (n_fraud^2 + n_normal^2) / (2 * n_fraud * n_normal)

       // Set p_in to achieve desired average degree
       p_in = avg_degree / n_normal  // Approximate
       p_out = r * p_in

       // Ensure probabilities are valid
       p_in = min(p_in, 1.0)
       p_out = min(p_out, 1.0)

    3. // Create block sizes and probability matrix
       sizes = [n_normal, n_fraud]
       probs = [[p_in, p_out],
                [p_out, p_in]]

    4. G = stochastic_block_model(sizes, probs, seed=42)

    5. // Assign labels
       Y = [0] * n_normal + [1] * n_fraud

    6. // Generate node features
       // Normal accounts: mean=0, fraud accounts: mean=shift (with overlap)
       feature_dim = 16
       shift = 0.5  // Partial overlap for realistic difficulty

       X_normal = np.random.randn(n_normal, feature_dim)
       X_fraud = np.random.randn(n_fraud, feature_dim) + shift
       X = np.vstack([X_normal, X_fraud])

    7. // Verify homophily
       actual_h = calculate_edge_homophily(G, Y)
       PRINT f"Target h: {target_homophily}, Actual h: {actual_h}"

    8. RETURN G, X, np.array(Y)

END FUNCTION
```

### 5.4 Statistical Testing Subroutine

```
FUNCTION test_hypotheses(results_df):

    INPUT:
      - results_df: DataFrame with experimental results

    OUTPUT:
      - test_results: Dictionary of hypothesis test outcomes

    PROCEDURE:

    1. test_results = {}

    2. // H1: Heterophily-aware vs homophily-assuming at h < 0.5

       2.1. low_h_data = results_df[results_df['homophily'] < 0.5]

       2.2. hetero_f1 = low_h_data[low_h_data['model_type'] == 'hetero'].groupby('seed')['f1'].max()
       2.3. homo_f1 = low_h_data[low_h_data['model_type'] == 'homo'].groupby('seed')['f1'].max()

       2.4. delta_f1 = hetero_f1.mean() - homo_f1.mean()

       2.5. t_stat, p_value = paired_ttest(hetero_f1.values, homo_f1.values)

       2.6. test_results['H1'] = {
                'delta_f1': delta_f1,
                'p_value': p_value,
                'confirmed': (delta_f1 >= 0.05) and (p_value < 0.05),
                'strength': 'strong' if (delta_f1 >= 0.075 and p_value < 0.01) else
                           'moderate' if (delta_f1 >= 0.05 and p_value < 0.05) else 'weak'
            }

    3. // H2: Monotonic relationship between h and performance gap

       3.1. deltas_by_h = []

       3.2. FOR EACH h in sorted(results_df['homophily'].unique()):
                h_data = results_df[results_df['homophily'] == h]
                hetero_mean = h_data[h_data['model_type'] == 'hetero']['f1'].mean()
                homo_mean = h_data[h_data['model_type'] == 'homo']['f1'].mean()
                deltas_by_h.append((h, hetero_mean - homo_mean))

       3.3. h_values = [d[0] for d in deltas_by_h]
            delta_values = [d[1] for d in deltas_by_h]

       3.4. spearman_rho, p_value = spearman_correlation(h_values, delta_values)

       3.5. test_results['H2'] = {
                'spearman_rho': spearman_rho,
                'p_value': p_value,
                'confirmed': (spearman_rho < -0.5) and (p_value < 0.05),
                'deltas_by_h': deltas_by_h
            }

    4. // H3: Robustness thresholds

       4.1. financial_h_range = results_df[(results_df['homophily'] >= 0.1) &
                                            (results_df['homophily'] <= 0.4)]

       4.2. hetero_min_f1 = financial_h_range[financial_h_range['model_type'] == 'hetero'].groupby('homophily')['f1'].mean().min()
       4.3. homo_min_f1 = financial_h_range[financial_h_range['model_type'] == 'homo'].groupby('homophily')['f1'].mean().min()

       4.4. test_results['H3a'] = {
                'min_hetero_f1': hetero_min_f1,
                'threshold': 0.75,
                'confirmed': hetero_min_f1 >= 0.75
            }

       4.5. test_results['H3b'] = {
                'min_homo_f1': homo_min_f1,
                'threshold': 0.65,
                'confirmed': homo_min_f1 < 0.65
            }

    5. // H4: Effect persists across imbalance ratios

       5.1. ir_deltas = []

       5.2. FOR EACH ir in sorted(results_df['imbalance_ratio'].unique()):
                ir_data = results_df[results_df['imbalance_ratio'] == ir]
                hetero_mean = ir_data[ir_data['model_type'] == 'hetero']['f1'].mean()
                homo_mean = ir_data[ir_data['model_type'] == 'homo']['f1'].mean()
                ir_deltas.append((ir, hetero_mean - homo_mean, hetero_mean, homo_mean))

       5.3. all_positive = all(d[1] > 0 for d in ir_deltas)
       5.4. min_delta = min(d[1] for d in ir_deltas)

       5.5. test_results['H4'] = {
                'ir_deltas': ir_deltas,
                'all_positive': all_positive,
                'min_delta': min_delta,
                'confirmed': all_positive and (min_delta >= 0.04)
            }

    6. // H5: FAGCN/LINKX vs H2GCN/GPR-GNN at very low homophily

       6.1. very_low_h = results_df[results_df['homophily'] < 0.2]

       6.2. negative_models = ['FAGCN', 'LINKX']
            higher_order_models = ['H2GCN', 'GPR-GNN']

       6.3. neg_f1 = very_low_h[very_low_h['model'].isin(negative_models)]['f1'].mean()
       6.4. ho_f1 = very_low_h[very_low_h['model'].isin(higher_order_models)]['f1'].mean()

       6.5. delta = neg_f1 - ho_f1

       6.6. test_results['H5'] = {
                'negative_corr_f1': neg_f1,
                'higher_order_f1': ho_f1,
                'delta': delta,
                'confirmed': delta >= 0.03
            }

    7. RETURN test_results

END FUNCTION
```

### 5.5 Visualization Subroutine

```
FUNCTION generate_visualizations(results_df, test_results):

    INPUT:
      - results_df: DataFrame with experimental results
      - test_results: Dictionary of hypothesis test outcomes

    OUTPUT:
      - figures: List of matplotlib figure objects

    PROCEDURE:

    1. figures = []

    2. // Figure 1: F1 vs Homophily by Model Type

       2.1. fig1, ax1 = plt.subplots(figsize=(10, 6))

       2.2. FOR model_type in ['hetero', 'homo']:
                type_data = results_df[results_df['model_type'] == model_type]
                grouped = type_data.groupby('homophily')['f1'].agg(['mean', 'std'])

                ax1.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                            label=f'{model_type}-aware', marker='o', capsize=3)

       2.3. ax1.axhline(y=0.75, color='red', linestyle='--', label='Threshold (0.75)')
       2.4. ax1.axvspan(0.1, 0.4, alpha=0.2, color='yellow', label='Financial Fraud Range')
       2.5. ax1.set_xlabel('Edge Homophily Ratio')
       2.6. ax1.set_ylabel('F1 Score')
       2.7. ax1.set_title('F1 Score vs Homophily: Heterophily-aware vs Homophily-assuming GNNs')
       2.8. ax1.legend()
       2.9. ax1.grid(True, alpha=0.3)

       2.10. figures.append(('f1_vs_homophily.png', fig1))

    3. // Figure 2: Performance Gap vs Homophily

       3.1. fig2, ax2 = plt.subplots(figsize=(10, 6))

       3.2. deltas = test_results['H2']['deltas_by_h']
       3.3. ax2.bar([d[0] for d in deltas], [d[1] for d in deltas], width=0.08)
       3.4. ax2.axhline(y=0.05, color='green', linestyle='--', label='H1 Threshold')
       3.5. ax2.set_xlabel('Edge Homophily Ratio')
       3.6. ax2.set_ylabel('F1 Improvement (Hetero - Homo)')
       3.7. ax2.set_title('Performance Advantage of Heterophily-aware Models')

       3.8. figures.append(('performance_gap.png', fig2))

    4. // Figure 3: Per-Model Comparison

       4.1. fig3, ax3 = plt.subplots(figsize=(12, 6))

       4.2. model_means = results_df.groupby(['model', 'homophily'])['f1'].mean().unstack()
       4.3. model_means.T.plot(ax=ax3, marker='o')
       4.4. ax3.set_xlabel('Edge Homophily Ratio')
       4.5. ax3.set_ylabel('F1 Score')
       4.6. ax3.set_title('Individual Model Performance Across Homophily Levels')
       4.7. ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

       4.8. figures.append(('per_model_comparison.png', fig3))

    5. // Figure 4: Imbalance Ratio Effect

       5.1. fig4, ax4 = plt.subplots(figsize=(10, 6))

       5.2. FOR model_type in ['hetero', 'homo']:
                type_data = results_df[results_df['model_type'] == model_type]
                grouped = type_data.groupby('imbalance_ratio')['f1'].mean()
                ax4.plot(grouped.index, grouped.values, marker='o', label=model_type)

       5.3. ax4.set_xlabel('Class Imbalance Ratio')
       5.4. ax4.set_ylabel('F1 Score')
       5.5. ax4.set_xscale('log')
       5.6. ax4.set_title('Effect of Class Imbalance on Model Performance')
       5.7. ax4.legend()

       5.8. figures.append(('imbalance_effect.png', fig4))

    6. // Figure 5: Heatmap of Results

       6.1. fig5, ax5 = plt.subplots(figsize=(12, 8))

       6.2. pivot = results_df.pivot_table(values='f1',
                                            index='model',
                                            columns='homophily',
                                            aggfunc='mean')
       6.3. sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax5)
       6.4. ax5.set_title('F1 Score Heatmap: Model vs Homophily')

       6.5. figures.append(('heatmap.png', fig5))

    7. RETURN figures

END FUNCTION
```

### 5.6 Main Execution Script

```
ALGORITHM: Main_Execution

1. // Configuration
   config = {
       'datasets': ['elliptic', 'synthetic'],
       'homophily_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
       'imbalance_ratios': [50, 100, 200, 500, 1000],
       'models_hetero': ['H2GCN', 'FAGCN', 'GPR-GNN', 'LINKX'],
       'models_homo': ['GCN', 'GraphSAGE', 'GAT'],
       'n_seeds': 5,
       'n_folds': 5,
       'hidden_dim': 64,
       'num_layers': 2,
       'learning_rate': 0.01,
       'weight_decay': 5e-4,
       'epochs': 200,
       'patience': 20,
       'device': 'cuda' if torch.cuda.is_available() else 'cpu'
   }

2. // Run main experiment
   results_df, summary = GNN_Financial_Anomaly_Detection_Experiment(
       datasets=config['datasets'],
       homophily_levels=config['homophily_levels'],
       imbalance_ratios=config['imbalance_ratios'],
       models_hetero=config['models_hetero'],
       models_homo=config['models_homo'],
       n_seeds=config['n_seeds'],
       n_folds=config['n_folds']
   )

3. // Test hypotheses
   test_results = test_hypotheses(results_df)

4. // Generate visualizations
   figures = generate_visualizations(results_df, test_results)

5. // Save outputs
   5.1. results_df.to_csv('results/full_results.csv', index=False)
   5.2. summary.to_csv('results/summary.csv')
   5.3. save_json(test_results, 'results/hypothesis_tests.json')
   5.4. FOR (filename, fig) in figures:
            fig.savefig(f'results/figures/{filename}', dpi=300, bbox_inches='tight')

6. // Print summary
   PRINT "=" * 60
   PRINT "HYPOTHESIS TEST RESULTS"
   PRINT "=" * 60

   FOR h_name, h_result in test_results.items():
       PRINT f"\n{h_name}: {'CONFIRMED' if h_result['confirmed'] else 'REJECTED'}"
       PRINT f"  Strength: {h_result.get('strength', 'N/A')}"
       PRINT f"  Key metric: {h_result.get('delta_f1', h_result.get('delta', 'N/A'))}"
       PRINT f"  p-value: {h_result.get('p_value', 'N/A')}"

7. RETURN results_df, test_results, figures

END ALGORITHM
```

---

## 6. Data Requirements

### 6.1 Primary Datasets

| Dataset | Nodes | Edges | Features | Fraud % | Est. Homophily |
|---------|-------|-------|----------|---------|----------------|
| Elliptic (Bitcoin) | ~203K | ~234K | 166 | ~2% | ~0.15-0.20 |
| IEEE-CIS | ~590K | Custom | 400+ | ~3.5% | ~0.25-0.35 |
| Synthetic (SBM) | 10K-100K | Variable | 16-64 | 1-2% | Controllable |

### 6.2 Feature Engineering (for raw transaction data)

```
Node Features (per account):
  - balance_mean, balance_std, balance_min, balance_max
  - txn_count_in, txn_count_out
  - txn_amount_in_mean, txn_amount_out_mean
  - account_age_days
  - unique_counterparties
  - txn_velocity_1d, txn_velocity_7d, txn_velocity_30d
  - deviation_from_typical_amount
  - time_since_last_txn

Edge Features (per transaction):
  - amount (log-transformed)
  - timestamp (normalized)
  - transaction_type_embedding
```

---

## 7. Expected Outcomes Table

| Hypothesis | Prediction | Confirmation Criterion | Expected Outcome |
|------------|------------|----------------------|------------------|
| H1 | Hetero > Homo by >= 5% F1 | p < 0.05, delta >= 0.05 | CONFIRM |
| H2 | Monotonic decrease of gap | rho < -0.5, p < 0.05 | CONFIRM |
| H3a | Hetero F1 >= 0.75 at h in [0.1,0.4] | min F1 >= 0.75 | CONFIRM |
| H3b | Homo F1 < 0.65 at some h in [0.1,0.4] | min F1 < 0.65 | CONFIRM |
| H4 | Gap persists across IR | delta >= 0.04 at all IR | CONFIRM |
| H5 | FAGCN/LINKX > H2GCN at h < 0.2 | delta >= 0.03 | UNCERTAIN |

---

## 8. Risk Assessment and Mitigation

### 8.1 Potential Confounds

| Confound | Mitigation |
|----------|------------|
| Dataset-specific effects | Test on multiple datasets (real + synthetic) |
| Hyperparameter sensitivity | Grid search with fixed budget for all models |
| Random variation | 5 seeds, report mean +/- std |
| Label noise | Sensitivity analysis with injected noise |
| Temporal leakage | Strict temporal train/test splits |

### 8.2 Alternative Explanations

If hypotheses are not confirmed, consider:
1. Feature signal dominates over structural signal
2. Class imbalance handling varies across architectures
3. Implementation differences in baseline models
4. Hyperparameter optimization favoring certain architectures

---

## 9. Timeline and Computational Budget

| Phase | Duration | GPU Hours |
|-------|----------|-----------|
| Data preparation | 1 day | 0 |
| Synthetic data generation | 0.5 day | 2 |
| Model implementation | 2 days | 0 |
| Main experiments (7 models x 8 h x 5 IR x 5 seeds) | 3 days | ~100 |
| Statistical analysis | 1 day | 0 |
| Visualization | 0.5 day | 0 |
| **Total** | **8 days** | **~102** |

---

## 10. Deliverables Checklist

- [ ] Full results CSV with all experimental runs
- [ ] Summary statistics by model, homophily, and IR
- [ ] Hypothesis test results (p-values, effect sizes, confidence intervals)
- [ ] 5 visualization figures
- [ ] Trained model checkpoints (optional, for reproducibility)
- [ ] Configuration files for exact reproducibility
