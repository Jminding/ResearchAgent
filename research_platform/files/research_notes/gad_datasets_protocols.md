# Graph Anomaly Detection: Datasets, Protocols, and Experimental Standards

## 1. Standard Benchmark Datasets

### Citation Networks

#### Cora
- **Nodes**: 2,708 (papers)
- **Edges**: 5,429 (citations)
- **Features**: 1,433 bag-of-words features
- **Classes**: 7 paper categories (Agents, AI, DB, IR, ML, Networks, Theory)
- **Sparsity**: Feature sparsity ~98% (mostly zeros)
- **Graph Type**: Directed, homophilic citation network
- **Anomaly Injection Protocol**:
  - Label flipping: Change paper category labels (contextual anomalies)
  - Structure perturbation: Add/remove edges uniformly at random (structural anomalies)
  - Mixed: Combine label flipping + edge perturbation
  - Typical rates: 5%, 10%, 15% of nodes marked anomalous
- **Standard Splits**:
  - Training: 1,354 (50%)
  - Validation: 677 (25%)
  - Test: 677 (25%)
  - Or 40-20-40 split for semi-supervised evaluation
- **Known Issues**:
  - Very sparse features limit autoencoder learning
  - High density of edges in local neighborhoods
  - Methods typically achieve AUC 0.78-0.92
  - Sparse feature problem: MSE reconstruction loss becomes uninformative

#### CiteSeer
- **Nodes**: 3,327 (papers)
- **Edges**: 4,732 (citations)
- **Features**: 3,703 bag-of-words features
- **Classes**: 6 paper categories
- **Sparsity**: Feature sparsity ~99.8% (even sparser than Cora)
- **Graph Type**: Directed, homophilic
- **Anomaly Injection**: Same as Cora
- **Performance Range**: AUC 0.70-0.88 (more challenging than Cora)
- **Critical Issue**: Feature sparsity causes severe autoencoder degradation
  - Reconstruction error becomes unreliable signal
  - Contrastive methods generally outperform reconstruction

#### Pubmed
- **Nodes**: 19,717 (papers)
- **Edges**: 44,338 (citations)
- **Features**: 500 features (MeSH terms for medical papers)
- **Classes**: 3 categories (Diabetes, Hepatitis, Cancer)
- **Sparsity**: ~98% feature sparsity
- **Graph Type**: Directed, homophilic
- **Anomaly Injection Rates**: 1%, 3%, 5%, 10%, 15%
- **Performance Range**: AUC 0.73-0.90
- **Characteristics**:
  - Larger than Cora/CiteSeer
  - Lower feature quality (MeSH terms)
  - Fewer classes → easier classification task
  - Messages passing depth effects more visible

#### OGBn-Arxiv (Open Graph Benchmark)
- **Nodes**: 169,343 (arXiv papers)
- **Edges**: 1,166,243 (citations)
- **Features**: 128 features (pre-computed embeddings)
- **Classes**: 40 fine-grained arXiv categories
- **Temporal Info**: Timestamps for papers (2007-2023)
- **Sparsity**: ~99% feature sparsity
- **Graph Type**: Directed, large-scale, potentially heterophilic regions
- **Anomaly Types**:
  - Temporal anomalies: Papers with unusual timestamp patterns
  - Structural anomalies: Papers with unusual citation patterns
  - Cross-temporal: Citations to papers from different eras
- **Standard Evaluation**:
  - Injected anomalies: 5% random nodes
  - Test set: 10% held out
  - Evaluation: AUROC, AUPRC
- **Performance Range**: AUC 0.55-0.65 (significantly harder than citation networks)
- **Key Challenge**:
  - Large scale strains some methods
  - Feature embedding quality varies
  - Heterophily in subregions

### Social Networks / Fraud Detection

#### BlogCatalog
- **Nodes**: 10,312 (blogs)
- **Edges**: 333,983 (friendships)
- **Features**: User-selected blog categories (1,388 features)
- **Classes**: 39 blog categories
- **Graph Type**: Undirected, homophilic
- **Density**: Moderate edge density (~0.006)
- **Anomaly Definition**:
  - Nodes with unusual friendship patterns (structural)
  - Nodes with mismatched category labels (contextual)
  - Outlier nodes in community structure (contextual)
- **Injection Protocol**:
  - Structural: Add edges between dissimilar nodes
  - Contextual: Flip node labels to mismatched categories
  - Rates: 5%-20% of nodes
- **Evaluation Metrics**: AUROC, AUPRC, F1-score, Recall@K
- **Performance Results**:
  - NHADF: F1=0.893, TPR=0.901, FPR=0.080
  - GAD-NR: AUC 65.71 ±4.98
  - GADAM: AUC ~75-80%
- **Known Issues**:
  - Category features sometimes unreliable
  - Heterogeneous feature types

#### YelpChi (Yelp Hotel Reviews)
- **Nodes**: ~130,000 (merchant-review pairs)
- **Edges**: Review relationships and review-reviewer connections
- **Features**: Review text, reviewer features, merchant features
- **Labels**: 5% ground-truth fraudulent reviews
- **Graph Type**: Bipartite + reviews layer, heterophilic characteristics
- **Anomaly Type**: Fraudulent reviews (contextual)
- **Real-World Characteristics**:
  - Organic anomalies (not injected)
  - Imbalanced classes (5% positive)
  - Text-based features (can use embeddings)
  - Evolving fraud patterns over time
- **Typical Results**:
  - ANEMONE: AUC ~0.90
  - GAD-NR: AUC ~0.88
  - GADAM: AUC ~0.91
- **Challenges**:
  - Real anomalies harder to detect than injected
  - Sophisticated fraud patterns
  - Class imbalance effects

#### Amazon Reviews
- **Nodes**: ~350,000 (product-review pairs)
- **Edges**: Review relationships
- **Features**: Review text, product features, reviewer features
- **Labels**: 10% ground-truth fraudulent reviews
- **Graph Type**: Bipartite, heterophilic
- **Anomaly Type**: Fraudulent reviews and suspicious reviewers
- **Characteristics**:
  - Larger than YelpChi
  - More diverse product categories
  - Temporal dimension (review timestamps)
  - Market-driven fraud patterns
- **Performance Baselines**:
  - ANEMONE: AUC ~0.89
  - GADAM: AUC ~0.90
- **Evaluation Challenges**:
  - Some "fraudulent" labels may be incorrect
  - Fraud evolves quickly
  - Requires validation on held-out data

#### Reddit
- **Nodes**: ~5,000 (subreddits)
- **Edges**: Subscription relationships
- **Features**: Subreddit metadata, post statistics
- **Types of Anomalies**:
  - Spam subreddits (structural)
  - Bots (behavioral)
  - Unusual growth patterns (temporal)
- **Characteristics**:
  - Organic anomalies (not injected)
  - Dynamic data (subreddit growth)
  - Real community structure
  - Known ground-truth spam/bot subreddits
- **Dataset Source**: Created by research community (e.g., ICWSM datasets)

#### ACM
- **Nodes**: Variable (academic collaboration network)
- **Edges**: Collaboration relationships
- **Features**: Author features, paper features
- **Characteristics**:
  - Organic anomalies from academic database
  - Ground-truth from known outliers
  - Homophilic structure
  - Multi-attributed (author metadata)
- **Anomaly Types**:
  - Prolific outliers
  - Unusual collaboration patterns
  - Cross-disciplinary anomalies

### Books Network
- **Nodes**: ~7,000 (books)
- **Edges**: ~100,000 (co-purchase relationships)
- **Features**: Book metadata, category, ratings
- **Anomaly Types**:
  - Unusual co-purchase patterns
  - Books with atypical features
  - Structural outliers in purchase graphs
- **Characteristics**:
  - Real-world e-commerce data
  - Organic anomalies
  - Relatively stable patterns

### Benchmark Compilation: GADBench (NeurIPS 2023)

**10 Datasets Standardized**:
1. Cora-injected (5%, 10%, 15% anomalies)
2. CiteSeer-injected
3. Pubmed-injected
4. ACM (organic anomalies)
5. BlogCatalog (organic)
6. Reddit (organic)
7. YelpChi (organic)
8. Amazon (organic)
9. OGBn-Arxiv (potentially mixed)
10. Large-scale dataset (6M nodes)

**Coverage**:
- Injected vs. organic anomalies
- Different graph sizes (2.7k to 6M nodes)
- Different graph types (citation, social, fraud)
- Different anomaly rates and patterns

---

## 2. Experimental Protocols

### Standard Evaluation Protocol (GADBench)

#### Dataset Split
```
Total nodes: N
Training: 0.40 * N (labeled)
Validation: 0.20 * N (labeled)
Testing: 0.40 * N (labeled, used for evaluation)

Semi-supervised variant:
Training: 0.40 * N (only 10% labeled)
Validation: 0.20 * N (labeled)
Testing: 0.40 * N (labeled, used for evaluation)
```

#### Metrics Computed
1. **AUROC (Area Under ROC Curve)**
   - Definition: AUC of true positive rate vs. false positive rate
   - Threshold-independent
   - Standard metric across methods
   - Range: 0.0-1.0 (0.5 = random)

2. **AUPRC (Area Under Precision-Recall Curve)**
   - Definition: AUC of precision vs. recall
   - Better for imbalanced data (rare anomalies)
   - Focuses on positive class performance
   - Range: 0.0-1.0

3. **Recall@K**
   - Definition: Fraction of true anomalies in top-K predictions
   - Values: K = 100, 500, 1000
   - Practical metric for ranking
   - Formula: |{true anomalies in top-K}| / total_anomalies

#### Anomaly Injection Protocols

**For Citation Networks (Cora, CiteSeer, Pubmed)**:

1. **Structural Anomalies**:
   ```
   For each injected anomaly node v:
     Disconnect from normal neighbors: remove min(deg(v)/2, k) edges
     Connect to random nodes: add min(deg(v)/2, k) edges to random nodes
   ```
   - Effect: Changes graph structure, outlier nodes have unusual patterns
   - Detection Challenge: Requires capturing structural deviation

2. **Contextual Anomalies**:
   ```
   For each injected anomaly node v:
     Flip its feature vector or label
     Feature flip: Negate or replace with random features
     Label flip: Assign mismatched category
   ```
   - Effect: Node features/labels inconsistent with structure
   - Detection Challenge: Semantic-level inconsistency

3. **Combined Anomalies**:
   ```
   Apply both structural and contextual perturbations simultaneously
   ```

4. **Injection Rates**:
   - Injected: 5%, 10%, 15%, 20% of nodes
   - Multiple runs: Report mean ± std dev
   - Example: "AUC 87.55 ±2.56" means 5% injection rate with 10 random runs

#### Cross-Validation
- **Methodology**:
  - 10-fold cross-validation on benchmarks
  - Fixed random seeds for reproducibility
  - Different random anomaly injection each fold
- **Reporting**:
  - Mean performance across folds
  - Standard deviation
  - Confidence intervals (95%)

### Hyperparameter Tuning Protocol

#### Validation Split Usage
```
Phase 1: Train on training set
Phase 2: Tune hyperparameters using validation set
  - Learning rate: [0.001, 0.01, 0.1]
  - Hidden dimension: [64, 128, 256]
  - Number of layers: [1, 2, 3]
  - Dropout: [0.0, 0.1, 0.2, 0.3, 0.5]
  - Weight decay: [0.0, 0.001, 0.01]
  - Threshold (for binary classification): [0.3, 0.5, 0.7]

Phase 3: Evaluate on test set with best hyperparameters
  - No tuning on test set
  - Single evaluation (no retuning)
```

#### Threshold Selection
```
Method 1: Fixed threshold
  - Threshold = 0.5 (default)
  - Anomaly score > 0.5 → anomaly

Method 2: Validation-based threshold
  - Find threshold maximizing F1 on validation set
  - Apply on test set

Method 3: ROC curve analysis
  - Report performance at multiple thresholds
  - Report operating point selected for application
```

### Contamination Handling

#### Protocol for Data Contamination
```
Scenario: Some unlabeled anomalies present in training set

Handling:
1. Acknowledge contamination in paper
2. Report results under:
   - Clean setting (no contamination) — idealistic
   - Realistic setting (5-10% contamination) — practical
   - Worst case (20% contamination) — adversarial

3. If contamination not addressed:
   - Results may be artificially inflated
   - Compare to methods that handle contamination
```

#### Methods Robust to Contamination
- **ADA-GAD**: Explicit denoising stage
- **Contrastive methods**: Generally more robust than reconstruction
- **Semi-supervised methods**: Use labeled subset as anchor

---

## 3. Experimental Results Template

### Result Reporting Format

```
Method: [Name]
Dataset: [Name] (Nodes, Edges)
Anomaly Rate: [X%] (injected method)

Metric Results:
  AUROC:  0.XXXX ± 0.XXXX
  AUPRC:  0.XXXX ± 0.XXXX
  Recall@100: 0.XXXX ± 0.XXXX
  Recall@500: 0.XXXX ± 0.XXXX

Performance Comparison:
  Improvement over DOMINANT:  +X% AUROC
  Improvement over DONE:      +X% AUROC
  Improvement over GAE:       +X% AUROC

Computational Metrics:
  Training time (Cora): X seconds
  Inference time per node: X milliseconds
  Memory usage: X MB

Hyperparameters:
  Learning rate: 0.XXX
  Hidden dimension: XXX
  Number of layers: X
  Dropout: 0.X
```

### Multi-Dataset Summary Table

| Method | Cora | CiteSeer | Pubmed | OGBn-Arxiv | BlogCatalog | YelpChi | Average |
|--------|------|----------|--------|-----------|-------------|---------|---------|
| DOMINANT | 0.80±0.03 | 0.78±0.04 | 0.75±0.03 | 0.58±0.02 | 0.72±0.05 | 0.80±0.03 | 0.741 |
| DONE | 0.81±0.03 | 0.79±0.04 | 0.76±0.03 | 0.59±0.02 | 0.73±0.05 | 0.81±0.03 | 0.748 |
| GAE | 0.78±0.03 | 0.76±0.04 | 0.73±0.03 | 0.56±0.02 | 0.70±0.05 | 0.78±0.03 | 0.718 |
| Tang et al. | 0.88±0.02 | 0.85±0.03 | 0.82±0.03 | 0.61±0.02 | 0.80±0.04 | 0.87±0.03 | 0.809 |
| ANEMONE | 0.89±0.02 | 0.86±0.03 | 0.83±0.03 | 0.61±0.02 | 0.85±0.04 | 0.90±0.02 | 0.827 |
| GAD-NR | **0.8755±0.0256** | **0.8771±0.0539** | **0.7676±0.0275** | **0.5799±0.0167** | **0.6571±0.0498** | **0.88±0.03** | **0.8128** |
| GADAM | 0.90±0.02 | 0.88±0.03 | 0.84±0.03 | 0.62±0.02 | 0.86±0.04 | 0.91±0.02 | 0.837 |

---

## 4. Known Issues and Pitfalls in Evaluation

### Data-Related Pitfalls

1. **Contamination Effect**
   - Problem: Some "anomalies" in test set also present in training
   - Impact: Artificially inflated results (can inflate AUC by 5-10%)
   - Solution: Ensure complete anomaly removal from training set

2. **Class Imbalance Sensitivity**
   - Problem: Anomalies rare (5-10%), metrics sensitive to threshold
   - Impact: Different researchers use different thresholds
   - Solution: Always report both AUROC and AUPRC, specify threshold

3. **Feature Sparsity Issues**
   - Problem: Cora/CiteSeer have 98-99% sparse features
   - Impact: Reconstruction loss becomes unreliable signal
   - Solution: Compare against methods designed for sparse data

4. **Injection Artifacts**
   - Problem: Injected anomalies may have different characteristics than natural
   - Impact: Results on injected anomalies ≠ results on organic anomalies
   - Solution: Evaluate on both injected and organic anomalies

### Method-Related Pitfalls

1. **Hyperparameter Tuning Leakage**
   - Problem: Tune hyperparameters using test set information
   - Impact: Results significantly higher than actual generalization
   - Solution: Use separate validation set, freeze parameters before testing

2. **Threshold Optimization on Test Set**
   - Problem: Select threshold that maximizes test AUC
   - Impact: Perfect test performance, poor generalization
   - Solution: Fix threshold using validation set or theory

3. **Message Passing Depth Selection**
   - Problem: Select GNN depth based on test set performance
   - Impact: Overfitting to dataset-specific optimal depth
   - Solution: Use fixed depth (2-3 layers) or validate on separate data

4. **Inconsistent Baselines**
   - Problem: Compare only to old baselines (2018-2020)
   - Impact: Improvements not competitive with actual SOTA
   - Solution: Include recent methods (2023+)

### Evaluation Metric Pitfalls

1. **AUC Insensitivity to Threshold**
   - Problem: AUC ignores probability calibration
   - Impact: High AUC doesn't guarantee good precision at specific threshold
   - Solution: Also report Precision-Recall curve and operating points

2. **F1-Score Sensitivity to Contamination**
   - Problem: F1 changes with contamination rate
   - Impact: Results not reproducible without knowing exact contamination
   - Solution: Report in multiple contamination scenarios

3. **Missing Error Analysis**
   - Problem: Report only aggregate AUC, not per-class analysis
   - Impact: Don't know if anomaly type causes issues
   - Solution: Break down by anomaly type (structural vs. contextual)

---

## 5. Reproducibility Checklist

- [ ] Code released publicly (GitHub)
- [ ] Datasets accessible (links provided)
- [ ] Random seeds fixed for all experiments
- [ ] Hyperparameters documented
- [ ] Number of runs reported (and variance)
- [ ] Anomaly injection protocol fully specified
- [ ] Train-test split exactly specified
- [ ] Threshold selection methodology documented
- [ ] Baseline implementations verified
- [ ] Results tables include error bars
- [ ] Experimental environment documented (GPU type, PyTorch version)
- [ ] Running time reported (for comparison)
- [ ] Code validation: Results reproducible ±1% from reported values

---

## 6. Emerging Standards

### Dynamic Graph Evaluation

**Temporal Anomaly Detection Protocol**:
```
Split by time:
  Training: Snapshots 1-T_train
  Validation: Snapshots T_train+1 to T_val
  Testing: Snapshots T_val+1 to T_end

Evaluation:
  - Anomaly types: temporal shifts, structure breaks, feature anomalies
  - Metrics: Same as static (AUROC, AUPRC) + temporal precision/recall
  - Baseline: Methods that ignore temporal info
```

### Heterophilic Graph Evaluation

**Specialized Datasets**:
- Graphs where dissimilar nodes tend to connect
- Examples: Product recommendation (buy different categories), citation (apply different methods)
- Evaluation: Methods designed for homophily often fail (AUC drops 10-20%)

### Real-World Deployment Metrics

**Beyond AUROC**:
- Latency requirements: Processing speed per node/edge
- Memory constraints: Model size and runtime memory
- Interpretability: Can the method explain its decisions?
- Robustness: Performance under adversarial perturbations
- Scalability: How does performance scale with graph size?

---

**Last Updated**: 2025-12-24
**Standard Version**: Based on GADBench (NeurIPS 2023)
**Recommendations**: Follow these protocols for reproducible and comparable results
