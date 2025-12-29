# Statistical Analysis Methodology for Quantum Error Correction

**Document Purpose:** Detailed methodology for threshold extraction and hypothesis testing

---

## 1. Threshold Extraction Methods

### 1.1 Exponential Scaling Fit

**Model:**
```
P_L(p, d) = A(p) · exp(-α(p) · d)
```

**Linearized Form:**
```
ln(P_L) = ln(A) - α · d
```

**Fitting Procedure:**
1. For each physical error rate p:
   - Extract P_L values for d = 3, 5, 7
   - Perform weighted linear regression: ln(P_L) vs d
   - Extract slope α(p) and intercept ln(A(p))
   - Calculate R² and residuals

2. Find threshold p_th:
   - Plot α(p) vs p
   - Interpolate to find p where α(p) = 0
   - Alternative: Find intersection of P_L curves

3. Error Propagation:
   ```
   σ_α² = Σ[w_i · (ln(P_L,i) - ln(A) + α·d_i)²] / (n-2)
   σ_p_th = |∂p_th/∂α| · σ_α
   ```

**Acceptance Criteria:**
- R² > 0.95 for valid exponential regime
- At least 3 data points per fit
- Residuals show no systematic bias

### 1.2 Bootstrap Confidence Intervals

**Algorithm:**
```
FOR b = 1 to B (typically B=1000):
  1. Resample data with replacement
  2. Fit exponential scaling
  3. Extract α_b and p_th,b
  4. Store values

5. Calculate percentile CI:
   CI_lower = 2.5th percentile of {p_th,b}
   CI_upper = 97.5th percentile of {p_th,b}
```

**Output:**
- p_th = median({p_th,b})
- 95% CI: [CI_lower, CI_upper]
- Bootstrap distribution plot

### 1.3 Curve Intersection Method

**Procedure:**
1. Interpolate P_L(p, d) curves for each distance
2. Find p* where |P_L(p*, d₁) - P_L(p*, d₂)| < ε
3. Check consistency across all pairs (d₁, d₂)
4. Average intersection points with uncertainty

**Validation:**
- All pairwise intersections should agree within error bars
- Threshold should be consistent with α(p)=0 method

---

## 2. Performance Comparison Statistical Tests

### 2.1 Two-Proportion Z-Test

**Hypothesis:**
- H₀: p_RL = p_MWPM (success rates equal)
- H₁: p_RL ≠ p_MWPM (two-sided test)

**Test Statistic:**
```
z = (p̂_RL - p̂_MWPM) / √[p̂(1-p̂)(1/n_RL + 1/n_MWPM)]

where p̂ = (x_RL + x_MWPM) / (n_RL + n_MWPM)
```

**Decision Rule:**
- Reject H₀ if |z| > 1.96 (α=0.05, two-tailed)
- p-value = 2·Φ(-|z|)

**Effect Size (Cohen's h):**
```
h = 2·[arcsin(√p̂_RL) - arcsin(√p̂_MWPM)]

Interpretation:
  |h| < 0.2: small effect
  0.2 ≤ |h| < 0.5: medium effect
  |h| ≥ 0.5: large effect
```

### 2.2 Confidence Interval on Difference

**Formula:**
```
CI = (p̂_RL - p̂_MWPM) ± z_α/2 · √[p̂_RL(1-p̂_RL)/n_RL + p̂_MWPM(1-p̂_MWPM)/n_MWPM]
```

**Interpretation:**
- If 0 ∉ CI: significant difference
- Width of CI indicates precision

### 2.3 Power Analysis

**Required Sample Size:**
```
n = [z_α/2·√(2p̄(1-p̄)) + z_β·√(p₁(1-p₁) + p₂(1-p₂))]² / (p₁-p₂)²

where:
  α = Type I error rate (0.05)
  β = Type II error rate (0.20 for 80% power)
  p₁, p₂ = expected success rates
  p̄ = (p₁+p₂)/2
```

**Minimum Detectable Effect:**
Given fixed n, calculate smallest Δp detectable with 80% power.

---

## 3. Goodness of Fit Metrics

### 3.1 R² (Coefficient of Determination)

```
R² = 1 - SS_res / SS_tot

where:
  SS_res = Σ(y_i - ŷ_i)²
  SS_tot = Σ(y_i - ȳ)²
```

**Thresholds:**
- R² > 0.95: excellent fit
- 0.90 < R² ≤ 0.95: good fit
- 0.80 < R² ≤ 0.90: acceptable fit
- R² ≤ 0.80: poor fit (model invalid)

### 3.2 Chi-Squared Test

```
χ² = Σ[(O_i - E_i)² / E_i]

where:
  O_i = observed value
  E_i = expected value from fit
```

**Decision:**
- Compare χ² to critical value χ²_crit(α=0.05, df=n-k)
- If χ² < χ²_crit: accept model
- p-value = P(X > χ² | H₀)

### 3.3 Residual Analysis

**Tests:**
1. Normality: Shapiro-Wilk test on residuals
2. Homoscedasticity: Plot residuals vs fitted values
3. Independence: Durbin-Watson statistic

**Warning Signs:**
- Systematic patterns in residuals
- Heteroscedasticity (funnel shape)
- Outliers with high leverage

---

## 4. Bayesian Analysis (Optional Enhancement)

### 4.1 Bayesian Threshold Estimation

**Model:**
```
P_L(p, d) ~ LogNormal(μ(p,d), σ)
μ(p,d) = ln(A(p)) - α(p)·d

Prior on p_th: p_th ~ Uniform(0.05, 0.15)
```

**MCMC Sampling:**
- Use Stan or PyMC3
- Generate posterior distribution P(p_th | data)
- Report median and 95% credible interval

**Advantages:**
- Natural uncertainty quantification
- Can incorporate prior knowledge
- Handles small sample sizes better

### 4.2 Model Comparison

**Bayes Factor:**
```
BF = P(data | M₁) / P(data | M₂)

Interpretation:
  BF > 10: strong evidence for M₁
  3 < BF ≤ 10: moderate evidence
  1 < BF ≤ 3: weak evidence
  BF ≈ 1: no preference
```

**Application:**
Compare RL vs MWPM using Bayesian model selection.

---

## 5. Latency Analysis

### 5.1 Statistical Comparison

**Test:** Two-sample t-test (if latencies normally distributed)
- H₀: μ_RL = μ_MWPM
- H₁: μ_RL ≠ μ_MWPM

**Alternative:** Mann-Whitney U test (non-parametric)
Use if latency distributions are skewed.

### 5.2 Practical Significance

Beyond statistical significance, assess:
- Speedup factor: t_MWPM / t_RL
- Acceptable overhead for accuracy gain
- Wall-clock time for realistic workloads

**Cost-Benefit Analysis:**
```
Efficiency Score = (Δ Accuracy) / (Δ Latency)

Higher score indicates better tradeoff.
```

---

## 6. Trajectory Clustering Analysis

### 6.1 DBSCAN Parameters

**Algorithm:**
```
DBSCAN(trajectories, ε, minPts):
  - ε: maximum distance for neighborhood
  - minPts: minimum points to form cluster
```

**Selection:**
- Use k-distance plot to determine ε
- Silhouette score to validate clustering quality

### 6.2 Coherence Metrics

**Phase Coherence Time T₂*:**
```
|ρ(t)| = |ρ(0)| · exp(-t/T₂*)

Fit exponential decay to trajectory correlations.
```

**Angular Deviation:**
```
δθ = √[<(θ_obs - θ_ideal)²>]

Low δθ indicates coherent evolution.
```

### 6.3 Statistical Testing

Compare trajectory properties between:
- Correctable vs uncorrectable errors
- Different error types (X, Y, Z)
- RL-decoded vs MWPM-decoded cases

Use multivariate ANOVA or permutation tests.

---

## 7. Error Graph Analysis

### 7.1 Graph Metrics

**Centrality Measures:**
- Degree centrality: Number of connections
- Betweenness centrality: Node importance in paths
- Eigenvector centrality: Influence based on neighbors

**Clustering:**
```
C = (# triangles) / (# connected triples)
```

**Small-World Index:**
```
SWI = (C/C_rand) / (L/L_rand)

where L = average path length
```

### 7.2 Graph Comparison

**Edge Weight Correlation:**
```
r = corr(W_RL, W_MWPM)

High r suggests RL learned similar structure.
```

**Spectral Analysis:**
- Compare eigenvalue distributions of adjacency matrices
- Graph Laplacian eigenvectors reveal community structure

### 7.3 Information-Theoretic Measures

**Mutual Information:**
```
MI(RL, MWPM) = Σ P(w_RL, w_MWPM) log[P(w_RL, w_MWPM) / (P(w_RL)P(w_MWPM))]

Measures dependence between learned weights.
```

---

## 8. Multiple Testing Correction

### 8.1 Problem
Testing multiple hypotheses inflates Type I error rate.

### 8.2 Bonferroni Correction

**Adjusted α:**
```
α_adj = α_family / m

where m = number of tests
```

**Conservative but simple.**

### 8.3 Benjamini-Hochberg (FDR Control)

**Procedure:**
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ p_m
2. Find largest k where p_k ≤ (k/m)·α
3. Reject H₀ for tests 1, ..., k

**Less conservative, controls false discovery rate.**

### 8.4 Application

When testing RL vs MWPM across multiple (p, d) conditions, apply correction to maintain family-wise error rate.

---

## 9. Reporting Standards

### 9.1 Required Elements

**For Each Hypothesis:**
1. Clear statement of H₀ and H₁
2. Test statistic and its value
3. p-value (exact, not just <0.05)
4. Effect size with interpretation
5. Confidence interval (95%)
6. Sample size and power

**For Each Model Fit:**
1. Model equation and parameters
2. Parameter estimates with standard errors
3. Goodness of fit (R², χ², etc.)
4. Residual plots
5. Assumptions checked

### 9.2 Visualization Standards

**Plots Must Include:**
- Error bars (standard error or confidence intervals)
- Axis labels with units
- Legend identifying all curves
- Grid for readability
- Caption explaining key findings

**Recommended Plots:**
1. P_L vs p for each d (log-linear scale)
2. α(p) vs p with threshold marked
3. Success rate comparison (bar chart with error bars)
4. Latency distributions (box plots or violin plots)
5. Trajectory clusters (2D projection)
6. Graph structure (force-directed layout)

### 9.3 Tabular Data

**Standard Table Format:**
| Condition | RL Metric | MWPM Metric | Difference | 95% CI | p-value | Effect Size |
|-----------|-----------|-------------|------------|--------|---------|-------------|
| (p, d) | X ± σ | Y ± σ | Δ | [L, U] | p | h or d |

---

## 10. Quality Checklist

Before finalizing analysis, verify:

- [ ] All data sources documented
- [ ] Sample sizes reported
- [ ] Missing data handled appropriately
- [ ] Assumptions of statistical tests verified
- [ ] Multiple testing correction applied if needed
- [ ] Effect sizes reported (not just p-values)
- [ ] Confidence intervals calculated
- [ ] Results reproducible from provided data
- [ ] Limitations acknowledged
- [ ] Alternative explanations considered
- [ ] Practical significance addressed
- [ ] Figures publication-ready
- [ ] Tables complete and readable
- [ ] Code/methods available for verification

---

## References

**Statistical Methods:**
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
- Wasserman, L. (2004). All of Statistics.

**Quantum Error Correction:**
- Fowler, A. G., et al. (2012). Surface codes: Towards practical large-scale quantum computation. Phys. Rev. A.
- Dennis, E., et al. (2002). Topological quantum memory. J. Math. Phys.

**Machine Learning for QEC:**
- Vargas-Hernández, R. A., et al. (2020). Neural Network Quantum Error Correction.
- Nautrup, H. P., et al. (2019). Optimizing Quantum Error Correction Codes with Reinforcement Learning.

---

**Document Version:** 1.0
**Last Updated:** 2025-12-22
**Purpose:** Methodology reference for quantum decoder analysis
