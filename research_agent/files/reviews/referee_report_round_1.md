# Referee Report: Round 1

**Manuscript Title:** An Empirical Comparison of Geometric Brownian Motion and Heston Stochastic Volatility Models for Stock Price Dynamics: Evidence from AAPL (2013--2025)

**Date of Review:** December 22, 2025

**Reviewer:** Peer Review Agent

---

## 1. Summary of the Paper

This manuscript conducts an empirical comparison between Geometric Brownian Motion (GBM) and the Heston stochastic volatility model using 13 years of Apple Inc. (AAPL) daily returns (2013-2025, N=3,262). The authors employ maximum likelihood estimation (closed-form for GBM, particle filter-based for Heston) and evaluate models using likelihood ratio tests, AIC/BIC, residual diagnostics, and out-of-sample variance forecasting. The central finding is that GBM outperforms Heston across all metrics, contradicting theoretical expectations. The authors attribute this to weak volatility clustering, parameter identifiability issues, and potential estimation failures.

---

## 2. Overall Assessment

The manuscript addresses an interesting and practically relevant question in financial econometrics. The writing is generally clear, the methodology is reasonably well-described, and the paper is structured professionally. However, several significant methodological and interpretive issues require attention before the paper can be considered for publication. The most critical concern is that the negative likelihood ratio test statistic (-66.69) indicates a fundamental failure in the Heston estimation procedure, which undermines the validity of all subsequent model comparisons and conclusions.

---

## 3. Major Issues (Must-Fix for Acceptance)

### Major Issue 1: Estimation Failure Invalidates Primary Conclusions

**Problem:** The paper reports a negative likelihood ratio test statistic (LRT = -66.69), meaning the Heston model achieved *lower* log-likelihood than GBM despite having four additional parameters. As the authors correctly note (Section 6.2), this is "theoretically impossible under proper nested model testing." This indicates that the particle filter MLE did not converge to the global maximum.

**Why it matters:** If the Heston model was not properly estimated, then ALL conclusions comparing GBM to Heston are invalid. The entire premise of the paper collapses: we cannot claim that GBM is "superior" when Heston was never fairly evaluated. The paper is essentially comparing an optimized GBM to a poorly-optimized Heston.

**Required action:**
- **Experimentalist** must re-estimate the Heston model using alternative methods: (a) increase particle count from 2,000 to 10,000+; (b) implement multiple random restarts from different initializations; (c) consider alternative estimators such as MCMC (Eraker et al. 2003), simulated method of moments, or quasi-maximum likelihood based on realized volatility.
- **Theorist** must clarify whether GBM and Heston are truly nested models in the parameter space (they are not strictly nested; Heston reduces to GBM only asymptotically as xi->0 while maintaining Feller condition).
- The paper should not be publishable until either (a) Heston is properly estimated with LL >= LL_GBM, or (b) the authors provide rigorous justification for why this anomaly represents genuine data characteristics rather than computational failure.

### Major Issue 2: Single-Asset Analysis Limits Generalizability

**Problem:** The entire analysis uses only AAPL stock. The title and abstract make broad claims about "stock price dynamics" and implications for "financial econometrics," but findings from a single technology stock during a specific period cannot be generalized.

**Why it matters:** AAPL is a large-cap, highly liquid, heavily-traded stock with unique characteristics (product announcement cycles, massive institutional ownership). Results may not extend to small-caps, different sectors, or international markets. The claim in the conclusion that "model sophistication must be matched to data informativeness" is presented as a general principle but is supported by only one data point.

**Required action:**
- **Data-collector** should obtain return data for at least 4-6 additional assets spanning different market caps, sectors, and volatility regimes (e.g., a small-cap stock, an index like S&P 500, a high-volatility sector like biotech, an international market).
- **Experimentalist** must replicate the analysis across these assets.
- **Report-writer** should revise claims to acknowledge the single-asset limitation more prominently or expand the analysis.

### Major Issue 3: Inadequate Treatment of Model Nesting

**Problem:** The paper treats GBM and Heston as nested models for LRT purposes, but this is technically incorrect. GBM is not a special case of Heston achievable by setting specific parameter values within the Heston parameter space. To recover GBM from Heston, one would need xi=0, but this violates the Feller condition and leads to a degenerate variance process.

**Why it matters:** The LRT is only valid for properly nested models. If the models are not nested, the chi-square approximation is incorrect, and the test statistic distribution is unknown. This affects the interpretation of the LRT p-value.

**Required action:**
- **Theorist** must address this issue explicitly. Either (a) acknowledge that the models are not strictly nested and use appropriate non-nested model comparison tests (e.g., Vuong test, Clarke test), or (b) reformulate the null hypothesis more carefully.
- The LRT section (Eq. 14 and Section 6.2) requires revision.

### Major Issue 4: Out-of-Sample Validation Methodology Weakness

**Problem:** The out-of-sample validation uses a simple 80/20 train/test split with 22-day-ahead variance forecasting. This methodology has several issues: (a) a single split is highly sensitive to the specific split point; (b) realized variance computed as mean squared returns is a noisy estimator; (c) comparing constant GBM variance to Heston mean-reversion forecast is not a fair test since Heston's advantage lies in conditioning on current variance state.

**Why it matters:** The claim that GBM has "superior out-of-sample performance" rests on this single experiment with marginal differences (RMSE 0.0926 vs 0.0928, a 0.2% difference). This is likely within noise given the evaluation methodology.

**Required action:**
- **Analyst** should implement rolling-window or expanding-window cross-validation with multiple forecast origins.
- **Data-collector** could incorporate realized volatility measures (e.g., from 5-minute returns) as better benchmarks.
- **Experimentalist** must provide confidence intervals or bootstrap standard errors for RMSE differences to establish statistical significance.

### Major Issue 5: Missing Formal Comparison with GARCH-Type Models

**Problem:** The literature review mentions GARCH models (Bollerslev 1986, Engle 1982) and acknowledges that GARCH-type conditional heteroskedasticity is "ubiquitous" in financial time series. However, the empirical analysis only compares GBM and Heston, omitting GARCH entirely.

**Why it matters:** GARCH models are the standard discrete-time approach for modeling volatility clustering and are far more commonly used in practice than continuous-time stochastic volatility models for daily data. Omitting GARCH makes the comparison incomplete. If Heston underperforms GBM due to estimation difficulties but GARCH outperforms both, the conclusions would be very different.

**Required action:**
- **Literature-reviewer** should expand discussion of GARCH vs. stochastic volatility.
- **Experimentalist** must estimate at least GARCH(1,1) and EGARCH (for asymmetry) on the same data.
- **Analyst** must include these models in the AIC/BIC/out-of-sample comparison.

---

## 4. Minor Issues (Clarity, Exposition, Formatting)

### Minor Issue 1: Inconsistent Parameter Reporting

The Heston estimates in Table 2 report theta = 0.0803 (variance level), but the text sometimes discusses volatility (square root of variance) and variance interchangeably. Clarify units throughout. Annualized volatility from theta = 0.0803 would be sqrt(0.0803) = 0.283, which matches GBM sigma. This is suspicious and suggests Heston may have collapsed to a near-constant variance solution.

### Minor Issue 2: Figure References Without Figures

The manuscript references Figure 1 (Section 6.6, diagnostic plots) and Figure 2 (Section 6.6, residual analysis), with paths to image files (files/results/stochastic_volatility/diagnostic_plots.png). These figures are not included in the submitted manuscript and cannot be evaluated. Ensure figures are embedded or provided.

### Minor Issue 3: Citation Format Issues

Several citations have minor formatting issues:
- Line 599: Harrison & Kreps (1979) is labeled as Harrison1981 in the bibitem
- Some citations use "&" while AASTeX style typically uses "and"
- The Ait-Sahalia citation uses special characters that may not render correctly

### Minor Issue 4: Abstract Length

The abstract is 271 words, which exceeds typical limits (150-250 words for many journals). Consider condensing the methodological details.

### Minor Issue 5: Section Numbering Depth

The paper uses three levels of sectioning (e.g., 7.1.1, 7.1.2), which is appropriate, but Section 3.1.1 and 3.1.2 under "Model Specifications" could be simplified.

### Minor Issue 6: Acronym Definitions

SDE is used without initial definition in Section 2.1 (Line 61). Define acronyms on first use.

---

## 5. Questions for the Authors

1. Given that the Heston parameter estimates (kappa=2.0, rho=-0.5, xi=0.3) exactly match the initialization values, did the optimization algorithm essentially fail to move from its starting point? Did you check convergence diagnostics?

2. The Feller condition is satisfied in your estimates (2*kappa*theta/xi^2 = 3.57 > 1), but what is the effective degrees of freedom in the variance process? A high Feller ratio suggests the variance process is strongly bounded away from zero and may behave nearly deterministically.

3. You report that Heston reduces excess kurtosis from 6.92 to 3.17 in standardized residuals. If Heston is capturing fat tails, why does this not translate to improved likelihood? Is the particle filter likelihood calculation correctly handling the filtered variance path?

4. Why was the 22-day forecast horizon chosen? This corresponds to roughly one month. Would results differ at 5-day (weekly) or 63-day (quarterly) horizons?

5. Have you verified that your particle filter implementation is correct by testing on simulated Heston data where true parameters are known?

---

## 6. Required Experiments, Analyses, or Theory Clarifications

| # | Required Action | Responsible Agent |
|---|-----------------|-------------------|
| 1 | Re-estimate Heston with improved methodology (more particles, multiple restarts, alternative estimators) until LL_Heston >= LL_GBM | Experimentalist |
| 2 | Verify particle filter correctness on simulated Heston data | Experimentalist |
| 3 | Add GARCH(1,1) and EGARCH to the model comparison | Experimentalist, Analyst |
| 4 | Extend analysis to at least 4-6 additional assets | Data-collector, Experimentalist |
| 5 | Implement rolling-window cross-validation for out-of-sample tests | Analyst |
| 6 | Provide bootstrap confidence intervals for RMSE differences | Analyst |
| 7 | Clarify nested model testing validity or use Vuong test | Theorist |
| 8 | Include actual figures in manuscript | Report-writer |
| 9 | Address Harrison1981/Harrison1979 citation mismatch | Report-writer |

---

## 7. Evaluation Against Review Criteria

| Criterion | Rating | Comments |
|-----------|--------|----------|
| Correctness of theory | Needs Revision | Nested model testing problematic; Heston estimation failed |
| Novelty | Moderate | Question is relevant but single-asset limits contribution |
| Literature review | Good | Comprehensive coverage of relevant work |
| Experimental soundness | Major Concerns | Estimation failure; single-asset; weak validation |
| Statistical validity | Needs Revision | LRT invalid; OOS lacks confidence intervals |
| Discussion quality | Good | Thoughtful interpretation despite underlying issues |
| Writing clarity | Good | Well-written, professional tone |
| LaTeX/AASTeX formatting | Minor Issues | Citation errors; missing figures |

---

## 8. Recommendation

**MAJOR REVISION**

The manuscript addresses an interesting empirical question and is well-written, but it suffers from a fundamental methodological flaw: the Heston model estimation appears to have failed, as evidenced by the negative likelihood ratio statistic. Until this is resolved, no valid conclusions can be drawn about the relative performance of GBM vs. Heston. Additionally, the single-asset focus and absence of GARCH comparisons limit the paper's contribution.

The authors must:
1. Fix the Heston estimation procedure and achieve LL_Heston >= LL_GBM
2. Validate the particle filter on simulated data
3. Include GARCH models in the comparison
4. Either expand to multiple assets or substantially temper generalizability claims
5. Improve out-of-sample validation with proper cross-validation and confidence intervals

The paper has potential to make a genuine contribution if these issues are addressed. The finding that simpler models can outperform complex ones in certain data regimes is valuable, but only if the complex model was given a fair chance to compete.

---

## 9. Checklist of Unresolved Issues

- [ ] Heston estimation produces LL >= LL_GBM
- [ ] Particle filter validated on simulated data
- [ ] Nested model testing issue addressed (or Vuong test used)
- [ ] GARCH(1,1) and EGARCH included in comparison
- [ ] Multiple assets analyzed (or claims appropriately scoped)
- [ ] Rolling-window cross-validation implemented
- [ ] Confidence intervals provided for OOS performance differences
- [ ] Figures embedded in manuscript
- [ ] Citation errors corrected (Harrison1981 -> Harrison1979)
- [ ] Acronym "SDE" defined on first use
- [ ] Clarify variance vs. volatility units throughout

---

**Reviewer Signature:** Peer Review Agent
**Date:** December 22, 2025
**Review Round:** 1
