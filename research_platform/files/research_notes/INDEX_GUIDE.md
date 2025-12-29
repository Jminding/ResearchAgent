# Research Files Index and Navigation Guide

**Financial Transaction Networks, Money Laundering Detection, and Fraud Patterns**

---

## Quick Navigation

### Four Core Documents in This Review:

| Document | Purpose | Length | Audience |
|----------|---------|--------|----------|
| **lit_review_financial_transaction_networks_aml_fraud.md** | Complete literature review for research paper | 10,000+ words | Researchers, students |
| **evidence_sheet_financial_aml.json** | Quantitative evidence and metric ranges | ~150 KB JSON | Experimentalists, engineers |
| **sources_bibliography.md** | Complete citations (55+ papers) | ~50 KB markdown | Citation managers, writers |
| **RESEARCH_SUMMARY.md** | Executive overview and quick reference | ~30 KB markdown | Busy researchers, managers |

---

## File Descriptions

### Document 1: Main Literature Review
**File**: `lit_review_financial_transaction_networks_aml_fraud.md`
**Purpose**: Complete, publication-ready literature review
**Use**: Copy verbatim into research paper's literature review section
**Key Sections**:
- Overview of research area
- Chronological development (2008-2025)
- Prior work table with methods and results
- Baseline methods performance
- Dataset characteristics
- Money laundering typologies
- Real-world challenges
- Research gaps and open problems
- State-of-the-art summary
- References

**Length**: 10,000+ words
**Time to Read**: 30-60 minutes
**Reusability**: Direct copy-paste into papers

---

### Document 2: Evidence Sheet (JSON)
**File**: `evidence_sheet_financial_aml.json`
**Purpose**: Structured quantitative evidence for experimental design
**Format**: JSON (machine and human readable)
**Key Sections**:
- metric_ranges: Performance bounds for all methods
- typical_sample_sizes: Dataset dimensions
- class_imbalance_statistics: Fraud percentages, treatment effectiveness
- known_pitfalls: 20 common methodological errors
- key_findings: 12 critical insights
- key_references: 20 seminal papers
- research_gaps: 10 open problems
- methodological_insights: Detailed guidance
- recommendations_for_practitioners: Implementation checklist

**Format**: JSON (can be parsed programmatically)
**Time to Read**: 15-20 minutes
**Use Case**: Setting realistic thresholds and hypotheses

---

### Document 3: Complete Bibliography
**File**: `sources_bibliography.md`
**Purpose**: Full citations with 55+ peer-reviewed papers
**Organization**: 11 categories
- Primary Dataset Papers (7)
- Graph Neural Network Methods (7)
- Unsupervised/Autoencoder (3)
- Temporal and Blockchain (5)
- Comparative Studies (5)
- Baseline Methods (4)
- Graph Anomaly Detection (6)
- AML Regulatory (7)
- False Positive Analysis (5)
- Additional Resources (4)

**Citation Format**: Author, Year, Venue, URL, Contribution
**Total References**: 55+
**Time to Read**: 20-30 minutes
**Use**: Complete citations for references

---

### Document 4: Research Summary
**File**: `RESEARCH_SUMMARY.md`
**Purpose**: Executive overview for quick reference
**Key Sections**:
- Executive Summary (10 key findings)
- Datasets Characterized (table)
- Baseline Methods and Performance
- Class Imbalance Mitigation
- Real-World Detection Metrics
- Money Laundering Typologies
- Critical Research Gaps
- Recommendations for Practitioners
- Quantitative Results Summary
- Critical Insights for Experimentalists

**Length**: ~30 KB
**Time to Read**: 15-25 minutes
**Use**: Quick briefing, experimental setup

---

## How to Use These Files

### Scenario 1: Writing a Literature Review Section
**Step 1**: Read RESEARCH_SUMMARY.md (15 min)
**Step 2**: Copy lit_review_financial_transaction_networks_aml_fraud.md into your paper
**Step 3**: Extract citations from sources_bibliography.md as needed
**Result**: Complete, publication-ready literature review

---

### Scenario 2: Designing Fraud Detection Experiments
**Step 1**: Read RESEARCH_SUMMARY.md → Key Findings (10 min)
**Step 2**: Open evidence_sheet_financial_aml.json
**Step 3**: Use "Baseline Selection Framework" section
**Step 4**: Set metric thresholds from "metric_ranges"
**Step 5**: Check "known_pitfalls" to avoid errors
**Result**: Realistic experimental design with proven baselines

---

### Scenario 3: Finding Specific Papers or Citations
**Step 1**: Open sources_bibliography.md
**Step 2**: Navigate to relevant category
**Step 3**: Find paper by author/year
**Step 4**: Copy full citation and URL
**Result**: Complete citation ready for use

---

### Scenario 4: Quick Reference (5-minute briefing)
**Step 1**: Read RESEARCH_SUMMARY.md → Executive Summary
**Step 2**: Check relevant table (Datasets, Methods, Metrics)
**Step 3**: Consult Recommendations section if needed
**Result**: Quick understanding of state-of-the-art

---

### Scenario 5: Deep Dive on Specific Topic
**Example**: "What are the limitations of Isolation Forest for AML?"

**Search**: lit_review_financial_transaction_networks_aml_fraud.md
**Section**: "Baseline Methods and Quantitative Results" → "Isolation Forest"
**Result**: Comprehensive analysis with citations

---

## Key Findings (Summary)

### Finding 1: Extreme Class Imbalance
- IEEE-CIS: 3.49% fraud (27.5:1 imbalance)
- Kaggle: 0.17% fraud (578:1 imbalance)
- Production: 0.0005-0.05% (2000-200000:1)
- **Implication**: All models require explicit imbalance handling

### Finding 2: High Accuracy ≠ Effectiveness
- Test AUC: 99-100%
- Real-world FPR: 90-98%
- **Implication**: Evaluate on precision-recall, not accuracy

### Finding 3: Graph Structure Provides Gain
- GNN improvement: 2-10% F1-score
- **Implication**: Exploit transaction networks

### Finding 4: Temporal Dynamics Critical
- Multi-phase structure: placement → layering → integration
- **Implication**: Use temporal models (TGN)

### Finding 5: Synthetic Data Valuable
- Complete labels in eMoney, SAML-D
- **Implication**: Multi-dataset validation recommended

### Finding 6: False Positive Optimization Dominates
- Traditional FPR: 90-98%
- Advanced methods FPR: 10-20%
- **Implication**: Cost savings 80%+ possible

### Finding 7: Isolation Forest Competitive
- Unsupervised, O(n log n)
- **Implication**: Minimum labeled data scenarios

### Finding 8: Banking Networks Large
- 1.6M nodes, 0.00014% density
- **Implication**: GPU scalability challenges

### Finding 9: Downsampling Most Effective
- +0.5% AUC improvement
- **Implication**: Preferred for >100K samples

### Finding 10: Ensemble Methods Win
- 99.94% accuracy, 100% AUC
- **Implication**: Multiple learners capture patterns

---

## Baseline Performance Ranges

| Method | Accuracy | AUC | F1 (Minority) | Time Complexity |
|--------|----------|-----|---------------|-----------------|
| Isolation Forest | 0.85-0.95 | 0.75-0.85 | 0.70-0.92 | O(n log n) |
| Random Forest | 0.85-0.98 | 0.80-0.95 | 0.70-0.90 | O(n × m) |
| XGBoost | 0.94-1.0 | 0.94-0.99 | 0.75-1.0 | O(n × m log n) |
| Ensemble Stack | 0.9994 | 1.0 | 0.9952 | O(n × 3m) |
| GCN | 0.70-0.80 | 0.70-0.80 | 0.60-0.75 | O(|E|) |
| TGN | 0.80-0.92 | 0.80-0.92 | 0.75-0.90 | O(|E| + t) |
| Graph Autoencoder | Unsupervised | - | +5.7% vs RF | O(|V|+|E|) |

---

## Dataset Quick Reference

| Dataset | Year | Size | Fraud % | Imbalance | Best For |
|---------|------|------|---------|-----------|----------|
| IEEE-CIS | 2019 | 590K | 3.49% | 27.5:1 | Card fraud |
| Kaggle CC | 2013 | 285K | 0.17% | 578:1 | Extreme imbalance |
| Elliptic | 2019 | 204K | 2.23% | 44:1 | Crypto AML |
| eMoney | 2023 | Var | Var | Tunable | Development |
| SAML-D | 2023 | Var | Var | 28 types | Typology |
| Banking | 2024 | 1.6M | Var | High | Scalability |

---

## Class Imbalance Mitigation Effectiveness

| Strategy | Best For | F1 Improvement | Risks |
|----------|----------|----------------|-------|
| Downsampling | >100K samples | +0.5% AUC | Information loss |
| SMOTE | 5-50:1 imbalance | +5% F1 | Synthetic correlations |
| Cost-Weighting | Any size | +3-5% F1 | Ratio-sensitive |
| Stratified KFold | Always | Essential | Temporal leakage |

---

## Research Opportunities (Priority Order)

### High Priority
1. Synthetic dataset validation (empirical comparison)
2. Concept drift quantification (pattern evolution)
3. Scalability to 1M+ networks (distributed inference)
4. Explainability for compliance (SHAP/LIME)
5. False positive optimization (FPR-recall tradeoff)

### Medium Priority
6. Cross-domain transfer (cross-bank generalization)
7. Minimal labeled data (active learning, weak supervision)
8. Blockchain-specific methods (UTXO-aware)
9. Collusive fraud detection (subgraph anomaly)
10. Real-time inference (online GNN, edge computing)

---

## Recommendations for Practitioners

### Step 1: Choose Your Baseline
- Minimal labels? → Isolation Forest
- 1-5% fraud? → XGBoost + downsampling
- <1% fraud? → Ensemble stacking
- Network available? → GCN or TGN
- Real-time? → Isolation Forest or XGBoost

### Step 2: Prepare Data
- Use stratified K-fold cross-validation
- Apply strict temporal train-test splits
- Handle class imbalance (downsampling for >100K)
- Avoid temporal data leakage

### Step 3: Evaluate Properly
- Use AUC-ROC and precision-recall curves
- Never optimize accuracy alone
- Report false positive rate at fixed recall
- Validate on temporal holdout (last 2-4 weeks)

### Step 4: Deploy Carefully
- Expect 10-20x higher FPR than test metrics
- Use ensemble voting to reduce FPR
- Implement domain expert rules
- Plan for quarterly model retraining
- Monitor for concept drift

---

## Literature Coverage

- **Total References**: 55+ peer-reviewed papers
- **Time Period**: 2001-2025 (emphasis on 2020-2025)
- **Top Venues**: NeurIPS, Expert Systems, Information Systems, arXiv, IEEE, ACM
- **Datasets**: 6 major benchmarks characterized
- **Baselines**: 7 distinct methods with quantitative results

---

## Absolute File Paths

```
/Users/jminding/Desktop/Code/Research Agent/research_platform/files/research_notes/

Core Documents:
├── lit_review_financial_transaction_networks_aml_fraud.md
├── evidence_sheet_financial_aml.json
├── sources_bibliography.md
├── RESEARCH_SUMMARY.md
└── INDEX_GUIDE.md (this file)
```

---

## File Selection Quick Decision Tree

```
START
  │
  ├─ "I need to write a literature review"
  │  └─→ Use: lit_review_financial_transaction_networks_aml_fraud.md
  │
  ├─ "I need to design experiments"
  │  └─→ Use: evidence_sheet_financial_aml.json + RESEARCH_SUMMARY.md
  │
  ├─ "I need citations for my paper"
  │  └─→ Use: sources_bibliography.md
  │
  ├─ "I need a quick overview (5 min)"
  │  └─→ Use: RESEARCH_SUMMARY.md (Executive Summary section)
  │
  ├─ "I need to understand a specific method"
  │  └─→ Use: lit_review_financial_transaction_networks_aml_fraud.md
  │         (search method name)
  │
  ├─ "I need baseline performance ranges"
  │  └─→ Use: evidence_sheet_financial_aml.json
  │         (metric_ranges section)
  │
  ├─ "I need to understand research gaps"
  │  └─→ Use: lit_review_financial_transaction_networks_aml_fraud.md
  │         (Section 8: Identified Research Gaps)
  │
  └─ "I need to find a specific paper"
     └─→ Use: sources_bibliography.md
            (search by author/year)
```

---

## Document Maintenance

**Version**: 1.0
**Created**: December 24, 2025
**Status**: Complete and ready for use
**Last Updated**: December 24, 2025

**To Update**:
1. Add new papers to sources_bibliography.md
2. Extract quantitative results to evidence_sheet_financial_aml.json
3. Update relevant sections in lit_review document
4. Revise RESEARCH_SUMMARY.md with new findings

---

## Quality Assurance

- ✓ 55+ references from peer-reviewed venues
- ✓ 6 major datasets characterized with node/edge counts
- ✓ 7 baseline methods with quantitative performance
- ✓ 10 research gaps identified with opportunities
- ✓ 20 known pitfalls documented
- ✓ Quantitative evidence ranges for all metrics
- ✓ Real-world false positive rate analysis
- ✓ Class imbalance treatment effectiveness measured
- ✓ Temporal and regulatory considerations included
- ✓ Practitioner recommendations provided

---

**Ready to use. Copy documents as needed for your research.**

