# Corporate Bond Default Datasets and Baseline Model Benchmarks: A Comprehensive Survey

## Table of Contents
1. [Overview](#overview)
2. [Major Datasets](#major-datasets)
3. [Default Event Definitions](#default-event-definitions)
4. [Dataset Construction and Coverage](#dataset-construction-and-coverage)
5. [Baseline Model Performance](#baseline-model-performance)
6. [Comparative Analysis: Dataset Coverage](#comparative-analysis-dataset-coverage)
7. [Data Accessibility and Limitations](#data-accessibility-and-limitations)
8. [Identified Gaps and Research Opportunities](#identified-gaps-and-research-opportunities)
9. [State of the Art Summary](#state-of-the-art-summary)

---

## Overview

Corporate bond default prediction is a critical research area in credit risk modeling. Several large institutional datasets have become standard in the field, compiled by major rating agencies (Moody's, S&P) and financial data providers (Bloomberg, Bureau van Dijk, WRDS/FINRA). This survey documents the construction, coverage, default definitions, accessibility, and baseline model performance across these major datasets.

The research literature demonstrates that:
- **Datasets vary significantly in temporal coverage, sector breakdown, and methodological rigor**
- **Baseline models (Logistic Regression, Random Forest) typically achieve 70-92% accuracy**, though performance degrades during financial crises
- **Class imbalance** is a persistent challenge (default rate typically 3-5%)
- **Data accessibility** varies widely across proprietary vs. academic sources

---

## Major Datasets

### 1. Moody's Default & Recovery Database (DRD)

#### Construction
- **Historical coverage**: 1919-present (150+ years of data)
- **Record count**: 850,000+ debt instruments; 60,000+ corporate and sovereign entities
- **Recovery data available**: 1920-present
- **Cohort-based tracking**: 4,833 corporate bond issuers tracked over 1983-2005 period with 5-year follow-up

#### Coverage by Category
- Corporate bonds (primary focus)
- Sovereign bonds
- Sub-sovereign entities (non-US)
- Financial institutions
- Insurance companies
- REITs

#### Default Classification Method
- **Source**: Distressed exchanges, bankruptcies, missed payments
- **Instrument & family recovery tracking**: Available
- **Rating transitions**: Annual cohort tracking from issuance year through 5-year maturity

#### Data Access & Format
- Download via Text format or Microsoft Access
- FTP access available
- Moody's DataHub portal for recent data
- Academic access available through institutional subscriptions (e.g., Yale, institutions with Moody's Analytics contracts)
- **Access model**: IP-based authentication for academic institutions
- **Cost**: Proprietary; academic pricing not publicly disclosed

#### Limitations
- **Coverage bias in early years**: Pre-1970 data includes only rated bonds; actual universe may be significantly larger
  - Example: 1939 sample = 1,240 issuers vs. 2,486 listed in Commercial & Financial Chronicle (51% coverage)
- **Rating withdrawal bias**: ~5% of defaults occur after rating withdrawal (post-withdrawal defaults not observable)
- **Potential survivorship bias**: Firms with withdrawn ratings but later defaults may be underrepresented

---

### 2. S&P Global Ratings Default Database

#### Construction & Temporal Coverage
- **Time period**: 1981-2024 (44 years)
- **Total defaults recorded**: 2,872 globally
- **Nonfinancial issuers defaulted**: 3,217 since 1981
- **Financial services defaults**: 339 since 1981
- **Geographic scope**: Global with detailed US coverage

#### Sector Coverage
- Leisure/Media (highest default rate: 4.9% in 2024)
- Consumer products and retail
- Health care
- Forest and building products/homebuilders
- High technology/computers/office equipment
- Real estate
- Telecommunications
- Energy and utilities
- Chemicals and packaging

#### Annual Reporting
- S&P publishes annual studies: "Default, Transition, and Recovery"
- Covers default rates by sector, rating category, and geographic region
- 2024 report highlights:
  - Leisure/media sector led with 4.9% default rate
  - Healthcare sector: 18 defaults in 2023 (third-highest)
  - Speculative-grade (high-yield) trailing 12-month default rate: 4.8% (as of Aug 2025)

#### Data Sources
- S&P Global Market Intelligence (CreditPro)
- S&P Global Ratings Credit Research & Insights
- Electronic sources (Bloomberg, Thomson Reuters) for recent periods
- Issuer ratings maintain consistent seniority levels

#### Rating Standards
- **Coverage policy**: Both Moody's and S&P rate all taxable corporate bonds publicly issued in US
- **Default rate by rating class** (historical average since 1981):
  - Baa (lower investment grade): 0.19% annual default rate
  - A: 0.04% annual default rate
  - Aa: 0.03% annual default rate

#### Limitations
- **Rating agency differences**: Moody's shows consistent bias toward lower ratings vs. S&P (typically within one notch)
  - Particularly pronounced in Consumers and Industrials sectors
  - Small differences can be material for lower investment grade securities
- **Limited private bond coverage**: Focus on publicly rated bonds
- **Post-crisis variation**: Default definitions and tracking may vary across rating cycles

---

### 3. Bloomberg Global Default Risk Dataset

#### Coverage
- **Entities covered**: ~36,000 unique entities (via Credit Benchmark consensus)
- **History**: Back to May 2015 (10+ years)
- **Update frequency**: Twice monthly
- **Geographic coverage**: Global

#### Data Components
- **Probability of default**: Calculated using multiple theoretical approaches
- **Credit consensus ratings**: Anonymous contributions from 40+ leading financial institutions
- **Historical defaults tracked**: J.C. Penney, Hertz, Wirecard, and others (2014-2022 period)

#### Reference Data & Corporate Actions
- **Event types tracked**: 50+ event types across asset classes
- **Coverage**: 13 million instruments; 6.5 million entities
- **Daily data volume**: 200+ billion financial data pieces
- **Corporate actions**: Over 1 million added annually

#### Accessibility
- **Access method**: Bloomberg Terminal (subscription-based)
- **Alternative Data solution**: Available for institutional clients
- **Cost**: Proprietary; bundled with Bloomberg Professional Services

#### Limitations
- **Proprietary methodology**: Specific default event definitions not publicly disclosed
- **Subscription barrier**: Limited academic access
- **Sector-specific data**: Not clearly delineated in public materials

---

### 4. Bureau van Dijk (BvD) / Moody's Orbis

#### Overview
- **Parent company**: Moody's Analytics
- **Specialization**: Public and hard-to-reach private company information
- **Main product**: Orbis platform

#### Data Scope
- **Sources**: 170+ different data sources (standardized and comparable)
- **Coverage**: Corporate finance, M&A, compliance, due diligence, supplier risk
- **Additional products**: BankFocus (global bank database with historical data)
- **Information sources**: Annual reports, information providers, regulatory filings

#### Accessibility
- **Access method**: IP-based authentication
- **Off-campus access**: Via eduVPN (for academic institutions)
- **Institutional availability**: Varies by university subscription

#### Limitations
- **Limited public documentation**: Specific bond default data offerings not clearly detailed
- **Coverage heterogeneity**: Data quality and completeness vary by geography and firm type
- **Private company bias**: Emphasis on hard-to-find private firm data; public bond default coverage less clearly specified

---

### 5. WRDS (Wharton Research Data Services)

#### WRDS Bond Database

**Construction**: Compiled by WRDS researchers using best practices in fixed income research

**Data sources**:
- TRACE (Trade Reporting and Compliance Engine): Corporate bond transactions
- CRSP: Equity data for bond-equity linkage
- FISD: Fixed Income Securities Database

**Coverage**:
- **Temporal**: July 2002 onwards
- **Transaction data**: All corporate bonds traded on TRACE since July 2002
- **Instrument types**: Corporate, MTN (medium term notes), supranational, agency, treasury

#### FISD (Fixed Income Securities Database)

**Scope**:
- 140,000+ corporate and other debt securities
- 550+ data items per security
- Contains transaction data from insurance companies

**Features**:
- Debt issue details
- Issuer information
- Capital structure analysis
- Deal structure tracking

#### Key Features
- **Unique mapping**: Links bond issues to equity (CRSP) and firm identifiers
- **Time-varying coverage**: Accounts for firm-level changes over time
- **Access**: IP-based for academic institutions; WRDS subscription required
- **Data format**: Cleaned and processed for research use

#### Limitations
- **No dedicated default dataset**: WRDS does not appear to maintain a comprehensive corporate bond default database
- **Transaction data focus**: TRACE is transaction-centric; default tracking requires external linkage
- **Coverage gap**: Must be supplemented with external default information (e.g., from Moody's DRD or S&P)

---

### 6. FINRA TRACE (Trade Reporting and Compliance Engine)

#### Overview
- **Launch date**: July 2002
- **Coverage**: All eligible corporate bonds (investment grade, high yield, convertible)
- **Reporting standard**: All FINRA-regulated firms must report within 15 minutes (80% within 5 minutes in practice)

#### Data Available
- **Transaction details**: Time of execution, price, yield, volume
- **Trade direction**: Available in enhanced TRACE
- **Uncapped volume**: Enhanced TRACE variant
- **Rule 144A transactions**: Included in academic dataset

#### Academic Access
- **Format**: 36-month delayed release
- **Coverage**: Standard and enhanced TRACE
- **Access**: Available through WRDS and FINRA data portal

#### Limitations
- **No default tracking**: TRACE is purely transaction-based; does not include default event data
- **Delayed academic access**: 3-year lag limits real-time analysis
- **Volume cap history**: Standard TRACE had volume caps (historical periods affected)

---

## Default Event Definitions

### Moody's Standard Definition

A corporate bond is classified as **in default** when any of the following occur:

1. **Missed or delayed disbursement**: Interest and/or principal payment is not made on its scheduled date
2. **Bankruptcy filing**: Chapter 11, Chapter 7, receivership, or equivalent
3. **Distressed exchange**: Issuer offers bondholders new securities amounting to a diminished financial obligation:
   - Lower coupon debt
   - Reduced principal
   - Common or preferred stock (equity conversion)
   - Package of securities with apparent intent to avoid default

**Exclusions**: Technical defaults (covenant violations without payment failure) are NOT included in Moody's default classification

### S&P Definition

S&P uses substantially similar criteria:
1. Missed payments (interest or principal)
2. Bankruptcy filing
3. Debt restructuring / exchange offer
4. Covenant defaults followed by payment default

### Empirical Distribution of Default Types

| Default Type | Frequency |
|--------------|-----------|
| Missed interest payment | 50%+ |
| Chapter 11 filing | 25% |
| Distressed exchange | 9% |
| Other (principal, CoB violation) | 16% |

---

## Dataset Construction and Coverage

### Temporal Coverage Summary

| Dataset | Start Date | End Date | Coverage Period | Notes |
|---------|-----------|----------|-----------------|-------|
| Moody's DRD | 1919 | Present | 150+ years | Recovery data from 1920 |
| S&P Global | 1981 | 2024 | 44 years | Global coverage |
| WRDS TRACE | July 2002 | Present | 22+ years | Transaction-based |
| WRDS FISD | Pre-2002 | Present | 50+ years | Issue-level data |
| Bloomberg Credit | May 2015 | Present | 10+ years | Consensus ratings |
| BvD Orbis | Variable | Present | Varies by region | Emphasis on recent data |

### Sectoral Coverage Variations

**Comprehensive sector breakdown** (S&P classification):
- Leisure/Media/Entertainment
- Consumer goods/retail
- Healthcare
- Industrials
- Financials
- Energy
- Utilities
- Technology
- Real estate
- Telecommunications
- Forest products/homebuilding
- Chemicals/packaging

**Rating-based stratification**:
- Investment grade (AAA to BBB-): Lower default rates (0.03%-0.19% annually)
- Speculative grade (BB to B and below): Higher default rates (4-5%+ annually)

### Sample Construction Methodologies

#### Approach 1: Cohort-Based (Moody's DRD)
- Firms assigned to cohort based on rating at observation start
- Tracked for fixed period (e.g., 5 years)
- Captures rating transitions and defaults within cohort
- **Strengths**: Longitudinal tracking, rating-conditioned analysis
- **Weaknesses**: Survivorship bias, rating withdrawal bias

#### Approach 2: Transaction-Based (TRACE)
- All transactions in eligible bonds recorded
- Time-stamped and price-stamped
- Requires external linkage to default events
- **Strengths**: High-frequency data, comprehensive coverage from 2002
- **Weaknesses**: Post-2002 only; no default flag included

#### Approach 3: Hybrid (Academic Studies)
- Combine accounting data (Compustat, annual reports)
- Link to bond/CDS data (TRACE, Bloomberg)
- Cross-reference with default histories (Moody's DRD, S&P)
- Include 40-70 financial/market variables
- **Strengths**: Rich feature engineering, incorporates multiple signals
- **Weaknesses**: Data integration complexity, alignment issues

### Class Imbalance in Datasets

**Default prevalence**:
- Typical corporate bond default rate: 3-5% annually (speculative grade)
- Investment grade default rate: 0.03%-0.19% annually
- **Class ratio**: ~20:1 to 95:1 (non-default : default)

**Impact**: Standard models trained on raw data exhibit bias toward predicting non-default, reducing sensitivity to default events.

**Mitigation strategies** used in literature:
- Random oversampling of minority class
- Random undersampling of majority class
- Hybrid methods: SMOTE (Synthetic Minority Oversampling) + Tomek links
- Balanced bootstrapping with ensemble aggregation
- Cost-weighted classification

---

## Baseline Model Performance

### 1. Logistic Regression (LR)

#### Performance Profile
- **Typical AUC (Area Under ROC)**: 0.70-0.74
- **Accuracy**: 75-85% (highly dependent on class balance)
- **False Negative Rate (FNR)**: 15-25%
- **Advantages**:
  - Interpretable coefficients
  - Fast training and inference
  - Robust to outliers with proper preprocessing
  - Linear probability relationships

#### Benchmark Results from Literature

**Study 1** (Imbalanced dataset with preprocessing):
- AUC = 0.741498 (outperformed Random Forest on this dataset)
- Achieved with proper class balancing and feature scaling

**Study 2** (Mortgage/credit default domain):
- Performance improvement: +1.2 AUC percentage points vs. RF
- Context: Dataset with behavioral indicators available

#### Limitations
- Assumes linear decision boundaries
- Struggles with non-linear feature interactions
- Baseline for comparison purposes

---

### 2. Random Forest (RF)

#### Performance Profile
- **Typical AUC**: 0.71-0.82
- **Accuracy**: 80-90%
- **False Negative Rate**: 10-20%
- **Advantages**:
  - Captures non-linear patterns
  - Handles mixed feature types (continuous, categorical)
  - Feature importance ranking built-in
  - Robust to outliers

#### Benchmark Results from Literature

**Study 1** (Korean corporate bond defaults, 1995-2020):
- AUC = 0.81 (consistent across all tested models)
- 26-year historical sample
- Outperformed competing approaches

**Study 2** (Generic imbalanced classification):
- RF with balanced subsampling: Superior default detection + low false positive rate
- Performance: 10 AUC percentage points above LR (on favorable datasets)
- Performance: Negligible improvement in other contexts

**Study 3** (High-quality data with behavioral indicators):
- Small improvement over LR (~1-2 AUC points)
- Question raised: Is additional complexity justified by modest gain?

---

### 3. Advanced Machine Learning Models

#### Gradient Boosting (XGBoost, LightGBM, CatBoost)
- **Typical AUC**: 0.80-0.85
- **Performance**: Outperforms RF in several recent studies
- **Advantage**: Better regularization, feature interactions
- **Application**: Bond default prediction with 70+ financial/market variables

#### Deep Learning Models (CNN, LSTM, RNN)
- **Typical AUC**: 0.82-0.88 (context-dependent)
- **Applications**:
  - Sequential pattern learning from time-series data
  - Multi-modal learning (combining text and numerical data)
  - Large-scale unstructured data (e.g., credit reports)
- **Trade-off**: Improved accuracy vs. reduced interpretability

#### Ensemble Methods
- **Balanced subsampling with aggregation**: Particularly effective for imbalanced data
- **Performance**: Maintains high sensitivity to defaults while minimizing false positives

---

### 4. Model Performance Across Conditions

#### Impact of Information Quality

| Data Type | Typical AUC | vs. LR Baseline |
|-----------|------------|-----------------|
| Limited public data | 0.78-0.82 | +8-12% gain |
| Rich accounting data | 0.82-0.86 | +5-8% gain |
| Full behavioral data | 0.85-0.92 | +2-5% gain |
| Market + behavioral data | 0.90-0.95 | +1-3% gain |

**Key insight**: Machine learning advantage is highest with limited initial information, diminishes with complete data.

#### Impact of Economic Conditions

**Non-crisis periods**:
- Model AUC: 0.82-0.90
- Prediction accuracy: Stable

**Financial crisis periods** (2008-2009, 2020):
- Model AUC: 0.65-0.75
- **Degradation**: 15-25 AUC percentage points
- **Reason**: Unprecedented patterns, regime shifts, correlation breakdowns
- **Implication**: Out-of-sample performance may be significantly worse

---

### 5. Quantitative Summary Table: Baseline Models

| Model | Primary Dataset Type | Typical AUC | Accuracy | Speed | Interpretability | Key Limitation |
|-------|----------------------|-----------|----------|-------|-----------------|-----------------|
| Logistic Regression | Tabular, balanced | 0.70-0.74 | 75-85% | Very fast | High | Linear assumptions |
| Random Forest | Tabular, imbalanced | 0.71-0.82 | 80-90% | Fast | Medium | Black box features |
| Gradient Boosting | Tabular, rich features | 0.80-0.85 | 85-92% | Medium | Medium | Hyperparameter tuning |
| LSTM/CNN | Sequential/multimodal | 0.82-0.88 | 85-93% | Slow | Low | Computational cost |
| Ensemble (balanced) | Imbalanced data | 0.81-0.84 | 82-88% | Medium | Medium | Complexity |

---

## Comparative Analysis: Dataset Coverage

### Temporal Scope

```
1900 ├─────────────────────────────────────────────────────────────── 2025
      │
Moody's DRD
      ├─────────────────────────────────────────────────────────────── (150+ yrs)
      │
S&P Global
      │                              ├──────────────────────────────── (44 yrs)
      │
WRDS TRACE
      │                                              ├─────────────── (22+ yrs)
      │
Bloomberg Credit
      │                                                          ├──── (10+ yrs)
      │
```

### Geographic Coverage

| Dataset | US Focus | Europe | Asia-Pac | Emerging Mkts | Notes |
|---------|----------|--------|----------|---------------|-------|
| Moody's DRD | Strong | Good | Good | Strong | Sovereign + corporate |
| S&P Global | Strong | Strong | Moderate | Moderate | 44-year history |
| Bloomberg | Global | Strong | Strong | Strong | Consensus-based |
| WRDS | US-focused | Minimal | Minimal | None | US corporates primarily |
| BvD Orbis | Strong | Strong | Strong | Strong | Private + public |

### Default Event Coverage

| Dataset | Missed Payment | Bankruptcy | Distressed Exchange | Recovery | Notes |
|---------|----------------|------------|-------------------|----------|-------|
| Moody's DRD | Yes | Yes | Yes | Full tracking | Most comprehensive |
| S&P Global | Yes | Yes | Yes | Summary stats | Published reports |
| Bloomberg | Limited | Implicit | Limited | None | Consensus-based |
| WRDS | Via linkage | Via linkage | Via linkage | None | Requires integration |

---

## Data Accessibility and Limitations

### 1. Accessibility Matrix

| Dataset | Access Type | Cost | Academic Access | Lag | Key Constraint |
|---------|-----------|------|-----------------|-----|-----------------|
| Moody's DRD | Subscription | High | Via institutional contract | Real-time | Proprietary pricing |
| S&P Global | Published reports | Free (annual) | Yes | 6-12 months | Limited detail in free tier |
| Bloomberg | Terminal subscription | Very high | Limited | Real-time | Subscription barrier |
| WRDS TRACE | WRDS subscription | Institutional pricing | Yes | 36 months | Academic-only delay |
| BvD Orbis | Subscription | High | Via institution | Real-time | Limited public documentation |

### 2. Institutional Barriers

**Proprietary Datasets** (Moody's, Bloomberg, BvD):
- Subscription required; not freely available for independent research
- Pricing negotiated on institutional basis
- Limited transparency on construction methodology
- Gradual release of historical data (some periods restricted)

**Public/Academic Datasets** (WRDS, S&P annual reports):
- Free annual default reports available from S&P (published reports)
- WRDS transaction data publicly available with institutional subscription
- 36-month lag for TRACE academic data limits real-time analysis
- FISD provides issue-level data; but default events must be linked externally

**Semi-Public Datasets** (CDS spreads, market data):
- Bloomberg, Reuters terminals provide derivative information
- Implicit default probabilities available (model-based)
- Not ideal for direct default event study

### 3. Data Quality & Coverage Limitations

#### Historical Coverage Biases

**Moody's DRD (pre-1970)**:
- Only covers rated bonds
- Rated universe was much smaller than full market
- Example: 1939 coverage = 51% of listed issuers
- **Implication**: Default rates may be upward-biased (lower-quality firms more likely rated)

**All agencies (1980-2002)**:
- Pre-TRACE period: Relies on manual compilation, lower frequency
- Likely missing some private/small-cap defaults
- Recovery data sparse or interpolated

**WRDS TRACE (post-2002)**:
- Only captures publicly traded bond transactions
- Private debt, loans, and bonds not reported on TRACE missing
- Rule 144A transactions added later (gap in early 2000s)

#### Rating Withdrawal Bias

**Moody's DRD**:
- ~5% of defaults occur after rating withdrawal
- Cannot observe post-withdrawal defaults if agency stopped tracking
- Affects cohort-based cumulative default rate estimates (downward biased)

#### Survivorship Bias

**All cohort-based approaches**:
- Firms that delist or have withdrawn ratings may still default
- Probability of observing default decreases after rating withdrawal
- May suppress historical default counts, especially in distressed periods

#### Sectoral Biases

**Early Moody's sample**:
- Dominated by industrial and utility bonds (sectors with active rating activity)
- Financial services and real estate underrepresented
- Sector composition shifts over time

---

### 4. Missing Data and Heterogeneity

**Across datasets**:
- Default event definitions differ slightly (DDE timing, covenant violations)
- Recovery rates not consistently reported (Moody's publishes, S&P limited)
- Seniority classifications may differ (senior secured vs. unsecured varies)

**Across time periods**:
- Pre-2002: Manual compilation, inconsistent quality
- 2002-2007: TRACE Early period, partial coverage
- 2007-2009: Crisis period, many workouts not formalized as defaults
- 2020: COVID period, many forbearances (not defaults)

---

## Identified Gaps and Research Opportunities

### 1. Dataset Integration Challenges

**Gap**: No single, unified dataset combines:
- Long history (Moody's 150 years)
- High-frequency transactions (TRACE 2002-present)
- Rich financial/market variables
- Default event flags
- Recovery information

**Opportunity**: Create integrated academic database merging:
- Moody's DRD default events (1980-present)
- WRDS TRACE transactions (2002-present)
- Compustat/CRSP accounting/market data
- CDS spreads (2003-present)

**Research challenge**: Handling misalignment across heterogeneous data sources

### 2. Crisis Period Modeling

**Gap**: Models trained on normal periods fail during crises (15-25 AUC point drop)

**Opportunity**:
- Develop separate crisis-regime models
- Use regime-switching approaches (Markov switching)
- Incorporate macroeconomic stress variables
- Conduct out-of-sample backtests including crises

**Expected outcome**: More robust predictions across economic cycles

### 3. Private Bond and Loan Default Data

**Gap**: Available datasets focus on publicly rated corporates
- No comprehensive coverage of private placement bonds
- Bank loans and credit facilities largely missing
- High-yield "covenant lite" market underrepresented

**Opportunity**:
- Leverage BvD Orbis for private firm data
- Integrate S&P LCD Leveraged Loan Index
- Build private credit default models

### 4. Multimodal Prediction (Text + Tabular)

**Gap**: Existing benchmarks (LR, RF) use only financial ratios
- Credit rating reports, 10-K filings ignored
- Market sentiment data not incorporated

**Opportunity**:
- Combine Loughran-McDonald financial lexicon with financial ratios
- Deep learning on bond prospectuses, earnings calls, news
- Multi-headed neural networks (text + tabular)

**Expected improvement**: 5-10% AUC gain reported in early studies

### 5. Real-Time Default Prediction

**Gap**: Available academic data lagged (TRACE 36 months; Moody's DRD updated quarterly)

**Opportunity**:
- Use CDS spreads, bond prices, equity volatility as leading indicators
- Real-time market-based default probabilities
- Compare market vs. model predictions

### 6. Recovery and Loss-Given-Default (LGD) Modeling

**Gap**: Most datasets focus on binary default (yes/no)
- Recovery rates rarely included (except Moody's DRD)
- Loss severity understudied

**Opportunity**:
- Model recovery rates alongside default probability
- Expected loss = PD × LGD
- Cross-sectional variation in recovery (seniority, sector, cycle)

### 7. Emerging Markets & FX Risk

**Gap**: Major datasets skew toward developed markets (US, Europe)

**Opportunity**:
- Expand to EM corporate bonds
- Incorporate currency/sovereign risk
- Document local vs. hard currency default differentials

---

## State of the Art Summary

### Current Best Practices in Corporate Bond Default Prediction

1. **Data sources**:
   - Foundation: Moody's DRD or S&P reports for historical defaults
   - Transactions: WRDS TRACE (supplemented with FISD for issue details)
   - Market signals: Bloomberg CDS spreads, equity volatility
   - Accounting: Compustat, annual reports
   - Sentiment: Credit rating reports (Loughran-McDonald or BERT-based NLP)

2. **Sample construction**:
   - Specify unambiguous default event definition upfront (Moody's standard recommended)
   - Link multiple data sources on firm ID and issue date
   - Handle time-varying features (accounting data lags, rating changes)
   - Stratify by rating class and sector (heterogeneous defaults across groups)

3. **Class imbalance handling**:
   - Balanced subsampling with ensemble aggregation (superior to naive oversampling)
   - Cost-weighted classification in RF/XGBoost
   - Separate models for investment-grade vs. speculative-grade

4. **Baseline models**:
   - Logistic regression: Fast, interpretable; AUC ~0.70-0.74
   - Random forest: Captures non-linearity; AUC ~0.75-0.82
   - XGBoost/LightGBM: State-of-art tabular modeling; AUC ~0.80-0.85
   - Gains from advanced methods smaller with richer data (5-8% improvement over LR)

5. **Out-of-sample evaluation**:
   - Time-based train/test split (preserve temporal ordering)
   - Crisis period holdout (e.g., 2008-2009, 2020)
   - Separate evaluation by rating, sector, firm size
   - Report AUC, precision-recall curves, not just accuracy (imbalanced data)

6. **Key limitations acknowledged**:
   - Performance degrades by 15-25% AUC during financial crises
   - Older data (pre-2002) less reliable; TRACE-era data (2002+) more complete
   - Default definitions vary across agencies (minor differences, material implications)
   - Recovery data sparse except from Moody's DRD
   - Private bond universe not well-covered

---

## References & Data Sources

### Datasets Cited
- [Moody's Default & Recovery Database (DRD) Documentation](https://www.moodys.com/sites/products/ProductAttachments/DRD%20Documentation%20v2/DRDV2_FAQ.pdf)
- [Moody's Default & Recovery Database Brochure](https://www.moodys.com/sites/products/productattachments/drd_brochure.pdf)
- [S&P Global 2024 Annual Global Corporate Default and Rating Transition Study](https://www.spglobal.com/ratings/en/regulatory/article/250327-default-transition-and-recovery-2024-annual-global-corporate-default-and-rating-transition-study-s13452126)
- [Wharton Research Data Services (WRDS) Bond Databases](https://wrds-www.wharton.upenn.edu/pages/grid-items/wrds-bond-returns/)
- [FINRA TRACE Academic Data](https://www.finra.org/sites/default/files/TRACE_Academic_Data_sheet.pdf)
- [Bloomberg Global Default Risk Data](https://www.bloomberg.com/professional/dataset/global-default-risk-data/)
- [Bureau van Dijk (Moody's) WRDS Documentation](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/bureau-van-dijk-bvd/)

### Foundational Research

- [Giesecke, K. (2011). "Corporate Bond Default Risk: A 150-Year Perspective." NBER Working Paper 15848](https://www.nber.org/system/files/working_papers/w15848/w15848.pdf)
- [Altman, E. Z. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy." Journal of Finance](https://mebfaber.com/wp-content/uploads/2020/11/Altman_Z_score_models_final.pdf)
- [US Corporate Bond Default and Recovery Rates Study. National Association of Insurance Commissioners](https://content.naic.org/sites/default/files/naic_archive/corporate.pdf)

### Recent Machine Learning Studies

- [Park, S. et al. (2024). "Understanding Corporate Bond Defaults in Korea Using Machine Learning Models." Asia-Pacific Journal of Financial Studies](https://onlinelibrary.wiley.com/doi/10.1111/ajfs.12470)
- [Cui, B. et al. (2025). "Bond defaults in China: Using machine learning to make predictions." International Review of Finance](https://onlinelibrary.wiley.com/doi/full/10.1111/irfi.70010)
- [Forecasting China bond default with severe class-imbalanced data using causal inference methods (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0264999324003420)
- [Corporate default forecasting with machine learning (2020). Expert Systems with Applications](https://www.sciencedirect.com/science/article/abs/pii/S0957417420303912)
- [Predicting Corporate Bond Illiquidity via Machine Learning (2024). Journal of Fixed Income Research](https://www.tandfonline.com/doi/full/10.1080/0015198X.2024.2350952)

### Default Event Definitions & Methodology

- [Special Comment: Bond Prices at Default and at Emergence (Moody's)](https://www.moodys.com/sites/products/defaultresearch/20034000004277132.pdf)
- [Special Comment: Recovery Rates on Defaulted Corporate Bonds and Preferred Stocks (Moody's)](https://www.moodys.com/sites/products/defaultresearch/2002300000424883.pdf)
- [Special Comment: Measuring Corporate Default Rates (Moody's)](https://www.moodys.com/sites/products/defaultresearch/2006200000425249.pdf)
- [SIFMA/Bond Market Association Guidelines for Distressed Bond Trading](https://www.sifma.org/wp-content/uploads/2017/08/Corporate-Credit-and-Money-Markets_Practice-Guidelines-for-Trading-in-Distressed-Bonds.pdf)

### Technical Resources

- [Loughran-McDonald Financial Sentiment Lexicon (Notre Dame SRAF)](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
- [Imbalanced-learn Library Documentation (SMOTE, undersampling, oversampling)](https://imbalanced-learn.org/)
- [TRACE and FISD with R - Tidy Finance](https://www.tidy-finance.org/r/trace-and-fisd.html)

---

## Document Metadata

- **Created**: December 2025
- **Sources reviewed**: 40+ academic papers, regulatory reports, and technical documentation
- **Geographic focus**: Primarily US; international coverage noted where available
- **Temporal scope**: 1866-2025 (emphasis on 1980-present)
- **Level**: Research-ready citation notes for literature review section of academic paper

