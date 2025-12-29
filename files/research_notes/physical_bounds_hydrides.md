# Literature Review: Physical Bounds on Superconducting Tc, Hydride Superconductors, and ML Validation

## 1. Overview of the Research Area

This review synthesizes literature spanning three interconnected domains:

1. **Fundamental Physical Bounds on Tc**: Theoretical and empirical limits governing the maximum achievable superconducting transition temperature across material classes, grounded in BCS-Eliashberg theory and thermodynamic constraints.

2. **Recent Hydride Superconductor Discoveries**: High-pressure hydrogen-rich compounds (H3S, LaH10, YH3, YH6, LaH9, and ternary systems) exhibiting anomalously high Tc values (203–250+ K), challenging conventional understanding of superconductivity.

3. **ML Validation for Tc Prediction**: Benchmarking methodologies, cross-validation protocols, and performance metrics for machine learning models predicting superconducting properties across databases of 5,000–16,000+ compounds.

The field has undergone a paradigm shift following H3S (Eremets et al., 2015), with subsequent discoveries in polyhydride systems demonstrating that conventional electron-phonon coupling, when optimized through hydrogen's high vibrational frequencies and strong coupling, can yield extraordinarily high Tc values under megabar pressures. Simultaneously, computational discovery pipelines combining DFT, structure prediction, and increasingly, machine learning, have accelerated candidate identification.

---

## 2. Chronological Summary of Major Developments

### 2.1 Theoretical Foundations (1957–2010s)

**BCS Theory (1957)**: Bardeen, Cooper, and Schrieffer established the microscopic theory of conventional superconductivity via electron-phonon-mediated Cooper pair formation, predicting Tc depends on electron-phonon coupling strength (λ) and characteristic phonon frequency (ω_D).

**McMillan Formula (1968)**: Extended BCS theory to account for strong coupling effects:
$$T_c = \frac{\Theta_D}{1.20} \exp\left[-\frac{1.04(1 + \lambda)}{\lambda - \mu^*(1 + 0.62\lambda)}\right]$$
where λ is electron-phonon coupling constant and μ* is Coulomb repulsion parameter.

**Eliashberg Theory (1960s)**: Migdal-Eliashberg formalism incorporates retarded nature of electron-phonon interaction, providing rigorous self-consistent gap equations. Historically believed to establish practical upper bounds on Tc for conventional superconductors around 30–40 K at ambient pressure.

**Historical Bounds**: Pre-1990s consensus: ambient-pressure conventional superconductivity above 40 K deemed extremely unlikely (Allen & Dynes, 1975).

### 2.2 Conventional and High-Tc Discoveries (1986–2015)

**Cuprate Revolution (1986–)**: YBa₂Cu₃O₇ (Tc = 92 K) shattered BCS predictions, prompting theories of unconventional (non-phonon-mediated) pairing. However, mechanism remains debated after 40 years.

**MgB₂ (2001)**: Discovery of Tc = 39 K in a conventional (weakly-correlated) material via two-gap electron-phonon pairing; reinforced validity of conventional mechanisms under optimal conditions.

### 2.3 Hydride Superconductor Era (2015–Present)

**H₃S Discovery (2015)**: Eremets et al. reported Tc = 203 K in hydrogen sulfide at 155 GPa—first superconductor exceeding 200 K without pressure-dependent cuprate systems. Triggered paradigm: extreme pressures stabilize hydrogen-rich structures with high electron-phonon coupling.

**LaH₁₀ (2019)**: Somayazulu et al. (Nature 2019) demonstrated Tc ≈ 250 K at 170 GPa in lanthanum hydride. Clathrate-like structure with H atoms forming cages around rare-earth centers. Predictions later validated experimentally.

**YH₉ and YH₃ (2021–2023)**: Yttrium systems achieving Tc = 243–244 K at lower pressures (e.g., YH₃ at 17.7 GPa). Strong anharmonicity of hydrogen vibrations identified as key factor.

**Ternary Systems (2022–2025)**: (La,Ce)H₉, (La,Ce)H₁₀, (Y,Ce)H₉, (La,Nd)H₁₀ clathrate hydrides showing promise; confirmed experimental superconductivity in doped variants.

**LK-99 Retraction (2023)**: Claimed ambient-pressure room-temperature superconductor comprehensively refuted by multiple independent groups (August 2023); identified as Cu₂S impurity artifacts. Demonstrates critical importance of validation protocols.

**Recent Advances (2024–2025)**:
- Prediction of (La,Th)H₁₀ at Tc = 242 K below 200 GPa (stabilization shift).
- LaSc₂H₂₄ predicted at Tc > 200 K.
- Direct tunneling spectroscopy confirmation of H₃S superconducting gap (December 2025, described as "most important work since H₃S discovery in 2015").
- Focus on pressure reduction toward ambient-pressure stabilization via doping and virtual pressure effects.

---

## 3. Fundamental Physical Bounds on Superconducting Tc

### 3.1 Theoretical Upper Limits

#### McMillan-Allen-Dynes Bound

The practically-used upper estimate for conventional superconductor Tc under BCS-Eliashberg framework:

$$T_c \approx 0.1 \omega_{\log} \exp\left[-\frac{1.04(1 + \lambda)}{\lambda - 0.62\mu^*}\right]$$

where:
- ω_log = logarithmic average phonon frequency (typically < 1000 K at ambient pressure)
- λ = electron-phonon coupling constant (observed max ~2–4)
- μ* ≈ 0.1–0.2 Coulomb repulsion parameter

**Classical Consensus Prediction**: Achievable Tc ≲ 30–40 K at ambient pressure, based on:
- Maximum observed λ ≈ 2–2.5 in conventional metals
- Fermi temperature constraints (TF ~ 10⁴ K, Tc/TF ≪ 1)
- Trade-off between phonon frequency and coupling strength

#### Heuristic Bounds (Refuted)

Recent theoretical work shows that proposed heuristic bounds on Tc as fractions of:
- Fermi temperature: Tc/TF
- Zero-temperature superfluid stiffness: Tc/ρs(0)
- Debye frequency: Tc/ω_D

are **not fundamental**—explicit models demonstrate these ratios can be unbounded. However, they remain useful practical guides in many regimes.

### 3.2 Fundamental Physical Limits from First Principles

#### Electron-Phonon Coupling Constraint

**Finding**: Maximum electron-phonon coupling constant λ ≤ 4, arising from an intrinsic lattice instability.

**Physical Origin**: As coupling increases, equilibrium between electrons and ions becomes unstable. Phase transition (metal → new state) occurs at first-order; metal becomes dynamically unstable to small thermal deviations. This sets a hard upper limit on λ regardless of material or pressure.

**Polyhydride Data**: Observed λ values in hydrides range 2–3, approaching but not exceeding this bound, explaining anomalously high Tc.

#### Phonon Frequency Bounds

**Fundamental Limit**: Phonon frequencies bounded by fundamental physical constants to ~10³–10⁴ K (Planck-scale constraints on condensed matter).

**Hydride Observation**: Maximum phonon frequencies observed in hydrides exceed 5000 K locally, but **logarithmic average frequency ω_log rarely exceeds 1800 K**. This reveals:

> An inherent trade-off: high peak frequencies cannot be simultaneously optimized with high average frequencies and strong coupling.

Mathematically, optimizing Eliashberg function α²F(ω) subject to phonon spectrum constraints shows that the "ideal" spectral function (maximizing Tc) is **physically unrealizable**.

#### Ambient Pressure Ceiling

**Consensus Finding** (2025): Room-temperature conventional superconductivity at ambient pressure is "extremely unlikely" based on current understanding. Reasoning:
1. At P = 0, typical ω_log ~ 500 K for best materials (noble metals, hydrides decompose or become insulating)
2. λ cannot exceed 4, and realistic λ at ambient P ~ 1–2
3. McMillan formula yields Tc_max ~ 10–20 K

**Implication**: Megabar pressures (100–400 GPa) remain necessary to achieve 200+ K superconductivity in conventional phonon-mediated systems.

### 3.3 Empirical Ranges by Superconductor Class

| **Class** | **Tc Range (K)** | **Mechanism** | **Examples** | **Pressure Req.** |
|-----------|-----------------|---------------|-------------|-------------------|
| **Conventional (weak coupling)** | 0.01–10 | Phonon-mediated (λ < 1) | Al, Pb, Nb | Ambient |
| **Conventional (strong coupling)** | 10–40 | Phonon-mediated (λ ≈ 1–2) | MgB₂ (39 K), Nb₃Sn (18 K) | Ambient |
| **Cuprate (layered oxide)** | 30–130 | Unconventional (magnetic?); **BCS framework fails** | YBa₂Cu₃O₇ (92 K), Bi₂Sr₂Ca₂Cu₃O₁₀ (110 K) | Ambient; pressure reduces Tc |
| **Heavy fermion** | 0.5–2 | Unconventional (Kondo pairing); strong correlations | CeCoIn₅ | Ambient–high P |
| **Organic** | 1–15 | Weakly unconventional; phonon + repulsion | κ-(ET)₂Cu(SCN)₂ (13 K) | Ambient–moderate P |
| **Hydride (ambient P prediction)** | 0–80 | Phonon-mediated (λ → 4) if stabilized | LiH, CaH₆ (hypothetical) | ~10–100 GPa to stabilize |
| **Hydride (high pressure)** | 200–250 | Phonon-mediated (λ ≈ 2.5–3.5); conventional BCS | H₃S (203 K), LaH₁₀ (250 K) | **155–400 GPa** |

---

## 4. Recent Discoveries in Hydride Superconductors

### 4.1 H₃S (Hydrogen Sulfide)

**Discovery**: Eremets et al. (2015), Nature 525, 73–76. Tc = 203 K at P = 155 GPa.

**Structure**: Cubic P6₃/mmc; conventional metallic H₃S with vibrational frequencies up to 550 cm⁻¹ (≈ 1000 K phonon freq).

**Electron-Phonon Properties**:
- λ ≈ 1.9–2.2
- ω_log ≈ 900–1050 K
- Eliashberg spectral function α²F(ω) dominated by H-H stretching modes

**Experimental Verification**:
- Infrared spectroscopy, electrical resistance measurements, magnetic susceptibility
- Isotope effect on Tc observed (∂ln Tc / ∂ln M ≈ 0.4–0.6)—confirms phonon-mediated mechanism

**Key Challenge**: Stability only at >150 GPa; material decomposes or loses superconductivity below ~100 GPa.

### 4.2 LaH₁₀ (Lanthanum Decahydride)

**Prediction & Synthesis** (Somayazulu et al., Nature 2019): Predicted via DFT/structure search; Tc = 250 K at 170 GPa; structure later confirmed.

**Structure**: Clathrate-like P4₂/mnm (or related); La atoms at cage centers, H atoms forming hydrogen sublattice.

**Superconducting Properties**:
- Tc = 250 K (borderline approaching ambient temperature)
- Tc/Tc(H₃S) enhancement factor ≈ 1.23× due to optimized electronic structure at Fermi level
- Magnetic transitions probed via SQUID magnetometry

**Electron-Phonon Properties**:
- λ ≈ 2.5–2.8
- ω_log ≈ 1100–1200 K
- Strong contribution from H-derived phonon modes

**Experimental Status**: Confirmed superconductivity via multiple groups; remains under pressure; phase stability diagram refined with anharmonic corrections.

### 4.3 YH₃ (Yttrium Trihydride)

**Distinction**: Among hydrides, YH₃ requires the **lowest pressure** for superconductivity onset: Tc = 40 K at 17.7 GPa (face-centered cubic fcc structure).

**Notable Features**:
- Lowest pressure hydride superconductor known (as of 2023)
- Cubic Y-H bond stability extends to lower pressures than rare-earth analogs

**Anharmonic Lattice Dynamics**:
- EXAFS measurements reveal **strong anharmonicity**: Y atoms vibrate in double-well potential
- XANES shows density of 4d Y states strongly modulated by H, increasing with pressure
- Quantum anharmonic effects enable phase stabilization even at modest pressures

**Electron-Phonon Coupling**:
- λ ≈ 1.8–2.0 (moderate, lower than LaH₁₀)
- Compensated by elevated ω_log ~ 1000 K and optimized density of states
- Doping (electron/hole) can further enhance Tc via band structure engineering

### 4.4 YH₆ (Yttrium Hexahydride)

**Claims**: Tc = 244 K at modest pressures (~130 GPa).

**Anomaly**: Superconducting properties notably **depart from conventional Migdal-Eliashberg predictions**. Spectroscopic data hints at additional mechanisms beyond simple phonon-mediated coupling.

**Open Questions**:
- Possible contributions from electronic correlations?
- Role of hydrogen sublattice instabilities?
- Requires independent reproduction to clarify.

### 4.5 Ternary Systems: (La,Ce)H₉, (La,Ce)H₁₀, (Y,Ce)H₉, (La,Nd)H₁₀

**Recent Progress (2022–2025)**:
- Experimental confirmation of superconductivity in rare-earth-doped clathrate hydrides
- Electron substitution via doping modulates electronic structure, filling, and band gaps
- Some systems predicted at reduced pressures (~150 GPa) vs. parent LaH₁₀

**Example**: (La,Th)H₁₀ predicted Tc = 242 K at <200 GPa (Usseinov et al., 2023).

**Advantage**: Compositional tuning enables pressure optimization and potential stabilization routes.

### 4.6 Ambient-Pressure Stabilization Efforts (Emerging)

**Virtual Pressure Effect**: Recent proposals exploit charge-transfer modulation (doping via alkali or transition metals) to mimic high-pressure electronic effects at lower pressures. Preliminary DFT predictions suggest Tc > 80 K may be achievable at P ~ 1 GPa for engineered ternary hydrides.

**Experimental Status**: Early-stage; no confirmed ambient-pressure superconductivity in hydrides yet.

---

## 5. Mechanisms Explaining Ultra-High Tc Values

### 5.1 Fundamental Electron-Phonon Pairing Mechanism

**Conventional BCS Scenario**:
Hydride superconductivity is phonon-mediated Cooper pairing under Eliashberg framework. Electrons near the Fermi level interact with lattice vibrations, forming bound pairs with net attractive interaction overcoming Coulomb repulsion.

**Hydrogen Advantage**:
1. **Ultra-High Phonon Frequencies**: H atoms (lowest mass) vibrate at extremely high frequencies (ω_H ~ 500–2000 cm⁻¹, or 1500–6000 K), contributing to ω_log ~ 1000–1800 K.
2. **Strong Electron-Phonon Coupling (λ ≈ 2.5–3.5)**: H-derived bands at Fermi level couple strongly to acoustic and optical H modes due to (i) small effective mass of H, (ii) high band-structure sensitivity to H displacements.
3. **High Density of States at EF**: Rare-earth d-orbitals + H-derived s-p bands create enhanced electronic density at Fermi level, boosting pairing attraction.

**Mathematical Expression (McMillan Formula Applied)**:
$$T_c = \frac{\omega_{log}}{1.2} \exp\left[-\frac{1.04(1 + 2.5)}{2.5 - 0.62 \times 0.1}\right] \approx \frac{1500 \text{ K}}{1.2} \exp[-1.62] \approx 200\text{-}250 \text{ K}$$

This shows that optimizing both λ and ω_log in hydrides approaches theoretical BCS limits, **without requiring unconventional mechanisms**.

### 5.2 Anharmonic Lattice Effects

**Key Discovery**: Conventional harmonic lattice dynamics **underestimate** stability and Tc in hydrides due to zero-point quantum and thermal anharmonicity.

**Anharmonic Corrections**:

- **Lattice Anharmonicity** (SSCHA calculations): H atoms experience not parabolic but quartic (or higher-order) potentials. Stochastic self-consistent harmonic approximation (SSCHA) accounts for:
  - Renormalization of phonon frequencies
  - Softening of acoustic modes
  - Hardening of optical modes
  - Thermal/quantum fluctuation effects

**Quantitative Impact (YH₃ example)**:
- Harmonic prediction: Lattice unstable <5 GPa
- Anharmonic correction: Lattice stabilized to ~17.7 GPa (matches experiment)
- Difference: ~10 GPa pressure shift

**Physical Interpretation**: H atoms in double-well potentials at cage boundaries. Anharmonicity enables "smearing" of instability, allowing superconductivity to persist to lower pressures. This is **not a new pairing mechanism** but rather a re-normalization of lattice parameters affecting electron-phonon coupling.

### 5.3 Quantum Nuclear Effects

**Isotope Effect Data**:
- H₃S: dTc/d(1/√M_H) ≈ 0.4–0.6, confirming phonon-mediated origin
- Expected value from BCS: 0.5; observed values consistent

**Deuterium Substitution**: H → D (doubling mass) → ω reduced by √2, Tc reduced by ~15–20%, matching BCS prediction. Confirms **phonon-mediated** superconductivity.

### 5.4 Band Structure and Electronic Effects

**Fermi Surface Properties**:
- Rare-earth d-bands (localized) + H s-p bands (delocalized) create multiband structures
- Heavy d-orbital contribution raises density of states N(EF)
- Typical N(EF) in LaH₁₀, YH₃ ~ 1–3 states/(eV·atom), vs. ~0.1 in simple metals

**Optimization**:
- Tc sensitively depends on band filling (position of μ relative to van Hove singularities)
- Doping via ternary substitution shifts μ, enabling Tc tuning
- Example: (La,Ce)H₉ shows Tc enhancement via electron doping

### 5.5 Breakdown of Approximations & Departures from Standard Theory

**Migdal Approximation Limits**:
Migdal theory assumes ω_phonon ≪ EF (adiabatic limit, ω_phonon/EF ~ 10⁻³).

In hydrides:
- ω_H ~ 2000 K, EF ~ 10,000 K
- Ratio ω/EF ~ 0.2–0.3 (less adiabatic than assumed)
- Migdal approximation becomes marginal; vertex corrections potentially important (~10–20% corrections to λ possible)

**YH₆ Anomaly**: Departures from Eliashberg predictions suggest:
1. Possible vertex corrections beyond Migdal
2. Potential excitonic effects from H vibration coupling to rare-earth correlations
3. Requires experimental validation (gap structure, spectroscopy)

### 5.6 Trade-off Between Tc-Enhancing Factors

**Critical Constraint**:
Optimizing Tc requires balancing:

$$T_c \propto \lambda \cdot \omega_{log}$$

But:
- **Increasing λ**: Hardens crystal, increases ω_log but reduces electronic mobility (increases electron-impurity scattering)
- **Increasing ω_log**: Requires smaller atomic mass (H), but further increasing H content destabilizes lattice at ambient pressure

**Result**: Optimal Tc occurs at finite pressure where:
- Lattice density maximizes N(EF)
- Phonon frequencies remain high
- Electron-phonon coupling reaches ~2.5–3 (below instability at λ ~ 4)

This explains **why hydride superconductivity requires extreme pressure**: it represents a finely-tuned balance impossible to achieve at ambient conditions.

---

## 6. Validation Criteria for ML Predictions of Superconducting Tc

### 6.1 Dataset Construction and Curation

#### Data Source & Quality

- **Primary Database**: Majority of ML studies use ISC (International Superconductivity Center) database or in-house compiled datasets
- **Typical Size**: 5,713–16,413 compounds (varies by study; larger datasets post-2022)
- **Composition Range**: Elemental superconductors, binary (e.g., MgB₂, FeSeₓ), ternary/quaternary, and high-entropy compounds
- **Pressure Coverage**: Mostly ambient-pressure superconductors; increasingly including high-pressure systems (2020+)

#### Data Cleaning Protocols

**Required Steps**:
1. **Duplicate Removal**: For compounds with multiple Tc measurements, compute **median Tc** (outliers removed via IQR or σ-clipping)
2. **Valid Tc Criterion**: Include only superconductors with experimentally-confirmed Tc and unambiguous transition signatures (resistivity drop + diamagnetism ideally)
3. **Exclude**:
   - Superconductors lacking clear experimental confirmation
   - Non-stoichiometric compounds without precise composition
   - Materials with disputed Tc values (e.g., LK-99)
   - Theoretical predictions not yet synthesized

**Data Distribution**:
- Most datasets exhibit heavy skew toward low Tc (0–20 K, ~70% of data)
- Tail toward high-Tc cuprates and hydrides (200+ K, ~1–2% of data)
- Class imbalance handling: weighted loss functions or stratified stratified splitting

### 6.2 Feature Engineering and Representation

#### Chemical Composition Features

**Elemental Descriptor Extraction**:
1. Extract atomic numbers, electronegativities, atomic radii, d-electron counts for each element
2. Generate statistics across constituents: mean, std, max, min, range for each property
3. Typical feature count: 20–80 features after engineering

**Example Feature Set**:
- Average atomic number
- Max/min electronegativity difference
- Total number of valence electrons
- Presence of d-block elements (binary indicator)
- Stoichiometry (atomic ratios as fractions)

#### Structural Descriptors

**DFT-Based Features** (when structure available):
- Band structure: N(EF) (density of states at Fermi level)
- Projected density of states (PDOS) on each atom type
- Electron-phonon coupling λ from Eliashberg calculations
- Characteristic phonon frequency ω_log

**Symmetry-Based**:
- Space group, point group symmetry
- Bravais lattice type

**Constraint**: Structural features only applicable to ~10–20% of database entries; most predictions rely on chemical composition alone.

#### Attention-Based & Graph Representations

**Recent (2023–2025)**:
- Atomic vectorizations (embeddings) learned from chemical data
- SOAP (Smooth Overlap of Atomic Positions) descriptors: encode local atomic environments
- Graph neural networks (GNNs): represent crystal structure as graph with atoms as nodes, bonds as edges
- Attention mechanisms: weight contributions of different elements/features adaptively

**Reported Advantage**: SOAP + cross-validation achieves 92.9% R² (vs. 88–90% for simpler features).

### 6.3 Train-Test Splitting and Cross-Validation

#### Standard Protocol

1. **Train-Test Split**: 85%:15% or 80%:20% ratio
   - Stratified sampling by Tc range to maintain distribution
   - No data leakage (ensure chemical composition uniqueness across splits)

2. **Cross-Validation Scheme**:
   - **K-Fold CV** (k = 5–10): partition training set into k equal folds; train k models, each leaving out one fold for validation. Report mean ± std of CV scores
   - **Nested CV**: outer loop for hyperparameter tuning, inner loop for generalization estimation (prevents overfitting to test set indirectly)

3. **Temporal Validation** (for time-series context):
   - Train on pre-2020 compounds, test on 2020–2025 discoveries (e.g., hydride systems)
   - Reveals model generalization to novel material classes

#### Stratification by Material Class

**Refinement (2023+)**: Some studies segment data by:
- Elemental vs. intermetallic vs. organic vs. cuprate vs. hydride
- Train separate models per class (improves accuracy, but requires larger datasets)
- Cross-class predictions tested separately

### 6.4 Performance Metrics & Reporting Standards

#### Regression Metrics (for continuous Tc prediction)

| **Metric** | **Formula** | **Interpretation** | **Target** |
|-----------|-----------|------------------|-----------|
| **R²** (Coeff. of Determination) | 1 - (SS_res / SS_tot) | % variance explained; max = 1 | > 0.90 |
| **RMSE** (Root Mean Squared Error) | √(Σ(y_true - y_pred)² / N) | Avg. absolute error in K | < 10 K |
| **MAE** (Mean Absolute Error) | Σ(\|y_true - y_pred\|) / N | Median absolute deviation | < 8 K |
| **MAPE** (Mean Absolute % Error) | Mean(\|y_true - y_pred\| / y_true) | Percentage error; problematic for low Tc | N/A (unreliable) |
| **Spearman Correlation (ρ)** | Rank-based correlation | Monotonic relationship; robust to outliers | > 0.90 |

**Best Practices**:
- Report **all** of R², RMSE, MAE (not just R²)
- For skewed distributions: report metrics separately for Tc < 50 K (majority) and Tc > 50 K (rare events)
- Median Absolute Error (MedAE) preferred over MAE for outlier robustness

#### Classification Metrics (if Tc thresholded, e.g., "high-Tc" vs. "low-Tc")

- **Accuracy**: (TP + TN) / Total; inflated on imbalanced datasets
- **Precision & Recall**: TP/(TP+FP) and TP/(TP+FN); report both to avoid gaming
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall); harmonic mean
- **AUC-ROC**: Area under receiver operating characteristic; robust to class imbalance

### 6.5 Hyperparameter Tuning and Regularization

#### Hyperparameter Search

**Methods**:
1. **Grid Search**: Exhaustive search over predefined parameter grid (computationally expensive)
2. **Random Search**: Sample parameter space randomly (more efficient for high-dimensional spaces)
3. **Bayesian Optimization** (e.g., Optuna, Hyperopt): model surrogate function, iteratively sample promising regions
4. **Genetic Algorithms**: evolutionary search (less common in recent work)

**Typical Hyperparameters** (model-dependent):
- **Neural Networks**: learning rate, batch size, dropout, number of layers/units, activation functions
- **Random Forests**: number of trees, max depth, min samples per leaf
- **Gradient Boosting**: learning rate (η), max depth, number of boosting rounds

**Nested CV for Tuning**: Hyperparameters selected based on inner CV loop to prevent test leakage.

#### Regularization Strategies

1. **L1/L2 Regularization**: Penalize large weights; reduce overfitting
2. **Dropout** (neural networks): randomly deactivate neurons during training; ensemble effect
3. **Early Stopping**: halt training when validation loss stops improving
4. **Feature Selection**: reduce feature count to discourage overfitting (next section)

### 6.6 Feature Selection and Dimensionality Reduction

#### Motivation

Excessive features lead to:
- **Overfitting**: model memorizes training data rather than learning generalizable patterns
- **Computational Cost**: training time and memory scale with feature count
- **Interpretability**: 50+ features difficult to interpret physically

#### Feature Selection Methods

1. **Univariate Selection** (chi-square, ANOVA F-test, mutual information)
   - Rank features by statistical significance
   - Select top k features

2. **Recursive Feature Elimination (RFE)**
   - Train model, remove lowest-weight feature, retrain iteratively
   - Captures feature interactions unlike univariate methods

3. **Gradient Boosted Feature Selection (GBFS)** (recent, 2023+)
   - Train gradient boosting model (e.g., CatBoost, XGBoost)
   - Extract feature importances from boosting rounds
   - Select features with cumulative importance > 95%
   - Integrates statistical evaluation and multicollinearity reduction

4. **Embedded Methods**
   - LASSO (L1 regression): automatically zeroes low-impact coefficients
   - Tree-based importance: use feature importances from random forests/XGBoost

**Reported Benefit**: GBFS + two-layer selection reduces feature set from 80 → 15–20 features, while maintaining R² > 0.95, improving generalization.

### 6.7 Addressing Class Imbalance and Outliers

#### Class Imbalance Strategy

Most Tc databases are **heavily skewed**: ~70% compounds with Tc < 20 K, ~1% with Tc > 100 K.

**Solutions**:
1. **Stratified Cross-Validation**: maintain Tc distribution in each fold
2. **Weighted Loss Functions**: up-weight rare high-Tc samples in training
3. **SMOTE (Synthetic Minority Over-sampling)**: generate synthetic high-Tc samples (use cautiously; may introduce artifacts)
4. **Quantile Regression**: predict conditional quantiles instead of mean Tc; captures uncertainty

#### Outlier Handling

- **Identification**: z-score > 3, IQR-based thresholding, or visual inspection
- **Treatment**:
  - Remove (if clearly erroneous, e.g., disputed Tc like LK-99)
  - Cap (replace with 99th percentile)
  - Separate models for outlier/non-outlier data
  - Robust loss functions (Huber, quantile) less sensitive to outliers

### 6.8 Validation Protocols for Generalization

#### Out-of-Sample Testing

1. **Test Set Performance**: Evaluate model on held-out test set (15–20% of data)
   - **Golden Standard**: provides unbiased estimate of generalization if test set independent and representative

2. **Temporal Validation** (material-discovery context):
   - Train on pre-2020 compounds (N ~ 5000)
   - Test on 2020–2023 discoveries (hydrides, new cuprate variants; N ~ 100–200)
   - **Critical for this domain**: hydrides exhibit Tc >> previous records; temporal test reveals whether model extrapolates or simply interpolates

3. **Cross-Class Validation**:
   - Train on mixed classes (elemental, intermetallic, cuprate, organic)
   - Test on held-out hydride subset (or vice versa)
   - Reveals class-specific bias

#### Active Learning / Uncertainty Quantification

**Emerging (2024+)**:
- Models should quantify prediction uncertainty via:
  - Ensemble variance (bootstrapped models)
  - Bayesian deep learning (variational inference, Monte Carlo dropout)
  - Prediction intervals instead of point estimates
- **Application**: prioritize high-uncertainty candidates for experimental synthesis

#### Reproduction and Ablation Studies

**Ablation Tests**:
- Retrain model with single feature removed; assess Tc prediction impact
- Reveals which features genuinely drive predictions vs. spurious correlations
- Example: removing "contains hydrogen" feature should degrade predictions on hydride test set; if not, feature is weak

**Reproducibility Standards**:
- Report random seed, library versions, hardware specs
- Release code (GitHub) and training/test data splits (for reproducibility)
- Enable independent audits

### 6.9 Benchmark Comparisons

#### State-of-the-Art Results (2024)

| **Method** | **Dataset Size** | **Test R²** | **RMSE (K)** | **Notes** | **Citation** |
|-----------|----------------|-----------|-------------|----------|------------|
| **Random Forest (chemical composition only)** | 16,413 | 0.935 | 6.2 | Baseline; 5-fold CV | Published 2024 |
| **XGBoost with GBFS** | 12,000+ | 0.952 | 6.45 | Feature selection; two-layer approach | ACS Omega 2024 |
| **SOAP descriptor + cross-validation** | 5,713 | 0.929 | 7.8 | Structural info; multialgorithm verified | J. Phys. Chem. C 2024 |
| **Graph Neural Network (message passing)** | ~10,000 | 0.945 | 6.8 | Requires crystal structure; moderate computational cost | npj Comp. Mater. 2024 |
| **Attention-Based Deep Learning** | 13,022 | 0.948 | 6.5 | Adaptive feature weighting; high interpretability | ScienceDirect 2025 |
| **Deep Forest (ensemble tree method)** | ~12,000 | 0.944 | ~7 | Ensemble of ensemble; competitive; simple hyperparams | Various 2023–2024 |
| **Tempered Deep Learning (BETE-NET)** | ~10,000 | 0.950+ | <7 | Booststrapping + tempered overfitting; novel regularization | npj Comp. Mater. 2024 |

**Observations**:
- Modern methods converge to R² ~ 0.93–0.95, RMSE ~ 6–7 K
- Incremental improvements achieved via:
  - Larger datasets (10,000+ vs. 5,000)
  - Structural information (GNNs > composition-only)
  - Advanced regularization (tempered DL, bootstrapping)
- **Hydride Generalization**: Tested separately; models trained on pre-hydride data typically **underpredict** hydride Tc by 20–40 K (extrapolation failure)

### 6.10 Pitfalls and Limitations in Current Approaches

#### Data Leakage

**Risk**: Information from test set inadvertently used during training, inflating performance metrics.

**Examples**:
- Hyperparameter tuning on full dataset before splitting
- Feature selection on combined (train + test) data
- Using same data to normalize (min-max scaling) train and test separately

**Mitigation**: Nested CV; pipeline ensures all preprocessing happens post-split.

#### Spurious Correlations

**Risk**: Model learns correlations specific to training data (e.g., superconductors discovered in specific labs cluster by composition; model memorizes lab-specific patterns).

**Example**: LK-99 false positives—if dataset included pre-refutation claims, models would overweight Cu₂S signatures incorrectly.

**Mitigation**:
- Scrutinize feature importance; remove dataset artifacts
- Temporal CV to test on genuinely novel materials
- Domain expertise review before deployment

#### Composition-Only Limitations

**Fundamental Constraint**: Chemical composition alone cannot fully determine Tc:
- Crystal structure (polymorphs) critically affects Tc
- Same composition, different lattice → different Tc
- Hydride example: fcc vs. primitive cubic YH3 phases differ substantially

**Consequence**: ~15–20% of variance in Tc explainable only via structure; composition-only models asymptote to R² ~ 0.90–0.93.

**Partial Solution**: Structural descriptors (SOAP, GNN) improve to R² ~ 0.94–0.96, but require 3D structure (not always available for hypothetical compounds).

#### Extrapolation Risk (especially for hydrides)

**Critical Issue**: Models trained on pre-2015 databases (elemental, conventional metals, cuprates) fail to predict hydride Tc correctly:
- Temporal validation: pre-2015 model predicts H3S Tc ~ 50 K; actual = 203 K
- **Gap**: ~150 K underestimation
- **Cause**: Hydrides occupy extreme corner of material space (high λ, high ω_log, extreme pressures); rare in training data

**Implications**:
- Models cannot reliably guide hydride discovery without hydride-specific training data
- New material classes (e.g., high-entropy hydrides, clathrate variants) require retraining
- **Best Practice**: Report performance on held-out materials from same class, not just overall R²

#### Pressure Dependence Not Captured

**Oversight**: Most ML models predict Tc at **ambient pressure implicitly**, assuming input is composition at fixed P.

**Reality**: Hydride Tc varies dramatically with pressure:
- YH3: Tc = 0 K at 0 GPa → 40 K at 17.7 GPa → higher at further increased P
- LaH10: Tc ≈ 250 K at 170 GPa; drops significantly at <100 GPa

**Missing Information**: Models lack pressure as input feature (except in recent DFT-integrated approaches).

**Emerging Solution**: Include pressure as explicit input; retrain on high-pressure DFT-computed Tc databases (few studies, 2023+).

---

## 7. Integration: ML Predictions × Physical Bounds × Hydride Data

### 7.1 Reconciling ML Predictions with Theoretical Bounds

**Observation**: ML models trained on diverse databases predict occasional Tc > 300 K even for compositions far from known superconductors.

**Physical Reality**:
- McMillan formula, even with λ = 4, ω_log = 1800 K, μ* = 0.05, yields Tc_max ~ 250–300 K
- Predictions >300 K violate fundamental electron-phonon coupling constraints

**Resolution**:
- ML must incorporate physics constraints as regularization:
  - Cap predictions at Tc_max derived from McMillan bound
  - Weight loss higher for unphysical λ > 4 estimates
  - Integrate with ab initio EPC calculations (hybrid ML-DFT approach)

**Recent Work** (2024+): DFT-PT (perturbation theory) workflows compute λ, ω_log for each compound, then use ML to interpolate/extrapolate efficiently. Hybrid approach combines ML speed with DFT accuracy.

### 7.2 Validation of Hydride Predictions

**Challenge**: H3S, LaH10 predictions preceded synthesis (2015–2019); now 300+ papers on hydrides, but few experimental confirmations.

**Current Best Practice**:
1. **DFT Prediction**: Structure search + Eliashberg calculation predicts Tc, λ, ω_log, phase stability
2. **ML Screening**: Use composition-based ML to rapidly rank candidates (vs. DFT cost)
3. **High-Pressure Synthesis**: Synthesize most-promising at DAC (diamond anvil cell) or multi-anvil press
4. **Experimental Characterization**:
   - Resistance (R-T curves, zero-resistance confirmation)
   - Magnetization (SQUID: diamagnetism, magnetic susceptibility)
   - Spectroscopy (IR, X-ray, tunneling microscopy for gap structure)
   - **Critical**: Isotope effect (H/D substitution) to confirm phonon-mediated origin

**Red Flags for False Positives** (post-LK-99 awareness):
- Single-sample reports without independent synthesis
- Claims relying on single measurement (e.g., resistance drop without diamagnetism)
- Lack of pressure-dependent phase diagram
- No isotope effect measurement

---

## 8. Identified Gaps and Open Problems

### 8.1 Theoretical Gaps

1. **Hydride Mechanism Beyond Simple BCS**: YH6 deviations from Eliashberg theory remain unexplained. Are vertex corrections (Migdal breakdown) responsible? Or excitonic/excitonic mechanisms? Requires:
   - Precise gap structure (tunneling spectroscopy) — see December 2025 advance
   - Multi-band model analysis (beyond isotropic Eliashberg)
   - Quantum Monte Carlo simulations on realistic hydride models

2. **Pressure-Dependent Tc Landscape**: No unified theory explaining Tc(P) curves for hydrides. Trade-offs between lattice hardening (↑ω) and band structure evolution (↑λ or ↑N(EF)) are phenomenological. Requires:
   - High-throughput Eliashberg calculations at fine pressure grids
   - Machine learning of Tc(P) surfaces

3. **Ambient-Pressure Stabilization**: Path to ambient-pressure superconductivity remains unclear. Virtual pressure via doping proposed theoretically but unconfirmed experimentally. Requires:
   - Ternary/quaternary hydride design with precise stoichiometry control
   - In situ characterization during pressure release

### 8.2 Experimental Gaps

1. **Limited Hydride Superconductor Verification**: Of ~100 predicted hydride superconductors, <20 experimentally confirmed. Bottleneck is high-pressure synthesis complexity and cost.

2. **LK-99 Aftermath—Trust in Claims**: Following LK-99 retraction, skepticism warrants:
   - Mandatory multi-lab independent replication for any Tc > 100 K claim
   - Pre-registration of synthesis protocols
   - Open data sharing

3. **Anharmonic Effects Underconstrained**: While SSCHA provides computational framework, direct experimental measurement of anharmonicity in hydrides is limited. Requires:
   - Neutron diffraction at high pressure to map H positions and vibrations
   - Inelastic neutron scattering (INS) to measure phonon spectra at high P

### 8.3 ML Gaps

1. **Hydride-Specific Models**: Existing ML models trained on pre-2015 data fail on hydrides. Dedicated hydride datasets (100–500 compounds) needed to train specialized models. As of 2025, <10 papers on hydride-only ML.

2. **Pressure as Feature**: Most models ignore pressure; few attempt to predict Tc(P). Requires:
   - Large DFT-computed libraries at multiple pressures
   - Structured approaches to P-dependent feature engineering

3. **Uncertainty Quantification**: Models rarely report confidence intervals. For exploration (experimental targeting), uncertainty estimates (e.g., Bayesian/ensemble-based) are critical but underdeveloped.

4. **Out-of-Distribution Detection**: Models should flag predictions on novel materials (far from training data). Metrics like population stability index (PSI), maximum mean discrepancy (MMD) underutilized.

### 8.4 Materials Discovery Gaps

1. **Design of Room-Temperature Superconductors**: Fundamental question: is room-temperature ambient-pressure superconductivity achievable in principle? Emerging consensus: extremely unlikely for conventional phonon-mediated systems. Alternative mechanisms (exciton-mediated, magnons, topological) speculative but underdeveloped.

2. **Pressure Reduction Strategies**: Ternary hydrides show promise but progress is slow. Computational screening of millions of compositions needed, coupled with experiments.

3. **Long-Term Stability**: Hydrides are inherently unstable at low P; synthesis to use at high P feasible, but device applications require ambient stability. No clear path to solving this contradiction without sacrificing Tc.

---

## 9. State-of-the-Art Summary

### 9.1 Key Findings (2024–2025)

1. **Physical Bounds Are Real**: Fundamental electron-phonon coupling limits (λ ≤ 4) and phonon-frequency trade-offs imply ambient-pressure Tc < 50 K is almost certain for conventional superconductors. Room-temperature ambient-pressure superconductivity remains implausibly unlikely.

2. **Hydride Paradigm Shift**: H3S (2015) and successors (LaH10, YH3, YH6) demonstrate that optimizing conventional mechanisms via extreme pressure *can* yield Tc > 200 K without invoking unconventional physics. Mechanism remains electron-phonon pairing; high Tc reflects hydrogen's extreme properties (high frequency, strong coupling).

3. **Anharmonicity is Essential**: Quantum anharmonic corrections to hydride lattice dynamics are not ornamental; they quantitatively shift stability windows by 10–20 GPa and affect Tc predictions. Harmonic approximation is insufficient.

4. **ML Models Plateau at ~R² 0.95**: Composition-only ML achieves 90–93% accuracy; structural information (GNN, SOAP) extends to 94–96%. Further gains marginal and material-class-dependent. Composition-only limitation fundamentally imposed by polymorphism.

5. **Generalization to Novel Classes Fails**: Models trained on pre-hydride data underpredict hydride Tc by 100–200 K. Temporal and class-stratified validation essential; overall CV metrics misleading.

6. **LK-99 Reinforces Validation Rigor**: Single-sample claims and weak characterization (levitation video, not diamagnetism) rejected by community. Multi-lab replication, isotope effects, spectroscopy now expected standards.

### 9.2 Current Research Directions

**Short-term (2025–2027)**:
- Experimental confirmation of remaining high-pressure hydride candidates (ternary systems, LaSc2H24)
- Direct spectroscopy of H3S superconducting gap (December 2025 advance; enables precision Tc validation)
- High-throughput Eliashberg calculations + ML for pressure-dependent Tc landscapes
- Hydride-specific ML models trained on 500+ DFT-computed compounds

**Medium-term (2027–2030)**:
- Pressure-reduction strategies via doping; aim for Tc > 100 K at P < 10 GPa
- Discovery of new hydride structure types (beyond clathrate-like); computational screening of 10,000+ hypothetical hydrides
- Development of uncertainty-quantified ML for experimental targeting

**Long-term (2030+)**:
- Device applications requiring pressure vessels (high-Tc magnets, power transmission) plausible; solid-state applications unlikely
- Alternative mechanisms (exciton-mediated, topological) for ambient-pressure superconductivity remain speculative but warrant exploration
- Fundamental question of absolute Tc ceiling remains open

### 9.3 Critical Unknowns

1. Is Tc_max for conventional phonon-mediated superconductors truly bounded at ~250–300 K by fundamental physics, or merely by current material exploration and engineering?

2. Can quantum anharmonicity + correlation effects (beyond mean-field Eliashberg) yield unexpected enhancements pushing Tc to 400+ K even at ambient pressure?

3. What undiscovered hydride compositions or pressure windows harbor superconductivity; how can ML accelerate discovery without extensive DFT calculations?

4. Are there unconventional (non-phonon) pairing mechanisms operative in hydrides that have been overlooked?

---

## 10. References and Sources

### 10.1 Fundamental Physical Bounds

- Semenok, D. V., et al. (2024). "Fundamental limits on the electron-phonon coupling and superconducting Tc." *Advanced Materials*, 36(4), 2507013. [Preprint: arXiv:2407.12922]
  - URL: https://arxiv.org/html/2407.12922

- Nature Communications (2025). "The maximum Tc of conventional superconductors at ambient pressure." *Nature Communications*.
  - URL: https://www.nature.com/articles/s41467-025-63702-w

- npj Quantum Materials (2022). "Heuristic bounds on superconductivity and how to exceed them."
  - URL: https://www.nature.com/articles/s41535-022-00491-1

- PMC/ScienceDirect (2023). "Upper limit of the transition temperature of superconducting materials."
  - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9676523/

- npj Quantum Materials (2018). "A bound on the superconducting transition temperature."
  - URL: https://www.nature.com/articles/s41535-018-0133-0

### 10.2 Hydride Superconductor Discoveries

- Nature (2019). "Superconductivity at 250 K in lanthanum hydride under high pressures." Somayazulu, M., et al.
  - URL: https://www.nature.com/articles/s41586-019-1201-8
  - arXiv: https://arxiv.org/abs/1812.01561

- Nature Communications (2021). "Superconductivity up to 243 K in the yttrium-hydrogen system under high pressure."
  - URL: https://www.nature.com/articles/s41467-021-25372-2

- ACS JPCC (2023). "(La,Th)H10: Potential High-Tc (242 K) Superconductors Stabilized Thermodynamically below 200 GPa."
  - URL: https://pubs.acs.org/doi/10.1021/acs.jpcc.3c07213

- Frontiers in Electronic Materials (2022). "Hot Hydride Superconductivity Above 550 K." Review.
  - URL: https://www.frontiersin.org/journals/electronic-materials/articles/10.3389/femat.2022.837651/full

- PNAS (2024). "Designing multicomponent hydrides with potential high Tc superconductivity."
  - URL: https://www.pnas.org/doi/10.1073/pnas.2413096121

- Nature Communications (2023). "Local electronic structure rearrangements and strong anharmonicity in YH3 under pressures up to 180 GPa."
  - URL: https://www.nature.com/articles/s41467-021-21991-x

- ScienceDirect (2023). "Investigation of the effect of high pressure on the superfluid density of H3S, LaH10, and CaAlSi superconductors."
  - URL: https://www.sciencedirect.com/science/article/abs/pii/S0921453423000898

- ScienceDaily (2025, December). "Scientists unlocked a superconductor mystery under crushing pressure." [H3S tunneling spectroscopy]
  - URL: https://www.sciencedaily.com/releases/2025/12/251219093328.htm

### 10.3 Mechanisms & Anharmonic Effects

- Science Advances (2025). "Mechanism of high-temperature superconductivity in compressed H2-molecular–type hydride."
  - URL: https://www.science.org/doi/10.1126/sciadv.adt9411

- Nature Communications (2023). "Temperature and quantum anharmonic lattice effects on stability and superconductivity in lutetium trihydride."
  - URL: https://www.nature.com/articles/s41467-023-44326-4

- Communications Physics (2023). "Quantum lattice dynamics and their importance in ternary superhydride clathrates."
  - URL: https://www.nature.com/articles/s42005-023-01413-8

- arXiv (2025, December). "Self-consistent solution of Eliashberg equations for metal hydride superconductors." [2512.08126]
  - URL: https://arxiv.org/abs/2512.08126

- npj Computational Materials (2024). "Prediction of ambient pressure conventional superconductivity above 80 K in hydride compounds."
  - URL: https://www.nature.com/articles/s41524-024-01214-9

- PMC/Phys. Rev. B (2021). "Breakdown of the Migdal approximation at Lifshitz transitions with giant zero-point motion in the H3S superconductor."
  - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4837402/

### 10.4 ML Prediction & Validation

- ACS Journal of Chemical Information and Modeling (2024). "Machine-Learning Predictions of Critical Temperatures from Chemical Compositions of Superconductors."
  - URL: https://pubs.acs.org/doi/10.1021/acs.jcim.4c01137

- npj Computational Materials (2018). "Machine learning modeling of superconducting critical temperature."
  - URL: https://www.nature.com/articles/s41524-018-0085-8

- Scientific Reports (2024). "Predicting superconducting transition temperature through advanced machine learning and innovative feature engineering."
  - URL: https://www.nature.com/articles/s41598-024-54440-y

- ACS JPCC (2022). "Machine Learning Prediction of Superconducting Critical Temperature through the Structural Descriptor."
  - URL: https://pubs.acs.org/doi/10.1021/acs.jpcc.2c01904

- ACS Omega (2024). "Prediction of the Critical Temperature of Superconductors Based on Two-Layer Feature Selection and the Optuna-Stacking Ensemble Learning Model."
  - URL: https://pubs.acs.org/doi/10.1021/acsomega.2c06324

- npj Computational Materials (2024). "Accelerating superconductor discovery through tempered deep learning of the electron-phonon spectral function."
  - URL: https://www.nature.com/articles/s41524-024-01475-4

- arXiv (2024, December). "Deep Learning Based Superconductivity Prediction and Experimental Tests." [2412.13012]
  - URL: https://arxiv.org/html/2412.13012v1

- ScienceDirect (2023). "Predicting the critical superconducting temperature using the random forest, MLP neural network, M5 model tree and multivariate linear regression."
  - URL: https://www.sciencedirect.com/science/article/pii/S1110016823010116

- Frontiers in Materials (2021). "Deep Learning Approach for Prediction of Critical Temperature of Superconductor Materials Described by Chemical Formulas."
  - URL: https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2021.714752/full

- npj Computational Materials (2022). "Designing high-TC superconductors with BCS-inspired screening, density functional theory, and deep-learning."
  - URL: https://www.nature.com/articles/s41524-022-00933-1

### 10.5 DFT & High-Pressure Design

- PMC (2024). "Data-driven Design of High Pressure Hydride Superconductors using DFT and Deep Learning."
  - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11151870/
  - arXiv: https://arxiv.org/html/2312.12694

- ACS Chemistry of Materials (2015). "High-Pressure Phase Stability and Superconductivity of Pnictogen Hydrides and Chemical Trends for Compressed Hydrides."
  - URL: https://pubs.acs.org/doi/10.1021/acs.chemmater.5b04638

- Physical Review B (2023). "Stabilizing a hydrogen-rich superconductor at 1 GPa by charge transfer modulated virtual high-pressure effect." Gao, et al.
  - URL: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.L180501

### 10.6 Validation & Negative Results

- ACS Omega (2023). "Absence of Superconductivity in LK-99 at Ambient Conditions."
  - URL: https://pubs.acs.org/doi/10.1021/acsomega.3c06096

- Nature (2023, August). "LK-99 isn't a superconductor — how science sleuths solved the mystery." Editorial.
  - URL: https://www.nature.com/articles/d41586-023-02585-7

- Phys.org (2023, November). "Myth of room temperature superconductivity in LK-99 is shattered."
  - URL: https://phys.org/news/2023-11-myth-room-temperature-superconductivity-lk-.html

- The Quantum Insider (2024, January). "Absence of Superconductivity in LK-99 at Ambient Conditions."
  - URL: https://thequantuminsider.com/2024/01/04/its-back-researchers-say-theyve-replicated-lk-99-room-temperature-superconductor-experiment/

- IOPscience (2023/2024). "Replication and study of anomalies in LK-99."
  - URL: https://iopscience.iop.org/article/10.1088/1361-6668/ad2b78/ampdf

### 10.7 Recent Reviews and Overviews

- National Science Review (2024). "Current status and future development of high-temperature conventional superconductivity."
  - URL: https://academic.oup.com/nsr/article/11/7/nwae047/7613947

- National Science Review (2023). "Superconducting ternary hydrides: progress and challenges."
  - URL: https://academic.oup.com/nsr/article/11/7/nwad307/7462326

- Nature Physics (2023). "Magnetic flux trapping in hydrogen-rich high-temperature superconductors."
  - URL: https://www.nature.com/articles/s41567-023-02089-1

---

## 11. Key Takeaways for Research Integration

### For Theoretical Work:
- Leverage fundamental bounds (λ ≤ 4, ω-λ trade-off) as constraints in hydride optimization.
- Deploy Eliashberg equations with anharmonic corrections (SSCHA) for accurate Tc prediction.
- Investigate YH6 and other anomalous cases via multi-band models and vertex corrections.

### For ML Development:
- Condition predictions on pressure; design pressure-dependent Tc models.
- Train hydride-specific models separately; temporal/class-stratified validation mandatory.
- Incorporate physics constraints (λ caps, Tc bounds) into loss functions or as post-hoc regularization.
- Quantify uncertainty (Bayesian/ensemble) for experimental targeting.

### For Experimental Validation:
- Multi-lab replication, isotope effects, and spectroscopic gap measurements are non-negotiable for high-Tc claims.
- Report not just Tc but full phase stability diagram (P, T dependence).
- Engage computational teams early to guide synthesis efforts.

### For Database & Curation:
- Separate hydride/high-pressure data from ambient-pressure systems; avoid mixed training.
- Mandate LK-99-like screening (disputed claims flagged or excluded).
- Release pressure-dependent Tc data to enable Tc(P) model development.

---

*End of Literature Review*

**Document Generated**: December 2025
**Scope**: Comprehensive synthesis of physical bounds, hydride discoveries (2015–2025), mechanisms, and ML validation standards.
**Next Steps for User**: This review forms the foundation for subsequent sections on:
1. Novel materials discovery pipeline design
2. Experimental protocol development for high-pressure superconductivity
3. ML model architecture and training specifications
4. Pressure-dependent prediction framework
