# Theoretical Framework: X-ray Classification of AGN vs Star-Forming Galaxies

## 1. Problem Formalization

### 1.1 Objective
Develop a rigorous mathematical framework for distinguishing Active Galactic Nuclei (AGN) from Star-Forming Galaxies (SFG) using multi-wavelength X-ray observations, spectral analysis, and derived physical parameters.

### 1.2 Notation and Definitions

#### Primary Observable Variables
| Symbol | Definition | Units |
|--------|------------|-------|
| F_X | Observed X-ray flux | erg s^{-1} cm^{-2} |
| L_X | X-ray luminosity | erg s^{-1} |
| Gamma | Photon index (power-law slope) | dimensionless |
| N_H | Hydrogen column density (absorption) | cm^{-2} |
| E | Photon energy | keV |
| z | Redshift | dimensionless |
| HR | Hardness ratio | dimensionless |
| kT | Plasma temperature | keV |
| Z | Metallicity | Z_solar |

#### Derived Parameters
| Symbol | Definition | Units |
|--------|------------|-------|
| alpha_OX | Optical-to-X-ray spectral index | dimensionless |
| L_X/L_IR | X-ray to infrared luminosity ratio | dimensionless |
| L_X/L_FIR | X-ray to far-infrared luminosity ratio | dimensionless |
| EW_Fe | Iron K-alpha equivalent width | eV |
| SFR | Star formation rate | M_solar yr^{-1} |

---

## 2. Physical Models

### 2.1 AGN X-ray Emission Model

The X-ray spectrum of an AGN is modeled as a composite of multiple physical components:

#### 2.1.1 Primary Continuum (Corona)
The hot corona above the accretion disk produces X-rays via inverse Compton scattering:

```
F_AGN,primary(E) = K_AGN * E^{-Gamma} * exp(-E/E_cut)
```

Where:
- K_AGN: Normalization constant (photons keV^{-1} cm^{-2} s^{-1} at 1 keV)
- Gamma: Photon index, typically 1.5 < Gamma < 2.5, with mean ~ 1.9
- E_cut: High-energy cutoff, typically 100-300 keV

#### 2.1.2 Reflection Component
Compton reflection from the accretion disk or torus:

```
F_reflection(E) = R * F_AGN,primary(E) * G(E, theta_i)
```

Where:
- R: Reflection fraction (0 < R < 2 typical)
- G(E, theta_i): Angle-dependent reflection Green's function
- theta_i: Inclination angle

#### 2.1.3 Iron K-alpha Emission
Fluorescent iron line at 6.4 keV (neutral) to 6.97 keV (H-like):

```
F_Fe(E) = A_Fe * phi(E - E_Fe, sigma_Fe)
```

Where:
- A_Fe: Line flux (photons cm^{-2} s^{-1})
- E_Fe: Line centroid energy
- sigma_Fe: Line width (Gaussian or relativistically broadened)
- phi: Line profile function

#### 2.1.4 Photoelectric Absorption
```
F_observed(E) = F_intrinsic(E) * exp(-N_H * sigma(E))
```

Where sigma(E) is the energy-dependent photoelectric cross-section:
```
sigma(E) ~ sigma_0 * (E/E_0)^{-3}  for E > 0.5 keV
```

#### 2.1.5 Complete AGN Model
```
F_AGN(E) = exp(-N_H * sigma(E)) * [F_primary(E) + F_reflection(E) + F_Fe(E) + F_soft(E)]
```

Where F_soft(E) represents soft excess emission (often modeled as blackbody or warm Comptonization).

### 2.2 Star-Forming Galaxy X-ray Emission Model

X-ray emission from SFGs arises from stellar endpoints and hot interstellar medium:

#### 2.2.1 X-ray Binary Population (XRB)
High-mass X-ray binaries (HMXB) dominate in actively star-forming regions:

```
L_HMXB = alpha_HMXB * SFR
```

Where alpha_HMXB ~ (2-3) x 10^{39} erg s^{-1} (M_solar yr^{-1})^{-1}

Low-mass X-ray binaries (LMXB) scale with stellar mass:
```
L_LMXB = alpha_LMXB * M_*
```

Where alpha_LMXB ~ (1-2) x 10^{29} erg s^{-1} M_solar^{-1}

#### 2.2.2 Hot Gas Emission
Thermal plasma from supernova-heated ISM:

```
F_thermal(E) = Sum_i [ EM_i * Lambda(E, kT_i, Z) / (4 * pi * D_L^2) ]
```

Where:
- EM_i: Emission measure of component i (n_e * n_H * V)
- Lambda(E, kT, Z): Cooling function (APEC/MEKAL models)
- kT_i: Plasma temperature (typically 0.2-1.0 keV)
- D_L: Luminosity distance

#### 2.2.3 Complete SFG Model
```
F_SFG(E) = exp(-N_H,Gal * sigma(E)) * [F_XRB(E) + F_thermal(E)]
```

Where:
```
F_XRB(E) = Sum_j w_j * F_j(E)
```

Representing a population of XRBs with spectral templates F_j and weights w_j.

### 2.3 Spectral Shape Parameterization

#### Power-Law Representation
```
F(E) = K * E^{-Gamma}

log(F) = log(K) - Gamma * log(E)
```

#### Hardness Ratio Definition
```
HR = (H - S) / (H + S)
```

Where:
- H: Count rate in hard band (typically 2-10 keV)
- S: Count rate in soft band (typically 0.5-2 keV)

For power-law spectrum:
```
HR = f(Gamma, N_H, z)
```

---

## 3. Discriminant Functions and Classification Boundaries

### 3.1 Luminosity-Based Discrimination

#### Hypothesis H1: Luminosity Threshold
AGN typically exhibit L_X(2-10 keV) > L_threshold

```
P(AGN | L_X) = sigmoid(beta_0 + beta_1 * log(L_X))
```

Threshold criterion:
```
Classification = AGN   if L_X > 10^{42} erg s^{-1}
               = SFG   if L_X < 10^{41} erg s^{-1}
               = Ambiguous otherwise
```

### 3.2 Spectral Index Discrimination

#### Hypothesis H2: Photon Index Distribution
AGN and SFG exhibit overlapping but distinguishable Gamma distributions:

```
Gamma_AGN ~ N(mu_AGN, sigma_AGN^2)  where mu_AGN ~ 1.9, sigma_AGN ~ 0.3
Gamma_SFG ~ N(mu_SFG, sigma_SFG^2)  where mu_SFG ~ 1.7, sigma_SFG ~ 0.4
```

Note: SFG spectra often appear harder due to combined HMXB population.

### 3.3 Multi-Wavelength Ratios

#### X-ray to Optical/IR Ratio
```
alpha_OX = -0.384 * log(L_X / L_2500A)
```

For AGN: alpha_OX correlates with L_2500A (Steffen relation)
```
alpha_OX = -0.137 * log(L_2500A) + 2.638
```

#### X-ray to Star Formation Correlation
For SFGs, X-ray luminosity follows star formation:
```
log(L_X) = log(alpha_SFR) + log(SFR) + epsilon
```

Where epsilon ~ N(0, sigma_scatter^2) with sigma_scatter ~ 0.4 dex

Diagnostic ratio:
```
R_XSF = L_X / (alpha_SFR * SFR)

Classification: AGN if R_XSF > R_threshold (typically R_threshold ~ 3-10)
```

### 3.4 Absorption Diagnostics

#### Column Density Distribution
```
N_H,AGN: Bimodal distribution
  - Unabsorbed: N_H < 10^{22} cm^{-2}
  - Absorbed: 10^{22} < N_H < 10^{24} cm^{-2}
  - Compton-thick: N_H > 10^{24} cm^{-2}

N_H,SFG: Typically N_H < 10^{22} cm^{-2} (host galaxy only)
```

### 3.5 Iron Line Diagnostics

#### Hypothesis H3: Iron K-alpha Detection
Significant Fe K-alpha emission (EW > 100 eV) strongly indicates AGN:

```
P(AGN | EW_Fe > 100 eV) > 0.95
```

For Compton-thick AGN: EW_Fe can exceed 1 keV.

---

## 4. Testable Hypotheses

### Hypothesis H1: Luminosity-SFR Excess
**Statement:** Sources with X-ray luminosity exceeding the SFR-predicted value by more than factor delta contain an AGN component.

**Formal Expression:**
```
H1: L_X,observed > delta * alpha_SFR * SFR  =>  P(AGN) > p_threshold

Where: delta = 3, p_threshold = 0.8
```

**Falsification Criterion:** If more than 20% of spectroscopically confirmed pure SFGs exceed this threshold, H1 is falsified.

### Hypothesis H2: Hardness-Luminosity Separation
**Statement:** In the (HR, L_X) plane, AGN and SFG occupy statistically separable regions.

**Formal Expression:**
```
H2: D_KL(P_AGN(HR, L_X) || P_SFG(HR, L_X)) > D_threshold

Where D_KL is the Kullback-Leibler divergence, D_threshold = 1.0
```

**Falsification Criterion:** If the 2D distributions show D_KL < 0.5, the populations are not separable using these diagnostics alone.

### Hypothesis H3: Spectral Curvature Indicator
**Statement:** AGN exhibit greater spectral complexity (curvature, reflection features) than SFGs when sufficient counts are available.

**Formal Expression:**
```
H3: Chi^2_powerlaw / Chi^2_complex > R_AGN for AGN
    Chi^2_powerlaw / Chi^2_complex ~ 1 for SFG

Where R_AGN > 1.5 indicates significant spectral complexity
```

**Falsification Criterion:** If the distribution of chi-squared ratios is indistinguishable between AGN and SFG samples.

### Hypothesis H4: Multi-Wavelength Diagnostic Convergence
**Statement:** Combining X-ray spectral parameters with optical/IR indicators increases classification accuracy compared to X-ray alone.

**Formal Expression:**
```
H4: Accuracy(X-ray + multiwavelength) - Accuracy(X-ray only) > delta_acc

Where delta_acc = 0.10 (10% improvement)
```

**Falsification Criterion:** If accuracy improvement is less than 5%, multi-wavelength data provides negligible benefit.

---

## 5. Statistical Classification Framework

### 5.1 Feature Vector Definition
```
x = [log(L_X), Gamma, N_H, HR, log(L_X/L_IR), log(L_X/SFR), EW_Fe, alpha_OX]^T
```

Dimension: d = 8 (or reduced via PCA if features correlated)

### 5.2 Probabilistic Classification Model

#### Bayesian Framework
```
P(AGN | x) = P(x | AGN) * P(AGN) / P(x)

P(x) = P(x | AGN) * P(AGN) + P(x | SFG) * P(SFG)
```

#### Likelihood Models
Option A: Gaussian Mixture Model
```
P(x | class) = Sum_k pi_k * N(x | mu_k, Sigma_k)
```

Option B: Kernel Density Estimation
```
P(x | class) = (1/n) * Sum_i K_h(x - x_i)
```

### 5.3 Decision Boundary
```
Classify as AGN if: P(AGN | x) > tau

Default threshold: tau = 0.5
Adjustable based on precision/recall requirements
```

### 5.4 Evaluation Metrics
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
AUC-ROC = Area under ROC curve
```

---

## 6. Pseudocode: AGN/SFG Classification Algorithm

### 6.1 Data Preparation

```
ALGORITHM: PrepareXrayData

INPUT:
  - raw_spectra: Array of X-ray spectral files (PHA/PI format)
  - catalog: Source catalog with positions, redshifts
  - ancillary_data: Optical/IR photometry, SFR estimates

OUTPUT:
  - feature_matrix: X (N x d matrix)
  - labels: y (N x 1 vector, for training set)

PROCEDURE:

1. FOR each source i in catalog:

   2. LOAD X-ray spectrum spectrum_i from raw_spectra[i]

   3. LOAD response matrices (ARF, RMF) for spectrum_i

   4. COMPUTE observed flux:
      F_soft = integrate(spectrum_i, E=0.5-2 keV)
      F_hard = integrate(spectrum_i, E=2-10 keV)
      F_total = integrate(spectrum_i, E=0.5-10 keV)

   5. COMPUTE hardness ratio:
      HR_i = (F_hard - F_soft) / (F_hard + F_soft)

   6. IF counts > 200 THEN:
      6a. FIT power-law model: F(E) = K * E^{-Gamma} * exp(-N_H * sigma(E))
      6b. EXTRACT: Gamma_i, N_H_i, K_i, chi2_i
      6c. FIT thermal model: F(E) = EM * Lambda(E, kT)
      6d. COMPARE chi2 values
      6e. IF Fe K-alpha detected: MEASURE EW_Fe_i
   ELSE:
      6f. USE hardness ratio to estimate Gamma_i
      6g. SET N_H_i = N_H_galactic (from HI maps)
      6h. SET EW_Fe_i = NaN (not measurable)

   7. COMPUTE luminosity:
      D_L = luminosity_distance(z_i)
      L_X_i = 4 * pi * D_L^2 * F_total * K_correction(z_i, Gamma_i)

   8. RETRIEVE ancillary data:
      L_IR_i = infrared_luminosity(ancillary_data[i])
      SFR_i = star_formation_rate(ancillary_data[i])
      L_opt_i = optical_luminosity(ancillary_data[i])

   9. COMPUTE multi-wavelength ratios:
      R_XIR_i = log10(L_X_i / L_IR_i)
      R_XSF_i = log10(L_X_i / (alpha_SFR * SFR_i))
      alpha_OX_i = -0.384 * log10(L_X_i / L_opt_i)

   10. CONSTRUCT feature vector:
       x_i = [log10(L_X_i), Gamma_i, log10(N_H_i), HR_i,
              R_XIR_i, R_XSF_i, EW_Fe_i, alpha_OX_i]

   11. HANDLE missing values:
       IF any(isnan(x_i)):
          IMPUTE using median/mode from training set
          OR flag for exclusion from certain analyses

12. STACK all feature vectors:
    X = vstack([x_0, x_1, ..., x_{N-1}])

13. NORMALIZE features:
    FOR each feature j:
       X[:, j] = (X[:, j] - mean(X[:, j])) / std(X[:, j])

14. RETURN X, y
```

### 6.2 Model Training

```
ALGORITHM: TrainClassifier

INPUT:
  - X_train: Training feature matrix (N_train x d)
  - y_train: Training labels (N_train x 1), where 1=AGN, 0=SFG
  - model_type: "RandomForest" | "GradientBoost" | "NeuralNet" | "Bayesian"

OUTPUT:
  - trained_model: Fitted classification model
  - feature_importance: Ranking of feature importance

PROCEDURE:

1. SPLIT data for cross-validation:
   folds = stratified_k_fold(X_train, y_train, k=5)

2. IF model_type == "RandomForest":
   2a. INITIALIZE: n_estimators=500, max_depth=10, min_samples_leaf=5
   2b. FOR each fold in folds:
       - TRAIN random forest on training fold
       - EVALUATE on validation fold
       - RECORD: accuracy, precision, recall, F1, AUC
   2c. SELECT hyperparameters via grid search over:
       {n_estimators: [100, 500, 1000], max_depth: [5, 10, 20],
        min_samples_split: [2, 5, 10]}

3. ELIF model_type == "GradientBoost":
   3a. INITIALIZE: n_estimators=200, learning_rate=0.1, max_depth=5
   3b. TRAIN with early stopping on validation set
   3c. TUNE: learning_rate in [0.01, 0.05, 0.1], max_depth in [3, 5, 7]

4. ELIF model_type == "NeuralNet":
   4a. ARCHITECTURE:
       Input(d) -> Dense(64, ReLU) -> Dropout(0.3) ->
       Dense(32, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)
   4b. LOSS: Binary cross-entropy
   4c. OPTIMIZER: Adam, learning_rate=0.001
   4d. TRAIN for max 100 epochs with early stopping (patience=10)

5. ELIF model_type == "Bayesian":
   5a. ESTIMATE class-conditional densities:
       P(x | AGN) using kernel density estimation
       P(x | SFG) using kernel density estimation
   5b. ESTIMATE priors from training set:
       P(AGN) = sum(y_train) / len(y_train)
       P(SFG) = 1 - P(AGN)
   5c. STORE density estimators and priors

6. COMPUTE feature importance:
   IF tree-based model:
      feature_importance = model.feature_importances_
   ELIF neural network:
      feature_importance = permutation_importance(model, X_train, y_train)
   ELIF Bayesian:
      feature_importance = mutual_information(X_train, y_train)

7. PERFORM final training on full training set with best hyperparameters

8. RETURN trained_model, feature_importance
```

### 6.3 Classification and Uncertainty Quantification

```
ALGORITHM: ClassifySource

INPUT:
  - x_new: Feature vector for new source (1 x d)
  - trained_model: Fitted classification model
  - threshold: Classification probability threshold (default 0.5)
  - uncertainty_method: "bootstrap" | "dropout" | "posterior"

OUTPUT:
  - classification: "AGN" | "SFG" | "Ambiguous"
  - probability: P(AGN | x_new)
  - confidence_interval: [p_lower, p_upper]
  - flag: Quality/reliability indicator

PROCEDURE:

1. PREDICT probability:
   p_AGN = trained_model.predict_proba(x_new)[AGN_class]

2. QUANTIFY uncertainty:
   IF uncertainty_method == "bootstrap":
      2a. FOR b = 1 to B (B=1000):
          - RESAMPLE training data with replacement
          - RETRAIN model (or use pre-trained bootstrap ensemble)
          - p_AGN_b = model_b.predict_proba(x_new)
      2b. confidence_interval = [percentile(p_AGN_b, 2.5),
                                  percentile(p_AGN_b, 97.5)]
      2c. uncertainty = std(p_AGN_b)

   ELIF uncertainty_method == "dropout":
      2d. ENABLE dropout at test time
      2e. FOR t = 1 to T (T=100):
          - p_AGN_t = model.predict_proba(x_new, training=True)
      2f. p_AGN = mean(p_AGN_t)
      2g. confidence_interval = [mean - 1.96*std, mean + 1.96*std]

   ELIF uncertainty_method == "posterior":
      2h. USE Bayesian model posterior:
          p_AGN = integral(P(AGN | x, theta) * P(theta | data) d_theta)
      2i. confidence_interval from posterior quantiles

3. APPLY classification rules:
   IF p_AGN > threshold + margin:
      classification = "AGN"
   ELIF p_AGN < threshold - margin:
      classification = "SFG"
   ELSE:
      classification = "Ambiguous"

   WHERE margin = 0.1 (adjustable)

4. SET quality flag:
   flag = "HIGH" if uncertainty < 0.1
   flag = "MEDIUM" if 0.1 <= uncertainty < 0.2
   flag = "LOW" if uncertainty >= 0.2

5. CHECK for anomalies:
   IF mahalanobis_distance(x_new, X_train) > chi2_threshold:
      flag = "OUTLIER"  // Source may not fit either category

6. RETURN classification, p_AGN, confidence_interval, flag
```

### 6.4 Diagnostic Diagram Generation

```
ALGORITHM: GenerateDiagnosticDiagrams

INPUT:
  - X: Feature matrix (N x d)
  - y_true: True labels (if available)
  - y_pred: Predicted labels
  - p_pred: Predicted probabilities

OUTPUT:
  - diagnostic_plots: Collection of visualization objects
  - separation_metrics: Quantitative separation measures

PROCEDURE:

1. LUMINOSITY-HARDNESS DIAGRAM:
   1a. EXTRACT: L_X = 10^X[:, 0], HR = X[:, 3]
   1b. PLOT scatter: x=log10(L_X), y=HR, color=y_pred
   1c. OVERLAY: Decision boundary contours from model
   1d. COMPUTE: Separation metric = silhouette_score(X[:, [0,3]], y_pred)

2. X-RAY vs SFR DIAGRAM:
   2a. EXTRACT: L_X, R_XSF = X[:, 5]
   2b. PLOT: log(L_X) vs log(SFR)
   2c. OVERLAY: Expected relation for pure SFGs: L_X = alpha_SFR * SFR
   2d. OVERLAY: AGN threshold line: L_X = 3 * alpha_SFR * SFR
   2e. COMPUTE: Fraction of each class above/below threshold

3. PHOTON INDEX HISTOGRAM:
   3a. EXTRACT: Gamma = X[:, 1]
   3b. PLOT: Overlapping histograms for AGN and SFG
   3c. FIT: Gaussian to each distribution
   3d. COMPUTE: KL divergence between distributions

4. MULTI-PARAMETER CORNER PLOT:
   4a. SELECT: Top 4 features by importance
   4b. PLOT: Corner plot with 2D density contours
   4c. COLOR: By classification (AGN/SFG)

5. ROC CURVE (if y_true available):
   5a. COMPUTE: fpr, tpr, thresholds = roc_curve(y_true, p_pred)
   5b. PLOT: ROC curve
   5c. COMPUTE: AUC = auc(fpr, tpr)
   5d. IDENTIFY: Optimal threshold (Youden's J or cost-based)

6. CONFUSION MATRIX:
   6a. COMPUTE: cm = confusion_matrix(y_true, y_pred)
   6b. PLOT: Heatmap of confusion matrix
   6c. REPORT: Precision, Recall, F1 for each class

7. PROBABILITY CALIBRATION:
   7a. BIN predictions by probability
   7b. COMPUTE: Fraction of true positives in each bin
   7c. PLOT: Calibration curve (reliability diagram)
   7d. COMPUTE: Brier score = mean((p_pred - y_true)^2)

8. RETURN diagnostic_plots, separation_metrics
```

### 6.5 Full Pipeline Execution

```
ALGORITHM: AGN_SFG_ClassificationPipeline

INPUT:
  - data_directory: Path to X-ray spectral data
  - catalog_file: Source catalog with metadata
  - training_labels: Known classifications for subset
  - config: Configuration parameters

OUTPUT:
  - classifications: Final source classifications
  - probabilities: Classification probabilities
  - diagnostics: Diagnostic plots and metrics
  - model: Trained classification model

PROCEDURE:

1. CONFIGURATION:
   1a. SET energy_bands = [(0.5, 2.0), (2.0, 10.0)]  // soft, hard
   1b. SET luminosity_threshold = 10^42 erg/s
   1c. SET classification_threshold = 0.5
   1d. SET model_type = "RandomForest"
   1e. SET uncertainty_method = "bootstrap"

2. DATA LOADING:
   2a. LOAD catalog from catalog_file
   2b. LOAD training labels
   2c. VALIDATE: Check for required columns (RA, Dec, z, SFR, etc.)

3. FEATURE EXTRACTION:
   3a. CALL PrepareXrayData(data_directory, catalog, ancillary_data)
   3b. RECEIVE: X, y_known (for sources with known labels)
   3c. SPLIT: X_train, X_test, y_train, y_test (stratified 80/20)

4. QUALITY FILTERING:
   4a. REMOVE sources with:
       - z < 0 or z > 5 (invalid redshift)
       - Gamma < 0 or Gamma > 4 (unphysical)
       - N_H < 10^{19} (below Galactic minimum)
   4b. FLAG sources with:
       - counts < 50 (low S/N)
       - large uncertainties in key parameters

5. MODEL TRAINING:
   5a. CALL TrainClassifier(X_train, y_train, model_type)
   5b. RECEIVE: trained_model, feature_importance
   5c. PRINT: Feature importance ranking

6. MODEL VALIDATION:
   6a. PREDICT on X_test
   6b. COMPUTE metrics: Accuracy, Precision, Recall, F1, AUC
   6c. IF AUC < 0.8:
       WARN: "Model performance may be insufficient"
       SUGGEST: "Consider additional features or data cleaning"

7. CLASSIFICATION:
   7a. FOR each source with unknown classification:
       7b. CALL ClassifySource(x_i, trained_model, threshold, uncertainty_method)
       7c. STORE: classification_i, probability_i, confidence_i, flag_i

8. POST-PROCESSING:
   8a. APPLY physical consistency checks:
       - IF L_X > 10^{45} AND classification == "SFG":
         FLAG as "Likely misclassified or blended"
       - IF EW_Fe > 500 eV AND classification == "SFG":
         RECLASSIFY as "AGN" with note
   8b. CROSS-CHECK with multi-wavelength indicators:
       - Optical spectroscopy (if available)
       - Radio loudness
       - Mid-IR colors (WISE W1-W2)

9. DIAGNOSTICS:
   9a. CALL GenerateDiagnosticDiagrams(X, y_true, y_pred, p_pred)
   9b. RECEIVE: diagnostic_plots, separation_metrics
   9c. SAVE plots to output directory

10. OUTPUT:
    10a. WRITE classifications to catalog_output.fits
    10b. INCLUDE columns: [source_id, classification, p_AGN,
                          confidence_lower, confidence_upper, flag]
    10c. WRITE model to model_output.pkl
    10d. WRITE diagnostic report to report.pdf

11. RETURN classifications, probabilities, diagnostics, trained_model
```

---

## 7. Confirmation and Falsification Criteria

### 7.1 Confirmation of Framework Validity

The theoretical framework is confirmed if:

1. **Classification Accuracy:** The algorithm achieves AUC-ROC > 0.85 on an independent test set with spectroscopically confirmed sources.

2. **Physical Consistency:** Classified AGN show properties consistent with theoretical expectations:
   - Mean Gamma within [1.7, 2.1]
   - Correlation between L_X and L_optical follows AGN scaling relations
   - Detection rate of Fe K-alpha emission > 50% for Compton-thick candidates

3. **Multi-wavelength Agreement:** Classification agrees with independent optical/IR AGN indicators (BPT diagram, mid-IR excess) in > 80% of cases.

4. **Hypothesis Testing:** Hypotheses H1-H4 produce statistically significant results (p < 0.05) in the predicted direction.

### 7.2 Falsification Criteria

The framework is falsified (requires revision) if:

1. **Poor Separation:** AUC-ROC < 0.70, indicating AGN and SFG populations are not distinguishable using these features.

2. **Biased Performance:** Systematic misclassification of specific source subpopulations (e.g., low-luminosity AGN consistently classified as SFG).

3. **Feature Irrelevance:** Feature importance analysis shows X-ray spectral parameters contribute less than 10% to classification, suggesting other wavelengths dominate.

4. **Physical Inconsistency:** Classified sources violate physical constraints (e.g., "AGN" with L_X < 10^{38} erg/s, or "SFG" with Compton-thick absorption).

5. **Temporal Instability:** Classification changes significantly (>20%) when applied to different epochs of data, indicating sensitivity to observational artifacts.

---

## 8. Data Requirements and Parameters

### 8.1 Minimum Data Requirements

| Requirement | Specification |
|-------------|---------------|
| X-ray observations | Chandra, XMM-Newton, or NuSTAR |
| Energy range | 0.5-10 keV (minimum), extended to 30 keV preferred |
| Minimum counts | 50 (hardness ratio only), 200+ (spectral fitting) |
| Spectral resolution | E/dE > 20 at 6 keV for Fe K detection |
| Redshift | Required for luminosity calculation |
| Infrared photometry | WISE W1-W4 or Spitzer for SFR estimation |
| Optical photometry | For alpha_OX calculation |

### 8.2 Algorithm Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| luminosity_threshold | 10^42 erg/s | 10^41 - 10^43 | AGN luminosity threshold |
| classification_threshold | 0.5 | 0.3 - 0.7 | Probability threshold |
| alpha_SFR | 2.6e39 | 2-4 x 10^39 | L_X/SFR normalization |
| Gamma_mean_AGN | 1.9 | 1.7 - 2.1 | Expected AGN photon index |
| N_H_threshold | 10^22 cm^-2 | 10^21 - 10^23 | Absorbed/unabsorbed boundary |
| EW_Fe_threshold | 100 eV | 50 - 200 | Fe K-alpha detection threshold |

---

## 9. Summary

This theoretical framework provides:

1. **Mathematical formalization** of X-ray emission models for AGN and SFG, including all relevant spectral components and physical parameters.

2. **Discriminant functions** based on luminosity, spectral shape, absorption, and multi-wavelength ratios.

3. **Four testable hypotheses** (H1-H4) with explicit falsification criteria.

4. **Complete pseudocode** for a classification pipeline including:
   - Data preparation and feature extraction
   - Model training with multiple algorithm options
   - Classification with uncertainty quantification
   - Diagnostic visualization
   - Quality control and validation

5. **Confirmation/falsification criteria** to validate the framework against observational data.

The framework is designed to be directly implementable by an experimentalist without requiring additional interpretation of the theoretical constructs.

---

## References and Physical Constants

### Constants Used
- alpha_SFR = 2.6 x 10^39 erg s^-1 (M_sun/yr)^-1 (Lehmer et al. 2010)
- alpha_LMXB = 1.5 x 10^29 erg s^-1 M_sun^-1 (Gilfanov 2004)
- sigma_T = 6.65 x 10^-25 cm^2 (Thomson cross-section)
- H_0 = 70 km/s/Mpc, Omega_M = 0.3, Omega_Lambda = 0.7

### Standard Spectral Models
- Power-law: XSPEC model "powerlaw"
- Absorbed power-law: XSPEC model "tbabs*powerlaw"
- Thermal plasma: XSPEC model "apec" or "mekal"
- Reflection: XSPEC model "pexrav" or "xillver"
