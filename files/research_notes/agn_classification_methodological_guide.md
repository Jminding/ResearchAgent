# Methodological Guide: Selecting and Applying AGN Classification Techniques

**Compiled:** December 22, 2025
**Purpose:** Practical guidance for researchers selecting classification methods for specific scientific goals

---

## 1. QUICK-START: WHICH METHOD TO USE?

### Decision Framework by Science Question

#### Question 1: "I want a census of AGN in my galaxy sample. How many are AGN vs. star-forming?"

**Constraints → Recommended Approach:**

- **If z < 0.3 and you have optical spectra:**
  - Use: **BPT diagram** (primary) + **WHAN diagram** (for weak emitters)
  - Add: WISE W1−W2 color (if available) to catch Type 2
  - Expected accuracy: 90% purity, 85% completeness
  - Time: < 1 minute per galaxy

- **If z < 0.3 but NO optical spectra (only photometry):**
  - Use: **WISE W1−W2 ≥ 0.8** (simple + fast)
  - Add: Random Forest ML classifier if multi-band photometry available
  - Expected accuracy: 88% purity, 75% completeness (biased toward Type 1)
  - Time: < 1 second per galaxy
  - **Caveat:** Type 2 AGN and cool-torus systems systematically missed

- **If 0.3 < z < 1 (intermediate redshift):**
  - Use: **Multi-component SED fitting** (CIGALE/AGNfitter) as primary
  - Add: **X-ray selection** (if Chandra/XMM available) for confirmation
  - Add: **Mid-IR colors** (WISE/Spitzer) as secondary diagnostic
  - Expected accuracy: 85–88% purity, 78–85% completeness
  - Time: 5–10 minutes per galaxy

- **If z > 1 (high redshift):**
  - Use: **X-ray + radio + mid-IR SED fitting** (multi-wavelength consensus)
  - Optical diagnostics unreliable; avoid BPT
  - For z > 3: **JWST spectroscopy** (if available) with new photoionization models
  - Expected accuracy: 78–85% purity, 70–80% completeness
  - Time: 10–30 minutes per galaxy (SED fitting)

---

#### Question 2: "How obscured are my AGN? Are any Compton-thick?"

**Constraints → Recommended Approach:**

- **Primary method: X-ray spectroscopy** (Chandra/XMM)
  1. Fit absorption column density N_H from soft X-ray (0.5–2 keV) suppression
  2. Check for reflection component (Fe Kα at 6.4 keV; equivalent width)
  3. Hardness ratio analysis (HR1, HR2) for quick estimates

  - N_H < 10^22: Unobscured (Type 1)
  - 10^22 < N_H < 10^24: Moderately obscured (Type 2)
  - N_H > 1.5×10^24: Compton-thick (signatures: strong reflection, high Fe Kα EW)

- **Secondary method: Mid-IR + X-ray combination**
  - If L_MIR > 10^11 L_sun but X-ray flux very low/undetected → Compton-thick candidate
  - Silicate 9.7 μm absorption feature (IRS): strong absorption indicates high obscuration

- **Tertiary method: Radio detection**
  - Unaffected by obscuration; if radio-detected but X-ray and optical weak → likely CT AGN

**Expected accuracies:**
- Obscured (N_H > 10^22) identification: 90% from X-ray spectroscopy
- Compton-thick (N_H > 1.5×10^24) identification: 60–70% (challenging)

**Recommendation:** Never rely on single method for Compton-thick classification; require convergence of X-ray + mid-IR + radio diagnostics.

---

#### Question 3: "What is the star-formation rate in my AGN hosts?"

**Constraints → Recommended Approach:**

- **Problem:** AGN heating of dust and photoionization contaminate SFR indicators
  - Hα luminosity enhanced by AGN ionization → SFR overestimated by factor 2–5
  - FIR luminosity partly from AGN → SFR overestimated
  - UV luminosity affected by AGN dust extinction

- **Solution A: SED decomposition**
  - Use multi-component fitting (CIGALE, AGNfitter) to separate AGN and starburst
  - SFR derived from starburst component only
  - Requires: Good photometric coverage (UV–FIR); redshift accuracy ±0.01
  - Expected SFR uncertainty: ±0.2–0.3 dex (intrinsic model degeneracy)

- **Solution B: AGN subtraction + correction**
  - Measure L_FIR (50–500 μm) from Herschel/WISE
  - Subtract AGN mid-IR contribution (measure from WISE W3 24 μm or SED)
  - Apply dust-corrected FIR→SFR conversion
  - Expected uncertainty: ±0.25 dex

- **Solution C: Composite-free sample selection**
  - Select only pure star-forming galaxies (BPT/WHAN classification)
  - Avoid all AGN (including composite) for SFR studies
  - Trade-off: Smaller sample but unambiguous SFR

**Recommendation for high-accuracy SFR:** Use SED fitting with multi-wavelength data; expect ±0.2 dex systematic uncertainty even with best data.

---

#### Question 4: "Are there AGN that I'm missing with standard methods?"

**Constraints → Recommended Approach:**

- **Type of AGN likely missed:**
  - Heavily obscured (Compton-thick): Optical/soft X-ray selection fails
  - Low-accretion (LLAGN): Weak emission across all bands; below survey sensitivities
  - Heavily reddened Type 1: Broad lines hidden by dust; classified as Type 2
  - AGN in dense starbursts: AGN emission diluted; hard to isolate

- **Recovery strategies:**

  1. **For Compton-thick:** Combine radio excess (from SFR prediction) + mid-IR detection + X-ray hardness
     - Radio-excess AGN (ΔR > 0.5): Finds ~15–20% additional AGN not detected in X-ray
     - Hard X-ray (10–40 keV) sensitive: NuSTAR, Swift/BAT detect CT AGN direct

  2. **For LLAGN:** Use radio, variability, or low-ionization spectroscopic features
     - Radio excess sensitive to accretion (even at L_bol < 10^40 erg/s)
     - Optical variability structure function (year timescales) identifies AGN
     - [Fe VII] forbidden lines (optical) emission indicative of AGN ionization

  3. **For reddened Type 1:** NIR spectroscopy (broad Paschen Hα at 1.28 μm accessible through dust)
     - GNIRS, FLAMINGOS-2 on Gemini reach through A_V ~ 5 magnitudes

  4. **For buried AGN in starbursts:** Full SED decomposition with multi-component models
     - Radio excess and X-ray hardness ratio most diagnostic

**Recommendation:** Single-method AGN surveys have selection biases; always apply multiple independent diagnostics (optical + mid-IR + X-ray + radio) to capture full AGN population.

---

## 2. DETAILED METHOD COMPARISONS

### BPT Diagram: Detailed Advantages & Limitations

#### When to Use BPT:
- **Primary requirement:** Four emission lines must be detected: Hα, [N II] λ6584, Hβ, [O III] λ5007
- **Redshift range:** z < 0.5 (Hα enters K-band at z ~ 0.16; [O III] difficult at z > 0.3)
- **Typical application:** Local universe SDSS galaxies; nearby IFU surveys

#### Practical Steps:
1. Extract emission-line fluxes from 1D spectroscopy or IFU data
2. Compute line ratios: x = log10([N II]/Hα); y = log10([O III]/Hβ)
3. Compare to demarcation lines:
   - **Kewley et al. (2001) maximum starburst line:** y = 0.61 / (x − 0.47) + 1.19
   - **Kauffmann et al. (2003) empirical line:** y = 0.61 / (x − 0.05) + 1.3
4. Classification:
   - Above Kewley line: Seyfert 2 (AGN)
   - Between Kauffmann and Kewley: Composite (AGN + star formation)
   - Below Kauffmann: Star-forming galaxy (HII region)
   - Off-diagram (e.g., x < -0.5): Low-ionization objects (LINER, retired galaxies)

#### Quantitative Performance:
- **Purity:** 85–92% (varies with line SNR cutoff)
- **Completeness:** 75–85% (misses LINER and composite regions)
- **False positive rate:** ~10% (contamination from shock ionization, jets)
- **Redshift evolution:** Effectiveness decreases 5–10% per unit Δz above z = 0.3

#### Critical Caveats:
1. **Metallicity dependence:** Line ratios change with Z; demarcation lines calibrated for Z ≈ 0.02 (solar)
   - Low-Z (Z < 0.1 Z_sun) AGN emit like star-forming galaxies
   - High-Z (Z > 2 Z_sun) galaxies show enhanced [N II] → appear more AGN-like

2. **Dust reddening:** Hα and Hβ affected differently; E(B−V) > 1 mag introduces >0.2 dex error in [O III]/Hβ
   - Correction: Apply Balmer decrement-derived extinction

3. **AGN+starburst composites:** Mixtures produce intermediate ratios; cannot quantify individual contributions from BPT alone
   - Solution: Combine with SED fitting for decomposition

4. **Low-luminosity AGN:** If [O III] flux very faint, noise-dominated ratios; impossible to classify with confidence

---

### WHAN Diagram: Detailed Advantages & Limitations

#### When to Use WHAN:
- **Primary advantage:** Only requires Hα and [N II] (two lines vs. four for BPT)
- **Key metric:** Hα equivalent width (W_Hα); measures line strength relative to continuum
- **Application:** Weak-emission-line galaxies (WELGs) not classifiable by BPT

#### Practical Steps:
1. Measure Hα flux and Hα equivalent width from spectrum
2. Measure [N II] λ6584 flux (to compute [N II]/Hα)
3. Compute: x = log10([N II]/Hα); y = log10(W_Hα [Å])
4. Classification:
   - x < -0.4, y > 3 Å: **Star-forming galaxy** (strong Hα from young stars)
   - x > -0.4, y > 6 Å: **Strong AGN** (Seyfert 1/2)
   - x > -0.4, 3 < y < 6 Å: **Weak AGN** (weak accretion; LINER-like)
   - y < 3 Å: **Retired/passive galaxy** (old stellar population; weak emission from hot evolved stars)

#### Quantitative Performance:
- **Purity:** 80–88% (depends on W_Hα measurement quality)
- **Completeness:** 70–80%
- **Key strength:** Identifies ~20% of galaxies missed by BPT (W_Hα < 3 Å objects)
- **LINER distinction:** Superior to BPT for separating weak AGN from post-AGB ionization

#### Critical Caveats:
1. **Equivalent width sensitivity:** W_Hα affected by underlying stellar absorption features
   - Typical error: ±20% in W_Hα → ±0.1 dex in log(W_Hα) → misclassification possible at boundaries

2. **Reddening:** [N II]/Hα ratio dependent on dust extinction (Hα more obscured than [N II])
   - Apply extinction correction: A_Hα ≈ 2.5 × A_V (Balmer decrement)

3. **Boundary ambiguity:** Sources near decision thresholds (x ~ -0.4, y ~ 3–6) inherently uncertain
   - Recommendation: Use confidence intervals; flag borderline cases

4. **Metallicity:** [N II] strength varies with Z; non-solar Z shifts classifications
   - Caveat: WHAN less sensitive to Z than BPT, but not immune

---

### WISE Mid-Infrared Colors: Detailed Advantages & Limitations

#### When to Use WISE W1−W2:
- **Primary advantage:** All-sky coverage; rapid; simple metric
- **Application:** Large-area AGN surveys; rapid AGN identification without spectroscopy
- **Redshift range:** Effective to z ~ 4 (W1 and W2 probing rest-frame 0.7–1 μm)

#### Practical Steps:
1. Obtain WISE W1 (3.4 μm) and W2 (4.6 μm) magnitudes
2. Compute color: C = W1 − W2 (Vega magnitudes)
3. Apply criterion: C ≥ 0.8 → AGN candidate
4. Optional refinement: Add W2−W3 color to reduce star-forming galaxy contamination
   - Combined criterion: (W1−W2 ≥ 0.8) AND (W2−W3 < 0.5) → AGN purity ~92%

#### Performance by AGN Type:

| **AGN Type** | **W1−W2 Mean** | **Detection Rate** | **False Positive Rate** |
|---|---|---|---|
| Type 1 (unobscured) | 1.0–1.3 | 90–95% | ~3–5% |
| Type 2 (moderately obscured) | 0.6–0.95 | 75–85% | ~8–12% |
| Compton-thick (heavily obscured) | 0.4–0.7 | 50–65% | ~15–20% |
| ULIRG starburst (no AGN) | 0.7–1.1 | — | ~5–10% (contamination) |
| Low-z cool-torus AGN | 0.3–0.6 | 40–55% | — (missed) |

#### Critical Caveats:
1. **Type 2 AGN underrepresented:** Cool torus (T ~ 100–200 K) has weak infrared bump; misses ~30–40% of Type 2
   - Mitigation: Combine WISE with mid-IR spectroscopy (IRS) for detailed diagnostics

2. **Starburst contamination:** Dusty starbursts produce W1−W2 colors overlapping with AGN
   - Distinction: Dust in SFG cooler (T ~ 40–60 K) than AGN torus (T ~ 150–300 K)
   - Diagnostic: Measure [3.6]−[4.5] IRAC color (if available); SFG steeper

3. **Redshift dependence:** Rest-frame wavelengths observed shift with z
   - z = 0: W1−W2 probes 3.4–4.6 μm rest (warm dust)
   - z = 2: W1−W2 probes ~1.1–1.5 μm rest (stellar continuum) → less AGN-sensitive

4. **AGN fraction evolution:** AGN W1−W2 color depends on Eddington ratio; variable over cosmic time
   - Low-accretion AGN (λ < 0.01): W1−W2 ~ 0.5; may be missed
   - High-accretion (λ > 0.1): W1−W2 > 1.0; reliably detected

#### Comparison to IRAC Colors (if Spitzer available):
- **WISE advantage:** All-sky; free; rapid
- **IRAC advantage:** Deeper (faint sources); better SNR for W1−W2-borderline objects
- **Recommendation:** Use WISE for initial survey; follow up IRAC borderline/faint sources

---

### SED Fitting: Detailed Methodology

#### When to Use SED Fitting:
- **Primary requirement:** Multi-band photometry (ideally 5+ bands spanning UV–submm)
- **Application:** Composite system decomposition; physical property extraction
- **Redshift requirement:** Accurate photometric or spectroscopic redshift (Δz < 0.05 critical)

#### Practical Steps:

**Step 1: Assemble observed photometry**
- Collect flux measurements from UV (GALEX, Swift/UVOT), optical (SDSS, Pan-STARRS), NIR (2MASS, WISE), MIR (Spitzer/IRAC, MIPS; WISE), FIR (Herschel), submm (ALMA, SCUBA)
- Check data quality; flag non-detections as upper limits
- Apply Galactic extinction correction (Schlegel et al. 1998)

**Step 2: Set redshift**
- If z known from spectroscopy: Use exactly
- If only photometric z: Adopt best estimate; run SED for z_range (z ± Δz_photo)

**Step 3: Select SED fitting code**
- For AGN: CIGALE or AGNfitter recommended
- For pure star-forming: MAGPHYS or Prospector
- For panchromatic (all types): AGNfitter-rx (includes X-ray)

**Step 4: Configure templates**
- **Stellar population:** Select age/metallicity grid (e.g., Bruzual & Charlot 2003, Maraston 2011)
  - Young galaxies (z > 2): Prefer 0.1–1 Gyr stellar population
  - Massive ellipticals: Prefer 5–13 Gyr stellar population

- **AGN component:**
  - For Type 1: Include accretion disk (hot continuum)
  - For Type 2: Include obscured torus (Nenkova, CAT3D, or SKIRTOR)

- **Dust attenuation:** Select extinction law
  - SMC, MW, LMC: For local galaxies
  - Calzetti: Starburst galaxies
  - Salim: Custom; often used for composite systems

**Step 5: Run fitting**
- Typically: Bayesian χ² minimization or grid search
- Output: PDF of each parameter; mode, mean, and credible intervals
- Time: 1–10 minutes per source (depending on code and parameter space)

**Step 6: Interpret results**
- **Stellar mass M_*:** Typically robust; ±0.15 dex typical uncertainty
- **Star-formation rate SFR:** Less certain for AGN hosts; ±0.2–0.3 dex expected
- **AGN luminosity:** Model-dependent; ±0.25–0.35 dex uncertainty even with good data
- **AGN bolometric luminosity:** From accretion disk or torus template scaling

#### Quantitative Uncertainties:

| **Parameter** | **Photometry Quality** | **With Good Z** | **With Photo-Z** |
|---|---|---|---|
| **Stellar Mass (M_*)** | ±0.15 dex (opt–MIR) | ±0.15 dex | ±0.25 dex |
| **SFR (no AGN)** | ±0.2 dex | ±0.2 dex | ±0.3 dex |
| **AGN Luminosity (composite)** | ±0.25 dex | ±0.3 dex | ±0.4 dex |
| **AGN Bolometric Luminosity** | ±0.25 dex (full SED) | ±0.3 dex | ±0.4 dex |

#### Critical Caveats:
1. **Degeneracies:** Multiple component combinations can fit identical SED
   - Example: Young stellar population (hot dust T ~ 5000 K) mimics AGN continuum
   - Mitigation: Use prior constraints (e.g., stellar mass from kinematics); fix age range

2. **Model choice:** Different torus models (Nenkova, CAT3D, SKIRTOR) yield L_AGN differing by ±0.2–0.3 dex
   - No "true" answer; intrinsic scatter reflects model limitations

3. **Photometric redshift:** Δz = 0.1 → ±0.3 dex error in all derived quantities
   - Spectroscopic z essential for reliable SED fitting

4. **Composite system uncertainty:** Separating AGN and starburst in mixed systems inherently ambiguous
   - Different models assume different geometries; results model-dependent
   - SFR and L_AGN uncertainties typically double (±0.4–0.5 dex) in composites

#### Recommendation for High-Accuracy Results:
- Require: Spectroscopic redshift, 10+ photometric bands, consistent apertures
- Use: Compare multiple torus models; test sensitivity
- Report: Parameter PDFs not just point estimates; quantify model dependence

---

### X-Ray Selection and Spectroscopy: Detailed Methodology

#### When to Use X-Ray Selection:
- **Primary advantage:** Unbiased against obscuration; efficient for all AGN types
- **Application:** Deep surveys (Chandra), wide-area surveys (XMM), hard X-ray (NuSTAR)
- **Typical depths:** Chandra to 10^41 erg/s; XMM to 10^42 erg/s

#### Practical Steps for X-Ray Classification:

**Method 1: Hardness Ratio (photometric approach)**
1. Extract X-ray counts in soft (0.5–2 keV) and hard (2–10 keV) bands
2. Compute hardness ratios: HR1 = (C_hard1 − C_soft1) / (C_hard1 + C_soft1); HR2 = (C_hard2 − C_soft2) / (C_hard2 + C_soft2)
   - HR1: 2–4 keV vs. 1–2 keV (distinguishes unobscured vs. scattered)
   - HR2: 4–16 keV vs. 2–4 keV (distinguishes power-law hardness)
3. Classify based on HR locus:
   - Soft HR1, hard HR2: Unobscured or moderately obscured AGN
   - Soft HR1, soft HR2: Unobscured Type 1; star-forming galaxy contamination
   - Hard HR1, hard HR2: Heavily obscured (Type 2, possibly Compton-thick)

**Method 2: Full X-Ray Spectroscopy (standard approach)**
1. Extract source and background spectra from FITS files
2. Rebin to ensure >20–50 counts per bin (statistics)
3. Fit spectral models (XSPEC, Sherpa):
   - **Unobscured power-law:** absorbed power law with Galactic N_H only
     - Free parameters: Photon index Γ, flux normalization
     - Expected: Γ ~ 1.9–2.0 for typical Type 1 AGN

   - **Absorbed power-law:** intrinsic N_H + Galactic N_H
     - Free parameters: N_H (intrinsic), Γ, flux
     - Expected: Γ slightly harder (Γ ~ 1.7–1.9) if N_H > 10^23

   - **Reflection-dominated:** includes Fe Kα line, Compton reflection
     - Free parameters: Fe Kα EW, reflection amplitude R
     - Diagnostic: R >> 1 indicates heavy obscuration or Compton-thick

4. Evaluate goodness-of-fit (reduced χ²; should be ~1)
5. Extract N_H, Γ, Fe Kα equivalent width
6. Classify AGN type based on N_H:
   - N_H < 10^22: Type 1 (unobscured)
   - 10^22 < N_H < 10^24: Type 2 (moderately obscured)
   - N_H > 1.5×10^24: Compton-thick (signs: high R, high Fe Kα EW, full spectrum depression)

#### Performance by Method:

| **X-Ray Method** | **Unobscured Detection** | **Type 2 Detection** | **Compton-Thick Detection** | **Photon Requirement** |
|---|---|---|---|---|
| **Hardness ratio** | 92% | 75% | 30% | >50 |
| **Spectroscopy (power-law)** | 95% | 85% | 45% | >100 |
| **Spectroscopy (reflection model)** | 90% | 88% | 70% | >200 |
| **Hardness + spectroscopy** | 98% | 92% | 75% | >200 |

#### Critical Caveats:
1. **Low photon statistics:** Sources with <50 counts have large uncertainty in derived N_H
   - Mitigation: Use hardness ratios (photometric approach) for faint sources

2. **Complex spectra:** Real AGN spectra often include:
   - Intrinsic absorption (obscuring gas)
   - Scattering off hot ionized medium
   - Warm absorber (partially ionized gas)
   - Partial covering (clumpy obscuration)
   - Compton reflection (iron Kα, hump at E > 10 keV)
   - Simple power-law assumption insufficient

3. **Redshift effects:** Rest-frame X-ray energy shifts; photometric z error impacts N_H measurement
   - At z > 2: 2–10 keV rest-frame shifted to observer 0.5–2.5 keV; soft band contaminated

4. **Confusion with star-forming galaxies:** High-luminosity X-ray-bright starbursts (~10^42 erg/s at 2–10 keV) mimic AGN flux
   - Distinction: Starburst spectra thermal (bremsstrahlung); AGN spectra power-law
   - With high SNR: Starburst shows soft excess; AGN power-law extends to 10 keV

#### Recommendation for X-Ray Classification:
- **For purity:** Require high signal-to-noise ratio X-ray data + full spectroscopy
- **For completeness:** Combine with mid-IR (WISE) and radio (VLA) to catch heavily obscured systems
- **For Compton-thick:** Never rely on X-ray alone; require hard X-ray (>10 keV; NuSTAR) + mid-IR mid-IR detection + radio confirmation

---

### Machine Learning: Detailed Methodology

#### When to Use Machine Learning:
- **Advantage:** Handles incomplete/noisy multi-wavelength data; rapid inference
- **Application:** Large catalogs (>1000 sources); automated classification
- **Requirement:** Training set of 100–1000 sources with ground-truth labels (spectroscopic classification)

#### Practical Steps:

**Step 1: Prepare training data**
- Collect sources with consensus AGN classification (from BPT, X-ray, or multi-wavelength)
- Extract features: photometric colors, spectral lines (if available), luminosities
- Split into training (70%), validation (15%), test (15%)
- Check for bias: Over-representation of bright nearby objects? Spectroscopic survey biases?

**Step 2: Feature engineering**
- **Physics-based features:**
  - BPT position (distance to demarcation line in [O III]/Hβ vs. [N II]/Hα space)
  - WISE colors (W1−W2, W2−W3)
  - X-ray hardness ratios
  - Radio spectral index
  - Optical variability (structure function)

- **Data-driven features:**
  - Raw photometric colors ([u−g], [g−r], [r−i], [i−z], [3.6]−[4.5])
  - Luminosity ratios (L_X / L_opt, L_radio / L_IR)
  - SED fitting residuals

**Step 3: Handle missing data**
- **Option A: Listwise deletion** (only use sources with all features)
  - Pros: Simple; unbiased toward available data
  - Cons: Discards >30% of survey (if typical ~30% data incompleteness)

- **Option B: Imputation** (estimate missing values)
  - kNN imputation: Replace with mean of k nearest neighbors (k=5 typical)
  - MICE (Multivariate Imputation by Chained Equations): Iterative modeling
  - Performance: Both achieve ~91% accuracy; MICE slightly better for rare classes

**Step 4: Select ML algorithm**
- **Random Forest:**
  - Pros: Fast; handles non-linear relationships; feature importance analysis
  - Cons: May overfit; no probabilistic output by default
  - Performance: 91% accuracy on SDSS/Fermi sample

- **XGBoost:**
  - Pros: Superior accuracy; probabilistic outputs; handles missing data natively
  - Cons: Slower; requires more hyperparameter tuning
  - Performance: 90% accuracy; slightly worse than Random Forest on SDSS

- **Support Vector Machines (SVM):**
  - Pros: Excellent for binary classification; low false positive rate
  - Cons: Slow; high-dimensional feature space problematic
  - Performance: 88% accuracy on SDSS

- **Neural Network (standard):**
  - Pros: Flexible; can learn complex patterns
  - Cons: Requires more training data; black-box interpretation
  - Performance: 87% accuracy; slower convergence than tree-based

- **Convolutional Neural Network (imaging):**
  - Pros: Learns morphological features; no hand-crafted features needed
  - Cons: Requires large image training set (>100k); computationally intensive
  - Performance: 78% accuracy on SDSS galaxy images; good for morphology-AGN link

**Step 5: Hyperparameter optimization**
- Use cross-validation (5-fold typical) to tune:
  - Random Forest: Number of trees (50–200), max depth (5–15)
  - XGBoost: Learning rate (0.01–0.1), max depth (3–7), number of rounds (100–1000)
  - SVM: Kernel (RBF typical), C regularization (0.1–100)

- Grid search or Bayesian optimization

**Step 6: Evaluate on test set**
- **Metrics:**
  - Overall accuracy = (TP + TN) / (TP + TN + FP + FN)
  - Precision (purity) = TP / (TP + FP) [fraction of predicted AGN that are true AGN]
  - Recall (completeness) = TP / (TP + FN) [fraction of true AGN identified]
  - F1-score = 2 × (Precision × Recall) / (Precision + Recall) [harmonic mean]

- Confusion matrix: TP, TN, FP, FN for each class

**Step 7: Interpret results**
- **Feature importance:** Which features most discriminative?
  - Random Forest: Measure impurity decrease per feature
  - Typical ranking: WISE W1−W2 > X-ray luminosity > optical color > radio spectral index

- **Error analysis:** Which sources misclassified?
  - Composite systems? High-z sources? Faint sources?
  - Identify systematic biases in training set

**Step 8: Apply to full sample**
- Use trained model on new data
- Flag low-confidence predictions (probability near 0.5)
- Do NOT extrapolate beyond training data distribution (e.g., z_train < 1 → don't apply to z_test > 2)

#### Performance Typical of Well-Optimized ML Models:

| **Data Completeness** | **Overall Accuracy** | **AGN Purity** | **AGN Completeness** | **Recommendation** |
|---|---|---|---|---|
| **100% complete** | 91–92% | 91% | 88% | Excellent; use for science |
| **80% complete (20% imputed)** | 90–91% | 90% | 87% | Good; note systematic ±0.5% |
| **60% complete (40% imputed)** | 88–89% | 88% | 84% | Acceptable; quantify uncertainty |
| **40% complete (60% imputed)** | 85–86% | 84% | 80% | Marginal; use with caution |

#### Critical Caveats:
1. **Training set bias:** If training sample biased (e.g., only bright objects, low-z), model will replicate bias
   - Mitigation: Characterize training set; weight samples by survey selection function

2. **Overfitting:** Model memorizes training noise instead of learning patterns
   - Diagnosis: Training accuracy >> test accuracy
   - Prevention: Use regularization (dropout, L1/L2); cross-validation

3. **Redshift evolution:** Model trained at z < 0.5 will fail at z > 1 if AGN properties change
   - Example: AGN fraction evolves; low-z training biased
   - Mitigation: Include z as feature; evaluate performance per redshift bin

4. **Class imbalance:** If training set unbalanced (e.g., 70% SFG, 30% AGN), model biased toward majority class
   - Mitigation: Weight classes inversely; use stratified sampling; adjust decision threshold

5. **Interpretability:** Machine learning predictions hard to validate against physics
   - Mitigation: Use SHAP (SHapley Additive exPlanations) values to explain individual predictions

---

## 3. MULTI-METHOD CONSENSUS APPROACH (Recommended Best Practice)

### Integrated Framework for Robust AGN Classification

When resources permit, apply multiple independent methods and synthesize results:

```
Step 1: Apply ALL available diagnostics
├─ Optical (if 0.3 < z < 1):      BPT/WHAN diagram
├─ Mid-IR (all z):                WISE W1−W2 ± Spitzer colors
├─ X-ray (if available):          X-ray hardness ± spectroscopy
├─ Radio (if available):          Radio excess; spectral index
├─ SED (all z with photometry):   Multi-component decomposition
└─ Variability (if light curves): Optical/IR structure function

Step 2: Assess agreement
├─ Count how many methods identify source as AGN
├─ Sources with ≥3/5 methods → Robust AGN (high confidence)
├─ Sources with 2/5 methods  → Probable AGN (moderate confidence)
├─ Sources with <2/5 methods → Ambiguous (flag as uncertain)

Step 3: Quantify confidence
├─ For robust: Confidence ~ 95%+ (use in all science)
├─ For probable: Confidence ~ 80–90% (use with caveats)
├─ For ambiguous: Confidence < 80% (exclude from main analysis)

Step 4: Report results
├─ List which methods applied
├─ Report individual method results
├─ State confidence level
└─ Quantify systematic uncertainties (±X dex in L_AGN, etc.)
```

### Example Application:

**Galaxy XYZ, z = 0.8:**

| **Method** | **Classification** | **Confidence** | **Details** |
|---|---|---|---|
| **BPT** (if emission lines in HST/grism) | Ambiguous/composite | ~70% | Near demarcation line |
| **WISE W1−W2** | AGN | 85% | W1−W2 = 0.82; slightly reddened |
| **Chandra X-ray** (if 4 Ms CDF data) | AGN | 92% | N_H = 10^23 cm^-2; moderate obscuration |
| **VLA 3 GHz** (if COSMOS field) | AGN | 88% | Radio excess ΔR = 0.55 |
| **SED fitting** (UV–submm) | AGN+starburst | 80% | L_AGN / L_total ~ 40% ± 20% |
| **Variability** (if multi-epoch optical) | AGN | 75% | X-ray variable; optical structure function shallow |

**Consensus:** AGN with high confidence (4/6 methods agree); moderate obscuration (N_H ~ 10^23); AGN fraction ~40% (rest starburst)

**Recommendation:** Use as AGN in demographics; quantify AGN fraction uncertainty as ±0.2 dex

---

## 4. SUMMARY TABLE: METHOD SELECTION DECISION TREE

| **Your Data** | **Redshift** | **Best Primary Method** | **Best Secondary Method** | **Expected Accuracy** | **Key Limitation** |
|---|---|---|---|---|---|
| **Optical spectrum (4 lines)** | z < 0.3 | BPT | WHAN + WISE | 90% purity, 85% complete | Type 2 bias |
| **Optical spectrum (2 lines)** | z < 0.5 | WHAN | WISE | 85% purity, 75% complete | Composites ambiguous |
| **No spectrum, photometry only** | z < 1 | WISE W1−W2 | Random Forest (ML) | 85% purity, 72% complete | Cool-torus missed |
| **Full photometry UV–submm** | 0.5 < z < 2 | SED fitting (CIGALE) | X-ray hardness | 86% purity, 80% complete | Model-dependent |
| **X-ray spectrum + photometry** | All z | X-ray spectroscopy | SED fitting | 92% purity, 80% complete | Type 2 underweighted |
| **Radio + mid-IR photometry** | z < 3 | Radio excess + WISE | ML classifier | 80% purity, 85% complete | AGN confirmation weak |
| **JWST spectroscopy** | z > 3 | UV emission lines ([Ne V], [He II]) | X-ray (if available) | 75–80% | Photoionization models incomplete |

---

**End of Methodological Guide**

*Practical guidance for AGN classification in observational surveys*
