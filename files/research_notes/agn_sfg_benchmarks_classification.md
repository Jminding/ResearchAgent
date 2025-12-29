# Benchmark Datasets and Classification Methods for AGN/SFG Studies

**Compiled:** December 22, 2025
**Scope:** Available labeled datasets, classification algorithms, and performance benchmarks

---

## 1. LABELED BENCHMARK DATASETS

### 1.1 SDSS Spectroscopic Sample

**Overview:**
The Sloan Digital Sky Survey represents the largest homogeneous spectroscopic database available for AGN/SFG classification studies.

**Sample Specifications:**
- **Total Spectra:** 930,000+ (DR7); 1,000,000+ (DR17+)
- **Sky Coverage:** 9,380 deg²
- **Spectral Range:** 3800-9200 Å (observed)
- **Spectral Resolution:** R ~ 1000-2000 (dispersion ~1.49 Å/pixel)
- **Redshift Range:** 0.01 < z < 5
- **Median Redshift:** z ~ 0.15

**Photometric Data:**
- Five-band imaging (u', g', r', i', z')
- Typical 5σ limits: u=22.0, g=22.2, r=22.2, i=21.3, z=20.5 mag
- Astrometry: ±0.1 arcsec accuracy

**Spectroscopic Classification:**

| Class | Criteria | Number | Fraction |
|-------|----------|--------|----------|
| Star | Stellar continuum + absorption lines | ~200,000 | ~22% |
| Galaxy | 4000 Å break; emission or absorption | ~600,000 | ~65% |
| QSO | High-z galaxy-like; emission lines + continuum | ~130,000 | ~13% |

**AGN Identification Methods in SDSS:**

1. **Broad-Line AGN:**
   ```
   Detect Hβ (λ4861) or Hα (λ6563) with FWHM > 1500 km/s (≥5σ)
   Flag: BROADLINE in spectral class
   Fraction: ~20% of emission-line galaxies
   ```

2. **BPT Diagnostic (Emission-Line AGN):**
   ```
   Log([O III] λ5007 / Hβ λ4861) vs Log([N II] λ6584 / Hα λ6563)

   Regions:
   - Star-forming (SF): below Kauffmann+2003 line
   - Composite: between SF and Kewley+2001 line
   - AGN: above Kewley+2001 line
   - LINER: right side (high [N II]/Hα)

   Requirement: All lines ≥3σ detection; z > 0.02 (rest-frame 5007 > 5000 Å in observed)
   ```

3. **X-ray/Infrared AGN (Matched Sample):**
   ```
   Cross-matched to Chandra/XMM or WISE
   Enables selection of obscured AGN unavailable via optical diagnostics
   ```

**Strengths for Machine Learning:**
- Largest homogeneous labeled dataset
- Well-characterized selection effects
- Multiple classification schemes (class + BPT + broad-line flag)
- Multi-wavelength cross-matches available
- Public, well-documented, standardized

**Limitations:**
- Optical extinction biases against dust-obscured AGN
- Sparse redshift coverage for z > 2 AGN (QSO subsample biased)
- No native NIR/MIR photometry
- BPT-dominated: less reliable for low S/N spectra
- Limited emission-line redshift range (Hβ/Hα barely visible at z > 0.4)

**Access:**
- SDSS Data Release Server: https://www.sdss4.org/dr17/spectro/catalogs/
- Query by object ID or coordinates
- Download FITS spectra + computed parameters

**Usage in Literature:**
- Default training set for AGN photo-z models (>100 papers)
- Validation set for machine learning classifiers
- AGN/SFG population studies at low-z

---

### 1.2 GAMA Spectroscopic Sample

**Overview:**
Higher spectroscopic quality at lower redshift; ideal for detailed AGN/SFG studies in z ~ 0-0.5 Universe.

**Sample Specifications:**
- **Total Redshifts:** 238,000
- **Magnitude Limit:** r < 19.8 mag (r-band selected)
- **Redshift Range:** 0 < z < 0.5 (median z ~ 0.2)
- **Sky Coverage:** ~286 deg² (three distinct fields: G02, G09, G12, G15)
- **Spectral Range:** 3750-8850 Å (observed)
- **Spectral Resolution:** R ~ 1300

**Spectroscopic Measurements:**
1. **Redshift:** σ_z ~ 70 km/s (repeatability)
2. **Emission-Line Fluxes:** Hα, Hβ, [O III], [N II], [S II]
3. **Star Formation Rate (Hα):**
   ```
   SFR [M_⊙/yr] = (SFR_Hα_uncorr × dust_factor)
   Dust correction via Balmer decrement (Hα/Hβ ratio)
   Typical uncertainties: ±0.3 dex
   ```
4. **Metallicity:** Via [O III]/[O II] and [N II]/Hα
5. **AGN Classification:**
   - BPT diagnostic (all four lines available)
   - X-ray matching (cross-referenced to Chandra, XMM)
   - Radio matching (VLA cross-identification)

**AGN Sample in GAMA:**
- Total AGN identified: ~15,000 (6.3% of sample)
- Type 1 (Unobscured): ~30% of AGN (broad Hα)
- Type 1.5-1.9: ~10% (intermediate)
- Type 2 (Obscured): ~60% (narrow lines only)
- LINER population: separately classified

**Strengths for Analysis:**
- Highest spectroscopic quality available (multiple exposures averaged)
- Detailed AGN properties + host SFR simultaneously measured
- Excellent redshift accuracy enables clustering studies
- Multi-wavelength photometry provided (UV, optical, NIR)
- Environment information (group membership)
- Dust corrections applied uniformly

**Limitations:**
- Limited to z < 0.5 (local universe only)
- Biased toward r-band selected galaxies
- Three separate fields → cosmic variance important
- Limited high-L AGN (too rare at low-z)

**Access:**
- GAMA Data Release Server: https://www.gama-survey.org/
- Direct redshift + spectral parameter downloads
- Cross-matched to external surveys (X-ray, radio, IR)

**Usage in Literature:**
- Local AGN demographics + feedback studies
- Validation for high-z AGN/SFG separation methods
- SFR calibration at z ~ 0

---

### 1.3 Chandra-COSMOS Legacy Multi-Wavelength Sample

**Overview:**
Deep X-ray survey with unprecedented multi-wavelength photometric coverage; ideal for training multi-wavelength AGN classifiers.

**Sample Specifications:**
- **X-ray Sources:** 4,016 (Chandra detection, 0.5-7 keV)
- **Spectroscopic Redshifts:** 652 (16% of sample)
- **Photometric Redshifts:** 3,364 (84% of sample)
- **Redshift Range:** 0 < z < 6 (median z ~ 1.0)
- **Sky Area:** 2.0 deg² (COSMOS field)

**Photometric Coverage (8 Bands):**
```
UV:  GALEX (NUV 2300 Å)
Optical: CFHT u*g'r'i'z' + Subaru z-deep + Suprime
NIR: CFHT Ks + Subaru NIR
MIR: Spitzer IRAC (3.6, 4.5, 5.8, 8 μm) + MIPS 24 μm
FIR: Herschel PACS (100, 160 μm) + SPIRE (250 μm)
Radio: VLA 1.4 GHz
```

**Classification in Chandra-COSMOS:**

| Type | N | Criteria | AGN Fraction |
|------|---|----------|-------------|
| X-ray AGN | 3,028 | X-ray detection + spectral fit | ~75% |
| Starburst | ~800 | Star-forming; no X-ray excess | — |
| Normal Galaxy | ~188 | X-ray detected but consistent with SFG | — |

**Physical Properties Derived:**
1. **Redshift:** Spectroscopic (high-z accuracy); photometric with σ_z/(1+z) ~ 5-10%
2. **Stellar Mass:** SED fitting (optical+NIR) → M_* (uncertainty ±0.3 dex)
3. **Star Formation Rate:**
   ```
   Method 1: L_IR → SFR (Herschel 100+ μm)
   Method 2: UV + IR SED decomposition
   Typical values: 0.1 < SFR < 1000 M_⊙/yr
   ```
4. **AGN Bolometric Luminosity:**
   ```
   From X-ray (2-10 keV) with bolometric correction
   or multi-component SED fitting (accretion disk + torus)
   Typical values: 10^44 < L_bol < 10^47 erg/s
   ```
5. **Dust Obscuration (A_V):** From SED fitting; separates AGN from dust-obscured SFGs

**Strengths for Multi-Wavelength ML:**
- Ideal multi-wavelength feature vector construction
- Rich SED enables decomposition of AGN + host
- X-ray confirmation provides clean AGN label
- High spectroscopic redshift purity (99%)
- Published derived properties catalogs
- Covers full AGN luminosity range (L_bol ~ 10^44-10^47 erg/s)

**Limitations:**
- Small area (2 deg²) → limited dynamic range in environment/galaxy properties
- Biased toward X-ray bright (unobscured + luminous) AGN
- Compton-thick systems underrepresented
- Radio-loud AGN sparse (radio-quiet biased)
- Requires multi-step cross-matching

**Data Access:**
1. **X-ray Catalog:** NASA HEASARC (CHANDRACOSMOS)
2. **Multi-wavelength Photometry:** COSMOS Portal (cosmos.astro.caltech.edu)
3. **Derived Properties:** Published in Civano+ 2016 (ApJ 819, 62)

**Usage in Literature:**
- Multi-wavelength photo-z training (>50 papers)
- AGN/SFG template SED construction
- Validation for AGN classification in high-z samples
- AGN-feedback studies (L_bol vs SFR analysis)

---

### 1.4 WISE AllWISE AGN Sample

**Overview:**
All-sky infrared AGN catalog enabling statistical studies of AGN host properties.

**Sample Specifications:**
- **Total WISE AGN (R90 purity):** 4,543,530
- **Alternative (C75 completeness):** 20,907,127
- **Sky Coverage:** 30,093 deg² (excludes Galactic plane)
- **Redshift Range:** 0 < z < 5+ (median z ~ 0.5)

**Selection Criteria:**

| Catalog | Criterion | Purity | Completeness | Contaminants |
|---------|-----------|--------|-------------|-------------|
| R90 (Reliable) | W1-W2 ≥ 0.8 + refinement | ~90% | ~65% | ~10% stars/SFGs |
| C75 (Complete) | W1-W2 ≥ 0.7 + refinement | ~75% | ~90% | ~25% stars/SFGs |
| Mixed | W1-W2: 0.5-0.8 (ambiguous) | ~50% | Variable | Many composites |

**Color-Color Space:**
```
W1-W2 vs W2-W3 two-dimensional diagrams separate:
- Unobscured AGN: W1-W2 > 0.8, W2-W3 < 1.0
- Obscured AGN: W1-W2 > 0.8, W2-W3 > 1.0
- Starbursts: W1-W2 < 0.5, W2-W3 > 0.5
- Composites: Intermediate colors
```

**Available Properties (Assef+ 2018 Catalog):**

For 695,273 WISE AGN with multi-wavelength SED fitting:
1. **Photometric Redshift:** σ_z/(1+z) ~ 0.05-0.10
2. **AGN Bolometric Luminosity:** From SED decomposition (10^44-10^48 erg/s)
3. **Host Stellar Mass:** 10^8-10^12 M_⊙
4. **Star Formation Rate:** Derived from infrared SED (0.1-1000 M_⊙/yr)
5. **AGN Contribution:** Fraction of total luminosity from accretion disk

**Strengths:**
- All-sky coverage; enables unbiased AGN statistics
- High-z sensitive (infrared selection probes dust torus regardless of optical extinction)
- Large sample supports rare AGN studies
- Pre-computed properties save processing time
- Well-validated in literature (>100 papers)

**Limitations:**
- WISE selection biased toward mid-IR bright sources
- Limited spectroscopic follow-up (most lack spec-z)
- Photo-z accuracy modest (σ_z ~ 5-10%)
- Incompleteness at high-z (W1-W2 colors shift; torus opacity effects)
- Limited for radio-loud AGN identification

**Access:**
- AllWISE Catalog: NASA IRSA (irsa.ipac.caltech.edu)
- WISE AGN Catalogs: NASA HEASARC (allwiseagn.html)
- Assef+ 2018 properties: IRSA or ADS

**Usage in Literature:**
- AGN host galaxy properties at z ~ 0-3
- High-z obscured AGN selection (z > 3)
- Statistical AGN luminosity function studies
- Training for infrared AGN classifiers

---

### 1.5 Radio Galaxy Zoo / RGC-Bent Dataset

**Overview:**
Crowdsourced radio galaxy morphology classifications enabling deep learning on radio images.

**Sample Specifications:**
- **Total Radio Images:** ~20,000 (Radio Galaxy Zoo main project)
- **RGC-Bent Subset:** ~2,500 bent vs straight radio galaxies
- **Source Survey:** FIRST (1.4 GHz) and ATLAS (1.4 GHz)
- **Image Resolution:** 5 arcsec (FIRST)
- **Image Size:** Varies; typically ~200×200 pixels

**Classifications Available:**

| Morphology | Description | Typical L_radio | Sample Size |
|-----------|------------|----------------|------------|
| Bent (NAT/WAT) | Narrow/Wide-Angle Tail; jets deflected by environment | 10^41-10^44 erg/s | ~1,200 |
| Straight (FRI/FRII) | Classical jets; core-lobe or edge-brightened | 10^40-10^44 erg/s | ~1,300 |

**Data Structure:**
- **Input:** Radio image (greyscale; 1.4 GHz continuum)
- **Label:** Boolean or multi-class (bent, straight, uncertain)
- **Crowdsourcing:** ~40 independent classifications per source (consensus voting)
- **Accuracy:** >95% agreement between independent annotators for clear cases

**Machine Learning Performance (2024):**

| Architecture | Accuracy (Bent Classification) | F1-Score | Training Data |
|-------------|------------------------------|----------|--------------|
| ResNet-50 | 0.92 | 0.91 | 2,000 labeled images |
| EfficientNet-B5 | 0.94 | 0.93 | 2,000 labeled images |
| ConvNeXT-Base | 0.96 | 0.95 | 2,000 labeled images |
| Ensemble (3 models) | 0.97 | 0.96 | All data combined |

**Strengths:**
- High-dimensional image data (CNN-optimal)
- Clear morphological phenotypes (easy interpretation)
- Crowdsourced consensus improves label quality
- Well-documented in literature
- Enables transfer learning from other image domains

**Limitations:**
- Single wavelength (radio only)
- Limited sample size (~2-20K) vs. other benchmarks
- Requires image pre-processing (cutout, normalization)
- Does not directly inform AGN/SFG separation

**Access:**
- Radio Galaxy Zoo: https://www.radiogalaxyzoo.org/
- RGC-Bent Dataset: Preprint + supplementary data (arXiv:2311.xxxxx)
- Download images + classifications from project repository

**Usage in Literature:**
- Convolutional neural network training
- Transfer learning base models
- Radio morphology classification automation
- Jet-AGN classification studies

---

### 1.6 LoTSS Deep Fields Multi-Wavelength Sample

**Overview:**
Radio survey with deep multi-wavelength follow-up enabling radio + optical AGN/SFG classification.

**Sample Specifications:**
- **Radio Survey:** LoTSS (LOFAR Two-metre Sky Survey) at 150 MHz
- **Angular Resolution:** 6 arcsec (exceptional)
- **Deep Fields:** COSMOS, ELAIS-N1, Bootes
- **Total Radio Sources (Deep):** ~50,000
- **Radio + Optical Photo-z Available:** Yes

**Multi-Wavelength Data Provided:**
1. **Radio:** 150 MHz continuum + spectral indices (where second freq available)
2. **Optical:** ugriz photometry (Pan-STARRS/SDSS)
3. **Photometric Redshift:** Photo-z + uncertainty from optical
4. **Radio Morphology:** Source classification (compact, extended, bent, etc.)

**AGN/SFG Classification Results:**

**Method 1: Radio Excess (Condon Relation)**
```
Predicts radio luminosity from L_IR (infrared-radio correlation)
Excess radio above prediction indicates AGN jet contribution
Enables AGN identification in radio-quiet systems
Performance: ~80% accuracy in separating radio AGN from SFGs
```

**Method 2: Multi-Band Machine Learning**
```
Input features:
  - Radio: flux density, spectral index, morphological class
  - Optical: colors (g-r, r-i, etc.), photometric redshift
  - Combined: radio-optical flux ratio, K-correction effects

Classifier: Random Forest or Gradient Boosting
Performance: >85% accuracy (AGN vs SFG in radio sources)
```

**Strengths for AGN/SFG Classification:**
- Combines single-wavelength radio with optical multi-band data
- Radio morphology provides physical motivation
- Large sample (~50K) enables statistical studies
- Published ML pipeline and training code available
- LOFAR ongoing (extended to all-sky planned)

**Limitations:**
- Limited to radio-detected sources (radio-quiet AGN missed)
- Photo-z accuracy moderate (~3-5%)
- Deep fields only (COSMOS, ELAIS, Bootes); limited statistics
- 150 MHz selection biased toward steep-spectrum sources

**Access:**
- LoTSS Data Release: https://lofar-surveys.org/
- Multi-wavelength catalogs: Published in papers + IRSA
- ML code: GitHub repositories (LoTSS team publications)

**Usage in Literature:**
- Training set for radio+optical ML classifiers
- Validation for high-z AGN identification methods
- Radio morphology studies (jets, environments)

---

## 2. CLASSIFICATION METHODS AND ALGORITHMS

### 2.1 Classical Spectroscopic Methods

#### BPT (Baldwin-Phillips-Terlevich) Diagnostic

**Mathematical Definition:**
```
x = log₁₀([N II] λ6584 / Hα λ6563)
y = log₁₀([O III] λ5007 / Hβ λ4861)

Kauffmann+ 2003 SF/Composite Boundary:
y = 0.61 / (x - 0.05) + 1.3

Kewley+ 2001 Composite/AGN Boundary:
y = 0.61 / (x - 0.47) + 1.19

LINER Separator:
y = 0.73 * x + 1.12
```

**Classification:**
- **Star-Forming (SF):** Below Kauffmann line
- **Composite:** Between Kauffmann and Kewley lines
- **AGN:** Above Kewley line
- **LINER:** Right of LINER line ([N II]/Hα > 0.6)

**Requirements:**
- Emission lines ≥3σ detection
- Rest-frame wavelengths (requires redshift knowledge)
- Best for z < 0.4 (Hα, Hβ become faint at higher z)

**Strengths:**
- Well-motivated physically (ionization mechanism)
- Quantitative classification
- Extensive literature validation
- Accounts for composite systems

**Limitations:**
- Fails at high-z (emission lines shift out of spectral window)
- Dust extinction affects lines differently (Balmer decrement correction needed)
- Cannot detect heavily obscured AGN (faint [O III])
- Composites intrinsically ambiguous (AGN + SFG mixed)

**Python Implementation:**
```python
import numpy as np

def bpt_classify(NII_Ha, OIII_Hb):
    """
    Classify galaxy based on BPT diagnostic.

    Input: log([N II]/Hα) and log([O III]/Hβ) ratios
    Output: Classification (SF, Composite, AGN, LINER)
    """
    # Kauffmann+ 2003 boundary
    kauffmann_y = 0.61 / (NII_Ha - 0.05) + 1.3

    # Kewley+ 2001 boundary
    kewley_y = 0.61 / (NII_Ha - 0.47) + 1.19

    # LINER boundary
    liner_x = 0.6

    if NII_Ha < liner_x:
        if OIII_Hb < kauffmann_y:
            return "Star-Forming"
        elif OIII_Hb < kewley_y:
            return "Composite"
        else:
            return "Seyfert"
    else:
        return "LINER"
```

---

#### Broad-Line Detection (Optical)

**Method:**
```
Measure FWHM of Hα λ6563 or Hβ λ4861
If FWHM > 1500 km/s (≥5σ): Broad-line AGN (Type 1)
If FWHM < 1000 km/s: Narrow-line AGN (Type 2)
FWHM ~ 1000-1500 km/s: Intermediate (Type 1.5-1.9)
```

**Physics:**
- Broad lines indicate close (~1 pc) to supermassive black hole
- Narrow lines from ~100 pc regions (obscured by torus)
- Line width related to black hole mass via virial theorem

**SDSS Broad-Line Identification:**
```
SDSS pipeline flags broad lines in spectra
Output: BROADLINE flag appended to spectral class
Automatic detection at >5σ significance
```

**Strengths:**
- Direct probe of accretion disk
- Unambiguous Type 1 identification
- Enables black hole mass estimation (virial mass)

**Limitations:**
- Only detects unobscured AGN
- Misses Type 2 systems entirely
- Requires high S/N spectroscopy

---

### 2.2 Multi-Wavelength Color-Based Methods

#### WISE Mid-Infrared Selection

**Standard Wedge (Stern+ 2012; Mateos+ 2012):**

```
Selection Criterion 1 (High Purity):
W1 - W2 >= 0.8 mag

Selection Criterion 2 (Higher Completeness):
W1 - W2 >= 0.7 mag

2D Diagnostic (Mateos+ 2012):
Boundaries in (W1-W2, W2-W3) color space:

Equation 1: (W1-W2) = 0.315*(W2-W3) - 0.222
Equation 2: (W1-W2) = 0.315*(W2-W3) + 0.796
Equation 3: (W1-W2) = -3.172*(W2-W3) + 7.624

AGN lies inside region bounded by these lines
SFG lies outside
```

**Physics:**
- W1 (3.4 μm) - W2 (4.6 μm): Detects hot dust torus emission
- W2 (4.6 μm) - W3 (12 μm): Distinguishes AGN from starburst SED
- Insensitive to optical extinction (infrared)

**Performance:**
```
W1-W2 >= 0.8 AGN Sample (R90 catalog):
- Purity: ~90% (true AGN)
- Completeness: ~65% (fraction of all AGN detected)
- Contamination: ~5-10% (stars, starburst SFGs)

W1-W2 >= 0.7 AGN Sample (C75 catalog):
- Purity: ~75%
- Completeness: ~90%
- Contamination: ~25%
```

**Advantages:**
- All-sky applicable
- Dust-insensitive
- Detects highly obscured AGN
- Fast computation
- No redshift needed

**Disadvantages:**
- Degeneracy at intermediate colors (0.5 < W1-W2 < 0.8)
- High-z incompleteness (torus emission shifts beyond W4)
- Cannot separate unobscured AGN from young starbursts
- Requires S/N ≥ 5 in W1, W2, W3

**Python Implementation:**
```python
def wise_agn_select(W1, W2, W3, method='Stern'):
    """
    WISE mid-IR AGN selection.

    Input: Magnitudes W1, W2, W3
    Output: AGN classification
    """
    W1_W2 = W1 - W2
    W2_W3 = W2 - W3

    if method == 'Stern':
        # High purity
        if W1_W2 >= 0.8:
            return "AGN (high purity)"
        elif W1_W2 >= 0.7:
            return "AGN (moderate purity)"
        else:
            return "Non-AGN"

    elif method == 'Mateos':
        # 2D diagnostic
        in_wedge = (
            (W1_W2 > 0.315*W2_W3 - 0.222) and
            (W1_W2 > 0.315*W2_W3 + 0.796) and
            (W1_W2 < -3.172*W2_W3 + 7.624)
        )
        return "AGN" if in_wedge else "Non-AGN"
```

---

#### X-Ray Hardness Ratio

**Definition:**
```
H = (Hard Counts - Soft Counts) / (Hard Counts + Soft Counts)

Soft Band: 0.5-2 keV
Hard Band: 2-7 keV
```

**Interpretation:**
```
H < 0.0: Unobscured AGN (power-law spectrum)
0.0 < H < 0.4: Mixed / low absorption
0.4 < H < 0.8: Obscured AGN (N_H ~ 10^22-10^24 cm^-2)
H > 0.8: Compton-thick (N_H > 10^24 cm^-2) or very soft SFG
```

**Advantages:**
- Quick classification (no detailed fitting)
- Indicates obscuration column density
- Identifies Compton-thick AGN

**Limitations:**
- Degenerate (soft SFGs also have high hardness)
- Requires significant X-ray counts (Chandra/XMM needed)
- AGN spectral modeling improves over simple hardness

---

### 2.3 Machine Learning Methods

#### Random Forest Classification

**Algorithm Overview:**
Ensemble of decision trees trained on multi-wavelength photometric features.

**Training Procedure:**
```
1. Feature vector: [mag_g, mag_r, mag_i, mag_z, W1, W2, W3, W4, photo_z]
2. Labels: "AGN" or "SFG" (from spectroscopy or X-ray)
3. Train 100-1000 trees; each uses random feature subset
4. Each tree predicts class; majority vote = final prediction
5. Out-of-bag (OOB) error estimates generalization
```

**Python Implementation (Scikit-learn):**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Prepare data
X = catalog[['mag_g', 'mag_r', 'mag_z', 'W1', 'W2', 'photo_z']]
y = catalog['agn_label']  # 0=SFG, 1=AGN

# Train classifier
rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
rf.fit(X, y)

# Evaluate
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Feature importance
importances = rf.feature_importances_
for name, imp in zip(X.columns, importances):
    print(f"{name}: {imp:.3f}")
```

**Performance (Typical):**
```
Accuracy: 85-92%
Precision: 88-95%
Recall: 80-90%
F1-Score: 0.85-0.92
```

**Strengths:**
- Non-parametric (no functional form assumptions)
- Handles non-linear relationships automatically
- Feature importance quantifiable
- Robust to outliers
- Fast prediction (~1M objects in seconds)

**Limitations:**
- Requires labeled training data (limited for high-z)
- Feature engineering critical (photometry preprocessing)
- Less interpretable than decision rules
- Can overfit to training set peculiarities

---

#### Support Vector Machines (SVM)

**Algorithm Overview:**
Finds optimal hyperplane separating classes in high-dimensional feature space.

**Training:**
```
Kernel: RBF (Radial Basis Function) typically optimal for photometry
Cost parameter C: Controls overfitting; tuned via cross-validation
Features: Standardized (zero mean, unit variance)
```

**Python Implementation:**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Scale features + train SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

pipeline.fit(X_train, y_train)

# Probability estimates
proba = pipeline.decision_function(X_test)  # Decision boundary distance
```

**Performance:**
```
Accuracy: 85-90%
Precision: 87-93%
Recall: 80-88%
Computational cost: Higher than RF (especially for large samples)
```

**Strengths:**
- Excellent for high-dimensional data
- Kernel methods enable non-linear classification
- Theoretical foundations (VC dimension, margin theory)

**Limitations:**
- Slower than RF for large datasets
- Less interpretable (weight vectors in kernel space)
- Hyperparameter tuning critical
- Memory intensive for 1M+ object catalogs

---

#### Convolutional Neural Networks (CNN)

**Architecture (Typical for Galaxy Images):**
```
Input: 224×224 pixel image (optical or radio)
    ↓
Conv1: 64 filters, 3×3 kernel, ReLU
    ↓
MaxPool1: 2×2, stride 2
    ↓
Conv2: 128 filters, 3×3 kernel, ReLU
    ↓
MaxPool2: 2×2, stride 2
    ↓
Conv3: 256 filters, 3×3 kernel, ReLU
    ↓
Flatten: 1D vector
    ↓
Dense1: 512 neurons, ReLU, Dropout(0.5)
    ↓
Dense2 (Output): 2 neurons, Softmax [P(AGN), P(SFG)]
```

**Training:**
```
Loss function: Cross-entropy
Optimizer: Adam (learning rate 1e-3)
Batch size: 32-128
Epochs: 50-200
Validation split: 20%
```

**Performance (Radio Galaxy Classification):**
```
Accuracy: 92-97% (bent vs straight classification)
Precision: 93-96%
Recall: 91-97%
```

**Strengths:**
- Optimal for image data (spatial feature extraction)
- Automatic feature learning (no manual engineering)
- Transfer learning available (pre-trained ImageNet weights)
- State-of-the-art performance

**Limitations:**
- Requires large labeled training sets (~1000+ images)
- Computationally expensive (requires GPU)
- Less interpretable ("black box")
- Prone to overfitting on small datasets

**Python Implementation (PyTorch):**
```python
import torch
import torch.nn as nn
from torchvision import models

# Use ResNet-50 pretrained on ImageNet
model = models.resnet50(pretrained=True)

# Modify final layer for AGN/SFG classification
model.fc = nn.Linear(2048, 2)

# Define loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(200):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

#### Ensemble Methods and Hybrid Approaches

**Multi-Method Voting:**
```
Combine outputs from multiple classifiers:
1. Random Forest (photometry)
2. SVM (photometry + photo-z)
3. WISE color cuts (infrared)
4. X-ray hardness ratio (if available)

Final class = majority vote
Confidence = fraction of classifiers agreeing
```

**Performance (Voting Ensemble):**
```
Accuracy: 90-95%
Precision: 92-97%
Recall: 85-92%
Robustness: Higher than individual methods
```

**Advantages:**
- Combines strengths of multiple methods
- More robust to systematic errors
- Quantifiable confidence scores

**Limitations:**
- Computationally expensive (multiple models)
- Correlation between methods reduces benefit
- Requires calibration of voting weights

---

### 2.4 Photometric Redshift Methods for AGN

#### Machine Learning Photo-z (PICZL)

**Algorithm:**
```
Convolutional Neural Network on imaging data

Input: ugriz optical + WISE W1/W2 photometry + uncertainty maps
       from DESI Legacy Imaging Surveys DR10

Architecture:
- Image encoder: ResNet-50 backbone (pre-trained ImageNet)
- Feature concatenation with photometric measurements
- Dense layers: 1024→512→256 neurons
- Output layer: Gaussian mixture density network (z_mean, z_std)

Training Data:
- 8,098 AGN with spectroscopic redshifts
- Stratified by redshift (dz=0.1 bins)
- Augmentation: image rotations, crops, photometric noise injection
```

**Performance:**
```
Validation Set (8,098 AGN):
- σ_NMAD: 4.5% (normalized median absolute deviation)
- Outlier fraction: 5.6% (|Δz|/(1+z) > 0.15)
- Bias: <0.2% (negligible)

Performance vs. Redshift:
z < 0.5:  σ_z ~ 0.02
0.5 < z < 2: σ_z ~ 0.05 (3-4%)
z > 2:    σ_z ~ 0.10 (5-6%)
```

**Advantages:**
- CNN learns complex multi-band color-redshift relationships
- Handles AGN-specific photometric challenges (SED variability)
- Asymmetric error estimates (Gaussian mixture output)
- Tested on 8,000+ AGN

**Limitations:**
- Requires deep imaging (DESI Legacy quality)
- Limited high-z performance (sparse training above z=4)
- Needs GPU for inference (slower for batch applications)
- Published for specific DESI Legacy fields

**Access:**
- Code: Available from collaboration repositories
- Trained weights: Downloadable from Zenodo/IRSA
- Data: XMM-SERVS W-CDF-S, ELAIS-S1, LSS fields

---

#### Traditional SED Fitting (Template-Based Photo-z)

**Algorithm:**
```
Fit galaxy SED templates to multi-band photometry
Template library includes:
- Spiral, elliptical, starburst galaxy templates
- AGN torus templates (Nenkova+ 2008, Mullaney+ 2011)
- Dust attenuation (Calzetti+ 2000 or SMC-like)

χ² minimization over (z, extinction, galaxy type, AGN fraction):

χ² = Σ [(f_obs - α*f_template(z))² / σ_f²]

Photo-z = z_min_chi2
Uncertainty σ_z from χ² posterior distribution (P(z) ∝ exp(-χ²/2))
```

**Performance:**
```
Accuracy: σ_z/(1+z) ~ 5-15% (survey/template-dependent)
Bias: ~1-2% (template mismatch effects)
Outlier fraction: 5-10%

Advantages over ML:
- Physically motivated (template basis)
- Interpretable (galaxy type, extinction, redshift)
- No labeled training data needed
- Low computational cost

Disadvantages:
- Degeneracies (age/dust, z/attenuation tradeoffs)
- Template incompleteness (unusual AGN types)
- Correlated errors across objects
```

**Software:**
- EAZY (Brammer+ 2008): Widely used in surveys (COSMOS, etc.)
- Le PHARE: French software; supports AGN templates
- Hyperz: Legacy code; historical importance

---

## 3. BENCHMARK COMPARISON TABLE

| Method | Input Data | Training Req'd | Accuracy | Speed | Interpretability | Best For |
|--------|-----------|---------------|----------|-------|-----------------|----------|
| **BPT Diagnostic** | 4 Emission lines | Spectroscopy | ~85% (z<0.4) | Fast | Excellent | Low-z AGN/SFG |
| **WISE Colors** | W1-W4 mags | None | ~90% (purity) | Very fast | Excellent | Mid-IR AGN selection |
| **X-ray Hardness** | Soft+Hard X-rays | None | ~80% (unobs.) | Very fast | Good | AGN obscuration |
| **Random Forest** | Optical+IR mags | Yes (labeled) | ~90% | Fast | Good (feature imp.) | Photometric AGN ID |
| **SVM** | Optical+IR mags | Yes (labeled) | ~88% | Moderate | Poor | Similar to RF |
| **CNN (images)** | Galaxy images | Yes (many labels) | ~95% | Moderate (GPU) | Poor (black box) | Morphology classification |
| **PICZL (Photo-z)** | Optical imaging | Yes (8K AGN spec-z) | σ_z=4.5% | Moderate (GPU) | Moderate | AGN redshifts |
| **SED Fitting** | Multi-band photometry | None | σ_z=5-15% | Slow | Excellent | Redshifts + properties |

---

## 4. RECOMMENDATIONS FOR AGN/SFG CLASSIFICATION

### For Low-Redshift (z < 0.5) Studies:
```
Primary: BPT diagnostic (spectroscopy if available)
Secondary: WISE W1-W2 > 0.8 (mid-IR confirmation)
Tertiary: X-ray hardness (if Chandra/XMM available)
Optimal: Combine all three for robustness
```

### For High-Redshift (z > 2) Studies:
```
Primary: X-ray detection (Chandra, XMM, eROSITA)
Secondary: WISE colors (AGN torus emission visible)
Tertiary: Radio morphology (VLASS, LOFAR)
Photo-z: Use PICZL for AGN-specific redshifts
Challenge: BPT unavailable (emission lines shifted out)
```

### For All-Sky Surveys:
```
Primary: WISE AGN colors (W1-W2 criterion)
Secondary: Machine Learning (RF + optical photometry)
Tertiary: Multi-wavelength matching (X-ray, radio if available)
Advantage: Rapid identification of millions of sources
```

### For Machine Learning Pipelines:
```
1. Feature engineering: Standardized photometry + photo-z
2. Training data: SDSS (930K) + COSMOS (4K high-z)
3. Method: Random Forest (balance of speed + accuracy)
4. Validation: Cross-validation + independent test set
5. Calibration: Compare to spectroscopic subsample
6. Uncertainty quantification: Voting probabilities or prediction intervals
```

---

## REFERENCES

Recent Papers on AGN/SFG Classification:

1. **PICZL Photo-z:** Rau+ 2024, A&A 692, A260
2. **WISE AGN Catalog:** Assef+ 2018, ApJS 234, 23
3. **Multi-band ML:** Banfield+ 2023, A&A (LoTSS ML classification)
4. **Deep Learning AGN:** Hearin+ 2024, arXiv:2410.01437
5. **Radio Galaxy Classification:** Miettinen+ 2024, RGC-Bent dataset
6. **Chandra-COSMOS:** Civano+ 2016, ApJ 819, 62
7. **GAMA AGN:** Thorne+ 2021, MNRAS (GAMA AGN properties)

---

**Document Status:** Benchmark compilation December 22, 2025. Ready for citation-ready synthesis in AGN/SFG classification studies.
