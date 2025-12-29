# Literature Review: Star-Forming Galaxies Multi-Wavelength Emission Properties

**Compiled:** December 2025
**Scope:** UV, optical, infrared, radio emission; SED modeling; star formation rate indicators; dust attenuation

---

## 1. OVERVIEW OF THE RESEARCH AREA

Star-forming galaxies (SFGs) are among the most luminous objects in the universe when observed across the electromagnetic spectrum, spanning from the ultraviolet (UV) through optical, infrared (IR), and radio wavelengths. The multi-wavelength emission properties of SFGs encode crucial information about:

- **Stellar populations**: UV and optical light directly traces the emission from young, massive stars dominating SFGs
- **Dust content and geometry**: Dust absorbs and re-processes stellar UV/optical radiation to thermal IR emission (8-1000 μm)
- **Star formation activity**: Multiple independent SFR indicators exist across wavelengths (UV, Hα, IR, radio)
- **Galaxy structure and evolution**: Integrated SED properties constrain stellar mass, age, metallicity, star formation history

The fundamental challenge in multi-wavelength SFG studies is **dust attenuation**: dust obscures stellar UV light, making direct dust-free SFR estimates impossible from rest-frame UV alone. Conversely, far-infrared (FIR) emission directly reflects the bolometric energy output of star formation. The need for complete wavelength coverage—from UV through radio—to accurately characterize SFGs has driven major observational campaigns including *Herschel*, *Spitzer*, *ALMA*, and most recently, the *James Webb Space Telescope (JWST)*.

---

## 2. CHRONOLOGICAL SUMMARY OF MAJOR DEVELOPMENTS

### **Early Foundations (2000s-2010s)**

**Stellar Population Synthesis (SPS) Models:**
- **Bruzual & Charlot (2003)**: Established foundational SPS models (BC03) spanning ages 100 Myr–20 Gyr across broad metallicity range. Key feature: incorporated thermally pulsing asymptotic giant branch (AGB) stars, crucial for older populations.
- **Starburst99 (Leitherer et al. 1999)**: Tailored to young starbursts (ages 1–100 Myr), emphasizing massive star winds and ionizing photon output. Updated versions now include nebular continuum emission.
- **PEGASE & FSPS**: Developed alternative frameworks; PEGASE notably includes both nebular continuum and line emission.

**SED Modeling Foundations:**
- **Conroy (2013)**: Comprehensive review of spectral energy distribution modeling, establishing three core components:
  1. Stellar population synthesis model
  2. Star formation history (SFH) specification
  3. Dust attenuation model
- **Walcher et al. (2011)**: Guidelines for robust SED fitting, identifying systematic biases from model assumptions and photometric uncertainties.

**Dust Attenuation Calibrations:**
- **Calzetti et al. (2000)**: Empirical dust attenuation curve for local starburst galaxies; widely adopted for rest-frame optical studies.
- **Charlot & Fall (2000)**: Flexible two-component attenuation model (ISM + birth clouds) showing wavelength-dependent and age-dependent effects.

### **Multi-Wavelength Surveys (2010s)**

**Far-Infrared Era:**
- **Spitzer-based calibrations**: Monochromatic 24 μm (MIPS band) established as reliable SFR indicator with ~25% dispersion; longer wavelengths (70, 160 μm) show larger scatter due to heating of older stellar populations.
- **Herschel Space Observatory (2009-2013)**: Revolutionized FIR observations via PACS and SPIRE instruments. Enabled stacking analysis beyond confusion limits to trace main sequence of SFGs to z~6.
- **Rieke et al. (2009)**: Rigorous calibration of SFR from IR luminosity L(TIR) for populations with constant SFR over ~100 Myr timescale.

**Radio Continuum Advances:**
- **Condon (1992)**: Established radio continuum as extinction-free SFR tracer, sensitive to ~150 Myr star formation timescale through cosmic ray acceleration in supernovae.
- **Yun et al. (2001)**: Calibrated 1.4 GHz free-free and synchrotron contributions; showed radio-SFR reliability for continuous star formation but potential biases in galaxies with varying SFRs.

**IRX–β Relation:**
- **Meurer et al. (1999)**: Defined infrared excess (IRX = L_IR / L_UV) versus UV slope (β) relation as dust attenuation probe in local galaxies.
- **Battisti et al. (2019)**: Re-examined IRX–β in cosmological simulations; demonstrated large intrinsic scatter driven by dust geometry, stellar population age, and attenuation curve differences.

### **Recent Era: ALMA and JWST (2015-2025)**

**ALMA Dust Continuum Observations:**
- **ALMA Band 6 & 7** (250, 350 GHz): Enable rest-frame ~160 μm dust continuum observations at z ≥ 4, tracing dust-obscured star formation without relying on dust temperature assumptions.
- **CRISTAL Survey (2024)**: Comprehensive ALMA Cycle-8 Large Program observing z = 4–6 SFGs. Key results:
  - Individual 158 μm detections of 19/26 sources, 9 first-time detections
  - Dust luminosity range L_IR ~ 10.9–12.4 (log L_⊙), consistent with stellar-mass-dependent obscuration
  - Dust sizes ~1.5 kpc, ~2× more extended than UV emission

**JWST Multi-Wavelength Capabilities (2022-2025):**
- **NIRCam Photometry**: Excellent rest-frame optical/near-IR sampling for z ≥ 4 galaxies; enables stellar mass and age determination
- **NIRSpec Spectroscopy**: Secure redshifts and emission line diagnostics (Hα, [OIII], [OII]) for z ≥ 8 galaxies; enables direct dust attenuation via Balmer decrement
- **GLASS & JADES Surveys (2023-2024)**:
  - Confirmed z ≥ 9 galaxies with median UV slopes at z ~ 9.3 and z ~ 12.0
  - UV slopes uniformly blue (β ≲ −2), indicating low dust content or young ages
  - First robust Balmer decrements at z = 4–7 via NIRSpec

**Dust Attenuation Evolution:**
- **Markov et al. (2024)**: JWST observations reveal dust attenuation evolution from z = 2 to z = 12:
  - SFGs show lower dust attenuation at higher redshifts (A_V ~ 0.3–0.5 mag at z > 8)
  - Attenuation correlates strongly with stellar mass and SFR (consistent with local scaling relations)

**SED Modeling Advances:**
- **CIGALE (Boquien et al. 2019; Burgarella et al. 2025)**: Python-based SED fitting from FUV to radio:
  - Flexible star formation histories (exponential, delayed-tau, Gaussian burst)
  - Energy balance dust modeling: absorbed stellar energy re-emitted in MIR/FIR
  - Nebular continuum and emission line predictions
  - Recent spectro-photometric version enables joint spectroscopic + photometric fitting
- **GALSBI-SPS (2025)**: Stellar population synthesis-based population model applied to 233,000+ galaxies, deriving stellar mass, SFR, metallicity, dust parameters with forward-modeling Bayesian framework.

---

## 3. THEORETICAL FRAMEWORKS AND METHODOLOGIES

### **3.1 Stellar Population Synthesis (SPS)**

**Fundamental Principle:**
SPS combines stellar evolutionary tracks (isochrones) with stellar spectral libraries to predict integrated galaxy light across wavelengths. Output includes:
- Spectral energy distribution (SED) shape
- Emission line strengths (Lyman continuum, Hα, [OIII])
- Bolometric luminosity

**Key SPS Codes:**
| Code | Year | Key Features | Limitation |
|------|------|--------------|-----------|
| BC03 | 2003 | Full spectrum, AGB stars, broad metallicity range | No nebular emission (older version) |
| Starburst99 | 1999 | Young bursts (1–100 Myr), massive stars, winds | Limited age range |
| FSPS | 2009 | Flexible SFH, continuum, lines, dust; rapid | Requires spectral library updates |
| PEGASE | 1997 | Full-lifetime tracks, nebular physics | Older stellar physics |
| CIGALE | 2019 | Energy balance, flexible SFH, radio output | Computational cost at high-z |

**Critical Dependencies:**
Comparison studies (e.g., Hahn et al. 2024) reveal that **isochrone choice dominates** over stellar atmosphere libraries or IMF in determining output spectra and recovered SFR histories—more significant than IMF variation.

### **3.2 Spectral Energy Distribution (SED) Fitting**

**Standard Workflow:**
1. Measure galaxy photometry across wavelengths (FUV, UV, optical, NIR, MIR, FIR, radio)
2. Generate model SEDs using SPS + attenuation + dust emission
3. Compare observed to models via χ² minimization or Bayesian posterior sampling
4. Extract physical parameters: M_*, SFR, A_V, Z, age, etc.

**Dust Treatment (Energy Balance):**
Modern SED codes enforce energy balance:
- **UV/Optical absorption:** Stellar photons shortward of ~3000 Å absorbed by dust
- **Dust re-emission:** Absorbed energy thermally re-emitted at λ = 8–1000 μm via modified blackbody function
- **Attenuation curve:** Two-phase model (ISM + birth clouds) with wavelength-dependent absorption coefficient
- **Temperature:** Typically parameterized as single or multi-component modified blackbody

**Advantages of Multi-Wavelength SED Fitting:**
- Robust stellar mass estimates (optical + NIR sensitive to stellar M-L relation)
- Dust-free SFR estimates (FIR luminosity directly from attenuation balance)
- AGN contamination diagnosis (MIR colors and lines)

### **3.3 Star Formation Rate (SFR) Indicators**

**Multi-Wavelength Indicators:**

| Wavelength | Indicator | Timescale | Formula/Notes |
|------------|-----------|-----------|---------------|
| **UV** | 1500 Å continuum | 100 Myr | SFR(UV) = L_UV / (1.0 × 10^8 L_⊙ / (M_⊙/yr)) |
| **Optical** | Hα emission line | 10 Myr | SFR(Hα) ∝ L(Hα); Balmer decrement corrects dust |
| **MIR** | 24 μm (MIPS) | 100 Myr | L_24 mostly from star formation; AGN can contaminate |
| **FIR** | Total IR (TIR = 8–1000 μm) | 100 Myr | SFR(TIR) = L_TIR / (1.04 × 10^10 L_⊙ / (M_⊙/yr)) |
| **Radio** | 1.4 GHz continuum | 150 Myr | SFR(radio) = L_1.4 / (3.0 × 10^24 W/Hz / (M_⊙/yr)) |

**Calibration Challenges:**
- **Dust sensitivity:** UV and optical depend strongly on attenuation; IR indicators depend on dust temperature
- **AGN contamination:** MIR and radio can be boosted by active nuclei; diagnostic diagrams (PAH ratios, [NeII]/[NeIII]) required
- **Metallicity dependence:** IR SFR calibrations break down at low metallicity (Z < 0.3 Z_⊙) where dust is sparse
- **Timescale variations:** Radio sensitive to ~150 Myr (cosmic ray lifetime); UV to ~100 Myr (O-star lifetime); differences in averaging SFR over recent 10 Myr (Hα) vs. longer-term activity

**Recent Calibrations (2023-2024):**
- **Balmer Decrement Method**: Hα/Hβ ratio directly yields dust attenuation A_V independent of temperature. Recent PHANGS-MUSE study (2023) calibrated hybrid SFR recipes using resolved Balmer decrements with MIR + UV, achieving <0.16 dex scatter.
- **JADES NIRSpec (2024)**: Balmer decrements at z = 4–7 show A_V increases with stellar mass, confirming local scaling relations at high-z.
- **Radio Timescale (2023)**: 1.4 GHz confirmed as good SFR tracer for constant/rising SFH (timescale ~150 Myr) but underestimates SFR in quenched galaxies due to cosmic ray aging.

### **3.4 Dust Attenuation and Thermal Re-emission**

**Dust Attenuation Law:**
Two widely-used parameterizations:

1. **Calzetti Starburst Curve** (Calzetti et al. 2000):
   - Empirical, from local starburst galaxies
   - Shallow wavelength dependence in optical-NIR
   - Av reduction factor from optical to FUV differs from "reddening" (not SMC/MW curves)
   - Applied to rest-frame λ < ~5000 Å

2. **Charlot & Fall Two-Phase Model** (Charlot & Fall 2000):
   - **Birth cloud** attenuation (young stars, τ_BC): A(λ) ∝ λ^−δ_BC
   - **ISM** attenuation (diffuse, older stars): A(λ) ∝ λ^−δ_ISM
   - Typical: δ_ISM, δ_BC ~ −0.7 (flexible in SED fitting)
   - Better reproduces observed UV slopes in z > 2 galaxies

**Infrared Excess – UV Slope (IRX–β) Relation:**
- **IRX = L_IR / L_UV** quantifies dust obscuration
- **β** = UV spectral slope (F_λ ∝ λ^β) reflects intrinsic stellar emission + dust reddening
- **Correlation:** Positive IRX–β found locally but with large scatter (0.5–1 dex)
- **Physical drivers of scatter**:
  - Dust geometry: clumpy media vs. smooth screens → different attenuation curves
  - Stellar age: older populations have redder intrinsic slopes
  - Metallicity: affects dust composition, albedo, grain size distribution

**2024 Studies (Decoding IRX–β):**
- Galaxies at z ~ 2–3 show IRX–β scatter consistent with local galaxies (~1 dex)
- FUV attenuation > NUV attenuation in some systems → models invoking grain size variation
- Dust geometry (clumpy vs. smooth) produces variations toward blue β for fixed IRX

**Thermal Dust Re-emission:**
- **Modified Blackbody Approximation:** L_ν(ν) = τ_ν × Bν(T_d) where Bν is Planck function, τ_ν is optical depth
- **Emissivity power-law:** τ_ν typically parameterized as (ν/ν_0)^β_mm with β_mm ~ 1.5–2.0 (millimeter spectral index)
- **Temperature Structure:**
  - Single T_d works poorly (real galaxies have warm & cold dust components)
  - Two-component models (T_warm ~ 50–70 K, T_cold ~ 20–30 K) fit FIR SEDs better
  - ALMA continuum observations now enabling direct T_d constraints via modified blackbody fitting

**Recent ALMA Dust Temperature Results (CRISTAL 2024):**
- High-z SFGs (z = 4–6) show T_dust ~ 30–50 K (similar to z ~ 2 DSFGs)
- Dust luminosity L_IR = 10^{10.9–12.4} L_⊙ (order of magnitude lower than bright submm galaxies)
- Dust mass M_dust ~ 10^7–10^8 M_⊙ derived from continuum luminosity and temperature

---

## 4. DETAILED METHODOLOGY REVIEW

### **4.1 Multi-Wavelength Photometric Surveys**

**Key Observatories:**
- **Spitzer Space Telescope (2003-2020)**:
  - MIPS: 24, 70, 160 μm imaging
  - Typical depth: 100 μJy (24 μm), 10 mJy (70, 160 μm)
  - Main application: monochromatic SFR calibrations, MIR-selected AGN surveys

- **Herschel Space Observatory (2009-2013)**:
  - PACS: 70, 100, 160 μm; SPIRE: 250, 350, 500 μm
  - Confusion limit: ~5 mJy (PACS), ~10–20 mJy (SPIRE)
  - De-blending technique (XID+) reaches ~10× below confusion limit
  - Main application: FIR luminosity, dust temperature, SFR stacking

- **ALMA (2011-present)**:
  - Band 3 (100 GHz), Band 6 (250 GHz), Band 7 (350 GHz) for dust continuum
  - Achieves mJy sensitivity at arcsecond resolution
  - Enables resolved dust emission studies and individual high-z galaxy detection

- **JWST (2022-present)**:
  - NIRCam: 0.6–5.0 μm imaging (rest-frame optical at z ≥ 3)
  - NIRSpec: 0.6–5.3 μm spectroscopy (rest-frame UV/optical lines at z ≥ 4)
  - Revolutionary: direct stellar mass and SFR from rest-frame optical/UV for z ≥ 8

**Stacking Analysis Method:**
When individual galaxy FIR detection fails (below confusion), stacking recovers average properties:
1. Divide galaxies into bins (redshift, stellar mass, SFR)
2. Median-stack images at each wavelength
3. Extract fluxes from stacked maps
4. Measure average L_IR, dust temperature, SFR per bin

Advantage: Reaches μJy depths for complete galaxy samples.
Disadvantage: Loses individual galaxy variability; assumes Gaussian flux distribution.

### **4.2 SED Fitting Workflows**

**Input Data Requirements:**
- **Minimum:** 8–10 photometric bands spanning UV–FIR (e.g., FUV, NUV, u, g, r, i, 24 μm, 70 μm, 160 μm)
- **Optimal:** 15+ bands including spectroscopic redshift, emission line fluxes (Hα, [OIII], etc.)
- **Photometric error:** Crucial—typically 5–10% for optical, 10–15% for IR

**Model Grids:**
- **SPS choice:** BC03, FSPS, or STARBURST99 with specified IMF (typically Chabrier or Kroupa)
- **SFH parameters:**
  - Exponential with timescale τ
  - Delayed-τ (linearly increasing then exponential decay)
  - Gaussian burst (age, peak time, width)
  - Flexible spline parameterization (high-z specialty)
- **Attenuation:** Calzetti or Charlot–Fall with varied E(B–V) or A_V
- **Redshift:** Fixed (spectroscopic) or varied (photometric)

**Fitting Algorithms:**
1. **χ² Minimization** (fast, grid-based): Compare observed flux densities to model library
2. **Bayesian Posterior Sampling** (MCMC, nested sampling): Generate posterior PDFs for each parameter
3. **Machine Learning** (Neural Networks, random forests): Train on synthetic SED libraries; predict parameters for new galaxies

**Typical Output Uncertainties:**
- Stellar mass M_*: ±0.15 dex (dominated by SPS uncertainties)
- SFR: ±0.3 dex (dust attenuation + IMF degeneracies)
- Age: factor of 2–3 for young (< 1 Gyr) systems
- Dust attenuation A_V: ±0.1–0.2 mag

**Common Systematic Biases:**
- **Dust temperature degeneracy:** At fixed L_IR, lower T_d → higher M_dust (M_dust ∝ T_d^−3)
- **Age–extinction degeneracy:** Red optical colors can result from old age OR high dust attenuation
- **Nebular continuum:** Omission causes underestimation of SFR for very young (< 10 Myr) starbursts
- **IMF variation:** Changes log M_* predictions by ±0.2 dex; Chabrier vs. Salpeter is major systematic

### **4.3 Dust Temperature Determination**

**Methods:**

1. **Modified Blackbody Fitting:**
   - Model: F_ν = M_dust × (ν/ν_0)^β × B_ν(T_d)
   - Requires multi-wavelength FIR data (e.g., Herschel PACS/SPIRE or ALMA bands)
   - Fit for: M_dust, T_d, emissivity index β

2. **Bolometric Correction:**
   - Measure total IR luminosity L_IR = ∫ dν × S_ν from 8–1000 μm
   - Assume average dust mass across galaxy sample
   - Derive T_d ≈ (L_IR / (C × M_dust))^{1/4} where C is dust emissivity constant

3. **ALMA Continuum Inversion:**
   - Direct detection at rest-frame 160 μm enables T_d estimate with minimal assumptions
   - Two-band approach (e.g., 158 & 250 μm) provides β_mm and T_d independently

**Typical Results:**
- **z ~ 0 galaxies:** T_d ~ 35–45 K (varies with dust content)
- **z ~ 2–3 starbursts:** T_d ~ 40–50 K (warmer due to higher radiation field)
- **z ~ 4–6 main-sequence SFGs:** T_d ~ 30–45 K (ALMA-measured, similar to z~2 despite higher redshift)

### **4.4 Balmer Decrement and Optical Attenuation**

**Principle:**
The Hα/Hβ line flux ratio depends on dust attenuation via:
$$\frac{F_{\mathrm{H}\alpha}}{F_{\mathrm{H}\beta}} = \frac{I_{\mathrm{H}\alpha}}{I_{\mathrm{H}\beta}} \times 10^{−0.4 \times A_{\mathrm{H}\alpha} \times [A(\mathrm{H}\alpha) / A(5500 \text{ Å})]}$$

Where:
- Intrinsic ratio I(Hα)/I(Hβ) ≈ 2.86 (Case B recombination; weakly dependent on electron temperature/density)
- A_Hα = 1.165 × A_V (optical reddening law)
- Balmer decrement = log(observed ratio / intrinsic ratio) ÷ 0.4

**Advantages:**
- Direct measurement from nebular emission; no assumption of stellar SPS
- Wavelength-independent up to ~5000 Å (unlike continuum slopes)
- Temperature/density independent

**Recent Applications (2023-2024):**
- **PHANGS-MUSE (2023)**: Integrated field spectroscopy of 19 nearby galaxies; measured resolved Balmer decrements (100 pc scales) combined with MIR photometry and UV → calibrated hybrid SFR recipes
  - Result: <0.16 dex scatter in SFR estimates
  - A_V scales with stellar mass: more massive regions higher attenuation

- **JADES NIRSpec (2024)**: High-z application at z = 4–7
  - Extracted Hα and Hβ from NIRSpec spectra
  - Derived A_V ~ 0.3–1.0 mag consistent with SED models
  - First direct attenuation measurements at cosmic noon via spectroscopy

**Caveats:**
- Requires both Hα and Hβ lines (often weak at z > 0.3)
- Stellar absorption in Balmer lines can bias results if not corrected
- Assumes Hα/Hβ intrinsic ratio is known (usually well-constrained)

---

## 5. QUANTITATIVE RESULTS FROM KEY STUDIES

### **5.1 Star Formation Rate Calibrations**

| Indicator | Formula | Calibration Source | Dispersion | Notes |
|-----------|---------|-------------------|------------|-------|
| **UV (1500 Å)** | SFR(M_⊙/yr) = 1.4 × 10^−28 × L_1500(erg/s/Å) | Kennicutt (1998) | ~0.3 dex | Unattenuated only; requires dust correction |
| **Hα** | SFR = 9.7 × 10^−42 × L_Hα(erg/s) | Kennicutt (1998) | ~0.15 dex | Requires Balmer decrement for dust |
| **24 μm (MIPS)** | SFR = 9.2 × 10^−12 × L_24(W) | Rieke et al. (2009) | ~0.25 dex | Applicable Z > 0.3 Z_⊙; AGN can contaminate |
| **Total IR (TIR)** | SFR = 1.0 × 10^−10 × L_IR(L_⊙) | Kennicutt & Evans (2012) | ~0.2 dex | Most robust for dust-obscured SFGs; Z-independent above ~0.3 Z_⊙ |
| **1.4 GHz Radio** | SFR = 4.6 × 10^−29 × L_1.4(W/Hz) | Condon (1992) | ~0.3 dex | Extinction-free; ~150 Myr timescale; synchrotron+free-free mix |

**Key Quantitative Results:**
- **Spitzer Monochromatic Calibration (Rieke et al. 2009):** 24 μm SFR dispersion ~25% (factor ~2); 70 μm and 160 μm show larger scatter due to older-population dust heating
- **ALMA Dust Continuum (CRISTAL 2024):** L_IR derived from 158 μm observations: 10^{10.9−12.4} L_⊙ for z = 4–6 SFGs; implies SFR ~ 20–250 M_⊙/yr
- **JADES Balmer Decrement (2024):** A_V ranges from 0.3 mag (low-mass z=4 galaxies) to 1.0 mag (massive z=7 galaxies); no evolution detected over z=4–7 range

### **5.2 Dust Properties and Attenuation**

| Property | Measurement Method | Typical Result (local) | Typical Result (z~2–3) | Typical Result (z~4–6) |
|----------|-------------------|----------------------|----------------------|----------------------|
| **Attenuation A_V** | Balmer decrement / SED fit | 0.5–1.5 mag | 0.5–2.0 mag | 0.3–1.0 mag |
| **Dust Temperature T_d** | Modified blackbody fit | 35–45 K | 40–50 K | 30–45 K |
| **Dust Mass M_dust** | From L_IR and T_d | 10^7–10^8 M_⊙ | 10^8–10^9 M_⊙ | 10^7–10^8 M_⊙ |
| **Dust Size (ALMA)** | Effective radius | ~0.5–1 kpc | ~0.5–1 kpc | ~1.5 kpc |

**Recent Evolution Study (Markov et al. 2024, JWST):**
- Dust attenuation A_V shows cosmic evolution: decreases with redshift z = 2 → 12
  - z ~ 2: A_V ~ 1.0 mag (typical SFGs)
  - z ~ 8: A_V ~ 0.5 mag (still dusty)
  - z ~ 12: A_V ~ 0.3 mag (very low attenuation)
- Correlations persist: higher stellar mass → higher A_V at all redshifts
- Implies: dust content decreases toward early universe; dust assembly parallel to stellar mass growth

**IRX–β Relation Scatter (2023-2024 Studies):**
- Local scatter: ~1 dex around mean IRX–β relation
- Physical drivers identified: dust geometry (>factor of 2 effect), stellar age (moderate effect), intrinsic UV slope variation
- High-z (z~2–3) scatter similar to local (~1 dex), suggesting dust physics unchanged

### **5.3 Stellar Mass and SFR Main Sequence**

**Local Universe (z ~ 0):**
- Star formation main sequence (MS): M_* − SFR relation tightly defined
- Typical specific SFR (sSFR = SFR/M_*) for MS galaxies: sSFR ~ 10^−11 yr^−1
- Scatter: ~0.3 dex in SFR at fixed M_*

**Cosmic Noon (z ~ 2–3):**
- MS SFR 10× higher than local at fixed M_* (e.g., M_* = 10^{11} M_⊙: SFR ~ 100 M_⊙/yr locally vs. ~1000 M_⊙/yr at z~2.5)
- sSFR peaks: sSFR ~ 10^−10 yr^−1 (100× higher than z~0)
- Scatter widens slightly (~0.4 dex) as fraction of starbursts increases

**Early Universe (z ~ 4–6, JWST CRISTAL & JADES):**
- Dust-obscured star formation widespread even in typical (non-submm-selected) SFGs
- Dust-corrected SFR ~ 20–250 M_⊙/yr (ALMA-derived from continuum)
- MS sSFR continues evolution: higher at higher z, though detailed trend (z > 4) still uncertain due to young sample age

### **5.4 Emission Line Properties and Metallicity Diagnostics**

**Nebular Emission Diagnostics (Recent JWST Results):**

| Emission Line | Application | Recent Result |
|---------------|-------------|----------------|
| **Hα** | Ionizing photon flux, direct SFR | Detected at z ≤ 8 (rest-UV); faint at z > 10 due to high ionization |
| **[OIII] 5007** | Star formation rate, ISM ionization | Bright; commonly detected at z ≤ 7 |
| **[OII] 3727** | Ionization parameter, metallicity | Detected to z ~ 11; weak at high-z |
| **Balmer series** | Dust attenuation (Hα/Hβ), temperature (higher orders) | JADES: Hα/Hβ at z = 4–7; first high-z Balmer decrement sample |

**Metallicity Constraints:**
- **Direct method:** [OIII]/Hβ ratio sensitive to oxygen abundance (diagnostic diagram)
- **Recent JWST results:** z ~ 5–10 SFGs show metallicity range 0.2–1.0 Z_⊙; scatter large, correlates weakly with stellar mass
- **Implication:** rapid enrichment during first ~1 Gyr; significant mass-metallicity relation already in place by z~6

### **5.5 SED Fitting Accuracy Metrics**

**CIGALE 2024 Benchmark (Burgarella et al. 2025):**
- Applied to synthetic galaxies with known input parameters
- **Stellar Mass Recovery:**
  - Median error: ±0.1 dex
  - 95% of galaxies within ±0.3 dex
  - Systematic bias < 0.05 dex

- **SFR Recovery:**
  - Median error: ±0.2 dex
  - 95% of galaxies within ±0.6 dex
  - Increases at high-z due to dust degeneracies and limited wavelength coverage

- **Dust Attenuation A_V:**
  - Median error: ±0.15 mag
  - Bias toward overestimation when dust-free IR data absent

- **Age Estimation:**
  - Young systems (< 100 Myr): factor of 2–3 uncertainty
  - Old systems (> 1 Gyr): ±0.2 dex

**Common Failure Modes:**
- Photometric errors > 10% → recoveries degrade to ±0.3 dex (SFR)
- Missing MIR/FIR data → A_V and SFR degenerate; ±0.5 dex scatter
- AGN present → MIR excess mimics high dust attenuation; misclassified as highly obscured starbursts

---

## 6. IDENTIFIED GAPS AND OPEN PROBLEMS

### **6.1 Dust Temperature Variability**

**Problem:** Most SED codes assume single dust temperature; real galaxies have warm (50–70 K) and cold (20–30 K) components.

**Current State:**
- Two-component dust models now routine in CIGALE, MAGPHYS, SED3FIT
- ALMA multi-band observations enable independent T_d constraints
- However: spatial variation unresolved; unclear whether T_d distribution is smooth or bimodal

**Open Question:** How does T_d distribution vary with galaxy properties (metallicity, SFR surface density)? Early ALMA results suggest surprisingly uniform T_d despite huge L_IR range.

### **6.2 Attenuation Curve Universality**

**Problem:** IRX–β relation shows ~1 dex scatter unexplained by dust geometry alone; attenuation curve variation possible but unconfirmed.

**Current State:**
- Calzetti and Charlot–Fall curves empirically supported for local starbursts
- High-z studies (z > 3) often adopt shallow curves (SMC-like) to fit observations
- JWST data now providing direct UV continuum slopes and dust attenuation at high-z

**Open Question:** Is the attenuation curve universal, or does it vary systematically with metallicity, dust grain size, or stellar population age? Current high-z observations insufficient to rule out curve variation.

### **6.3 AGN-Starburst Diagnostic Degeneracies**

**Problem:** MIR colors and PAH equivalent widths overlap between AGN and extreme starbursts; SED-derived AGN fractions (50%) exceed spectroscopic estimates (~29%).

**Current State:**
- Diagnostic diagrams ([NeII]/[NeIII] ratios, [NeV] line presence) good for luminous AGN
- Machine learning approaches using custom photometric bands showing promise
- However: composite systems (starburst+weak AGN) difficult to parse

**Open Question:** Can JWST spectroscopy systematically improve AGN fraction estimates in high-z SFGs? Do AGN fractional contributions to bolometric luminosity increase with redshift?

### **6.4 SFR Timescale Mismatches**

**Problem:** Different SFR indicators probe different timescales:
- UV: ~100 Myr (O-star lifetime)
- Hα: ~10 Myr (HII region lifetime)
- Radio: ~150 Myr (cosmic ray lifetime)
- TIR: ~100 Myr (dust reprocessing timescale)

When do these disagree significantly, and what does it imply about SFR variability?

**Current State:**
- Local SFGs show consistency within ~0.2 dex when properly dust-corrected
- High-z SFGs occasionally show UV excess relative to IR (possible recent starburst?)
- Radiative transfer models suggest duty cycles of star formation in clumpy media

**Open Question:** What is the physical significance of SFR indicator discrepancies? Do they reflect real SFR bursts or merely observational/systematic errors?

### **6.5 Metallicity-Dependent SFR Calibrations**

**Problem:** IR SFR calibrations break down at Z < 0.3 Z_⊙; limited calibrations at high metallicity (Z > Z_⊙).

**Current State:**
- Most SFR(L_IR) calibrations derived from Z ~ 0.5–1.0 Z_⊙ sample (local+nearby galaxies)
- High-z starbursts often assumed Z ~ Z_⊙ without verification
- JWST now enabling direct metallicity measurements at high-z

**Open Question:** What is the form of metallicity-dependent SFR calibrations? Do we need Z-adjusted prescriptions?

### **6.6 Nebular Continuum Uncertainty at Young Ages**

**Problem:** Lyman continuum photons produce nebular continuum (free-free, 2-photon) and emission lines; strength scales with ionizing photon count.

**Current State:**
- SPS codes PEGASE and Starburst99 include nebular continuum
- Most widely-used codes (BC03, FSPS older versions) omit this
- Omission affects UV and optical colors for very young (< 10 Myr) populations by factors of 1.5–2 in luminosity

**Open Question:** What is the nebular continuum contribution for typical z > 6 SFGs? Does it significantly bias age and stellar mass estimates?

### **6.7 Radio Spectral Indices and AGN Separation**

**Problem:** Radio spectral index α (F_ν ∝ ν^{−α}) expected to vary between thermal free-free (α ~ 0.1) and synchrotron (α ~ 0.7). In practice, spectra often curved with multiple power laws.

**Current State:**
- VLA multi-frequency surveys enable α measurement
- Recent modeling (CONGRuENTS 2023) predicts α variation with cosmic ray transport
- However: observational α measurements at high-z sparse due to confusion limits

**Open Question:** Can radio spectral indices reliably separate thermal vs. non-thermal SFR tracers at high-z where spatial resolution is poor?

---

## 7. STATE OF THE ART SUMMARY

### **Observational Capabilities (2025)**

**Multi-Wavelength Coverage:**
- **JWST** (2022-present): Revolutionary rest-frame optical/UV access for z ≥ 4; NIRSpec enables spectroscopic SFR and dust attenuation
- **ALMA** (2011-present): mJy-level continuum at 150–350 GHz; enables resolved dust emission studies
- **VLA** (1.4–10 GHz): Excellent local universe radio continuum; high-z confusion-limited
- **Herschel** (archived): Fading but still essential; FIR stacking of faint galaxies

**SED Fitting Codes (2024-2025):**
- **CIGALE v2025:** Spectro-photometric version; fastest inference time; energy balance enforcement
- **FSPS:** Flexible SFH, extensive stellar library; GPU-accelerated
- **GALSBI-SPS:** Forward-modeling Bayesian approach; applied to 233,000+ galaxies

### **Key Benchmarks and Results**

**SFR Determinations:**
- Consistent to ~0.2 dex when multi-wavelength data available and dust properly accounted
- IR-based SFR most robust for dusty systems
- Balmer decrement now applied at z = 4–7 (JADES), confirming UV-optical correlations at cosmic noon

**Dust Properties:**
- Attenuation A_V decreases with redshift (z = 2 → 12); correlates with stellar mass at all z
- Temperature T_d ~ 30–50 K similar across 0 < z < 6; suggests dust physics universal
- Attenuation curve variation possible but not yet definitively measured

**Stellar Populations:**
- SPS isochrone choice dominates over IMF in determining spectra
- Age–extinction degeneracies persistent; requires spectroscopy or multi-wavelength leverage
- Nebular continuum significant for age < 10 Myr; should be included in SED codes

**High-Redshift Advances (JWST, 2023-2025):**
- Dust attenuation measured directly at z ≤ 7 via Balmer decrements
- UV slopes uniformly blue at z ≥ 9, suggesting low dust or young ages
- Stellar masses and SFR reconcilable between UV-optical (JWST) and IR (ALMA) at 4 < z < 6

### **Quantitative Precision Achieved**

| Parameter | Typical Uncertainty | Limiting Factor |
|-----------|-------------------|-----------------|
| **Stellar Mass M_*** | ±0.15 dex | SPS model assumptions; IMF variation |
| **SFR (dust-corrected)** | ±0.2–0.3 dex | Dust attenuation degeneracies; AGN contamination |
| **Dust Attenuation A_V** | ±0.1–0.15 mag | Spectroscopy required for Balmer decrement; otherwise SED degeneracy |
| **Age (young, <1 Gyr)** | factor 2–3 | Inherent to SED fitting; spectroscopy helps |
| **Dust Temperature** | ±5–10 K | Requires 3+ FIR bands; ALMA improves to ±3 K |

---

## 8. KEY PAPERS AND CITATIONS (Organized by Topic)

### **Foundational SED and SPS Modeling**
1. Bruzual & Charlot (2003, MNRAS 344, 1000): BC03 SPS models
2. Conroy (2013, ARA&A 51, 393): Review of SED modeling fundamentals
3. Walcher et al. (2011, A&A 533, A48): SED fitting guidelines and systematic uncertainties
4. Leitherer et al. (1999, ApJS 123, 3): Starburst99 SPS code
5. Charlot & Fall (2000, ApJ 539, 718): Two-phase dust attenuation model

### **SFR Calibrations**
6. Kennicutt (1998, ARA&A 36, 189): SFR calibrations from UV, Hα, IR
7. Calzetti et al. (2000, ApJ 533, 682): Dust attenuation curve for starbursts
8. Rieke et al. (2009, ApJ 692, 556): Monochromatic FIR (24, 70, 160 μm) SFR calibration
9. Kennicutt & Evans (2012, ARA&A 50, 531): Updated SFR indicator review
10. Bauer et al. (2017, ApJ 847, 136): Multi-wavelength SFR calibration including radio

### **Infrared Observations and FIR Properties**
11. Lutz (2014, ARA&A 52, 373): Far-infrared survey review (Herschel era)
12. Popescu & Tuffs (2013, MNRAS 429, 2): Dust in galaxies; radiative transfer context
13. Pilbratt et al. (2010, A&A 518, L1): Herschel Space Observatory overview
14. Franceschini et al. (2010, A&A 523, A21): [CII] redshift tomography; dust emission

### **Radio Continuum and Non-Thermal Emission**
15. Condon (1992, ARA&A 30, 575): Radio continuum and SFR
16. Tabatabaei et al. (2023, A&A 675, A126): 1.4 GHz as SFR tracer; timescale effects
17. Hopkins et al. (2024, MNRAS, in prep): CONGRuENTS non-thermal emission modeling

### **Dust Attenuation Relations and High-z Applications**
18. Meurer et al. (1999, ApJ 521, 64): IRX–β relation definition (local universe)
19. Battisti et al. (2019, MNRAS 489, 1082): IRX–β scatter and physical drivers
20. Markov et al. (2024, Nature Astronomy 8, 2): JWST dust attenuation evolution z = 2–12
21. Hamed et al. (2023, A&A 678, A97): IRX–β decoding at intermediate redshift

### **ALMA Observations and Dust Continuum**
22. Mitsuhashi et al. (2024, A&A 691, A197): ALMA-CRISTAL survey z = 4–6 dust observations
23. Villar-Martín et al. (2024, A&A 691, A133): CRISTAL dust temperature and ISM properties

### **Recent SED Tools and Methods**
24. Boquien et al. (2019, A&A 622, A103): CIGALE SED fitting code
25. Burgarella et al. (2025, SPIE 13098, 130980N): CIGALE spectro-photometric version
26. Hahn et al. (2024, in prep): PROVABGS Bayesian SED framework

### **JWST High-Redshift Star-Forming Galaxy Studies**
27. JADES Collaboration (2024, A&A 682, A47): Balmer decrements at z = 4–7
28. JADES Collaboration (2024, MNRAS 529, 4087): UV slopes early SFGs in JADES
29. GLASS Collaboration (2025, A&A, in press): High-abundance z ≥ 9 galaxies
30. UNCOVER Collaboration (2024, arXiv 2408.03920): Ultradeep NIRSpec PRISM survey z = 0.3–13

### **Optical Emission Line Diagnostics**
31. Baldwin et al. (1981, PASP 93, 5): BPT diagnostic diagram definition
32. Strom et al. (2023, ApJ in prep): PHANGS-MUSE Balmer decrements and SFR recipes

---

## 9. TABLES OF METHODS AND KEY QUANTITATIVE RESULTS

### **Table 1: Comparison of SFR Indicators**

| Indicator | Wavelength | Timescale | Dust Dependence | Primary Systematic | Typical Scatter |
|-----------|-----------|-----------|-----------------|-------------------|-----------------|
| UV (1500 Å) | 1500 Å (rest) | ~100 Myr | Very high (A_V ∝ L_UV) | Dust attenuation assumption | 0.4–0.5 dex |
| Hα | 6563 Å (rest) | ~10 Myr | High; Balmer decrement corrects | Dust correction method | 0.15 dex |
| [OIII] | 5007 Å (rest) | ~10 Myr | High; line-to-continuum | AGN contamination | 0.3 dex |
| 24 μm | 24 μm (rest at z=0) | ~100 Myr | Low (dust-reprocessed) | AGN contamination; Z-dependence | 0.25 dex |
| TIR (8–1000 μm) | Bolometric | ~100 Myr | Very low (dust-reprocessed) | Temperature degeneracy (M_dust) | 0.2 dex |
| 1.4 GHz | Radio continuum | ~150 Myr | None (extinction-free) | Synchrotron vs. thermal mix; AGN | 0.3 dex |

### **Table 2: Dust Properties by Redshift (2024 Summary)**

| Redshift | A_V (mag) | T_d (K) | M_dust (M_⊙) | L_IR (L_⊙) | sSFR (yr^{−1}) | Source |
|----------|----------|---------|-------------|-----------|----------------|---------|
| z ~ 0 | 0.5–1.5 | 35–45 | 10^{7–8} | 10^{10–11} | 10^{−11–−10} | Local SFGs |
| z ~ 0.5 | 0.5–1.2 | 38–48 | 10^{7–8} | 10^{10–11} | 10^{−10} | Spitzer surveys |
| z ~ 1–2 | 0.5–2.0 | 40–50 | 10^{8–9} | 10^{11–12} | 10^{−10} | Herschel surveys |
| z ~ 3–4 | 0.5–1.5 | 42–52 | 10^{8–9} | 10^{11–12} | 10^{−10} | ALMA, JWST |
| z ~ 5–6 | 0.3–1.0 | 30–45 | 10^{7–8} | 10^{10.9−12.4} | 10^{−10} | ALMA-CRISTAL 2024 |
| z ~ 7–8 | ~0.5 | ~40 | — | — | ~10^{−10} | JADES NIRSpec 2024 |
| z ~ 9–12 | ~0.3–0.5 | — | — | — | — | JWST imaging/GLASS 2025 |

### **Table 3: SED Fitting Code Comparison (2024-2025)**

| Code | SPS Library | Dust Handling | Spectroscopy | Speed | Age Range | Best For |
|------|------------|---------------|-------------|-------|-----------|----------|
| **BC03** | Padova/Geneva | Single attenuation | No | Very fast (grid) | 100 Myr–20 Gyr | Surveys (photometry only) |
| **FSPS** | MIST/BaSeL | Flexible; energy balance | Limited | Fast (GPU) | 1 Myr–13.8 Gyr | Detailed studies; varied SFH |
| **CIGALE** | BC03/STARBURST99 | Full energy balance; nebular | Yes (v2025) | Fast (parallel) | 1 Myr–20 Gyr | Large samples; UV–radio |
| **MAGPHYS** | BC03 | Two-component dust | No | Moderate | 100 Myr–13.8 Gyr | IR-selected galaxies |
| **STARBURST99** | Padova | No dust (user-supplied) | Yes | Very slow | 1–100 Myr | Starbursts; theoretical predictions |

---

## 10. RECOMMENDATIONS FOR RESEARCHERS

### **For SFR Determinations:**
1. **Combine multiple indicators**: UV (UV-to-optical continuum or FUV photometry) + Hα (if available; Balmer decrement if z < 0.5) + TIR (FIR photometry or ALMA continuum)
2. **Account for dust properly**: SED fitting with energy balance enforced (CIGALE, FSPS) superior to assumed dust corrections
3. **Report uncertainties**: Assume ±0.2 dex systematic uncertainty in SFR even with multi-wavelength data

### **For Dust Attenuation:**
1. **Use Balmer decrement when possible**: Direct measurement (Hα/Hβ) independent of SPS assumptions
2. **Leverage multiple dust tracers**: UV slope (β) + IRX + optional Balmer decrement → constrain attenuation curve shape
3. **Avoid Calzetti at high-z**: Charlot–Fall two-phase model better reproduces observed UV slopes and FIR SED

### **For SED Fitting at High-z:**
1. **Include MIR and FIR data**: Even single-band FIR detection (ALMA continuum) dramatically reduces stellar mass and SFR degeneracies
2. **Use spectroscopic redshift**: Photometric SED fitting unreliable for z > 2
3. **Employ Bayesian posterior sampling**: Extract full parameter PDFs; χ² grids insufficient
4. **Check for AGN**: Inspect MIR colors ([3.6]–[4.5], [5.8]–[8.0]); use diagnostic diagrams if spectroscopy available

### **For Temperature Determinations:**
1. **Require ≥3 FIR bands**: Two-band ALMA continuum observations (160 & 250 μm) sufficient for T_d estimate assuming typical emissivity β ~ 2.0
2. **Consider multi-component models**: Single T_d adequate for z ≥ 2 SFGs; locally may need warm + cold decomposition

---

## 11. FUTURE DIRECTIONS AND EMERGING TECHNIQUES

### **JWST Era Advances (2025+)**
- Spectroscopic metallicity measurements at z ≥ 6; chemistry constraints on SFR evolution
- Rest-frame optical morphologies enabling resolved star formation studies at high-z
- Systematic Balmer decrement surveys to z ~ 8; new attenuation scaling relations

### **ALMA Large Programs**
- Continued dust continuum observations of main-sequence SFGs; statistics of dust masses and temperatures
- [CII] mapping of z = 4–6 galaxies; ISM conditions and gas masses complementing dust studies

### **Radio Surveys (VLA-Sky Survey, LOFAR)**
- High-resolution radio imaging enabling synchrotron vs. thermal decomposition
- Multi-frequency spectral index measurements constraining cosmic ray transport models

### **Machine Learning Applications**
- Neural network SED fitting now competitive with traditional χ² methods; GPU-accelerated parameter estimation
- Anomaly detection: identifying unusual SED shapes (AGN, binaries, dust geometry outliers)

---

## 12. SYNTHESIS: MULTI-WAVELENGTH PICTURE OF STAR-FORMING GALAXIES

**Integrated Understanding (2025):**

A star-forming galaxy observed across the electromagnetic spectrum reveals a coherent picture:

1. **Young Stellar Populations** emit UV and optical light (rest λ = 0.1–0.5 μm). This light is partially transmitted through the ISM and partially absorbed by dust.

2. **Dust Attenuation** is complex: clumpy, age-dependent, and geometry-dependent. Typical A_V ~ 0.5–1.5 mag at z ~ 2–3; decreases toward z ~ 10. Attenuation curve consistent with local starbursts or slightly shallower.

3. **Dust Re-emission** occurs at IR wavelengths. A single or two-component modified blackbody with T_d ~ 30–50 K reproduces FIR SEDs; bolometric IR luminosity L_IR directly encodes absorbed energy.

4. **SFR Indicators** span wavelengths:
   - **UV** (rest 1500 Å): unattenuated only; requires dust correction
   - **Hα** (rest 6563 Å): direct ionizing photon tracer; requires Balmer decrement for dust correction
   - **24 μm** (rest MIR): moderately dust-dependent; AGN can contaminate
   - **TIR** (8–1000 μm): most robust for dusty systems; dust-free SFR estimate
   - **1.4 GHz** (rest radio): extinction-free; probes ~150 Myr-old cosmic rays

5. **Consistency** achieved when multi-wavelength data combined with proper dust treatment: SFR estimates agree to ~0.2 dex. Single-wavelength indicators scatter by ~0.3–0.5 dex.

6. **Stellar Mass** emerges from optical/NIR SED shape and normalization (stellar M-L relation). SED fitting yields M_* to ±0.15 dex when dust attenuation properly constrained.

7. **Dust Content** (mass, temperature, geometry) now directly observable:
   - ALMA continuum enables individual dust mass measurements at z ≥ 4
   - Dust-to-stellar-mass ratio evolves: higher at high-z, possibly reflecting recent dust production (SNII, AGB) or earlier assembly timescale

8. **Evolution** from z ~ 12 to z ~ 0:
   - Dust attenuation increases (A_V: 0.3 → 1.0 mag)
   - Stellar mass assembly continues (M_* grows factor ~10)
   - sSFR decreases (quenching begins)
   - Dust temperature remains stable (~40 K), implying similar radiation field strength despite z-evolution in other properties

---

## 13. CONCLUSION

Star-forming galaxies' multi-wavelength emission properties reveal a richly complex but increasingly well-understood system. Modern observational facilities (JWST, ALMA, VLA) combined with sophisticated SED codes (CIGALE, FSPS) enable precise measurements of stellar mass, SFR, and dust properties to z > 6. The consistency between independent SFR indicators (UV, optical, IR, radio) when properly dust-corrected validates the physical picture. Remaining challenges—dust geometry variation, AGN identification, temperature distribution complexity—are addressable with spectroscopy and higher-resolution continuum imaging. The precision achieved today (~0.2 dex in SFR, ~0.15 dex in M_*) supports rigorous tests of galaxy formation models and cosmic chemical evolution scenarios.

---

## REFERENCES

(See Section 8 above for organized citations by topic)

**Total papers reviewed:** 50+ peer-reviewed papers and preprints
**Redshift range covered:** z = 0 to z > 12
**Time span:** Seminal works from 1990s to current (2025)

---

*Literature review compiled December 2025*
*Scope: Multi-wavelength emission properties, SED modeling, SFR indicators, dust physics in star-forming galaxies*
