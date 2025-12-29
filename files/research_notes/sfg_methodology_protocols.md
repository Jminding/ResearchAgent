# Star-Forming Galaxies: Methodology Protocols and Implementation Guide

**Compiled:** December 2025

---

## 1. MULTI-WAVELENGTH DATA COLLECTION PROTOCOL

### 1.1 Photometric Survey Selection Criteria

| Wavelength Range | Recommended Survey | Depth | Resolution | Pros | Cons |
|------------------|-------------------|-------|-----------|------|------|
| **UV (1500 Å)** | GALEX FUV | ~22 mag | 1.5" | Direct O-star light | Limited to z < 3 rest-frame |
| **Optical (0.4–0.9 μm)** | SDSS / Pan-STARRS / CFHTLS | ~22–24 mag | 0.5–1.5" | Multiple bands; deep | Extinction-affected |
| **NIR (0.9–2.5 μm)** | 2MASS / VHS / VISTA | ~17–20 mag (2MASS) | 2–3" | Stellar mass tracer | 2MASS shallow; need ground-based follow-up |
| **MIR (3–25 μm)** | Spitzer IRAC/MIPS | ~22 μJy (IRAC) | 1.2–2" | AGN diagnostics; warm dust | Limited depth; Spitzer aging |
| **MIR (5–25 μm)** | JWST MIRI | ~0.1 mJy (5 σ) | 0.1–0.8" | Revolutionary; new capability | Program-dependent availability |
| **FIR (70–500 μm)** | Herschel PACS/SPIRE | ~5 mJy (PACS) | 6–35" | Bolometric SFR direct | Confusion-limited; Herschel ended 2013 |
| **Radio 1.4 GHz** | VLA Sky Survey | ~20 μJy | 2.5" | Extinction-free SFR | High confusion at small scales |

### 1.2 Redshift-Dependent Bandpass Considerations

**For z = 0–1 Galaxies:**
```
Preferred filters:
  - UV: GALEX FUV (1500 Å rest → 3000 Å observed at z=1)
  - Optical: SDSS u, g, r, i (rest-frame optical colors preserved)
  - NIR: K-band (rest-frame ~2.2 μm, M/L relation well-calibrated)
  - MIR: Spitzer IRAC (warm dust; stellar continuum)
  - FIR: Herschel (bolometric)

Avoidance:
  - GALEX NUV too long (rest ~2300 Å → very faint at z=1)
```

**For z = 2–3 Galaxies (Cosmic Noon):**
```
Preferred filters:
  - UV: Ground-based optical u-band (rest ~1500 Å observed → 6000 Å at z=3, so use optical z-band rest → 4000 Å observed)
  - Optical: CFHTLS u*g'r'i'z' multi-band (rest-frame optical colors)
  - NIR: JWST NIRCam 1–5 μm or ground-based K (rest-frame 200–500 nm to 2 μm)
  - MIR: Spitzer IRAC 3.6–8 μm (rest-frame ~0.7–1.6 μm; stellar continuum + PAHs)
  - FIR: Herschel or ALMA (rest-frame ~16–100 μm dust emission)

Critical:
  - Include FIR to constrain dust attenuation and SFR robustly
  - K-band for stellar mass; avoid optical-only M_* (degeneracy with dust)
```

**For z = 4–6 Galaxies (Early Universe):**
```
Preferred filters:
  - Rest-frame UV: JWST NIRCam 0.6–2.0 μm observed (rest 0.12–0.4 μm at z=4–6)
  - Rest-frame optical: JWST NIRCam 3–5 μm observed (rest 0.6–1.0 μm)
  - Dust continuum: ALMA Band 6/7 (250–350 GHz observed, rest-frame ~160 μm)
  - Spectroscopy: JWST NIRSpec 0.6–5 μm (Hα, [OIII], etc. at rest-frame optical)

Avoidance:
  - Ground-based optical (too faint; low signal-to-noise)
  - Old Spitzer data (limited depth for z>4)

Best practice:
  - Combine JWST photometry + ALMA continuum + NIRSpec spectrum → maximize constraints
```

**For z ≥ 7 Galaxies (Reionization Era):**
```
Mandatory:
  - JWST NIRCam (only option for rest-frame UV/optical)
  - JWST NIRSpec if spectroscopy needed (emission lines)

Optional but valuable:
  - ALMA Band 6 for dust continuum (if SFR > 100 M_⊙/yr to be detectable)
  - Multiple NIRCam bands to sample UV slope (β determination)

Warning:
  - FIR data sparse; stacking analysis may be necessary
  - Radio data difficult; VLA confusion-limited
```

---

## 2. SED FITTING PROTOCOL

### 2.1 Pre-Fitting Data Preparation

**Step 1: Photometry Quality Checks**
```
For each galaxy, for each photometric band:
  a) Verify flux ≥ 3σ detection; flag upper limits
  b) Check for nearby contaminants (other sources blended in beam)
  c) Cross-reference multiple surveys for consistency
     → If inconsistency > 0.2 dex: investigate aperture size, PSF issues
  d) Apply Galactic extinction corrections (Schlegel et al. 1998 dust maps)
     Typical A_V ~ 0.05–0.1 mag for high-latitude fields
  e) Assign photometric errors:
     Conservative: 10% if official quoted error < 5%
     Realistic: use official errors if > 5%
```

**Step 2: Redshift Assignment**
```
Option A (Spectroscopic redshift, preferred):
  z_spec from emission lines (e.g., Hα, [OIII]) or absorption (Lyman break)
  Typical accuracy: Δz ~ 0.0005 (excellent)
  Action: FREEZE redshift in SED fitting

Option B (Photometric redshift, if z_spec unavailable):
  Estimate from broad SED shape (e.g., 4000 Å break)
  Typical accuracy: Δz ~ 0.05 × (1+z) (0.05 for z~0, 0.25 for z~4)
  Action: Let z float ± 0.1 range around photo-z estimate
  Caution: Can introduce systematic biases in age/dust if z poorly constrained
```

**Step 3: Outlier Removal**
```
For each galaxy:
  a) Check for infrared excess (flux in FIR bands unexpectedly high)
     → May indicate AGN, unusual dust, or measurement error
  b) Check UV excess (UV flux high relative to optical continuum)
     → Possible indicator of recent starburst or anomalous SED
  c) Remove bands where S/N < 2σ (or flag as upper limits)
  d) Verify no duplicate observations from different surveys
```

### 2.2 SED Fitting Workflow (Using CIGALE as Example)

**Configuration File Setup:**
```
[SFR configurations]
star_formation_histories:
  - delayed  (linear rise → exponential decline; age, τ parameter)
  - constant (flat SFR)
  - burst    (Gaussian spike; age, peak age, duration)

[Stellar population synthesis]
sps_library: bc03  (or 'starburst99', 'fsps', etc.)
imf: chabrier  (0.1–100 M_⊙ range)
metallicity: solar  (or array of [0.005, 0.02, 0.04])

[Dust attenuation]
model: charlot_fall
delta_ISM: [-0.5, -0.7, -1.0]  (ISM attenuation slope)
delta_BC:  [-0.5, -0.7, -1.0]  (birth cloud slope)
E_BV_ISM: [0.0–0.5]  (color excess, ISM)
E_BV_BC:  [0.0–2.0]  (color excess, birth clouds)

[Dust emission]
dust_type: modified_blackbody
T_min: 20 K, T_max: 80 K
beta: 1.8  (emissivity; fixed)

[Nebular emission]
nebular_continuum: yes
nebular_lines: yes  (recommended)

[Redshift]
redshift: [z_spec]  or [z_photo_min:z_photo_max:Δz]
```

**Fitting Procedure:**
```
1. Generate model library:
   - Iterate through all parameter combinations (SFH, age, Z, E_BV, T_dust)
   - Calculate spectral energy distribution for each combination
   - Convolve with observed filter transmission curves
   - Normalize to observed photometry

2. χ² Minimization:
   χ² = Σ_i [(f_obs,i - f_model,i) / σ_i]^2
   where:
     f_obs,i = observed flux in band i
     f_model,i = model prediction (before adding photometric errors)
     σ_i = photometric uncertainty

3. Posterior extraction:
   Extract PDF (probability distribution function) for each parameter
   Report: median + 68% confidence interval (±1σ equivalent)

4. Quality assessment:
   a) Reduced χ² should be ~1 (if errors realistic)
      If χ² >> 1: underestimated errors or poor model
      If χ² << 1: overestimated errors or redundant bands
   b) Visual inspection: overlay best-fit SED on observed photometry
      Check for systematic misfits (residuals not randomly scattered)
   c) Parameter correlations: examine if (age, A_V) degenerate
      (Common; increases uncertainty)
```

### 2.3 Post-Fitting Analysis

**Parameter Validation:**
```
a) Stellar Mass: M_*
   Expected range: 10^8–10^12 M_⊙ (galaxies span ~4 orders of magnitude)
   Red flag if:
     - M_* > 10^13 M_⊙ (unrealistic; check SPS model normalization)
     - Uncertainty > 0.3 dex (indicates degeneracy; add data)

b) Star Formation Rate: SFR
   Expected range: 0.1–1000 M_⊙/yr (depends on galaxy selection)
   Cross-check:
     - SFR vs. UV SFR (if available): should agree ±0.2 dex
     - SFR vs. Hα SFR (if available): should agree ±0.15 dex
     - SFR vs. radio SFR (if available): should agree ±0.3 dex (timescale caveat)

c) Dust Attenuation: A_V
   Expected range: 0.0–3.0 mag (rarely > 3 mag in SFGs)
   Red flag if:
     - A_V > 3 mag: likely obscured AGN, not SFG
     - A_V = 0 ± 0.1 mag: verify no dust by independent method (IRX?)

d) Age: t_age
   Expected range:
     - z~0 SFGs: 0.1–10 Gyr (broad; age-dust degeneracy)
     - z~2 SFGs: 0.1–1 Gyr (typically younger; higher sSFR)
     - z>4 SFGs: 0.01–0.5 Gyr (younger still; universe not old)
   Red flag if:
     - Age > age_of_universe at given z (impossible)
     - Age uncertainty factor > 10 for young systems (degenerate with dust)

e) Metallicity: Z
   Expected range: 0.1–2 Z_⊙ (most galaxies)
   Red flag if:
     - Z > 3 Z_⊙: possible in starburst nuclei, rare in large apertures
     - Z < 0.01 Z_⊙: dwarf/low-metallicity systems; check emission lines
```

**SED Residual Analysis:**
```
Calculate residuals: r_i = (f_obs,i - f_model,i) / σ_i

Plot residuals vs. wavelength:
  - Smooth, random scatter: good model
  - Systematic pattern (e.g., "S" shape in residuals): inadequate SPS model or dust law
  - Large outliers (|r_i| > 3σ): investigate measurement error or data quality

Common issues:
  1) Steep UV residuals: may indicate nebular continuum omission
  2) FIR residuals: multi-component dust or temperature variation unmodeled
  3) Optical residuals: AGN contamination or unusual stellar population
```

---

## 3. DUST ATTENUATION MEASUREMENT PROTOCOLS

### 3.1 Balmer Decrement Method (z < 0.5 or JWST Spectroscopy at z > 4)

**Observational Requirements:**
```
Spectroscopic observation of Hα (6563 Å) and Hβ (4861 Å) emission lines
Signal-to-noise ratio (S/N) ≥ 5 for both lines (to measure flux ratio to ±20%)
Wavelength calibration accuracy ± 1 Å (ensures correct line identification)
```

**Data Reduction:**
```
Step 1: Emission Line Extraction
  a) Fit continuum under each line (polynomial fit to flux adjacent to line)
  b) Subtract continuum; measure net line flux = integral of (f_line − f_continuum)
  c) Correct for stellar absorption: account for Balmer absorption from continuum stars
     (Typically ~10–20% correction for young populations; ~30–50% for old populations)
  d) Measure uncertainty: propagate flux measurement noise + continuum subtraction uncertainty

Step 2: Balmer Decrement Calculation
  Observed ratio: R_obs = F_Hα / F_Hβ
  Intrinsic ratio: R_int = 2.86 (Case B; T_e ~ 10,000 K)

  Balmer decrement (reddening): C = R_obs / R_int

  Note: If R_obs < R_int, set C = 0 (no attenuation; possible if error large)

Step 3: Attenuation Derivation
  Using Calzetti attenuation law:
    A_Hα / A_5500Å = 1.165
    A_Hβ / A_5500Å = 1.196

  Attenuation difference:
    A_Hα − A_Hβ = 0.4 × log10(C) × (A_Hα/A_5500Å − A_Hβ/A_5500Å)

  Solving:
    A_Hα − A_Hβ = 0.4 × log10(C) × 0.031
    0.4 × [−2.156 × (1/1500−1/1500) + ...] × (1500 Å to optical range scaling)

  Simplified (standard formula):
    A_V = [log10(R_obs / R_int)] / [f_Hα − f_Hβ]

  where f_Hα, f_Hβ from Calzetti curve at 6563, 4861 Å respectively.
  Numerically: A_V ≈ 2.36 × [log10(R_obs / 2.86)]  (using Calzetti)
```

**Uncertainty Propagation:**
```
σ(A_V) depends on flux measurement errors:
  σ(A_V) = 2.36 / ln(10) × σ(log R_obs)
  σ(log R_obs) = sqrt[(σ_Hα / F_Hα)^2 + (σ_Hβ / F_Hβ)^2]

Typical S/N = 10 for each line:
  σ_Hα / F_Hα ≈ 0.1, σ_Hβ / F_Hβ ≈ 0.1
  σ(log R_obs) ≈ 0.07
  σ(A_V) ≈ 0.16 mag
```

**Validation:**
```
Check: Compare A_V from Balmer decrement to SED fitting result
  Should agree within ±0.2 mag if both methods valid
  If discrepancy > 0.3 mag: investigate dust geometry (possible clumpy media)
```

### 3.2 Infrared Excess (IRX) Method

**Requirements:**
```
UV flux measurement at rest-frame ~1500 Å (or nearby)
Bolometric IR luminosity L_IR (8–1000 μm, via FIR photometry or SED fitting)
```

**Calculation:**
```
Step 1: Measure Rest-Frame UV Luminosity
  L_UV = 4π d_L^2 × f_ν(1500 Å)  (distance luminosity, observed flux)
  or directly from SED model at 1500 Å wavelength

Step 2: Compute Infrared Excess
  IRX = L_IR / L_UV

  Example:
    L_IR = 10^11 L_⊙
    L_UV = 10^10 L_⊙
    IRX = 10  (or log(IRX) = 1.0)

Step 3: Estimate Dust Attenuation from IRX–β Relation
  Using empirical relation (Meurer et al. 1999):
    log(IRX) = 0.68 × β + log(C)

  where C ~ 0.5 is normalization

  Solve for A_V via dust attenuation law:
    If knowing IRX and β, can invert to A_V
    (Requires assumption of attenuation curve shape)

  Approximate conversion:
    A_V(Calzetti) ≈ 2.5 × log(IRX) if using Calzetti law
    (Rough; depends on intrinsic UV slope)
```

**Uncertainties:**
```
σ(IRX) dominates due to FIR measurement errors:
  σ(L_IR) / L_IR ~ 0.15–0.3 (typical FIR photometry)
  σ(L_UV) / L_UV ~ 0.05–0.1 (UV well-measured)

  Result: σ(IRX) ~ 0.2 dex (factor ~1.6 uncertainty in IRX)
           → σ(A_V) ~ ±0.3 mag (larger than Balmer decrement)

Advantage of IRX: Extinction-free measurement of attenuation
Disadvantage: Requires FIR data (not available at high-z in early JWST observations)
```

### 3.3 UV Continuum Slope Method

**Principle:**
```
Dust reddening steepens UV continuum slope β (makes it more negative):
  Intrinsic β ~ −2 to −2.5 (young stellar populations)
  Observed β can range −0.5 to −3 (depending on dust + age)

Relationship:
  A_V ≈ 10 × [−0.4 × (β_obs − β_intrinsic) / A(1500 Å)]

  where A(1500 Å) is reddening per magnitude attenuation
```

**Measurement:**
```
Step 1: Measure UV Continuum Slope
  β = d(log F_λ) / d(log λ) ≈ slope of log-log plot of flux vs. wavelength

  Practical: Measure flux at λ1 = 1500 Å and λ2 = 2800 Å (commonly available)
    β ≈ [log(F_1500) − log(F_2800)] / log(λ_2800 / λ_1500)
    β ≈ [log(F_1500) − log(F_2800)] / log(1.867)
    β ≈ 1.86 × [log(F_1500 / F_2800)]

Step 2: Compare to Intrinsic β
  Stellar population synthesis predicts β_intrinsic based on age, Z
  Example (age 100 Myr, solar Z):
    β_intrinsic ≈ −2.2

Step 3: Estimate Dust Attenuation
  Δβ = β_obs − β_intrinsic = reddening effect
  A_V estimate from empirical calibrations

  Rough relation (Meurer et al. 1999):
    A_V ≈ 3.5 × |Δβ|  (very approximate)
```

**Caveats:**
```
1) Intrinsic β depends on age, metallicity, IMF (uncertainties ±0.2–0.3 dex)
2) Dust curve shape affects relationship (not universal)
3) Clumpy dust geometry can produce blue slopes despite high attenuation
4) Younger populations have steeper (more negative) intrinsic β

Recommendation:
  Use UV slope as SECONDARY check on attenuation, not primary indicator
  Combine with Balmer decrement or IRX for robust A_V estimate
```

---

## 4. STAR FORMATION RATE DETERMINATION MATRIX

### 4.1 Decision Tree: Which SFR Indicator to Use?

```
START: Galaxy properties available?

├─ Have Hα line flux (z < 0.5 or JWST)?
│  ├─ YES → Use Hα + Balmer decrement (if Hβ available)
│  │        SFR_Hα = 9.7×10^-42 × L_Hα(erg/s) [M_⊙/yr]
│  │        UNCERTAINTY: ±0.15 dex (if Balmer decrement corrects dust)
│  │
│  └─ NO → Continue

├─ Have FIR data (Herschel, ALMA, or SED L_IR)?
│  ├─ YES → Use TIR SFR
│  │        SFR_TIR = 1.0×10^-10 × L_IR(L_⊙) [M_⊙/yr]
│  │        UNCERTAINTY: ±0.2 dex (BEST OPTION if available)
│  │
│  └─ NO → Continue

├─ Have 24 μm flux (Spitzer or JWST/MIRI)?
│  ├─ YES → Use 24 μm SFR
│  │        Check metallicity: is Z > 0.3 Z_⊙?
│  │        ├─ YES (sufficient metallicity) → Apply monochromatic calibration
│  │        │        SFR_24 = 9.2×10^-12 × L_24(W) [M_⊙/yr]
│  │        │        UNCERTAINTY: ±0.25 dex
│  │        │        WARNING: Check for AGN (PAH EW)
│  │        │
│  │        └─ NO (low metallicity) → Avoid; use alternative
│  │
│  └─ NO → Continue

├─ Have 1.4 GHz radio data?
│  ├─ YES → SFR_radio = 4.6×10^-29 × L_1.4GHz(W/Hz) [M_⊙/yr]
│  │        UNCERTAINTY: ±0.3 dex
│  │        CAVEAT: ~150 Myr timescale; underestimates if SFR declining
│  │        USE AS: Cross-check / confirmation (not primary)
│  │
│  └─ NO → Continue

├─ Have UV flux (GALEX, optical u-band, or rest-UV from SED)?
│  ├─ YES → Dust-corrected UV SFR
│  │        Step 1: Measure L_UV at 1500 Å
│  │        Step 2: Estimate A_V from dust models (SED fitting, IRX)
│  │        Step 3: Correct UV luminosity for attenuation
│  │        Step 4: Apply Kennicutt calibration
│  │                SFR_UV,corrected = 1.4×10^-28 × L_UV,intrinsic(erg/s/Å) [M_⊙/yr]
│  │        UNCERTAINTY: ±0.3–0.5 dex (depends on dust correction method)
│  │        WARNING: Large uncertainty; use with other indicators
│  │
│  └─ NO → INSUFFICIENT DATA

END: Calculate SFR uncertainty
     If single indicator: stated uncertainty ± systematic
     If multiple indicators: AVERAGE weighted by inverse uncertainties
                           Report agreement/disagreement
```

### 4.2 Multi-Indicator SFR Averaging

**When Multiple Indicators Available:**

```
SFR_best = weighted average of individual SFR estimates

Weight for each indicator: w_i = 1 / σ_i^2
where σ_i is uncertainty (dex) for indicator i

Example:
  SFR_Hα = 50 M_⊙/yr, σ_Hα = 0.15 dex
  SFR_TIR = 80 M_⊙/yr, σ_TIR = 0.2 dex

  w_Hα = 1/(0.15)^2 = 44.4
  w_TIR = 1/(0.2)^2 = 25

  log(SFR_best) = [44.4 × log(50) + 25 × log(80)] / (44.4 + 25)
                = [44.4 × 1.699 + 25 × 1.903] / 69.4
                = (75.4 + 47.6) / 69.4 = 1.779
  SFR_best = 10^1.779 ≈ 60 M_⊙/yr

  σ_best = 1 / sqrt(44.4 + 25) = 1 / sqrt(69.4) = 0.12 dex
  Final SFR: 60 ± 0.12 dex M_⊙/yr (or 60_{−7}^{+7} M_⊙/yr in linear form)

Sanity check:
  If individual SFR estimates disagree > 0.3 dex, investigate cause:
  - Measurement error?
  - Dust properties anomalous (e.g., very clumpy)?
  - AGN contamination of MIR/radio?
  - SFR variability (burst vs. average)?
```

---

## 5. QUALITY ASSURANCE AND VALIDATION PROTOCOLS

### 5.1 Consistency Checks

**Check 1: Specific SFR (sSFR)**
```
sSFR = SFR / M_*

Expected ranges:
  Local main sequence: sSFR ~ 10^{-11} yr^-1
  z ~ 2 main sequence: sSFR ~ 10^{-10} yr^-1
  Starburst: sSFR > 10^{-9} yr^-1

If sSFR falls outside expected range:
  ├─ sSFR > 10^{-8} yr^-1 (extreme starburst)
  │  → Likely genuine; check for youth, recent merger
  │
  └─ sSFR < 10^{-12} yr^-1 (quiescent)
     → Likely contaminated; check for AGN, old stellar population

Red flag if sSFR varies by > 0.5 dex between SED model and alternative measurements
```

**Check 2: Dust-to-Stellar Mass Ratio**
```
M_dust / M_* expected ranges:
  z ~ 0: 10^{-3} to 10^{-2} (typically ~0.01)
  z ~ 2: 10^{-2} to 10^{-1} (higher dust content)
  z > 4: 10^{-3} to 10^{-2} (lower dust; less metal-enriched)

If M_dust / M_* > 0.5:
  → Possible error in dust temperature (too cold assumed → M_dust overestimated)
  → Or actual young, dust-rich starburst

If M_dust / M_* < 10^{-4}:
  → Unrealistically low; check FIR measurement and dust model
```

**Check 3: Age vs. Specific SFR Consistency**
```
Old star formation (age > 1 Gyr):
  sSFR should be low (< 10^{-11} yr^-1)
  If high sSFR predicted, age–dust degeneracy likely
  → Solution: Add spectroscopic data or longer-wavelength imaging

Young burst (age < 100 Myr):
  sSFR should be high (> 10^{-10} yr^-1)
  If low sSFR predicted, check dust attenuation
  → Very dusty bursts can appear old/quiescent
```

### 5.2 Comparison to Independent Measurements

**If Available: Compare to:**
1. **Emission line SFR** (Hα, [OII])
2. **Radio SFR** (1.4 GHz)
3. **Mid-infrared SFR** (24 μm)
4. **UV continuum SFR** (1500 Å, dust-corrected)

**Acceptance Criterion:**
```
If all indicators agree within ±0.3 dex:
  → PASS; report weighted average as best estimate

If scatter > 0.3 dex:
  → INVESTIGATE
  ├─ Check for AGN contamination
  ├─ Verify dust attenuation is reasonable
  ├─ Examine SED fit quality (residuals)
  ├─ Consider SFR variability (recent starburst vs. average)
  └─ Document outliers; discuss in paper

If one indicator strongly deviates (> 0.5 dex from others):
  → FLAG as potential systematic
  → Exclude if clear measurement error
  → Include with caution if systematic source unknown
```

---

## 6. COMMON PITFALLS AND HOW TO AVOID THEM

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **Missing MIR/FIR data** | Large SFR uncertainty (±0.4 dex) | Add ALMA obs or use better photometry |
| **Dust attenuation undefined** | Age and M_* range wildly | Measure Balmer decrement or IRX; add spectroscopy |
| **AGN contamination in MIR** | 24 μm flux anomalously high | Check [NeII]/[NeIII] line ratio; use PAH EW |
| **Photometric errors underestimated** | χ² >> 1 in SED fit | Inflate errors to 10–15%; repeat fitting |
| **Model SEDs not sampling parameter space** | Best-fit seems reasonable but uncertain | Expand SFH/metallicity/dust parameter grids |
| **Confusion with rest-frame vs. observed frame** | Using observed photometry directly | Always convert to rest-frame or account for k-correction |
| **Stellar mass bugged by young age + high dust** | Huge M_* uncertainty | Add spectroscopy to break age-dust degeneracy |
| **SFR varies wildly between indicators** | SFR_UV ≠ SFR_IR by factor 5+ | Check dust attenuation careful; inspect SED residuals |
| **Starburst99 at z > 4** | Crashes or warnings about redshift | Use BC03 or FSPS instead; Starburst99 optimized for z~0 |
| **Forgetting dust energy balance** | SED doesn't conserve energy | Use code enforcing balance (CIGALE); avoid BC03 without careful dust treatment |

---

## 7. RECOMMENDED PRACTICE WORKFLOWS

### 7.1 Typical High-z SFG Study (Minimal Resources)

```
Available:
  - JWST NIRCam photometry (2–4 bands)
  - ALMA continuum (1–2 bands, 250–350 GHz)
  - Spectroscopic redshift (JWST NIRSpec or other)

Workflow:
1. Collect photometry (quality control, errors)
2. Run CIGALE with:
   - SFH: delayed-tau
   - SPS: BC03
   - Dust: Charlot & Fall, A_V ∈ [0, 2] mag
   - Nebular: on

3. Extract: M_*, SFR, A_V, age
4. Compare to:
   - ALMA dust continuum → check L_IR consistency
   - Literature values (if available)

5. Report:
   - Best-fit parameters + 68% confidence intervals
   - Reduced χ²
   - Justification for any outliers

Expected time: 1–2 hours per galaxy (once photometry assembled)
```

### 7.2 Precision Study (Full Multi-wavelength)

```
Available:
  - GALEX UV
  - Ground-based optical (u, g, r, i, z bands)
  - Ground-based NIR (J, H, K)
  - Spitzer IRAC (3.6, 4.5, 5.8, 8.0 μm)
  - Herschel PACS/SPIRE (70, 100, 160, 250, 350, 500 μm)
  - JWST spectroscopy (if z < 7)
  - VLA 1.4 GHz radio (if detected)

Workflow:
1. Assemble photometry; cross-reference for consistency
2. If spectroscopy available:
   - Extract emission lines (Hα, [OIII], [OII], Hβ)
   - Measure Balmer decrement for A_V
   - Secure redshift

3. Run SED fitting:
   - SFH: multiple (constant, delayed, burst)
   - SPS: BC03 + FSPS (compare)
   - Dust: Calzetti + Charlot & Fall (compare)
   - Metallicity: fixed (from lines) or fitted (0.2–2 Z_⊙)

4. Extract physical parameters:
   - M_*, SFR, A_V from main model
   - Compute SFR from multiple indicators (Hα, TIR, UV corrected, radio)
   - Compare; report spread

5. Residual analysis:
   - Plot observed vs. model
   - Inspect residuals; look for systematic patterns
   - Iterate if needed (change model, check data quality)

6. Report:
   - Full parameter PDFs (median + 68% CI)
   - All SFR estimates + final value
   - Comparison to literature
   - Full transparency on assumptions (SPS, dust law, IMF)

Expected time: 3–5 hours per galaxy (including iteration and write-up)
```

---

## 8. DATA ARCHIVING AND DOCUMENTATION

### 8.1 Required Documentation Per Galaxy

For reproducibility, archive:
```
1. PHOTOMETRY TABLE:
   - All bands measured (λ, S_ν, σ_S_ν)
   - Galactic extinction applied? (A_V^gal value)
   - Flags: detection, upper limit, flagged bad, etc.
   - Source of data (survey name, reference)

2. SED FITTING RESULTS:
   - Best-fit parameters: M_*, SFR, A_V, age, Z, T_d
   - 68% confidence intervals (or full posterior samples)
   - χ² and reduced χ²
   - SED model used (code, SPS library, dust law)
   - Residuals (obs - model) / σ for each band

3. INDEPENDENT SFR MEASUREMENTS:
   - SFR_UV (if applicable): formula, dust correction applied
   - SFR_Hα (if applicable): Balmer decrement, dust correction
   - SFR_24 (if applicable): calibration used
   - SFR_TIR: L_IR derivation method
   - SFR_radio (if applicable): formula, timescale notes

4. NOTES:
   - Redshift source (spectroscopic vs. photometric)
   - Known issues (AGN? unusual dust? merger?)
   - Comparison to literature values (if available)
```

### 8.2 File Organization

```
Project/
  ├── data/
  │   ├── photometry/
  │   │   ├── source_ID_photometry.fits
  │   │   └── photometry_catalog.fits
  │   ├── spectra/ (if applicable)
  │   │   └── source_ID_spec.fits
  │   └── raw_images/ (for quality control)
  │
  ├── sed_fitting/
  │   ├── cigale_config.cfg
  │   ├── sed_best_fits/
  │   │   └── source_ID_SED_results.txt
  │   └── sed_posteriors/ (optional; chain files)
  │
  ├── results/
  │   ├── physical_parameters_table.csv
  │   ├── sfr_comparison_table.csv
  │   └── plots/
  │       ├── sed_1.pdf, sed_2.pdf, ...
  │       └── parameter_distributions.pdf
  │
  └── manuscript/
      ├── paper.tex
      ├── tables.tex
      └── figures/ (publication-quality plots)
```

---

## 9. REFERENCE IMPLEMENTATIONS

### 9.1 Pseudo-Code: SED Fitting Main Loop

```python
import numpy as np
from scipy.optimize import minimize
import stellar_population_synthesis as sps
import dust_models

# Load galaxy data
photometry = load_photometry(galaxy_id)
redshift = photometry['z_spec']

# Initialize parameter grid
params = {
    'age': np.linspace(0.01, 13.8, 50),  # Gyr
    'sfh_tau': np.linspace(0.1, 10, 30),  # Gyr (for delayed-tau)
    'metallicity': np.array([0.005, 0.02, 0.04]),  # Z_⊙
    'A_V_ISM': np.linspace(0, 1.5, 20),
    'A_V_BC': np.linspace(0, 2.0, 20),
    'T_dust': np.linspace(20, 80, 15),  # K
}

# Generate SED templates
models = []
for age in params['age']:
    for tau in params['sfh_tau']:
        for Z in params['metallicity']:
            for A_V_ISM in params['A_V_ISM']:
                for A_V_BC in params['A_V_BC']:
                    for T_d in params['T_dust']:

                        # SED computation
                        sed_rest = sps.bc03(age=age, sfh='delayed-tau', tau=tau, Z=Z)
                        sed_attenued = dust_models.apply_attenuation(
                            sed_rest, A_V_ISM, A_V_BC, curve='charlot_fall'
                        )

                        # Dust emission
                        sed_total = sed_attenued + dust_models.dust_emission_bb(T_d, ...)

                        # Convolve with filters
                        model_fluxes = convolve_with_filters(sed_total, photometry['filter_list'])

                        models.append({
                            'parameters': (age, tau, Z, A_V_ISM, A_V_BC, T_d),
                            'model_fluxes': model_fluxes,
                            'observed_fluxes': photometry['fluxes'],
                            'flux_errors': photometry['flux_errors']
                        })

# Fit: χ² minimization
chi2_grid = []
for model in models:
    chi2 = np.sum(((model['observed_fluxes'] - model['model_fluxes'])
                   / model['flux_errors'])**2)
    chi2_grid.append(chi2)

# Extract best fit and posteriors
chi2_array = np.array(chi2_grid)
best_idx = np.argmin(chi2_array)
best_params = models[best_idx]['parameters']

# Posterior: convert χ² to probability
prob = np.exp(-chi2_array / 2)
prob /= np.sum(prob)  # Normalize

# Parameter PDFs: marginalize
age_posterior = integrate_probability_over(prob, 'age')
sfr_posterior = compute_sfr_from_posterior(prob, models)

# Report
print(f"Best-fit age: {best_params[0]:.2f} Gyr")
print(f"Best-fit SFR: {sfr_from_params(best_params):.1f} M_sun/yr")
print(f"16-84 percentile SFR: {percentile(sfr_posterior, 16)} – {percentile(sfr_posterior, 84)}")
```

---

## 10. TROUBLESHOOTING GUIDE

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| χ² > 3 (poor fit) | Photometry error overestimated | Reduce error estimates; check data quality |
| χ² << 0.1 (too good) | Photometry error underestimated | Increase errors to 10–15% |
| Huge parameter uncertainties (factor 100+) | Degenerate model space | Add spectroscopic data; freeze some parameters |
| Best-fit age = 13.8 Gyr (maximum) | Grid boundary reached | Extend age grid |
| SFR = 0 or undefined | No star formation in model | Check A_V; may be quiescent |
| M_* jumps by factor 10 between runs | Numerical instability | Check filter transmission curves; rebuild library |
| SED residuals show "S" shape | SPS model inadequate | Switch to FSPS; include nebular continuum |
| Radio SFR factor 2 higher than TIR | Cosmic ray aging long | OK; ~150 Myr timescale; document |
| Balmer decrement undefined (R < 2.86) | No dust or measurement error | Check line S/N; verify continuum subtraction |

---

*End of Methodology Protocols Document*
*December 2025*
