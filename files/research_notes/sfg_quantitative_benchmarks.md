# Star-Forming Galaxies: Quantitative Benchmarks and Data Tables

**Compiled:** December 2025

---

## 1. STAR FORMATION RATE CALIBRATIONS (DETAILED)

### 1.1 UV-Based SFR (Rest-Frame 1500 Å)

**Formula:**
```
SFR(M_⊙/yr) = 1.4 × 10^{−28} × L_1500 (erg/s/Å)
            = 1.4 × 10^{−43} × L_1500 (W/Å)
```

**Source:** Kennicutt (1998, ARA&A 36, 189)

**Key Parameters:**
- Assumes IMF range 0.1–100 M_⊙ (Salpeter-like)
- Constant SFR for age > 100 Myr (older than O-star lifetime)
- Zero dust attenuation (unattenuated UV)

**Typical Dispersion:** ±0.3–0.4 dex

**Application Range:** Low-z UV-selected galaxies; must correct for dust via SED fitting or other methods for dusty systems

**Critical Note:** Requires dust-attenuation-corrected UV luminosity; uncorrected UV severely underestimates SFR in dusty systems (factors of 2–10 error)

---

### 1.2 Hα-Based SFR (Rest-Frame 6563 Å Emission Line)

**Formula:**
```
SFR(M_⊙/yr) = 9.7 × 10^{−42} × L_Hα (erg/s)
            = 1.09 × 10^{−11} M_⊙/yr / (erg/s) × L_Hα
```

**Source:** Kennicutt (1998)

**Key Parameters:**
- Ionizing photon rate ∝ L_Hα from young massive stars
- Recombination dominated by hydrogen II regions
- Requires Case B recombination (T_e ~ 10^4 K, n_e ~ 10^2–10^4 cm^{−3})

**Dust Correction:**
```
A_Hα = 1.165 × A_V  (optical depth scaling)

F_Hα_obs = F_Hα_intrinsic × 10^{−0.4 × A_Hα}

Balmer Decrement: C = (I_Hα/I_Hβ)_intrinsic / (F_Hα/F_Hβ)_obs
                    ≈ 2.86 / (F_Hα/F_Hβ)_obs

A_V = C / 0.4 (Calzetti reddening)
```

**Intrinsic Hα/Hβ Ratio:**
- Case B (T_e = 10,000 K): 2.86 (weak T_e dependence ±0.05)
- Standard assumption in literature

**Typical Dispersion:** ±0.15 dex (when dust-corrected via Balmer decrement)

**Application Range:** z < 0.5 (Hα in optical spectra); z = 4–7 with JWST NIRSpec (rest-frame optical)

**Recent Calibration (PHANGS-MUSE 2023):**
- Applied Balmer decrements on 100 pc scales in 19 nearby galaxies
- Combined with 24 μm photometry and UV observations
- Result: Hybrid SFR recipe achieving <0.16 dex scatter
- Formula (resolved scales):
  ```
  SFR = 0.68 × [L(Hα_dust_corrected) + L(24μm)] / (erg/s)  [M_⊙/yr]
  ```

---

### 1.3 Mid-Infrared (24 μm, MIPS) SFR

**Formula (Monochromatic):**
```
SFR(M_⊙/yr) = 9.2 × 10^{−12} × L_24 (W)
            = 0.92 × 10^{−11} M_⊙/yr × L_24 (W)
```

**Source:** Rieke et al. (2009, ApJ 692, 556)

**Key Physics:**
- 24 μm emitted by warm dust (T ~ 70–100 K) heated by star formation
- Primarily arising from dust associated with birth clouds of young stars
- Age weighting favors recent star formation (~100 Myr)

**Metallicity Dependence:**
- **Calibration valid for:** 12+log(O/H) ≥ 8.1 (Z ≥ 0.3 Z_⊙)
- **At lower metallicity:** Dust content drops; 24 μm flux becomes unreliable SFR tracer
- **Correction factor:** None published; avoid use at Z < 0.3 Z_⊙

**Systematic Uncertainties:**
- Dust temperature variation (48–80 K range) introduces ~0.15 dex uncertainty
- AGN contribution can boost 24 μm by factor 2–5; requires diagnostic separation

**AGN Diagnostic (PAH Equivalence Width):**
```
EW_6.2μm (6.2 μm PAH) < 0.2 μm  →  AGN-dominated
EW_6.2μm > 0.2 μm              →  > 50% flux from star formation
```

**Typical Dispersion:** ±0.25 dex (more scattered than Hα due to dust temperature variation and AGN contamination)

**Wavelength-Dependent Scatter:**
- 24 μm: ±25% (factor ~2)
- 70 μm: ±50% (factor ~2.5)
- 160 μm: ±60% (factor ~4)

Longer wavelengths show larger scatter due to increased contribution from older dust populations not tied to current star formation.

---

### 1.4 Total Infrared (TIR) SFR (8–1000 μm)

**Formula:**
```
SFR(M_⊙/yr) = 1.0 × 10^{−10} × L_IR (L_⊙)
            = 1.09 × 10^{−12} × L_IR (erg/s)
```

**Source:** Kennicutt & Evans (2012, ARA&A 50, 531)

**Key Physics:**
- L_IR = total bolometric energy reprocessed by dust
- Age-weighted over ~100 Myr (dust equilibration timescale)
- Metallicity-independent above Z ~ 0.3 Z_⊙ (sufficient dust opacity)

**Determination of L_IR:**

**Method 1: Integration from Observed SEDs**
```
L_IR = ∫_{8μm}^{1000μm} λ F_λ d(log λ)
```
Requires multi-band FIR photometry (e.g., Herschel PACS 70/100/160 + SPIRE 250/350/500 μm).

**Method 2: Modified Blackbody Template Fit**
```
L_IR = ∫ M_dust × (ν/ν_0)^β × B_ν(T_d) dν

Typical parameters:
  β (emissivity index) ~ 1.5–2.0
  T_d (dust temperature) ~ 30–50 K (varies; single-component approximation)
```

**Method 3: Flux at Single Far-IR Band + Interpolation**
```
L_IR ≈ C × L_250μm  (for high-z; correction factor C ~ 2–3 depending on redshift and SED shape)
```

**Typical Dispersion:** ±0.2 dex (most robust; dust-free measurement of energy balance)

**Key Advantage:** Dust-free SFR; independent of dust temperature assumptions (energy is conserved)

**Temperature Degeneracy Caveat:**
If T_d unknown (single FIR band only), M_dust ∝ T_d^{−3}, so M_dust highly uncertain; L_IR still valid if determined via integration or bolometric correction.

---

### 1.5 Radio Continuum (1.4 GHz) SFR

**Formula:**
```
SFR(M_⊙/yr) = 4.6 × 10^{−29} × L_1.4GHz (W/Hz)
            = 3.0 × 10^{−24} M_⊙/yr / (W/Hz) × L_1.4GHz
```

**Source:** Condon (1992, ARA&A 30, 575)

**Key Physics:**
- Free-free emission from ionized gas + non-thermal synchrotron from cosmic ray electrons
- Cosmic ray acceleration in supernova remnants (age ~150 Myr timescale)
- Extinction-free: unaffected by dust attenuation

**Timescale and Temporal Sensitivity:**
- **Free-free component:** Directly linked to current ionizing photons (young starbursts); decays after ~10 Myr
- **Synchrotron component:** Cosmic ray residence time ~150 Myr; older star formation contributes
- **Net sensitivity:** ~150 Myr (dominated by synchrotron aging timescale)
  - Reliable for constant or rising SFR
  - Underestimates SFR in quenched/declining systems (old cosmic rays still present)

**Thermal vs. Non-Thermal Decomposition:**
```
L_1.4GHz = L_free-free + L_synchrotron

Free-free (thermal): α_ff ~ −0.1  (slowly varying with frequency)
Synchrotron (non-thermal): α_s ~ −0.7  (steeper spectrum)

Spectral index measurement (α from F_ν ∝ ν^{−α}) can attempt separation;
in practice, curved spectra common → decomposition uncertain.
```

**AGN Contamination:**
- AGN contribute synchrotron and free-free (jets); can boost 1.4 GHz by factor 2–10
- Requires optical/IR diagnostic to identify AGN
- Radio morphology (jets vs. compact) helps separate

**Typical Dispersion:** ±0.3 dex (larger than IR due to thermal-nonthermal mix variation)

**Application:** Local universe and low-z (z < 0.5) due to confusion limits at high-z; LOFAR and new facilities improving high-z radio statistics

---

### 1.6 Multi-Wavelength SFR Calibration (Hybrid Approach)

**Recent Consensus (2023-2024):**

When combining UV, Hα, 24 μm, and TIR:

```
Dust-corrected UV SFR = SFR_UV,unatt + SFR_dust
                      ≈ (observed UV flux corrected to intrinsic) via attenuation law
```

```
Hα SFR = SFR_Hα with A_Hα from Balmer decrement
       = 9.7 × 10^{−42} × L_Hα(intrinsic) [M_⊙/yr]
```

```
IR SFR = 1.0 × 10^{−10} × L_IR(8-1000 μm) [M_⊙/yr]
```

**Consistency Check:**
- Typically agree within ±0.2 dex when dust properly accounted
- If > 0.3 dex scatter: indicates data quality issue or anomalous dust/AGN contribution

**Recommended Usage:**
- **UV-only:** Highly uncertain; use only for quiescent (dust-free) galaxies
- **Hα-only:** z-dependent; not available at high-z
- **24 μm-only:** Moderate uncertainty; avoid if Z < 0.3 Z_⊙
- **TIR:** Most reliable if FIR data available; use as primary indicator
- **Radio:** Extinction-free; useful as sanity check; timescale caveat noted

---

## 2. DUST ATTENUATION CURVES AND PARAMETERS

### 2.1 Calzetti Starburst Attenuation Curve

**Form:**
```
A(λ) / A_V = 2.659 × [-2.156 + 1.509/λ - 0.198/λ^2 + 0.011/λ^3] + R_V

where λ is in μm, R_V = A_V / E(B-V) ≈ 4.05 (Calzetti et al. 2000, empirical)
```

**Wavelength Coverage:** 0.12–2.2 μm (UV to near-IR)

**Key Properties:**
- Shallow wavelength dependence (different from MW/SMC curves)
- Better describes dust in star-forming regions (birth clouds)
- Empirically derived from local starburst galaxies

**Application:**
- Optical and near-IR primarily; caution at λ < 0.15 μm (extrapolation)

**Typical A_V Range:** 0.5–2.0 mag

---

### 2.2 Charlot & Fall Two-Phase Attenuation Model

**Form:**
```
A(λ) = A_ISM(λ) + [1 + μ] × A_BC(λ)

where:
  A_ISM(λ) = τ_ISM(λ) × attenuation_law(λ)
  A_BC(λ) = τ_BC(λ) × attenuation_law(λ)

  τ_ISM ∝ (λ/λ_0)^{−δ_ISM}  (typically δ_ISM ~ −0.7)
  τ_BC ∝ (λ/λ_0)^{−δ_BC}   (typically δ_BC ~ −0.7)

  μ = ratio of stellar light originating in birth clouds
      (younger population has higher μ)
```

**Physical Interpretation:**
- **ISM attenuation:** Diffuse dust affecting all stars; age-independent
- **Birth cloud attenuation:** Young massive stars still embedded; fades with age (~10 Myr)

**Key Advantages:**
- Age-dependent: better reproduces UV slope evolution with stellar population age
- Flexible: δ_ISM and δ_BC can vary to accommodate observed UV slopes
- Realistic: two-phase structure matches observations and theory

**Typical Parameters (from SED fits):**
- δ_ISM = −0.5 to −1.0 (shallow to steep)
- δ_BC = −0.5 to −1.0 (similar range)
- A_V(ISM) ~ 0.2–0.5 mag
- A_V(BC) ~ 0.2–1.0 mag (additional attenuation for young stars)

**Application:** Preferred at high-z (z > 2) where dust properties more uncertain

---

### 2.3 Infrared Excess – UV Slope Relation (IRX–β)

**Definition:**
```
IRX = L_IR / L_UV    (dimensionless; L_IR = 8–1000 μm, L_UV ~ rest 1500 Å)

β = ultraviolet slope (F_λ ∝ λ^β)
  Typically β ∈ [−3, 0]
  β = −2: intrinsic young stellar population
  β = −1: dust-reddened, older population
  β = 0: heavily reddened
```

**Empirical Relation (Local Universe, Meurer et al. 1999):**
```
log(IRX) = 0.68 × β + log(C)
where C ~ 0.5 (normalization constant)

Or approximately:
IRX ≈ 0.5 × 10^{0.68 × β}
```

**Scatter and Deviations:**
- Local galaxies scatter ~1 dex around mean relation
- High-z galaxies (z ~ 2–6) show similar scatter
- Primary drivers of scatter:
  1. **Dust geometry:** Clumpy media permit low IRX at fixed β; smooth screen forces higher IRX
  2. **Age effect:** Older stellar populations have redder intrinsic β; for fixed IRX, older → β more negative
  3. **Attenuation curve:** Shallow curves (SMC-like) shift IRX–β locus

**2024 Decoding Studies (Hamed et al. 2023, arXiv:2312.16700):**
```
Relative FUV attenuation > NUV attenuation  →  grain size variation or AGN heating
Dust geometry clumpy                        →  β offset −0.2 to +0.5 mag bluer
Stellar age > 100 Myr                       →  β offset −0.5 mag redder
```

**Quantitative Example:**
```
Case 1: Young starburst (age 10 Myr), smooth dust screen
  Intrinsic β_0 ≈ −2.5, A_V ≈ 0.5 mag → β_obs ≈ −1.8, IRX ≈ 3 (log IRX ≈ 0.5)

Case 2: Age 100 Myr, smooth dust screen
  Intrinsic β_0 ≈ −2.0, A_V ≈ 0.5 mag → β_obs ≈ −1.2, IRX ≈ 3 (log IRX ≈ 0.5)

Case 3: Young starburst, clumpy dust geometry
  Intrinsic β_0 ≈ −2.5, A_V ≈ 0.5 mag (effective) → β_obs ≈ −1.4, IRX ≈ 1 (log IRX ≈ 0)
  (Clumpiness permits escape of UV photons, reducing effective attenuation)
```

---

## 3. DUST TEMPERATURE MEASUREMENTS

### 3.1 Modified Blackbody Parameterization

**Standard Form:**
```
S_ν (flux density) = τ_ν × Ω × B_ν(T_d)

where:
  τ_ν = dust optical depth ≈ (ν/ν_0)^β  (frequency dependence)
  Ω = solid angle (or equivalently, M_dust × κ_ν)
  B_ν(T_d) = Planck function
  β = emissivity index, typically 1.5–2.0
```

**Fitting Procedure:**
1. Select FIR observations (e.g., Herschel 70, 100, 160 μm or ALMA 160, 250 μm)
2. Parameterize: T_d and β (or equivalently, T_d and M_dust if β fixed)
3. Minimize χ² between observed and model fluxes
4. Extract best-fit parameters and uncertainties from posterior distribution

### 3.2 Single-Component Dust Temperature Results

**Local Universe (z ~ 0):**
```
Typical star-forming galaxies: T_d ~ 35–45 K
Dust mass: M_dust ~ 10^7–10^8 M_⊙
Dust-to-stellar mass ratio: M_dust / M_* ~ 10^{−3}–10^{−2}
```

**Cosmic Noon (z ~ 2–3):**
```
SFGs: T_d ~ 40–50 K (slightly warmer; radiation field stronger)
DSFGs (submm-selected): T_d ~ 30–40 K (cooler; can be older starbursts)
Dust mass: M_dust ~ 10^8–10^9 M_⊙
Dust-to-stellar mass ratio: M_dust / M_* ~ 10^{−2}–10^{−1} (higher, suggesting efficient dust production)
```

**High Redshift (z ~ 4–6, ALMA-CRISTAL 2024):**
```
Typical SFGs (not submm-selected): T_d ~ 30–50 K (range, similar to lower-z)
Dust mass: M_dust ~ 10^7–10^8 M_⊙ (lower than z~2–3 DSFGs; main-sequence objects)
L_IR: 10^{10.9}–10^{12.4} L_⊙

Key finding: T_d remarkably uniform across cosmic time despite large L_IR variation
Implication: radiation field temperature regulated by dust properties (composition, grain size) not strongly z-dependent
```

### 3.3 Multi-Component Dust Models

**Two-Component Model:**
```
S_ν = τ_warm × Ω_warm × B_ν(T_warm) + τ_cold × Ω_cold × B_ν(T_cold)

Typical parameters:
  T_warm ~ 70 K,  M_warm ~ 10^6–10^7 M_⊙
  T_cold ~ 25 K,  M_cold ~ 10^8 M_⊙ (dominates total mass)

Two-component model ~10–20% improvement in fit quality compared to single T_d
Preferred for local galaxies; high-z SFGs often approximated by single component
```

**Emissivity Index β:**
```
Typical range: 1.5–2.0
β ~ 1.8–2.0 common in high-z sources (steep grain opacity)
β ~ 1.5 in some local galaxies (flatter; possible grain size variation)

Degeneracy with T_d: (T_d, β) pairs can produce identical SED fits
Recommended: fix β = 1.8 unless multiple independent FIR bands available
```

---

## 4. STELLAR MASS DETERMINATION FROM SED FITTING

### 4.1 Stellar Mass–Luminosity Relationship (M/L)

**Rest-Frame Near-Infrared (K-band, λ ~ 2.2 μm) Sensitivity:**

```
M_* / M_⊙ ∝ L_K / L_⊙  (approximately; M/L ~ 0.5–2 M_⊙ / L_⊙ for different ages)

Empirically, at fixed stellar population age:
  M/L_K increases with age (old stars: M/L ~ 1–2)
  M/L_K decreases with youth (young stars: M/L ~ 0.2–0.5 due to massive O-star light domination)
```

**SED Fitting Approach:**
1. Measure galaxy flux at multiple wavelengths (UV through NIR minimum: optical + K-band)
2. Generate SPS models at various ages, metallicity, IMF
3. Normalize models to observed optical/NIR flux
4. Extract M_* from best-fit SPS model

### 4.2 Systematic Uncertainties in M_*

| Source of Uncertainty | Typical Shift in log(M_*) | Dominance |
|---|---|---|
| **SPS model choice** | ±0.1 dex | Major |
| **IMF variation** (Chabrier vs. Salpeter) | ±0.2 dex | Major |
| **Dust attenuation** (unknown A_V) | ±0.15 dex | Moderate |
| **Age degeneracy** (young + high-dust vs. old + low-dust) | ±0.2–0.3 dex | Major |
| **Stellar mass loss** (instantaneous vs. delayed) | ±0.1 dex | Minor |
| **Photometric errors** (5% vs. 10% accuracy) | ±0.05 dex (5%) to ±0.1 dex (10%) | Minor |

**Typical Total Uncertainty (Combined):** ±0.15 dex (optimistic; 10% photometry, proper dust correction)
to ±0.3 dex (realistic; systematic SPS and dust uncertainties)

### 4.3 Age-Extinction Degeneracy

**Example:**
```
Galaxy A: Young (age 10 Myr) + low dust (A_V = 0)
  → (B−V)_obs ≈ 0.1 mag (blue)
  → M_* ≈ 10^{10} M_⊙ (if L_K measured)

Galaxy B: Old (age 2 Gyr) + high dust (A_V = 1.0 mag)
  → (B−V)_obs ≈ 0.8 mag (red)
  → M_* ≈ 10^{11} M_⊙ (higher luminosity due to age → larger M/L)

Both produce reddened colors, but very different ages and M_*
Resolution: Multi-wavelength SED covering UV + optical + NIR + FIR
  → Dust attenuation measurable from IRX or Balmer decrement
  → Age/metallicity constrained from UV slope + optical colors
```

---

## 5. SUMMARY TABLE: TYPICAL MEASUREMENT UNCERTAINTIES (2024-2025)

| Quantity | Measurement Method | Typical Uncertainty | Limiting Factor |
|----------|-------------------|-------------------|-----------------|
| **SFR** | UV (dust-corrected) | ±0.3–0.5 dex | Dust attenuation assumption |
| **SFR** | Hα + Balmer decrement | ±0.15 dex | Emission line flux measurement |
| **SFR** | 24 μm monochromatic | ±0.25 dex | Dust temperature; AGN contamination |
| **SFR** | TIR (8–1000 μm) | ±0.2 dex | FIR photometry accuracy |
| **SFR** | 1.4 GHz radio | ±0.3 dex | Synchrotron/thermal mix; AGN |
| **M_*** | SED fitting (with K-band) | ±0.15 dex | SPS model, IMF, dust attenuation |
| **A_V** | Balmer decrement | ±0.1 mag | Line flux ratio measurement |
| **A_V** | SED fitting | ±0.15 mag | SPS, dust model assumptions |
| **T_d** | Modified BB (3+ bands) | ±5–10 K | FIR photometry calibration |
| **Z** | Optical emission lines | ±0.1–0.2 dex | Line flux measurement; ISM temperature assumption |
| **Age** | SED fitting (young < 1 Gyr) | factor 2–3 | Degeneracy with dust; SPS model |
| **Age** | SED fitting (old > 1 Gyr) | ±0.2 dex | Breaks in SPS model |

---

## 6. REDSHIFT-DEPENDENT OBSERVATIONAL EFFECTS

### 6.1 K-Correction and Redshift Effects

**K-Correction Definition:**
```
K(z) = −2.5 log10[(1+z) × L(λ_obs) / L(λ_rest)]

where λ_rest and λ_obs = rest and observed wavelengths related by (1+z)
```

**Physical Effect:** As galaxy redshifts, observed bandpass shifts to longer rest wavelengths.
Example: z = 2 galaxy observed at 4.5 μm (IRAC 2) actually shows rest-frame ~0.9 μm (NIR).

**Impact on SFR Measurements:**
- Rest-frame UV shifts from UV bandpass to optical (z ~ 2–3)
- Rest-frame optical shifts to NIR (z ~ 3–6)
- Dust attenuation changes with wavelength → K-correction essential

### 6.2 Cosmological Dimming and Flux Limits

**Surface Brightness Dimming:**
```
S_obs = S_intrinsic / (1+z)^4   (two factors of (1+z): luminosity distance effect + bandpass shift)
```

**Observable Flux Limit at Fixed Depth (e.g., 1 μJy):**
```
L_intrinsic ∝ (1+z)^4  (flux dimming requires L increase to stay detectable)

Example:
  z = 0: L_1500Å = 10^8 L_⊙ → F_UV ≈ 1 mJy (easily detected by GALEX)
  z = 2: L_1500Å = 10^8 L_⊙ → F_UV ≈ 0.1 μJy (undetectable by current UV satellites)
         but rest-frame ~3500 Å (optical) ≈ 1 mJy in observed-frame (easily detected)
```

### 6.3 Dust Attenuation Evolution (JWST Results, Markov et al. 2024)

| Redshift | A_V (mag) | A_V / log(M_*/M_⊙) | Implication |
|----------|----------|-------------------|------------|
| z ~ 0–1 | 0.5–1.5 | 0.05–0.1 | Dust common; older galaxies dustier |
| z ~ 2–3 | 0.5–2.0 | 0.05–0.15 | Cosmic noon: high dust content |
| z ~ 4–6 | 0.3–1.0 | 0.03–0.08 | ALMA-CRISTAL: dust present but lower amounts |
| z ~ 7–8 | ~0.5 | ~0.03 | JADES: Balmer decrements show moderate attenuation |
| z ~ 9–12 | ~0.3–0.5 | — | GLASS: blue UV slopes, low inferred attenuation |

**Key Finding:** A_V decreases with redshift (z = 2 → 12), but scatter remains large (factor 3–5) at fixed mass.

---

## 7. COMPARISON TABLE: SED FITTING CODES (2024-2025 Status)

| Code | SPS | Dust | Nebular | Speed | Multi-λ | Z-range | Notes |
|------|-----|------|---------|-------|---------|---------|-------|
| **BC03** | Padova/FIRES | Calzetti | No | Very fast | Photo | 0–10 | Industry standard; no nebular |
| **FSPS** | MIST | Flexible | Some | Fast (GPU) | Photo/Spec | 0–13.8 | Excellent flexibility; nebular limited |
| **CIGALE** | BC03/SB99 | Full balance | Yes | Fast | Photo/Spec (v2025) | 0–20 | Best for complete UV–radio SED |
| **MAGPHYS** | BC03 | Two-T | No | Moderate | Photo | 0–10 | Good for IR-heavy data |
| **Starburst99** | Padova | User | Yes | Slow | Spec | 0–100 Myr | Theory-focused; young systems |
| **Prospector** | MIST | Flexible | No | Slow (MCMC) | Photo/Spec | 0–10 | Bayesian; full uncertainty quantification |

---

## 8. REQUIRED PHOTOMETRY FOR ROBUST SED FITS

### 8.1 Minimum Configuration

For **high-z star-forming galaxies** (z ≥ 3):

```
UV/Optical:
  - UV: GALEX FUV (rest ~1500 Å) or NUV
  - Optical: g, r, i or similar (rest-frame wavelengths depend on z)

NIR:
  - J, H, K (or NIRCam equivalent at high-z)

MIR (if available):
  - Spitzer IRAC 3.6, 4.5 μm (or JWST equivalent)

FIR:
  - Herschel PACS 160 μm OR
  - ALMA continuum (160, 250, 350 μm) for individual detection
```

**Expected Results with Minimum Config:**
- M_* to ±0.2 dex
- SFR to ±0.3 dex (if FIR data; ±0.5 dex without)
- A_V to ±0.2 mag

### 8.2 Optimal Configuration (for precision science)

```
UV/Optical (8–10 bands):
  GALEX FUV, NUV + optical multi-band (u, g, r, i, z) or narrow-band filters

NIR (3–4 bands):
  J, H, K (or NIRCam equivalent)

MIR (3–4 bands):
  IRAC [3.6], [4.5], [5.8], [8.0] μm (or MIRI at high-z)

FIR (4–6 bands):
  Herschel PACS [70, 100, 160] + SPIRE [250, 350, 500] or ALMA equivalent

Radio (1–2 bands):
  1.4 GHz VLA or similar

Spectroscopy (ideal):
  Redshift + emission lines (Hα, [OIII], [OII] or rest-UV equivalents)
```

**Expected Results with Optimal Config:**
- M_* to ±0.1 dex
- SFR to ±0.2 dex
- A_V to ±0.1 mag
- T_d to ±5 K

---

## 9. RECOMMENDED READING ORDER FOR PRACTITIONERS

1. **Start with calibrations:** Kennicutt & Evans (2012) — comprehensive review of SFR indicators
2. **For SED fitting basics:** Conroy (2013) — stellar population synthesis overview
3. **For dust physics:** Charlot & Fall (2000) — two-phase attenuation model
4. **For practical tools:** Boquien et al. (2019) + Burgarella et al. (2025) — CIGALE code papers
5. **For recent high-z results:** JADES 2024 papers (NIRSpec spectroscopy) + ALMA-CRISTAL 2024
6. **For systematic comparisons:** Hahn et al. (2024) — SPS code/isochrone dependencies
7. **For Balmer decrement application:** PHANGS-MUSE 2023 paper on resolved SFR

---

## 10. QUICK REFERENCE: UNIT CONVERSIONS

```
1 L_⊙ = 3.828 × 10^26 W = 3.828 × 10^33 erg/s

1 W/m^2 = 10^7 erg/(s·cm^2)

Magnitudes ↔ Flux:
  m_AB = −2.5 log10(f_ν) − 48.60   (f_ν in Jy)
  1 Jy = 10^{−26} W/(m^2·Hz) = 10^{−23} erg/(s·cm^2·Hz)

Luminosity Distance:
  L_ν = 4π d_L^2 × S_ν  (S_ν = observed flux, L_ν = luminosity)

Specific SFR:
  sSFR = SFR / M_* = (M_⊙/yr) / (M_⊙) = [1/yr]
  sSFR ~ 10^{−11} yr^{−1} (local main sequence)
  sSFR ~ 10^{−10} yr^{−1} (z ~ 2 main sequence)
  sSFR ~ 10^{−9} yr^{−1} (starburst galaxies)
```

---

*This table serves as a companion to the main literature review.*
*Updated December 2025.*
