# Diagnostic Techniques for Distinguishing AGN from Starburst X-ray Sources: Technical Summary

## 1. X-ray Spectroscopic Diagnostics

### 1.1 Photon Index Measurement

**Definition**: The photon index (Γ) characterizes the power-law spectral steepness: F(E) ∝ E^(-Γ)

**Measurement Method**:
- Fit X-ray energy spectrum (0.5-10 keV range typically) to absorbed power-law model: F(E) = K × (E/1keV)^(-Γ) × exp(-N_H × σ(E))
- Requires ≥100-500 source counts for reliable measurement (source-dependent)
- Typical fitting tools: XSPEC, ISIS, PyXspec

**Diagnostic Thresholds**:
| Source Type | Typical Γ | Notes |
|-------------|----------|-------|
| **Type 1 AGN** | 1.7-2.0 | Intrinsic power-law from hot corona |
| **Type 2 AGN** | 1.5-2.0 | Often harder due to scattering |
| **X-ray Binaries** | 1.5-2.5 | Overlaps AGN range; spectral state dependent |
| **Starburst (thermal)** | - | Dominated by plasma cutoff (kT~0.1-1 keV), not power-law |
| **Starburst (XRB-dominated)** | >1.8 | Steeper than typical AGN due to population mixing |

**Systematic Uncertainties**:
- Photon index uncertainty: ±0.1-0.3 (depends on source counts, spectral resolution)
- Absorption effects: N_H uncertainties typically 20-30% affecting hardness

**Practical Example**:
- A source with Γ = 1.6 ± 0.2 is ambiguous (could be AGN or hard XRB)
- Combined with Fe Kα line detection (see below), likely AGN
- Without Fe Kα but with high N_H (>10^22 cm^-2), indicates obscured AGN

### 1.2 Iron K-alpha (Fe Kα) Emission Line Detection

**Physical Origin**:
- Nuclear continuum photoionizes inner-disk iron atoms
- Fluorescence line at 6.4 keV from neutral/low-ionization Fe (Fe I-XVII)
- Broad component (FWHM ~10,000-50,000 km/s) from relativistic Doppler broadening in accretion disk
- Narrow component (FWHM <10,000 km/s) from more distant material (torus, NLR)

**Measurement Procedure**:
1. Fit continuum model (power-law + thermal components) to data excluding Fe Kα region (6.0-6.8 keV)
2. Add Gaussian line component at 6.4 keV; allow FWHM and equivalent width (EW) as free parameters
3. Measure line EW with respect to underlying continuum: EW = ∫(flux_line)/continuum

**Diagnostic Quantification**:

| Parameter | AGN | Starburst |
|-----------|-----|-----------|
| **EW (Fe Kα)** | >1 keV (often >2 keV) | <0.5 keV if present at all |
| **Line FWHM** | Typically 3,000-15,000 km/s | Broader (if Fe XXV) or absent |
| **Ionization State** | Fe I-XVII (neutral/low-ion) | Fe XXV (highly ionized), ~6.7 keV |
| **Line Variability** | Tracks continuum on months-years | Stable or absent variability |

**Detection Requirements**:
- Need ≥5-10σ detection of line above continuum
- For AGN: requires 5,000-50,000 net source counts depending on EW
- For starburst: rarely detectable unless extremely strong thermal plasma

**Recent Key Result** (2021-2024):
Studies using Chandra HETGS show:
- AGN Fe Kα EW independent of X-ray luminosity
- Starburst Fe line (primarily Fe XXV) EW correlates with infrared luminosity, NOT X-ray luminosity
- This distinction provides powerful AGN/starburst separation even in composite systems

### 1.3 Spectral Hardness Ratios

**Definition**: Ratio of hard (H, typically 2-10 keV) to soft (S, typically 0.5-2 keV) band counts

HR = (H - S)/(H + S)

**Interpretation Guide**:

| HR Range | Interpretation | Caveats |
|----------|----------------|---------|
| HR > 0.5 | Hard source; AGN likely or heavily obscured starburst | Moderate spectral resolution only |
| 0.2 < HR < 0.5 | Mixed AGN/XRB or Compton-thick AGN | Degeneracy between source type and absorption |
| -0.2 < HR < 0.2 | Soft source; starburst or thermal plasma dominant | Many unobscured stars and normal galaxies here |
| HR < -0.2 | Very soft; likely thermal plasma, not AGN | Clear starburst/normal galaxy signature |

**Advantages**:
- Requires no spectral fitting; available even for faint sources with few counts
- Fast, objective classification metric for large samples

**Limitations**:
- Only reflects 2-band ratio; loses detailed spectral information
- High N_H absorption can make starburst appear hard
- XRB population hardness overlaps with AGN range

**Practical Example**:
- HR = 0.7 with low N_H: high confidence AGN
- HR = 0.7 with high N_H (>10^23 cm^-2): could be heavily obscured starburst; Fe Kα line presence breaks degeneracy

### 1.4 Spectral Component Modeling (Multi-Component Fits)

**Canonical Starburst Model** (Ptak et al. 1999):
```
Model = PHABS * [RAYMOND + POWERLAW]
```
- RAYMOND: Thermal plasma component, kT ~ 0.1-1 keV (diffuse hot gas)
- POWERLAW: Power-law component, Γ ~ 1.7-2.2 (X-ray binaries)
- Photon index Γ typically > 1.8 in starburst-dominated systems

**Canonical AGN Model** (base):
```
Model = PHABS * POWERLAW + REFLECTION [+ GAUSS(Fe Kα) + GAUSS(other lines)]
```
- POWERLAW: Direct nuclear continuum, Γ ~ 1.7-2.0
- REFLECTION: Compton reflection hump at E > 10 keV (if present)
- GAUSS: Iron line and other discrete components

**Composite ULIRG Model** (4+ components):
```
Model = PHABS * [POWERLAW_AGN + SCATTERED_AGN + RAYMOND + POWERLAW_XRB]
+ [reflection + Fe Kα + other lines]
```

**Fitting Strategy**:
1. Start with minimal model (photon index + NH) and evaluate goodness of fit (χ²/dof)
2. Add thermal component if significant improvement and residuals show soft excess
3. Test for Fe Kα line; keep if ΔC-stat > 9 (roughly 3σ) or visual inspection shows significant peak
4. For composite systems, use independent physical priors (e.g., Fe Kα EW must match reflection models)

**Model Selection Criteria**:
- F-test: Compare nested models; require ΔC-stat > 9 for component addition (roughly 3σ significance)
- Information criteria: AIC, BIC penalize additional parameters
- Residual analysis: Visual inspection for systematic deviations

### 1.5 Absorption Column Density (N_H) Diagnostics

**Measurement**:
- X-ray photoelectric absorption cross-section σ(E) tabulated (Verner et al. 1996; Wilms et al. 2000)
- Fit N_H as parameter in spectral model: τ(E) = N_H × σ(E)
- Units: 10^22 cm^-2 (standard)

**Diagnostic Interpretation**:

| N_H (10^22 cm^-2) | Source Classification | Notes |
|------------------|----------------------|-------|
| < 1 | Unobscured source | AGN Type 1, star-forming galaxy, or normal galaxy |
| 1-10 | Moderately obscured | Could be Type 2 AGN, dusty starburst, or composite |
| 10-100 | Heavily obscured AGN | Compton-thin; typical Type 2 Seyferts |
| > 100 | Compton-thick AGN | High equivalent width Fe Kα expected |

**AGN vs. Starburst Distinction**:
- **AGN**: Wide range of N_H due to obscuration by torus; no strong correlation with stellar mass
- **Starburst**: N_H typically 10^20-10^22 cm^-2 reflecting galactic dust column; correlates with star formation rate
- High N_H in optically soft source indicates obscured AGN (Compton-thick candidate)

**Systematic Issues**:
- Large uncertainties in N_H for sources with few counts (factor ~2-3)
- Assumes cosmic abundance ratios; enriched ISM can affect cross-sections
- Line-of-sight absorption can hide starburst XRB population (makes it appear AGN-like)

---

## 2. Multi-Wavelength Diagnostic Approaches

### 2.1 X-ray to Optical Flux Ratio Diagnostic

**Formulation**:
log₁₀(L_X/L_Hα) where L_X is 2-10 keV X-ray luminosity and L_Hα is Hα optical line luminosity

**Physical Basis**:
- AGN: Produce hard X-rays from accretion disk corona; optical emission from BLR/NLR relatively weak compared to X-ray power
- Starburst: Produce abundant Hα from star-forming regions; hard X-rays from XRB population limited relative to optical emission

**Quantitative Diagnostic Threshold** (Yan et al. 2011):
- **log₁₀(L_X/L_Hα) > 1.0**: AGN classification; ~90% of sources in this regime are AGN
- **log₁₀(L_X/L_Hα) < 1.0**: Star-forming classification; ~80% are pure starbursts
- **Transition zone (0.5 < log₁₀(L_X/L_Hα) < 1.5)**: Composite AGN+starburst systems

**Advantages Over Optical BPT Alone**:
- Reduces optical misclassification of dust-obscured AGN by ~30-40%
- Less affected by AGN obscuration than optical lines (depends only on X-ray flux)
- Quantitatively separates populations without need for detailed model fitting

**Application Method**:
1. Measure Hα flux from optical spectroscopy (rest-frame equivalent width > 6Å for reliable detection)
2. Correct for dust attenuation using optical continuum slope or Balmer decrement
3. Calculate L_Hα from flux and luminosity distance
4. Extract L_X(2-10keV) from X-ray spectral fit or observed flux scaling
5. Compute ratio and compare to threshold

**Observational Requirements**:
- Optical spectroscopy with resolution sufficient to resolve Hα (Δλ ~ 1-5 Å)
- X-ray detection with ≥30-100 counts in 2-10 keV band for flux reliability
- Spectroscopic redshift for accurate luminosity distances

**Refinements and Extensions**:
- Can be extended to [O III], [N II], or other emission lines
- Ratio depends weakly on redshift (ionization parameter evolution)
- Dust correction introduces ~0.3 dex systematic uncertainty

### 2.2 Infrared-to-X-ray Luminosity Correlation

**Physical Principle**:
Star-forming galaxies follow tight L_IR ∝ L_X correlation; AGN deviate significantly due to accretion power dominating over star formation

**Parametric Form**:
For pure starburst: log(L_X) ≈ 0.6-0.8 × log(L_IR) - offset

Typical empirical relation: log(L_X/L_IR) ≈ -4.5 for starbursts; higher ratios indicate AGN contribution

**Quantitative Application**:
1. Measure total infrared luminosity L_IR (8-1000 μm) from infrared SED
2. Obtain L_X from X-ray 2-10 keV observations
3. Calculate L_X/L_IR ratio
4. **Starburst-dominated**: L_X/L_IR ≈ 10^-4.5 to 10^-4.0
5. **AGN-dominated**: L_X/L_IR > 10^-3.5 (AGN adding to starburst baseline)

**Advantages**:
- Largely dust-independent (infrared from heated dust, X-ray penetrates dust)
- Exploits physical differences in radiation production mechanisms
- Less biased by absorption than optical diagnostics
- Works even for heavily obscured systems

**Limitations**:
- Requires infrared photometry at multiple wavelengths (Herschel, WISE, AKARI)
- Intrinsic scatter ~0.5 dex in starburst L_IR-L_X relation
- AGN contribution to mid-IR dust heating affects L_IR (complicates inference)

**Practical Example**:
- ULIRG with L_IR = 10^12.5 L_sun and L_X(2-10keV) = 10^43.5 erg/s
- L_X/L_IR = 10^43.5 / 10^12.5×(3.8×10^26 W) ≈ 10^-4.2
- Close to starburst locus; indicates starburst-dominated system with possible modest AGN contribution

### 2.3 Spectral Energy Distribution (SED) Decomposition

**Methodology**:
Fit broadband photometry (UV to submillimeter) with composite models for stellar populations, AGN, and dust

**Software Tools**:

**CIGALE** (Code Investigating GALaxy Emission):
- Fits SEDs with templates for stellar populations, dust, and AGN
- Can include X-ray flux in fitting process
- Accounts for dust extinction in AGN polar regions
- Output: Quiescent and star-formation stellar mass, AGN bolometric luminosity, dust luminosity
- Typical accuracy: ±0.2-0.3 dex on luminosities

**AGNFITTER-RX** (recent 2024):
- Bayesian code fitting radio-to-X-ray broadband SEDs
- Uses theoretical templates for AGN (disk, corona, jet, torus) and empirical stellar/dust models
- Provides probability distributions on parameters, not point estimates
- Computational cost: hours per source on modern hardware

**Fundamental Approach**:
1. Construct SED: Gather photometry at ≥10 wavelengths from UV (0.1 μm) to millimeter (1 mm)
2. Specify model components: Stellar population template (SSP), AGN template (disk+torus or simple power law), dust attenuation law
3. Fit photometry to model; minimize χ² or maximize Bayesian likelihood
4. Extract component luminosities: L_*,quiescent, L_*,SF, L_AGN, L_dust

**Component Separation Uncertainties**:
- Pure starburst: AGN luminosity uncertainty factor ~2-3 (limited by template variety)
- Pure AGN: Starburst SFR uncertainty ~30-50% (dust/stellar mass degeneracy)
- Composite systems: Uncertainties increase to factor ~3-5 for each component

**Quantitative Example**:
- Composite galaxy: CIGALE fit yields:
  - L_AGN = 10^44.5 ± 0.3 erg/s
  - L_IR(starburst) = 10^12.2 ± 0.2 L_sun
  - Implies SFR ≈ 100 M_sun/yr (using standard calibrations)
  - Dust temperature: T_d ≈ 50 K (from SED shape)

**Model-Dependence Issues**:
- Results depend strongly on choice of AGN template (accretion efficiency, torus geometry)
- Stellar population age-dust degeneracy can affect derived parameters
- Different extinction laws (Calzetti, SMC, etc.) produce ~0.2 dex differences in L_AGN

### 2.4 Optical Emission-Line Diagnostics (BPT and Extensions)

**Classical BPT Diagram**:
Axes: [O III]λ5007/Hβ vs. [N II]λ6584/Hα

**Classification Regions**:
- **HII (Star-forming)**: Below KEWLEY line; electron density Ne ~ 10² cm^-3
- **Seyfert (Type 2 AGN)**: Above Seyfert line; [O III]/Hβ > 3
- **LINER**: Low-ionization emission-line region; distinct from Seyferts
- **Composite**: Transitional regions between star-forming and AGN

**Quantitative Boundaries** (Kewley et al. 2001, 2006):
- KEWLEY line (upper boundary of HII): log([O III]/Hβ) = 0.61 / (log([N II]/Hα) - 0.05) + 1.3
- SEYFERT line: log([O III]/Hβ) = 0.61 / (log([N II]/Hα) - 0.47) + 1.19

**AGN vs. Starburst Confusion**:
- ~30-50% of optically-classified HII galaxies harbor X-ray detected AGN (low-ionization/obscured AGN invisible optically)
- Dust extinction toward BLR can shift AGN sources into HII region
- Low-mass AGN produce weak optical emission lines, mimicking starburst diagnostics

**Extensions to Reduce Bias**:

**Near-Infrared Diagnostics** (less dust-affected):
- Pa-alpha (1.875 μm) / Br-gamma (2.166 μm) analogue to Hα/Hβ
- Uses [Fe II](1.644 μm) and [P II](1.188 μm) instead of [N II]
- Reduces dust attenuation effects by factor ~2-5 compared to optical

**X-ray Enhanced BPT**:
- Color-code BPT sources by X-ray/optical ratio
- Sources with high X-ray/optical ratio cluster in Seyfert region
- Provides visual separation even for marginal optical diagnostics

**Ionization Parameter Analysis**:
- Measure ionization parameter U from emission-line ratios (e.g., [O III]/[O II])
- AGN photoionization: U ~ 0.01-1 (high ionization)
- Starburst HII regions: U ~ 0.01-0.1 (lower ionization)
- Requires careful dust correction and detailed plasma modeling

---

## 3. Summary Diagnostic Decision Tree

### Quick AGN/Starburst Classification Algorithm

```
1. X-ray Detection Available?
   NO → Use optical BPT + infrared SED alone
   YES → Continue to step 2

2. X-ray Spectral Fitting Possible? (≥500 counts)
   NO → Use hardness ratio (HR)
       IF HR > 0.5 → Likely AGN
       IF HR < -0.2 → Likely Starburst
       ELSE → Ambiguous; need additional data
   YES → Continue to step 3

3. Fe Kα Line Detected at >3σ?
   YES → Fe Kα EW > 1 keV → AGN confirmed; Compton-thick if high NH
   NO → Continue to step 4

4. Photon Index Measurement
   IF Γ < 1.6 and high NH → Obscured AGN likely
   IF Γ > 1.9 → XRB-dominated starburst or Type 2 AGN; use optical data
   IF 1.6 < Γ < 1.9 → Ambiguous; continue to step 5

5. Multi-wavelength Diagnostics

   a) Optical Available?
      Calculate log₁₀(L_X/L_Hα)
      IF > 1.0 → AGN
      IF < 0.5 → Starburst
      ELSE → Composite; proceed to SED fitting

   b) Infrared Available?
      Calculate L_X/L_IR ratio
      IF log(L_X/L_IR) > -3.5 → AGN-dominated
      IF log(L_X/L_IR) < -4.0 → Starburst-dominated
      ELSE → Composite

   c) Perform SED Fitting (CIGALE/AGNFITTER)
      If L_AGN/L_total > 0.3 → AGN-dominated
      If L_AGN/L_total < 0.1 → Starburst-dominated
      Else → Composite AGN+Starburst

6. Final Classification:
   Combine diagnostics with weighted voting
   AGN if ≥3 independent diagnostics favor AGN
   Starburst if ≥3 independent diagnostics favor starburst
   Composite if AGN and starburst diagnostics split
```

### Diagnostic Confidence Levels

| Confidence | Criteria |
|-----------|----------|
| **Very High (>95%)** | Multiple independent diagnostics agree (Fe Kα + high Γ + X-ray/optical ratio > 1 + SED) |
| **High (80-95%)** | Two independent diagnostics agree (e.g., Fe Kα + X-ray/optical ratio) |
| **Moderate (60-80%)** | Single robust diagnostic (e.g., X-ray hardness ratio with caveats) |
| **Low (<60%)** | Single ambiguous diagnostic or contradictory indicators |

---

## 4. Quantitative Results Summary Table

### Diagnostic Performance Metrics

| Diagnostic Method | AGN Detection Rate | Starburst Detection Rate | Confusion Rate | Notes |
|------------------|-------------------|------------------------|-----------------|-------|
| **Optical BPT alone** | 50-70% | 70-80% | 20-30% | High confusion in composite systems |
| **Hard X-ray HR** | 75-85% | 60-70% | 15-25% | Spectral hardness overlaps XRB/AGN |
| **Fe Kα line (EW)** | 85-95% | >99% (no false positives) | <5% | Requires high S/N; few AGN without Fe Kα |
| **X-ray/Hα ratio** | 85-90% | 80-85% | 10-15% | Robust; dust-correction dependent |
| **L_X/L_IR ratio** | 75-85% | 70-80% | 15-25% | Model-dependent; good for IR-luminous sources |
| **SED decomposition** | 80-90% | 75-85% | 15-20% | Template-dependent; requires multi-wavelength data |
| **Combined multi-wavelength** | 90-95% | 85-90% | 5-10% | State-of-the-art; computationally intensive |

---

## 5. Key Literature References by Technique

### Photon Index Diagnostics
- Ptak & Griffiths (1999), ApJS 120:179
- Multiple X-ray survey papers citing canonical Γ values

### Fe Kα Line Diagnostics
- Fabian et al. (2000), MNRAS 315:L8
- Nandra et al. (2007), MNRAS 382:194
- Recent: Younes et al. (2021), ApJ 914:83 (Chandra Fe K line in starbursts)

### X-ray/Optical Ratios
- Yan et al. (2011), ApJ 728:38
- Panessa et al. (2012), A&A 544:B139

### SED Decomposition
- Berta et al. (2013), A&A 551:A100 (CIGALE first paper)
- AGNFITTER: Calistro Rivera et al. (2016), ApJ 833:98
- AGNFITTER-RX: Recent 2024 papers

### Infrared-X-ray Correlations
- Multiple Chandra papers; correlations established empirically in large surveys

