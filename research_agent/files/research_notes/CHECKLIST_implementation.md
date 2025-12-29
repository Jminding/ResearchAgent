# Implementation Checklist: AGN vs. Starburst X-ray Analysis

## Pre-Analysis Preparation

### Data Requirements Verification
- [ ] **X-ray data available** (Chandra, XMM-Newton, or eROSITA)
- [ ] **X-ray exposure time ≥ 5 ks** (for spectral fitting)
- [ ] **Source net counts ≥ 30-100** (minimum for flux measurement)
- [ ] **Source net counts ≥ 500** (desired for detailed spectral fitting)
- [ ] **Optical spectroscopy available** (Hα, [N II], [O III], Hβ)
- [ ] **Optical S/N ≥ 20 per Angstrom** (for reliable emission-line ratios)
- [ ] **Spectroscopic redshift available** (for accurate luminosity distances)
- [ ] **Infrared photometry available** (≥3 bands preferred; WISE minimum)
- [ ] **Far-infrared data** (Herschel/AKARI preferred for starburst diagnostics)
- [ ] **Radio data optional** (VLA/LOFAR helpful for high-z confirmation)

### Software Installation and Setup
- [ ] **XSPEC installed** (for X-ray spectral fitting)
  - Version: Check latest (v12.12+)
  - Test: Run simple POWERLAW model
- [ ] **ISIS or alternative** (optional for advanced Bayesian fitting)
- [ ] **CIGALE installed** (for SED decomposition)
  - Configuration: Set AGN+starburst templates
  - Test: Run sample galaxy SED
- [ ] **AGNFITTER-RX available** (optional; latest SED tool)
- [ ] **Basic tools**: Python (numpy, scipy, matplotlib), IDL or similar
- [ ] **Reference data**: X-ray calibration files (RMF, ARF files downloaded)

---

## Stage 1: Data Preparation and Validation

### X-ray Data Processing
- [ ] **Extract source spectrum** (source + background regions defined)
  - Source region: Circle or optimized shape around nucleus
  - Background region: Adjacent source-free area (same chip)
  - Minimum source region: 5-arcsec radius
- [ ] **Extract background spectrum** (multiple background regions recommended)
- [ ] **Generate response files** (RMF, ARF)
  - RMF: Instrumental response matrix
  - ARF: Effective area file (accounts for vignetting, QE)
- [ ] **Verify spectral extraction** (check net counts and S/N)
  - Net source counts: [ ] > 100, [ ] > 500, [ ] > 1000
  - Count-loss < 1%: [ ] Confirmed
- [ ] **Check for pile-up** (if using CCD detectors)
  - Expected at count rate > ~1 count/s in point source
  - [ ] Pile-up negligible, [ ] Pile-up correction applied
- [ ] **Define energy ranges for analysis**
  - Soft band: 0.5-2 keV
  - Hard band: 2-10 keV
  - Full spectrum: 0.5-10 keV (minimum)

### Optical Spectroscopy Reduction
- [ ] **Emission-line measurements made** (if not from literature)
  - Hα flux: [ ] Measured, uncertainty ±__%
  - [N II] λ6584 flux: [ ] Measured
  - [O III] λ5007 flux: [ ] Measured
  - Hβ flux: [ ] Measured
- [ ] **Dust extinction correction applied**
  - Method used: [ ] Balmer decrement, [ ] Optical continuum slope, [ ] External model
  - Extinction value: A_Hα = ____ (mag)
  - Uncertainty: ±____ mag
- [ ] **Emission-line ratios calculated**
  - [O III]/Hβ = ____ ± ____
  - [N II]/Hα = ____ ± ____
  - Hα luminosity: ____ ± ____ erg/s
  - Uncertainties propagated: [ ] Yes, [ ] No

### Infrared Photometry Compilation
- [ ] **Multi-band infrared photometry assembled** (minimum WISE, preferred Herschel)
  - WISE bands (3.4, 4.6, 12, 22 μm): [ ] Available
  - Herschel bands (70, 100, 160 μm): [ ] Available
  - AKARI or Spitzer (24 μm): [ ] Available
  - Additional bands: [ ] Specify _______
- [ ] **Flux uncertainty estimates obtained** (≥10% assumed if not specified)
- [ ] **Corrected for Galactic extinction** (e.g., Schlegel dust maps)
- [ ] **Redshift verified** (consistent across bands)

### Redshift Confirmation
- [ ] **Redshift source documented** (spectroscopic preferred)
  - Source: [ ] Spectroscopic, [ ] Photometric
  - Redshift value: z = ____ ± ____
  - Reference: _______
- [ ] **Luminosity distances calculated** (H₀=67.4 km/s/Mpc assumed)
  - Distance modulus: DM = ____ mag
  - Luminosity distance: D_L = ____ Mpc

---

## Stage 2: X-ray Spectral Analysis

### Quick Classification (Minimal Analysis)
- [ ] **Hardness ratio calculated** (if high-count source)
  - Hard band (2-10 keV) counts: ____
  - Soft band (0.5-2 keV) counts: ____
  - HR = (H-S)/(H+S) = ____ ± ____
  - Interpretation: [ ] Likely AGN (HR > 0.5), [ ] Ambiguous, [ ] Likely Starburst (HR < -0.2)

### Detailed Spectral Fitting
- [ ] **Source spectral analysis initiated** (XSPEC or equivalent)
  - Fitting method: [ ] χ² minimization, [ ] Bayesian (stat parameter)
  - Energy range for fit: ____ - ____ keV
  - Rebinning: [ ] No rebinning, [ ] ≥ 20 counts/bin, [ ] Optimal (Cash statistic)

- [ ] **Simple power-law model fitted first**
  ```
  Model: PHABS*POWERLAW
  Absorption (PHABS):
    - NH (fixed or free): ______ ± ______ (10²² cm⁻²)
    - Confidence: [ ] Fixed, [ ] Fitted
  Continuum (POWERLAW):
    - Photon index (Γ): ______ ± ______
    - Normalization: ______ photons/keV/cm²/s
  ```
  - Goodness of fit: χ²/dof = ______ (acceptable if <1.5)
  - [ ] Good fit, [ ] Requires additional components

- [ ] **Thermal component tested** (if soft excess evident)
  ```
  Model: PHABS*(RAYMOND + POWERLAW)
  Thermal component (RAYMOND):
    - Temperature (kT): ______ ± ______ keV (~0.1-1 keV for starbursts)
    - Abundance (Z): ______ (relative to solar)
  Improvement in fit: ΔC-stat = ______ (significant if > 9)
  ```
  - [ ] Thermal component significant, [ ] Not required

- [ ] **Fe Kα line searched for** (if adequate spectral resolution)
  ```
  Model: PHABS*(POWERLAW + GAUSS(Fe Kα))
  Fe Kα line:
    - Energy (fixed): 6.4 keV
    - FWHM (free): ______ ± ______ km/s
    - Equivalent Width (EW): ______ ± ______ keV
    - Confidence: ______ σ (significance)
  ```
  - [ ] Fe Kα detected (EW > 1.0 keV, >3σ), indication: **AGN likely**
  - [ ] Fe Kα marginal (0.5-1.0 keV, 2-3σ), indication: **Composite possible**
  - [ ] Fe Kα not detected (<0.5 keV), indication: **Starburst more likely**

- [ ] **Compton reflection tested** (if high-quality hard X-ray data)
  - [ ] Reflection component evident (E > 10 keV)
  - [ ] Absorption column density high (N_H > 10²³ cm⁻²): **Obscured AGN**

- [ ] **Best-fit spectral model selected**
  - Recommended model: ______ (e.g., PHABS*POWERLAW)
  - Model complexity justified: [ ] Yes (fewer dof penalties)

- [ ] **Spectral parameters and uncertainties documented**
  - Photon index (Γ): ______ ± ______ (note: AGN ~1.7-2.0, SB ~1.8-2.2)
  - Absorption (N_H): ______ ± ______ (10²² cm⁻²)
  - X-ray flux (2-10 keV): ______ ± ______ erg/s/cm²
  - X-ray luminosity (L_X, 2-10 keV): ______ ± ______ erg/s

### Hardness Ratio Calculation (Alternative for faint sources)
- [ ] **Hardness ratio computed**
  - HR = (H-S)/(H+S) = ______ ± ______
  - [ ] HR > 0.5 (likely AGN)
  - [ ] -0.2 < HR < 0.5 (ambiguous; need additional diagnostics)
  - [ ] HR < -0.2 (likely starburst)

---

## Stage 3: Multi-Wavelength Diagnostic Application

### X-ray/Optical Ratio Diagnostic
- [ ] **X-ray/Hα ratio calculated**
  - L_X (2-10 keV): ______ ± ______ erg/s (from spectral fit)
  - L_Hα: ______ ± ______ erg/s (from optical spectroscopy)
  - Ratio: log₁₀(L_X/L_Hα) = ______ ± ______

  - **Classification**:
    - [ ] log₁₀(L_X/L_Hα) > 1.0: **AGN-dominated** (confidence: high)
    - [ ] 0.5 < log₁₀(L_X/L_Hα) < 1.0: **Composite AGN+Starburst** (confidence: moderate)
    - [ ] log₁₀(L_X/L_Hα) < 0.5: **Starburst-dominated** (confidence: high)

  - Uncertainty source: [ ] X-ray flux ±0.2 dex, [ ] Optical dust ±0.3 dex, [ ] Both

### Infrared-X-ray Luminosity Correlation
- [ ] **Total infrared luminosity calculated** (8-1000 μm)
  - Method used: [ ] SED integration, [ ] Template fitting
  - L_IR: ______ ± ______ L_sun (or erg/s)

- [ ] **X-ray to infrared ratio computed**
  - L_X/L_IR ratio: 10^______ ± ______

  - **Classification**:
    - [ ] log(L_X/L_IR) > -3.5: **AGN-dominated**
    - [ ] -4.5 < log(L_X/L_IR) < -3.5: **Composite**
    - [ ] log(L_X/L_IR) < -4.5: **Starburst-dominated**

### Optical Diagnostic Diagram (BPT)
- [ ] **BPT classification performed** (if optical lines available)
  - [O III]/Hβ: ______ ± ______
  - [N II]/Hα: ______ ± ______

  - **Position on BPT diagram**:
    - [ ] HII region (star-forming)
    - [ ] Seyfert (AGN)
    - [ ] LINER
    - [ ] Composite/transition

  - Note: **Check if X-ray detected AGN misclassified as HII** (known issue; Panessa et al. 2012)

---

## Stage 4: SED Decomposition (If Multi-wavelength Data Available)

### CIGALE SED Fitting
- [ ] **CIGALE configuration prepared**
  - Template library: [ ] Include AGN, [ ] Include starburst, [ ] Include dust
  - Redshift: z = ____ (fixed or free)
  - Fitting method: [ ] Chi-square, [ ] Bayesian

- [ ] **SED assembled** (minimum 10 bands recommended)
  - UV: [ ] Available (e.g., GALEX)
  - Optical: [ ] Available (g, r, i, z bands)
  - Near-IR: [ ] Available (J, H, K)
  - Mid-IR: [ ] Available (WISE bands)
  - Far-IR: [ ] Available (Herschel 70-500 μm)
  - Millimeter: [ ] Available (optional)
  - X-ray: [ ] Included (recommended if possible)

- [ ] **SED fitting executed**
  - Output photometry predicted: [ ] Reviewed for consistency
  - Chi-square per band: [ ] All < 3σ outliers checked

- [ ] **Component luminosities extracted**
  - L_AGN (AGN bolometric): ______ ± ______ erg/s
  - L_SB (starburst): ______ ± ______ erg/s
  - SFR (star formation rate): ______ ± ______ M_sun/yr
  - Dust mass: ______ ± ______ M_sun
  - AGN fraction: L_AGN/(L_AGN + L_SB) = ______ ± ______

### AGNFITTER-RX (If Advanced Bayesian Fitting Desired)
- [ ] **AGNFITTER-RX setup** (optional; more computationally intensive)
  - [ ] Installation verified
  - [ ] Broadband SED assembled (radio-to-X-ray if possible)
  - [ ] Bayesian fitting executed
  - [ ] Posterior distributions examined for AGN and starburst parameters

---

## Stage 5: Final Classification and Confidence Assessment

### Decision Tree Classification
- [ ] **Evidence summary tabulated**

  | Diagnostic | Result | AGN Indicator | SB Indicator | Ambiguous |
  |-----------|--------|---|---|---|
  | **Fe Kα line** | EW = ____ keV | [ ] | [ ] | [ ] |
  | **Photon index** | Γ = ____ | [ ] | [ ] | [ ] |
  | **Hardness ratio** | HR = ____ | [ ] | [ ] | [ ] |
  | **X-ray/Hα ratio** | log(L_X/L_Hα) = ____ | [ ] | [ ] | [ ] |
  | **X-ray/L_IR** | log(L_X/L_IR) = ____ | [ ] | [ ] | [ ] |
  | **BPT class** | ______ | [ ] | [ ] | [ ] |
  | **SED AGN fraction** | ______ | [ ] | [ ] | [ ] |

- [ ] **Consensus determination**
  - Number of diagnostics indicating AGN: ____/7
  - Number of diagnostics indicating SB: ____/7
  - Number ambiguous: ____/7

- [ ] **Confidence level assigned**
  - [ ] **Very High (>95%)**: ≥5 diagnostics agree on classification
  - [ ] **High (80-95%)**: 3-4 diagnostics agree; 1-2 ambiguous
  - [ ] **Moderate (60-80%)**: Mixed signals; 2-3 diagnostics each direction
  - [ ] **Low (<60%)**: Majority ambiguous or conflicting

- [ ] **Final classification**
  - Source type: [ ] **AGN-dominated**, [ ] **Composite**, [ ] **Starburst-dominated**
  - Confidence: ______ (very high/high/moderate/low)
  - Justification: _______________________________________________________

### Systematic Uncertainty Summary
- [ ] **Uncertainties propagated and documented**
  - X-ray luminosity uncertainty: ±____ dex
  - Optical luminosity uncertainty: ±____ dex
  - Infrared luminosity uncertainty: ±____ dex
  - Combined multi-wavelength uncertainty: ±____ dex

- [ ] **Principal systematic sources identified**
  - [ ] Dust extinction correction (optical)
  - [ ] X-ray flux calibration
  - [ ] Distance/redshift determination
  - [ ] SED fitting model degeneracies
  - [ ] Spectral fitting degeneracies

---

## Stage 6: Documentation and Reporting

### Analysis Report Components
- [ ] **Source identification section**
  - Object name, coordinates (RA, Dec)
  - Redshift (z = ____), distance (D_L = ____ Mpc)

- [ ] **Data description section**
  - X-ray: Instrument, exposure, net counts (source, background)
  - Optical: Wavelength coverage, spectral resolution, S/N
  - Infrared: Bands available, flux errors

- [ ] **X-ray spectral analysis section**
  - Best-fit model and parameters (with uncertainties)
  - Goodness of fit (χ²/dof)
  - Important spectral features (Fe Kα, absorption, reflection)

- [ ] **Diagnostic results section**
  - Each diagnostic applied documented
  - Quantitative thresholds compared
  - Classification from each method

- [ ] **Multi-wavelength integration section**
  - SED fit results (if applicable)
  - Component decomposition (L_AGN, L_SB, SFR)
  - AGN fraction and uncertainties

- [ ] **Final classification and confidence section**
  - Evidence summary table
  - Final source type determination
  - Confidence level justification
  - Alternative interpretations discussed

- [ ] **References section**
  - Diagnostic thresholds cited (Yan et al. 2011, Panessa et al. 2012, etc.)
  - Observational facilities and pipelines cited
  - Calibration references (extinction, luminosity distance conventions)

### Data Products to Archive
- [ ] **X-ray spectrum and fit** (ASCII or FITS format)
- [ ] **Best-fit spectral model parameters** (table)
- [ ] **SED and best-fit model** (plot + data)
- [ ] **BPT diagram with source marked** (plot)
- [ ] **Diagnostic comparison table** (summary)
- [ ] **Spectral energy distribution components** (plot showing AGN, starburst, dust)

---

## Stage 7: Advanced Analysis (Optional)

### Detailed Spectral Decomposition
- [ ] **Four-component X-ray model tested** (for composite systems)
  - Direct AGN power-law
  - Scattered AGN power-law
  - Thermal starburst component
  - Power-law XRB population
  - [ ] Model improvement > 9 Δχ²: Justifies additional parameters

### High-Resolution X-ray Spectroscopy (If Available)
- [ ] **XRISM or similar high-resolution data analyzed**
  - Fe Kα line profile: Broad + narrow components?
  - Resonance line absorption from ionized wind?
  - Oxygen and nitrogen lines diagnostic for ionization

### Temporal Variability (If Multiple Observations)
- [ ] **X-ray variability on source timescale analyzed**
  - Flux variations: ____% amplitude
  - Variability timescale: ____ (hours/days/years)
  - Interpretation: AGN (variable) vs. starburst (stable)

---

## Quality Assurance Checklist

### Results Verification
- [ ] **Uncertainties realistic** (compare to similar published sources)
- [ ] **Physical parameters sensible**
  - Photon index 1.5-2.5 (within AGN/starburst range)
  - N_H < 10²⁶ cm⁻² (beyond Compton-thick regime)
  - Temperature (if thermal) 0.1-10 keV

- [ ] **Cross-checks performed**
  - X-ray flux consistency across energy ranges
  - Optical emission-line ratios consistent with literature BPT
  - Infrared SED shape consistent with source type

- [ ] **Comparison to sample** (if part of larger study)
  - [ ] Parameters within expected ranges
  - [ ] Classification consistent with population trends

### Peer Review Readiness
- [ ] **Analysis reproducible**
  - [ ] All software versions documented
  - [ ] Calibration files identified
  - [ ] Fitting procedure clearly described

- [ ] **Figures publication-quality**
  - [ ] Spectra with best-fits clearly displayed
  - [ ] Residuals shown below main plot
  - [ ] All axes labeled with units
  - [ ] Error bars visible (if data allow)

- [ ] **Tables clear and complete**
  - [ ] All parameters with uncertainties
  - [ ] Thresholds and classifications explicitly stated
  - [ ] Footnotes explain abbreviations

---

## Common Pitfalls to Avoid

- [ ] **Do NOT assume X-ray luminosity alone** determines AGN (many starbursts bright in soft X-rays)
- [ ] **Do NOT ignore dust extinction** in optical diagnostics (30-40% misclassification if uncorrected)
- [ ] **Do NOT apply single diagnostic** to composite systems (minimum 2-3 independent methods required)
- [ ] **Do NOT forget background subtraction** in spectral fits (can bias photon index by >0.3)
- [ ] **Do NOT trust SED fits without input verification** (garbage in, garbage out; inspect photometry)
- [ ] **Do NOT neglect uncertainties** in classification (report confidence levels)
- [ ] **Do NOT mix spectral definitions** (0.5-2 keV vs. 0.3-8 keV have very different interpretations)
- [ ] **Do NOT assume high S/N** (source > 500 counts) if soft source contamination or pile-up present

---

## Recommended Literature References to Cite

For methods documentation, cite:

**X-ray Spectroscopy**:
- Ptak & Griffiths (1999) - Canonical spectral models
- Ho (2008) - AGN/LINER distinction

**Diagnostic Ratios**:
- Yan et al. (2011) - X-ray/Hα ratio thresholds
- Panessa et al. (2012) - Optical misclassification problem

**SED Fitting**:
- Menanteau et al. (2007) - Multi-component decomposition
- CIGALE papers - SED fitting tools

**Survey Context**:
- Brandt & Alexander (2015) - Comprehensive X-ray survey review
- Lançon et al. (2022) - Modern eFEDS AGN survey

---

## Final Sign-Off

- [ ] **All analysis stages completed**
- [ ] **Results internally consistent**
- [ ] **Uncertainties properly estimated**
- [ ] **Classification well-justified**
- [ ] **Documentation complete**
- [ ] **Ready for publication or archival**

**Analysis Completed**: _________________ (date)
**Analyst**: _________________________________
**Verification (if applicable)**: _________________________

---

**This checklist ensures comprehensive, reproducible AGN vs. starburst X-ray analysis.**

