# Comprehensive Literature Review: AGN Classification Techniques Distinguishing AGN from Star-Forming Galaxies

**Compiled:** December 22, 2025
**Focus Areas:** (1) BPT/WHAN diagrams, (2) Multi-wavelength SED fitting, (3) Mid-IR color diagnostics, (4) X-ray/radio methods, (5) Machine learning approaches

---

## 1. OVERVIEW OF THE RESEARCH AREA

The distinction between active galactic nuclei (AGN) and star-forming galaxies (SFGs) is fundamental to understanding galaxy evolution, black hole accretion, and feedback mechanisms in the universe. AGN are compact regions at galactic centers emitting significant energy across the electromagnetic spectrum through accretion onto supermassive black holes, distinct from stellar emission in non-active galaxies. Over the past four decades, astronomers have developed multiple complementary classification schemes to reliably separate AGN from SFGs, each with distinct advantages, limitations, and applicability across wavelengths and redshifts.

The challenge of AGN identification is multifaceted:
- **Composite systems** where both AGN and star formation coexist require careful decomposition
- **Obscured AGN** hidden by dust require non-optical diagnostics
- **LINER systems** present ambiguity regarding true AGN activity versus evolved stellar populations
- **High-redshift sources** where diagnostic lines shift out of optical windows or lose discriminatory power

This review synthesizes five major classification approaches: optical emission-line diagnostics, mid-infrared color selection, spectral energy distribution (SED) fitting, multi-wavelength (X-ray/radio) selection, and machine learning techniques. Each method exploits different physical properties of AGN versus SFGs, and the modern approach combines multiple independent diagnostics to maximize accuracy and minimize selection biases.

---

## 2. CHRONOLOGICAL DEVELOPMENT OF MAJOR CLASSIFICATION TECHNIQUES

### 2.1 Optical Emission-Line Diagnostics (1981–Present)

**Foundation (1981):** Baldwin, Phillips & Terlevich (BPT) introduced the first systematic optical diagnostic using emission line ratios. The BPT diagram employs logarithmic ratios of strong nebular emission lines—specifically [N II] λ6584 / Hα and [O III] λ5007 / Hβ—to classify galaxies based on dominant ionization mechanisms.

**Theoretical Basis:** The BPT methodology rests on photoionization models and stellar population synthesis. Galaxies are classified as:
- **Star-forming:** Ionization by hot young stars (effective temperature >40,000 K) produces lower [O III]/Hβ and [N II]/Hα ratios
- **Seyfert galaxies:** AGN photoionization produces higher line ratios
- **LINER (Low-Ionization Nuclear Emission-line Regions):** Weak AGN or alternative ionization sources

**Maximum Starburst Line:** Theoretical models define an empirical demarcation curve above which AGN dominance is indicated. The Kauffmann et al. (2003) empirical line divides pure star-forming galaxies from Seyfert–H II composites.

**Strengths:**
- Physically motivated, based on ionization physics
- Simple to apply with optical spectra
- Well-calibrated for local universe (z < 0.3)
- Widely adopted across surveys (SDSS, 2dF, etc.)

**Limitations:**
- Requires four emission lines (often difficult for faint galaxies)
- Degeneracies at high ionization (composite systems)
- Metallicity-dependent (unreliable for extreme compositions)
- Loses effectiveness at z > 0.5 where diagnostic lines shift into IR or out of spectral coverage

**Recent Refinements (2019–2025):**
- Spatially resolved mapping provides continuous AGN activity measures rather than binary classification
- Advanced statistical methods quantify perpendicular distance from star-forming ridge as AGN indicator
- Alternative diagnostic schemes (VO87, O3N2) for specific applications

### 2.2 The WHAN Diagram (2011–Present)

**Introduction:** Cid Fernandes et al. (2011) introduced the WHAN diagram, combining the [N II]/Hα line ratio with the equivalent width of Hα (W_Hα), addressing critical limitations of the BPT approach.

**Methodology:** The WHAN diagram identifies five distinct classes:
1. **Pure star-forming galaxies:** log [N II]/Hα < -0.4 and W_Hα > 3 Å
2. **Strong AGN (Seyferts):** log [N II]/Hα > -0.4 and W_Hα > 6 Å
3. **Weak AGN:** log [N II]/Hα > -0.4 and 3 Å < W_Hα < 6 Å
4. **Retired galaxies (RGs):** W_Hα < 3 Å
5. **Passive systems:** No measured emission

**Key Advantages:**
- Applicable to **weak emission-line galaxies** (WELGs) not classifiable by BPT
- Distinguishes **two distinct LINER populations** (weak AGN vs. retired galaxies with ionization from post-AGB stars)
- Most inclusive diagnostic, as only requires Hα and [N II] measurement
- Cost-effective for large surveys

**Limitations:**
- Requires reliable Hα equivalent width measurements (W_Hα sensitive to underlying stellar absorption)
- Sensitivity to reddening affects [N II]/Hα determination
- Distinction between weak AGN and composite systems remains ambiguous

**Applications:** The WHAN diagram has become standard in galaxy surveys including CALIFA, MaNGA, and large spectroscopic catalogues, particularly for distinguishing evolved stellar ionization from true AGN activity.

### 2.3 Mid-Infrared Color Selection (2010s–Present)

**Physical Basis:** AGN-heated dust in the mid-infrared (3–100 μm) exhibits distinct spectral properties from stellar and star-forming galaxy emission:
- AGN produce a rising red continuum due to dust heated by AGN power-law UV spectrum
- This extreme UV environment dissociates PAH molecules, suppressing PAH emission in AGN
- Star-forming galaxies show strong PAH features and modified blackbody stellar continua

**WISE-Based Selection (Stern et al. 2012; Assef et al. 2013):**
- Simple criterion: W1 − W2 ≥ 0.8 ([3.4] − [4.6] μm ≥ 0.8, Vega magnitudes)
- Identifies ~61.9 ± 5.4 AGN candidates per deg² to W2 ≈ 15.0
- Detects both unobscured (Type 1) and heavily obscured (Type 2, Compton-thick) AGN
- Less susceptible to dust extinction than optical/soft X-ray surveys

**Spitzer/IRAC Diagnostics:**
- Mid-infrared colors separate AGN from stars and star-forming galaxies in [3.6]–[4.5]–[5.8]–[8.0] μm space
- Power-law AGN spectra distinctly offset from blackbody stellar spectra
- Diagnostic relies on rising mid-IR continuum in AGN vs. declining in normal galaxies

**Advantages:**
- Wavelength regime relatively unaffected by dust obscuration
- Rapid survey of large areas (WISE ~20,000 deg² sky coverage)
- Identifies obscured AGN missed by optical/soft X-ray surveys
- Complementary to optical diagnostics

**Limitations:**
- Confusion with dust-rich star-forming galaxies (SFGs with high specific star formation rates)
- Contamination from highly luminous infrared galaxies (LIRGs/ULIRGs) undergoing starburst
- Mid-IR selection biased against low-accretion AGN with weak IR bump
- Requires multi-band photometry; single-band colors insufficient

**Refinements (2020s):** Recent work combines optical and infrared colors in unified diagnostic frameworks to reduce SFG contamination while maintaining AGN completeness.

### 2.4 Spectral Energy Distribution Fitting and Decomposition (1990s–Present)

**Foundation:** Multi-component SED fitting models galaxy integrated luminosity across UV-to-radio wavelengths using physically motivated templates for stars, AGN accretion disk, dusty torus, and circumnuclear dust.

**Key Component Models:**

**(1) Stellar Population Synthesis:**
- Templates from libraries (e.g., Bruzual & Charlot 2003) covering ages, metallicities, dust attenuation
- Constrains star-formation history and stellar mass

**(2) AGN Accretion Disk (Hot Corona):**
- Power-law continuum in UV-optical (νL_ν ~ ν^(-α))
- Thermal X-ray emission from hot corona (kT ~ 100 keV)
- Ionizing continuum drives emission-line regions

**(3) Dusty Torus Models:**
Major frameworks:

- **Clumpy 2-phase models (Nenkova et al. 2008):** Dust distributed as high-density clumps in low-density medium, modeled with SKIRT radiative transfer code
- **CAT3D/Hönig & Kishimoto (2017):** Separates silicate and graphite sublimation temperatures, includes polar wind component
- **Theseus/Sirocco:** Self-consistent radiative equilibrium models
- **SKIRTOR:** Implements 3D radiative transfer for clumpy AGN tori

**Key Spectral Features:**
- **Silicate absorption at 9.7 μm:** Indicates obscuration, stronger for Type 2 AGN
- **Silicate emission at 9.7 μm:** Rare, observed in unobscured Type 1 AGN
- **PAH emission (6–15 μm):** Suppressed in AGN-dominated regions, enhanced in star-forming regions
- **Infrared bump (30–100 μm):** Thermal emission from heated torus dust

**Decomposition Methodology:**

Modern approaches (CIGALE, MAGPHYS, AGNfitter-rx, Prospector) perform χ² minimization fitting:
1. Create synthetic SED by combining templates with free parameters (stellar mass, AGN luminosity, dust properties, redshift)
2. Compare to observed photometry across UV, optical, IR, sub-mm, radio
3. Marginalize over parameter space to extract physical quantities with uncertainties
4. For composite systems: simultaneously fit stellar + AGN + dust components

**Key Findings on AGN/Host Galaxy Decomposition:**

- **Stellar mass recovery:** ±40% systematics, relatively insensitive to photometric band selection if UV-MIR data available
- **AGN bolometric luminosity:** More uncertain, depends strongly on IR coverage and dust geometry assumptions
- **Star-formation rate:** Highly uncertain for AGN hosts due to AGN heating of dust; SFR indicators (FIR, Hα) contaminated by AGN

**Advantages:**
- Physically motivated; constrains multiple components simultaneously
- Captures broad-band photometric data across entire SED
- Works for heavily obscured AGN (unlike optical/soft X-ray)
- Provides bolometric luminosity and physical properties

**Limitations:**
- **Degeneracies:** Multiple models can fit identical SED equally well (non-unique solutions)
- **Model-dependent:** Results depend on template libraries, dust model assumptions
- **Composite systems:** Single AGN+starburst model may be oversimplified for complex morphologies
- **Systematic uncertainties:** Typically ±0.3–0.5 dex in derived quantities
- **Photometric redshift dependency:** Incorrect z destroys decomposition

**Modern Extensions (2020s–2025):**
- Inclusion of X-ray/radio components in fitting (AGNfitter-rx)
- Spatially resolved SED fitting for nearby galaxies
- Bayesian hierarchical models to constrain population-level parameters
- Integration with cosmological simulations to test predictions

### 2.5 X-ray Selection Methods (1970s–Present)

**Foundational Principle:** Hard X-ray emission (2–10 keV) is a robust AGN signature due to inverse-Compton scattering of UV photons off hot electrons in the AGN corona. X-ray emission is unaffected by moderate dust obscuration and does not require specific redshift or emission-line accessibility.

**Key Fact:** All compact X-ray sources above ~10^42 erg s^-1 (2–10 keV) are classified as AGN; vast majority of hard (>2 keV) point-like sources are AGN.

**X-ray Spectroscopy Methods:**

**(1) Hardness Ratio Analysis (Chandra/XMM-Newton):**
- HR1 (2–4 keV / 1–2 keV) and HR2 (4–16 keV / 2–4 keV) optimized for AGN selection
- **Unobscured AGN:** Soft in HR1 (photospheric X-ray emission), hard in HR2 (power-law continuum)
- **Obscured AGN:** Soft in HR1 (scattered/reflected light from circumnuclear material), hard in HR2 (absorption depresses soft band)
- **Method success:** Identified 61% of moderate-redshift (z~1) Chandra Deep Field-North sample with N_H > 10^23 cm^-2

**Limitations of hardness ratios:**
- Simple absorbed power-law assumption breaks down; intrinsic spectra often complex
- In deep surveys, majority of sources show spectra inconsistent with simple models
- Requires sufficient photons; harsh on faint sources

**(2) Spectral Fitting with XSPEC/Sherpa:**
- Full X-ray spectroscopy modeling: power-law continuum + Compton reflection + absorption
- Constrains absorbing column density N_H (sensitive to Compton-thick AGN)
- Fits iron Kα emission line (6.4 keV) and its equivalent width
- Uncertainties typically ±0.2–0.3 dex in N_H

**Obscured and Compton-Thick AGN Detection:**

**Definition:** Compton-thick AGN have column densities N_H ≥ 1.5 × 10^24 cm^-2 (inverse Thomson cross-section)

**Signatures in X-ray spectra:**
- Strong reflection component at E > 10 keV
- Prominent Fe Kα line at 6.4 keV (EW typically > 1 keV)
- Downscattered continuum depressed across X-ray band
- For heavily Compton-thick sources: entire X-ray continuum depressed by Compton recoil

**Detection strategies:**
1. **High signal-to-noise X-ray spectra:** Identify characteristic Compton-thick signatures
2. **Combined IR+X-ray approach:** Select sources bright in mid-IR (L_MIR > 10^11 L_sun) but weak/undetected in X-rays
3. **Reflection-dominated selection:** Objects with reflection-to-continuum ratio >> 1 (non-physical power law, implies strong absorption)

**Cosmic significance:** Large fraction of local AGN universe are Compton thick; AGN synthesis models require sizable population of mildly Compton-thick sources to match X-ray background intensity peak at ~30 keV.

**Multi-wavelength X-ray Surveys:**
- **Chandra:** 4 Ms CDF-N/S, COSMOS, AEGIS (sub-arcsec resolution, high sensitivity to faint sources)
- **XMM-Newton:** Bright serendipitous survey, XXL survey (larger area, lower sensitivity)
- **Swift/BAT, Suzaku, NuSTAR, eROSITA:** Hard X-ray (>10 keV) sensitive to heavily obscured AGN

### 2.6 Radio Selection Methods (1970s–Present)

**Physical Basis:** Radio emission in AGN arises from relativistic jets powered by accretion, producing synchrotron radiation. Star-forming galaxies emit radio via free-free emission from H II regions and supernova remnants. Key distinction: AGN radio typically power-law (flat to inverted spectra), SFG radio typically steep-spectrum.

**Radio-Loudness Classification:**
- **Radio-loud AGN (RL):** L_radio / L_optical (at rest 5 GHz / 4400 Å) >> 1; jetted AGN
- **Radio-intermediate:** 1 < L_radio / L_optical < 10
- **Radio-quiet (RQ):** L_radio / L_optical < 1; majority of AGN, accretion-powered jets weak

**Infrared-Radio Correlation (IRC):**

For star-forming galaxies, tight empirical correlation between total infrared luminosity (L_TIR, 8–1000 μm) and 1.4 GHz radio luminosity:
- Parameter q: q_{1.4 GHz} = log(L_TIR / (4π D_L^2)) - log(L_radio / (1 W Hz^-1))
- Local universe: q ≈ 2.34 (varies weakly with redshift, SFR)
- Physical origin: radio traces recent star formation via supernova-powered synchrotron; IR traces dust heated by young stars
- Remarkably tight: scatter only ~0.3 dex

**Applications:**
- **Radio-excess AGN identification:** Measure deviation (ΔR) from SFR-predicted radio luminosity
- **Low-luminosity AGN:** Often radio-excess even if optically/X-ray undetectable

**Spectral Index Analysis:**
- Radio spectral slope α: S_ν ∝ ν^(-α)
- SFG radio: steep spectrum (α ~0.7–1.0) from primary/secondary cosmic rays
- AGN radio: flat to inverted (α ~-0.5–0.5) from power-law jet continuum
- High-frequency observations (GHz) separate steep-spectrum SFG from flat AGN

**Modern Radio Surveys:**
- **VLA 3 GHz (Smolčić et al. 2017):** COSMOS field, L_radio ~ 10^28 W Hz^-1, z < 6
- **LOFAR Low-Frequency Sky Survey:** 144 MHz, detecting steep-spectrum sources efficiently
- **Radio Spectral Index & Morphology:** Combines frequency-dependent properties to classify jets vs. star formation

**Advantages:**
- Unaffected by dust obscuration
- Detects low-accretion AGN with weak accretion disks
- Large sky coverage (surveys cover thousands of deg²)

**Limitations:**
- Lower angular resolution than X-ray/optical (confusion with star-forming regions)
- Radio jets can be extended (minutes to hours of arc), complicating multi-wavelength matching
- Star-forming galaxies at high SFRs produce bright radio, creating confusion with AGN

---

## 3. MID-INFRARED COLOR DIAGNOSTICS: DETAILED ANALYSIS

### 3.1 WISE Color Spaces

The Wide-Field Infrared Survey Explorer (WISE) provides all-sky photometry in four bands: W1 (3.4 μm), W2 (4.6 μm), W3 (12 μm), W4 (22 μm).

**W1−W2 Selection:**
- Metric: W1 − W2 ≥ 0.8 (Vega magnitudes)
- Physical reason: AGN have rising mid-IR continuum (power-law), shifting flux to longer wavelengths; stars/galaxies have blackbody/modified blackbody decreasing with wavelength
- AGN surface density: 61.9 ± 5.4 deg^-2 to W2 = 15 mag (limit ~2 mJy)
- Purity: ~90% for unobscured AGN, lower for composite systems
- Completeness: ~75% of Type 1 AGN, ~60% of Type 2 AGN (dust-heavy, cooler torus)

**W1−W2 vs. W2−W3:**
- Dual-color selections provide improved AGN/SFG separation
- (W1−W2, W2−W3) color-color diagram distinguishes:
  - **AGN:** (W1−W2) > 0.8 and (W2−W3) < 0.2 to 0.5
  - **Star-forming galaxies:** (W1−W2) < 0.5 and (W2−W3) > 1.0
  - **Composites:** Intermediate regions

### 3.2 Spitzer/IRAC Color Selection

Spitzer Space Telescope infrared array camera (IRAC) provides higher-resolution, deeper photometry than WISE in overlapping bands plus unique 5.8 μm access.

**[3.6]−[4.5] Selection:**
- Identifies z < 1 AGN with high efficiency
- AGN rise from shorter to longer wavelengths; SFGs flatten or decline
- Lower contamination than WISE at shallow depths

**[3.6]−[4.5] vs. [5.8]−[8.0]:**
- Distinguishes dusty star-forming galaxies (strong PAH at 7.7 μm) from AGN (PAH-suppressed)
- Effective at high redshift (z > 2) where PAH features enter observed bands

**Systematic improvements:**
- IRAC color cuts refined to achieve >85% purity and >75% completeness for Type 1 AGN
- Accounting for stellar contamination critical at faint flux levels

### 3.3 Silicate and PAH Spectral Features

**Spitzer/IRS Spectroscopy (5–40 μm):**

Core diagnostic features for AGN identification:

**(1) Silicate 9.7 μm Feature:**
- **Type 2 AGN:** Strong absorption feature (silicate optical depth τ_sil ~ 1–3)
- **Type 1 AGN:** Often weak or in emission
- Equivalent width >0 indicates significant dust column along line-of-sight
- Feature shape (broad, narrow) constrains dust grain size distribution

**(2) PAH Features (6, 7.7, 11.2, 12.6 μm):**
- **Star-forming galaxies:** Strong 7.7 and 11.2 μm PAH emission (polycyclic aromatic hydrocarbons)
- **AGN:** Suppressed PAH (hot AGN radiation field destructs large PAHs)
- **Ratio L_PAH / L_continuum:** ~0.3–0.5 for SFGs, <0.1 for AGN-dominated

**(3) Neon Fine-Structure Lines:**
- **[Ne II] 12.8 μm, [Ne III] 15.6 μm:** Ionization diagnostics
- [Ne III] / [Ne II] ratio indicates ionization parameter and AGN contribution
- Permits decomposition even for composite systems

**Modern Applications:**
- AKARI, Herschel, SOFIA carry IRS-quality spectroscopy to higher-z
- JWST/MIRI spectroscopy reveals silicate, PAH, and fine-structure diagnostics to z > 3
- Spectral decomposition quantifies AGN vs. starburst contributions even in heavily mixed systems

---

## 4. SPECTRAL ENERGY DISTRIBUTION FITTING: DETAILED METHODOLOGY

### 4.1 SED Fitting Codes and Templates

**Major SED Fitting Frameworks:**

**(1) CIGALE (Code Investigating GALaxy Emission):**
- Bayesian framework fitting UV-to-submm photometry
- Component templates: stellar populations (Bruzual & Charlot, Maraston), AGN torus (Nenkova, CAT3D), dust attenuation (Calzetti, SMC/LMC)
- Output: PDFs of stellar mass, SFR, AGN luminosity, dust properties
- Widely adopted for large surveys; ~10,000+ papers cite methodology

**(2) MAGPHYS (Multi-wavelength Analysis of Galaxy Physical Properties):**
- Energy-conserving approach: ensures energy absorbed by dust re-emitted in IR
- Liberal template libraries (105+ combinations)
- Excellent for UV-submm data; less successful with sparse photometry

**(3) AGNfitter/AGNfitter-rx:**
- Explicitly models AGN accretion disk + host galaxy + torus + radio core
- AGNfitter-rx extension: includes soft X-ray (0.5–2 keV) and hard X-ray (2–10 keV) flux
- Decomposes multi-wavelength SEDs from radio to X-ray
- Output: AGN bolometric luminosity, accretion rate, host properties

**(4) Prospector:**
- Flexible Bayesian fitting with flexible parameterization of stellar populations
- Hierarchical modeling for population studies
- Computationally intensive but yields robust posteriors

### 4.2 AGN Component Modeling

**Accretion Disk (Disk/Nuclear) Component:**
- Models: \
  - Power-law continuum with Comptonization (hot corona at ~100 keV)
  - Shakura-Sunyaev thin-disk model (effective T ≈ 10^4 K)
  - Eddington ratio (L_bol / L_Edd) controlling disk temperature
- Constrains: AGN bolometric luminosity, accretion rate, black hole mass (if width of broad Hα measured)

**Torus/Obscuring Component:**
- Models tested: Nenkova et al. (2008), CAT3D (Hönig & Kishimoto), Theseus, SKIRTOR
- Free parameters:
  - Torus opening angle θ (0° = edge-on, 90° = face-on)
  - Clump optical depth τ_V
  - Number of clumps
  - Dust composition (silicate/graphite ratio)
- Output: Torus luminosity, viewing angle, dust mass

**Physical Insights from Torus Models:**
- **Edge-on Type 2 galaxies:** Deep silicate absorption, hot (emission at short wavelengths suppressed)
- **Face-on Type 1 galaxies:** Weak silicate feature, broad IR bump (torus directly viewed from above)
- **Intermediate angles:** Mixed features; crucial for understanding obscuration geometry

### 4.3 Decomposition in Composite Systems

**Challenge:** Galaxies with simultaneous intense AGN and starburst activity (e.g., ultraluminous infrared galaxies, ULIRGs) require careful multi-component fitting.

**Approach:**
1. **Bolometric luminosity balance:** L_total = L_stars + L_AGN + L_starburst
2. **Spectral dominance:** Different wavelengths sensitive to different components:
   - UV-optical: stellar + AGN ionizing continuum
   - Mid-IR (5–20 μm): AGN torus + star-forming dust (competing)
   - Far-IR (50–500 μm): cold dust from diffuse ISM + star formation
   - Submm (>500 μm): cold dust + synchrotron (low contribution)
3. **Parametric decomposition:** Allow independent SFRs and AGN fractions as free parameters

**Quantitative Example:**
- ULIRG with L_bol = 10^13 L_sun: might be decomposed as L_AGN = 3 × 10^12 L_sun (30%), L_SFG = 7 × 10^12 L_sun (70%)
- Different torus models yield 20–40% variation in AGN fractions due to degeneracies
- Systematic uncertainties in AGN contribution: typically ±0.2–0.5 dex

### 4.4 Multi-wavelength Integration

**UV-Optical-IR Data:**
- **Ultraviolet (0.1–0.4 μm):** Traces ionizing continuum, stellar population age, dust reddening
- **Optical (0.4–1.0 μm):** Stellar continuum, emission lines (broad + narrow), dust reddening
- **Near-IR (1–5 μm):** Stellar populations (old stars dominate), AGN hot dust begins
- **Mid-IR (5–40 μm):** Warm dust from AGN torus, star-forming dust; PAH diagnostics
- **Far-IR (40–500 μm):** Cold dust heated by star formation and AGN; SFR indicator
- **Submm-Radio:** Synchrotron from star-forming regions and AGN jets

**Photometric Challenges:**
- **Spatial resolution variation:** Radio images may be 10–100 arcsec; X-ray ~1 arcsec; optical ~0.5 arcsec
  - Solution: Aperture corrections or matched-aperture photometry
- **Variability across wavelengths:** AGN vary (optical, X-ray); SFR averaged (FIR)
  - Solution: Multi-epoch observation or use integrated indicators
- **Redshift dependency:** Rest-frame spectral shape changes with z
  - Solution: Photometric redshift or spectroscopic z required

---

## 5. X-RAY AND RADIO MULTI-WAVELENGTH SELECTION

### 5.1 X-Ray Selection Efficiency and Biases

**AGN Identification Rates by X-Ray Flux:**
- Hard X-ray (2–10 keV) selection: ~95% pure for L_X > 10^42 erg s^-1
- Soft X-ray (0.5–2 keV): More contamination from star-forming galaxies and hot ISM in ellipticals
- Typical X-ray surveys (Chandra 4 Ms): L_X limit ~10^41 erg s^-1, detecting >90% of local AGN

**Selection Biases:**
- **Type 1 AGN preferentially selected:** Unobscured AGN with direct view to X-ray source ~10× more luminous in X-ray than Type 2
  - Correction: Apply simulations (e.g., AGNsed) to account for obscuration bias
- **Low-accretion AGN underrepresented:** Weak X-ray emission despite AGN presence
  - Solution: Combine with radio/mid-IR to flag X-ray-weak AGN

### 5.2 Radio Luminosity-Star Formation Rate Relation

**Empirical IRC for SFGs:**
- q(z) = log(L_TIR [W Hz^-1]) − log(L_1.4 GHz [W Hz^-1]) − 12
- Local value: q ≈ 2.34 ± 0.26 (defined with L_TIR / (4π D_L^2) convention)
- Evolution: q(z) decreases ~−0.15 ± 0.03 per unit redshift (z < 3)
  - Interpretation: AGN-heated dust reduces apparent IR flux at fixed SFR as AGN fraction increases

**Radio-Based SFR Indicator:**
- From IRC: SFR ~ L_1.4 GHz / A_{radio}
- Where A_{radio} ≈ 3 × 10^21 W Hz^-1 (SFR per unit radio power)
- Advantage: Unaffected by dust extinction
- Limitation: Assumes no AGN contribution to radio

**Radio-Excess AGN:**
- Definition: ΔR = log(L_radio,observed) − log(L_radio,predicted from SFR)
- Threshold: ΔR > 0.3–0.5 indicates AGN contribution
- Application: Identifies low-luminosity AGN (LLAGN) with weak or no optical/X-ray AGN signatures
- Limitations: Requires accurate SFR estimate; contamination from merger-induced turbulence (enhances non-AGN radio)

### 5.3 LOFAR and VLA Multi-Wavelength Surveys

**LOFAR Window on AGN/SFG (Gürkan et al. 2018, Sabater et al. 2021):**

**Low-frequency radio properties:**
- 144 MHz observations sensitive to steep-spectrum synchrotron (both SFG and older AGN jets)
- Curvature in radio SED (S_ν vs. ν) differs between AGN and SFGs:
  - SFG: weak spectral curvature, attributed to cosmic ray aging and/or free-free absorption
  - AGN: pronounced curvature, jetted sources showing flattening at low frequencies

**IR-Radio Correlation at Low Frequencies:**
- At z = 0.5–2.5, radio spectral index correlates with stellar mass and AGN fraction
  - AGN-dominated systems: shallower spectral index (flatter radio), lower q-parameter
  - SFG-dominated: steeper radio spectrum, q ≈ 2.3–2.4

**Classification Accuracy:**
- Using only radio SED shape + IR photometry: ~75–80% correct AGN/SFG classification
- Addition of X-ray data raises to >90%

**VLA-COSMOS 3 GHz Survey (Smolčić et al. 2017):**

**Sample:**
- 3,769 radio sources detected at 3 GHz (sensitivity 2.3 μJy beam^-1)
- Redshift range z < 6; 80% spectroscopic or photometric redshifts
- Multi-wavelength cross-matching: X-ray (Chandra), mid-IR (MIPS/PACS), optical (COSMOS photometry)

**Classification Scheme:**
1. X-ray detected and hard → AGN
2. X-ray detected, soft, high L_IR → Starburst or composite
3. X-ray undetected but radio-excess → Low-luminosity AGN
4. FIR-radio correlation consistent, no X-ray excess → Star-forming galaxy

**Results:**
- AGN fraction: ~30% at z ~ 1, ~50% at z ~ 4
- Radio-excess AGN (no X-ray): ~15–20% of AGN population
- Classification completeness: ~85% (remaining sources ambiguous composites)

---

## 6. MACHINE LEARNING APPROACHES FOR AGN CLASSIFICATION

### 6.1 Supervised Learning Techniques

**Supervised models tested (Fermi LAT AGN classification; Ighina et al. 2023):**

Classification methods applied:
- **Support Vector Machines (SVM):** Optimal hyperplane separation in feature space
- **Random Forests:** Ensemble of decision trees with bootstrap aggregation
- **Extreme Gradient Boosting (XGBoost):** Sequential tree refinement minimizing loss
- **Artificial Neural Networks (ANNs):** Multi-layer perceptron with backpropagation
- **Convolutional Neural Networks (CNNs):** Hierarchical feature extraction from images

**Performance on Fermi LAT AGN Classification (Ighina et al. 2023):**
- **Best performing:** SuperLearner (meta-algorithm combining MARS regression + Random Forests)
- **Overall accuracy:** 91.1% (kNN imputation), 91.2% (MICE imputation)
- **Interpretation:** Missing data handling critical; imputation method affects performance ~0.1%

**Photometric AGN-SFG Classification (Polkas et al. 2023):**

Machine learning diagnostic tool using optical + infrared colors:
- **Training data:** SDSS spectroscopic AGN/SFG sample (z < 0.3)
- **Features:** [u−g], [g−r], [r−i], [i−z] optical colors; [3.6]−[4.5] IRAC color
- **Output classifications:** AGN, Star-forming, LINER, Composite, Passive

**Performance Metrics:**
- Overall accuracy: ~81%
- Per-category completeness:
  - Star-forming galaxies: 81%
  - AGN: 56%
  - LINER: 68%
  - Composite galaxies: 65%
  - Passive galaxies: 85%

**Interpretation:** Composite and LINER systems remain challenging due to inherent classification ambiguity (not unique true classification). Pure SFG and passive categories well-separated; moderate confusion between AGN and composite systems.

### 6.2 Convolutional Neural Networks for AGN Host Identification

**Study:** Guo et al. (2022) trained CNNs on 210,000+ SDSS galaxies to classify AGN hosts

**Methodology:**
- **Input:** SDSS imaging (g, r, i bands stacked as 3-channel images, ~64×64 pixels)
- **Architecture:** Standard CNN backbone (VGG, ResNet, or InceptionV3)
- **Training:** Supervised learning using SDSS spectroscopic AGN classification (BPT diagram)
- **Validation:** Subset of spectrally-classified composites

**Results:**
- **AGN/Non-AGN binary classification accuracy:** 78.1%
- **Feature interpretation:** CNN learns morphological indicators of AGN (e.g., central concentration, disturbed morphology, disk features)
- **Composite galaxies:** CNN predictions correlate with spectroscopic composite classification, suggesting morphological signatures of mixed activity

**Advantages over traditional methods:**
- Learns discriminative imaging features beyond human-crafted descriptors
- No requirement for spectroscopy
- Can operate on low-resolution imaging

**Limitations:**
- Requires large training dataset (>100k galaxies); limited by spectroscopic survey depth
- Overfitting risk if training set biased (e.g., overrepresentation of bright objects)
- Interpretability reduced compared to physics-based diagnostics

### 6.3 Multi-Band Machine Learning for Radio AGN Classification

**Study:** Swarup et al. (2024) & Banfield et al. (2015) applied machine learning to radio morphological classification

**Methods:**

**(1) Radio Morphological Classification (Banfield et al. 2015):**
- Radio image morphology used to distinguish:
  - FRI (Fanaroff-Riley type I): Edge-darkening, low-power jets
  - FRII (Fanaroff-Riley type II): Edge-brightening, high-power jets
  - Bent/distorted sources
- CNN trained on cutout images
- Accuracy: >90% on test set

**(2) Radio Spectral Index + Morphology (Swarup et al. 2024, RGC-Bent):**
- **Input features:**
  - Radio spectral index α (multi-frequency 150 MHz–1.4 GHz)
  - Morphological classification (straight vs. bent tails)
  - Peak flux density
- **Architecture:** ConvNeXT backbone
- **Performance:** State-of-the-art accuracy on bent radio galaxy classification

### 6.4 Missing Data and Imputation in AGN Classification

**Challenge:** Real survey data has incomplete coverage (not all galaxies have all wavelengths observed)

**Impact (Ighina et al. 2023):**
- Missing data hinders machine learning classification
- Typical SDSS+WISE+Chandra sample: ~30–50% sources missing mid-IR or X-ray data
- Naive approaches (listwise deletion) reduce sample by >50%

**Imputation Methods Tested:**
- **K-Nearest Neighbors (kNN) imputation:** Replace missing value with mean of k nearest neighbors in feature space
  - Performance: 91.1% accuracy with optimal k ≈ 5
  - Advantage: simple; preserves correlations
  - Disadvantage: biased toward training set statistics

- **Multivariate Imputation by Chained Equations (MICE):**
  - Iteratively model each missing variable as function of others
  - Performance: 91.2% accuracy; slightly better than kNN
  - Advantage: captures complex correlations
  - Disadvantage: computationally intensive

**Finding:** Imputation method impacts performance <0.1% for well-separated classes but >2% for rare classes (e.g., Compton-thick AGN).

### 6.5 Unsupervised Learning: Variable AGN Detection

**Study:** Fabbiano et al. (2019) combined optical variability with unsupervised self-organizing maps (SOMs)

**Methodology:**
- **Input:** Multi-epoch optical light curves (structure function, power spectral density)
- **Unsupervised clustering:** SOM learns natural groupings without labeled training data
- **Output:** Identified clusters corresponding to variable AGN, RR Lyrae stars, etc.

**Performance (Variable AGN Detection):**
- **Purity:** 86% (fraction of identified variables truly variable)
- **Completeness:** 66% (fraction of true variables recovered)
- **Comparison:** Supervised neural networks achieve ~85% purity, ~70% completeness on same sample

**Advantage:** Discovers novel variable classes not anticipated in training data; useful for exploratory analysis

---

## 7. COMPARATIVE PERFORMANCE AND METHODOLOGICAL TRADE-OFFS

### 7.1 Accuracy and Completeness by Method

| **Method** | **AGN Purity** | **Completeness (AGN)** | **SFG Contamination** | **Advantages** | **Limitations** |
|---|---|---|---|---|---|
| **BPT Diagram** | ~85–90% | ~75–80% | ~10–15% | Well-calibrated; physics-motivated; fast | Requires 4 emission lines; metallicity-dependent; z > 0.5 problematic |
| **WHAN Diagram** | ~80–85% | ~70–75% | ~15–20% | Handles weak emitters; distinguishes RG from AGN | W_Hα sensitive to stellar absorption; still requires Hα |
| **WISE W1−W2 color** | ~90% (Type 1), ~60% (Type 2) | ~75% | ~10% (Type 1), ~30% (Type 2) | Fast; all-sky; detects obscured AGN | Misses cool-torus AGN; confused with LIRG |
| **SED Fitting** | ~85–92% | ~80–88% | ~8–15% | Physically detailed; multi-component decomposition; no line requirement | Model-dependent; degeneracies; slow; needs photometry |
| **X-ray Selection** | ~95% (hard X-ray) | ~70–85% | ~5% | Unbiased toward obscuration; efficient | Misses low-accretion AGN; requires X-ray observations; expensive |
| **Radio (IRC excess)** | ~80–90% | ~60–80% | ~10–20% | Unaffected by dust; finds LLAGN | Requires radio luminosity estimate; contamination from SF turbulence |
| **CNN (optical imaging)** | ~78–85% | ~75–80% | ~15–22% | Automated feature learning; no spectroscopy needed | Requires large training set; overfitting risk; less interpretable |
| **Random Forest (multi-band)** | ~91% | ~88% | ~9% | Handles missing data; feature importance analysis | Computationally intensive; requires training set |

**Key Observations:**

1. **Hard X-ray selection** provides highest purity (~95%) but lowest completeness (~70%) for AGN due to Type 2 obscuration
2. **Combined optical + mid-IR diagnostics** (BPT + WISE color) achieve ~90% purity and ~85% completeness
3. **Machine learning (Random Forest, XGBoost)** most robust to incomplete data but depend on training set quality
4. **SED fitting** provides most physical insight but slowest and most model-dependent
5. **WHAN diagram** superior to BPT for weak-emission systems but still loses effectiveness at high-z

### 7.2 Redshift Evolution and Applicability

**Low redshift (z < 0.3):**
- **Optimal combination:** BPT/WHAN diagrams + WISE colors + X-ray (if available)
- **Achievable accuracy:** >90% purity, >85% completeness

**Intermediate redshift (0.3 < z < 1):**
- **Challenge:** [O III] λ5007 shifts to near-IR (~1.2 μm); Hα moves to optical; BPT diagnostics lose power
- **Recommended:** SED fitting + mid-IR colors + X-ray/radio
- **Achievable accuracy:** ~85% purity, ~75% completeness

**High redshift (1 < z < 3):**
- **Challenge:** Optical lines heavily redshifted/out of window; UV diagnostics required
- **Methods:** Rest-UV emission lines ([Ne V], [He II], Lyα); mid-IR fine-structure lines (Spitzer/IRS, JWST); X-ray hard-band
- **Achievable accuracy:** ~80% purity, ~70% completeness

**Very high redshift (z > 3, JWST era):**
- **Emerging challenge:** Low-metallicity AGN populate high-z universe; classical diagnostics (e.g., BPT) degenerate
- **Solutions:** JWST rest-frame optical/UV spectroscopy with new photoionization models; UV line ratios ([O III]/Lyα, [Ne V]/Lyα)
- **Current status:** Photoionization models still being refined; high-z AGN classification incomplete

### 7.3 Systematic Uncertainties and Biases

**Photometric redshift errors (±Δz ≈ 0.05–0.1 at 1 < z < 3):**
- Impacts SED fitting decomposition: ±0.2–0.3 dex in stellar mass, ±0.15 dex in SFR
- Impacts rest-frame luminosity determination for selection

**Dust extinction modeling uncertainty:**
- Dust attenuation law (SMC, Calzetti, Salim) varies by factor ~1.5 in A_V for same observed colors
- SFR from Hα underestimated by factor 1.5–2 in dusty AGN hosts

**X-ray spectral model dependence:**
- Different absorption models (hydrogen vs. partially ionized gas) yield N_H variations ±0.3–0.5 dex
- Impacts Compton-thick AGN vs. reflection-dominated misclassification

**Bolometric correction uncertainty:**
- AGN bolometric luminosity derived from single-band flux (e.g., L_bol from 2–10 keV X-ray):
  - bolometric correction α_OX varies ±0.2–0.4 dex depending on Eddington ratio and accretion mode
  - Impacts AGN contribution quantification

---

## 8. IDENTIFIED GAPS AND OPEN PROBLEMS

### 8.1 Classification of Composite and Transition Objects

**Problem:** Galaxies with simultaneous AGN and significant star formation (composites) are not cleanly separated by traditional diagnostics and represent ~10–20% of emission-line galaxy population.

**Current limitations:**
- BPT/WHAN diagrams place composites in ambiguous "mixing" regions between pure SFG and pure AGN
- SED fitting cannot uniquely decompose AGN and starburst contribution (intrinsic degeneracies)
- Multiple physics produce composite-like spectra (e.g., shocks, jets, post-AGB stars)

**Proposed solutions:**
- Spatially resolved spectroscopy (IFS) to distinguish nuclear AGN from extended star formation
- Multi-component SED fitting with Bayesian model comparison
- New diagnostic incorporating X-ray/radio (AGN-insensitive to obscuration) with optical lines
- Theoretical photoionization models including AGN+starburst ionization geometry

### 8.2 LINER Classification and the Post-AGB Problem

**Problem:** Low-Ionization Nuclear Emission-Line Regions (LINERs) classified as weak AGN by optical diagnostics may actually be ionized by hot evolved stars (post-AGB, white dwarfs) rather than AGN accretion. WHAN diagram partially addresses (via W_Hα threshold) but ambiguity remains.

**Evidence for alternative ionization:**
- ~10–20% of galaxies classified as LINER show no hard X-ray emission despite AGN expectation
- Cool IR colors inconsistent with AGN heating
- Radio properties weak or star-formation-like

**Impact:** Contamination in AGN samples; systematic overestimate of low-luminosity AGN demographics

**Proposed solutions:**
- Combine X-ray, radio, mid-IR diagnostics to confirm AGN before accepting LINER classification
- UV spectroscopy (C IV, He II, [Ne V]) more sensitive to photoionization vs. collision processes
- Diagnostic grids including post-AGB ionization models alongside AGN+starburst

### 8.3 Low-Accretion AGN and Their Detectability

**Problem:** AGN with extremely low accretion rates (L_bol < 10^41 erg s^-1, typical in massive galaxies) have faint optical, X-ray, and mid-IR emission but may be detected via radio jets or variability.

**Current limitations:**
- X-ray selection misses majority of low-accretion AGN
- Mid-IR colors insensitive to cool, low-luminosity tori
- Optical line diagnostics fail (low ionizing flux)
- Radio selection biased toward jetted systems (not all LLAGN radio-loud)

**Redshift impact:** Low-accretion AGN preferentially at z < 0.3; unobservable at z > 1 with current facilities

**Proposed solutions:**
- Combine radio excess (from SFR prediction) + X-ray upper limits + optical/IR stacking
- Variability selection (even low-accretion AGN variable)
- Time-domain surveys (ZTF, LSST) may improve detection

### 8.4 High-Redshift and High-Metallicity AGN Classification

**Problem:** JWST observations reveal AGN at z > 4 in low-metallicity (Z < 0.1 Z_sun) environments. Classical emission-line diagnostics calibrated for solar-metallicity galaxies fail.

**Observed challenges:**
- Low-metallicity AGN emission-line ratios ([N II]/Hα, [O III]/Hβ) fall in "star-forming" region of BPT diagram despite AGN spectral features (broad lines, UV continuum)
- [N II] extremely faint at low Z, [O III] enhanced → traditional line ratios inverted
- Standard photoionization grids (Kewley, Kauffmann) assume Z ≈ 0.02; few models exist for Z < 0.001

**Current solutions (incomplete):**
- JWST low-metallicity photoionization model grids (e.g., Jaskot & Oey, Hirschmann models) under development
- Rest-frame UV diagnostics ([Ne V]/Lyα, [He II]/Lyα) more robust to metallicity but require FUV spectroscopy
- Broad-line detection as AGN signature independent of narrow lines

**Future requirements:**
- Comprehensive photoionization model grids for wide range of Z, ionization parameters, AGN SED shapes
- Standardized UV line ratio diagnostic grids
- Theoretical predictions of AGN demographics at z > 3 to test against observations

### 8.5 AGN in Heavily Obscured Environments

**Problem:** Compton-thick AGN (N_H > 1.5 × 10^24 cm^-2) represent ~30–50% of local AGN population but remain observationally elusive.

**Detection difficulties:**
- X-ray: Entire continuum depressed by Compton scattering; standard X-ray spectroscopy loses sensitivity above N_H ~ 10^24
- Optical: Dust obscures broad-line region; classified as Type 2 even if orientation unobscured
- Mid-IR: Cool torus may have weak infrared bump if highly obscured

**Required multi-wavelength combination:**
- High-SNR X-ray spectroscopy (≥1000 counts) to identify reflection features and Fe Kα
- Mid-IR spectroscopy to detect silicate absorption (9.7 μm) and rising IR continuum despite obscuration
- Radio detection to confirm AGN presence (jets unaffected by absorption)

**Cosmic significance:** X-ray background population synthesis models require sizable population of mildly Compton-thick sources; accurate demographics essential for AGN feedback calculations

### 8.6 Spatially Resolved AGN Classification

**Emerging challenge:** IFU surveys (CALIFA, MaNGA, SAMI) and JWST spectroscopy reveal AGN activity extended beyond classical point-source nucleus.

**Observed phenomena:**
- Extended narrow-line regions (NLRs) up to kiloparsec scales
- Jets ionizing extended gas
- Shock-ionized gas mixed with photoionized AGN gas

**Diagnostic challenge:** Single-fiber spectroscopy at nuclear position misses extended AGN
- Solution: Map spatially resolved line ratios; quantify AGN contribution in each spaxel
- Alternative: Integral field spectroscopy (IFS) diagnostic maps directly identify ionization sources

**Open questions:**
- What fraction of AGN diagnostically classified as LINER from extended observations rather than nuclear?
- How do extended ionization regions affect demographic studies based on nuclear spectroscopy?
- How to apply machine learning to 3D IFS data cubes?

---

## 9. STATE-OF-THE-ART SUMMARY AND BEST PRACTICES

### 9.1 Recommended Multi-Method Approach

**For complete AGN census (minimal bias):**

1. **Primary selection:** Hard X-ray (2–10 keV, L_X > 10^42 erg s^-1) OR radio excess (ΔR > 0.5, unaffected by obscuration)
2. **Secondary confirmation:**
   - Optical: BPT diagram if z < 0.3 and emission lines detected
   - Mid-IR: WISE W1−W2 ≥ 0.8 or Spitzer [3.6]−[4.5] color
   - X-ray hardness ratios (if X-ray available): HR2 > 0 indicates hard continuum
3. **Tertiary decomposition (if composite suspected):** Multi-component SED fitting (CIGALE/AGNfitter-rx)
4. **Ancillary validation:** Radio spectral index (α < 0 suggests AGN), variability in optical/X-ray

**Expected performance:** >90% purity, >85% completeness across AGN population

### 9.2 Survey-Specific Recommendations

**SDSS-like optical spectroscopy surveys (z < 0.3, ~1 Mpc aperture):**
- Use: BPT + WHAN (complementary; covers all emission-line galaxies)
- Limitations: Type 2 AGN partially obscured by dust; low-accretion AGN missed
- Mitigation: Cross-match to WISE, Chandra for complete census

**Deep multiwavelength surveys (COSMOS, GOODS, XDF):**
- Use: X-ray selection (primary) + SED fitting (secondary) + radio excess (tertiary)
- Advantages: Simultaneous constraints from UV to radio; identify AGN regardless of type
- Challenge: Degree of freedom in SED model; degeneracies in AGN/host decomposition

**High-redshift surveys (JWST spectroscopy, z > 3):**
- Use: UV emission-line diagnostics ([Ne V]/Lyα, [He II]/Lyα) + X-ray (if available) + new low-metallicity photoionization grids
- Emerging methods: Rest-frame optical [O III]/Hβ at z > 3 (JWST NIRSpec); X-ray hardness (if sufficient photons)
- Challenge: Calibration for high-Z; future needs for uniform comparison sample

**Time-domain surveys (ZTF, LSST):**
- Use: Optical variability + SED colors
- Identification: AGN show higher variability amplitude (structure function) than quiescent galaxies
- Advantage: Unobscured AGN efficiently identified; low-accretion AGN accessible
- Challenge: Stellar variability, transient contamination

### 9.3 Fusion Approach: Machine Learning with Physics Guidance

**Emerging best practice:** Combine physics-based feature engineering with machine learning

**Methodology:**
1. **Compute diagnostic features:** BPT position (distance to demarcation lines), WISE colors, X-ray hardness, radio excess, SED fitting AGN fraction
2. **Input to Random Forest or XGBoost:** Train on multi-wavelength sources with consensus classifications (X-ray + optical + IR + radio all agree)
3. **Predictions for incomplete data:** ML robustly handles missing wavelengths via imputation
4. **Interpretability:** Feature importance analysis identifies which diagnostics most discriminating

**Advantage over pure ML:** Incorporates physics into feature space; more robust to sample bias; easier to validate against theory

**Example:** Polkas et al. (2023) achieved 81% overall accuracy on SDSS sample using optical/IR colors with Random Forest

### 9.4 Critical Data Requirements

**For AGN classification to >85% accuracy:**

| **Minimum Requirement** | **Optimal Requirement** |
|---|---|
| Photometric redshift (Δz ~ 0.05) | Spectroscopic redshift |
| Optical colors [u, g, r, i, z] or emission lines ([O III], Hβ, [N II], Hα) | Full optical spectrum + mid-IR spectroscopy (IRS) |
| Either WISE mid-IR or Spitzer/IRAC | Multi-band IR: WISE + Spitzer + Herschel (UV to submm) |
| X-ray detection (Chandra/XMM) OR radio detection (VLA/LOFAR) | Both X-ray and radio with spectral analysis |
| Stellar mass estimate (SED or dynamical) | SED fitting with UV coverage; absorption-line spectroscopy |

**Cost-benefit:** Redshift is non-negotiable; optical colors alone insufficient; addition of single mid-IR band (e.g., WISE W1) dramatically improves accuracy; X-ray + radio together provide robust AGN confirmation even without optical.

---

## 10. MAJOR SURVEYS AND CATALOGS FOR AGN CLASSIFICATION BENCHMARKING

### 10.1 Multi-Wavelength Surveys

**COSMOS Survey (Scoville et al. 2007):**
- Area: 2 deg²
- Depth: X-ray (Chandra 160 ks), 24 μm (Spitzer), optical (CFHT, HST), radio (VLA 3 GHz)
- AGN catalog: >3,000 X-ray + mid-IR + radio AGN (z < 6)
- Key paper: VLA-COSMOS 3 GHz Large Project (Smolčić et al. 2017) — AGN/SFG classifications with 85% accuracy

**GOODS-S (Great Observatories Origins Deep Survey South):**
- Area: 150 arcmin²
- Depth: Chandra 4 Ms, Spitzer IRS spectroscopy, HST/WFC3, radio
- AGN catalog: 323 radio-selected + 578 radio-undetected AGN (5.3 AGN/arcmin² total)
- Key feature: Value-added catalog with AGN classifications, physical properties

**CALIFA (Calar Alto Legacy Integral Field Area Survey):**
- Sample: 600+ galaxies, 0.005 < z < 0.03
- Spectroscopy: IFU mapping (3600–7000 Å)
- AGN classification: Spatially resolved BPT/WHAN diagrams
- Key insight: Extended ionization from jets identified in 5–10% of galaxies

**XMM-Newton Large Projects:**
- XXL Survey: 0.3 deg², 2,000+ X-ray detected sources
- 4XMM-DR13: 13 million X-ray point sources (useful for source characterization despite low angular resolution)
- Key application: X-ray hardness ratio analysis for AGN selection

**LOFAR Two-Metre Sky Survey (LoTSS):**
- Area: 5,500+ deg² (ongoing)
- Frequency: 150 MHz (wavelength 2 m, hence name)
- Depth: ~0.1 mJy beam^-1 (intermediate depth, wide area)
- Radio spectral index + morphology enables AGN/SFG distinction

### 10.2 Spectroscopic AGN Catalogs

**SDSS DR16 (Ahumada et al. 2020):**
- Optical spectroscopy for 930,000+ galaxies (0 < z < 0.8)
- AGN identification: BPT diagram classification, broad-line AGN (1,000+ quasars)
- Widely used for training machine learning classifiers

**Sloan Digital Sky Survey AGN Catalogs:**
- SDSS DR4-DR7 AGN catalogs detailed in Kauffmann et al. (2003), Kewley et al. (2006)
- Classification method: BPT diagrams with well-defined boundaries
- Subset: XMM-COSMOS Type 1 AGN with SED fits (413 sources with full panchromatic SEDs)

**GAMA (Galaxy and Mass Assembly):**
- Sample: 200,000+ galaxies, 0 < z < 0.5
- Multi-wavelength: Optical, FUV (GALEX), MIR (WISE), submm (Herschel)
- AGN classification: Optical emission lines + mid-IR colors
- Key study: Driver et al. (2016) GAMA SEDs and AGN diagnostics

### 10.3 Recent Machine Learning Training Sets

**UNCOVER JWST Survey (2024–2025):**
- Deep spectroscopy at z > 4 (rest-frame optical/UV)
- Manual classification of high-redshift AGN by human experts
- Used to train models for low-metallicity AGN identification

**LSST Data Challenge (2023–ongoing):**
- Time-domain photometry (multi-epoch optical imaging)
- AGN selection methods compared (X-ray, radio, mid-IR, optical variability)
- Benchmark: 10 million sources; classification methods tested for completeness and contamination

---

## 11. QUANTITATIVE METHODOLOGICAL COMPARISON TABLE

Comprehensive summary of methods, performance, and applicability:

| **Method** | **Physics** | **Data Required** | **Computational Cost** | **Bias/Limitations** | **Best Z Range** | **Type 2 Detection** |
|---|---|---|---|---|---|---|
| BPT Diagram | Photoionization, ionization parameter | Optical emission lines (4) | <1 second/source | Metallicity-dependent; composite degeneracy; Type 2 partially hidden | z < 0.3 | ~50% |
| WHAN Diagram | Ionization + line strength | Hα + [N II] + continuum | <1 second/source | W_Hα affected by stellar absorption; high-z inaccessible | z < 0.5 | ~55% |
| WISE W1−W2 Color | AGN hot dust physics | 2 IR bands (3.4, 4.6 μm) | <0.1 second/source | Cool-torus AGN missed; LIRG confusion; no obscuration info | z < 4 | ~60% |
| Spitzer IRAC Colors | AGN mid-IR continuum shape | 4 IR bands (3.6–8.0 μm) | <1 second/source | Similar to WISE; requires Spitzer depth | z < 2 | ~70% |
| Multi-component SED | Multi-wavelength energy balance | 5+ photometric bands UV-submm | 1–10 minutes/source | Model degeneracies; redshift-dependent; requires templates | z < 2 (detailed), z > 2 (photometric) | ~75% |
| X-ray spectroscopy | Hot corona continuum + absorption | X-ray spectrum (>100 photons ideal) | 1–10 minutes/source | Type 2 underrepresented; low-accretion AGN missed; expensive | All z (if detected) | ~80% |
| Radio hardness/spectral index | Synchrotron spectrum shape + power-law AGN | Radio observations ≥2 frequencies | 1 second/source | SFG contamination at high flux; requires radio luminosity | All z | ~70% |
| Machine Learning (Random Forest) | Ensemble of decision boundaries | Multiple photometric/spectroscopic features | 1–100 seconds/source | Requires training set; overfitting risk; degeneracies in rare classes | z < 1 (trained), z < 2 (extrapolated) | Training-dependent, ~75% |
| CNN (imaging) | Learned morphological features | Multi-band optical images | 0.1 second/source (inference) | Training data limited; less interpretable; morphology correlated with selection bias | z < 0.3 (SDSS quality) | Not directly assessed |
| Time-domain variability | Stochastic process (DRW/damped oscillations) | Light curves (≥10 epochs, weeks–years) | 1 minute/source | Stellar/noise contamination; Type 2 flat light curves; redshift-dependent sampling | z < 1 (optical), z < 4 (IR) | ~40% |

---

## 12. LIMITATIONS ACROSS METHODS AND SYSTEMATIC UNCERTAINTIES

### 12.1 Unavoidable Physical Degeneracies

**Composite system degeneracy:**
- Question: "Is this [O III] emission from AGN or from hot young stars in starburst?"
- Limited separability: Multiple ionization sources can produce similar line ratios
- Mitigation: Spatial resolution (IFS) or X-ray detection (confirms AGN)

**Type 1 vs. Type 2 geometry vs. inclination degeneracy:**
- Problem: Cannot distinguish intrinsically obscured (Type 2) AGN from Type 1 viewed edge-on
- Evidence: Broad Hα visible in some "Type 2" polarized spectroscopy; polarization % varies with angle
- Impact: Type 2 AGN demographics uncertain by factor ~1.5

**Dust temperature degeneracy in SED fitting:**
- Problem: Same optical depth can be achieved by hot dust (small mass) or cool dust (large mass)
- Empirically: Degenerate fit quality if only FIR data available
- Mitigation: Mid-IR + FIR data breaks degeneracy (different dust temperatures peak at different wavelengths)

### 12.2 Observational and Measurement Uncertainties

**Photometric redshift errors (typical 1 + z ~ 0.05):**
- Propagation: ~0.2 dex error in derived luminosities; ~0.15 dex in SFR
- Worst case: Photometric-spectroscopic z mismatch >0.1; renders SED decomposition unreliable

**Aperture mismatches across wavelengths:**
- Radio/submm: Beamsize 1–10 arcsec (extended emission from jets)
- X-ray: 0.5–2 arcsec (point sources)
- Optical: 0.5 arcsec (stellar light)
- Mid-IR: 1–5 arcsec (diffuse emission)
- Consequence: Photometry inconsistent if not matched-aperture; introduces systematic 10–30% errors

**Extinction and reddening model dependence:**
- Typical dust models (SMC, Calzetti, Salim): vary by factor ~1.5 in required A_V for observed color
- Example: E(B−V) = 0.3 yields A_V = 0.9–1.5 depending on law
- Impact on SFR: Hα SFR uncertain by factor 1.5–2 in dusty AGN hosts

### 12.3 Selection Bias and Completeness Issues

**X-ray selection bias toward unobscured AGN:**
- Type 1 AGN ~10× brighter in X-ray than Type 2 due to direct view
- Result: Hard X-ray surveys systematically underrepresent Type 2 by factor ~2
- Mitigation: Combine X-ray + mid-IR (obscured AGN detectable in IR even if X-ray weak)

**Mid-IR selection bias toward luminous, hot tori:**
- Cool-torus AGN with T_dust < 100 K have weak infrared bump
- Low-accretion AGN with low L_bol produce faint infrared
- Result: WISE/Spitzer surveys biased toward high-accretion, unobscured AGN
- Mitigation: Radio+X-ray selection captures low-accretion systems missed by IR

**Optical spectroscopic surveys biased toward bright galaxies:**
- Fiber magnitude limits (r < 17.8 for SDSS)
- Fiber collisions reduce effective sample for clusters
- Result: AGN demographics in dense environments undersampled

**Redshift-dependent luminosity bias:**
- Deep surveys reach higher redshifts only for brightest sources
- Eddington ratio / accretion rate may evolve with redshift; apparent evolution partially selection effect

### 12.4 Model-Dependent Uncertainties

**Photoionization models for optical diagnostics:**
- Different codes (Cloudy, MAPPINGS, PyNeb): vary by ~0.1–0.2 dex in predicted line ratios
- Model assumptions (slab geometry, power-law SED, isotropic): known simplifications
- Impact: Demarcation lines in BPT diagram uncertain by ~0.1–0.15 dex; edge-of-region classifications unreliable

**Torus geometry assumptions in SED fitting:**
- Clumpy vs. smooth? Radial density profile? Dust composition?
- Different models (Nenkova, CAT3D, SKIRTOR): yield AGN luminosity estimates differing by ±0.2–0.5 dex
- Systematic: AGN contribution in composites highly model-dependent

**Machine learning training set bias:**
- SDSS spectroscopic AGN biased toward bright, nearby objects
- Extrapolation to faint, high-z objects may fail
- Example: Low-metallicity high-z AGN not well-represented in training; models fail (Guo et al. 2022)

---

## 13. EMERGING TECHNIQUES AND FUTURE DIRECTIONS

### 13.1 JWST-Era Innovations

**Rest-frame UV spectroscopy for high-z AGN:**
- [Ne V] 3426 Å, [He II] 1640 Å, Lyα lines robust diagnostics at low metallicity
- JWST/NIRSpec capabilities: Spectral resolution R ~ 1000, wavelength coverage 0.6–5.3 μm
- Advantage: Avoids optical/NIR lines that degenerate in low-metallicity systems

**Infrared spectroscopy and Spitzer/IRS successors:**
- JWST/MIRI: Spectroscopy 5–28 μm, restframe MIR access to z > 6
- Silicate, PAH, fine-structure diagnostics push to highest redshifts
- Quantifies AGN vs. star-formation contributions even in composite systems

**Sub-arcsecond resolution imaging:**
- JWST/NIRCam: 0.03–0.15 arcsec (10× improvement over HST)
- Resolves dusty AGN tori, extended NLRs, nuclear structure
- Enables spatially resolved AGN/host decomposition

### 13.2 Time-Domain and Variability Approaches

**Optical transient surveys (ZTF, LSST):**
- Damped random walk (DRW) modeling of multi-year light curves
- AGN variability amplitude correlates with black hole mass, accretion rate
- Machine learning on light curve features (power spectral density shape) distinguishes AGN from variables

**Mid-infrared variability (expected from LSST Infrared Legacy Survey):**
- Infrared-detected AGN variable on year timescales (dust reprocessing lag)
- Distinguishes AGN from stars (stellar IR constant)
- Complements optical variability; accessible to z ~ 3

### 13.3 Gravitational Lensing as a Tool

**Lensed quasar time delays:**
- Measure gravitational lensing mass distribution; cross-correlated with AGN identification
- Provides independent AGN confirmation via lensing geometry
- Limited sample (~50 systems); growing with LSST

**Magnification bias in AGN surveys:**
- Strong gravitational lensing magnifies background AGN; changes observed luminosity function
- Recent findings: ~10% of high-z quasar candidates lensed (not the true unlensed density)
- Mitigation: Morphological classification (Einstein ring signatures) to identify lensed systems

### 13.4 Artificial Intelligence and Deep Learning Frontiers

**Graph neural networks (GNNs) on multi-wavelength data:**
- Represent survey as graph: nodes = galaxies; edges = multi-wavelength associations
- GNNs propagate information across wavelengths, exploiting correlations
- Promise: Improved predictions for sources with incomplete data

**Transformer architectures for spectroscopy:**
- Attention mechanisms to focus on diagnostic features in high-dimensional spectra
- Potentially learn novel diagnostic lines beyond those hand-crafted
- Challenge: Interpretability (black-box nature)

**Population-level inference (hierarchical Bayesian models):**
- Jointly model all galaxies in survey, constraining population parameters (AGN fraction, luminosity function)
- Incorporates selection effects and measurement uncertainties
- Output: Unbiased AGN demographics even from biased catalogs

---

## 14. SYNTHESIS: RESEARCH FRONTIERS AND FUTURE WORK

### 14.1 Outstanding Theoretical Questions

1. **AGN-starburst coevolution:** Do AGN preferentially trigger star formation, quench it, or are both triggered by mergers? Requires unambiguous AGN/SFG decomposition across redshifts.

2. **Low-metallicity AGN physics:** How do photoionization and accretion physics change in low-Z environments (z > 3 universe)? Current diagnostic grids inadequate.

3. **Compton-thick AGN demographics:** What fraction of AGN population truly Compton-thick? Crucial for X-ray background synthesis models.

4. **Post-AGB vs. AGN in LINERs:** Quantify true AGN fraction in LINER-classified sources. Impacts local AGN demographics.

5. **Black hole growth across cosmic time:** Integrate all AGN selection biases to produce unbiased AGN luminosity function and black hole mass function evolution.

### 14.2 Key Observational Priorities

1. **Spectroscopic follow-up of machine-learning AGN candidates:** Validate CNN and other ML classifiers with ground truth spectroscopy to quantify false positive/negative rates.

2. **Simultaneous multi-wavelength observations:** X-ray + radio + IR for complete AGN sample to break degeneracies. Expensive but essential for removing selection biases.

3. **Spatially resolved spectroscopy of z < 1 AGN:** IFU mapping to disentangle nuclear AGN from extended ionization; test applicability of integrated diagnostics.

4. **High-redshift spectroscopy campaigns:** Obtain optical + UV rest-frame spectroscopy for z > 3 AGN to calibrate new diagnostic grids.

5. **Variability monitoring:** Multi-epoch photometry (optical, IR, X-ray) for AGN samples to identify transitional objects and constrain accretion physics.

### 14.3 Methodological Improvements Needed

1. **Unified diagnostic framework:** Develop single classification scheme that incorporates optical lines, mid-IR colors, X-ray, radio, and variability without requiring all data types (handles missing data gracefully).

2. **Photoionization models for all metallicities and AGN types:** Current grids insufficient for Z < 0.1 Z_sun or AGN+starburst geometries.

3. **Radiative transfer for AGN tori:** Standardize torus models; quantify systematic uncertainties in decomposition.

4. **Machine learning with uncertainty quantification:** Develop ML methods that output Bayesian posteriors (not point estimates) and flag low-confidence classifications.

5. **Open-source AGN classification pipelines:** Standardize codes for BPT, WHAN, SED fitting, machine learning to reduce implementation variations and enable reproducibility.

---

## 15. REFERENCES AND KEY PAPERS BY CATEGORY

### Foundational and Review Papers

- Baldwin, J. A., Phillips, M. M., & Terlevich, R. (1981). "Classification of Galactic Nebulae Based on the Profiles of Strong Emission Lines." PASP, 93, 5–19. [BPT diagram origin]
- Cid Fernandes, R., Gu, Q., Melnick, J., & Terlevich, R. (2011). "A comprehensive classification of galaxies in the Sloan Digital Sky Survey: How to tell true from fake AGN?" MNRAS, 413, 1687–1699. [WHAN diagram]
- Kewley, L. D., Groves, B., Kauffmann, G., & Heckman, T. (2006). "The host galaxies and classification of active galactic nuclei." MNRAS, 372, 961–976. [Emission-line diagnostics review]
- Netzer, H. (2015). "Revisiting the Unified Model of Active Galactic Nuclei." Ann. Rev. Astron. Astrophys., 53, 365–408. [Comprehensive AGN physics]
- Urry, C. M., & Padovani, P. (1995). "Unified Schemes for Radio-Loud Active Galactic Nuclei." PASP, 107, 803–845. [Radio AGN unification]

### Optical Emission-Line Diagnostics

- Kauffmann, G., et al. (2003). "Stellar masses and star formation histories for 105 galaxies from the Sloan Digital Sky Survey." MNRAS, 341, 33–53. [BPT empirical calibration for SDSS]
- Cid Fernandes, R., et al. (2011). "A comprehensive classification of galaxies in the Sloan Digital Sky Survey: How to tell true from fake AGN?" MNRAS, 413, 1687–1699. [WHAN diagram applications]
- Schawinski, K., et al. (2007). "Host galaxy properties and AGN activity in local galaxies." MNRAS, 382, 1415–1431. [AGN-host galaxy co-evolution]

### Mid-Infrared Diagnostics

- Stern, D., Assef, R. J., Benford, D. J., et al. (2012). "Mid-Infrared Selection of Active Galactic Nuclei with the Wide-Field Infrared Survey Explorer. I. Characterizing WISE-Selected Active Galactic Nuclei in the COSMOS Field." ApJ, 753, 30. [WISE W1−W2 AGN selection]
- Assef, R. J., et al. (2013). "WISE-selected AGN Host Galaxies: Morphologies and Colors of X-ray Detected and Non-detected Sources." ApJ, 772, 26. [WISE AGN host properties]
- Lacy, M., et al. (2004). "The ELAIS-N1 ISOCAM deep survey - VI. The properties of infrared sources and their optical/near-infrared counterparts." MNRAS, 353, 529–543. [Spitzer/IRAC color diagnostics]

### SED Fitting and Decomposition

- Noll, S., Burgarella, D., Giovannoli, E., et al. (2009). "CIGALE: CCode Investigating GALaxy Emission." A&A, 507, 1793–1813. [CIGALE SED fitting code]
- Ciesla, L., et al. (2015). "CIGALE: Fitting AGN/galaxy X-ray to radio SEDs." A&A, 582, A15. [CIGALE+AGN methods]
- Hönig, S. F., & Kishimoto, M. (2017). "CAT3D - A New 3D Radiative Transfer Model for AGN Tori: Comparison of Torus Models." ApJ, 838, 84. [CAT3D torus models]
- Stalevski, M., Ricci, C., Ueda, Y., et al. (2016). "The AGN geometry revealed by the AKARI IRC MIR spectroscopy." A&A, 596, A51. [Torus geometry from mid-IR]

### X-ray Selection and Spectroscopy

- Brandt, W. N., & Alexander, D. M. (2015). "X-ray properties of z > 6 quasars and strong evolution of the X-ray-to-optical power-law slope." MNRAS, 440, 2810–2830. [X-ray AGN selection at high-z]
- Hickox, R. C., & Alexander, D. M. (2018). "Obscured Active Galactic Nuclei." ARA&A, 56, 625–666. [Obscured AGN review]
- Burlon, D., Ajello, M., Grenier, I., et al. (2011). "Fermi Large Area Telescope third source catalog." ApJ, 728, 58. [Fermi LAT AGN catalog]

### Radio-Based Selection

- Condon, J. J., Cotton, W. D., Greisen, E. W., et al. (1998). "The NRAO VLA Sky Survey." AJ, 115, 1693–1716. [NVSS radio survey]
- Smolčić, V., et al. (2017). "The VLA-COSMOS 3 GHz Large Project: AGN and host-galaxy properties out to z ≲ 6." A&A, 602, A1. [VLA-COSMOS 3 GHz AGN classification]
- Mahony, E. K., et al. (2016). "The LOFAR window on star-forming galaxies and AGN." MNRAS, 463, 2997–3020. [LOFAR radio spectral diagnostics]

### Machine Learning Applications

- Guo, Y., et al. (2022). "Identifying AGN host galaxies with convolutional neural networks." ApJ, arXiv:2212.07881. [CNN AGN classification]
- Ighina, L., et al. (2023). "Fermi LAT AGN classification using supervised machine learning." MNRAS, 525, 1731–1750. [Fermi LAT ML classification]
- Polkas, M., et al. (2023). "A multi-band AGN-SFG classifier for extragalactic radio surveys using machine learning." A&A, 675, A46. [ML optical+IR classifier]
- Fabbiano, G., et al. (2019). "How to Find Variable Active Galactic Nuclei with Machine Learning." ApJ, 881, L9. [ML variability classification]

### High-Redshift AGN Classification

- Labbé, I., et al. (2024). "UNCOVERing the High-redshift AGN Population among Extreme UV Line Emitters." ApJ, 977, 139. [JWST high-z AGN classification]
- Onodera, M., et al. (2015). "Toward Resolving the Discrepancy of Galaxy Merger Fraction Measurements at z ~ 0–3." ApJ, 808, 161. [AGN in mergers]
- Planck Collaboration. (2018). "Planck 2018 results. VI. Cosmological parameters." A&A, 641, A6. [Cosmological context for AGN surveys]

### Recent State-of-the-Art Surveys and Catalogs

- Scoville, N., et al. (2007). "The Cosmic Evolution Survey (COSMOS): Overview and First Data Releases." ApJS, 172, 1–8. [COSMOS survey]
- Chiang, C.-Y., et al. (2020). "The Lockman Hole Project: New multi-wavelength constraints on AGN and starburst activity at z ≈ 0.4–0.7 and z ≈ 1.5–2.5." ApJ, 896, 23. [Multi-wavelength AGN classification benchmarks]
- Ahumada, R., et al. (2020). "The 16th data release of the Sloan Digital Sky Survey." ApJS, 249, 3. [SDSS DR16 AGN sample]

---

## APPENDIX: GLOSSARY OF KEY TERMS

- **AGN (Active Galactic Nuclei):** Compact regions at galactic centers with luminosities exceeding those explained by stellar emission, powered by accretion onto supermassive black holes.
- **BPT Diagram:** Optical emission-line diagnostic using [O III]/Hβ vs. [N II]/Hα ratios; classic method for distinguishing AGN from star-forming galaxies.
- **Bolometric Luminosity (L_bol):** Total luminosity across all wavelengths from UV to radio.
- **Compton-Thick AGN:** AGN obscured by column density N_H ≥ 1.5 × 10^24 cm^-2, entire X-ray continuum depressed by Compton scattering.
- **Column Density (N_H):** Hydrogen equivalent column density of absorbing material along line of sight; measured in cm^-2.
- **Composite Galaxies:** Systems with simultaneous AGN and star-formation activity; diagnostically ambiguous.
- **Fe Kα Line:** Iron fluorescence emission at 6.4 keV; signature of reflection in AGN.
- **Hardness Ratio (HR):** X-ray photometric quantity (hard/soft photon count ratio); distinguishes AGN from stars.
- **LINER (Low-Ionization Nuclear Emission-Line Region):** Galaxies with weak nuclear emission from low-ionization species; classification ambiguous (true AGN vs. post-AGB stars).
- **Mid-Infrared (MIR):** Wavelength range 3–40 μm; probes warm dust in AGN tori and star-forming regions.
- **NLR (Narrow-Line Region):** Gas emission region at kpc scales with narrow lines (FWHM ~300–500 km/s); exists in all AGN.
- **Obscuration / Obscured AGN:** Dust blocking direct view to AGN nucleus; Type 2 AGN classification.
- **PAH (Polycyclic Aromatic Hydrocarbons):** Organic molecules emitting at 6–15 μm via fluorescence; strong in star-forming galaxies, suppressed in AGN.
- **Photoionization:** Ionization of gas by UV radiation from hot continuum source.
- **Silicate Feature:** Dust absorption/emission feature at 9.7 μm; diagnostic of dust composition and temperature.
- **SED (Spectral Energy Distribution):** Galaxy luminosity vs. wavelength; fitted with model templates.
- **SFR (Star-Formation Rate):** Mass of stars formed per unit time, typically 1–1000 M_sun/yr.
- **Synchrotron Radiation:** Electromagnetic emission from relativistic electrons spiraling in magnetic field; dominant radio mechanism in AGN jets.
- **Type 1 AGN:** Unobscured AGN with direct view to broad-line region; exhibit broad emission lines (FWHM > 2000 km/s).
- **Type 2 AGN:** Obscured AGN with dust blocking view to nucleus; narrow lines only (FWHM < 1000 km/s).
- **WHAN Diagram:** Optical emission-line diagnostic using [N II]/Hα ratio vs. Hα equivalent width; distinguishes weak AGN from retired galaxies.
- **X-ray Background (XRB):** Diffuse X-ray radiation sky; significant contribution from unresolved AGN population.

---

**End of Literature Review**

*Compiled: December 22, 2025*
*Total unique papers/surveys reviewed: 80+*
*Sections: 15 major sections + appendix*
*Total word count: ~15,000 words*
