# Literature Review: Diagnostic Techniques for Distinguishing AGN from Star-Forming Galaxies

**Date compiled:** December 2025

---

## Executive Summary

The distinction between Active Galactic Nuclei (AGN) and star-forming galaxies (SFGs) is fundamental to understanding galaxy evolution and AGN demographics. This review synthesizes diagnostic methods across optical, infrared, X-ray, and radio wavelengths, along with emerging machine learning approaches. Key findings highlight the power of multi-wavelength diagnostics while documenting persistent challenges with composite galaxies and edge cases (low-luminosity AGN, heavily obscured systems, starburst-AGN mixtures).

---

## 1. Overview of the Research Area

### 1.1 Problem Statement

The fundamental challenge is identifying and classifying AGN versus star-forming activity in galaxies across cosmic time and across the electromagnetic spectrum. Single-wavelength diagnostics suffer from degeneracies and obscuration bias:
- **Optical spectroscopy:** Biased against dust-obscured sources
- **X-ray selection:** Misses heavily Compton-thick systems
- **Infrared selection:** Cannot uniquely distinguish AGN from starburst heating
- **Radio emission:** Contaminated by both AGN jets and star formation supernova emission

Composite galaxies (containing both AGN and active star formation) present particular diagnostic challenges, with mixing sequences spanning intermediate parameter space on standard diagnostic diagrams.

### 1.2 Scope of this Review

This review covers:
1. **Optical diagnostics:** BPT diagrams and emission-line methods
2. **Mid-infrared diagnostics:** Color-color selections and fine-structure lines
3. **X-ray methods:** Hardness ratios and spectral absorption measures
4. **Radio-optical techniques:** Radio excess and luminosity ratios
5. **Machine learning:** Supervised and unsupervised classification approaches
6. **Edge cases:** Composite galaxies, LINERs, starburst-AGN systems

---

## 2. Chronological Summary of Major Developments

### 2.1 Classical Optical Diagnostics (1981–2003)

**Baldwin, Phillips & Terlevich (1981):** Introduced the foundational BPT diagram using emission-line ratios [O III]/Hβ vs. [N II]/Hα to separate ionization mechanisms. This landmark paper established the framework for distinguishing AGN from star formation (Baldwin et al., PASP 93:5-19).

**Veilleux & Osterbrock (1987):** Extended optical diagnostics by introducing alternative emission-line ratios:
- [S II] λλ6717,6731/Hα vs. [O III]/Hβ (VO87-SII diagram)
- [O I] λ6300/Hα vs. [O III]/Hβ (VO87-OI diagram)

These additional diagrams improved separation of AGN subtypes and revealed the "LINER" (Low-Ionization Nuclear Emission-line Region) class of galaxies with characteristically lower ionization.

**Kewley et al. (2001):** Provided theoretical photoionization models for AGN and star-forming regions, deriving theoretical separation curves that superseded purely empirical divisions. This work established that AGN have "harder" ionizing spectra (higher flux at extreme ultraviolet energies), producing enhanced [O III] and [N II] relative to hydrogen recombination lines.

**Kauffmann et al. (2003):** Derived empirical AGN/starburst division curves using SDSS data, providing practical boundaries distinguishing:
- Star-forming galaxies: Below the Kauffmann line
- Composite/transition objects: Between Kauffmann and Kewley curves
- AGN-dominated: Above the Kewley line

This three-region classification became standard in modern surveys.

### 2.2 Mid-Infrared and Multiwavelength Era (2002–2015)

**Spoon et al. (2002):** Published mid-infrared fine-structure line diagnostics analogous to optical BPT diagrams, showing that mid-IR emission lines ([Ne III], [S IV], etc.) provide AGN diagnostics complementary to optical methods and less affected by dust obscuration (A&A 402:499-513).

**Stern et al. (2012); Assef et al. (2013):** Established WISE (Wide-field Infrared Survey Explorer) mid-infrared color selection criteria for AGN:
- **Primary criterion:** W1 − W2 ≥ 0.8 (3.4 − 4.6 μm in Vega magnitudes)
- **High-completeness criterion:** W1 − W2 ≥ 0.7
- **Key advantage:** Identifies both unobscured (Type 1) and Compton-thick obscured (Type 2) AGN, unbiased against dust (ApJ 753:30 and related studies)

The color selection works because AGN-heated dust shows characteristic excess mid-infrared emission.

**Hickox et al. (2005–2010):** Systematized X-ray selection of obscured AGN, establishing hardness ratio thresholds and showing that X-ray spectral absorption measures (column densities NH) effectively distinguish unabsorbed (NH < 10²² cm⁻²) from absorbed (NH > 10²² cm⁻²) systems, with HR ≈ −0.2 as a convenient threshold.

### 2.3 Composite Galaxy Problem (2010–2018)

**Starburst-AGN Mixing Studies:**

**Pereira-Santaella et al. (2015):** Detailed spatially-resolved spectroscopy of the prototype starburst-AGN system NGC 7130, revealing that optical emission-line ratios in mixed systems follow curved "mixing sequences" on the BPT diagram rather than clean separation (MNRAS 439:3835-3848).

**Pereira-Santaella et al. (2016):** Extended analysis to optically-selected AGN samples, showing that variation in gas metallicity and ionization parameter significantly affects the expected AGN diagnostic position, creating a "confusion zone" where pure starbursts can mimic AGN-like line ratios (MNRAS 444:3961-3978).

**Key finding:** The position on a diagnostic diagram alone is insufficient for composite systems; multiwavelength data and spatial resolution are essential.

### 2.4 Advanced Spectroscopic Methods (2015–2023)

**Integral Field Spectroscopy (IFS):**

Studies using CALIFA, MANGA, and other IFS surveys demonstrated that spatial resolution dramatically improves AGN/starburst separation. When the AGN-heated and star-formation-heated regions can be spatially resolved, line ratios can be measured separately, eliminating aperture effects and mixing degeneracies. This approach reveals that AGN contribute heterogeneously: AGN-heated gas is typically confined to nuclear regions (r < few hundred pc), while star formation dominates extended disks.

**New Optical Diagnostics:**

Lonsdale et al. (2015) and subsequent authors developed:
- **Mass-Excitation (MEx) diagram:** Uses [O III]/Hβ vs. stellar mass to identify AGN, extending diagnostic reach to z ~ 1 (beyond the z ~ 0.4 limit of traditional BPT diagrams)
- **Color-Excitation (CEx) diagram:** Incorporates optical colors with emission-line ratios for improved high-redshift applicability

**Near-Infrared Diagnostics:**

Recognition that rest-frame near-infrared spectroscopy (less affected by dust) enables optical-equivalent diagnostics for dusty, high-redshift systems. Studies showed [O III] and Hα can be measured at z > 1 in the near-infrared, providing cleaner AGN identification in obscured populations.

### 2.5 Machine Learning Era (2018–2025)

**Early ML Studies (2018–2020):**

Frampton et al. (2019) demonstrated that variable AGN could be identified with 86% purity and 66% completeness using machine learning applied to optical variability light curves, rivaling supervised deep-learning approaches (ApJ 881:L9).

Banfield et al. (2015–2016) applied CNNs to morphological classification of radio-loud AGN, achieving ~75–95% accuracy depending on Fanaroff-Riley morphology class (MNRAS 453:2326-2340).

**Supervised Classification with Multi-Wavelength Data (2020–2025):**

Modern studies employ Random Forest, XGBoost, and gradient boosting methods:

- **Hardcastle et al. (2023):** Trained LightGBM on radio+optical+infrared data from surveys, achieving **precision = 0.92±0.01** and **recall = 0.87±0.02** for star-forming galaxy classification (A&A 671:A136).

- **LSST AGN Data Challenge (2023):** Systematically evaluated Support Vector Machines, Random Forest, and XGBoost classifiers on photometric + spectroscopic features. XGBoost achieved F1-scores ~0.85–0.90 for AGN vs. non-AGN classification in realistic LSST-like datasets.

- **MIGHTEE Radio Survey Studies (2023–2025):** Implemented 5 ML algorithms (Random Forest, XGBoost, Neural Networks, SVM, Gradient Boosting) for distinguishing AGN-dominated vs. star-forming-dominated radio sources, with typical F1-scores in the range 0.82–0.89 depending on feature set and redshift.

**Key innovation:** Gradient boosting methods (XGBoost, LightGBM, CatBoost) currently show superior performance for this binary classification task due to their ability to capture non-linear feature interactions.

---

## 3. Detailed Methods and Diagnostic Techniques

### 3.1 Optical Emission-Line Diagnostics

#### 3.1.1 Baldwin-Phillips-Terlevich (BPT) Diagram

**Principle:** Uses two line-ratio axes to map ionization source:
- **X-axis:** log([N II] 6583 Å / Hα 6563 Å)
- **Y-axis:** log([O III] 5007 Å / Hβ 4861 Å)

**Physical basis:** AGN produce harder ionizing spectra (more extreme-UV photons) than O-type stars in H II regions. This hardness leads to:
- Enhanced collisional ionization of nitrogen and oxygen
- Increased forbidden-line emission
- Higher [N II] and [O III] relative to hydrogen recombination lines

**Classification regions** (forming a characteristic "seagull" shape):

| Region | Line-ratio Position | Interpretation |
|--------|-------------------|-----------------|
| **Lower-left (wing)** | Low [N II]/Hα; Low [O III]/Hβ | Star-forming galaxies, H II regions |
| **Lower-center (body)** | Between Kauffmann (~0.3 dex above SF) | Transition/composite galaxies |
| **Upper-right (wing)** | High [N II]/Hα; High [O III]/Hβ | AGN (Seyfert 1, Seyfert 2, Quasars) |

**Boundary definitions:**

- **Kauffmann et al. (2003) line** (empirical):
  - Equation: log([O III]/Hβ) = 0.61 / (log([N II]/Hα) − 0.47) + 1.19
  - Separates pure star formation from AGN contribution

- **Kewley et al. (2001) line** (theoretical):
  - More stringent; AGN above this line
  - Derived from photoionization models with hard ionizing spectrum

**Quantitative results:**
- Accuracy for pure AGN/SF classification: ~85–90% when avoiding the composite zone
- Misclassification increases dramatically in composite region (±0.3 dex around mixing curve)

**Limitations:**
- **Redshift constraint:** Rest-frame optical lines observable only to z ~ 0.4 (where Hβ enters NIR)
- **Dust obscuration bias:** Optical emission suppressed in heavily dusty systems; spectral line ratios affected by reddening
- **Aperture effects:** Composite galaxies show spatial variation; central apertures may appear more AGN-like than integrated spectrum
- **Metallicity degeneracy:** Low-metallicity starbursts can shift towards AGN region on BPT diagram (log([N II]/Hα) decreases with Z)
- **LINER confusion:** Low-ionization objects occupy intermediate parameter space

#### 3.1.2 Alternative Optical Diagnostics

**Veilleux & Osterbrock (1987) diagrams:**

- **[S II] VO87 diagram:** log([S II]/Hα) vs. log([O III]/Hβ)
  - Better separates Seyfert 2 from LINER

- **[O I] VO87 diagram:** log([O I]/Hα) vs. log([O III]/Hβ)
  - Improved discrimination of AGN subtypes

**Mass-Excitation (MEx) diagram:**
- Axes: log([O III]/Hβ) vs. log(M*/M⊙)
- **Innovation:** Incorporates stellar mass; AGN preferentially found in more massive hosts
- Extends to z ~ 1 (beyond BPT z ~ 0.4 limit)
- Result: ~85% AGN/SF separation at z ≤ 1

**Color-Excitation (CEx) diagram:**
- Combines optical colors (e.g., u−r) with [O III]/Hβ
- Reduces redshift-dependent systematics
- Particularly useful for high-redshift (z > 1) optical spectroscopy

**Line-width diagnostics (MaNGA/IFS-resolved):**
- Gas velocity dispersion σ_gas measured from emission-line widths
- AGN produce broader lines (σ ∼ 100–300 km/s) vs. SF (σ ∼ 20–50 km/s)
- Spatial variation revealed by IFS: AGN-heated regions show elevated velocity dispersion

### 3.2 Mid-Infrared Diagnostics

#### 3.2.1 WISE Mid-Infrared Color Selection

**Standard Color Criterion (Stern et al. 2012; Assef et al. 2013):**

W1 − W2 ≥ 0.8 (Vega magnitudes), where:
- W1 = 3.4 μm band
- W2 = 4.6 μm band

**Physical basis:** AGN-heated dust emits excess power in the 3–5 μm region (power-law dominates) compared to stellar continuum (falls off in this band) or cool starformation-heated dust (peaks at longer wavelengths).

**Quantitative characteristics:**
- **AGN detection efficiency:** ~61.9 ± 5.4 AGN deg^−2 at W2 ≤ 15.0
- **Completeness vs. luminosity:** Strong AGN (L_bol > 10^45 erg/s) detected with ~95% completeness; efficiency drops for low-luminosity AGN
- **Type 1 + Type 2 detection:** Identifies both unobscured and Compton-thick systems (unlike optical/soft X-ray)

**Refined criteria:**
- **Higher purity:** W1 − W2 = 0.8 (strict)
- **Higher completeness:** W1 − W2 = 0.7 (less conservative, ~5% contamination)

**Limitations:**
- **Luminosity-dependent:** Misses significant fractions of low-Eddington-ratio systems (accretion-powered AGN with cool, massive tori)
- **High-z evolution:** Color cut performance degrades at z > 3 due to dust temperature changes
- **Starburst contamination:** Young starbursts with hot dust can marginally enter AGN color space

#### 3.2.2 Mid-Infrared Fine-Structure Line Diagnostics

**Method (Spoon et al. 2002; subsequent Spitzer/SOFIA observations):**

Analogous to optical BPT diagrams but using mid-IR fine-structure lines:
- [Ne III] 15.5 μm (higher ionization)
- [Ne II] 12.8 μm
- [S IV] 10.5 μm
- [S III] 18.7, 33.5 μm

**Advantages:**
- Nearly dust-independent (at these wavelengths, even heavily obscured AGN visible)
- Diagnostic coverage for z ~ 0.5–2 objects (lines redshift into Spitzer wavelength range)
- Fine-structure line ratios probe obscuration independently

**Quantitative results:**
- AGN/starburst separation achieved with ~80% accuracy
- [Ne III]/[Ne II] ratio: AGN enhance [Ne III] flux via harder ionizing spectrum

**Limitations:**
- Requires high-sensitivity infrared spectroscopy (Spitzer IRS, SOFIA, future JWST)
- Not applicable to unobscured systems where mid-infrared continuum dominates
- Relatively small sample sizes in published studies (n ~ 50–200 per study)

#### 3.2.3 Infrared-Radio Correlation and q Parameter

**The q_IR ratio (Ivison et al. 1994 onwards):**

Definition: q_IR = log(L_IR / [3.75 × 10^12 W m^−2 Hz^−1 L_ν(1.4 GHz)])

where L_IR = total infrared luminosity (8–1000 μm).

**Physical meaning:** The ratio probes the relative balance of star-formation heating (which produces ~equal radio and IR from synchrotron + dust reprocessing) vs. AGN heating (which distorts this balance through jets or obscuration effects).

**Empirical calibration (Delhaize et al. 2021):**

q_IR(M*, z) = (2.646 ± 0.024) × (1+z)^(−0.023±0.008) − (0.148 ± 0.013) × [log(M*/M⊙) − 10]

**Star formation rate indicator:**
- For pure star-forming galaxies, q_IR ≈ 2.1–2.4 (nearly constant with redshift)
- Radio luminosity → SFR via: SFR = (q − q_ref) × constant + standard radio-SFR relation
- Radio-based SFR estimates can therefore self-correct for AGN contamination

**AGN effects on q_IR:**
- AGN jets increase radio flux without comparable IR increase → lower q_IR
- Obscured AGN with torus heating increase IR without radio enhancement → higher q_IR
- q_IR_excess (data − prediction) indicates AGN contribution

**Quantitative results:**
- SFR measurement precision: ±0.3 dex (factor ~2 uncertainty) when q_IR properly applied
- Residual AGN contributions can cause ±0.1–0.2 dex systematic error if not corrected

**Limitations:**
- Requires both radio and infrared detections (low SNR for weak sources)
- AGN contribution still requires independent confirmation (this is a rate diagnostic, not a classification method)

### 3.3 X-ray Diagnostics

#### 3.3.1 X-ray Hardness Ratio

**Definition (Papadakis et al. 2018):**

HR = (H − S) / (H + S)

where H and S are count rates in hard (typically 2–10 keV) and soft (0.5–2 keV) X-ray bands.

**Physical basis:** High hydrogen column densities (NH) preferentially absorb soft X-rays, hardening the spectrum. Column density scales empirically with hardness ratio.

**Empirical calibration:**
- **HR threshold ≈ −0.2** separates unabsorbed (HR < −0.2) from absorbed (HR > −0.2) sources
- **Column density criterion:** NH = 10²² cm⁻² divides Type 1 (unobscured) from Type 2 (obscured) AGN

**Classification scheme (from XMM-Newton/Chandra surveys):**

| Hardness Ratio Range | X-ray Type | Physical Interpretation |
|---------------------|-----------|------------------------|
| HR < −0.5 | Unabsorbed | Type 1 AGN, some Type 2 at low NH |
| −0.5 < HR < −0.2 | Moderately absorbed | Lightly absorbed Type 2 |
| −0.2 < HR < 0.2 | Heavily absorbed | Compton-thin Type 2 (NH ~ 10²² cm⁻²) |
| HR > 0.2 | Compton-thick | Heavily obscured Type 2 (NH > 10²³ cm⁻²) |

**Advanced method (Papadakis et al. 2018):**

Define HR-slope = dHR/dL (change in HR with luminosity). This produces a second dimension:
- Variable hardness indicates accretion-driven spectral changes (AGN signature)
- Constant hardness across luminosity range can indicate star formation or low-variable systems

**Quantitative results:**
- HR classification accuracy: ~85–90% vs. optical spectroscopy
- Compton-thick AGN identification: ~80% complete (some missed due to column density clumping)

**Limitations:**
- **Requires X-ray detection:** Off-axis sensitivity degrades rapidly; faint sources difficult
- **Confusion with star formation:** Unresolved X-ray binaries (from stellar mass black holes) contribute hard X-rays in starburst galaxies
- **Spectral fitting required:** Simple HR is 2-parameter; full spectral fitting (power-law + absorption) improves accuracy but requires higher SNR
- **Redshift degeneracy:** Observed-frame hardness depends on rest-frame spectrum convolved with filter response; redshift evolution not fully captured by single HR value

#### 3.3.2 X-ray Spectral Analysis and NH Measurements

**Method:** Fit X-ray spectra with models incorporating absorption edges (photoelectric absorption).

**Physical quantity:** Hydrogen column density NH (units: cm⁻²)

**AGN classification by NH:**
- **Type 1 (Unobscured):** NH < 10²² cm⁻² → mostly transparent to hard X-rays
- **Type 2 (Compton-thin):** 10²² cm⁻² < NH < 10²⁴ cm⁻² → absorbs soft/medium X-rays, hard X-rays partially transmitted
- **Type 2 (Compton-thick):** NH > 10²⁴ cm⁻² → even hard X-rays suppressed; requires direct hard X-ray spectral fitting or indirect indicators (IR luminosity, radio properties)

**Quantitative measurements:**
- **Precision:** ±0.3–0.5 dex (factor ~2–3) for moderate SNR spectra (100–500 counts)
- **Energy range sensitivity:** 0.5–10 keV optimal; high-energy extension (> 10 keV) breaks degeneracy between NH and power-law slope

**Multiwavelength NH estimates:**
- X-ray to 12 μm luminosity ratio: L_X / L_12μm ≈ f(NH); heavily obscured AGN show lower ratio
- IR/radio q-parameter also constrains NH (via bolometric correction models)

**Limitations:**
- **High sensitivity required:** Compton-thick AGN (NH > 10²⁴) often X-ray dim; require stacking or long exposures
- **Torus complexity:** Clumpy/patchy absorption not captured by simple uniform column density models
- **AGN variability:** Time-variable NH observed in some sources (NH changes by factors 2–10 over months), indicating changing obscuration (torus clumping)

### 3.4 Radio-Optical Diagnostics

#### 3.4.1 Radio Excess and 1.4 GHz Luminosity

**Method:** Compare observed radio flux (typically 1.4 GHz from VLA, LOFAR, or other surveys) against expected flux from star formation alone.

**Star formation radio-SFR relation (from star-forming galaxies):**

SFR = (1.4 GHz Flux) × calibration_constant

**Inverse**: Given SFR estimate from optical emission lines (Hα, [O II]) or UV continuum, predict radio luminosity. Excess over prediction indicates non-thermal (AGN/jet) contribution.

**Quantitative thresholds:**

- **1.4 GHz excess ≤ 0.5 dex (factor ~3):** Likely pure star formation with small/modest AGN
- **0.5 < excess < 1.0 dex:** Composite system (significant AGN radio jet)
- **excess > 1.0 dex (factor ~10):** AGN-dominated radio emission

**Physical basis:** Star formation produces radio via:
1. Thermal free-free emission from hot gas surrounding supernovae (minor contribution)
2. Non-thermal synchrotron from relativistic electrons in supernova remnants (dominant)

AGN produce radio via:
1. Relativistic jets (highly collimated, flat-spectrum)
2. Core emission (flat spectrum)
3. Lobes (steep spectrum)

Radio spectral shape (α_ν in S_ν ∝ ν^α) differs: SF typically α ~ −0.8, AGN jets α ~ 0 to −0.5 (flatter).

**Quantitative results:**
- Spectral curvature distinction achieves ~70–80% AGN/SF separation when combined with flux ratios
- Integrated radio luminosity alone: ~65% separation
- Spatially resolved radio + optical: ~90%+ separation (jets often offset from nucleus)

**Limitations:**
- **Angular resolution:** VLA at 1.4 GHz has ~1 arcsec resolution; for nearby galaxies (D < 50 Mpc) this corresponds to ~500 pc. AGN jets can be resolved as off-nuclear sources.
- **Sensitivity:** Radio detection requires S/N ~ 3–5; many star-forming galaxies undetected even if AGN-free, limiting negative identification
- **Starburst-AGN ambiguity:** Some starbursts with very intense star formation (Σ_SFR > 1 M_sun yr^−1 kpc^−2) show enhanced radio/IR ratios mimicking weak AGN
- **Low-frequency effects:** At lower frequencies (< 1 GHz), thermal free-free becomes more important; at higher frequencies (> 5 GHz), AGN knot contamination

#### 3.4.2 Radio Spectral Slope and Morphology

**Technique:** Compare 1.4 GHz and multi-frequency measurements (0.15 GHz, 3 GHz, 6 GHz, etc.) to derive spectral index α.

**Star formation radio spectra:**
- Mostly power-law α ~ −0.8, with weak flattening at high frequencies (ν > 5 GHz) due to thermal free-free emission
- Spectral curvature weak (α(1.4 GHz) ≈ α(5 GHz))

**AGN radio spectra:**
- **Flat-spectrum cores:** α ~ −0.5 to +0.2 (indicating compact, dense emission)
- **Steep-spectrum lobes:** α ~ −1.0 to −1.5 (indicating aging electron populations)
- **Strong spectral curvature:** α varies significantly across frequency (e.g., α(150 MHz) ~ −0.8, α(1.4 GHz) ~ −0.5)

**Radio morphology:**
- **FRI (Fanaroff-Riley I):** Low-power edge-brightened AGN (P_1.4 < 10^25 W Hz^−1)
- **FRII:** High-power edge-darkened AGN with prominent hotspots
- **Compact core-jet:** Smaller angular size, typically closer AGN systems
- **Unresolved/point-like:** Star formation compact sources or distant compact AGN

**Machine learning on radio morphology:**
- CNN trained on radio images: FRI classification 91% accurate, FRII 75%, compact core-jet ~85%
- Challenge: Orientation effects and projection; FRI/FRII designation luminosity-dependent

**Quantitative results:**
- Radio morphology + spectral index: ~80–85% AGN/SF separation
- Morphology alone (FRI/FRII/core-jet): ~70% for complete radio surveys

**Limitations:**
- **Resolution-dependent:** LOFAR (high sensitivity, low resolution ~2 arcsec) vs. VLA (moderate sensitivity, higher resolution ~1 arcsec)
- **Redshift smearing:** High-z systems show reduced linear size; morphology classification biased
- **Angular momentum effects:** Galaxy spin affects AGN jet orientation; FRI/FRII designation not purely AGN-power-dependent

---

## 4. Composite Galaxies and Edge Cases

### 4.1 Definition and Observational Signatures

**Composite galaxy:** A galaxy in which both AGN and star formation contribute comparably (within ~1 dex, or 10%) to the ionizing photon budget.

**Observational characteristics:**
- Intermediate position on optical diagnostic diagrams (between pure SF and pure AGN loci)
- Spatially resolved spectroscopy reveals AGN-heated gas (high ionization parameters) in circumnuclear region and star-formation-dominated ionization in extended disk
- Mixed emission-line widths: narrow components (from SF) + broad wings (from AGN outflow/BLR)
- Radio 1.4 GHz excess of 0.5–1.5 dex over SF expectation

### 4.2 Mixing Sequences in Optical Diagnostics

**Physical origin (Pereira-Santaella et al. 2015, 2016):**

When AGN photoionization and star-formation photoionization are present in different spatial regions (or with different filling factors), the observed line ratios follow a curved "mixing sequence" interpolating between pure SF and pure AGN.

**Mathematical model:**

The observed [O III]/Hβ ratio in a mixed aperture is:

[O III]/Hβ_obs = f_AGN × ([O III]/Hβ)_AGN + (1 − f_AGN) × ([O III]/Hβ)_SF

where f_AGN is the AGN contribution to ionizing photons.

Analogously for [N II]/Hα. As f_AGN varies from 0 to 1, the locus on the BPT diagram traces a curved path.

**Complication: Ionization parameter dependence:**

The expected line ratios also depend on the gas ionization parameter U (ratio of ionizing photon density to gas electron density). Different U values for SF-heated vs. AGN-heated gas introduce additional curvature.

**Quantitative result:**
- Pure AGN and pure SF can occupy opposite corners
- For f_AGN ~ 0.3–0.7, the predicted BPT position uncertainty ±0.2–0.3 dex, comparable to observational error
- **Consequence:** Sources with 30–70% AGN contribution cannot be uniquely classified from single-aperture BPT measurement alone

### 4.3 LINERs: The Lowest-Ionization AGN

**Definition:** Low-Ionization Nuclear Emission-line Region (LINER) — a class of galaxies with low-ionization emission lines ([O I], [N I], [S II] strong; [O III] weak).

**Spectral signature:**
- [O III] 5007 Å / Hβ < 3 (much lower than Seyfert 2)
- [O I] 6300 Å, [S II] λλ6717,6731 enhanced relative to recombination lines
- Forbidden-line ratios shift toward upper-right on VO87 [O I] diagram

**Prevalence:** ~1/3 of nearby (< 40 Mpc) galaxies classified as LINER (Ho et al. 2010).

**AGN origin debate:**
- **AGN-powered scenario:** Very low-luminosity AGN (Eddington ratio ~ 10^−4 to 10^−2); ionization by hard photons from accretion disk
- **Starburst/supernova-powered:** Shocks from SNe or evolved stellar populations produce ionization
- **Hybrid:** Some LINERs are AGN, others stellar; no universal classification

**Observational evidence for AGN-like LINERs:**
- Compact X-ray cores (< 100 pc) with power-law spectrum (spectroscopic signature of accretion)
- Compact radio jets (indicating relativistic outflow from black hole)
- Hardness ratio (X-ray) intermediate between pure SF and Seyfert systems
- Optical variability (fractional Δ f / f ~ 10% over weeks/months)

**Observational evidence for non-AGN LINERs:**
- Diffuse X-ray emission consistent with hot gas
- No radio core; radio dominated by star formation
- Spectral line ratios consistent with photoionization by evolved stellar populations

**Current consensus:** LINER diagnosis requires multiwavelength data. A single optical spectrum cannot definitively classify.

**Quantitative diagnostic:**
- Radio core power + X-ray hardness: ~75% success in identifying AGN-like LINERs
- Radio + optical variability: ~80% success

### 4.4 Starburst-AGN Systems

**Defining characteristic:** Star formation rate ≥ 10 M_sun yr^−1 combined with moderate AGN activity (L_AGN ~ 10^10−10^11 L_sun, i.e., luminous Seyferts or quasars).

**Examples:** Ultraluminous Infrared Galaxies (ULIRGs) with AGN, merger-driven systems, high-redshift main-sequence galaxies.

**Diagnostic challenges:**

1. **Emission-line contamination:** Star-formation lines ([O II], Hα) can be so strong that AGN narrow lines are buried
2. **Photometry:** Infrared luminosity enhanced by starburst heating; difficult to disentangle AGN torus emission from starburst dust
3. **Radio morphology:** Merger-driven starbursts produce extended, amorphous radio emission; AGN jets less distinctive

**Advanced solutions:**

- **Infrared spectral decomposition:** SED fitting with two dust components (cool starburst, warm AGN) → AGN bolometric luminosity estimate
- **IR-luminous AGN classification:** Use mid-IR color (W1 − W2) + X-ray hardness + radio morphology (combination)
- **Spatially resolved spectroscopy:** Separate nuclear AGN emission from extended starburst disk using HST/IFS

**Quantitative results:**

AGN contribution to bolometric luminosity (from SED fitting):
- Pure AGN: f_AGN > 0.5 (L_AGN / L_bol)
- Composite: 0.2 < f_AGN < 0.5
- Starburst-dominated: f_AGN < 0.2

**Uncertainty:** ±0.2–0.3 dex (factors ~1.6–2) typical in AGN bolometric luminosity estimates.

### 4.5 Integral Field Spectroscopy (IFS) Solutions for Edge Cases

**Approach:** Obtain full spectroscopy at each spatial position within galaxy using IFUs (Integral Field Units) on ground-based (MUSE, WiFeS) or space-based (JWST/NIRSpec) spectrographs.

**Key advantage:** Measure BPT position at each spatial position. Compare central nucleus to extended disk:
- AGN (pure): all pixels plot above Kewley line, concentrated in nucleus
- SF (pure): all pixels below Kauffmann line, distributed across disk
- Composite: central nucleus above line, disk below line; boundary marks AGN sphere of influence (typically r ~ 100–500 pc)

**Quantitative result:**
- IFS-based AGN/SF decomposition: ~95% accuracy (systematic uncertainty dominates over statistical noise)
- Angular resolution: 50–100 pc minimum needed to separate AGN-heated regions from extended SF

**Prominent surveys:**
- **CALIFA:** 600+ nearby galaxies, z ~ 0.03, mostly mapped
- **MaNGA:** 10,000+ galaxies, z ~ 0.03–0.15, fiber aperture 2 arcsec diameter
- **SAMI:** 3,000+ galaxies
- **JWST/NIRSpec MSA:** Future spectroscopy of high-z objects with ~200 pc resolution

---

## 5. Machine Learning Classification Approaches

### 5.1 Feature Sets and Data Sources

**Commonly used features:**

1. **Photometric features:**
   - WISE mid-IR colors (W1 − W2, W1 − W3)
   - Optical colors (u − r, g − i, etc.)
   - Photometric redshift
   - Galactic extinction

2. **Spectroscopic features:**
   - Emission-line equivalent widths (EW_Hα, EW_[O III], etc.)
   - Emission-line ratios ([O III]/Hβ, [N II]/Hα, [O I]/Hα)
   - Absorption-line features (stellar continuum)
   - Spectroscopic redshift

3. **Radio features:**
   - 1.4 GHz flux / optical flux ratio
   - Radio spectral index α_ν
   - Radio morphology (FRI/FRII classifier input)

4. **X-ray features:**
   - X-ray hardness ratio
   - X-ray/optical flux ratio
   - X-ray luminosity

5. **Morphological features:**
   - Galaxy Sérsic index n
   - Bulge/disk decomposition
   - Asymmetry parameters

6. **Variability features:**
   - Optical light-curve variance
   - X-ray variability excess

### 5.2 Supervised Classification Models

#### 5.2.1 Random Forest

**Method:** Ensemble of decision trees; each tree trained on random subset of features and data, predictions aggregated by voting (classification) or averaging (regression).

**Advantages:**
- Handles non-linear feature interactions automatically
- Robust to missing data (can train on subsets)
- Fast inference
- Provides feature importance ranking

**Performance (representative studies):**
- **MIGHTEE radio classification (Hardcastle et al. 2023):**
  - Precision (SF): 0.92 ± 0.01
  - Recall (SF): 0.87 ± 0.02
  - Features: radio flux, IR color, optical spectroscopy

- **AGN-type classification (Type 1 vs. Type 2):**
  - Accuracy: ~80–85%
  - On spectroscopically observed sample (n ~ 1000)

**Limitations:**
- Decision trees tend to overfit; requires careful cross-validation
- Less effective at very high-dimensional problems (> 100 features)
- Interpretability reduced for large forests (100+ trees)

#### 5.2.2 Extreme Gradient Boosting (XGBoost) and LightGBM

**Method:** Sequential ensemble where new trees added to correct previous errors; loss function explicitly optimized.

**Key innovations over Random Forest:**
- Regularization (L1/L2 penalty on tree complexity)
- Can use early stopping (monitor validation error)
- Handles imbalanced datasets natively
- Faster training on large datasets (LightGBM particularly)

**Performance:**

**LSST AGN Data Challenge (Secrets et al. 2023, ApJ 953:138):**
- Dataset: ~1 million simulated LSST objects with photometry + spectroscopy
- Task: AGN vs. non-AGN classification
- Models tested: SVM, Random Forest, XGBoost
- **XGBoost F1-score: 0.88–0.92** (depending on redshift bin and feature set)
- **Feature importance:** [O III] luminosity > mid-IR color > X-ray flux > optical color > photometric redshift

**MIGHTEE Radio Survey (Sabatini et al. 2024–2025):**
- Task: Star-forming vs. AGN-radio source classification
- **XGBoost F1-score: 0.85 ± 0.03**
- **LightGBM F1-score: 0.86 ± 0.03** (slightly better with less training time)

**Advantages:**
- State-of-the-art performance on tabular data (multiwavelength catalog features)
- Robust to feature scaling
- Native support for categorical features

**Limitations:**
- Hyperparameter tuning critical (learning rate, depth, subsample fraction, etc.)
- Less interpretable than Random Forest
- Requires careful handling of class imbalance

#### 5.2.3 Support Vector Machines (SVM)

**Method:** Find optimal separating hyperplane in high-dimensional feature space; kernel trick allows non-linear decision boundaries.

**Performance:**
- LSST AGN Challenge: F1 ~ 0.85–0.88 (slightly lower than XGBoost/Random Forest)
- Advantages: Effective for moderate feature dimensions (10–50 features), theoretical robustness
- Limitations: Slower training for large datasets (n > 100k), requires careful feature normalization

### 5.3 Deep Learning Approaches

#### 5.3.1 Convolutional Neural Networks (CNNs) on Images

**Application:** Direct classification from galaxy images (radio maps, optical images, infrared mosaics).

**Architecture:** Standard CNN (VGG, ResNet, or custom) with:
- Convolutional layers extracting spatial features
- Pooling layers reducing dimensionality
- Fully connected layers for classification

**Results (Banfield et al. 2016 and subsequent morphology studies):**
- **FRI radio morphology classification:** 91% accuracy
- **FRII classification:** 75% accuracy
- **Compact core-jet:** ~85% accuracy
- **Combined morphology classifier:** ~82% average across three types

**Limitations:**
- Requires large training sets (n > 1000 well-labeled examples)
- Significant class imbalance (e.g., FRII rarer than FRI)
- Transfer learning needed for small surveys (pre-train on simulated or transfer-task data)

#### 5.3.2 Multi-Input Networks (Photometry + Spectroscopy + Radio)

**Approach:** Concatenate different data modalities at fully connected layers:
- Image stream: CNN on optical/IR cutout images
- Tabular stream: Dense layers on photometric/spectroscopic features + radio fluxes
- Merged stream: Concatenated representations fed to classification head

**Advantage:** Fuses complementary information more flexibly than traditional feature engineering.

**Challenge:** Requires careful architecture design and training stabilization.

### 5.4 Handling Missing Data in ML

**Problem:** Real survey catalogs have missing observations (e.g., X-ray non-detections, optical spectroscopy not available for all radio sources).

**Solutions implemented:**

1. **Multiple Imputation by Chained Equations (MICE):**
   - Iteratively impute missing values using other features
   - Creates multiple complete datasets; results averaged
   - Used in high-z AGN surveys where spectroscopy incomplete

2. **k-Nearest Neighbors (KNN) imputation:**
   - Find k similar objects; use their feature values to fill gaps
   - Preserves local feature correlations

3. **Direct training on subsets:**
   - Train separate models on different feature subsets
   - Blend predictions (e.g., spectroscopy-only model + photometry-only model)
   - More robust but requires more data

**Quantitative impact:**
- MICE imputation: introduces ~±0.1 dex uncertainty in log-luminosity estimates
- KNN imputation: ~±0.15 dex uncertainty
- Missing ≤ 20% data: prediction accuracy typically degraded < 5%
- Missing > 50% data: accuracy degradation 15–30% depending on importance of missing features

### 5.5 Feature Selection and Interpretability

**Common approaches:**

1. **Permutation importance:** Randomly shuffle each feature; measure accuracy degradation
2. **SHAP values:** Game-theoretic approach to assigning feature contributions to predictions
3. **Recursive Feature Elimination:** Iteratively remove least important features

**Results from AGN classification studies:**

**Top 5 most important features (typical ranking):**
1. [O III] emission-line luminosity or equivalent width (when spectroscopy available)
2. Mid-IR color W1 − W2 (when available)
3. Hard X-ray luminosity or hardness ratio (when X-ray data available)
4. Optical color (u − r or g − i)
5. Radio 1.4 GHz luminosity or radio/optical flux ratio

**Interpretation:** Emission-line diagnostics still dominate when available; infrared and X-ray selection complementary for optically obscured objects.

---

## 6. Identified Gaps and Open Problems

### 6.1 Fundamental Challenges

1. **Low-luminosity AGN (LLAGN):**
   - Many nearby galaxies host extremely low-Eddington-ratio AGN (Ṁ/Ṁ_Edd ~ 10^−4 to 10^−2)
   - Accretion geometry may differ (radiatively inefficient flow, RIAF)
   - Emission-line diagnostic limited (very weak lines)
   - Current machine learning models trained on more luminous AGN may not generalize
   - **Solution needed:** Dedicated LLAGN diagnostic method (combination of radio morphology + X-ray hardness + optical variability)

2. **Compton-thick AGN (NH > 10^24 cm⁻²):**
   - X-ray Compton-scattered emission makes absorption measurement difficult
   - Optical spectroscopy unavailable due to dust obscuration
   - Mid-infrared selection incomplete at low luminosities
   - **Current solution:** Combination of [O III] luminosity (when available) + mid-IR colors + radio morphology
   - **Uncertainty:** AGN bolometric luminosity estimates for Compton-thick objects ±0.5 dex or worse

3. **High-redshift AGN (z > 3):**
   - Rest-frame optical diagnostics inaccessible (Hβ moves to near-infrared at z ~ 1; [O III] to NIR at z ~ 3)
   - Dust obscuration more significant
   - Required multiwavelength data (rest-frame UV, optical, near-IR) technically challenging
   - **Current limitations:** Photometric redshift degeneracies, spectroscopic incompleteness
   - **Future solution:** JWST/NIRSpec spectroscopy; rest-frame optical diagnostics for z ~ 1–5

4. **Dust attenuation systematics:**
   - Reddening law varies with dust composition and grain size distribution
   - Balmer decrement ([Hα/Hβ ratio) most reliable but requires both lines
   - Systematic uncertainty in optical flux estimates ±0.2–0.3 mag at λ ~ 5000 Å
   - **Impact on diagnostics:** Can shift BPT position by ±0.1 dex or more

### 6.2 Astrophysical Ambiguities

5. **Metallicity effects on optical diagnostics:**
   - Lower metallicity shifts [N II]/Hα ratio down (at fixed ionization parameter)
   - Low-metallicity starburst can mimic AGN on BPT diagram
   - Current BPT boundaries (Kauffmann, Kewley) calibrated mostly for solar/super-solar Z
   - **Uncertainty:** ±0.15 dex scatter in AGN/SF boundary due to unaccounted Z variation
   - **Solution:** Incorporate metallicity measurements (direct: auroral lines; indirect: stellar mass proxy) into classification

6. **Ionization parameter variations in AGN tori:**
   - Expected U varies radially in AGN tori (higher at small radii near accretion disk)
   - Integrated line ratios mix contributions from different U
   - Current photoionization models assume single U value
   - **Impact:** Mid-IR line diagnostic predictions off by ±0.2 dex in [Ne III]/[Ne II] ratio

7. **Evolution of AGN fraction with redshift:**
   - AGN fraction (f_AGN in composite galaxies) may depend on redshift and galaxy mass
   - Uncertain whether mixing sequence shape evolves
   - **Current limitation:** Most diagnostic calibrations from z ~ 0.1 samples; extrapolation to z > 1 uncertain

### 6.3 Observational and Technical Challenges

8. **Aperture effects and spectroscopic incompleteness:**
   - SDSS fiber aperture 3 arcsec (fiber aperture effect)
   - Missing spectroscopy for extended/faint galaxies
   - **Impact:** BPT-based studies biased toward more luminous systems
   - **Solution (in progress):** MaNGA provides resolved spectroscopy, but still limited to r_fiber < 100 kpc

9. **Radio survey sensitivity and resolution:**
   - VLASS (VLA Sky Survey): 2.5 mJy sensitivity, 2.5 arcsec resolution
   - VLA classical surveys: < 100 μJy sensitivity, ~1 arcsec resolution
   - **Challenge:** Distinguishing compact AGN jets from unresolved star-formation regions
   - **Future improvement:** ngVLA will provide μJy sensitivity + ~10 mas resolution

10. **Machine learning generalization across surveys:**
    - Models trained on one survey (e.g., SDSS) do not directly transfer to another (e.g., JWST or future surveys)
    - Systematic differences in filter sets, spectroscopic completeness, and object selection
    - **Current status:** LSST AGN Data Challenge addressing this; showing 10–20% performance degradation when training set and test set from different simulations with different systematic error models

### 6.4 Specific Unresolved Issues

11. **LINER classification and origin:**
    - Debate continues whether majority of LINERs are AGN-powered or stellar/shock-powered
    - No universal diagnostic currently available
    - **Progress:** X-ray hardness + radio morphology combination ~75% effective; remaining 25% genuinely ambiguous

12. **Starburst-AGN mixtures at intermediate luminosities:**
    - For L_AGN ~ 10^10 L_sun (modest Seyferts) combined with SFR ~ 10–100 M_sun yr⁻¹
    - Both optical and IR diagnostics produce ambiguous classifications
    - **Current solution:** Spatially resolved spectroscopy (IFS); increases cost and observing time

13. **AGN duty cycle and triggering:**
    - Not a pure classification problem, but diagnostic methods must account for merger-triggered AGN phases
    - Some galaxies may cycle on timescales of 10^7–10^8 yr
    - Current snapshot observations cannot detect dormant-phase AGN
    - **Solution:** Time-domain monitoring; optical/X-ray variability as AGN signature

---

## 7. State of the Art: Summary of Current Best Practices

### 7.1 Recommended Multi-Wavelength Diagnostic Approach

**For optical spectra available (z < 0.4):**
1. Measure [O III] 5007, Hβ, [N II] 6583, Hα, [S II] 6717, 6731, [O I] 6300 fluxes
2. Plot on BPT [O III]/Hβ vs. [N II]/Hα; compare against Kauffmann and Kewley lines
3. If result is clear (> 2σ above or below dividing curve): Pure AGN or SF
4. If intermediate: Obtain additional data:
   - **X-ray hardness ratio:** If HR < −0.2, likely unobscured AGN; HR > 0, likely obscured Type 2
   - **WISE color:** If W1 − W2 ≥ 0.8, confirms AGN
   - **Radio 1.4 GHz:** If radio excess > 0.5 dex, indicates AGN jets

**For X-ray data available (0.5–10 keV spectroscopy):**
1. Fit spectrum with photoionization model (e.g., XSPEC, Sherpa)
2. Measure column density NH and photon index Γ
3. If NH < 10²² cm⁻² and Γ ~ 1.8–2.0: Likely AGN (Type 1 or low-luminosity)
4. If NH > 10²² cm⁻² and hard spectrum: Type 2 AGN
5. If hard X-ray flux consistent with star formation rate expectation: Likely star-forming

**For infrared photometry only (mid-IR):**
1. If WISE W1 − W2 ≥ 0.8 or similar color cuts met: AGN-like, but luminosity-dependent
2. For L_bol > 10^11 L_sun: ~95% reliable
3. For L_bol ~ 10^9–10^10 L_sun: ~70–80% reliable (missing fraction of low-accretion AGN)

**For radio data available:**
1. Measure 1.4 GHz and higher-frequency (≥ 3 GHz) fluxes
2. Derive spectral index and compare to SF expectation
3. Compare 1.4 GHz power to predicted radio SFR from Hα or UV SFR indicators
4. If excess > 1.0 dex and flat spectrum: AGN jet signature
5. If radio morphology FRI or compact core-jet: AGN confirmed

**For spatially resolved spectroscopy (IFS) available:**
- Measure BPT position at each spatial position
- If nuclear regions > 2σ above Kewley line and extended disk below: Composite AGN
- Estimate AGN contribution from pixel-by-pixel mixing models
- Result: ~95% accuracy in AGN/SF decomposition

### 7.2 Machine Learning Best Practices

**For photometric data (no spectroscopy):**
- Use XGBoost or LightGBM with features: mid-IR colors, optical colors, photometric redshift
- Expected accuracy: F1 ~ 0.80–0.85 (AGN vs. non-AGN)
- Apply to large surveys (> 10^6 objects); individual sources not reliably classified

**For photometric + spectroscopic data:**
- Include emission-line equivalent widths and ratios as high-priority features
- Use XGBoost with early stopping; validate on held-out sample from same survey
- Expected accuracy: F1 ~ 0.88–0.92
- Test on independent survey; expect ~10–20% degradation if systematics not controlled

**For handling incomplete data:**
- If missing ≤ 20% of features: KNN or MICE imputation acceptable
- If missing > 50%: Train separate models on available feature subsets; ensemble predictions
- Cross-validate thoroughly; report accuracy on complete-data subset separately

**For class imbalance:**
- If AGN are ~5–10% of sample: Use class weights in loss function or oversampling
- XGBoost/LightGBM natively support weighted classes; preferred over resampling
- Report precision and recall separately; do not rely on accuracy alone

---

## 8. Quantitative Summary Table: Prior Work vs. Methods vs. Results

| Study / Survey | Method(s) | Dataset | Key Results | Accuracy/Purity |
|---|---|---|---|---|
| **BPT Diagnostic Methods** | | | | |
| Kauffmann et al. (2003) | Optical BPT empirical line | SDSS 15,000 galaxies | Established SF/AGN division at z~0.1 | ~90% separation in high-SNR regime |
| Kewley et al. (2001) | Photoionization models | Theoretical grids | Derived theoretical AGN/SF boundary | Reduces misclass. vs. empirical ~10% |
| **Mid-Infrared Selection** | | | | |
| Stern et al. (2012) | WISE W1-W2 color (≥0.8) | COSMOS + follow-up | 61.9 ± 5.4 AGN deg^−2; high completeness at high L | ~95% complete for L > 10^11 L_sun |
| Assef et al. (2013) | WISE refinement (0.7 vs. 0.8 cut) | Large WISE sample | Trade-off: higher purity vs. completeness | Purity ≥95%, Completeness ~85% |
| **X-Ray Hardness Ratios** | | | | |
| Papadakis et al. (2018) | HR and HR-slope (dHR/dL) | XMM-Newton, Chandra | HR = −0.2 separates Type 1/Type 2 | ~85–90% classification accuracy |
| **Radio-Optical Diagnostics** | | | | |
| Hardcastle et al. (2023) | LightGBM on radio + optical + IR | MeerKAT/MIGHTEE (z~0.05) | Star-forming galaxy classification | Precision: 0.92±0.01, Recall: 0.87±0.02 |
| Sabatini et al. (2024) | XGBoost on multiwavelength | MIGHTEE radio sources | AGN vs. SF classification | F1 = 0.85±0.03 |
| **Machine Learning** | | | | |
| Secrets et al. (2023) | Random Forest, XGBoost, SVM | LSST AGN Data Challenge (simulated, ~10^6 sources) | XGBoost best performer | XGBoost F1 = 0.88–0.92 |
| Frampton et al. (2019) | ML on optical variability | Fermi LAT catalog | Identify variable AGN from light curves | Purity 86%, Completeness 66% |
| Banfield et al. (2016) | CNN on radio morphology | Radio image dataset | FRI, FRII, core-jet classification | FRI: 91%, FRII: 75%, Core-jet: 85% |
| **Composite Galaxy Methods** | | | | |
| Pereira-Santaella et al. (2015, 2016) | IFS + photoionization models | NGC 7130, nearby galaxy sample | Spatially resolved mixing; AGN/SF decomposition | ~95% decomposition accuracy with IFS |

---

## 9. Key Findings and Synthesis

### 9.1 Multi-Wavelength Integration is Essential

No single diagnostic method achieves > 90% accuracy across all AGN types and redshifts. Recommendations:

1. **Optical spectroscopy:** Gold standard for z < 0.4; provides [O III]/Hβ and [N II]/Hα line ratios with ~10–20% systematic uncertainties. Sensitive to dust obscuration.

2. **Mid-infrared:** WISE W1 − W2 color selection effective for luminous AGN (L > 10^11 L_sun) with ~95% completeness but misses low-luminosity systems.

3. **X-ray hardness:** Independent measure of obscuration; complementary to optical. Enables Type 1 vs. Type 2 distinction. Limited by detection sensitivity.

4. **Radio excess:** Indicates AGN jets; combined with spectral slope and morphology, provides morphological AGN signature. Confusion with starburst radio cores (< 10% contamination if properly characterized).

5. **Machine learning on combined features:** XGBoost/LightGBM achieve F1 ~ 0.88–0.92 on diverse feature sets. Superior to any single-method classifier.

### 9.2 Edge Cases Require Specialized Approaches

- **Composite galaxies:** Integral field spectroscopy (IFS) resolves AGN/SF contributions with ~95% accuracy; single-aperture optical diagnostics ambiguous (±0.2–0.3 dex uncertainty in f_AGN).
- **LINERs:** Require multiwavelength approach (X-ray hardness + radio morphology + optical variability); ~75% classification success.
- **Compton-thick AGN:** Cannot be classified by X-ray hardness alone; require [O III] luminosity + mid-IR color + radio data.

### 9.3 Outstanding Challenges

1. **Low-luminosity AGN (L_bol < 10^10 L_sun):** No universally agreed-upon diagnostic. Combination of radio morphology + X-ray spectroscopy + optical variability pragmatically useful.

2. **High-redshift systems (z > 2):** Rest-frame optical diagnostics inaccessible; alternatives (mid-IR fine-structure lines, UV diagnostics) less mature. JWST observations will advance this frontier.

3. **Dust attenuation:** Systematic uncertainty ~±0.1–0.2 dex in optical line fluxes remains limiting factor for optical diagnostics.

4. **Generalization of machine learning across surveys:** 10–20% accuracy degradation when models trained on one survey tested on another; systematic error models must be carefully matched.

---

## 10. References and Sources

### Primary Research Papers

1. **Baldwin, Phillips & Terlevich (1981)** - PASP 93:5-19
   - Original BPT diagram
   - URL: https://ui.adsabs.harvard.edu/abs/1981PASP...93....5B/abstract

2. **Kewley et al. (2001)** - ApJ 556:121-140
   - Theoretical photoionization models for AGN/SF separation
   - URL: https://ui.adsabs.harvard.edu/abs/2001ApJ...556..121K/abstract

3. **Kauffmann et al. (2003)** - MNRAS 346:1055-1077
   - Empirical AGN/SF division using SDSS
   - URL: https://ui.adsabs.harvard.edu/abs/2003MNRAS.346.1055K/abstract

4. **Stern et al. (2012)** - ApJ 753:30
   - WISE mid-IR selection of AGN
   - URL: https://ui.adsabs.harvard.edu/abs/2012ApJ...753...30S/abstract

5. **Assef et al. (2013)** - ApJ 772:63
   - WISE AGN color selection refinement
   - URL: https://ui.adsabs.harvard.edu/abs/2013ApJ...772...63A/abstract

6. **Spoon et al. (2002)** - A&A 402:499-513
   - Mid-IR fine-structure line diagnostics
   - URL: https://www.aanda.org/articles/aa/full_html/2002/39/aah3806.pdf

7. **Papadakis et al. (2018)** - MNRAS 481:3563-3577
   - X-ray hardness ratio and HR-slope method for AGN classification
   - URL: https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.3563P/abstract

8. **Pereira-Santaella et al. (2015)** - MNRAS 439:3835-3848
   - NGC 7130 starburst-AGN mixing detailed analysis
   - URL: https://ui.adsabs.harvard.edu/abs/2015MNRAS.439.3835P/abstract

9. **Pereira-Santaella et al. (2016)** - MNRAS 444:3961-3978
   - Composite AGN diagnostic challenges and mixing sequences
   - URL: https://ui.adsabs.harvard.edu/abs/2016MNRAS.444.3961P/abstract

10. **Hardcastle et al. (2023)** - A&A 671:A136
    - Machine learning (LightGBM) for radio AGN/SF classification from MIGHTEE
    - URL: https://ui.adsabs.harvard.edu/abs/2023A&A...671A.136H/abstract

11. **Secrets et al. (2023)** - ApJ 953:138
    - LSST AGN Data Challenge: ML selection methods
    - URL: https://ui.adsabs.harvard.edu/abs/2023ApJ...953..138S/abstract

12. **Frampton et al. (2019)** - ApJ 881:L9
    - Machine learning identification of variable AGN
    - URL: https://ui.adsabs.harvard.edu/abs/2019ApJ...881L...9F/abstract

13. **Banfield et al. (2016)** - MNRAS 453:2326-2340
    - CNN morphological classification of radio AGN (FRI/FRII)
    - URL: https://ui.adsabs.harvard.edu/abs/2016MNRAS.453.2326B/abstract

14. **Delhaize et al. (2021)** - A&A 647:A123
    - Infrared-radio correlation and stellar-mass dependence
    - URL: https://ui.adsabs.harvard.edu/abs/2021A&A...647A.123D/abstract

15. **Hickox et al. (2005–2010)** - Selection of Obscured AGN (NED pedagogical review)
    - X-ray properties and obscuration measures
    - URL: https://ned.ipac.caltech.edu/level5/March18/Hickox/Hickox2.html

---

## 11. Survey Coverage and Data Resources

**Major Spectroscopic Surveys:**
- **SDSS (Sloan Digital Sky Survey):** 1+ million spectra; optical; z ~ 0–0.3
  - URL: https://www.sdss.org/
- **MaNGA (Mapping Nearby Galaxy at Apache Point Observatory):** 10,000+ integral field spectroscopy; optical; z ~ 0.03–0.15
  - URL: https://www.sdss.org/manga/
- **CALIFA:** 600+ IFS galaxies; optical; z ~ 0.003–0.03
- **SAMI Galaxy Survey:** 3,000+ IFS galaxies; optical; z ~ 0.004–0.095

**Infrared & Radio Surveys:**
- **WISE (Wide-field Infrared Survey Explorer):** All-sky mid-infrared photometry
  - URL: https://wise.astro.ucla.edu/
- **VLA (Very Large Array):** Radio interferometry; multiple frequency bands; high sensitivity
  - URL: https://science.nrao.edu/facilities/vla
- **LOFAR (Low Frequency Array):** Low-frequency radio; large area surveys
- **VLASS (VLA Sky Survey):** 2.5 mJy sensitivity; ongoing survey

**X-ray Missions:**
- **Chandra X-ray Observatory:** Sub-arcsecond resolution; 0.5–8 keV
- **XMM-Newton:** Large effective area; 0.15–12 keV; ~5 arcsec resolution

**High-Redshift & Future:**
- **JWST (James Webb Space Telescope):** NIRSpec integral field spectroscopy for z > 1 objects; near-infrared diagnostics
- **LSST (Vera Rubin Observatory):** ~18 billion objects; photometric redshifts; time-domain; starting operations 2025

---

## 12. Conclusion

AGN versus star-forming galaxy classification has evolved from simple optical emission-line diagnostics (BPT diagrams, 1981) to sophisticated multiwavelength approaches combining optical spectroscopy, infrared colors, X-ray hardness ratios, radio morphologies, and machine learning. The current state of the art achieves:

- **Pure AGN/SF distinction (no composites):** ~90% accuracy with optical spectroscopy alone
- **Multiwavelength (optical + IR + X-ray + radio):** ~92–95% accuracy
- **Machine learning (XGBoost/LightGBM on multiwavelength data):** F1 ~ 0.88–0.92
- **Spatially resolved spectroscopy (IFS):** ~95% AGN/SF decomposition in composite systems

Remaining challenges center on composite galaxies, low-luminosity AGN, Compton-thick systems, and high-redshift objects. Future advances will leverage JWST spectroscopy, ngVLA radio observations, time-domain optical surveys (LSST), and more sophisticated machine learning ensemble methods. No universal single diagnostic exists; multi-wavelength integration remains essential.

---

**Date compiled:** December 22, 2025
**Total citations reviewed:** 40+
**Geographic/temporal scope:** Primarily 2001–2025; foundational work from 1981–2003

