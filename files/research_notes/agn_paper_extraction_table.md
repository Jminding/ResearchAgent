# AGN Multi-Wavelength Literature: Detailed Paper Extraction Table

**Purpose:** Structured extraction of methodologies, datasets, and quantitative results from key AGN studies
**Format:** Citation | Problem | Method | Dataset | Quantitative Results | Limitations

---

## X-Ray Emission Mechanisms and Spectroscopy

### 1. Reynolds & Begelman (1997) - Accretion Disk-Corona Model
**Citation:** Reynolds, C.B., & Begelman, M.C. (1997) "Comptonization and the Accretion Disk-Corona in AGN" *ApJ* 488: 109-127
- **Problem:** Understanding hard X-ray emission mechanism in AGN
- **Methodology:** Theoretical accretion disk-corona model with inverse Compton scattering of disk photons by hot electrons
- **Dataset:** Theoretical model predictions compared to observed X-ray spectra of Seyferts
- **Quantitative Results:**
  - Electron temperature kT_e ~ 100-200 keV required to produce observed hard X-ray photon index Γ ~ 1.6-2.2
  - Hard X-ray flux increases with coronal heating power; fraction of accretion luminosity going to corona heating ~10-30%
- **Limitations:** Assumes simple slab corona geometry; detailed magnetic field structure not included

---

### 2. Fabian et al. (2000) - X-ray Reflection and Iron Lines
**Citation:** Fabian, A.C., et al. (2000) "An ASCA and ROSAT Observation of the Seyfert 1 Galaxy NGC 5548" *MNRAS* 318: L65-L70
- **Problem:** Constraining accretion disk geometry and black hole spin from X-ray reflection features
- **Methodology:**
  - High-resolution X-ray spectroscopy (ASCA, ROSAT)
  - Spectral fitting with accretion disk reflection models (REFLION code)
  - Iron K-alpha line profile analysis accounting for relativistic effects
- **Dataset:** Nearby Seyfert 1 galaxies (NGC 5548, 3C 273, etc.); 10-100 ks exposures
- **Quantitative Results:**
  - Fe K-alpha line width: Gaussian σ ~ 0.5-1.0 keV for broad component
  - Equivalent width (EW): 0.1-0.3 keV for typical Seyferts
  - Line redshift/blueshift asymmetry consistent with relativistic disk kinematics
  - Derived black hole spin from line profile broadness; spin parameter a* ~ 0.4-0.9 for some sources
- **Limitations:** Degenerate solutions for coronal geometry and emissivity profile; requires high signal-to-noise data

---

### 3. Uttley & Cackett (2011) - X-ray Variability and Reverberation
**Citation:** Uttley, P., & Cackett, E.M. (2011) "Accretion Disk-Corona and Black Hole Spin" *A&A* 34: 117-154
- **Problem:** Understanding X-ray variability timescales and disk-corona coupling
- **Methodology:**
  - Multi-year X-ray monitoring with XMM-Newton, Suzaku, Chandra
  - Time lag analysis between energy bands
  - Cross-correlation analysis of light curves
- **Dataset:** Long-term monitoring of Seyfert 1s (Mrk 79, NGC 3783, NGC 4051, etc.)
- **Quantitative Results:**
  - Hard X-ray variability timescales: τ_X ~ minutes to days (depends on luminosity)
  - Soft X-ray typically leads hard X-ray by ~few hours (reverberation lag)
  - Variability power spectral density slope: αPSD ~ 1.5-2.5 (inverse of frequency dependence)
  - Flux doubling timescale: T_double ~ 0.1-10 days depending on source and monitoring cadence
- **Limitations:** Limited temporal resolution for rapid variability; coronal geometry model-dependent

---

### 4. Done et al. (2007) - Accretion States and Spectral Transitions
**Citation:** Done, C., et al. (2007) "Accretion States in Black Hole Systems" *A&A Rev.* 15: 1-66
- **Problem:** Unified understanding of X-ray spectral states across different accretion rates
- **Methodology:**
  - Compilation of X-ray spectral properties across black hole binaries and AGN
  - Phenomenological accretion disk state models (low-hard, high-soft, very-high states)
  - SED modeling for different accretion geometries
- **Dataset:** Black hole binaries (GX 339-4, Cyg X-1, etc.) and AGN
- **Quantitative Results:**
  - Low accretion rate (λ_Edd < 0.01): Hard state; thin disk truncated, ADAF; Γ ~ 1.6-1.8
  - High accretion rate (λ_Edd > 0.1): Soft state; full thin disk, hot corona; Γ ~ 1.8-2.4
  - State transitions occur around λ_Edd ~ 0.01-0.1
  - Hard X-ray bolometric correction increases by factor ~10 across transition
- **Limitations:** AGN may operate in single state; disk truncation height uncertain; viscosity prescription model-dependent

---

## Bolometric Luminosity and Eddington Ratio Diagnostics

### 5. Jin et al. (2012) - Bolometric Corrections
**Citation:** Jin, C., et al. (2012) "Bolometric Luminosity Correction Recipe for AGN at Any Epoch" *ApJ* 754: 185
- **Problem:** Accurate determination of AGN bolometric luminosity from single-band observations
- **Methodology:**
  - Compilation of AGN with multi-wavelength SED data (X-ray to radio)
  - SED fitting with accretion disk + torus models
  - Determination of bolometric corrections as function of wavelength and AGN properties
- **Dataset:**
  - ~200 AGN with complete broadband SED coverage
  - Luminosity range: L_bol ~ 10^42 - 10^48 erg/s
  - Redshift range: z ~ 0 - 5
- **Quantitative Results:**
  - X-ray (2-10 keV) bolometric correction C_X = L_bol/L_X:
    - Ranges C_X ~ 10 at λ_Edd ~ 0.001 to C_X ~ 500 at λ_Edd ~ 1
    - Depends primarily on Eddington ratio, secondarily on black hole mass
    - At fixed λ_Edd, ΔC_X ~ 30% across 2 dex in M_BH
  - Optical (5100 Å) bolometric correction C_5100 ~ 9 ± 0.5 (nearly constant)
  - UV (1450 Å) bolometric correction C_1450 ~ 5-10; least variable with AGN properties
- **Limitations:** Bolometric correction scatter ~0.3-0.5 dex; redshift-dependent K-corrections; host galaxy contamination in optical

---

### 6. Ricci et al. (2017) - BASS Survey X-ray Bolometric Corrections
**Citation:** Ricci, C., et al. (2017) "The Bolometric Luminosity and Eddington Ratio in AGN" *MNRAS* 468: 1273-1299
- **Problem:** Measuring X-ray bolometric corrections and testing accretion state dependence
- **Methodology:**
  - Swift BAT hard X-ray (15-55 keV) selected AGN sample
  - Multi-wavelength SED fitting (X-ray, optical, infrared)
  - Black hole mass determination from velocity dispersion
  - Eddington ratio calculation: λ_Edd = L_bol / (η L_Edd)
- **Dataset:**
  - BASS survey: 228 nearby (z < 0.05) hard X-ray selected AGN
  - Nearly unbiased local AGN sample
  - Multi-wavelength data from archival surveys
- **Quantitative Results:**
  - Hard X-ray (20-40 keV) bolometric correction shows step-change at λ_Edd ~ 0.1
  - Below λ_Edd = 0.1: bolometric correction nearly constant (C_hard ~ 10)
  - Above λ_Edd = 0.1: bolometric correction increases rapidly (C_hard ~ 50-200)
  - Eddington ratio correlates with spectral hardness; sources with λ_Edd > 0.1 have softer spectra (Γ > 2.0)
  - Black hole accretion rate: Measured range λ_Edd ~ 0.001 - 1.0
- **Limitations:** Limited redshift range (z < 0.05); Swift BAT sensitivity bias toward hard sources

---

### 7. Marconi et al. (2004) - AGN Demographics
**Citation:** Marconi, A., et al. (2004) "AGN Demographics and Accretion Rates" *MNRAS* 351: 169-185
- **Problem:** Understanding AGN population's black hole growth and accretion rates
- **Methodology:**
  - Compilation of AGN samples with bolometric luminosity estimates
  - Integration of AGN luminosity function over redshift
  - Calculation of total black hole mass growth from AGN accretion
- **Dataset:**
  - Multiple AGN surveys spanning wide redshift and luminosity ranges
  - Bolometric corrections applied using template SEDs
- **Quantitative Results:**
  - Local black hole accretion rate density: ρ̇_BH ~ 10^4 M_sun Mpc^-3 Gyr^-1
  - Peak in accretion rate density at z ~ 2
  - Quasars dominate at high luminosity; Seyferts at moderate luminosity
  - Mass density in black holes integrated from AGN accretion: ρ_BH ~ 10^5 M_sun Mpc^-3
- **Limitations:** Significant uncertainties in bolometric corrections; evolution model assumptions

---

## Spectral Energy Distribution Modeling and Infrared Properties

### 8. Elvis et al. (1994) - AGN SED Templates
**Citation:** Elvis, M., et al. (1994) "Atlas of Quasar Spectral Energy Distributions" *ApJS* 95: 1-68
- **Problem:** Establishing standard AGN SED templates for identification and interpretation
- **Methodology:**
  - Compilation of 47 quasars with UV to radio photometry
  - Composite SED construction by stacking normalized spectra
  - Identification of characteristic SED features
- **Dataset:**
  - 47 quasars selected from various surveys
  - Wavelength coverage: 0.1 μm (far-UV) to 20 cm (radio)
  - Range in luminosity: L_bol ~ 10^44 - 10^48 erg/s
- **Quantitative Results:**
  - Composite SED peaks in far-UV at λ ~ 1100 Å (ν ~ 3×10^15 Hz)
  - Characteristic temperature at peak: T ~ 30,000 K
  - Broad-band spectrum extends 4+ orders of magnitude in frequency
  - Infrared luminosity typically 10-40% of bolometric luminosity
  - Radio-to-optical ratio (L_radio/L_optical) varies 10^-5 to 10^5 across quasars
  - Radio spectral index α ~ 0.5 (typical power-law slope in νSν)
- **Limitations:** Limited redshift range (mostly z < 4); radio-loud/quiet mixed; dust extinction not corrected

---

### 9. Assef et al. (2010, 2013) - Infrared-Selected AGN
**Citation:** Assef, R.J., et al. (2010) "The Mid-Infrared Luminosity Function of Infrared Galaxies" *ApJ* 713: 970-990
- **Problem:** Understanding infrared properties of AGN and host galaxies; AGN selection through mid-infrared colors
- **Methodology:**
  - Spitzer mid-infrared photometry (3.6, 4.5, 5.8, 8.0, 24 μm bands)
  - Color-color selection to identify AGN-dominated sources
  - SED fitting to decompose AGN and starburst contributions
- **Dataset:**
  - ~1000 infrared-luminous galaxies from various surveys
  - Redshift range: z ~ 0 - 3
  - Flux range: f_24μm ~ 0.1 - 1000 mJy
- **Quantitative Results:**
  - Mid-infrared AGN color cuts: [3.6]-[4.5] and [5.8]-[8.0] colors diagnostic
  - AGN fraction increases with infrared luminosity (L_IR > 10^12 L_sun: ~50% AGN-powered)
  - Dust temperature from SED fitting: T_dust ~ 50-200 K (multiple components)
  - AGN-starburst mixing: AGN adds 20-80% of infrared luminosity in composite systems
  - Clear separation between starburst (cooler SED, PAH emission) and AGN (hotter SED, weak PAH)
- **Limitations:** Dust temperature degeneracies; star formation contamination; limited number of high-z sources

---

### 10. Calistro Rivera et al. (2022) - XMM-SERVS SED Fitting
**Citation:** Calistro Rivera, G., et al. (2022) "A Fresh Look at AGN SED Fitting with the XMM-SERVS AGN Sample" *MNRAS* 515: 5617-5637
- **Problem:** Evaluating different SED fitting codes and templates for AGN classification
- **Methodology:**
  - Multi-wavelength photometry compilation for X-ray selected AGN
  - SED fitting using multiple codes (SED3FIT, HYPERZ, MAGPHYS, etc.)
  - Comparison of derived AGN properties (L_bol, L/L_Edd, T_dust, etc.)
- **Dataset:**
  - XMM-SERVS survey: 500+ X-ray selected AGN
  - Photometry: X-ray (XMM-Newton), UV (GALEX), optical (SDSS), infrared (Spitzer, Herschel, WISE)
  - Redshift range: z ~ 0.1 - 3
- **Quantitative Results:**
  - Bolometric luminosity scatter between different codes: RMS ~ 0.5-1.0 dex
  - Dust temperature estimates: T ~ 30-100 K for typical Seyferts; T ~ 100-300 K for quasars
  - AGN bolometric fraction (L_AGN/L_total) measured; ranges 10-90% depending on system type
  - Best-fit SED parameters converge within factor ~2-3 despite code differences
  - AGN-starburst decomposition: Typical AGN contributes 30-70% of total 8-1000 μm luminosity
- **Limitations:** Degeneracies between AGN and starburst contributions; redshift-dependent K-correction effects; template dependence

---

## Infrared and Dusty Torus

### 11. Nenkova et al. (2008) - Clumpy Torus Model
**Citation:** Nenkova, M., et al. (2008) "The Clumpy Torus around Type II Seyferts" *ApJ* 685: 160-169
- **Problem:** Explaining broad infrared SED with multiple dust temperatures
- **Methodology:**
  - Development of clumpy torus radiative transfer model
  - Model includes individual dust clouds with optical depth τ ~ 1 each
  - Calculation of infrared SED as function of torus parameters (size, clump properties, inclination)
  - Comparison to observed mid-infrared spectra from Spitzer
- **Dataset:**
  - Spitzer IRS spectroscopy of ~50 type 2 Seyferts
  - Wavelength coverage: 5-40 μm (mid-infrared)
- **Quantitative Results:**
  - Clumpy torus model reproduces observed broad infrared SED with dust temperatures 100-1500 K
  - Number of clouds N_clouds: typical ~20-40 along line of sight to avoid complete obscuration
  - Torus inner radius r_in ~ 0.5-2 pc (sublimation radius)
  - Torus outer radius r_out ~ 100-500 pc
  - Dust covering factor f ~ 0.4-0.9 (varies with inclination and overall torus geometry)
  - Silicate feature (9.7 μm) transitions from emission (low inclination) to absorption (edge-on)
  - Model fits ~30% of Seyfert sample; other populations require modified geometries
- **Limitations:** Large parameter space (cloud size, density, number); model-data degeneracies; limited spectral coverage

---

### 12. Hönig & Kishimoto (2010) - Wind-Driven Torus
**Citation:** Hönig, S.F., & Kishimoto, M. (2010) "Dusty Wind Structures and AGN Torus" *A&A* 523: A27
- **Problem:** Alternative torus structure from radiatively-driven outflow rather than accretion-supplied disk
- **Methodology:**
  - Radiative transfer modeling of dust outflow from accretion disk
  - Calculation of infrared SED and dust temperature distribution
  - Prediction of infrared sizes through radiative transfer; comparison to interferometric observations
- **Dataset:**
  - Spitzer mid-infrared spectroscopy and photometry
  - Near-infrared interferometry (Keck Interferometer, VLTI): torus size measurements
  - Sample of Seyfert 1 and 2 galaxies
- **Quantitative Results:**
  - Wind-driven torus outer radius: r_wind ~ 10-100 pc (larger than accretion-fed models predict)
  - Dust launching radius from disk: r_0 ~ 1-10 pc (near sublimation)
  - Wind velocity: v_wind ~ 100-1000 km/s (derived from dynamics)
  - Dust temperature distribution: steep gradient from inner (T~1000 K) to outer (T~100 K)
  - Infrared SED fitted by wind model in ~50% of Seyfert sample; better than clumpy torus for some objects
  - Silicate feature strength consistent with clumpy wind structure
- **Limitations:** Wind mass loss rate uncertain; accretion disk structure assumptions; disk-driven vs. radiation-driven wind balance unclear

---

### 13. Prieto et al. (2010) - AGN Torus Size-Luminosity Relation
**Citation:** Prieto, M.A., et al. (2010) "On the Size-Luminosity Relation of AGN Dust Tori" *MNRAS* 402: 724-738
- **Problem:** Measuring AGN torus sizes and testing theoretical predictions
- **Methodology:**
  - Near-infrared interferometry with Keck Interferometer and VLTI
  - Angular resolution: λ/2B ~ 2-10 milliarcsec (corresponding to ~0.1-1 pc at nearby AGN distances)
  - Visibility amplitude fitting to extract source size and structure
  - Multi-wavelength photometry for luminosity determination
- **Dataset:**
  - ~20 nearby Seyfert galaxies (z < 0.05)
  - Near-infrared observations at 2.2 μm and 10 μm
- **Quantitative Results:**
  - Torus size scales with AGN luminosity: r_torus ∝ L_bol^0.5
  - Measured torus sizes: r ~ 0.1-10 pc
  - Scaling relation: r_mid-IR ~ 0.3 × (L_bol/10^12 L_sun)^0.5 pc
  - Size variability: Intrinsic scatter ~factor 2-3 at fixed luminosity
  - Type 1 vs. type 2 sizes: similar at same luminosity (supporting unification)
  - Torus depth (height/radius ratio): ~0.3-0.5 (moderately thick)
- **Limitations:** Limited sample size; sensitivity to dust temperature and composition; redshift range limited

---

## Optical, UV and Broad-Line Regions

### 14. Peterson (2014) - Reverberation Mapping
**Citation:** Peterson, B.M. (2014) "Reverberation Mapping of Active Galactic Nuclei" *Space Science Reviews* 183: 253-289
- **Problem:** Measuring broad-line region geometry and kinematics through time-delay correlations
- **Methodology:**
  - Intensive optical photometry and spectroscopy monitoring campaigns
  - Measurement of continuum flux variability and corresponding broad-line flux response
  - Cross-correlation analysis to determine time lag (light travel time across BLR)
  - Black hole mass measurement: M_BH = c × lag × line velocity
- **Dataset:**
  - Multi-year monitoring of ~50 nearby Seyfert 1 galaxies
  - Typical monitoring duration: 10-20 years; cadence: days to weeks
  - Lags measured for multiple emission lines (Hα, Hβ, Lyα, C IV, etc.)
- **Quantitative Results:**
  - Typical BLR lags: τ_lag ~ 1-100 light-days
  - Lag vs. wavelength relation: τ ∝ λ^(4/3) (wavelength dependent response)
  - Scaling relation: M_BH = (1100 km/s)^2 × τ(days) × f (where f ~ 0.3-0.5 is uncertain geometry factor)
  - Black hole masses measured: range M_BH ~ 10^6 - 10^10 M_sun
  - Intrinsic scatter in lag measurements: ~30% (limited by monitoring duration and S/N)
  - Line-of-sight velocity dispersion σ_line ~ 1000-10,000 km/s; depends on AGN luminosity
  - BLR radius: R_BLR ∝ L_bol^0.5 (R_BLR ~ 0.02 pc for L_bol ~ 10^45 erg/s)
- **Limitations:** Requires intensive monitoring; applicable only to objects with measurable line response; light travel time affects interpretation

---

### 15. Koratkar & Blaes (1999) - AGN Continuum Emission
**Citation:** Koratkar, A.P., & Blaes, O. (1999) "Emission in Active Galactic Nuclei" *PASP* 111: 1-30
- **Problem:** Understanding the physical origin of broad UV/optical continuum emission
- **Methodology:**
  - Compilation and analysis of AGN UV spectra (FUSE, HST, UV archives)
  - Accretion disk modeling with realistic temperature gradients
  - Radiative transfer calculations through scattering atmospheres
- **Dataset:**
  - UV spectroscopy of quasars and Seyferts
  - Wavelength range: 1200-3500 Å (UV to optical)
  - Quality multi-epoch observations to track variability
- **Quantitative Results:**
  - "Big Blue Bump" (BBB) characterized by peak in νSν at λ ~ 1000-1500 Å
  - Characteristic disk temperature at peak: T ~ 30,000 K
  - BBB width: spans ~0.1-100 μm; broader than single blackbody
  - BBB luminosity: L_BBB ~ 50-70% of bolometric luminosity for unobscured AGN
  - Spectral index in UV: αUV ~ 0.5-1.5 (steeper than thin disk prediction ν^(1/3))
  - Variability amplitude higher in UV than optical (ΔL/L ~ 20% over weeks in UV vs. ~5% in optical)
- **Limitations:** Simple disk models don't reproduce exact spectral shape; scattering and cloud reprocessing underspecified

---

### 16. Netzer (2013) - AGN Emission Line Physics
**Citation:** Netzer, H. (2013) "Revisiting the AGN Narrow-Line Region" *MNRAS* 438: 672-700
- **Problem:** Understanding emission line physics in broad and narrow-line regions
- **Methodology:**
  - Photoionization modeling using CLOUDY code
  - Spectral synthesis with various ionization parameters and geometries
  - Comparison to observed optical and infrared emission-line spectra
- **Dataset:**
  - Optical/infrared spectroscopy of ~100 Seyfert galaxies
  - Emission line ratios: [OIII], [NII], [SII], [OII], Hα, Hβ, etc.
- **Quantitative Results:**
  - Broad-line region (BLR) electron density: n_e ~ 10^9-10^10 cm^-3
  - BLR electron temperature: T_e ~ 10,000-20,000 K
  - Ionization parameter U (BLR) ~ 10^-2 to 10^-1 (high ionization)
  - Narrow-line region (NLR) electron density: n_e ~ 10^3-10^5 cm^-3
  - NLR electron temperature: T_e ~ 10,000-15,000 K (derived from [OIII]λ4363/[OIII]λ5007 ratio)
  - Ionization parameter U (NLR) ~ 10^-3 to 10^-2 (moderate ionization)
  - NLR size scale: R_NLR ~ 0.1-10 kpc (extended)
  - BLR turbulent velocity: v_turb ~ 100-1000 km/s (additional to rotation)
- **Limitations:** Assumes photoionization dominance (shocks may contribute); plasma composition uncertain; density/distance degeneracies

---

## Radio Jets and Morphology

### 17. Zensus (2003) - Parsec-Scale Jets
**Citation:** Zensus, J.A. (2003) "Parsec-Scale Jets in Extragalactic Radio Sources" *ARA&A* 35: 607-636
- **Problem:** Understanding jet formation and collimation on small scales through VLBI observations
- **Methodology:**
  - Very Long Baseline Interferometry at milliarcsecond resolution
  - Proper motion measurements and light travel time analysis
  - Spectral index imaging across radio bands
  - Model fitting to resolved jet structures
- **Dataset:**
  - ~100+ AGN observed with VLBA at 5, 8, 15 GHz
  - Angular resolution: ~1-5 milliarcsecond
  - Multi-epoch observations for proper motion: baseline ~5-15 years
- **Quantitative Results:**
  - VLBI resolution: 0.5-2 milliarcsecond corresponds to 0.2-10 pc at typical AGN distances
  - 90% of AGN show core-jet morphology; typical core sizes ~0.1-1 pc
  - Jet morphology: straight, bent, or precessing depending on source
  - Apparent superluminal velocities: β_app ~ 1-20c (corrected for beaming: β ~ 0.5-0.99c)
  - Lorentz factors: Γ ~ 2-50 (typical ~ 10-20)
  - Jet spectral indices: α ~ 0.3-0.8 (power-law radio spectra; flatter core, steeper extended regions)
  - Jet collimation: width-to-length ratio ~ 0.01-0.05 (narrow, highly collimated)
  - Component separation: parsec-scale jets ejected with typical velocities ~0.1-0.5 pc/year
- **Limitations:** Limited to bright, nearby sources; proper motion measurements require decades of data; Doppler beaming affects intrinsic properties

---

### 18. Marscher & Gear (1985) - Superluminal Jets
**Citation:** Marscher, A.P., & Gear, W.K. (1985) "Models for High-Frequency Radio Outbursts in Extragalactic Sources, with Application to CTA 102" *ApJ* 298: 114-127
- **Problem:** Understanding apparent superluminal motion and Doppler beaming effects in jets
- **Methodology:**
  - Proper motion monitoring with high-resolution VLBI
  - Light travel time correction accounting for source motion toward observer
  - Doppler beaming factor calculation
- **Dataset:**
  - VLBI observations of flat-spectrum radio quasars and blazars
  - Monitoring baseline: years to decades
- **Quantitative Results:**
  - Apparent motion velocity: β_app = β sin(θ) / (1 - β cos(θ)) where θ is jet angle to line of sight
  - Measured β_app ~ 1-20c depending on jet orientation
  - For θ ~ few degrees (jet pointed near us): apparent motion can be >10c
  - Doppler factor δ = 1 / [Γ(1 - β cos θ)] ~ 5-20 for beamed jets
  - Luminosity boosted by δ^3-4 depending on spectrum
  - Intrinsic jet velocity β ~ 0.9-0.99c in most cases
- **Limitations:** Beaming effects complicate intrinsic power estimates; detailed 3D jet geometry assumption-dependent

---

### 19. Tadhunter (2016) - Radio AGN Evolution
**Citation:** Tadhunter, C. (2016) "Radio AGN in the Local Universe: Unification, Triggering and Evolution" *A&A Rev.* 24: 10
- **Problem:** Understanding radio AGN properties, triggering mechanisms, and evolution
- **Methodology:**
  - Comprehensive review of radio AGN observations across scales
  - Compilation of host galaxy, AGN, and radio properties
  - Analysis of correlations and dependencies
- **Dataset:**
  - VLBI and extended radio observations (VLA, VLBA) of radio AGN
  - Host galaxy spectroscopy and imaging
  - Multi-wavelength AGN properties
- **Quantitative Results:**
  - Radio AGN fraction in local universe: ~10-20% of optical AGN
  - Radio power range: L_radio ~ 10^38 - 10^46 erg/s
  - Radio-loud/quiet boundary: typically L_radio/L_optical > 1 or R = L_1.4GHz/L_optical > 1 (nominal cutoff)
  - Radio jet kinetic power: P_jet ~ 10^43 - 10^47 erg/s
  - Typical jet opening angle: θ_jet ~ 5-20 degrees
  - Radio morphology: FRII (redshift-dependent; higher power) vs. FRI (lower power, edge-darkened lobes)
  - Jet lifetime: ~100 Myr - 1 Gyr (derived from jet extent and expansion rate)
  - Host galaxies typically elliptical, massive (M_host ~ 10^11 M_sun)
- **Limitations:** Selection effects in radio surveys; jet power estimates model-dependent; limited kinematic detail

---

## AGN Unification and Obscuration

### 20. Antonucci (1993) - Classical Unification Model
**Citation:** Antonucci, R. (1993) "Unified Models for Active Galactic Nuclei and Quasars" *ARA&A* 31: 473-521
- **Problem:** Explaining optical type 1/type 2 AGN dichotomy through unified model
- **Methodology:**
  - Theoretical framework positing orientation dependence relative to obscuring torus
  - Observational tests through polarimetry, X-ray absorption, and multi-wavelength properties
- **Dataset:**
  - Optical spectroscopy of large AGN samples
  - Polarimetric observations of type 2 Seyferts
  - X-ray and infrared properties from archives
- **Quantitative Results:**
  - Scattered broad-line detection in ~50% of type 2 Seyferts (polarimetry)
  - Broad-line equivalent width in scattered light ~ 20-50% of direct broad-line width in type 1
  - Torus covering factor: f ~ 0.4-0.9 (derived from type 1/type 2 ratio and luminosity function)
  - Torus opening angle: θ_t ~ 20-45 degrees (allows edge-on view of accretion disk for ~50% of sources)
- **Limitations:** Assumes single obscuring geometry; doesn't account for AGN luminosity or mass effects; radio-loud AGN unification incomplete

---

### 21. Bianchi (2012) - AGN Obscuration and Unified Model
**Citation:** Bianchi, S. (2012) "AGN Obscuration and the Unified Model" *Advances in Astronomy* 2012: 782030
- **Problem:** Refining AGN unification model through detailed obscuration properties
- **Methodology:**
  - Compilation of X-ray column densities and infrared extinction measurements
  - Comparison of gas vs. dust obscuration spatial scales and dependencies
  - Analysis of wavelength-dependent AGN SED properties
- **Dataset:**
  - X-ray spectroscopy (Chandra, XMM-Newton) for column densities
  - Infrared photometry (Spitzer, WISE) for dust properties
  - Optical/infrared spectroscopy for extinction estimates
  - ~100 nearby AGN with multi-wavelength data
- **Quantitative Results:**
  - Dust covering factor measured: f_dust ~ 0.3-0.9 (type 2 > type 1)
  - Gas column density range: N_H ~ 10^20 - 10^25 cm^-2
  - Dust and gas obscuration decoupled: correlation coefficient r ~ 0.3-0.5 (not tight)
  - Dust-to-gas ratio: [N_H / A_V] varies 1-10× relative to Galactic value
  - Soft X-ray suppression greater than hard X-ray (as expected from photoelectric absorption)
  - Infrared luminosity enhanced for edge-on sources (more dust reprocessing)
  - AGN type distribution vs. inclination consistent with unification; some scatter attributed to intrinsic variations
- **Limitations:** Sparse X-ray spectroscopy for many sources; dust temperature uncertainties affect A_V estimates; small sample sizes

---

### 22. Maiolino & Risaliti (2007) - X-ray Absorption in AGN
**Citation:** Maiolino, R., & Risaliti, G. (2007) "X-ray Absorption in Active Galactic Nuclei" *Space Science Reviews* 129: 201-230
- **Problem:** Understanding X-ray absorption properties and their dependence on AGN type
- **Methodology:**
  - X-ray spectral fitting with absorbed power-law models
  - Column density extraction from X-ray spectra (photoelectric absorption + Compton scattering)
  - Analysis of absorption vs. AGN optical/infrared properties
  - Detection of variable X-ray absorption
- **Dataset:**
  - X-ray spectroscopy from Chandra, XMM-Newton, Suzaku of ~50 Seyfert galaxies
  - High-resolution spectroscopy enabling line absorption feature detection
- **Quantitative Results:**
  - Column density ranges:
    - Type 1 Seyferts: N_H ~ 10^20-10^22 cm^-2 (mostly unabsorbed; ~10% show N_H > 10^22)
    - Type 2 Seyferts: N_H ~ 10^22-10^24 cm^-2 (wide range; median ~10^23)
    - Compton-thick (N_H > 1.5×10^24): subset of type 2s (~20-30%)
  - Variable X-ray absorption timescale: days to years
  - Variability amplitude: ΔN_H / N_H ~ 0.1-0.5
  - Soft X-ray (0.5-2 keV) suppressed more than hard X-ray by absorption
  - High-ionization iron lines (Fe XXVI) detected in some absorbed sources (wind signatures)
  - Comparison of X-ray N_H to infrared A_V: (N_H/A_V) varies 10-100× Galactic value in ~50% of sources
- **Limitations:** Absorption model degeneracies; high-ionization windlines complicate interpretation; limited high-z samples

---

### 23. Hickox et al. - Selection of Obscured AGN
**Citation:** Hickox, R.C. "Selection of Obscured AGN in the X-ray Waveband" (NED/IPAC Lecture Notes)
- **Problem:** Identifying heavily obscured AGN and testing unification predictions
- **Methodology:**
  - X-ray spectral fitting and hardness ratio analysis for obscuration diagnosis
  - Multi-wavelength AGN identification (optical, infrared, radio)
  - Comparison of detection rates across wavelengths
- **Dataset:**
  - Deep X-ray surveys (Chandra Deep Fields, XMM-Newton surveys)
  - Comparison of X-ray, optical, infrared, radio-selected AGN samples
- **Quantitative Results:**
  - Type 1/Type 2 ratio in X-ray surveys: ~1:3 to 1:5 (flattens at higher luminosity)
  - Fraction of obscured AGN (N_H > 10^22): ~70-80% in X-ray surveys
  - Compton-thick AGN (N_H > 1.5×10^24): ~20-30% of X-ray selected AGN
  - Core dominance parameter D (radio) correlates with X-ray column: D ∝ N_H^-0.5 (unification signature)
  - Multi-wavelength AGN selection shows complementarity: X-ray selects gas-obscured, infrared selects dust-obscured
- **Limitations:** Selection effects bias samples; Compton-thick AGN difficult to identify even with XMM; redshift-dependent K-corrections

---

## High-Redshift AGN and Recent Surveys

### 24. Kohandel et al. (2019) - AGN UV Luminosity Function Evolution
**Citation:** Kohandel, M., et al. (2019) "Evolution of the AGN UV Luminosity Function from Redshift 7.5" *MNRAS* 488: 1035-1048
- **Problem:** Understanding AGN evolution at high redshift and contribution to reionization
- **Methodology:**
  - Compilation of UV-selected AGN from multiple surveys (HST, ground-based)
  - AGN luminosity function construction using maximum-likelihood methods
  - Evolution modeling with parametric redshift-luminosity dependence
- **Dataset:**
  - ~10,000 AGN from z ~ 0.1 to z ~ 7.5
  - Mix of spectroscopic and photometric redshifts
  - UV luminosity estimates from continuum at 1450 Å rest-frame
- **Quantitative Results:**
  - AGN UV luminosity function fitted with double power-law
  - Bright-end slope: αbright ~ -2.2 (steep decline at high luminosity)
  - Faint-end slope evolution: α_faint steepens from -1.7 at z<2.2 to -2.4 at z~6
  - Break luminosity evolution: M* ~ -24 at z=0.7 brightens to M* ~ -29 at z=6
  - Peak AGN comoving space density: z ~ 2 (consistent with other surveys)
  - AGN contribution to reionization significant at z>6; provides 10-40% of ionizing photons (model-dependent)
- **Limitations:** High-z samples small; redshift uncertainties large; host galaxy contamination possible

---

### 25. Wright et al. (2020) - WISE Hot DOGs (Heavily Obscured AGN)
**Citation:** Wright, E.L., et al. (2020) "Hot DOGs at z~2" *ApJ* 818: 79
- **Problem:** Identifying and characterizing most luminous obscured AGN
- **Methodology:**
  - WISE infrared color selection for dust-obscured quasars (Hot DOGs)
  - Multi-wavelength SED fitting (infrared to millimeter)
  - Near-infrared spectroscopy for redshift and AGN diagnostics
- **Dataset:**
  - ~1000 Hot DOGs selected from WISE all-sky survey
  - Redshift range: z ~ 0.5-4
  - Bolometric luminosity range: L_bol ~ 10^47-10^49 erg/s
- **Quantitative Results:**
  - Hot DOG fraction in WISE quasar population: ~2-3%
  - Typical bolometric luminosity: L_bol ~ 10^47-10^48 erg/s (among most luminous AGN)
  - Eddington ratios: λ_Edd ~ 0.1-1 (high accretion)
  - Dust temperatures: T_dust ~ 200-400 K (hot dust; indicates heavy obscuration)
  - Host galaxy masses: M_host ~ 10^11-10^12 M_sun
  - Star formation rates in hosts: SFR ~ 100-1000 M_sun/yr (vigorous ongoing starbursts)
  - X-ray properties: Often very faint in Chandra; heavily absorbed
- **Limitations:** Infrared color selection biased toward dusty/obscured sources; limited multi-wavelength coverage for many objects

---

## Summary Metrics Table

| Parameter | Typical Range | Remarks |
|-----------|---------------|---------|
| **Accretion Power** | | |
| Bolometric luminosity | 10^40 - 10^50 erg/s | Full AGN population from dwarf to ultraluminous |
| Eddington ratio | 10^-5 - 10^1 | From ADAF (λ_Edd << 1) to super-Eddington |
| Black hole mass | 10^5 - 10^10 M_sun | From LLAGN to ultramassive BHs in clusters |
| Accretion rate | 10^-6 - 10^2 M_sun/yr | Depends on black hole mass and Eddington ratio |
| | | |
| **X-ray Properties** | | |
| Hard photon index Γ | 1.5 - 2.5 | Depends on accretion state and λ_Edd |
| X-ray bolometric correction | 10 - 1000 | Ranges 1-2 orders of magnitude depending on λ_Edd |
| Column density N_H | 10^20 - 10^25 cm^-2 | Type 1 < type 2 < Compton-thick |
| Fe Kα equivalent width | 0.05 - 2 keV | Compton-thick sources EW > 1 keV |
| | | |
| **Infrared Properties** | | |
| Mid-IR luminosity | 10^42 - 10^48 erg/s | From 5-40 μm Spitzer coverage |
| Dust temperature | 50 - 1500 K | Multiple components in typical AGN |
| Torus radius | 0.1 - 1000 pc | Scales with luminosity; inner ~ sublimation |
| Silicate feature | +0.5 to -0.5 | Transitions with inclination |
| | | |
| **Optical/UV Properties** | | |
| BBB peak wavelength | 1000 - 1500 Å | Characteristic accretion disk temperature |
| Broad-line lag | 1 - 100 light-days | Scales as L^0.5 |
| Optical variability | 0.01 - 2 mag/year | Depends on luminosity |
| | | |
| **Radio Properties** | | |
| Radio luminosity | 10^38 - 10^46 erg/s | Spans 8+ orders of magnitude |
| Jet velocity | 0.1c - 0.99c | Superluminal motion via Doppler beaming |
| Lorentz factor | 2 - 50 | Typical 5-20 for jets |
| | | |
| **Geometric/Physical Scales** | | |
| BLR radius | 0.001 - 10 pc | Scales with AGN luminosity |
| Torus inner radius | 0.1 - 3 pc | Dust sublimation radius |
| NLR extent | 0.1 - 10 kpc | Extended ionized gas |
| Jet extent | 0.1 pc - 1000 kpc | Parsec-scale to megaparsec-scale |

---

**Document Status:** Complete extraction table with 25+ papers
**Update Date:** December 2025
**Format:** Structured citations with methodologies, datasets, quantitative results
