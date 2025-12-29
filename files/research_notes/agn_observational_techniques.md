# AGN Multi-Wavelength Observation Techniques and Instruments

**Purpose:** Reference guide for observational methods, instruments, and sensitivity limits for AGN studies
**Scope:** X-ray, optical, infrared, radio, and gamma-ray techniques

---

## I. X-RAY OBSERVATIONS

### Current Missions (Active)

#### XMM-Newton
**Capabilities:**
- Energy range: 0.15-12 keV (soft to hard X-ray)
- Effective area: ~1500 cm² at 1.5 keV (peak); ~500 cm² at 10 keV
- Field of view: 30 arcmin diameter circular
- Spectral resolution: ΔE/E ~ 0.05 at 1 keV (EPIC pn); ~0.07 (EPIC MOS)
- Timing resolution: 70 ms readout; 2.6 s for full frame

**AGN Studies:**
- Spectroscopic analysis of X-ray absorption/reflection
- Broad-band continuum fitting (soft to hard X-ray)
- Time-variability studies (seconds to years)
- Extended source mapping (outflows, jets)

**Typical AGN Observations:**
- Bright AGN: 10-100 ks observations reach signal-to-noise ~10-100
- Fainter AGN (L_X ~ 10^42 erg/s): require 100-500 ks exposures
- Spectral parameter uncertainties: ±10-20% for well-exposed sources

#### Chandra X-ray Observatory
**Capabilities:**
- Energy range: 0.3-10 keV (soft to moderately hard X-ray)
- Effective area: ~600 cm² at 1 keV; ~300 cm² at 5 keV
- Spatial resolution: 0.5 arcsec (HPD) at 1.5 keV; exceptional for AGN nuclei
- Field of view: 17 arcmin square (ACIS-S); 16.9 arcmin square (ACIS-I)
- Spectral resolution: ΔE/E ~ 0.15 at 1 keV (best with grating instruments)

**AGN Studies:**
- Highest spatial resolution X-ray imaging of AGN nuclei and jets
- Detailed spectroscopy of hot gas outflows and kiloparsec-scale emission
- Weak AGN detection in crowded fields (galaxies, clusters)
- Multi-year monitoring of nearby AGN (variability, flux changes)

**Typical AGN Observations:**
- Bright Seyferts: 10-50 ks sufficient for spectral analysis
- Faint/distant AGN: 50-200 ks to reach L_X ~ 10^42-10^43 erg/s
- Positional accuracy: ~1 arcsec absolute astrometry

#### NuSTAR (Nuclear Spectroscopic Telescope Array)
**Capabilities:**
- Energy range: 3-79 keV (hard X-ray; unique mission capability)
- Effective area: ~1000 cm² at 10 keV; ~100 cm² at 50 keV
- Spatial resolution: 18 arcsec HPD (modest but sufficient for point sources)
- Field of view: 12 arcmin circular
- Spectral resolution: ΔE/E ~ 0.04 at 10 keV
- Timing resolution: 2.5 μs

**AGN Studies:**
- Primary mission for hard X-ray spectroscopy
- Compton-thick AGN identification (reflection dominated >10 keV)
- High-energy cutoff measurements in X-ray spectra
- Faint source detection above 10 keV (other missions less sensitive)

**Typical AGN Observations:**
- Bright AGN: 10-50 ks reach high signal-to-noise (>100) in 10-50 keV
- Faint AGN: 100-300 ks for spectral fitting above 20 keV
- Hard X-ray bolometric correction determination (most direct method)

#### Suzaku (ASTRO-H)
**Heritage Capabilities** (Suzaku; mission ended 2015; archived data still analyzed):
- Energy range: 0.2-600 keV (broad X-ray and soft gamma-ray)
- Spectral resolution: ΔE/E ~ 0.05 at 5 keV (HXD; world-class for iron line studies)
- Hard X-ray imaging possible with simultaneous XIS soft X-ray data

**AGN Studies:**
- Broad iron K-alpha line profile measurements for black hole spin
- Hard X-ray variability timescale studies
- Combined soft+hard X-ray spectral fitting

#### eROSITA (Spektrum-Roentgen-Gamma; SRG)
**Capabilities:**
- Energy range: 0.2-10 keV
- Field of view: 1°diameter (exceptionally large; ~30× Chandra ACIS)
- Effective area: ~2000 cm² at 1 keV (competitive with XMM at soft X-rays)
- All-sky survey capability

**AGN Studies:**
- All-sky AGN census; detection of faint, extended sources
- AGN in groups and clusters
- Variability monitoring of thousands of AGN simultaneously
- Soft X-ray variability timescale studies (first-ever systematic sample)

**Survey Depth:** All-sky survey reaches L_X ~ 10^44 erg/s; deeper targeted fields reach ~10^42 erg/s

### Planned/Future X-ray Missions

#### Athena (Advanced Telescope for High-resolution Astrophysics; ESA; launch ~2037)
**Planned Capabilities:**
- Energy range: 0.2-12 keV
- Effective area: ~12,000 cm² at 1 keV (9× larger than XMM-Newton)
- Spectral resolution: ΔE/E ~ 0.002 at 5 keV (1000× improvement over current missions; will resolve atomic transitions)
- High-resolution imaging spectrometer (XIFU) + wide-field imager
- Field of view: 40 arcmin (XIFU), 40 arcmin (WFI)

**Revolutionary AGN Studies Enabled:**
- Detailed ionized outflow spectroscopy (resolve individual lines, kinematics)
- Precise black hole spin measurements from iron line (factor ~3 improvement)
- Weak AGN detection in z>3-4 galaxies
- High-resolution X-ray spectroscopy of AGN accretion disks

---

## II. OPTICAL SPECTROSCOPY

### Ground-Based Spectrographs

#### 8-10m Telescopes (VLT, Keck, Gemini, Subaru)
**Capabilities:**
- Wavelength range: 0.3-10 μm (UV-optical-infrared with various spectrographs)
- Spectral resolution: R = λ/Δλ from 100 (low-res imaging spectroscopy) to ~10,000 (high-res echelle)
- Signal-to-noise: Can achieve S/N ~ 100-1000 per pixel for bright objects in hours
- Spatial resolution: Diffraction-limited with adaptive optics (~0.05 arcsec)

**AGN-Relevant Instruments:**
- VIMOS/VLT: Multi-object spectroscopy; survey capability
- MOSFIRE/Keck: Near-infrared spectroscopy for redshifted optical lines
- FMOS/Subaru: Fiber-fed multi-object NIR spectroscopy

**AGN Studies:**
- Optical AGN classification (BPT diagrams) using [OIII], [NII], Hα, Hβ
- Broad-line kinematics (FWHM, equivalent width, line profile fitting)
- Emission-line variability (months to years monitoring)
- Host galaxy spectral synthesis (stellar mass, age, star formation)

#### SDSS (Sloan Digital Sky Survey) and Legacy Surveys
**Capabilities:**
- Wavelength coverage: 3800-9200 Å (optical)
- Spectral resolution: R ~ 1800 (sufficient for emission-line diagnostics)
- Sample size: ~120,000 spectroscopic AGN in SDSS main survey
- Depth: g < 19.1 mag for spectroscopy

**AGN Catalog Properties:**
- Automated AGN classification based on spectral features
- Broad-line and narrow-line AGN photometrically and spectroscopically selected
- Black hole mass estimates from broad-line widths (virial method)
- Eddington ratio calculations possible

**Quantitative Benchmarks:**
- Broad-line detection completeness: ~95% for L_bol > 10^44 erg/s
- AGN classification accuracy: ~95% using emission-line ratios
- Black hole mass uncertainties: ~0.3-0.5 dex (factor 2-3)

### Space-Based Optical/UV Spectroscopy

#### HST (Hubble Space Telescope)
**Capabilities:**
- Wavelength range: 0.1-1 μm (UV-optical-NIR)
- Spectral resolution: R from ~500 (grism; wide-field) to ~17,000 (echelle; STIS)
- Spatial resolution: Diffraction-limited ~0.05 arcsec (optical)

**Key Instruments:**
- STIS (Space Telescope Imaging Spectrograph): High-res echelle; sensitive UV
- COS (Cosmic Origins Spectrograph): UV spectroscopy; low/moderate resolution
- WFC3 (grism): Low-res spectroscopy of faint objects; survey capability

**AGN Studies:**
- Ultraviolet continuum characterization (UV peak, "big blue bump")
- High-ionization lines (C IV, N V, Si IV) in broad-line regions
- Stellar population analysis in AGN hosts (age, mass, SFR)
- Reverberation mapping for black hole masses

**Typical Program:** 10-50 orbits (10-50 ks) for spectroscopy of typical Seyfert; 100+ orbits for faint/distant AGN

---

## III. INFRARED OBSERVATIONS

### Mid-Infrared Spectroscopy and Photometry

#### Spitzer Space Telescope (Deactivated 2020; archived data)
**Legacy Capabilities:**
- Wavelength range: 3.6-160 μm (near to far-infrared)
- Spectral resolution: R ~ 50-100 (IRS spectroscopy; low resolution)
- Photometric bands: 3.6, 4.5, 5.8, 8.0, 24, 70, 160 μm

**Key Instruments:**
- IRAC (Infrared Array Camera): Photometry at 3.6, 4.5, 5.8, 8.0 μm; point-spread function ~2 arcsec
- IRS (Infrared Spectrograph): Spectroscopy 5.2-40 μm; silicate feature characterization
- MIPS (Multiband Imaging Photometer): Photometry 24-160 μm; far-infrared AGN/starburst contributions

**AGN Studies:**
- Mid-infrared AGN selection using colors (e.g., WISE colors)
- Silicate feature (9.7 μm) characterization for inclination and dust properties
- Torus temperature distribution from SED fitting (5-40 μm)
- AGN/starburst decomposition in infrared-luminous galaxies

**Quantitative Benchmarks:**
- Sensitivity: Point source detection ~10 μJy (IRAC), ~0.1 mJy (IRS), ~10 mJy (MIPS 24 μm)
- Silicate feature measurement accuracy: ~10% for mid-strength features
- Dust temperature uncertainty: ±50 K typical (from multi-band photometry)

#### Herschel Space Observatory (Deactivated 2013; archival analysis ongoing)
**Legacy Capabilities:**
- Wavelength range: 70-500 μm (far-infrared; unique capability)
- Spectral resolution: R ~ 40 (FTS; spectroscopy); photometry in 70, 100, 160, 250, 350, 500 μm bands
- Beam size: 7 arcsec (70 μm) to 37 arcsec (500 μm)

**Key Instruments:**
- PACS (Photodetector Array Camera and Spectrometer): 70-160 μm
- SPIRE (Spectral and Photometric Imaging Receiver): 250-500 μm

**AGN Studies:**
- Far-infrared AGN luminosity (total dust reprocessing; not affected by obscuration)
- Far-infrared SED peak measurement (cool dust temperature from 100-500 μm data)
- Star formation rate in AGN hosts (estimated from far-IR total luminosity)
- AGN bolometric luminosity (integrating complete SED from X-ray to far-IR)

**Quantitative Benchmarks:**
- Far-IR point source sensitivity: ~50 mJy (PACS), ~100 mJy (SPIRE)
- Temperature determination from PACS/SPIRE colors: ±10 K (for dust T ~ 30-50 K)
- Far-infrared SED fitting uncertainty: ~20-30% (limited by foreground/background subtraction)

### Contemporary and Future Infrared Missions

#### JWST (James Webb Space Telescope; launched 2021)
**Revolutionary Capabilities:**
- Wavelength range: 0.6-28 μm (optical-infrared; NIR/MIR emphasis)
- Spectral resolution: R from 100 (MIRI low-res) to 2700 (NIRSpec high-res); MIRI spectroscopy 5-28 μm unprecedented
- Spatial resolution: Diffraction-limited ~0.1 arcsec (NIR)
- Sensitivity: Dramatically improved (10-100× better than predecessors at near-infrared)

**Key Instruments:**
- NIRCam: 0.6-5.0 μm imaging; 4 arcsec^2 field
- NIRSpec: 0.6-5.3 μm spectroscopy; integral field unit mode
- MIRI: 5-28 μm imaging + spectroscopy; exquisite sensitivity

**Revolutionary AGN Studies (first results arriving 2023-2025):**
- Heavily obscured (Compton-thick) AGN at z>3 detectable via infrared
- Discovery of "little red dots" (LRDs): abundant heavily-obscured AGN population at z>6
- Detailed AGN SED fitting for z>3-4 objects (previously limited to bolometric luminosity estimates)
- Mid-infrared emission-line diagnostics (high-ionization lines [Ne V], [Ar III], etc.)
- AGN host galaxy properties for z>3 (stellar mass, star formation; new regime)
- Torus characterization with unprecedented detail (PAH features, silicates, continuum)

**Quantitative Benchmarks:**
- Mid-infrared point source sensitivity: ~0.1 μJy (achieving 5σ in 10,000 s MIRI imaging)
- Spectroscopic sensitivity: Line flux detection limit ~10^-18 W/m²
- Impact: Rewriting AGN evolution picture at z>6; hidden black hole growth more prolific than expected

#### ALMA (Atacama Large Millimeter/submillimeter Array)
**Capabilities:**
- Wavelength range: 0.3-3.6 mm (millimeter/submillimeter)
- Spectral resolution: R up to ~10^6 (excellent for molecular lines)
- Angular resolution: 0.3-20 arcsec depending on frequency and configuration
- Field of view: 20-60 arcsec

**Key Capabilities:**
- CO and other molecular gas observations (tracing cold gas and star formation)
- Far-infrared continuum at 1.3 mm, 870 μm (dust continuum; unaffected by heavy obscuration)
- High-redshift galaxy continuum detection (ALMA sensitive to high-z objects)

**AGN Studies:**
- AGN-starburst decomposition (far-IR continuum; molecular gas)
- Black hole accretion rate from CO observations (indirect via gas dynamics)
- Outflow kinematics (CO outflows mapped to 10+ kpc)
- High-z AGN bolometric luminosity (accessible via submm continuum)

**Quantitative Benchmarks:**
- Continuum sensitivity at 1.3 mm: ~0.1 mJy (in single 1 GHz channel, 1 hour integration)
- Spectral line sensitivity: flux limits ~1 mJy for narrow lines

---

## IV. RADIO OBSERVATIONS

### Very Long Baseline Interferometry (VLBI)

#### VLBA (Very Long Baseline Array)
**Capabilities:**
- 10 antennas spanning North America; baseline ~8600 km
- Frequency bands: 4-80 GHz (L, S, C, X, Ku, K, Q bands)
- Angular resolution: 0.3-5 milliarcsec depending on frequency (0.5-10 pc at typical AGN distances)
- Sensitivity: ~0.1-1 mJy for point sources (achieved in hours of observation)

**AGN Studies:**
- Parsec-scale jet imaging and structural analysis
- Proper motion measurements for superluminal motion (requires multi-epoch, years apart)
- Core dominance measurements (D = S_core / S_total)
- Jet collimation angle and morphology

**Observational Timescales:**
- Single-epoch imaging: 1-4 hours of VLBA time
- Multi-epoch monitoring: typically 4-8 epochs over 1-5 years
- Proper motion measurement uncertainty: ~10-50 μas depending on source brightness

#### EHT (Event Horizon Telescope)
**Revolutionary Emerging Capability:**
- Global milliarcsecond VLBI at 230 GHz and above
- Achieves angular resolution ~20-50 microarcseconds (approaching black hole shadow size)
- Sparse array (8-13 participating telescopes); signal is not traditional imaging

**Breakthrough AGN Results:**
- M87 black hole shadow imaging (2019; first direct black hole imaging)
- Sgr A* black hole shadow imaging (2022; in Milky Way center)
- Constraints on black hole mass, spin, and spacetime geometry

#### MERLIN, e-MERLIN (UK facility)
**Capabilities:**
- UK-based VLBI array; baseline ~200 km
- Frequency bands: 1.4-22 GHz
- Angular resolution: 10-100 milliarcsec

**AGN Studies:**
- Moderate-resolution jet imaging (complementary to VLBA)
- Accessible alternative to VLBA for some bright sources

### Centimeter-Wavelength Surveys

#### VLA (Very Large Array)
**Capabilities:**
- 27 antennas; multiple configurations (A, B, C, D; largest baseline ~36 km)
- Frequency bands: 1-50 GHz (L through Q bands)
- Angular resolution: 0.1-45 arcsec (depending on frequency and configuration)
- Field of view: 30-50 arcmin (frequency-dependent)

**AGN Studies:**
- Extended radio source morphology (kpc-scale jets and lobes)
- Core-jet flux measurements
- Radio spectral index determination (multi-frequency observations)
- Weak/faint AGN detection (mJy and μJy levels)

**Survey Capabilities:**
- NVSS (NRAO VLA Sky Survey): 1.4 GHz all-sky survey; ~1 mJy sensitivity; ~45 arcsec resolution
- FIRST (Faint Images of Radio Sky at Twenty-cm): 1.4 GHz; deeper (~1 mJy), higher resolution (~5 arcsec) in northern hemisphere
- VLA Stripe 82 survey: repeated observations for variability studies

**Quantitative Benchmarks:**
- Point source detection: ~10 μJy (C configuration, 1 hour)
- Extended source mapping: down to ~100 μJy/beam
- Radio spectral index measurement uncertainty: ±0.05-0.1 (from multi-frequency data)

### Millimeter/Submillimeter Observations

#### ALMA (See Infrared Section - dual capability)
**Millimeter-wave capability:**
- Observes AGN jets/outflows at mm wavelengths
- Continuum sensitivity to synchrotron emission and dust
- Molecular line observations (jet-cloud interactions)

#### MeerKAT (South Africa; recently operational)
**Capabilities:**
- 64 dishes providing strong sensitivity improvement over older facilities
- Frequency bands: 0.6-14.5 GHz
- Angular resolution: 0.7-45 arcsec (frequency-dependent)
- High sensitivity to faint radio sources

**AGN Studies:**
- Very faint radio AGN surveys (formerly undetectable)
- High-z AGN radio properties
- Radio-loud/radio-quiet AGN population statistics

#### SKA Precursors (ASKAP, LOFAR)
**Emerging Capabilities:**
- ASKAP (Australia): 36 dishes; wide field (30 deg²); transient variable AGN
- LOFAR (Europe): Low-frequency (10-240 MHz); unique sensitivity to diffuse emission

**AGN Science:**
- Time-domain radio AGN variability (transient jets?)
- Ultra-faint radio AGN census (contributing to CXB understanding)
- Low-frequency spectral indices for AGN populations

---

## V. GAMMA-RAY OBSERVATIONS

### High-Energy Gamma-Ray Missions

#### Fermi-LAT (Fermi Large Area Telescope)
**Capabilities:**
- Energy range: 20 MeV - 300 GeV (high-energy gamma-ray)
- Large field of view: ~2.4 sr (monitors entire gamma-ray sky daily)
- Angular resolution: 0.1-1 degree (energy-dependent; coarser at low energy)
- Point source detection sensitivity: ~3×10^-13 cm^-2 s^-1 (10 GeV; steady sources)

**AGN Detection Capability:**
- ~4000+ sources in 4FGL catalog; ~56% associated with AGN
- Dominated by blazars (jet-aligned AGN)
- Narrow-line Seyfert 1s (NLS1) surprisingly bright gamma-ray sources
- Misaligned AGN recently detected (challenging beaming paradigm)

**Quantitative Results:**
- Fermi-detected AGN typically L_gamma > 10^44 erg/s (high accretion, powerful jets)
- Gamma-ray variability timescale: days to months (rapid, requiring detailed monitoring)
- Spectral hardness index: Γ ~ 1.5-2.5 (varying with source state)

#### MAGIC and VERITAS (Ground-Based Cherenkov Telescopes)
**Capabilities:**
- Energy range: 30 GeV - 300 TeV (very high-energy gamma-ray)
- Angular resolution: 0.05-0.1 degree (excellent for point source identification)
- Sensitivity: ~0.1 mCrab (10-100 GeV equivalent sensitivity; mCrab = 10^-13 cm^-2 s^-1 cm Crab flux)

**AGN Detection Capability:**
- ~100+ AGN detected in TeV regime (dominated by blazars at close distances, z<0.3)
- Extended sources and jets resolved at TeV energies
- High-z objects rarely detected (absorption by cosmic microwave background at TeV)

**Recent Discoveries:**
- Misaligned AGN TeV detection (radio galaxies M87, Cen A, NGC 1275)
- Challenges standard beaming models

---

## VI. MULTI-WAVELENGTH SURVEY COMBINATIONS

### Synoptic Surveys Enabling AGN Multi-Wavelength Characterization

#### Panoramic Surveys

| Survey | Wavelength(s) | Coverage | Depth (5σ limit) | Key AGN Capability |
|--------|--------------|----------|------------------|-------------------|
| **SDSS** | Optical (3800-9200 Å) | 14,555 deg² | r ~ 19.1 mag | 120k spectroscopic AGN; black hole masses |
| **2MASS** | NIR (1.2-2.2 μm) | All-sky | K ~ 14 mag | Host galaxy stellar mass; dusty AGN colors |
| **WISE** | Mid-IR (3.4-22 μm) | All-sky | W1 ~ 17.1 mag | AGN/starburst selection via colors; hot dust |
| **XMM-NEWTON** | X-ray (0.15-12 keV) | Deep fields | L_X ~ 10^40-10^43 erg/s | X-ray selected AGN; spectroscopy |
| **Spitzer Legacy** | Mid-IR (3.6-160 μm) | Large surveys | S_24 ~ 0.1 mJy | Infrared-selected AGN; SED fitting |
| **Herschel** | Far-IR (70-500 μm) | COSMOS, etc. | S_100 ~ 50 mJy | Far-IR AGN luminosity; starburst SFR |
| **ALMA** | mm/submm (0.3-3.6 mm) | Targeted deep | S_1.3mm ~ 0.1 mJy | Molecular gas; unobscured AGN luminosity |
| **VLA** | Radio (1-50 GHz) | NVSS, FIRST | S_1.4GHz ~ 1-10 mJy | Radio jets; core dominance |
| **Chandra** | X-ray (0.3-10 keV) | Deep fields | L_X ~ 10^41-10^42 erg/s (deep) | Highest spatial resolution AGN imaging |
| **HST** | Optical-NIR (0.1-1 μm) | Deep fields | AB ~ 29-31 mag | High-z AGN host galaxies; UV continuum |
| **JWST** | NIR-MIR (0.6-28 μm) | Deep fields | AB ~ 29-32 mag | High-z heavily-obscured AGN; new era |

### Deep Field Multi-Wavelength Surveys

#### COSMOS Field
**Coverage:**
- Area: 2 deg² (deep)
- Multi-wavelength data: X-ray, UV, optical, NIR, MIR, FIR, mm, radio
- Depth: X-ray L_X ~ 10^42 erg/s at z~1-3
- Redshift coverage: z ~ 0-6+
- AGN identification: ~5000+ AGN in COSMOS

**AGN Characterization Enabled:**
- AGN luminosity function evolution from z~0 to z~6
- Host galaxy properties vs. AGN properties
- AGN-starburst connection; quenching timescales
- Black hole mass and accretion rate demographics

#### Chandra Deep Field South (CDF-S)
**Coverage:**
- Area: 0.084 deg²
- Deepest X-ray survey: 7 Ms exposure (XMM+Chandra combined)
- Multiwavelength: optical, NIR, MIR, FIR (Spitzer, Herschel), radio (ALMA, VLA)
- Redshift coverage: z ~ 0-6+
- AGN density: ~2000-3000 AGN in survey area

**AGN Studies:**
- Faintest AGN detection limit: L_X ~ 10^40-10^41 erg/s at z~3
- Compton-thick AGN census (reflection-dominated)
- High-z (z>4) AGN properties and space density
- X-ray/optical/infrared color-color analysis

#### GOODS-South (Great Observatories Origins Deep Survey)
**Coverage:**
- Area: 0.16 deg²
- Multiwavelength from X-ray to radio
- Notable: Includes CDF-S central region; excellent spectroscopic redshifts
- AGN statistics: ~500-700 AGN in GOODS region

---

## VII. OBSERVATIONAL SENSITIVITIES AND DETECTION LIMITS

### X-ray Detection Limits (Fluxes and Luminosities)

| Instrument | Exposure | E range | Flux Limit (cgs) | L_X at z=0.1 | L_X at z=1 | L_X at z=3 |
|-----------|----------|---------|------------------|--------------|-----------|-----------|
| Chandra ACIS-S | 1 ks | 0.5-7 keV | 10^-14 | 3×10^40 | 3×10^42 | 3×10^44 |
| Chandra ACIS-S | 100 ks | 0.5-7 keV | 10^-15 | 3×10^39 | 3×10^41 | 3×10^43 |
| XMM-Newton EPIC | 10 ks | 0.5-10 keV | 5×10^-14 | 1.5×10^40 | 1.5×10^42 | 1.5×10^44 |
| NuSTAR | 100 ks | 3-20 keV | 10^-13 | 3×10^40 | 3×10^42 | 3×10^44 |
| eROSITA all-sky | - | 0.2-10 keV | 10^-12 | 3×10^41 | 3×10^43 | 3×10^45 |

*Note: Luminosity scaled assuming standard cosmology (H₀=70 km/s/Mpc, Ω_m=0.3)*

### Optical/NIR Spectroscopy Detection Limits

| Wavelength | Instrument | Exposure | Magnitude Limit | Flux (10^-17 erg/cm²/s) | AGN Class |
|-----------|-----------|----------|-----------------|------------------------|-----------|
| Optical | SDSS/fiber | - | r ~ 19.1 | ~100 | Bright AGN |
| Optical | HST/STIS | 1 orbit (2.6 ks) | V ~ 20 | ~0.5 | Faint Seyfert |
| Optical | VLT/VIMOS | 1 hour | r ~ 22 | ~0.05 | Distant AGN |
| NIR | Keck/MOSFIRE | 1 hour | K ~ 19 | ~0.1 | High-z AGN |
| NIR | JWST/NIRSpec | 10k s | K ~ 29 | ~10^-4 | z>6 AGN |

### Infrared Sensitivity

| Wavelength | Instrument | Exposure | Flux Limit | λ L_λ at z=1 (L_sun) |
|-----------|-----------|----------|-----------|----------------------|
| 3.6 μm | Spitzer/IRAC | 100 s | 10 μJy | 10^11 |
| 24 μm | Spitzer/MIPS | 100 s | 0.1 mJy | 10^12 |
| 70 μm | Herschel/PACS | 100 s | 50 mJy | 10^12 |
| 5-40 μm | JWST/MIRI | 10k s | 1 μJy | 10^9 |

### Radio Sensitivity

| Frequency | Instrument | Exposure | Flux Limit | Detection Type |
|-----------|-----------|----------|-----------|-----------------|
| 1.4 GHz | NVSS | Survey | 2.5 mJy | All-sky survey |
| 1.4 GHz | FIRST | Survey | 1 mJy | Targeted imaging |
| 5 GHz | VLA/A config | 1 hour | 0.1 mJy | Compact sources |
| 50 GHz | ALMA | 1 hour | 0.01 mJy | Point sources |
| 230 GHz | EHT | - | Below mJy | Black hole shadows |

---

## VIII. OBSERVATIONAL STRATEGY GUIDELINES

### Multi-Wavelength AGN Campaign Design

#### For Well-Studied Nearby Seyferts (L_bol ~ 10^44 erg/s; z ~ 0.01-0.1)

**X-ray Spectroscopy:**
- XMM-Newton: 50-100 ks (achieve Γ measurement ±0.1; Fe line EW ~0.1 keV)
- OR Chandra: 100-200 ks (higher spatial resolution for kiloparsec-scale extended emission)
- Cadence: Annual snapshots for variability monitoring

**Optical/UV Spectroscopy:**
- Ground telescope: 1-2 hr/month for emission-line variability and broad-line monitoring
- HST/STIS: 5-10 orbits/year for UV continuum and C IV monitoring

**Infrared:**
- Spitzer (archival): Use existing data; Herschel archival for FIR
- JWST: 10-20 ks/year for high-resolution MIR spectroscopy and SED refinement

**Radio:**
- VLA: 1-2 monthly monitoring at 1.4, 5 GHz
- VLBA: Quarterly 4-hour epochs (if sufficient flux for proper motions)

#### For Distant High-Luminosity Quasars (L_bol ~ 10^46-10^47 erg/s; z ~ 2-4)

**X-ray:**
- Chandra/XMM: 50-100 ks (detect sources at L_X ~ 10^44-10^45 erg/s)
- Stacking if individual sources too faint

**Optical/NIR:**
- JWST/NIRSpec: 5-10 ks (high-res spectroscopy; AGN-host decomposition)
- Ground-based NIR: Keck/MOSFIRE or VLT/EMIR for redshifted optical lines

**Infrared:**
- JWST/MIRI: 5-20 ks (MIR spectroscopy; torus characterization at z~2-4)
- Herschel FIR (archival): Bolometric luminosity, dust mass

**Radio:**
- VLA 1-10 GHz: Deep imaging (hours); flux measurement, structure
- ALMA: 30-60 min (detect continuum; map CO for gas dynamics)

#### For High-z (z>4) Heavily-Obscured AGN

**X-ray:**
- Chandra or XMM deep fields: 100-500 ks (redshifted hard X-rays; luminosity constraint)
- NuSTAR: Hard X-ray identification (if Compton-thick)

**Infrared (critical):**
- JWST/NIRCam: Imaging (detect obscured nucleus; colors)
- JWST/MIRI: Spectroscopy (high-ionization lines; torus diagnostics)
- ALMA: Submm continuum (unobscured bolometric luminosity proxy; molecular gas)

**Optical/NIR:**
- JWST/NIRSpec: Red-optical lines (Hα, [OIII]) for AGN identification
- Ground-based NIR: Follow-up spectroscopy for redshift confirmation

**Radio:**
- VLA 1-5 GHz: Jet structure (if radio-loud); morphology
- ALMA: Alignment with infrared for system characterization

---

## IX. DATA ARCHIVE RESOURCES

### Major Data Archives

| Archive | Mission(s) | Interface | Capabilities |
|---------|-----------|-----------|--------------|
| NASA HEASARC | X-ray (XMM, Chandra, NuSTAR, Suzaku) | Web + command-line | Query, download, pipeline products |
| XSA (XMM Science Archive) | XMM-Newton | Web | Browse, download; enhanced products |
| Chandra Data Archive | Chandra | Web (CDS) | Browse, download; reprocessed data |
| MAST | HST, GALEX, IUE | Web | Multi-mission, multi-wavelength queries |
| IRSA | Spitzer, Herschel, WISE, Planck | Web | Browse, download; image/spectra cutouts |
| NED | All wavelengths | Web | Compilation; object properties; references |
| SDSS | SDSS data | Web + CasJobs SQL | Query >4 billion objects; photometry, spectroscopy |

### Virtual Observatory Standards

- **VO Cone Search:** Query AGN catalog sources within cone radius
- **SED access:** Multi-wavelength photometry compilation
- **Spectral query:** Access to spectroscopic data across missions

---

**Document Status:** Complete observational techniques reference (December 2025)
**Purpose:** Citation-ready for methods sections and observational strategy discussions
