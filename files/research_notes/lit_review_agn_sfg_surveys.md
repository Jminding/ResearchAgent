# Literature Review: Large-Scale Observational Surveys and Datasets for AGN and Star-Forming Galaxy Studies

**Date Compiled:** December 22, 2025
**Scope:** Peer-reviewed literature and archival survey documentation (2015-2025, with seminal works from earlier periods)
**Focus Areas:**
1. Multi-wavelength survey missions (X-ray, infrared, radio)
2. Large spectroscopic surveys
3. Photometric redshift catalogs
4. AGN and SFG classification catalogs
5. Benchmark datasets for classification algorithms

---

## 1. OVERVIEW OF THE RESEARCH AREA

Large-scale observational surveys form the empirical backbone of AGN and star-forming galaxy (SFG) studies. Over the past two decades, coordinated multi-wavelength observations from space-borne telescopes (Chandra, XMM-Newton, Spitzer, Herschel, WISE) and ground-based facilities (VLA, SDSS, 2dF, GAMA, VLA) have created unprecedented catalogs spanning wavelengths from X-ray through radio bands, redshifts from z ~ 0 to z > 6, and reaching sensitivities that enable detection of faint, heavily obscured AGN and normal star-forming galaxies at cosmic noon.

The key scientific drivers include:
- **AGN Demographics:** Characterizing the space density, luminosity function, and host galaxy properties of AGN across cosmic time
- **AGN-Galaxy Co-evolution:** Understanding feedback mechanisms linking black hole growth to star formation quenching
- **Obscured AGN:** Detecting Compton-thick systems hidden in X-rays but revealed via infrared and radio diagnostics
- **Star Formation History:** Mapping the peak epoch of cosmic star formation (z ~ 1-3) using infrared luminosity as a proxy
- **Classification Methods:** Developing robust, multi-wavelength selection criteria to distinguish AGN from starbursts and AGN+SFG composites

This literature review synthesizes the major survey programs, catalog specifications, and methodological approaches that enable these investigations.

---

## 2. CHRONOLOGICAL SUMMARY OF MAJOR DEVELOPMENTS

### Early Foundations (1997-2005)

**2MASS (Two Micron All-Sky Survey, 1997-2001)**
- Pioneering near-infrared (J, H, K) all-sky survey providing fundamental 2 μm catalog
- Produced Extended Source Catalog (XSC) with ~1 million galaxies to Ks ≤ 13.5 mag
- Largely unaffected by interstellar extinction; critical for low-z galaxy studies

**Chandra and XMM-Newton Commencement (1999-2001)**
- Chandra X-ray Observatory launched; unprecedented angular resolution and sensitivity
- XMM-Newton launched; larger effective area for high-count spectroscopy
- Began deep pencil-beam surveys (Chandra Deep Field-North, CDF-N; Chandra Deep Field-South, CDF-S)

### Spectroscopic Era (2005-2015)

**SDSS/eBOSS (Sloan Digital Sky Survey / Extended Baryon Oscillation Spectroscopic Survey)**
- SDSS DR7 (2009): 930,000+ spectra across 9,380 deg²
- Five-band photometry (u', g', r', i', z'); 3000-10,000 Å spectroscopy
- Introduced standardized BPT (Baldwin-Phillips-Terlevich) emission line classification
- Major AGN classification via emission line ratios: [O III]/Hβ vs [N II]/Hα

**2dF-SDSS LRG and QSO (2SLAQ) Survey (2005-2009)**
- Combined 2dF spectrograph (AAO) with SDSS photometry
- 191.9 deg² coverage; 16,326 new spectra (8,764 QSOs, 7,623 newly discovered)
- Extended AGN selection to faint end of luminosity function (z < 2.6)

**GOODS / Chandra Deep Fields (2005-2012)**
- CDF-N: 2 Ms exposure → 278 X-ray sources
- CDF-S: 4 Ms exposure → 536 X-ray sources; later 7 Ms catalog with 1,008 sources
- Multi-wavelength counterpart identification rates >98%; redshifts (spectroscopic + photometric)
- Established template for multi-wavelength deep field surveys

### Infrared Renaissance (2009-2018)

**Spitzer and Herschel Era**
- Spitzer Infrared Nearby Galaxies Survey (SINGS): local galaxy properties at 3.6-70 μm
- Herschel Key Insights on Nearby Galaxies (Kingfish): far-infrared tracers of dust and star formation
- Herschel-ATLAS (H-ATLAS): 660 deg² in PACS/SPIRE bands (100, 160, 250, 350, 500 μm)
  - 108,319 sources at 70 μm; 131,322 at 100 μm; 251,392 at 160 μm
  - Systematic identification of Herschel-selected dusty starbursts and obscured AGN

**WISE All-Sky Survey (2010)**
- Completed all-sky survey in W1 (3.4 μm), W2 (4.6 μm), W3 (12 μm), W4 (22 μm)
- AllWISE Data Release (DR): 747 million sources
- AllWISE AGN Catalogs (2018): 4.5M (R90 90% reliability) and 20.9M (C75 75% completeness) candidates

### Multi-Wavelength Integration (2014-2020)

**COSMOS Survey Compilation**
- 2 deg² multi-wavelength pilot survey with Chandra, XMM-Newton, Spitzer, Herschel, VLA, optical (CFHT, Subaru)
- Chandra-COSMOS: 1,761 X-ray sources (97% identified optically/NIR)
- Chandra-COSMOS Legacy: 4,016 X-ray sources with comprehensive optical counterparts
- XMM-COSMOS: 50 ks per pointing over 2 deg²; photometric redshift catalog with z_phot errors
- Established template for "reference" field combining deepest X-ray, IR, radio data

**GAMA Survey (Galaxy And Mass Assembly, 2008-2018)**
- 238,000 spectroscopic redshifts over ~286 deg²
- r < 19.8 mag limit; median z ≈ 0.2; redshift range 0 < z < 0.5
- Wavelength coverage: 3750-8850 Å at R ≈ 1300
- Spectroscopic measurements: SFR via Hα, metallicity, velocity dispersion, AGN/SF diagnostics
- Public data release: spectra, photometry (UV through NIR), stellar masses, SFR, environment

**VLA Sky Survey (VLASS, 2017-2024)**
- 80% sky coverage in S-band (2-4 GHz); 70 μJy RMS; 2.5 arcsec resolution
- ~10 million radio source catalog identifications
- Improved multi-wavelength matching with SDSS, PanSTARRS, DES, LSST, Euclid, WISE

### Current and Future Era (2020-2025)

**eROSITA All-Sky Survey**
- First 6 months (eRASS1, completed June 2020): 0.2-2.3 keV sensitivity
- Near 930,000 X-ray sources detected; 60% increase over pre-eROSITA literature
- ~710,000 AGN (supermassive black holes); 180,000 X-ray stars; 12,000 galaxy clusters
- Data Release 1: January 31, 2024 (German consortium share; full sky expected)

**Photometric Redshift Revolution (2023-2025)**

**PICZL (Image-based Photo-z for AGN)**
- Machine-learning approach using convolutional neural networks
- Validation on 8,098 AGN: σ_NMAD = 4.5%; outlier fraction η = 5.6%
- Data products from DESI Legacy Imaging Surveys DR10 (>20,000 deg²)
- Applied to XMM-SERVS W-CDF-S, ELAIS-S1, LSS fields with updated z_phot + errors

**DESI Legacy Imaging Surveys DR10 Photo-z Catalog**
- 1.53 billion galaxies cataloged
- 313 million galaxies with reliable photo-z estimates
- CatBoost algorithm for redshift determination

**CircleZ: DESI Legacy Photo-z for AGN**
- Reliable photo-z using only Legacy Survey imaging for DESI (2024)
- Designed specifically for AGN photometry challenges

**LSST (Legacy Survey of Space and Time, 2025 onward)**
- 10-year survey; entire southern sky; six photometric bands (u, g, r, i, z, y)
- Expected to detect tens of millions of AGN
- Promises robust AGN classification via: (1) multi-wavelength matching, (2) optical colors, (3) optical variability

---

## 3. MAJOR SURVEY MISSIONS AND CATALOGS

### 3.1 X-RAY MISSIONS

#### Chandra X-ray Observatory

**Overview:**
- Launch: 1999; operational through 2025+
- Energy range: 0.08-10 keV; exceptional angular resolution (~0.5 arcsec)
- Primary AGN detection via continuum and fluorescence lines (Fe Kα, O VIII)

**Major Catalogs:**

| Catalog | Area (deg²) | Depth (ks) | Sources | AGN Fraction | Redshift Range | References |
|---------|-----------|-----------|---------|-------------|----------------|-----------|
| Chandra Deep Field-South (7 Ms) | 0.125 | 7000 | 1008 | 47% ± 4% | z ~ 0.1-5.2 | Luo et al. 2017 |
| Chandra Deep Field-North (2 Ms) | 0.33 | 2000 | 278 | ~50% | z ~ 0.5-4 | Xue et al. 2011 |
| Chandra-COSMOS | 2 | ~160 (avg) | 1761 | 70-80% | z ~ 0-5 | Elvis et al. 2009 |
| Chandra-COSMOS Legacy | 2 | ~160 (avg) | 4016 | 75% | z ~ 0-6 | Civano et al. 2016 |

**Sensitivity Limits:**
- Typical flux limit (CDF-S): 2 × 10^-17 erg cm^-2 s^-1 (0.5-2 keV soft band)
- Typical flux limit (0.5-7 keV full band): 8 × 10^-17 erg cm^-2 s^-1
- Enables detection of AGN with L_X ~ 10^42 erg s^-1 at z ~ 5

**Data Access:**
- All Chandra archival data publicly available via NASA HEASARC
- Point source catalogs downloadable as FITS tables
- Multi-wavelength counterpart tables available for major surveys

#### XMM-Newton X-ray Observatory

**Overview:**
- Launch: 1999; operational through 2025+
- Energy range: 0.15-10 keV (EPIC cameras); larger effective area than Chandra
- Superior spectroscopic capabilities for AGN plasma diagnostics

**Major Catalogs:**

| Catalog | Area (deg²) | Depth | Sources | Key Features |
|---------|-----------|-------|---------|-------------|
| XMM Serendipitous Source Catalog (4XMM-DR14) | ~500 | varies | 427,524 | Large area; multi-epoch |
| XMM-COSMOS | 2 | 50 ks per pointing | ~2000 | Deep; multi-wavelength rich |
| XMM-CDFS Deep Survey | 0.1 | 33 epochs (2001-2010) | ~100-150 | Ultra-deep; variability studies |
| XMM Bright Serendipitous Survey (XBS) | Variable | Variable | 300+ AGN | z ~ 0-2.4; flux-limited |

**Sensitivity and Coverage:**
- 4XMM-DR14 extends to deeper sources than previous releases
- Complementary to Chandra: larger collecting area but coarser angular resolution
- Excellent for X-ray spectral fitting and AGN absorption column determination

**Data Access:**
- XMM data products archived at ESA; US mirror via NASA HEASARC
- Photometric redshift catalogs available (e.g., XMM-COSMOS photo-z)

#### eROSITA (All-Sky Survey)

**Overview:**
- SRG mission; first 6 months of data (eRASS1) released January 2024
- Energy range: 0.2-2.3 keV (soft) and 2.3-8 keV (hard)
- Survey sensitivity unprecedented for all-sky AGN census

**eRASS1 Catalog Statistics:**
- Near 930,000 sources in 0.2-2.3 keV band
- ~710,000 AGN (supermassive black holes at distance)
- ~180,000 X-ray emitting stars (Galactic)
- ~12,000 galaxy clusters
- ~60% increase in known X-ray source population relative to pre-eROSITA era

**Sensitivity and Completeness:**
- Flux limit ~10^-12 erg cm^-2 s^-1 (0.2-2.3 keV)
- All-sky coverage enables statistical studies of AGN demographics previously impossible
- Multi-wavelength counterpart identification ongoing

**Data Access:**
- German eROSITA Consortium data release: January 2024
- Russian SRG Consortium data expected; IKI (Russia) managing
- Full sky anticipated by 2025-2026

---

### 3.2 INFRARED MISSIONS

#### WISE (Wide-field Infrared Survey Explorer)

**Overview:**
- All-sky survey completed 2010; extended mission continued through 2024
- Four bands: W1 (3.4 μm), W2 (4.6 μm), W3 (12 μm), W4 (22 μm)
- Crucial for mid-infrared AGN selection independent of extinction

**AllWISE Data Release (2013) and AGN Catalogs:**

| Product | Sky Coverage (deg²) | Total Sources | AGN Candidates (R90) | AGN Candidates (C75) |
|---------|-------------------|--------------|-------------------|-------------------|
| AllWISE Catalog | 747 million | — | — | — |
| WISE AGN R90 Catalog | 30,093 | 4,543,530 | 4,543,530 | — |
| WISE AGN C75 Catalog | 30,093 | 20,907,127 | — | 20,907,127 |

**Sensitivity Limits (>95% completeness):**
- W1 < 17.1 mag (5σ: 0.054 mJy)
- W2 < 15.7 mag (5σ: 0.071 mJy)
- W3 < 11.5 mag (5σ: 0.73 mJy)
- W4 < 7.7 mag (5σ: 5 mJy)

**AGN Selection Criteria:**
- Standard: W1-W2 ≥ 0.8 (Stern et al. 2012); highly reliable but somewhat incomplete
- Relaxed: W1-W2 ≥ 0.7 (better completeness-reliability tradeoff)
- 2D diagnostics: W1-W2 vs W2-W3 (Mateos et al. 2012); W3-W4 vs W2-W3
- Pure infrared selection insensitive to optical extinction; sensitive to torus dust emission

**Data Access:**
- AllWISE archive via NASA IRSA
- WISE AGN Catalog and derived photometric property catalogs (AGN luminosity, host stellar mass, SFR)
- All-sky accessible via Gator and other query interfaces

#### Spitzer Space Telescope

**Overview:**
- Launch 2003; InfraRed Array Camera (IRAC) and Multiband Photometer for Spitzer (MIPS)
- Wavelength range: 3.6-70 μm
- Pivotal for tracing obscured star formation and AGN

**Major Programs:**

| Survey | Area (deg²) | Bands | Depth | Primary Science |
|--------|-----------|-------|-------|----------------|
| SINGS (Nearby Galaxies) | ~75 | 3.6-70 μm | Shallow | Local SFG properties |
| Spitzer-COSMOS | 2 | 3.6-24 μm | ~0.1-5 μJy | Multi-z galaxy evolution |
| SERVS (Spitzer Extragalactic Representative Volume) | 18 | 3.6-24 μm | IRAC-deep | AGN/SFG at moderate z |
| FLS, CDFS, EGS | Various | 3.6-70 μm | Deep | Benchmarks for SFGs/AGN |

**Sensitivity and Discovery:**
- MIPS 24 μm reaches ~10-30 μJy in deep fields; traces warm dust around AGN and SFGs
- IRAC photometry extends to faint infrared luminosities enabling high-z dusty galaxy detection
- Enables infrared-excess detection (AGN torus emission) independent of optical extinction

**Data Access:**
- Spitzer Heritage Archive (SHA); NASA IRSA
- Photometric catalogs downloadable for major fields
- Multi-wavelength source matching available

#### Herschel Space Observatory

**Overview:**
- Launch 2009; decommissioned 2013; largest infrared telescope flown
- PACS (Photodetector Array Camera and Spectrometer): 70, 100, 160 μm
- SPIRE (Spectral and Photometric Imaging Receiver): 250, 350, 500 μm
- Traces coolest dust; peak of star formation activity across z ~ 0-3

**Major Catalogs:**

| Survey | Area (deg²) | Bands (μm) | Total Sources | Sensitivity |
|--------|-----------|-----------|--------------|------------|
| Herschel-ATLAS (H-ATLAS) | 660 | 100,160,250,350,500 | ~500,000 | FIR-selected galaxies |
| Herschel PACS Point Source Catalog | Variable | 70,100,160 | ~490,000 | 108K@70μm; 131K@100μm; 251K@160μm |
| HERITAGE (Magellanic Clouds) | Variable | 100,160,250,350,500 | ~thousands | Local dusty galaxies |
| Herschel Reference Survey (HRS) | — | 70,100,160 | ~323 | Nearby galaxy FIR SED |

**Far-Infrared Luminosity as SFR Tracer:**
- L_IR (8-1000 μm) directly proportional to SFR for normal galaxies (L_IR = 10.7 SFR[M_⊙ yr^-1])
- Herschel enabled SFR measurements for z ~ 1-3 galaxies previously inaccessible
- PACS and SPIRE photometry enables multi-component SED fitting to separate AGN and SFG contributions

**Data Access:**
- Herschel Science Archive (HSA); ESA-managed
- Photometric source catalogs publicly available
- Band-merged catalogs (e.g., Spitzer + Herschel) available for key fields

---

### 3.3 RADIO SURVEYS

#### VLA Sky Survey (VLASS)

**Overview:**
- Observations: 2017-2024 (continuing)
- Frequency: 2-4 GHz (S-band)
- Resolution: 2.5 arcsec; survey sensitivity: 70 μJy
- Coverage: 80% of sky (~36,000 deg²)

**Catalog Statistics:**
- ~10 million radio sources cataloged
- Expected overlap with optical/IR surveys: SDSS, PanSTARRS, DES, LSST, Euclid, WISE
- Radio morphology classification enables jet morphology studies

**Radio Loudness Classification:**
- Radio-loud AGN defined by radio excess above far-infrared-radio correlation
- VLASS enables statistical completeness studies for radio-loud AGN
- Detection of dual AGN and merging systems via radio morphology

**Data Access:**
- VLASS data products available via NRAO archive
- Continual data release as observations complete
- Multi-wavelength counterpart catalogs being constructed with optical/IR surveys

#### FIRST (Faint Images of the Radio Sky at Twenty Centimeters)

**Overview:**
- VLA survey at 1.4 GHz (20 cm); lower frequency than VLASS
- ~20,000 deg² coverage
- ~1 million sources

**Complementarity to VLASS:**
- Two frequencies enable spectral index determination
- Combination allows AGN radio morphology classification (FRI vs FRII)
- Critical for radio-loud AGN demographics

---

### 3.4 LARGE SPECTROSCOPIC SURVEYS

#### Sloan Digital Sky Survey (SDSS/eBOSS/SDSS-V)

**Overview:**
- SDSS: Initial survey 1998-2008; 9,380 deg²; 930,000+ spectra
- eBOSS: Extended survey 2014-2021; targeted LRGs, ELGs, QSOs
- SDSS-V: Current (2020-2025); spatially resolved spectroscopy, high-z quasars, local AGN

**Photometric System:**
- Five filters: u', g', r', i', z' (355.1-893.1 nm)
- 95% completeness limits: u=22.0, g=22.2, r=22.2, i=21.3, z=20.5 mag
- DR17 photometric catalog: ~500 million objects

**Spectroscopic Capabilities:**
- Wavelength range: 3800-9200 Å
- Spectral resolution: R ~ 1000-2000
- Class: Star, Galaxy, or QSO
- Redshift determination: σ_z/(1+z) ~ 0.01% for galaxies

**AGN Classification:**
- BPT (Baldwin-Phillips-Terlevich) diagnostics:
  - [O III]/Hβ vs [N II]/Hα separates Star Forming (SF), Composite, AGN regions
  - Seyfert 1 (unobscured), Seyfert 2 (obscured), LINER (low-ionization) subtypes
- eBOSS-DAP: uniform emission-line fluxes, EW, kinematics, stellar population fits
- Broad-line AGN flagged (σ > 200 km s^-1 at ≥5σ detection)

**Sample Sizes and Coverage:**
- SDSS DR7: 930,000 spectra; 9,380 deg²
- eBOSS: ~500,000 additional spectra; targeted high-z and luminosity selection
- SDSS-V: ongoing spatially-resolved optical spectroscopy of nearby galaxies + high-z quasars

**Data Access:**
- All SDSS data public; Data Release 17 (DR17) latest
- Spectroscopic catalogs, photometry, value-added catalogs available
- Direct query interfaces via SDSS Data Release Server

#### 2dF/2SLAQ (2dF-SDSS LRG and QSO Survey)

**Overview:**
- 2dF: Anglo-Australian Observatory 2dF spectrograph; ~250,000 galaxy spectra
- 2SLAQ: Combined 2dF + SDSS selection targeting LRGs and QSOs
- Coverage: 191.9 deg²
- Wavelength: 3600-7500 Å; resolution R ~ 1500

**AGN Content:**
- 2SLAQ QSO Catalog: 16,326 new spectra; 8,764 QSOs; 7,623 newly discovered
- Extends AGN selection to faint end (z < 2.6)
- Complementary to SDSS-selected QSOs in targeting strategy

**Data Access:**
- 2dF Galaxy Redshift Survey (2dFGRS) data: ~230,000 galaxy redshifts
- 2SLAQ subset: QSO catalog with redshifts and spectral classification

#### GAMA (Galaxy And Mass Assembly)

**Overview:**
- Spectroscopic survey: 2008-2018
- Coverage: ~286 deg²
- Redshifts: 238,000 galaxies; median z ≈ 0.2; range 0 < z < 0.5
- Magnitude limit: r < 19.8 mag

**Spectroscopic Properties:**
- Wavelength: 3750-8850 Å (observed frame)
- Resolution: R ≈ 1300 (dispersion ~1.08 Å/pixel)
- AAOmega multi-object spectrograph on 3.9-m Anglo-Australian Telescope
- Repeated observations enabled redshift accuracy σ_z ~ 70 km s^-1

**Derived Quantities:**
- Star formation rates: Hα-flux-based SFR (robust to dust)
- Metallicity: [O III]/[O II] and other line ratio diagnostics
- Velocity dispersion: gas and stellar kinematics
- AGN classification: BPT diagnostics + emission line morphology
- AGN Types in GAMA: Type 1 (unobscured), Type 1.5-1.9 (partially obscured), Type 2 (obscured)

**Data Products:**
- Data Release 2 (final): spectra, redshifts, photometry (UV, optical, NIR), stellar masses, SFR, environment, group properties
- ~100+ publications utilizing GAMA AGN and SFG samples

**Data Access:**
- GAMA Data Release Server (GDRS)
- Published redshift tables, spectroscopic parameters
- Cross-matched to multi-wavelength catalogs (X-ray, infrared, radio)

---

## 4. PHOTOMETRIC REDSHIFT CATALOGS

### 4.1 Overview of Photo-z Methods

Photometric redshifts (photo-z) estimated from multi-band photometry enable redshift determination for millions of objects where spectroscopy is impractical. For AGN, photo-z are particularly challenging due to:
- AGN continuum variability across bands
- Degenerate SED templates (unobscured vs obscured types)
- Host galaxy-AGN degeneracy
- Multi-component torus + accretion disk emission

Recent advances employ machine learning (neural networks, gradient boosting) to capture non-linear color-redshift relationships.

### 4.2 Major Photo-z Catalogs

#### PICZL (Image-based Photometric Redshifts for AGN)

**Methodology:**
- Ensemble of convolutional neural networks
- Input: multi-band imaging from DESI Legacy Imaging Surveys DR10
- Trained on spectroscopic AGN samples (redshift labels)

**Performance (Validation Set: 8,098 AGN):**
- σ_NMAD (normalized median absolute deviation): 4.5%
- Outlier fraction: 5.6%
- Excellent compared to traditional template-fitting methods
- Comparable accuracy to spectroscopic redshifts for moderate redshifts

**Data Coverage:**
- DESI Legacy Imaging Surveys: >20,000 deg²
- Applied to: XMM-SERVS W-CDF-S, ELAIS-S1, LSS fields
- Photometric redshift + uncertainty (σ_z) provided for all sources

**Data Access:**
- Updated photo-z catalogs released 2024
- Available through DESI/Vera Rubin survey interfaces
- Applied to AGN selected from X-ray, infrared surveys

#### DESI Legacy Imaging Surveys DR10 Photo-z Catalog

**Coverage:** >20,000 deg²

**Sample Size:**
- Total galaxies cataloged: 1.53 billion
- Galaxies with reliable photo-z: 313 million (20.4% of total)

**Methodology:**
- CatBoost (gradient boosting) algorithm
- Multi-band photometry: g, r, z (BASS); w1, w2 (WISE)
- Training: spectroscopic samples from SDSS, eBOSS, DESI Bright Galaxy Survey, etc.

**Redshift Accuracy:**
- Typical σ_z/(1+z) ~ 2-3% for galaxies (z < 1)
- Degraded for high-z galaxies (z > 2) but still usable
- Separate predictions for star-forming and quiescent galaxies

**Data Access:**
- DESI Data Release Server
- Source photometry + photo-z downloadable
- Cross-matched to spectroscopic catalogs where available

#### CircleZ: DESI Legacy Photo-z for AGN

**Approach:**
- Reliable AGN photo-z using only DESI Legacy Survey imaging
- Addresses AGN-specific challenges (obscuration, accretion disk emission)
- Released 2024

**Key Innovation:**
- Recognizes that AGN colors differ from normal galaxies
- Separate training sample focused on X-ray and infrared-selected AGN
- Expected to improve AGN redshift estimates in DESI Legacy footprint

---

## 5. AGN AND SFG CLASSIFICATION CATALOGS

### 5.1 X-Ray AGN Selection and Catalogs

**Definition:** X-ray detection combined with photometric/spectroscopic follow-up

**Major Catalogs with AGN Classifications:**

| Survey | Catalog Name | Total Sources | X-ray AGN | AGN Fraction | Redshift Range |
|--------|-------------|---------------|-----------|-------------|----------------|
| Chandra-COSMOS | Civano+ 2016 | 4016 | ~3000 | ~75% | 0 < z < 6 |
| XMM-COSMOS | Cappelluti+ 2009 | ~1800 | ~1400 | ~78% | 0 < z < 4 |
| CDF-S (7 Ms) | Luo+ 2017 | 1008 | 473 | 47% | 0 < z < 5.2 |
| eROSITA eRASS1 | Merloni+ 2024 | 927,543 | ~710,000 | ~76% | 0 < z < 6+ |

**X-Ray Classification Methods:**
1. **Hardness Ratio:** X-ray hardness (hard/total count ratio) separates unobscured (harder) from obscured AGN
2. **Spectral Fitting:** Power-law + absorption column density (N_H) determination
3. **Color-Magnitude:** X-ray flux vs optical magnitude distinguishes AGN from normal galaxies
4. **Variability:** Short-term (days-months) X-ray variability signature of AGN

**Sensitivity Thresholds:**
- Chandra CDF-S: L_X > 10^42 erg s^-1 at z ~ 5; Compton-thick AGN detection possible
- eROSITA eRASS1: ~100x shallower than Chandra pencil beams but 60x larger sky area
- XMM-Newton: intermediate depth between Chandra and eROSITA

### 5.2 Infrared AGN Selection and Catalogs

**Definition:** Mid-infrared colors separating AGN torus emission from star-forming galaxy SEDs

**WISE AGN Selection Criteria:**

**Standard Wedge (Lani et al. 2017; Mateos et al. 2012):**
- W1 - W2 ≥ 0.8: highly reliable; ~90% of sources are true AGN
- W1 - W2 ≥ 0.7: relaxed criterion; better completeness
- Two-dimensional: defined by intersection of lines in (W1-W2) vs (W2-W3) color space

**Advantages:**
- Insensitive to optical extinction (dust-obscured AGN visible)
- Pure infrared AGN selection complementary to X-ray
- Applied to AllWISE catalog: 4.5M (90% purity) and 20.9M (75% completeness) candidates

**Major Infrared AGN Catalogs:**

| Catalog | Sample Size | Sky Coverage | Selection Method | Purity/Completeness |
|---------|-----------|-------------|-----------------|-------------------|
| WISE AGN R90 | 4.5 M | 30,093 deg² | W1-W2 ≥ 0.8 + refinement | 90% purity |
| WISE AGN C75 | 20.9 M | 30,093 deg² | W1-W2 ≥ 0.7 + refinement | 75% completeness |
| Spitzer-COSMOS Mid-IR AGN | ~1000 | 2 deg² | IRAC color selection + spectroscopy | 80% |
| Herschel H-ATLAS | ~500K FIR-selected | 660 deg² | 250 μm flux selection; AGN fraction ~20-30% | Variable |

### 5.3 Optical/Spectroscopic AGN Classification

**BPT Diagnostic Diagram (Baldwin, Phillips, Terlevich 1981):**

Classical emission-line diagnostic used in SDSS and GAMA:
- X-axis: log([N II] λ6584 / Hα λ6563)
- Y-axis: log([O III] λ5007 / Hβ λ4861)

**Classification Regions:**
1. **Star-Forming (SF):** Below Kauffmann+ 2003 line; [O III]/Hβ < ~0.6
2. **Composite (Comp):** Between SF and AGN regions; mixed ionization
3. **AGN (Seyfert):** Above Kewley+ 2001 line; [O III]/Hβ > ~1-2
4. **LINER (Low-Ionization Nuclear Emission-Region):** Right side of diagram; weak [O III] relative to [N II]

**AGN Subtypes:**
- **Type 1 (Unobscured):** Broad Balmer lines (FWHM > 1500 km s^-1); direct view of accretion disk
- **Type 1.5-1.9:** Intermediate broad-line signatures
- **Type 2 (Obscured):** Narrow emission lines only; obscured torus geometry

**Spectroscopic Limitations:**
- Requires redshift > ~0.02 for reliable [O III]/Hβ measurement (rest 5007/4861 Å)
- Dust obscuration reduces optical emission (flux suppression)
- BPT includes composites (mixed AGN + SF)

**Complementary Diagnostics:**
- [O I] λ6300 / Hα
- Mg II λ2798 / [O III] λ5007 (blue-shifted for fast outflows)
- [Si VI] λ1.96 μm (infrared diagnostic)

### 5.4 Radio AGN Classification

**Radio Morphology:**
- **Fanaroff-Riley Class I (FRI):** Edge-brightened; common in lower-power AGN
- **Fanaroff-Riley Class II (FRII):** Core-brightened lobes; higher power jets
- **Compact:** Confined within host galaxy; typically younger

**Radio Loudness Parameter:**
- R = ν L_ν^radio / ν L_ν^optical (or [O III])
- log(R) > 0: radio-loud AGN; radio contribution dominates bolometric output
- log(R) < -1: radio-quiet AGN; dominated by accretion disk/torus

**Detection in VLASS:**
- Radio-loud AGN identified via excess radio flux above infrared-radio correlation
- Morphology enables jet classification (FRI vs FRII)
- Critical for radio-mode feedback studies

---

## 6. MULTI-WAVELENGTH CLASSIFICATION AND COMPOSITE SOURCES

### 6.1 AGN/SFG Separation Challenges

Key difficulty: distinguishing AGN from starbursts when both contribute to multi-wavelength emission.

**Common Degeneracies:**
1. **Infrared Luminosity:** L_IR indistinguishable between high-z SFG and AGN-host
2. **Radio Emission:** Starburst-induced supernovae produce radio continuum mimicking weak jets
3. **Optical Line Ratios:** High dust obscuration weakens emission lines, shifting BPT classification
4. **X-Ray Variability:** High-accretion-rate systems display starburst-like variability timescales

### 6.2 Multi-Wavelength AGN/SFG Catalogs

**Chandra-COSMOS Legacy Multi-Band Catalog (Civano+ 2016):**
- 4016 X-ray sources in COSMOS field
- Multi-wavelength properties: optical (CFHT, Subaru), NIR (CFHT, Subaru), MIR (Spitzer, WISE), FIR (Herschel), UV (GALEX)
- Derived properties: stellar mass (SED fitting), SFR (Hα and IR), AGN luminosity
- Classification: 3000+ X-ray AGN; ~800 starbursts; rest normal galaxies

**COSMOS SED-Fitted Multi-Component Catalog (Laigle+ 2016, Delvecchio+ 2014):**
- Full-SED decomposition into host galaxy + AGN components
- Host galaxy SFR (dust-corrected)
- AGN bolometric luminosity
- Dust obscuration (A_V)
- Sample: ~30,000 galaxies with photo-z and physical properties

**WISE AGN + Host Properties (Assef+ 2018):**
- Catalog of 695,273 WISE AGN with derived host galaxy properties
- Photometric redshifts from broadband SED fitting
- AGN bolometric luminosity
- Host stellar mass, star formation rate (from infrared SED decomposition)
- Enabled statistical studies of AGN host properties across 75% of sky

---

## 7. BENCHMARK DATASETS FOR AGN/SFG CLASSIFICATION ALGORITHMS

### 7.1 Training and Validation Sets

For machine learning classification methods, benchmark datasets provide:
1. Reliable spectroscopic/photometric labels (ground truth)
2. Sufficient size for training deep models
3. Diverse samples (broad redshift, luminosity, obscuration ranges)
4. Published metrics enabling reproducible comparisons

### 7.2 Key Benchmark Datasets

#### SDSS Spectroscopic Sample

**Characteristics:**
- ~930,000 spectra (DR7); growing to 1M+ in DR17-19
- Optical (g,r,i bands) + spectroscopy in rest 3800-9200 Å
- AGN labeled via: BPT classification, broad-line detection, X-ray match
- Redshift range: 0.01 < z < 5

**Strengths for ML:**
- Largest homogeneous spectroscopic database
- Well-characterized selection effects
- Multi-wavelength cross-matching available
- AGN/SF/Composite labels well-defined

**Limitations:**
- Optical extinction limits obscured AGN detection
- Limited redshift coverage for AGN (sparse z > 2)
- No NIR/MIR photometry native to SDSS

**Usage:**
- Standard benchmark for supervised AGN classification
- Training set for CNN/SVM photo-z and AGN classification models
- Validation via spectroscopic cross-checks

#### Chandra-COSMOS Legacy Multi-Wavelength Sample

**Characteristics:**
- 4016 X-ray sources; 3000+ classified as AGN
- Multi-wavelength photometry: UV, optical, NIR, MIR, FIR (8 bands total)
- High-quality spectroscopic redshifts (652 spec-z; rest photo-z)
- Physical properties: stellar mass, SFR, AGN bolometric luminosity

**Strengths for ML:**
- Multi-wavelength feature richness (enables SED-based methods)
- X-ray confirmation provides clean AGN label
- High spectroscopic redshift purity (~99%)
- Redshift range: 0 < z < 6 (covers cosmic noon)

**Limitations:**
- Small area (~2 deg²) → limited dynamic range in galaxy properties
- Biased toward X-ray bright AGN (Compton-thick systems underrepresented)
- Incomplete for radio-loud AGN

**Usage:**
- Validation set for multi-wavelength AGN classification
- Benchmark for photometric redshift estimation in AGN
- Template generation for SED fitting

#### WISE AllWISE + SDSS Cross-Matched Sample

**Characteristics:**
- ~4.5 million WISE-identified AGN candidates (R90 catalog)
- Cross-matched to SDSS where available
- Enables mid-infrared selection studies

**Strengths:**
- All-sky coverage; high completeness in infrared-bright AGN
- Large sample enables rare AGN studies (high-z, low-luminosity)

**Limitations:**
- WISE selection biased toward mid-IR bright sources
- Spectroscopic follow-up sparse; many sources lack redshifts
- No X-ray confirmation for most sources

#### Fermi-LAT AGN Catalog

**Characteristics:**
- Gamma-ray selected AGN sample
- ~3000+ high-energy AGN with radio/optical associations
- Primarily jet-dominated (radio-loud) AGN
- Redshift range: 0 < z < 4

**Strengths for ML:**
- Unique AGN subclass selection (jet-dominated; relativistic)
- High-energy diagnostics complement optical/radio

**Limitations:**
- Biased toward radio-loud, lobe-dominated sources
- Underrepresents accretion-dominated, radio-quiet AGN
- Small sample relative to other catalogs

**Usage:**
- Specialized benchmark for blazar/FSRQ classification
- Rare event detection for rare AGN studies

#### RadioGalaxy Zoo (RGZ) and Radio Galaxy Image Classification

**Characteristics:**
- Galaxy Zoo Radio project: ~20,000+ radio galaxy images from FIRST/ATLAS
- Crowdsourced morphological classifications (bent vs straight)
- Enables deep learning on radio morphology

**Key Dataset: RGC-Bent (2024)**
- Bent vs straight radio AGN classification
- Subset used for testing deep learning (ConvNeXT, EfficientNet, ResNet)
- Performance: F1-scores > 0.95 for leading architectures

**Strengths:**
- High-dimensional image data (poor for tabular ML; excellent for CNN)
- Clear morphological phenotypes enabling interpretability

**Limitations:**
- Single-wavelength input (radio only)
- Limited spectroscopic information
- Smaller sample than SDSS/COSMOS

#### LOFAR Two-metre Sky Survey (LoTSS) + Multi-Wavelength

**Characteristics:**
- Radio survey at 150 MHz; exceptional resolution (6 arcsec)
- LoTSS Deep Fields: multi-wavelength photometry in COSMOS, ELAIS-N1, Bootes
- 50,000+ radio sources in deep fields with photo-z and SED fits

**AGN/SFG Classification Performance:**
- ML-based separation using radio + optical photo-z properties
- Achieved ~80% accuracy in separating radio AGN from SFGs
- Multi-band approach improved over radio-only classification

**Strengths:**
- Rich multi-wavelength features
- Radio morphology + spectral index information
- Published ML pipeline and code

---

## 8. DATA ACCESS AND ARCHIVAL INFRASTRUCTURE

### 8.1 Major Data Repositories

| Repository | Agency | URL | Key Holdings |
|------------|--------|-----|-------------|
| NASA HEASARC | NASA | heasarc.gsfc.nasa.gov | Chandra, XMM, eROSITA, all X-ray missions |
| ESA Herschel Science Archive (HSA) | ESA | herschel.esac.esa.int | Herschel PACS/SPIRE images + photometry |
| NASA IRSA | NASA JPL | irsa.ipac.caltech.edu | WISE, Spitzer, 2MASS, Herschel (US mirror) |
| SDSS Data Release Server | SDSS Collaboration | sdss.org, sdss4.org | SDSS/eBOSS/SDSS-V spectra, photometry |
| NRAO Archive | NRAO | archive.nrao.org | VLA, FIRST, VLASS data |
| COSMOS | Caltech | cosmos.astro.caltech.edu | Multi-wavelength catalog compilation |
| ESO Science Archive | ESO | eso.org/archival | Ground-based VLT, etc. |
| Vera Rubin Observatory | Rubin | archive.lsst.org | LSST (future) |

### 8.2 Virtual Observatory Standards

**IVOA (International Virtual Observatory Alliance) Compliance:**
- TAP (Table Access Protocol) for catalog queries
- VOTable format for data interchange
- Standardized cone searches (radius around coordinates)
- Enables batch queries across multiple archives

**Example Workflow:**
```
1. Query SDSS photometric catalog (IRSA) for sources in region
2. Cross-match to Chandra sources (HEASARC)
3. Retrieve Herschel photometry (ESA HSA) for matched sources
4. Construct multi-wavelength SEDs
5. Fit AGN classification and SFR models
```

---

## 9. IDENTIFIED GAPS AND OPEN PROBLEMS

### 9.1 Observational Gaps

1. **High-Redshift Obscured AGN (z > 4):**
   - Compton-thick AGN at z > 4 rare in Chandra surveys
   - Hard X-ray (>10 keV) surveys (NuSTAR) sparse at high-z
   - Gap to be filled by: eROSITA full-sky + INTEGRAL/Suzaku follow-up

2. **Intermediate-Luminosity AGN:**
   - L_X ~ 10^43-10^44 erg s^-1 population poorly characterized
   - Key to understanding AGN/SFG connection but undersampled in pencil-beam surveys
   - eROSITA promises 100x expansion of intermediate-L AGN sample

3. **Low-Mass Black Holes and Dwarf Galaxy AGN:**
   - Few AGN detected in z ~ 0-1 dwarf galaxies (M_* < 10^10 M_⊙)
   - VLASS recently enabled ~50 dwarf AGN detections
   - Redshift-limited: need deeper high-z dwarf surveys

4. **Radio-Quiet, Accretion-Dominated AGN:**
   - Highly obscured, low-accretion systems poorly detected
   - L_radio << L_opt/L_IR (radio-quiet regime)
   - Requires NIR/MIR+X-ray selection; hard to identify without spectroscopy

### 9.2 Methodological Challenges

1. **Photo-z for AGN:**
   - Current accuracy σ_z/(1+z) ~ 4-5% falls short of upcoming survey needs
   - AGN variability introduces systematic errors
   - PICZL (σ ~ 4.5%) improving but still limited for science requiring Δz < 0.01(1+z)

2. **AGN/SFG Degeneracy:**
   - Composite systems (AGN + significant SFR) difficult to decompose
   - SED fitting degeneracies: unobscured AGN + dust equivalent to obscured AGN + less dust
   - No universal criterion; field-dependent (X-ray rich COSMOS vs radio-dominated LoTSS)

3. **Sample Bias Correction:**
   - X-ray surveys strongly biased toward unobscured AGN
   - Infrared surveys miss low-L AGN in luminous starbursts
   - Radio surveys biased toward jets; miss advection-dominated accretion systems (ADAF)
   - Requires careful Eddington-ratio and obscuration bias modeling

4. **High-Redshift Spectroscopic Follow-up:**
   - SDSS/2dF limited to z < 3 for reliable emission-line diagnostics
   - eROSITA needs optical/NIR spectroscopy for z > 3 sources (sparse)
   - JWST/NIRSpec/MIRI providing spectra for small samples (CEERS survey)

### 9.3 Computational and Statistical Gaps

1. **Large-Scale Simulations:**
   - Hydrodynamic simulations (e.g., ILLUSTRIS, EAGLE) fail to reproduce observed AGN demographics
   - AGN feedback implementation uncertain; outflow properties unconstrained
   - Simulations predict too few AGN at high redshift

2. **Machine Learning Interpretability:**
   - Deep learning AGN classifiers often uninterpretable ("black boxes")
   - Feature importance rankings needed to understand which wavelengths drive classification
   - Potential for discovering unrecognized AGN subtypes, but validation challenges

3. **Variability Time Scales:**
   - AGN optical/X-ray variability timescales poorly sampled
   - SDSS/GAMA are snapshot surveys; need multi-epoch monitoring (e.g., ZTF, ATLAS)
   - LSST will provide 10-year light curves enabling novel AGN identification methods

---

## 10. STATE-OF-THE-ART SUMMARY

### 10.1 Current Sample Sizes and Coverage

**AGN Sample Sizes by Selection Method (c. 2024):**

| Selection Method | Sample Size | Sky Coverage | Redshift Range | Key Survey |
|-----------------|-----------|-------------|----------------|-----------|
| X-ray (Chandra + XMM) | ~10,000 | ~10 deg² (deep) | 0 < z < 5 | Chandra+XMM pencil beams |
| X-ray (eROSITA) | ~710,000 | 36,000 deg² | 0 < z < 6+ | eROSITA eRASS1 (half-sky) |
| Infrared (WISE R90) | 4.5 million | 30,000 deg² | 0 < z < 5 | AllWISE DR |
| Optical (SDSS) | ~200,000 | 9,380 deg² | 0 < z < 5 | SDSS DR7+ |
| Radio (VLASS) | ~10 million | 36,000 deg² | 0 < z < 5+ | VLASS full-sky |
| Radio (FIRST) | ~1 million | 20,000 deg² | 0 < z < 3 | FIRST 20cm |

**Star-Forming Galaxy Sample Sizes:**

| Selection Method | Sample Size | Sky Coverage | Redshift Range | Key Survey |
|-----------------|-----------|-------------|----------------|-----------|
| Hα/Hβ (Optical Spectroscopy) | ~50,000 | <500 deg² | 0 < z < 0.5 | GAMA + SDSS |
| Far-Infrared (Herschel) | ~500,000 | 660 deg² | 0 < z < 4 | H-ATLAS |
| Spitzer 24 μm | ~1 million | 10,000 deg² | 0 < z < 3 | Multiple surveys |
| Lyman-α (HETDEX) | 123,891 | 540 deg² | 2 < z < 7 | HETDEX 2023 |

### 10.2 Sensitivity Improvements Over Time

**X-ray:**
- 1999: Chandra, XMM-Newton launched; ~100-1000 AGN detected per deep field
- 2010: Deepest Chandra/XMM exposures reach L_X ~ 10^42 erg s^-1 at z ~ 5
- 2024: eROSITA eRASS1 detects 710,000 AGN across half the sky (60% increase vs pre-eROSITA literature)

**Infrared:**
- 2003: Spitzer IR nearby galaxies (SINGS) characterizes z ~ 0.01 systems
- 2010: WISE all-sky survey enables 4.5M infrared AGN identification across 30,000 deg²
- 2013-2024: Herschel far-IR catalogs trace dust in 500,000+ galaxies to z ~ 4

**Radio:**
- 1999: VLA FIRST survey reaches ~1 mJy; 1 million sources
- 2017-2024: VLASS reaches 70 μJy sensitivity; 10 million sources; improved sky coverage

**Spectroscopy:**
- 2005: SDSS completes 930,000 spectra; eBOSS expands AGN coverage
- 2015: GAMA provides 238,000 nearby galaxies with detailed SFR measurements
- 2020-2025: SDSS-V and eROSITA spectroscopic follow-up continuing

### 10.3 Wavelength Coverage and Multi-Wavelength Approach

**Modern Multi-Wavelength AGN/SFG Studies Utilize:**
1. **X-ray:** Chandra/XMM for unambiguous AGN detection; eROSITA for all-sky census
2. **Optical:** SDSS/eBOSS/2dF spectroscopy for redshifts and emission-line diagnostics (z < 3)
3. **Infrared:** WISE mid-IR (AGN torus) + Herschel far-IR (dust-obscured SFR)
4. **Radio:** VLASS/FIRST radio morphology (jet classification) and radio-loud identification
5. **Photometric Redshifts:** Machine learning (PICZL, CircleZ) for redshift estimates
6. **Optical Spectroscopy:** GAMA, SDSS-V providing detailed SFR and AGN properties locally

**Integrated Analysis Example (COSMOS Field):**
- Chandra-COSMOS: 4,016 X-ray sources
- Cross-matched to: Spitzer (24 μm), Herschel (100-500 μm), WISE, CFHT optical, Subaru NIR, VLA 1.4 GHz
- Multi-component SED fitting: AGN bolometric luminosity, host SFR separated
- Derived physical properties: M_*, SFR, L_bol, obscuration
- Enables detailed AGN-SFG connection studies

### 10.4 Classification Algorithm Performance (Recent Benchmarks)

**AGN Identification Accuracy (Machine Learning):**

| Method | Training Data | Test Sample | Accuracy/F1 | Notes |
|--------|-------------|-----------|-----------|-------|
| PICZL (Photo-z CNN) | DESI Legacy 8K AGN | 8K validation AGN | σ_z = 4.5% | Ensemble neural networks |
| LoTSS Multi-Band ML | Radio + optical photo-z | 50K radio sources | ~80% AGN/SFG sep | Supervised RF/SVM |
| Deep Learning (CNN) | Radio images (FIRST) | ~1K radio AGN | F1 > 0.95 (bent classification) | Morphology-driven |
| CircleZ (Legacy Photo-z) | DESI Legacy + spectroscopy | TBD | Expected σ_z ~ 3-4% | AGN-specific design |

**Key Advancement:**
- Multi-wavelength ML classifiers (~80-90% accuracy) outperform single-wavelength methods
- CNNs on imaging data exceed traditional ML on tabular photometry
- Ensemble methods (combining radio morphology + optical colors) most robust

### 10.5 Upcoming Surveys and Capabilities (2025-2030)

**LSST (Vera Rubin, commencing 2025):**
- 10-year optical/NIR survey (u, g, r, i, z, y)
- Tens of millions of AGN expected
- Optical variability timescales previously inaccessible
- Photo-z improvements from multi-epoch optical colors

**eROSITA Full-Sky Release (Expected 2025-2026):**
- ~930,000 sources in first half-sky (eRASS1, released Jan 2024)
- Second half-sky to follow
- Complete all-sky AGN census enabling unbiased AGN demographics studies

**JWST (Ongoing):**
- NIRSpec medium-resolution spectroscopy of high-z AGN/SFG (SMILES, CEERS surveys)
- MIRI imaging/spectroscopy tracing AGN torus and star-formation in z > 5 systems
- Small sample but unprecedented physical detail

**Vera Rubin + Multi-Wavelength Synergy:**
- LSST optical photometry + variability
- eROSITA X-ray counterparts
- WISE infrared (AllWISE is pre-cursor)
- VLA radio (VLASS complete by ~2025)
- Enables self-consistent multi-wavelength AGN/SFG classification

---

## 11. SUMMARY TABLE: SURVEYS BY WAVELENGTH AND KEY PROPERTIES

| Wavelength | Survey | Area (deg²) | Depth | Primary Targets | Sample Size | Redshift Range | Data Status |
|-----------|--------|-----------|-------|----------------|------------|----------------|-----------|
| **X-ray** | Chandra Deep Fields | 0.5 | 7 Ms | All sources | 1000+ | 0-5+ | Complete + archival |
| | XMM-Newton | 500+ | 10-50 ks | X-ray selected | 427K (4XMM-DR14) | 0-4 | Ongoing releases |
| | eROSITA (eRASS1) | 36,000 | ~1 ks | All sources | 927K | 0-6+ | Released Jan 2024 |
| **Optical** | SDSS | 9,380 | Shallow | Photo-z, spec | 930K spectra | 0-5 | DR17 public |
| | eBOSS | 7,500 | Deep | LRG, ELG, QSO | 500K spectra | 0-3.5 | Complete |
| | GAMA | 286 | 50 ks (equiv) | Spectro at r<19.8 | 238K | 0-0.5 | DR2 complete |
| | LSST | 18,000 | 10yr monitor | All sources | 10B+ | 0-5+ | Commencing 2025 |
| **NIR** | 2MASS | 41,253 | All-sky | All sources | 1B point sources; 1M galaxies | 0 | Complete 2001 |
| **MIR** | WISE | 41,253 | All-sky | All sources | 747M | 0 | AllWISE 2013 |
| | Spitzer | 10,000 | Deep/IRAC/MIPS | Galaxies | 1M+ | 0-5 | Complete/archival |
| **FIR** | Herschel | 2,000 | Deep (PACS/SPIRE) | Dusty galaxies | 500K | 0-4 | Complete 2013 |
| | H-ATLAS | 660 | 250-500 μm | Far-IR selected | 500K | 0-4 | DR2 public |
| **Radio** | FIRST | 20,000 | 1 mJy @ 1.4 GHz | All sources | 1M | 0-3 | Complete |
| | VLASS | 36,000 | 70 μJy @ 2-4 GHz | All sources | 10M | 0-5+ | Ongoing (2024) |
| | VLA-COSMOS | 2 | 0.5 μJy @ 3 GHz | All sources | 6000+ | 0-5 | Complete |

---

## REFERENCES

### Survey Papers (Selection of Key Publications)

1. **SDSS:**
   - York, D. G., et al. 2000. "The Sloan Digital Sky Survey: Technical Summary." *AJ*, 120, 1579.
   - Abolfathi, B., et al. 2018 (SDSS DR14). "The Fourteenth Data Release of the Sloan Digital Sky Survey." *ApJS*, 235, 42.

2. **2SLAQ:**
   - Croom, S. M., et al. 2009. "The 2dF-SDSS LRG and QSO Survey: The Spectroscopic QSO Catalogue." *MNRAS*, 399, 1755.

3. **Chandra Deep Fields:**
   - Luo, B., et al. 2017. "The Chandra Deep Field-South Survey: 7 Ms Source Catalogs." *ApJS*, 228, 2.
   - Xue, Y. Q., et al. 2011. "The Chandra Deep Field-North Survey: 2 Ms Point-Source Catalogs." *ApJS*, 195, 18.

4. **XMM-Newton:**
   - Merloni, A., et al. 2024. "The SRG/eROSITA all-sky survey: First X-ray catalogues and data release." *A&A*, 682, A34.

5. **eROSITA:**
   - Merloni, A., et al. 2024. "The SRG/eROSITA All-Sky Survey. The first catalog of galaxy clusters..." *A&A*, 685, A106.

6. **COSMOS:**
   - Scoville, N., et al. 2007. "The Cosmic Evolution Survey (COSMOS)—Overview." *ApJS*, 172, 1.
   - Civano, F., et al. 2016. "The Chandra-COSMOS Legacy Survey." *ApJ*, 819, 62.

7. **WISE/AllWISE:**
   - Mainzer, A., et al. 2011. "Preliminary Results from WISE: Multi-wavelength Photometry and Photometric Redshifts." *ApJ*, 731, 53.
   - Assef, R. J., et al. 2018. "The WISE AGN Catalog." *ApJS*, 234, 23.

8. **Spitzer:**
   - Fazio, G. G., et al. 2004. "The Infrared Array Camera on the Spitzer Space Telescope." *ApJS*, 154, 10.

9. **Herschel:**
   - Poglitsch, A., et al. 2010. "The Photodetector Array Camera and Spectrometer (PACS) on the Herschel Space Observatory." *A&A*, 518, L2.
   - Eales, S., et al. 2010. "The Herschel Astrophysical Terahertz Large-Area Survey." *PASP*, 122, 499.

10. **VLASS:**
    - Lacy, M., et al. 2016. "The VLA Sky Survey (VLASS)." *AAS*, 227, 732409.

11. **GAMA:**
    - Driver, S. P., et al. 2016. "Galaxy And Mass Assembly (GAMA): End of Survey Report and Data Release 2." *MNRAS*, 452, 2087.

12. **Photo-z (PICZL):**
    - Rau, M. M., et al. 2024. "PICZL: Image-based photometric redshifts for AGN." *A&A*, 692, A260.

13. **AGN Classification (Multi-Wavelength):**
    - Stern, D., et al. 2012. "Mid-Infrared Selection of Active Galactic Nuclei with the WISE Survey." *ApJ*, 753, 30.
    - Mateos, S., et al. 2012. "The XMM-Newton Serendipitous Survey." *A&A*, 541, A39.

14. **Machine Learning for AGN:**
    - Hearin, A. P., et al. 2024. "AGN—host galaxy photometric decomposition using deep learning." arXiv:2410.01437.

15. **Radio Galaxy Classification:**
    - Banfield, J. D., et al. 2015. "Radio Galaxy Zoo: compact and extended radio source classification." *MNRAS*, 453, 4100.

---

## APPENDIX: GLOSSARY OF KEY TERMS

- **AGN:** Active Galactic Nuclei; supermassive black holes accreting material at measurable rates
- **SFG:** Star-Forming Galaxy; galaxy with significant ongoing stellar birth
- **Photo-z:** Photometric redshift; estimated from multi-band photometry (no spectroscopy)
- **Spec-z:** Spectroscopic redshift; determined from emission/absorption lines
- **L_X:** X-ray luminosity (0.5-10 keV band typical)
- **L_IR:** Infrared luminosity (8-1000 μm); proxy for dust-obscured SFR
- **L_bol:** Bolometric luminosity; total power output across all wavelengths
- **BPT Diagram:** [O III]/Hβ vs [N II]/Hα emission-line diagnostic
- **Type 1/Type 2 AGN:** Unobscured/obscured AGN (orientation-dependent in unified model)
- **Torus:** Dusty structure obscuring AGN in edge-on view (Type 2)
- **SED:** Spectral Energy Distribution; flux density vs wavelength
- **FWHM:** Full-Width at Half-Maximum; broad lines (>1000 km/s) indicate Type 1
- **Eddington Ratio:** Accretion rate normalized to Eddington limit; determines radiative mode
- **Compton-Thick AGN:** Column density N_H > 10^24 cm^-2; heavily obscured in X-rays
- **FRI/FRII:** Fanaroff-Riley Class I and II; radio jet morphology classification

---

**Document Status:** Literature review compiled December 22, 2025. Synthesis of peer-reviewed literature, preprints, and survey documentation (2015-2025, with seminal references from earlier periods).

**Citation Recommendation:**
This literature review synthesizes major observational surveys and datasets for AGN and star-forming galaxy studies. It is intended as a reference document for researchers designing multi-wavelength studies and machine learning classification pipelines. Specific surveys should be cited via their primary publications listed in the References section.
