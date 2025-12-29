# Practical Data Guide: AGN and Star-Forming Galaxy Survey Catalogs

**Compiled:** December 22, 2025
**Purpose:** Actionable reference for accessing, utilizing, and interpreting major survey catalogs

---

## 1. QUICK-ACCESS CATALOG DIRECTORY

### 1.1 By Wavelength

#### X-RAY CATALOGS

**Chandra Deep Field-South (7 Ms)**
- **URL:** https://heasarc.gsfc.nasa.gov/ (browse catalog "CDFS7MS")
- **Data Format:** FITS tables
- **Key Columns:** RA, Dec, Flux (0.5-2keV, 2-7keV, 0.5-7keV), Counts, Hardness ratio
- **Size:** 1,008 sources; 992 with multi-wavelength counterparts
- **AGN Fraction:** 47% ± 4%
- **How to Query:**
  ```
  HEASARC: Browse CDFS7MS catalog → Download region of interest (cone search)
  Python: from astroquery.heasarc import Heasarc; heasarc.query_region()
  ```
- **Caveats:**
  - Requires spectroscopic/photometric redshift determination
  - Multi-wavelength counterpart catalog essential (separate table)

**Chandra-COSMOS Legacy Survey**
- **URL:** https://heasarc.gsfc.nasa.gov/ (browse "CHANDRACOSMOS")
- **Key Columns:** RA, Dec, Flux (soft/hard bands), Significance, Spectroscopic ID, Photometric ID
- **Size:** 4,016 X-ray sources (3,000+ AGN)
- **Associated Multi-Wavelength Data:** IRSA COSMOS portal (cosmos.astro.caltech.edu)
- **Query Method:**
  ```
  COSMOS portal: Select X-ray, retrieve photometry (optical, NIR, MIR, FIR)
  Direct download: FITS tables from archive
  ```
- **Key Advantage:** Deep multi-wavelength photometry enables SED fitting

**eROSITA eRASS1 (Half-Sky Catalog)**
- **URL:** https://erosita.mpe.mpg.de/releases/ (German consortium) OR HEASARC mirror
- **Catalog:** ERASS1MAIN
- **Data Format:** FITS
- **Key Columns:** RA, Dec, Flux (0.2-2.3keV, 2.3-8keV), ML Detection, Spectral hardness
- **Size:** 927,543 point sources; ~710,000 classified as AGN
- **Sky Coverage:** Western Galactic hemisphere (50% of sky); Eastern half pending
- **Query:**
  ```
  HEASARC: Search ERASS1MAIN by coordinates
  Python: from astroquery.heasarc import Heasarc
  ```
- **Caveats:**
  - Preliminary photometric/spectroscopic redshifts (ongoing spectroscopic follow-up)
  - Flux limits vary across survey (not uniform sensitivity)
  - Multiwavelength counterpart identification ongoing

**XMM-Newton Serendipitous Source Catalog (4XMM-DR14)**
- **URL:** https://heasarc.gsfc.nasa.gov/ (browse "XMMSSC")
- **Size:** 427,524 sources (XMM observations overlap detected)
- **Spectral Info:** Hardness ratios, spectral fit parameters available
- **How to Use:**
  ```
  Cone search around source of interest
  Cross-match to optical/infrared for counterpart identification
  Spectral fitting parameters enable hardness-based AGN classification
  ```
- **Data Products:** Source detection maps, exposure maps (science-grade)

---

#### INFRARED CATALOGS

**AllWISE Catalog**
- **URL:** https://irsa.ipac.caltech.edu/Missions/wise.html
- **Data Format:** FITS, ASCII tables, VOTable
- **Sky Coverage:** 41,253 deg² (entire sky outside Galactic plane exclusion)
- **Query Tools:**
  ```
  Gator: cone/box search, cross-ID by position
  Jupyter notebooks: from astroquery.irsa import Irsa
  TAP: Table Access Protocol for batch queries
  ```
- **Key Columns:** RA, Dec, W1, W2, W3, W4 magnitudes + uncertainties
- **Sensitivity Limits (>95% completeness):**
  ```
  W1 < 17.1 mag (3.4 μm)    [5σ: 0.054 mJy]
  W2 < 15.7 mag (4.6 μm)    [5σ: 0.071 mJy]
  W3 < 11.5 mag (12 μm)     [5σ: 0.73 mJy]
  W4 < 7.7 mag (22 μm)      [5σ: 5 mJy]
  ```
- **AGN Selection:**
  ```
  # Standard criterion (high purity)
  W1 - W2 >= 0.8

  # Relaxed (better completeness)
  W1 - W2 >= 0.7

  # 2D diagnostic (Mateos+2012)
  (W1-W2) > 0.315*(W2-W3) - 0.222
  (W1-W2) > 0.315*(W2-W3) + 0.796
  (W1-W2) < -3.172*(W2-W3) + 7.624
  ```
- **WISE AGN Catalogs (Pre-computed):**
  - R90 Catalog: 4.5M AGN candidates (30,093 deg²; 90% purity)
  - C75 Catalog: 20.9M AGN candidates (75% completeness)
  - Available: https://heasarc.gsfc.nasa.gov/w3browse/all/allwiseagn.html
  - Derived Properties Catalog (Assef+2018): Host stellar mass, SFR, AGN L_bol for 695K sources

**Spitzer IRAC/MIPS Photometry**
- **URLs:**
  - COSMOS field: https://irsa.ipac.caltech.edu/data/SPITZER/COSMOS/
  - General archive: https://irsa.ipac.caltech.edu/Missions/spitzer.html
- **Data Types:**
  - IRAC: 3.6, 4.5, 5.8, 8.0 μm imaging
  - MIPS: 24, 70, 160 μm imaging
- **Query:**
  ```
  Browse by field (ELAIS, CDFS, GOODS, COSMOS, etc.)
  Retrieve photometric catalogs (mosaic-based source lists)
  Individual source extraction available
  ```
- **COSMOS Specific:** Multi-wavelength catalog available at COSMOS portal

**Herschel PACS/SPIRE Photometry**
- **URL:** https://herschel.esac.esa.int/ (ESA Herschel Science Archive)
- **US Mirror:** https://irsa.ipac.caltech.edu/Missions/herschel.html
- **Data Formats:** FITS maps, source catalogs (PACS PSC, SPIRE catalogs)
- **Key Products:**
  - PACS: 70, 100, 160 μm point source catalogs
  - SPIRE: 250, 350, 500 μm catalogs
  - H-ATLAS: 660 deg² pre-generated catalogs
- **Herschel Reference Survey (HRS):**
  - ~323 nearby galaxies with complete PACS/SPIRE coverage
  - Ideal SED templates for high-z studies
  - Available: https://irsa.ipac.caltech.edu/data/Herschel/HERITAGE/

**Accessing Herschel Data:**
```
1. Query Herschel Science Archive for field/source
2. Download FITS maps (or pre-extracted catalogs)
3. For COSMOS: Use COSMOS portal (pre-matched photometry)
4. PSF/confusion limits provided in documentation
```

---

#### OPTICAL/SPECTROSCOPIC CATALOGS

**SDSS Photometric Catalog (DR17)**
- **URL:** https://www.sdss4.org/dr17/ (SDSS-IV official)
- **Alternative Access:** https://datalab.noirlab.edu/data/sdss
- **Query Methods:**
  ```
  Web interface: Direct cone search
  Python (datalab): from dl import queryClient
  TAP: sql2csv via Virtual Observatory
  ```
- **Key Columns:** RA, Dec, u, g, r, i, z magnitudes + errors
- **Coverage:** 9,380 deg²; ~500M objects photometrically detected
- **Spectroscopic Subset:** 930,000+ objects with redshifts (DR7) → 1M+ (DR17+)
- **How to Access Spectra:**
  ```
  SDSS Spectroscopic Catalog browser
  Download FITS spectra by plate/fiber ID
  MPA-JHU value-added products (emission lines, stellar mass, SFR, AGN class)
  ```
- **AGN/SF Classification (Value-Added):**
  ```
  BPT Classification: AGN vs Composite vs Star-Forming
  Broad-line flag: Binary (narrow or broad detected)
  Spectral classes: QSO, Galaxy, Star
  Access: Galaxy properties from MPA-JHU (https://www.sdss4.org/dr17/spectro/galaxy_mpajhu/)
  ```

**SDSS eBOSS Spectroscopic Catalogs**
- **Coverage:** ~7,500 deg²; LRGs, ELGs, QSOs
- **Size:** 500,000+ new spectra
- **eBOSS-DAP (Data Analysis Pipeline):**
  - Uniform emission-line measurements
  - Stellar population properties
  - Kinematic parameters (gas + stellar dispersion)
  - **Access:** https://www.sdss4.org/dr17/spectro/ (eBOSS-DAP products)

**2dF Galaxy Redshift Survey (2dFGRS)**
- **URL:** http://www.2dfgrs.net/ OR https://datalab.noirlab.edu/data/2dF
- **Size:** ~230,000 galaxy redshifts
- **Magnitude Limit:** bJ < 19.45 mag
- **How to Query:**
  ```
  NoirLab DataLab interface
  Direct download from archive (ASCII format available)
  ```

**GAMA Survey (Galaxy And Mass Assembly)**
- **URL:** https://www.gama-survey.org/
- **Data Release 2:** Final data release (complete)
- **Sample Size:** 238,000 spectroscopic redshifts
- **Access Methods:**
  ```
  GAMA Data Release Server (query interface)
  Direct downloads: redshifts, spectra, photometry, derived properties
  Cross-matching tools to external surveys
  ```
- **Key Value-Added Properties:**
  ```
  - Stellar mass (SED fitting)
  - Star formation rate (Hα-based + infrared)
  - Metallicity
  - AGN classification (BPT + X-ray match)
  - Group/environment membership
  ```
- **Publication Requirement:** Cite GAMA team papers per terms of use

---

#### RADIO CATALOGS

**VLASS (VLA Sky Survey)**
- **URL:** https://archive.nrao.org/
- **Frequency:** 2-4 GHz (S-band)
- **Resolution:** 2.5 arcsec
- **Sensitivity:** ~70 μJy (target); varies with position
- **Data Format:** FITS images, source catalog (increasingly available)
- **How to Access:**
  ```
  1. NRAO archive: query by coordinates/source name
  2. Download FITS imaging data
  3. Source catalogs (ongoing construction) available via archive
  4. Multi-wavelength counterpart matching: cross-reference to SDSS, PanSTARRS, WISE
  ```
- **Caveats:**
  - Survey ongoing (2017-2024+)
  - Not all regions have equal sensitivity (weather-dependent)
  - Source catalog construction ongoing
- **Radio Morphology Classification:**
  - Via manual inspection or AI (CNN-based bent galaxy classification available)
  - FRI vs FRII classification possible with morphological fitting

**FIRST (Faint Images of Radio Sky at Twenty cm)**
- **URL:** https://archive.nrao.org/archive/archiveoverview.html OR https://skyview.gsfc.nasa.gov/
- **Frequency:** 1.4 GHz
- **Resolution:** 5 arcsec
- **Source Catalog:** ~1 million sources
- **Query:**
  ```
  SkyView: Retrieve FIRST images by coordinates
  NRAO archive: Direct source catalog query
  Gator (IRSA): Cross-match FIRST to infrared sources
  ```

**VLA-COSMOS 3 GHz Survey**
- **URL:** https://cosmos.astro.caltech.edu/news/52
- **Depth:** Extremely deep (3σ ~ 2.5 μJy)
- **Area:** 2 deg² (COSMOS field only)
- **Content:** ~6,000 sources with radio morphology
- **Access:** COSMOS portal multi-wavelength download

---

### 1.2 By Scientific Application

#### **For AGN Identification:**

**Best Options:**
1. **X-ray (Unambiguous):** Chandra-COSMOS Legacy, eROSITA eRASS1
2. **Infrared (Dust-Unbiased):** WISE AllWISE (W1-W2 ≥ 0.8), Assef+2018 catalog
3. **Optical Spectroscopy (Diagnostic):** SDSS DR17, GAMA
4. **Radio (Jets):** VLASS, FIRST (radio-loud subsample)

**Recommended Multi-Method Approach:**
```
Step 1: Identify X-ray detected sources (Chandra or eROSITA)
Step 2: Cross-match to WISE for mid-IR confirmation
Step 3: Retrieve optical spectra (SDSS/GAMA if available)
Step 4: Apply BPT diagnostic if z < 3 and emission lines detected
Step 5: Estimate SFR from infrared luminosity (Herschel if available)
Result: Robust AGN+host SFR determination
```

#### **For Star-Forming Galaxy Identification:**

**Best Options:**
1. **Infrared Luminosity Proxy:** Herschel 250+ μm (direct SFR), Spitzer 24 μm
2. **Emission-Line Spectroscopy:** SDSS Hα, GAMA Hα (z < 0.5)
3. **Radio Continuum (Avoided SFG):** VLASS (SFG radio-quiet unless starburst)

**Pure SFG Selection (AGN-Avoided):**
```
1. Select infrared-bright sources (Herschel L_IR > 10^11 L_⊙)
2. Avoid X-ray detection (eROSITA/Chandra)
3. Avoid WISE AGN colors (W1-W2 < 0.5)
4. Avoid broad emission lines (spectroscopy if available)
Result: Star-forming galaxy sample with minimal AGN contamination
```

#### **For Photometric Redshift Training:**

**Recommended Datasets:**
1. **Primary Training Set:** SDSS DR17 (930K spectra) + SDSS photometry
2. **High-z Extension:** COSMOS multi-wavelength (4,000 spec-z at z > 1)
3. **AGN-Specific:** LoTSS Deep Fields (50K with photometry + photo-z quality assessment)
4. **Validation:** GAMA local universe (z < 0.5; high accuracy bench)

#### **For Machine Learning Classification:**

**Ideal Feature Sets (by completeness):**

| Task | Required Inputs | Optimal Survey Source |
|------|-----------------|----------------------|
| AGN/SFG (optical only) | ugriz magnitudes | SDSS DR17 |
| AGN/SFG (full SED) | optical+NIR+MIR | COSMOS multi-wavelength |
| Photo-z for AGN | optical imaging + photo-z truth | PICZL training set |
| Radio morphology classification | Radio images | FIRST, VLASS (imaging) |
| Multi-band AGN classification | Radio + optical photo-z | LoTSS Deep Fields |

---

## 2. SENSITIVITY LIMITS AND FLUX THRESHOLDS

### 2.1 Detection Limits by Wavelength

| Wavelength | Survey | Flux Limit | Luminosity (z=1)* | Sky Coverage | Notes |
|-----------|--------|-----------|------------------|-------------|-------|
| **0.5-2 keV** | Chandra CDF-S (7Ms) | 2×10^-17 | 10^42 erg/s | 0.125 deg² | Ultra-deep |
| | eROSITA eRASS1 | 1×10^-12 | 10^44 erg/s | 18,000 deg² | All-sky |
| | XMM-Newton (typical) | 1×10^-14 | 10^43 erg/s | 500+ deg² | Large area |
| **3.4 μm (W1)** | WISE AllWISE | 0.054 mJy (17.1 mag) | 10^44 L_⊙ | 30,000 deg² | All-sky |
| **4.6 μm (W2)** | WISE AllWISE | 0.071 mJy (15.7 mag) | 10^44 L_⊙ | 30,000 deg² | All-sky |
| **24 μm** | Spitzer MIPS | 10-30 μJy (deep) | 10^11 L_⊙ SFR** | 10,000 deg² | Field-dependent |
| **100 μm** | Herschel PACS | 5 mJy (catalog limit) | 10^12 L_⊙ SFR** | ~2,000 deg² | Deep fields |
| **250 μm** | Herschel SPIRE | 30 mJy | 10^12 L_⊙ SFR** | 660 deg² | H-ATLAS |
| **1.4 GHz** | FIRST | ~1 mJy | 10^23 W/Hz @ z=1 | 20,000 deg² | Single epoch |
| **2-4 GHz (S-band)** | VLASS | 70 μJy | Lower L_radio | 36,000 deg² | Ongoing |

*Rough L_X conversion assuming typical AGN spectrum; **SFR conversion via L_IR ∝ SFR

### 2.2 Source Detectability Criteria

**X-ray Source Detection:**
```
Chandra: Typically require ≥3σ significance in 0.5-2 keV
eROSITA: Maximum Likelihood detection; ML ≥ 6-10 for reliable sources
XMM-Newton: Source significance or hardness ratio classification
```

**Infrared Source Detection (Herschel/Spitzer):**
```
MIPS 24 μm: S/N ≥ 5 for secure detection
Herschel PACS: 70 μm typically S/N ≥ 3; 100,160 μm S/N ≥ 3
Herschel SPIRE: 250+ μm S/N ≥ 3-5 (field-dependent confusion limit)
```

**Radio Source Detection:**
```
FIRST: Typically 5 mJy single-epoch (1.4 GHz)
VLASS: ~70 μJy; deeper than FIRST but different epoch/frequency
```

**Optical/NIR Magnitude Limits:**
```
SDSS: ugriz typically r < 22 mag (95% completeness)
GAMA: r < 19.8 mag (spectroscopy)
WISE: W1 < 17.1 mag (95% completeness)
```

### 2.3 Redshift Accuracy by Method

| Method | Accuracy σ_z/(1+z) | Redshift Range | Best Use |
|--------|-------------------|----------------|----------|
| SDSS Spectroscopy | 0.01% | 0 < z < 5 | Ground truth for training |
| GAMA Spectroscopy | 0.02-0.03% | 0 < z < 0.5 | Nearby galaxies |
| PICZL Photo-z (AGN) | 4.5% | 0 < z < 4+ | AGN-specific photo-z |
| DESI Legacy Photo-z | 3-4% (z<1) | 0 < z < 2+ | General galaxy catalog |
| COSMOS Photo-z | 5-10% | 0 < z < 6 | Deep field benchmark |
| BPT Diagnostic (redshift) | Direct from spectra | 0.02 < z < 3 | AGN classification (requires Hβ,Hα, [O III], [N II]) |

---

## 3. CATALOG CROSS-MATCHING AND WORKFLOWS

### 3.1 Standard Multi-Wavelength Matching

**Typical Positional Offsets:**
```
Chandra vs optical: ±0.5 arcsec (excellent astrometry)
Herschel PACS 100 μm: ±2-3 arcsec (beam size ~9 arcsec)
Herschel SPIRE 250 μm: ±6-8 arcsec (beam size ~18 arcsec)
WISE: ±0.1 arcsec (space-based astrometry)
SDSS: ±0.1 arcsec (excellent astrometry)
VLA FIRST: ±0.5 arcsec (1.4 GHz resolution)
VLASS: ±0.5 arcsec (2.5 arcsec resolution)
```

**Recommended Matching Radius (99% accuracy):**
```
Chandra + optical/NIR: 0.6 arcsec
Chandra + WISE: 1 arcsec
Herschel + Spitzer: 4 arcsec
Herschel + WISE: 6 arcsec
Radio (VLASS) + optical: 2-3 arcsec
```

### 3.2 Python-Based Workflow Example

```python
from astroquery.heasarc import Heasarc
from astroquery.irsa import Irsa
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# Step 1: Query Chandra-COSMOS AGN
chandra = Heasarc()
agn = chandra.query_region("02:59:36 +02:24:00", radius=30*u.arcmin,
                            table="CHANDRACOSMOS")

# Step 2: Match WISE infrared photometry
coords = SkyCoord(agn['RA'], agn['DEC'])
wise = Irsa()
# Use Gator cross-match service
# Result: Matched WISE W1, W2, W3, W4 magnitudes

# Step 3: Retrieve Herschel photometry from COSMOS portal
# (Download pre-matched multi-wavelength catalog)

# Step 4: Construct SED
# Extract wavelengths: 3.4, 4.6, 12, 22 (WISE), 100, 160 (Herschel PACS)
# AGN selection: W1 - W2 ≥ 0.8

# Step 5: SED fitting
# Fit AGN torus + host galaxy templates
# Derive: L_AGN, SFR_host, A_V
```

### 3.3 Data Format Standards

**Common Formats Across Archives:**

| Format | Pros | Cons | Tools |
|--------|------|------|-------|
| FITS (Binary Table) | Preserves metadata, binary-efficient | Large files | astropy.io.fits, fitsio |
| FITS (ASCII Table) | Human-readable, preserved format | Larger file size | astropy.io.fits |
| VOTable (XML) | VO-compliant, hierarchical | Verbose | astropy.io.votable |
| ASCII (CSV) | Simple, portable | Loss of metadata | pandas.read_csv() |
| HDF5 | Efficient for large tables | Non-standard in astronomy | h5py, pandas |

**Recommendation:** Use FITS Binary Tables for raw catalogs; convert to CSV for analysis.

---

## 4. PRACTICAL TIPS FOR CATALOG USAGE

### 4.1 Avoiding Common Pitfalls

**Flux Density Units:**
- Chandra: Flux in erg cm^-2 s^-1 (integrated across band)
- WISE: Magnitudes (convert to mJy: F[mJy] = 10^(-(mag + 2.5 log(zp))/2.5) where zp is magnitude offset)
- Herschel: Flux in mJy (per beam; requires flux loss correction for extended sources)
- Radio: Flux in mJy or Jy (1 Jy = 1 mJy/beam for point sources)

**Redshift Definitions:**
- Heliocentric vs. rest frame: SDSS uses heliocentric; convert if needed
- Photo-z uncertainty: Report σ_z (not relative error); asymmetric errors for outliers

**AGN Classification:**
- BPT classification requires emission lines at >3σ; low S/N spectra unreliable
- WISE W1-W2 selection incomplete at high-z (torus SED shift toward longer wavelengths)
- X-ray hardness ratio depends on N_H; spectral fitting preferred for column density

### 4.2 Quality Flags and Selection Criteria

**SDSS Photometry Flags:**
- Use `objtype='GALAXY'` to avoid stars
- Apply `photoz_status=0` for reliable photometry
- Check `FLAGS` for bad photometry (saturation, blending, etc.)

**Chandra Source Selection:**
- Use sources from main catalog (higher significance)
- `significance ≥ 3σ` standard; eROSITA `ML ≥ 6-10`
- Hardness ratio H = (H-S)/(H+S) where H=hard counts, S=soft counts
  - H > 0.4: likely obscured AGN
  - H < 0.2: likely unobscured AGN or star

**WISE AGN Selection:**
- Require `S/N ≥ 5` in W1, W2, W3 (apply SNR cuts before color selection)
- Apply Galactic latitude cut (`|b| > 10°`) to avoid stellar confusion
- Known star contamination ~5% in R90 catalog; ~20% in C75

**Herschel Source Selection:**
- Use PSF-fitted fluxes (not aperture) to avoid confusion noise
- Check quality flag (CONFMAP value); avoid high-confusion regions
- For PACS: report "flux loss correction" (typically ~10-20% for point sources)

### 4.3 Uncertainty Propagation

**Magnitude to Flux Conversion Errors:**
```
If m_err = 0.1 mag, then flux error ~ 0.096 × flux (10% approx.)
```

**Photometric Redshift Errors:**
```
If photo-z has σ_z = 0.045(1+z), then at z=1: σ_z ~ 0.09
Luminosity distance error: σ_dL/dL ~ 0.18 (doubles uncertainty)
```

**SED Fitting Uncertainties:**
```
Stellar mass: ±0.3 dex typical
SFR: ±0.5 dex (dominated by dust obscuration assumption)
AGN luminosity: ±0.5 dex (decomposition degeneracy)
```

---

## 5. SURVEY COMPARISONS

### 5.1 AGN Sample Completeness

| Selection Method | N_AGN | Completeness | Missing AGN Type |
|-----------------|-------|-------------|-----------------|
| X-ray (Chandra deep) | 100-500 | ~90% (unobscured+moderate) | Compton-thick |
| X-ray (eROSITA) | ~710K | ~80% (all types) | Compton-ultra-thick; low-L AGN |
| WISE (W1-W2≥0.8) | 4.5M | ~90% (MIR-bright) | Dust-free AGN, radio-loud jets |
| Optical Spectroscopy (SDSS) | 200K | ~70% (broad-line biased) | Heavily obscured, low-z LINERs |
| Radio (VLASS) | ~10M potential | ~70% (radio-loud) | Radio-quiet, low-power AGN |

### 5.2 Field-to-Field Variations

**Cosmic Variance Considerations:**

| Survey | Area (deg²) | Sampling | Variance | Solution |
|--------|-----------|----------|----------|----------|
| Chandra pencil beams | ~0.5 | Ultra-deep/small | High (few Mpc scale) | Multiple fields (CDFS, CDFN, COSMOS, ECDFS) |
| eROSITA | 18,000 | All-sky shallow | Low (~0.1%) | Large sample averages variance |
| WISE | 30,000 | All-sky | Low (~0.1%) | Systematic variance minimal |
| SDSS | 9,380 | All-sky | Low (~0.1%) | Well-characterized |
| GAMA | 286 | 3 distinct fields | Medium (~5%) | Use GAMA weights for clustering |

---

## 6. DATA RELEASE TIMELINE AND EXPECTATIONS

**Recent Releases:**
- SDSS DR17 (2024)
- eROSITA eRASS1 (January 2024; German consortium)
- DESI Legacy DR10 Photo-z (2024)
- PICZL AGN Photo-z (2024)

**Expected Future Releases:**
- eROSITA Full-Sky (Eastern hemisphere): 2025-2026
- LSST First Data Release (Y1): ~2026
- VLA/VLASS Final Catalog: ~2025-2026
- JWST/CEERS AGN Catalog (NIRSpec spectra): 2024-2025 (ongoing)

**Data Access Best Practices:**
```
1. Check data release documentation for known issues
2. Verify sensitivity/completeness limits for your science case
3. Apply recommended quality cuts (survey team guidance)
4. Account for cosmic variance if using limited area samples
5. Cross-reference multi-wavelength data with published papers
```

---

## APPENDIX: QUICK REFERENCE URLS

```
NASA HEASARC (X-ray archives): https://heasarc.gsfc.nasa.gov/
NASA IRSA (Infrared): https://irsa.ipac.caltech.edu/
ESA HSA (Herschel): https://herschel.esac.esa.int/
SDSS (Optical): https://www.sdss4.org/dr17/
GAMA (Spectroscopy): https://www.gama-survey.org/
NRAO (Radio): https://archive.nrao.org/
COSMOS Multi-wavelength: https://cosmos.astro.caltech.edu/
NoirLab DataLab: https://datalab.noirlab.edu/
eROSITA: https://erosita.mpe.mpg.de/releases/
Vera Rubin/LSST: https://www.lsst.org/
```

---

**Last Updated:** December 22, 2025
