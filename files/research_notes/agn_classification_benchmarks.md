# AGN Classification Methods: Quantitative Benchmarks and Performance Metrics

**Compiled:** December 22, 2025
**Purpose:** Summary of published performance metrics, accuracy rates, and methodological comparisons

---

## TABLE 1: OPTICAL EMISSION-LINE DIAGNOSTICS PERFORMANCE

| **Method** | **Sample** | **Year** | **Purity (%)** | **Completeness (%)** | **Key Metric** | **Redshift Range** | **Reference** |
|---|---|---|---|---|---|---|---|
| BPT Diagram | SDSS (z<0.3) | 2003 | 85–92 | 75–85 | [O III]/Hβ vs. [N II]/Hα | z < 0.3 | Kauffmann et al. 2003 |
| BPT + empirical line | Local universe | 2006 | 88–95 | 80–88 | Kewley demarcation + starburst line | z < 0.2 | Kewley et al. 2006 |
| WHAN Diagram | SDSS NELGS | 2011 | 80–88 | 70–80 | [N II]/Hα vs. Hα EW | z < 0.5 | Cid Fernandes et al. 2011 |
| BPT (z~0.5) | GOODS | 2012 | 78–85 | 70–78 | Optical lines shifted to NIR | z ~ 0.4–0.7 | Xue et al. 2012 |
| Composite distinction | SDSS composite | 2011 | 60–70 (composites only) | 65–75 | Spatial distance in BPT plane | z < 0.3 | Stasińska et al. 2011 |
| High-z optical lines | COSMOS z~1 | 2013 | 72–80 | 68–76 | Hα + [O III] from grism | z ~ 0.8–1.2 | Momcheva et al. 2013 |
| JWST UV lines (z>3) | UNCOVER | 2024 | 75–85 (estimated) | 70–80 (estimated) | [Ne V], [He II] diagnostics | z > 3 | Labbé et al. 2024 |

**Key Findings:**
- BPT remains gold standard at z < 0.3 but loses discriminatory power at z > 0.5
- WHAN diagram superior for weak-emission and LINER systems
- Composite galaxies remain systematically misclassified (~30–40% correct rate)
- High-z optical diagnostics degraded; UV diagnostics needed for z > 2

---

## TABLE 2: MID-INFRARED COLOR DIAGNOSTICS

| **Method** | **Survey/Sample** | **Year** | **Purity (%)** | **Completeness (%)** | **AGN Surface Density** | **Notes** | **Reference** |
|---|---|---|---|---|---|---|---|
| WISE W1−W2 ≥ 0.8 | All-sky | 2012 | 90 (Type 1), 60 (Type 2) | 75 (Type 1), 60 (Type 2) | 61.9±5.4 deg^-2 | Simple color; misses cool tori | Stern et al. 2012 |
| WISE W1−W2 + W2−W3 | COSMOS | 2013 | 92 | 78 | Variable | Dual-color selection better | Assef et al. 2013 |
| Spitzer [3.6]−[4.5] | z<1 | 2010 | 88 | 80 | — | Spitzer depth ~1 μJy | Lacy et al. 2004 |
| Spitzer [3.6]−[4.5] + [5.8]−[8.0] | z>1 | 2012 | 85 | 75 | — | PAH features distinguish SFG | Donley et al. 2012 |
| Silicate 9.7 μm | IRS spectroscopy | 2011 | 95 (Type 2) | 70 (Type 2) | — | Requires IRS; excellent Type 2 ID | Spoon et al. 2011 |
| PAH equivalent width | IRS spectroscopy | 2013 | 92 | 78 | — | PAH-suppressed in AGN | Shirahata et al. 2013 |

**Key Findings:**
- WISE W1−W2 efficient but biased: Type 1 well-detected; Type 2 (especially cool-torus) missed
- Mid-IR color single most efficient AGN survey metric for large samples
- IRS spectroscopy provides superior diagnostics but at cost of lower sample size (Spitzer limited)
- Systematic confusion with luminous IR starburst galaxies (LIRGs/ULIRGs) at ~10–15% level

---

## TABLE 3: X-RAY SELECTION AND SPECTROSCOPY

| **Survey** | **Area (deg²)** | **Depth (ks)** | **Flux limit (erg/cm²/s)** | **N_AGN** | **X-ray Purity (%)** | **AGN Density (arcmin^-2)** | **Year** | **Notes** |
|---|---|---|---|---|---|---|---|---|
| Chandra Deep Field-North | 0.32 | 4000 | 2×10^-17 (0.5–8 keV) | 915 | 95 | 50 | 2017 | Deepest; full spectroscopy |
| Chandra Deep Field-South | 0.32 | 4000 | 1×10^-17 (0.5–8 keV) | 773 | 95 | 45 | 2017 | Matched sensitivity to CDF-N |
| COSMOS-XMM/Chandra | 2.0 | 160 (XMM) | 5×10^-15 (0.5–2 keV) | 1910 | 93 | 27 | 2019 | Best multi-wavelength ancillary data |
| AEGIS-XD | 0.2 | 800 | 1.9×10^-16 (0.5–8 keV) | 200 | 92 | 30 | 2011 | Deep field with spectroscopy |
| XXL-North | 25 | 10 | 6×10^-14 (2–10 keV) | 1646 | 91 | 2.1 | 2016 | Wide area; shallow |
| XMM-Newton Bright Survey | 0.5 (pointings) | 30 | 1×10^-12 (0.5–12 keV) | 50–200 per field | 89 | — | 2008–2011 | Spectral quality excellent |

**Performance Metrics for X-ray Classification:**

| **Method** | **Type Distinction** | **Obscuration Detection** | **Compton-thick ID** | **Limit** |
|---|---|---|---|---|
| Hardness ratio (HR1, HR2) | ~85% (Type 1 vs. 2) | N_H from HR; N_H < 10^24 | ~30% completeness | Simple power-law assumption fails |
| Full X-ray spectroscopy | ~92% (Type 1 vs. 2) | N_H measured ±0.3 dex | ~60% completeness | Requires >100 photons; expensive |
| Reflection dominance (R > 1) | N/A | Indicates heavy obscuration | ~75% completeness for CT | Non-physical fits possible |
| Fe Kα equivalent width | AGN confirmation | EW > 1 keV → heavy obscuration | Signature present | Requires high SNR |

**Key Findings:**
- Hard X-ray (2–10 keV) selection: 95% pure, 70–85% complete for L_X > 10^42 erg s^-1
- Type 1/Type 2 distinction: 85–92% accuracy from spectroscopy; lower for photometric
- Compton-thick detection: Challenging; combined IR+X-ray approach ~60% complete
- Selection bias: Type 1 AGN overrepresented by factor ~2 in X-ray surveys

---

## TABLE 4: RADIO SELECTION AND INFRARED-RADIO CORRELATION

| **Parameter** | **Local (z~0)** | **z ~ 0.5–1** | **z ~ 1–2** | **z ~ 2–3** | **Notes** | **References** |
|---|---|---|---|---|---|---|---|
| **IRC slope q (radio-based SFR)** | 2.34±0.26 | 2.30±0.15 | 2.20±0.18 | 2.10±0.20 | Weak redshift evolution; AGN contamination increases with z | Ivison et al. 2010; Magnelli et al. 2015 |
| **Radio spectral index α (SFG)** | 0.7–1.0 | 0.75±0.15 | 0.8±0.15 | 0.85±0.20 | Steep spectrum; primary cosmic rays dominate | Condon 1992; VLA-COSMOS |
| **Radio spectral index α (AGN)** | -0.5–0.5 | -0.3±0.4 | 0.0±0.4 | 0.2±0.5 | Flatter or inverted; power-law jets | VLA-COSMOS; LOFAR |
| **Radio-excess AGN threshold** | ΔR > 0.3–0.5 | ΔR > 0.4 | ΔR > 0.5 | ΔR > 0.6 | Deviation from SFR-radio relation; corrects for AGN | Iono et al. 2016; Mahony et al. 2016 |

**Radio Survey Performance:**

| **Survey** | **Frequency** | **Flux limit (μJy)** | **Area (deg²)** | **Radio-excess AGN ID efficiency** | **Year** |
|---|---|---|---|---|---|
| VLA 3 GHz (COSMOS) | 3 GHz | 2.3 μJy beam^-1 | 2 | 78% (z < 4) | 2017 |
| LOFAR 150 MHz | 150 MHz | 100 μJy beam^-1 | 5500+ (ongoing) | 72% (z < 2.5) | 2021+ |
| FIRST 1.4 GHz | 1.4 GHz | 1 mJy | 10000 | 65% (nearby AGN) | 2014 |
| NVSS 1.4 GHz | 1.4 GHz | 2.5 mJy | 37500 | 70% (z < 0.5) | 1998 |

**Key Findings:**
- Infrared-radio correlation q evolves weakly with redshift (−0.15/Δz); AGN increases q uncertainty
- Radio-excess AGN selection: 65–78% complete for z < 2; lower at z > 2
- Spectral index analysis effective (AGN vs. SFG separation ~75–80% accuracy)
- Low-accretion AGN (LLAGN) efficiently detected via radio excess even if X-ray weak

---

## TABLE 5: MULTI-WAVELENGTH SED FITTING DECOMPOSITION

| **Code** | **Components** | **Typical AGN Uncertainty (dex)** | **SFR Uncertainty (dex)** | **Stellar Mass Uncertainty (dex)** | **Computational Time per Source** | **Notes** |
|---|---|---|---|---|---|---|
| **CIGALE** | Stars, AGN, dust (clumpy) | ±0.25–0.35 | ±0.20–0.30 | ±0.15–0.25 | 1–5 min (CPU) | Bayesian; widely used; active development |
| **MAGPHYS** | Stars, dust (SMC/MW/LMC) | ±0.3–0.4 | ±0.25–0.35 | ±0.15–0.20 | 1–2 min (CPU) | Energy-conserving; excellent for UV-submm |
| **AGNfitter** | Accretion disk, torus, host, jet | ±0.2–0.3 | ±0.20–0.30 | ±0.20–0.30 | 10–30 min (CPU) | AGN-focused; multi-component explicit |
| **AGNfitter-rx** | +X-ray hard/soft components | ±0.15–0.25 | ±0.20–0.30 | ±0.20–0.30 | 20–60 min (CPU) | Radio-to-X-ray SED; best for complete data |
| **Prospector** | Flexible SFH, AGN, dust | ±0.25–0.35 | ±0.20–0.30 | ±0.15–0.25 | 60–300 min (CPU) | Hierarchical Bayesian; slow but flexible |
| **BEAGLE** | Stars, dust, nebular emission | ±0.30–0.40 | ±0.25–0.35 | ±0.15–0.25 | 5–15 min (CPU) | Nebular continuum/lines; for composite |

**Decomposition Accuracy by Photometric Coverage:**

| **Data Coverage** | **Stellar Mass** | **SFR** | **AGN L_bol** | **AGN Fraction** | **Example Survey** |
|---|---|---|---|---|---|
| **Optical only** (5 bands, U–K) | ±0.3 dex | ±0.4 dex (high systematic) | ±0.5 dex | ±0.3 dex | SDSS optical |
| **Optical + WISE (2 MIR)** | ±0.2 dex | ±0.3 dex | ±0.3 dex | ±0.25 dex | SDSS+WISE |
| **UV–optical–MIR–FIR** (10+ bands) | ±0.15 dex | ±0.2 dex | ±0.2 dex | ±0.15 dex | COSMOS full SED |
| **UV–optical–MIR–FIR–submm** (15+ bands) | ±0.1 dex | ±0.15 dex | ±0.15 dex | ±0.1 dex | Deep surveys (CDF-N/S) |

**Torus Model Variation (same data, different models):**
- Nenkova et al. (clumpy, 2008): L_AGN estimate
- CAT3D (Hönig & Kishimoto, 2017): L_AGN estimate ±0.2 dex different
- SKIRTOR (radiative transfer, 2012): L_AGN estimate ±0.25 dex different
- **Impact:** ~30% systematic variation in AGN contribution estimate from model choice alone

**Key Findings:**
- SED fitting provides detailed physical decomposition but model-dependent
- Systematic uncertainties ±0.2–0.3 dex typical for AGN bolometric luminosity
- Photometric redshift crucial: Δz = 0.1 → ±0.3 dex error in all derived quantities
- AGN contribution in composites highly uncertain (factor ~2–3 variation possible)

---

## TABLE 6: MACHINE LEARNING PERFORMANCE METRICS

### Supervised Learning Comparison

| **Method** | **Training Sample** | **Test Accuracy** | **AGN Purity** | **AGN Completeness** | **Computational Cost (test)** | **Reference** |
|---|---|---|---|---|---|---|
| **Random Forest** | SDSS 20k sources | 91% | 91% | 88% | <0.1 s/source | Ighina et al. 2023 |
| **XGBoost** | SDSS 20k sources | 90% | 89% | 86% | <0.1 s/source | Ighina et al. 2023 |
| **SVM** | SDSS 20k sources | 88% | 86% | 82% | 0.01 s/source | Ighina et al. 2023 |
| **Neural Network (3-layer)** | SDSS 20k sources | 87% | 84% | 80% | 0.05 s/source | Ighina et al. 2023 |
| **CNN (imaging)** | SDSS 210k galaxies | 78% (AGN vs. non-AGN) | 75% | 78% | 0.1 s/source (GPU) | Guo et al. 2022 |
| **SuperLearner ensemble** | Fermi LAT 2k sources | 91% | — | — | 1–10 s/source | Ighina et al. 2023 |

### Multi-Class Classification Performance (Polkas et al. 2023)

| **Galaxy Type** | **Precision** | **Recall (Completeness)** | **F1-Score** | **Notes** |
|---|---|---|---|---|
| Star-forming | 88% | 81% | 0.84 | Well-separated from AGN/passive |
| AGN | 72% | 56% | 0.63 | Confusion with composites; incomplete |
| LINER | 68% | 68% | 0.68 | Intermediate class; ambiguous |
| Composite | 60% | 65% | 0.62 | Inherent classification ambiguity |
| Passive | 92% | 85% | 0.88 | Well-separated; no emission |
| **Overall Accuracy** | — | — | **81%** | Random Forest on 5-class problem |

### Missing Data Impact (Ighina et al. 2023)

| **Imputation Method** | **Accuracy (kNN=5)** | **Accuracy (MICE)** | **AGN Purity** | **AGN Completeness** |
|---|---|---|---|---|
| **No missing data (baseline)** | 92.3% | 92.3% | 92% | 91% |
| **30% data missing → kNN imputation** | 91.1% | — | 90% | 89% |
| **30% data missing → MICE imputation** | — | 91.2% | 91% | 90% |
| **50% data missing → kNN** | 88.5% | — | 87% | 85% |

**Finding:** Imputation method negligible (0.1% difference); robustness to missing data critical for real surveys

### CNN Radio Morphology Classification (Swarup et al. 2024 / Banfield et al. 2015)

| **Task** | **Architecture** | **Accuracy** | **Notes** | **Sample** |
|---|---|---|---|---|
| **FRI vs. FRII classification** | VGG-16 | 94% | Classical radio morphology | ~3000 radio sources |
| **Bent radio galaxy detection** | ConvNeXT | 91% | State-of-the-art; handles complex morphology | ~500 bent sources |
| **Morphology + spectral index** | ResNet-50 + MLP | 88% | Combined image + tabular features | Radio survey subset |

**Key Findings:**
- Machine learning achieves ~90% accuracy for AGN classification with complete multi-wavelength data
- Performance degrades 2–4% with 30% missing data; 8–10% with 50% missing
- Supervised models outperform unsupervised (SOM achieves ~86% purity, 66% completeness)
- CNN on imaging competitive with spectroscopic methods but requires large training sets (>100k galaxies)
- Composite and LINER systems inherently difficult (F1 scores <0.65); pure SFG/passive >0.85

---

## TABLE 7: REDSHIFT-DEPENDENT CLASSIFICATION CHALLENGES AND SOLUTIONS

| **Redshift Range** | **Primary Challenge** | **Diagnostic Lines Accessibility** | **Recommended Method(s)** | **Achievable Accuracy** | **Key Uncertainties** |
|---|---|---|---|---|---|
| **z < 0.3 (Local Universe)** | Incomplete spectra for faint galaxies | All optical lines (λ > 3700 Å) | BPT/WHAN + WISE + X-ray | >90% | Dust extinction variation |
| **0.3 < z < 0.7** | Redshift-dependent [N II] line strength | Hα→NIR; [O III] shifting | [O III] optical + [N II] NIR + SED fitting | 85–88% | Metallicity effects; aperture mismatch |
| **0.7 < z < 1.5** | [O III] out of optical band | Hα+[N II] in NIR/grism | Rest-frame UV diagnostics (if available) + X-ray + radio | 80–85% | Photometric z error (Δz~0.05) |
| **1.5 < z < 3.0** | Optical lines in IR; UV lines in visible | [Ne II] IRS; rest-frame [O III] in NIRSPEC (JWST) | Mid-IR fine-structure lines + X-ray + radio | 75–82% | Model predictions incomplete |
| **z > 3.0** | Low metallicity; diagnostic lines degenerate | [Ne V], [He II] in optical; JWST spectroscopy | JWST rest-frame optical/UV + low-Z photoionization grids | 70–78% | Photoionization models insufficient |

---

## TABLE 8: SYSTEMATIC UNCERTAINTIES IN AGN CLASSIFICATION

| **Source of Uncertainty** | **Typical Magnitude** | **Impact on AGN Identification** | **Mitigation Strategy** |
|---|---|---|---|
| **Photometric redshift error** | Δz ~ 0.05–0.1 at z~1 | ±0.2–0.3 dex in derived luminosities; wrong rest frame | Spectroscopic redshift; photo-z training on AGN |
| **Dust extinction law choice** | Factor ~1.5 variation in A_V | ±0.2 dex in SFR from Hα; affects SED fitting | Multi-extinction models; model comparison |
| **X-ray spectral model** | ±0.3–0.5 dex in N_H | Misclassification of Compton-thick AGN | Multiple models (absorbed power-law, reflection, partial covering) |
| **Bolometric correction α_OX** | ±0.2–0.4 dex | ±0.3 dex error in L_bol from 2–10 keV | Use multiple bands; SED fitting instead |
| **Torus model choice** | ±0.2–0.3 dex in L_AGN | AGN fraction in composites: factor ~2 uncertainty | Compare multiple torus models; test sensitivity |
| **Aperture mismatch (radio vs. X-ray)** | ~10–30% photometry error | Misestimate AGN vs. SFG radio flux | Matched-aperture photometry; beam-convolution corrections |
| **Training set bias (ML)** | Model bias ~5–10% | Lower accuracy on underrepresented objects (high-z, low-mass, low-Z) | Augment training with diverse samples; regularization |
| **Composite system degeneracy** | Factor ~3 in AGN fraction | Cannot uniquely separate AGN and starburst | IFS for spatial resolution; multi-component fitting |

---

## TABLE 9: COMPARISON OF MAJOR AGN SURVEYS

| **Survey** | **Area (deg²)** | **Depth** | **Multi-wavelength Coverage** | **N_AGN** | **Redshift Range** | **Key Strength** | **Key Limitation** |
|---|---|---|---|---|---|---|---|
| **SDSS** | 14,000 | g,r,i,z ~ 23 mag | Optical + WISE cross-match | ~200k AGN | z < 0.8 | Largest spectroscopic sample | Optical biased; Type 2 underrepresented |
| **COSMOS** | 2 | X-ray: Chandra 160 ks | X-ray, optical, IR, radio panchromatic | ~4000 AGN | z < 6 | Best multi-wavelength ancillary data | Small area; cost-prohibitive to replicate |
| **GOODS-S/HUDF** | 0.11 | Chandra 4 Ms; deepest | Panchromatic UV–submm; spectroscopy | ~900 AGN | z < 8 | Deepest multi-wavelength | Ultra-small area; limited statistics |
| **VLA-COSMOS** | 2 | 3 GHz: 2.3 μJy beam^-1 | Radio-primary; X-ray/IR cross-match | ~3800 sources | z < 6 | Unbiased radio selection | Limited spectroscopy; ~30% AGN not X-ray detected |
| **LOFAR LoTSS** | 5500+ (ongoing) | 150 MHz: ~0.1 mJy | Radio-primary; optical cross-match | ~100k sources (ongoing) | z < 2.5 | Huge area; low-frequency unique | Moderate depth; spectroscopy limited |
| **XMM-Newton 4XMM-DR13** | All-sky | Serendipitous (varied) | X-ray primary; limited ancillary | ~13 million X-ray sources | All z | All-sky coverage; huge sample | Lower resolution; minimal spectroscopy |
| **CALIFA** | Small patches | IFU mapping (3600–7000 Å) | Optical spectroscopy only | ~600 galaxies (IFU) | z < 0.03 | Spatially-resolved diagnostics | Very small sample; nearby only |

---

## TABLE 10: COMPOSITE SYSTEM CHALLENGES AND DECOMPOSITION ACCURACY

| **System Type** | **AGN Fraction (%)** | **SED Fitting Uncertainty** | **BPT Classification** | **WISE Color Indication** | **Example** |
|---|---|---|---|---|---|
| **Pure starburst** | 0 | <±0.05 dex | Star-forming region | W1−W2 < 0.5 | M83 |
| **Low-AGN composite** | 10–30 | ±0.2 dex | Mixing region | W1−W2 ~ 0.6–0.8 | NGC 1068 (partial) |
| **High-AGN composite** | 50–70 | ±0.3 dex | Upper mixing region | W1−W2 ~ 0.8–1.0 | IRAS F13224-3809 |
| **AGN-dominated ULIRG** | 80–95 | ±0.25 dex | AGN region | W1−W2 > 1.0 | IRAS F08572+3915 |
| **AGN-only** | 100 | ±0.15 dex | Pure AGN region | W1−W2 > 1.2 | 3C 273 (quasar) |

**Decomposition Method Accuracy for Composites:**
- Single AGN+starburst SED model: L_AGN uncertainty ±0.3 dex (intrinsic degeneracy)
- Comparison of multiple models: L_AGN uncertainty ±0.2 dex (method variation)
- IFS + SED fitting combined: L_AGN uncertainty ±0.15 dex (spatial separation)
- Consensus from X-ray+radio+optical: L_AGN uncertainty ±0.2 dex (multi-wavelength)

---

## TABLE 11: AGN CLASSIFICATION DECISION TREE (Quick Reference)

```
START: Do you have AGN to classify?

├─ YES, redshift-dependent: Select method by z range
│  ├─ z < 0.3: Optical priority
│  │  ├─ Have emission lines? → BPT/WHAN
│  │  ├─ No emission lines? → WISE W1−W2 + X-ray (if available)
│  │  └─ Composite suspected? → SED fitting
│  │
│  ├─ 0.3 < z < 1: Multi-wavelength essential
│  │  ├─ Have X-ray? → X-ray primary + optical (if accessible)
│  │  ├─ Have radio? → Radio-excess diagnostic
│  │  ├─ Have IR? → SED fitting (best for this regime)
│  │  └─ Have all three? → Consensus classification (>90% accuracy)
│  │
│  ├─ 1 < z < 3: IR + X-ray + radio
│  │  ├─ Have JWST spectroscopy? → Rest-frame diagnostics
│  │  ├─ Mid-IR available? → Fine-structure lines + colors
│  │  └─ X-ray only? → Hardness ratios + spectroscopy
│  │
│  └─ z > 3: Frontier territory
│     ├─ JWST spectroscopy? → New low-Z diagnostic grids
│     ├─ X-ray detection? → Hard-band selected; rare
│     └─ Estimate photometric SED? → Caution advised; high uncertainty
│
└─ NO, just want benchmark: Use TABLE 6 (ML) or TABLE 1–5 (method-specific)
```

---

## APPENDIX: QUANTITATIVE DEFINITIONS

### AGN Luminosity Classifications (Bolometric)

- **Ultra-luminous AGN:** L_bol > 10^46 erg/s (quasars)
- **Luminous AGN:** 10^45 < L_bol < 10^46 erg/s (bright Seyferts, high-z AGN)
- **Moderate-luminosity AGN:** 10^43 < L_bol < 10^45 erg/s (typical Seyferts; M87-like LLAGN)
- **Low-luminosity AGN (LLAGN):** 10^40 < L_bol < 10^43 erg/s (Sgr A*-analog; M31)

### Accretion Rate (Eddington Ratio)

- λ_Edd = L_bol / L_Edd = (L_bol / c²) / (M_BH × G / R_s)
- **High accretion (λ > 0.1):** Quasar-like; thin disk; hot corona; Type 1 broad lines
- **Moderate (0.01 < λ < 0.1):** Seyferts; mixed disk/advection-dominated flow
- **Low (λ < 0.01):** LLAGN; advection-dominated; weak jets; weak lines

### Column Density Classifications (N_H in cm^-2)

- **Unobscured (Type 1):** N_H < 10^22 cm^-2
- **Moderately obscured:** 10^22 < N_H < 10^24 cm^-2 (Type 2)
- **Heavily obscured:** 10^24 < N_H < 1.5×10^24 cm^-2 (Compton-thick candidate)
- **Compton-thick:** N_H ≥ 1.5×10^24 cm^-2 (entire X-ray spectrum affected)

### SED Fitting Parameters (Typical Definitions)

- **Star-formation rate (SFR):** Mass formed per unit time [M_sun/yr]
  - From dust-corrected Hα: SFR = 7.9×10^-42 × L_Hα [erg/s] (M_sun/yr)
  - From FIR: SFR = 1.49×10^-10 × (L_IR / 8–1000μm) [L_sun] in M_sun/yr
  - FIR method preferred (dust-unbiased)

- **Stellar mass:** Total mass in stars [M_sun]
  - From SED: M_* typically ±0.15–0.3 dex when UV–NIR data available

- **Black hole mass:** From broad-line dispersion (Type 1) or stellar kinematics [M_sun]
  - M_BH ~ σ^4 relation; ~0.1% of M_* typical

- **Bolometric AGN luminosity:** Total power from accretion [erg/s]
  - From SED: L_bol = σ T^4 × integrated L; uncertainties ±0.2–0.3 dex typical

---

**End of Quantitative Benchmarks**

*Data compiled from 50+ peer-reviewed papers, 2000–2025*
*All metrics subject to revision as new surveys complete*
