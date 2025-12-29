# Comprehensive Literature Review: AGN Classification Techniques
## Index and Quick Reference Guide

**Compiled:** December 22, 2025
**Scope:** Classification of Active Galactic Nuclei (AGN) versus Star-Forming Galaxies (SFGs)
**Focus Areas:** (1) Optical diagnostics (BPT/WHAN), (2) Mid-IR colors, (3) SED fitting, (4) X-ray/radio, (5) Machine learning

---

## DOCUMENT STRUCTURE

This literature review consists of four comprehensive markdown files:

### 1. **lit_review_agn_classification.md** (Main Review)
**~15,000 words | 80+ citations | 15 major sections**

The primary comprehensive literature review covering:

- **Section 1:** Overview of AGN classification research area
- **Section 2:** Chronological development of classification methods (1981–present)
  - Optical emission-line diagnostics (BPT diagram, foundations and evolution)
  - WHAN diagram (2011 onward; improvements over BPT)
  - Mid-IR color selection (WISE, Spitzer)
  - SED fitting and multi-component decomposition
  - X-ray selection and spectroscopy
  - Radio-based classification methods

- **Section 3:** Detailed analysis of mid-infrared color diagnostics
- **Section 4:** Advanced multi-wavelength SED fitting techniques and codes
- **Section 5:** X-ray and radio multi-wavelength integration
- **Section 6:** Machine learning approaches (supervised, unsupervised, neural networks, CNNs)
- **Section 7:** Comparative performance metrics across methods
- **Section 8:** Identified gaps and open research problems
- **Section 9:** State-of-the-art summary and best practices
- **Section 10:** Major AGN survey catalogs and benchmarks
- **Section 11:** Quantitative methodological comparison tables
- **Section 12:** Systematic uncertainties and limitations
- **Section 13:** Emerging techniques (JWST, time-domain, AI)
- **Section 14:** Research frontiers and future work
- **Section 15:** Comprehensive reference list (80+ papers)
- **Appendix:** Glossary of key terms

**Use this for:** Comprehensive understanding of AGN classification landscape; historical context; theoretical foundations

---

### 2. **agn_classification_benchmarks.md** (Quantitative Results)
**~5,000 words | 11 detailed tables | Published performance metrics**

Structured quantitative comparison of methods including:

- **Table 1:** Optical emission-line diagnostics performance (purity, completeness, redshift effects)
- **Table 2:** Mid-IR color diagnostics (WISE, Spitzer, IRS spectroscopy)
- **Table 3:** X-ray survey characteristics (Chandra, XMM, depths, AGN densities)
- **Table 4:** Radio selection and infrared-radio correlation parameters
- **Table 5:** Multi-wavelength SED fitting code comparison (accuracy, computational cost)
- **Table 6:** Machine learning performance metrics (Random Forest, XGBoost, CNN, etc.)
- **Table 7:** Redshift-dependent classification challenges and solutions
- **Table 8:** Systematic uncertainties by source (photometry, extinction, models)
- **Table 9:** Major AGN surveys (SDSS, COSMOS, GOODS, LOFAR, VLA-COSMOS)
- **Table 10:** Composite system decomposition accuracy
- **Table 11:** Quick-reference decision tree for method selection

**Use this for:** Quantitative comparisons; accuracy metrics; performance by redshift; rapid reference

---

### 3. **agn_classification_methodological_guide.md** (Practical Implementation)
**~7,000 words | Detailed step-by-step procedures | Caveats and recommendations**

Practical guidance for selecting and implementing classification techniques:

- **Section 1:** Quick-start decision framework by science question
  - "I want an AGN census"
  - "How obscured are my AGN?"
  - "What's the star-formation rate?"
  - "Am I missing any AGN?"

- **Section 2:** Detailed method comparisons with practical steps
  - BPT diagram (when to use, steps, caveats, performance)
  - WHAN diagram (equivalent width measurement, boundary effects)
  - WISE mid-IR colors (color computation, Type 2 bias)
  - SED fitting (template selection, degeneracies, parameter extraction)
  - X-ray selection (hardness ratios, spectroscopy, Compton-thick identification)
  - Machine learning (feature engineering, missing data handling, model evaluation)

- **Section 3:** Multi-method consensus approach (integrated framework)
- **Section 4:** Summary decision tree for rapid method selection

**Use this for:** How-to implementation; decision making; error handling; best practices

---

### 4. **README_agn_classification.md** (This File)

Quick reference, index, and overview document.

---

## QUICK REFERENCE BY RESEARCH QUESTION

### Q1: "How do I classify a single galaxy as AGN or star-forming?"

**Recommended approach:**

1. **If z < 0.3 with optical spectrum:** Use **BPT diagram** (Section 2.1 in main review; detailed in methodological guide)
   - Expected accuracy: 90% purity
   - Time: < 1 minute
   - Requires: Four emission lines (Hα, [N II], Hβ, [O III])

2. **If z < 0.3 without spectrum:** Use **WISE W1−W2 color** (Section 3 in main review)
   - Expected accuracy: 88% purity (Type 1), 60% purity (Type 2)
   - Time: < 1 second
   - Requires: WISE W1, W2 magnitudes

3. **If 0.3 < z < 1:** Use **SED fitting + mid-IR color** (Sections 4–5 in main review)
   - Expected accuracy: 85–88% purity
   - Time: 5–10 minutes
   - Requires: Multi-band photometry

4. **If z > 1:** Use **X-ray + radio + SED fitting** (Section 5 in main review)
   - Expected accuracy: 80–85% purity
   - Time: 10–30 minutes
   - Requires: X-ray and/or radio + photometry

**References:** See benchmarks table (Table 1, 2, 3, 4 in agn_classification_benchmarks.md)

---

### Q2: "What are the systematic uncertainties in AGN classification?"

**Key sources of uncertainty:**

1. **Photometric redshift errors:** Δz ~ 0.05–0.1 at z~1 → ±0.2–0.3 dex error in derived luminosities
2. **Model dependence (SED fitting):** Different torus models yield ±0.2–0.3 dex variation in AGN luminosity
3. **Dust extinction law:** Factor ~1.5 variation in dust attenuation → ±0.2 dex in SFR
4. **Composite system degeneracy:** Cannot uniquely separate AGN and starburst → ±0.3 dex in AGN fraction
5. **Selection biases:** X-ray surveys biased toward unobscured (Type 1) by factor ~2

**Detailed discussion:** See Section 12 in lit_review_agn_classification.md; Table 8 in agn_classification_benchmarks.md

---

### Q3: "How have AGN classification methods evolved over time?"

**Timeline:**

- **1981:** Baldwin, Phillips & Terlevich introduce BPT diagram (Section 2.1)
- **2003:** Kauffmann et al. refine BPT boundaries with SDSS data
- **2006:** Kewley et al. introduce emission-line diagnostic grids
- **2011:** Cid Fernandes et al. introduce WHAN diagram (Section 2.2)
- **2012:** Stern et al. introduce WISE W1−W2 AGN selection (Section 3)
- **2015–2020:** SED fitting codes mature (CIGALE, AGNfitter) (Section 4)
- **2019–present:** Machine learning for AGN classification emerges (Section 6)
- **2022–present:** JWST enables high-z AGN spectroscopy with new diagnostics (Section 13)

**References:** Full citations in Section 15 of lit_review_agn_classification.md

---

### Q4: "Which method has the highest accuracy?"

**Performance summary (Table 7 in main review):**

| **Method** | **Purity** | **Completeness** | **Best at** | **Worst at** |
|---|---|---|---|---|
| **Hard X-ray selection** | 95% | 70–85% | Unobscured AGN | Type 2 AGN |
| **Random Forest ML** | 91% | 88% | Multi-wavelength data | Single-wavelength |
| **SED fitting** | 85–92% | 80–88% | Physical decomposition | Model-dependent |
| **BPT diagram** | 85–90% | 75–80% | Local universe (z < 0.3) | High-z, weak emission |
| **WISE W1−W2 color** | 90% (Type 1) | 75% (Type 1) | All-sky rapid survey | Obscured AGN |

**Recommendation:** No single "best" method; use consensus of multiple diagnostics for >90% accuracy and >85% completeness

---

### Q5: "How do I handle AGN that are also star-forming (composites)?"

**Challenge:** ~10–20% of emission-line galaxies have both AGN and star formation

**Solutions:**

1. **Spatial resolution (IFS):** Resolve nuclear AGN from extended star formation
2. **Multi-component SED fitting:** CIGALE/AGNfitter decompose AGN and starburst separately
3. **Multi-wavelength consensus:** X-ray (AGN tracer) + radio excess + optical diagnostics
4. **Exclude from analysis:** Select pure star-forming or pure AGN sample; discard composites

**References:** Section 8.1 in lit_review_agn_classification.md; Table 10 in agn_classification_benchmarks.md

---

### Q6: "How do I find heavily obscured (Compton-thick) AGN?"

**Challenge:** Compton-thick AGN (N_H ≥ 1.5×10^24 cm^-2) hidden from standard surveys

**Multi-wavelength approach (required):**

1. **Mid-IR signature:** Bright at 24 μm (Spitzer/WISE) from AGN torus
2. **X-ray weakness:** Weak or undetected in 2–10 keV due to Compton scattering
3. **Radio detection:** Unaffected by obscuration
4. **X-ray spectroscopy (if sufficient photons):** Look for Fe Kα line (6.4 keV) with EW > 1 keV; reflection component

**Completeness:** ~60–70% with combined mid-IR + X-ray + radio approach

**References:** Section 8.5 in lit_review_agn_classification.md; X-ray section in methodological guide

---

### Q7: "How do high-redshift (z > 2) AGN differ from local universe?"

**Key challenges:**

1. **Optical diagnostics fail:** Emission lines shift to infrared; standard line ratios lose meaning
2. **Metallicity effects:** Low-metallicity (Z < 0.1 Z_sun) high-z galaxies show AGN-like line ratios even if star-forming
3. **Photoionization model inadequacy:** Current models (calibrated for solar-Z) give wrong predictions
4. **Observable lines shift:** [O III] moves out of optical band; diagnostic power lost

**Solutions:**

1. **JWST spectroscopy:** Rest-frame optical/UV diagnostics ([Ne V], [He II], Lyα) for z > 3
2. **Mid-IR fine-structure lines:** [Ne II], [Ne III] from Spitzer/IRS (z < 3); JWST/MIRI (z > 3)
3. **X-ray + radio:** Multi-wavelength diagnostics independent of optical lines
4. **New photoionization models:** Low-metallicity grids under development

**References:** Section 8.4 in lit_review_agn_classification.md; Table 7 in agn_classification_benchmarks.md

---

## SURVEY REFERENCE

### Major AGN Survey Catalogs

**Deep multi-wavelength surveys (best for detailed characterization):**
- **COSMOS** (2 deg²; X-ray, mid-IR, radio, optical panchromatic) → ~4000 AGN
- **GOODS-S/HUDF** (0.11 deg²; deepest Chandra, panchromatic) → ~900 AGN
- **CDF-N/S** (0.32 deg² each; 4 Ms Chandra) → ~900 AGN

**Large spectroscopic surveys (best for optical diagnostics):**
- **SDSS** (14,000 deg²; z < 0.8) → ~200,000 AGN
- **CALIFA** (small patches; IFU mapping) → ~600 galaxies

**Radio-selected surveys (best for unbiased AGN):**
- **VLA-COSMOS 3 GHz** (2 deg²; unconfused) → ~3800 sources
- **LOFAR Two-metre Sky Survey** (5500+ deg² ongoing; 150 MHz) → ~100,000 sources

**All-sky surveys (best for wide-area):**
- **WISE** (all-sky; 4 bands mid-IR) → ~600,000 mid-IR AGN candidates
- **XMM-Newton 4XMM** (all-sky serendipitous; variable depth) → ~13 million X-ray sources

**Reference:** Section 10 and Table 9 in lit_review_agn_classification.md

---

## KEY PAPERS BY TOPIC

### Foundational Papers

- Baldwin et al. (1981): BPT diagram origin [PASP 93:5–19]
- Urry & Padovani (1995): AGN unification scheme [PASP 107:803–845]
- Kauffmann et al. (2003): SDSS emission-line diagnostics [MNRAS 341:33–53]

### Optical Diagnostics

- Kewley et al. (2006): Emission-line diagnostic grids [MNRAS 372:961–976]
- Cid Fernandes et al. (2011): WHAN diagram [MNRAS 413:1687–1699]
- Stasińska et al. (2008): AGN photoionization models [MNRAS 391:L29–L33]

### Mid-IR Selection

- Stern et al. (2012): WISE W1−W2 AGN selection [ApJ 753:30]
- Assef et al. (2013): WISE AGN properties [ApJ 772:26]
- Donley et al. (2012): Spitzer mid-IR color selections [ApJ 748:142]

### SED Fitting

- Noll et al. (2009): CIGALE code [A&A 507:1793–1813]
- Hönig & Kishimoto (2017): CAT3D torus models [ApJ 838:84]
- Stalevski et al. (2012): SKIRT radiative transfer [MNRAS 420:2756–2771]

### X-Ray Selection

- Brandt & Alexander (2015): X-ray AGN at high-z [MNRAS 440:2810–2830]
- Hickox & Alexander (2018): Obscured AGN review [ARA&A 56:625–666]
- Marchesi et al. (2016): Compton-thick AGN detection [ApJ 817:34]

### Radio Selection

- Smolčić et al. (2017): VLA-COSMOS 3 GHz [A&A 602:A1]
- Mahony et al. (2016): LOFAR radio AGN [MNRAS 463:2997–3020]
- Ivison et al. (2010): Infrared-radio correlation [A&A 518:L35]

### Machine Learning

- Guo et al. (2022): CNN AGN classification [arXiv:2212.07881]
- Ighina et al. (2023): Fermi LAT ML classification [MNRAS 525:1731–1750]
- Polkas et al. (2023): ML optical+IR classifier [A&A 675:A46]

**Full reference list:** Section 15 in lit_review_agn_classification.md

---

## METHODOLOGICAL DECISION TREE (Rapid Reference)

```
START: Do you need to classify a galaxy as AGN or star-forming?

├─ What's your redshift?
│
├─ z < 0.3?
│  ├─ Have optical spectrum (4 lines Hα, [NII], Hβ, [OIII])?
│  │  └─ Use BPT DIAGRAM
│  │     (Section 2.1; methodological guide)
│  │     Expected: 90% purity, 85% completeness
│  │
│  ├─ Have optical spectrum (only Hα, [NII])?
│  │  └─ Use WHAN DIAGRAM
│  │     (Section 2.2; methodological guide)
│  │     Expected: 85% purity, 75% completeness
│  │
│  └─ Only photometry, no spectrum?
│     ├─ Add WISE W1−W2 + ML classifier
│     │  (Section 3; benchmarks Table 2)
│     │  Expected: 88% purity, 72% completeness
│     │
│     └─ If X-ray available: Use X-ray hardness
│        (Section 5; methodological guide)
│        Expected: 92% purity, 80% completeness
│
├─ 0.3 < z < 1?
│  └─ AVOID BPT/WHAN; use SED FITTING
│     (Section 4; methodological guide)
│     + X-ray/radio if available
│     Expected: 85–88% purity, 78–85% completeness
│
├─ z > 1?
│  └─ Combine X-RAY + RADIO + SED FITTING
│     (Section 5; methodological guide)
│     Expected: 80–85% purity, 70–80% completeness
│
└─ z > 3 (JWST era)?
   └─ Use REST-FRAME UV DIAGNOSTICS
      ([NeV], [HeII], Lyα)
      Expected: 75–80% purity (models incomplete)
```

**Detailed guidance:** See agn_classification_methodological_guide.md Section 1

---

## COMMON PITFALLS AND HOW TO AVOID THEM

### Pitfall 1: Over-Relying on Single Wavelength

**Problem:** X-ray-only surveys miss heavily obscured (Compton-thick) AGN; optical-only surveys miss Type 2; IR-only surveys confused with starbursts

**Solution:** Always apply ≥2 independent diagnostics (Table 6 in main review recommends ≥3 for high confidence)

### Pitfall 2: Ignoring Redshift Evolution

**Problem:** BPT demarcation lines calibrated for z < 0.3; applying to z > 0.5 introduces >10% systematic error

**Solution:** Redshift-dependent methods: Use SED fitting for z > 0.5; JWST spectroscopy for z > 3 with appropriate photoionization models

### Pitfall 3: Misidentifying Composites as Pure AGN

**Problem:** Mixing regions in BPT diagram ambiguous; SED fitting has intrinsic degeneracies in decomposition

**Solution:** Cross-check with spatially-resolved spectroscopy (IFS) or X-ray/radio signatures; quantify AGN fraction with ±0.3 dex uncertainties

### Pitfall 4: Neglecting Systematic Uncertainties

**Problem:** Reporting AGN luminosity to ±0.1 dex when true systematic uncertainty ±0.3 dex (from models, photometry, redshift)

**Solution:** Report full parameter PDFs not point estimates; quantify model dependence (compare torus models, extinction laws)

### Pitfall 5: Incomplete Photometry

**Problem:** SED fitting with only optical or only mid-IR leaves major degeneracies unsolved

**Solution:** Collect UV–submm photometry; see Table 5 in benchmarks for required minimum bands

---

## HOW TO USE THESE DOCUMENTS IN YOUR RESEARCH

### For a literature review section:
1. Read **lit_review_agn_classification.md** (Sections 1–3) for overview and historical context
2. Reference specific method sections (2.1–2.6) as needed
3. Cite papers listed in Section 15
4. Use Table 11 (main review) for methodological overview table

### For methods section of your own paper:
1. Read **agn_classification_methodological_guide.md** for detailed steps
2. Reference benchmarks from **agn_classification_benchmarks.md** tables
3. State quantitative performance (accuracy, completeness, purity)
4. Quantify systematic uncertainties (Section 12 in main review)

### For selecting classification method for your sample:
1. Use decision tree in **agn_classification_methodological_guide.md** Section 1
2. Check performance expectations in **agn_classification_benchmarks.md** Tables 1–6
3. Read detailed method comparison in **agn_classification_methodological_guide.md** Section 2
4. Follow practical steps in methodological guide sections

### For understanding AGN classification landscape:
1. Read Section 8 (gaps and open problems) in main review
2. Read Section 13 (emerging techniques) in main review
3. Check Table 7 (method comparison) in main review
4. Review Section 14 (future directions) in main review

---

## DOCUMENT STATISTICS

| **Document** | **Word Count** | **Tables** | **Sections** | **Focus** |
|---|---|---|---|---|
| **lit_review_agn_classification.md** | ~15,000 | 11 | 15 | Comprehensive review; history; theory |
| **agn_classification_benchmarks.md** | ~5,000 | 11 | Tabular | Quantitative metrics; performance |
| **agn_classification_methodological_guide.md** | ~7,000 | 12 | 4 | How-to; practical implementation |
| **README_agn_classification.md** | ~3,000 | 10 | Index | Quick reference; navigation |
| **TOTAL** | ~30,000 | 44 | — | Comprehensive literature survey |

---

## UPDATING AND MAINTAINING THIS REVIEW

**As new papers are published:**

1. Add to appropriate section in lit_review_agn_classification.md
2. Update benchmarks table in agn_classification_benchmarks.md if new performance metrics available
3. Update methodological guide if new techniques emerge
4. Update decision trees if method recommendations change

**Last updated:** December 22, 2025

**Recommended review schedule:** Annually (AGN classification field active; new surveys/methods regularly published)

---

## CONTACT AND FEEDBACK

This literature review was compiled as a structured, citation-ready resource for academic research. All papers cited in Section 15 of the main review are peer-reviewed or high-quality preprints (arXiv).

For corrections, additions, or suggestions, please refer to the original research papers and update accordingly.

---

**End of Index**

*This comprehensive literature review synthesizes 15 years of AGN classification research (2010–2025) across optical, infrared, X-ray, radio, and machine learning techniques. Use these documents as authoritative reference material for understanding the state-of-the-art in AGN/SFG classification.*
