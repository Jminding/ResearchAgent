# Star-Forming Galaxies Multi-Wavelength Emission: Literature Review Index

**Compiled:** December 2025
**Scope:** Comprehensive literature survey of UV, optical, infrared, and radio emission from star-forming galaxies; SED modeling; SFR indicators; dust physics

---

## Document Overview

This directory contains four complementary literature review documents designed for researchers studying the multi-wavelength properties of star-forming galaxies (SFGs):

### 1. **lit_review_star_forming_galaxies_multiwavelength.md** (Main Review)
**Length:** ~10,000 words
**Contents:**
- Overview of research area and key concepts
- Chronological development (2000s–2025)
- Theoretical frameworks and methodologies
- Detailed methodology review (SED fitting, dust, SFR indicators)
- Quantitative results from 50+ peer-reviewed papers
- Identified gaps and open problems
- State-of-the-art summary
- 30+ key paper citations organized by topic

**Use For:**
- Understanding the broader context of multi-wavelength SFG studies
- Finding seminal papers on SED modeling, SFR calibrations, dust physics
- Learning about recent JWST and ALMA discoveries
- Writing literature review sections
- Identifying research gaps

**Key Sections:**
- Section 2: Chronological summary (easily skimmable)
- Section 3: Theoretical frameworks (physics background)
- Section 4: Detailed methodologies (technical details)
- Section 5: Quantitative results (numbers and metrics)
- Section 6: Open problems (future research directions)

---

### 2. **sfg_quantitative_benchmarks.md** (Reference Tables)
**Length:** ~5,000 words
**Contents:**
- Detailed SFR calibrations for all wavelengths (UV, Hα, 24 μm, TIR, radio)
- Dust attenuation curve parameterizations (Calzetti, Charlot & Fall)
- Infrared excess – UV slope (IRX–β) relation analysis
- Dust temperature measurement methods and typical results
- Stellar mass determination via SED fitting
- Summary tables of systematic uncertainties
- Redshift-dependent observational effects
- SED fitting code comparison matrix
- Unit conversions and quick reference

**Use For:**
- Looking up SFR calibration formulas
- Understanding dust attenuation laws
- Finding typical uncertainties in parameters
- Choosing an SED fitting code
- Troubleshooting parameter values (are they realistic?)

**Key Tables:**
- Table 1: Comparison of SFR indicators (6 wavelengths)
- Table 2: Dust properties by redshift (z = 0 to z > 12)
- Table 3: SED fitting code comparison (2024-2025)
- Table: Typical measurement uncertainties (10+ parameters)

---

### 3. **sfg_methodology_protocols.md** (Implementation Guide)
**Length:** ~6,000 words
**Contents:**
- Multi-wavelength data collection protocol (survey selection)
- Redshift-dependent bandpass selection (z = 0–1, 2–3, 4–6, ≥7)
- SED fitting workflow (configuration, fitting, validation)
- Dust attenuation measurement protocols (Balmer decrement, IRX, UV slope)
- SFR determination decision tree and multi-indicator averaging
- Quality assurance and validation procedures
- Common pitfalls and solutions
- Recommended practice workflows (minimal vs. precision studies)
- Data archiving and documentation standards
- Troubleshooting guide

**Use For:**
- Planning a multi-wavelength SFG study
- Implementing SED fitting on a new galaxy sample
- Measuring dust attenuation reliably
- Comparing multiple SFR indicators
- Debugging SED fitting issues
- Setting up analysis pipelines

**Key Workflows:**
- Section 1.2: Bandpass selection by redshift
- Section 2.2: SED fitting procedure (step-by-step)
- Section 3: Dust attenuation protocols (three methods)
- Section 4: SFR indicator decision tree
- Section 7: Recommended practice workflows

---

### 4. **This File (README.md)** (Index & Navigation)
Quick reference for finding information across all documents.

---

## Quick Navigation by Topic

### **Star Formation Rate (SFR) Indicators**

| Topic | Primary Source | Section | Key Formulas |
|-------|---|---|---|
| UV-based SFR | Benchmarks | 1.1 | SFR(UV) = 1.4×10^-28 × L_1500 |
| Hα SFR (optical) | Benchmarks | 1.2 | SFR(Hα) = 9.7×10^-42 × L_Hα |
| 24 μm SFR (MIR) | Benchmarks | 1.3 | SFR(24) = 9.2×10^-12 × L_24 |
| TIR SFR (FIR) | Benchmarks | 1.4 | SFR(TIR) = 1.0×10^-10 × L_IR |
| Radio SFR (1.4 GHz) | Benchmarks | 1.5 | SFR(radio) = 4.6×10^-29 × L_1.4 |
| Comparative analysis | Main Review | 5.1 | Table of all indicators |
| Decision tree | Protocols | 4.1 | Which indicator to use? |
| Multi-indicator averaging | Protocols | 4.2 | Weighted combination method |

---

### **Dust Physics**

| Topic | Primary Source | Section | Keywords |
|-------|---|---|---|
| Dust attenuation curves | Benchmarks | 2.1–2.2 | Calzetti, Charlot & Fall |
| IRX–β relation | Benchmarks | 2.3 | Infrared excess vs. UV slope |
| Dust temperature | Benchmarks | 3 | Modified blackbody, multi-component |
| Balmer decrement method | Protocols | 3.1 | Hα/Hβ ratio, A_V measurement |
| IRX method | Protocols | 3.2 | L_IR / L_UV attenuation proxy |
| UV slope method | Protocols | 3.3 | β parameter, reddening |
| Dust attenuation evolution | Main Review | 5.2 | JWST results z = 2–12 |
| Dust in high-z galaxies | Main Review | 3.4 | Recent ALMA & JWST findings |

---

### **SED Fitting and Modeling**

| Topic | Primary Source | Section | Keywords |
|-------|---|---|---|
| SPS fundamentals | Main Review | 3.1 | BC03, FSPS, Starburst99 |
| SED fitting workflow | Main Review | 3.2 | χ² minimization, energy balance |
| CIGALE code | Main Review | 3.2; Protocols | Modern SED fitting tool |
| SED fitting protocol | Protocols | 2 | Step-by-step procedure |
| Data preparation | Protocols | 2.1 | Quality checks, redshift |
| Post-fitting validation | Protocols | 2.3 | Parameter checks, residuals |
| Age-extinction degeneracy | Benchmarks | 4.3 | Breaking the degeneracy |
| Code comparison | Benchmarks | 7.2 | Which code to use? |

---

### **High-Redshift Galaxies (JWST Era)**

| Topic | Primary Source | Section | Keywords |
|-------|---|---|---|
| JWST capabilities | Main Review | 2 (Recent Era) | NIRCam, NIRSpec observations |
| JADES survey results | Main Review | 5.4 | Balmer decrements z = 4–7 |
| High-z dust evolution | Main Review | 5.2 | A_V trend with z |
| z ≥ 4 bandpass selection | Protocols | 1.2 | Which filters to measure? |
| ALMA dust continuum (z > 4) | Main Review | 3.4 | 158 μm observations |
| Early universe SFGs | Main Review | 5.3 | z = 4–6 properties |
| CRISTAL survey | Main Review | 2 | Dust properties at z = 4–6 |

---

### **Systematic Uncertainties and Validation**

| Topic | Primary Source | Section | Keywords |
|-------|---|---|---|
| Parameter uncertainties | Benchmarks | 5 | Typical error budgets |
| SFR consistency checks | Protocols | 5.1 | Multi-indicator agreement |
| SED quality assurance | Protocols | 5 | χ², residuals, validation |
| Common pitfalls | Protocols | 6 | Mistakes to avoid |
| Troubleshooting | Protocols | 10 | Fixing analysis problems |

---

### **Observational Facilities**

| Topic | Primary Source | Section | Keywords |
|-------|---|---|---|
| Survey selection | Protocols | 1.1 | Depth, resolution, wavelengths |
| Redshift-dependent surveys | Protocols | 1.2 | Best data for each z range |
| JWST imaging | Main Review | 2 | NIRCam capabilities |
| ALMA continuum | Main Review | 2 | Dust observations 250–350 GHz |
| Herschel photometry | Main Review | 2 | FIR SED constraints |
| VLA radio | Benchmarks | 1.5 | SFR and AGN diagnostics |
| Historical development | Main Review | 2 | Spitzer, GALEX, evolution |

---

### **Key Redshift Ranges**

**z ~ 0 (Local Universe)**
- Main Review: Section 2 (Foundations)
- Benchmarks: Tables 2, typical z~0 values
- Protocols: 1.2 (z = 0–1 bandpass selection)

**z ~ 0.5–2 (Intermediate)**
- Main Review: Section 5 (Quantitative Results)
- Benchmarks: Section 2.3 (IRX–β at intermediate z)
- Protocols: 1.2 (z = 2–3 bandpass selection)

**z ~ 2–4 (Cosmic Noon)**
- Main Review: Sections 3.2 (Dust), 5.3 (Stellar mass & SFR)
- Benchmarks: Table 2 (z~2–3 dust properties)
- Protocols: 1.2 (z = 2–3 strategy); 4.2 (multi-indicator verification)

**z ~ 4–6 (Early Reionization)**
- Main Review: Sections 2 (ALMA-CRISTAL), 5.3 (ALMA results)
- Benchmarks: Table 2 (z~4–6 values)
- Protocols: 1.2 (z = 4–6 bandpass selection); 7.1 (minimal resources study)

**z ≥ 7 (JWST Frontier)**
- Main Review: Sections 2 (JWST era), 5.4 (JADES results)
- Benchmarks: Table 2 (z~7–12 dust evolution)
- Protocols: 1.2 (z ≥ 7 strategy); 7.2 (full multi-wavelength study)

---

## Recommended Reading Paths

### **Path 1: For Researchers New to SFG Multi-Wavelength Studies**

1. **Start:** Main Review, Section 1 (Overview)
2. **Read:** Main Review, Section 2 (Chronological Development)
3. **Study:** Main Review, Section 3 (Theoretical Frameworks)
4. **Reference:** Benchmarks, Sections 1–2 (SFR & dust formulas)
5. **Implement:** Protocols, Section 2 (SED fitting workflow)
6. **Validate:** Protocols, Section 5 (Quality assurance)

**Estimated time:** 4–6 hours; achieves working understanding

---

### **Path 2: For Practitioners Planning an SFG Analysis**

1. **Quick Review:** Main Review, Section 5 (State of the Art)
2. **Data Planning:** Protocols, Section 1 (Data collection)
3. **Implementation:** Protocols, Section 2 (SED fitting)
4. **Reference:** Benchmarks, Sections 1–5 (Formulas & tables)
5. **Troubleshoot:** Protocols, Section 10 (Debugging)

**Estimated time:** 3–4 hours; ready to start analysis

---

### **Path 3: For Literature Review Writers**

1. **Overview:** Main Review, Sections 1–2
2. **Deep Dive:** Main Review, Sections 3–6
3. **Recent Results:** Main Review, Section 5
4. **Gaps:** Main Review, Section 6
5. **Cite:** Main Review, Section 8 (Papers organized by topic)

**Estimated time:** 2–3 hours; comprehensive foundation for writing

---

### **Path 4: For Code Users (SED Fitting)**

1. **Concepts:** Main Review, Section 3.2 (SED fitting theory)
2. **Tools:** Benchmarks, Section 7.2 (Code comparison table)
3. **Setup:** Protocols, Section 2.2 (CIGALE workflow example)
4. **Validation:** Protocols, Section 2.3 (Post-fitting checks)
5. **Reference:** Benchmarks, Sections 1–5 (Parameter formulas)

**Estimated time:** 2–3 hours; ready to use SED fitting tools

---

### **Path 5: For Dust Physics Specialists**

1. **Background:** Main Review, Sections 2–3 (Historical + theoretical)
2. **Detailed Physics:** Main Review, Section 3.4 (Dust attenuation & re-emission)
3. **Formulas:** Benchmarks, Sections 2–3 (Attenuation curves, temperature)
4. **Measurement:** Protocols, Section 3 (Dust attenuation protocols)
5. **High-z Results:** Main Review, Section 5.2 (Dust evolution)

**Estimated time:** 3–4 hours; thorough dust understanding

---

## Citation Format

**For individual documents:**

```
Smith, J. A. et al. (2025). Star-Forming Galaxies Multi-Wavelength Emission:
Literature Review. arXiv preprint (or University Archive).
```

**For specific sections, cite like:**

```
As discussed in the SED modeling framework review (Main Review, Section 3.2),
modern codes enforce energy balance to constrain dust simultaneously with
stellar populations.
```

**For quantitative results:**

```
Recent SFR calibrations (Benchmarks, Section 1.4) indicate TIR-based SFR
measurements achieve ±0.2 dex precision when FIR data available.
```

---

## Key Statistics

| Metric | Value |
|--------|-------|
| **Total papers reviewed** | 50+ peer-reviewed papers + preprints |
| **Redshift range covered** | z = 0 to z > 12 |
| **Wavelength coverage** | UV (0.1 μm) to radio (GHz) |
| **SED fitting codes discussed** | 7 major codes (BC03, FSPS, CIGALE, MAGPHYS, etc.) |
| **SFR indicators detailed** | 6 wavelength regimes (UV, Hα, 24 μm, TIR, radio, combinations) |
| **Dust models explained** | 2 major attenuation laws + variations |
| **High-z studies (z>4)** | 15+ JWST and ALMA results documented |
| **Typical parameter uncertainties documented** | 10+ major parameters |
| **Practice workflows provided** | 2 complete workflows |
| **Troubleshooting entries** | 15+ common issues and solutions |

---

## How to Update This Review

As new papers and surveys emerge, this review can be extended:

1. **New papers:** Add citations to relevant sections (organize by topic)
2. **New facilities:** Update Section 2 (Recent era)
3. **New benchmarks:** Add results to Section 5 (Quantitative results)
4. **New methodologies:** Extend Protocols accordingly
5. **New open problems:** Update Main Review Section 6

**Version History:**
- **v1.0** (December 2025): Initial comprehensive review

---

## Contact and Questions

For questions about specific sections or methodologies, refer to:
- **Conceptual questions:** Main Review, Sections 1–3
- **Quantitative questions:** Benchmarks, all sections
- **Implementation questions:** Protocols, all sections
- **Paper finding:** Main Review, Section 8 (indexed citations)

---

## Legal / Citation Note

These documents synthesize peer-reviewed literature and are intended as reference material for academic research. All findings, formulas, and calibrations credit original authors (see Section 8 of main review for full citations). Use these documents in accordance with academic integrity standards.

---

## Summary Table: Which Document to Use?

| Question | Document | Section |
|----------|----------|---------|
| What are the main concepts? | Main Review | 1 |
| What is the history of this field? | Main Review | 2 |
| What are the SFR formulas? | Benchmarks | 1 |
| What are typical uncertainties? | Benchmarks | 5 |
| How do I measure dust attenuation? | Protocols | 3 |
| Which filters should I use? | Protocols | 1.2 |
| How do I run SED fitting? | Protocols | 2 |
| What code should I choose? | Benchmarks | 7.2 |
| What are recent JWST results? | Main Review | 2, 5 |
| How do I validate my results? | Protocols | 5 |
| I'm stuck, what went wrong? | Protocols | 6, 10 |
| What papers should I read? | Main Review | 8 |

---

**Last Updated:** December 22, 2025
**Total Review Length:** ~20,000 words across 4 documents
**Intended Audience:** Astronomers, astrophysicists, researchers studying galaxy evolution and star formation
