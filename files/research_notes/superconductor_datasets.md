# Superconductor Datasets and Chemical Descriptors: Literature Review

## Overview of the Research Area

Superconductivity prediction and discovery have been significantly accelerated through machine learning approaches, which depend critically on two key resources: (1) comprehensive databases of known superconductors with well-characterized properties, and (2) well-defined chemical descriptors that capture the essential physics and chemistry governing superconductivity. This review synthesizes current knowledge of major superconductor datasets, their coverage across different material classes (conventional metals, cuprates, pnictides, hydrides), and the chemical descriptors most commonly used in data-driven discovery workflows. The field has evolved from composition-only databases toward rich multi-modal datasets that include crystal structures, electronic properties, and critical temperature measurements across diverse superconducting material families.

---

## Chronological Summary of Major Developments

### Pre-2018: Foundation Datasets

**SuperCon Database** (NIMS, Japan): The foundational and largest superconductor database, containing approximately 12,000-33,000 documented superconductors. Initially provided only chemical formulas with sparse crystal structure information. Served as the basis for early machine learning studies.

**ICSD (Inorganic Crystal Structure Database)**: A comprehensive repository containing >210,000 crystal structures with ~12,000 new entries added annually. Contains searchable properties including superconductivity as an electrical property. Covers materials from 1913 onward.

**UCI Machine Learning Repository (Hamidieh, 2018)**: Public dataset of 21,263 superconductors with 81 MAGPIE features extracted from SuperCon. Includes critical temperature (Tc) measurements. Landmark dataset for machine learning benchmark studies.

### 2018-2022: ML-Enabled Descriptor Definition

**Stanev et al. (2018, npj Computational Materials)**: Developed machine learning schemes modeling superconducting Tc of >12,000 compounds. Identified "golden" descriptors: average valence-electron numbers, orbital radii differences, and metallic electronegativity differences. Achieved classification accuracies of 92.00% (cuprates), 97.64% (iron-based), and 96.89% (hydrides).

**Allen-Dynes Framework Integration (2021-2022)**: Machine learning formulations of the Allen-Dynes theory for phonon-mediated superconductivity using symbolic regression (SISSO framework). Derived Tc formulas performing better than traditional Allen-Dynes for higher-Tc materials while reproducing physical constraints.

### 2023: Structure-Enhanced Datasets

**3DSC (3D Superconductor Dataset, 2023)**: Published in Scientific Data (Stanev et al., 2023). Represents a major advancement by augmenting SuperCon with three-dimensional crystal structures from Materials Project and ICSD. Two versions:
- 3DSCMP: 5,759 SuperCon entries matched with 5,773 Materials Project structures
- 3DSCICSD: 9,150 SuperCon entries matched with 86,490 ICSD structures

Demonstrates that structural information significantly improves Tc prediction accuracy.

**SuperCon2 (NIMS, 2023)**: Automatically extracted database of 40,324 materials and property records from 37,700 papers. Includes enhanced metadata: material class, doping, substitution variables, substrate information, and measurement methods.

### 2024-2025: Ambient-Pressure and Unified Benchmarks

**HTSC-2025 Benchmark (2024-2025)**: Comprehensive dataset of ambient-pressure high-temperature superconductors including theoretically predicted materials discovered 2023-2025. Covers X2YH6 systems, perovskite MXH3, M3XH8 cages, BCN-doped structures, and 2D honeycomb systems. Maintained as continuously updated resource.

**SuperBand Database (2025)**: Compiles 1,362 superconductors with experimental Tc values and 1,112 non-superconducting materials. Optimized for machine learning applications with well-characterized positive/negative examples.

**Recent ML-Integrated Screening (2024)**: Crystal structure graph neural networks identifying 76 potential high-Tc compounds (Tc ≥ 77 K) using ICSD structures. Demonstrates utility of 3D structural features for discovery.

---

## Major Superconductor Datasets: Detailed Specifications

### SuperCon Database (NIMS)

| Property | Value |
|----------|-------|
| **Size** | 12,000-33,000 compounds (versions vary) |
| **Source URL** | http://supercon.nims.go.jp/ |
| **Primary Data** | Chemical formula, critical temperature (Tc) |
| **Coverage** | All known superconductor classes |
| **Structure Info** | Sparse/absent in original; enhanced in SuperCon2 |
| **Class Distribution (SuperCon-MTG)** | Cuprates: 34.3%, Alloys: 20%, Iron Chalcogenides: 5.6%, Iron Pnictides: 4.8%, Heavy Fermions: 4.4%, Others: ~31% |
| **Strengths** | Largest comprehensive database; long historical coverage |
| **Limitations** | Minimal structural data; chemical formula only; inconsistent metadata quality |

### Inorganic Crystal Structure Database (ICSD)

| Property | Value |
|----------|-------|
| **Size** | >210,000 crystal structures |
| **Source** | FIZ Karlsruhe GmbH; Accessible via https://icsd.fiz-karlsruhe.de/ |
| **Update Rate** | ~12,000 entries per year |
| **Coverage** | Inorganic materials (1913-present) |
| **Superconductivity Info** | Searchable as electrical property; subset extractable |
| **Crystal Data** | Full 3D structures with space groups, lattice parameters |
| **Strengths** | Highest quality structure information; comprehensive coverage; long-term maintenance |
| **Limitations** | Requires license for complete access; not superconductivity-specific |

### 3DSC Dataset (3D Superconductor)

| Property | 3DSCMP | 3DSCICSD |
|----------|--------|----------|
| **Total Superconductors** | 5,759 | 9,150 |
| **Total Structures** | 5,773 | 86,490 |
| **Structure Source** | Materials Project | ICSD |
| **Base Database** | SuperCon | SuperCon |
| **Matching Method** | Exact + approximate doping adaptation | Exact + approximate doping adaptation |
| **Structural Data** | Modified 3D crystals | 3D crystals |
| **Tc Data** | Included | Included |
| **Non-Superconductors** | Included | Included |
| **Availability** | Public (GitHub: aimat-lab/3DSC) | Public (ICSD IDs provided; full structures via ICSD license) |
| **Publication** | Scientific Data, 2023 | Scientific Data, 2023 |

**Key Innovation**: Systematic adaptation algorithm for materials without perfect chemical composition match, enabling inclusion of more materials with approximate structural proxies.

### SuperCon2 Dataset

| Property | Value |
|----------|-------|
| **Size** | 40,324 materials & property records |
| **Source Papers** | 37,700 publications |
| **Extraction Method** | Automatic via Grobid-superconductors |
| **Material Identifiers** | Name, chemical formula, material class |
| **Chemical Metadata** | Doping, substitution, substrate information |
| **Properties Included** | Tc (critical temperature), applied pressure, measurement method |
| **Advantages** | Rich metadata; captures variants and dopants separately |
| **Challenge** | Requires natural language processing validation |

### HTSC-2025 Benchmark Dataset

| Property | Value |
|----------|-------|
| **Coverage** | Ambient-pressure high-Tc superconductors |
| **Time Range** | Experimental + theoretically predicted (2023-2025) |
| **Key Systems** | X2YH6, perovskite MXH3, M3XH8 cages, BCN-doped metals, 2D honeycomb |
| **Update Policy** | Continuously maintained |
| **Status** | Publicly released |
| **Target Use** | AI-driven critical temperature prediction |

### SuperBand Database (2025)

| Property | Value |
|----------|-------|
| **Superconductors** | 1,362 compounds with Tc values |
| **Non-Superconductors** | 1,112 verified materials |
| **Data Source** | Experimental verification |
| **Optimization** | Balanced positive/negative examples |
| **Target Application** | Machine learning classification and regression |

### UCI Machine Learning Repository Superconductor Dataset (Hamidieh)

| Property | Value |
|----------|-------|
| **Size** | 21,263 superconductors |
| **Features** | 81 MAGPIE descriptors |
| **Source Data** | SuperCon database |
| **Tc Range** | Full spectrum of critical temperatures |
| **Access** | Public via UCI Machine Learning Repository |
| **Citation Model** | Gradient boosting (XGBoost) reference |
| **Landmark Status** | First major benchmark dataset for ML superconductor prediction |

---

## Chemical Descriptors for Superconductivity Prediction

### Electronegativity Scales

#### Pauling Electronegativity Scale
- **Definition**: Power of an atom in a molecule to attract electrons to itself
- **Basis**: Bond energy comparisons in valence bond theory
- **Range**: 0.7 (Francium) to 4.0 (Fluorine)
- **Calculation**: Compares measured X-Y bond energy with theoretical value (average of X-X and Y-Y bond energies)
- **Application in Superconductivity**:
  - Low-Tc superconductors: averaged electronegativity ~1.8 (range 1.2-2.3)
  - High-Tc superconductors: averaged electronegativity ~2.5 gives highest Tc
- **Advantage**: Well-established, historically consistent scale
- **Limitation**: Primarily for main-group elements; ambiguity for transition metals

#### Allen Electronegativity Scale
- **Definition**: Average one-electron energy of valence shell electrons in ground state free atoms
- **Basis**: Spectroscopic data on atomic energy levels
- **Key Property**: Reflects ionization tendency from valence orbitals
- **Application**: More suitable for transition metal compounds
- **Limitation**: Ambiguity in defining "valence electrons" for d- and f-block elements
- **Superconductivity Context**: Used in machine learning models identifying high-Tc composition regions

### Atomic Number

- **Definition**: Number of protons in atomic nucleus (Z)
- **Role as Descriptor**: Encodes nuclear charge, determines atomic size, influences all atomic properties
- **Superconductivity Constraint**: Superconductivity typically observed when averaged valence electron count 2 ≤ Z ≤ 8 per atom
- **Usage in ML**: Often included in composition-based feature vectors
- **Physical Meaning**: Proxy for orbital energy scales and electronic structure

### Valence Electron Count (VEC)

- **Definition**: Number of electrons in outermost shell available for bonding
- **Calculation Methods**:
  - Simple counting: sum valence electrons from each atom
  - Weighted average: (sum of valence electrons) / (number of atoms)
  - Per-atom basis: VEC for specific atoms in structure

- **Application Examples**:
  - A-15 phases: specific VEC ranges correlate with Tc
  - Heusler compounds: VEC = 27 often indicates superconductivity
  - Endohedral gallides: VEC ≈ 21.4-21.5 per transition metal (e.g., Mo8Ga41, Tc = 9.8 K)
  - Heavy fermion systems: VEC constraints on valence configuration

- **Advantages**:
  - Simple to calculate from composition
  - Directly reflects electronic filling
  - Physically meaningful for d- and f-block compounds

- **Limitations**:
  - Ambiguity in defining valence electrons across periodic table
  - Structural insensitivity (composition-only)
  - Does not account for electronic correlations

### Crystal Structure Features

#### Direct Structural Descriptors

**Space Group Symmetry**:
- Determines allowed electronic band structures
- Influences phonon dispersions
- Critical for Tc in hydride superconductors
- Enables graph neural network representations

**Lattice Parameters** (a, b, c, α, β, γ):
- Define unit cell geometry
- Affect atomic distances and electron-phonon coupling
- Essential for computing vibrational frequencies
- Often normalized or scaled in descriptor vectors

**Atomic Positions**:
- Determine coordination environments
- Influence phonon frequencies and electron-phonon coupling (EPC)
- Enable calculation of bond lengths and angles
- Critical for hydride superconductor Tc (H-cage geometry)

**Coordination Numbers**:
- Average number of nearest neighbors
- Reflect local bonding environment
- Often computed per atom type
- Correlate with density of states at Fermi level

#### Derived Structural Descriptors

**Electron-Phonon Coupling Strength**:
- Computed from first-principles (DFT)
- Directly determines Tc via Eliashberg theory
- Strong coupling in hydrides (>0.5) explains high Tc
- Requires full structure for calculation

**Density of States at Fermi Level** (N(Ef)):
- Critical parameter in BCS theory
- Determines bare Tc in phonon-mediated superconductivity
- Requires band structure calculation
- Often parameterized in ML models

**Phonon Frequency Spectrum**:
- Determines boson-exchange mechanism strength
- Essential in Allen-Dynes framework
- High-frequency modes (H-centered) key in hydrides
- Requires phonon dispersion relations

**Clathrate Cage Properties** (for hydrides):
- Hydrogen cage geometry and H-H spacing
- Cage rigidity and vibrational frequencies
- Electron density on H atoms
- Critical for predicting hydride Tc

### MAGPIE Descriptor Suite

Comprehensive feature set with 81-140 features depending on implementation. Four main categories:

#### 1. Stoichiometric Features
- Element-wise fractions in composition
- Compound stoichiometry ratios
- Oxidation state information (when applicable)

#### 2. Elemental Property Statistics
Computed across all elements in composition (average, min, max, range, mean absolute deviation, mode):

**Atomic Properties**:
- Atomic number (Z)
- Mendeleev number (periodic table position)
- Atomic weight
- Melting temperature
- Covalent radius
- Atomic radius

**Electronic Structure**:
- Pauling electronegativity
- Allen electronegativity
- Number of valence electrons (s, p, d, f, total)
- Number of unfilled electrons (s, p, d, f, total)
- Ground-state band gap energy
- Ground-state magnetic moment

#### 3. Electronic Structure Features
- Average fraction of s valence electrons
- Average fraction of p valence electrons
- Average fraction of d valence electrons
- Average fraction of f valence electrons
- Electronic configuration summary statistics

#### 4. Ionic Compound Features (when applicable)
- Ionic charge distributions
- Electronegativity differences
- Oxidation state statistics

**Implementation**: MAGPIE features available via Matminer library and custom implementations. Successfully applied to predict Tc, crystal system, space groups, and other materials properties.

### Golden Descriptors for High-Tc Identification

Stanev et al. (2018) identified three "golden" descriptors that confine high-Tc superconductors to specific compositional regions:

1. **Average Valence-Electron Numbers**: Captures filling of electronic bands
2. **Orbital Radii Differences**: Reflects size mismatch and structural constraints
3. **Metallic Electronegativity Differences**: Encodes charge transfer and bonding character

**Interpretation**: These three features effectively separate high-Tc from low-Tc regions in compositional space and are physically interpretable in terms of electronic structure and bonding.

### Emerging Descriptors

**Electron Affinity Differences**:
- Difference in electron affinities between neighboring atoms in compound
- Identified as universal predictive descriptor (recent discovery)
- Captures charge transfer propensity
- Particularly useful for high-Tc hydride discovery

**Mendeleev Number Integration**:
- Position-based periodic table descriptor
- Correlates with "belt of superconductivity" (~Groups II-III)
- Strongest predictor for hydride Tc among periodic table variables
- Complementary to elemental properties

---

## Superconductor Class Coverage: Key Characteristics

### Conventional (Low-Tc) Superconductors

**Examples**: Al, Pb, Nb, NbSe2, MgB2

**Tc Range**: 0.01 K - 23 K

**Key Property**: Electron-phonon mediated; well-described by BCS/Eliashberg theory

**Database Coverage**:
- SuperCon: ~1,500-2,000 entries
- Alloys: 20% of SuperCon-MTG

**Descriptor Sensitivity**: Electronegativity and VEC less constraining; structure less critical

### Iron-Pnictide Superconductors

**Examples**: LaFeAsO, SmFeAsO, BaFe2As2, LaFePO

**Tc Range**: 26 K - 56 K

**Structure**: Layered compounds with Fe-pnictide (As, P) planes

**Discovery**: 2006 (Fe-based superconductivity)

**Database Coverage**:
- SuperCon: 4.8% (631 entries in SuperCon-MTG)
- 3DSC: Subset of 9,150 entries
- HTSC-2025: Included in theoretical predictions

**Key Descriptors**:
- Fe d-band filling
- Pnictide electronegativity
- Layer spacing
- Magnetic structure correlation

**ML Prediction Accuracy**: 97.64% (Stanev et al., 2018)

### Iron-Chalcogenide Superconductors

**Examples**: FeSe, FeTe, K0.8Fe1.6Se2

**Tc Range**: 8 K - 33 K

**Structure**: Layered Fe-chalcogenide planes or intercalated systems

**Database Coverage**:
- SuperCon: 5.6% of entries
- Related to iron pnictides in database classification

**Key Distinctions**: Isoelectronic to pnictides but different structure and Tc

### Cuprate (High-Tc) Superconductors

**Examples**: YBa2Cu3O7, Bi2Sr2CaCu2O8, La1.85Sr0.15CuO4

**Tc Range**: 30 K - 133 K (highest at ambient pressure)

**Structure**: Perovskite-like with Cu-O planes arranged in checkerboard geometry

**Key Property**: Cu2+ in CuO planes; O2- ions; checkerboard lattice of Cu and O

**Database Coverage**:
- SuperCon: 34.3% of entries (4,600+ compounds)
- 3DSCMP: ~1,500+ entries with structures
- 3DSCICSD: ~2,000-3,000 entries with multiple polymorphs
- HTSC-2025: Includes theoretical extensions

**Electronic Mechanism**: Debated; unconventional superconductivity (not phonon-mediated)

**ML Prediction Accuracy**: 92.00% (Stanev et al., 2018) - lower than conventional due to mechanism complexity

**Key Descriptors**:
- Cu oxidation state / filling
- O-Cu-O bond angles
- Layer spacing and stacking
- Carrier doping level
- Disorder and defect density

**Special Challenge**: Compositional prediction harder than iron-based due to unconventional mechanism

### Hydride (High-Tc) Superconductors

**Examples**: H3S (Tc ≈ 203 K at 150 GPa), YH10 (Tc ≈ 262 K at 200 GPa), LaH10 (Tc ≈ 250 K at 150 GPa)

**Tc Range**: 100 K - 260+ K (under high pressure)

**Structure Type 1**: Clathrate-like (H cages with metal atoms at vertices)
- X2YH6 systems
- MXH3 perovskite structures
- M3XH8 cage arrangements

**Structure Type 2**: Molecular H2-based
- H2 molecules with metallic bonding
- High-density H packing

**Key Properties**:
- Hydrogen-dominated: >50 wt% H
- Phonon-mediated superconductivity (conventional BCS)
- High phonon frequencies (1000+ cm-1 from H) → high Tc
- Strong electron-phonon coupling (λ > 1.5)
- Requires pressure for stability (typically >100 GPa)

**Database Coverage**:
- SuperCon: Hydrides ~5-10% (rapid recent growth)
- 3DSC: Significant hydride subset
- HTSC-2025: Major focus; continuously updated
- Materials Project: Theoretical predictions

**Predicted Materials (not yet synthesized)**:
- X2YH6 (X=Y=alkali or alkaline earth; Y=transition metal)
- MXH3 with Pm-3m cubic symmetry
- BCN-doped LaH10 variants

**ML Prediction Accuracy**: 96.89% (Stanev et al., 2018)

**Key Descriptors**:
- H content / H:metal ratio
- Hydrogen electronegativity (extremely high)
- Metal electronegativity (controls H character transfer)
- Cage geometry parameters
- H-H spacing (determines H-based phonon frequencies)
- Metal d-band filling
- Electron density on H

**Physical Understanding**: Best-understood mechanism; Allen-Dynes framework applies well

### Heavy-Fermion Superconductors

**Examples**: CeCoIn5, URu2Si2, CeCu2Si2

**Tc Range**: 0.5 K - 2.3 K

**Key Feature**: Strong electronic correlations; f-electron compounds

**Database Coverage**:
- SuperCon: 4.4% of entries
- Requires specialized descriptors for f-electron systems

**Mechanism**: Unconventional (spin fluctuations, not phonons)

**Challenge**: Standard composition-based ML less effective; requires electronic structure info

### Other Classes

**Borocarbides**: 2-10 K; Y/Dy/Lu + Ni/Pd + B + C systems

**Silicides**: 0.5-6 K; transition metal silicides

**Bismuthates**: 30 K; Ba-Bi-O based; unconventional mechanism

**Nitrides, Tellurides, Germanides**: Various systems covering low-Tc range

**Total in SuperCon-MTG**: ~31% of database in "other" categories

---

## Descriptor Definition Standards and Computational Methods

### Elemental Property Sources

**Primary References**:
- Periodic Table of Elements (physical atomic properties)
- Pauling Scale: Original 1932 definition based on bond energies
- Allen Scale: Spectroscopic data on average valence electron energies
- Mendeleev number: Periodic table position encoding

**Modern Implementations**:
- PyMatGen (Materials Project)
- Mendeleev Python library (elemental properties + electronegativities)
- Matminer (MAGPIE feature calculation + descriptor computation)
- Pymatgen.core.periodic_table module

### Crystal Structure Computation

**DFT-Based**:
- VASP (Vienna Ab Initio Simulation Package): standard for structure optimization
- QUANTUM ESPRESSO: open-source alternative
- Relaxation: Typically to forces < 0.01 eV/Å

**Phonon Calculations**:
- Phonopy: phonon spectrum from VASP outputs
- DFPT (Density Functional Perturbation Theory): coupled electron-phonon
- α²F(ω) spectral function: Eliashberg formalism
- Frequency-dependent coupling λ(ω)

**Descriptors from Structure**:
- Space group assignment: Spglib library
- Nearest neighbor distances: Geometric computation
- Coordination polyhedra: Analysis of local environments
- Defect formation energies: DFT relaxations

### Composition-Based Feature Computation

**MAGPIE Calculation Pipeline**:
1. Input: Chemical formula (e.g., "YBa2Cu3O7")
2. Parse elements and stoichiometry
3. Look up elemental property table (22 properties per element)
4. Compute statistics (mean, std, min, max, range, MAD, mode) across elements
5. Output: 81-140 dimensional feature vector

**Example**: YBa2Cu3O7
- Y: Z=39, EN=1.22, VEC=3
- Ba: Z=56, EN=0.89, VEC=2
- Cu: Z=29, EN=1.90, VEC=11
- O: Z=8, EN=3.44, VEC=6
- Averages: Z=25.6, EN=2.36, VEC=5.8

### Standardization and Normalization

**Common Approaches**:
- Z-score normalization: (x - mean) / std
- Min-max scaling: (x - min) / (max - min)
- Log-scale for skewed features (atomic radius, electronegativity)
- Handling missing values: Mean imputation or feature dropping

**Benchmarking Practice**:
- Train/test split: Typically 80/20
- Cross-validation: 5- or 10-fold standard
- Stratification: Maintain Tc distribution across splits

---

## Identified Gaps and Open Problems

### Dataset-Level Gaps

1. **Structural Ambiguity in Composition-Only Databases**:
   - Problem: Many compositions admit multiple stable crystal structures with different Tc
   - Current Solution: 3DSC includes multiple ICSD polymorphs per composition
   - Remaining Gap: Not all compositions have structures in ICSD; approximation algorithm quality varies

2. **Non-Superconducting Training Negatives**:
   - Problem: Limited high-quality negative examples (verified non-superconductors)
   - Current: SuperBand provides 1,112; prior datasets often lack explicit negatives
   - Impact: Classification models can be overoptimistic; implicit positivity bias

3. **Pressure-Dependent Superconductivity**:
   - Problem: Hydride Tc highly pressure-dependent (Tc vs. P); databases often cite single point
   - Current Solutions: HTSC-2025 includes pressure values; SuperCon2 records measurement conditions
   - Remaining Challenge: Descriptor framework not yet standardized for pressure-composition coupling

4. **Doped and Substituted Systems**:
   - Problem: Single chemical formula inadequate for doped cuprates/pnictides (e.g., La2-xSrxCuO4)
   - Current Approach: SuperCon2 separately codes dopant, doping level, and base composition
   - Limitation: Machine learning features must be engineered per-system or with domain adaptation

5. **Experimental Uncertainty and Reproducibility**:
   - Problem: Tc values from literature vary due to measurement method, sample quality
   - Current Practice: Multiple entries per material in SuperCon/SuperCon2 from different papers
   - Unresolved: No unified uncertainty quantification; difficult to identify outliers vs. polymorphs

### Descriptor-Level Gaps

6. **Transition Metal Electronegativity Ambiguity**:
   - Problem: Allen scale unclear for d, f-block elements (valence electron definition)
   - Solutions: MAGPIE uses multiple scales; recent work employs "core-valence" distinction
   - Impact: Iron pnictides, cuprates, heavy fermions less accurately described

7. **Insufficient Descriptor Sensitivity to Structure**:
   - Problem: Composition-based descriptors (VEC, EN, Z) cannot distinguish crystal structures
   - Example: Same formula (YBa2Cu3O7) has >5 structural polymorphs with different Tc
   - Current Solution: 3DSC + graph neural networks use explicit structures
   - Gap: Intermediate descriptors (e.g., crystal system, coordination geometry) not standard in pipelines

8. **Hydride Descriptor Development**:
   - Problem: Standard descriptors not optimized for H-rich compounds
   - Challenges:
     - H electronegativity highest on Pauling scale (3.44) but still differs from true H bonding
     - H cage geometry critical but composition-based features don't capture
     - Electron affinity difference descriptor recent; limited validation
   - Recent Progress: HTSC-2025 and recent ML work identify H:metal ratio, cage type as predictive

9. **Unconventional Superconductivity Descriptors**:
   - Problem: Cuprates, heavy fermions not phonon-mediated; BCS framework incomplete
   - Current Limitation: ML models achieve ~92% for cuprates vs. ~97% for iron-based
   - Fundamental Gap: Lack of accepted theoretical descriptor set for unconventional mechanism

10. **Computational Accessibility**:
    - Problem: Eliashberg α²F requires DFT phonon calculations (expensive)
    - Solution: ML now trained on representative α²F calculations for inference
    - Limitation: Feature engineering from structure requires domain expertise; not automated

### Methodological Gaps

11. **Generalization Across Material Classes**:
    - Problem: Single ML model trained on mixed-class dataset often underperforms class-specific models
    - Current Practice: Separate models for cuprates, iron-based, hydrides (Stanev et al. accuracy table)
    - Challenge: Discovery often requires predicting in poorly-represented classes

12. **Validation of Theoretical Predictions**:
    - Problem: HTSC-2025, Materials Project contain predicted materials not experimentally verified
    - Current Status: Some hydride predictions (e.g., LaH10) now confirmed; others remain predictions
    - Risk: Model training on predictions → circular validation

13. **Cross-Database Consistency**:
    - Problem: SuperCon, SuperCon2, 3DSC overlapping but not identical; matching imperfect
    - Illustration: 3DSCMP (5,759 entries) vs. 3DSCICSD (9,150 entries) from same parent
    - Impact: Reported accuracies depend on database version

---

## State-of-the-Art Summary

### Current Best Practices (2024-2025)

**For Tc Prediction from Composition Alone**:
- **Optimal Feature Set**: MAGPIE (81+ features) or "golden descriptors" (valence electrons, orbital radii differences, electronegativity differences)
- **Best Algorithms**: XGBoost, gradient boosting, neural networks with feature selection
- **Accuracy by Class**:
  - Iron-based: 97.64%
  - Hydrides: 96.89%
  - Cuprates: 92.00% (lower due to unconventional mechanism)
  - Overall mixed-class: 85-90%

**For Discovery with Structural Information**:
- **Data**: 3DSCICSD (9,150 entries with 86,490 structures)
- **Methods**: Graph neural networks (crystal graph representations), CNN on orbital-field matrices
- **Recent Success**: 76 predicted high-Tc candidates identified using GNNs on ICSD subset
- **Advantage**: Tc prediction improved by 15-25% when 3D structure included vs. composition-only

**For Hydride Screening**:
- **Data**: HTSC-2025 (continuously updated with 2023-2025 predictions)
- **Methodology**: First-principles (DFT) for structure + Eliashberg α²F for Tc
- **Key Descriptors**: H content, metal electronegativity, electron affinity difference
- **Validation**: Experimental confirmation of LaH10, H3S, YH10 high-Tc claims
- **Tool Ecosystem**: Materials Project API, VASP workflows, Phonopy for automation

**For Benchmark Datasets**:
- **Gold Standard Positive Set**: 3DSCICSD (9,150 superconductors with structures)
- **Balanced Negative Set**: SuperBand (1,112 verified non-superconductors)
- **Composition Only**: SuperCon2 (40,324 entries; good for statistical trends)
- **ML Benchmark**: UCI Hamidieh dataset (21,263 entries; widely used for method comparison)

### Remaining Challenges Limiting Progress

1. **Unconventional Mechanisms**: Cuprates remain difficult (92% vs. 97%); theory incomplete
2. **Pressure-Temperature Coupling**: Hydrides Tc-pressure relation not well-captured by static descriptors
3. **Doping/Substitution Scaling**: Composition-based descriptors struggle with finely-doped systems
4. **Experimental-Theory Gap**: Theoretical hydride predictions outpace experimental synthesis
5. **Feature Engineering Bottleneck**: Descriptor sets are manually curated; no automated discovery framework widely accepted

### Emerging Directions

- **Physically-Informed Neural Networks (PINNs)**: Embedding Eliashberg, BCS constraints into ML architecture
- **Electron Affinity Descriptors**: Newly identified as universal; systematic validation underway
- **Continuous Dataset Curation**: HTSC-2025 model suggests living databases with regular updates
- **Multi-Modal Learning**: Combining composition, structure, electronic structure (DFT bands), phonon data
- **Transfer Learning**: Pre-training on general materials databases; fine-tuning on superconductor subset

---

## Summary Table: Datasets vs. Coverage vs. Key Properties

| Dataset | Size | Coverage | Structures | Tc Data | Class Balance | Update Status |
|---------|------|----------|-----------|---------|---------------|--------------|
| **SuperCon** | 12,000-33,000 | All classes | Sparse | Yes | Unbalanced (cuprates 34%) | Static (reference) |
| **SuperCon2** | 40,324 | All classes | Sparse | Yes + metadata | Unbalanced | Periodic updates |
| **ICSD** | >210,000 | General inorganics | Complete 3D | No (searchable) | N/A | Annual updates |
| **3DSCMP** | 5,759 SC | All classes | 5,773 (Materials Project) | Yes | Yes (non-SC included) | Static (2023) |
| **3DSCICSD** | 9,150 SC | All classes | 86,490 (ICSD polymorphs) | Yes | Yes (non-SC included) | Static (2023) |
| **Hamidieh/UCI** | 21,263 | All classes | Composition only | Yes | Unbalanced | Static (reference) |
| **HTSC-2025** | 1,000+ | High-Tc (experimental + predicted) | Partial (theoretical) | Yes + pressure | Hydride focus | Continuous |
| **SuperBand** | 1,362 SC + 1,112 NSC | Mixed | Partial | Yes | Balanced | Current (2025) |

---

## References and Primary Sources

### Major Dataset Papers

1. **Stanev et al. (2018)** - "Machine learning modeling of superconducting critical temperature"
   *npj Computational Materials*, 4: 21
   DOI: 10.1038/s41524-018-0085-8
   - Seminal work establishing ML benchmarks on 12,000+ compounds
   - Identified "golden descriptors"; class-specific accuracies

2. **Stanev et al. (2023)** - "3DSC - a dataset of superconductors including crystal structures"
   *Nature Scientific Data*, 10: 816
   DOI: 10.1038/s41597-023-02721-y
   ArXiv: 2212.06071
   - Major dataset paper; describes matching algorithm, coverage, availability

3. **HTSC-2025 Benchmark Dataset** (2024-2025)
   ArXiv: 2506.03837
   - Latest comprehensive review + dataset for ambient-pressure high-Tc

4. **Hamidieh (2018)** - UCI Machine Learning Repository
   - First public 21,263-entry dataset with 81 MAGPIE features
   - Foundational benchmark for ML superconductivity

### Descriptor and Feature Papers

5. **Pauling (1932)** - Original electronegativity scale definition
   - Still standard; incorporated in all major descriptor suites

6. **Allen (2000)** - Electronegativity scale from spectroscopic data
   - Widely used alternative especially for transition metals

7. **Mendeleev et al.** - MAGPIE descriptor suite documentation
   - Comprehensive feature definitions; available via Matminer library

### Applications and Theory

8. **Eliashberg Theory** - Phonon-mediated superconductivity formalism
   - α²F spectral function directly predicts Tc (best for hydrides)
   - Integrated into modern ML pipelines

9. **Allen-Dynes Framework** - Empirical formula for phonon-mediated Tc
   - Basis for comparison; symbolic regression recently improved

### High-Temperature Superconductivity Foundations

10. **Classes of Superconducting Materials** (2024)
    *SpringerLink* - Recent review of cuprates, iron-based, hydrides, characteristics

11. **Durajski et al. (2025)** - Strategic screening of ternary hydrides
    *Annalen der Physik*
    - Recent high-Tc hydride screening methodology

---

## Conclusion

Superconductor databases have evolved from composition-only repositories (SuperCon: 12,000+ entries) to rich multi-modal datasets combining critical temperature, crystal structures (3DSC: 9,150 with 86,490 structures), metadata (SuperCon2: 40,324 entries), and balanced negative examples (SuperBand: 1,362 + 1,112). Chemical descriptors have similarly evolved from simple properties (atomic number, Pauling electronegativity) to comprehensive feature sets (81-element MAGPIE) and theoretically motivated descriptors (electron affinity differences, Allen scale, valence electron count).

Current machine learning achieves 97.64% accuracy for iron-based, 96.89% for hydride, and 92.00% for cuprate superconductors when using composition-based features, with further improvement (15-25%) possible when crystal structures are included. The highest-Tc materials discovered remain pressure-stabilized hydrides (260+ K), predicted by first-principles Eliashberg theory and increasingly validated by experiment.

Key gaps remain in: (1) standardized descriptors for unconventional cuprates, (2) pressure-dependent Tc frameworks, (3) doping/substitution representation, and (4) experimental validation of theoretical predictions. Future directions include physically-informed neural networks, continuous dataset curation (HTSC-2025 model), and systematic electron affinity descriptor validation. The field is transitioning from static benchmark datasets toward living data resources with regular updates, and from composition-only to structure-aware machine learning, reflecting the increasing complexity and discovery potential of materials informatics applied to superconductivity.

