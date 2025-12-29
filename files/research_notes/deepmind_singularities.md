# Literature Review: Google DeepMind's 2025 Work on Unstable Singularities in Fluid Dynamics

**Date Compiled:** December 2025
**Focus:** Google DeepMind's systematic discovery of unstable singularity families in 3D incompressible Navier-Stokes and related fluid PDEs

---

## 1. Overview of the Research Area

### Problem Statement

The existence of finite-time blow-up (singularity formation) in 3D incompressible Navier-Stokes equations is one of the six Clay Millennium Prize Problems (unsolved). A foundational conjecture states that if singularities exist in the boundary-free 3D Euler and Navier-Stokes equations, they must be unstable; stable singularities are hypothesized to be absent or structurally forbidden in these systems.

**Key Distinction:**
- **Stable singularity:** Forms robustly; persists even under small perturbations to initial conditions
- **Unstable singularity:** Requires initial conditions tuned with infinite precision; infinitesimal perturbations divert the solution from the blow-up trajectory

Prior to 2025, no unstable singularities had been discovered numerically for incompressible fluid equations.

### Significance

Understanding unstable singularities is critical for:
1. Resolving the Navier-Stokes Millennium Prize Problem
2. Determining whether finite-time blow-up is possible in bounded and unbounded domains
3. Characterizing the structure of the solution manifold near singularities

---

## 2. Chronological Summary of Major Developments

### September 18, 2025: DeepMind Publication

**Title:** "Discovery of Unstable Singularities" (arXiv:2509.14185)

**Institutions Involved:**
- Google DeepMind (lead organization)
- Brown University (mathematics)
- New York University (mathematics)
- Stanford University (geophysics/applied mathematics)

**Key Milestone:** First systematic discovery of new families of unstable singularities across multiple fluid PDE systems using Physics-Informed Neural Networks (PINNs).

---

## 3. Detailed Research Methodology

### 3.1 Core Computational Framework

**Approach:** Physics-Informed Neural Networks (PINNs) with mathematical embedding

**Architecture Design:**
- Neural networks trained to satisfy known PDE constraints
- Direct embedding of mathematical symmetries, boundary conditions, and asymptotic behaviors into network architecture
- Strong inductive biases reduce the hypothesis space and guide optimization

**Rationale:** Unlike conventional neural networks trained on large datasets, PINNs enforce physical laws through the loss function. The output is continuously checked against PDE requirements.

### 3.2 Optimization Strategy

**Optimizer:** Full-matrix Gauss-Newton method (second-order)

**Advantages over first-order optimizers:**
- Convergence to solutions with residuals at or near double-precision machine epsilon
- Enables unprecedented precision required for computer-assisted mathematical proofs
- Overcomes vanishing gradient issues in highly nonlinear PDE regimes

**Training Procedure:**
- Curated neural network architectures specific to each equation
- Custom training schemes combining global and local optimization
- Achieved accuracies equivalent to predicting Earth's diameter to within a few centimeters

### 3.3 Key Mathematical Embedding Techniques

**Self-Similar Solutions:** Solutions represented as \( u(t,x) = (T-t)^{\alpha} U(\eta) \) where \(\eta = x/(T-t)^{\beta}\) and \(\alpha, \beta\) are self-similar exponents.

**Symmetry Constraints:** Embedding axisymmetry, swirl properties, and other geometric structures directly into network weight matrices.

**Boundary Conditions:** Incorporated into network architecture rather than as penalty terms in loss function.

**Regularity Constraints:** Smoothness and monotonicity properties of solutions encoded as network parameterizations.

---

## 4. Singularities Identified and Studied

### 4.1 Three Target Fluid Systems

#### (1) Córdoba-Córdoba-Fontelos (CCF) Equation
- **Form:** A 1D nonlocal transport equation: \(\partial_t u + \mathcal{H}(u^2)_x = 0\)
- **Significance:** Simplified model for 3D Euler and Navier-Stokes singularity mechanisms
- **Results:**
  - Discovered stable and multiple unstable self-similar solutions
  - Achieved near double-float machine precision (residual ~10⁻¹³)
  - First systematic discovery of unstable solutions for CCF

#### (2) Incompressible Porous Media (IPM) Equation
- **Form:** 2D generalized porous medium equation with nonlocal advection
- **Physical context:** Related to geophysical flows and nonlocal fluid models
- **Results:**
  - Multiple new unstable self-similar solutions discovered
  - Equation residuals: ~10⁻⁸ to 10⁻⁷
  - Clear linear pattern between blow-up rate and instability order

#### (3) Boussinesq Equation
- **Form:** 2D Boussinesq equations with buoyancy coupling
- **Context:** Model for density-stratified incompressible flow
- **Results:**
  - Systematic discovery of unstable singularities
  - Residual accuracy: ~10⁻⁸ to 10⁻⁷
  - Reveals empirical asymptotic formula relating blow-up rate to instability order

### 4.2 Singularity Families Discovered

**Structure:** For each equation, multiple families parameterized by "order of instability" \(n\):
- \(n=0\): Stable branch
- \(n=1,2,3,...\): First, second, third unstable branches, etc.

**Self-Similarity:** All discovered solutions exhibit self-similar blow-up: \( \|u(t)\|_{\infty} \sim (T-t)^{-\alpha} \) as \( t \to T^- \)

---

## 5. Geometric Structures

### 5.1 Axisymmetric Vortex Filaments

**Relevance:** Axisymmetric flows with concentrated vorticity are central to understanding 3D Euler and Navier-Stokes singularities.

**Characteristics in Unstable Solutions:**
- Vortex filaments exhibit specific radial and azimuthal structure
- Concentrated vorticity regions (antiparallel vortex configurations)
- Geometrical structures preserved by self-similarity

### 5.2 Swirl Symmetry

**Definition:** Azimuthal velocity component proportional to swirl strength parameter.

**Role in Instability:**
- Swirl amplifies nonlinear effects in Euler equations
- Localized instabilities of vortex rings with swirl drive blow-up
- Rayleigh centrifugal instability associated with swirl profiles

**Embedding:** DeepMind embedded swirl symmetry directly into neural network parameterizations to enforce geometric realism.

### 5.3 Self-Similar Scaling

**Radial-Azimuthal Coupling:** Self-similar profiles couple radial velocity \(u_r\), azimuthal velocity \(u_\theta\), and pressure through nonlocal interactions.

**Similarity Variables:** Solutions depend only on similarity coordinate \(\eta = r/(T-t)^{\beta}\), reducing PDEs to ODEs or simpler systems.

---

## 6. Stability Analysis and Order of Instability

### 6.1 Definition of Instability Order

**Order of Instability \(n\):** The number of unstable directions in the linearization about the self-similar solution.

**Mathematical Characterization:**
- Eigenvalue analysis of the PDE linearization about self-similar profiles
- Count of positive-real-part eigenvalues (unstable modes)
- \(n\)-th order unstable solution: \(n\) independent unstable eigenmodes

### 6.2 Stability Structure

**Stable Branch (\(n=0\)):**
- All eigenvalues of linearization have negative real parts (in appropriate Hilbert space)
- Solution is linearly stable to all perturbations
- Attracting manifold in the PDE flow near the singularity

**Unstable Branches (\(n \geq 1\)):**
- Exactly \(n\) eigenvalues with positive real part (codimension-\(n\) manifold)
- Solution lies at the intersection of \(n\) unstable manifolds and a stable manifold
- Fine-tuned initial conditions required for blow-up: measure-zero set in initial condition space

### 6.3 Blow-Up Rate Parameter \(\lambda\)

**Definition:** Scaling exponent in self-similar ansatz; characterizes speed of singularity formation.

**Self-Similar Form:** \( u(t,x) = (T-t)^{-\lambda} U\left(\frac{x}{(T-t)^{\mu}}\right) \)

**Empirical Finding:** Linear relationship between inverse scaling rate and instability order:
\[ \lambda^{-1} \propto \text{(order of instability)} + \text{constant} \]

This linear pattern was observed in IPM, Boussinesq, and CCF systems, suggesting a universal principle governing unstable singularity families.

### 6.4 Quantitative Blow-Up Rate Results

**Córdoba-Córdoba-Fontelos:**
- Stable solution: \(\lambda_0 = \) [specific value from numerical solution]
- 1st unstable: \(\lambda_1 = \) [higher than stable]
- Pattern: as instability order increases, blow-up rate increases

**Incompressible Porous Media:**
- Multiple unstable branches discovered with distinct \(\lambda\) values
- Clear linear trend in \(\lambda\) vs. instability order
- Residual error: 10⁻⁸ to 10⁻⁷ (suitable for mathematical validation)

**Boussinesq:**
- Similar linear pattern to IPM
- Slightly lower precision achieved (10⁻⁷ to 10⁻⁸ residuals)

---

## 7. Numerical Evidence and Results

### 7.1 Precision Metrics

**Equation Residuals:**
- CCF stable/1st unstable: **~10⁻¹³** (near double-float machine precision)
- IPM branches: **10⁻⁸ to 10⁻⁷**
- Boussinesq branches: **10⁻⁸ to 10⁻⁷**

**Error Magnitude:** Equivalent to predicting Earth's diameter to within a few centimeters globally.

**Computational Precision:** Constrained only by GPU round-off errors (float64), meeting stringent requirements for computer-assisted proofs.

### 7.2 Solution Families

| Equation | System Type | Stable Solutions | Unstable Orders Discovered | Highest Residual |
|----------|-------------|------------------|---------------------------|------------------|
| CCF      | 1D nonlocal | 1 (exact)        | n=1, n=2, ...             | ~10⁻¹³           |
| IPM      | 2D nonlocal | 1                | Multiple (n≥1)            | ~10⁻⁷            |
| Boussinesq | 2D coupled | 1                | Multiple (n≥1)            | ~10⁻⁷            |

### 7.3 Linear Pattern Discovery

**Key Empirical Finding:**

As the order of instability \(n\) increases, the blow-up rate parameter \(\lambda(n)\) follows a linear pattern:
\[ \lambda(n) \approx \lambda_0 + c \cdot n \]

where \(\lambda_0\) is the stable rate and \(c\) is a model-dependent constant.

**Implications:**
- Suggests universal organizing principle for unstable singularities
- Provides predictive formula for higher-order unstable branches
- May generalize to 3D Euler and Navier-Stokes with boundary

### 7.4 Numerical Validation Procedure

1. **Solution Representation:** Neural network parameterization \( \hat{u}_\theta(x,t) \) trained via PINNs
2. **Residual Evaluation:** Compute \( \| \mathcal{L}[\hat{u}_\theta] \|_{\infty} \) where \(\mathcal{L}\) is the PDE operator
3. **Refinement:** Apply Gauss-Newton optimizer until convergence
4. **Verification:** Confirm self-similarity and eigenvalue structure numerically

---

## 8. Key Findings Summary

### 8.1 Fundamental Discoveries

1. **First systematic discovery** of unstable singularities for incompressible fluid equations
2. **Multiple families** organized by instability order
3. **Unprecedented numerical precision** suitable for rigorous computer-assisted proofs
4. **Universal linear pattern** relating blow-up rate to instability order

### 8.2 Technical Achievements

1. **PINN Precision:** Extended PINNs from typical 10⁻⁴ to 10⁻⁷ or 10⁻¹³ accuracy
2. **Gauss-Newton Optimization:** Full-matrix second-order method enables convergence to machine epsilon
3. **Mathematical Embedding:** Direct encoding of symmetries, boundary conditions, and regularity into architecture
4. **Multi-Scale Structure:** Resolved both global singularity profiles and local fine structure

### 8.3 Validation and Rigor

- Precision achieved meets requirements for computer-assisted proofs
- Stability properties (eigenvalue structure) numerically verified
- Solutions reproducible across independent implementations
- Residual accuracies sufficient for interval arithmetic validation

---

## 9. Specific Quantitative Results

### 9.1 CCF Equation Results

**Stable Branch:**
- Self-similar exponent \(\alpha_0\)
- Equation residual: ~10⁻¹³
- Linearization: all stable eigenvalues (confirmed numerically)

**First Unstable Branch:**
- Self-similar exponent \(\alpha_1 > \alpha_0\)
- Equation residual: ~10⁻¹³
- Single unstable eigenvalue
- Blow-up rate \(\lambda_1\) higher than stable

**Higher Orders:** Discovered up to \(n=2,3,\ldots\) with trend \(\lambda(n) > \lambda(n-1)\)

### 9.2 IPM and Boussinesq Results

**Residual Accuracy:**
- CCF: log₁₀(max residual) ≈ -13.0
- IPM: log₁₀(max residual) ≈ -7.0 to -8.0
- Boussinesq: log₁₀(max residual) ≈ -7.0 to -8.0

**Pattern Identification:**
- IPM shows clear linear trend: multiple unstable branches discovered
- Boussinesq exhibits similar pattern
- Both confirm empirical \(\lambda\) vs. instability order relationship

### 9.3 Comparison with Prior Work

| Aspect | Prior Work | DeepMind 2025 |
|--------|-----------|---------------|
| Unstable singularities discovered | 0 | Multiple families |
| Numerical residuals | ~10⁻³ to 10⁻⁵ | 10⁻⁷ to 10⁻¹³ |
| Systems studied | Limited | 3 canonical systems |
| Order of instability characterized | No | Yes, with linear pattern |
| Computer-assisted proof ready | No | Potentially yes (CCF) |

---

## 10. Limitations and Stated Caveats

### 10.1 Scope Limitations

1. **Bounded vs. Unbounded Domains:** Initial work focused on bounded domains (e.g., CCF is 1D, IPM and Boussinesq are 2D)
   - Extension to 3D unbounded Navier-Stokes remains challenging
   - Boundary conditions critical; results may not directly transfer

2. **Simplified Models:** CCF, IPM, and Boussinesq are reduced-complexity analogues
   - While related to 3D Euler/Navier-Stokes, not identical
   - Mechanisms of blow-up may differ in full 3D systems

3. **Self-Similarity Assumption:** All discovered solutions are self-similar
   - General blow-up (non-self-similar) not addressed
   - Self-similarity is restrictive but canonical ansatz

### 10.2 Numerical Precision Trade-offs

1. **Residual Accuracy vs. Computation Cost:** Achieving 10⁻¹³ residuals required specialized optimization; 10⁻⁷ more practical
2. **Grid Resolution:** Domain discretization and mesh refinement influence achievable precision
3. **Floating-Point Limitations:** GPU float64 precision inherently limits further refinement

### 10.3 Stability Analysis Limitations

1. **Linear Stability:** Eigenvalue analysis captures local stability; global stability behavior may differ
2. **Perturbation Direction:** Stability may depend on perturbation direction; analysis assumes worst-case perturbations
3. **Codimension:** High-codimension unstable manifolds (large \(n\)) are increasingly difficult to access numerically

### 10.4 Proof Completeness

1. **Computer-Assisted Proofs:** While residuals support existence, rigorous interval-arithmetic proofs remain future work
2. **Gap to 3D Navier-Stokes:** DeepMind has not directly addressed the full unbounded 3D equations
3. **Uniqueness:** Numerically discovered solutions confirmed to exist; uniqueness/multiplicity not fully characterized

---

## 11. Methodological Innovations

### 11.1 PINN Enhancements

**Standard PINN Loss:**
\[ L = \alpha_{\text{PDE}} \|\mathcal{L}[u]\|^2 + \alpha_{\text{BC}} \|\text{BC}\|^2 + \alpha_{\text{data}} \|u - u_{\text{data}}\|^2 \]

**DeepMind Modifications:**
- Removed data-fitting term (no experimental data used)
- Embedded boundary/symmetry constraints into network structure
- Incorporated self-similarity ansatz directly into parameterization
- Added regularization terms for numerical stability

### 11.2 Architecture Design Principles

1. **Equivariant Networks:** Axisymmetry built into architecture (e.g., via radius-based parameterization)
2. **Asymptotic Prescaling:** Initial behavior near singularity prescribed analytically; neural network captures corrections
3. **Smooth Parameterization:** Network outputs smooth, differentiable functions (required for residual evaluation)

### 11.3 Optimization Refinement

**Full-Matrix Gauss-Newton:**
- Computes Jacobian of residuals: \( J \in \mathbb{R}^{m \times p} \)
- Newton step: \( \Delta \theta = -(J^T J)^{-1} J^T r \)
- Avoids first-order optimizer issues (vanishing gradients near sharp features)
- Convergence criterion: residual stagnation

---

## 12. Identified Gaps and Open Problems

### 12.1 Immediate Extensions

1. **3D Bounded Domains:** Extend from 2D to 3D equations with rigid boundaries
2. **Unbounded Domains:** Remove boundary conditions; address 3D Euler and Navier-Stokes
3. **Non-Self-Similar Blow-Up:** Discover or rule out non-self-similar singularities
4. **Higher Instability Orders:** Discover and characterize \(n > 2\) or \(n > 3\) unstable branches

### 12.2 Theoretical Questions

1. **Universality of Linear Pattern:** Does \(\lambda(n) \propto n\) hold universally?
2. **Minimum Instability Order:** Is there a minimum instability order in 3D systems?
3. **Blow-Up Rates:** Do higher instability orders correspond to faster blow-up?
4. **Perturbation Direction Dependence:** How does stability vary with perturbation type?

### 12.3 Practical/Computational Challenges

1. **Scalability:** Can Gauss-Newton optimization scale to larger systems?
2. **High-Dimensional Geometry:** Is computational cost prohibitive for higher dimensions?
3. **Eigenvalue Stability:** Numerical computation of unstable eigenvalues near singularities (ill-conditioning)
4. **Computer-Assisted Proofs:** Complete interval-arithmetic validation pipeline for CCF equation

### 12.4 Physical Interpretation

1. **Mechanism of Instability:** Why do higher-order modes lead to faster blow-up?
2. **Energy Cascade:** Connection to energy transfer and vortex dynamics
3. **Navier-Stokes Relevance:** How do unstable singularities relate to viscous dissipation?
4. **Experimental Observation:** Can unstable singularities be detected or created in laboratory fluid experiments?

---

## 13. State-of-the-Art Summary

### 13.1 Before DeepMind 2025

- **Known:** Theoretical arguments for existence of unstable singularities in Euler/Navier-Stokes
- **Unknown:** Explicit, numerically realized unstable singularities for any incompressible fluid equation
- **Challenge:** Extremely high codimension; measure-zero sets in initial condition space

### 13.2 DeepMind 2025 Contributions

1. **Empirical Confirmation:** Unstable singularities exist and can be computed numerically
2. **Systematic Families:** Multiple unstable branches organized by instability order
3. **Precision Breakthrough:** Residuals 10⁻⁷ to 10⁻¹³ enable future computer-assisted proofs
4. **Universal Patterns:** Linear relationship between blow-up rate and instability order

### 13.3 Remaining Frontier

- **Full 3D Unbounded Systems:** Extend to Navier-Stokes without boundaries (the Clay Problem)
- **Rigorous Proofs:** Complete computer-assisted proof for at least one system
- **Physical Mechanisms:** Understand why unstable singularities emerge and their role in the PDE landscape
- **Broader Classes:** Extend methods to other nonlinear PDEs (reaction-diffusion, kinetic equations, etc.)

---

## 14. Technical References and Key Papers

### Primary Source

**DeepMind 2025:**
- "Discovery of Unstable Singularities" (arXiv:2509.14185, September 2025)
- Authors: DeepMind, Brown, NYU, Stanford collaborators
- First systematic discovery; multiple fluid systems studied

### Related Foundational Work

**Physics-Informed Neural Networks:**
- Raissi et al. (2019): Original PINN formulation for solving PDEs
- Follow-up work on PINN precision and optimization

**Self-Similar Blow-Up in Fluids:**
- Córdoba, Córdoba, Fontelos (early 2000s): CCF model introduced
- Prior computational studies on bounded Euler equations
- Boussinesq singularities: prior existence results

**Stability and Instability Theory:**
- Linear and nonlinear stability of self-similar solutions
- Eigenvalue analysis of PDE linearizations
- Codimension and manifold structure theory

---

## 15. Future Research Directions

### 15.1 Immediate Follow-Up Work

1. **Higher-Precision CCF:** Complete computer-assisted proof for CCF stable branch
2. **3D Extensions:** Adapt methodology to axisymmetric Euler with boundary
3. **Navier-Stokes Progress:** Attempt discovery of unstable singularities in viscous regime

### 15.2 Methodological Advances

1. **Hybrid Approaches:** Combine PINNs with analytical perturbation theory
2. **Adaptive Networks:** Dynamic network architecture adjustment based on solution structure
3. **Multi-Fidelity Learning:** Use lower-precision solutions to accelerate high-precision searches

### 15.3 Theoretical Implications

1. **Existence vs. Uniqueness:** Characterize families of unstable singularities in Navier-Stokes
2. **Transverse Instability:** Study stability to three-dimensional perturbations of 2D solutions
3. **Physical Realism:** Determine whether unstable singularities are dynamically relevant (given viscosity, small-scale perturbations)

---

## 16. Key Takeaways

1. **First Systematic Discovery:** DeepMind's work represents the first computational discovery of unstable singularity families for incompressible fluid equations.

2. **Precision Breakthrough:** Physics-informed neural networks with second-order optimization achieve 10⁻⁷ to 10⁻¹³ residuals, meeting computer-assisted proof standards.

3. **Universal Structure:** Linear relationship between blow-up rate and instability order suggests underlying organizing principles.

4. **Pathway to Millennium Prize:** While not solving Navier-Stokes directly, the work provides a computational pathway for studying singular behavior in related systems.

5. **Methodological Milestone:** Demonstrates AI-assisted mathematical discovery with full numerical rigor; potential model for other open PDE problems.

---

## 17. Citation Summary Table

| Paper/Source | Year | Focus | Key Result |
|-------------|------|-------|-----------|
| DeepMind et al. (arXiv:2509.14185) | 2025 | Unstable singularities in fluid PDEs | First systematic discovery; multiple families; 10⁻¹³ residuals (CCF) |
| Google DeepMind Blog | 2025 | Research announcement and summary | Three systems (CCF, IPM, Boussinesq); linear λ vs. instability order |
| Physics World Article | 2025 | Popularization and commentary | Context on Millennium Prize; significance of unstable vs. stable |
| Quanta Magazine | 2022 | Earlier work on AI and Navier-Stokes | Background on deep learning applications to fluid equations |

---

## 18. Appendix: Notation and Definitions

**Notation:**
- \(u(t,x)\): velocity field at time \(t\) and position \(x\)
- \(T\): blow-up time (time of singularity formation)
- \(\lambda\): blow-up rate exponent in self-similar scaling
- \(n\): order of instability (number of unstable eigenvalues)
- \(\eta = x/(T-t)^{\beta}\): similarity variable
- \(\mathcal{L}[u]\): PDE operator applied to \(u\)
- Residual: \(\|\mathcal{L}[\hat{u}]\|_{\infty}\) (maximum error in satisfying PDE)

**Key Concepts:**
- **Self-Similar Solution:** Scaling-invariant solutions that depend on similarity coordinates
- **Blow-Up / Finite-Time Singularity:** Growth of solution to infinity in finite time
- **Codimension:** Dimension of subspace of initial conditions leading to singularity
- **Unstable Manifold:** Set of initial conditions evolving toward blow-up under linearized dynamics
- **Computer-Assisted Proof:** Combination of analytical argument and numerical verification with rigorous error bounds

---

## 19. Document Metadata

**Compilation Date:** December 23, 2025
**Primary Focus:** Google DeepMind's September 2025 publication on unstable singularities
**Search Strategy:** 10 targeted web searches covering methodology, specific equations, stability analysis, numerical evidence, and applications
**Source Quality:** Peer-reviewed arXiv preprint, official DeepMind blog post, science journalism coverage, technical documentation
**Total Sources Reviewed:** 30+ unique sources (blogs, arXiv preprints, journalism, technical databases)

---

## 20. Sources

The following sources were consulted in the preparation of this literature review:

- [Discovering new solutions to century-old problems in fluid dynamics - Google DeepMind](https://deepmind.google/discover/blog/discovering-new-solutions-to-century-old-problems-in-fluid-dynamics/)
- [Discovery of Unstable Singularities - arXiv (HTML)](https://arxiv.org/html/2509.14185v1)
- [[2509.14185] Discovery of Unstable Singularities - arXiv (Abstract)](https://arxiv.org/abs/2509.14185)
- [Neural networks discover unstable singularities in fluid systems – Physics World](https://physicsworld.com/a/neural-networks-discover-unstable-singularities-in-fluid-systems/)
- [Meet The New Family of Blow-Ups Discovered By Google DeepMind – Circular Astronomy](https://circularastronomy.com/2025/09/21/meet-the-new-family-of-blow-ups-discovered-by-google-deepmind/)
- [Google AI's Potential Win of Millennium Prize - 36kr](https://eu.36kr.com/en/p/3473272279865732)
- [Google DeepMind AI Cracks Century-Old Fluid Mysteries – Decrypt](https://decrypt.co/340451/google-deepmind-ai-cracks-fluid-mysteries-new-era-science)
- [Deep Learning Poised to 'Blow Up' Famed Fluid Equations - Quanta Magazine](https://www.quantamagazine.org/deep-learning-poised-to-blow-up-famed-fluid-equations-20220412/)
- [Open-Sourcing the Universe's Code: Unstable Singularities in Fluid Dynamics - Medium](https://medium.com/@m.alfaro.007/open-sourcing-the-universes-code-unstable-singularities-in-fluid-dynamics-9f5cdb5e5e9e)
- [Google's DeepMind Just Cracked a Century-Old Math Mystery – Code Coup (Medium)](https://medium.com/@CodeCoup/googles-deepmind-just-cracked-a-century-old-math-mystery-in-fluid-dynamics-fe860eab58bc)
- [Using Physics-Informed neural networks for solving Navier-Stokes equations - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197625003471)
- [Physics Informed Neural Networks, A Proven PINNs Guide 2025](https://binaryverseai.com/physics-informed-neural-networks-pinns-explained/)
- [Physics-informed neural networks - Wikipedia](https://en.wikipedia.org/wiki/Physics-informed-neural-networks)
- [GitHub - PINNs: Physics Informed Deep Learning](https://github.com/maziarraissi/PINNs)
- [Localized instabilities of vortex rings with swirl — NYU Scholars](https://nyuscholars.nyu.edu/en/publications/localized-instabilities-of-vortex-rings-with-swirl)
- [Bifurcation analysis for axisymmetric capillary water waves with vorticity and swirl](https://arxiv.org/html/2202.01754)
- [Vortex Rings with Swirl: Axisymmetric Solutions of the Euler Equations - SIAM Journal](https://epubs.siam.org/doi/10.1137/0520005)
- [Finite-time singularities in the axisymmetric three-dimension Euler equations - Phys. Rev. Lett.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.68.1511)
- [Unstable Singularities in Fluid PDEs - Emergent Mind](https://www.emergentmind.com/papers/2509.14185)
- [Unstable Singularities in Fluid PDEs - Topic Overview](https://www.emergentmind.com/topics/unstable-singularities-in-nonlinear-fluid-pdes)
- [Stable self-similar singularity formation for infinite energy solutions - arXiv 2507.17381](https://arxiv.org/abs/2507.17381)
- [Discovery of Unstable Singularities - ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250914185W/abstract)
- [Finite time blow-up in a 1D model of the incompressible porous media equation - arXiv](https://arxiv.org/html/2412.16376)
- [Exact Gauss-Newton Optimization for Training Deep Neural Networks - arXiv 2405.14402](https://arxiv.org/abs/2405.14402)
- [Exact Gauss-Newton optimization for training deep neural networks - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231225024105)
- [Deep Neural Network Learning with Second-Order Optimizers - arXiv 2004.03040](https://ar5iv.labs.arxiv.org/html/2004.03040)
- [Unstable Eigenvectors & Eigenvalues - Emergent Mind](https://www.emergentmind.com/topics/unstable-eigenvector-and-eigenvalues)
- [Linear stability - Wikipedia](https://en.wikipedia.org/wiki/Linear_stability)
- [Blending Neural Networks with Physics: PINNs - SISSA mathLab (Medium)](https://medium.com/sissa-mathlab/blending-neural-networks-with-physics-the-physics-informed-neural-network-d681b6b44eb8)
- [Learning dynamical systems from data - PNAS](https://www.pnas.org/doi/10.1073/pnas.2311808121)
- [Embedding Physics into Deep Learning: A Structured Review of PINNs - Preprints.org](https://www.preprints.org/manuscript/202504.2577)
- [Physics-Informed Computer Vision: A Review - ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3689037)
- [Physical Symmetries Embedded in Neural Networks - Academia.edu](https://www.academia.edu/107363525/Physical_Symmetries_Embedded_in_Neural_Networks)
- [Physics-informed neural networks for PDE problems: A comprehensive review - Artificial Intelligence Review](https://link.springer.com/article/10.1007/s10462-025-11322-7)
- [Automatic network structure discovery via knowledge distillation - Nature Communications](https://www.nature.com/articles/s41467-025-64624-3)
- [Simplifying Hamiltonian and Lagrangian Neural Networks - Semantic Scholar](https://www.semanticscholar.org/paper/Simplifying-Hamiltonian-and-Lagrangian-Neural-via-Finzi-Wang/b639d2c614219f3de1e6a21091a1ad8d443916e8)
- [Understanding Mathematical Inductive Bias - NYU](https://cs.nyu.edu/media/publications/Marc_Finzi_Thesis__12_.pdf)
- [Google DeepMind discovers new solutions - Lifeboat News](https://lifeboat.com/blog/2025/09/google-deepmind-discovers-new-solutions-to-century-old-problems-in-fluid-dynamics)
- [Keryc News - AI Discovers Singularities](https://keryc.com/en/news/ai-discovers-unprecedented-singularities-fluid-dynamics-0aa35601)

---

**End of Literature Review**
