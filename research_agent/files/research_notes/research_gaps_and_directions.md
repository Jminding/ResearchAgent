# Research Gaps and Future Directions
## Stock Pricing Models Literature Survey

**Compilation Date:** December 2025
**Purpose:** Identify unresolved questions, methodological limitations, and promising research directions in foundational stock pricing models

---

## SECTION 1: IDENTIFIED GAPS IN CURRENT LITERATURE

### 1.1 Volatility Modeling Gaps

#### Gap 1.1a: Dynamic Model Selection Problem
**Issue:**
Literature provides multiple volatility models (GBM, Heston, GARCH, rough volatility, regime-switching) but no principled framework for selecting which model applies in a given context.

**Current State:**
- Each model supported by different empirical studies
- Model selection criteria (AIC, BIC, cross-validation) underexplored for derivatives pricing
- Practitioners use heuristics or software defaults

**Research Opportunity:**
- Develop unified framework for model comparison across different loss functions (pricing error, risk management, hedging performance)
- Information-theoretic approach: When does increased model complexity (Heston vs. GBM) justify additional parameters?
- Out-of-sample testing: Compare models' ability to predict future option prices on held-out data

#### Gap 1.1b: Stochastic Volatility Calibration Instability
**Issue:**
Heston and other stochastic volatility models have 5+ parameters; estimation from option prices is ill-posed (non-unique solutions).

**Specific Problem:**
- Correlation parameter ρ in Heston highly sensitive to optimization algorithm and starting values
- Different subsets of options (short-maturity vs. long-maturity, ITM vs. OTM) yield conflicting parameter estimates
- Day-to-day parameter "jumps" despite smooth market price changes

**Research Opportunity:**
- Regularization techniques to stabilize parameter estimates
- Hierarchical Bayesian methods for pooling information across options and time
- Functional data analysis approach: treat implied volatility surface as function, estimate parameters from functional principal components
- Constraints leveraging economic theory (e.g., ρ related to leverage effect magnitude)

#### Gap 1.1c: Rough Volatility Calibration Methods
**Issue:**
Rough volatility (Gatheral et al., 2018) theoretically superior but computationally more challenging than classical models.

**Current State:**
- Hurst exponent H estimated via realized volatility, not from option prices
- Most papers use fixed H ≈ 0.1; time-varying H not studied
- Integration with other features (jumps, multiscale roughness) in early stages

**Research Opportunity:**
- Joint estimation of H from option prices using characteristic function methods
- Sequential estimation: Update H as new options data arrives (filtering/learning)
- Hybrid rough-Markovian model: Short-term rough, long-term mean-reverting behavior

### 1.2 Jump Risk and Tail Risk Gaps

#### Gap 1.2a: Jump Parameter Estimation
**Issue:**
Merton jump-diffusion requires estimating λ (jump intensity), jump size distribution parameters; standard MLE approaches unstable.

**Current State:**
- Jump detection methods (BPV, realized variance cleaners) rely on high-frequency data not always available
- Jump arrival times treated as latent variables; inference computationally intensive
- Conflicting estimates across literature: λ ranges 0.5-5 jumps/year depending on method

**Research Opportunity:**
- Bayesian nonparametrics: Place Dirichlet process priors on jump size distribution (don't assume lognormal)
- Realized jumps: Use high-frequency tick data to directly identify jumps, reduce latent variable problem
- Multi-scale jump detection: Small jumps vs. large jumps may have different dynamics
- Option-implied jump information: Extract jump parameters from smile/skew properties

#### Gap 1.2b: Leverage Effect and Correlation Structure
**Issue:**
Heston model assumes fixed correlation ρ between price and volatility; empirically, leverage effect (∂σ/∂S) time-varying and asymmetric.

**Current State:**
- Negative correlation ρ ≈ -0.5 to -0.7 typical for equities
- Magnitude varies across stocks and time periods
- Time-varying leverage documented (stronger during stress) but not modeled in standard frameworks

**Research Opportunity:**
- Regime-switching leverage: Different ρ in bull vs. bear markets
- Asymmetric leverage: |∂σ/∂S| larger for negative returns than positive
- Jump-leverage coupling: Leverage effect stronger immediately after jumps
- Volatility feedback loops: Model endogenous increase in volatility from price declines (amplification mechanism)

#### Gap 1.2c: Tail Risk Quantification
**Issue:**
Models generate kurtosis and skewness but effectiveness for tail risk (1% VaR, expected shortfall) incomplete.

**Current State:**
- Black-Scholes severely underestimates tail risk
- Jump-diffusion improves but parameter uncertainty large
- Rough volatility shows promise empirically but limited theoretical work on extreme quantile properties

**Research Opportunity:**
- Extreme value theory integration: Model conditional distribution in tails separately
- Stress test design: Which model assumptions most critical for tail outcomes?
- Implicit tail risk: Extract tail expectations from deep OTM option prices (option-implied tail risk metrics)
- Path simulation and tail events: Monte Carlo study of model performance under adverse scenarios

### 1.3 Multi-Asset and Dependence Gaps

#### Gap 1.3a: Correlation Structure in Multi-dimensional Models
**Issue:**
Basket options, portfolio derivatives require joint modeling of multiple assets; correlation structure simplified in most models.

**Current State:**
- Most models assume constant correlation
- Empirical evidence: Correlations increase in market stress (contagion effect)
- Dynamic correlation models (DCC-GARCH) exist but not integrated into standard option pricing

**Research Opportunity:**
- Stochastic correlation models coupled with stochastic volatility (multi-variate Heston with stochastic correlations)
- Copula methods: Model dependence structure separately from marginal distributions
- Factor-based correlation: Assets driven by common factors; factor structure more stable than pairwise correlations

#### Gap 1.3b: High-Dimensional Scaling
**Issue:**
For large portfolios (e.g., 100+ assets), computational feasibility of pricing/hedging breaks down.

**Current State:**
- Typically use PCA to reduce dimensionality
- Loss of information from dimensionality reduction not quantified
- Most papers focus on 2-5 asset case

**Research Opportunity:**
- Curse of dimensionality analysis: How many factors really needed?
- Approximate pricing methods for large-scale portfolios
- Machine learning for dimensionality: Learn low-dimensional representation of high-dimensional option surfaces

### 1.4 Time-Varying Parameters and Non-Stationarity

#### Gap 1.4a: Parameter Time-Variation
**Issue:**
Models assume constant parameters (μ, σ, λ, ρ, etc.); empirically, parameters drift over time.

**Current State:**
- Rolling window estimation used ad-hoc; no principled framework
- Breakpoint detection (Chow test, CUSUM) identifies changes but doesn't model smooth drift
- Few papers explicitly model parameter evolution

**Research Opportunity:**
- State-space models: Treat parameters as latent states, estimate via Kalman filter
- Bayesian adaptive learning: Update parameter beliefs as data arrives
- Piecewise-constant parameters: Identify regimes and optimize regime-switching model
- Information dynamics: How new market information (news, earnings) changes parameters?

#### Gap 1.4b: Regime-Switching Integration
**Issue:**
Market behavior differs across regimes (bull/bear, crisis/normal, high/low volatility); single-parameter models inadequate.

**Current State:**
- Regime-switching models (Hamilton) exist for equity returns but not fully integrated with option pricing
- Heston model with regime-switching rare (computational complexity)
- Transition probabilities between regimes usually assumed constant or ad-hoc

**Research Opportunity:**
- Stochastic volatility with regime-switching: Different Heston parameters in each regime
- Observable regime indicators: Use leading indicators (VIX, credit spreads, term structure slope) to improve regime inference
- Asymmetric regimes: Crisis regimes have different dynamics than normal regimes
- Mean reversion across regimes: Mean volatility level different by regime

### 1.5 Interest Rate and Bond Model Gaps

#### Gap 1.5a: Vasicek/CIR Limitations
**Issue:**
Vasicek allows negative rates (addressed post-2008); CIR complex to estimate and calibrate.

**Current State:**
- Vasicek still widely used despite theoretical flaws
- CIR requires Bessel function evaluations (numerical stability issues at boundaries)
- Hull-White model adopted as compromise but not well-studied theoretically

**Research Opportunity:**
- Shifted models (Hull-White with lower bound): Characterize impact of bounds on pricing, hedging
- Quadratic models: Quadratic term structure models (QTSMs) provide flexibility; integration with stochastic volatility
- Affine term structure models: General framework with analytical tractability

#### Gap 1.5b: Term Structure Slope and Curvature
**Issue:**
Models focus on short rate; forward rate curve slope/curvature often ignored or oversimplified.

**Current State:**
- Empirical evidence: 3-4 factors explain yield curve (level, slope, curvature, twist)
- Single-factor Vasicek/CIR miss slope/curvature dynamics
- Multi-factor models exist but calibration challenging

**Research Opportunity:**
- Factor-based yield curve models: Directly model principal components
- Slope as state variable: Include (long - short) rate differential as second state
- Term structure implications: Infer market expectations of future rate levels, volatility, inflation from term structure

#### Gap 1.5c: Inflation and Real Rates
**Issue:**
Classical models ignore inflation; post-2021, inflation surge renewed interest in inflation-adjusted models.

**Current State:**
- Fisher hypothesis: Nominal = real + expected inflation; weakly supported empirically
- Break-even inflation rates (TIPS spreads) distorted by liquidity and hedging flows
- Few papers model real-inflation dynamics jointly

**Research Opportunity:**
- Joint real-inflation model: Two-factor short rate model (real + inflation components)
- Stochastic inflation: Inflation volatility varies; capturing inflation regimes
- Purchasing power and derivatives: Real value of nominal derivatives affected by inflation

### 1.6 Empirical Implementation Gaps

#### Gap 1.6a: Model Validation Procedures
**Issue:**
No standard approach for validating models on out-of-sample data; back-testing methodologies differ.

**Current State:**
- Papers compare models on same dataset (in-sample)
- Few papers report out-of-sample pricing errors
- Overfitting risk: Complex models (Heston, rough volatility) may have more parameters than justifiable

**Research Opportunity:**
- Standardized benchmarks: Define holdout test sets for validation
- Cross-validation for options: Walk-forward testing (temporal validation)
- Information criteria properly applied: AIC, BIC with correct degrees of freedom
- Pricing error metrics: Compare MSE, MAE, RMSE across models with statistical tests

#### Gap 1.6b: Algorithmic and Implementation Issues
**Issue:**
Literature focuses on theory; practical implementation challenges underappreciated.

**Current State:**
- Numerical stability of PDEs not thoroughly studied for all models
- Monte Carlo variance reduction techniques scattered across papers
- Code availability limited; reproducibility concerns

**Research Opportunity:**
- Benchmark implementations: Release calibrated models with transparent code
- Stability analysis: Characterize when numerical algorithms fail (e.g., near boundaries in CIR)
- Hybrid algorithms: Combine analytical (where available) with numerical methods
- Real-time pricing: How to update option prices as new market data arrives?

#### Gap 1.6c: Transaction Costs and Market Microstructure
**Issue:**
Virtually all foundational models ignore transaction costs, discrete tick sizes, bid-ask spreads.

**Current State:**
- Classical no-arbitrage theory assumes frictionless markets
- Bid-ask spreads: 0.01-0.05 per share (1-10% of option value for some options)
- Impact of these frictions on optimal hedging ratios not fully characterized

**Research Opportunity:**
- Bid-ask adjusted pricing: Adjust Black-Scholes formula for realistic spreads
- Hedging with transaction costs: Optimal rehedging frequency/bands
- Discrete tick effects: How pricing formulas change under discrete price movements
- Market impact: Large trades affect prices; implications for replication hedging

---

## SECTION 2: THEORETICAL GAPS

### 2.1 Mathematical Foundations

#### Gap 2.1a: Non-Semimartingale Processes
**Issue:**
Rough volatility (fractional Brownian motion) is not a semimartingale; classical stochastic calculus machinery doesn't apply directly.

**Current State:**
- Ad-hoc approaches for pricing under rough volatility
- Pathwise integration developed but less standard
- Arbitrage pricing under non-semimartingales still evolving

**Research Opportunity:**
- Extend fundamental theorems of asset pricing to non-semimartingales
- Pathwise integral properties: Fully characterize when classical results extend
- Change of measure for rough processes: Develop Girsanov-type theorems
- No-arbitrage characterization under roughness

#### Gap 2.1b: Levy Processes and Infinite Activity Jumps
**Issue:**
Standard jump-diffusion (Poisson) assumes finite activity; empirically, infinite activity (many small jumps) may be more realistic.

**Current State:**
- Variance Gamma, Normalized Inverse Gaussian models developed but complex
- Limited empirical comparison of finite vs. infinite activity
- Calibration methods for Levy processes challenging

**Research Opportunity:**
- Jump decomposition: Separate large discrete jumps from continuous small-jump component
- Optimal representation: Which Levy process for which market/asset class?
- Characteristic function methods: Extend analytical techniques to broader class of Levy processes

### 2.2 No-Arbitrage Theory

#### Gap 2.2a: Model-Free Bounds
**Issue:**
Options prices must satisfy no-arbitrage bounds (e.g., call price ≤ stock price); models must respect these.

**Current State:**
- Classical bounds well-known (call-put parity, intrinsic value bounds)
- Model-free implied variance bounds less discussed
- Connection between model assumptions and feasible bound ranges unexplored

**Research Opportunity:**
- Derive tightest model-free bounds under specific assumptions (e.g., stochastic volatility)
- Identify which models naturally satisfy bounds vs. requiring constraints
- Use bounds as diagnostic: If model-implied prices violate bounds, diagnose assumption failures

#### Gap 2.2b: Implied Process Characterization
**Issue:**
Given observed option prices, what do they imply about the underlying process?

**Current State:**
- Implied volatility well-established; implied volatility surface documented
- Implied skewness/kurtosis less standard; extraction methods ad-hoc
- Implied distribution (Breeden-Litzenberger) rarely computed in practice

**Research Opportunity:**
- Algorithm to extract implied distributions from option prices
- Implied jump probability and intensity from smile properties
- Consistency check: Are implied processes economically sensible?
- Information content: What market information embedded in option prices vs. spot/forward prices?

### 2.3 Equilibrium and Foundations

#### Gap 2.3a: Equilibrium Pricing Models
**Issue:**
Most models are "reduced-form" (assume price process); few derive pricing from economic equilibrium.

**Current State:**
- Cox-Ingersoll-Ross (1985) provided equilibrium model for term structure; extended versions rare
- Merton (1973) equilibrium option pricing; seldom extended
- Limited models with consumption-based framework

**Research Opportunity:**
- Equilibrium models with habit formation (model preferences, not just return distribution)
- Jump risk premiums from equilibrium: What jump intensity/size emerges from rational agents?
- Stochastic volatility in equilibrium: Derive Heston model from agent preferences
- Multi-agent equilibrium: Heterogeneous agents, learning, information asymmetries

#### Gap 2.3b: Behavioral Finance Integration
**Issue:**
Foundational models assume rationality; behavioral evidence (overconfidence, loss aversion, momentum) suggests otherwise.

**Current State:**
- Behavioral finance literature largely separate from derivatives pricing
- Behavioral models exist but don't lead to explicit option pricing formulas
- Connection between behavioral biases and volatility smile unclear

**Research Opportunity:**
- Behavioral volatility smile: Do behavioral biases explain observed smile/skew?
- Prospect theory and options: How loss-averse agents value options?
- Sentiment and prices: Incorporate investor sentiment as latent state
- Information cascades and jumps: Behavioral herding as explanation for jump risk

---

## SECTION 3: EMPIRICAL AND PRACTICAL GAPS

### 3.1 Empirical Testing

#### Gap 3.1a: Cross-Asset Generalization
**Issue:**
Most empirical studies focus on equity options; generalization to other assets limited.

**Current State:**
- Currencies: Some work (Merton on FX options)
- Commodities: Limited studies; mean reversion literature stronger
- Credit: CDS pricing models; default risk introduces complications
- Crypto: Emerging; very high volatility, different distribution properties

**Research Opportunity:**
- Standardized empirical comparison across assets
- Asset-specific calibration: Which model best for each asset class?
- Cross-asset correlation studies: Do models that work for equities transfer to commodities, currencies?
- Volatility surface properties by asset: Systematic comparison of smile characteristics

#### Gap 3.1b: Time Period Robustness
**Issue:**
Models estimated on calm-market data may fail in stressed markets; sensitivity to sample period understudied.

**Current State:**
- Few papers systematically test model stability across market regimes
- COVID-19 (2020), 2008 financial crisis offer natural stress tests; limited analysis
- Parameter estimates from pre-crisis vs. during-crisis periods differ substantially

**Research Opportunity:**
- Robust estimation methods: Minimize worst-case pricing error across regimes
- Crisis calibration: Special treatment of risk premium changes in stress
- Forward-testing: Train model on historical data, test on future crisis period
- Learning and adaptation: Do markets learn to price options correctly after crises?

#### Gap 3.1c: Distributional Assumptions Validation
**Issue:**
Models assume specific distributions (lognormal, normal); empirical distributions differ.

**Current State:**
- Fat tails, skewness widely documented
- Goodness-of-fit tests (KS, Anderson-Darling) show normal/lognormal rejected
- Impact on option pricing not fully quantified

**Research Opportunity:**
- Semiparametric methods: Don't assume specific distribution; use empirical CDF
- Quantile-based pricing: Price options using estimated quantiles directly
- Distribution misspecification risk: How much do pricing errors stem from distribution assumption?

### 3.2 Practical Applications

#### Gap 3.2a: Hedging Performance
**Issue:**
Models tell how to price; less emphasis on whether hedging based on model predictions works in practice.

**Current State:**
- Delta hedging is standard but assumes Black-Scholes; robustness to model misspecification unclear
- Vega hedging (long volatility) more relevant for Heston but implementation complex
- Backtesting of hedging strategies limited

**Research Opportunity:**
- Hedging under model uncertainty: Robust hedges valid across multiple models
- Empirical hedging performance: Compare delta vs. delta-gamma vs. delta-vega hedging
- Rehedging frequency: How often should hedges be rebalanced? (Trade cost vs. hedge effectiveness)
- Hedging with constraints: Impact of trading limits, leverage constraints on hedging

#### Gap 3.2b: Risk Management and Stress Testing
**Issue:**
Models used for pricing; how to use them for risk management not fully specified.

**Current State:**
- VaR computations typically ad-hoc (historical simulation, parametric)
- Greeks (delta, gamma, vega) important but not sufficient for multi-factor risk
- CVA (counterparty credit adjustment) overlaid on option prices; integration limited

**Research Opportunity:**
- Integrated risk: Model price risk, credit risk, counterparty risk jointly
- Tail risk metrics: Expected shortfall, expected tail loss from models
- Stress scenarios: Generate pathwise scenarios from model; risk profiles under stressed paths
- Dynamic risk management: How do Greeks evolve over time? Gamma-scalping opportunities?

#### Gap 3.2c: Volatility Trading
**Issue:**
Options markets allow trading volatility directly; models should inform volatility trading strategies.

**Current State:**
- Volatility ETPs (VXX, UVXY) track realized volatility; pricing of these products understudied
- Variance swaps: Fixed payoff on realized variance; pricing and hedging models available but complex
- Volatility forecasting: Separate literature; limited integration with option pricing models

**Research Opportunity:**
- Volatility trading under model constraints: Optimal positions when volatility follows Heston?
- Variance swap pricing: Connect to option prices; extract forward variance via moment matching
- Volatility index (VIX) properties: Is VIX pricing consistent with option model?
- Vol-of-vol trading: Trade volatility of volatility (requires rough volatility or higher-order models)

### 3.3 Computational and Algorithmic Gaps

#### Gap 3.3a: Machine Learning Integration
**Issue:**
Deep learning, neural networks emerging for option pricing; not yet integrated into foundational framework.

**Current State:**
- Neural network option pricing gaining traction (papers 2020+)
- Typically used to approximate PDE solutions or calibration
- Limited theoretical understanding of when/why neural networks outperform classical methods

**Research Opportunity:**
- Interpretability: What does neural network learn? Can extract economic insights?
- Hybrid methods: Classical model + neural network correction
- Transfer learning: Train on liquid (SPX) options; fine-tune for less-liquid underlyings
- Calibration via learning: Inverse problem of option pricing via neural nets

#### Gap 3.3b: Fast Algorithms for New Models
**Issue:**
Rough volatility, regime-switching, multi-factor models computationally challenging; fast algorithms needed for real-time trading.

**Current State:**
- Heston: Fourier inversion methods ~0.1s per option
- Rough volatility: Monte Carlo or approximations; slower
- Regime-switching: Filter update expensive at each time step

**Research Opportunity:**
- FFT/COS methods for new models: Extend fast Fourier transform to rough processes
- Surrogate models: Train fast approximation (polynomial, rational function) to slow model
- GPU/parallel computing: Leverage modern hardware for Monte Carlo
- Asymptotic approximations: Develop analytic approximations for short/long maturities

---

## SECTION 4: SYNTHESIS AND PRIORITY RESEARCH DIRECTIONS

### Highest Priority (Foundation for Future Work)

**4.1 Volatility Parameter Stability and Estimation**
- *Why*: Volatility is core input to all models; unreliable estimation undermines entire framework
- *What to do*: Develop robust, stable calibration procedures; benchmark on standard datasets
- *Expected impact*: Practical models practitioners trust and use consistently

**4.2 Multi-Factor Option Pricing**
- *Why*: Single-factor (BS/Heston) insufficient; equities exhibit size, value, momentum factors; bonds have level/slope/curvature
- *What to do*: Develop tractable multi-factor option pricing models; calibration procedures
- *Expected impact*: Better explanation of cross-sectional option prices; improved hedging

**4.3 Out-of-Sample Validation Framework**
- *Why*: Overfitting risk high; models need objective validation on unseen data
- *What to do*: Establish standardized testing protocols; public benchmarks
- *Expected impact*: Accurate assessment of which models truly superior; guide model selection

### Medium Priority (Refinements and Extensions)

**4.4 Regime-Switching Stochastic Volatility**
- *Why*: Evidence of different regimes in volatility; unified model needed
- *What to do*: Heston model with regime-switching; efficient calibration
- *Expected impact*: Better hedging in crisis; improved risk management

**4.5 Option Pricing under Transaction Costs**
- *Why*: Transaction costs significant; models ignoring them unrealistic
- *What to do*: Develop option pricing with explicit bid-ask, rehedging costs
- *Expected impact*: More realistic valuation; better understand bid-ask widths

**4.6 Jump-Leverage Coupling**
- *Why*: Leverage effect stronger after jumps; not modeled in Merton
- *What to do*: Develop models where jump probability/size depends on state
- *Expected impact*: Better tail risk understanding; improved crisis pricing

### Longer-Term (Fundamental Questions)

**4.7 Behavioral Option Pricing**
- *Why*: Behavioral biases documented; implications for derivatives not clear
- *What to do*: Integrate behavioral economics with derivatives pricing theory
- *Expected impact*: Explain anomalies (smile, skew); match human decision-making

**4.8 Quantum Computing for Finance**
- *Why*: Emerging technology; potential for exponential speedup on certain problems
- *What to do*: Identify which financial problems suited to quantum; develop quantum algorithms
- *Expected impact*: Ability to solve larger problems; new models previously intractable

**4.9 Systemic Risk and Network Effects**
- *Why*: Options on multiple correlated assets; network structure matters (e.g., financial contagion)
- *What to do*: Develop pricing models incorporating systemic risk; agent-based frameworks
- *Expected impact*: Understand tail risks; systemic risk measurement and hedging

---

## SECTION 5: RECOMMENDATIONS FOR RESEARCHERS

### For Empirical Studies
1. Report out-of-sample pricing errors, not just in-sample fit
2. Compare models on same dataset with consistent metrics
3. Test robustness across time periods and market regimes
4. Provide code/data for reproducibility
5. Quantify parameter uncertainty (confidence intervals, sensitivity analysis)

### For Theoretical Work
1. Clearly state assumptions; justify why realistic
2. Provide conditions under which model properties hold
3. Characterize limitations; don't hide failures
4. Connect to empirical implications
5. Relate to equilibrium foundations where possible

### For Practitioners
1. Understand model limitations before using
2. Backtest on your own data, markets, time periods
3. Use ensemble of models, not single model
4. Monitor parameter stability; alert on changes
5. Combine model-based pricing with market prices (trust observed when model uncertain)

---

## SECTION 6: OPEN QUESTIONS

### Fundamental Questions
1. Is there a single "best" option pricing model, or is it asset/time/regime dependent?
2. How much of the volatility smile can be explained by jumps vs. stochastic volatility vs. leverage effect?
3. Do agents truly price options correctly, or are there persistent pricing errors? (EMH question)
4. What is the equilibrium level of option implied volatility? (Why is it what it is?)
5. How to optimally combine model prices with market prices for practical trading?

### Methodological Questions
1. How to handle parameter time-variation? State-space vs. regime-switching vs. adaptive learning?
2. What is the "right" loss function for option pricing (MSE, MAE, proportional error)?
3. How to validate models when true data-generating process unknown?
4. How to incorporate market microstructure without losing tractability?
5. When should simpler models (BS) be preferred to complex ones (Heston) in practice?

### Practical Questions
1. Can neural networks improve on classical models? By how much? When?
2. How to hedge options in illiquid markets where rebalancing is costly?
3. How to forecast volatility? (Separate literature; integration with pricing?)
4. How to price options on illiquid underlyings? (Extrapolate from liquid ones?)
5. How to account for funding costs and repo rates in option pricing? (Increasingly important post-2008)

---

**End of Document**

*This research agenda synthesizes gaps across 15+ major papers and identifies 40+ specific research directions. The highest-priority items address foundational challenges (volatility estimation, multi-factor modeling, validation) without which progress on other areas is limited.*

