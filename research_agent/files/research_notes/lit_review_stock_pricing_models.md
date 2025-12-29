# Literature Review: Foundational Stock Pricing Models
## Black-Scholes, Geometric Brownian Motion, and Mean Reversion Models

**Date Compiled:** December 2025
**Scope:** Peer-reviewed literature, academic preprints, and authoritative technical sources
**Focus:** Mathematical formulations, theoretical assumptions, empirical validation, and limitations

---

## 1. OVERVIEW OF THE RESEARCH AREA

Stock pricing models form the mathematical foundation of modern quantitative finance, enabling valuation of derivatives, portfolio optimization, and risk management. The core framework rests on three interconnected pillars:

1. **Continuous-Time Stochastic Processes**: Geometric Brownian motion (GBM) as the primary model for stock price evolution
2. **Derivatives Pricing**: The Black-Scholes formula and its extensions for option valuation
3. **Mean Reversion Models**: Stochastic differential equations capturing mean-reverting behavior (Ornstein-Uhlenbeck, Vasicek processes)

The theoretical underpinnings rely heavily on Itô calculus, martingale pricing theory, and stochastic differential equations (SDEs). Despite widespread adoption, empirical evidence reveals significant discrepancies between theoretical assumptions and market reality, including non-constant volatility, fat-tailed distributions, and jump discontinuities.

---

## 2. CHRONOLOGICAL DEVELOPMENT OF MAJOR MODELS

### 2.1 The Black-Scholes Framework (1973)

**Seminal Work:**
- **Black, F. & Scholes, M. (1973)** "The Pricing of Options and Corporate Liabilities," *Journal of Political Economy*, Vol. 81, No. 3, pp. 637–654.
  - Proposed the first successful closed-form option pricing formula for European-style derivatives
  - Assumes: (1) geometric Brownian motion for underlying asset, (2) constant volatility, (3) no transaction costs, (4) log-normal returns
  - Mathematical basis: Dynamic portfolio replication and risk-neutral valuation
  - Impact: Launched the field of financial engineering; became the most widely used option pricing model globally

**Historical Context:**
- Black and Scholes (1968) developed the risk-neutral argument through dynamic portfolio revision, showing that expected returns are not relevant for pricing (foundational for modern finance)
- Publication was difficult; eventually required intervention from Eugene Fama and Merton Miller to be accepted by Journal of Political Economy

### 2.2 Geometric Brownian Motion: Mathematical Foundation

**Stochastic Differential Equation Formulation:**

For a stock price S_t, the GBM is defined as:
```
dS_t = μ S_t dt + σ S_t dW_t
```

Where:
- μ = drift (expected return)
- σ = volatility (constant)
- dW_t = Wiener process increment
- Analytical solution: S_t = S_0 exp[(μ - σ²/2)t + σ W_t]

**Key Properties of GBM:**
1. **Positive Values**: S_t > 0 for all t (unlike arithmetic Brownian motion)
2. **Log-Normal Distribution**: log(S_T) ~ N(log(S_0) + (μ - σ²/2)T, σ²T)
3. **Scale Independence**: Expected returns independent of process value (realistic assumption)
4. **Computational Tractability**: Analytical solution exists; amenable to Monte Carlo simulation

**Mathematical Justification:**
- GBM ensures all simulated prices remain positive, reflecting real stock behavior
- Logarithmic returns follow normal distribution, consistent with empirical observations (with caveats)
- Itô's lemma facilitates derivative pricing under this specification

**Critical Limitations:**
- **Constant Volatility**: Real markets exhibit time-varying, stochastic volatility
- **No Jumps**: Assumes continuous price paths; reality includes discontinuities (earnings announcements, market shocks)
- **Normal Assumption**: Returns exhibit fat tails and skewness violating normality

### 2.3 Itô Calculus and Stochastic Integration

**Foundational Theory:**
- **Itô, K. (1951)** Established the stochastic integral and calculus results (Itô's lemma)
- Itô's lemma is the stochastic counterpart to the chain rule in ordinary calculus

**Mathematical Statement (Itô's Lemma):**

For a stochastic process dX_t = a(X_t)dt + b(X_t)dW_t and a smooth function f(t,x):
```
df(t, X_t) = [∂f/∂t + a(X_t)∂f/∂x + (1/2)b(X_t)²∂²f/∂x²] dt + b(X_t)∂f/∂x dW_t
```

**Critical Distinction from Deterministic Calculus:**
- Includes second-order term (1/2)b(X_t)²∂²f/∂x² due to non-zero quadratic variation of Brownian motion
- This arises from property: (dW_t)² = dt (not zero as in standard calculus)
- Application: Derivation of Black-Scholes PDE and option pricing formulas

**Historical Significance:**
- Transforms SDEs systematically without manual derivations
- Enables closed-form solutions for many financial models
- Essential for modern derivatives pricing theory

### 2.4 Merton's Extension: Jump-Diffusion Models (1976)

**Research:**
- **Merton, R. C. (1976)** Extended Black-Scholes to include jump processes

**Model Structure:**
Asset price evolution:
```
dS_t = μ S_t dt + σ S_t dW_t + (Y_t - 1)S_t dN_t
```

Where:
- N_t = Poisson process with intensity λ (jump arrival rate)
- Y_t = jump size factor (multiplicative, log-normal distribution)
- Independent of Brownian motion

**Key Contributions:**
1. **Captures Tail Risk**: Adds kurtosis and skewness to return distribution
2. **Volatility Smile Generation**: Jump component produces observed option volatility smile
3. **Market Anomalies**: Explains extreme price movements from news/shocks

**Quantitative Results:**
- Jump-diffusion model better fits observed option prices far from the money
- Generates implied volatility surface consistent with market data (vs. flat volatility under pure GBM)

**Limitations:**
- Parameter estimation challenging (jump intensity λ and jump size distribution)
- Still assumes specific distributional forms

### 2.5 Mean Reversion Models: Ornstein-Uhlenbeck and Vasicek

**Ornstein-Uhlenbeck Process:**

Stochastic differential equation:
```
dX_t = θ(μ - X_t)dt + σ dW_t
```

Where:
- θ = mean reversion speed
- μ = long-term mean level
- σ = volatility
- Key property: X_t exhibits oscillations around μ

**Analytical Solution:**
```
X_t = μ + (X_0 - μ)e^(-θt) + σ ∫₀ᵗ e^(-θ(t-s)) dW_s
```

**Vasicek Model for Interest Rates (1977):**
- **Vasicek, O. A. (1977)** Applied OU process to short-rate modeling
- Model: dr_t = a(b - r_t)dt + σ dW_t
- First model to capture mean reversion in interest rates
- Parameters: a = reversion speed, b = long-term mean

**Key Properties:**
1. **Mean Reversion**: Process reverts to long-term equilibrium
2. **Analytical Tractability**: Closed-form bond pricing formula exists
3. **Continuous Paths**: Smooth, no discontinuities

**Critical Limitation:**
- **Negative Interest Rates**: Normal distribution component allows r_t < 0 (unrealistic)
- Partially addressed in later models (CIR, Hull-White)

**Applications Beyond Rates:**
- Commodity prices (oil, agricultural futures)
- Currency pairs
- Stochastic volatility modeling
- Pairs trading and statistical arbitrage

### 2.6 Heston's Stochastic Volatility Model (1993)

**Research:**
- **Heston, S. L. (1993)** Proposed two-factor model with mean-reverting volatility

**Model Specification:**
```
dS_t = μ S_t dt + √(v_t) S_t dW_t^S
dv_t = κ(θ - v_t)dt + ξ√(v_t) dW_t^v
```

Where:
- S_t = stock price
- v_t = variance (stochastic)
- κ = volatility mean reversion speed
- θ = long-term variance level
- ξ = volatility of volatility
- Correlation ρ between W^S and W^v (typically negative for equities)

**Key Advantages:**
1. **Volatility Smile/Smirk**: Generates realistic implied volatility surfaces (vs. flat BS)
2. **Mean Reversion**: Volatility reverts to equilibrium level
3. **Closed-Form Solution**: Characteristic function available; European option prices via numerical integration
4. **Empirical Fit**: Superior to Black-Scholes for out-of-the-money options

**Practical Acceptance:**
- Widely adopted by practitioners as compromise between theoretical rigor and computational tractability
- Handles European options; American pricing requires approximation methods

**Limitations:**
- Estimation of 5 parameters (μ, κ, θ, ξ, ρ) requires robust calibration procedures
- Does not address leverage effect fully in all regimes

---

## 3. MATHEMATICAL FOUNDATIONS AND KEY EQUATIONS

### 3.1 Black-Scholes Option Pricing Formula

**European Call Option Price:**
```
C(S,t) = S N(d₁) - K e^(-r(T-t)) N(d₂)
```

**European Put Option Price:**
```
P(S,t) = K e^(-r(T-t)) N(-d₂) - S N(-d₁)
```

Where:
```
d₁ = [ln(S/K) + (r + σ²/2)(T-t)] / [σ√(T-t)]
d₂ = d₁ - σ√(T-t)
```

**Parameters:**
- S = current stock price
- K = strike price
- r = risk-free rate
- σ = volatility (annualized)
- T - t = time to expiration
- N(·) = cumulative standard normal distribution

**Derivation Method:**
1. Assume stock follows GBM: dS = μS dt + σS dW
2. Form risk-neutral portfolio: Δ shares + 1 short option
3. Apply Itô's lemma to option value V(S,t)
4. Eliminate randomness through delta-hedging
5. Risk-neutral pricing (μ replaced by r)
6. Solve resulting PDE with boundary conditions

### 3.2 Risk-Neutral Valuation Framework

**Core Principle:**
Under the risk-neutral measure Q (equivalent martingale measure):
```
V_t = E^Q[e^(-r(T-t)) V_T | F_t]
```

Where:
- V_T = payoff at maturity T
- r = risk-free rate
- E^Q = expectation under risk-neutral measure
- F_t = information set at time t

**Key Insight:**
- Actual drift μ is irrelevant for pricing (replaced by r)
- Investors are assumed risk-neutral in pricing measure (not in reality)
- This is the fundamental theorem of asset pricing in continuous time

---

## 4. PRIOR WORK: COMPREHENSIVE TABLE OF KEY PAPERS

| Author(s) & Year | Venue | Model/Topic | Key Result | Limitation |
|---|---|---|---|---|
| Black & Scholes (1973) | J. Political Economy | European option pricing | Closed-form formula; launched derivatives industry | Assumes constant volatility; lognormal returns |
| Merton (1976) | J. Financial Economics | Jump-diffusion processes | Captures tail risk and volatility smile | Parameter estimation challenging |
| Cox, Ross & Rubinstein (1979) | J. Financial Economics | Binomial tree pricing | Converges to BS; handles American options | Computational complexity with many steps |
| Vasicek (1977) | J. Financial Economics | Interest rate mean reversion | First tractable OU-based rate model | Allows negative rates |
| Heston (1993) | Rev. Financial Studies | Stochastic volatility | Closed-form European option prices; generates smile | Restricted to European options; calibration difficult |
| Fama & French (2004) | J. Economic Perspectives | CAPM limitations and extensions | Documents size, value, momentum anomalies | Does not fully explain all cross-sectional returns |
| Merton (1973) | Bell J. Economics | Option pricing and CAPM | Extends option pricing to continuous-time equilibrium | Model assumptions restrictive |
| Ito (1951) | — | Stochastic calculus (Itô's lemma) | Foundation for solving SDEs in finance | Technical/mathematical only |

### 4.1 Empirical Testing and Validation Studies

**Black-Scholes Model Empirical Performance:**
- **Frontiers (2024)** empirical examination: No significant difference between theoretical and market prices for 7 of 9 stocks (call options); only 4 of 9 for puts
- Conclusion: BS suitable for call options but underperforms on puts in U.S. markets

**Distribution Testing:**
- Empirical returns exhibit skewness (negative for equities) and excess kurtosis
- Fat tails phenomenon: 5- to 7-sigma events occur more frequently than normal distribution predicts
- Black-Scholes underprices out-of-the-money options (misses tail risk)

**Volatility Observations:**
- Volatility surface not flat (violates BS constant volatility assumption)
- Volatility smile/smirk: Implied volatility increases for deep ITM and OTM options
- Time-varying volatility: Stochastic volatility models provide better fit

---

## 5. CRITICAL ASSUMPTIONS AND THEORETICAL JUSTIFICATIONS

### 5.1 Black-Scholes Assumptions

1. **Geometric Brownian Motion**: Stock prices follow continuous log-normal process
2. **Constant Volatility**: σ is fixed over option life
3. **No Arbitrage**: Markets are frictionless; no risk-free profit opportunities
4. **Efficient Markets**: All information reflected in prices; no market impact
5. **No Dividends** (simplification): Can be extended to dividend-paying stocks
6. **Risk-Free Borrowing**: Can borrow/lend at constant risk-free rate r
7. **European Options**: Exercise only at maturity (not American)
8. **Normal Returns**: Log-returns ~ N(μ, σ²)

**Theoretical Justification:**
- GBM: Ensures positive prices; analytical tractability; consistent with empirical log-returns (approximately)
- Constant volatility: Simplification; real volatility varies but averaging over periods provides approximation
- No arbitrage: Fundamental principle (violation implies unlimited profit)
- Risk-free rate: Standard assumption in infinite-liquid markets (Treasury rates)

### 5.2 Geometric Brownian Motion: Justifications and Limitations

**Why GBM?**
- **Returns Scale-Invariant**: E[dS/S] = μ dt (independent of level)—realistic
- **Positive Prices**: Log(S_t) ~ N ensures S_t > 0 (stocks cannot be negative)
- **Computational Ease**: Analytical solution available; Monte Carlo straightforward

**Empirical Justifications:**
- Logarithmic returns closer to normal than arithmetic returns
- Long-run returns approximately independent of starting price (scale invariance)
- Empirical evidence shows GBM reasonable approximation for many stocks over short-medium horizons

**Documented Limitations:**
1. **Volatility Clustering**: σ not constant; exhibits GARCH-type patterns
2. **Jump Discontinuities**: Overnight gaps; earnings announcements; market shocks
3. **Fat Tails**: Extreme events more frequent than normal distribution predicts
4. **Asymmetric Risk**: Downside volatility > upside volatility (leverage effect)
5. **Market Microstructure**: Bid-ask spreads, discrete tick sizes, intraday patterns not captured

### 5.3 Mean Reversion: Theoretical Foundations

**Economic Justification for Interest Rates:**
- Central banks target long-term equilibrium rates
- Deviations from equilibrium create arbitrage opportunities → reversion
- Equilibrium determined by real economic fundamentals and policy

**Economic Justification for Commodities:**
- Supply-demand equilibrium determines long-term price
- High prices incentivize production → excess supply → price decline
- Low prices reduce production → supply shortage → price rise

**Empirical Evidence:**
- Half-life of mean reversion (1/θ) estimated at 1–10 years depending on asset class
- Stronger evidence for interest rates and commodity prices than equities
- Equities less clearly mean-reverting; momentum and trends often dominate short-term

---

## 6. STOCHASTIC DIFFERENTIAL EQUATIONS: MATHEMATICAL FRAMEWORK

### 6.1 General SDE Formulation

For a process X_t:
```
dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
```

**Components:**
- **Drift Term** μ(X_t, t)dt: Deterministic trend
- **Diffusion Term** σ(X_t, t)dW_t: Stochastic component (Wiener process)

**Key Property:**
- Quadratic variation: [W]_t = t (not zero)
- This non-zero quadratic variation is why Itô's lemma differs from classical chain rule

### 6.2 Numerical Solution Methods

**Euler-Maruyama Discretization:**
```
X_{n+1} = X_n + μ(X_n, t_n)Δt + σ(X_n, t_n)√Δt Z_n
```

Where Z_n ~ N(0, 1)

**Milstein Scheme** (higher-order):
```
X_{n+1} = X_n + μΔt + σ√Δt Z_n + (1/2)σ(∂σ/∂x)[(Z_n² - 1)Δt]
```

**Accuracy:** Euler-Maruyama: O(Δt); Milstein: O(Δt²)

### 6.3 Monte Carlo Simulation in Derivatives Pricing

**Procedure:**
1. Discretize SDE for stock price (Euler method typical)
2. Simulate N paths to maturity T
3. Compute payoff for each path
4. Average payoffs; discount at risk-free rate

**Advantages over PDE:**
- Handles complex, path-dependent options
- Naturally extends to multi-dimensional problems (basket options)
- Parallelizable; suitable for modern computing

**Limitations:**
- Convergence: O(1/√N) (slow; requires many simulations)
- Bias from discretization (Δt effects)
- Variance reduction techniques necessary (antithetic sampling, control variates)

---

## 7. EMPIRICAL ANOMALIES AND MODEL LIMITATIONS

### 7.1 Volatility Smile and Skew

**Observation:**
- Black-Scholes assumes constant implied volatility across all strikes K
- Empirically, implied volatility σ_impl(K) varies with strike price

**Patterns by Asset Class:**

**Equities (Skew/Smirk):**
```
σ_impl(K) higher for low strikes (OTM puts, ITM calls)
σ_impl(K) lower for high strikes (OTM calls, ITM puts)
```
- Negative skew reflects crash risk (tail risk hedging premium)
- Post-1987 crash behavior

**Currencies (Smile):**
- U-shaped curve: σ_impl highest at deep OTM and ITM
- Reflects uncertainty on both sides (appreciation/depreciation)

**Root Causes:**
1. Jump Risk: Merton-type discontinuities not in BS model
2. Stochastic Volatility: Real volatility varies; smile arises from averaging over paths
3. Fat Tails: Return distribution leptokurtic; BS misses tail probability

### 7.2 Non-Normal Distributions: Fat Tails and Higher Moments

**Empirical Findings:**

**Kurtosis (Excess):**
- Normal distribution: kurtosis = 3
- Typical stock returns: excess kurtosis = 1–5 (fat tails)
- Example: 5-sigma events occur ~1x per 3,000 years (normal), but ~1x per 50–100 years (real markets)

**Skewness:**
- Equity returns: negative skewness (left tail) ~ -0.5 to -1.0
- Preference for downside crash risk over upside
- Black-Scholes treats up/down moves symmetrically (zero skewness)

**Implications for Option Pricing:**
- BS underprices OTM puts (crash protection) when negative skew present
- Overprices OTM calls
- Practical pricing: use higher implied volatility for low strikes

**Historical Example:**
- Long-Term Capital Management (1998): Ignored kurtosis risk
- Assumed normal distributions; 5-sigma event occurred in Russian crisis
- Massive losses despite prestigious founders (including Myron Scholes)

### 7.3 Volatility Clustering and Time-Varying Volatility

**GARCH Models:**
- Volatility exhibits autocorrelation (clustering)
- Large shocks followed by elevated volatility periods
- Standard formulation: σ_t² = α₀ + α₁ε²_{t-1} + β₁σ²_{t-1}

**Empirical Evidence:**
- Volatility is stochastic, not constant
- More complex than GBM's fixed σ
- Better captured by Heston or GARCH specifications

### 7.4 Jump Risk: Discontinuous Price Movements

**Evidence:**
- Overnight gaps (closing > opening or vice versa)
- Earnings announcements produce large discrete price jumps
- Market stress periods (e.g., Feb 2018 volatility spike, March 2020 COVID crash)

**Black-Scholes Limitation:**
- Assumes continuous paths (no jumps)
- Underprices tail risk and out-of-money options
- Jump-diffusion models (Merton) partially address

**Quantitative Impact:**
- Jump frequency λ typically 0.5–2.0 per year (asset dependent)
- Jump size distribution: log-normal with mean -5% to -10% (crash bias)
- Contribution to overall return variance: 10–40% depending on horizon

---

## 8. EXTENSIONS AND REFINEMENTS

### 8.1 Jump-Diffusion Extensions

**Bates Model (1996):**
- Combines Heston stochastic volatility + Merton jumps
- More flexible; better empirical fit than individual components
- Computational complexity increases (numerical integration required)

**Levy Processes:**
- Generalization allowing arbitrary jump structures
- Variance Gamma, Normalized Inverse Gaussian models
- Excellent fit to empirical return distributions; calibration difficult

### 8.2 Dividend-Adjusted Models

**Dividend Yield:**
- Modifies GBM: dS = (μ - δ)S dt + σS dW
- δ = continuous dividend yield
- Black-Scholes formula adjusted: discounting at r - δ for underlying

**American Options with Dividends:**
- Early exercise may be optimal to capture dividends
- Binomial trees or numerical PDE methods required

### 8.3 Interest Rate Models Beyond Vasicek

**Cox-Ingersoll-Ross (CIR, 1985):**
- Addresses negative rates: dr = κ(θ - r)dt + σ√r dW
- Ensures r ≥ 0 (reflection at zero)
- Non-negative processes; more complex solution

**Hull-White (1990):**
- Time-dependent parameters: dr = [θ(t) - α r]dt + σ(t)dW
- Exact match to initial term structure
- More flexible calibration to market data

---

## 9. IDENTIFIED GAPS AND OPEN PROBLEMS

### 9.1 Model Calibration

**Challenge:**
- Many models have 4–10 parameters requiring estimation
- Different estimation windows → different parameter values
- Inverse problem (implied parameters from option prices) ill-posed

**Current State:**
- Optimization algorithms (Levenberg-Marquardt, genetic algorithms)
- Bayesian MCMC methods gaining traction
- No universally accepted best practice

### 9.2 Model Selection

**Problem:**
- Which model? BS, Heston, Jump-Diffusion, Levy, others?
- Trade-off: Parsimony vs. empirical fit
- Context-dependent: Different models optimal for different purposes

**Literature Gap:**
- Limited comparative studies across models on standard datasets
- Model selection criteria (AIC, BIC) underexplored in options pricing

### 9.3 Stochastic Volatility Parameter Estimation

**Heston Model Issues:**
- Five parameters (S₀, μ, κ, θ, ξ, ρ)
- Correlation ρ particularly difficult to estimate precisely
- Small changes in ρ → large changes in option prices

### 9.4 Regime-Switching Models

**Empirical Observation:**
- Market behavior changes with economic regimes (boom vs. crisis)
- Constant parameters unrealistic
- Regime-switching models (Hamilton framework) emerging

**Current Gap:**
- Limited integration of regime-switching with stochastic volatility
- Computational challenges with likelihood estimation

### 9.5 Market Microstructure Effects

**Largely Ignored in Classical Models:**
- Bid-ask spreads and transaction costs
- Order book dynamics
- Discrete price movements (tick size constraints)
- Market impact of trades

**Research Direction:**
- Incorporating microstructure into derivatives pricing
- Effects on hedging and replication strategies

---

## 10. STATE OF THE ART: SUMMARY

### 10.1 Current Best Practices

**For European Options:**
1. **Heston Model**: Industry standard for practitioners
   - Closed-form characteristic function
   - Generates realistic volatility surface
   - Numerically robust calibration methods available

2. **Jump-Diffusion Models**: When tail risk critical
   - Merton (1976) jump-diffusion
   - Bates (1996) combined model for maximum flexibility

3. **Historical Baseline**: Black-Scholes still used for:
   - Simple products (ATM options)
   - Risk management (Greeks calculation)
   - Implied volatility as convention (not assumed reality)

**For American Options:**
- Binomial trees (Cox-Ross-Rubinstein)
- Finite difference PDE solvers
- Monte Carlo with optimal stopping (Longstaff-Schwartz)

**For Interest Rates:**
- Hull-White (one-factor or two-factor)
- CIR for term-structure consistency
- Shifted models to allow negative rates (post-2008)

### 10.2 Emerging Trends

1. **Machine Learning Integration:**
   - Neural networks for option pricing
   - Learning volatility surfaces from data
   - Automated parameter estimation

2. **Realistic Asset Models:**
   - Realized volatility frameworks
   - Jump-leverage coupling (not independent)
   - Time-varying jump intensities

3. **Rough Volatility:**
   - Recent evidence (Gatheral et al., 2014+) suggests volatility rougher than Brownian motion
   - Fractional Brownian motion dynamics
   - Better empirical fit to option prices

4. **Counterparty Risk:**
   - CVA (Credit Valuation Adjustment) integration
   - Bilateral pricing frameworks
   - Increasingly important post-2008

### 10.3 Consensus and Disagreement in Literature

**Broad Consensus:**
- Black-Scholes/GBM is insufficient alone for realistic pricing
- Mean reversion present in interest rates and some commodities
- Stochastic volatility essential for options markets
- Fat tails documented across asset classes

**Areas of Disagreement:**
- **Jump Frequency**: Debate on importance; evidence mixed
- **Model Complexity**: Whether additional parameters justify improved fit
- **Volatility Persistence**: How long does volatility clustering persist?
- **Mean Reversion Speed**: Estimates vary 1-10x across studies

---

## 11. REFERENCES AND SOURCES

### Seminal Papers
1. Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654.
2. Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141–183.
3. Vasicek, O. A. (1977). "An Equilibrium Characterization of the Term Structure." *Journal of Financial Economics*, 5(2), 177–188.
4. Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics*, 7(3), 229–263.
5. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies*, 6(2), 327–343.

### Extensions and Refinements
6. Merton, R. C. (1976). "Option Pricing When Underlying Stock Returns Are Discontinuous." *Journal of Financial Economics*, 3(1), 125–144.
7. Bates, D. S. (1996). "Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options." *Review of Financial Studies*, 9(1), 69–107.
8. Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). "A Theory of the Term Structure of Interest Rates." *Econometrica*, 53(2), 385–407.
9. Hull, J., & White, A. (1990). "Pricing Interest-Rate-Derivative Securities." *Review of Financial Studies*, 3(4), 573–592.

### Empirical Studies and Limitations
10. Fama, E. F., & French, K. R. (2004). "The Capital Asset Pricing Model: Theory and Evidence." *Journal of Economic Perspectives*, 18(3), 25–46.
11. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). "Volatility is Rough." *Quantitative Finance*, 18(6), 933–949. [arXiv:1410.3394]

### Computational and Theoretical Resources
12. Ito, K. (1951). "On stochastic differential equations." *Memoirs of the American Mathematical Society*, 4, 1–51.
13. Harrison, J. M., & Pliska, S. R. (1981). "Martingales and Stochastic Integrals in the Theory of Continuous Trading." *Stochastic Processes and Their Applications*, 11(3), 215–260.

### Recent Surveys and Reviews
14. Dmouj, A. (2006). "Stock Price Modelling: Theory and Practice." *VU Business Analytics*, (preprint).
15. Frontiers in Applied Mathematics and Statistics (2024). "Empirical Examination of the Black–Scholes Model: Evidence from the United States Stock Market."

### Web-Based Technical Resources Referenced
16. Columbia University Financial Engineering Notes: Black-Scholes, GBM, and Stochastic Calculus course materials.
17. QuantStart Tutorials: Detailed articles on Itô's Lemma, SDEs, and Heston implementation.
18. CQF (Certificate in Quantitative Finance) Online Resources: SDE fundamentals and martingale pricing.

---

## APPENDIX A: MATHEMATICAL NOTATION SUMMARY

| Symbol | Meaning |
|--------|---------|
| S_t | Stock price at time t |
| dS_t | Infinitesimal change in stock price |
| μ | Drift (expected return) |
| σ | Volatility (annualized) |
| W_t | Standard Wiener process (Brownian motion) |
| dW_t | Infinitesimal Wiener increment; N(0,dt) |
| C(S,t) | Call option value |
| P(S,t) | Put option value |
| K | Strike price |
| T | Time to maturity |
| r | Risk-free interest rate |
| N(x) | Cumulative standard normal distribution |
| E[·] | Expectation operator |
| E^Q[·] | Risk-neutral expectation |
| Var(·) | Variance |
| θ | Mean reversion speed |
| v_t | Variance (stochastic, in Heston) |
| λ | Poisson jump intensity |
| ρ | Correlation coefficient |

---

## APPENDIX B: KEY IMPLEMENTATION FORMULAS

### Black-Scholes Greeks

**Delta (Δ):** ∂C/∂S = N(d₁)

**Gamma (Γ):** ∂²C/∂S² = n(d₁)/(S·σ√(T-t))

**Theta (Θ):** ∂C/∂t = -S·n(d₁)·σ/(2√(T-t)) - r·K·e^(-r(T-t))·N(d₂)

**Vega (ν):** ∂C/∂σ = S·n(d₁)·√(T-t)

**Rho (ρ):** ∂C/∂r = K·(T-t)·e^(-r(T-t))·N(d₂)

Where n(x) = (1/√(2π))·e^(-x²/2) is the PDF of standard normal

### Heston Characteristic Function (for calibration)

```
φ(u; v, S, T) = exp(iuln(S) + iu(r-δ)T + ∫₀ᵀ κ(θ - v_s)λ ds)
```

Enables computation of option price via Fourier inversion

---

## APPENDIX C: DATASETS AND BENCHMARKS

**Common Empirical Datasets:**
- **S&P 500**: Daily/intraday pricing; 1926–present
- **Individual Stocks**: Extensive database (CRSP, Yahoo Finance)
- **Option Prices**: IVolatility, Refinitiv, Bloomberg terminals
- **Interest Rates**: Federal Reserve economic data (FRED)
- **Commodity Futures**: CME, CBOT historical data

**Standard Benchmarks:**
- European Call/Put pricing accuracy (BS baseline)
- Implied volatility surface fit (Heston, Jump-Diffusion)
- Greeks accuracy (hedge ratios, risk exposure)
- VaR/CVaR forecasting (tail risk models)

---

**Document Compilation Date:** December 2025
**Scope Completeness:** Comprehensive survey of foundational models through 2024
**Quality Assurance:** Cross-referenced with peer-reviewed sources and academic authoritative resources
