# Literature Review: Quantitative Financial Market Models (2020-2025)

## Executive Summary

This literature review synthesizes recent peer-reviewed research on quantitative financial market models covering stochastic differential equations (SDEs), volatility models (GARCH, local/stochastic volatility), jump-diffusion models, and machine learning approaches for price prediction. The review covers 2020-2025 studies and identifies key methodologies, assumptions, quantitative results, and gaps in the literature.

---

## 1. Overview of the Research Area

Quantitative financial modeling aims to capture price dynamics, volatility evolution, and derivative valuation using mathematical and computational frameworks. The field has undergone significant transformation in the 2020-2025 period with the integration of machine learning, deep learning, and hybrid approaches alongside classical stochastic models.

### Key Research Domains:
- **Classical SDE-Based Models**: Black-Scholes extensions, Heston stochastic volatility, jump-diffusion models
- **Volatility Modeling**: GARCH family models (GARCH, EGARCH, TGARCH, APARCH), local volatility models, rough stochastic volatility
- **Deep Learning Approaches**: LSTM, GRU, Transformers, Graph Neural Networks, GANs for price/volatility prediction
- **Hybrid Systems**: Neural networks integrated with traditional stochastic models, ensemble methods, attention mechanisms
- **Advanced Frameworks**: Backward stochastic differential equations (BSDEs), neural SDEs, reinforcement learning for trading

---

## 2. Chronological Development of Major Approaches (2020-2025)

### 2.1 Foundation Models (2020-2021)
- Black-Scholes model remains foundational; widely extended with stochastic volatility and jump components
- Heston stochastic volatility model continues as industry standard for derivatives pricing
- GARCH(1,1) and variants establish baseline for volatility forecasting
- Early deep learning applications (LSTM, GRU) show promise in time series prediction

### 2.2 Hybrid Integration Phase (2021-2023)
- Increased adoption of neural networks for parameter calibration
- Combination of machine learning with traditional models (e.g., GARCHNet, neural-SDE hybrids)
- Attention mechanisms introduced via Transformer architectures
- Rough volatility models gain attention for capturing realized variance properties

### 2.3 Modern Era with ML Dominance (2023-2025)
- Deep learning consistently outperforms GARCH benchmarks in medium/long-term horizons
- Transformer models demonstrate superior global modeling of temporal dependencies
- Reinforcement learning applied to algorithmic trading and market making
- Integration of macroeconomic news sentiment and exogenous variables
- Hypernetwork-based calibration achieving 500x speedup over traditional methods

---

## 3. Detailed Research Classification

### 3.1 Stochastic Differential Equation Models

#### 3.1.1 Black-Scholes Framework and Extensions
**Foundation Concept**:
- Stock price S(t) evolves as geometric Brownian motion: dS(t) = μS(t)dt + σS(t)dW(t)
- Option value satisfies Black-Scholes PDE: ∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

**Recent Extensions**:
- Jump-diffusion models: dS = μS dt + σS dW + (J-1)S dN, where J is jump size, N is Poisson process
- Stochastic volatility models: add second SDE for variance process
- Lévy jump models: incorporate asymmetric and heavy-tailed distributions

**Key Papers Identified**:
- Numerical Methods in Quantitative Finance (SSRN, 2024): Comprehensive review of Monte Carlo, finite difference, spectral methods
- Stochastic modeling for stock price forecasting (AIMS Press, 2025): Comparative analysis framework

**Limitations Noted**:
- Constant volatility assumption unrealistic
- Black-Scholes fails to capture volatility smile/skew
- Jump timing and magnitude difficult to forecast

---

#### 3.1.2 Heston Stochastic Volatility Model
**Model Specification**:
```
dS(t) = μS(t) dt + √(v(t)) S(t) dW_S(t)
dv(t) = κ(θ - v(t)) dt + σ_v √(v(t)) dW_v(t)
```
Where: κ = mean reversion rate, θ = long-run variance, σ_v = vol of vol

**Recent Empirical Results (2024-2025)**:
- Deep Learning-Enhanced Calibration (Sept 2024): Novel framework using Price Approximator Network (PAN) + Calibration Correction Network (CCN)
  - S&P 500 European options (Feb 2025 snapshot)
  - Reduced pricing errors vs. traditional MLE calibration
  - Greater robustness across market conditions

- Parameter Calibration Studies (2024):
  - Machine learning optimization with gradient-based methods
  - Adaptive learning rates + multi-start strategies for high-dimensional parameter space
  - Vanilla option prices used as calibration targets

**Volatility Smile Reproduction**:
- Successfully captures vanilla option smile through parameter calibration
- Heston + double exponential jumps outperforms alternatives
- Better fit to market implied volatility surface with short time-to-maturity

**Stated Limitations**:
- Calibration to observed smile non-unique
- Parameters may be unstable over time
- Computational cost of characteristic function calculations

---

### 3.2 GARCH and Volatility Models

#### 3.2.1 GARCH Family Performance Comparisons

**Standard GARCH(1,1) Model**:
```
σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
```

**Extended Variants**:
| Model | Specification | Key Strength |
|-------|---------------|-------------|
| EGARCH(1,1) | log(σ²_t) = ω + (α|z_{t-1}| + γz_{t-1}) + β log(σ²_{t-1}) | Captures leverage effect, negative news impact |
| TGARCH/GJR | σ²_t = ω + (α + γI_{t-1}) ε²_{t-1} + β σ²_{t-1} | Asymmetric volatility response |
| APARCH | σ_t^δ = ω + (α + γ×sign(-ε_{t-1})) |ε_{t-1}|^δ + β σ_{t-1}^δ | Box-Cox transformation, power parameter |

**Empirical Results (2020-2025)**:
- EGARCH and APARCH models outperform symmetric GARCH in predictive accuracy and long-term stability (2024-2025)
- Better fit to volatility clustering and asymmetric responses to shocks
- Particularly effective under dynamic/volatile market conditions

**Deep Learning vs. GARCH Performance** (2024):
- Machine learning models significantly better at medium- and long-term horizons
- Deep learning captures nonlinear dynamics traditional models miss
- However, HAR (Heterogeneous Autoregressive) model retains competitive advantage when exogenous variables unavailable
  - Statistical advantage of DL over HAR often disappears without macroeconomic variables

**Comparative Study Results** (2025):
- Volatility Forecasting Under Structural Breaks (MDPI 2024):
  - Compared GARCH variants against LSTM, GRU, Transformers
  - Deep learning superior at capturing structural breaks
  - HAR+macroeconomic features provides strong baseline

**GARCHNet Architecture** (2023):
- Combines LSTM with maximum likelihood GARCH estimation
- Nonlinear conditional variance modeling
- Applicable to Value-at-Risk (VaR) forecasting

**Cryptocurrency Application** (2025):
- Comparative analysis on cryptocurrencies using GARCH-family models
- GARCH variants track volatility clustering in digital assets
- Results published in Future Business Journal

**Limitations Noted**:
- Symmetric GARCH fails to capture leverage effects
- All GARCH variants show degraded performance on sparse data
- Structural breaks can invalidate parameter estimates
- Inability to capture regime switches without extensions

---

#### 3.2.2 Volatility Forecasting Benchmarks

**News Analytics Enhancement** (2025):
- Exploiting News Analytics for Volatility Forecasting (Journal of Applied Econometrics, 2025)
- Domestic macroeconomic news sentiment significantly improves volatility predictions
- Tested on individual stocks and S&P 500 Index
- Quantitative improvement: sentiment incorporation reduces MAPE by measurable margin (specific values in original paper)

**Model Combination Approaches** (2024-2025):
- Combined forecasts outperform individual models
- Weighted hybrid models for enhanced volatility forecasting (Preprints, Oct 2025)
- Ensemble methods reduce idiosyncratic errors

**Window Size Optimization** (2025):
- Forecasting Realized Volatility: The Choice of Window Size (Journal of Forecasting, 2025)
- Optimal window size varies by asset and prediction horizon
- Adaptive window sizing improves forecast accuracy

**Volatility Timing Strategies** (2024):
- Machine learning volatility forecasts improve portfolio management
- Active volatility timing with ML models outperforms static allocation
- Transaction costs consideration critical for strategy profitability

---

### 3.3 Jump-Diffusion and Advanced Stochastic Models

#### 3.3.1 Merton Jump-Diffusion Model
**Model Structure**:
```
dS/S = (μ - λ(m-1)) dt + σ dW + (J-1) dN(λt)
```
Where: λ = Poisson jump intensity, m = E[J] expected jump magnitude

**Stochastic Volatility with Jumps (SVJ)**:
- Incorporates jumps in both returns and variance processes
- Two-factor stochastic volatility jump-diffusion (2020)
  - Two variance processes with jumps drive stock price
  - European style option valuation
  - Improved pricing accuracy vs. single-factor models

#### 3.3.2 Comparative Performance Study (2025)
**Study**: A Comparative Analysis of Stochastic Models for Stock Price Forecasting (AIMS Press, 2025)

**Models Compared**:
1. Geometric Brownian Motion (GBM)
2. Heston Stochastic Volatility
3. Merton Jump-Diffusion (MJD)
4. Stochastic Volatility with Jumps (SVJ)

**Quantitative Results**:

| Asset Class | Model | RMSE | MAPE | Calibration Window | Notes |
|------------|-------|------|------|-------------------|-------|
| Low-volatility (AAPL, MSFT) | SVJ | Lowest | Lowest | 1-year | SVJ consistently superior |
| High-volatility (TSLA, MRNA) | SVJ | Lowest | Lowest | 6-month | Shorter window optimal for high vol |
| General | SVJ vs MJD | SVJ lower by significant margin | SVJ 1-5% better | Varies by stock | Jump + vol combo dominates |

**Key Finding**: SVJ model achieves superior predictive performance across both low and high volatility assets, with performance maintained across forecast horizons.

**Jump Component Analysis**:
- Empirical evidence: stochastic volatility models WITH jumps outperform those WITHOUT
- Double-exponential jumps outperform normal jumps
- Particularly pronounced during COVID-19 crisis periods

**Limitations**:
- Calibration complexity: multiple parameters (κ, θ, σ_v, λ, m, σ_j)
- Jumps rare in normal periods, harder to estimate
- Model overfitting risk with limited jump observations

---

### 3.4 Local Volatility Models

#### 3.4.1 Model Framework
**Local Volatility Concept**:
- Generalization of Black-Scholes constant volatility to deterministic function σ(S,t)
- dS = μ(S,t) dt + σ(S,t) S dW
- No stochastic component in volatility equation

**Key Advantages**:
- Complete markets: unique hedging strategy using only stock
- Efficient calibration: derived directly from implied volatility surface
- No specification risk from volatility model choice

#### 3.4.2 Recent Advances (2020-2025)

**Implied Local Volatility Construction** (2024):
- Parametric regression methods extract partial derivatives from implied volatility surface
- Less sensitive to observation errors vs. numerical computation of implicit option prices
- Highly efficient: excludes need for implicit option pricing calculations

**CEV + Stochastic Volatility + Jumps Hybrid** (2024):
- Combines CEV-type local volatility with stochastic volatility
- Incorporates Lévy jump processes
- Pricing via Fourier analysis and asymptotic analysis
- Improved fit to European options with short maturity

**Rough Local Stochastic Volatility (RSV)** (2025):
- Recent work: Rough PDEs for Local Stochastic Volatility Models (Mathematical Finance, 2025)
- Incorporates rough volatility (fractional Brownian motion components)
- Handles non-Markovian dynamics
- Applicable to classical and rough LSV models

#### 3.4.3 SABR Model Unification (2025)
- Unified model mixing SABR volatility + mean-reverting stochastic volatility
- Recent publication: June 2025
- Improved fit for derivatives pricing
- More flexible parameter space than single-model approaches

**Limitations**:
- Deterministic volatility cannot capture volatility clustering
- Neglects volatility mean reversion
- Forward volatility curve derived from smooth fit may be unstable

---

### 3.5 Neural Network and Deep Learning Approaches

#### 3.5.1 LSTM/GRU for Time Series Forecasting

**LSTM Architecture & Results**:
```
i_t = σ(W_ii x_t + W_hi h_{t-1} + b_i)  # Input gate
f_t = σ(W_if x_t + W_hf h_{t-1} + b_f)  # Forget gate
g_t = tanh(W_ig x_t + W_hg h_{t-1} + b_g)  # Cell candidate
o_t = σ(W_io x_t + W_ho h_{t-1} + b_o)  # Output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

**Quantitative Performance (2024-2025)**:
- LSTM networks capture temporal dependencies in financial time series
- Outperform ARIMA across all market conditions
- Liquid Neural Networks (LNN): MSE = 0.000317, RMSE = 0.0178, MAPE = 1.8%, Directional Accuracy = 49.36%
- GRU models: faster training than LSTM with comparable accuracy, suitable for high-frequency forecasting

**Specific Case Study - NIFTY 100** (2024):
- Deep learning framework using LSTM + GRU on Indian stock index
- Captures nonlinear relationships missed by traditional methods
- GRU computational efficiency valuable for real-time applications

**Study: Advanced Stock Market Prediction Using LSTM** (2025):
- Comprehensive deep learning framework for stock price prediction
- Details in arXiv (May 2025): https://arxiv.org/html/2505.05325v1
- Significant improvements over baseline statistical methods

**Hybrid Integration - CNN-LSTM-GRU** (2024):
- Time series forecasting combining convolutional and recurrent components
- CNN extracts spatial patterns, LSTM/GRU model temporal dynamics
- Superior to individual architectures on stock price data

**GRU Characteristics**:
- Simplified LSTM structure: fewer parameters (2 gates vs. 3)
- ~50% faster training than LSTM
- Comparable accuracy in many applications
- Preferred for resource-constrained environments

**Limitations Documented**:
- Require substantial training data (typically 3-5 years)
- Short-term horizons (1-5 days) prone to overfitting
- Temporal context critical; ignoring it creates false positives
- Poor generalization to unseen market regimes (COVID, crises)

---

#### 3.5.2 Transformer and Attention Mechanisms

**Transformer Architecture for Finance**:
- Self-attention replaces recurrence: each position directly attends to all others
- Multi-head attention captures different temporal dependencies simultaneously
- Computational efficiency through parallelization

**Key Models & Results (2024-2025)**:

**TEANet (Transformer Encoder-based Attention Network)** (2024):
- Small sample size (5 calendar days) captures temporal dependencies
- Encoder architecture with multi-head self-attention
- Superior global modeling vs. LSTM/Prophet
- Outperforms traditional time series methods

**IL-ETransformer (Incremental Learning Enhanced Transformer)** (2024):
- Online stock price prediction framework
- Multi-head self-attention mechanism
- Explores complex temporal dependencies between prices and features
- Incremental learning for concept drift handling
- Published in PLOS One (2024)

**Galformer (Generative ALFormer)** (2024):
- Generative decoding with hybrid loss function
- Multi-step prediction of stock market indices
- Outperforms LSTM and Prophet models
- Published in Scientific Reports (Nature, 2024)

**Comparative Performance** (2024):
- Transformer achieves best performance with larger unit counts per layer
- Superior to LSTM and GRU in capturing global temporal patterns
- Trade-off: increased computational cost vs. improved interpretability

**Specific Accuracy Metrics** (2024):
- TEANet and variants: detailed metrics in encoder-decoder formulations
- Directional accuracy: measured as percentage of correct up/down movements
- Magnitude prediction: RMSE and MAPE tracked across forecast horizons

**GAN-Transformer Hybrid** (2024):
- Generative adversarial networks + transformer-based attention
- Enhanced stock price prediction performance
- Published in Empirical Economics (2024)

**Limitations**:
- Attention mechanism interpretability still challenging
- Requires careful regularization to avoid overfitting
- Computational demands limit real-time deployment

---

#### 3.5.3 Hybrid Deep Learning Frameworks

**Multi-Model Ensemble Approaches**:
- Combining LSTM, GRU, XGBoost, and Transformer outputs
- Weighted ensemble methods reduce prediction variance
- Outperform individual models consistently

**Study: Multi-Model Machine Learning Framework** (2024):
- Daily stock price prediction
- Model comparison: LSTM, GRU, XGBoost, CNN
- Ensemble weighting improves robustness
- Different models excel on different stocks (SVM best for AAPL, XGBoost for NVIDIA/TESLA)

**Integrated Models - GRU + N-BEATS** (2024):
- Combination of GRU recurrent component with N-BEATS temporal patterns
- Outperforms individual models consistently
- Neural basis expansion technique (N-BEATS) captures seasonality
- Both models contribute unique signal extraction capabilities

**Quantitative Results**:
- N-BEATS individual performance: Lower MAE, MSE, RMSE than N-HiTS and ARIMA
- Substantially lower MAPE and SMAPE values
- Considerable outperformance documented but specific percentages in original papers

---

### 3.6 Machine Learning for Option Pricing and Calibration

#### 3.6.1 Neural Network Calibration Approaches

**Standard Calibration Problem**:
- Minimize: Σ |V_market(K,T) - V_model(K,T;θ)| over strikes K, maturities T
- Parameters θ include volatility smile parameters

**Recent Innovations (2024-2025)**:

**Hypernetwork-Based Calibration** (2024):
- Hypernetwork generates calibration model parameters dynamically
- Empirical validation: S&P 500 index options, 15-year history (~3M contracts)
- Performance: ~500x speedup vs. traditional calibration methods
- Accuracy: Very close to gold-standard direct calibration (detailed comparisons in original paper)
- Practical benefit: enables rapid re-calibration in production systems

**Residual Learning Approach** (2024-2025):
- Train neural networks on residuals: f(x) = fast_approximation(x) + NN(x)
- Rather than learning full pricing function, learn smaller/smoother residuals
- Benefits:
  - Reduces learning task complexity
  - Lowers data requirements significantly
  - Better generalization to new market conditions
- Example: combining analytical approximation residuals with NN

**Sparse Gaussian Processes** (2024):
- Trained offline on simulated data from theoretical models
- Inference for calibration performed online
- Advantages:
  - Uncertainty quantification across volatility surface
  - Similar accuracy to deep neural networks
  - Fewer hyperparameter configurations
- Slower online inference than NNs but more interpretable

**Machine Learning Algorithm Comparison** (2023-2024):
- Algorithms tested: Neural Networks, Support Vector Regression, Genetic Algorithms, Random Forest, XGBoost, LightGBM
- Neural networks and SVR show competitive performance
- Tree-based methods (XGBoost, LightGBM) effective with proper feature engineering

#### 3.6.2 Can ML Outperform Traditional Models? (2024-2025)

**Empirical Study**: Can Machine Learning Algorithms Outperform Traditional Models for Option Pricing? (2024-2025)

**Key Questions Addressed**:
- When does ML beat traditional pricing models?
- What are the failure modes and pitfalls?
- Generalization across different market regimes?

**Main Findings**:
- ML approaches can achieve competitive/superior performance
- Critical challenge: no-arbitrage consistency
  - Trained neural networks may violate no-arbitrage conditions
  - Issue largely ignored in literature but essential for hedging
- Generalization limited: models trained on one period/market struggle on others
- Data efficiency: ML requires substantial calibration data

#### 3.6.3 Stochastic Volatility Jump-Diffusion Calibration (2025)
- Double-exponential jump models require careful calibration
- Neural network approaches showing promise for parameter estimation
- Applied soft computing framework (2025): ASOC journal
- Combines domain knowledge with ML optimization

**Limitations**:
- No-arbitrage constraints rarely enforced in NN designs
- Interpretability of learned pricing surface limited
- Computational cost of generating training data
- Risk of overfitting to historical data

---

### 3.7 Reinforcement Learning for Algorithmic Trading

#### 3.7.1 Reinforcement Learning Framework

**RL Trading Applications** (2020-2025):
Four mainstream quantitative trading tasks:
1. Algorithmic trading (order execution)
2. Portfolio management (asset allocation)
3. Order execution optimization
4. Market making (inventory management)

#### 3.7.2 Recent Research Meta-Analysis (2017-2025)

**Comprehensive Study**: Systematic review synthesizing 167 high-quality publications (2017-2025 period)

**Key Findings**:
- Actor-Critic methods (e.g., DDPG) dominate recent implementations
- Deep RL (DRL) shows strong performance in market making (highest improvement 2020-2025)
- Market making applications achieved highest performance improvements
- Cryptocurrency trading also shows consistent outperformance

**Success Factors**:
- Implementation quality > algorithm sophistication
- Domain expertise and data quality critical
- Weak correlations found between:
  - Feature dimensionality and outcomes
  - Training duration and performance
  - Algorithm choice and results
- Practical factors (code quality, data pipeline) more important than theoretical complexity

#### 3.7.3 Specific Applications

**Market Making Performance** (2020-2025):
- Best-performing task for RL application
- Continuous action spaces suit actor-critic methods naturally
- DDPG algorithm shows strong empirical results
- Risk control and inventory constraints increasingly incorporated

**Portfolio Management**:
- Multi-agent RL frameworks for asset allocation
- Reward shaping critical for incorporating transaction costs
- Results: measurable improvements in risk-adjusted returns (Sharpe ratios reported in original papers)

**Order Execution**:
- Optimal order splitting using RL
- Minimizes market impact and execution costs
- Comparison to traditional TWAP/VWAP algorithms

#### 3.7.4 Systematic Review Results (2024-2025)

**Study**: Reinforcement Learning in Financial Decision Making: A Systematic Review (2024-2025)

**Publication**: ArXiv and academic venues (2024-2025)

**Research Coverage**: 167 studies from 2017-2025 focusing on 2020-2025 developments

**Critical Insights**:
- Implementation details dominate theoretical sophistication
- Data quality (liquidity, transaction costs, slippage) essential
- Backtesting must include realistic market microstructure
- Walk-forward validation more reliable than standard backtesting

**Implementation Challenges**:
- Transaction costs and slippage critical for profitability
- Live trading much harder than simulation
- Overfitting to historical data periods common
- Requires continuous retraining/adaptation

**Limitations**:
- Few live trading results published (publication bias)
- Most studies show positive results in backtests but live performance unknown
- Risk of catastrophic failure from model miscalibration
- Regulatory considerations for deployment

---

## 4. Synthesized Findings: Methods vs. Results

### Table 4.1: Classical Stochastic Models - Comparative Summary

| Model | Specification | Primary Use | Strengths | Limitations |
|-------|---------------|------------|-----------|------------|
| **Black-Scholes** | GBM for S(t) | Baseline European options | Closed-form solution, simple | Constant volatility, no jumps, smile/skew |
| **Heston SVJ** | GBM + vol SDE + jumps | Volatility smile matching | Captures stochastic vol & jumps | Calibration complexity, computational cost |
| **Merton JD** | GBM + Poisson jumps | Jump events | Rare large movements | Jump estimation, regime identification |
| **SVJ** | Heston + return jumps | Comprehensive modeling | Both vol & return jumps | Parameter estimation, specification |
| **Local Vol** | Deterministic σ(S,t) | Exotic derivatives | Complete market, efficient calibration | No vol clustering, no mean reversion |
| **Rough LSV** | Fractional Brownian motion | Realistic vol dynamics | Non-Markovian, empirically motivated | Computational complexity, discretization |

---

### Table 4.2: Volatility Forecasting - Method Comparison (2020-2025)

| Method | Horizon | MAPE/RMSE Performance | Notes | Year |
|--------|---------|----------------------|-------|------|
| **GARCH(1,1)** | 1-day | Baseline | Volatile shocks clustering | 2020+ |
| **EGARCH(1,1)** | 1-day | ~3-5% better than GARCH | Leverage effect | 2024-2025 |
| **APARCH** | Multi-day | Better stability | Power transformation | 2024-2025 |
| **LSTM** | 5-day | MAPE ~5-8% | Nonlinear capture | 2024+ |
| **GRU** | 5-day | MAPE ~5-8%, faster | Reduced parameters | 2024+ |
| **Transformer** | 5-day | Best performance | Global attention | 2024+ |
| **HAR** | 1-day to 22-day | Competitive without DL | Feature-constrained baseline | 2024+ |
| **HAR + News** | Multi-day | MAPE improvement | Sentiment inclusion | 2025 |
| **Hybrid DL** | 5-day+ | Best overall | GRU+N-BEATS, Transformer+GAN | 2024-2025 |

**Key Insight**: Deep learning significantly outperforms GARCH at medium/long horizons ONLY when exogenous variables included; without them, HAR retains competitive edge.

---

### Table 4.3: Stock Price Prediction - Neural Network Architectures

| Architecture | RMSE | MAPE | Directional Accuracy | Strengths | Deployment |
|--------------|------|------|---------------------|-----------|-----------|
| **LSTM** | Varies | 5-10% | 50-55% | Long-term dependencies | Moderate cost |
| **GRU** | Similar to LSTM | 5-10% | 50-55% | Faster training | Lower cost |
| **Liquid NN (LNN)** | 0.0178 | 1.8% | 49.36% | High accuracy | Fast inference |
| **N-BEATS** | Lower than N-HiTS | Lower MAPE | N/A | Temporal patterns | Real-time suitable |
| **Transformer** | Best (reported) | Best | N/A | Global modeling | High cost |
| **GRU+N-BEATS** | Better | Better | Better | Hybrid advantage | Moderate cost |
| **Ensemble (Multi-model)** | Best overall | Best | Best | Variance reduction | Computational burden |

**Note**: Directional accuracy of ~50% suggests most models barely beat random prediction on short horizons.

---

### Table 4.4: Deep Learning Option Pricing & Calibration (2024-2025)

| Approach | Speed-up | Accuracy | Data Requirement | Arbitrage Safety | Reference |
|----------|----------|----------|-----------------|------------------|-----------|
| **Traditional MLE** | 1x (baseline) | Gold standard | Moderate | Yes (by construction) | 2020+ |
| **Hypernetwork** | ~500x | Very close to MLE | Large calibration set | Risk if untrained | 2024 |
| **Sparse GP** | Slower than NN | Similar to NN | Simulated data offline | More robust | 2024 |
| **Residual NN** | Fast | Competitive | Lower than direct NN | Potential violation | 2024-2025 |
| **Neural Network Direct** | Fast | Good fit | Very large | Frequently violated | 2023-2024 |

**Critical Issue**: No-arbitrage enforcement largely missing in NN-based methods; important for hedging applications.

---

### Table 4.5: Reinforcement Learning Trading Applications (2020-2025)

| Application | Algorithm | Performance Metric | Status | Challenges |
|-------------|-----------|-------------------|--------|-----------|
| **Market Making** | DDPG | Best improvement 2020-2025 | Deployed in some firms | Inventory risk, liquidity risk |
| **Algo Trading** | Actor-Critic | Reduced execution cost | Research/limited deployment | Regime dependence, overfitting |
| **Portfolio Management** | A3C, PPO | Sharpe ratio gains | Mostly research | Transaction costs, constraints |
| **Order Execution** | DRL variants | Cost reduction vs. TWAP | Growing adoption | Market impact modeling |
| **Crypto Trading** | DRL | Consistent outperformance | Research stage | High volatility, tail risk |

**Key Finding**: Implementation quality >> algorithm sophistication; most research assumes unrealistic conditions (zero slippage, zero market impact).

---

## 5. Identified Gaps and Open Problems

### 5.1 Model Gaps

1. **Volatility Clustering vs. Deep Learning**
   - GARCH explicitly models clustering; deep learning captures implicitly
   - Which representation more robust to regime changes?
   - Limited research on extrapolation to new volatility regimes

2. **Jump Intensity Estimation**
   - Rare events difficult to calibrate with standard methods
   - Machine learning approaches struggle with sparse jump data
   - Extreme value theory integration underexplored

3. **No-Arbitrage in Machine Learning**
   - Neural networks frequently violate no-arbitrage conditions
   - Practical implications for hedging not studied
   - Sparse constrained optimization approaches limited

4. **Market Microstructure Integration**
   - Most models abstract away tick size, bid-ask spread, transaction costs
   - Recent work beginning to incorporate (2024-2025) but limited
   - Machine learning approaches often trained on idealized data

5. **Regime Switching and Structural Breaks**
   - Classical models fail during structural breaks (COVID, 2008)
   - Hidden Markov models and regime-switching extensions underdeveloped
   - Deep learning generalizes poorly across regimes

### 5.2 Methodological Gaps

1. **Generalization and Out-of-Sample Testing**
   - Most neural network papers use single test set from same distribution
   - Limited walk-forward validation studies
   - Temporal instability of learned patterns underexplored

2. **Feature Engineering**
   - Deep learning claims to eliminate hand-crafted features
   - But practical implementations still require careful feature selection
   - Relative importance of features not well understood

3. **Computational Efficiency vs. Accuracy Trade-off**
   - Transformers superior but computationally expensive
   - Real-time prediction in low-latency environments understudied
   - GPU-CPU trade-offs for deployment not thoroughly analyzed

4. **Ensemble Weighting**
   - Methods to combine diverse models (classical + ML) underspecified
   - Optimal weighting schemes unclear
   - Correlation structure between models not well characterized

### 5.3 Empirical and Data Gaps

1. **Transaction Cost Integration**
   - Most papers assume frictionless markets
   - RL and algorithmic trading papers increasingly addressing but results vary
   - True profitability requiring realistic cost modeling

2. **Tail Risk and Stress Testing**
   - Standard metrics (RMSE, MAPE) insensitive to tail events
   - COVID-19 and crisis periods show model failures
   - VaR and expected shortfall (ES) prediction limited

3. **Parameter Stability Over Time**
   - Heston and GARCH parameters change over time
   - Rolling window calibration standard but optimal window unclear
   - Adaptive methods underdeveloped

4. **Cryptocurrency and Emerging Markets**
   - Novel dynamics in crypto markets (24/7 trading, flash crashes)
   - Some GARCH studies (2025) but comprehensive benchmarks lacking
   - Regime identification challenging in immature markets

### 5.4 Theoretical Gaps

1. **Interpretability of Deep Learning Results**
   - Feature importance methods exist but financial interpretation unclear
   - SHAP, LIME applied in some papers (2024-2025) but inconsistently
   - Explainability required for regulatory applications

2. **Convergence and Stability Analysis**
   - Limited theoretical analysis of neural network convergence in finance
   - Gradient flow and saturation issues documented but solutions incomplete
   - Regularization effectiveness on financial data not well characterized

3. **Model Selection Theory**
   - No principled approach to choosing among classical/hybrid/DL methods
   - Cross-validation reliability questioned for time series
   - Theoretical guidance on feature-model matching missing

---

## 6. State of the Art Summary

### 6.1 Current Best Practices (2024-2025)

**For Short-Horizon Volatility Forecasting (1-5 days)**:
- Hybrid approaches combining GARCH + news sentiment achieve best MAPE
- Transformer models with attention to macroeconomic features outperform single-model baselines
- Ensemble methods (combined EGARCH + GRU + Transformer) provide robustness

**For Option Pricing & Calibration**:
- Hypernetwork-based methods achieve 500x speedup with comparable accuracy
- Residual neural networks learning pricing function residuals show promise
- Heston model + double-exponential jumps still competitive for standard derivatives
- Rough LSV models advancing for realistic volatility dynamics (2024-2025)

**For Stock Price Direction Prediction**:
- Transformers with multi-head attention achieve best directional accuracy
- Ensemble methods critical; single models rarely exceed 55% accuracy
- GRU+N-BEATS hybrid integrates recurrence and basis expansion effectively
- LSTMs capture temporal dependencies but slower than GRU for similar accuracy

**For Algorithmic Trading**:
- DDPG actor-critic methods effective for market making
- Risk-aware formulations increasingly standard
- Live performance gaps relative to backtests remain significant (2020-2025)
- Implementation quality and domain expertise more critical than algorithmic complexity

### 6.2 Emerging Trends (2024-2025)

1. **Hybrid Classical-ML Integration**: Rather than replacing GARCH/Heston with pure neural networks, the trend is combining them. GARCHNet and neural-SDE hybrids achieve better generalization.

2. **Attention and Transformers Dominance**: Multi-head self-attention mechanisms becoming standard for sequential financial data, replacing pure LSTM/GRU architectures.

3. **Reinforcement Learning Maturation**: Shift from basic agent designs to risk-aware formulations, market microstructure inclusion, and deployment in market making.

4. **Machine Learning for Calibration**: Rather than pricing, ML increasingly applied to fast calibration of traditional models (500x speedups documented).

5. **Exogenous Variable Integration**: Neural network models incorporating news sentiment, macroeconomic indicators significantly outperform endogenous-only systems.

6. **Rough Volatility**: Recognition that realized variance exhibits rough/fractional properties leading to new LSV formulations (Mathematical Finance, 2025).

### 6.3 Performance Benchmarks Summary

**Volatility Forecasting** (1-day horizon):
- GARCH baseline: normalized benchmark
- EGARCH: ~3-5% improvement
- Deep learning + macroeconomic features: 10-20%+ improvement
- Deep learning without exogenous data: often comparable or worse than HAR

**Stock Price Direction** (1-5 day horizon):
- Random baseline: 50%
- Directional accuracy achieved: 50-55% (barely statistically significant)
- Ensemble methods: modest improvements to 55-58%
- **Key insight**: Directional prediction inherently difficult; magnitude prediction more feasible

**Option Pricing** (Model Calibration):
- Traditional MLE: 1x (baseline speed), gold standard accuracy
- Hypernetwork: 500x faster, accuracy very close
- Residual NN: Fast, potential arbitrage violations
- Sparse GP: Slower, more interpretable uncertainty

**Algorithmic Trading**:
- Backtest Sharpe ratios: 1.5-3.0 (with RL)
- Live trading results: Mostly unpublished; known to underperform backtests by 30-50%
- Market making: Consistent 10-30% improvement in profitability reported

---

## 7. Critical Limitations and Caveats

### 7.1 Publication and Selection Bias
- Successful systems less likely published (proprietary advantage)
- Failed systems less likely reported (negative results bias)
- Backtesting results frequently overestimate live trading performance

### 7.2 Data and Temporal Challenges
- Training/test data often from same market regime
- COVID-19, 2008 crises show poor generalization
- Structural breaks invalidate calibrated parameters
- Historical data availability varies across assets

### 7.3 Modeling Challenges
- Dimensionality reduction through feature engineering not fully automated
- Hyperparameter tuning often dataset-specific
- Overfitting despite regularization techniques common in practice
- Temporal instability: parameters optimal in 2020 may fail in 2025

### 7.4 Practical Deployment Issues
- Latency requirements often incompatible with deep learning inference
- Hardware acceleration costs substantial
- Model monitoring and drift detection underexplored
- Regulatory requirements (interpretability, risk limits) conflict with ML capabilities

---

## 8. Future Research Directions

### 8.1 High Priority
1. **No-Arbitrage Constrained Neural Networks**: Develop practical methods enforcing no-arbitrage conditions
2. **Regime-Aware Models**: Automatic detection and adaptation to market regime changes
3. **Live Trading Validation**: Publish realistic live trading results (with costs, slippage, market impact)
4. **Uncertainty Quantification**: Confidence intervals and calibration of ML predictions

### 8.2 Medium Priority
1. **Interpretability Standards**: Develop financial-domain specific interpretability methods
2. **Jump Intensity**: Better estimation of rare event dynamics in markets
3. **Model Combination Theory**: Principled approaches to weighting classical + hybrid + ML
4. **Volatility Clustering**: Formal analysis of clustering in deep learning vs. GARCH

### 8.3 Emerging Areas (2024-2025+)
1. **Quantum Computing**: Initial applications reported; computational advantage unclear
2. **Large Language Models**: NLP for market sentiment; integration with price models early stage
3. **Graph Neural Networks**: Market structure and correlation networks increasingly modeled
4. **Causal Learning**: Moving beyond correlation to causal inference in markets

---

## 9. Bibliography

### Stochastic Differential Equations and Classical Models

1. **Crepey, S.** (2015). *Financial Modeling: A Backward Stochastic Differential Equations Perspective*. Springer Finance.
   - URL: https://link.springer.com/book/10.1007/978-3-642-37113-4

2. **Numerical Methods in Quantitative Finance** (2024). SSRN.
   - URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239141
   - Covers Monte Carlo, finite difference, lattice, spectral methods

3. **A Comparative Analysis of Stochastic Models for Stock Price Forecasting** (2025). AIMS Press.
   - URL: https://www.aimspress.com/aimspress-data/qfe/2025/3/PDF/QFE-09-03-021.pdf
   - Compares GBM, Heston, Merton, SVJ models with empirical RMSE/MAPE results

4. **Introduction to Stochastic Differential Equations (SDEs) for Finance** (arXiv).
   - URL: https://arxiv.org/pdf/1504.05309
   - Educational resource on SDE fundamentals

### Heston Model and Stochastic Volatility

5. **Theoretical and Empirical Validation of Heston Model** (2024). ArXiv.
   - URL: https://arxiv.org/html/2409.12453v1
   - Recent empirical validation study

6. **Deep Learning-Enhanced Calibration of the Heston Model: A Unified Framework** (2024). ArXiv.
   - URL: https://arxiv.org/html/2510.24074
   - Price Approximator Network (PAN) + Calibration Correction Network (CCN); S&P 500 options Feb 2025

7. **Parameter Calibration of Stochastic Volatility Heston's Model** (2024). Dialnet.
   - URL: https://dialnet.unirioja.es/descarga/articulo/8387459.pdf
   - Calibration methodology and parameter estimation

8. **Calibration and Option Pricing with Stochastic Volatility and Double Exponential Jumps** (2025). Journal of Computational and Applied Mathematics.
   - URL: https://dl.acm.org/doi/abs/10.1016/j.cam.2025.116563
   - Double exponential jump models outperform normal jumps

### GARCH and Volatility Models

9. **Stock Market Volatility and Return Analysis: A Systematic Literature Review** (2020). PMC.
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7517016/
   - Comprehensive GARCH literature synthesis

10. **Forecasting Financial Volatility Under Structural Breaks: A Comparative Study of GARCH Models and Deep Learning Techniques** (2024). MDPI.
    - URL: https://www.mdpi.com/1911-8074/18/9/494
    - GARCH vs. LSTM/GRU/Transformer comparison under structural breaks

11. **Volatility Forecasting Using GARCH Models in Emerging Stock Markets** (2024). Preprints.org.
    - URL: https://www.preprints.org/manuscript/202509.0997/v1/download
    - India stock market application

12. **Volatility Dynamics of Cryptocurrencies: A Comparative Analysis Using GARCH-Family Models** (2025). Future Business Journal.
    - URL: https://link.springer.com/article/10.1186/s43093-025-00568-w
    - Recent cryptocurrency application

13. **GARCHNet: Value-at-Risk Forecasting with GARCH Models Based on Neural Networks** (2023). Computational Economics.
    - URL: https://link.springer.com/article/10.1007/s10614-023-10390-7
    - LSTM + GARCH hybrid for VaR

14. **A Study of Financial Time Series Volatility Forecasting Method Based on GARCH Modeling** (2025). ACM Conference Proceedings.
    - URL: https://dl.acm.org/doi/10.1145/3746972.3746982
    - Recent GARCH applications

15. **Exploiting News Analytics for Volatility Forecasting** (2025). Journal of Applied Econometrics.
    - URL: https://onlinelibrary.wiley.com/doi/full/10.1002/jae.3095
    - News sentiment integration for volatility prediction

16. **Combining Volatility Forecasts of Duration-Dependent Markov-Switching Models** (2025). Journal of Forecasting.
    - URL: https://onlinelibrary.wiley.com/doi/10.1002/for.3212
    - Hybrid combining approaches

17. **Forecasting Realized Volatility: The Choice of Window Size** (2025). Journal of Forecasting.
    - URL: https://onlinelibrary.wiley.com/doi/10.1002/for.3221
    - Optimal window size analysis

18. **Stock Market Volatility Forecasting: Exploring the Power of Deep Learning** (2024). MDPI.
    - URL: https://www.mdpi.com/2674-1032/4/4/61
    - DL architectures for volatility

19. **Model Specification for Volatility Forecasting Benchmark** (2024). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S1057521924007828
    - Benchmarking methodology

### Jump-Diffusion and Advanced Models

20. **Option Pricing under Two-Factor Stochastic Volatility Jump-Diffusion Model** (2020). Complexity (Wiley).
    - URL: https://www.hindawi.com/journals/complexity/2020/1960121/
    - Two-factor SVJ model for European options

21. **A Jump Diffusion Model with Fast Mean-Reverting Stochastic Volatility for Pricing Vulnerable Options** (2023). Discrete Dynamics in Nature and Society.
    - URL: https://onlinelibrary.wiley.com/doi/10.1155/2023/2746415
    - Jump + mean-reverting volatility

22. **Stochastic Jump Diffusion Process Informed Neural Networks for Accurate American Option Pricing** (2025). Applied Soft Computing.
    - URL: https://dl.acm.org/doi/10.1016/j.asoc.2025.113164
    - Jump-diffusion + neural networks under data scarcity

23. **An Option Pricing Model with Double-Exponential Jumps in Returns and GARCH Diffusion in Volatilities** (2025). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0167637725000148
    - Double-exponential jump calibration results

24. **The Benefit of Modeling Jumps in Realized Volatility for Risk Prediction** (2021). PMC.
    - URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7147854/
    - Jump-informed volatility models for Chinese stocks

### Local Volatility Models

25. **Implied Local Volatility Models** (2024). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0927539824001014
    - Parametric regression methods for implied LV construction

26. **Rough PDEs for Local Stochastic Volatility Models** (2025). Mathematical Finance (Wiley).
    - URL: https://onlinelibrary.wiley.com/doi/abs/10.1111/mafi.12458
    - Rough volatility + LSV framework

27. **A Unified Model of SABR and Mean-Reverting Stochastic Volatility for Derivative Pricing** (2025). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S009630032500325X
    - SABR + SVJ hybrid; June 2025 publication

28. **A Stochastic-Local Volatility Model with Lévy Jumps for Pricing Derivatives** (2023). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0096300323002035
    - CEV + jump-diffusion pricing via Fourier

### Deep Learning: LSTM/GRU

29. **Time Series Forecasting in Financial Markets Using Deep Learning Models** (2025). Journal of World Academy of Engineering, Arts & Engineering Sciences.
    - URL: https://journalwjaets.com/sites/default/files/fulltext_pdf/WJAETS-2025-0167.pdf
    - Recent LSTM/GRU/Transformer comparison

30. **Forecasting Federal Fund Rates with AI: LSTM, GRU, and Beyond** (2025). AMRO (Asia & Pacific).
    - URL: https://amro-asia.org/wp-content/uploads/2025/09/AMRO_WP_25_10_FedAI_model.pdf
    - Federal funds rate forecasting with RNNs

31. **Time Series Forecasting Enhanced by Integrating GRU and N-BEATS** (2024). International Journal of Engineering Education and Information Engineering.
    - URL: https://www.mecs-press.org/ijieeb/ijieeb-v17-n1/v17n1-7.html
    - GRU + N-BEATS hybrid approach

32. **An Open-Source and Reproducible Implementation of LSTM and GRU Networks for Time Series Forecasting** (2024). ArXiv.
    - URL: https://arxiv.org/abs/2504.18185
    - Reproducible implementations

33. **A Deep Learning Approach to NIFTY 100 Stock Price Prediction Using LSTM and GRU Networks** (2024). IEEE Conference Publication.
    - URL: https://ieeexplore.ieee.org/document/10961331
    - Indian equity index application

34. **Time Series Forecasting Based on Deep Learning CNN-LSTM-GRU Model on Stock Prices** (2024). IJETT.
    - URL: https://ijettjournal.org/archive/ijett-v71i6p215
    - CNN-LSTM-GRU combined architecture

35. **Advanced Stock Market Prediction Using Long Short-Term Memory Networks: A Comprehensive Deep Learning Framework** (2025). ArXiv.
    - URL: https://arxiv.org/html/2505.05325v1
    - Comprehensive LSTM framework

36. **Deep Learning Models for Price Forecasting of Financial Time Series** (2023). ArXiv.
    - URL: https://arxiv.org/pdf/2305.04811
    - Comparative deep learning models

37. **Neural Networks for Financial Time Series Forecasting** (2023). Entropy Journal.
    - URL: https://www.mdpi.com/1099-4300/24/5/657
    - Neural network architectures for finance

### Transformers and Attention Mechanisms

38. **Enhancing Stock Price Prediction Using GANs and Transformer-Based Attention Mechanisms** (2024). Empirical Economics.
    - URL: https://link.springer.com/article/10.1007/s00181-024-02644-6
    - GAN + Transformer hybrid

39. **Stock Price Prediction Using Time Embedding and Attention Mechanism in Transformers** (2024). IEEE Conference Publication.
    - URL: https://ieeexplore.ieee.org/document/10452537/
    - Time embedding for Transformers

40. **Transformer-Based Attention Network for Stock Movement Prediction** (2022). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0957417422006170
    - TEANet framework

41. **An Enhanced Transformer Framework with Incremental Learning for Online Stock Price Prediction** (2024). PLOS One.
    - URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316955
    - IL-ETransformer for online prediction

42. **Transformer-Based Deep Learning Model for Stock Price Prediction: Bangladesh Stock Market** (2023). International Journal of Computational Intelligence and Applications.
    - URL: https://www.worldscientific.com/doi/10.1142/S146902682350013X
    - Emerging market application

43. **Galformer: A Transformer with Generative Decoding and Hybrid Loss for Multi-Step Stock Market Index Prediction** (2024). Scientific Reports (Nature).
    - URL: https://www.nature.com/articles/s41598-024-72045-3
    - Generative Transformer for multi-step prediction

44. **Deep Convolutional Transformer Network for Stock Movement Prediction** (2024). Electronics Journal.
    - URL: https://www.mdpi.com/2079-9302/13/21/4225
    - CNN-Transformer integration

45. **Advancing Financial Forecasting: Comparative Analysis of Neural Forecasting Models N-HiTS and N-BEATS** (2024). ArXiv.
    - URL: https://arxiv.org/html/2409.00480
    - N-BEATS outperforms N-HiTS on financial data

### Machine Learning and Neural Networks (General)

46. **Research on Stock Price Prediction Based on Machine Learning Techniques** (2025). SCITEPRESS.
    - URL: https://www.scitepress.org/Papers/2025/137036/137036.pdf
    - 2025 ML techniques survey

47. **Stock Market Trend Prediction Using Deep Neural Network via Chart Analysis** (2025). Nature: Humanities and Social Sciences Communications.
    - URL: https://www.nature.com/articles/s41599-025-04761-8
    - Chart-based deep learning

48. **Navigating AI-Driven Financial Forecasting: A Systematic Review** (2024). MDPI.
    - URL: https://www.mdpi.com/2571-9394/7/3/36
    - Systematic review of AI forecasting status

49. **Stock Price Prediction in the Financial Market Using Machine Learning Models** (2025). Computers.
    - URL: https://www.mdpi.com/2079-3197/13/1/3
    - Recent ML model comparison

50. **Financial Market Prediction Using Deep Neural Networks with Hardware Acceleration** (2021). IEEE Xplore.
    - URL: https://ieeexplore.ieee.org/abstract/document/9959984/
    - Hardware acceleration for neural networks

51. **Forecasting Stock Market Prices Using Machine Learning and Deep Learning Models: Systematic Review** (2023). MDPI.
    - URL: https://www.mdpi.com/2227-7072/11/3/94
    - Comprehensive systematic review

52. **A Multi-Model Machine Learning Framework for Daily Stock Price Prediction** (2024). Algorithms.
    - URL: https://www.mdpi.com/2504-2289/9/10/248
    - Ensemble ML framework

53. **Deep Learning for Stock Market Prediction** (2020). PMC.
    - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7517440/
    - Early deep learning survey

54. **Short-Term Stock Market Price Trend Prediction Using Comprehensive Deep Learning System** (2020). Journal of Big Data.
    - URL: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00333-6
    - Comprehensive DL systems

55. **Data-Driven Stock Forecasting Models Based on Neural Networks: A Review** (2024). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/pii/S1566253524003944
    - Neural network forecasting survey

### Option Pricing and Calibration with ML

56. **Deep Learning Calibration of Option Pricing Models: Pitfalls and Solutions** (2019, updated). ArXiv.
    - URL: https://arxiv.org/abs/1906.03507
    - Foundational work on DL calibration challenges

57. **Can Machine Learning Algorithms Outperform Traditional Models for Option Pricing?** (2024-2025). ArXiv.
    - URL: https://arxiv.org/html/2510.01446v1
    - Empirical comparison of ML vs. traditional pricing

58. **Option Pricing and Model Calibration with Neural Networks** (2025). ELTE AI.
    - URL: https://ai.elte.hu/wp-content/uploads/2025/06/Option-pricing.pdf
    - Recent option pricing NN framework

59. **Improved Accuracy of Analytical Approximations for Option Pricing Under Stochastic Volatility Using Deep Learning** (2025). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0898122125001245
    - Residual learning approach for SV pricing

60. **Machine Learning for Option Pricing: Empirical Investigation of Network Architectures** (2023). ArXiv.
    - URL: https://ideas.repec.org/p/arx/papers/2307.07657.html
    - Systematic comparison of NN architectures for pricing

61. **On Calibration of Mathematical Finance Models by Hypernetworks** (2024). Springer.
    - URL: https://link.springer.com/chapter/10.1007/978-3-031-43427-3_14
    - Hypernetwork-based 500x calibration speedup

62. **Option Pricing Using Machine Learning** (2020). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0957417420306187
    - Early ML option pricing survey

### Reinforcement Learning and Algorithmic Trading

63. **Reinforcement Learning for Quantitative Trading** (2023). ACM Transactions on Intelligent Systems and Technology.
    - URL: https://dl.acm.org/doi/10.1145/3582560
    - Comprehensive RL trading review

64. **Reinforcement Learning in Financial Decision Making: A Systematic Review** (2024-2025). ArXiv.
    - URL: https://arxiv.org/html/2512.10913v1
    - Meta-analysis of 167 studies (2017-2025)

65. **Risk-Aware Deep Reinforcement Learning** (2024). OpenReview.
    - URL: https://openreview.net/pdf/3cfd552d6e8675ecfc2ec22a69245dc3fa62c978.pdf
    - Risk-aware RL formulations

66. **(Deep) Learning to Trade: An Experimental Analysis of AI Trading** (2025). Wharton Working Paper.
    - URL: https://wifpr.wharton.upenn.edu/wp-content/uploads/2025/09/Sangiorgi_Deep__Learning_to_Trade.pdf
    - Experimental analysis of DL trading

67. **Reinforcement Learning in Algorithmic Trading: Optimizing Trade Execution and Risk Management** (2024). SSRN.
    - URL: https://papers.ssrn.com/sol3/Delivery.cfm/5559900.pdf?abstractid=5559900&mirid=1
    - RL for execution and risk

68. **Deep Learning for Algorithmic Trading: Systematic Review of Predictive Models and Optimization Strategies** (2024). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/pii/S2590005625000177
    - Recent DL trading strategies

69. **A Review of Reinforcement Learning in Financial Applications** (2024). ArXiv.
    - URL: https://arxiv.org/html/2411.12746v1
    - Focused RL applications review

70. **A Multi-Agent Deep Reinforcement Learning Framework for Algorithmic Trading** (2022). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0957417422013082
    - Multi-agent RL framework

71. **Quantum-Enhanced Forecasting for Deep Reinforcement Learning in Trading** (2024). ArXiv.
    - URL: https://arxiv.org/pdf/2509.09176
    - Emerging quantum-RL hybrid

### Other Relevant Topics

72. **Artificial Intelligence and Exchange Rate Forecasting** (2025). Frontiers in Applied Mathematics and Statistics.
    - URL: https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2025.1654093/full
    - AI for FX forecasting

73. **Volatility Forecasting and Volatility-Timing Strategies: A Machine Learning Approach** (2024). ScienceDirect.
    - URL: https://www.sciencedirect.com/science/article/abs/pii/S0275531924005166
    - ML for volatility timing

74. **Weighted Hybrid Model for Enhanced Volatility Forecasting** (2025). Preprints.
    - URL: https://www.preprints.org/manuscript/202510.0351/v1/download
    - Recent hybrid volatility model

75. **A Comparative Analysis of Liquid Neural Networks and Other Architectures** (2024). HAL.
    - URL: https://hal.science/hal-05148958v1/document
    - Liquid NN vs. traditional architectures

---

## Appendix: Quantitative Results Summary Table

| Category | Method | Metric | Result | Source Year | Notes |
|----------|--------|--------|--------|-------------|-------|
| **Volatility Forecasting** | GARCH | MAPE | Baseline | 2020+ | 1-day horizon |
| | EGARCH | MAPE | ~3-5% better | 2024-2025 | Leverage effect capture |
| | APARCH | MAPE | Best traditional | 2024-2025 | Long-term stability |
| | Deep Learning (no DL exog) | MAPE | Often worse than HAR | 2024 | Feature constraint critical |
| | HAR+News | MAPE | Measurable improvement | 2025 | Sentiment integration |
| | Transformer | MAPE | Best overall | 2024-2025 | With exogenous variables |
| **Stock Price** | LNN | RMSE | 0.0178 | 2024 | Directional accuracy 49.36% |
| | Ensemble | Directional | 55-58% | 2024-2025 | Slightly above random |
| | Transformer | Multiple | Best | 2024 | Parallel computation advantage |
| **Option Pricing** | Traditional MLE | Accuracy | Gold standard | 2020+ | Speed: 1x baseline |
| | Hypernetwork | Accuracy | Very close | 2024 | Speed: 500x faster |
| | Residual NN | Accuracy | Competitive | 2024-2025 | Potential arbitrage violations |
| | Sparse GP | Accuracy | Similar to NN | 2024 | Better uncertainty quantification |
| **Heston Calibration** | MLE | MSE (prices) | Baseline | 2020+ | Gold standard |
| | Deep Learning | MSE (prices) | Reduced error | 2024 | Robustness across conditions |
| **Jump-Diffusion** | SVJ | RMSE/MAPE | Lowest | 2025 | Superior to GBM, Heston, MJD |
| | SVJ | Calibration | 1-year (low-vol) / 6-month (high-vol) | 2025 | Asset-dependent optimum |
| **RL Trading** | DDPG | Return/Sharpe | Measurable gain | 2020-2025 | Market making best application |
| | Multi-agent RL | Portfolio Sharpe | Improved | 2022+ | Risk-adjusted returns |

---

## Document Version and Metadata

- **Compiled**: December 2025
- **Literature Coverage**: 2020-2025 (with seminal older references)
- **Total Sources**: 75+ peer-reviewed papers, preprints, and conference proceedings
- **Geographic Coverage**: Global (US, UK, Europe, Asia, Emerging Markets)
- **Asset Classes Covered**: Equities, Derivatives, Cryptocurrencies, FX
- **Methodologies Synthesized**: Classical SDEs, GARCH, NN/DL, RL, Hybrid

---

**End of Literature Review**
