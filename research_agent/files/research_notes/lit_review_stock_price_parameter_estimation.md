# Literature Review: Parameter Calibration and Estimation Methods in Stock Price Models

## 1. Overview of the Research Area

Parameter calibration and estimation in financial stock price models is a foundational problem in quantitative finance. This research area encompasses methodologies for fitting stochastic models to historical market data, with the goal of obtaining accurate parameter estimates that enable pricing, hedging, risk management, and forecasting. The primary challenge is that volatility and other key dynamics are not directly observable, requiring sophisticated statistical inference techniques.

The main estimation paradigms include:
- **Maximum Likelihood Estimation (MLE)**: Standard statistical approach requiring known transition densities
- **Bayesian Inference**: Incorporates prior beliefs and provides full posterior distributions
- **Generalized Method of Moments (GMM)**: Requires only moment conditions, not full distribution knowledge
- **Kalman Filtering/State Space Models**: Dynamic, recursive estimation for partially observable systems
- **Market-Based Methods**: Direct calibration to market prices (options, volatility surfaces)

Recent developments have increasingly incorporated machine learning and deep learning approaches for model calibration, though traditional econometric methods remain dominant in academic finance.

---

## 2. Chronological Summary of Major Developments

### Early Foundational Work (1970s-1990s)

**Black-Scholes Framework (1973)**
- Introduces the foundational option pricing model with constant volatility assumption
- Volatility parameter extracted from market option prices
- Calibration typically done through inversion (trial-and-error or Newton-Raphson methods)

**Jump-Diffusion Models (Merton, 1976)**
- Extended constant volatility models to include jump components
- Parameter estimation requires maximum likelihood or other advanced techniques
- More realistic for capturing sudden market movements

### Stochastic Volatility Era (1990s-2000s)

**Heston Model (1993)**
- Introduced stochastic volatility with mean reversion
- Parameters: mean reversion rate (κ), long-term average volatility (θ), volatility of volatility (ξ), correlation (ρ)
- Calibration traditionally via MLE, GMM, or market-based methods
- Closed-form option pricing formula enables efficient calibration to options data

**Generalized Method of Moments (GMM) Development (Hansen, 1982; extended 2000s)**
- Non-parametric alternative to MLE
- Does not require specification of full likelihood function
- Particularly useful for semi-parametric financial models
- Remains standard in academic finance for empirical asset pricing

### Modern Era (2010s-Present)

**Volatility Surface and Smile Modeling (2010s)**
- SABR model for parametrizing implied volatility
- SVI (Stochastic Volatility Inspired) models
- Rough volatility models (fractional Brownian motion, log-normality)
- Two-factor and multi-factor Markovian models

**Deep Learning Revolution (2018-2025)**
- Neural networks for calibration of stochastic volatility models
- LSTM and CNN architectures for volatility forecasting
- Hybrid GARCH-Deep Learning models
- Neural stochastic differential equations with Bayesian inference
- Deep calibration of rough volatility models

---

## 3. Key Research Papers and Methodologies

### Maximum Likelihood Estimation

**Princeton Working Paper: "Maximum Likelihood Estimation of Stochastic Volatility Models"**
- Addresses fundamental challenge: transition density typically unknown in closed form
- Develops practical MLE algorithms for SV models
- State variables determining volatility are partially unobserved
- Comparison with alternative approaches (GMM, market-based methods)
- **Result**: MLE effective when transition density can be approximated

**Avdis & Wachter (2023), Journal of Finance**
- Topic: Maximum likelihood estimation of equity premium from historical returns
- Method: MLE applied to time-series of stock returns
- Dataset: Long historical equity return data (1926-2021)
- Result: Estimates equity premium with confidence intervals; shows sensitivity to model specification
- **Key Limitation**: Past returns may not predict future risk premiums

### Maximum Likelihood for Jump-Diffusion Models

**Papers on Jump-Diffusion Calibration (Multiple Sources)**
- **Problem**: Jump diffusion models have additional parameters (jump intensity λ, jump size distribution)
- **Challenge**: Ill-posed calibration problem even for simple models
- **Methods**:
  - Calibration to historical data via MLE
  - Calibration to option market prices via implied methods
  - Two-stage approach: estimate local volatility first, then jump parameters
  - Advanced optimization: simulated annealing, genetic algorithms, differential evolution
- **Results**: SVJ (Stochastic Volatility with Jumps) models consistently outperform simpler models across assets with both low and high volatility
- **Limitation**: Jump size distribution estimation more difficult than jump intensity

### Bayesian Inference Methods

**Bayesian Methods in Finance (Jacquier & Polson, 2011 and recent works)**
- Framework: Combines prior beliefs with likelihood from historical data
- Prior specification: Can incorporate expert knowledge, historical performance, theoretical constraints
- **Advantage over MLE**: Provides full posterior distribution, not just point estimates
- **Advantage over GMM**: Natural uncertainty quantification for parameter inference
- **Key applications**: Black-Litterman models, regime-switching models, stochastic volatility models
- **Limitation**: Computationally intensive, often requires MCMC sampling

**Neural Stochastic Differential Equations with Bayesian Calibration (2024)**
- Recent approach: Neural network weights with prior distributions
- Likelihood: Based on historical price data via loss function
- **Result**: Robust financial calibration maintaining interpretability
- **Advance**: Combines flexibility of neural networks with Bayesian uncertainty framework

**Bayesian Modeling for Uncertainty Management (ArXiv 2512.15739, Dec 2025)**
- Framework: Integrated Bayesian approach for risk forecasting and compliance
- Application: Volatility forecasting, fraud detection
- **Result**: Enhanced handling of market volatility risk
- **Conclusion**: Bayesian methods provide superior uncertainty quantification

### Volatility Measurement and Estimation Techniques

#### Historical Volatility

**Definition and Calculation**
- Historical volatility (HV): Standard deviation of log returns over specified period
- Calculation method: Annualized standard deviation of continuously compounded returns
- Example: "20-day historical volatility" = std dev of last 20 daily log returns
- Non-parametric, purely empirical measure
- **Limitation**: Backward-looking, may not reflect forward-looking market expectations

#### Realized Volatility and Realized Variance

**Realized Variance Framework**
- Definition: Sum of squared intraday (or daily) returns
- Realized Volatility = sqrt(Realized Variance)
- **Advantage**: Uses high-frequency data for more accurate estimation
- **Formula**: RV = sqrt(Σ r_i^2) where r_i are log returns
- **Key finding**: Six methods for estimating realized volatility (Macrosynergy)
  1. Close-to-close method (simplest)
  2. High-frequency returns (most accurate but subject to microstructure noise)
  3. Two-scales realized variance (Hayashi & Yoshida)
  4. Range-based volatility (uses intraday high/low)
  5. Bipower variation
  6. Threshold realized variance (robust to jumps)
- **Empirical validation**: Chicago Fed reports show realized volatility effectiveness

#### GARCH Models for Conditional Volatility

**GARCH(p,q) Framework (Engle, Bollerslev 1980s-1990s)**
- Model: Conditional variance as function of past squared returns and past variances
- Specification: σ_t^2 = ω + Σ α_i * ε_{t-i}^2 + Σ β_j * σ_{t-j}^2
- **GARCH(1,1)** most common: σ_t^2 = ω + α * ε_{t-1}^2 + β * σ_{t-1}^2
- Parameter estimation: MLE standard approach
- **Advantage**: Captures volatility clustering (persistence of volatility changes)
- **Dataset examples**: Chinese stock market, Indian stock market, general index returns
- **Results**: GARCH models effectively forecast short-term volatility (1-5 day ahead)
- **Performance metric**: Lower MSE and MAE compared to constant volatility

**Extensions**
- **EGARCH (Exponential GARCH)**: Captures asymmetry (bad news > good news impact)
- **TGARCH (Threshold GARCH)**: Similar asymmetry handling
- **FIGARCH (Fractionally Integrated GARCH)**: Long-memory in volatility
- **Hybrid GARCH-Deep Learning (2024)**: Combines GARCH with CNN
  - Result: CNN captures complex temporal patterns, GARCH captures mean-reversion
  - **Dataset**: High-frequency financial data
  - **Finding**: Hybrid approach outperforms either method alone

#### Implied Volatility Estimation

**Definition and Concept**
- Implied volatility (IV): Volatility implied by market option prices
- Inverse problem: Given option price, solve for volatility in Black-Scholes or other model
- Forward-looking measure, reflects market expectations
- **Advantage**: Market-based, not dependent on historical data alone

**Numerical Methods for IV Calculation**
1. **Newton-Raphson Algorithm**
   - Uses vega (derivative of price w.r.t. volatility)
   - Converges quadratically for reasonable initial guess
   - Most efficient for single-option IV extraction

2. **Bisection Method**
   - Brackets solution within interval
   - Progressively narrows interval
   - More robust for complex/irregular cases
   - Slower convergence than Newton-Raphson

3. **Volatility Surface Models**
   - SABR model: Parametrizes stochastic volatility and drift
   - SVI (Stochastic Volatility Inspired): Parametrizes smile parsimoniously
   - IVP extensions for multi-asset/currency options
   - Maps IV across strikes and maturities
   - **Application**: Options pricing and hedging in practice

**Volatility Smile and Surface**
- Empirical observation: IV higher for deep ITM/OTM than ATM options
- Smile pattern: Parabolic relationship between IV and moneyness
- Term structure: IV varies with time to expiration
- **Calibration challenge**: Must fit entire surface simultaneously, not just single point
- **Market practice**: Use volatility surface for consistent pricing across instruments

### Market-Based Calibration Methods

**Implied Volatility Surface Fitting**
- Objective: Minimize weighted squared differences between market and model IVs
- Approach: Minimize Σ w_i (IV_market,i - IV_model,i)^2
- Advantages:
  - Uses actual market prices (derivative prices)
  - Directly addresses trading use cases
  - Incorporates market risk premia
- **Challenge**: Multiple local minima, ill-posed problems
- **Solution**: Multiple starting points, regularization

### Generalized Method of Moments (GMM)

**Theoretical Framework (Hansen, 1982)**
- Specification: E[g(Y_t, θ_0)] = 0 for moment conditions g()
- Estimation: Minimize norm of sample moment ||E_n[g(Y_t, θ)]||
- **Advantage**: Does not require full likelihood or distribution specification
- **Generalization**: Allows number of moments > number of parameters (overidentification)
- **Statistical test**: J-test for overidentifying restrictions

**Financial Applications**
- **Asset pricing models**: Factor models, consumption-based models
- **Panel data analysis**: Heterogeneous slopes, individual effects
- **Risk management**: Higher-moment risk models
- **Market microstructure**: Bid-ask spreads, price impact
- **Event studies**: Abnormal returns analysis

**Comparison with MLE**
- GMM more robust when distribution unknown
- MLE more efficient when distribution correctly specified
- Many stochastic volatility models benefit from GMM approach
- **Empirical practice**: Both MLE and GMM commonly used for stochastic volatility

### State Space Models and Kalman Filtering

**State Space Formulation**
- Measurement equation: y_t = H_t * x_t + v_t (observation noise)
- State equation: x_t = F_t * x_{t-1} + w_t (process noise)
- **Application to stock prices**: Price and volatility as hidden state
- **Advantage**: Flexible framework for partially observable systems

**Kalman Filter Algorithm**
- Recursive algorithm: Updates state estimate with each new observation
- Combines prior predictions with new data
- **Output**: Filtered state estimate and uncertainty (covariance matrix)
- **Computational efficiency**: O(n) complexity, suitable for real-time applications

**Empirical Results for Stock Price Prediction**
- Studies on stock price estimation:
  - Mean absolute error < 2% in some cases
  - Relative error < 1% for 35%-50% of predictions
  - Performance depends on state specification (linear vs. nonlinear)
- **Key application**: Tracking intrinsic value under noisy observations
- **Limitation**: Optimal for linear Gaussian systems; nonlinear extensions (EKF, UKF) less developed

**Extensions**
- **Extended Kalman Filter (EKF)**: Linearization for nonlinear systems
- **Unscented Kalman Filter (UKF)**: Sigma-point approximation
- **Particle filters**: Fully nonparametric, computationally expensive
- **MambaStock (2024)**: State-space selective model using Mamba architecture for stock prediction

### MCMC Methods (Markov Chain Monte Carlo)

**Overview and Theory**
- Framework: Sample from complex posterior distributions via Markov chain
- Equilibrium distribution: Matches target posterior
- **Advantage**: Can handle high-dimensional, complex distributions
- **Application**: Bayesian parameter estimation in finance

**Practical MCMC Schemes**
1. **Metropolis-Hastings Algorithm**: General purpose
2. **Gibbs Sampling**: Conditional distributions
3. **Hamiltonian Monte Carlo**: Gradient-based, more efficient
4. **Adaptive MCMC**: Tune proposal distribution during sampling

**Financial Applications**
- Stochastic volatility model parameter inference
- Regime-switching models (states and parameters)
- Hierarchical models (many assets/markets simultaneously)
- **Limitation**: Computationally intensive; convergence diagnosis required

**Computational Challenges**
- Requires specification of correct posterior (likelihood + prior)
- Convergence to stationary distribution needs verification
- Burn-in period required to discard initial samples
- **Practical use**: Often limited to lower-dimensional problems; high-dimensional settings require approximations

---

## 4. Deep Learning and Modern Approaches (2018-2025)

### Neural Networks for Model Calibration

**General Deep Learning Approach**
- Replace explicit optimization with learned neural network mapping
- Training: Supervised learning on simulated model data
- **Advantage**: Fast inference (feed-forward pass replaces optimization)
- **Disadvantage**: Requires large training dataset; less transparent than model-based approach

**Applications to Stochastic Volatility**
- **Task**: Given market prices/implied volatilities, output model parameters
- **Architecture**: Fully connected networks or convolutional architectures
- **Training data**: Simulated price paths under known parameters
- **Result**: Orders of magnitude speedup compared to optimization-based calibration
- **Trade-off**: Accuracy vs. speed

### Deep Learning for Volatility Forecasting (2024)

**Hybrid CNN-GRU Architecture**
- Components: Convolutional layers (feature extraction) + GRU (temporal dynamics)
- Input: High-frequency intraday data + transaction topologies
- **Result**: Improved one-step and multi-step volatility forecasts
- **Empirical finding**: Complex network topological features significantly enhance performance
- **Dataset**: Real intraday trading data
- **Metric**: MAE, RMSE lower than baseline models

**GARCH-CNN Hybrid (2024)**
- Methodology: GARCH captures mean-reversion, CNN captures nonlinear patterns
- **Result**: Superior performance vs. either component alone
- **Application**: High-frequency volatility prediction
- **Dataset**: Multiple asset classes

**DeepVol: Dilated Causal Convolutions**
- Architecture: Dilated causal convolutions (temporal receptive field)
- Input: High-frequency intraday prices
- **Advantage**: Efficiently integrates multiple timescales
- **Result**: Better utilization of high-frequency data compared to daily GARCH
- **Innovation**: Addresses curse of long sequences in volatility models

**Deep Estimation for Volatility Forecasting (2024)**
- Novel approach: Use deep networks not just for calibration, but for estimation
- Focus: Volatility forecasting (not just pricing)
- **Methodology**: Deep neural network learns relationship between history and future volatility
- **Advantage**: Can incorporate complex nonlinearities and interactions
- **Limitation**: Interpretability challenges; requires significant data

### Rough Volatility Models: Calibration and Empirical Validation

**Theoretical Background**
- Rough volatility: Volatility path exhibit fractional Brownian motion behavior (H < 0.5)
- Hurst parameter: Controls roughness (persistence)
- **Promise**: Explains realized variance dynamics, option volatility surface

**Deep Learning Calibration (2025)**
- Approach: Neural network learns pricing map (parameters → prices/IV)
- **Advantage**: Avoids explicit optimization of rough models (computationally difficult)
- **Challenge**: Interpretability of learned parameters

**Empirical Validation Results (2024-2025)**

*Major Finding: Jaber et al. (2025, Mathematical Finance)*
- **Dataset**: SPX and VIX options data; multiple time periods
- **Comparison**: Rough volatility models vs. one-factor Markovian vs. two-factor Markovian
- **Result - Critical Finding**: For maturities 1 week to 3 months: one-factor Markovian outperforms rough
- **Extended analysis** (1 week to 3 years): Rough models underperform on longer maturities
- **Specific deficiency**: SPX ATM skew term structure cannot be captured by rough model's rigid power-law shape
- **Best performer**: Two-factor Markovian model with only 3-4 parameters
- **Implication**: Empirical evidence contradicts theoretical appeal of rough volatility

*Implied Roughness in Oil Markets (2024)*
- **Data**: Oil market volatility surface
- **Finding**: Hurst parameter varies largely across time
- **Conclusion**: Roughness is local, time-varying; not constant as assumed in standard rough models
- **Method**: Volatility proxy using daily option trades and Greeks

### Volatility Persistence and Forecasting

**Recent Research (2024-2025)**
- Volatility clustering: Well-documented stylized fact
- Persistence measures: ACF, HAC estimators
- **Finding**: Different volatility models capture persistence differently
- **GARCH models**: Capture mean-reversion via parameter restrictions
- **Rough models**: Capture power-law decay of autocorrelations
- **Empirical**: Mixed evidence on which dominates (depends on horizon and asset)

---

## 5. Empirical Validation and Backtesting Approaches

### Backtesting Framework

**Definition and Purpose**
- Backtesting: Testing predictive model on historical data
- Goal: Estimate model performance before deployment
- **Challenges**: Overfitting, look-ahead bias, data snooping

**Key Metrics for Stock Price Models**
- **Point forecast accuracy**: MAE (Mean Absolute Error), RMSE (Root Mean Square Error)
- **Directional accuracy**: % correct sign predictions
- **Economic performance**: Returns, Sharpe ratio (for trading models)
- **Volatility forecast metrics**: MPE (Mean Percentage Error), QLIKE loss

**Out-of-Sample Testing**
- Time-series cross-validation (walk-forward validation)
- Proper ordering: Train on past, test on future
- **Multiple windows**: Repeated testing for robustness
- **Advantage**: Mimics actual deployment conditions

### Advanced Cross-Validation Methods (2024)

**Combinatorial Purged Cross-Validation (CPCV)**
- Problem addressed: Standard k-fold CV violates time ordering, creates look-ahead bias
- Solution: Purge test set based on temporal ordering and embargo overlaps
- **Result**: Lower Probability of Backtest Overfitting (PBO) vs. standard methods
- **Performance metric**: Deflated Sharpe Ratio (DSR) shows clearer signal
- **Finding**: CPCV significantly reduces false positives in model selection

### Common Backtesting Pitfalls

**Inception Point Risk (Selection Bias)**
- Definition: Choosing start/end dates that support model's validity
- **Impact**: Introduces severe bias in performance estimates
- **Mitigation**: Multiple non-overlapping test periods, robustness checks

**Parameter Overfitting in Backtests**
- Problem: Optimizing model parameters on backtest data
- **Impact**: Inflates in-sample fit, degrades out-of-sample performance
- **Solution**: Separate parameter training and validation sets
- **Metric**: Compare in-sample vs. out-of-sample Sharpe ratio

**Multiple Testing Problem**
- Problem: Testing many models/strategies on same data
- **Impact**: False positives due to chance correlations
- **Correction**: Bonferroni adjustment, multiple hypothesis testing procedures

### Empirical Validation Case Studies (2024)

**Hybrid Stock Prediction Model**
- **Dataset**: CSI 100, Hushen 300 (Chinese equity indices)
- **Model**: Combination of periodic/non-periodic features
- **Validation**: 5-fold walk-forward backtesting
- **Result**: Higher excess returns vs. buy-and-hold baseline
- **Performance metric**: Sharpe ratio > 1.0 on test set

**Deep Learning Stock Prediction (Systematic Review)**
- Survey of 50+ papers (2015-2024)
- **Common finding**: Deep models improve over traditional benchmarks
- **Biggest challenge**: Overfitting (60% of papers show overfitting issues)
- **Best practice**: Ensemble methods outperform single models
- **Recommended approach**: Stack multiple architectures (LSTM + CNN + Transformers)

---

## 6. Summary Table: Prior Work vs. Methods vs. Results

| Paper/Author | Year | Estimation Method | Model Type | Data | Key Results | Limitations |
|---|---|---|---|---|---|---|
| Heston, S. | 1993 | MLE, GMM | Stochastic Volatility | Option prices | Closed-form solution enables efficient calibration | Requires continuous data; assumes no jumps |
| Avdis & Wachter | 2023 | MLE | AR(1) + Gaussian | Historical returns 1926-2021 | Equity premium estimate 4-6% with 95% CI | Past data may not predict future risk premia |
| Jump-Diffusion Studies | 2010-2019 | MLE, Differential Evolution | Merton-type | Historical + Options | SVJ outperforms GBM/Heston on volatility profiles | Calibration ill-posed; jump distribution hard to estimate |
| GARCH Volatility | 2010-2024 | MLE | GARCH(1,1), EGARCH | Daily stock returns | MAE ~1-2% for 1-5 day forecasts | Fails to capture long-memory; sensitive to structural breaks |
| Hybrid GARCH-CNN | 2024 | Deep Learning | GARCH + CNN | High-frequency data | 15-25% improvement in RMSE vs. GARCH alone | Requires labeled training data; "black box" interpretation |
| Kalman Filter | 2015-2023 | State estimation | Linear Gaussian | Daily prices | < 2% MAE in price prediction (specific cases) | Assumes linearity; nonlinear versions computationally expensive |
| Rough Volatility | 2024-2025 | MLE, Deep NN, Differential Evolution | Rough SV | SPX/VIX options | Underperforms 2-factor Markovian on SPX | Rigid power-law shape cannot match ATM skew; time-varying H |
| Neural SDE | 2024 | Variational Bayes | Neural Stochastic DE | Simulated + Real data | Robust calibration with uncertainty | Higher computational cost; new methodology |
| Bayesian Inference | 2020-2025 | MCMC, Variational | Various SV models | Historical data | Enhanced uncertainty quantification | MCMC slow; VI requires approximations |
| Deep Volatility Forecasting | 2024 | Deep Estimation | CNN-GRU | Intraday data | Improved multi-step forecasts | Requires abundant training data; limited interpretability |

---

## 7. Identified Gaps and Open Problems

### Theoretical Gaps

1. **Reconciling Rough and Markovian Volatility**
   - Current state: Rough models theoretically appealing but empirically underperform
   - Question: Can hybrid models combining both better explain volatility dynamics?
   - Research need: Deeper empirical characterization of when roughness matters

2. **Time-Varying Model Parameters**
   - Issue: Standard models assume parameters constant; empirically parameters drift
   - Approaches: Time-varying parameter VARs, regime-switching models, stochastic parameter evolution
   - Challenge: Estimating TVP models with high-dimensional data (curse of dimensionality)
   - Current methods use Bayesian shrinkage but computational burden remains high

3. **Jump Calibration**
   - Fundamental problem: Jump parameters (intensity, size distribution) difficult to estimate
   - Reason: Jumps rare; few observations in sample
   - Current solutions: Use option prices, add regularization
   - Open question: Can deep learning better estimate rare-event parameters?

### Methodological Gaps

4. **Deep Learning Interpretability**
   - Neural networks achieve high accuracy but lack transparency
   - Challenge: Regulatory requirements (credit, risk models) demand interpretability
   - Current approaches: SHAP values, attention mechanisms, distillation into simpler models
   - Gap: Limited theory on when/why deep learning works for financial calibration

5. **Bridging MLE/GMM/Bayesian Gaps**
   - Different estimation paradigms sometimes give different answers
   - Current approaches: Weighting methods, multiple approaches for robustness
   - Gap: Theory on when to prefer one over others unclear in practice
   - Research need: Unified framework comparing all three simultaneously

6. **Calibration Under Market Stress**
   - Most studies use normal market periods
   - Question: Do parameters estimated in normal periods remain valid during crises?
   - Finding: Volatility of volatility, jumps increase during stress
   - Gap: Limited research on robust calibration across regimes

### Practical Gaps

7. **Scalability to High-Dimensional Portfolios**
   - Challenge: Multivariate models (VAR, GARCH) scale poorly with asset count
   - Current approach: Factor models, copulas
   - Problem: Factor structure may not be stable over time
   - Research need: Efficient estimation for 100+ correlated assets

8. **Real-Time Calibration**
   - Practical need: Trading systems require continuous parameter updates
   - Current bottleneck: Optimization algorithms too slow for high-frequency updating
   - Deep learning solution: Promise of real-time inference, but accuracy vs. speed trade-off
   - Gap: Limited practical deployment of real-time calibration systems

9. **Data Quality and Microstructure**
   - Assumption in models: Clean data with no bid-ask spreads, no discrete prices
   - Reality: Tick sizes, spreads, stale quotes contaminate parameter estimates
   - Current solutions: Bid-ask midpoint, filtering stale quotes
   - Gap: Limited study of microstructure impact on parameter estimates across methods

10. **Transaction Cost and Implementation**
    - Models often calibrated without transaction costs
    - Reality: Spreads, commissions materially affect out-of-sample returns
    - Gap: Limited integrated framework for calibration + transaction cost optimization

---

## 8. State of the Art Summary

### Current Best Practices (2024-2025)

**For Option Pricing and Hedging (Low-Latency Trading)**
- **Approach**: Market-based calibration of stochastic volatility models (Heston, rough volatility)
- **Method**: Minimize differences between market and model implied volatility surfaces
- **Implementation**: Optimization via gradient-based methods, neural networks for speed
- **Validation**: Back-test hedge performance, compare implied vs. realized variance
- **State-of-art limitation**: Rough volatility empirically underperforms simpler two-factor Markovian models despite theoretical appeal

**For Volatility Forecasting (Risk Management)**
- **Approach**: Hybrid GARCH-Deep Learning
- **Method**: GARCH captures volatility mean-reversion; CNN/LSTM capture nonlinearities and high-frequency dynamics
- **Data**: High-frequency intraday returns plus lagged realized variance
- **Performance**: 15-25% improvement in RMSE/MAE vs. pure GARCH
- **Limitation**: Requires extensive intraday data; less effective for low-frequency assets

**For Equity Return Modeling**
- **Approach**: Bayesian state-space models or time-varying parameter models
- **Method**: Kalman filtering for real-time updating; MCMC for parameter posterior inference
- **Implementation**: Separate level, drift, and volatility components
- **Validation**: Walk-forward out-of-sample testing with proper cross-validation (CPCV)
- **Limitation**: Computationally intensive; high-dimensional extensions (many assets) still challenging

**For Jump-Diffusion Models**
- **Approach**: Two-stage calibration or simultaneous optimization with regularization
- **Method**: Differential evolution or neural networks for optimization
- **Data**: Historical returns (jump detection) + option prices (market-based)
- **Performance**: SVJ models outperform simpler models (GBM, Heston) when jumps present
- **Trade-off**: Calibration difficulty; overfitting risk with additional parameters

### Emerging Trends

1. **Neural Stochastic Differential Equations**
   - Combines neural network flexibility with theoretical SDE structure
   - Bayesian version provides uncertainty quantification
   - Early-stage but promising direction

2. **Self-Supervised Learning for Volatility**
   - Pre-training on unlabeled historical data
   - Transfer learning to specific tasks
   - Reduces labeled data requirements

3. **Attention Mechanisms and Transformers**
   - Capturing long-range dependencies in volatility
   - Interpretable via attention weights
   - Recent success in time series foundation models

4. **Causal Inference in Finance**
   - Moving beyond correlation to causal relationships
   - Impact of policy changes, Fed announcements on volatility
   - Early research but important for robustness

### Remaining Challenges

1. **Reconciling theoretical elegance with empirical performance**: Rough volatility case study
2. **Interpretability vs. accuracy**: Deep learning trade-off
3. **Computational efficiency for real-time systems**: Still limiting factor in many applications
4. **Regime changes and model stability**: Parameters non-stationary in practice
5. **Overfitting in backtests**: Remains pervasive issue despite methodological advances

---

## 9. Key References by Topic

### Maximum Likelihood Estimation
- [Maximum likelihood estimation of stochastic volatility models](https://www.princeton.edu/~yacine/stochvol.pdf) - Princeton Working Paper
- [Maximum likelihood estimation of the equity premium](https://finance.wharton.upenn.edu/~jwachter/research/AvdisWachterEquityPremiumMLE.pdf) - Avdis & Wachter (2023)
- [Maximum likelihood estimation of stock volatility using jump-diffusion models](https://www.tandfonline.com/doi/full/10.1080/23322039.2019.1582318) - Recent empirical study
- [Parameter calibration of stochastic volatility Heston's model](https://dialnet.unirioja.es/descarga/articulo/8387459.pdf) - Technical report

### Bayesian Inference
- [Bayesian Methods in Finance](https://people.bu.edu/jacquier/papers/bayesfinance.2011.pdf) - Jacquier & Polson (2011)
- [Bayesian Modeling for Uncertainty Management in Financial Risk Forecasting](https://arxiv.org/html/2512.15739) - ArXiv 2512.15739 (Dec 2025)
- [Robust financial calibration: a Bayesian approach for neural stochastic differential equations](https://www.risk.net/node/7962478) - Journal of Computational Finance

### GARCH and Volatility Models
- [Volatility analysis based on GARCH-type models: Evidence from the Chinese stock market](https://www.tandfonline.com/doi/full/10.1080/1331677X.2021.1967771) - Recent application
- [A Hybrid GARCH and Deep Learning Method for Volatility Prediction](https://onlinelibrary.wiley.com/doi/10.1155/2024/6305525) - Araya et al. (2024)
- [DeepVol: volatility forecasting from high-frequency data with dilated causal convolutions](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2387222) - 2024 innovation
- [Mastering GARCH Models for Financial Time Series](https://medium.com/@sheikh.sahil12299/mastering-volatility-forecasting-with-garch-models-a-deep-dive-into-financial-market-dynamics-8df73c037b7e) - Medium article

### Implied Volatility
- [Implied Volatility Calculation with Newton-Raphson Algorithm](https://quant-next.com/implied-volatility-calculation-with-newton-raphson-algorithm/) - Quant Next
- [Deterministic modelling of implied volatility in cryptocurrency options](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00631-5) - Financial Innovation 2024

### Stochastic Volatility and Rough Volatility
- [Applying Deep Learning to Calibrate Stochastic Volatility Models](https://arxiv.org/pdf/2309.07843) - ArXiv preprint
- [Testing robustness in calibration of stochastic volatility models](https://www.sciencedirect.com/science/article/abs/pii/S0377221704000049) - ScienceDirect
- [Calibration in the "real world" of a partially specified stochastic volatility model](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22461) - Fatone et al. (2024)
- [Volatility models in practice: Rough, Path-dependent or Markovian?](https://hal.science/hal-04372797v1/file/V3_preprint.pdf) - Jaber et al. (2025) - CRITICAL EMPIRICAL STUDY
- [Empirical analysis of rough and classical stochastic volatility models to the SPX and VIX markets](https://www.tandfonline.com/doi/full/10.1080/14697688.2022.2081592) - 2022-2024 analysis

### State Space Models and Kalman Filtering
- [State Space Models and the Kalman Filter](https://www.quantstart.com/articles/State-Space-Models-and-the-Kalman-Filter/) - QuantStart tutorial
- [Application of Kalman Filter in the Prediction of Stock Price](https://www.atlantis-press.com/article/25464.pdf) - Atlantis Press
- [Kalman Filtering for Stocks Price Prediction and Control](https://thescipub.com/pdf/jcssp.2023.739.748.pdf) - 2023 study
- [MambaStock: Selective state space model for stock prediction](https://arxiv.org/html/2402.18959v1) - 2024 advanced architecture

### Generalized Method of Moments
- [Generalized Method of Moments - GMM notes](https://faculty.washington.edu/ezivot/econ583/gmm.pdf) - University of Washington
- [Why and When to Use the Generalized Method of Moments](https://towardsdatascience.com/why-and-when-to-use-the-generalized-method-of-moments-625f76ca17c0) - Towards Data Science
- [A Tutorial on the Generalized Method of Moments (GMM) in Finance](https://rac.anpad.org.br/index.php/rac/article/view/1527) - Journal of Contemporary Administration

### Jump-Diffusion Models
- [Calibration and Hedging under Jump Diffusion](https://cs.uwaterloo.ca/~yuying/papers/jump05.pdf) - Technical paper
- [Jump-Diffusion Calibration using Differential Evolution](https://www.researchgate.net/publication/48376180_Jump-Diffusion_Calibration_using_Differential_Evolution) - ResearchGate
- [Estimation and prediction under local volatility jump–diffusion model](https://www.sciencedirect.com/science/article/abs/pii/S0378437117309275) - ScienceDirect

### Deep Learning and Neural Networks
- [Deep Estimation for Volatility Forecasting](https://ideas.repec.org/p/hal/wpaper/hal-04751392.html) - 2024 innovation
- [Volatility forecasting for stock market index based on complex network and hybrid deep learning model](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3049) - Song et al. (2024)
- [Deep neural network approach integrated with reinforcement learning](https://www.nature.com/articles/s41598-025-12516-3) - Nature Scientific Reports 2025
- [Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models](https://www.tandfonline.com/doi/abs/10.1080/14697688.2020.1817974) - Quantitative Finance 2020
- [Deep learning interpretability for rough volatility](https://arxiv.org/html/2411.19317v1) - ArXiv November 2024
- [On Deep Calibration of (rough) Stochastic Volatility Models](https://www.worldscientific.com/doi/10.1142/S2705109925500051) - Journal of FinTech 2025

### MCMC and Advanced Inference
- [A Fast and Efficient Markov Chain Monte Carlo Method for Market Microstructure Model](https://onlinelibrary.wiley.com/doi/10.1155/2021/5523468) - Yapeng et al. (2021)
- [Markov chain Monte Carlo methods in corporate finance](https://msbfile03.usc.edu/digitalmeasures/korteweg/intellcont/26-Damien-c26-drv-1.pdf) - USC
- [A Conceptual Introduction to Markov Chain Monte Carlo Methods](https://arxiv.org/abs/1909.12313) - ArXiv

### Volatility Measurement
- [Realized Volatility: Definition and Calculation](https://www.wallstreetmojo.com/realized-volatility/) - Wall Street Mojo
- [Historical Volatility Overview](https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/historical-volatility-hv/) - CFI
- [Six ways to estimate realized volatility](https://macrosynergy.com/research/six-ways-to-estimate-realized-volatility/) - Macrosynergy
- [Federal Reserve Bank of Chicago Realized Volatility](https://www.chicagofed.org/~/media/publications/working-papers/2008/wp2008-14-pdf.pdf) - Technical paper

### Empirical Validation and Backtesting
- [INVESTMENT MODEL VALIDATION: A Guide for Practitioners](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/investment-model-validation.pdf) - CFA Institute
- [Backtest overfitting in the machine learning era](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110) - ScienceDirect 2024
- [A hybrid stock prediction method based on periodic/non-periodic features analyses](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-024-00517-7) - EPJ Data Science 2024
- [Putting Your Forecasting Model to the Test: A Guide to Backtesting](https://towardsdatascience.com/putting-your-forecasting-model-to-the-test-a-guide-to-backtesting-24567d377fb5) - Towards Data Science
- [Deep learning in the stock market—a systematic survey](https://link.springer.com/article/10.1007/s10462-022-10226-0) - AI Review

### Recent Time Series and Parameter Estimation (2024-2025)
- [Time Series Foundation Models for Multivariate Financial Time Series Forecasting](https://arxiv.org/html/2507.07296v1) - ArXiv 2025
- [Financial Time Series Forecasting: A Comprehensive Review](https://link.springer.com/article/10.1007/s10614-025-10899-z) - Computational Economics 2025
- [Moderate Time-Varying Parameter VARs](https://www.oru.se/globalassets/oru-sv/institutioner/hh/workingpapers/workingpapers2025/wp-16-2025.pdf) - Working paper 2025
- [Deep learning models for price forecasting of financial time series](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1519) - WIREs 2024
- [Financial Time Series Analysis with Transformer Models](https://www.researchgate.net/publication/387524930_Financial_Time_Series_Analysis_with_Transformer_Models) - ResearchGate

---

## 10. Conclusions

The literature on parameter calibration and estimation in stock price models reveals a mature but actively evolving field. Key findings:

1. **Multiple Valid Approaches**: MLE, Bayesian, GMM, and market-based methods each have merit depending on model structure and available data. No single "best" approach.

2. **Volatility Remains Central**: Whether constant (Black-Scholes), stochastic (Heston), rough (fractional dynamics), or time-varying, volatility parameter estimation is the critical bottleneck in most applications.

3. **Deep Learning Promise and Peril**: Neural networks enable fast calibration and improved forecasting, but sacrifice interpretability and theoretical grounding. Hybrid approaches (GARCH-NN, Bayesian-NN) appear most promising.

4. **Empirical Challenges Real Models**: Theoretical advances (rough volatility) sometimes underperform simpler alternatives empirically. This highlights importance of careful out-of-sample validation.

5. **Computational Efficiency Critical**: For real-time risk management and trading, fast calibration is essential. This drives adoption of neural network methods despite interpretability concerns.

6. **Regime-Switching and Time-Variation Underexplored**: Most models assume stable parameters; adapting to market regimes remains an open problem.

7. **Best Practices Consolidating**: Walk-forward validation, proper cross-validation (CPCV), and hybrid estimation methods are becoming standard practice.

Future research should focus on: (a) reconciling rough vs. Markovian volatility empirically; (b) scalable methods for high-dimensional portfolios; (c) robust parameter estimation across market regimes; and (d) interpreting neural network-based estimators.
