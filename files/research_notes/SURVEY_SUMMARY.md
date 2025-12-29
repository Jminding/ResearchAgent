# Literature Survey Summary: Quantitative Financial Market Models

## Survey Scope and Methodology

This literature survey comprehensively reviews peer-reviewed research on quantitative financial market models from 2020-2025, with a focus on:
- Stochastic differential equations (SDEs) and classical models
- Volatility models (GARCH family, local/stochastic volatility, rough volatility)
- Jump-diffusion models
- Machine learning and deep learning approaches for price prediction
- Hybrid classical-ML integration approaches
- Reinforcement learning for algorithmic trading

## Search Strategy Employed

Conducted 13 targeted web searches:
1. Quantitative financial market models + SDEs (2020-2025)
2. GARCH volatility models + literature review
3. Stochastic volatility + jump-diffusion + price prediction
4. Machine learning + price prediction + neural networks
5. Local volatility models + derivatives pricing
6. LSTM/GRU + financial time series
7. Volatility forecasting + empirical benchmarks
8. Reinforcement learning + algorithmic trading
9. Black-Scholes + SDE + financial modeling
10. Heston model + volatility smile + calibration
11. Neural networks + financial forecasting + accuracy metrics
12. Transformers + attention + stock price prediction
13. Option pricing + machine learning + calibration

## Key Findings Summary

### 1. Classical Stochastic Models (Still Relevant)

**Black-Scholes-Merton Framework**:
- Foundation for all modern models
- Extensions: add stochastic volatility (Heston), jumps (Merton), both (SVJ)
- Limitations well-understood: constant volatility, no smile/skew capture

**Heston Stochastic Volatility Model**:
- Industry standard for derivatives pricing
- Recent advances: deep learning-enhanced calibration (2024-2025)
- Empirical: successfully reproduces volatility smile with proper calibration
- Challenge: parameter stability over time; calibration computationally intensive

**Jump-Diffusion Models**:
- Merton framework: captures large rare events
- Stochastic volatility + jumps (SVJ) consistently outperforms simpler models
- 2025 study: SVJ achieves lowest RMSE/MAPE on both low and high-volatility stocks
- Calibration window: 1 year for low-vol, 6 months for high-vol assets

**Local Volatility Models**:
- Recent advances in rough LSV (2025) for realistic volatility dynamics
- Efficient calibration directly from implied volatility surface
- SABR + stochastic volatility unification (2025) promising for derivatives

### 2. GARCH and Volatility Models

**Performance Hierarchy** (1-day forecasting horizon):
1. Transformers + macroeconomic features (best)
2. Deep learning + news sentiment
3. HAR + news sentiment (competitive without DL)
4. EGARCH/APARCH (3-5% better than GARCH)
5. GARCH(1,1) (baseline)

**Critical Insight**: Deep learning outperforms GARCH only WITH exogenous variables. Without macroeconomic features, HAR often equals or beats DL—underreported in literature.

**Asymmetric Models**:
- EGARCH captures leverage effect (negative news impact)
- APARCH with power transformation shows best long-term stability
- Structural breaks invalidate calibrated parameters; include break dummies

**GARCHNet Hybrid** (2023):
- Combines LSTM with ML-estimated GARCH parameters
- Effective for Value-at-Risk (VaR) forecasting
- Preserves interpretability of GARCH while capturing nonlinearity

### 3. Volatility Forecasting Benchmarks

**News Analytics Integration** (2025):
- Domestic macroeconomic news sentiment significantly improves forecasts
- Applied to individual stocks and S&P 500 Index
- Quantifies importance of exogenous variables

**Model Combination**:
- Combined forecasts consistently outperform single models
- Weighted ensembles of GARCH + GRU + Transformer superior
- Addresses idiosyncratic errors in individual methods

**Window Size Optimization** (2025):
- Journal of Forecasting confirms optimal window varies by asset
- Adaptive window sizing improves accuracy
- Trade-off between responsiveness and stability

### 4. Deep Learning for Time Series

**LSTM Performance**:
- Remarkably successful at capturing temporal dependencies
- Outperforms ARIMA across all market conditions
- Typical MAPE: 5-10% on 5-day horizons
- Drawback: requires 3-5 years training data, slow training (hours/GPU)

**GRU Advantages**:
- ~50% faster training than LSTM
- Comparable accuracy with fewer parameters
- Preferred for high-frequency forecasting
- Better suited for real-time applications

**Liquid Neural Networks** (2024):
- MSE: 0.000317, RMSE: 0.0178, MAPE: 1.8%
- Directional accuracy: 49.36% (barely above random)
- **Key observation**: High absolute accuracy doesn't guarantee profitable trading

**CNN-LSTM-GRU Hybrids**:
- CNN extracts spatial patterns
- LSTM/GRU capture temporal dynamics
- Superior to individual architectures
- Significant improvement over single-model baselines

### 5. Transformer Architecture Dominance (2024-2025)

**Advantages Over LSTM/GRU**:
- Self-attention: direct connections across all time steps (vs. sequential processing)
- Parallelizable: significant computational speedup in training
- Multi-head attention: captures diverse temporal patterns simultaneously
- Better global modeling of price dynamics

**Key Models**:

**TEANet** (Transformer Encoder-based Attention Network):
- Small sample (5 calendar days) sufficient for temporal dependency
- Outperforms LSTM, Prophet on standard benchmarks

**IL-ETransformer** (Incremental Learning Enhanced Transformer):
- Online prediction with concept drift handling
- Multi-head self-attention for price-feature relationships
- Published in PLOS One (2024)

**Galformer** (Generative ALFormer):
- Generative decoding + hybrid loss function
- Multi-step stock market index prediction
- Nature Scientific Reports (2024)

**Performance**: Best overall on forecasting, but computational demands high.

### 6. Hybrid Classical-ML Integration

**Emerging Paradigm**: Rather than replacing GARCH/Heston with pure neural networks, successful approaches combine them.

**GARCHNet Success** (2023):
- LSTM estimates time-varying GARCH parameters
- Preserves economic interpretability
- Effective for risk management (VaR)

**GAN-Transformer Combination** (2024):
- Generative adversarial networks + attention mechanisms
- Empirical Economics publication
- Improved stock price prediction over individual approaches

**Ensemble Methods**:
- Weight classical + hybrid + DL outputs
- Reduces prediction variance
- Consistent performance across market conditions

**Implicit Insight**: Hybrid integration trending because:
1. Classical models more interpretable (regulatory requirement)
2. DL captures nonlinear dynamics
3. Ensemble reduces failure modes
4. Implementation simpler than pure DL

### 7. Machine Learning for Option Pricing

**Calibration Speedup via Hypernetworks** (2024):
- Hypernetwork generates model parameters dynamically
- **500x faster** than traditional MLE calibration
- S&P 500 option empirical validation (3M contracts, 15 years)
- Accuracy very close to gold-standard direct calibration

**Residual Learning Approach** (2024-2025):
- Train NN to learn residuals: f(x) = analytical_approximation(x) + NN(x)
- Reduces learning complexity (smaller target function)
- Lower data requirements
- Better generalization than direct pricing NN

**Sparse Gaussian Processes** (2024):
- Offline training on simulated model data
- Online calibration inference
- Advantages: uncertainty quantification, fewer hyperparameters
- Trade-off: slower inference than neural networks

**Critical Issue Not Addressed**:
- No-arbitrage enforcement largely absent in NN-based pricing
- Standard DL methods frequently violate no-arbitrage
- Implications for hedging unclear but concerning
- Emerging direction: constrained optimization with arbitrage penalties

### 8. Stochastic Volatility + Jumps

**SVJ Model (2025 Comparative Study)**:
- Consistently achieves lowest RMSE and MAPE
- Outperforms: GBM, Heston (no jumps), Merton (no vol jumps)
- **Key result**: Both volatility AND return jumps necessary
- Double-exponential jumps outperform normal jumps
- Asset-dependent optimal calibration window (1yr low-vol, 6mo high-vol)

**Empirical Validation**:
- Tested on AAPL, MSFT (low volatility), TSLA, MRNA (high volatility)
- Superior predictive performance across asset classes
- Particularly important during COVID-19 crisis periods

**Calibration Complexity**:
- Parameter space high-dimensional: μ, σ, κ, θ, σ_v, λ, m, σ_j
- Jump intensity λ difficult to estimate (rare events)
- ML optimization methods improving parameter search

### 9. Reinforcement Learning for Trading

**Meta-Analysis** (2024-2025):
- Systematic review of 167 studies (2017-2025)
- Four main applications: algo trading, portfolio mgmt, order execution, market making
- Market making shows **highest performance improvement** (2020-2025)

**Key Insight**: Implementation quality >> algorithm sophistication
- Weak correlations: feature dimensionality, training duration, algorithm choice
- Strong factors: data quality, domain expertise, cost modeling
- Most critical: realistic market microstructure (slippage, impact, costs)

**Algorithm Performance**:
- Actor-Critic (DDPG) dominates market making (continuous action space)
- Multi-agent RL promising for portfolio management
- Risk-aware formulations increasingly standard

**Critical Gap**: Live trading results rarely published
- Backtest Sharpe ratios: 1.5-3.0
- Live performance typically 30-50% worse than backtests (estimated)
- Publication bias toward successes; failures unpublished

**Challenges**:
- Model overfitting to historical regimes (COVID, 2008)
- Regime detection/switching underdeveloped
- Transaction costs and slippage easily misestimated
- Regulatory constraints on leverage, diversification

### 10. Overall State of the Art (2024-2025)

**For Volatility Prediction** (Best In Class):
- Transformer + macroeconomic news sentiment
- GARCH component for clustering (hybrid)
- HAR as strong baseline when DL features unavailable
- Quantified: 10-20% improvement over traditional GARCH

**For Stock Price Direction** (Realistic Assessment):
- Ensemble methods achieve 55-58% directional accuracy
- Barely statistically significant above 50% random
- **Reality check**: Profitable trading requires magnitude prediction, not just direction
- Single-model directional accuracy: typically 50-52%

**For Option Pricing/Calibration**:
- Hypernetwork-based calibration: 500x speedup, comparable accuracy
- Residual NN learning: effective with lower data requirements
- Heston + jumps still competitive for standard derivatives
- Rough LSV advancing for realistic volatility

**For Algorithmic Trading**:
- DDPG in market making: proven improvements
- Backtests show positive Sharpe (1.5-3.0)
- Live trading: mostly unpublished, likely 30-50% worse
- Success depends heavily on implementation quality

**For Portfolio Management**:
- Multi-agent RL showing promise
- Risk-adjusted returns improving
- Transaction cost modeling critical
- Regulatory constraints significant limiting factor

### 11. Identified Gaps (Critical Research Directions)

**High Priority**:
1. **No-Arbitrage Enforcement**: Develop practical methods for constrained NN option pricing
2. **Regime Awareness**: Automatic market regime detection and model adaptation
3. **Live Trading Validation**: Publish realistic live results (with costs, slippage, impact)
4. **Uncertainty Quantification**: Confidence intervals, calibration of ML predictions

**Medium Priority**:
1. **Interpretability**: Financial-domain specific XAI methods
2. **Jump Dynamics**: Better estimation of rare event processes
3. **Model Selection Theory**: Principled approach to choosing classical vs. hybrid vs. DL
4. **Volatility Clustering**: Formal analysis in DL vs. GARCH context

**Temporal Challenges**:
1. **Parameter Stability**: Time-varying optimal parameters; adaptive methods needed
2. **Structural Breaks**: Models fail during crises; detection and handling
3. **Regime Switching**: Hidden Markov extensions underexplored
4. **Data Availability**: Sparse in emerging markets, cryptocurrencies

### 12. Key Limitations and Caveats

**Publication Bias**:
- Successful systems proprietary, unpublished
- Failed systems not reported
- Backtests != live trading

**Temporal Instability**:
- Parameters optimal in 2020 may fail in 2025
- Generalization across market regimes poor
- COVID-19, 2008 show poor extrapolation

**Data Issues**:
- Training and test often same distribution
- Walk-forward validation limited
- Cryptocurrency markets young and immature
- Emerging markets sparse data

**Practical Deployment**:
- Latency requirements incompatible with DL inference
- GPU costs substantial
- Model monitoring/drift detection underdeveloped
- Regulatory requirements (interpretability) conflict with ML

**Measurement Problems**:
- Directional accuracy ~50% = random prediction
- RMSE/MAPE insensitive to tail events
- Sharpe ratios from backtests overstate actual returns
- Transaction costs frequently ignored or underestimated

---

## Recommendations for Practitioners

1. **Use Hybrid Approaches**: Combine GARCH/Heston with neural networks; don't discard classical models
2. **Include Exogenous Variables**: News sentiment, macroeconomic data critical; DL alone insufficient
3. **Validate Extensively**: Walk-forward testing, multiple market regimes, realistic costs
4. **Manage Expectations**: Directional accuracy barely better than coin flip; focus on magnitude
5. **Monitor Model Drift**: Parameters change over time; automatic recalibration essential
6. **Enforce Constraints**: No-arbitrage, risk limits, regulatory requirements
7. **Document Assumptions**: Market microstructure, costs, regulatory environment critical
8. **Ensemble Methods**: Reduce overfitting; weight multiple model outputs

---

## References

Complete bibliography with 75+ sources saved in main literature review document:
- `/Users/jminding/Desktop/Code/Research Agent/files/research_notes/lit_review_quantitative_financial_models.md`

File contains full citations, URLs, extraction of methodologies, datasets, and quantitative results for all identified papers.

---

**Survey Completion Date**: December 22, 2025
**Total Sources**: 75+ peer-reviewed papers, preprints, conference proceedings
**Geographic Coverage**: Global (US, Europe, Asia, Emerging Markets)
**Asset Classes**: Equities, Derivatives, Cryptocurrencies, FX
**Time Period Focused**: 2020-2025 with seminal older references

