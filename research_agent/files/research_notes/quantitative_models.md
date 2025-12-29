# Literature Review: Quantitative Models for Financial Markets (2018-2025)

## Overview of the Research Area

Quantitative modeling of financial markets has undergone a paradigm shift over the past 5-7 years, transitioning from purely traditional econometric approaches (ARIMA, GARCH, factor models) to sophisticated machine learning and deep learning architectures. The field now encompasses several major research directions: (1) deep neural network architectures for time series prediction; (2) hybrid econometric-neural models combining classical stochastic processes with neural networks; (3) transformer-based attention mechanisms for capturing market dynamics; (4) reinforcement learning for portfolio optimization and trading; (5) generative models (GANs, VAEs) for synthetic market simulation; and (6) factor mining using algorithmic approaches and graph neural networks. This literature review synthesizes 45+ recent papers and preprints from 2018-2025, with emphasis on methodological innovations, empirical performance metrics, and identified limitations.

---

## Chronological Summary of Major Developments

### 2018-2020: Foundation Period
- Establishment of LSTM and RNN as baseline deep learning approaches for stock price forecasting
- Early exploration of hybrid GARCH-neural network models combining econometric and statistical learning approaches
- Introduction of ensemble methods (bagging, boosting, stacking) for improved financial forecasting
- Credit risk prediction using gradient boosting methods (XGBoost, LightGBM)

### 2021-2022: Transformer and Architecture Diversification
- Emergence of transformer-based architectures (PatchTST, iTransformer) adapting vision transformers to time series
- Physics-informed neural networks (PINNs) for option pricing and calibration of stochastic volatility models
- Neural Ordinary Differential Equations (NODEs) for continuous-time financial modeling
- Variational Autoencoders (VAEs) for synthetic financial data generation and dimensionality reduction
- Generative Adversarial Networks (GANs) for volatility surface computation and market microstructure simulation
- Graph Neural Networks (GNNs) applied to market microstructure and volatility spillover prediction

### 2023-2024: Advanced Hybrid Systems and Explainability
- Refined multi-task learning frameworks for cross-stock prediction leveraging inter-stock relationships
- Transfer learning approaches for domain adaptation across markets and asset classes
- Comprehensive surveys on reinforcement learning in quantitative finance (167+ publications reviewed)
- Integration of sentiment analysis (NLP, FinBERT) with deep learning for return prediction
- SHAP/LIME explainability methods applied to trading and credit risk models
- Hybrid GARCH-informed neural networks combining domain knowledge with learning capacity
- Large Language Models (LLMs) for fundamental factor discovery and news-based predictions

### 2025: Emerging Frontiers
- Quantum machine learning applications in portfolio optimization and risk analysis
- Neural differential equations for high-frequency trading (1-, 5-, 30-minute prediction)
- Extreme learning machines for rapid training of quantitative models
- Dynamic graph neural networks for real-time volatility spillover monitoring
- Multi-agent reinforcement learning frameworks for coordinated portfolio management and execution
- Diffusion models and score-based generative models for market scenario simulation

---

## Detailed Survey of Prior Work

### A. Deep Learning Architectures for Time Series Prediction

#### LSTM and RNN Models
**Papers:**
- Stock Market Prediction Using LSTM Recurrent Neural Network (PMC, peer-reviewed)
- Advanced Stock Market Prediction Using Long Short-Term Memory Networks (arXiv 2505.05325, 2025)
- Exploring Different Dynamics of Recurrent Neural Network Methods for Stock Market Prediction - A Comparative Study (2024)

**Problem Statement:** Traditional time series models (ARIMA, exponential smoothing) assume linearity and cannot capture long-range temporal dependencies in nonlinear financial data.

**Methodology:**
- LSTM networks introduce memory cells and gating mechanisms to selectively retain/forget information over arbitrary time horizons
- Bidirectional LSTM (BiLSTM) processes sequences in both forward and backward directions to capture bidirectional temporal dependencies
- Variants include Gated Recurrent Units (GRU) reducing computational cost while maintaining expressiveness

**Datasets:**
- S&P 500, NASDAQ indices
- Google stock prices
- DAX, DOW, S&P500 closing prices
- CSI 300 Index (Chinese market)

**Key Results:**
- LSTM consistently outperforms ARIMA and simple ANN models, demonstrating robust accuracy during volatile phases
- BiLSTM-GARCH hybrid outperforms LSTM-GARCH due to bidirectional dependency capture
- Ensemble of CNN-LSTM and GRU-CNN models produces superior forecasts for multi-step ahead prediction
- LSTM with genetic algorithm (GA) optimization: improved prediction accuracy through hyperparameter tuning

**Limitations and Assumptions:**
- Requires substantial historical data (typically 3-5 years minimum)
- Sensitive to data normalization and feature scaling
- Memory and computational burden increases with sequence length
- Struggles with distribution shifts in market regimes
- Assumes stationarity or suitable preprocessing (differencing)

---

#### Convolutional Neural Networks (CNNs) and Hybrid Architectures
**Papers:**
- Data-driven stock forecasting models based on neural networks: A review (2024)
- Stock Price Prediction Using CNN-BiLSTM-Attention Model (MDPI 2024)
- Enhanced stock market forecasting using dandelion optimization-driven 3D-CNN-GRU classification (Nature Scientific Reports 2024)
- CNN-based stock price forecasting using chart images (2023)

**Problem Statement:** CNNs excel at spatial feature extraction from images and high-dimensional structured data; their application to financial time series exploits local temporal patterns and hierarchical feature learning.

**Methodology:**
- Graph Convolutional Feature-based CNN (GC-CNN) combines inter-stock correlations with individual price dynamics
- CNN-BiLSTM architecture: CNN extracts temporal features (filters); BiLSTM captures sequential patterns; Attention layer assigns dynamic weights to historical periods
- 3D-CNN variants for multi-resolution time series (treating time as depth dimension)
- CNN applied to stock chart images (candlestick patterns) to identify technical patterns

**Datasets:**
- Various stock indices and individual stocks
- Chinese A-share market
- Intraday price chart images (1-minute to daily candlesticks)

**Key Results:**
- GC-CNN outperforms baseline LSTM and CNN models due to correlation modeling
- CNN-BiLSTM-Attention: achieves higher accuracy than pure LSTM or CNN baselines
- Dandelion optimization of 3D-CNN-GRU hyperparameters: improved RMSE vs. standard configurations
- Chart-based CNN strategy: relatively high Sharpe ratio (>0.5) outperforming short-term momentum strategies

**Limitations and Assumptions:**
- Chart patterns are partially random; technical analysis claims may not have reliable predictive power
- Computational cost of multi-dimensional convolutions
- Feature interpretability reduced compared to traditional econometric models
- Requires careful engineering of input representations (chart normalization, time window selection)

---

#### Transformer Models and Attention Mechanisms
**Papers:**
- A novel transformer-based dual attention architecture for financial time series prediction (Journal of King Saud University 2024)
- Deep context-attentive transformer transfer learning for financial forecasting (PeerJ 2024)
- A Financial Time-Series Prediction Model Based on Multiplex Attention and Linear Transformer Structure (Applied Sciences 2024)
- PatchTST: Patch-time series transformer for revisiting long sequence time-series forecasting (arXiv, referenced in 2023-2024 literature)
- iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (2023)

**Problem Statement:** Transformers capture global long-range dependencies via self-attention; recent innovations adapt vision transformer designs to time series by treating channels/variables as "patches" rather than tokens.

**Methodology:**
- Standard transformer: multi-head self-attention over time steps, then feed-forward networks
- PatchTST: divides time series into overlapping or non-overlapping patches; attention operates on patches, reducing sequence length and computational cost
- iTransformer: inverted attention operates across variables (channels) rather than time steps, suitable for multivariate time series
- Dual attention mechanisms: temporal attention (across time) + feature/channel attention (across variables)
- Signal decomposition (trend + seasonal + residual) + transformer encoder-decoder
- Integration with cross-entropy loss for return direction classification

**Datasets:**
- Stock indices (S&P 500, NASDAQ, CSI 300)
- Multivariate financial time series (OHLCV + sentiment features)
- Transfer learning: pre-training on large benchmark datasets (e.g., electricity consumption) then fine-tuning on specific stocks

**Key Results:**
- Transformer models capture longer temporal dependencies (100+ time steps) than LSTMs
- PatchTST achieved best out-of-sample R² on CSI 300 vs. ARIMA, GARCH, and linear models
- iTransformer outperforms standard transformer on many financial benchmarks
- Dual attention: 3-5% improvement in MAE/RMSE vs. single-attention variants
- Transfer learning fine-tuning reduces required training data by ~30-40% while maintaining performance

**Limitations and Assumptions:**
- High computational demand (O(n²) complexity in sequence length for self-attention)
- Extensive hyperparameter tuning required (attention heads, patch size, depth)
- Data preprocessing complexity (signal decomposition, normalization)
- Interpretability of attention weights remains contested (not all attention corresponds to causal relationships)
- Limited theoretical justification for why transformers should outperform domain-aware econometric models

---

### B. Hybrid Econometric-Neural Models

#### GARCH and Neural Network Integration
**Papers:**
- A Hybrid GARCH and Deep Learning Method for Volatility Prediction (Journal of Applied Mathematics 2024)
- Volatility Forecasting using Hybrid GARCH Neural Network Models: The Case of the Italian Stock Market (2021)
- Forecasting the volatility of stock price index: A hybrid model integrating LSTM with multiple GARCH-type models (2021)
- GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets (ACM 2024)
- Volatility forecasting using deep recurrent neural networks as GARCH models (Computational Statistics 2023)
- The Sentiment Augmented GARCH-LSTM Hybrid Model for Value-at-Risk Forecasting (Computational Economics 2025)

**Problem Statement:** GARCH models capture conditional heteroscedasticity and volatility clustering but assume linear dynamics; neural networks capture nonlinear patterns but lack interpretability. Hybrid models combine both.

**Methodology:**
- LSTM-GARCH: LSTM predicts residuals from GARCH fit, capturing nonlinear deviations
- Bidirectional LSTM (BiLSTM) + EGARCH (Exponential GARCH) for asymmetric volatility
- CNN-GARCH: CNN extracts features from multivariate inputs; GARCH models conditional variance
- GRU-GARCH: Gated Recurrent Unit variant of LSTM-GARCH
- Sentiment-augmented GARCH-LSTM: incorporates text sentiment scores as additional input to GARCH equation
- Residual learning: neural network learns additive correction to GARCH baseline
- Information pooling: GARCH outputs (conditional variance, risk measures) as features to neural network

**Datasets:**
- Italian stock market indices
- S&P 500, NASDAQ futures
- EUR-USD exchange rates
- Multiple-asset portfolios with cross-sectional dependencies

**Key Results:**
- Hybrid LSTM-GARCH: 15-30% improvement in volatility forecast RMSE vs. pure GARCH
- BiLSTM-GARCH: further 5-10% improvement over LSTM-GARCH (better bidirectional capture)
- CNN-GARCH: effectively solves GARCH shortcomings in capturing complex temporal patterns
- ANN-GARCH hybrid: 30.6% improvement in prediction over GARCH baseline
- Sentiment-GARCH-LSTM: reduces Value-at-Risk (VaR) forecast errors by 12-18% during high-sentiment periods
- Outperforms: ARIMA, EGARCH, simple neural networks, and ensemble baselines

**Limitations and Assumptions:**
- Assumes GARCH component remains valid under regime shifts
- Requires careful calibration of GARCH orders (p, q) and neural network depth
- Computational cost higher than pure GARCH or pure neural network
- Residuals from GARCH must exhibit nonlinear patterns for hybrid to outperform
- Overfitting risk if neural network component is too expressive relative to GARCH baseline
- Assumes additive decomposition of volatility (linear + nonlinear) may not hold under extreme events

---

#### Neural Ordinary Differential Equations (NODEs)
**Papers:**
- Phase Space Reconstructed Neural Ordinary Differential Equations Model for Stock Price Forecasting (PACIS 2024)
- Building a High-Frequency Trading Algorithm Using an Ordinary Differential Equation Recurrent Neural Network (SSRN 2024)
- Neural Ordinary Differential Equation Networks for Fintech Applications Using Internet of Things (IEEE 2024)
- Financial Time Series Prediction via Neural Ordinary Differential Equations Approach (IEEE 2024)
- Forecasting with an N-dimensional Langevin equation and neural-ordinary differential equation (arXiv 2405.07359, 2024)

**Problem Statement:** Neural ODEs provide continuous-time dynamics for financial modeling, avoiding discrete time step artifacts and enabling natural incorporation of stochastic processes.

**Methodology:**
- Neural ODE: represents hidden state evolution as solution to ODE: dh(t)/dt = f_θ(h(t), t), solved via adjoint method
- Phase Space Reconstruction (PSR): reconstructs high-dimensional manifold from univariate time series using delay embedding
- PSR-NODE: combines PSR to enrich state space with NODE for continuous prediction
- ODE-LSTM hybrid: LSTM for feature extraction, NODE for temporal dynamics
- Langevin NODE: incorporates stochastic differential equation with added noise term for volatility modeling

**Datasets:**
- Technology stocks (Apple, Microsoft, Google)
- Finance sector stocks
- Pharmaceutical stocks
- Intraday price data (1-minute, 5-minute, 30-minute frequencies)

**Key Results:**
- PSR-NODE achieves superior performance across technology, finance, and pharmaceutical sectors
- Outperforms LSTM, RNN, CNN, and standard Transformer models
- Better handling of chaotic price dynamics and regime changes
- ODE-LSTM for high-frequency: effective prediction across multiple time scales (1-min to 30-min)
- Langevin NODE: captures fat-tailed distributions and volatility clustering observed in real data

**Limitations and Assumptions:**
- Requires numerical ODE solver (adjoint method) adding computational overhead
- Fewer empirical studies compared to LSTM/Transformer literature
- Assumes underlying continuous dynamics, which may not hold for discrete exchange systems
- Hyperparameter selection (ODE solver tolerance, integration method) impacts performance
- Limited theoretical understanding of when continuous vs. discrete models are appropriate
- Scalability to very high-dimensional systems (>1000 variables) remains unexplored

---

### C. Factor Models and Machine Learning

#### Traditional Factor Models and ML Enhancements
**Papers:**
- Factor Models, Machine Learning, and Asset Pricing (Annual Review of Financial Economics 2022)
- From Factor Models to Deep Learning: Machine Learning in Reshaping Empirical Asset Pricing (arXiv 2403.06779, 2024)
- Fundamental Factor Models Using Machine Learning (ResearchGate/SCIRP 2018)
- The pricing ability of factor model based on machine learning: Evidence from high-frequency data in China (ScienceDirect 2025)
- The Fama 3 and Fama 5 factor models under a machine learning framework (2018-2019)

**Problem Statement:** Linear factor models (Fama-French 3/5) assume constant factor loadings and linear risk premia; machine learning extends these to nonlinear mappings and time-varying parameters.

**Methodology:**
- Non-linear extensions: SVM, random forests, neural networks approximating E[R_i] = α + Σ_k β_ik * F_k (nonlinearly)
- Dimensionality reduction: PCA, VAE, autoencoders to extract implicit factors from high-dimensional firm characteristics
- Factor discovery via regularized regression (elastic net, lasso) on firm characteristics
- Genetic programming for algorithmic factor mining (automated feature engineering)
- Neural symbol regression: discovers interpretable mathematical expressions for factors
- Stochastic discount factor (SDF) estimation: neural networks directly model m(s) rather than linear approximations

**Datasets:**
- Fama-French factor library (HML, SMB, RMW, CMA factors)
- CSI 300 Index (China) for high-frequency tests
- Firm characteristics (momentum, value, quality, investment, profitability)
- US market 1926-2020, international markets

**Key Results:**
- Deep learning methods outperform linear models on CSI 300: neural networks with 2 hidden layers achieve highest out-of-sample R²
- Non-linear factor models: 5-15% improvement in alpha detection vs. linear baselines
- Factor mining: genetic algorithms discover novel factors uncorrelated with standard factors
- Improved return predictions: combining traditional factors with machine-learned factors yields 20-30% reduction in OOS RMSE
- SDF estimation: neural networks provide more accurate state price density estimation than parametric assumptions

**Limitations and Assumptions:**
- Out-of-sample performance depends heavily on test period; results sensitive to regime shifts
- Overfitting risk: high-dimensional nonlinear models with limited data (historical periods)
- Interpretability: machine-learned factors lack economic intuition vs. manually-constructed factors
- Survivor bias: historical characteristic datasets exclude delisted firms
- Assumes factors remain stable across time; factor premiums may have dissipated in recent decades
- Computational cost of exhaustive nonlinear exploration prohibitive for some approaches

---

### D. Reinforcement Learning for Portfolio Optimization and Trading

**Papers:**
- A novel multi-agent dynamic portfolio optimization learning system based on hierarchical deep reinforcement learning (Complex & Intelligent Systems 2025)
- A Systematic Approach to Portfolio Optimization: Comparative Study of RL Agents, Market Signals, and Investment Horizons (Algorithms 2025)
- Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative (ICAPS FinPlan 2023)
- Multiagent-based deep reinforcement learning framework for multi-asset adaptive trading and portfolio management (ScienceDirect 2024)
- Deep Reinforcement Learning for Portfolio Optimization using Latent Feature State Space (LFSS) Module (arXiv 2102.06233)
- Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization: A Multi-reward Approach (International Journal Computational Intelligence Systems 2025)
- The Evolution of Reinforcement Learning in Quantitative Finance: A Survey (arXiv 2408.10932, 2024)

**Problem Statement:** Traditional portfolio optimization (Markowitz) assumes known covariance matrices and linear constraints; RL enables dynamic, adaptive allocation responding to changing market conditions via Markov Decision Processes.

**Methodology:**
- Deep Q-Network (DQN): learns action-value function Q(s,a) for asset allocation decisions
- Policy Gradient Methods (A3C, PPO, DDPG): directly optimize portfolio weight policy π(a|s)
- Actor-Critic: combines policy gradient (actor) with value function baseline (critic) for reduced variance
- Multi-agent RL: independent agents for each asset or portfolio component with coordination mechanisms
- Model-based RL: learn environment dynamics p(s'|s,a) to enable planning
- State space: portfolio holdings, market prices, volatility, sentiment, macro indicators
- Action space: continuous asset weights; discrete buy/sell/hold actions
- Reward specification: cumulative return, Sharpe ratio, return - λ×risk, or other risk-adjusted metrics
- Meta-learning: adapt policy across different market regimes without full retraining

**Datasets:**
- S&P 500, NASDAQ stocks
- Cryptocurrency portfolios
- Multi-asset classes (equities, bonds, commodities)
- Backtest periods: 2010-2020+, validation 2020-2024

**Key Results:**
- DQN agents: consistent outperformance of S&P 500 benchmarks in annualized returns
- DDPG for continuous allocation: smoother portfolio transitions, lower transaction costs
- Multi-agent RL: outperforms single-agent for N-asset portfolios (N>10)
- Risk-adjusted returns: multi-reward RL (return + Sharpe + Sortino) produces balanced solutions
- Training efficiency: meta-learning reduces convergence time for new market regimes by 30-50%
- Practical trading: RL policies successfully deployed in live trading environments with modified reward constraints

**Limitations and Assumptions:**
- Reward specification is task-dependent; poor reward design leads to unintended solutions (gaming)
- Sample efficiency: requires millions of environment interactions (simulated or historical)
- Sim-to-real gap: backtesting assumes perfect execution; real trading has slippage, bid-ask spreads
- Non-stationary environment: market regime changes invalidate learned policies; adaptation lag
- Computational resources: training DQN/DDPG agents CPU-intensive; real-time deployment limits model complexity
- Benchmark selection: different benchmarks yield different apparent outperformance
- Regulatory constraints: RL policies may violate concentration limits or sector restrictions

---

### E. Generative Models: GANs and VAEs

#### Generative Adversarial Networks (GANs)
**Papers:**
- Factor-GAN: Enhancing stock price prediction and factor investment with Generative Adversarial Networks (PMC 2024)
- Computing Volatility Surfaces using Generative Adversarial Networks with Minimal Arbitrage Violations (arXiv 2304.13128, 2023)
- Towards Realistic Market Simulations: a Generative Adversarial Networks Approach (arXiv 2110.13287, 2021)
- Can GANs Learn the Stylized Facts of Financial Time Series? (ACM 2024)
- VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces (Quantitative Finance 2024)
- Enhancing stock price prediction using GANs and transformer-based attention mechanisms (Empirical Economics 2024)
- Fin-GAN: forecasting and classifying financial time series via generative adversarial networks (Quantitative Finance 2023)
- Modeling financial time-series with generative adversarial networks (2019)

**Problem Statement:** GANs learn data distribution p(x) by adversarial game between generator (creates synthetic data) and discriminator (distinguishes real vs. synthetic). For finance, enables realistic scenario generation and market simulation.

**Methodology:**
- Standard GAN: min_G max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]
- Conditional GAN (CGAN): condition on observable state (current market conditions) to generate next prices
- Wasserstein GAN (WGAN): replaces JS divergence with Wasserstein distance, improving training stability
- Spectral normalization: stabilizes discriminator training
- Factor-GAN: GAN component extracts factors; prediction component uses factors for return forecasting
- VolGAN: specialized architecture for volatility surfaces; constraints to ensure arbitrage-free surfaces
- Fin-GAN: jointly optimizes forecasting loss + adversarial loss for better calibration
- Architecture: generator = MLP or temporal conv; discriminator = CNN or RNN

**Datasets:**
- S&P 500, NASDAQ, Dow Jones daily returns
- Options data: implied volatility surfaces across strikes/maturities
- Synthetic market simulator: generate limit order books, trade execution flows
- Long time periods: 10-20 years to capture rare events

**Key Results:**
- CGAN-generated limit orders: realistic stylized facts closer to historical data than simple baselines
- VolGAN: computes arbitrage-free volatility surfaces from limited data; improves interpolation vs. traditional splines
- GAN-based returns: recover statistical properties (linear unpredictability, heavy tails, volatility clustering, leverage effects, gain/loss asymmetry)
- Fin-GAN: 10-15% improvement in return classification accuracy vs. non-adversarial deep learning
- Factor-GAN: jointly optimized factors and return predictions; outperforms sequential approaches
- Scenario generation: GAN-generated paths useful for stress testing and risk management

**Limitations and Assumptions:**
- Mode collapse: generator learns to produce limited variety of outputs
- Training instability: requires careful tuning of learning rates, architectures, regularization
- Computational cost: adversarial training slower than supervised baselines
- Evaluation difficulty: assessing whether synthetic data "realistic" is subjective; limited metrics
- Overfitting to training distribution: GANs may not extrapolate to out-of-distribution scenarios (e.g., extreme events)
- Constraint satisfaction: ensuring arbitrage-free surfaces requires additional penalty terms, not always effective
- Limited theoretical understanding of what GANs learn about market microstructure

---

#### Variational Autoencoders (VAEs)
**Papers:**
- Time-Causal VAE: Robust Financial Time Series Generator (arXiv 2411.02947, 2024)
- Hybrid variational autoencoder for time series forecasting (2023)
- Variational Autoencoders for Completing the Volatility Surfaces (MDPI 2024)
- An Overview of Variational Autoencoders for Source Separation, Finance, and Bio-Signal Applications (PMC 2022)

**Problem Statement:** VAEs learn latent representations q(z|x) of financial time series; can forecast future observations p(x_t|z) and generate synthetic data for stress testing.

**Methodology:**
- VAE structure: encoder q_φ(z|x) maps observations to latent distribution N(μ, σ); decoder p_θ(x|z) reconstructs observations
- Loss: reconstruction + KL divergence regularizer ensuring latent distribution matches N(0, I)
- Recurrent VAE (RVAE / CVAE): uses LSTM/GRU in encoder/decoder for sequential data
- Time-Causal VAE: ensures future values don't influence past latent representations (causal masking)
- Hybrid VAE: combines VAE with traditional forecasting (e.g., ARIMA) components
- Dimensionality reduction: learn compact latent representation of high-dimensional financial data
- Volatility surface completion: VAE imputes missing option prices while preserving arbitrage-free constraints

**Datasets:**
- Daily stock returns (multiple assets)
- Quarterly firm financial metrics (debt ratios, profitability, valuations)
- Options data: implied volatility across strikes and maturities
- Synthetic data evaluation: compare generated distributions to historical

**Key Results:**
- Time-Causal VAE: generates realistic financial time series preserving autocorrelation, heavy tails, volatility clustering
- Hybrid VAE-ARIMA: achieves better forecasting than pure VAE or ARIMA
- Volatility surface completion: successful interpolation while maintaining arbitrage-free properties
- Synthetic data quality: distributions closely match historical; useful for backtesting and data augmentation
- Dimensionality reduction: latent factors interpretable as market regimes, volatility states
- Generative performance: outperforms GANs on some metrics (e.g., distribution matching); slower training than GANs

**Limitations and Assumptions:**
- Posterior collapse: VAE may ignore latent variable if decoder powerful enough
- Linear assumption: Gaussian posterior may not capture multimodal or heavy-tailed latent distributions
- Forecasting performance: generally lower than supervised deep learning models (LSTM, Transformer)
- Hyperparameter sensitivity: β parameter balancing reconstruction/regularization crucial
- Evaluation: assessing synthetic data quality remains open; limited metrics beyond distribution tests
- Scalability: computational cost increases with latent dimensionality and sequence length

---

### F. Graph Neural Networks for Market Microstructure

**Papers:**
- A Review on Graph Neural Network Methods in Financial Applications (arXiv 2111.15367, 2021)
- Graph Theory Application in Market Microstructure Analysis (SSRN 2025)
- Statistical analysis and applications of financial network data in the era of digital intelligence (AIM Press 2025)
- Attention based dynamic graph neural network for asset pricing (PMC 2023)
- Forecasting cryptocurrency volatility: a novel framework based on the evolving multiscale graph neural network (Financial Innovation 2025)
- Dynamic graph neural networks for enhanced volatility prediction in financial markets (arXiv 2410.16858, 2024)

**Problem Statement:** Financial markets exhibit complex networked relationships (stock correlations, sector linkages, contagion channels); GNNs capture these topological structures alongside node features for improved predictions.

**Methodology:**
- Graph construction: nodes = assets (stocks); edges = correlation, causality, common sector, ownership links
- Graph Convolutional Network (GCN): aggregates neighboring node features via spectral convolutions
- Graph Attention Network (GAT): learns dynamic edge weights via attention mechanism
- Temporal Graph Attention Network (TGAT): captures time-varying volatility spillovers
- Heterogeneous GNNs: handle different node/edge types (firms, sectors, macro indicators)
- Dynamic graphs: edges/features update over time; uses recurrent mechanisms or time-attention
- Asset pricing GNN: predicts cross-sectional returns incorporating market structure
- Volatility spillover: predicts how volatility propagates across connected assets

**Datasets:**
- Stock correlation networks (daily returns)
- Supply chain networks: firm-to-firm connections
- Contagion networks: during crisis periods (2008, 2020)
- Cryptocurrency networks: transaction flows, price correlations
- Sector and industry classifications

**Key Results:**
- GCN for return prediction: captures market structure effects; outperforms single-asset models
- GAT for asset pricing: attention weights reveal important relationships; dynamic weights adapt to regime shifts
- TGAT for volatility: predicts spillovers 1-5 days ahead; incorporates macro shocks
- Cryptocurrency volatility: multiscale graph model captures short and long-range dependencies
- Contagion prediction: GNN identifies highly connected assets prone to systemic risk

**Limitations and Assumptions:**
- Graph construction is ad-hoc: correlation-based edges may reflect spurious relationships
- Scalability: full-graph attention O(N²) infeasible for 5000+ assets
- Interpretability: graph-learned representations less interpretable than factor models
- Temporal alignment: assumes consistent network topology; edges may appear/disappear over time
- Assumes Markovian property: future depends only on current graph state
- Causal inference: cannot distinguish correlation from causation in learned edges
- Evaluation difficulty: hard to benchmark against simpler baselines (e.g., time-varying correlation models)

---

### G. Option Pricing and Derivative Valuation

#### Neural Networks for Option Pricing
**Papers:**
- Neural Network Learning of Black-Scholes Equation for Option Pricing (arXiv 2405.05780, 2024)
- Option pricing with neural networks vs. Black-Scholes under different volatility forecasting approaches (ScienceDirect 2021)
- Physics-Informed Neural Networks (PINNs) for Option Pricing (MATLAB 2025)
- Calibrating the Heston Model with Deep Differential Networks (arXiv 2407.15536, 2024)
- Accelerated American Option Pricing with Deep Neural Networks (SSRN 2023)
- Option Pricing Based on the Residual Neural Network (Computational Economics 2023)
- Considering Appropriate Input Features of Neural Network to Calibrate Option Pricing Models (Computational Economics 2024)
- Pricing options with a new hybrid neural network model (ScienceDirect 2024)

**Problem Statement:** Black-Scholes assumes constant volatility and log-normal distribution; neural networks approximate option prices without parametric assumptions, enabling calibration and pricing under realistic dynamics.

**Methodology:**
- Supervised learning: train network to map (S, K, T, r, σ) → option price
- Physics-Informed Neural Networks (PINNs): incorporate Black-Scholes PDE as constraint in loss function
- Deep Differential Networks (DDN): gradient-based learning of pricing formula and partial derivatives
- Residual Neural Networks (ResNet): approximate option pricing formula with residual blocks
- Feature engineering: use implied volatility as input (better-conditioned) vs. option prices
- Bijective transformation: rescale inputs to ensure well-conditioned learning problem
- Unsupervised approach: learn from market prices without parametric assumption

**Datasets:**
- European and American options on S&P 500, individual equities
- Options on currency pairs, commodities
- Implied volatility surfaces (strikes × maturities)
- Heston stochastic volatility model simulations

**Key Results:**
- PINN for Black-Scholes: learns pricing PDE with comparable accuracy to analytical solution
- ResNet pricing: higher prediction accuracy than DNN and fully convolutional networks
- DDN for Heston calibration: learns calibrated parameters from option surface data
- American options: neural network acceleration from 12 hours (Monte Carlo) to 1.5 hours (NN) for 100-dimensional problems
- Feature engineering: bijective transformation reduces calibration error vs. raw features
- Regime-dependent pricing: neural networks adapt to changing volatility regimes better than fixed Black-Scholes

**Limitations and Assumptions:**
- Requires labeled training data (market prices or benchmark models)
- Extrapolation risk: trained outside historical range of strikes/maturities may be inaccurate
- Arbitrage enforcement: networks may not learn arbitrage-free prices without explicit constraints
- Computational cost: training slower than closed-form Black-Scholes; inference speed varies
- Volatility assumption: PINN with constant volatility assumption loses flexibility
- Theoretical gaps: no convergence guarantees or approximation error bounds
- Practical deployment: regulatory acceptance of neural network valuations uncertain

---

### H. Credit Risk and Default Prediction

**Papers:**
- Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers (MDPI 2024)
- Applying machine learning algorithms to predict default probability in the online credit market (ScienceDirect 2021)
- Measuring the model risk-adjusted performance of machine learning algorithms in credit default prediction (Financial Innovation 2022)
- A machine learning-based credit risk prediction engine system using a stacked classifier and filter-based feature selection (Journal of Big Data 2024)
- Machine learning techniques for default prediction: an application to small Italian companies (Risk Management 2023)
- Machine Learning and Credit Risk Modelling (S&P Global 2020)

**Problem Statement:** Traditional credit models (logistic regression) assume linear risk factors; machine learning captures nonlinear interactions and complex dependencies for more accurate default prediction.

**Methodology:**
- Classification algorithms: logistic regression, SVM, random forest, gradient boosting (XGBoost, LightGBM)
- Deep learning: feedforward neural networks, autoencoders for fraud detection
- Ensemble stacking: layer multiple base learners (RF, SVM, NB) with meta-learner (LR, GB)
- Feature selection: filter-based (correlation, mutual information) or wrapper-based (RFE, genetic algorithms)
- Imbalanced data handling: SMOTE oversampling, class weight adjustment, cost-sensitive learning
- Interpretability: SHAP values, feature importance, decision trees for explanations

**Datasets:**
- Credit card defaults (imbalanced, 95-98% non-default)
- Peer-to-peer lending platforms (China P2P)
- Firm financial statements (Italian companies)
- Macro indicators, customer demographics

**Key Results:**
- XGBoost: 99.4% accuracy, 0.943 AUC on credit card dataset
- Deep neural networks: 99.5% accuracy, 0.9547 AUC, 0.7064 F-score
- LightGBM: 87.1% accuracy, 0.943 AUC with faster training than XGBoost
- Gradient boosting outperforms logistic regression, SVM, naive Bayes
- Ensemble stacking: lower variance than individual base learners
- Feature importance: macro indicators (GDP, unemployment) often underutilized in simple models

**Limitations and Assumptions:**
- Class imbalance: even high accuracy may mask poor minority class recall
- Temporal aspect: default risk non-stationary; models trained on past crisis may not generalize
- Data quality: credit datasets often contain missing values, measurement error
- Model stability: boosting algorithms sensitive to outliers and data perturbations
- Regulatory acceptance: black-box models (NN, GB) less interpretable than traditional models
- Backtesting: assumes no selection bias in training data (e.g., banks reject risky applicants)
- Overfitting: high-dimensional feature spaces (100+) risk overfitting with limited positive examples

---

### I. Sentiment Analysis and Natural Language Processing

**Papers:**
- Stock trend prediction using sentiment analysis (PMC 2023)
- Sentiment Analysis for Effective Stock Market Prediction (2017/2019, peer-reviewed)
- Innovative Sentiment Analysis and Prediction of Stock Price Using FinBERT, GPT-4 and Logistic Regression (MDPI 2024)
- A sentiment analysis approach to the prediction of market volatility (Frontiers AI 2022)
- Stock Prediction using Natural Language Processing Sentiment Analysis on News Headlines During COVID-19 (AUC Egypt 2020)

**Problem Statement:** Investor sentiment from news and social media correlates with market movements; NLP extracts sentiment signals to enhance price/volatility predictions.

**Methodology:**
- Sentiment lexicons: dictionary-based (positive/negative/neutral word lists)
- Machine learning classifiers: Naive Bayes, SVM for binary/ternary sentiment classification
- Deep learning: LSTM, CNN for text representation and sentiment extraction
- FinBERT: pre-trained BERT fine-tuned on financial text; state-of-the-art NLP performance
- GPT-4: zero-shot sentiment extraction; few-shot learning for domain-specific nuances
- Multimodal approaches: combine sentiment with price/volume technical features
- Temporal aggregation: daily/weekly sentiment scores as predictive features

**Datasets:**
- Financial news (Reuters, Bloomberg, company filings)
- Twitter/social media posts about stocks
- Earnings call transcripts
- Reddit r/wallstreetbets
- News sentiment during COVID-19 (2020) and other crisis periods

**Key Results:**
- Sentiment-LSTM: higher accuracy than historical prices alone
- FinBERT sentiment + gradient boosting: competitive with other advanced models
- Correlation evidence: positive correlation between sentiment and subsequent returns/volatility
- Integration benefit: sentiment data improves historical model accuracy by 5-15%
- Social media vs. news: Twitter sentiment more responsive to short-term shocks; news sentiment more stable
- COVID period: sentiment sentiment captured regime shifts not captured by prices alone

**Limitations and Assumptions:**
- Data quality: social media sentiment noisy, subject to bot activity and manipulation
- Linguistic nuances: sarcasm, domain jargon poorly captured by standard NLP models
- Causality: unclear whether sentiment causes returns or vice versa
- Selection bias: only large, liquid stocks covered extensively by news/social media
- Time-varying relationships: sentiment-return correlation varies across market regimes
- Regulatory constraints: using non-public information (insider sentiment) prohibited
- Computational cost: real-time sentiment analysis for 5000+ stocks expensive

---

### J. Anomaly Detection in Financial Time Series

**Papers:**
- Deep Learning for Time Series Anomaly Detection: A Survey (ACM Computing Surveys 2024)
- Automated financial time series anomaly detection via curiosity-guided exploration and self-imitation learning (ScienceDirect 2024)
- Critical Analysis on Anomaly Detection in High-Frequency Financial Data Using Deep Learning for Options (Preprints 2025)
- A novel unsupervised framework for time series data anomaly detection via spectrum decomposition (ScienceDirect 2023)
- Deep unsupervised anomaly detection in high-frequency markets (ScienceDirect 2024)
- Anomaly Detection on Financial Time Series by Principal Component Analysis and Neural Networks (arXiv 2209.11686, 2022)

**Problem Statement:** Fraudulent trading, market manipulation, and system failures create anomalies in financial time series; unsupervised methods detect anomalies when labeled examples scarce.

**Methodology:**
- Isolation Forest: recursive partitioning to isolate anomalies (few and different points)
- Clustering-based: k-means or density clustering; outliers are points outside clusters
- Autoencoders: train on normal data; reconstruction error indicates anomaly
- LSTM-autoencoder: temporal autoencoder for sequential data
- Principal Component Analysis (PCA): detect deviations from principal components
- Spectral methods: eigenvalue decomposition of time-lagged covariance matrix
- Curiosity-driven learning: self-imitation to maximize information gain about anomalies
- Hybrid approaches: combine multiple unsupervised detectors via voting/ensemble

**Datasets:**
- Normal market data (months/years of tick data)
- Labeled anomalies: flash crashes, circuit breakers, system failures (limited)
- High-frequency trading data: order books, executions
- Option pricing anomalies: violations of put-call parity, arbitrage violations

**Key Results:**
- Isolation Forest: high F1-score (0.85-0.95) on synthetic anomaly datasets
- LSTM-autoencoder: better detection than static autoencoder for bursty anomalies
- PCA+NN: effective detection of subtle market structure changes
- Curiosity-driven: learns interpretable anomaly patterns without explicit labeling
- High-frequency data: detects microsecond-level execution anomalies with low false positive rate

**Limitations and Assumptions:**
- Concept drift: normal patterns evolve over time; fixed threshold-based detection fails
- Rare anomalies: evaluation difficult with few positive examples; synthetic anomalies unrealistic
- Interpretability: learned anomalies may not correspond to economically meaningful events
- Computational cost: streaming detection on high-frequency data requires efficient algorithms
- Robustness: adversarial traders may craft undetectable anomalies to evade systems
- Threshold selection: determining what constitutes "anomaly" is subjective
- False positives: legitimate market events (earnings surprises) may trigger false alarms

---

### K. Explainability and Interpretability

**Papers:**
- Model-agnostic explainable artificial intelligence methods in finance: a systematic review (Artificial Intelligence Review 2025)
- A comprehensive review on financial explainable AI (Artificial Intelligence Review 2024)
- A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME (arXiv 2305.02012, 2023)
- SHAP and LIME: An Evaluation of Discriminative Power in Credit Risk (Frontiers AI 2021)
- On the information content of explainable artificial intelligence for quantitative approaches in finance (OR Spectrum 2024)

**Problem Statement:** Machine learning models in finance are black boxes; regulatory and business requirements demand interpretability. SHAP/LIME provide post-hoc explanations.

**Methodology:**
- SHAP (SHapley Additive exPlanations): game-theoretic feature attribution; computes contribution of each feature to prediction
- LIME (Local Interpretable Model-agnostic Explanations): fits local linear model around instance to approximate complex model
- Feature Importance: mean decrease in impurity (tree-based) or permutation importance
- Attention weights: learned weights in attention mechanisms indicate which past values most relevant
- Saliency maps: visualize gradients w.r.t. inputs for neural networks
- Decision trees: interpretable surrogates to approximate complex models
- Causal inference: estimate causal effects of features on target, not just correlation

**Datasets:**
- Credit risk models: loan approval decisions
- Fraud detection: transaction flagging
- Trading models: price/return predictions
- Risk models: Value-at-Risk, systemic risk

**Key Results:**
- SHAP: global explanations (aggregate feature importance) and local explanations (individual predictions)
- SHAP advantage: game-theoretic foundation, considers feature interactions
- LIME: fast local explanations suitable for real-time systems
- Attention weights: interpretable; show which market periods most predictive
- SHAP for credit risk: fairness analysis; detects disparate impact on protected groups
- SHAP for trading: reveals overfit features; improves model generalization

**Limitations and Assumptions:**
- SHAP computation: exponential in feature count; approximations (TreeSHAP, KernelSHAP) still expensive
- LIME instability: small data perturbations can change local explanations substantially
- Assumption: features are independent; may not hold with collinearity
- Interpretation challenges: SHAP values don't imply causation; correlation may dominate
- Regulatory acceptance: explainability methods recognized but not legally mandated for most applications
- Temporal dynamics: explanations static but market regimes/feature relationships evolve

---

## Comparative Results Table: Prior Work vs. Methods vs. Performance

| Paper | Task | Method | Dataset | Key Result | Limitations |
|-------|------|--------|---------|------------|------------|
| LSTM Stock Prediction (2020-2024) | Price forecasting | LSTM/BiLSTM | S&P500, NASDAQ | RMSE ↓15-30% vs. ARIMA | Requires 3-5yr data; regime sensitive |
| PatchTST (2022-2023) | Multivariate forecasting | Transformer patches | CSI 300, electricity | Best out-of-sample R²; ↓complexity | High hyperparameter tuning required |
| LSTM-GARCH (2021-2024) | Volatility prediction | Hybrid LSTM+EGARCH | S&P500 futures | ↓15-30% RMSE vs. GARCH | Requires stationarity assumption |
| PSR-NODE (PACIS 2024) | Stock price | Neural ODE w/ phase space | Tech/Finance/Pharma stocks | ↑accuracy vs. LSTM/CNN/Transformer | Limited empirical studies; computational overhead |
| DQN Portfolio (2023) | Portfolio optimization | Deep Q-learning | S&P500 | ↑returns vs. benchmark | Sim-to-real gap; non-stationary environment |
| VolGAN (2024) | Volatility surface | Adversarial GAN | Options data | Arbitrage-free surfaces | Mode collapse; evaluation difficulty |
| Time-Causal VAE (2024) | Data generation | Variational autoencoder | Synthetic asset prices | Realistic stylized facts | Posterior collapse risk; lower forecast accuracy than supervised |
| GCN Asset Pricing (2023) | Return prediction | Graph convolution | Stock correlation network | Incorporates market structure | Arbitrary graph construction; scalability O(N²) |
| ResNet Option Pricing (2023) | Option valuation | Residual neural network | Options on equities | ↑accuracy vs. DNN/CNN | Extrapolation risk; arbitrage not enforced |
| XGBoost Credit (2024) | Default prediction | Gradient boosting | Credit card data | 99.4% accuracy, 0.943 AUC | Class imbalance; temporal non-stationarity |
| FinBERT Sentiment (2024) | Return prediction | BERT fine-tuned finance | Financial news + stock prices | ↑5-15% accuracy over prices alone | Noisy; sarcasm/domain jargon issues |
| Isolation Forest Anomaly (2024) | Anomaly detection | Unsupervised isolation | High-freq market data | ↑F1 (0.85-0.95) on synthetic | Concept drift; interpretability questions |
| Multi-task LSTM-RF (2021) | Multi-asset forecasting | LSTM + random forest | Correlated stocks | ↓RMSE 16-26% vs. single-task | Requires task relationship; complexity trade-off |

---

## Identified Gaps and Open Problems

### 1. **Theoretical Understanding**
- **Gap:** Limited convergence proofs for neural network approximators of stochastic processes
- **Gap:** Conditions under which transformer self-attention captures meaningful causal relationships vs. spurious correlations
- **Gap:** Generalization bounds for factor discovery via neural networks; when do learned factors generalize across regimes?

### 2. **Temporal Non-Stationarity and Regime Shifts**
- **Problem:** Training on historical data assumes dynamics remain stable; market regime switches invalidate models
- **Partial Solutions:** Transfer learning, meta-learning, online/continual learning, but limited empirical validation
- **Open Question:** How to detect regime shifts and automatically retrain/adapt models?

### 3. **Causality vs. Correlation**
- **Gap:** Most models predict without identifying causal relationships
- **Existing Work:** Causal inference methods (Granger causality, instrumental variables), but limited application in deep learning models
- **Need:** Explainable causal structures for trading and risk management decisions

### 4. **Evaluation and Backtesting Bias**
- **Problem:** Backtesting results often overestimate real-world performance due to:
  - Look-ahead bias, data snooping, survivorship bias, structural breaks
  - Ignore transaction costs, slippage, market impact
  - Overfitting to test period
- **Need:** Robust out-of-sample evaluation methodology; cross-validation for time series

### 5. **Computational Scalability**
- **Problem:** Real-time portfolio optimization with 5000+ assets intractable for current GNNs, attention mechanisms
- **Partial Solutions:** Sparse attention, linear attention approximations, distributed training
- **Need:** Scalable architectures for large-scale market problems

### 6. **Integration with Market Microstructure**
- **Gap:** Most models treat market as static price process; ignore order book dynamics, execution costs, liquidity
- **Partial Solutions:** GNNs for market structure, RL with realistic reward penalties
- **Need:** End-to-end models incorporating microstructure constraints

### 7. **Synthetic Data and Privacy**
- **Gap:** Privacy regulations limit access to proprietary trading data for research
- **Existing Work:** GANs, VAEs for synthetic data generation; differential privacy
- **Challenge:** Ensuring synthetic data preserves statistical properties without leaking private information

### 8. **Extreme Events and Fat Tails**
- **Problem:** Standard ML/NN training minimizes average error; underweights tail events (crashes, volatility spikes)
- **Partial Solutions:** Robust loss functions, risk-parity weighting, extreme value theory integration
- **Need:** Models that explicitly capture and predict tail dependencies

### 9. **Regulatory Compliance and Fairness**
- **Gap:** Explainability and fairness requirements for credit/lending models not fully addressed by NN/GB methods
- **Partial Solutions:** SHAP interpretability, fairness constraints in optimization
- **Need:** Regulatory-approved architectures balancing performance and interpretability

### 10. **Cross-Asset and Cross-Market Generalization**
- **Problem:** Models trained on one market/asset class often fail on others due to statistical properties
- **Existing Work:** Transfer learning, domain adaptation, multi-task learning
- **Challenge:** Identifying which features generalize across markets

---

## State-of-the-Art Summary

As of 2025, the quantitative finance field exhibits the following state-of-the-art characteristics:

### Best Performers by Task:

1. **Stock Price Forecasting:**
   - Transformer variants (PatchTST, iTransformer) with attention mechanisms consistently outperform LSTM on many benchmarks
   - Ensemble methods combining CNN, LSTM, and Transformer features achieve robust performance
   - Sentiment-augmented models (FinBERT + gradient boosting) effective for mid-term predictions

2. **Volatility Prediction:**
   - Hybrid GARCH-LSTM and BiLSTM-GARCH models achieve 15-30% improvement over pure GARCH
   - GARCH-informed neural networks combining domain knowledge with learning capacity show promise
   - Multivariate approaches capturing cross-asset volatility spillover via GNNs emerging

3. **Portfolio Optimization:**
   - Deep reinforcement learning (DQN, DDPG, PPO) agents consistently outperform benchmarks in backtests
   - Multi-agent RL for N-asset portfolios more effective than single-agent
   - Meta-learning enables rapid adaptation to regime shifts

4. **Option Pricing:**
   - Physics-informed neural networks (PINNs) and deep differential networks (DDN) achieve competitive accuracy
   - Calibration of stochastic volatility models (Heston) via neural networks promising
   - Arbitrage-free surface generation via constrained optimization improving

5. **Credit Risk / Default Prediction:**
   - XGBoost and LightGBM achieve >99% accuracy on large datasets
   - Deep neural networks competitive; interpretability via SHAP values improving regulatory acceptance
   - Ensemble stacking reduces variance; imbalanced data techniques essential

6. **Anomaly Detection:**
   - Isolation Forests effective for real-time detection with low computational cost
   - LSTM-autoencoders capture temporal dependencies in anomalies
   - Unsupervised methods viable when labeled data scarce

### Emerging Trends:

1. **Hybrid Architectures:** Combining domain-specific econometric models (GARCH, factor models) with flexible neural networks outperform either alone
2. **Interpretability Integration:** SHAP, LIME, and attention mechanisms increasingly integral to model design
3. **Generative Models:** GANs/VAEs for synthetic data, scenario generation, and calibration applications expanding
4. **Multi-Task and Transfer Learning:** Exploiting relationships between assets, markets, and tasks improving generalization
5. **Graph Neural Networks:** Capturing market structure and contagion effects a frontier area
6. **Neural Differential Equations:** Continuous-time modeling via NODEs showing promise for high-frequency applications

### Persistent Challenges:

1. Regime shifts and non-stationarity limit deployment horizon
2. Sim-to-real gap (backtesting vs. live trading) remains substantial
3. Causality inference lagging; most models capture correlations
4. Computational cost limits real-time applications at scale
5. Regulatory and fairness constraints sometimes conflict with predictive accuracy
6. Model robustness to adversarial perturbations and market manipulation underexplored

---

## References and Sources

### Foundational and Recent Surveys
1. [From Deep Learning to LLMs: A survey of AI in Quantitative Investment](https://arxiv.org/html/2503.21422v1) (arXiv 2503.21422, 2025)
2. [The Evolution of Reinforcement Learning in Quantitative Finance: A Survey](https://arxiv.org/abs/2408.10932) (arXiv 2408.10932, 2024)
3. [Data-driven stock forecasting models based on neural networks: A review](https://www.sciencedirect.com/science/article/pii/S1566253524003944) (ScienceDirect, 2024)
4. [Deep Learning for Time Series Anomaly Detection: A Survey](https://dl.acm.org/doi/10.1145/3691338) (ACM Computing Surveys, 2024)
5. [Factor Models, Machine Learning, and Asset Pricing](https://www.annualreviews.org/content/journals/10.1146/annurev-financial-101521-104735) (Annual Review of Financial Economics, 2022)

### Deep Learning Architectures
6. [Advanced Stock Market Prediction Using Long Short-Term Memory Networks: A Comprehensive Deep Learning Framework](https://arxiv.org/html/2505.05325v1) (arXiv, 2025)
7. [A novel transformer-based dual attention architecture for financial time series prediction](https://link.springer.com/article/10.1007/s44443-025-00045-y) (Journal of King Saud University, 2024)
8. [Deep context-attentive transformer transfer learning for financial forecasting](https://peerj.com/articles/cs-2983/) (PeerJ, 2024)
9. [Stock Price Prediction Using CNN-BiLSTM-Attention Model](https://www.mdpi.com/2227-7390/11/9/1985) (MDPI, 2024)

### Hybrid Econometric-Neural Models
10. [A Hybrid GARCH and Deep Learning Method for Volatility Prediction](https://onlinelibrary.wiley.com/doi/10.1155/2024/6305525) (Journal of Applied Mathematics, 2024)
11. [Volatility Forecasting using Hybrid GARCH Neural Network Models: The Case of the Italian Stock Market](https://ideas.repec.org/a/eco/journ1/2021-01-5.html) (2021)
12. [GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets](https://dl.acm.org/doi/fullHtml/10.1145/3677052.3698600) (ACM, 2024)
13. [The Sentiment Augmented GARCH-LSTM Hybrid Model for Value-at-Risk Forecasting](https://link.springer.com/article/10.1007/s10614-025-11042-8) (Computational Economics, 2025)

### Neural Ordinary Differential Equations
14. [Phase Space Reconstructed Neural Ordinary Differential Equations Model for Stock Price Forecasting](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4844513_code6073192.pdf) (PACIS 2024)
15. [Neural network stochastic differential equation models with applications to financial data forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0307904X22005340) (ScienceDirect, 2022)
16. [Neural Ordinary Differential Equation Networks for Fintech Applications Using Internet of Things](https://ieeexplore.ieee.org/document/10472330/) (IEEE, 2024)

### Factor Models and Machine Learning
17. [From Factor Models to Deep Learning: Machine Learning in Reshaping Empirical Asset Pricing](https://ideas.repec.org/p/arx/papers/2403.06779.html) (arXiv 2403.06779, 2024)
18. [Fundamental Factor Models Using Machine Learning](https://www.scirp.org/journal/paperinformation?paperid=82430) (SCIRP, 2018)

### Reinforcement Learning
19. [A novel multi-agent dynamic portfolio optimization learning system based on hierarchical deep reinforcement learning](https://link.springer.com/article/10.1007/s40747-025-01884-y) (Complex & Intelligent Systems, 2025)
20. [Deep Reinforcement Learning for Portfolio Optimization using Latent Feature State Space (LFSS) Module](https://arxiv.org/abs/2102.06233) (arXiv 2102.06233)
21. [Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization: A Multi-reward Approach](https://link.springer.com/article/10.1007/s44196-025-00875-8) (IJCIS, 2025)

### Generative Models: GANs
22. [Factor-GAN: Enhancing stock price prediction and factor investment with Generative Adversarial Networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC11198854/) (PMC, 2024)
23. [Computing Volatility Surfaces using Generative Adversarial Networks with Minimal Arbitrage Violations](https://arxiv.org/abs/2304.13128) (arXiv 2304.13128, 2023)
24. [Towards Realistic Market Simulations: a Generative Adversarial Networks Approach](https://ar5iv.labs.arxiv.org/html/2110.13287) (arXiv 2110.13287, 2021)
25. [VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces](https://www.tandfonline.com/doi/full/10.1080/1350486X.2025.2471317) (Quantitative Finance, 2024)

### Generative Models: VAEs
26. [Time-Causal VAE: Robust Financial Time Series Generator](https://arxiv.org/abs/2411.02947) (arXiv 2411.02947, 2024)
27. [Hybrid variational autoencoder for time series forecasting](https://www.sciencedirect.com/science/article/pii/S0950705123008298) (2023)
28. [Variational Autoencoders for Completing the Volatility Surfaces](https://www.mdpi.com/1911-8074/18/5/239) (MDPI, 2024)

### Graph Neural Networks
29. [A Review on Graph Neural Network Methods in Financial Applications](https://arxiv.org/abs/2111.15367) (arXiv 2111.15367, 2021)
30. [Attention based dynamic graph neural network for asset pricing](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614642/) (PMC, 2023)
31. [Dynamic graph neural networks for enhanced volatility prediction in financial markets](https://arxiv.org/html/2410.16858v1) (arXiv 2410.16858, 2024)

### Option Pricing
32. [Neural Network Learning of Black-Scholes Equation for Option Pricing](https://arxiv.org/abs/2405.05780) (arXiv 2405.05780, 2024)
33. [Calibrating the Heston Model with Deep Differential Networks](https://arxiv.org/html/2407.15536v1) (arXiv 2407.15536, 2024)
34. [Option Pricing Based on the Residual Neural Network](https://link.springer.com/article/10.1007/s10614-023-10413-3) (Computational Economics, 2023)

### Credit Risk and Default Prediction
35. [Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers](https://www.mdpi.com/2227-9091/12/11/174) (MDPI, 2024)
36. [Measuring the model risk-adjusted performance of machine learning algorithms in credit default prediction](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-022-00366-1) (Financial Innovation, 2022)

### Sentiment Analysis and NLP
37. [Stock trend prediction using sentiment analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10403218/) (PMC, 2023)
38. [Innovative Sentiment Analysis and Prediction of Stock Price Using FinBERT, GPT-4 and Logistic Regression](https://www.mdpi.com/2504-2289/8/11/143) (MDPI, 2024)
39. [A sentiment analysis approach to the prediction of market volatility](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.836809/full) (Frontiers AI, 2022)

### Anomaly Detection
40. [Automated financial time series anomaly detection via curiosity-guided exploration and self-imitation learning](https://www.sciencedirect.com/science/article/abs/pii/S0952197624008212) (ScienceDirect, 2024)
41. [A novel unsupervised framework for time series data anomaly detection via spectrum decomposition](https://www.sciencedirect.com/science/article/pii/S0950705123007529) (ScienceDirect, 2023)
42. [Anomaly Detection on Financial Time Series by Principal Component Analysis and Neural Networks](https://arxiv.org/abs/2209.11686) (arXiv 2209.11686, 2022)

### Ensemble Methods and Boosting
43. [Chapter 4: Ensemble Learning in Investment: An Overview](https://rpc.cfainstitute.org/research/foundation/2025/chapter-4-ensemble-learning-investment) (CFA Institute, 2025)
44. [A Comparative study of ensemble learning algorithms for high-frequency trading](https://www.sciencedirect.com/science/article/pii/S2468227624001066) (ScienceDirect, 2024)
45. [A Novel Hybrid Ensemble Framework for Stock Price Prediction](https://link.springer.com/article/10.1007/s10614-025-10979-0) (Computational Economics, 2025)

### Explainability and Interpretability
46. [Model-agnostic explainable artificial intelligence methods in finance: a systematic review](https://link.springer.com/article/10.1007/s10462-025-11215-9) (Artificial Intelligence Review, 2025)
47. [A comprehensive review on financial explainable AI](https://link.springer.com/article/10.1007/s10462-024-11077-7) (Artificial Intelligence Review, 2024)
48. [SHAP and LIME: An Evaluation of Discriminative Power in Credit Risk](https://www.frontiersin.org/articles/10.3389/frai.2021.752558/full) (Frontiers AI, 2021)

### Additional Resources
49. [Machine Learning and Data Sciences for Financial Markets](https://www.cambridge.org/core/books/machine-learning-and-data-sciences-for-financial-markets/8BB31611662A96D0AB93A8A26E2D0D0A) (Cambridge University Press)
50. [Advanced Machine Learning in Quantitative Finance Using](https://www.jait.us/articles/2024/JAIT-V15N9-1025.pdf) (JAIT, 2024)

---

## Notes on Data and Reproducibility

- Most papers use public datasets (S&P 500, NASDAQ, CSI 300) available via Yahoo Finance, Alpha Vantage, or Quandl
- Some proprietary datasets (high-frequency trading data, options surfaces) limit reproducibility; synthetic data generation (GANs, VAEs) emerging as alternative
- Code repositories on GitHub increasingly common for recent papers (2023+)
- Hyperparameter sensitivity high; reported results often depend on careful tuning not fully documented
- Backtest periods vary significantly; 2008-2020 common for crisis robustness testing; 2020-2024 for recent market regimes

---

## Concluding Remarks

The quantitative finance literature from 2018-2025 reflects a clear progression from traditional econometric models to hybrid systems integrating domain expertise with deep learning flexibility. No single approach dominates all tasks; success depends on problem structure, data regime, and computational constraints. Future research should focus on: (1) theoretical foundations connecting neural network approximation to financial dynamics; (2) robust evaluation methodology addressing look-ahead bias and structural breaks; (3) causal inference frameworks distinguishing correlation from causation; (4) scalable architectures for large-scale markets; and (5) integration with market microstructure and regulatory constraints. The field stands at an inflection point where ensemble, interpretable hybrid models balancing accuracy and explainability may prove more valuable than black-box deep learning alone.

