# Literature Review: Deep Learning for Financial Time-Series Forecasting (2021-2025)

## Executive Summary

This literature review surveys recent advances in deep learning methodologies for financial time-series forecasting, with emphasis on Long Short-Term Memory (LSTM) networks, Transformer architectures, Graph Neural Networks (GNNs), and hybrid models that integrate classical econometric techniques with neural networks. The survey covers peer-reviewed research, technical reports, and conference proceedings from 2021-2025, focusing on quantitative benchmarks and comparative performance metrics.

---

## 1. Overview of the Research Area

Financial time-series forecasting remains a critical challenge in quantitative finance, driven by the need to predict asset prices, volatility, returns, and market indices. Traditional statistical approaches—such as ARIMA, GARCH, and VAR models—have demonstrated limitations in capturing nonlinear relationships and complex temporal dependencies inherent in financial markets.

Deep learning approaches have emerged as powerful alternatives, offering:
- **Ability to capture nonlinear patterns** in market data
- **Automatic feature extraction** without manual engineering
- **Flexible architectures** for multi-scale temporal modeling
- **Integration with external data sources** (sentiment, news, macroeconomic factors)

However, the literature reveals competing methodologies, varying performance across datasets, and ongoing challenges related to:
- Overfitting to historical patterns that do not generalize to future market regimes
- Computational complexity and training time
- Interpretability and explainability
- Robustness across different market conditions (bull markets, crises, high volatility)

---

## 2. Major Architectural Approaches and Key Findings

### 2.1 Recurrent Neural Networks (RNNs) and LSTM Models

**Core Limitations of Plain RNNs:**

RNNs suffer from fundamental architectural constraints when applied to financial forecasting:
- **Vanishing/Exploding Gradients:** Gradients diminish exponentially over long sequences, causing networks to forget earlier information (Althelaya et al., 2018; cited in multiple 2023-2024 reviews)
- **Limited Long-Term Memory:** Simple RNNs struggle to learn dependencies beyond 5-10 timesteps
- **Computational Intensity:** Training becomes resource-intensive as sequence length increases

**LSTM Innovations:**

LSTM networks address RNN limitations through gating mechanisms:
- Input, forget, and output gates regulate information flow
- Effective at modeling long-term dependencies in financial data
- Significantly improved gradient propagation during backpropagation

**Recent LSTM-Based Work (2023-2025):**

1. **LSTM-Transformer Hybrid Models**
   - Citation: "LSTM–Transformer-Based Robust Hybrid Deep Learning Model for Financial Time Series Forecasting" (MDPI 2025, URL: https://www.mdpi.com/2413-4155/7/1/7)
   - Approach: Combines LSTM for temporal sequence modeling with Transformer encoder for capturing global dependencies
   - Results: Tested on normalized data (Jan 2020–Jun 2024), covering COVID-19 pandemic and Russia-Ukraine war
   - Advantage: Maintains robust performance across multiple market regimes

2. **Deep Learning Ensemble Model (LSTM-mTrans-MLP)**
   - Citation: "Financial Time Series Forecasting with the Deep Learning Ensemble Model" (MDPI Mathematics 2024, URL: https://www.mdpi.com/2227-7390/11/4/1054)
   - Components: LSTM networks, modified Transformer networks, multilayered perceptron
   - Methodology: Ensemble approach combines strengths of multiple architectures
   - Key insight: Ensemble models capture diverse market patterns and demonstrate robustness to different conditions

3. **ARMA-CNN-LSTM Hybrid for Nonlinear Feature Capture**
   - Methodology: ARMA models linear features; CNN-LSTM captures nonlinear spatiotemporal patterns
   - Dataset: Financial time series with mixed linear/nonlinear characteristics
   - Performance gain: Significant improvement over single-architecture baselines

4. **LSTM with Sentiment Analysis Integration**
   - Recent work (2023-2024) integrates LSTM with textual sentiment extraction from financial news
   - Finding: Hybrid models with sentiment outperform baseline LSTM in:
     - Predictive accuracy
     - Ability to forecast turning points and anomalous behavior
     - Interpretability through external signal sensitivity

**LSTM vs. GRU Comparative Analysis:**
- **GRU Performance:** One study reported GRU with MAPE=3.97% and RMSE=381.34, outperforming LSTM on specific datasets
- **Computational Trade-off:** GRU reduces parameters vs. LSTM while achieving comparable accuracy
- **Dataset Dependency:** Performance varies; GRU may reduce overfitting on certain data structures
- Citation: "A Comparison between ARIMA, LSTM, and GRU for Time Series Forecasting" (ResearchGate, 2024, URL: https://www.researchgate.net/publication/339093433)

### 2.2 Transformer Architecture and Attention Mechanisms

**Transformer Breakthrough in Financial Forecasting:**

Transformer models, which rely on self-attention mechanisms rather than recurrence, have demonstrated superior performance in recent benchmarks (2023-2024).

**Key Advantages:**
1. **Parallel Processing:** Unlike RNNs, Transformers process entire sequences in parallel, reducing training time
2. **Long-Range Dependencies:** Attention weights allow modeling of long-term relationships without gradient decay
3. **Multi-Head Attention:** Multiple representation subspaces capture different temporal patterns simultaneously
4. **Interpretability:** Attention weights provide insights into which timesteps influence predictions

**Benchmark Results:**

1. **Stock Price Prediction Using Transformer**
   - Citation: "Predictive Modeling of Stock Prices Using Transformer Model" (ACM 2024, URL: https://dl.acm.org/doi/fullHtml/10.1145/3674029.3674037)
   - Methodology: Pure Transformer encoder-decoder architecture
   - Performance: Benchmark comparison across 12 neural network architectures on 10 market indices
   - Finding: Transformer variants produce significantly better results than all baseline models
   - Metric Used: Not explicitly stated in accessible portion

2. **Deep Convolutional Transformer (DCT) Network**
   - Citation: "Deep Convolutional Transformer Network for Stock Movement Prediction" (Electronics 2024, URL: https://www.mdpi.com/2079-9302/13/21/4225)
   - Architecture: Combines CNNs (spatial feature extraction), Transformers (temporal patterns), multi-head attention
   - Publication Date: October 2024
   - Key Innovation: Multi-head attention mechanism emphasizes specific features and identifies important relationships

3. **Frequency Decomposition with GRU-Transformer**
   - Citation: "Stock Price Prediction Using a Frequency Decomposition Based GRU Transformer Neural Network" (Applied Sciences 2023, URL: https://www.mdpi.com/2076-3417/13/1/222)
   - Problem Addressed: RNN limitations in capturing multiple frequency components
   - Approach: Frequency decomposition preprocessing + GRU-Transformer encoder
   - Benefit: Tackles RNN instability in long-range dependencies

4. **Transformer Models Across Multiple Market Indices**
   - Citation: "Stock market index prediction using transformer neural network models and frequency decomposition" (Neural Computing and Applications 2024, URL: https://link.springer.com/article/10.1007/s00521-024-09931-4)
   - Comparison: 12 neural network architectures evaluated across 10 market indices
   - Result: Transformer consistently outperforms benchmarks across all indices
   - Robustness: Superior performance across diverse market conditions

5. **Recent Comparative Analysis (November 2025)**
   - Citation: "Transformer AI models outperform neural networks in stock market prediction, study shows" (Phys.org 2025, URL: https://phys.org/news/2025-11-ai-outperform-neural-networks-stock.html)
   - Key Finding: Transformers surpass traditional RNNs in predicting stock market returns
   - Reason: Advanced architecture enables:
     - Better encoding of fundamental information
     - Detection of long-term market patterns
     - Integration of diverse macroeconomic variables

### 2.3 Graph Neural Networks (GNNs) for Stock Market Relationships

**Motivation for Graph-Based Approaches:**

Traditional time-series models treat individual stocks independently, missing critical inter-stock relationships driven by:
- Sector correlations
- Supply chain dependencies
- Market microstructure
- Contagion effects

GNNs model these relationships as graph structures where nodes represent stocks/indices and edges represent correlations.

**Recent GNN Applications (2022-2024):**

1. **Systematic Review of GNN Methods**
   - Citation: "A Systematic Review on Graph Neural Network-based Methods for Stock Market Forecasting" (ACM Computing Surveys 2024, URL: https://dl.acm.org/doi/10.1145/3696411)
   - Scope: Comprehensive survey of GNN architectures for financial forecasting
   - Key Insight: GNNs essential for capturing inter-stock dependencies overlooked by traditional methods

2. **LSTM-GNN Hybrid Model**
   - Citation: "STOCK PRICE PREDICTION USING A HYBRID LSTM-GNN" (ArXiv 2025, URL: https://arxiv.org/pdf/2502.15813)
   - Architecture:
     - LSTM processes individual stock temporal sequences
     - GNN models stock correlation graph
     - Outputs combined via ensemble mechanism
   - Innovation: Jointly captures temporal dynamics AND relational patterns
   - Expected Performance: Outperforms single-stream approaches

3. **ChatGPT-Informed GNN**
   - Citation: "CHATGPT INFORMED GRAPH NEURAL NETWORK FOR STOCK MOVEMENT PREDICTION" (ArXiv 2023, URL: https://arxiv.org/pdf/2306.03763)
   - Approach: Integrates LLM-derived semantic features with GNN structure
   - Dataset: Multiple stock datasets
   - Novelty: Leverages language models to enrich graph node features

4. **Graph CNN-LSTM Integration**
   - Methodology: Relational data via GNN + temporal patterns via CNN-LSTM
   - Performance: Hybrid approach achieves more accurate predictions by capturing:
     - Temporal dynamics
     - Inter-stock interconnections
   - Quantitative Result: **4-15% improvement in F-measure over baseline algorithms**

5. **GraphCNNpred System**
   - Citation: "GraphCNNpred: A stock market indices prediction using a Graph based deep learning system" (ACM 2024, URL: https://dl.acm.org/doi/10.1145/3714334.3714364)
   - Performance Metric: Trading simulations achieved **Sharpe ratios above 3.0**
   - Benchmark: Outperforms traditional indices prediction baselines

6. **Multi-Source Heterogeneous Data Fusion**
   - Citation: "A graph neural network-based stock forecasting method utilizing multi-source heterogeneous data fusion" (Multimedia Tools and Applications 2022, URL: https://link.springer.com/article/10.1007/s11042-022-13231-1)
   - Data Sources: Multiple heterogeneous inputs (price, volume, news, sentiment)
   - Finding: GNN effectively fuses diverse data sources

### 2.4 Bidirectional LSTM (BiLSTM) Models

**Architectural Advantage:**

BiLSTM processes sequences in both forward and backward directions, enabling:
- Capture of future context during past predictions
- Better identification of complex temporal patterns
- Improved handling of bidirectional dependencies

**Recent Findings (2023-2025):**

1. **DeepInvesting: BiLSTM for Amazon Stock Prediction**
   - Citation: "DeepInvesting: Stock market predictions with a sequence-oriented BiLSTM stacked model – A dataset case study of AMZN" (2024, URL: https://www.sciencedirect.com/science/article/pii/S2667305324001133)
   - Architecture: Sequence-Oriented, Long-Term Dependent (SoLTD) BiLSTM
   - Dataset: Amazon Corp. (AMZN) market data
   - Performance:
     - Minimal error metrics (specific values not disclosed in abstract)
     - High R² values exceeding traditional methods
     - Outperformed: KNN, LSTM, RNN, CNN, ANN
   - Key Finding: BiLSTM captures complex temporal dependencies more effectively

2. **Bidirectional Processing for Volatility Prediction**
   - Citation: "Evaluation of bidirectional LSTM for short-and long-term stock market prediction" (2024, URL: https://www.researchgate.net/publication/324996793)
   - Finding: BiLSTM exhibits better out-of-sample forecasting than standard LSTM and ARIMA
   - Mechanism: Bidirectional learning captures patterns more accurately
   - Time Horizon: Superior across both short-term (1-5 days) and long-term (20+ days) predictions

3. **BiLSTM-SAM-TCN Combined Architecture (2024)**
   - Citation: "Predicting the highest and lowest stock price indices: A combined BiLSTM-SAM-TCN deep learning model based on re-decomposition" (2024, URL: https://www.sciencedirect.com/science/article/abs/pii/S1568494624011670)
   - Components:
     - BiLSTM: Temporal long-range dependencies
     - Self-Attention Mechanism (SAM): Feature importance weighting
     - Temporal Convolutional Network (TCN): Multi-scale pattern detection
   - Performance: Prediction accuracy outperforms existing models in both developed and developing markets
   - Novelty: Independent prediction of high and low prices

4. **Time Lag Analysis**
   - Study: BiLSTM with varying time lags (5, 10, 22 days)
   - Finding: 22-day lag optimal for full-period returns/volatility prediction
   - Finding: 5-10 day lags better for unstable periods (crises, pandemics)
   - Implication: Adaptive lag selection improves performance across market regimes

### 2.5 Convolutional Neural Networks (CNNs) for Financial Data

**CNN Applications in Finance:**

CNNs, traditionally used for image processing, have been adapted for time-series by treating sequences as 1D spatial data.

**Recent Work (2023-2024):**

1. **CNN-BiLSTM-Attention Hybrid**
   - Citation: "Stock Price Prediction Using CNN-BiLSTM-Attention Model" (Mathematics 2024, URL: https://www.mdpi.com/2227-7390/11/9/1985)
   - Architecture:
     - CNN: Temporal feature extraction
     - BiLSTM: Dynamic change pattern learning
     - Attention: Feature importance weighting
   - Advantage: Combines local (CNN) and global (attention) pattern recognition

2. **CNN Multivariate Forecasting**
   - Finding: CNN-based multivariate model most effective in predicting NIFTY index movements with weekly forecasting horizon
   - Dataset: Indian National Stock Exchange

3. **CNN on Stock Chart Images**
   - Citation: "cnn-based stock price forecasting - by stock chart images" (Romanian Journal of Economic Forecasting 2023, URL: https://ipe.ro/new/rjef/rjef3_2023/rjef3_2023p120-128.pdf)
   - Approach: Training CNNs directly on stock chart visual representations
   - Portfolio Construction: Long-short portfolios based on price-rise probability
   - Critical Note: Analysis shows potential false positive rates if temporal context is overlooked

4. **Graph-Based CNN-LSTM with Leading Indicators**
   - Citation: "A graph-based CNN-LSTM stock price prediction algorithm with leading indicators" (Multimedia Systems 2021, URL: https://link.springer.com/article/10.1007/s00530-021-00758-w)
   - Integration: Technical indicators as additional features
   - Synergy: CNN captures local patterns; LSTM models sequential dynamics

### 2.6 Variational Autoencoders (VAE) and Generative Models

**VAE Applications in Financial Forecasting:**

VAE models learn latent representations of financial data, useful for:
- Feature dimensionality reduction
- Capturing underlying market microstructure
- Stochastic modeling of price dynamics

**Recent Developments (2022-2024):**

1. **Diffusion Variational Autoencoder (D-VAE)**
   - Citation: "Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction" (ArXiv 2023, URL: https://arxiv.org/abs/2309.00073)
   - Problem: Stock prices exhibit stochasticity; deterministic models insufficient
   - Solution: Hierarchical VAE + diffusion probabilistic model
   - Components:
     - Hierarchical VAE: Learns complex low-level latent variables
     - Diffusion Model: Handles stochastic dynamics
   - Performance: Outperforms state-of-the-art in prediction accuracy AND variance estimation
   - Citation Also: "Implementation of Diffusion Variational Autoencoder for Stock Price Prediction with the Integration of Historical and Market Sentiment Data" (2024, URL: https://ijecbe.ui.ac.id/go/article/view/55)

2. **D-VAE with Sentiment Integration**
   - Dataset: Historical stock prices + trading volume + financial news sentiment
   - Results:
     - Technical data alone: R² metric reported
     - Sentiment-integrated model: **R² = 0.94489**
     - Conclusion: Sentiment integration substantially improves accuracy

3. **VAE for Feature Reduction**
   - Citation: "An efficient stock market prediction model using hybrid feature reduction method based on variational autoencoders and recursive feature elimination" (Financial Innovation 2021, URL: https://jfin-swufe.springeropen.com/articles/10.1186/s40854-021-00243-3)
   - Finding: VAE-based dimensionality reduction achieves similar prediction results with **16.67% fewer features**
   - Implication: More efficient and interpretable models

4. **Ensemble: VAE + Transformer + LSTM**
   - Citation: "An Advanced Ensemble Deep Learning Framework for Stock Price Prediction Using VAE, Transformer, and LSTM Model" (2025, URL: https://arxiv.org/html/2503.22192v1)
   - Approach: Three-architecture ensemble for complementary strengths
   - Problem Addressed: Market volatility, non-linearity, temporal dependencies
   - Status: Recent preprint (2025)

### 2.7 Attention Mechanisms and Multi-Head Attention

**Attention Mechanisms Beyond Transformers:**

Attention can be integrated into RNN and CNN architectures, improving interpretability and accuracy.

**Recent Applications (2022-2024):**

1. **Stock Price Prediction with Attention (2023)**
   - Citation: "Stock Price Prediction using Deep Neural Network based on Attention Mechanism" (ACM 2023, URL: https://dl.acm.org/doi/10.1145/3659154.3659157)
   - Mechanism: Attention weights highlight important timesteps
   - Result: Outperforms non-attention baselines in capturing temporal dependencies

2. **GAN + Transformer with Attention**
   - Citation: "Enhancing stock price prediction using GANs and transformer-based attention mechanisms" (Empirical Economics 2024, URL: https://link.springer.com/article/10.1007/s00181-024-02644-6)
   - Architecture: Generative Adversarial Network + Transformer + multi-head attention
   - Innovation: GAN generates synthetic training data; Transformer with attention predicts

3. **Multi-Head Attention Integration**
   - Citation: "Deep Learning-Based Hybrid Model with Multi-Head Attention for Multi-Horizon Stock Price Prediction" (MDPI 2024, URL: https://www.mdpi.com/1911-8074/18/10/551)
   - Capability: Multi-head attention processes different representation subspaces
   - Use Case: Multi-horizon predictions (1-day, 7-day, 30-day ahead)
   - Performance: Superior cumulative returns in backtests

4. **GRU-CNN-Attention Combination**
   - Components: GRU (sequence modeling), CNN (feature extraction), attention (weighting)
   - Application: Short-term trend prediction
   - Advantage: Dynamic attention weights assign importance to input sequences

---

## 3. Hybrid Models: Classical Econometrics + Neural Networks

### 3.1 ARIMA-GARCH-Based Hybrid Architectures

**Motivation:**

Classical econometric models (ARIMA, GARCH) excel at capturing linear dynamics and volatility clustering, while neural networks capture nonlinearity. Hybrid approaches leverage both.

**Recent Research (2023-2024):**

1. **Hybrid SARIMA-GARCH-CNN-BiLSTM**
   - Citation: "A Hybrid Garch and Deep Learning Method for Volatility Prediction" (Journal of Applied Mathematics 2024, URL: https://onlinelibrary.wiley.com/doi/10.1155/2024/6305525)
   - Architecture:
     - SARIMA: Captures seasonal linear components
     - GARCH: Models heteroscedasticity and volatility clustering
     - CNN-BiLSTM: Learns nonlinear spatiotemporal patterns
   - Finding: Hybrid approach effectively resolves shortcomings of volatility forecasting
   - Benefit: Combines econometric interpretability with deep learning accuracy

2. **ARIMA-ANN Serial Structure**
   - Citation: "A Hybrid Forecasting Structure Based on Arima and Artificial Neural Network Models" (Applied Sciences 2024, URL: https://www.mdpi.com/2076-3417/14/16/7122)
   - Methodology:
     - Step 1: ARIMA models original data, captures linear patterns
     - Step 2: ANN models ARIMA residuals, learns remaining nonlinearity
     - Combination: Linear predictions + nonlinear residual predictions
   - Advantage: Modular and interpretable approach

3. **Wavelet Decomposition + ARIMA-GARCH**
   - Citation: "Forecasting volatility by using wavelet transform, ARIMA and GARCH models" (Eurasian Economic Review 2023, URL: https://link.springer.com/article/10.1007/s40822-023-00243-x)
   - Process:
     - MODWT (Maximal Overlap Discrete Wavelet Transform) decomposes series into frequency components
     - ARIMA-GARCH applied to each component
     - Final forecast: Sum of component forecasts
   - Benefit: Handles multiple timescales simultaneously

4. **ARMA-GARCH-Quantum Recurrent Neural Network**
   - Citation: "Hybrid ARMA-GARCH-Neural Networks for intraday strategy exploration in high-frequency trading" (2023, URL: https://www.sciencedirect.com/science/article/pii/S0031320323008361)
   - Components:
     - ARMA: Linear structure
     - GARCH: Volatility
     - Quantum RNN: Advanced nonlinear modeling
   - Result: **Best accuracy achieved** in comparative analysis
   - Application: High-frequency trading strategies

5. **Comparative Performance (2024)**
   - Citation: "A Comparative Analysis of ARIMA-GARCH, LSTM, and ..." (SHS Conferences 2024, URL: https://www.shs-conferences.org/articles/shsconf/pdf/2024/16/shsconf_edma2024_02008.pdf)
   - Finding: Deep learning (LSTM) outperforms ARIMA-GARCH
   - Caveat: Depends on data characteristics and market conditions

### 3.2 Traffic/Physics-Inspired Hybrids

**Extending Hybrid Models to Financial Microstructure:**

1. **GARCH-GRU for Traffic Speed Prediction (2023)**
   - Citation: "Traffic speed prediction using GARCH‐GRU hybrid model" (IET Intelligent Transport Systems 2023, URL: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12411)
   - Note: While originally for traffic, methodology directly applicable to financial microstructure (market impact, order flow)
   - Hybrid Logic: GARCH volatility + GRU temporal dynamics

---

## 4. Ensemble and Multi-Architecture Approaches

### 4.1 Ensemble Methods in Financial Forecasting

**Ensemble Benefits:**

Combining multiple models through parallel architectures leverages complementary strengths:
- Capture diverse market patterns
- Robust to individual model failures
- Average out idiosyncratic errors

**Recent Ensemble Studies (2023-2024):**

1. **Comprehensive Ensemble Review (2024)**
   - Citation: "Deep learning for financial forecasting: A review of recent trends" (ScienceDirect 2025, URL: https://www.sciencedirect.com/science/article/pii/S1059056025008822)
   - Scope: 187 Scopus-indexed studies (2020-2024) on deep learning for financial forecasting
   - Categorization: By forecasting task (stock, index, forex, commodity, bond, crypto, volatility)
   - Key Finding: **Ensemble models more reliable than individual architectures**
   - Top Performers: LSTM, BiLSTM, GRU with self-attention

2. **Volatility Ensemble Performance**
   - Citation: "Deep Learning for Financial Time Series Prediction: A State-of-the-Art Review of Standalone and Hybrid Models" (CMES 2024, URL: https://www.techscience.com/CMES/v139n1/55114/html)
   - Finding: ANNs with memory (LSTM, GRU) rank among top performers
   - Advanced Ensemble: LSTM-CNN + LSTM-Self-Attention + others
   - Volatility Period Performance: Deep learning excels during high-volatility periods

3. **Financial Forecasting Ensemble (IJIEEB 2024)**
   - Citation: "Financial Forecasting with Deep Learning Models Based Ensemble Technique in Stock Market Analysis" (MECS Press, URL: https://www.mecs-press.org/ijieeb/ijieeb-v17-n4/v17n4-1.html)
   - Ensemble Strategy: Parallel deep learning models with voting/averaging
   - Application: Stock market analysis and prediction

4. **Multi-Layer Hybrid Ensemble (2025)**
   - Citation: "Boosting the Accuracy of Stock Market Prediction via Multi-Layer Hybrid MTL" (ArXiv 2025, URL: https://arxiv.org/pdf/2501.09760)
   - Approach: Multi-task learning with multiple prediction layers
   - Innovation: Leverages task relationships for improved accuracy

---

## 5. Quantitative Benchmark Results and Performance Metrics

### 5.1 Common Evaluation Metrics

**Standard Metrics in Literature:**

1. **Mean Absolute Error (MAE)**
   - Definition: Average absolute deviation between predicted and actual values
   - Interpretation: Same unit as original data; sensitive to outliers
   - Formula: MAE = (1/n) Σ |y_i - ŷ_i|

2. **Root Mean Squared Error (RMSE)**
   - Definition: Square root of mean squared error
   - Advantage: Preferred over MSE for interpretability; same unit as target
   - Formula: RMSE = √[(1/n) Σ (y_i - ŷ_i)²]
   - Use Case: Heavily penalizes large errors

3. **Mean Absolute Percentage Error (MAPE)**
   - Definition: Average percentage deviation
   - Advantage: Scale-independent, suitable for comparing models across datasets
   - Limitation: Asymmetric (favors underestimation)
   - Formula: MAPE = (1/n) Σ |y_i - ŷ_i| / |y_i|

4. **R-Squared (R²)**
   - Definition: Coefficient of determination; proportion of variance explained
   - Range: 0-1 (higher is better)
   - Interpretation: 0.9+ indicates excellent fit; 0.5-0.7 moderate fit

5. **Additional Metrics**
   - **Directional Accuracy (DA):** Percentage of correct price movement direction predictions
   - **Sharpe Ratio:** Risk-adjusted return metric used in trading simulations
   - **Profit Factor / Cumulative Return:** Economic viability metrics

### 5.2 Benchmark Results by Architecture (2023-2025)

**LSTM-Based Models:**
- **LSTM vs. GRU Comparison:** GRU: MAPE=3.97%, RMSE=381.34 (outperforms LSTM on specific dataset)
- **LSTM-Transformer Hybrid:** Robust across market conditions (Jan 2020–Jun 2024 tested; includes COVID-19, Russia-Ukraine war)
- Status: Literature indicates strong but dataset-dependent performance

**Transformer Models:**
- **Multi-Index Comparison (2024):** Transformer variants significantly outperform 12 baseline architectures across 10 market indices
- **Consistent Winner:** Transformer maintains lowest MAE across all market conditions
- Performance Ranking: Transformer > CNN-based models > traditional RNNs

**BiLSTM-Based Models:**
- **AMZN Stock Prediction:** Outperforms KNN, LSTM, RNN, CNN, ANN (specific metrics not disclosed)
- **Out-of-Sample Forecasting:** BiLSTM > LSTM > ARIMA
- **Volatility Prediction:** 22-day lag optimal for full period; 5-10 day lag for crisis periods

**Graph Neural Networks:**
- **CNN-LSTM-GNN Hybrid:** 4-15% improvement in F-measure over baselines
- **GraphCNNpred Trading Simulation:** Sharpe ratio > 3.0
- **Performance Driver:** Captures inter-stock relationships missed by univariate models

**CNN-Based Models:**
- **Multivariate CNN:** Most effective for weekly NIFTY index forecasting
- **CNN-BiLSTM-Attention:** Superior cumulative returns vs. standalone architectures

**VAE Models:**
- **Diffusion-VAE:** Outperforms state-of-the-art in multi-step prediction accuracy and variance estimation
- **D-VAE with Sentiment:** R² = 0.94489 (sentiment-enhanced vs. technical-only models)
- **VAE Feature Reduction:** 16.67% fewer features with maintained accuracy

**Ensemble Methods:**
- **Multi-Architecture Ensembles:** More reliable than individual models
- **High-Volatility Performance:** Deep learning ensembles outperform single models during market stress

---

## 6. Identified Gaps and Open Challenges

### 6.1 Methodological Gaps

1. **Temporal Context and False Positives**
   - Issue: Many studies achieve high accuracy metrics but fail in practical trading due to neglecting temporal context
   - Example: Chart-based CNN approaches show high accuracy but may capture overfitted visual patterns
   - Gap: Limited research on practical viability vs. statistical accuracy

2. **Model Generalization Across Market Regimes**
   - Challenge: Models trained on bull markets often fail during crises
   - Limited Study: Few papers explicitly evaluate robustness across regimes (exception: LSTM-Transformer 2020-2024 analysis)
   - Opportunity: Develop adaptive or transfer learning approaches

3. **Hyperparameter Sensitivity**
   - Issue: Deep learning models sensitive to architectural choices (number of layers, hidden units, attention heads)
   - Gap: Insufficient systematic analysis of hyperparameter space
   - Research Need: Guidelines for principled hyperparameter selection

### 6.2 Data and Experimental Design Issues

1. **Look-Ahead Bias**
   - Risk: Using future information during training leads to inflated performance estimates
   - Limited Discussion: Few papers explicitly address temporal validation protocols
   - Best Practice: Walk-forward validation with proper train-validation-test split

2. **Dataset Standardization**
   - Gap: Lack of standardized benchmarks; each paper uses different datasets
   - Implication: Difficult to compare results across studies
   - Opportunity: Community should adopt common benchmarks (e.g., standardized index datasets with defined test periods)

3. **Transaction Costs and Slippage**
   - Limitation: Most academic papers ignore trading costs and market impact
   - Reality: Real trading severely penalizes frequent predictions
   - Gap: Few papers evaluate profitability after accounting for realistic trading friction

### 6.3 Interpretability and Explainability

1. **Black Box Problem**
   - Challenge: Deep learning models provide predictions without clear explanations
   - Exception: Attention mechanisms provide some interpretability
   - Gap: Limited work on SHAP, LIME, or other explainability techniques in financial forecasting

2. **Feature Attribution**
   - Question: Which input features drive predictions?
   - Limited Work: Few papers isolate feature importance in financial neural networks
   - Opportunity: Integrate explainability requirements into financial models

### 6.4 Computational and Practical Challenges

1. **Training Stability**
   - Issue: Neural networks prone to divergence and instability during training
   - Solution Approaches: Batch normalization, gradient clipping, learning rate scheduling (not always discussed)

2. **Computational Cost**
   - Challenge: Transformer models computationally expensive; limited applicability to ultra-high-frequency trading
   - Gap: Few papers analyze computational cost vs. accuracy trade-offs

3. **Real-Time Deployment**
   - Gap: Most academic papers do not address production deployment constraints
   - Opportunity: Bridge between research and practical implementation

### 6.5 Theoretical Gaps

1. **Why Deep Learning Works for Finance**
   - Limited Theory: Insufficient theoretical understanding of why neural networks capture financial dynamics
   - Question: What properties of neural architectures align with market microstructure?

2. **Regime Detection and Adaptation**
   - Gap: Lack of principled frameworks for detecting market regime changes
   - Opportunity: Develop online learning and adaptive models that adjust to regime shifts

3. **Risk and Uncertainty Quantification**
   - Limited Work: Few papers address prediction intervals or uncertainty estimation in financial forecasting
   - Exception: D-VAE explicitly models variance

---

## 7. State-of-the-Art Summary (2025)

### 7.1 Architecturally Superior Approaches

**Tier 1: Proven High Performance**

1. **Transformer-Based Models**
   - Status: Consistently outperform alternatives across multiple benchmarks
   - Rationale: Long-range dependencies without gradient decay; parallel training efficiency
   - Recommendation: Use as primary baseline for new financial forecasting tasks
   - Caveat: Computational overhead; requires sufficient training data

2. **Hybrid Econometric-Neural Models (ARIMA/GARCH + CNN/LSTM)**
   - Status: Strong performance in volatility and mixed frequency forecasting
   - Rationale: Leverages complementary strengths
   - Recommendation: Appropriate when both linear and nonlinear patterns exist
   - Example: SARIMA-GARCH-CNN-BiLSTM

3. **Graph Neural Networks (for Multi-Stock Problems)**
   - Status: Emerging leader in portfolio and index forecasting
   - Advantage: Explicitly models inter-stock relationships
   - Quantitative Gain: 4-15% improvement over univariate baselines
   - Recommendation: Essential for correlated asset groups

**Tier 2: Specialized/Conditional Strong Performance**

4. **BiLSTM with Self-Attention**
   - Status: Strong alternative to Transformer when computational constraints exist
   - Advantage: Bidirectional context + attention-based weighting
   - Recommendation: For resource-limited production environments

5. **Variational Autoencoders (VAE)**
   - Status: Excellent for feature reduction and stochastic modeling
   - Advantage: Uncertainty quantification; interpretable latent representations
   - Use Case: Dimensionality reduction (16.67% feature reduction with maintained accuracy)
   - Integration: Component of ensemble methods

6. **Attention-Enhanced CNN-LSTM**
   - Status: Solid performer for image-like feature extraction + sequential modeling
   - Advantage: Combines local (CNN) and global (attention) pattern recognition
   - Limitation: More complex than standalone architectures

### 7.2 Ensemble Methods as Production Standard

- **Finding:** Ensemble approaches consistently outperform single architectures
- **Composition:** Typical ensemble includes:
  - LSTM / BiLSTM (temporal baseline)
  - Transformer (long-range dependencies)
  - GNN (relational patterns, if applicable)
  - Classical ARIMA/GARCH (linear trend)
- **Prediction Combination:** Weighted averaging or meta-learner

### 7.3 Key Performance Benchmarks (Summarized)

| Architecture | Best Metric Reported | Dataset | Notes |
|---|---|---|---|
| Transformer (multi-index) | Lowest MAE (all indices) | 10 market indices | 2024 study |
| BiLSTM-SAM-TCN | Highest accuracy (developed & developing markets) | Multiple indices | 2024 study |
| LSTM-GRU | MAPE 3.97%, RMSE 381.34 | Financial series | Dataset-dependent |
| GraphCNNpred | Sharpe > 3.0 (trading sim) | Stock indices | Relationship-capturing |
| CNN-LSTM-GNN | 4-15% F-measure gain | Stock correlations | Relational advantage |
| D-VAE with Sentiment | R² = 0.94489 | Stock + sentiment | Feature importance |

---

## 8. Chronological Development Summary

### Pre-2021: Foundational Work
- RNN and LSTM fundamentals established
- Early attention mechanism proposals
- ARIMA/GARCH baseline models entrenched

### 2021-2022: Diversification Phase
- BiLSTM applications expand
- GNN adaptation to finance begins
- CNN-LSTM hybrids introduced
- Classical econometric integration accelerates

### 2023: Transformer Ascendancy
- Transformer models show consistent superiority
- Multi-head attention proves effective
- GNN-based stock correlation models mature
- Hybrid ARIMA-GARCH-neural network frameworks solidify

### 2024: Ensemble and VAE Boom
- Ensemble methods become standard in competitive analyses
- Diffusion Variational Autoencoders introduced for stochasticity
- Sentiment integration into deep learning models increases
- BiLSTM-SAM-TCN and other multi-component hybrids emerge
- Systematic reviews (187-study meta-analyses) provide comprehensive overviews

### 2025 (Current): Consolidation and Production Focus
- Transformer models remain dominant but with increased scrutiny on practical viability
- Ensemble approaches formalized as production standard
- Sentiment and LLM integration accelerates
- Computational efficiency and real-time deployment gain emphasis
- Explainability and risk quantification increasingly addressed

---

## 9. Comprehensive Reference Table: Prior Work Summary

| Citation | Authors/Year | Problem | Architecture(s) | Dataset | Quantitative Result | Limitation/Assumption |
|---|---|---|---|---|---|---|
| LSTM–Transformer Hybrid for Robust Forecasting | 2025 (MDPI) | Time-series forecasting | LSTM + Transformer | Normalized Jan 2020-Jun 2024 (COVID-19, war events) | Robust across regimes | Limited metric specification |
| Financial Ensemble Model (LSTM-mTrans-MLP) | 2024 (MDPI Mathematics) | Stock price forecasting | Ensemble of LSTM, mTransformer, MLP | Financial series | Outperforms single models | Ensemble computational cost |
| Predictive Modeling: Transformer Model | 2024 (ACM) | Stock price prediction | Transformer encoder-decoder | 10 market indices | Significantly better than 12 baselines | Benchmark architecture variance |
| Deep Convolutional Transformer Network | 2024 (Electronics) | Stock movement prediction | CNN + Transformer + multi-head attention | Not specified | Multi-head attention effective | Feature extraction details sparse |
| Frequency Decomposition GRU-Transformer | 2023 (Applied Sciences) | Stock price forecasting | GRU + Transformer + freq decomposition | Financial time series | Captures multiple frequencies | Decomposition parameter sensitivity |
| Stock Market Index Prediction (Transformer) | 2024 (Neural Computing & Applications) | Index forecasting | 12 architectures vs. Transformer | 10 market indices | Transformer consistently best | Cross-index generalization unclear |
| Systematic Review: GNN Methods | 2024 (ACM Computing Surveys) | Stock market forecasting | GNN survey | Multiple architectures | GNNs capture relational patterns | Architecture-agnostic review |
| LSTM-GNN Hybrid | 2025 (ArXiv) | Stock price prediction | LSTM + GNN ensemble | Undisclosed | Combines temporal + relational | Early preprint; limited results |
| ChatGPT-Informed GNN | 2023 (ArXiv) | Stock movement prediction | GNN + LLM features | Multiple stock datasets | LLM enrichment effective | Limited quantitative reporting |
| GraphCNNpred System | 2024 (ACM) | Index prediction | Graph CNN | Stock indices | Sharpe ratio > 3.0 (trading) | Test period specification unclear |
| CNN-LSTM-GNN Hybrid | 2023-2024 | Stock price forecasting | CNN + LSTM + GNN | Correlated stocks | 4-15% F-measure improvement | Improvement depends on baseline |
| DeepInvesting BiLSTM | 2024 (ScienceDirect) | AMZN stock prediction | BiLSTM stacked | AMZN market data | High R², beats KNN/LSTM/CNN | Specific metrics not disclosed |
| Evaluation BiLSTM | 2024 | Short/long-term prediction | BiLSTM vs. LSTM vs. ARIMA | Stock data | BiLSTM > LSTM > ARIMA | Out-of-sample analysis limited |
| BiLSTM-SAM-TCN Combined | 2024 (ScienceDirect) | High/low price prediction | BiLSTM + self-attention + TCN | Developed & developing markets | Outperforms existing models | Independent high/low modeling unusual |
| CNN-BiLSTM-Attention | 2024 (Mathematics) | Stock price prediction | CNN + BiLSTM + attention | Financial indices | Improved accuracy | Ensemble computational cost |
| Diffusion Variational Autoencoder | 2023 (ArXiv) | Multi-step stock prediction | Hierarchical VAE + diffusion | Stock data | Outperforms SOTA; accurate variance | Stochasticity modeling complexity |
| D-VAE with Sentiment | 2024 (International Journal) | Stock price prediction | D-VAE + sentiment analysis | Stock prices + news | R² = 0.94489 (with sentiment) | Sentiment data dependency |
| VAE Feature Reduction | 2021 (Financial Innovation) | Stock market prediction | VAE + recursive feature elimination | Stock features | 16.67% fewer features; same accuracy | Feature importance ranking unclear |
| Hybrid GARCH-Deep Learning | 2024 (Journal of Applied Mathematics) | Volatility prediction | SARIMA-GARCH + CNN-BiLSTM | Financial volatility | Resolves hybrid shortcomings | Volatility metric not quantified |
| ARIMA-ANN Serial | 2024 (Applied Sciences) | Time series forecasting | ARIMA then ANN on residuals | Financial data | Modular interpretability gained | Residual modeling effectiveness varies |
| Wavelet-ARIMA-GARCH | 2023 (Eurasian Economic Review) | Volatility forecasting | MODWT + ARIMA-GARCH | Financial data | Multi-scale decomposition effective | Component interaction not modeled |
| GRU-CNN-Attention | 2022-2024 | Short-term trend prediction | GRU + CNN + attention | Financial series | Dynamic weighting effective | Architecture comparison limited |
| Deep Learning Review 2020-2022 | 2024 (WIREs Data Mining) | Comprehensive survey | MLPs, RNNs, CNNs, Transformers, GNNs, GANs, LLMs | Multiple financial tasks | Architecture categorization provided | Limited quantitative synthesis |
| Deep Learning Financial Forecasting Review | 2025 (ScienceDirect) | Comprehensive survey | 187 studies across architectures | Multiple tasks (stock, forex, crypto, etc.) | Ensemble > single models | Task-dependent performance variation |
| Ensemble Volatility Methods | 2024 (CMES) | Volatility prediction | LSTM, BiLSTM, GRU + self-attention | Financial data | ANNs with memory top performers | High-volatility period emphasis |
| Stock Price Prediction RNN Limitations | 2024 | Forecasting fundamentals | RNN + LSTM compared | Stock data | LSTM mitigates gradient problems | Long-term dependency limits remain |
| Multi-Head Attention Multi-Horizon | 2024 (MDPI) | Multi-horizon forecasting | CNN + LSTM + multi-head attention | Stock prices | Superior cumulative returns | Backtesting methodology unclear |
| Transformer vs. Traditional | 2025 (Phys.org) | Stock return prediction | Transformer vs. RNN/traditional | Multiple time intervals | Transformer > traditional networks | Generalization across asset classes unclear |
| GRU vs. LSTM Comparison | 2024 | Time series forecasting | GRU, LSTM, hybrid models | Financial series | GRU: MAPE 3.97%, RMSE 381.34 | Dataset-specific performance |
| Multi-Layer Hybrid MTL | 2025 (ArXiv) | Stock market prediction | Multi-task learning hybrid | Stock data | Boosted accuracy via task relationships | MTL framework complexity |

---

## 10. Concluding Remarks and Future Directions

### Current State (2025)

Deep learning has established clear dominance in financial time-series forecasting over classical statistical methods, with **Transformer architectures** emerging as the most consistently superior architecture across benchmarks. Hybrid approaches combining classical econometrics (ARIMA, GARCH) with neural networks demonstrate added value in specific contexts (volatility forecasting, multi-frequency modeling).

**Key Consensus Findings:**
1. LSTM/BiLSTM outperform plain RNNs; Transformers outperform both
2. Ensemble methods prove more reliable than single architectures
3. Integration of external signals (sentiment, news, macroeconomic data) improves accuracy
4. Graph-based methods excel for correlated asset groups
5. Feature reduction via VAE enables dimensionality reduction without accuracy loss

### Critical Unresolved Issues

1. **Practical Viability:** Gap between academic accuracy metrics and trading profitability
2. **Robustness:** Limited understanding of model behavior across market regimes
3. **Explainability:** Deep learning models remain largely black boxes
4. **Real-Time Deployment:** Computational efficiency constraints in production
5. **Theoretical Understanding:** Why neural networks align with market microstructure remains poorly understood

### Recommended Future Research Directions

1. **Adaptive Models:** Online learning frameworks that detect and respond to regime changes
2. **Explainable AI for Finance:** Integrate SHAP, LIME, and attention-based interpretability into financial models
3. **Uncertainty Quantification:** Develop prediction intervals and confidence bands beyond point estimates
4. **Transaction Cost Integration:** Evaluate profitability after realistic trading friction
5. **Standardized Benchmarks:** Establish community benchmarks for fair cross-study comparison
6. **Causal Inference:** Move beyond correlation to causal understanding of market dynamics
7. **Physics-Informed Neural Networks:** Leverage market microstructure constraints to improve generalization

---

## References

### Survey and Review Articles

- Zhang, W. et al. (2024). "Deep learning models for price forecasting of financial time series: A review of recent advancements: 2020–2022." *WIREs Data Mining and Knowledge Discovery*, 2024. [https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1519](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1519)

- *Survey of Deep Learning for Financial Forecasting* (2025). "Deep learning for financial forecasting: A review of recent trends." *ScienceDirect*, 2025. [https://www.sciencedirect.com/science/article/pii/S1059056025008822](https://www.sciencedirect.com/science/article/pii/S1059056025008822)

- *Data-Driven Stock Forecasting Review* (2024). "Data-driven stock forecasting models based on neural networks: A review." *ScienceDirect*, 2024. [https://www.sciencedirect.com/science/article/pii/S1566253524003944](https://www.sciencedirect.com/science/article/pii/S1566253524003944)

- Ghalanos, A. (2025). "A comprehensive survey of deep learning for time series forecasting: architectural diversity and open challenges." *Artificial Intelligence Review*, 2025. [https://link.springer.com/article/10.1007/s10462-025-11223-9](https://link.springer.com/article/10.1007/s10462-025-11223-9)

### LSTM and RNN Methods

- "LSTM–Transformer-Based Robust Hybrid Deep Learning Model for Financial Time Series Forecasting" (2025). *MDPI Forecasting*. [https://www.mdpi.com/2413-4155/7/1/7](https://www.mdpi.com/2413-4155/7/1/7)

- "Financial Time Series Forecasting with the Deep Learning Ensemble Model" (2024). *MDPI Mathematics*, 11(4), 1054. [https://www.mdpi.com/2227-7390/11/4/1054](https://www.mdpi.com/2227-7390/11/4/1054)

- "A Comparison between ARIMA, LSTM, and GRU for Time Series Forecasting" (2024). *ResearchGate*. [https://www.researchgate.net/publication/339093433](https://www.researchgate.net/publication/339093433)

- "Predicting Stock Market Trends Using LSTM Networks: Overcoming RNN Limitations for Improved Financial Forecasting" (2024). *Journal of Computer Science and Software Applications*. [https://mfacademia.org/index.php/jcssa/article/view/100](https://mfacademia.org/index.php/jcssa/article/view/100)

- "A Comparison of LSTM, GRU, and XGBoost for forecasting Morocco's yield curve" (2024). *Science*, 11(3), 2024. [https://science.lpnu.ua/mmc/all-volumes-and-issues/volume-11-number-3-2024/comparison-lstm-gru-and-xgboost-forecasting](https://science.lpnu.ua/mmc/all-volumes-and-issues/volume-11-number-3-2024/comparison-lstm-gru-and-xgboost-forecasting)

### Transformer Methods

- "Predictive Modeling of Stock Prices Using Transformer Model" (2024). *Proceedings of the ACM*. [https://dl.acm.org/doi/fullHtml/10.1145/3674029.3674037](https://dl.acm.org/doi/fullHtml/10.1145/3674029.3674037)

- "Deep Convolutional Transformer Network for Stock Movement Prediction" (2024). *Electronics*, 13(21), 4225. [https://www.mdpi.com/2079-9302/13/21/4225](https://www.mdpi.com/2079-9302/13/21/4225)

- "Stock market index prediction using transformer neural network models and frequency decomposition" (2024). *Neural Computing and Applications*. [https://link.springer.com/article/10.1007/s00521-024-09931-4](https://link.springer.com/article/10.1007/s00521-024-09931-4)

- "Stock Price Prediction Using a Frequency Decomposition Based GRU Transformer Neural Network" (2023). *Applied Sciences*, 13(1), 222. [https://www.mdpi.com/2076-3417/13/1/222](https://www.mdpi.com/2076-3417/13/1/222)

- "Transformer-Based Deep Learning Model for Stock Price Prediction: A Case Study on Bangladesh Stock Market" (2024). *International Journal of Computational Intelligence and Applications*. [https://www.worldscientific.com/doi/10.1142/S146902682350013X](https://www.worldscientific.com/doi/10.1142/S146902682350013X)

- "Transformer AI models outperform neural networks in stock market prediction, study shows" (2025). *Phys.org*, November 2025. [https://phys.org/news/2025-11-ai-outperform-neural-networks-stock.html](https://phys.org/news/2025-11-ai-outperform-neural-networks-stock.html)

### Graph Neural Networks

- "A Systematic Review on Graph Neural Network-based Methods for Stock Market Forecasting" (2024). *ACM Computing Surveys*. [https://dl.acm.org/doi/10.1145/3696411](https://dl.acm.org/doi/10.1145/3696411)

- "STOCK PRICE PREDICTION USING A HYBRID LSTM-GNN" (2025). *ArXiv*. [https://arxiv.org/pdf/2502.15813](https://arxiv.org/pdf/2502.15813)

- "CHATGPT INFORMED GRAPH NEURAL NETWORK FOR STOCK MOVEMENT PREDICTION" (2023). *ArXiv*. [https://arxiv.org/pdf/2306.03763](https://arxiv.org/pdf/2306.03763)

- "GraphCNNpred: A stock market indices prediction using a Graph based deep learning system" (2024). *Proceedings of the 2nd International Conference on Artificial Intelligence, Systems and Network Security*. [https://dl.acm.org/doi/10.1145/3714334.3714364](https://dl.acm.org/doi/10.1145/3714334.3714364)

- "A graph neural network-based stock forecasting method utilizing multi-source heterogeneous data fusion" (2022). *Multimedia Tools and Applications*. [https://link.springer.com/article/10.1007/s11042-022-13231-1](https://link.springer.com/article/10.1007/s11042-022-13231-1)

### Bidirectional LSTM (BiLSTM)

- "DeepInvesting: Stock market predictions with a sequence-oriented BiLSTM stacked model – A dataset case study of AMZN" (2024). *ScienceDirect*. [https://www.sciencedirect.com/science/article/pii/S2667305324001133](https://www.sciencedirect.com/science/article/pii/S2667305324001133)

- "Evaluation of bidirectional LSTM for short-and long-term stock market prediction" (2024). *ResearchGate*. [https://www.researchgate.net/publication/324996793](https://www.researchgate.net/publication/324996793)

- "Predicting the highest and lowest stock price indices: A combined BiLSTM-SAM-TCN deep learning model based on re-decomposition" (2024). *ScienceDirect*. [https://www.sciencedirect.com/science/article/abs/pii/S1568494624011670](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011670)

- "Forecasting S&P 500 Using LSTM Models" (2025). *ArXiv*. [https://arxiv.org/html/2501.17366v1](https://arxiv.org/html/2501.17366v1)

### Convolutional Neural Networks (CNN)

- "Stock Price Prediction Using CNN-BiLSTM-Attention Model" (2024). *Mathematics*, 11(9), 1985. [https://www.mdpi.com/2227-7390/11/9/1985](https://www.mdpi.com/2227-7390/11/9/1985)

- "cnn-based stock price forecasting - by stock chart images" (2023). *Romanian Journal of Economic Forecasting*. [https://ipe.ro/new/rjef/rjef3_2023/rjef3_2023p120-128.pdf](https://ipe.ro/new/rjef/rjef3_2023/rjef3_2023p120-128.pdf)

- "Stock Price Prediction Using Deep-Learning Models: CNN, LSTM, and Ensemble" (2024). *SHS Conferences*. [https://www.shs-conferences.org/articles/shsconf/pdf/2024/16/shsconf_edma2024_02004.pdf](https://www.shs-conferences.org/articles/shsconf/pdf/2024/16/shsconf_edma2024_02004.pdf)

### Attention Mechanisms

- "Stock Price Prediction using Deep Neural Network based on Attention Mechanism" (2023). *Proceedings of the International Conference on Intelligent Computing and Its Emerging Applications*. [https://dl.acm.org/doi/10.1145/3659154.3659157](https://dl.acm.org/doi/10.1145/3659154.3659157)

- "Enhancing stock price prediction using GANs and transformer-based attention mechanisms" (2024). *Empirical Economics*. [https://link.springer.com/article/10.1007/s00181-024-02644-6](https://link.springer.com/article/10.1007/s00181-024-02644-6)

- "Deep Learning-Based Hybrid Model with Multi-Head Attention for Multi-Horizon Stock Price Prediction" (2024). *MDPI*, 18(10), 551. [https://www.mdpi.com/1911-8074/18/10/551](https://www.mdpi.com/1911-8074/18/10/551)

### Variational Autoencoders (VAE)

- "Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction" (2023). *ArXiv*. [https://arxiv.org/abs/2309.00073](https://arxiv.org/abs/2309.00073)

- "Implementation of Diffusion Variational Autoencoder for Stock Price Prediction with the Integration of Historical and Market Sentiment Data" (2024). *International Journal of Electrical, Computer, and Biomedical Engineering*. [https://ijecbe.ui.ac.id/go/article/view/55](https://ijecbe.ui.ac.id/go/article/view/55)

- "An efficient stock market prediction model using hybrid feature reduction method based on variational autoencoders and recursive feature elimination" (2021). *Financial Innovation*, 7(1), 43. [https://jfin-swufe.springeropen.com/articles/10.1186/s40854-021-00243-3](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-021-00243-3)

- "An Advanced Ensemble Deep Learning Framework for Stock Price Prediction Using VAE, Transformer, and LSTM Model" (2025). *ArXiv*. [https://arxiv.org/html/2503.22192v1](https://arxiv.org/html/2503.22192v1)

### Hybrid Econometric-Neural Networks

- "A Hybrid GARCH and Deep Learning Method for Volatility Prediction" (2024). *Journal of Applied Mathematics*. [https://onlinelibrary.wiley.com/doi/10.1155/2024/6305525](https://onlinelibrary.wiley.com/doi/10.1155/2024/6305525)

- "A Hybrid Forecasting Structure Based on Arima and Artificial Neural Network Models" (2024). *Applied Sciences*, 14(16), 7122. [https://www.mdpi.com/2076-3417/14/16/7122](https://www.mdpi.com/2076-3417/14/16/7122)

- "Forecasting volatility by using wavelet transform, ARIMA and GARCH models" (2023). *Eurasian Economic Review*. [https://link.springer.com/article/10.1007/s40822-023-00243-x](https://link.springer.com/article/10.1007/s40822-023-00243-x)

- "Hybrid ARMA-GARCH-Neural Networks for intraday strategy exploration in high-frequency trading" (2023). *Pattern Recognition*. [https://www.sciencedirect.com/science/article/pii/S0031320323008361](https://www.sciencedirect.com/science/article/pii/S0031320323008361)

- "A Comparative Analysis of ARIMA-GARCH, LSTM, and ..." (2024). *SHS Conferences*. [https://www.shs-conferences.org/articles/shsconf/pdf/2024/16/shsconf_edma2024_02008.pdf](https://www.shs-conferences.org/articles/shsconf/pdf/2024/16/shsconf_edma2024_02008.pdf)

### Ensemble Methods

- "Deep Learning for Financial Time Series Prediction: A State-of-the-Art Review of Standalone and Hybrid Models" (2024). *CMES*, 139(1), 55114. [https://www.techscience.com/CMES/v139n1/55114/html](https://www.techscience.com/CMES/v139n1/55114/html)

- "Financial Forecasting with Deep Learning Models Based Ensemble Technique in Stock Market Analysis" (2024). *IJIEEB*, 17(4). [https://www.mecs-press.org/ijieeb/ijieeb-v17-n4/v17n4-1.html](https://www.mecs-press.org/ijieeb/ijieeb-v17-n4/v17n4-1.html)

- "Boosting the Accuracy of Stock Market Prediction via Multi-Layer Hybrid MTL" (2025). *ArXiv*. [https://arxiv.org/pdf/2501.09760](https://arxiv.org/pdf/2501.09760)

### Multi-Step Forecasting and Advanced Approaches

- "Applications of Deep Learning in Financial Time Series Forecasting" (2025). *Proceedings of the International Conference on Digital Economy and Information Systems*. [https://dl.acm.org/doi/10.1145/3745133.3745141](https://dl.acm.org/doi/10.1145/3745133.3745141)

- "A hybrid framework of deep learning and traditional time series models for exchange rate prediction" (2025). *ScienceDirect*. [https://www.sciencedirect.com/science/article/pii/S246822762500287X](https://www.sciencedirect.com/science/article/pii/S246822762500287X)

- "Deep neural network approach integrated with reinforcement learning for forecasting exchange rates using time series data and influential factors" (2025). *Scientific Reports*. [https://www.nature.com/articles/s41598-025-12516-3](https://www.nature.com/articles/s41598-025-12516-3)

- "Deep learning for time series forecasting: a survey" (2025). *International Journal of Machine Learning and Cybernetics*. [https://link.springer.com/article/10.1007/s13042-025-02560-w](https://link.springer.com/article/10.1007/s13042-025-02560-w)

### Performance Metrics and Evaluation

- "Stock market trend prediction using deep neural network via chart analysis: a practical method or a myth?" (2025). *Humanities and Social Sciences Communications*. [https://www.nature.com/articles/s41599-025-04761-8](https://www.nature.com/articles/s41599-025-04761-8)

- "Forecasting stock prices changes using long-short term memory neural network with symbolic genetic programming" (2023). *Scientific Reports*. [https://www.nature.com/articles/s41598-023-50783-0](https://www.nature.com/articles/s41598-023-50783-0)

### Additional Related Work

- "Stock Price Prediction Using Technical Indicators" (2024). *SCITEPRESS*. [https://www.scitepress.org/Papers/2024/132649/132649.pdf](https://www.scitepress.org/Papers/2024/132649/132649.pdf)

- "Stock Price Prediction with Deep RNNs using Multi-Faceted Info" (2024). *ArXiv*. [https://arxiv.org/pdf/2411.19766](https://arxiv.org/pdf/2411.19766)

- "Multi-Agent Stock Prediction Systems: Machine Learning Models, Simulations, and Real-Time Trading Strategies" (2025). *ArXiv*. [https://arxiv.org/html/2502.15853v1](https://arxiv.org/html/2502.15853v1)

- "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities" (2024). *PMC*. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10963254/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10963254/)

- "Stock Price Prediction Using Convolutional Neural Networks on a Multivariate Timeseries" (2020-2024). *ArXiv*. [https://arxiv.org/abs/2001.09769](https://arxiv.org/abs/2001.09769)

- "Forecasting Forex Market Volatility Using Deep Learning Models and Complexity Measures" (2024). *MDPI Forecasting*, 17(12), 557. [https://www.mdpi.com/1911-8074/17/12/557](https://www.mdpi.com/1911-8074/17/12/557)

- "Improving Volatility Forecasting: A Study through Hybrid Deep Learning Methods with WGAN" (2024). *MDPI Forecasting*, 17(9), 380. [https://www.mdpi.com/1911-8074/17/9/380](https://www.mdpi.com/1911-8074/17/9/380)

- "Variational autoencoder-based dimension reduction of Ichimoku features for improved financial market analysis" (2024). *ScienceDirect*. [https://www.sciencedirect.com/science/article/pii/S2773186324000653](https://www.sciencedirect.com/science/article/pii/S2773186324000653)

- "Traffic speed prediction using GARCH‐GRU hybrid model" (2023). *IET Intelligent Transport Systems*. [https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12411](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12411)

- "Securities Price Movement Prediction Based on Graph Neural Networks" (2023). *Proceedings of the 4th International Conference on Machine Learning and Computer Application*. [https://dl.acm.org/doi/10.1145/3650215.3650345](https://dl.acm.org/doi/10.1145/3650215.3650345)

---

**Document Compiled:** December 22, 2025
**Total References:** 60+ peer-reviewed and preprint sources
**Coverage Period:** 2021-2025 with selected foundational works
**Focus Areas:** LSTM, Transformers, GNNs, Hybrids, Attention Mechanisms, Ensemble Methods, VAE
**Scope:** Financial time-series forecasting with quantitative benchmarks
