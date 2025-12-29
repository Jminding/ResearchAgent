# Formal Mathematical Framework: Transformer-Based Order Flow Microstructure Model (TOFM)

## 1. Problem Formalization

### 1.1 Objective

Construct a predictive model that leverages high-frequency order flow data through a transformer architecture to forecast short-term price movements, while explicitly incorporating market microstructure dynamics including adverse selection, inventory effects, and information asymmetry.

### 1.2 State Space Definition

Let the market state at discrete time t be characterized by the tuple:

```
S_t = (LOB_t, OF_t, P_t, V_t, I_t)
```

Where:
- **LOB_t**: Limit Order Book state (bid/ask prices and quantities across L levels)
- **OF_t**: Order Flow history (signed trade sequence)
- **P_t**: Mid-price at time t
- **V_t**: Realized volatility estimate
- **I_t**: Latent information state (unobserved)

---

## 2. Variable Definitions

### 2.1 Input Variables (Observable)

| Variable | Definition | Dimension |
|----------|------------|-----------|
| p^a_{t,l} | Ask price at level l at time t | R |
| p^b_{t,l} | Bid price at level l at time t | R |
| q^a_{t,l} | Ask quantity at level l | R+ |
| q^b_{t,l} | Bid quantity at level l | R+ |
| s_t | Bid-ask spread: s_t = p^a_{t,1} - p^b_{t,1} | R+ |
| m_t | Mid-price: m_t = (p^a_{t,1} + p^b_{t,1}) / 2 | R |
| OFI_t | Order Flow Imbalance | R |
| VOI_t | Volume Order Imbalance | R |
| TI_t | Trade Imbalance (signed volume) | R |
| lambda^+_t | Buy arrival rate | R+ |
| lambda^-_t | Sell arrival rate | R+ |

### 2.2 Derived Microstructure Features

**Order Flow Imbalance (OFI):**
```
OFI_t = (Delta_q^b_{t,1} - Delta_q^a_{t,1}) / (Delta_q^b_{t,1} + Delta_q^a_{t,1} + epsilon)
```

**Volume-Weighted Price Pressure:**
```
VPP_t = sum_{l=1}^{L} w_l * (q^b_{t,l} - q^a_{t,l}) / sum_{l=1}^{L} (q^b_{t,l} + q^a_{t,l})
```
where w_l = exp(-alpha * l) are exponentially decaying weights.

**Kyle's Lambda Estimate (Information Content):**
```
lambda_t = Cov(Delta_m_{t:t+k}, TI_{t:t+k}) / Var(TI_{t:t+k})
```

**Adverse Selection Component (Roll Decomposition):**
```
AS_t = sqrt(max(0, -Cov(Delta_m_t, Delta_m_{t-1})))
```

**Inventory Imbalance Proxy:**
```
INV_t = cumsum_{s=t-W}^{t}(TI_s) / (W * sigma_TI)
```

### 2.3 Target Variable

**Forward Price Movement:**
```
y_t = sign(m_{t+H} - m_t) in {-1, 0, +1}  (Classification)
```
or
```
y_t = (m_{t+H} - m_t) / m_t  (Regression)
```

where H is the prediction horizon (in ticks or time units).

### 2.4 Latent Variables

| Variable | Interpretation |
|----------|----------------|
| z^inf_t | Latent informed trader activity |
| z^liq_t | Latent liquidity state |
| z^vol_t | Latent volatility regime |

---

## 3. Mathematical Framework

### 3.1 Microstructure Foundation

We adopt a hybrid information model combining elements of Kyle (1985), Glosten-Milgrom (1985), and Hasbrouck (1991).

**Price Dynamics:**
```
m_t = m_{t-1} + lambda * x_t + eta_t
```

Where:
- x_t is the signed order flow (positive = buy initiated)
- lambda is the permanent price impact (adverse selection)
- eta_t ~ N(0, sigma^2_eta) is pricing noise

**Information Asymmetry Measure:**
The probability of informed trading (PIN-like measure):
```
PI_t = alpha * mu / (alpha * mu + 2 * epsilon)
```

Where alpha is probability of information event, mu is informed arrival rate, epsilon is uninformed arrival rate.

### 3.2 Transformer Architecture Formulation

**Input Embedding:**

Let the raw input at time t be:
```
X_t = [OFI_t, VOI_t, TI_t, VPP_t, s_t, lambda_t, AS_t, INV_t, RV_t, q^b_{t,1:L}, q^a_{t,1:L}]
```

Dimension: d_input = 8 + 2L

**Sequence Construction:**
For context window of length T:
```
X_{1:T} = [X_{t-T+1}, X_{t-T+2}, ..., X_t] in R^{T x d_input}
```

**Linear Projection to Model Dimension:**
```
E = X_{1:T} * W_E + b_E
```
where W_E in R^{d_input x d_model}, E in R^{T x d_model}

**Positional Encoding (Learnable Temporal):**
```
PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

**Embedded Input:**
```
H^{(0)} = E + PE
```

### 3.3 Attention Mechanism with Microstructure Bias

**Standard Self-Attention:**
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

**Microstructure-Informed Attention (Novel Contribution):**

We introduce a bias term B_micro that encodes microstructure relationships:

```
Attention_micro(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + B_micro) * V
```

Where B_micro in R^{T x T} is constructed as:

```
B_micro[i,j] = gamma_1 * Corr(OFI_i, OFI_j) + gamma_2 * |lambda_i - lambda_j| + gamma_3 * I[regime_i = regime_j]
```

This biases attention toward:
1. Time steps with correlated order flow patterns
2. Similar adverse selection environments
3. Same volatility regime

### 3.4 Multi-Head Attention

```
MultiHead(H) = Concat(head_1, ..., head_h) * W^O
```

where:
```
head_i = Attention(H * W^Q_i, H * W^K_i, H * W^V_i)
```

### 3.5 Transformer Block

```
H' = LayerNorm(H + MultiHead(H))
H^{(l+1)} = LayerNorm(H' + FFN(H'))
```

where:
```
FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
```

### 3.6 Output Layer

After N transformer blocks:
```
z_t = H^{(N)}[T, :]  (Take last position)
```

**Classification Head:**
```
logits = z_t * W_out + b_out
P(y_t = k) = softmax(logits)_k
```

**Regression Head:**
```
y_hat_t = z_t * W_reg + b_reg
```

---

## 4. Assumptions

### 4.1 Market Microstructure Assumptions

**A1 (Continuous Trading):** Markets operate in continuous time, discretized at sufficiently high frequency that tick-by-tick data approximates the true process.

**A2 (Price Efficiency):** The mid-price follows a martingale with respect to public information:
```
E[m_{t+1} | F_t] = m_t
```
where F_t is the public information filtration.

**A3 (Adverse Selection):** There exist informed traders whose trades convey private information, leading to permanent price impact.

**A4 (Inventory Effects):** Market makers adjust quotes based on inventory, creating temporary price pressure.

**A5 (Order Flow Informativeness):** Order flow contains predictive information about future price movements:
```
E[Delta_m_{t+H} | OF_{1:t}] != 0
```

### 4.2 Statistical Assumptions

**A6 (Stationarity):** Within a trading session, the joint distribution of features and returns is approximately stationary.

**A7 (Ergodicity):** Time averages converge to ensemble averages, enabling learning from historical sequences.

**A8 (Finite Memory):** Relevant information for prediction is contained within a finite lookback window T:
```
P(y_t | X_{1:t}) approx P(y_t | X_{t-T+1:t})
```

### 4.3 Modeling Assumptions

**A9 (Sufficient Representation):** The transformer's learned representation captures the essential microstructure dynamics.

**A10 (Label Quality):** The target variable y_t is accurately constructed from observed prices, accounting for bid-ask bounce.

---

## 5. Hypotheses

### Hypothesis H1: Order Flow Feature Superiority

**Statement:** A transformer model using microstructure-derived order flow features (OFI, VOI, Kyle's lambda) will achieve higher predictive accuracy than a model using raw LOB snapshots alone.

**Formalization:**
```
Acc(TOFM_{microstructure}) > Acc(TOFM_{raw_LOB}) + delta_1
```

where delta_1 >= 0.02 (2 percentage points improvement in directional accuracy).

**Falsification Criterion:** If the microstructure model fails to outperform by at least 2% on out-of-sample data across multiple assets and time periods.

### Hypothesis H2: Attention Pattern Interpretability

**Statement:** The learned attention weights in a trained TOFM will exhibit systematic patterns corresponding to known microstructure phenomena (clustering around high-information events, adverse selection spikes).

**Formalization:**
```
Corr(Attention_weights, Information_proxy) > rho_crit
```

where Information_proxy = |OFI| * lambda (product of imbalance and price impact), and rho_crit = 0.3.

**Falsification Criterion:** If attention patterns show no statistical correlation (p > 0.05) with microstructure events.

### Hypothesis H3: Regime-Dependent Performance

**Statement:** TOFM performance varies systematically with market volatility regimes, with higher accuracy in high-volatility regimes where information asymmetry is elevated.

**Formalization:**
```
Acc(TOFM | RV_t > RV_median) > Acc(TOFM | RV_t <= RV_median) + delta_2
```

where delta_2 >= 0.03.

**Falsification Criterion:** If accuracy is equal or lower in high-volatility regimes.

### Hypothesis H4: Microstructure Attention Bias Improvement

**Statement:** The novel microstructure-informed attention bias (B_micro) improves model performance over standard attention mechanisms.

**Formalization:**
```
Acc(TOFM_{with_bias}) > Acc(TOFM_{standard}) + delta_3
```

where delta_3 >= 0.01.

**Falsification Criterion:** If adding B_micro degrades or does not improve performance.

### Hypothesis H5: Cross-Asset Generalization

**Statement:** A TOFM trained on multiple assets exhibits positive transfer learning, outperforming single-asset models on held-out assets.

**Formalization:**
```
Acc(TOFM_{multi}^{asset_new}) > Acc(TOFM_{single}^{asset_new})
```

**Falsification Criterion:** If multi-asset models underperform single-asset models on new assets.

---

## 6. Loss Functions and Optimization

### 6.1 Classification Loss (Cross-Entropy)

```
L_CE = -1/N * sum_{i=1}^{N} sum_{k in {-1,0,+1}} y_{i,k} * log(p_{i,k})
```

### 6.2 Regression Loss (MSE with Asymmetric Weighting)

```
L_MSE = 1/N * sum_{i=1}^{N} w_i * (y_i - y_hat_i)^2
```

where w_i = 1 + beta * |y_i| to upweight larger moves.

### 6.3 Auxiliary Microstructure Loss

To regularize learned representations toward microstructure-meaningful features:

```
L_aux = ||z_t - f(OFI_t, lambda_t)||_2
```

where f is a simple linear projection of known informative features.

### 6.4 Total Loss

```
L_total = L_CE + alpha_1 * L_aux + alpha_2 * L_reg
```

where L_reg = ||theta||_2^2 is L2 regularization.

---

## 7. Evaluation Metrics

### 7.1 Prediction Accuracy

| Metric | Formula |
|--------|---------|
| Directional Accuracy | sum(sign(y_hat) == sign(y)) / N |
| Precision (per class) | TP / (TP + FP) |
| Recall (per class) | TP / (TP + FN) |
| F1 Score | 2 * Precision * Recall / (Precision + Recall) |
| Cohen's Kappa | (Acc - Acc_random) / (1 - Acc_random) |

### 7.2 Financial Performance

| Metric | Formula |
|--------|---------|
| PnL | sum_{t} signal_t * (m_{t+H} - m_t) |
| Sharpe Ratio | mean(returns) / std(returns) * sqrt(252 * N_daily) |
| Maximum Drawdown | max_{t} (peak_t - trough_t) / peak_t |
| Hit Rate | P(PnL_trade > 0) |

### 7.3 Microstructure Alignment

| Metric | Description |
|--------|-------------|
| Attention-OFI Correlation | Correlation between attention weights and |OFI| |
| Lambda Sensitivity | d(prediction) / d(lambda) |
| Regime Consistency | Accuracy gap between volatility regimes |

---

## 8. Experimental Design Pseudocode

### 8.1 Data Preparation

```
PROCEDURE DataPreparation(raw_data, config):

1. INPUT:
   - raw_data: TAQ/LOBSTER format tick data
   - config: {L=10, T=100, H=10, train_ratio=0.7, val_ratio=0.15}

2. PARSE raw limit order book data:
   - Extract bid/ask prices at L levels: p^a_{t,1:L}, p^b_{t,1:L}
   - Extract bid/ask quantities: q^a_{t,1:L}, q^b_{t,1:L}
   - Extract trade timestamps, prices, volumes, and directions

3. COMPUTE mid-price series:
   - m_t = (p^a_{t,1} + p^b_{t,1}) / 2

4. COMPUTE spread:
   - s_t = p^a_{t,1} - p^b_{t,1}

5. COMPUTE Order Flow Imbalance (OFI):
   - Delta_q^b_t = q^b_{t,1} - q^b_{t-1,1}
   - Delta_q^a_t = q^a_{t,1} - q^a_{t-1,1}
   - OFI_t = (Delta_q^b_t - Delta_q^a_t) / (|Delta_q^b_t| + |Delta_q^a_t| + 1e-8)

6. COMPUTE Volume Order Imbalance (VOI):
   - IF p^b_{t,1} > p^b_{t-1,1}: Delta_q^b_t = q^b_{t,1}
   - ELIF p^b_{t,1} < p^b_{t-1,1}: Delta_q^b_t = -q^b_{t-1,1}
   - ELSE: Delta_q^b_t = q^b_{t,1} - q^b_{t-1,1}
   - (Similar for ask side)
   - VOI_t = Delta_q^b_t - Delta_q^a_t

7. COMPUTE Trade Imbalance (TI):
   - TI_t = sum(signed_volume) over trades in interval t
   - Sign determined by Lee-Ready algorithm or exchange flag

8. COMPUTE Volume-Weighted Price Pressure (VPP):
   - weights = exp(-0.5 * [1, 2, ..., L])
   - VPP_t = sum(weights * (q^b_{t,l} - q^a_{t,l})) / sum(q^b + q^a)

9. COMPUTE Kyle's Lambda (rolling window W=100):
   - FOR each window [t-W, t]:
     - lambda_t = Cov(Delta_m, TI) / (Var(TI) + 1e-8)

10. COMPUTE Adverse Selection Component:
    - rolling_cov = Cov(Delta_m_t, Delta_m_{t-1}) over window W
    - AS_t = sqrt(max(0, -rolling_cov))

11. COMPUTE Inventory Proxy:
    - INV_t = cumsum(TI_{t-W:t}) / (W * std(TI))

12. COMPUTE Realized Volatility:
    - RV_t = sqrt(sum((Delta_m)^2) over window W)

13. CONSTRUCT feature matrix X:
    - X_t = [OFI_t, VOI_t, TI_t, VPP_t, s_t, lambda_t, AS_t, INV_t, RV_t,
             q^b_{t,1:L}, q^a_{t,1:L}]
    - Dimension: d_input = 9 + 2*L

14. NORMALIZE features:
    - FOR each feature f in X:
      - X[:,f] = (X[:,f] - mean(X[:,f])) / (std(X[:,f]) + 1e-8)
    - USE rolling normalization for online deployment

15. CONSTRUCT target variable y:
    - y_t = m_{t+H} - m_t  (raw return)
    - y_class_t = sign(y_t) with threshold tau for neutral class
      - IF |y_t| < tau: y_class_t = 0
      - ELIF y_t >= tau: y_class_t = +1
      - ELSE: y_class_t = -1
    - tau = 0.5 * median(|y|)

16. CREATE sequences:
    - FOR t = T to N-H:
      - X_seq[t] = X[t-T+1 : t+1]  # Shape: (T, d_input)
      - y_seq[t] = y_class_t

17. SPLIT data temporally (no shuffle to preserve time structure):
    - train_idx = [0 : int(0.70 * N)]
    - val_idx = [int(0.70 * N) : int(0.85 * N)]
    - test_idx = [int(0.85 * N) : N]

18. OUTPUT:
    - X_train, y_train, X_val, y_val, X_test, y_test
    - feature_stats (for normalization in inference)
    - metadata (timestamps, asset_id)

END PROCEDURE
```

### 8.2 Model Architecture

```
PROCEDURE BuildTOFMModel(config):

1. INPUT config:
   - d_input: input feature dimension (9 + 2*L)
   - d_model: transformer hidden dimension (128)
   - n_heads: number of attention heads (8)
   - n_layers: number of transformer blocks (4)
   - d_ff: feedforward dimension (512)
   - T: sequence length (100)
   - dropout: dropout rate (0.1)
   - n_classes: output classes (3 for {-1, 0, +1})
   - use_micro_bias: boolean for microstructure attention bias

2. DEFINE InputEmbedding layer:
   - Linear(d_input -> d_model)
   - LayerNorm(d_model)

3. DEFINE PositionalEncoding:
   - PE = zeros(T, d_model)
   - FOR pos in range(T):
     - FOR i in range(0, d_model, 2):
       - PE[pos, i] = sin(pos / 10000^(i/d_model))
       - PE[pos, i+1] = cos(pos / 10000^(i/d_model))
   - REGISTER as buffer (non-trainable)

4. DEFINE MicrostructureAttentionBias (if use_micro_bias):
   - gamma = learnable parameters (3,)
   - FUNCTION compute_bias(X_batch):
     - ofi = X_batch[:, :, 0]  # OFI feature
     - lambda_feat = X_batch[:, :, 5]  # Kyle's lambda
     - corr_matrix = compute_pairwise_correlation(ofi)
     - lambda_diff = |lambda_feat.unsqueeze(2) - lambda_feat.unsqueeze(1)|
     - B_micro = gamma[0] * corr_matrix + gamma[1] * lambda_diff
     - RETURN B_micro  # Shape: (batch, T, T)

5. DEFINE TransformerBlock:
   - MultiHeadAttention(d_model, n_heads, dropout, bias=B_micro)
   - LayerNorm(d_model)
   - FeedForward: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
   - LayerNorm(d_model)
   - Residual connections

6. DEFINE ClassificationHead:
   - Linear(d_model -> d_model // 2)
   - GELU activation
   - Dropout(dropout)
   - Linear(d_model // 2 -> n_classes)

7. DEFINE AuxiliaryHead (for microstructure regularization):
   - Linear(d_model -> 2)  # Predict OFI and lambda

8. ASSEMBLE full model:
   - input -> InputEmbedding -> add PositionalEncoding
   - FOR layer in range(n_layers):
     - H = TransformerBlock(H, B_micro)
   - z = H[:, -1, :]  # Last timestep representation
   - logits = ClassificationHead(z)
   - aux_pred = AuxiliaryHead(z)

9. OUTPUT: model with parameters theta

END PROCEDURE
```

### 8.3 Training Procedure

```
PROCEDURE TrainTOFM(model, data, config):

1. INPUT:
   - model: initialized TOFM
   - data: (X_train, y_train, X_val, y_val)
   - config: {
       epochs: 100,
       batch_size: 256,
       lr: 1e-4,
       weight_decay: 1e-5,
       patience: 10,
       alpha_aux: 0.1,
       warmup_steps: 1000,
       gradient_clip: 1.0
     }

2. INITIALIZE:
   - optimizer = AdamW(model.parameters, lr=config.lr, weight_decay=config.weight_decay)
   - scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)
   - criterion_main = CrossEntropyLoss(weight=class_weights)  # Handle imbalance
   - criterion_aux = MSELoss()
   - best_val_loss = infinity
   - patience_counter = 0

3. COMPUTE class weights:
   - class_counts = count(y_train per class)
   - class_weights = max(class_counts) / class_counts
   - NORMALIZE so sum = n_classes

4. CREATE data loaders:
   - train_loader = DataLoader(train_data, batch_size, shuffle=True)
   - val_loader = DataLoader(val_data, batch_size, shuffle=False)

5. FOR epoch in range(1, epochs + 1):

   5.1 TRAINING PHASE:
   - model.train()
   - epoch_loss = 0
   - FOR batch_idx, (X_batch, y_batch) in enumerate(train_loader):

     a. Forward pass:
        - logits, aux_pred = model(X_batch)

     b. Compute losses:
        - loss_main = criterion_main(logits, y_batch)
        - ofi_true = X_batch[:, -1, 0]  # Last timestep OFI
        - lambda_true = X_batch[:, -1, 5]  # Last timestep lambda
        - loss_aux = criterion_aux(aux_pred, stack([ofi_true, lambda_true]))
        - loss_total = loss_main + config.alpha_aux * loss_aux

     c. Backward pass:
        - optimizer.zero_grad()
        - loss_total.backward()
        - clip_gradient_norm(model.parameters, config.gradient_clip)
        - optimizer.step()
        - scheduler.step()

     d. Accumulate:
        - epoch_loss += loss_total.item()

   - avg_train_loss = epoch_loss / len(train_loader)

   5.2 VALIDATION PHASE:
   - model.eval()
   - val_loss = 0
   - all_preds = []
   - all_labels = []
   - WITH no_gradient:
     - FOR X_batch, y_batch in val_loader:
       - logits, aux_pred = model(X_batch)
       - loss = criterion_main(logits, y_batch)
       - val_loss += loss.item()
       - preds = argmax(logits, dim=1)
       - all_preds.extend(preds)
       - all_labels.extend(y_batch)

   - avg_val_loss = val_loss / len(val_loader)
   - val_accuracy = accuracy(all_preds, all_labels)
   - val_f1 = f1_score(all_preds, all_labels, average='macro')

   5.3 LOGGING:
   - LOG(epoch, avg_train_loss, avg_val_loss, val_accuracy, val_f1)

   5.4 EARLY STOPPING CHECK:
   - IF avg_val_loss < best_val_loss:
     - best_val_loss = avg_val_loss
     - SAVE model checkpoint
     - patience_counter = 0
   - ELSE:
     - patience_counter += 1
     - IF patience_counter >= config.patience:
       - LOG("Early stopping triggered")
       - BREAK

6. LOAD best checkpoint

7. OUTPUT: trained model, training_history

END PROCEDURE
```

### 8.4 Evaluation Procedure

```
PROCEDURE EvaluateTOFM(model, X_test, y_test, config):

1. INPUT:
   - model: trained TOFM
   - X_test, y_test: test data
   - config: evaluation parameters

2. SET model to evaluation mode

3. GENERATE predictions:
   - all_logits = []
   - all_probs = []
   - all_preds = []
   - all_labels = []
   - all_attention_weights = []

   - WITH no_gradient:
     - FOR batch in test_loader:
       - logits, _, attention = model(batch.X, return_attention=True)
       - probs = softmax(logits, dim=1)
       - preds = argmax(logits, dim=1)
       - STORE logits, probs, preds, batch.y, attention

4. COMPUTE classification metrics:
   - accuracy = sum(preds == labels) / N
   - precision_per_class = compute_precision(preds, labels)
   - recall_per_class = compute_recall(preds, labels)
   - f1_per_class = compute_f1(preds, labels)
   - f1_macro = mean(f1_per_class)
   - confusion_matrix = compute_confusion_matrix(preds, labels)
   - cohens_kappa = compute_kappa(preds, labels)

5. COMPUTE financial metrics:
   - signals = preds - 1  # Map {0,1,2} to {-1,0,+1}
   - returns = actual_price_changes[H:]
   - strategy_returns = signals * returns
   - cumulative_pnl = cumsum(strategy_returns)
   - sharpe_ratio = mean(strategy_returns) / std(strategy_returns) * sqrt(annualization_factor)
   - max_drawdown = compute_max_drawdown(cumulative_pnl)
   - hit_rate = mean(strategy_returns > 0)

6. COMPUTE microstructure alignment metrics:

   6.1 Attention-OFI Correlation:
   - FOR each sample:
     - avg_attention = mean(attention_weights, axis=heads)
     - ofi_sequence = X_test[sample, :, 0]
     - corr = pearson_correlation(avg_attention[-1, :], abs(ofi_sequence))
   - attention_ofi_corr = mean(correlations)

   6.2 Regime Analysis:
   - high_vol_idx = where(RV > median(RV))
   - low_vol_idx = where(RV <= median(RV))
   - acc_high_vol = accuracy(preds[high_vol_idx], labels[high_vol_idx])
   - acc_low_vol = accuracy(preds[low_vol_idx], labels[low_vol_idx])
   - regime_gap = acc_high_vol - acc_low_vol

7. STATISTICAL SIGNIFICANCE TESTS:
   - bootstrap_accuracies = []
   - FOR i in range(1000):
     - sample_idx = random_sample_with_replacement(N)
     - boot_acc = accuracy(preds[sample_idx], labels[sample_idx])
     - bootstrap_accuracies.append(boot_acc)
   - confidence_interval = percentile(bootstrap_accuracies, [2.5, 97.5])

   - baseline_accuracy = 1/3  # Random guessing for 3 classes
   - p_value = proportion_test(accuracy, baseline_accuracy, N)

8. COMPILE results:
   - results = {
       'classification': {accuracy, precision, recall, f1, kappa, confusion_matrix},
       'financial': {sharpe, max_dd, hit_rate, cumulative_pnl},
       'microstructure': {attention_ofi_corr, regime_gap},
       'statistical': {confidence_interval, p_value}
     }

9. OUTPUT: results, predictions, attention_weights

END PROCEDURE
```

### 8.5 Hypothesis Testing Procedure

```
PROCEDURE TestHypotheses(results, baseline_results, config):

1. INPUT:
   - results: evaluation results for TOFM variants
   - baseline_results: results for baseline models
   - config: significance thresholds

2. TEST Hypothesis H1 (Microstructure Features):
   - acc_micro = results['TOFM_microstructure']['accuracy']
   - acc_raw = baseline_results['TOFM_raw_LOB']['accuracy']
   - delta = acc_micro - acc_raw
   - se = sqrt(acc_micro*(1-acc_micro)/N + acc_raw*(1-acc_raw)/N)
   - z_stat = delta / se
   - p_value = 1 - normal_cdf(z_stat)

   - H1_supported = (delta >= 0.02) AND (p_value < 0.05)
   - LOG("H1: delta = {}, p = {}, supported = {}".format(delta, p_value, H1_supported))

3. TEST Hypothesis H2 (Attention Interpretability):
   - attention_corr = results['TOFM']['microstructure']['attention_ofi_corr']
   - t_stat = attention_corr * sqrt(N-2) / sqrt(1 - attention_corr^2)
   - p_value = 2 * (1 - t_cdf(abs(t_stat), df=N-2))

   - H2_supported = (attention_corr > 0.3) AND (p_value < 0.05)
   - LOG("H2: corr = {}, p = {}, supported = {}".format(attention_corr, p_value, H2_supported))

4. TEST Hypothesis H3 (Regime Dependence):
   - acc_high = results['TOFM']['acc_high_volatility']
   - acc_low = results['TOFM']['acc_low_volatility']
   - delta = acc_high - acc_low
   - (Perform paired t-test or McNemar test)

   - H3_supported = (delta >= 0.03) AND (p_value < 0.05)
   - LOG("H3: delta = {}, supported = {}".format(delta, H3_supported))

5. TEST Hypothesis H4 (Microstructure Bias):
   - acc_with_bias = results['TOFM_with_bias']['accuracy']
   - acc_standard = results['TOFM_standard']['accuracy']
   - delta = acc_with_bias - acc_standard
   - (Perform significance test)

   - H4_supported = (delta >= 0.01) AND (p_value < 0.05)
   - LOG("H4: delta = {}, supported = {}".format(delta, H4_supported))

6. TEST Hypothesis H5 (Cross-Asset Transfer):
   - FOR each held_out_asset:
     - acc_multi = results['TOFM_multi'][asset]['accuracy']
     - acc_single = results['TOFM_single'][asset]['accuracy']
     - record delta
   - avg_delta = mean(deltas)
   - (Perform paired t-test across assets)

   - H5_supported = (avg_delta > 0) AND (p_value < 0.05)
   - LOG("H5: avg_delta = {}, supported = {}".format(avg_delta, H5_supported))

7. COMPILE hypothesis testing report:
   - report = {
       'H1': {supported, delta, p_value, effect_size},
       'H2': {supported, correlation, p_value},
       'H3': {supported, regime_gap, p_value},
       'H4': {supported, delta, p_value},
       'H5': {supported, avg_delta, p_value}
     }

8. OUTPUT: hypothesis_report

END PROCEDURE
```

### 8.6 Ablation Study Design

```
PROCEDURE AblationStudy(base_config, data):

1. DEFINE ablation variants:
   - variants = [
       ('full_model', base_config),
       ('no_OFI', remove_feature(base_config, 'OFI')),
       ('no_lambda', remove_feature(base_config, 'lambda')),
       ('no_VOI', remove_feature(base_config, 'VOI')),
       ('no_micro_bias', set(base_config, 'use_micro_bias', False)),
       ('no_aux_loss', set(base_config, 'alpha_aux', 0)),
       ('shallow_1layer', set(base_config, 'n_layers', 1)),
       ('deep_8layer', set(base_config, 'n_layers', 8)),
       ('small_T50', set(base_config, 'T', 50)),
       ('large_T200', set(base_config, 'T', 200)),
       ('LSTM_baseline', switch_architecture('LSTM')),
       ('MLP_baseline', switch_architecture('MLP'))
     ]

2. FOR each (variant_name, variant_config) in variants:

   2.1 BUILD model with variant_config
   2.2 TRAIN model on training data
   2.3 EVALUATE model on test data
   2.4 STORE results[variant_name] = evaluation_results

3. COMPUTE relative importance:
   - base_accuracy = results['full_model']['accuracy']
   - FOR each variant:
     - importance[variant] = base_accuracy - results[variant]['accuracy']
   - SORT by importance (descending)

4. STATISTICAL COMPARISON:
   - FOR each variant vs full_model:
     - Perform McNemar test on prediction disagreements
     - Record p-value and effect size

5. OUTPUT: ablation_results, feature_importance_ranking

END PROCEDURE
```

---

## 9. Implementation Parameters

### 9.1 Recommended Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 128 | Balance between capacity and overfitting |
| n_heads | 8 | d_model / n_heads = 16 (standard) |
| n_layers | 4 | Sufficient depth for temporal patterns |
| d_ff | 512 | 4 * d_model (standard ratio) |
| T (sequence length) | 100 | Captures ~1-5 minutes at tick level |
| L (LOB levels) | 10 | Standard in literature |
| H (horizon) | 10 | 10 ticks forward prediction |
| batch_size | 256 | Memory-efficient, stable gradients |
| learning_rate | 1e-4 | Standard for transformers |
| dropout | 0.1 | Regularization |
| weight_decay | 1e-5 | L2 regularization |

### 9.2 Data Requirements

| Requirement | Specification |
|-------------|---------------|
| Minimum training samples | 1,000,000 ticks |
| Minimum assets | 5 (for multi-asset experiments) |
| Time span | >= 1 year (for regime diversity) |
| Data format | LOBSTER, TAQ, or equivalent |
| Frequency | Tick-by-tick or 100ms snapshots |

### 9.3 Computational Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU Memory | 8 GB | 16+ GB |
| Training Time | ~4 hours | ~8 hours (full hyperparameter search) |
| Inference Latency | < 10 ms | < 1 ms (for live trading) |

---

## 10. Confirmation and Falsification Criteria Summary

| Hypothesis | Confirmation | Falsification |
|------------|--------------|---------------|
| H1 | delta >= 2%, p < 0.05 | delta < 2% or p >= 0.05 |
| H2 | corr > 0.3, p < 0.05 | corr <= 0.3 or p >= 0.05 |
| H3 | regime_gap >= 3%, p < 0.05 | regime_gap < 3% or p >= 0.05 |
| H4 | delta >= 1%, p < 0.05 | delta < 1% or p >= 0.05 |
| H5 | avg_delta > 0, p < 0.05 | avg_delta <= 0 or p >= 0.05 |

---

## 11. Expected Outcomes and Contingencies

### 11.1 If Hypotheses Are Confirmed

- The TOFM framework provides a principled approach to combining microstructure theory with deep learning
- Microstructure features provide interpretable, economically meaningful signal
- The model can be extended to other assets and markets

### 11.2 If Hypotheses Are Falsified

- **H1 Falsified:** Raw LOB data may contain sufficient information; feature engineering provides marginal benefit
- **H2 Falsified:** Transformer attention may not align with human-interpretable microstructure; consider attention-free architectures
- **H3 Falsified:** Information asymmetry theory may not translate to predictive advantage; reconsider theoretical foundations
- **H4 Falsified:** Inductive biases may not help; rely on data-driven learning alone
- **H5 Falsified:** Market microstructure may be highly asset-specific; focus on single-asset models

---

## 12. References (Theoretical Foundations)

1. Kyle, A.S. (1985). Continuous Auctions and Insider Trading. *Econometrica*.
2. Glosten, L.R. & Milgrom, P.R. (1985). Bid, Ask and Transaction Prices. *Journal of Financial Economics*.
3. Hasbrouck, J. (1991). Measuring the Information Content of Stock Trades. *Journal of Finance*.
4. Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events. *Journal of Financial Econometrics*.
5. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
6. Zhang, Z. et al. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing*.

---

*Document Version: 1.0*
*Framework: Transformer-Based Order Flow Microstructure Model (TOFM)*
*Status: Ready for Implementation*
