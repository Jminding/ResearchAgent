# Unified Mathematical Framework: GNNs for Financial Anomaly Detection

## 1. Problem Formulation

### 1.1 Graph Construction

Let G = (V, E, X, W, T) be a dynamic attributed financial graph where:

- **V** = {v_1, v_2, ..., v_n}: Set of n account nodes
- **E** subset of V x V: Set of directed edges representing transactions
- **X** in R^{n x d}: Node feature matrix where x_i in R^d represents features of account v_i
- **W**: E -> R^+: Edge weight function mapping transactions to amounts
- **T**: E -> R: Timestamp function assigning temporal ordering to edges

**Node Features (x_i):**
- Account age, balance statistics, transaction frequency
- Behavioral features: login patterns, device fingerprints
- Derived features: velocity metrics, deviation from historical patterns

**Edge Attributes:**
For edge e_{ij} from v_i to v_j:
- w_{ij}: Transaction amount
- t_{ij}: Timestamp
- c_{ij}: Transaction category/type
- Optional: memo embeddings, location data

### 1.2 Adjacency Matrix Representation

Define the weighted adjacency matrix A in R^{n x n}:

```
A_{ij} = sum over e in E_{ij} of f(w_e, t_e)
```

where E_{ij} is the set of edges from v_i to v_j, and f is an aggregation function (e.g., sum, count, recency-weighted sum).

**Normalized Adjacency:**
```
A_hat = D^{-1/2} (A + I_n) D^{-1/2}
```
where D is the degree matrix and I_n is the identity matrix.

### 1.3 Classification Objective

Let Y = {0, 1}^n be the binary label vector where:
- y_i = 1: Account v_i is anomalous (fraudulent)
- y_i = 0: Account v_i is normal (legitimate)

**Objective:** Learn a function f_theta: G -> [0,1]^n that minimizes:

```
L(theta) = L_cls(f_theta(G), Y) + lambda_1 * R(theta) + lambda_2 * L_struct(G)
```

where:
- L_cls: Classification loss (cross-entropy with class weighting)
- R(theta): Regularization term
- L_struct: Structural consistency loss
- lambda_1, lambda_2: Hyperparameters

---

## 2. Theoretical Foundations from GNN Literature

### 2.1 Message Passing Neural Networks (MPNN) Framework

General form of layer l message passing:

```
m_i^{(l)} = AGGREGATE({h_j^{(l-1)} : j in N(i)})
h_i^{(l)} = UPDATE(h_i^{(l-1)}, m_i^{(l)})
```

**GCN (Kipf & Welling, 2017):**
```
H^{(l)} = sigma(A_hat H^{(l-1)} W^{(l)})
```

**GraphSAGE (Hamilton et al., 2017):**
```
h_i^{(l)} = sigma(W^{(l)} * CONCAT(h_i^{(l-1)}, AGG({h_j^{(l-1)} : j in N(i)})))
```

**GAT (Velickovic et al., 2018):**
```
h_i^{(l)} = sigma(sum over j in N(i) of alpha_{ij} W h_j^{(l-1)})
```
where alpha_{ij} are learned attention coefficients.

### 2.2 Homophily vs. Heterophily in Graphs

**Definition (Node Homophily Ratio):**
```
h(G) = (1/|E|) * sum over (i,j) in E of 1[y_i = y_j]
```

**Definition (Edge Homophily):**
```
h_edge = |{(i,j) in E : y_i = y_j}| / |E|
```

**Key Insight from Literature:**
- Standard GNNs (GCN, GraphSAGE) perform well when h(G) > 0.7 (homophilic)
- Performance degrades significantly when h(G) < 0.3 (heterophilic)
- Financial fraud networks exhibit h(G) in [0.1, 0.4] (strongly heterophilic)

### 2.3 Heterophily-Aware GNN Architectures

**H2GCN (Zhu et al., 2020):**
- Separates ego and neighbor embeddings
- Uses higher-order neighborhoods
- Combines intermediate representations

```
h_i^{final} = COMBINE(h_i^{(0)}, h_i^{(1)}, ..., h_i^{(K)}, h_{N(i)}^{(1)}, ..., h_{N(i)}^{(K)})
```

**FAGCN (Bo et al., 2021):**
- Learns edge-wise aggregation coefficients in [-1, 1]
- Allows negative correlation modeling

```
h_i^{(l)} = epsilon * h_i^{(l-1)} + sum over j in N(i) of alpha_{ij}^{(l)} h_j^{(l-1)}
```
where alpha_{ij} in [-1, 1].

**GPR-GNN (Chien et al., 2021):**
- Generalized PageRank with learnable weights
- Adaptive polynomial filters

```
Z = sum_{k=0}^{K} gamma_k * (A_hat)^k * H
```
where gamma_k are learnable coefficients.

**LINKX (Lim et al., 2021):**
- Decouples feature and structure processing
- MLP on concatenated representations

```
h_i = MLP(CONCAT(MLP_X(x_i), MLP_A(A^K x_i)))
```

### 2.4 Theoretical Analysis: Why Standard GNNs Fail on Heterophilic Graphs

**Theorem (Informal):** Under k-layer message passing with averaging aggregation, node representations converge to:

```
H^{(k)} approx (A_hat)^k X W_1 W_2 ... W_k
```

For heterophilic graphs where connected nodes have different labels, this averaging operation:
1. Mixes features of dissimilar nodes
2. Reduces class separability in embedding space
3. Creates indistinguishable representations for different classes

**Spectral Perspective:**
- Low-frequency signals (smooth over graph): Captured by standard GNNs
- High-frequency signals (vary rapidly): Contain class-discriminative information in heterophilic graphs
- Standard GNNs act as low-pass filters, discarding high-frequency information

---

## 3. Financial Network Characteristics

### 3.1 Structural Properties

**Property 1 (Heterophily):** Financial fraud networks exhibit heterophily:
- Fraudulent accounts transact with legitimate accounts (victims)
- Estimated homophily ratio: h(G) in [0.15, 0.40]

**Property 2 (Scale-Free Degree Distribution):**
```
P(k) proportional to k^{-gamma}, gamma in [2, 3]
```
Few hub accounts, many peripheral accounts.

**Property 3 (Temporal Burstiness):**
Fraudulent activity concentrates in short time windows:
```
B = (sigma_tau - mu_tau) / (sigma_tau + mu_tau)
```
where tau is inter-event time. Fraud exhibits B > 0.6.

**Property 4 (Community Structure):**
- Legitimate accounts form dense communities (family, business)
- Fraud rings form transient, sparse structures

### 3.2 Class Imbalance Characteristics

**Anomaly Prevalence:** p_fraud in [0.001, 0.02] (0.1% to 2%)

**Imbalance Ratio:** IR = n_normal / n_fraud in [50, 1000]

**Implications:**
- Standard cross-entropy loss biased toward majority class
- F1-score and AUPRC more informative than accuracy
- Need class-weighted losses or sampling strategies

---

## 4. Key Assumptions

### Assumption A1 (Graph Connectivity)
The financial transaction graph G is weakly connected, with giant component containing > 90% of nodes.

### Assumption A2 (Feature Informativeness)
Node features X contain signal for classification:
```
I(X; Y) > 0
```
where I denotes mutual information.

### Assumption A3 (Structural Signal)
Graph structure provides additional signal beyond features:
```
I(A; Y | X) > 0
```

### Assumption A4 (Heterophily Dominance)
Edge homophily ratio satisfies:
```
h_edge(G) < 0.5
```
implying more cross-class edges than same-class edges.

### Assumption A5 (Temporal Stationarity)
Within evaluation windows, the data generating process is approximately stationary:
```
P(Y | X, A, T in [t, t+Delta]) approx P(Y | X, A, T in [t', t'+Delta])
```
for reasonable Delta (e.g., 1 month).

### Assumption A6 (Anomaly Prevalence Bounds)
```
0.001 <= P(Y=1) <= 0.02
```

### Assumption A7 (Label Quality)
Training labels are noisy with false negative rate FNR < 0.3 and false positive rate FPR < 0.05.

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

**F1-Score (Fraud Class):**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Area Under Precision-Recall Curve (AUPRC):**
```
AUPRC = integral from 0 to 1 of P(r) dr
```

**Area Under ROC Curve (AUROC):**
```
AUROC = P(score(x_pos) > score(x_neg))
```

### 5.2 Secondary Metrics

**Precision at k% (P@k):**
Precision when top k% scored samples are classified as fraud.

**Value-Weighted Recall (VWR):**
```
VWR = (sum over i:y_i=1 and y_hat_i=1 of v_i) / (sum over i:y_i=1 of v_i)
```
where v_i is monetary value at risk.

### 5.3 Fairness Metrics (if applicable)

**Demographic Parity Difference:**
```
DPD = |P(Y_hat=1 | A=0) - P(Y_hat=1 | A=1)|
```

---

## 6. Experimental Variables

### 6.1 Independent Variables

**IV1: GNN Architecture Type**
- Homophily-assuming: GCN, GraphSAGE, GAT
- Heterophily-aware: H2GCN, FAGCN, GPR-GNN, LINKX

**IV2: Homophily Level**
- Controlled via synthetic modification or dataset selection
- Levels: h in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

**IV3: Class Imbalance Ratio**
- IR in {50, 100, 200, 500, 1000}

**IV4: Temporal Split Strategy**
- Random split vs. temporal split

### 6.2 Dependent Variables

- F1-score (primary)
- AUPRC
- AUROC
- Precision@1%
- Training time
- Inference time

### 6.3 Control Variables

- Number of GNN layers: K = 2
- Hidden dimension: d_h = 64
- Learning rate: lr = 0.01
- Epochs: 200 with early stopping (patience=20)
- 5-fold cross-validation or 5 random seeds

---

## 7. Loss Functions for Class Imbalance

### 7.1 Weighted Cross-Entropy

```
L_WCE = -(1/n) * sum_i [w_1 * y_i * log(p_i) + w_0 * (1-y_i) * log(1-p_i)]
```

where w_1 = n / (2 * n_fraud), w_0 = n / (2 * n_normal).

### 7.2 Focal Loss

```
L_focal = -(1/n) * sum_i [(1-p_i)^gamma * y_i * log(p_i) + p_i^gamma * (1-y_i) * log(1-p_i)]
```

with gamma in {1, 2, 3}.

### 7.3 LDAM Loss (Label-Distribution-Aware Margin)

```
L_LDAM = -(1/n) * sum_i log(exp(z_{y_i} - Delta_{y_i}) / sum_j exp(z_j))
```

where Delta_j = C / n_j^{1/4} and C is a hyperparameter.

---

## 8. Computational Complexity Analysis

### 8.1 Time Complexity

| Model | Training (per epoch) | Inference |
|-------|---------------------|-----------|
| GCN | O(|E| * d + n * d^2) | O(|E| * d + n * d^2) |
| GAT | O(|E| * d^2) | O(|E| * d^2) |
| H2GCN | O(K * |E| * d + n * K * d^2) | O(K * |E| * d + n * K * d^2) |
| LINKX | O(|E| * d + n * d^2) | O(|E| * d + n * d^2) |

### 8.2 Space Complexity

| Model | Memory |
|-------|--------|
| GCN | O(n * d + |E|) |
| GAT | O(n * d + |E| * H) |
| H2GCN | O(n * K * d + |E|) |
| LINKX | O(n * d + |E|) |

where H is number of attention heads, K is number of hops.

---

## 9. References for Implementation

### 9.1 Required Libraries
- PyTorch Geometric (PyG) for GNN implementations
- DGL as alternative
- scikit-learn for evaluation metrics
- NetworkX for graph analysis

### 9.2 Datasets
- Elliptic (Bitcoin): ~200K nodes, ~234K edges, h approx 0.15
- IEEE-CIS Fraud Detection: Requires graph construction
- Synthetic: Controllable homophily via stochastic block model

---

## 10. Notation Summary

| Symbol | Description |
|--------|-------------|
| G = (V, E) | Graph with vertices V and edges E |
| n = |V| | Number of nodes |
| m = |E| | Number of edges |
| X in R^{n x d} | Node feature matrix |
| A in R^{n x n} | Adjacency matrix |
| Y in {0,1}^n | Label vector |
| h(G) | Homophily ratio |
| H^{(l)} | Node embeddings at layer l |
| theta | Model parameters |
| N(i) | Neighborhood of node i |
| d | Feature dimension |
| d_h | Hidden dimension |
| K | Number of GNN layers/hops |
