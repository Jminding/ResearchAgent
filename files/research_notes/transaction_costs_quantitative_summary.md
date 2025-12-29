# Transaction Costs & Momentum Strategies: Quantitative Summary & Data Tables

## Quick Reference: Key Quantitative Findings

This document provides a concise tabular summary of quantitative results from the literature on transaction costs' impact on momentum strategies.

---

## 1. Gross Momentum Returns (Pre-Cost Baseline)

| **Study** | **Sample Period** | **Formation Period** | **Holding Period** | **Gross Return** | **Frequency** | **Notes** |
|-----------|------------------|----------------------|-------------------|-----------------|--------------|----------|
| Jegadeesh & Titman (1993) | 1965–1989 | 6 months | 6 months | 1.0% / month | Ongoing | Seminal work; equal-weighted |
| Jegadeesh & Titman (1993) | 1965–1989 | 3 months | 12 months | 1.0% / month | Ongoing | Best-performing window |
| Jegadeesh & Titman (1993) | 1965–1989 | 12 months | 12 months | 1.0% / month | Annual | 12-month momentum variant |

---

## 2. Transaction Cost Components: Bid-Ask Spreads

| **Stock Characteristic** | **Typical Spread (bps)** | **Variation Range** | **Drivers** | **Source** |
|-------------------------|------------------------|--------------------|-----------|-----------|
| Large-cap, high volume | 1–5 bps | 0.5–10 bps | High liquidity | Market data (2020+) |
| Mid-cap, moderate volume | 5–20 bps | 2–50 bps | Moderate liquidity | Typical |
| Small-cap, low volume | 20–100+ bps | 10–500 bps | Low liquidity | Momentum stock characteristic |
| Momentum winners (avg.) | 10–30 bps | 5–50 bps | Mix of sizes | Korajczyk & Sadka (2004) |
| Momentum losers (avg.) | 10–40 bps | 5–100 bps | Often smaller/less liquid | Lesmond et al. (2004) |

**Key Insight:** Bid-ask spreads are **symmetric** between winners and losers but **asymmetric in frequency** (losers traded more often in momentum rebalancing) and **absolute dollar impact** (position sizes matter).

---

## 3. Transaction Cost Components: Market Impact

### 3.1 Impact Scaling with Order Size

| **Order Size (% of ADV)** | **Estimated Impact (bps)** | **Scaling Model** | **Sources** |
|--------------------------|---------------------------|------------------|-----------|
| 0.5% | 3–7 bps | Power law: ~α × (OS/ADV)^0.6 | Korajczyk & Sadka (2004) |
| 1.0% | 5–10 bps | β ≈ 0.5–0.7 | Multiple sources |
| 2.0% | 10–20 bps | Nonlinear / convex | Standard assumption |
| 5.0% | 20–40 bps | Higher exponent regime | Kearns et al. (market impact) |
| 10.0% | 50–100 bps | Steep increase | Korajczyk & Sadka (2004) |
| 20.0% | 150–300 bps | Severely nonlinear | Extreme order size; rare |

**Formula (approximation):**
```
Impact (bps) ≈ 10 × (OrderSize / ADV) ^ 0.65
```

### 3.2 Impact Components (Breakdown)

| **Component** | **Typical Magnitude** | **Persistence** | **Driver** |
|---------------|----------------------|-----------------|-----------|
| **Temporary (bid-ask related)** | 30–50% of total | Immediate to minutes | Market maker compensation |
| **Permanent (information-based)** | 50–70% of total | Hours to days | Price discovery; informed trading |

**Note:** Temporary impact can sometimes be recovered if order is cancelled; permanent impact is irrecoverable.

---

## 4. Total Round-Trip Transaction Costs: By Strategy Type

| **Strategy Type** | **Formation Period** | **Holding Period** | **Turnover (annual)** | **Avg. Trade Cost (bps)** | **Cost per Rebalance** | **Source** |
|------------------|----------------------|-------------------|----------------------|------------------------|-----------------------|-----------|
| **Equal-Weighted Momentum** | 3 months | 3 months | 150–200% | 50–100 bps | 0.75–1.25% | Korajczyk & Sadka (2004) |
| **Value-Weighted Momentum** | 3 months | 3 months | 100–130% | 40–70 bps | 0.50–0.90% | Korajczyk & Sadka (2004) |
| **Liquidity-Weighted Momentum** | 3 months | 3 months | 50–80% | 25–50 bps | 0.30–0.60% | Korajczyk & Sadka (2004) |
| **Momentum (6-month hold)** | 6 months | 6 months | 70–100% | 35–65 bps | 0.40–0.70% | Implied from turnover |
| **Momentum (12-month hold)** | 12 months | 12 months | 40–60% | 25–45 bps | 0.25–0.50% | Novy-Marx & Velikov (2016) |

---

## 5. Fund Size and Scalability: Break-Even Points

| **Strategy** | **Break-Even Fund Size** | **Market Baseline** | **Estimated Dollar Return (at break-even)** | **Source** |
|--------------|------------------------|--------------------|-------------------------------------------|-----------|
| **Equal-Weighted Momentum** | ~$300–500M | Dec 1999 market cap | ~Zero alpha | Korajczyk & Sadka (2004) |
| **Value-Weighted Momentum** | ~$2–3B | Dec 1999 market cap | ~Zero alpha | Korajczyk & Sadka (2004) |
| **Liquidity-Weighted Momentum** | ~$5B+ | Dec 1999 market cap | ~Zero alpha | Korajczyk & Sadka (2004) |
| **Momentum (liquidity-adjusted, modern)** | ~$10–15B | 2015–2020 market cap | ~Zero alpha (estimated) | Extrapolated from Novy-Marx & Velikov (2016) |

**Interpretation:**
- At break-even size, market impact costs consume all alpha
- Larger fund → higher market impact → lower net returns
- Equal-weighted scales worst; liquidity-weighted best

**Updated 2020s Estimate:** Market capacity increases ~2–3× due to:
- Improved algorithmic execution
- Higher average daily volumes
- More diverse trading venues

---

## 6. Annual Cost Burden: Fund Size vs. Market Impact

| **Fund Size (AUM)** | **Annual Market Impact Cost (bps)** | **Strategy Type** | **Implied Gross Return Needed** | **Source** |
|--------------------|------------------------------------|------------------|-------------------------------|-----------|
| $500M | 50–100 bps | Liquidity-weighted | 1.0%+ / month | Implied |
| $1B | 100–150 bps | Liquidity-weighted | 1.2%+ / month | Korajczyk & Sadka (2004) |
| $5B | 200–300 bps | Liquidity-weighted | 1.5%+ / month | Korajczyk & Sadka (2004) |
| $10B | 200–270 bps | Standard momentum | 2.0%+ / month | Novy-Marx & Velikov (2016) |
| $10B | 270+ bps | Risk-adjusted Sharpe momentum | 2.5%+ / month | Novy-Marx & Velikov (2016) |

---

## 7. Turnover Thresholds and Profitability Survival

| **Monthly Turnover (One-Sided)** | **Annual Turnover** | **Survives Transaction Costs?** | **Typical Execution Cost** | **Notes** |
|----------------------------------|-------------------|--------------------------------|--------------------------|----------|
| 0–10% | 0–120% | **Yes (easily)** | < 20 bps | Very low turnover |
| 10–25% | 120–300% | **Yes (usually)** | 20–40 bps | Medium turnover |
| 25–50% | 300–600% | **Yes (marginal)** | 40–70 bps | Higher turnover |
| 50–100% | 600–1200% | **No (rarely)** | 70–150 bps | Novy-Marx & Velikov threshold |
| 100%+ | 1200%+ | **No** | 150+ bps | Impractical for most |

**Key Finding (Novy-Marx & Velikov 2016):** Strategies with **<50% monthly turnover** survive; **>50%** typically do not.

---

## 8. Rebalancing Frequency: Cost Impact Analysis

### 8.1 Calendar-Based Rebalancing (Stock/Bond Portfolio)

| **Rebalancing Frequency** | **Typical Annual Cost (per 0.5% spread)** | **Annual Cost Drag (bps)** | **Pros/Cons** |
|--------------------------|----------------------------------------|--------------------------|--------------|
| **Monthly** | 0.5% × 12 = 6% | 600 bps | Cons: Very high cost drag; Pros: Low tracking error |
| **Quarterly** | 0.5% × 4 = 2% | 200 bps | Cons: High drag; Pros: Reasonable tracking error |
| **Semi-annual** | 0.5% × 2 = 1% | 100 bps | Cons: Moderate drag; Pros: Lower cost |
| **Annual** | 0.5% × 1 = 0.5% | 50 bps | Cons: Higher drift; Pros: Low cost |

**Data source:** Vanguard (2022)

### 8.2 Threshold-Based Rebalancing

| **Rebalancing Trigger (Drift)** | **Average Rebalances/Year** | **Annual Cost Drag (bps)** | **Avg. Tracking Error** | **Recommendation** |
|--------------------------------|---------------------------|--------------------------|----------------------|------------------|
| **2% threshold** | 12–15 | 100–150 | < 0.5% | Tight; higher cost |
| **5% threshold** | 4–6 | 30–50 | 0.5–1.0% | **OPTIMAL (Vanguard finding)** |
| **10% threshold** | 2–3 | 15–25 | 1.0–2.0% | Loose; higher drift risk |
| **15% threshold** | 1–2 | 10–15 | 2.0%+ | Very loose; not recommended |

**Key Takeaway (Vanguard 2022):** 5% threshold balances cost efficiency with risk management; near-optimal for most investors.

---

## 9. Profitability Scenarios: Net Returns After Costs

### 9.1 Equal-Weighted Momentum (3-month formation/3-month hold)

| **Gross Return (pre-cost)** | **Transaction Cost (round-trip)** | **Net Return** | **Profitability** | **Feasibility** |
|---------------------------|----------------------------------|----------------|-------------------|-----------------|
| 1.0% / month | 1.0% / rebalance | ~0% / month | **Break-even** | Marginal |
| 1.0% / month | 0.75% / rebalance | ~0.25% / month | Minimal | Difficult |
| 1.2% / month | 1.0% / rebalance | ~0.2% / month | Minimal | Difficult |

**Conclusion:** Equal-weighted, high-turnover momentum generally **not viable**.

### 9.2 Liquidity-Weighted Momentum (6-month formation/6-month hold)

| **Gross Return (pre-cost)** | **Transaction Cost (round-trip)** | **Net Return** | **Profitability** | **Fund Size Constraint** |
|---------------------------|----------------------------------|----------------|-------------------|------------------------|
| 0.8% / month | 0.35% / rebalance (2×/yr) | ~0.65% / month | **Healthy** | < $2B (easily) |
| 0.8% / month | 0.50% / rebalance | ~0.60% / month | **Healthy** | < $5B (challenging) |
| 0.8% / month | 0.75% / rebalance | ~0.50% / month | Acceptable | > $5B (difficult) |

**Conclusion:** Liquidity-weighted, moderate-turnover momentum **viable up to ~$5B**.

### 9.3 Momentum (12-month formation/12-month hold)

| **Gross Return (pre-cost)** | **Annual Turnover** | **Transaction Cost (annual)** | **Net Annual Return** | **Profitability** |
|---------------------------|-------------------|------------------------------|-----------------------|-------------------|
| 9.6% (0.8% / month) | 40–60% | 60–100 bps | 8.6–9.4% | **Robust** |
| 9.6% (0.8% / month) | 50–80% | 80–120 bps | 8.4–8.8% | **Robust** |
| 9.6% (0.8% / month) | 100%+ | 150–200 bps | 7.6–8.1% | **Still viable** |

**Conclusion:** Long-horizon momentum (12-month+) **survives costs well** and scales to larger fund sizes.

---

## 10. Cost Mitigation Techniques: Effectiveness

| **Technique** | **Cost Reduction vs. Baseline** | **Implementation Difficulty** | **Trade-offs** | **Reference** |
|---------------|--------------------------------|-------------------------------|----------------|--------------|
| **Liquidity Weighting** | 30–50% cost reduction | Medium | Slight return reduction; better Sharpe | Korajczyk & Sadka (2004) |
| **Buy/Hold Spread** | 20–40% cost reduction | Low | Tracking error vs. cost trade-off | Novy-Marx & Velikov (2016) |
| **Longer Holding Period** | 40–60% cost reduction | Low | Lower gross return (but more than offset) | Implied from turnover |
| **Smart Order Routing** | 10–30% cost reduction | Medium | Execution complexity | Algorithmic trading literature |
| **TWAP/VWAP Execution** | 5–20% cost reduction | Low-Medium | Slower execution; market impact risk | Market microstructure |
| **Threshold-based Rebalancing** | 20–50% cost reduction | Low | Higher portfolio drift | Vanguard (2022) |

**Overall Ranking (effectiveness × simplicity):**
1. **Liquidity weighting** - Highest impact, relatively simple
2. **Longer holding periods** - Highest impact, simplest (design choice)
3. **Threshold-based rebalancing** - Good impact, very simple
4. **Buy/hold spreads** - Moderate impact, simple

---

## 11. Market Impact Costs by Stock Liquidity Decile

| **Liquidity Decile** | **Avg. Daily Volume** | **Estimated Impact (5% order)** | **Characteristic Stocks** |
|--------------------|-----------------------|--------------------------------|--------------------------|
| **Top 10% (most liquid)** | $500M+ daily | 5–10 bps | Large-cap index constituents |
| **Decile 2–3** | $100–500M daily | 10–20 bps | Large-cap growth/value |
| **Decile 4–5 (median)** | $20–100M daily | 20–40 bps | Mid-cap core holdings |
| **Decile 6–7** | $5–20M daily | 40–80 bps | Mid-cap / small-cap |
| **Decile 8–9** | $1–5M daily | 80–150 bps | Small-cap; momentum common |
| **Bottom 10% (least liquid)** | <$1M daily | 150–500+ bps | Micro-cap; very illiquid |

**Note:** Momentum portfolios are skewed toward **lower-liquidity deciles** (8–10), making average costs higher than broad market.

---

## 12. Statistical Significance After Costs

| **Metric** | **Before Costs** | **After Realistic Costs** | **Change** | **T-stat Impact** |
|-----------|----------------|-----------------------|-----------|------------------|
| Momentum return (gross) | 1.0% / month | 0.6–0.8% / month | -30–40% | t ≈ 3–4 |
| Excess return (after risk) | 0.6% / month | 0.3–0.5% / month | -40–50% | t ≈ 2–3 |
| Annualized Sharpe (gross) | 0.7–0.9 | 0.4–0.6 | -35–45% | Materially lower |

**Key Finding:** Transaction costs reduce t-statistics by ~30–50%, pushing strategies that border statistical significance into marginal territory. **Harvey et al. (2015) t-stat threshold of 3.0** (vs. traditional 2.0) makes costs even more critical.

---

## 13. Geographic Variation in Momentum Costs

| **Market** | **Avg. Bid-Ask Spread** | **Market Impact Intensity** | **Estimated Total Cost (round-trip)** | **Momentum Profitability** |
|-----------|----------------------|----------------------------|-----------------------------------------|--------------------------|
| **U.S.A** | 2–20 bps | Low-moderate | 30–100 bps | **Robust** |
| **U.K.** | 5–25 bps | Low-moderate | 40–120 bps | **Robust** (studies confirm) |
| **Developed Europe** | 5–30 bps | Moderate | 50–150 bps | **Viable** |
| **Japan** | 3–15 bps | Low | 25–80 bps | **Strong** (high liquidity) |
| **Emerging Markets** | 20–100+ bps | High | 100–300+ bps | **Questionable** |

**Note:** Limited empirical research on emerging markets; costs likely higher; momentum profitability less certain.

---

## 14. High-Frequency Rebalancing: Cost vs. Drift Trade-off

| **Rebalancing Period** | **Annual # Events** | **Tracking Error to Target** | **Transaction Cost per Event** | **Total Annual Cost** | **Optimal Regime** |
|----------------------|-------------------|-------------------------------|------------------------------|-----------------------|------------------|
| **Daily** | 252 | <0.1% | 50–75 bps | 20–30% of capital | Never (impractical) |
| **Weekly** | 52 | 0.5–1% | 50–75 bps | 2–4% of capital | Only HFT |
| **Monthly** | 12 | 1–2% | 50–75 bps | 600–900 bps | Not recommended |
| **Quarterly** | 4 | 2–3% | 50–75 bps | 200–300 bps | Viable for large funds |
| **Annual** | 1 | 3–5% | 50–75 bps | 50–75 bps | **Optimal for most** |
| **Threshold (5% drift)** | 4–6 | 1–2% | 50–75 bps | 30–50 bps | **Optimal (Vanguard)** |

---

## 15. Breakeven Analysis: Momentum Fund Profitability

### Scenario: Liquidity-Weighted Momentum, 6-Month Hold, Different Fund Sizes

| **Fund AUM** | **Estimated Annual Market Impact** | **Gross Return Needed** | **With 0.8% / month expected** | **Net Profitability** |
|-----|---------|--------|---------|------------|
| **$100M** | 20–40 bps | 0.25% / month | 0.55–0.75% / month | ✓ Profitable |
| **$500M** | 50–100 bps | 0.50% / month | 0.30–0.75% / month | ✓ Marginally profitable |
| **$1B** | 100–150 bps | 0.75–1.0% / month | 0.05–0.75% / month | ✓ Marginal; size-dependent |
| **$2B** | 150–200 bps | 1.0–1.25% / month | -0.25–0.75% / month | ✗ Breakeven to unprofitable |
| **$5B** | 200–300 bps | 1.25–1.75% / month | -0.45–0.75% / month | ✗ Likely unprofitable |
| **$10B** | 300–400+ bps | 1.75–2.25% / month | -0.95–0.75% / month | ✗ Unprofitable |

**Note:** Assumes 0.8% / month gross momentum return; actual returns vary by period and stock selection.

---

## Key Takeaways for Practitioners

1. **Transaction costs are nonlinear in fund size**: Doubling AUM can more than double costs
2. **Turnover management is critical**: >50% monthly turnover rarely survives costs
3. **Longer holding periods are powerful cost reducers**: 12-month hold ≈ 50% cost reduction vs. 3-month
4. **Liquidity weighting beats equal weighting by 30–50% in net returns**
5. **Threshold-based rebalancing (5% drift) often superior to calendar-based** and reduces costs by 20–50%
6. **Fund size capacity limits are real**: Aggregate momentum AUM facing systematic capacity constraints

---

## Sources for Quantitative Data

- Korajczyk & Sadka (2004): Break-even fund sizes, market impact scaling
- Novy-Marx & Velikov (2016): Turnover thresholds, annual market impact costs at fund scale
- Vanguard (2022): Rebalancing frequency, threshold optimization
- Lesmond et al. (2004): Bid-ask spreads for momentum stocks
- Market microstructure literature: Impact modeling and slippage estimation

---

**Document updated:** December 23, 2024
**Data coverage:** 1993–2025
**Quantitative metrics:** ~80+ specific data points

