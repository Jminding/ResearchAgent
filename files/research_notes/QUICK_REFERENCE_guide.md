# Multi-Factor Momentum Strategies: Quick Reference Guide

## One-Page Strategy Overview

### Core Strategy Concept
Combine 4-6 return premia (momentum, value, quality, low-volatility, liquidity) into single portfolio to achieve 5-8% gross annual outperformance with improved risk-adjusted returns via diversification.

### Why It Works
- **Factor Correlation:** Value-momentum correlation = -0.49 (strong negative)
- **Diversification Benefit:** Multi-factor Sharpe ratio 0.60-0.80 vs. 0.30-0.50 single-factor
- **Regime Coverage:** Different factors excel in different economic environments
- **Risk Reduction:** Maximum drawdown -35% to -40% vs. -50% to -60% single-factor

---

## Quick Specification Reference

### Factor Definitions (Standard Approach)

| Factor | Score Calculation | Update | Rebalance | Solo Return |
|--------|---|---|---|---|
| **Momentum** | 12-month return (exclude month +1) | Monthly | Quarterly | +2.0% |
| **Value** | Composite: P/B, P/E, EV/EBITDA, P/FCF (z-score normalized) | Annual | Annual | +2.0% |
| **Quality** | Composite: Profitability (40%) + Investment (35%) + Stability (25%) | Annual | Annual | +1.5% |
| **Low Vol** | 12-month rolling volatility (or EWMA) | Monthly | Quarterly | +1.5% |
| **Liquidity** | Applied as screening filter (min liquidity threshold) | Annual | Annual | Baseline |

### Weighting Scheme (Recommended)

**Risk Parity (Inverse Volatility):**
```
Weight_i = (1 / Factor_Volatility_i) / Sum(1 / Factor_Volatility_j)
```

**Example with 4 factors (assuming volatilities: Mom=12%, Val=10%, Qual=8%, LowVol=9%):**
- Momentum: 22% (high volatility → lower weight)
- Value: 27% (medium volatility → medium weight)
- Quality: 28% (low volatility → higher weight)
- Low Volatility: 23%

**Update:** Monthly (volatility estimates); rebalance: Quarterly

### Portfolio Construction (Step-by-Step)

**Option A: Separate Factor Portfolios (Recommended)**
1. Construct each factor as long-short quintile spread (L5-S1 or L4-S2)
2. Weight each factor portfolio per risk parity scheme
3. Combine: Portfolio = 0.22×Mom + 0.27×Val + 0.28×Qual + 0.23×LowVol

**Option B: Integrated Composite Score (Simpler)**
1. Z-score normalize each factor score
2. Composite = 0.22×Mom_z + 0.27×Val_z + 0.28×Qual_z + 0.23×LowVol_z
3. Quintile sort by composite score
4. Portfolio = Long Q5 / Short Q1 (130/70 or market-neutral)

---

## Performance Expectations (Realistic)

### Gross Annual Returns (Before Costs)
- **Domestic Large Cap:** 5-7%
- **Domestic All Cap:** 6-8%
- **Global Developed:** 4-6%
- **Includes Emerging Markets:** 3-5%

### Risk-Adjusted Metrics
- **Sharpe Ratio:** 0.65-0.75 (vs. 0.25-0.35 for market)
- **Information Ratio:** 0.40-0.50
- **Sortino Ratio:** 1.5-1.8
- **Maximum Drawdown:** -35% to -40%
- **Volatility:** 12-15% (vs. 15-16% for market)

### After Transaction Costs
- **Typical Annual Costs:** 50-100 bps
- **Net Annual Return:** 4-6% (typical estimate)
- **Sharpe After Costs:** 0.50-0.60

---

## Decision Matrix: Quick Selection Guide

### How to Choose Your Factors

**Want Maximum Simplicity?**
→ Use 4 factors: Momentum + Value + Quality + Low Vol
→ Use equal weighting (25% each)
→ Rebalance annually

**Want Optimized Risk-Adjusted Returns?**
→ Use 4-6 factors as specified
→ Use risk parity weighting (inverse volatility)
→ Rebalance quarterly

**Want Downside Protection (Conservative Investor)?**
→ Emphasize Quality (35%) + Low Vol (35%)
→ Reduce Momentum (15%)
→ Use long-only or 110/10 structure

**Want Maximum Outperformance (Aggressive)?**
→ Equal weight all factors
→ Use 130/70 or market-neutral structure
→ Accept higher transaction costs (100+ bps)
→ Monitor crowding

**Want Tactical Allocation Capability?**
→ Implement factor momentum (rank factors by prior year return)
→ Monthly updates to factor weights
→ Overlay on base multi-factor portfolio

---

## Critical Parameter Checklist

### Must-Haves
- [ ] Use point-in-time (PIT) data (avoid look-ahead bias)
- [ ] Include transaction costs in backtests (assume 75-100 bps)
- [ ] Test out-of-sample, not just in-sample
- [ ] Use risk parity or inverse volatility weighting
- [ ] Quarterly rebalancing minimum
- [ ] Position limits (max 2-5% per position)

### Should-Haves
- [ ] 4-6 factors for diversification
- [ ] Staggered rebalancing (Q-Momentum, A-Fundamentals)
- [ ] Liquidity screening (top 80% of universe)
- [ ] Annual strategy review
- [ ] Drawdown limits (stop if >-50%)

### Nice-to-Haves
- [ ] Dynamic regime-based weighting (+5-10% improvement)
- [ ] Quality momentum acceleration signal
- [ ] Factor momentum overlay for tactical allocation
- [ ] Geographic diversification (international)
- [ ] Quarterly performance attribution

---

## Cost Breakdown (Per Year)

| Item | Typical Cost | High Estimate |
|------|---|---|
| Market impact | 15 bps | 30 bps |
| Bid-ask spreads | 10 bps | 20 bps |
| Commissions | 2 bps | 5 bps |
| Opportunity cost | 8 bps | 15 bps |
| Rebalancing frequency (quarterly) | 15 bps | 30 bps |
| **Total** | **50 bps** | **100 bps** |

**Impact on Returns:** 6.5% gross → 6.0% net (75 bps costs) → 5.0-6.0% after advisory fees

---

## Rebalancing Schedule (Recommended)

### Monthly
- [ ] Update momentum scores
- [ ] Update volatility estimates (for low-vol rebalancing)
- [ ] Monitor portfolio drift (alert if >2% from target)

### Quarterly (End of Month: Mar, Jun, Sep, Dec)
- [ ] Rebalance momentum factor
- [ ] Rebalance low-volatility factor
- [ ] Rebalance weights (if using dynamic allocation)
- [ ] Review factor attribution

### Annually (July)
- [ ] Update value scores (valuations)
- [ ] Update quality scores (fundamentals)
- [ ] Annual strategy review
- [ ] Update factor definitions (if needed)

---

## Performance Benchmarking

### How to Track Results

**Monthly Report:**
```
Portfolio Return:          +1.2%
Benchmark Return:          +0.8%
Active Return:             +0.4% (outperformance)
Factor Attribution:
  - Momentum:              +0.15%
  - Value:                 +0.10%
  - Quality:               +0.08%
  - Low Vol:               +0.07%
Expected Value Added:      +0.30% (consistent with 0.40 IR)
```

**Quarterly Report:**
```
Cumulative Return:         +3.2%
Cumulative Benchmark:      +1.9%
Sharpe Ratio YTD:          0.65
Maximum Drawdown:          -8% (acceptable)
Transaction Costs:         25 bps (on track)
Factor Weights:
  - Momentum:              22% (vs. 22% target) ✓
  - Value:                 27% (vs. 27% target) ✓
  - Quality:               28% (vs. 28% target) ✓
  - Low Vol:               23% (vs. 23% target) ✓
```

**Annual Report:**
```
Full Year Return:          +6.8%
Benchmark Return:          +0.5%
Outperformance:            +6.3% (gross)
Information Ratio:         0.43
Annual Costs:              75 bps
Net Outperformance:        +5.55%
Sharpe Ratio:              0.62
Factor Contributions:
  - Momentum:              +2.0%
  - Value:                 +2.1%
  - Quality:               +1.5%
  - Low Vol:               +1.2%
```

---

## Red Flags to Watch

### Immediate Action Required
- [ ] Maximum drawdown exceeds -50% (reassess strategy)
- [ ] Monthly costs exceed 50 bps (portfolio too large or illiquid)
- [ ] Factor weights drift >5% from target for >1 month (rebalance)
- [ ] Information ratio declines below 0.20 (factor crowding or degradation)

### Warning Signals (Monitor Closely)
- [ ] Correlations between factors increasing toward +1.0 (crisis regime)
- [ ] Drawdown from peak approaches -40% (increase quality/low-vol weights)
- [ ] Performance gap from benchmark widens unexpectedly (attribution analysis needed)
- [ ] Single factor outperforming by >5% (concentration risk)

### Long-Term Concerns
- [ ] Factor crowding metrics deteriorating (check fund inflows)
- [ ] Historical factor premia eroding (structural regime change?)
- [ ] Out-of-sample performance consistently below -40% of backtest
- [ ] Correlation to market approaching 1.0 (strategy losing independence)

---

## Common Questions Answered

**Q: How much capital can I manage with this strategy?**
A: Depends on universe. Typical: $5-50B for domestic large-cap. Liquidity constraints become binding above position limits; capacity grows with universe size.

**Q: What's the optimal rebalancing frequency?**
A: Quarterly is sweet spot (balances performance vs. costs). Monthly for momentum; annual for value. More frequent = higher costs; less frequent = more drift.

**Q: Should I use long-only or long-short?**
A: 130/70 optimal for most investors. Long-only reduces outperformance by ~2%; market-neutral increases costs by 30+ bps.

**Q: How much estimation error is in my backtest?**
A: Plan for 40-60% performance degradation from in-sample to out-of-sample. Conservative estimates: multiply backtest Sharpe by 0.6.

**Q: Is factor momentum worth it?**
A: Yes. Adds ~1-2% annually with ~50% increase in complexity and 20-40 bps additional costs. IR improves from 0.40→0.50. Consider overlay approach.

**Q: What about emerging markets?**
A: Factors work but with higher costs and volatility. Recommend 15-25% allocation to emerging markets if adding them.

**Q: How do I know if factors are crowded?**
A: Watch for: (1) factor return compression, (2) spreads widening, (3) fund inflows exceeding factor capacity. Typical crowding indicators: momentum showing extreme negative correlation or reversals.

**Q: Can I use machine learning for factor selection?**
A: Risky without careful validation. Avoid data mining. Better to use proven factors (Fama-French). If using ML, use walk-forward validation and hold-out test set.

**Q: What's the ideal number of factors?**
A: Start with 4-6 (momentum, value, quality, low-vol). Adding 7th+ factors introduces estimation risk without clear benefit. Stick to well-documented factors.

---

## Key Formulas Reference

### Momentum Score
```
Momentum = Return(t-1 to t-12) - Rf
Exclude most recent month (month +1 to 0)
```

### Value Score (Composite)
```
Value_z = 0.25*Z(P/B) + 0.25*Z(P/E) + 0.25*Z(EV/EBITDA) + 0.25*Z(P/FCF)
Where Z() = z-score normalization = (X - Mean(X)) / Std(X)
```

### Quality Score
```
Quality_z = 0.40*Profitability_z + 0.35*Investment_z + 0.25*Stability_z
```

### Risk Parity Weight
```
Weight_i = (1/sigma_i) / Sum(1/sigma_j)
sigma = volatility (standard deviation)
```

### Information Ratio
```
IR = (Return_strategy - Return_benchmark) / Std(Return_strategy - Return_benchmark)
Typical target: IR > 0.40
```

### Sharpe Ratio
```
Sharpe = (Return - Risk_free_rate) / Std(Return)
Typical target: Sharpe > 0.60
```

---

## Implementation Checklist (Fast Track)

**Week 1: Planning**
- [ ] Decide on factors (recommend: Mom, Value, Quality, LowVol)
- [ ] Choose weighting (recommend: Risk parity)
- [ ] Define universe (recommend: Top 1500 US stocks)
- [ ] Set position limits (recommend: 2-5% max position)

**Week 2-3: Data & Backtesting**
- [ ] Obtain 20+ years historical data
- [ ] Verify point-in-time (PIT) data quality
- [ ] Backtest with transaction costs (75-100 bps)
- [ ] Out-of-sample validation (walk-forward)

**Week 4: Implementation Setup**
- [ ] Build portfolio construction algorithm
- [ ] Set up monitoring dashboards
- [ ] Document all specifications
- [ ] Approval from investment committee

**Week 5+: Operational Launch**
- [ ] Start with pilot portfolio (smaller size)
- [ ] Monitor daily risk metrics
- [ ] Monthly performance review
- [ ] Annual strategy optimization

---

## Further Reading (Top 5 Must-Read Papers)

1. **Asness, Frazzini & Pedersen (2013). Value and Momentum Everywhere.**
   → Foundational work on factor correlations across asset classes

2. **Fama & French (2018). Choosing Factors.**
   → Academic authority on six-factor model including momentum

3. **Ehsani & Linnainmaa (2022). Factor Momentum and the Momentum Factor.**
   → Factor-level momentum strategy with empirical validation

4. **DeMiguel et al. (2021). What Alleviates Crowding in Factor Investing?**
   → Crowding effects and mitigation strategies

5. **Dynamic Factor Allocation using Regime-Switching (2024, arXiv).**
   → Latest research on adaptive factor weighting

All papers referenced in main literature review document with URLs.

---

## Summary: 10 Key Takeaways

1. **Multi-factor beats single-factor:** 0.65 vs. 0.40 Sharpe ratio
2. **Negative correlations matter:** Value-momentum = -0.49 drives benefits
3. **Risk parity outperforms equal weight:** 5-10% improvement
4. **Costs are real:** Assume 75-100 bps; momentum most expensive at 200+ bps
5. **Out-of-sample matters:** Plan for 40-60% degradation from backtest
6. **Quarterly rebalancing optimal:** Balances performance vs. costs
7. **4-6 factors is enough:** Avoid over-complication; use proven factors
8. **Position limits essential:** 2-5% max per position
9. **Diversification is free lunch:** Different factors excel in different regimes
10. **Monitoring is critical:** Daily risk tracking; monthly attribution; annual review

---

**For detailed specifications, full references, and implementation guidance, see the main literature review documents.**

All materials copyright 2025 | Research synthesis from academic literature, arXiv, and institutional research
