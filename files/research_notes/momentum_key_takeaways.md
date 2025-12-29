# Momentum Investing: Key Takeaways for Research

## Quick Reference: Core Findings

### Canonical Definition (Jegadeesh & Titman, 1993)
Stocks with the highest returns over a 3-12 month formation period tend to outperform stocks with the lowest returns over the subsequent 3-12 month holding period. This effect is robust across all major tested combinations of formation and holding periods.

### Two Primary Implementations
1. **Cross-Sectional Momentum**: Long winners, short losers (relative strength)
2. **Time-Series Momentum**: Long if trending up, short if trending down (absolute momentum)

### Historical Performance (Key Metrics)
- **Long-term annualized return**: 8-9% (1926-2024)
- **Median Sharpe ratio**: 0.61 (range: 0.38-0.94)
- **Jegadeesh-Titman (1993) finding**: ~1% per month (12% annualized)
- **Earnings momentum (Novy-Marx, 2012)**: 90 bps/month (1,080% annualized pre-cost)
- **Factor momentum (Ehsani & Linnainmaa, 2020)**: +53 bps/month after gains, +1 bp/month after losses

### Academic Integration
- **1997**: Carhart adds momentum (UMD) to Fama-French three-factor model → four-factor model
- **2018**: Fama-French officially include momentum in five-factor model
- **Status**: Canonical fourth/sixth pricing factor in modern asset pricing

### Competing Theoretical Explanations

**Behavioral Finance (Underreaction Hypothesis)** - BETTER SUPPORTED
- Information diffuses gradually; prices underreact to news
- Hong & Stein (1999): Momentum traders profit before overreaction occurs
- Conservatism bias (Barberis et al., 1998): Investors slow to update beliefs
- Overconfidence + self-attribution (Daniel et al., 1998): Sustains mispricing
- Evidence: Post-earnings-announcement drift, earnings momentum > price momentum
- Geographic variation: Better explains differences across countries

**Risk-Based (Rational Pricing)** - PARTIALLY SUPPORTED
- Momentum stocks have higher conditional market beta
- Time-varying risk exposure justifies higher returns
- Crash risk: -88% max drawdown suggests tail risk compensation
- Macroeconomic risk factors: Partial explanatory power
- Magnitude: Explains ~30-50% of momentum premium, insufficient alone

**Consensus**: Hybrid explanation (behavioral underreaction + rational risk compensation)

### Key Limitations & Challenges

1. **Transaction Costs**
   - High turnover reduces net returns
   - Concentrated in high-spread stocks
   - Quarterly rebalancing vs. monthly reduces costs significantly

2. **Crash Risk**
   - Maximum documented drawdown: -88%
   - Occurs during regime shifts (2001, 2009, 2023)
   - Mitigation: Volatility scaling roughly doubles Sharpe ratio

3. **Performance Degradation Post-2000**
   - Returns lower than 1960s-1990s period
   - Hypotheses: Arbitrage crowding, improved information dissemination
   - More pronounced in developed markets; persists in emerging markets

4. **Unresolved Theoretical Questions**
   - Few direct tests of mutual exclusivity between competing explanations
   - Geographic variation difficult to explain with single framework
   - Optimal formation/holding period selection still open question

### Earnings vs. Price Momentum (Critical Distinction)

**Earnings Momentum (Superior)**
- Returns: 90 bps/month (1972-1999)
- Survives after FF3 controls and transaction costs
- Economically significant correlation with real GDP, industrial production
- Market incorporates cash flow information too slowly

**Price Momentum (Subordinate)**
- Returns: Insignificant when controlling for earnings momentum in same quintiles
- May be noisy proxy for earnings momentum
- Less reliable signal for future performance

### Geographic Evidence

**Developed Markets** (1990-2004)
- Country-neutral momentum: 56 bps/month (6.7% annualized)
- Effect documented but weaker than emerging markets

**Emerging Markets** (1990-2004)
- Country-neutral momentum: 79 bps/month (9.5% annualized)
- Daily momentum strong (14 of 21 markets)
- Country-switching returns: Up to 35% annualized
- Larger diversification benefit than developed markets

### Implementation Best Practices (Academic Consensus)

1. **Formation Period**: 12 months (canonical)
2. **Rebalancing Frequency**: Quarterly vs. monthly (reduces turnover, maintains performance)
3. **Skip Period**: 1 month (eliminates short-term reversal noise)
4. **Construction**: Decile portfolios (long top 10%, short bottom 10%)
5. **Risk Management**:
   - Volatility-scale momentum signals
   - Diversify across implementations
   - Monitor drawdown exposure
   - Consider hedging during regime transitions

### Most Critical Papers (Essential Reading)

1. **Jegadeesh & Titman (1993)** - Original empirical discovery (foundational)
2. **Carhart (1997)** - Factor integration into FF model
3. **Hong & Stein (1999)** - Behavioral theoretical framework
4. **Novy-Marx (2012)** - Earnings momentum subsumes price momentum
5. **Jegadeesh & Titman (2023)** - 30-year retrospective assessment
6. **Ehsani & Linnainmaa (2020)** - Factor momentum phenomenon

### Current Research Frontiers (Active Areas)

1. **Performance Prediction**: Explaining post-2000 return degradation
2. **Risk Management**: Optimal momentum scaling and hedging
3. **Theoretical Integration**: Unified framework for multiple explanations
4. **Cross-Asset Consistency**: Explaining momentum across equities, commodities, FX, bonds
5. **Signal Optimization**: Dynamic formation/holding period selection
6. **Information Decomposition**: Separating information momentum from behavioral momentum

### Practical Implementation Considerations

| Consideration | Impact | Mitigation |
|---------------|--------|-----------|
| Transaction Costs | Reduces net returns 0.5-2% annually | Quarterly rebalancing, portfolio concentration |
| Momentum Crashes | -88% max drawdown, 1x per decade | Volatility scaling, hedging, diversification |
| Performance Degradation | Lower returns post-2000 | International diversification, emergin market focus |
| Timing Risk | Difficult to time entry/exit | Long-term allocation, avoid market timing |
| Crowding Risk | Growing institutional adoption | Unique implementation approach, size focus |

### Questions for Further Research

1. Why does earnings momentum outperform price momentum so dramatically?
2. What specific behavioral mechanisms dominate in different geographic regions?
3. Can machine learning improve momentum signal generation and timing?
4. How does momentum interact with other factors in multi-factor portfolios?
5. What is the optimal volatility-scaling approach for crash risk mitigation?
6. How does momentum perform during high-inflation regimes?
7. Can real-time earnings surprise capture replace lagged price momentum?

---

## Full Literature Review Location
Complete detailed review: `/Users/jminding/Desktop/Code/Research Agent/files/research_notes/lit_review_momentum_investing.md`

This document contains:
- Comprehensive chronological development of research
- Detailed theoretical frameworks (behavioral vs. risk-based)
- Quantitative results tables
- Full bibliography with working links
- International evidence
- State-of-the-art summary
