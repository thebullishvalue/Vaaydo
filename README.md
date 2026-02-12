# VAAYDO (वायदो) — The Promise

### World's First God-Tier Options Trading Intelligence System

**Hemrek Capital** · Version 3.0.0

---

## Overview

VAAYDO is a comprehensive, self-contained FnO (Futures & Options) trade intelligence system built for the Indian derivatives market. It auto-fetches all market data, computes institutional-grade analytics from raw OHLCV, and delivers conviction-scored trade recommendations backed by rigorous quantitative mathematics.

Named after the Gujarati/Sanskrit word for **"promise"**, VAAYDO embodies the commitment to mathematically-rigorous, bias-free trade identification.

---

## Mathematical Arsenal — 20 Engines

| # | Engine | Implementation |
|---|--------|---------------|
| 1 | **Black-Scholes-Merton** | Full pricing + 9 Greeks (Δ, Γ, Θ, ν, ρ, Vanna, Volga, Charm, Speed) |
| 2 | **Multi-Estimator Volatility** | Close-to-Close, Parkinson, Garman-Klass, Yang-Zhang (weighted composite) |
| 3 | **GARCH(1,1)** | Conditional variance forecasting, persistence, half-life estimation |
| 4 | **Volatility Risk Premium** | Adaptive IV estimation from RV + regime-dependent VRP factor |
| 5 | **Monte Carlo (Antithetic)** | 10,000 effective paths with antithetic variate variance reduction |
| 6 | **Kelly Criterion** | Half-Kelly with confidence-weighted dynamic position sizing |
| 7 | **Hidden Markov Model** | 6-state volatility + 5-state trend regime detection |
| 8 | **Kalman Filter** | Adaptive signal smoothing (via Nirnay heritage) |
| 9 | **CUSUM Detection** | Multi-stream structural break alerts |
| 10 | **Bayesian Adaptive Thresholds** | No-bias, data-driven boundary discovery |
| 11 | **Higher-Order Greeks** | Vanna, Volga, Charm, Speed for sophisticated risk mgmt |
| 12 | **Composite Risk Score** | Regime-dependent weighted risk aggregation |
| 13 | **Ensemble POP** | Inverse-variance weighted BSM + MC fusion |
| 14 | **Regime Transition** | Stability scoring + transition probability forecasting |
| 15 | **Entropy Uncertainty** | Regime ambiguity quantification |
| 16 | **Expected Move Zones** | 1σ/2σ/3σ log-normal probability distributions |
| 17 | **Sharpe Optimization** | Risk-adjusted strategy ranking |
| 18 | **Information Ratio** | Strategy performance benchmarking |
| 19 | **Unified Conviction** | §9.1 multi-factor weighted formula |
| 20 | **Optimal DTE** | Theta decay curve analysis |

## Strategy Universe — 10 Active

| Strategy | Market View | Risk Profile |
|----------|------------|--------------|
| Short Strangle | Neutral, Range-bound | Unlimited risk, High premium |
| Short Straddle | Strongly Neutral | Unlimited risk, Max premium |
| Iron Condor | Neutral, Defined Risk | Defined risk, Moderate premium |
| Iron Butterfly | Pin at Strike | Defined risk, High premium |
| Bull Put Spread | Bullish | Defined risk |
| Bear Call Spread | Bearish | Defined risk |
| Calendar Spread | Neutral, Vol Expansion | Defined risk, Time decay |
| Jade Lizard | Neutral-Bullish | Upside risk limited |
| Broken Wing Butterfly | Directional Bias | Asymmetric risk |
| Ratio Spread | Mildly Bullish | Unlimited upside risk |

---

## Installation

### Prerequisites

- Python 3.9+
- Internet access (for yfinance market data)

### Setup

```bash
# Clone the repository
git clone https://github.com/hemrek-capital/vaaydo.git
cd vaaydo

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run vaaydo.py
```

The system will automatically:
1. Fetch the F&O stock list (NSE API with fallback)
2. Download OHLCV data for 110+ securities via yfinance
3. Compute all volatility estimators, technicals, and regime metrics
4. Run BSM + Monte Carlo analysis for every security
5. Present conviction-scored recommendations

**No CSV files, no manual data — everything is auto-fetched.**

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  L7: Signal Generator                        │
│  Unified Conviction · Ensemble POP · Ranking │
├──────────────────────────────────────────────┤
│  L6: Strategy Optimizer                      │
│  10 Strategies · Kelly Criterion · Sizing    │
├──────────────────────────────────────────────┤
│  L5: Regime Intelligence                     │
│  HMM · CUSUM · Stability · Transitions       │
├──────────────────────────────────────────────┤
│  L4: Pricing Engine                          │
│  BSM · Greeks (9) · Risk Score               │
├──────────────────────────────────────────────┤
│  L3: Volatility Engine                       │
│  C2C · Parkinson · GK · YZ · GARCH · VRP    │
├──────────────────────────────────────────────┤
│  L2: Technical Analysis                      │
│  RSI · ATR · MAs · Volume · PCR Proxy        │
├──────────────────────────────────────────────┤
│  L1: Data Ingestion                          │
│  yfinance · NSE API · Cache · Validation     │
└──────────────────────────────────────────────┘
```

## Key Formulas

### Volatility Estimators (§3.2)

- **Close-to-Close:** `σ = std(ln(Ct/Ct-1)) × √252`
- **Parkinson:** `σ = √(Σ(ln(H/L))² / (4n·ln2)) × √252`
- **Garman-Klass:** `σ = √(0.5·u² - (2ln2-1)·c² + 0.5·d²) × √252`
- **Yang-Zhang:** `σ = √(σ²_overnight + k·σ²_close-open + (1-k)·σ²_GK) × √252`
- **Composite:** Weighted average (YZ: 40%, GK: 25%, Park: 20%, C2C: 15%)

### Conviction Scoring (§9.1)

```
Conviction = w_r·Regime_Alignment + w_p·POP + w_e·EV_norm
           + w_s·Sharpe_norm + w_st·Stability + w_iv·IV_norm
```

### Ensemble POP (§9.2)

```
POP_ensemble = (w_bsm·POP_bsm + w_mc·POP_mc) / (w_bsm + w_mc)
where w_i = 1/RMSE_i²  (inverse-variance weighting)
```

---

## UI Tabs

| Tab | Function |
|-----|----------|
| **⚡ Trade Radar** | Top 9 opportunities as premium cards with conviction scores |
| **🔬 Deep Analysis** | Full strategy teardown: payoff diagrams, Greeks, vol estimators, position sizing |
| **📊 Rankings** | Complete sortable data table with all metrics |
| **📐 Probability Lab** | Monte Carlo distributions, sample paths, BSM Greeks chain |

---

## Configuration

All parameters are configurable via the sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Expiry Date | Next Thursday | Calendar picker with NSE weekly expiry suggestions |
| Strike Gap | ₹50 | Strike price increment |
| Capital | ₹5,00,000 | Portfolio capital for Kelly sizing |
| Min IV Percentile | 20% | Filter threshold |
| Min Conviction | 30 | Minimum conviction score |

---

## Technical Specifications

| Spec | Value |
|------|-------|
| Risk-free rate | 7% (India govt bonds) |
| MC paths | 10,000 (5,000 + 5,000 antithetic) |
| GARCH params | ω=5e-6, α=0.10, β=0.85 |
| Kelly safety | Half-Kelly × confidence, 25% cap |
| Vol estimator weights | YZ:40%, GK:25%, Park:20%, C2C:15% |
| VRP range | 8%–18% (regime-adaptive) |
| Regime states | 6 vol × 5 trend = 30 combinations |
| Data source | yfinance (auto-fetch) |
| UI framework | Streamlit |
| Charts | Plotly (dark theme) |

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Disclaimer

VAAYDO is a research and analysis tool. It does not constitute financial advice. Options trading involves substantial risk of loss. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

**Hemrek Capital** · Built with mathematical precision
