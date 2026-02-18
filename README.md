# VAAYDO (वायदो) — FnO Trade Intelligence

Quantitative options strategy screener and analytics platform for NSE F&O.

---

## What It Does

Screens **207+ NSE F&O securities** across **14 option strategies** using 20 quantitative engines. Produces conviction-scored trade recommendations with lot-adjusted financials (₹ P&L per lot, ROM%, daily theta income).

The system automatically selects the optimal strategy per stock by cross-referencing IV regime, trend direction, volatility state, and DTE fitness.

## Architecture

```
DATA          NSE F&O Universe → Yahoo Finance → 252-day analytics
                ↓
ANALYTICS     Multi-Vol (C2C/Park/GK/YZ) → GARCH(1,1) → VRP → Kalman → CUSUM
                ↓
GATING        IVP Gate → DTE Gate → Premium Quality Check
                ↓
ENGINE        14 Strategies × BSM Pricing × MC(10K Antithetic) × Kelly
                ↓
SCORING       9-Factor Conviction: RA · POP · EV · Sharpe · Stab · IV · PQ · DTE · CUSUM
                ↓
OUTPUT        Trade Radar → Deep Analysis → Rankings → Probability Lab
```

## Strategy Universe

| Bias | Credit | Debit |
|------|--------|-------|
| ▲ Bullish | Bull Put Spread | Bull Call Spread |
| ▼ Bearish | Bear Call Spread | Bear Put Spread |
| ◆ Neutral | Iron Condor · Iron Butterfly · Short Strangle · Short Straddle · Jade Lizard | Calendar Spread |
| ⚡ Volatile | — | Long Straddle · Long Strangle |

Hybrid: Broken Wing Butterfly · Ratio Spread

## Intelligence Layers

| Layer | What It Does |
|-------|-------------|
| IVP Gate | Blocks credit strategies when IV too low; blocks debit when IV too high |
| DTE Gate | Filters by viable expiry range (Calendar needs 14+ DTE, spreads work at 1+) |
| Regime Alignment | Cross-refs IV(50%) × trend(40%) × vol(10%) per strategy |
| Premium Quality | Strategy-specific credit/risk thresholds (6% for naked, 20% for IC, 25% for spreads) |
| CUSUM Penalty | Structural break → 0.60× neutral strats, 1.05× volatile strats |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run vaaydo.py
```

Requires Python 3.10+ and network access to Yahoo Finance.

## Engines

| Engine | Purpose |
|--------|---------|
| BSM | Full pricing + 9 Greeks (Δ Γ Θ ν ρ Vanna Volga Charm Speed) |
| Multi-Vol | Close-to-Close, Parkinson, Garman-Klass, Yang-Zhang estimators |
| GARCH(1,1) | Conditional variance, persistence (α+β), half-life |
| VRP | Regime-adaptive IV from realized vol composite |
| Monte Carlo | 10K antithetic paths, strategy-specific payoff simulation |
| Kelly | Continuous f* = μ/σ², capital-normalized, half-Kelly scaled |
| Ensemble POP | Inverse-variance BSM + MC probability fusion |
| SPAN Margin | DTE-scaled max-loss for unlimited-risk strategies |

## Tabs

| Tab | Content |
|-----|---------|
| ⚡ Trade Radar | Top 9 cards: lot-adjusted P&L, ROM%, Θ/day, alternative strategy |
| 🔬 Deep Analysis | Per-security: vol estimators, expected move, 5-strategy ranking with payoffs |
| 📊 Rankings | Full sortable table with CSV export |
| 📐 Probability Lab | MC terminal distribution, sample paths, BSM Greeks chain |

## Configuration

- **Expiry Date** — defaults to next Thursday with ≥3 DTE
- **Min IV Percentile** — filter by IV environment (drives credit/debit selection)
- **Min Conviction** — minimum score threshold

---

Version 3.3.0 · Hemrek Capital
