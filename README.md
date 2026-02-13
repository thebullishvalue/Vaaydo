# VAAYDO (वायदो) — FnO Trade Intelligence

**Version 3.1.0** · Hemrek Capital

20 mathematical engines, 10 strategy evaluators, ensemble probability fusion, regime-aware conviction scoring.

---

## Architecture (7-Layer Stack)

```
L7 │ Kelly Criterion — Half-Kelly, confidence-weighted, capital-capped
L6 │ Strategy Engine — 10 strategies, real MC for all, full 9-Greeks
L5 │ Regime Intelligence — 6-Vol × 5-Trend + ADX + Kalman + CUSUM
L4 │ BSM Pricing — Analytical + 9 Greeks (Δ Γ Θ ν ρ Vanna Volga Charm Speed)
L3 │ Monte Carlo — 10K antithetic paths, generic payoff engine
L2 │ Technical Analysis — RSI, ATR, ADX, MAs, Volume, Kalman Filter
L1 │ Data Ingestion — Auto-fetch, Multi-Estimator Vol, GARCH, VRP
```

## v3.1.0 Bugfixes (14 Fixes)

| Fix | Bug | Resolution |
|-----|-----|------------|
| #1 | Strike collapse when em < gap | Enforce max(em, gap) for all strike placement |
| #2 | Iron Butterfly wing width = 0 | ww = max(snap(em*1.2, gap), min_wing) |
| #3 | Sharpe 10+ digit values | clamp_sharpe(): return 0 if std < 0.01, cap [-5, 5] |
| #4 | Arbitrary max loss for unlimited risk | span_margin(): max(15% underlying, 2σ monthly) |
| #5 | EV normalization wrong for ₹1000+ stocks | Normalize as ev/max_profit ratio, not absolute ₹ |
| #6 | Risk-Reward explosion (ml=0.01 floor) | clamp_rr(): cap at [0, 50] |
| #7 | Bull Put/Bear Call strikes collapse | Enforce lp ≤ sp - min_wing |
| #8 | 5/10 strategies used fake MC | All 10 now use MC.analyze() with real simulation |
| #9 | IC/IB/Calendar Greeks = theta only | compute_full_greeks() for all 9 Greeks on every strategy |
| #10 | Calendar max_profit fabricated | Proper BSM back month residual at front expiry |
| #11 | BWB net credit had ×0.1 multiplier | Correct payoff: 2×center - low - high |
| #12 | Ratio Spread mp overstated | Correct: (sK-lK) + net_credit at short strike |
| #13 | Zero-premium strategies displayed | Skip when net_credit < ₹0.50 |
| #14 | Missing lot sizes | 110+ NSE lot sizes integrated |

## New in v3.1

- **Kalman Filter** (§5.3) — Adaptive trend smoothing for regime detection
- **ADX** (§4.3) — Trend strength confirmation (threshold 25)
- **SPAN Margin** — Realistic max loss for unlimited-risk strategies
- **Generic MC Engine** — MC.analyze() works for any multi-leg payoff
- **Greeks Algebra** — Greeks dataclass with add/negate/scale operators
- **Lot Sizes** — 110+ NSE F&O lot sizes for position sizing

## Quick Start

```bash
pip install -r requirements.txt
streamlit run vaaydo.py
```

## Key Formulas

**Sharpe** (clamped): `clamp(EV/σ, -5, 5)` — returns 0 if σ < 0.01
**Conviction** (§9.1): `20%×RA + 25%×POP + 15%×(EV/MaxProfit) + 20%×Sharpe + 10%×Stability + 10%×IV`
**Ensemble POP** (§9.2): Inverse-variance weighted BSM+MC fusion (MC gets ~2.56× weight)
**SPAN Margin**: `max(0.15×S, 2σ_monthly)` per lot

## License

MIT License · Copyright 2026 Hemrek Capital
