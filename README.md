# VAAYDO (वायदो) — FnO Trade Intelligence

Quantitative options strategy screener and analytics platform for NSE F&O.

---

## What It Does

Screens **207+ NSE F&O securities** across **14 option strategies** using a fully adaptive intelligence engine. Every parameter is derived from the universe distribution at runtime — zero hardcoded thresholds, zero assumed weights.

## Architecture

```
SIGNAL SPACE     → Orthogonalize signals, measure information, detect crowding
FUZZY REGIME     → Continuous probability distributions (not discrete labels)
ADAPTIVE GATING  → Sigmoid/Beta viability curves (never binary)
ENSEMBLE         → Disagreement-weighted BSM + MC fusion
SCORING          → Bayesian conviction with full uncertainty (mean ± std, CI)
META-INTELLIGENCE → Reflexivity, edge decay, exploration, anti-fragility
ENTROPY GOVERNOR → System-wide risk budget scales with uncertainty
ADAPTIVE KELLY   → Bayesian sizing × drawdown sensitivity
PORTFOLIO        → Cross-sectional diversification, ruin prevention
```

## What Makes v4.0 Different

| v3.x (Threshold Machine) | v4.0 (Adaptive Intelligence) |
|---|---|
| `if ivp < 20: BLOCK` | Sigmoid viability from universe percentile |
| Conviction = 67.3 (point estimate) | Conv = 67.3 ± 8.2, CI = [56.8, 77.8] |
| Fixed weights: RA 20%, POP 22%, EV 12% | Certainty-weighted: uncertain factors contribute less |
| `MIN_PREMIUM = ₹0.50` (same for ₹11 and ₹35,000 stock) | 0.02% of stock price × DTE factor |
| `vol_regime = HIGH` (discrete) | P(high) = 0.38, P(elevated) = 0.30, ... |
| Always picks highest conviction | Thompson sampling for exploration |
| 139/204 = Bull Put Spread | Reflexivity detector penalizes concentration |

## Files

| File | Purpose |
|------|---------|
| `vaaydo.py` | Main Streamlit application (2,067 lines) |
| `adaptive_engine.py` | v4.0 intelligence engine (897 lines, 14 classes, 57 methods) |
| `ARCHITECTURE.md` | Full system design document |
| `CHANGELOG.md` | Version history |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run vaaydo.py
```

## Requirements

- Python 3.10+
- Network access to Yahoo Finance

---

Version 4.0.0 · Hemrek Capital
