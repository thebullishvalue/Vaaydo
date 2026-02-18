# Changelog

## v3.3.0 — Intelligence Overhaul

### New Intelligence Layers
- **DTE Gate**: Each of 14 strategies has a viable DTE range; strategies outside range auto-eliminated
- **Premium Quality Scoring**: Strategy-type-aware credit/risk thresholds (6% naked, 20% IC, 25% spreads)
- **CUSUM Penalty**: Structural breaks penalize neutral strategies (0.60×), bonus volatile (1.05×)
- **Alternative Strategy**: Every stock shows best + second-best strategy for optionality

### Mathematical Fixes
- **Kelly Criterion**: Capital-normalized (μ/capital) / (σ/capital)² — produces meaningful position sizes
- **Sharpe Ratio**: Normalized by max_loss for fair cross-strategy comparison
- **BSM POP**: Fixed dual-condition formulas — addition-1 for credit strats, addition for debit strats (was incorrect product formula)
- **SPAN Margin**: Scales with DTE (√T), less aggressive for near-expiry
- **Strike Placement**: IC at 0.6×EM, IB wings at 0.8×EM, Strangle at 0.7×EM (were all at 1.0×EM, causing poor premium)

### Conviction Upgrade (6 → 9 factors)
- Added: premium quality (8%), DTE fitness (7%), CUSUM penalty (multiplier)
- Reweighted: RA(20%), POP(22%), EV(12%), Sharpe(15%), Stability(8%), IV(8%)

### Lot-Adjusted Financials
- Cards show ₹ Profit (lot), ₹ Risk (lot), ROM%, Θ/Day ₹
- Ranking table includes lot-adjusted columns
- Deep Analysis shows per-strategy lot-adjusted panel

### UI Enhancements
- Market Regime + Strategy Mix + Top-5 Θ/Day + Top-5 Capital metric cards
- Bias tags (▲▼◆⚡) + Type tags (₹ CREDIT / ₹ DEBIT / ₹ HYBRID) on cards
- Alternative strategy shown on each card
- Default expiry ensures minimum 3 DTE

## v3.2.0 — IV-Regime-Aware Strategy Selection

### Intelligent Strategy Selection (replaces manual credit/debit toggle)
- **IVP Gate**: Credit strategies blocked when IVP < 20; debit blocked when IVP > 85
- **Unified Regime Alignment Engine**: 3-signal composite (IV 50% + Trend 40% + Vol 10%)
- **Strategy Type Classification**: CREDIT (7) / DEBIT (5) / HYBRID (2)

### Strategy Expansion (10 → 14)
- Added: Bull Call Spread, Bear Put Spread, Long Straddle, Long Strangle
- Strategy bias system: BULLISH / BEARISH / NEUTRAL / VOLATILE
- Dynamic card accent colors by bias

## v3.1.0 — Mathematical Foundation

### Kelly Criterion Fix
- Monte Carlo-based continuous Kelly: f* = μ/σ² (half-Kelly, confidence-scaled)
- Eliminates binary Kelly oscillation for spread strategies

### UI Redesign
- Single sidebar button ("Run Analysis" / "Refresh Data")
- Expanded lot size database (256 stocks)
- CSV data quality verification (7/7 health score)

### Core Engines
- BSM + MC(10K antithetic) + GARCH(1,1) + Kelly
- 9 Greeks: Δ Γ Θ ν ρ Vanna Volga Charm Speed
- 6-state vol regime + 5-state trend regime
- CUSUM structural break detection
- Ensemble POP (BSM + MC fusion)
