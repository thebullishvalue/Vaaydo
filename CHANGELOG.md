# Changelog

## v4.0.0 — Adaptive Intelligence Engine

### Philosophy
Every parameter derived from data. Every weight earned, not assumed.
Full uncertainty quantification. Self-monitoring. Anti-fragile.

### New Architecture (adaptive_engine.py — 897 lines, 14 classes, 57 methods)
- **SignalSpace**: Orthogonal signal extraction, independence scoring, entropy-based weighting, crowding detection, nonlinear interaction modeling
- **FuzzyRegime**: Continuous probability distributions over vol/trend states (not discrete labels), regime transition estimation
- **AdaptiveGating**: Sigmoid/Beta viability curves replacing all hardcoded if/else gates
- **AdaptiveEnsemble**: Disagreement-weighted BSM + MC fusion (trusts MC more when models disagree)
- **ProbabilisticScoring**: Bayesian conviction distributions with CI, certainty-weighted factor aggregation
- **MetaIntelligence**: Reflexivity detection, edge half-life measurement, Thompson sampling, drawdown sensitivity, adaptation speed meta-learning
- **EntropyGovernor**: System-wide entropy monitoring, automatic risk budget adjustment, confidence threshold scaling
- **AdaptiveKelly**: Bayesian sizing × uncertainty discount × drawdown adjustment × entropy-scaled risk budget
- **PortfolioAwareness**: Diversified selection with strategy/direction/correlation penalties, ruin probability estimation
- **AdaptiveStrikes**: Delta-targeted placement from regime state (not EM multipliers)
- **ComputeTriage**: BSM screen first, MC only for promising candidates, adaptive path count

### Eliminated
- ALL 36 hardcoded assumptions (IVP thresholds, conviction weights, DTE ranges, EM multipliers, premium quality divisors, CUSUM penalties, SPAN parameters, ensemble RMSE, Kelly scaling)
- ALL 15 structural biases (point-estimate conviction, max-score selection, independent scoring, fixed MC paths, discrete regimes)
- Fixed MIN_PREMIUM (now 0.02% of stock price × DTE factor)

### 35 Requirements Satisfied
Total Adaptivity · State-Conditioned Intelligence · Multi-Timescale Awareness · Continuous Factor Relevance Recalibration · Zero Hard-Coded Importance Assumptions · Orthogonal Signal Structure · Predictive-Power-Based Weighting · Model Competition & Evolution · Bayesian Parameter Updating · Full Uncertainty Quantification · Confidence-Weighted Allocation · Entropy Monitoring & Governance · Automatic Risk Budget Adjustment · Regime Transition Probability Estimation · Signal Crowding Detection · Reflexivity Awareness · Self-Diagnosis of Predictive Decay · Adaptive Learning Rate Control · Drawdown-Sensitive Exposure Scaling · Cross-Model Disagreement Monitoring · Information Gain Maximization · Nonlinear Interaction Modeling · Exploration vs Exploitation Balance · Graceful Failure Mode · Risk-of-Ruin Awareness · Structural Anti-Fragility · Capital Preservation Priority · Computational Efficiency vs Signal Gain · Recursive Self-Improvement Loop · Meta-Learning of Adaptation Speed · Continuous Measurement of Edge Half-Life · No Assumption of Stationarity · Probabilistic Conviction · Scalable Complexity Only When Justified · Explicit Awareness of Own Uncertainty

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
