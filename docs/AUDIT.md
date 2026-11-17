# Audit of VAAYDO v4.x (the pilot) — why it was rebuilt

The pilot's goal was right: one screen that says which F&O trade to put on, in which
name, at what size. Its engine could not do that, for reasons that no amount of tuning
would fix.

## Fatal: it never looked at an option price

| What the pilot called it | What it actually was |
|---|---|
| `ATMIV` | 20-day realised vol × 1.08–1.18 (a fixed "VRP factor") |
| `IVPercentile` | percentile of *realised* vol over the past year |
| `PCR` | down-day volume ÷ up-day volume of the stock |
| every premium | Black-Scholes price of the synthetic IV above |

An options screener that never reads option prices cannot find mispriced options.

## Fatal: the edge was manufactured

Premiums were priced at `RV × 1.12` and then simulated at `sim_vol ≈ RV × 1.05`, with a
constant +7% annual drift in every Monte Carlo path. Selling premium, especially bull put
spreads, therefore showed positive expected value **by construction**. The "139 of 204
stocks = Bull Put Spread" concentration that the v4 architecture document complained about
was this artefact, not a market signal.

## Wrong market structure

* Defaulted to a **weekly Thursday expiry** for stock options. Single-stock F&O in India
  is monthly only, expiries have been on Tuesday since Sep 2025 (earlier on holidays),
  and stock options are **physically settled**.
* Lot sizes came from a hardcoded table that was stale (RELIANCE 250; it is 500).
* Strike intervals came from a price-bucket rule instead of the listed strikes.
* Allowed naked short strangles and straddles on single stocks.

## Bugs that changed outputs

* The NseKit (live data) path was dead: `st.session_state.use_nse = False` ran
  unconditionally right after the data-source radio button.
* "Regime stickiness" smoothed each stock's regime with **the previous stock in the
  loop**, so a stock's regime depended on its alphabetical neighbour.
* The engine was calibrated once per server process and never again.
* Iron butterfly Greeks were computed with the wings at the body strike (net Greeks ≈ 0).
* The cost model passed a breakeven price as "spot" (zero slippage on call spreads) and
  used the pre-Oct-2024 STT rate.

## "Adaptive intelligence" that did nothing

Edge half-life, drawdown sensitivity, Thompson sampling and IC tracking were defined but
never called: no outcome was ever recorded. The validation tab ran a 17-test suite on
`_score_proxy`, a stub that returned a hardcoded in-sample Sharpe of 2.0 and
`conviction = 50 + 5·z`. Walk-forward was run on a single day's cross-section. "Zero
hardcoded thresholds" coexisted with dozens of hardcoded sigmoid slopes, affinity vectors
and Beta parameters.

## What the rebuild keeps

The intent, the name, the trader-facing summary (one plan, sized, with reasons), and the
instinct that a system should watch itself. Here, "watching itself" means a journal
that compares predicted to realised on every trade, and a walk-forward backtest on real
prices that anyone can rerun.
