# Pre-registration: intraday structure as stock signals (Dhan data)

Written 22 Nov 2026 before any of these features was computed against returns. Changes after this point are
listed at the bottom with the reason.

## Why these, and not Dhan's greeks or PCR

Dhan's put/call ratios, IV and greeks are the same information VAAYDO already reads, more completely, from
NSE's end-of-day files (every strike of every stock option since 2019): the score *is* put/call volume and OI,
put and call skew, the call-put IV spread and net option flow, and dealer gamma and delta-hedge flow were
registered and tested on 17 Nov 2026 (docs/PREREGISTRATION.md). Re-testing them from Dhan would re-test the same
numbers. What NSE's daily files do not carry is **how a stock trades inside the session**. Dhan does, for five
years; the vault holds it per stock-session (open, 30/60-minute and session VWAP, close; 274 stocks, Jul 2024 on).

## Features (known at the close of t)

| | Feature | Expected sign | Reason |
|---|---|---|---|
| I1 | `on10`: sum of the last 10 overnight returns, log(open_t / close_{t-1}) | + | Returns earned overnight persist (momentum lives overnight) |
| I2 | `id10`: sum of the last 10 intraday returns, log(close_t / open_t) | − | Intraday moves tend to reverse |
| I3 | `tug10`: on10 − id10 | + | The two together ("tug of war") |
| I4 | `cv5`: mean of the last 5 log(close / session VWAP) | + | Strength into the close is informed buying |

## Outcome, split and bar (the v6 standard)

* Outcome: the futures return from the next session's open to the close h sessions later (contract active on
  the fill day; NaN if it expires first), minus that day's cross-sectional mean, divided by the stock's trailing
  63-session futures volatility × √h. **h = 10** (the book's horizon) and h = 5.
* **Discovery: Jul 2024 – Dec 2025. Holdout: 2026**, computed once, only for discovery survivors.
* Daily cross-sectional Spearman IC, Newey-West t (lags = h).
* **Survives discovery:** |t| ≥ 3.0 at h = 10 or 5 (4 features × 2 horizons), **and** the IC stays at t ≥ 2 after
  removing the existing score (each day, the feature's rank residualised on the score's rank).
* **Confirmed:** holdout IC of the same sign with t ≥ 2.
* **Book gate:** a confirmed feature enters the score as a seventh equal-weight rank (signed as found). It is
  adopted only if the one-lot book's Sharpe rises by ≥ 0.10 in discovery and is not lower in the holdout.

## Changes after registration

(none)
