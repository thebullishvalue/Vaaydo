# Pre-registration: VAAYDO v6 signal research

Written before any v6 signal was computed (17 Nov 2026). Changes after this point are
listed at the bottom with the reason.

## Question

Do the end-of-day F&O data (futures price, basis and open interest; option prices, IV
surface, volume and open interest by strike; spot volume and delivery) predict the
**cross-section** of single-stock F&O returns over the next 1–20 trading days, by enough
to trade after costs?

## Data and split

* Universe: every NSE single-stock F&O underlying on each day (no look-ahead membership).
* **Discovery: 8 Jul 2024 – 31 Dec 2025.** All exploration, feature design and
  composite fitting happen here.
* **Holdout: 1 Jan 2026 – 17 Nov 2026.** Touched once, at the end, for the final
  candidate set only. Results are reported whatever they are.

## Outcome

Forward return from the **next session's open** to the close h sessions later
(h = 1, 5, 10, 20), corporate-action adjusted, minus that day's cross-sectional mean
(market-neutral), divided by the stock's trailing 63-day volatility.

## Test

* Daily cross-sectional Spearman rank IC per feature, averaged over days, with
  Newey-West t-statistics (lags = h).
* **A feature survives discovery** with |t| ≥ 3.0 at its horizon (about 40 features × 4
  horizons are tested, so t ≥ 3 keeps the expected false discoveries below one).
* **A feature is confirmed** if the holdout IC has the same sign and t ≥ 2.0.
* Only confirmed features may enter the live curator. Composite weights are fitted on
  discovery only.

## Trading test (after confirmation)

Long-short futures portfolio on the composite (top vs bottom quintile, equal risk), next
open entry, real costs (STT, exchange, stamp, GST, brokerage, 3–10 bp slippage by
liquidity), Pragyam stops. Reported separately for discovery and holdout.

## Frozen candidate (17 Nov 2026, after discovery, BEFORE any holdout computation)

Discovery survivors (|t| >= 3): put/call volume, put/call OI, put skew, basis, 5-day
basis change, delivery % and its surprise, call OI flow, net option flow, call-put IV
spread, ATM IV, 5-day OI change, turnover surprise. Joint Fama-MacBeth keeps two
independent cores (put/call volume t = 3.9, basis t = 4.8 at 5d) and two weaker ones
(call OI flow, call-put IV spread).

**Composite (frozen):** `rank(pcr_vol) + rank(basis) + 0.5 * (1 - rank(call_oi_flow)) + 0.5 * (1 - rank(cpiv))`,
ranks taken within each day's cross-section. **Horizon: 10 sessions** (5 also reported).
**Trade:** long top quintile, short bottom quintile, next-open entry, 0.11% round-trip cost.

Holdout tests run once, on: each of the four components, the composite, and the
quintile portfolio. Pass = composite holdout IC same sign with t >= 2, and the net
long-short return positive.

## Holdout result, look 1 (spot outcome)

Composite holdout IC 0.043 at 10 sessions, t = 3.1; quintiles monotone; long-short
+0.45% per trade net on SPOT returns. Passed as specified.

## Changes after registration

1. **Outcome changed from spot to futures returns** (17 Nov 2026, after holdout look 1).
   Reason: the trades are in futures. Re-simulated on real futures prices, the frozen
   composite earned +0.18% (discovery, t = 1.3) and +0.06% (holdout, t = 0.3) per pair
   net: the basis component predicts spot drifting up to the futures price, which a futures
   holder pays away as the premium converges. Features are re-selected on DISCOVERY with
   futures outcomes; the holdout is looked at a second time. Because it is a second look,
   the bar for holdout confirmation rises from t >= 2.0 to t >= 2.5.

## Frozen candidate v2 (after re-selection on discovery with futures outcomes, BEFORE holdout look 2)

On futures outcomes, basis drops to IC 0.000 (it was carry). The surviving family is
option-market positioning. **OPI (Option Positioning Index), equal weights, no fitting:**

`rank(pcr_vol) + rank(pcr_oi) + rank(skew) + (1 - rank(cskew)) + (1 - rank(cpiv)) + rank(net_opt_flow)`

Discovery (futures, 10 sessions): IC 0.054, t = 4.3. Trade: long top 10% / short bottom
10% by OPI, next-open futures entry, contract clear of expiry, held 10 sessions, disaster
stop at 2x Pragyam's 20%-touch stop distance, 0.126% round-trip cost.
Discovery: +0.50% per long-short pair net, t = 2.8. (Pragyam's 1x stop cost 0.13%/pair:
rejected. Quintiles and a liquidity filter were weaker: rejected.)

Pass on holdout look 2: OPI 10-session IC t >= 2.5 AND pair net > 0 with t >= 2.0.

## Holdout look 2 result: FAILED
OPI 10-session IC 0.016 (t = 1.0); long-short pair +0.08% net (t = 0.3), Jan-Nov 2026.

## Long out-of-sample test (2019 - Jul 2024, pre-UDiFF archive, never used): FAILED
Frozen v2 rules exactly: pair +0.03% net per trade, t = 0.22, 42,509 trades, 1,325 days.
By year: 2019 +0.51%, 2020 -0.27%, 2021 +0.31%, 2022 +0.04%, 2023 -0.45%, 2024 +0.03%.
Option positioning had no durable edge; 2024-25 was a period effect. **Rejected.**

## Protocol v3 (registered before any computation on the 2019-2024 sample)
* Discovery: 2019-01-01 to 2022-12-31. Validation: 2023-01-01 to 2024-06-30.
  Final check: 2024-07-08 onward (already seen; reported but not relied on).
* Outcome: FUTURES returns, next open to close h (h = 5, 10), market-neutral, vol-scaled.
* A signal must have |t| >= 3 in discovery, same sign t >= 2 in validation, and same sign
  in the final check. A strategy must be net-positive after costs in every stage.

## Frozen v3 (after discovery 2019-2022, BEFORE validation)
Discovery IC screen (futures, 2019-22): the option-positioning family persists (pcr_vol
t = 4.9, pcr_oi 3.6, skew 3.5, cskew -3.5, net_opt_flow 3.1 at 5 sessions) but at half the
2024-25 strength; trade-by-trade it did not clear costs (gross 0.29% vs 0.25% per pair).
Daily OPI is noisy (rank autocorrelation 0.46) and a daily book churns.

**Score:** OPI = equal-weight rank of pcr_vol, pcr_oi, skew, -cskew, -cpiv, net_opt_flow;
daily cross-sectional percentile, then its **10-session mean** (`opi_s10`).
**Book:** futures, enter top/bottom 10%, hold while in top/bottom 20%, equal notional per
side, next-open execution, roll 3 sessions before expiry, disaster stop 2x Pragyam,
0.126% round trip per name charged on entries, exits and rolls.
Discovery: Sharpe 0.88, every year positive (0.99, 0.70, 1.17, 0.73), 8.6 trades/day.
**Validation pass:** Sharpe > 0.5 net and positive in 2023 and in 2024 H1.

## Correction: stop look-ahead (18 Nov 2026)
`book.simulate` checked a stop against the close of day t+1 but exited at the open of t+1,
before that close was known. All stop variants were inflated, tight ones most (a 1x
trailing stop showed Sharpe 2.45). Fixed: stops are decided on the close of d and
executed at the next open; a stopped name is barred until its signal leaves the band.
Fixed results, 12-name book 2019-2026: no stop 0.69; fixed 1x/1.5x/2x 0.58/0.71/0.65;
trailing 1x/1.5x/2x 0.41/0.66/0.66. Stops add no return; the 1.5x trailing stop was
chosen for risk behaviour (it follows the price), not for its backtest. Published track
record regenerated: model book 0.77, 20 names 0.76, 12 names 0.66, 6 names 0.30.

## Option-flow readings (registered 18 Nov 2026, before the book test below)
Tested on futures outcomes, discovery 2019-22 / validation 2023-24H1 / final 24H2-26:
buyers-vs-writers opening (contract build-up) weak (abs t <= 2.3, mixed); unusual volume
sign flips; **call-vs-put volume surge (surge_tilt) negative in all three periods**
(t at 5 sessions: -3.5 / -4.5 / -2.0), and it held while the score was flat (2023-24H1).
**Hypothesis (one, fixed now):** combined = rank(score) + 0.5 x rank(-surge_tilt), same
book rules (5% / 15% hysteresis, trailing 1.5x stop). Adopted only if discovery Sharpe is
not lower than the score alone AND validation improves.
**Result: rejected.** Score alone: Sharpe 1.23 / -0.81 / 0.67 (disc / valid / final), 5.7
trades a day. Score + surge: 0.06 / -1.88 / -0.19, 26.8 trades a day: the daily component
churns the book and costs take it. Surge, buyers/writers and unusual volume are shown as
context on every name; none enters the signal.
