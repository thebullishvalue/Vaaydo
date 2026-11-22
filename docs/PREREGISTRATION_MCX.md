# Pre-registration: MCX commodities

Written 22 Nov 2026 before any commodity signal or strategy return was computed. The only
data look so far is structural (history depth, contract stitching, expiry pattern; docs/DHAN.md).
Changes after this point are listed at the bottom with the reason.

## Question

Do MCX futures prices, term structure and open interest predict the **direction** of each
commodity's next 1–3 months, by enough to trade one mini lot per commodity after MCX costs?

## Universe (fixed now)

Nine non-agricultural commodities with history from Jan 2012 and a liquid future today:
**GOLD, SILVER, CRUDEOIL, NATURALGAS, COPPER, ZINC, ALUMINIUM, LEAD, NICKEL.**
Agricultural contracts (cotton, mentha, cardamom, kapas) are excluded: thin, and repeatedly
suspended by regulators. Minis trade the same price; the full contract's series is used.

## The return of a held position (no look-ahead, no delivery month)

Each position is held in the **second-listed contract** (Dhan's rank-2 series). That contract is
never in its delivery or tender period, so the book never faces physical delivery. When the
front contract expires, the held contract becomes the front: that day's return is
`front_t / second_{t-1}`, and the position rolls to the new second contract at that close.
Expiry days are recovered from the data (the new front continues yesterday's second contract in
price and open interest). Returns are log, daily, on the settlement close.

## Hypotheses (each tested on its own; no combination unless both pass)

| | Signal (known at the close of t) | Position from t+1 | Rebalance |
|---|---|---|---|
| H1 | **Trend, 12-1:** sign of the return from t−252 to t−21 | long if +, short if − | monthly (first session of the month) |
| H2 | **Trend, 3 months:** sign of the return from t−63 to t | long / short | weekly (first session of the week) |
| H3 | **Carry:** `log(front / second)`, z-scored against the commodity's own trailing 252 sessions | long if z > 0 (backwardation above normal), short if z < 0 | monthly |
| H4 | **OI build-up:** 20-session change in total OI (ranks 1–3) × sign of the 20-session return | long if price up with OI up, short if price down with OI up, flat if OI falls | weekly |
| H5 | **Cross-sectional momentum:** rank the nine by their 12-1 return | long top 3, short bottom 3 | monthly |

Each position is sized to equal risk (inverse trailing 63-session volatility, capped at 3× the
median) for the statistics; the live book is one mini lot each, reported separately.

## Split (one look at the last)

* **Discovery: Jan 2013 – Dec 2018** (2012 is warm-up for the 252-session signals).
* **Validation: Jan 2019 – Dec 2022.**
* **Final: Jan 2023 – 20 Nov 2026.** Computed once, for the hypotheses that pass validation.
  Reported whatever it shows.

## Costs

0.05% of notional per side (MCX CTT 0.01% on sells, exchange and SEBI fees, GST, stamp duty, and
one tick of slippage on the mini contracts), charged on every entry, exit and roll.

## Test and pass rule

Portfolio of the nine (equal risk), monthly non-overlapping net returns:

* **Survives discovery:** annualised Sharpe > 0.3 **and** Newey–West t ≥ 2.0 (lags 3).
* **Passes validation:** same sign, Sharpe > 0.2, block-bootstrap p < 0.10.
* **Passes final:** net return > 0 and Sharpe > 0. A hypothesis that passes all three is traded;
  everything else is shown on paper only.
* **Benchmark check:** H1–H4 must also beat the equal-risk long-only book of the nine over the
  same period (long-only bullion has been a strong decade; a long-biased signal must earn more
  than its bias).

Five hypotheses are tested, so a single pass at the validation stage is weak evidence. It is
reported as such.

## Changes after registration

(none)
