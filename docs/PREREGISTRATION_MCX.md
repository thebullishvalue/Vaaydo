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

**22 Nov 2026, data cleaning, decided from the return-series audit and before any signal or strategy
return was computed:**
1. A daily return beyond ±30% that reverses (sum within ±5%) inside three sessions is a data fault:
   both days are set to zero (one case: COPPER, 17–20 Jun 2019, +166% / −158%, at the change from
   two-monthly to monthly contracts).
2. A commodity-year in which fewer than 60% of the expected expiries are detected is excluded for that
   commodity (NICKEL 2023, 2024, 2025: 2, 1 and 4 of 12; the contract barely traded). Its positions are
   flat in those years.
3. Real extreme days are kept: crude in March–April 2020 (its negative front settlement never enters, as
   the book holds the second contract), nickel's March 2022 squeeze, the March 2020 base-metal moves.

## Result (22 Nov 2026): no hypothesis survives discovery

Discovery, 2013–2018, monthly net returns of the equal-risk book of the nine (rule: Sharpe > 0.3 and NW t ≥ 2):

| | Return / yr | Sharpe | NW t | |
|---|---|---|---|---|
| H1 trend 12-1 | −1.6% | −0.16 | −0.43 | fails |
| H2 trend 3-month | +0.4% | +0.04 | +0.13 | fails |
| H3 carry | +2.2% | +0.25 | +0.58 | fails |
| H4 OI build-up | +6.7% | +0.52 | +1.73 | fails (closest; t below 2) |
| H5 cross-sectional momentum | +0.5% | +0.06 | +0.13 | fails |
| Long-only (benchmark) | −8.1% | −0.56 | −1.50 | |

Validation (2019–2022) and the final period (2023–2026) were **not computed**: the gates are enforced in code,
and nothing reached them. They remain unseen.

Engine check: the held-contract return of each commodity trails its front price by the cost of carry, as a
rolled futures position must (2013–18: gold −11% vs +2%, silver −60% vs −33%, crude −71% vs −37%, natural gas
−66% vs +15%: INR futures sit above spot by roughly Indian interest rates, more in natural gas's steep
contango). The long-only loss is that carry plus 2013–15's commodity bear market.

**Consequence:** VAAYDO gives no commodity trades. The Commodities universe is a desk (term structure, open
interest, the option chain, the daily record) with the five readings tracked on paper, as the Indices desk
does. A hypothesis may be re-registered only with a new idea, never by retuning these five.
