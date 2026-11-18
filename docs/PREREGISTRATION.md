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

---

## Index desk: NIFTY and BANKNIFTY futures (registered 17 Nov 2026, before any index outcome was computed)

Instruments: NIFTY and BANKNIFTY near-month futures, rolled 3 sessions before expiry (index
futures are monthly). Decisions on the close of d, fills at the open of d+1. Outcome: the
roll-continuous futures return from the open of d+1 to the open of d+1+h, h = 1, 5, 10
sessions; **primary horizon 5**. Periods as for stocks: discovery 2019–2022, validation
2023 – Jun 2024, final Jul 2024 – Nov 2026.

Every reading is standardised without look-ahead: z = (x − mean) / sd over the previous
250 sessions of the same index (minimum 120). Test: pooled time-series regression of the
forward return on z (both indices stacked), Newey–West t with h lags, per period.

| # | Reading | Registered sign | Why |
|---|---|---|---|
| H1 | trend: sign of the 60-session futures return | + | time-series momentum in equity indices |
| H1b | Pragyam trend state T | + | the house indicator, tested on the index it was built for |
| H2 | vol timing: realised 5-day variance on iv_30² | slope > 0, and a 1/iv_30-scaled long beats plain long on Sharpe in each period | Moreira–Muir volatility-managed exposure |
| H3a | gex_dealer > 0 → next-5 realised vol / iv_front lower | − (vol) | dealers long gamma damp moves |
| H3b | 1-day return × sign(gex_dealer): next-day autocorrelation | − in positive gamma, + in negative | damping vs amplification |
| H4 | pcr_oi z | + | put writing is support (Indian convention) |
| H5a | FII index-futures long share z | + | the best-capitalised participant |
| H5b | Client index-futures long share z | − | the crowd on the other side |
| H6a | FII index-options bias z | + | |
| H6b | Client index-options bias z | − | |
| H7 | max-pain pull: max_pain_d at 2 sessions before a weekly/monthly expiry vs the spot return to expiry | + | writers defend the strike that pays them most |
| H8 | hedge_flow z (writers' delta hedge of the day's OI change) | + | hedging demand pushes the future |
| H9a | VRP: iv_30 − realised 20-day vol, z | + | a fat premium pays the holder of risk |
| H9b | India VIX z | + | fear is followed by recovery |
| H10 | basis z | two-sided (exploratory) | |
| H11 | skew z (IV 3% below − 3% above) | + | fear premium |
| H12 | stock breadth: share of stock futures in long build-up minus short build-up over 5 sessions | + | participation |

**Acceptance (directional readings):** registered sign with t ≥ 1.0 in all three periods
**and** |t| ≥ 2.0 over 2019–2026 at h = 5. H3 and H2 are judged on volatility, not
direction, and may enter only as sizing.

**Strategies, judged on net Sharpe per period and worst drawdown, against S0:**
S0 long the future, always (the benchmark: being long Indian equities has paid);
S1 S0 scaled by 0.15 / iv_30 (cap 2×); S2 trend (H1) long/short, scaled as S1;
S3 S2 plus accepted positioning readings, equal-weighted z, no fitted weights.
The simplest strategy that is not beaten in any period is the one that trades.
Costs: 0.03% per side (STT, exchange, stamp, GST, brokerage, a tick of slippage); an
exposure change smaller than a quarter of the target is not traded.

### Index desk, round 1 results (17 Nov 2026), recorded as computed

No directional reading met the acceptance rule at h = 5 (t: discovery / validation / final / all):
trend 60d 0.70/−0.87/−0.67/0.42; Pragyam trend 0.37/−2.85/−1.23/−0.42; PCR OI 1.56/−0.75/1.18/1.59;
FII futures 1.83/−1.69/−0.53/1.36; Client futures −1.76/1.15/0.56/−1.35; **FII options bias
3.20/−0.58/0.42/3.20** (fails validation); Client options bias −2.60/0.52/−0.84/−2.92; hedge flow
−1.36/0.38/−1.19/−1.61 (wrong sign); VRP 0.08/−2.06/−0.76/−0.80; VIX −0.90/1.53/1.01/−0.41; basis
0.27/0.45/−1.91/0.05; skew −0.90/0.27/0.68/−0.67; stock breadth 0.90/0.12/−0.56/0.66.
H2: implied vol forecasts realised vol (t 4.6/1.6/6.1). H3a: positive dealer gamma goes with
lower realised/implied (0.835 vs 0.887), t −1.8, not enough. H3b null. H7 max-pain pull: spot
moved toward max pain into expiry 47.5% of 737 expiries; rejected.
Strategies (50/50 NIFTY/BANKNIFTY, net SR disc/val/final/all): S0 long 0.48/0.66/−0.10/0.36;
S1 vol-managed 0.38/0.79/−0.30/0.25; S2 trend 0.27/0.29/−0.21/0.14. **No index futures
strategy is adopted.** Realised 5-day vol averaged 84% of front implied in every period: the
volatility risk premium is the one robust regularity in the index data.

## Index desk, round 2: the volatility risk premium (registered 17 Nov 2026, before testing)

Trade: a defined-risk short-premium position on each index's options, every expiry cycle.

* **Entry**: the open of the first session at which the nearest expiry has 5 or fewer sessions
  left (≥ 2), one position per expiry. Strikes from the previous close: forward F, ATM IV σ,
  T to expiry; expected move m = F·σ·√T.
* **Structure (registered: iron condor)**: sell the call nearest F + 0.75m and the put nearest
  F − 0.75m; buy the call nearest F + 2m and the put nearest F − 2m. Max loss defined.
  Comparison, not for adoption unless it dominates: iron fly (sell ATM, wings at ±1.5m).
* **Exit**: held to expiry, settled at NSE's final settlement price (the expiring option's
  settle). No discretionary stop.
* **Fills and costs**: each leg at its open on the entry session; a leg with no trade at the open
  uses the previous close plus a half-spread; half-spread max(₹0.5, 1% of premium) per leg;
  STT on sold premium (0.0625% to Sep 2024, 0.1% after), exchange 0.035%, GST 18% on charges,
  stamp 0.003% on bought premium, ₹20 a leg.
* **Measure**: P&L per position as % of max loss (the capital at risk), per period; Sharpe of
  the weekly series; worst cycle; worst drawdown.
* **Conditioning, registered with signs**: C1 iv_30 / rv20 above its 250-session median →
  better; C2 gex_dealer > 0 → better (H3a direction); C3 skip when front IV is in its top decile
  of 250 sessions (stress). Each adopted only if it improves mean P&L in all three periods.
* **Adoption**: mean P&L per cycle after costs > 0 in all three periods, and the worst cycle no
  worse than −100% of max loss by construction. Otherwise the index desk ships analytics only.

### Index desk, round 2 results (17 Nov 2026), recorded as computed

Settlement check: 16,146 expiring-option rows in 2019–Feb 2020 carry no settle price; those
cycles settle at the index close instead (found when a cycle settled at 0).
Mean P&L per cycle as % of max loss, after costs (n; t):
iron condor: NIFTY −2.9% / −4.6% / −5.9% (disc/val/final), all −4.2% (402; −2.2);
BANKNIFTY −6.0% / −6.2% / −10.4%, all −6.7% (323; −2.8). Wins 59–68% of cycles; the losses
are larger. Iron fly: NIFTY +2.0% / −8.9% / −12.3%; BANKNIFTY −2.9% / −5.1% / −7.8%.
Conditioning: C1 (IV/RV above median) and C2 (positive dealer gamma) do not make any period
positive; C3 (skip top-decile IV) makes it worse: selling only in the top decile was −2.1% /
+6.7% / +7.7% (condor), the opposite of the registered sign, on small samples; not adopted.
**Round 2 rejected.** The 16% gap between implied and realised 5-day vol does not survive
next-open fills, the tails, and costs on these structures.

**Decision:** the index desk trades nothing on 2019–2026 evidence. It ships as the index
market read (positioning, volatility, participants, futures and rollover, levels) with every
test's verdict on screen, and forward-tracks two candidates on paper only, without capital:
the FII index-options bias (best in discovery, failed validation) and the condor sold only in
top-decile IV (C3 reversed). Either is promoted only if it earns it out of sample from 18 Nov 2026.

## Stocks: gamma and hedge flow (registered 17 Nov 2026, before computing)

Per stock and session, from the same chain arithmetic as the index desk: `gex_dealer`
(dealers long calls, short puts; Σγ·OI·S²·1% over options ≤ 60 days, per unit of futures OI
notional) and `hedge_flow` (writers' delta hedge of the day's option OI change, same units).
Cross-sectional Fama–MacBeth ICs on futures outcomes, discovery 2019–22 / validation
2023–24H1 / final 24H2–26, 10-session means like the score.
* SG1 `hedge_flow`: sign **−** (put-side OI growth precedes outperformance, as in the score's
  option family); adopt into the score only if same-sign t ≥ 1 in all periods, |t| ≥ 2 overall
  at 5 or 20 sessions, **and** the book's net Sharpe improves in all three periods.
* SG2 `gex_dealer`: volatility, not direction: higher gamma → lower next-20-session realised
  volatility relative to ATM IV, cross-sectionally. If it holds (t ≥ 2 overall, same sign in
  every period), it may scale the trailing stop; it is not a directional input.
* F&O ban: a stock in the exchange's ban period cannot take a fresh position. Entries on a
  banned fill day are deferred until the ban lifts (implementation, not a hypothesis).

### Stocks: gamma and hedge flow, results (17 Nov 2026), recorded as computed

10-session means, daily rank IC on futures outcomes (t; disc / val / final / all), h = 5:
`hedge_flow` −0.019 (−2.28) / −0.011 (−1.03) / −0.029 (−3.14) / −0.020 (−3.76): passes the IC
rule with the registered sign (h = 10: −1.59 / −0.95 / −2.56 / −2.90). Rank correlation with the
score's own option-flow input −0.47. **Book gate:** the score with −hedge_flow as a seventh
equal-weight component, model book with the F&O ban gate, net Sharpe 0.40 / −1.15 / 1.06 / 0.34
against 1.20 / −0.94 / 0.73 / 0.74 now, worst drawdown −25.5% against −13.5%. **Not adopted.**
`gex_dealer` (SG2, volatility): IC with log(forward 20-day realised / ATM IV) −0.005 / +0.025 /
−0.005 / +0.001: rejected. Observed, not registered as direction and not adopted: `gex_dealer`
as a directional rank had IC t −3.61 / −0.80 / −2.78 / −4.43 (fails validation).
(Horizon 20 is not available on the 2019–24 store; 5 and 10 were used.)
F&O ban gate adopted (implementation): model book 0.78 → 0.74, 20 names 0.77 → 0.75,
12 names 0.71 → 0.71, 6 names 0.31 → 0.27.

## NIFTY breadth and regime workbook (registered 18 Nov 2026, before any outcome was computed)

Source: a user-supplied NIFTY workbook (daily, Jun 2006 – Nov 2026): market-wide NSE
advance/decline ratio (verified against NSE's equity bhavcopy: it counts all NSE stocks, not
NIFTY 50), NIFTY close, PE/PB/DY, VIX, rates and macro. Its derived columns are recomputed
causally from the raw inputs, with the workbook's own formulas, except:
* the fair-PE regression is fitted on an expanding window (the workbook fits it on all of
  2006–2026, so its PE_DEV, REGIME and SIGNAL_STR carry look-ahead), as a true multiple
  regression on the workbook's selected variables (it sums univariate slopes), excluding
  COUNT (selected despite its changelog), on PE adjusted for NSE's 31 Mar 2021 switch to
  consolidated earnings (pre-switch PE × 33.20 / 40.43, the day's ratio net of the index move);
* stale NIFTY prints are replaced by NSE closes where they disagree by more than 0.2%.

Outcomes: (a) NIFTY spot excess return over repo, close d → close d+h, 2006–2026, in three
periods P1 2006–2012, P2 2013–2018, P3 2019–2026; (b) NIFTY futures from the next open as in
the index desk (2019–2026, discovery / validation / final). Newey–West t, h lags; z-scores over
the previous 250 sessions. Primary horizon 5; valuation readings at 20.

| # | Reading | Sign | Why |
|---|---|---|---|
| W1 | BREADTH (10-day EMA of advancing share) | + | broad participation |
| W2 | REL_BREADTH | + | the same, smoothed across horizons |
| W3 | REL_AD_RATIO | + | breadth momentum |
| W4 | REGIME (causal) | + | the workbook's composite |
| W5 | SIGNAL_STR (causal) | + | as the workbook reads it: buying pressure |
| W6 | BTD events | + (next 10 sessions vs all days) | oversold bounce |
| W7 | STT events | − | distribution at the top |
| W8 | SPREAD flag (above 90- and 200-day MA by 1.5%) | + | trend |
| W9 | PE_DEV (causal), h = 20 | − | cheap beats dear |
| W10 | earnings yield − IN10Y, h = 20 | + | equity risk premium (Fed model) |
| W11 | IN_YC (10y − 2y), h = 20 | + | curve steepness, growth |
| W12 | COUNT (OSC − its 50-day mean) | + | the workbook's momentum count |

Accept a reading if: registered sign with t ≥ 1 in P1, P2, P3 **and** |t| ≥ 2 on 2006–2026
spot, **and** same sign with t ≥ 1 on 2019–2026 futures. Events (W6, W7): mean 10-session
excess return after the event minus the unconditional mean, same rules.
Strategies against S0 (long NIFTY futures always), net of 0.03% a side, 2019–2026 futures and
2006–2026 spot: A1 long when causal REGIME > 0, flat otherwise; A2 long when BREADTH is above
its 250-session median, flat otherwise. Adopted only if net Sharpe beats S0 in every period.

### Workbook results (18 Nov 2026), recorded as computed

Fidelity of the causal rebuild against the workbook (2008 on): breadth, rel_breadth, rel_ad,
count correlation 1.000; RSI 0.9999; REGIME 0.78 and SIGNAL_STR 0.91 (the difference is the
workbook's look-ahead fair PE). PE switch factor 0.830. Causal fair PE exists from Jul 2015
(inflation series starts 2013). 7 stale NIFTY prints replaced. VAAYDO's own NSE advance/decline
count from the equity bhavcopy matches the workbook's AD_RATIO at 0.978 (daily) and 0.998
(10-day EMA of the advancing share), Jul 2024 – Nov 2026.

t by reading (spot P1 / P2 / P3 / all; futures disc / val / final / all):
W1 breadth 0.47/0.03/1.36/1.19; 1.63/−0.57/−0.57/1.33 · W2 rel_breadth 0.84/0.18/1.19/1.40;
1.48/−0.56/−0.88/1.13 · W3 rel_ad 0.13/1.22/1.57/1.41; 0.58/0.89/0.95/1.20 · W4 regime
0.86/−1.46/0.70/0.63; 0.91/−1.66/0.05/0.76 · W5 signal_str —/−0.64/0.18/−0.03; 0.38/−1.29/−0.03/0.25 ·
W8 spread flag 2.46/−0.13/0.67/2.17; 0.18/−0.74/0.76/0.53 · W9 pe_dev (h 20, sign −)
—/0.05/−0.80/−0.75; −0.88/−1.46/−0.23/−0.74 · W10 EY gap −0.24/0.45/0.81/0.45; −0.12/7.74/1.41/0.90
(validation is 18 months of a slow variable: one observation, not evidence) · W11 yield curve
1.86/−0.05/0.76/1.78; 1.61/0.93/0.59/0.71 · W12 count 1.66/−1.10/1.14/1.50; 0.46/1.42/0.87/1.09.
**No reading accepted.**
Events, 10-session excess return after the event minus all days: BTD −0.38% / −0.18% / −0.69%
(214 events, t −0.68; futures −0.40%, t −0.42): buying the workbook's dips did worse than any
day. STT +2.47% / +0.03% / +0.13% (86 events, t +1.47): returns after "sell the top" were
higher, the opposite of the registered sign. Both rejected.
Strategies, net Sharpe (spot P1 / P2 / P3 / all; futures disc / val / final / all):
S0 long 0.34/0.31/0.41/0.34; 0.58/1.15/−0.22/0.45 · A1 long when regime > 0 0.85/0.12/0.29/0.49;
0.86/0.33/−0.31/0.47 · **A2 long when breadth is above its 250-session median 0.72/0.31/0.90/0.66;
1.33/1.00/−0.37/0.83**, in the market 49% of days. A2 beats or ties S0 on 2006–2026 spot in
every period but loses to it on futures in validation and final: **not adopted**; added to the
paper track from 18 Nov 2026, with the FII options stance and the high-IV condor.

## NIFTY, Nishkarsh round (registered 18 Nov 2026, before any of these was computed)

Nishkarsh (the owner's NIFTY research system) independently tested the same workbook and
reached the same verdicts VAAYDO did: the fair-PE levels model is falsified (its signal
inverted), macro inputs degrade it, and its predictive lead did not replicate on purged code.
What VAAYDO takes from it is its inference standard and two ways to combine weak readings
without fitting weights. All tests on NIFTY only; outcome = NIFTY futures from the next open
(2019–2026) unless stated; h = 5 primary, 10 secondary.

**Inference (adopted for these tests):** IC = mean of the h phase-wise Spearman ICs on
non-overlapping h-spaced rows; Fisher t; Newey–West(h) t of the daily rank IC; 95%
stationary block bootstrap CI (mean block 2h, seeded); yearly IC; the label's own 20-day
momentum as a baseline; and the minimum detectable IC at 80% power, one-sided 5%.
**Acceptance (Nishkarsh's rule):** HAC t ≥ 2, bootstrap CI excluding 0, ≥ 70% of years
positive, IC above the momentum baseline; and, as a trade, a net Sharpe above always-long in
every period with break-even cost ≥ 2× the 0.03% assumed.

* **N1 re-grade** (descriptive, changes no earlier verdict): market breadth, A/D momentum, the
  breadth rule A2, FII options stance; reported with their minimum detectable IC, so "no
  edge" and "too little data to tell" are distinguished.
* **N2 composite**: the equal-weight mean of every reading registered with a sign in rounds 1
  and W (each z-scored over 250 sessions, sign applied as registered): trend 60d, Pragyam
  trend, PCR OI, FII futures, −Client futures, FII options stance, −Client options stance,
  hedge flow, VRP, VIX, skew, stock breadth, market breadth, relative breadth, A/D
  momentum, ±1% count. No weights fitted, no reading chosen by its result.
* **N3 precedent**: state = six family z-scores (60-day trend, market breadth, PCR OI, FII
  options stance, 30-day IV, dealer gamma); for each day the 10 nearest past states by
  Mahalanobis distance (OAS-shrunk covariance of the candidate pool), candidates only where
  the h-forward outcome is realised, Theiler gap h; read = mean forward return of the
  analogs minus the candidate pool's mean. First read after 500 candidates.
* **N4 agreement**: two independent families, crowd/participant positioning (FII options
  stance z) and market participation (breadth z): read = their common sign when they agree,
  0 otherwise (Nishkarsh's signed-agreement idea, L-12, with VAAYDO's families).

### Nishkarsh round results (18 Nov 2026), recorded as computed

NIFTY futures 2019–2026, h = 5 (h = 10 in brackets). IC / n / minimum detectable IC / HAC t /
95% bootstrap CI / years positive / momentum baseline IC:
N1 market breadth 0.025 / 364 / 0.134 / 0.59 / [−0.06, +0.10] / 62% / 0.061 · A/D momentum
0.012 / 0.60 · A2 rule 0.024 / 0.63 · **FII options stance 0.155 / HAC 3.97 / [+0.08, +0.23] /
62% of years** (fails the 70% rule; period ICs +0.199 / −0.035 / +0.053) · N2 composite 0.088 /
2.12 / [−0.001, +0.165] / 62% · **N3 precedent 0.106 / n 251 / MDE 0.161 / HAC 2.56 / [+0.03,
+0.19] / 83% of 6 / baseline 0.002: passes the statistical rule** (h 10: 0.110, 2.07, 67%, fails)
· **N4 agreement 0.102 / 2.69 / [+0.02, +0.18] / 50%** (h 10: **0.122 / 2.44 / [+0.02, +0.22] /
75% / baseline 0.077: passes**).
Trades, tranche book h = 5, net Sharpe disc / val / final / all, break-even one-way cost:
always long +0.59 / +1.15 / −0.25 / +0.45 · A2 +1.19 / +0.38 / −0.46 / +0.67, 38 bp ·
N2 +0.65 / +0.25 / +0.71 / +0.60, 48 bp · N3 +0.33 / −0.77 / +1.23 / +0.49, 18 bp ·
N4 +1.20 / −0.23 / +0.19 / +0.77, 41 bp. **None beats always-long in every period** (all lose in
validation, 2023–mid 2024, a one-way rise): **no NIFTY trade adopted.**
Reading: the combinations (N2–N4) all contain the FII options stance, so they are not
independent confirmations of one another. Single readings' ICs (≈ 0.01–0.03 for breadth) are
far below the 0.13 this history can detect: "rejected" there means unproven at this sample size.
N3 and N4 join the paper track; the desk shows their state daily.
Rerun on NSE-only data (18 Nov 2026): breadth rebuilt from NSE's equity bhavcopy (advances /
declines of EQ-series stocks, Jun 2006 on, 5,073 sessions) and daily index-close file (NIFTY
close, P/E, P/B, dividend yield, 2013 on). Against the workbook: NIFTY, P/E, P/B, DY identical
(correlation 1.0000 on 2,704 sessions), A/D 0.984, 10-day breadth 0.993; the workbook's 20
dates absent from NSE are weekends. Verdicts unchanged: N3 precedent h 5 IC 0.100, HAC 2.30,
CI [+0.012, +0.181], 83% (passes); h 10 IC 0.126, HAC 2.43, CI [+0.019, +0.228], 83% (passes);
N4 agreement h 10 IC 0.120, HAC 2.42 (passes); trades disc / val / final: A2 +1.16 / +0.60 /
−0.40, N2 +0.56 / +0.20 / +0.63, N3 +0.16 / −1.23 / +1.40, N4 +1.20 / −0.11 / +0.17 against
always-long +0.59 / +1.15 / −0.25. No NIFTY trade adopted. The desk now uses only NSE files.
Correction (coherence audit, 18 Nov 2026): the precedent engine admitted candidates whose
outcome needed the next session's open (one-session look-ahead in the backtest; the live read
was unaffected). Fixed to candidates realised by the decision close. Rerun: N3 h 5 IC 0.099,
HAC 2.29, CI [+0.011, +0.180], 83% (passes); h 10 IC 0.131, HAC 2.52 (passes); trade
−0.01 / −1.14 / +1.41 (all +0.38): still loses to always-long in two periods. Verdicts unchanged.
