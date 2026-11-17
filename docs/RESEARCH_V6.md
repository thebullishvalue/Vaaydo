# Research log, v6: from data to a tradeable signal

Everything below is reproducible from the code in `vaaydo/` (features, alpha, simulate,
book, longtest, optexpr) on NSE bhavcopies Jan 2019 – Nov 2026. The protocol and every
change to it are in `docs/PREREGISTRATION.md`, in the order they happened, including
the failures.

## 1. What was tested

42 end-of-day features per F&O stock per session (124k stock-days 2024-26, 236k in
2019-24): the option surface (ATM IV, skew, call-put IV spread, term structure), option
volume/OI by side and flows, max pain, OI walls, futures basis, OI and build-up,
rollover, delivery %, VWAP gap, and price features including all of Pragyam's (trend,
surprise, risk regime, risk momentum). Outcome: next-open to close h sessions later,
cross-sectionally demeaned, vol-scaled. Daily rank IC, Newey-West t.

## 2. The path, including the dead ends

| Step | Result |
|---|---|
| Discovery Jul-24 – Dec-25, spot returns | 25 feature-horizons with abs(t) >= 3; basis and put/call volume strongest |
| Holdout 1 (2026), spot | composite IC t = 3.1, monotone quintiles: **passed** |
| Same trades on real futures prices | +0.06% per pair, t = 0.3: **basis is carry**, the futures holder pays it away |
| Re-selection on futures outcomes | basis IC 0.000; the option-positioning family survives |
| Holdout 2 (2026), futures | IC t = 1.0, pair t = 0.3: **failed** |
| Long test 2019 – Jun 2024 (never used) | trade-by-trade pair +0.03%, t = 0.2: **failed** |
| ...but its IC on 2019-22 | pcr_vol t = 4.9, pcr_oi 3.6, skew 3.5, cskew -3.5: the information **persists**, at half strength |
| Gross vs cost | 0.29% vs 0.25% per pair: the edge was being spent on churn (daily rank autocorrelation 0.46) |
| 10-session smoothing + hysteresis book (fixed on 2019-22) | Sharpe 0.88, every year positive |
| Validation 2023 – Jun 2024 | Sharpe ~0: **flat** |
| Factor timing by trailing IC | hurt (0.51 vs 0.77): rejected |
| Debit spreads as the instrument | -6% to -16% of premium per trade in every period: rejected |

## 3. Where it ended

* **Option positioning predicts the cross-section of F&O stock futures returns.** Its
  composition was chosen on 2024-25; on 2019-22 (unseen) its IC had t ≈ 5. The sign held
  in every period; the strength did not (≈0 in 2023 – mid-2024).
* **Implemented as a smoothed, hysteresis futures book** with a 1.5x trailing stop it
  earned, 2019 – Nov 2026, net: model book (~30 names) Sharpe 0.77, 5.8%/yr, worst
  drawdown -13.5%; 20 names 0.76; 12 names 0.66, 7.0%/yr, -18.9%; 6 names 0.30, -25.9%.
* **Correction (18 Nov 2026).** Earlier versions of these figures (Sharpe ~1.0 / 0.89 /
  0.55) came from a simulator that decided a stop on a day's close but exited at that
  same day's open: look-ahead. Fixed in `book.simulate` (stop decided on the close of d,
  exit at the next open), with a regression test in tests/test_ledger.py. Every figure
  above is from the fixed simulator.
* **Not found:** any edge from price charts (Pragyam's and Sanket's own findings,
  re-confirmed cross-sectionally), futures build-up, OI walls, max pain, basis (after
  carry), volatility risk premium, or any option-selling structure (docs/RESEARCH.md).

## 4. Risks a trader should hold in mind

* **It is intermittent.** Expect losing stretches of a year or more.
* **Shorts lean to cheap, retail-favoured names** (often PSUs). A melt-up in those, like
  2023, hurts the short side.
* **Many looks were taken at 2024-26.** The cleanest evidence is the IC on 2019-22 and the
  flat 2023-24 validation; weigh those over the full-period backtest.
* **Fills are modelled** (0.126% round trip, liquidity-scaled slippage, next open).
