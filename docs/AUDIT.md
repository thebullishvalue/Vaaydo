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

## Coherence audit, 18 Nov 2026

Question: does every trade the system shows correspond to something that could actually have
been traded, in the contract shown, at the price shown, on the date shown?

**Method.** Every leg of every position the ledger has booked since Jul 2024 (1,361 positions,
2,200 legs) was checked against the exchange files: the contract was listed on the entry day
and printed on every day held; the fill is that contract's actual open (fills) or close (rolls);
the first leg is the rule's contract on the fill day; no leg is held past its expiry; roll
chains are continuous (exit at the old close, entry at the new close, same day); each leg's lot
is its contract's lot; entry is the session after the signal. Twelve past-dated runs were
compared with the full replay on the same date (positions, legs, contracts, prices, and that no
later data is read). The backtest simulator and the live ledger were compared day by day
(97 checkpoints).

**Found and fixed.**
1. *Tickets mixed contracts.* A held position showed its current contract (e.g. FUT Nov) next to
   the first leg's entry date and price (paid in, e.g., the June contract). Tickets now show what
   was first traded (contract, date, exchange price), the contract held now (since its last roll,
   at that roll's price), P&L across every leg, and the next roll. Closed trades show the entry and
   exit contracts. Leg tables show the actual exchange fill, and the adjusted equivalent after a
   split or bonus.
2. *Stocks leaving F&O.* When NSE listed no next contract (11 cases: PVRINOX, CHAMBLFERT, BSOFT,
   IRB, ABFRL, CESC, TATACHEM, HFCL, IRCTC, SYNGENE, SAMMAANCAP), both engines held the expiring
   contract past expiry; the ledger then "exited" with no price and the trade fell out of the
   statistics. Now: when the roll falls due and no later contract is listed, the position closes
   at that close, as a roll would.
3. *Contracts terminated early* (IDFC merger, GMRINFRA rename): closed at the last exchange print.
4. *Post-stop re-entry differed between the engines.* The ledger kept a stopped name out while its
   signal stayed in the band (the registered rule); the simulator built the band from its trading
   book and re-entered early. The simulator now tracks signal membership like the ledger. The
   two engines now hold identical books at all 97 checkpoints.
5. *Exchange holidays beyond the last session.* Sessions-to-expiry beyond the calendar counted
   plain weekdays, so a holiday inside the last four sessions (e.g. 25 Dec 2026) would roll a day
   late live and announce the wrong date. Both engines now count NSE sessions: past holidays from
   the files, announced ones in data/nse_holidays.csv, and holidays implied by NSE moving a listed
   expiry off its last Tuesday (24 Nov 2026, 26 Jan 2027).
6. *Rolls announced in advance.* A roll executes at the close three sessions before expiry, after
   which the evening run is too late to act. Tickets now show each position's next roll date and
   whether a next contract is listed, and the Signals tab lists the rolls due at the next close.
7. *Past-dated runs showed later performance.* The track record is now cut at the run's date.
8. *Precedent look-ahead (index research).* One session; fixed; verdicts unchanged.
9. *Interface.* Picking a position (Book), a scan (Contracts) or an index (desk) reran the page and
   returned to the first tab; these controls now rerun only their own block. Option contracts are
   labelled with their full expiry date.

**After the fixes.** 0 leg issues in 2,200 legs; 0 closed trades without a return; 12/12 past runs
identical to the full replay with no later data read; 97/97 checkpoints identical between engines.
Track record regenerated: model book Sharpe 0.80 (was 0.74), 20 names 0.74, 12 names 0.70,
6 names 0.24.
