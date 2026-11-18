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

## System audit, 18 Nov 2026 (every layer)

**Data.** (1) NSE answers a request for a holiday's equity file with the previous session's
file under the new date: 22 such copies sat in the equity store (doubling those days in delivery
features) and 23 fake sessions in the breadth history. Both fetchers now reject a file whose own
trade date is not the date asked for; the copies were removed. (2) 30 Mar 2021 is a real session
missing from NSE's own F&O archive (404): recorded as a known one-session gap, and removed from
the derived holiday list where it had been counted as a holiday. (3) Symbol renames (17 since
2019: ZOMATO → ETERNAL, MCDOWELL-N → UNITDSPR, SRTRANSFIN → SHRIRAMFIN, LTIM → LTM, …) are
detected from the futures file (same expiries, same lot, continuous price on consecutive
sessions) and mapped to one identity in every layer, live and in the backtest: positions are no
longer cut at the rename and the signal no longer restarts; past legs show the name they traded
under. (4) Pragyam's levels depended on where the computation started (median 0.34% of price in
stop distance, up to 3.7%): every run now starts at one fixed date, so a session's level is the
same however it was computed. (5) A regression had dropped the fill of Yahoo's late bars from
NSE's equity file: restored in the history loader, so no stock loses its level for a day.
Checked and clean: no duplicate rows in any shipped file; every F&O session has inputs, futures
and levels; no inverted stop ratios; no thin sessions; Yahoo-adjusted and NSE prices join without
jumps.

**Execution rules.** (6) Rolls during an F&O ban (7 cases): a roll opens the next contract, which
the ban forbids; the contract is now held and rolled on the first unbanned session, or settles at
expiry if the ban outlasts it. (7) Positions without a stop (5): a missing Pragyam level now falls
back to the last level within 5 sessions, else the day's median stop distance. (8) Capital-limited
sizing valued every past decision at today's contract size; it now uses each day's own price and
lot in the contract the order fills in.

**Presentation.** (9) Skew and call skew were shown with a % sign; they are differences of implied
volatilities and are now shown in vol points. (10) Index desk "basis" was the carry over the
risk-free rate; it is labelled so, and blanked inside 5 days of expiry. (11) Counts said "in the
book" while including tomorrow's unfilled entries; every count now says "after next open". (12)
The NIFTY trade evidence was typed into the page and had gone stale after a correction; it is now
read from the file the research script writes. (13) Margin is labelled as a rough guide.

**Operations.** (14) The nightly refresh never updated the index futures history the precedent
read draws its outcomes from: added. A replay of the refresh (drop the last session from a copy of
every shipped file, refresh, compare) now rebuilds all eight files exactly; the refresh warns by
name if any stock lacks a level. (15) The app cached "latest" for 6 hours, so a run before NSE's
evening file kept serving the previous session: the cache now turns over hourly.

**After the fixes.** 0 leg issues (2,198 legs, now also checking: no entry or roll during a ban,
no position without a stop); past runs identical to the full replay; simulator and ledger
identical at 97 checkpoints; the nightly refresh reproduces every shipped file. Track record
regenerated: model book Sharpe 0.76 (was 0.80) with a deeper worst drawdown, −17.4% (was −13.4%);
20 names 0.69; 12 names 0.80; 6 names 0.36. NIFTY verdicts unchanged on the cleaned data.
Reproduce: scripts/audit/{data,legs,asof,equiv,refresh}.py.

## Third pass, 18 Nov 2026: one lot per position, rupee reconciliation

**Capital sizing removed.** Every position is one lot; the book is the whole model book. The
sizing rail, the capital-limited replay and its helpers are gone.

**The track record now measures what is traded.** The backtest weighted every name equally in
rupees; one lot of each is not equal rupees. The simulator now also books every position at
its lot and reports the one-lot book's return on gross exposure and its rupee P&L. NSE's F&O
file before Jul 2024 has no lot size: lots for 2019 – Jul 2024 are estimated from traded value
and contracts on each contract's busiest day (within 1%: INFY 302 vs 300, SBIN 1,504 vs 1,500),
used only as weights (scripts/legacy_lots.py). Without them the first five years of the
one-lot book were silently zero.

**Found by reconciling rupees between the engines** (ledger legs vs simulator, ₹50.3 L of P&L,
Jul 2024 – Nov 2026): (1) on 1 Nov 2024 NSE revised the lots of six contracts already trading
(PETRONET 3,000 → 1,500, COFORGE 150 → 75, …) with no price change. A held position keeps its
shares; the simulator re-read the lot daily and booked the wrong quantity. It now carries the
quantity, changed only by a split/bonus or a roll; tickets show the quantity and, when NSE's
current lot differs, what it is in today's lots. (2) An exit decided at the previous close now
takes precedence over a same-day forced close of a stock leaving F&O (AARTIIND, Jul 2025).
After the fixes the engines agree to the rupee (difference 0) as well as in holdings.

**Found in the nightly refresh:** (3) the refresh appended Yahoo's still-forming bar for the
current day (220 rows dated the next session); every Pragyam computation is now capped at the
session being built. (4) Yahoo sometimes skips a day in the middle of a series while already
showing the next; the NSE fill only filled gaps at the end, so 31 stocks lost a level. The fill
now covers any missing session in the last 10 weekdays. (5) Renamed stocks are refreshed under
their current names.

**Holidays.** NSE's 2025–26 holiday circular dates (from the option-hub repo's table) match
VAAYDO's own derivation on all 29 dates; 2027 dates are carried as provisional and the refresh
warns until NSE's circular replaces them.

**Results.** One lot of every name: Sharpe 0.85, 7.1% a year on gross, worst drawdown −14.2%,
one losing year of eight; net long 12% of gross on average, beta 0.04, Sharpe 0.81 without it.
Equal rupees (reference): 0.76, −17.4%.
**Checks:** data audit clean; 0 leg issues in 2,198; past runs identical to the full replay
(stocks and index desk, nothing read after the run date); holdings identical at 97/97
checkpoints; rupee P&L identical; 30 invariants of the screen, tickets, book and index desk
hold; the nightly refresh rebuilds every shipped file exactly.
