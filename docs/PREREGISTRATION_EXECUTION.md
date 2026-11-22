# Pre-registration: execution of the F&O Stocks book (stops and fills)

Written 22 Nov 2026, before any of these variants was computed. The signal, the bands, the stop distance
(1.5 × Pragyam's level) and the one-lot book are NOT re-tuned here: only how an exit is triggered and at
what price an order fills. Changes after this point are listed at the bottom with the reason.

## Question

Does VAAYDO trade its proven book better with (S1) stops watched during the session, or (S2) orders
filled later in the session than the 09:15 open?

## Book and data

* The live book exactly (book.simulate with the live rules: enter 5% / stay 15%, trailing stop 1.5 ×
  Pragyam, ban after a stop, F&O ban, rolls at T−3, one lot each, costs as in the track record).
* Futures daily open, high, low, close from NSE's bhavcopy (each contract's own). Intraday prices of the
  underlying stock (5-minute bars) from Dhan, through the vault (Dhan data never leaves it).
* **Discovery: 8 Jul 2024 – 31 Dec 2025. Holdout: 1 Jan 2026 – 20 Nov 2026**, looked at once, only for a
  variant that passes discovery.

## Variants

| | Change from the live book |
|---|---|
| S0 | The live book (baseline): stop decided on the close, filled at the next open; every order at the open. |
| S1 | **Intraday stop.** The stop level (from closes up to the previous session, as now) is a resting order: if the session's low (long) or high (short) reaches it, the position exits that session at the stop level, or at the open if the open is already through it. |
| S2a | **Fill at the first 30 minutes' VWAP** (09:15–09:45) instead of the open, for entries and exits alike. |
| S2b | **Fill at the first hour's VWAP** (09:15–10:15). |
| S2c | **Fill at the session's VWAP.** |

For S2 the futures fill is the futures open scaled by the stock's own intraday move:
`futures_open × (stock_vwap_window / stock_open)` (the basis does not change within a session to first order).
Costs are unchanged; VWAP fills are not credited with lower slippage (conservative for S2).

## Test and pass rule

Daily one-lot net return of each variant against S0, on the same days.

* **Passes discovery:** Sharpe higher than S0's by at least 0.10, and the daily return difference positive
  with block-bootstrap p < 0.10 (blocks of 10 sessions).
* **Passes holdout:** the difference has the same sign and the variant's net return is not below S0's.
* A variant that passes both changes the live book (the change is announced in the app, with the evidence).
  Four variants are tested, so a pass at p < 0.10 is reported as moderate evidence, not proof.

## Changes after registration

(none)

## Results

### S1, intraday stop (22 Nov 2026): passes discovery narrowly, **fails the holdout**. The live book is unchanged.

| | S0 (live) | S1 (intraday stop) | |
|---|---|---|---|
| Discovery, Jul 2024 – Dec 2025: Sharpe | +1.29 | +1.41 | +0.12; bootstrap p 0.087: passes |
| Holdout, 2026 (looked at once): Sharpe / return a year | +0.19 / +1.2% | +0.05 / +0.4% | −0.84% a year: fails |
| Whole period: Sharpe, worst drawdown, rupees (one lot) | 0.86, −6.3%, ₹51.7 L | 0.87, −7.4%, ₹51.5 L | no difference worth a change |

Stops stay decided on the close and filled at the next open. (Research flag `intraday_stop` in book.simulate;
the live book does not use it.)

### S2, fills later in the session (22 Nov 2026): **all three fail discovery**. The live book keeps the open.

Intraday data: Dhan 5-minute bars of 274 stocks (every stock in the futures history since Jul 2024), reduced
inside the vault to one row per stock-session (162,069 sessions); opens equal NSE's official opens exactly.

| Discovery, Jul 2024 – Dec 2025 | Sharpe | Return / yr | vs S0 | bootstrap p | |
|---|---|---|---|---|---|
| S0, the 09:15 open (live) | 1.29 | 7.7% | | | |
| S2a, VWAP 09:15–09:45 | 1.17 | 7.1% | −0.12 | 0.80 | fails |
| S2b, VWAP 09:15–10:15 | 1.20 | 7.3% | −0.09 | 0.69 | fails |
| S2c, session VWAP | 1.50 | 9.2% | +0.21 | 0.112 | fails (p above 0.10) |

The holdout (2026) was not computed for any of them. Filling early in the session is worse than the open (the
move the signal predicts starts at the open). The session VWAP is the one that looks better, but it misses
the evidence bar, and it was one of four variants: it is not adopted. It may be **tracked forward** on paper
from 23 Nov 2026 (the vault records each session's VWAP), and re-registered only on that new data.
