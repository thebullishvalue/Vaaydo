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
