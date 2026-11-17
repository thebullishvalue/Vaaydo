# VAAYDO (वायदो)

**An F&O trade screener for NSE.** Pick a universe and a date, press Run: every NSE F&O
stock is screened on where its options market is positioned, and the stock-futures
trades that follow come out as tickets: what to enter, exit and hold at the next open,
in which contract, at what price, with a stop.

> Long the stocks where the options crowd leans hardest to puts; short the ones where
> it leans hardest to calls; equal rupees each side.

## What a run returns

| Tab | |
|---|---|
| **Signals** | New ▲ / ▼ signals, exits, and the open book as tickets: contract with its expiry date, lot, signal date, entry price and date, last, P&L, trailing stop and its distance, risk to stop, Pragyam's 80% range. |
| **Book** | Every position leg by leg: the contract it was entered in, each roll (with its own lot size), split/bonus adjustments, realised and open P&L; exposure by industry. |
| **Watchlist** | Names in the 5–15% band moving toward an entry, nearest first; names stopped out and waiting. |
| **Flow** | Breadth, futures build-up (day and five sessions), sectors, option flow (buyers vs writers opening, unusual volume, call/put volume surge). |
| **Contracts** | The option-contract scanner: most active, OI gainers/losers, buyers/writers opening, premium gainers/losers, volume above OI. |
| **Screen** | Every F&O stock with every reading: score, put/call ratios, skews, IV, event flag, OI walls, build-up, delivery %, Pragyam regime. |
| **System Data** | Track record 2019 – today, every closed trade since Jul 2024, exports, definitions. |

**Historical Range** mode returns every closed trade whose signal fell in the range,
month by month, and the track record over it. Sizing to your capital is optional: the
screen is the same either way, and sizing only limits the book to the most extreme
names that fit, long and short balanced.

## The evidence, briefly (docs/RESEARCH_V6.md)

| Book (1× gross, net of costs, 2019 – Nov 2026) | Sharpe | Return / yr | Volatility | Worst drawdown |
|---|---|---|---|---|
| model book, ~30 names | 0.78 | 5.8% | 7.4% | −13.5% |
| 20 names | 0.77 | 6.4% | 8.4% | −13.0% |
| 12 names | 0.71 | 7.3% | 10.4% | −16.4% |
| 6 names | 0.31 | 4.5% | 14.6% | −25.3% |

A modest, market-neutral edge. Below ~12 names it is mostly stock-specific noise.
Designed on 2024–25 and confirmed on 2019–22, which no design choice had seen. Flat in
2023 – mid-2024. Expect losing years.

## How the book is kept honest

* Signals are decided on a session's close; fills are at the **next open**, in the
  contract that is actually active on the fill day.
* Contracts roll at the close **3 sessions before expiry**, counting expiries as NSE
  lists them (holiday-shifted expiries included). Each leg keeps its own contract's lot.
* Stops are decided on the close and filled at the next open; no intraday look-ahead.
* Splits, bonuses and demergers rescale entry, stop and lot by the exchange's factor,
  confirmed by the price jump, so P&L is continuous and lot revisions aren't mistaken
  for corporate actions.
* The same engine produces the backtest, the live book and the tickets (checked equal
  at 58 dates).

## Run it

**Streamlit Cloud:** point the app at `app.py`.

**Locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

`data/` ships the inputs since Jul 2024 (option-positioning inputs, futures prices,
Pragyam levels, sectors) so a run takes seconds. `.github/workflows/refresh.yml` appends
each evening's session (`scripts/refresh_bundle.py`); without it, the app fetches
missing days itself.

## Layout

```
app.py                  the screener (Streamlit, Graphite design system in ui/)
vaaydo/
  curator.py            one run: data, scores, the book replay, flow, the screen
  ledger.py             positions as chains of contract legs: fills, rolls, stops, corporate actions
  signal.py             Option Positioning Index, smoothing, model-book membership
  features.py           per-stock option/futures readings from one day's bhavcopy
  flowscan.py           contract-level option flow: buyers/writers opening, unusual, surge
  session.py            recent sessions, futures build-up, delivery
  bundle.py · universe.py   shipped data; index universes
  pragyam.py            Pragyam risk engine (thebullishvalue/Indicator), verbatim
  data/                 NSE F&O bhavcopy (UDiFF + legacy), equity bhavcopy, Yahoo history
  alpha.py · simulate.py · book.py · longtest.py · optexpr.py   the research that chose the signal
docs/  VISION.md · METHOD.md · RESEARCH_V6.md · PREREGISTRATION.md · RESEARCH.md · AUDIT.md
```

## Terms

Risk levels: **Pragyam** (thebullishvalue/Indicator). NSE data is subject to NSE's
terms. Decision support, not investment advice. Futures can lose more than the margin
posted.
