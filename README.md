# VAAYDO (वायदो)

**An F&O trade screener for NSE.** Pick a universe and a date, press Run: every NSE F&O
stock is screened on where its options market is positioned, and the stock-futures
trades that follow come out as tickets: what to enter, exit and hold at the next open,
in which contract, at what price, with a stop.

> Long the stocks where the options crowd leans hardest to puts; short the ones where
> it leans hardest to calls; one lot of each.

## Two universes

* **F&O Stocks**: the trades. Every NSE F&O stock screened on option positioning; the
  stock-futures book as tickets. Index-constituent lists filter it.
* **Indices · NIFTY & BANKNIFTY**: the index desk. Option chain (OI by strike, max pain,
  walls, dealer gamma and the gamma flip, hedge flow), volatility (IV term structure, smile,
  skew, priced move, India VIX), futures (basis, roll cost, rollover against past expiries),
  participants (FII / DII / Pro / Client), and the evidence. **It gives no index trades:**
  every index hypothesis VAAYDO registered — sixteen directional readings, trend and
  volatility-managed futures, an iron condor and an iron fly on real fills — failed on
  2019–2026 (docs/PREREGISTRATION.md). Market breadth (NSE advance/decline, 2006 on) and
  the owner's Nishkarsh standard of inference were added: combining weak readings
  without fitted weights (nearest-past-state precedent; FII stance × breadth agreement)
  passes the statistical bar, but no combination beat simply holding NIFTY in 2023–24,
  so none trades. Five candidates are tracked on paper and shown daily.

## What a run returns (F&O Stocks)

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
month by month, and the track record over it. There is no capital sizing: every
position is one lot of its futures contract, and the book is the whole model book.

## The evidence, briefly (docs/RESEARCH_V6.md)

| Book (~30 names, on gross exposure, net of costs, 2019 – Nov 2026) | Sharpe | Return / yr | Volatility | Worst drawdown | Losing years |
|---|---|---|---|---|---|
| **one lot of every name** (what VAAYDO trades) | 0.85 | 7.1% | 8.3% | −14.2% | 1 of 8 |
| equal rupees in every name (research reference) | 0.76 | 5.5% | 7.3% | −17.4% | 2 of 8 |

Lots differ in rupee size, so the one-lot book is not exactly balanced: it has averaged 12%
net long of gross (−4% to +28%), with a beta to NIFTY of 0.04; with that removed its Sharpe
is 0.81. A modest edge from stock selection, not from the market. (2019 – Jul 2024 lot sizes
are estimated from NSE's traded value and contracts, within 1%: NSE's file then had none.)
Designed on 2024–25 and confirmed on 2019–22, which no design choice had seen. Flat in
2023 – mid-2024. Expect losing years.

## How the book is kept honest

* Signals are decided on a session's close; fills are at the **next open**, in the
  contract that is actually active on the fill day.
* Contracts roll at the close **3 sessions before expiry**, counting expiries as NSE
  lists them (holiday-shifted expiries included). Each leg keeps its own contract's lot.
* Stops are decided on the close and filled at the next open; no intraday look-ahead.
* No fresh position in a stock during its **F&O ban** (open interest above 95% of the
  market-wide limit): the entry waits until the ban lifts. Tomorrow's ban list is read the
  evening before and shown on the screen.
* Splits, bonuses and demergers rescale entry, stop and lot by the exchange's factor,
  confirmed by the price jump, so P&L is continuous and lot revisions aren't mistaken
  for corporate actions.
* The backtest simulator and the live ledger hold identical books (checked at 97 dates),
  and every leg ever booked is checked against the exchange files (docs/AUDIT.md).
* A stock leaving F&O (no next contract listed) is closed at its due roll; a contract that
  stops printing early (merger) at its last print. A renamed stock (ZOMATO → ETERNAL and 16
  others since 2019) is one continuous position and signal. No entry and no roll during an
  F&O ban. Rolls count NSE sessions,
  holidays included, and are announced on the ticket before they happen.

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
app.py                  the screener (Streamlit, Graphite design system in ui/; ui/desk.py is the index desk)
vaaydo/
  curator.py            one run: data, scores, the book replay, flow, the screen
  ledger.py             positions as chains of contract legs: fills, rolls, stops, corporate actions
  signal.py             Option Positioning Index, smoothing, model-book membership
  features.py           per-stock option/futures readings from one day's bhavcopy
  flowscan.py           contract-level option flow: buyers/writers opening, unusual, surge
  session.py            recent sessions, futures build-up, delivery
  bundle.py · universe.py   shipped data; index-constituent filters
  banlist.py            NSE F&O ban list (entries wait out a ban)
  indices.py            the index desk's readings: chain, gamma, volatility, futures, participants
  index_desk.py         one desk run: history + newer sessions, chain views, paper candidates
  breadth.py            market breadth from NSE's equity bhavcopy and index closes, 2006 on
  inference.py          non-overlapping IC, HAC t, block bootstrap, detectable IC, precedent, tranche backtest
  index_research.py · index_options.py · gamma.py   the index and gamma tests
  pragyam.py            Pragyam risk engine (thebullishvalue/Indicator), verbatim
  data/                 NSE F&O bhavcopy (UDiFF + legacy), equity bhavcopy, Yahoo history
  alpha.py · simulate.py · book.py · longtest.py · optexpr.py   the research that chose the signal
docs/  VISION.md · METHOD.md · RESEARCH_V6.md · PREREGISTRATION.md · RESEARCH.md · AUDIT.md
```

## Terms

Risk levels: **Pragyam** (thebullishvalue/Indicator). NSE data is subject to NSE's
terms. Decision support, not investment advice. Futures can lose more than the margin
posted.
