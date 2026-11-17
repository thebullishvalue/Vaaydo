# VAAYDO (वायदो)

**An evidence-gated F&O desk for NSE options.** It reads every option chain each evening,
says which trades have *earned* a place on real prices, sizes them against your
capital, and keeps an honest book.

Most days it will tell you: **no trades**. On 2+ years of NSE data, every mechanical
premium-selling strategy tested lost money after costs, despite winning 70–90% of the
time. VAAYDO exists to keep you out of those trades, and to tell you when that changes.

## What's in it

| Tab | What it shows |
|---|---|
| **Today** | The plan: trades from live rules (sized, stress-tested), or "no trades" with the reason, plus a watchlist of what the system is waiting on |
| **Playbook** | Every rule's net record on real prices, gross vs net of costs, and the walk-forward record of the playbook itself |
| **Market** | Implied vs realised vol across all F&O names, event flags, futures positioning |
| **Name** | One underlying: smile, OI by strike, positioning, every defined-risk structure valued with costs and tail risk |
| **Book** | Your positions, marked daily at NSE settlement prices, with exit actions and predicted-vs-realised calibration |
| **How it works** | The method in plain words |

## Quick start

```bash
pip install -r requirements.txt
python -m vaaydo sync        # ~2 years of NSE F&O bhavcopy (~400 MB, ~7 min, once)
python -m vaaydo research    # builds the evidence from real prices (~15 min, once)
streamlit run app.py
```

Daily, after ~18:00 IST: `python -m vaaydo sync && python -m vaaydo research && python -m vaaydo scan`
(or click **Sync** in the app). Data is stored in `~/.vaaydo` (override with `VAAYDO_DATA`).

## The findings that shaped it (`docs/RESEARCH.md`)

* **Implied vol was efficient.** Mean realised variance / implied variance = 1.00. The
  "IV is 9% too high" premium exists only at the median; single-stock jumps pay it back.
  Out of sample, ATM IV beat every realised-vol forecast tried.
* **Short premium lost after costs**: stock iron condors −9% of risk per trade (t = −3.4),
  NIFTY weekly put spreads −10% (t = −3.2). They lost money even before costs.
* **Futures build-up and option-OI walls did not predict** forward returns or breaches.
* **Watching:** long straddles on stocks with the cheapest IV (+17% net per trade, t = 1.8).
  Not yet reliable enough to trade.

## Layout

```
vaaydo/
  data/bhavcopy.py   NSE F&O bhavcopy: download, slim, cache (UDiFF, Jul 2024+)
  data/store.py      panels: spot (corporate-action adjusted), lots, futures positioning, chains
  pricing.py         Black-76 on the futures forward, vectorised implied vol
  vol.py             smiles from real option prices, fitted smile, realised vol
  forecast.py        filtered historical simulation (fat tails, centred on the forward)
  strategies.py      defined-risk structures from liquid listed strikes only
  costs.py           Indian F&O charges + liquidity-dependent spread model
  evaluate.py        EV, P(profit), tail loss, max loss (incl. exit) per lot
  research.py        fixed-rule studies on real prices (the evidence)
  playbook.py        rules, live/watch status, walk-forward record
  engine.py          the daily scan
  portfolio.py       sizing, correlation clusters, NIFTY-gap stress
  journal.py         your book: positions, daily marks, exit actions, calibration
  backtest.py        settlement machinery, positioning study
  flow.py            futures build-up and option OI walls
app.py               Streamlit dashboard
docs/                METHOD.md · RESEARCH.md · AUDIT.md (why v4 was replaced)
```

## Credits and terms

The positioning concepts (futures build-up, option-OI walls) follow
[Order Flow](https://github.com/MrChartist/order-flow) by Mr. Chartist. VAAYDO
reimplements the ideas in Python and does not include its code or data. NSE data is
subject to NSE's terms. VAAYDO is a research and decision-support tool, not investment
advice. Options can lose more than you expect, faster than you expect.
