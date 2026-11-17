# VAAYDO (वायदो)

**An F&O trade curator for NSE.** Every evening it reads the option chain of every F&O
stock, scores where the options crowd is positioned, and gives you tomorrow's stock
futures orders, sized to your capital, with a stop and a reason for each.

> Long the stocks where the options crowd leans hardest to puts; short the ones where
> it leans hardest to calls; equal rupees each side.

## What you get each evening

* **Orders**: what to buy, sell, close or hold tomorrow, one lot each, with the futures
  contract, reference price, notional, margin, disaster stop and Pragyam's 10-session
  range, plus Order Flow context (futures build-up, put/call ratios, IV, event flag,
  delivery %).
* **Book**: your positions and the full model book, long/short balance, stop proximity.
* **Name**: why a stock is in the book: its positioning score over time, option-market
  readings, price with risk levels.
* **Track record**: 2019 – today, net of costs, for the book size your capital supports.

## The evidence, briefly (docs/RESEARCH_V6.md)

| Book (1× gross, net of costs, 2019 – Nov 2026) | Sharpe | Return / yr | Worst drawdown |
|---|---|---|---|
| model book, ~30 names | ~1.0 | ~9% | −14% |
| 12 names | 0.89 | 9.0% | −17% |
| 6 names | 0.55 | 7.7% | −21% |

The signal was designed on 2024–25 and confirmed on 2019–22, which no design choice had
seen. It was flat in 2023 – mid-2024 and is weak in 2026 so far. Expect losing years.

## Run it

**Streamlit Cloud:** point the app at `app.py`. First load takes about a minute (it
fetches the latest NSE files and price history); after that it's cached.

**Locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

`data/opi_inputs.parquet` ships the score inputs since Jul 2024 so the app starts fast.
The workflow in `.github/workflows/refresh.yml` appends each evening's session
(`scripts/refresh_bundle.py`); without it, the app computes missing days itself.

## Layout

```
app.py                  the curator (Streamlit)
vaaydo/
  curator.py            prepare (data, scores, model book, levels, context) + personalize (your orders)
  signal.py             Option Positioning Index, smoothing, model-book and your-book replay
  features.py           per-stock option/futures features from one day's bhavcopy
  pragyam.py            Pragyam risk engine (thebullishvalue/Indicator), verbatim
  session.py            what the live app fetches: recent bhavcopies, build-up, delivery
  data/                 NSE F&O bhavcopy (UDiFF + legacy), equity bhavcopy, Yahoo history
  alpha.py · simulate.py · book.py · longtest.py · optexpr.py   the research that chose the signal
docs/  METHOD.md · RESEARCH_V6.md · PREREGISTRATION.md · RESEARCH.md · AUDIT.md
```

## Credits and terms

Risk levels: **Pragyam** (thebullishvalue/Indicator). Positioning concepts: **Order
Flow** by Mr. Chartist (github.com/MrChartist/order-flow), reimplemented, with no code or
data copied. NSE data is subject to NSE's terms. This is decision support, not
investment advice. Futures can lose more than the margin posted.
