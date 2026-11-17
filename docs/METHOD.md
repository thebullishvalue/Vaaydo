# How VAAYDO picks trades

## The idea in one paragraph

Every evening NSE publishes the close of every option on every F&O stock. In those
chains you can see, stock by stock, which way the options crowd leans: whether people
are buying puts or calls, whether downside protection is expensive or cheap, and whether
put or call open interest is being added. **When the options crowd leans hardest to puts
on a stock, that stock has tended to beat the other F&O stocks over the following weeks;
when it leans hardest to calls, it has tended to lag.** VAAYDO goes long the first group
and short the second, through stock futures, in equal rupee amounts, so a NIFTY move
largely cancels out.

## The score (Option Positioning Index)

For every F&O stock, each session, six readings from its option chain, each ranked
against all other F&O stocks that day:

| Reading | Leans bullish for the score when… |
|---|---|
| put/call contracts traded | more puts than usual are traded |
| put/call open interest | more puts than usual are open |
| put skew (5% OTM put IV − ATM IV) | protection is expensive |
| call skew (5% OTM call IV − ATM IV) | upside calls are cheap |
| call IV − put IV at the same strikes | calls are cheap relative to puts |
| put OI added − call OI added | today's new positions lean to puts |

The six ranks are averaged into the day's reading, and **the score is its 10-session
average**. Daily readings are noisy, because option closes are not simultaneous; the
crowd's positioning moves slowly. Averaging keeps the information and removes most of
the noise, which also cuts trading by 5×.

## The book

* **Enter long** when a stock's score is in the top 5% of F&O stocks; **hold** while it
  stays in the top 15%. Shorts mirror this at the bottom. (Entering and exiting at
  different thresholds stops the book churning on small wiggles.)
* **One lot per name**, futures contract at least a week from expiry, rolled 3 sessions
  before expiry. Stock futures are physically settled, so nothing is held into expiry.
* **Disaster stop** at twice Pragyam's stop distance (Pragyam's stop is touched 20% of
  the time within 10 sessions; twice that is rarely touched). Tighter stops were tested
  and cut returns: this trade needs room.
* **Your book** is the subset that fits your capital: names you hold stay while they are
  in the model book; freed capital goes to the most extreme new names, alternating to
  keep long and short notional equal. The app replays this every day, so your orders are
  the same however often you open it.

## Why futures, not options

The signal comes *from* options, but options are a poor way to *trade* it. On the same
entry signals 2019–2026, bull call and bear put spreads lost 6–16% of the premium paid
per trade in every period, because two option spreads in and out cost more than the
edge. Futures cost about 0.13% round trip. Credit spreads, condors and straddles lost
money even unconditionally (docs/RESEARCH.md).

## What Pragyam and Order Flow contribute

* **Pragyam** (thebullishvalue/Indicator) supplies the risk: its calibrated stop sets
  each trade's disaster stop, and its 80% 10-session cone is shown on every order.
  Pragyam's own research found price charts don't forecast direction, and VAAYDO's tests
  agree: every price-only signal (trend, momentum, reversal, Pragyam's surprise and
  trend tapes) was null cross-sectionally.
* **Order Flow** (MrChartist/order-flow) supplies the reading of positioning: futures
  build-up, option-OI walls, put/call ratios. Shown on every order as context. Futures
  build-up and OI walls alone did not predict returns; the option-chain readings did.

## What to expect

Backtest 2019 – Nov 2026, net of costs, at 1× gross exposure: about 8–9% a year with
8–10% volatility for 12–30 names, worst drawdown 13–17%, one or two losing years. It
was flat in 2023 – mid-2024 and weak in 2026 so far. Plan for a live Sharpe of 0.5–0.8,
and pause if the book loses more than 1.5× its worst backtest drawdown.
