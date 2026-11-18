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
* **Entry at the next session's open** after the signal (the blotter shows that price).
* **Trailing stop**: 1.5× Pragyam's 20%-touch stop distance below the best close since
  entry (above it for shorts), checked on each close, exit at the next open. Typically
  5–20% from the last price, set by each stock's volatility. A stopped-out name stays out
  until its signal leaves the band. Stops were tested from none to tight trailing: none
  added return (Sharpe 0.58–0.72 for every sensible variant; tight trailing stops hurt),
  so the stop is there for risk control, and trailing keeps it close to the price.
* **One lot of every name.** There is no capital sizing: the book is the whole model book,
  one lot of each stock's futures contract. Lots differ in rupee size, so the book is not
  exactly balanced long against short (12% net long of gross on average since 2019). A
  position keeps its quantity in shares: a roll takes the new contract's lot, a split or
  bonus rescales it, and an exchange lot revision on a running contract changes nothing.

## Why futures, not options

The signal comes *from* options, but options are a poor way to *trade* it. On the same
entry signals 2019–2026, bull call and bear put spreads lost 6–16% of the premium paid
per trade in every period, because two option spreads in and out cost more than the
edge. Futures cost about 0.13% round trip. Credit spreads, condors and straddles lost
money even unconditionally (docs/RESEARCH.md).

## What Pragyam and the positioning readings contribute

* **Pragyam** (thebullishvalue/Indicator) supplies the risk: its calibrated stop sets
  each trade's disaster stop, and its 80% 10-session cone is shown on every order.
  Pragyam's own research found price charts don't forecast direction, and VAAYDO's tests
  agree: every price-only signal (trend, momentum, reversal, Pragyam's surprise and
  trend tapes) was null cross-sectionally.
* **Positioning readings** supply the context: futures
  build-up, option-OI walls, put/call ratios. Shown on every order as context. Futures
  build-up and OI walls alone did not predict returns; the option-chain readings did.

## What to expect

Backtest 2019 – Nov 2026, net of costs, at 1× gross exposure: 6–7% a year with 8–11%
volatility for 12–30 names (Sharpe 0.66–0.77), worst drawdown 13–19%, two losing years.
A 6-name book is mostly noise (Sharpe 0.30). It was flat in 2023 – mid-2024 and weak in
2026 so far. Plan for a live Sharpe of 0.4–0.6, and pause if the book loses more than
1.5× its worst backtest drawdown.
