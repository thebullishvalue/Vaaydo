# How VAAYDO decides

## The principle

**A trade must have earned its place on real prices before VAAYDO will suggest it.**

Most option dashboards rank trades by probability of profit, or by a model's opinion of
"fair" volatility. VAAYDO's own research (Playbook tab, `docs/RESEARCH.md`) showed why
that fails here. On 2+ years of NSE prices, short-premium structures won 70–90% of the
time and still lost money after costs, and no volatility forecast beat the market's own
implied volatility out of sample.

So the system is split in two:

| | decides | from |
|---|---|---|
| **Playbook** | *what* to trade | each rule's net record on real prices, judged only on data before the decision |
| **Risk model** | *how much*, and what can go wrong | market-implied volatility + the stock's own fat tails |

## Each day, after the close (`sync → research → scan → mark`)

1. **Data.** The NSE F&O bhavcopy has every contract's close, settlement price, volume,
   open interest and lot size. Expiries, lots and corporate actions all come from it.
2. **Read every chain.** For each underlying: the first expiry 20–60 days out, implied
   vol of every traded option (Black-76 off the futures price), a fitted smile, 25Δ skew,
   the IV term structure, 1-month realised vol, chain liquidity, futures positioning, and
   option OI walls.
3. **Features → terciles.** Each stock's ATM IV, IV / 1M realised, put skew and chain
   liquidity are ranked into thirds within today's cross-section, the same definitions
   the rules were tested on.
4. **Match the playbook.**
   * **Live** rule (net mean > 0, t ≥ 2.5, ≥ 12 prior cycles): matching names become
     trade candidates.
   * **Watch** rule (net mean > 0, t ≥ 1.5): matching names go on the watchlist, shown and
     never traded.
   * Names with an IV term inversion (near-month ATM IV > next month's by 8%: an event is
     priced) get no short-premium trades.
5. **Value each candidate.** Entry at the worse of the close and the fitted-smile price,
   ± a liquidity-dependent half-spread, plus every statutory charge. Max loss includes the
   cost of closing at the worst point. P(profit), tail loss and breakevens come from a
   distribution centred on the futures price, scaled to ATM IV, with the stock's own
   bootstrapped fat tails.
6. **Build the book.** Max loss per trade ≤ 1.5% of capital. At most 2 positions among
   stocks whose returns correlate > 0.6. Worst NIFTY gap (±3/6/10%, each stock by its
   beta) ≤ 8% of capital. At most 10 positions and 20% of capital at risk.
7. **Manage.** Short premium: close at 50% of max profit. Long premium: close at +100%.
   Stock options: close 3 trading days before expiry (physical settlement). NIFTY: cash
   settled, can be held. The Book tab marks every position at NSE settlement prices and
   flags the action due.

## What you'll usually see

**"No trades."** Over the history studied, no rule qualified, and the walk-forward
playbook stood aside throughout. That saved the 3–16% of risk per trade that every
mechanical short-premium rule lost. Standing aside is the system working, not failing.
Every new cycle is added to the evidence (`python -m vaaydo research`), so a rule that
starts working will go live on its own record.

## Limits

* Fills are modelled, not observed. Work limit orders at or better than the plan's
  entry; calibrate the spread model from your broker's quotes (`costs.calibrate_spread`).
* End-of-day only. VAAYDO doesn't watch intraday moves; your broker's stop orders do.
* 27 monthly cycles is one market regime. Treat every number as provisional.
