# Research log: what the data says (Jul 2024 – Nov 2026)

All results come from NSE's F&O bhavcopy (584 sessions, every listed contract) and can be
reproduced with `python -m vaaydo research` and the functions named below. "Return on risk"
is P&L divided by the structure's max loss, including the cost of closing at the worst
point. Means are taken per cycle first and t-tested across cycles, because trades in the
same month share one market.

## 1. Implied volatility is an efficient forecast here

At every monthly entry, for every name, ATM IV against the volatility that followed:

| | |
|---|---|
| median IV / realised | 1.09 |
| share of names with IV > realised | 61.5% |
| **mean realised variance / mean implied variance** | **1.00** |

The median says options are 9% "rich". The mean, which is what option P&L follows,
says they are fairly priced. Single stocks jump, and the jumps repay the premium.

Out of sample (fit on the first half of cycles, test on the second), raw ATM IV was the
best forecast of realised vol (log-RMSE 0.317, bias +0.002). Blends of 1M/3M/1Y realised
vol did worse (0.336–0.354), as did IV plus realised. The sorting signal "IV minus
forecast" had no relationship with subsequent IV minus realised across quintiles.

**Consequence:** v5's first engine traded "IV vs our forecast". It was retired. The
model now uses the market's IV for risk only.

## 2. The first engine's backtest (retired)

Forecast-gated, portfolio-sized, 22 monthly cycles: +16.8% total, Sharpe 0.76, max
drawdown −11%. But predicted EV was ₹10.5L against ₹1.7L realised, and the profit came
from long strangles whose realised return on risk (+13%) bore no relation to the
predicted +87%. Passing the gates did not separate winners from losers (t = 0.5 across
cycles). A profitable backtest from a model that doesn't predict is luck, and luck
doesn't get traded.

## 3. Fixed-rule structures on real prices (`research.harvest`)

Every liquid name, every cycle, the same structures. Exits: 50% of max profit, or 3
trading days before expiry (stocks are physically settled); NIFTY weeklies are held to
expiry (cash settled).

**Single-stock monthlies, 27 cycles, ~5,400 trades per structure**

| structure | gross | net of costs | t (net) | cycles positive |
|---|---|---|---|---|
| Put credit spread 15Δ | −1.6% | −5.5% | −2.0 | 44% |
| Call credit spread 15Δ | −0.2% | −3.2% | −1.7 | 44% |
| Iron condor 20Δ | −0.7% | −9.1% | −3.4 | 26% |
| Iron butterfly | −5.7% | −31.7% | −7.6 | 4% |
| Long straddle | +8.1% | +4.8% | +0.8 | 44% |

**NIFTY weeklies, 123 cycles**

| structure | gross | net | t (net) | trades won |
|---|---|---|---|---|
| Put credit spread 15Δ | −7.7% | −10.1% | −3.2 | 79% |
| Call credit spread 15Δ | −3.1% | −5.5% | −2.0 | 80% |
| Iron condor 20Δ | −11.1% | −16.1% | −3.4 | 67% |
| Long straddle | −3.4% | −4.6% | −0.7 | 37% |

Short premium won 67–88% of trades and still lost money. A high probability of profit
is not an edge.

## 4. Conditional rules (`playbook`)

98 rules: each structure × each tercile of IV level, IV / 1M realised, put skew and chain
liquidity. A rule goes **live** only with positive net mean, t ≥ 2.5 over ≥ 12 cycles,
judged on earlier cycles only.

* **No rule was live at any of the 123 decision dates.** The walk-forward playbook would
  have stood aside throughout, which was the correct call given §3.
* Watch (positive, t ≈ 1.8, not traded): long straddles on stocks in the **cheapest IV
  tercile** (+16.6% net per trade), and on the **most liquid chains** (+15.8%). With ~100
  rules tested, a couple of t ≈ 1.8 results are expected by chance. They become tradable
  only if the record holds up as new cycles arrive.

## 5. Positioning (futures build-up, OI walls) (`backtest.flow_study`)

Futures build-up over 5 days (long build-up, short build-up, short covering, long
unwinding, neutral) against forward 5- and 20-day returns relative to the day's average
stock, across ~200 names and ~580 days. No state had |t| > 1.2. Build-up describes what
happened. On this data it doesn't say what happens next.

The only significant effect: names with **no** meaningful move see volatility rise
afterwards (t ≈ 5.7). That's ordinary vol mean reversion, not positioning information.

## 6. Option OI walls

At each monthly entry, the heaviest put-OI strike below spot and call-OI strike above,
against the market-implied probability (from ATM IV) of finishing beyond them:

| | implied | realised | t |
|---|---|---|---|
| finished below the put wall | 31.1% | 36.1% | +1.0 |
| finished above the call wall | 28.2% | 25.7% | −0.7 |

Walls did not "hold" more than option prices implied. The put-side excess is roughly
what put skew prices in (ATM IV understates downside odds).

## 7. Data lessons (each was a bug before it was a lesson)

* **Closes are not simultaneous.** Each contract's close is its own last-half-hour
  average, and untraded contracts repeat an old close. Structures built from raw closes
  showed credits larger than their width. Entries are now priced at the worse of the close
  and a fitted smile, leg by leg.
* **Mark at settlement prices**, the clearing corporation's daily valuation, and clip
  every structure to its payoff bounds.
* **On expiry day the "settlement price" of every option is the underlying's final
  settlement price**, not the option's value. Value at intrinsic against it.
* **Corporate actions**: splits/bonuses show as a price jump with an inverse lot change.
  Demergers (SIEMENS, ABFRL, TATAMOTORS, VEDL) keep the lot and are caught as >40%
  single-day moves. Real crashes (IEX −35%, INDUSINDBK −32%) stay in: they are the risk.

## Limits

* Fills are modelled (close ± a liquidity-dependent half-spread), not observed. Calibrate
  `spread_k` from your own broker's quotes with `costs.calibrate_spread`. Note that even at
  zero cost, short premium did not make money (§3).
* 27 monthly cycles is a short history, and 2024–26 is one regime. The conclusions are
  about this period. The system re-tests every cycle, so if the regime changes, a rule
  will earn live status on its own record.
