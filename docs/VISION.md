# VAAYDO — vision and design

*वायदो, vāydo: a forward contract, a promise for a later date.*

## What it is

**A screener for the NSE F&O segment that hands you trades.** You open it, pick a
universe and a date, press Run, and it screens every F&O stock the way Sanket screens a
universe: what fired today, what you are holding and how it is doing, what is about to
fire, and where every name stands. Each trade comes as a complete ticket: contract,
side, entry, stop, expected range, size, and the reason.

It belongs to the same family as Sanket, Tattva and Nishkarsh: the same rail, the same
Graphite design system, the same habit of measuring a claim before showing it, and the
same refusal to call a state "proven" when it is not.

## What it reads

Everything an end-of-day F&O desk can see, from NSE's own files:

| Layer | What it reads | Used for |
|---|---|---|
| **Option positioning** | each stock's whole option chain: put/call volume and open interest, put skew, call skew, call-vs-put IV, which side's OI was added | **the signal** |
| **Option flow** | every option contract: buyers or writers opening or closing, unusual volume (above starting OI), volume surges | context; tested as signal |
| **Futures flow** | price and open interest together: long build-up, short build-up, short covering, long unwinding; 1-day and 5-day | context, conviction checks |
| **Option levels** | heaviest put OI below (support), heaviest call OI above (resistance), max pain, OI by strike | context, ticket levels |
| **Market board** | breadth, sector medians and leans, build-up census | context |
| **Cash market** | delivery %, VWAP | context |
| **Risk** (Pragyam) | per-bar risk forecast, calibrated stop distance, 80% range, trend, risk regime | every stop and range |

## How it decides

One signal, measured before shipping (docs/RESEARCH_V6.md): when the options crowd
leans hardest to puts on a stock, the stock has tended to outperform the other F&O
stocks over the following weeks; when it leans hardest to calls, it has lagged. The
score is the 10-session average of that positioning, ranked across all F&O stocks.

* **Signal ▲** a stock enters the top 5% of the score. **Signal ▼** the bottom 5%.
* **Held** while it stays in the top / bottom 15%.
* **Exit** when it leaves that band, or on its trailing stop.
* **Ready** (watchlist) a stock in the 5–15% band that is moving toward an entry.

Trades are stock futures, long ▲ and short ▼ in equal rupees. Options are where the
signal comes from, not how it is traded: options as the instrument lost to their own
spreads in every test.

## What a run returns

| Tab | What it shows |
|---|---|
| **Signals** | the signal board: new ▲ and ▼ today, exits today, each as a ticket with its reason, plus the open book with roll history |
| **Book** | every open position: legs per contract (entries, rolls, lot changes), realised and open P&L, trailing stop, risk; optional sizing to your capital |
| **Watchlist** | names closest to firing, nearest first |
| **Flow** | the market board: breadth, sectors, futures build-up census, option flow, unusual activity |
| **Contracts** | the option-contract scanner: most active, OI gainers/losers, buyers/writers opening, unusual volume, surges |
| **Screen** | every F&O stock: score, state, readings, levels |
| **System Data** | track record, the measured evidence, exports, definitions |

## Modes

* **Single Date** — the screen as of any session since July 2024.
* **Historical Range** — every signal in a range, with what followed each, against peers.

## Principles carried from the family

1. **A position is exactly what the backtest would hold.** One engine replays the rules
   on real futures contracts for both the backtest and the screen.
2. **Contracts are real.** A position entered in June was entered in a June contract
   and rolled each month; the ticket shows every leg, with the lot size of that leg.
3. **Nothing is adjusted silently.** Splits, bonuses and demergers are applied with the
   exchange's own factor, and shown.
4. **Evidence beside every claim.** The track record is shown at the size of book the
   reader would run; a book too small to be the strategy says so.
5. **No look-ahead anywhere.** Every reading uses only data published by that close;
   decisions at the close, fills at the next open.

## Data architecture

* NSE F&O bhavcopy (UDiFF, from 8 Jul 2024), NSE equity bhavcopy, Yahoo daily history.
* Shipped with the repo, refreshed nightly by a workflow: score inputs and the futures
  price history since July 2024. The app fetches only sessions after the bundle.
* Research store (not shipped): the 2019–2024 archive, every study's inputs.
