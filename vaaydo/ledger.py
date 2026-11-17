"""
The trading ledger: the exact rules of the backtest (`book.simulate`), replayed day by day
on real prices so that every position has an entry date, an entry price and a stop.

    signal    decided at the close of day d (score membership from `signal.replay`)
    entry     the futures OPEN of the next session
    stop      trailing: k x (best close since entry), k = 1 + 1.5 x (Pragyam's 20%-touch stop
              distance at the signal date). Checked on each close; exit at the next open.
    ban       a stopped-out name is not re-entered on that side until its signal leaves the
              band (a new signal, a new trade)
    exits     the score leaves the hold band, or the stop

Stateless: the same inputs always replay to the same book, so the app needs no database.
Stops are tracked on adjusted spot closes (Yahoo) and shown on the futures price by the
day's futures/spot ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STOP_MULT = 1.5


@dataclass
class Pos:
    symbol: str
    side: int
    signal: pd.Timestamp            # decision date
    exec: pd.Timestamp | None       # entry session (next open); None until it happens
    k: float | None                 # stop multiplier on the best close
    best: float | None = None       # best close since entry (spot)

    @property
    def stop(self) -> float | None:
        return self.best * self.k if (self.best is not None and self.k is not None) else None


@dataclass
class Ledger:
    held: dict = field(default_factory=dict)        # symbol -> Pos (after today's decisions)
    prev: dict = field(default_factory=dict)        # symbol -> Pos (yesterday's book)
    exits_today: dict = field(default_factory=dict)  # symbol -> (Pos, reason)
    log: list = field(default_factory=list)


def replay(dates, model: dict, closes: pd.DataFrame, opens: pd.DataFrame, ratios: dict,
           ext: dict | None = None, notional: dict | None = None, cap: float | None = None,
           mult: float = STOP_MULT) -> Ledger:
    """dates: sessions to replay (ascending). model: {date: {symbol: side}} (score membership).
    closes/opens: adjusted spot, date x symbol. ratios: {(symbol, side): Series of Pragyam
    stop ratio by date}. With cap: your book (capital-limited, most extreme first,
    long/short notional balanced); without: the full model book."""
    held: dict[str, Pos] = {}
    banned: dict[str, int] = {}
    led = Ledger()
    nxt = {d: dates[i + 1] if i + 1 < len(dates) else None for i, d in enumerate(dates)}
    for d in dates:
        prev = dict(held)
        exits = {}
        m = model.get(d, {})
        # 1) yesterday's entries happen at today's open; then today's close updates stops
        for s, p in held.items():
            if p.exec is None and d > p.signal:
                p.exec = d
                o = opens.at[d, s] if (s in opens and d in opens.index) else np.nan
                p.best = float(o) if np.isfinite(o) else None
            c = closes.at[d, s] if (s in closes and d in closes.index) else np.nan
            if p.exec is not None and np.isfinite(c):
                p.best = c if p.best is None else (max(p.best, c) if p.side > 0 else min(p.best, c))
                stp = p.stop
                if stp is not None and ((c <= stp) if p.side > 0 else (c >= stp)):
                    exits[s] = (p, "trailing stop hit on the close")
                    banned[s] = p.side
        # 2) bans last until the signal leaves the band on that side
        banned = {s: sd for s, sd in banned.items() if m.get(s) == sd}
        # 3) keep what is still in the model on the same side and not stopped
        keep = {s: p for s, p in held.items() if s not in exits and m.get(s) == p.side}
        for s, p in held.items():
            if s not in exits and s not in keep:
                exits[s] = (p, "signal left the band")
        # 4) add new names
        cand = [s for s, sd in m.items() if s not in keep and banned.get(s) != sd]
        if cap is None:
            add = cand
        else:
            e = (ext or {}).get(d, {})
            used = sum(notional.get(s, np.inf) for s in keep)
            nl = sum(notional.get(s, 0) for s, p in keep.items() if p.side > 0)
            ns = sum(notional.get(s, 0) for s, p in keep.items() if p.side < 0)
            L = sorted([s for s in cand if m[s] > 0 and s in notional], key=lambda s: -e.get(s, 0))
            S = sorted([s for s in cand if m[s] < 0 and s in notional], key=lambda s: -e.get(s, 0))
            add = []
            while L or S:
                pick_long = bool((nl <= ns and L) or not S)
                s = L.pop(0) if pick_long else S.pop(0)
                if used + notional[s] > cap:
                    continue
                add.append(s)
                used += notional[s]
                if pick_long:
                    nl += notional[s]
                else:
                    ns += notional[s]
        for s in add:
            r = ratios.get((s, m[s]))
            rv = r.get(d, np.nan) if r is not None else np.nan
            k = 1 + mult * (rv - 1) if np.isfinite(rv) else None
            keep[s] = Pos(s, m[s], d, None, k)
        for s, (p, why) in exits.items():
            led.log.append(dict(date=d, symbol=s, side=p.side, action="exit", why=why))
        for s in add:
            led.log.append(dict(date=d, symbol=s, side=m[s], action="enter", why="signal"))
        held = keep
        led.prev, led.exits_today = prev, exits
    led.held = held
    return led
