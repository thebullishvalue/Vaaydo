"""
The position ledger: the backtest's trading rules (`book.simulate`) replayed on real
futures contracts, so every position carries its full history.

A position is a chain of LEGS, one per futures contract it was held in:

    signal    decided at the close of day d (score membership)
    entry     the OPEN of session d+1, in the contract that was active at d (the nearest
              expiry with more than 3 sessions left, the same rule as the backtest)
    roll      when the active contract changes, the leg is closed at that session's close
              and a new leg opened in the new contract at its close; each leg keeps the
              lot size of ITS contract (NSE revises lots for new contracts)
    stop      trailing on futures closes: k x best close since entry,
              k = 1 + 1.5 x (Pragyam's 20%-touch stop distance at the signal), decided on the
              close, executed at the next open; a stopped name stays out until its signal
              leaves the band
    corporate actions   a split/bonus/demerger rescales the open leg's entry price and the
              trailing levels by the exchange's adjustment factor, so P&L is continuous

Stateless: the same inputs always replay to the same ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STOP_MULT = 1.5
ROLL_BEFORE = 3


def _expiry_positions(cal: pd.DatetimeIndex, expiries) -> np.ndarray:
    """Session index of each expiry. Expiries beyond the known calendar (the future, at the
    latest date) are placed by counting business days past its end; without this every
    future expiry looks one session away and no contract qualifies as tradable."""
    out = []
    for e in expiries:
        e = pd.Timestamp(e)
        if e <= cal[-1]:
            out.append(int(np.searchsorted(cal, e)))
        else:
            out.append(len(cal) - 1 + int(np.busday_count(cal[-1].date(), e.date())))
    return np.array(out)


@dataclass
class Leg:
    contract: pd.Timestamp
    entry_date: pd.Timestamp
    entry_px: float
    lot: int
    exit_date: pd.Timestamp | None = None
    exit_px: float | None = None
    exit_why: str | None = None
    adj: float = 1.0                     # cumulative corporate-action factor applied to entry_px

    def pnl(self, side: int, last: float | None = None) -> float:
        px = self.exit_px if self.exit_px is not None else last
        if px is None or not np.isfinite(px) or not np.isfinite(self.entry_px):
            return np.nan
        # entry_px x adj is the entry in today's (post-action) price terms; lot is already today's lot
        return side * (px - self.entry_px * self.adj) * self.lot


@dataclass
class Position:
    symbol: str
    side: int
    signal: pd.Timestamp
    k: float | None
    legs: list = field(default_factory=list)
    best: float | None = None            # best close since entry (futures, adjusted)
    pending: bool = True                 # decided, not yet filled

    @property
    def leg(self) -> Leg | None:
        return self.legs[-1] if self.legs else None

    @property
    def stop(self) -> float | None:
        return self.best * self.k if (self.best is not None and self.k is not None) else None

    def realised(self) -> float:
        return float(np.nansum([l.pnl(self.side) for l in self.legs[:-1]])) if len(self.legs) > 1 else 0.0


class Market:
    """Futures prices in the shape the ledger needs: (date, symbol, expiry) -> open/close/lot,
    the active contract per (date, symbol), and corporate-action factors."""

    def __init__(self, F: pd.DataFrame, ca: dict | None = None, roll_before: int = ROLL_BEFORE):
        F = F.copy()
        F["date"] = pd.to_datetime(F["date"])
        F["expiry"] = pd.to_datetime(F["expiry"])
        for c in ("open", "close"):
            F[c] = F[c].where(F[c] > 0)
        self.cal = pd.DatetimeIndex(sorted(F["date"].unique()))
        pos = pd.Series(np.arange(len(self.cal)), index=self.cal)
        F["i"] = F["date"].map(pos)
        exps = sorted(F["expiry"].unique())
        exp_i = pd.Series(_expiry_positions(self.cal, exps), index=exps)
        F["left"] = F["expiry"].map(exp_i) - F["i"]
        act = F[F["left"] > roll_before].sort_values("expiry").groupby(["date", "symbol"]).first().reset_index()
        self.active = act.set_index(["date", "symbol"])["expiry"].to_dict()
        # the contract an order decided today will fill in tomorrow: today's active one, unless
        # it rolls tomorrow (then the next listed contract)
        nxt = F[F["left"] > roll_before + 1].sort_values("expiry").groupby(["date", "symbol"]).first().reset_index()
        self.fill_contract = nxt.set_index(["date", "symbol"])["expiry"].to_dict()
        Fi = F.set_index(["date", "symbol", "expiry"])
        self.o = Fi["open"].to_dict()
        self.c = Fi["close"].to_dict()
        self.lot = Fi["lot"].to_dict() if "lot" in Fi else {}
        from .book import corporate_action_factors
        self.ca = corporate_action_factors(F)
        self.ca.update(ca or {})

    def px(self, d, s, e, which="close"):
        v = (self.c if which == "close" else self.o).get((d, s, e))
        return float(v) if v is not None and np.isfinite(v) else np.nan


@dataclass
class Ledger:
    held: dict = field(default_factory=dict)          # symbol -> Position after the last decision
    closed: list = field(default_factory=list)        # (Position) closed positions
    decisions: dict = field(default_factory=dict)     # symbol -> ("ENTER"|"EXIT"|"HOLD", reason) at the last date
    banned: dict = field(default_factory=dict)        # symbol -> side: stopped out, waiting for the signal to reset
    asof: pd.Timestamp | None = None


def replay(M: Market, dates, model: dict, ratios: dict, ext: dict | None = None, notional: dict | None = None,
           cap: float | None = None, mult: float = STOP_MULT, fno_ban: frozenset | None = None) -> Ledger:
    """dates: decision sessions, ascending. model: {date: {symbol: side}}. ratios:
    {(symbol, side): Series} Pragyam stop ratio. With cap: capital-limited book (most
    extreme first, long/short notional balanced), else the full model book."""
    held: dict[str, Position] = {}
    banned: dict[str, int] = {}
    led = Ledger()
    for i, d in enumerate(dates):
        # ── 1) fills and marks of day d: entries/exits decided at d-1 execute at d's open ──
        for s, p in list(held.items()):
            if p.pending:
                e = M.active.get((d, s))                   # the contract active on the FILL day (no instant roll)
                o = M.px(d, s, e, "open") if e is not None else np.nan
                if not np.isfinite(o):
                    o = M.px(d, s, e, "close") if e is not None else np.nan
                if e is None or not np.isfinite(o) or (fno_ban and (d, s) in fno_ban):
                    del held[s]                                   # unfillable or in an F&O ban: no position;
                    continue                                      # re-decided at today's close
                p.legs.append(Leg(e, d, o, int(M.lot.get((d, s, e), 0) or 0)))
                p.best = o
                p.pending = False
            leg = p.leg
            f = M.ca.get((d, s))
            if f:                                                 # exchange adjusted the contract today
                leg.adj *= f
                p.best = p.best * f if p.best is not None else None
                leg.lot = int(round(leg.lot / f)) if leg.lot else leg.lot
            c = M.px(d, s, leg.contract)
            if np.isfinite(c):
                p.best = c if p.best is None else (max(p.best, c) if p.side > 0 else min(p.best, c))
            nxt_contract = M.active.get((d, s))
            if nxt_contract is not None and nxt_contract != leg.contract:   # roll at today's close
                c_old, c_new = M.px(d, s, leg.contract), M.px(d, s, nxt_contract)
                if np.isfinite(c_old) and np.isfinite(c_new):
                    leg.exit_date, leg.exit_px, leg.exit_why = d, c_old, "rolled"
                    p.legs.append(Leg(nxt_contract, d, c_new, int(M.lot.get((d, s, nxt_contract), 0) or 0)))
                    p.best = p.best * (c_new / c_old) if p.best is not None else c_new   # trail in new-contract terms
        # ── 2) decisions at the close of d ──
        m = model.get(d, {})
        decisions = {}
        exits = {}
        for s, p in held.items():
            if p.pending:
                continue
            c = M.px(d, s, p.leg.contract)
            stp = p.stop
            if stp is not None and np.isfinite(c) and ((c <= stp) if p.side > 0 else (c >= stp)):
                exits[s] = "trailing stop hit on the close"
                banned[s] = p.side
        banned = {s: sd for s, sd in banned.items() if m.get(s) == sd}
        keep = {s: p for s, p in held.items() if s not in exits and m.get(s) == p.side}
        for s, p in held.items():
            if s not in exits and s not in keep:
                exits[s] = "signal left the band"
        cand = [s for s, sd in m.items() if s not in keep and banned.get(s) != sd and M.active.get((d, s)) is not None]
        if cap is None:
            add = cand
        else:
            e_ = (ext or {}).get(d, {})
            used = sum(notional.get(s, np.inf) for s in keep)
            nl = sum(notional.get(s, 0) for s, p in keep.items() if p.side > 0)
            ns = sum(notional.get(s, 0) for s, p in keep.items() if p.side < 0)
            L = sorted([s for s in cand if m[s] > 0 and s in notional], key=lambda s: -e_.get(s, 0))
            S = sorted([s for s in cand if m[s] < 0 and s in notional], key=lambda s: -e_.get(s, 0))
            add = []
            while L or S:
                pick_long = bool((nl <= ns and L) or not S)
                s = L.pop(0) if pick_long else S.pop(0)
                if used + notional[s] > cap:
                    continue
                add.append(s)
                used += notional[s]
                nl, ns = (nl + notional[s], ns) if pick_long else (nl, ns + notional[s])
        # exits fill at the next open: record them now with the leg still open
        for s, why in exits.items():
            p = held[s]
            decisions[s] = ("EXIT", why)
            led.closed.append((p, why, d))
        for s, p in keep.items():
            decisions[s] = ("HOLD", "")
        new_held = dict(keep)
        for s in add:
            r = ratios.get((s, m[s]))
            rv = r.get(d, np.nan) if r is not None else np.nan
            k = 1 + mult * (rv - 1) if np.isfinite(rv) else None
            new_held[s] = Position(s, m[s], d, k)
            decisions[s] = ("ENTER", "")
        # previously-closed positions get their exit fill at the next open (step 1 of next day)
        if i + 1 < len(dates):
            nd = dates[i + 1]
            for s in exits:
                p = held[s]
                leg = p.leg
                f = M.ca.get((nd, s), 1.0)
                o = M.px(nd, s, leg.contract, "open")
                if f != 1.0:
                    leg.adj *= f
                leg.exit_date, leg.exit_px, leg.exit_why = nd, o, exits[s]
        held = new_held
        led.decisions = decisions
    led.held, led.asof, led.banned = held, dates[-1] if dates else None, banned
    led.closed = [(p, why, d) for p, why, d in led.closed]
    return led
