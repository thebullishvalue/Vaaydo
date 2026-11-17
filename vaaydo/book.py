"""
Daily portfolio simulation of a cross-sectional futures book with hysteresis.

Each close d, every stock has a score. Decided at d, executed at d+1's open:
    enter long   score rank in the top `enter` fraction        (short: bottom)
    stay long    while rank stays in the top `stay` fraction    (short: bottom)
    exit         otherwise, or on a disaster stop (2x Pragyam), at the next open
Positions are equal-notional within each side; the long side and the short side each
carry 1.0 of notional, so the book is market-neutral by notional.
Contracts roll 3 sessions before expiry (charged a round trip).
Costs are charged only on actual entries, exits and rolls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import futures_cost



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

def corporate_action_factors(F: pd.DataFrame) -> dict:
    """{(date, symbol): f}: on a split or bonus NSE adjusts EXISTING futures contracts
    (price x f, lot / f) but never back-adjusts history. An existing contract's lot can
    only change for a corporate action (regular revisions apply to new contracts), so
    f = lot before / lot after, exactly. Multiply the previous close by f to compare."""
    if "lot" not in F or not (F["lot"] > 0).any():
        return {}
    x = F[F["lot"] > 0].sort_values(["symbol", "expiry", "date"])
    g = x.groupby(["symbol", "expiry"])
    prev_lot, prev_close = g["lot"].shift(1), g["close"].shift(1)
    ch = x[prev_lot.notna() & (x["lot"] != prev_lot)].copy()
    ch["f"] = prev_lot[ch.index] / ch["lot"]
    ch["px"] = ch["close"] / prev_close[ch.index]
    # a corporate action moves the price by the same ratio; a plain lot revision does not
    ch = ch[(np.abs(np.log(ch["px"] / ch["f"])) < 0.15) & (np.abs(np.log(ch["f"])) > 0.1)]
    out = {}
    for d, s, v in zip(ch["date"], ch["symbol"], ch["f"]):
        out.setdefault((pd.Timestamp(d), s), float(v))
    return out


def confirm_on_futures(F: pd.DataFrame, candidates: dict) -> dict:
    """Keep spot-detected corporate actions (demergers keep the lot) where the futures
    contract's own close jumped by the same ratio."""
    F = F.copy()
    F["date"] = pd.to_datetime(F["date"])
    F = F.sort_values(["symbol", "expiry", "date"])
    F["px"] = F["close"] / F.groupby(["symbol", "expiry"])["close"].shift(1)
    jump = F.dropna(subset=["px"]).groupby(["date", "symbol"])["px"].median()
    out = {}
    for (d, s), f in candidates.items():
        j = jump.get((pd.Timestamp(d), s))
        if j is not None and np.isfinite(j) and abs(np.log(j / f)) < 0.15:
            out[(pd.Timestamp(d), s)] = float(f)
    return out


class Prices:
    """Active-contract daily open/close per symbol, with roll flags."""

    def __init__(self, F: pd.DataFrame, roll_before: int = 3, extra_ca: dict | None = None):
        F = F.copy()
        F["date"] = pd.to_datetime(F["date"])
        F["expiry"] = pd.to_datetime(F["expiry"])
        for c in ("open", "close"):
            F[c] = F[c].where(F[c] > 0)
        self.ca = corporate_action_factors(F)
        for k, v in (extra_ca or {}).items():
            self.ca.setdefault(k, v)
        cal = pd.DatetimeIndex(sorted(F["date"].unique()))
        self.cal = cal
        pos = pd.Series(np.arange(len(cal)), index=cal)
        F["i"] = F["date"].map(pos)
        exps = sorted(F["expiry"].unique())
        exp_i = pd.Series(_expiry_positions(cal, exps), index=exps)
        F["left"] = F["expiry"].map(exp_i) - F["i"]
        act = F[F["left"] > roll_before].sort_values("expiry").groupby(["date", "symbol"]).first().reset_index()
        self.active = act.set_index(["date", "symbol"])["expiry"]
        F = F.set_index(["date", "symbol", "expiry"])
        self.o = F["open"]
        self.c = F["close"]

    def px(self, d, s, e, which="close"):
        try:
            v = (self.c if which == "close" else self.o).loc[(d, s, e)]
            return float(v) if np.isfinite(v) else np.nan
        except KeyError:
            return np.nan


def simulate(scores: pd.DataFrame, P: Prices, enter=0.10, stay=0.20, cost=None, stops: pd.DataFrame | None = None,
             stop_mult=2.0, start=None, end=None, membership: dict | None = None, trail: bool = False,
             ban_after_stop: bool = True) -> pd.DataFrame:
    """scores: date, symbol, score (higher = long). Returns a daily frame of book returns."""
    cost = futures_cost() if cost is None else cost
    half = cost / 2
    scores = scores.dropna(subset=["score"]).copy()
    scores["date"] = pd.to_datetime(scores["date"])
    scores["pct"] = scores.groupby("date")["score"].rank(pct=True)
    by_day = {d: g.set_index("symbol")["pct"] for d, g in scores.groupby("date")}
    stp = None
    if stops is not None:
        stp = stops.set_index(["date", "symbol"])[["stopL", "stopS"]]
    cal = [d for d in P.cal if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
    book = {}       # symbol -> dict(side, expiry, ref, stop)
    banned = {}     # stopped-out names: no re-entry on that side until the signal leaves the band
    rows = []
    for k in range(len(cal) - 1):
        d, nxt = cal[k], cal[k + 1]
        pct = by_day.get(d)
        # 1) decide at close d
        want = {}
        if membership is not None:
            want = dict(membership.get(d, {s: v["side"] for s, v in book.items()}))
        elif pct is not None and len(pct) >= 50:
            for s, p in pct.items():
                cur = book.get(s, {}).get("side")
                if p > 1 - enter or (cur == 1 and p > 1 - stay):
                    want[s] = 1
                elif p <= enter or (cur == -1 and p <= stay):
                    want[s] = -1
        else:
            want = {s: v["side"] for s, v in book.items()}
        if ban_after_stop:
            banned = {s: sd for s, sd in banned.items() if want.get(s) == sd}
            want = {s: sd for s, sd in want.items() if banned.get(s) != sd or s in book}
        # 2) execute at open nxt; P&L for the day nxt
        pnl = {1: [], -1: []}
        trades = 0.0
        new_book = {}
        for s in set(book) | set(want):
            held = book.get(s)
            w = want.get(s)
            if held is not None:
                e = held["expiry"]
                o, c_prev, c_now = P.px(nxt, s, e, "open"), P.px(d, s, e), P.px(nxt, s, e)
                # stops are decided on the close of d (known at decision time) and executed at
                # the next open, like every other exit. (Checking the close of nxt and exiting at
                # its open would be look-ahead.)
                stopped = False
                f_d = P.ca.get((d, s))
                if f_d and held.get("stop"):            # close of d is post-adjustment: rescale levels
                    held = dict(held, best=held["best"] * f_d, stop=held["stop"] * f_d)
                f_n = P.ca.get((nxt, s), 1.0)
                c_prev_adj = c_prev * f_n if np.isfinite(c_prev) else c_prev
                if trail and held.get("stop") and np.isfinite(c_prev):
                    best = max(held["best"], c_prev) if held["side"] > 0 else min(held["best"], c_prev)
                    held = dict(held, best=best, stop=best * held["k"])
                if held.get("stop") and np.isfinite(c_prev):
                    lvl = held["stop"]
                    stopped = (c_prev <= lvl) if held["side"] > 0 else (c_prev >= lvl)
                if w == held["side"] and not stopped:
                    if not np.isfinite(c_now) or not np.isfinite(c_prev):
                        new_book[s] = held
                        continue
                    r = held["side"] * (c_now / c_prev_adj - 1)
                    act = P.active.get((nxt, s))
                    if act is not None and act != e:           # roll at the close
                        r -= cost
                        trades += 1
                        c_new = P.px(nxt, s, act)
                        if held.get("stop") and np.isfinite(c_new) and np.isfinite(c_now) and c_now > 0:
                            g = c_new / c_now                   # trail in the new contract's terms
                            held = dict(held, best=held["best"] * g, stop=held["stop"] * g)
                        held = dict(held, expiry=act)
                    pnl[held["side"]].append(r)
                    new_book[s] = held
                else:                                           # exit at the open
                    if np.isfinite(o) and np.isfinite(c_prev):
                        pnl[held["side"]].append(held["side"] * (o / c_prev_adj - 1) - half)
                    trades += 1
                    if stopped:
                        banned[s] = held["side"]
                        w = None
            if w is not None and s not in new_book and (held is None or held["side"] != w or True):
                if held is not None and held["side"] == w:
                    continue
                e = P.active.get((nxt, s))                 # the contract active on the fill day (no instant roll)
                if e is None:
                    continue
                o, c_now = P.px(nxt, s, e, "open"), P.px(nxt, s, e)
                if not (np.isfinite(o) and np.isfinite(c_now)):
                    continue
                stop, k = None, None
                if stp is not None and stop_mult and (d, s) in stp.index:
                    sl, ss = stp.loc[(d, s)]
                    base = sl if w > 0 else ss
                    if np.isfinite(base):
                        k = 1 + stop_mult * (base - 1)
                        stop = o * k
                pnl[w].append(w * (c_now / o - 1) - half)
                trades += 1
                new_book[s] = dict(side=w, expiry=e, stop=stop, k=k, best=o)
        book = new_book
        nl, ns = len(pnl[1]), len(pnl[-1])
        ret = (np.mean(pnl[1]) if nl else 0.0) + (np.mean(pnl[-1]) if ns else 0.0)
        rows.append(dict(date=nxt, ret=ret, names=",".join(sorted(f"{k}{'+' if v['side'] > 0 else '-'}" for k, v in book.items())),
                         n_long=sum(v["side"] == 1 for v in book.values()),
                         n_short=sum(v["side"] == -1 for v in book.values()), trades=trades))
    return pd.DataFrame(rows).set_index("date")


def stats(R: pd.DataFrame) -> dict:
    r = R["ret"]
    eq = (1 + r / 2).cumprod()   # each side's notional = half of a 2x-gross book on capital
    dd = (eq / eq.cummax() - 1).min()
    return dict(ann_ret=float(r.mean() * 252 / 2), ann_vol=float(r.std() * np.sqrt(252) / 2),
                sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
                max_dd=float(dd), days=len(r), trades_per_day=float(R["trades"].mean()),
                avg_names=float((R["n_long"] + R["n_short"]).mean()))
