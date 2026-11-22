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
    """The ledger's: one definition of sessions-to-expiry (NSE holidays included) for both engines."""
    from .ledger import _expiry_positions as f
    return f(cal, expiries)


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
        for c in ("open", "close", "high", "low"):
            if c in F:
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
        self.roll_before = roll_before
        self.cal_pos = {**{e: int(v) for e, v in exp_i.items()}, **{d: i for i, d in enumerate(cal)}}
        self.present = set(zip(F["date"], F["symbol"]))          # the symbol is in the day's file
        F = F.set_index(["date", "symbol", "expiry"])
        self.o = F["open"]
        self.c = F["close"]
        self.h = F["high"] if "high" in F else None       # the session's range: only the intraday-stop research uses it
        self.l = F["low"] if "low" in F else None
        # rupee weight of one lot: the exchange's lot where the file has one, else the estimate
        # (lot_w, 2019 - Jul 2024: NSE's old file carries no lot). Never used to detect corporate actions.
        lw = F["lot"].where(F["lot"] > 0) if "lot" in F else pd.Series(np.nan, index=F.index)
        if "lot_w" in F:
            lw = lw.fillna(F["lot_w"])
        self.lots = lw.to_dict()

    def lot(self, d, s, e) -> float:
        v = self.lots.get((d, s, e))
        return float(v) if v is not None and v > 0 else np.nan

    def px(self, d, s, e, which="close"):
        try:
            src = {"close": self.c, "open": self.o, "high": self.h, "low": self.l}[which]
            if src is None:
                return np.nan
            v = src.loc[(d, s, e)]
            return float(v) if np.isfinite(v) else np.nan
        except KeyError:
            return np.nan


def simulate(scores: pd.DataFrame, P: Prices, enter=0.10, stay=0.20, cost=None, stops: pd.DataFrame | None = None,
             stop_mult=2.0, start=None, end=None, membership: dict | None = None, trail: bool = False,
             ban_after_stop: bool = True, fno_ban: frozenset | None = None, attrib: dict | None = None,
             intraday_stop: bool = False, fill_adj: dict | None = None) -> pd.DataFrame:
    """scores: date, symbol, score (higher = long). Returns a daily frame:
      ret        equal rupees in every name: (mean long return + mean short return), the research measure
      ret_lot    ONE LOT of every name (what VAAYDO trades): rupee P&L after costs / gross rupee exposure
      pnl_rs     one-lot rupee P&L before costs (reconciles with the live ledger's legs)

    Research variants (docs/PREREGISTRATION_EXECUTION.md; the live book uses neither):
      intraday_stop  the stop is a resting order: exit the session its low (long) / high (short) reaches it,
                     at the stop level, or at the open if the open is already through it (needs high/low)
      fill_adj       {(date, symbol): ratio}: every open fill (entries and exits) at open × ratio"""
    cost = futures_cost() if cost is None else cost
    half = cost / 2
    scores = scores.dropna(subset=["score"]).copy()
    scores["date"] = pd.to_datetime(scores["date"])
    scores["pct"] = scores.groupby("date")["score"].rank(pct=True)
    by_day = {d: g.set_index("symbol")["pct"] for d, g in scores.groupby("date")}
    stp = stpm = None
    if stops is not None:
        stp = stops.set_index(["date", "symbol"])[["stopL", "stopS"]]
        stpm = stops.groupby("date")[["stopL", "stopS"]].median()      # fallback: the day's median distance
    cal = [d for d in P.cal if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
    book = {}       # symbol -> dict(side, expiry, ref, stop)
    banned = {}     # stopped-out names: no re-entry on that side until the signal leaves the band
    model = {}      # signal-level membership (hysteresis on the signal alone)
    rows = []
    for k in range(len(cal) - 1):
        d, nxt = cal[k], cal[k + 1]
        pct = by_day.get(d)
        # 1) decide at close d
        want = {}
        if membership is not None:
            want = dict(membership.get(d, {s: v["side"] for s, v in book.items()}))
        elif pct is not None and len(pct) >= 50:
            # signal membership is the MODEL's (enter 5%, stay 15%), kept independently of what
            # the trading book holds, so a stopped-out name stays banned while its signal stays
            # in the band: the same rule as the live ledger
            new_model = {}
            for s, p in pct.items():
                cur = model.get(s)
                if p > 1 - enter or (cur == 1 and p > 1 - stay):
                    new_model[s] = 1
                elif p <= enter or (cur == -1 and p <= stay):
                    new_model[s] = -1
            model = new_model
            want = dict(model)
        else:
            want = {s: v["side"] for s, v in book.items()}
        if ban_after_stop:
            banned = {s: sd for s, sd in banned.items() if want.get(s) == sd}
            want = {s: sd for s, sd in want.items() if banned.get(s) != sd or s in book}
        # 2) execute at open nxt; P&L for the day nxt
        pnl = {1: [], -1: []}
        lotw = []                 # (return after costs, rupee weight) per position for the one-lot book
        lotside = []              # signed rupee weights: the one-lot book's net exposure
        rs = 0.0                  # one-lot rupee P&L before costs
        trades = 0.0
        new_book = {}

        def book_lot(side, r_net, gross_rs, weight_rs, sym=None):
            nonlocal rs
            if attrib is not None and sym is not None and np.isfinite(gross_rs):
                attrib[(sym, nxt)] = attrib.get((sym, nxt), 0.0) + gross_rs
            if np.isfinite(weight_rs) and weight_rs > 0:
                lotw.append((r_net, weight_rs))
                lotside.append(side * weight_rs)
            if np.isfinite(gross_rs):
                rs += gross_rs
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
                if f_n != 1.0 and np.isfinite(held.get("qty", np.nan)):
                    held = dict(held, qty=round(held["qty"] / f_n))   # split/bonus: more shares, same position
                q = held.get("qty", np.nan)          # shares held: fixed through an exchange lot revision
                c_prev_adj = c_prev * f_n if np.isfinite(c_prev) else c_prev
                if trail and held.get("stop") and np.isfinite(c_prev):
                    best = max(held["best"], c_prev) if held["side"] > 0 else min(held["best"], c_prev)
                    held = dict(held, best=best, stop=best * held["k"])
                if held.get("stop") and np.isfinite(c_prev) and not intraday_stop:
                    lvl = held["stop"]
                    stopped = (c_prev <= lvl) if held["side"] > 0 else (c_prev >= lvl)
                if fill_adj is not None and np.isfinite(o):
                    o = o * fill_adj.get((nxt, s), 1.0)
                if (nxt, s) not in P.present and nxt < e:
                    # the contract stopped printing before expiry (merger, rename): closed at its last print
                    pnl[held["side"]].append(-half)
                    book_lot(held["side"], -half, 0.0, q * c_prev if np.isfinite(c_prev) else np.nan, sym=s)
                    trades += 1
                    continue
                act_n = P.active.get((nxt, s))
                if act_n is None and P.cal_pos.get(e, 10 ** 9) - P.cal_pos.get(nxt, 0) <= P.roll_before \
                        and np.isfinite(c_now) and np.isfinite(c_prev) and w == held["side"] and not stopped:
                    # (an exit already decided fills at this open instead: handled below)
                    # roll due, no later contract listed (leaving F&O): closed at this close, like a roll
                    ra = held["side"] * (c_now / c_prev_adj - 1)
                    pnl[held["side"]].append(ra - half)
                    book_lot(held["side"], ra - half, held["side"] * q * (c_now - c_prev_adj), q * c_prev_adj, sym=s)
                    trades += 1
                    continue
                if intraday_stop and w == held["side"] and held.get("stop") and np.isfinite(c_prev) and np.isfinite(o):
                    lvl = held["stop"] * f_n                       # in the session's own (post-adjustment) terms
                    lo_, hi_ = P.px(nxt, s, e, "low"), P.px(nxt, s, e, "high")
                    hit = (np.isfinite(lo_) and lo_ <= lvl) if held["side"] > 0 else (np.isfinite(hi_) and hi_ >= lvl)
                    if hit:
                        fill = (min(o, lvl) if held["side"] > 0 else max(o, lvl))
                        rx = held["side"] * (fill / c_prev_adj - 1)
                        pnl[held["side"]].append(rx - half)
                        book_lot(held["side"], rx - half, held["side"] * q * (fill - c_prev_adj), q * c_prev_adj, sym=s)
                        trades += 1
                        if ban_after_stop:
                            banned[s] = held["side"]
                        continue
                if w == held["side"] and not stopped:
                    if not np.isfinite(c_now) or not np.isfinite(c_prev):
                        new_book[s] = held
                        continue
                    r = held["side"] * (c_now / c_prev_adj - 1)
                    lt = q
                    g_rs = held["side"] * lt * (c_now - c_prev_adj)
                    act = P.active.get((nxt, s))
                    if act is not None and act != e and fno_ban and (nxt, s) in fno_ban:
                        if nxt >= e:                            # ban through expiry: settles, position closes
                            pnl[held["side"]].append(r - half)
                            book_lot(held["side"], r - half, g_rs, lt * c_prev_adj, sym=s)
                            trades += 1
                            continue
                        act = None                              # hold the current contract, roll later
                    if act is not None and act != e:           # roll at the close
                        r -= cost
                        trades += 1
                        c_new = P.px(nxt, s, act)
                        if held.get("stop") and np.isfinite(c_new) and np.isfinite(c_now) and c_now > 0:
                            g = c_new / c_now                   # trail in the new contract's terms
                            held = dict(held, best=held["best"] * g, stop=held["stop"] * g)
                        held = dict(held, expiry=act, qty=P.lot(nxt, s, act))   # the new contract's lot
                    pnl[held["side"]].append(r)
                    book_lot(held["side"], r, g_rs, lt * c_prev_adj, sym=s)
                    new_book[s] = held
                else:                                           # exit at the open
                    if np.isfinite(o) and np.isfinite(c_prev):
                        rx = held["side"] * (o / c_prev_adj - 1)
                        pnl[held["side"]].append(rx - half)
                        book_lot(held["side"], rx - half, held["side"] * q * (o - c_prev_adj), q * c_prev_adj, sym=s)
                    trades += 1
                    if stopped:
                        banned[s] = held["side"]
                        w = None
            if w is not None and s not in new_book and (held is None or held["side"] != w or True):
                if held is not None and held["side"] == w:
                    continue
                e = P.active.get((nxt, s))                 # the contract active on the fill day (no instant roll)
                if e is None or (fno_ban and (nxt, s) in fno_ban):   # no fresh position during an F&O ban
                    continue
                o, c_now = P.px(nxt, s, e, "open"), P.px(nxt, s, e)
                if fill_adj is not None and np.isfinite(o):
                    o = o * fill_adj.get((nxt, s), 1.0)
                if not (np.isfinite(o) and np.isfinite(c_now)):
                    continue
                stop, k = None, None
                if stp is not None and stop_mult:
                    sl, ss = stp.loc[(d, s)] if (d, s) in stp.index else (np.nan, np.nan)
                    base = sl if w > 0 else ss
                    if not np.isfinite(base) and d in stpm.index:
                        base = stpm.loc[d, "stopL" if w > 0 else "stopS"]
                    if np.isfinite(base):
                        k = 1 + stop_mult * (base - 1)
                        stop = o * k
                pnl[w].append(w * (c_now / o - 1) - half)
                lt = P.lot(nxt, s, e)
                book_lot(w, w * (c_now / o - 1) - half, w * lt * (c_now - o), lt * o, sym=s)
                trades += 1
                new_book[s] = dict(side=w, expiry=e, stop=stop, k=k, best=o, qty=lt)
        book = new_book
        nl, ns = len(pnl[1]), len(pnl[-1])
        ret = (np.mean(pnl[1]) if nl else 0.0) + (np.mean(pnl[-1]) if ns else 0.0)
        wsum = sum(w for _, w in lotw)
        ret_lot = sum(r_ * w for r_, w in lotw) / wsum if wsum > 0 else 0.0
        net_lot = sum(lotside) / wsum if wsum > 0 else 0.0
        rows.append(dict(date=nxt, ret=ret, ret_lot=ret_lot, net_lot=net_lot, pnl_rs=rs, names=",".join(sorted(f"{k}{'+' if v['side'] > 0 else '-'}" for k, v in book.items())),
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
