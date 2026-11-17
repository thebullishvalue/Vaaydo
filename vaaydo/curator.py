"""
The daily curator: from the latest NSE close to tomorrow's order list.

    prepare() -> Curation                     (slow: data, scores, model book, levels, context)
    personalize(cur, capital, leverage)       (fast: your book and your orders)
    run(capital, leverage) -> Curation         (both)
        .actions   ENTER / EXIT / HOLD for every model-book name, with the futures contract,
                   lots, reference price, disaster stop, Pragyam's 10-session range and why
        .yours     the subset that fits your capital (most extreme first, long/short balanced)
        .context   Order Flow reading for every name: build-up, OI walls, PCR, IV, skew, event
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import session, signal
from .data import bhavcopy

STOP_MULT = 2.0
ROLL_BEFORE = 3     # sessions before expiry at which a position rolls


@dataclass
class Curation:
    asof: pd.Timestamp
    actions: pd.DataFrame
    yours: pd.DataFrame
    context: pd.DataFrame
    scores: pd.DataFrame
    log: pd.DataFrame
    notes: list = field(default_factory=list)


def _futures_today(d: date) -> pd.DataFrame:
    f = bhavcopy.load_range([d], columns=["date", "inst", "symbol", "expiry", "close", "settle", "lot", "spot", "oi", "contracts"],
                            filters=[("inst", "==", "STF")])
    f["expiry"] = pd.to_datetime(f["expiry"])
    return f


def _contract(fut: pd.DataFrame, sym: str, asof: pd.Timestamp, sessions_left) -> pd.Series | None:
    x = fut[fut["symbol"] == sym].sort_values("expiry")
    for _, r in x.iterrows():
        if sessions_left(r["expiry"]) > max(ROLL_BEFORE, 7):
            return r
    return None


def pragyam_levels(symbols: list[str], asof: pd.Timestamp) -> pd.DataFrame:
    """Pragyam on ~4 years of daily history per symbol, read at `asof`."""
    from . import pragyam as PG
    from .data import history
    h = history.load(symbols)
    rows = []
    for s, d in h.groupby("symbol"):
        if s not in symbols:
            continue
        d = d.set_index("date").sort_index()
        d = d[d.index <= asof][["open", "high", "low", "close"]].iloc[-1100:]
        if len(d) < 300:
            continue
        R = PG.run(d).iloc[-1]
        c = d["close"].iloc[-1]
        rows.append(dict(symbol=s, prg_close=c, stopL_r=R["stopL"] / c, stopS_r=R["stopS"] / c,
                         up_r=R["upper"] / c, lo_r=R["lower"] / c, sig=R["sig"], trend=R["T"],
                         regime=["calm", "normal", "stressed"][int(R["reg"])] if np.isfinite(R["reg"]) else None))
    return pd.DataFrame(rows)


def prepare(margin_rate: float = 0.20, n_context_days: int = 6, progress=None) -> Curation:
    """Everything that does not depend on your capital: data, scores, model book, levels,
    context. Leverage (gross notional / capital) is applied in `personalize`; 1.0 is the
    tested default: at 3x a -17% book drawdown becomes about -50% of capital."""
    notes = []
    days = session.recent_sessions(max(n_context_days, 2))
    asof = pd.Timestamp(days[-1])
    inputs = signal.extend_inputs(days, progress=progress)
    sc = signal.scores(inputs)
    daily: dict = {}
    book, log = signal.replay(sc, daily=daily)
    last = sc[sc["date"] == sc["date"].max()].set_index("symbol")
    if sc["date"].max() != asof:
        notes.append(f"Scores end {sc['date'].max().date()} but the latest session is {asof.date()}.")

    cal = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in bhavcopy.available_days()))

    def sessions_left(exp):
        exp = pd.Timestamp(exp)
        if exp <= cal[-1]:
            return int(((cal > asof) & (cal <= exp)).sum())
        return int(np.busday_count(asof.date(), exp.date()))

    fut = _futures_today(asof.date())
    today_log = log[log["date"] == log["date"].max()] if len(log) else log
    exits = today_log[today_log["action"] == "exit"]["symbol"].tolist() if len(log) else []
    names = sorted(set(book) | set(exits))
    prg = pragyam_levels(names, asof)
    prg = prg.set_index("symbol") if len(prg) else pd.DataFrame()

    # entry reference for held names: futures close on the entry session (for P&L and stop)
    entry_px = {}
    for s, p in book.items():
        try:
            f0 = _futures_today(p.since.date()) if p.since.date() in set(bhavcopy.available_days()) else None
            if f0 is None:
                bhavcopy.fetch_day(p.since.date())
                f0 = _futures_today(p.since.date())
            c0 = f0[(f0["symbol"] == s)].sort_values("expiry")
            c0 = c0[c0["expiry"] > p.since + pd.Timedelta(days=7)]
            if len(c0):
                entry_px[s] = float(c0["close"].iloc[0])
        except Exception:
            pass
    entry_prg = {}
    try:
        for s, p in book.items():
            if p.since != asof:
                lv = pragyam_levels([s], p.since)
                if len(lv):
                    entry_prg[s] = lv.iloc[0]
    except Exception:
        pass

    rows = []
    for s in names:
        p = book.get(s)
        side = p.side if p else int(today_log[today_log["symbol"] == s]["side"].iloc[0])
        c = _contract(fut, s, asof, sessions_left)
        if c is None:
            notes.append(f"{s}: no futures contract far enough from expiry")
            continue
        ref = float(c["close"])
        lot = int(c["lot"])
        pr = prg.loc[s] if len(prg) and s in prg.index else None
        # disaster stop: 2x Pragyam's 20%-touch distance, set at entry, on the futures price
        if p and s in entry_px and s in entry_prg:
            e_r = entry_prg[s]["stopL_r"] if side > 0 else entry_prg[s]["stopS_r"]
            stop = entry_px[s] * (1 + STOP_MULT * (e_r - 1))
        elif pr is not None:
            stop = ref * (1 + STOP_MULT * ((pr["stopL_r"] if side > 0 else pr["stopS_r"]) - 1))
        else:
            stop = np.nan
        breached = bool(p and np.isfinite(stop) and ((ref <= stop) if side > 0 else (ref >= stop)))
        if not p:
            action = "EXIT"
            why = "left the band (score no longer in the top/bottom 15%)"
        elif breached:
            action = "EXIT"
            why = "disaster stop breached"
        elif p.since == asof:
            action = "ENTER"
            why = "score entered the top 5%" if side > 0 else "score entered the bottom 5%"
        else:
            action = "HOLD"
            why = f"in the book since {p.since.date()}"
        e0 = entry_px.get(s)
        stop_today = (ref * (1 + STOP_MULT * ((pr["stopL_r"] if side > 0 else pr["stopS_r"]) - 1))
                      if pr is not None else np.nan)
        rows.append(dict(stop_today=stop_today,
            action=action, symbol=s, side="LONG" if side > 0 else "SHORT", contract=c["expiry"].date(),
            lot=lot, ref_price=ref, notional=ref * lot, margin_est=ref * lot * margin_rate,
            stop=stop, stop_dist=abs(stop / ref - 1) if np.isfinite(stop) else np.nan,
            range_lo=ref * pr["lo_r"] if pr is not None else np.nan, range_hi=ref * pr["up_r"] if pr is not None else np.nan,
            since=p.since.date() if p else None,
            pnl_pct=(side * (ref / e0 - 1)) if (p and e0 and p.since != asof) else np.nan,
            score_pct=float(last["pct"].get(s, np.nan)), opi_today=float(last["opi_pct"].get(s, np.nan)),
            trend=pr["trend"] if pr is not None else np.nan, risk_regime=pr["regime"] if pr is not None else None,
            sessions_to_expiry=sessions_left(c["expiry"]), why=why))
    actions = pd.DataFrame(rows)
    order = {"EXIT": 0, "ENTER": 1, "HOLD": 2}
    if len(actions):
        actions["extreme"] = (actions["score_pct"] - 0.5).abs()
        actions = actions.sort_values(["action", "extreme"], key=lambda s: s.map(order) if s.name == "action" else -s)

    # lot notional for every name the model book held in the last 40 sessions (for your book)
    notional = {}
    for s_ in set().union(*[set(v) for v in list(daily.values())[-40:]]) if daily else []:
        c_ = _contract(fut, s_, asof, sessions_left)
        if c_ is not None:
            notional[s_] = (float(c_["close"]) * int(c_["lot"]), c_["expiry"].date(), int(c_["lot"]), float(c_["close"]))
    ext = {d: (g.set_index("symbol")["pct"] - 0.5).abs().to_dict()
           for d, g in sc[sc["date"].isin(list(daily)[-40:])].groupby("date")}
    context = build_context(asof, days, names)
    cur = Curation(asof, actions, pd.DataFrame(), context, sc, log, notes)
    cur.daily, cur.book, cur.notional, cur.ext, cur.margin_rate = daily, book, notional, ext, margin_rate
    return cur


def run(capital: float = 5_000_000, leverage: float = 1.0, margin_rate: float = 0.20, progress=None) -> Curation:
    cur = prepare(margin_rate=margin_rate, progress=progress)
    cur.yours = personalize(cur, capital, leverage)
    return cur


def personalize(cur: Curation, capital: float, leverage: float = 1.0) -> pd.DataFrame:
    """Your book and your orders for this capital and leverage (gross notional / capital)."""
    actions = cur.actions
    if actions is None or actions.empty:
        return pd.DataFrame()
    stopped = set(actions[(actions["action"] == "EXIT") & (actions["why"] == "disaster stop breached")]["symbol"])
    daily = {d: {s: v for s, v in m.items() if not (d == max(cur.daily) and s in stopped)} for d, m in cur.daily.items()}
    nmap = {s: v[0] for s, v in cur.notional.items()}
    prev_y, now_y = signal.personal_replay(daily, cur.ext, nmap, nmap, capital * leverage)
    rows = []
    for s_, sd in now_y.items():
        r = actions[actions["symbol"] == s_]
        if len(r):
            r = r.iloc[0].copy()
            r["your_action"] = "HOLD" if prev_y.get(s_) == sd else "ENTER"
            if r["your_action"] == "ENTER" and r["action"] == "HOLD":
                # in the model book already; joins yours now that capital is free
                r["why"] = f"in the model book since {r['since']}; joins your book as capital frees up"
                r["pnl_pct"] = np.nan
                if np.isfinite(r.get("stop_today", np.nan)):          # your risk starts at today's price
                    r["stop"] = r["stop_today"]
                    r["stop_dist"] = abs(r["stop"] / r["ref_price"] - 1)
            rows.append(r)
    for s_, sd in prev_y.items():
        if now_y.get(s_) == sd:
            continue
        r = actions[actions["symbol"] == s_]
        if len(r):
            r = r.iloc[0].copy()
        elif s_ in cur.notional:
            n, exp, lot, px = cur.notional[s_]
            r = pd.Series(dict(symbol=s_, side="LONG" if sd > 0 else "SHORT", contract=exp, lot=lot, ref_price=px,
                               notional=n, margin_est=n * cur.margin_rate, why="left the model book"))
        else:
            continue
        r["your_action"] = "EXIT"
        if s_ in cur.book and cur.book[s_].side == sd and s_ not in stopped:
            r["why"] = "displaced by a more extreme name (capital limit)"
        rows.append(r)
    y = pd.DataFrame(rows)
    if len(y):
        order = {"EXIT": 0, "ENTER": 1, "HOLD": 2}
        y = y.sort_values("your_action", key=lambda s: s.map(order))
    return y


def fit_capital(actions: pd.DataFrame, capital: float, budget: float) -> pd.DataFrame:  # kept for API compatibility
    """1 lot per name. Most extreme first, alternating long and short, until the margin
    budget (capital x budget) is used; keeps long and short notional within one lot."""
    if actions is None or actions.empty:
        return pd.DataFrame()
    live = actions[actions["action"] != "EXIT"].copy()
    longs = live[live["side"] == "LONG"].sort_values("extreme", ascending=False)
    shorts = live[live["side"] == "SHORT"].sort_values("extreme", ascending=False)
    cap = capital * budget
    chosen, used, nl, ns = [], 0.0, 0.0, 0.0
    li, si = 0, 0
    while li < len(longs) or si < len(shorts):
        take_long = (nl <= ns and li < len(longs)) or si >= len(shorts)
        r = longs.iloc[li] if take_long else shorts.iloc[si]
        if take_long:
            li += 1
        else:
            si += 1
        if used + r["margin_est"] > cap:
            continue
        chosen.append(r)
        used += r["margin_est"]
        if take_long:
            nl += r["notional"]
        else:
            ns += r["notional"]
    return pd.DataFrame(chosen)


def build_context(asof: pd.Timestamp, days, names) -> pd.DataFrame:
    """Order Flow's reading for the names: positioning, options, delivery."""
    from .features import day_features
    f = day_features(asof.date())
    f = f[f["symbol"].isin(names)][["symbol", "atm_iv", "skew", "cskew", "cpiv", "pcr_vol", "pcr_oi", "put_oi_flow",
                                    "call_oi_flow", "term", "max_pain_d", "wall_put_d", "wall_call_d", "basis", "fwd"]]
    fc = session.futures_context(days)
    dl = session.delivery(asof.date())
    ctx = f.merge(fc, on="symbol", how="left").merge(dl, on="symbol", how="left")
    ctx["event"] = ctx["term"] > 1.08
    ctx["wall_put"] = ctx["fwd"] * np.exp(ctx["wall_put_d"])
    ctx["wall_call"] = ctx["fwd"] * np.exp(ctx["wall_call_d"])
    ctx["max_pain"] = ctx["fwd"] * np.exp(ctx["max_pain_d"])
    return ctx
