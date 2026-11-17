"""
The daily curator: from the latest NSE close to tomorrow's order list.

    prepare() -> Prepared                   slow, cached: data, scores, Pragyam, prices
    orders(prep, capital, leverage) -> Orders   fast: your book and your orders (and the model's)

Every position comes from `ledger.replay`, the backtest's own rules on real prices: entry
at the futures open after the signal, a trailing stop, exits on the signal or the stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import ledger as LG
from . import session, signal
from .data import bhavcopy

WINDOW = 160          # sessions replayed (covers every live position's history)
MIN_SESSIONS_TO_EXPIRY = 7


@dataclass
class Prepared:
    asof: pd.Timestamp
    dates: list
    model: dict                 # {date: {symbol: side}} score membership
    ext: dict                   # {date: {symbol: |pct - 0.5|}}
    scores: pd.DataFrame
    closes: pd.DataFrame        # adjusted spot
    opens: pd.DataFrame
    ratios: dict                # {(symbol, side): Series} Pragyam stop ratio
    levels: pd.DataFrame        # today's Pragyam levels per symbol
    fut: pd.DataFrame           # today's futures (contract to trade per symbol)
    context: pd.DataFrame
    margin_rate: float
    notes: list = field(default_factory=list)


@dataclass
class Orders:
    yours: pd.DataFrame
    model: pd.DataFrame


def _calendar():
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in bhavcopy.available_days()))


def _futures_day(d) -> pd.DataFrame:
    if bhavcopy.fetch_day(pd.Timestamp(d).date()) is None:
        return pd.DataFrame()
    f = bhavcopy.load_range([pd.Timestamp(d).date()], columns=["inst", "symbol", "expiry", "open", "close", "lot"],
                            filters=[("inst", "==", "STF")])
    f["expiry"] = pd.to_datetime(f["expiry"])
    return f


def _tradable_contracts(f: pd.DataFrame, asof: pd.Timestamp, cal) -> pd.DataFrame:
    """Per symbol: the nearest future with more than a week of sessions left."""
    def left(e):
        return int(((cal > asof) & (cal <= e)).sum()) if e <= cal[-1] else int(np.busday_count(asof.date(), e.date()))
    f = f.copy()
    f["left"] = f["expiry"].map(left)
    f = f[f["left"] > MIN_SESSIONS_TO_EXPIRY].sort_values("expiry")
    return f.groupby("symbol").first()


def _fill_from_nse(h: pd.DataFrame, days, names) -> pd.DataFrame:
    """Add sessions Yahoo has not posted yet from NSE's equity bhavcopy (same exchange
    prices; scaled to Yahoo's adjusted level by the last common close)."""
    from .data import equity
    h = h.copy()
    h["date"] = pd.to_datetime(h["date"])
    have = set(zip(h["symbol"], h["date"]))
    add = []
    for d in days:
        e = equity.fetch_day(d)
        if e is None:
            continue
        e = e[e["symbol"].isin(names)]
        for r in e.itertuples():
            if (r.symbol, pd.Timestamp(d)) in have:
                continue
            prior = h[(h["symbol"] == r.symbol) & (h["date"] < pd.Timestamp(d))].tail(1)
            k = 1.0
            if len(prior):
                pe = equity.fetch_day(prior["date"].iloc[0].date())
                pc = pe[pe["symbol"] == r.symbol]["close"] if pe is not None else pd.Series(dtype=float)
                if len(pc) and pc.iloc[0] > 0:
                    k = prior["close"].iloc[0] / pc.iloc[0]
            add.append(dict(date=pd.Timestamp(d), symbol=r.symbol, open=r.open * k, high=r.high * k, low=r.low * k,
                            close=r.close * k, adj=r.close * k, volume=r.qty))
    if add:
        h = pd.concat([h, pd.DataFrame(add)], ignore_index=True).sort_values(["symbol", "date"])
    return h


def prepare(margin_rate: float = 0.20, progress=None) -> Prepared:
    from . import pragyam as PG
    from .data import history
    notes = []
    days = session.recent_sessions(6)
    asof = pd.Timestamp(days[-1])
    sc = signal.scores(signal.extend_inputs(days, progress=progress))
    model: dict = {}
    signal.replay(sc, daily=model)
    dates = sorted(model)[-WINDOW:]
    if dates[-1] != asof:
        notes.append(f"Scores end {dates[-1].date()}; latest session {asof.date()}.")
    ext = {d: (g.set_index("symbol")["pct"] - 0.5).abs().to_dict() for d, g in sc[sc["date"].isin(dates)].groupby("date")}
    names = sorted(set().union(*[set(model[d]) for d in dates]))
    h = history.load(names, until=asof)
    h = h[h["symbol"].isin(names)]
    h = _fill_from_nse(h, days, names)
    closes = h.pivot(index="date", columns="symbol", values="close")
    opens = h.pivot(index="date", columns="symbol", values="open")
    ratios, lv = {}, []
    for s, d in h.groupby("symbol"):
        d = d.set_index("date").sort_index()[["open", "high", "low", "close"]]
        d = d[d.index <= asof].iloc[-1100:]
        if len(d) < 300:
            continue
        R = PG.run(d)
        ratios[(s, 1)] = R["stopL"] / d["close"]
        ratios[(s, -1)] = R["stopS"] / d["close"]
        last = R.iloc[-1]
        c = d["close"].iloc[-1]
        lv.append(dict(symbol=s, spot_close=c, lo_r=last["lower"] / c, up_r=last["upper"] / c, trend=last["T"],
                       regime=["calm", "normal", "stressed"][int(last["reg"])] if np.isfinite(last["reg"]) else None))
    cal = _calendar()
    fut = _tradable_contracts(_futures_day(asof), asof, cal)
    context = build_context(asof, days, names)
    return Prepared(asof, dates, model, ext, sc, closes, opens, ratios, pd.DataFrame(lv).set_index("symbol"),
                    fut, context, margin_rate, notes)


def _entry_price(sym: str, exec_date, cache: dict) -> tuple[float, object]:
    """Futures open on the entry session, for the contract that was traded then."""
    if exec_date is None:
        return np.nan, None
    key = pd.Timestamp(exec_date)
    if key not in cache:
        cache[key] = _futures_day(key)
    f = cache[key]
    if f.empty:
        return np.nan, None
    x = f[(f["symbol"] == sym) & (f["expiry"] > key + pd.Timedelta(days=10))].sort_values("expiry")
    if x.empty:
        return np.nan, None
    o = x["open"].iloc[0]
    return (float(o) if o > 0 else float(x["close"].iloc[0])), x["expiry"].iloc[0].date()


def _rows(prep: Prepared, led: LG.Ledger, capital: float) -> pd.DataFrame:
    asof, fut, lv = prep.asof, prep.fut, prep.levels
    cache: dict = {}
    rows = []
    items = [(s, p, "HOLD" if s in led.prev and led.prev[s].side == p.side else "ENTER", "") for s, p in led.held.items()]
    items += [(s, p, "EXIT", why) for s, (p, why) in led.exits_today.items()]
    for s, p, action, why in items:
        if s not in fut.index:
            continue
        c = fut.loc[s]
        last = float(c["close"])
        lot = int(c["lot"])
        spot = prep.closes[s].dropna().iloc[-1] if s in prep.closes else np.nan
        f2s = last / spot if np.isfinite(spot) and spot > 0 else np.nan
        stop = p.stop * f2s if (p.stop is not None and np.isfinite(f2s)) else np.nan
        if action == "ENTER":
            r = prep.ratios.get((s, p.side))
            rv = r.iloc[-1] if r is not None and len(r) else np.nan
            stop = last * (1 + LG.STOP_MULT * (rv - 1)) if np.isfinite(rv) else np.nan
        entry, entry_contract = _entry_price(s, p.exec, cache) if action != "ENTER" else (np.nan, None)
        pnl = p.side * (last - entry) * lot if np.isfinite(entry) else np.nan
        risk = abs(last - stop) * lot if np.isfinite(stop) else np.nan
        lvl = lv.loc[s] if s in lv.index else None
        if not why:
            why = ("new signal: score entered the top 5%" if p.side > 0 else "new signal: score entered the bottom 5%") \
                if action == "ENTER" and p.signal == asof else (
                "joins your book: signal since " + f"{p.signal:%d %b}" if action == "ENTER" else f"signal since {p.signal:%d %b}")
        rows.append(dict(
            action=action, symbol=s, side="LONG" if p.side > 0 else "SHORT", contract=pd.Timestamp(c["expiry"]).date(),
            lot=lot, signal_date=p.signal.date(), entry_date=p.exec.date() if p.exec is not None else None,
            entry_price=entry, entry_contract=entry_contract, last=last, pnl=pnl,
            pnl_pct=p.side * (last / entry - 1) if np.isfinite(entry) else np.nan,
            stop=stop, stop_from_last=(stop / last - 1) if np.isfinite(stop) else np.nan,
            stop_from_entry=(stop / entry - 1) if (np.isfinite(stop) and np.isfinite(entry)) else np.nan,
            risk_to_stop=risk, risk_pct_capital=risk / capital if np.isfinite(risk) else np.nan,
            range_lo=last * lvl["lo_r"] if lvl is not None else np.nan, range_hi=last * lvl["up_r"] if lvl is not None else np.nan,
            notional=last * lot, margin=last * lot * prep.margin_rate, risk_regime=lvl["regime"] if lvl is not None else None,
            score_pct=float(prep.scores[(prep.scores["date"] == prep.scores["date"].max()) & (prep.scores["symbol"] == s)]["pct"].iloc[0])
            if ((prep.scores["date"] == prep.scores["date"].max()) & (prep.scores["symbol"] == s)).any() else np.nan,
            why=why))
    out = pd.DataFrame(rows)
    if len(out):
        out["order"] = out["action"].map({"EXIT": 0, "ENTER": 1, "HOLD": 2})
        out["extreme"] = (out["score_pct"] - 0.5).abs()
        out = out.sort_values(["order", "side", "extreme"], ascending=[True, True, False]).drop(columns="order")
    return out


def orders(prep: Prepared, capital: float, leverage: float = 1.0) -> Orders:
    notional = {s: float(r["close"]) * int(r["lot"]) for s, r in prep.fut.iterrows()}
    model_led = LG.replay(prep.dates, prep.model, prep.closes, prep.opens, prep.ratios)
    your_led = LG.replay(prep.dates, prep.model, prep.closes, prep.opens, prep.ratios, ext=prep.ext,
                         notional=notional, cap=capital * leverage)
    return Orders(yours=_rows(prep, your_led, capital), model=_rows(prep, model_led, capital))


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
    return ctx
