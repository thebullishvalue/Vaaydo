"""
Settlement machinery shared by the research harness, the playbook and the journal:
daily marks at NSE settlement prices, exits (profit target / time), expiry-day handling.

History: v5.0's first engine ran a forecast-gated portfolio backtest here. It showed
the forecast had no edge (docs/RESEARCH.md §2), so selection moved to the evidence-gated
playbook and that runner was removed.

Original design notes:
Walk-forward backtest on real NSE option prices.

For every monthly stock-option expiry E in the window:
    entry   first trading day after the previous monthly expiry (+ `entry_lag`)
    scan    with data up to the entry close only
    book    `portfolio.construct` on the candidates that clear the gates
    manage  mark every leg daily at its bhavcopy close; exit on profit target, stop,
            or time (`exit_days_before` trading days before expiry: physical settlement)
    record  predicted EV / POP vs realised P&L, for the book AND for the best structure
            on every scanned name (gated or not), so the gates themselves can be judged

Honest limits, stated where results are shown:
    * fills are the close +/- an estimated half-spread, not real bid/ask
    * ~2 years of UDiFF data: ~25 monthly cycles; trades within a cycle share one market,
      so the effective sample is the number of cycles, not the number of trades
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from . import costs as C
from . import engine
from .config import DEFAULT, TRADING_DAYS, Config
from .data import bhavcopy, store
from . import flow as F

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Management:
    take_profit: float | None = 0.50   # close short-vol at this fraction of max profit
    stop_loss: float | None = None     # close when loss reaches this fraction of max loss
    long_take_profit: float | None = 1.0  # long vol: close at +100% of premium paid
    exit_days_before: int = 3          # trading days before expiry


def monthly_expiries() -> list[pd.Timestamp]:
    f = store.raw()
    stk = f[f["inst"] == "STF"]
    return sorted(stk["expiry"].unique())


def _leg_key(l):
    return (l.otype, float(l.strike))


MARK_COLS = ["inst", "symbol", "expiry", "strike", "otype", "close", "settle", "date"]


def expiry_day_marks(o: pd.DataFrame) -> pd.Series:
    """On its expiry day an option's 'settlement price' column holds the UNDERLYING's
    final settlement price, not the option's value. The option is worth its intrinsic
    value against that price."""
    if "date" not in o or "settle" not in o:
        return o["mark"]
    on_exp = pd.to_datetime(o["date"]).dt.normalize() == pd.to_datetime(o["expiry"]).dt.normalize()
    intrinsic = np.where(o["otype"] == "CE", np.maximum(o["settle"] - o["strike"], 0),
                         np.maximum(o["strike"] - o["settle"], 0))
    return o["mark"].where(~on_exp, intrinsic)


def _index_day(o: pd.DataFrame) -> dict:
    """symbol -> Series of marks indexed by (otype, strike), for one expiry on one day.

    Marks are NSE's daily settlement price: the price the clearing corporation marks every
    position at. An untraded contract's *close* just repeats an old close; its settlement
    price is recomputed from the day's underlying.
    """
    o = o.copy()
    o["mark"] = o["settle"].where(o["settle"] > 0, o["close"]) if "settle" in o else o["close"]
    o["mark"] = expiry_day_marks(o)
    return {s: g.set_index(["otype", "strike"])["mark"] for s, g in o.groupby("symbol", observed=True)}


def _mark(day_opts: dict, sym: str, expiry: pd.Timestamp, legs) -> np.ndarray | None:
    px = day_opts.get(sym)
    if px is None:
        return None
    out = []
    for l in legs:
        v = px.get(_leg_key(l))
        if v is None or not np.isfinite(v):
            return None
        out.append(float(v))
    return np.array(out)


def _settle_position(cand: engine.Candidate, days: list[pd.Timestamp], opts_by_day: dict, spot_px: pd.DataFrame,
                     mgmt: Management, cfg: Config) -> dict:
    e = cand.evaluation
    st = e.structure
    qty = np.array([l.qty for l in st.legs], dtype=float)
    lot = e.lot
    entry_value = e.entry_net                   # ₹/lot (+credit)
    K = np.array([l.strike for l in st.legs])
    Vc = np.array([l.contracts or 1 for l in st.legs], dtype=float)
    entry_cost = C.charges(C.fill([l.entry_ref for l in st.legs], qty, cfg, K, Vc), qty, lot, cfg)
    exit_reason, exit_day, pnl = "time", None, None
    from .evaluate import _grid, payoff
    bounds = payoff(st, _grid(st, cand.forward))
    lo, hi = float(bounds.min()), float(bounds.max())
    last_marks = None
    for d in days:
        marks = _mark(opts_by_day[d], cand.symbol, cand.expiry, st.legs)
        if marks is None:
            continue
        last_marks = marks
        # a defined-risk structure is never worth more/less than its payoff bounds
        held = float(np.clip((qty * marks).sum(), lo, hi))
        slip = float((np.abs(qty) * C.half_spread(marks, cfg, K, Vc)).sum())
        value = (held - slip) * lot                       # ₹ received (+) to unwind
        exit_cost = C.charges(C.fill(marks, -qty, cfg, K, Vc), -qty, lot, cfg)
        cur = entry_value + value - entry_cost - exit_cost
        exit_day = d
        if st.vol_side == "short":
            if mgmt.take_profit is not None and cur >= mgmt.take_profit * e.max_profit:
                exit_reason, pnl = "target", cur
                break
            if mgmt.stop_loss is not None and cur <= -mgmt.stop_loss * e.max_loss:
                exit_reason, pnl = "stop", cur
                break
        else:
            if mgmt.long_take_profit is not None and cur >= mgmt.long_take_profit * abs(entry_value):
                exit_reason, pnl = "target", cur
                break
        pnl = cur
    if pnl is None:   # never marked (should not happen): settle at intrinsic on last spot
        exit_reason = "intrinsic"
        S = spot_px[cand.symbol].dropna()
        S = S.loc[:days[-1]].iloc[-1]
        from .evaluate import payoff
        pnl = (payoff(st, np.array([S]))[0]) * lot + entry_value - entry_cost
    # realised vol over the holding window (the thing the forecast was about)
    sp = spot_px[cand.symbol].loc[days[0] - pd.Timedelta(days=1):days[-1]].dropna()
    r = np.log(sp / sp.shift(1)).dropna()
    rv = float(np.sqrt((r ** 2).mean() * TRADING_DAYS)) if len(r) > 3 else np.nan
    return dict(pnl=pnl, exit_reason=exit_reason, exit_day=exit_day, realised_vol=rv)


def flow_study(horizons=(5, 20)) -> pd.DataFrame:
    """Forward returns and forward realised vol by 5-day build-up state (Fama-MacBeth:
    cross-sectional mean each day, then t-stat across days)."""
    pos = F.history(store.positioning())
    sp = store.spot()
    r = np.log(sp / sp.shift(1))
    rows = []
    for h in horizons:
        fwd = (np.log(sp.shift(-h) / sp)).stack().rename("fwd").reset_index()
        fvol = (r[::-1].rolling(h).std()[::-1].shift(-1) * np.sqrt(TRADING_DAYS)).stack().rename("fvol").reset_index()
        hvol = (r.rolling(21).std() * np.sqrt(TRADING_DAYS)).stack().rename("hvol").reset_index()
        x = pos.merge(fwd, on=["date", "symbol"]).merge(fvol, on=["date", "symbol"]).merge(hvol, on=["date", "symbol"])
        x = x[x["symbol"] != "NIFTY"]
        x["fwd_x"] = x["fwd"] - x.groupby("date")["fwd"].transform("mean")   # vs the day's average stock
        x["vol_ratio"] = np.log(x["fvol"] / x["hvol"])
        daily = x.groupby(["date", "build5"]).agg(fwd_x=("fwd_x", "mean"), vol_ratio=("vol_ratio", "mean")).reset_index()
        rel = daily.copy()
        rel["vol_ratio"] = rel["vol_ratio"] - rel.groupby("date")["vol_ratio"].transform("mean")
        for state, g in rel.groupby("build5"):
            # overlapping horizons: use every h-th day for the t-stat
            gg = g.sort_values("date").iloc[::h]
            n = len(gg)
            rows.append(dict(horizon=h, state=state, days=n,
                             fwd_excess_ret=float(gg["fwd_x"].mean()),
                             t_ret=float(gg["fwd_x"].mean() / (gg["fwd_x"].std() / np.sqrt(n))) if n > 2 else np.nan,
                             fwd_vol_vs_avg=float(gg["vol_ratio"].mean()),
                             t_vol=float(gg["vol_ratio"].mean() / (gg["vol_ratio"].std() / np.sqrt(n))) if n > 2 else np.nan))
    return pd.DataFrame(rows)
