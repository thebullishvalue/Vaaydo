"""
Panels built from the daily bhavcopy store.

    spot      date x symbol   underlying close, back-adjusted for splits/bonuses
    lot       date x symbol   lot size of the near-month contract
    futures   long table      near-month future close/return + total futures OI per symbol/day
    chain(d)  long table      options on day d, with tradability flags

Splits and bonuses: the bhavcopy carries raw prices. A 1:2 split halves the price and
doubles the lot overnight. We detect a day where |log return| is large AND the lot size
moved by the inverse ratio, and back-adjust everything before it. This keeps a split from
being read as a -50% crash (which would poison every volatility estimate).
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import DATA_DIR, DEFAULT
from . import bhavcopy

FUT = ("STF", "IDF")
DEMERGER_MOVE = 0.40  # |daily log move| no F&O stock has made for a non-corporate reason here
OPT = ("STO", "IDO")


def _panel_path(name: str):
    p = DATA_DIR / "panels"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.parquet"


def build_futures_long(days: list[date]) -> pd.DataFrame:
    """Every future contract on every day: the raw material for spot, lots and positioning."""
    cols = ["date", "inst", "symbol", "expiry", "close", "spot", "oi", "oi_chg", "contracts", "lot"]
    df = bhavcopy.load_range(days, columns=cols, filters=[("inst", "in", list(FUT))])
    return df.sort_values(["symbol", "expiry", "date"]).reset_index(drop=True)


def _refresh() -> pd.DataFrame:
    days = bhavcopy.available_days()
    p = _panel_path("futures_raw")
    if p.exists():
        old = pd.read_parquet(p)
        have = set(pd.to_datetime(old["date"]).dt.date.unique())
        new_days = [d for d in days if d not in have]
        if not new_days:
            return old
        df = pd.concat([old, build_futures_long(new_days)], ignore_index=True)
    else:
        df = build_futures_long(days)
    df = df.drop_duplicates(["date", "symbol", "expiry"]).sort_values(["symbol", "expiry", "date"])
    df.to_parquet(p, index=False)
    return df


@lru_cache(maxsize=1)
def futures_raw(_stamp: int = 0) -> pd.DataFrame:
    df = _refresh()
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def _stamp() -> int:
    days = bhavcopy.available_days()
    return int(days[-1].strftime("%Y%m%d")) * 10000 + len(days) if days else 0


def raw() -> pd.DataFrame:
    return futures_raw(_stamp())


def detect_corporate_actions(spot: pd.DataFrame, lot: pd.DataFrame, max_abs: float | None = None) -> pd.DataFrame:
    """Return factor table (date x symbol): multiply prices BEFORE `date` by factor."""
    max_abs = max_abs or DEFAULT.forecast.max_abs_return
    r = np.log(spot / spot.shift(1))
    lot_ratio = lot / lot.shift(1)
    px_ratio = spot / spot.shift(1)
    # split/bonus: price ratio ~ 1 / lot ratio, and a big move
    consistent = (np.abs(np.log(px_ratio * lot_ratio)) < 0.15) & (lot_ratio > 1.2)
    # demergers (SIEMENS Apr-25, ABFRL May-25, TATAMOTORS Oct-25) keep their lot size, so
    # a move beyond DEMERGER_MOVE is also treated as a corporate action. Genuine crashes
    # (IEX -35% Jul-25, INDUSINDBK -32% Mar-25) stay in the data: they are the tail risk.
    ca = ((np.abs(r) > max_abs) & consistent) | (np.abs(r) > DEMERGER_MOVE)
    return px_ratio.where(ca)


@lru_cache(maxsize=2)
def _spot_lot(_stamp: int):
    f = raw()
    # spot: identical on every row of a symbol/day
    spot = f.groupby(["date", "symbol"], observed=True)["spot"].first().unstack()
    # lot: from the nearest expiry on that day
    near = f.sort_values("expiry").groupby(["date", "symbol"], observed=True).first()
    lot = near["lot"].unstack().reindex_like(spot)
    spot = spot.where(spot > 0)
    factors = detect_corporate_actions(spot, lot)
    adj = spot.copy()
    for sym in factors.columns[factors.notna().any()]:
        for d, fac in factors[sym].dropna().items():
            adj.loc[adj.index < d, sym] *= fac
    return adj, lot, factors


def spot() -> pd.DataFrame:
    return _spot_lot(_stamp())[0]


def lots() -> pd.DataFrame:
    return _spot_lot(_stamp())[1]


def corporate_actions() -> pd.DataFrame:
    f = _spot_lot(_stamp())[2]
    s = f.stack().dropna()
    return s.rename("price_ratio").reset_index()


def returns() -> pd.DataFrame:
    s = spot()
    return np.log(s / s.shift(1))


@lru_cache(maxsize=2)
def _positioning(_stamp: int) -> pd.DataFrame:
    """Per symbol/day: near-month future return (same-contract) and total futures OI change.

    Using the same contract's previous close avoids reading the roll gap as a move; summing
    OI across expiries avoids reading the rollover (near-month OI moving to next month) as
    unwinding.
    """
    f = raw().copy()
    f["prev_close"] = f.groupby(["symbol", "expiry"], observed=True)["close"].shift(1)
    f["ret"] = np.log(f["close"] / f["prev_close"])
    # near-month: earliest expiry strictly after today (expiry day contracts are dying)
    alive = f[f["expiry"] > f["date"]]
    near = alive.sort_values("expiry").groupby(["date", "symbol"], observed=True).first()
    tot = f.groupby(["date", "symbol"], observed=True).agg(oi=("oi", "sum"), oi_chg=("oi_chg", "sum"),
                                                          contracts=("contracts", "sum"))
    out = near[["expiry", "close", "ret", "lot"]].join(tot)
    out["oi_prev"] = out["oi"] - out["oi_chg"]
    return out.reset_index()


def positioning() -> pd.DataFrame:
    return _positioning(_stamp())


def chain(d: date, symbol: str | None = None) -> pd.DataFrame:
    """Options for one day (optionally one symbol), with the near future as forward."""
    filters = [("inst", "in", list(OPT))]
    if symbol:
        filters.append(("symbol", "==", symbol))
    opts = bhavcopy.load_range([d], filters=filters)
    if opts.empty:
        return opts
    opts["expiry"] = pd.to_datetime(opts["expiry"])
    fut = raw()
    fut = fut[fut["date"] == pd.Timestamp(d)][["symbol", "expiry", "close"]].rename(columns={"close": "fwd"})
    opts = opts.merge(fut, on=["symbol", "expiry"], how="left")
    return opts
