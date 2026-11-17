"""
What the live curator needs, and nothing more, so it runs on Streamlit Community Cloud:

    * the latest few NSE F&O bhavcopies (today's for the signal; the previous five for
      futures build-up context), fetched on demand (~1.4 MB each)
    * the latest equity bhavcopy (delivery %), fetched on demand
    * ~3 years of Yahoo daily history for the names that make the list (Pragyam's stops)

All cached on local disk for the life of the container.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from .data import bhavcopy, equity


def recent_sessions(n: int = 6, today: date | None = None, max_back: int = 20) -> list[date]:
    """The last `n` sessions whose F&O bhavcopy is published, newest last."""
    today = today or date.today()
    s = requests.Session()
    got = []
    d = today
    for _ in range(max_back + n * 2):
        if d.weekday() < 5 and bhavcopy.fetch_day(d, s) is not None:
            got.append(d)
            if len(got) >= n:
                break
        d -= timedelta(days=1)
        if (today - d).days > max_back + n * 2:
            break
    return sorted(got)


def futures_context(days: list[date]) -> pd.DataFrame:
    """The positioning reading of each future, end of day: 1-day and 5-day build-up."""
    from . import flow as F
    fr = bhavcopy.load_range(days, columns=["date", "inst", "symbol", "expiry", "close", "oi", "oi_chg", "spot"],
                             filters=[("inst", "in", ["STF", "IDF"])])
    fr["date"] = pd.to_datetime(fr["date"])
    fr["expiry"] = pd.to_datetime(fr["expiry"])
    fr = fr.sort_values(["symbol", "expiry", "date"])
    fr["ret"] = np.log(fr["close"] / fr.groupby(["symbol", "expiry"])["close"].shift(1))
    alive = fr[fr["expiry"] > fr["date"]]
    near = alive.sort_values("expiry").groupby(["date", "symbol"]).first()
    tot = fr.groupby(["date", "symbol"]).agg(oi=("oi", "sum"), oi_chg=("oi_chg", "sum"))
    x = near[["ret", "close"]].join(tot).reset_index().sort_values(["symbol", "date"])
    x["oi_pct"] = x["oi_chg"] / (x["oi"] - x["oi_chg"]).where(lambda v: v > 0)
    g = x.groupby("symbol")
    x["ret5"] = g["ret"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    x["oi5"] = x["oi"] / g["oi"].shift(min(5, len(days) - 1)) - 1
    last = x[x["date"] == x["date"].max()].copy()
    last["build1"] = F.classify(last["ret"].fillna(0).to_numpy(), last["oi_pct"].fillna(0).to_numpy(), 1)
    last["build5"] = F.classify(last["ret5"].fillna(0).to_numpy(), last["oi5"].fillna(0).to_numpy(), 5)
    return last[["symbol", "ret", "oi_pct", "ret5", "oi5", "build1", "build5"]].rename(
        columns={"ret": "fut_ret1", "oi_pct": "fut_oi1", "ret5": "fut_ret5", "oi5": "fut_oi5"})


def delivery(d: date) -> pd.DataFrame:
    e = equity.fetch_day(d)
    if e is None:
        return pd.DataFrame(columns=["symbol", "deliv_pct", "vwap_gap"])
    e = e.copy()
    e["vwap_gap"] = np.log(e["close"] / e["vwap"])
    return e[["symbol", "deliv_pct", "vwap_gap", "close"]].rename(columns={"close": "eq_close"})
