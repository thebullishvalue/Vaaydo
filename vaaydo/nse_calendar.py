"""
NSE's trading calendar beyond the last session on file.

Sessions to an expiry decide when a position rolls (3 sessions before expiry). For dates
already traded, the sessions are the files themselves. For dates ahead, they are weekdays
minus the exchange's holidays, known in advance:

    data/nse_holidays.csv   past holidays (weekdays with no NSE file) and announced ones
    listed expiries         NSE moves an expiry off its standard weekday (the last Tuesday of the
                            month for stock F&O) only for a holiday, so a listed expiry earlier
                            than its month's last Tuesday marks that Tuesday a holiday

Edit the CSV when NSE publishes the next year's list; the nightly refresh warns when it runs
out (no listed future holiday in the next 60 days is not an error, a missing year is).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "nse_holidays.csv"


@lru_cache(maxsize=1)
def listed() -> tuple:
    if not CSV.exists():
        return ()
    x = pd.read_csv(CSV)
    return tuple(pd.to_datetime(x["date"]).dt.date)


def implied_by_expiries(expiries) -> set:
    out = set()
    for e in pd.to_datetime(pd.Series(list(expiries))).dropna().unique():
        e = pd.Timestamp(e)
        last_tue = (e + pd.offsets.MonthEnd(0))
        last_tue -= pd.Timedelta(days=(last_tue.weekday() - 1) % 7)
        if e < last_tue and e.weekday() < 5 and (last_tue - e).days < 7:
            out.add(last_tue.date())
    return out


def holidays(expiries=()) -> np.ndarray:
    h = set(listed()) | implied_by_expiries(expiries)
    return np.array(sorted(h), dtype="datetime64[D]")


def sessions_after(last: pd.Timestamp, until: pd.Timestamp, hol: np.ndarray) -> int:
    """Number of NSE sessions in (last, until]."""
    return int(np.busday_count((pd.Timestamp(last) + pd.Timedelta(days=1)).date(),
                               (pd.Timestamp(until) + pd.Timedelta(days=1)).date(), holidays=hol))


def session_offset(last: pd.Timestamp, k: int, hol: np.ndarray) -> pd.Timestamp:
    """The k-th NSE session after `last` (k >= 1)."""
    return pd.Timestamp(np.busday_offset(pd.Timestamp(last).date(), k, roll="forward", holidays=hol))
