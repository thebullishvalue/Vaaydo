"""
The VAAYDO signal: Option Positioning Index (OPI), and the model book it implies.

    OPI (one day) = equal-weight average of six cross-sectional ranks, each read from that
                    day's NSE option chain of the stock:
        + put/call contracts traded            (put demand)
        + put/call open interest               (standing put positions)
        + 5%-OTM put IV minus ATM IV           (price of downside protection)
        - 5%-OTM call IV minus ATM IV          (price of upside calls)
        - call IV minus put IV at same strikes (call richness)
        + put OI added minus call OI added     (today's flow)
    score         = the 10-session mean of the day's OPI percentile
    model book    long when score is in the top 5%, held while in the top 15%;
                  short when in the bottom 5%, held while in the bottom 15%.

Why it works, and when it doesn't: see docs/RESEARCH_V6.md. In short, when the options
crowd leans to puts on a stock, that stock has tended to outperform over the next weeks,
and the reverse for calls (2019-2026, futures returns, net of costs). It is intermittent:
2023 and 2026 were losing years.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

COMPONENTS = [("pcr_vol", 1), ("pcr_oi", 1), ("skew", 1), ("cskew", -1), ("cpiv", -1), ("net_opt_flow", 1)]
INPUTS = ["pcr_vol", "pcr_oi", "skew", "cskew", "cpiv", "put_oi_flow", "call_oi_flow"]
SMOOTH = 10
ENTER, STAY = 0.05, 0.15
INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI"}
BUNDLED = Path(__file__).resolve().parent.parent / "data" / "opi_inputs.parquet"


def opi_day_pct(day: pd.DataFrame) -> pd.Series:
    """One session's OPI percentile per symbol (index: symbol)."""
    d = day[~day["symbol"].isin(INDICES)].set_index("symbol").copy()
    d["net_opt_flow"] = d["put_oi_flow"] - d["call_oi_flow"]
    s = 0.0
    for f, sign in COMPONENTS:
        r = d[f].rank(pct=True)
        s = s + (r if sign > 0 else 1 - r)
    return s.rank(pct=True)


def scores(inputs: pd.DataFrame) -> pd.DataFrame:
    """inputs: date, symbol, INPUTS -> date, symbol, opi_pct, score, pct (score percentile)."""
    rows = []
    for d, g in inputs.groupby("date"):
        p = opi_day_pct(g)
        rows.append(pd.DataFrame({"date": d, "symbol": p.index, "opi_pct": p.values}))
    x = pd.concat(rows).sort_values(["symbol", "date"])
    x["score"] = x.groupby("symbol")["opi_pct"].transform(lambda s: s.rolling(SMOOTH, min_periods=SMOOTH // 2).mean())
    x["pct"] = x.groupby("date")["score"].rank(pct=True)
    return x


@dataclass
class Position:
    symbol: str
    side: int
    since: pd.Timestamp


def replay(sc: pd.DataFrame, enter: float = ENTER, stay: float = STAY, daily: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Replay the model book's membership rules over the score history.
    Returns (book on the last day, log of every entry/exit). If `daily` is a dict, it is
    filled with {date: {symbol: side}} for every session."""
    book: dict[str, Position] = {}
    log = []
    for d, g in sc.dropna(subset=["pct"]).groupby("date"):
        pct = g.set_index("symbol")["pct"]
        if len(pct) < 50:
            continue
        new = {}
        for s, p in pct.items():
            cur = book.get(s)
            if p > 1 - enter or (cur and cur.side == 1 and p > 1 - stay):
                new[s] = cur if (cur and cur.side == 1) else Position(s, 1, d)
            elif p <= enter or (cur and cur.side == -1 and p <= stay):
                new[s] = cur if (cur and cur.side == -1) else Position(s, -1, d)
        for s in set(book) - set(new):
            log.append(dict(date=d, symbol=s, action="exit", side=book[s].side))
        for s, pos in new.items():
            if s not in book or book[s].side != pos.side:
                log.append(dict(date=d, symbol=s, action="enter", side=pos.side))
        book = new
        if daily is not None:
            daily[d] = {s: p.side for s, p in book.items()}
    return book, pd.DataFrame(log)


def load_inputs() -> pd.DataFrame:
    """Bundled history (shipped with the repo) + anything computed locally since."""
    frames = []
    if BUNDLED.exists():
        frames.append(pd.read_parquet(BUNDLED))
    local = DATA_DIR / "panels" / "opi_inputs_local.parquet"
    if local.exists():
        frames.append(pd.read_parquet(local))
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", *INPUTS])
    x = pd.concat(frames, ignore_index=True)
    x["date"] = pd.to_datetime(x["date"])
    return x.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"])


def extend_inputs(days, progress=None) -> pd.DataFrame:
    """Compute OPI inputs for any of `days` not yet in the store (needs their bhavcopies)."""
    from .features import day_features
    have = load_inputs()
    done = set(have["date"].dt.date) if len(have) else set()
    todo = [d for d in days if d not in done]
    new = []
    for i, d in enumerate(todo):
        if progress:
            progress(i, len(todo), d)
        f = day_features(d)
        new.append(f[["date", "symbol", *INPUTS]])
    if new:
        local = DATA_DIR / "panels" / "opi_inputs_local.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        old = pd.read_parquet(local) if local.exists() else pd.DataFrame()
        pd.concat([old, *new], ignore_index=True).to_parquet(local, index=False)
    return load_inputs()
