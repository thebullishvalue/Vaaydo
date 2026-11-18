"""
Long daily OHLC history from Yahoo Finance (split/dividend adjusted), for Pragyam's
warm-up (~750 bars) and long-horizon price features. NSE files cover only Jul 2024+.
Cached as one parquet panel; refreshed incrementally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DATA_DIR

YF = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"}


def ticker(sym: str) -> str:
    return YF.get(sym, f"{sym}.NS")


def _path():
    p = DATA_DIR / "panels"
    p.mkdir(parents=True, exist_ok=True)
    return p / "history.parquet"


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning as Pragyam's research loader: adjust OHLC by adj/close, fix bad opens."""
    d = d[(d["close"] > 0) & (d["high"] >= d["low"])].copy()
    k = (d["adj"] / d["close"]).where(d["adj"] > 0, 1.0)
    for c in ("open", "high", "low", "close"):
        d[c] = d[c] * k
    bad_open = ~((d["open"] > 0) & (np.log(d["open"] / d["close"]).abs() < 0.3))
    d.loc[bad_open, "open"] = d.loc[bad_open, "close"]
    return d[d["close"].pct_change().abs().fillna(0) < 0.6]


def download(symbols: list[str], period: str = "10y", chunk: int = 60) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        raw = yf.download([ticker(s) for s in part], period=period, interval="1d", auto_adjust=False,
                          group_by="ticker", progress=False, threads=True)
        for s in part:
            t = ticker(s)
            if t not in raw.columns.get_level_values(0):
                continue
            d = raw[t].rename(columns=str.lower).rename(columns={"adj close": "adj"}).dropna(subset=["close"])
            if len(d) < 50:
                continue
            d = _clean(d[["open", "high", "low", "close", "adj", "volume"]])
            d["symbol"] = s
            frames.append(d.rename_axis("date").reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load(symbols: list[str] | None = None, refresh: bool = False, until=None) -> pd.DataFrame:
    """Cached daily history. `until`: the latest session needed; a cache that ends before it
    is refreshed (so a long-lived container never runs stops on stale prices)."""
    p = _path()
    if p.exists() and not refresh:
        h = pd.read_parquet(p)
        h["date"] = pd.to_datetime(h["date"])
        # Yahoo usually posts a session's bar hours after NSE: tolerate one session of lag
        # (the curator fills it from NSE's own equity bhavcopy) and refresh beyond that.
        stale = until is not None and h["date"].max() < pd.Timestamp(until) - pd.offsets.BDay(1)
        missing = [s for s in (symbols or []) if s not in set(h["symbol"])]
        if stale:
            want = sorted(set(symbols or []) | set(h["symbol"]) if symbols else set(h["symbol"]))
            h = pd.concat([h, download(want, period="1y")], ignore_index=True)
        elif missing:
            h = pd.concat([h, download(missing)], ignore_index=True)
        else:
            return fill_from_nse(h, symbols, until)
    else:
        h = download(symbols or [])
    h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize()
    h = h.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    h.to_parquet(p, index=False)
    return fill_from_nse(h, symbols, until)


def fill_from_nse(h: pd.DataFrame, symbols=None, until=None, lookback: int = 10) -> pd.DataFrame:
    """Yahoo posts a session's bar hours after NSE, and sometimes skips one in the middle of a
    series. For every NSE session in the last `lookback` weekdays up to `until`, any wanted symbol
    without a bar that day gets NSE's own bar from the equity bhavcopy (a file whose own trade date
    is that day: holidays return nothing). Bars after `until` (a still-forming day) are dropped.
    Recent Yahoo bars match NSE's raw prices (adjustments apply to older history)."""
    if until is None or h is None or h.empty:
        return h
    from . import equity
    until = pd.Timestamp(until)
    h = h[h["date"] <= until]
    want = set(symbols) if symbols else set(h["symbol"])
    have = set(zip(h["date"], h["symbol"]))
    rows = []
    for d in pd.bdate_range(end=until, periods=lookback):
        miss = [x for x in want if (d, x) not in have]
        if not miss:
            continue
        e = equity.fetch_day(d.date())
        if e is None or e.empty:
            continue                                   # not a session
        e = e[e["symbol"].isin(miss) & (e["close"] > 0)]
        if len(e):
            rows.append(pd.DataFrame({"date": d, "symbol": e["symbol"].values, "open": e["open"].values,
                                      "high": e["high"].values, "low": e["low"].values, "close": e["close"].values,
                                      "adj": e["close"].values, "volume": e["qty"].values}))
    if not rows:
        return h
    out = pd.concat([h, *rows], ignore_index=True)
    return out.drop_duplicates(["symbol", "date"], keep="first").sort_values(["symbol", "date"]).reset_index(drop=True)
