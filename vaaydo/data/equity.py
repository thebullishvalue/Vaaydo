"""
NSE equity bhavcopy with delivery data (sec_bhavdata_full): spot OHLC, VWAP, traded and
delivered quantity for every listed stock, one file per day. Slimmed to EQ-series rows.
"""
from __future__ import annotations

import io
import time
from datetime import date, timedelta

import pandas as pd
import requests

from ..config import DATA_DIR

URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"
HEADERS = {"User-Agent": "vaaydo/6 (end-of-day research)"}


def _dir():
    p = DATA_DIR / "equity"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(d: date):
    return _dir() / f"{d:%Y%m%d}.parquet"


def parse(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["DATE1"].astype(str).str.strip(), format="%d-%b-%Y"),
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "prev_close": pd.to_numeric(df["PREV_CLOSE"], errors="coerce"),
        "open": pd.to_numeric(df["OPEN_PRICE"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH_PRICE"], errors="coerce"),
        "low": pd.to_numeric(df["LOW_PRICE"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
        "vwap": pd.to_numeric(df["AVG_PRICE"], errors="coerce"),
        "qty": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
        "turnover_lacs": pd.to_numeric(df["TURNOVER_LACS"], errors="coerce"),
        "trades": pd.to_numeric(df["NO_OF_TRADES"], errors="coerce"),
        "deliv_qty": pd.to_numeric(df["DELIV_QTY"], errors="coerce"),
        "deliv_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
    })
    return out.reset_index(drop=True)


def fetch_day(d: date, s: requests.Session | None = None) -> pd.DataFrame | None:
    p = _path(d)
    if p.exists():
        return pd.read_parquet(p)
    if d.weekday() >= 5:
        return None
    s = s or requests.Session()
    for attempt in range(4):
        try:
            r = s.get(URL.format(d=d), headers=HEADERS, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200 and len(r.content) > 1000:
            df = parse(r.content)
            df.to_parquet(p, index=False)
            return df
        time.sleep(2 ** attempt)
    return None


def sync(days: list[date]) -> int:
    s = requests.Session()
    return sum(fetch_day(d, s) is not None for d in days)


def available_days() -> list[date]:
    return sorted(pd.Timestamp(p.stem).date() for p in _dir().glob("2*.parquet"))


def load(days=None, symbols=None) -> pd.DataFrame:
    days = days or available_days()
    filters = [("symbol", "in", list(symbols))] if symbols else None
    frames = [pd.read_parquet(_path(d), filters=filters) for d in days if _path(d).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
