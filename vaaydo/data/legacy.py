"""
The pre-UDiFF NSE F&O bhavcopy (before 8 Jul 2024): same contracts, older layout,
no underlying price and no lot size. Slimmed to the same schema as `bhavcopy`, with
spot and lot left empty, in a separate store ($VAAYDO_DATA/bhav_legacy) so the live
panels are never mixed with it. Used for long out-of-sample research.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

from ..config import DATA_DIR

URL = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"
INST = {"FUTSTK": "STF", "OPTSTK": "STO", "FUTIDX": "IDF", "OPTIDX": "IDO"}
HEADERS = {"User-Agent": "vaaydo/6 (end-of-day research)"}


def _dir():
    p = DATA_DIR / "bhav_legacy"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(d: date):
    return _dir() / f"{d:%Y%m%d}.parquet"


def parse(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["INSTRUMENT"].isin(INST)]
    out = pd.DataFrame({
        "inst": df["INSTRUMENT"].map(INST), "symbol": df["SYMBOL"].astype(str).str.strip(),
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": df["STRIKE_PR"].astype(float), "otype": df["OPTION_TYP"].astype(str).str.strip().replace({"XX": "FX"}),
        "open": df["OPEN"].astype(float), "high": df["HIGH"].astype(float), "low": df["LOW"].astype(float),
        "close": df["CLOSE"].astype(float), "settle": df["SETTLE_PR"].astype(float), "spot": np.nan,
        "oi": df["OPEN_INT"].astype("int64"), "oi_chg": df["CHG_IN_OI"].astype("int64"),
        "contracts": df["CONTRACTS"].astype("int64"), "lot": 0,
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y"),
    })
    dte = (out["expiry"] - out["date"]).dt.days
    out = out[~out["inst"].isin(["STO", "IDO"]) | (dte <= 120)]
    for c in ("inst", "symbol", "otype"):
        out[c] = out[c].astype("category")
    return out.reset_index(drop=True)


def fetch_day(d: date, s: requests.Session) -> bool:
    p = _path(d)
    if p.exists():
        return True
    if d.weekday() >= 5:
        return False
    url = URL.format(d=d, mon=d.strftime("%b").upper())
    for attempt in range(4):
        try:
            r = s.get(url, headers=HEADERS, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return False
        if r.status_code == 200 and r.content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                parse(z.read(z.namelist()[0])).to_parquet(p, index=False)
            return True
        time.sleep(2 ** attempt)
    return False


def sync(start: date, end: date, progress=None) -> int:
    s = requests.Session()
    n = 0
    for i, d in enumerate(pd.bdate_range(start, end)):
        n += fetch_day(d.date(), s)
        if progress and i % 50 == 0:
            progress(i, d.date(), n)
    return n


def available_days():
    return sorted(pd.Timestamp(p.stem).date() for p in _dir().glob("2*.parquet"))


def load_day(d: date) -> pd.DataFrame:
    return pd.read_parquet(_path(d))
