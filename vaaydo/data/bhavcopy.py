"""
NSE F&O bhavcopy (UDiFF format, published daily since 8 Jul 2024).

This is VAAYDO's ground truth. One file per trading day holds the official close,
settlement price, volume, open interest and lot size of every listed future and option,
plus the underlying's closing price. From it we get, with no hardcoded tables:

    * the expiry calendar (holiday-shifted expiries come straight from the file)
    * lot sizes (they change; the old VAAYDO table was stale)
    * real option prices -> real implied volatility
    * futures price + OI history -> positioning (build-up) history
    * underlying close history -> realised volatility

Raw zips are not kept. Each day is slimmed to a parquet file of the columns we use.
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from ..config import DATA_DIR

log = logging.getLogger(__name__)

URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
FIRST_UDIFF_DAY = date(2024, 7, 8)
HEADERS = {"User-Agent": "vaaydo/5 (end-of-day research)"}

COLS = {
    "FinInstrmTp": "inst", "TckrSymb": "symbol", "XpryDt": "expiry", "StrkPric": "strike",
    "OptnTp": "otype", "OpnPric": "open", "HghPric": "high", "LwPric": "low", "ClsPric": "close",
    "SttlmPric": "settle", "UndrlygPric": "spot", "OpnIntrst": "oi", "ChngInOpnIntrst": "oi_chg",
    "TtlTradgVol": "contracts", "NewBrdLotQty": "lot",
}
KEEP_INST = {"STO", "IDO", "STF", "IDF"}
MAX_OPTION_DTE = 120  # far expiries are illiquid and only bloat the store


def day_dir() -> Path:
    p = DATA_DIR / "bhav"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(d: date) -> Path:
    return day_dir() / f"{d:%Y%m%d}.parquet"


def _holiday_file() -> Path:
    return day_dir() / "_missing.txt"


def _missing() -> set[str]:
    f = _holiday_file()
    return set(f.read_text().split()) if f.exists() else set()


def parse(raw_csv: bytes | str | Path) -> pd.DataFrame:
    src = io.BytesIO(raw_csv) if isinstance(raw_csv, bytes) else raw_csv
    df = pd.read_csv(src, usecols=list(COLS) + ["TradDt"])
    trade_date = pd.Timestamp(df["TradDt"].iloc[0]).normalize()
    df = df.rename(columns=COLS).drop(columns="TradDt")
    df = df[df["inst"].isin(KEEP_INST)].copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    is_opt = df["inst"].isin({"STO", "IDO"})
    dte = (df["expiry"] - trade_date).dt.days
    df = df[~is_opt | (dte <= MAX_OPTION_DTE)]
    df["otype"] = df["otype"].fillna("FX")
    df["date"] = trade_date
    for c in ("strike", "open", "high", "low", "close", "settle", "spot"):
        df[c] = df[c].astype("float64")
    for c in ("oi", "oi_chg", "contracts", "lot"):
        df[c] = df[c].fillna(0).astype("int64")
    for c in ("inst", "symbol", "otype"):
        df[c] = df[c].astype("category")
    return df.reset_index(drop=True)


def fetch_day(d: date, session: requests.Session | None = None, force: bool = False) -> pd.DataFrame | None:
    """Download, slim and cache one trading day. Returns None for holidays / unpublished days."""
    p = _path(d)
    if p.exists() and not force:
        return pd.read_parquet(p)
    if d.weekday() >= 5 or d < FIRST_UDIFF_DAY:
        return None
    key = f"{d:%Y%m%d}"
    if key in _missing() and not force and (date.today() - d).days > 3:
        return None
    s = session or requests.Session()
    for attempt in range(4):
        try:
            r = s.get(URL.format(d=d), headers=HEADERS, timeout=60)
        except requests.RequestException as e:  # network blip: back off
            log.warning("fetch %s failed (%s), retry %d", d, e, attempt)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            if (date.today() - d).days > 3:  # only remember settled absences
                with _holiday_file().open("a") as f:
                    f.write(key + "\n")
            return None
        if r.status_code != 200:
            time.sleep(2 ** attempt)
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0])
        df = parse(raw)
        df.to_parquet(p, index=False)
        return df
    raise RuntimeError(f"could not fetch bhavcopy for {d}")


def sync(start: date | None = None, end: date | None = None, progress=None) -> list[date]:
    """Ensure every trading day in [start, end] is cached. Returns the days available."""
    end = end or date.today()
    start = max(start or FIRST_UDIFF_DAY, FIRST_UDIFF_DAY)
    s = requests.Session()
    days, d = [], start
    all_days = [start + timedelta(i) for i in range((end - start).days + 1)]
    for i, d in enumerate(all_days):
        if fetch_day(d, s) is not None:
            days.append(d)
        if progress:
            progress(i + 1, len(all_days), d)
    return days


def available_days() -> list[date]:
    return sorted(pd.Timestamp(p.stem).date() for p in day_dir().glob("2*.parquet"))


def load_day(d: date) -> pd.DataFrame:
    return pd.read_parquet(_path(d))


def latest_day(on_or_before: date | None = None) -> date:
    days = [d for d in available_days() if on_or_before is None or d <= on_or_before]
    if not days:
        raise FileNotFoundError("no bhavcopy cached; run `python -m vaaydo sync` first")
    return days[-1]


def load_range(days: Iterable[date], columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    frames = []
    for d in days:
        p = _path(d)
        if p.exists():
            frames.append(pd.read_parquet(p, columns=columns, filters=filters))
    if not frames:
        return pd.DataFrame(columns=columns or list(COLS.values()))
    out = pd.concat(frames, ignore_index=True)
    for c in ("inst", "symbol", "otype"):
        if c in out:
            out[c] = out[c].astype(str)
    return out
