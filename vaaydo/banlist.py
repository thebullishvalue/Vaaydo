"""
NSE's F&O ban list: stocks whose open interest crossed 95% of the market-wide position limit.
During a ban no fresh position may be opened in the stock's futures or options; only
reducing trades are allowed. VAAYDO defers an entry whose fill day falls in a ban.

    archives/fo/sec_ban/fo_secban_DDMMYYYY.csv   one file per trade date, 2019 on
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd
import requests

from .config import DATA_DIR

URL = "https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_{d:%d%m%Y}.csv"
HEADERS = {"User-Agent": "vaaydo/6 (end-of-day research)"}
PATH = DATA_DIR / "panels" / "banlist.parquet"


def day(d, session=None) -> list[str] | None:
    s = session or requests.Session()
    for attempt in range(3):
        try:
            r = s.get(URL.format(d=d), headers=HEADERS, timeout=30)
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        return None
    if r.status_code != 200 or not r.text.startswith("Securities in Ban"):
        return None
    out = []
    for line in r.text.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            out.append(parts[1])
    return out


def build(days, workers: int = 4) -> pd.DataFrame:
    have = pd.read_parquet(PATH) if PATH.exists() else pd.DataFrame(columns=["date", "symbol"])
    done = set(pd.to_datetime(have["date"]).dt.date) if len(have) else set()
    todo = [d for d in days if d not in done]

    def one(d):
        x = day(d)
        return [(pd.Timestamp(d), s) for s in (x or [])] + ([(pd.Timestamp(d), "")] if x is not None else [])

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            rows += r
    new = pd.DataFrame(rows, columns=["date", "symbol"])
    out = pd.concat([have, new], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates().sort_values(["date", "symbol"])
    PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PATH, index=False)
    return out


@lru_cache(maxsize=1)
def banned() -> frozenset:
    """{(date, symbol)} in an F&O ban: the shipped bundle, else the research store. Rows with
    an empty symbol only record that the day was fetched."""
    from .bundle import ROOT
    for p in (ROOT / "banlist.parquet", PATH):
        if p.exists():
            x = pd.read_parquet(p)
            x = x[x["symbol"] != ""]
            return frozenset(zip(pd.to_datetime(x["date"]), x["symbol"]))
    return frozenset()


def live(dates) -> frozenset:
    """The shipped list plus any sessions after it (fetched from NSE), for the live replay."""
    base = set(banned())
    known = max((d for d, _ in base), default=pd.Timestamp("2024-07-01"))
    s = requests.Session()
    for d in dates:
        d = pd.Timestamp(d)
        if d > known:
            for sym in day(d.date(), s) or []:
                base.add((d, sym))
    return frozenset(base)


def next_session(d) -> tuple[pd.Timestamp, list[str] | None]:
    """NSE publishes the next trade date's ban list the evening before: (date, symbols) or None."""
    nd = pd.Timestamp(d) + pd.offsets.BDay(1)
    for _ in range(4):
        x = day(nd.date())
        if x is not None:
            return nd, x
        nd += pd.offsets.BDay(1)
    return nd, None
