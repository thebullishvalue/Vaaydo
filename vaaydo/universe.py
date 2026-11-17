"""Universes and sectors: NSE index constituents (niftyindices.com), cached for the day."""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd
import requests

INDEX_FILES = {
    "NIFTY 50": "ind_nifty50list.csv", "NIFTY Next 50": "ind_niftynext50list.csv", "NIFTY 100": "ind_nifty100list.csv",
    "NIFTY Bank": "ind_niftybanklist.csv", "NIFTY Financial Services": "ind_niftyfinancelist.csv",
    "NIFTY IT": "ind_niftyitlist.csv", "NIFTY Pharma": "ind_niftypharmalist.csv", "NIFTY Auto": "ind_niftyautolist.csv",
    "NIFTY Metal": "ind_niftymetallist.csv", "NIFTY FMCG": "ind_niftyfmcglist.csv", "NIFTY Energy": "ind_niftyenergylist.csv",
    "NIFTY PSU Bank": "ind_niftypsubanklist.csv", "NIFTY Realty": "ind_niftyrealtylist.csv", "NIFTY Midcap 100": "ind_niftymidcap100list.csv",
}
UNIVERSES = ["All F&O stocks", *INDEX_FILES]


@lru_cache(maxsize=32)
def constituents(name: str) -> frozenset | None:
    """Symbols in an NSE index; None for 'All F&O stocks' or if the list cannot be fetched."""
    f = INDEX_FILES.get(name)
    if not f:
        return None
    try:
        r = requests.get(f"https://www.niftyindices.com/IndexConstituent/{f}", headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        return frozenset(pd.read_csv(io.StringIO(r.text))["Symbol"].astype(str).str.strip())
    except Exception:
        return None
