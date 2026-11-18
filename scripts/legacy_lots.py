"""Estimate stock-futures lot sizes for 2019 - Jul 2024, where NSE's old F&O file has none.

    lot ~ traded value / (contracts x settle price), taken from each contract's busiest day

Accurate to well under 1% (INFY 302 vs 300, SBIN 1,504 vs 1,500): used only as the rupee
weight of a position in the one-lot backtest, never as a traded lot. Writes
$VAAYDO_DATA/panels/legacy_lots.parquet (symbol, expiry, lot_w).
"""
import io, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vaaydo.config import DATA_DIR  # noqa: E402
from vaaydo.data import legacy  # noqa: E402

def one(d):
    s = requests.Session()
    url = legacy.URL.format(d=d, mon=d.strftime("%b").upper())
    for k in range(3):
        try:
            r = s.get(url, headers=legacy.HEADERS, timeout=60)
            if r.status_code == 200 and r.content[:2] == b"PK":
                z = zipfile.ZipFile(io.BytesIO(r.content))
                x = pd.read_csv(z.open(z.namelist()[0]))
                x = x[x["INSTRUMENT"].astype(str).str.strip() == "FUTSTK"]
                return pd.DataFrame({"symbol": x["SYMBOL"].astype(str).str.strip(),
                                     "expiry": pd.to_datetime(x["EXPIRY_DT"], format="%d-%b-%Y"),
                                     "contracts": x["CONTRACTS"], "lot_est": x["VAL_INLAKH"] * 1e5 / (x["CONTRACTS"] * x["SETTLE_PR"])})
            return None
        except Exception:
            time.sleep(2 * (k + 1))
    return None

days = legacy.available_days()
with ThreadPoolExecutor(6) as ex:
    parts = [p for p in ex.map(one, days) if p is not None]
X = pd.concat(parts, ignore_index=True)
X = X[(X["contracts"] > 0) & X["lot_est"].notna()]
best = X.sort_values("contracts").groupby(["symbol", "expiry"]).tail(1)
best = best.rename(columns={"lot_est": "lot_w"})[["symbol", "expiry", "lot_w"]]
best.to_parquet(DATA_DIR / "panels/legacy_lots.parquet", index=False)
print(len(parts), "days;", len(best), "contracts")
