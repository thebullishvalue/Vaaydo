"""The vault's nightly MCX collection.  python -m vaaydo.dhan.nightly --vault PATH

Upserts, into the private vault:
    mcx/futures.parquet        official end-of-day bar per real contract (verified against its own quote)
    mcx/quotes.parquet         every listed future's end-of-day quote (far months too): OHLC, last, volume, OI
    mcx/contracts.parquet      every MCX future seen: symbol, security id, expiry, first and last seen
    mcx/chain/<year>.parquet   every strike of the three nearest option expiries of each commodity, once per session
    runs.csv                   one line per run: what was collected
Idempotent: a run with no new session (weekend, holiday, or a second run) changes nothing but runs.csv.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import mcx
from .client import Dhan


def _upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False, compression="zstd")
    return new


def run(vault: Path, dh: Dhan | None = None, today=None, log=print) -> dict:
    vault = Path(vault)
    dh = dh or Dhan()
    today = today or mcx.now_ist().date()
    m = mcx.master()
    fl = mcx.futures_list(m)
    fpath, cpath = vault / "mcx" / "futures.parquet", vault / "mcx" / "contracts.parquet"
    have = pd.read_parquet(fpath) if fpath.exists() else pd.DataFrame()
    # the latest expiry already passed, per commodity: a fetch never reaches back across it
    last_exp = {}
    if len(have):
        past = have[pd.to_datetime(have["expiry"]) < pd.Timestamp(today)]
        last_exp = past.groupby("root")["expiry"].max().to_dict()
    lookback = 7 if len(have) else 0                     # first run: the latest verified session only
    q = mcx.quotes(dh, fl)
    if len(q):
        _upsert(vault / "mcx" / "quotes.parquet", q.dropna(subset=["date"]), ["date", "symbol"])
    new = mcx.pull_futures(dh, fl, q, last_exp, today, lookback=lookback)
    out = {"run_ist": mcx.now_ist().strftime("%Y-%m-%d %H:%M"), "futures_rows": len(new), "session": "",
           "chain_rows": 0, "contracts": len(fl)}
    if new.empty:
        log("no futures bars returned")
        return out
    for k in ("open", "high", "low", "close"):
        new[k] = new[k].astype("float64")
    new["volume"] = new["volume"].fillna(0).astype("int64")
    new["oi"] = new["oi"].fillna(0).astype("int64")
    new["oi_close"] = new["oi_close"].astype("float64")
    F = _upsert(fpath, new, ["date", "symbol"])
    session = new["date"].max()
    out["session"] = f"{session:%Y-%m-%d}"
    seen = fl.assign(first_seen=pd.Timestamp(today), last_seen=pd.Timestamp(today))
    if cpath.exists():
        old = pd.read_parquet(cpath)
        seen = seen.merge(old[["symbol", "first_seen"]], on="symbol", how="left", suffixes=("", "_old"))
        seen["first_seen"] = seen["first_seen_old"].fillna(seen["first_seen"])
        gone = old[~old["symbol"].isin(seen["symbol"])]          # expired contracts stay on the record
        seen = seen.drop(columns="first_seen_old")
        seen = pd.concat([gone, seen], ignore_index=True) if len(gone) else seen
    seen.sort_values(["root", "expiry"]).to_parquet(cpath, index=False)
    chpath = vault / "mcx" / "chain" / f"{session:%Y}.parquet"
    done = set(pd.read_parquet(chpath, columns=["date"])["date"]) if chpath.exists() else set()
    if session not in done:
        ch = mcx.pull_chains(dh, fl, session)
        out["chain_rows"] = len(ch)
        if len(ch):
            _upsert(chpath, ch, ["date", "root", "expiry", "strike", "type"])
    log(f"session {out['session']}: {len(new)} futures bars ({new['symbol'].nunique()} contracts), "
        f"{out['chain_rows']} option rows; store holds {F['date'].nunique()} sessions")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    a = ap.parse_args(argv)
    out = run(Path(a.vault))
    runs = Path(a.vault) / "runs.csv"
    pd.DataFrame([out]).to_csv(runs, mode="a", header=not runs.exists(), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
