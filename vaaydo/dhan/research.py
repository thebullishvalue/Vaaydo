"""On-demand research pulls, run inside the vault (the only holder of the Dhan token).

    python -m vaaydo.dhan.research intraday --vault PATH [--since 2024-07-01]

intraday: 5-minute bars of every stock in VAAYDO's futures history since `since`, reduced to one row per
stock-session (open, VWAP of 09:15-09:45, VWAP of 09:15-10:15, session VWAP, close, volume): what
docs/PREREGISTRATION_EXECUTION.md (S2) needs. Written to research/nse_intraday_summary.parquet. The raw bars
are not kept.
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .client import MASTER_URL, Dhan, DhanError

FUT_URL = "https://raw.githubusercontent.com/thebullishvalue/Vaaydo/main/data/futures.parquet"
UA = {"User-Agent": "vaaydo/7 (end-of-day research)"}


def equity_ids(symbols) -> dict:
    m = pd.read_csv(io.StringIO(requests.get(MASTER_URL, headers=UA, timeout=180).text), low_memory=False)
    e = m[(m["SEM_EXM_EXCH_ID"] == "NSE") & (m["SEM_INSTRUMENT_NAME"] == "EQUITY") & (m["SEM_SERIES"] == "EQ")]
    ids = dict(zip(e["SEM_TRADING_SYMBOL"].astype(str), e["SEM_SMST_SECURITY_ID"].astype(int)))
    return {s: ids[s] for s in symbols if s in ids}


def _intraday(dh: Dhan, sid: int, a: date, b: date) -> pd.DataFrame:
    body = {"securityId": str(sid), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "interval": "5", "oi": False}
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):          # the docs do not say which form the dates take
        try:
            j = dh._call("POST", "/charts/intraday", {**body, "fromDate": pd.Timestamp(a).strftime(fmt),
                                                      "toDate": pd.Timestamp(b).strftime(fmt)})
            break
        except DhanError as e:
            if "DH-905" not in str(e) or fmt == "%Y-%m-%d":
                raise
    if not j or "timestamp" not in j:
        return pd.DataFrame()
    d = pd.DataFrame({k: j[k] for k in ("timestamp", "open", "high", "low", "close", "volume")})
    d["t"] = pd.to_datetime(d["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return d.drop(columns="timestamp")


def summarise(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per session: the windows S2 fills at (typical price × volume VWAPs)."""
    if bars.empty:
        return bars
    b = bars.copy()
    b["date"] = b["t"].dt.normalize()
    b["tp"] = (b["high"] + b["low"] + b["close"]) / 3
    b["pv"] = b["tp"] * b["volume"]
    mins = b["t"].dt.hour * 60 + b["t"].dt.minute
    out = []
    for d, g in b.groupby("date"):
        g = g.sort_values("t")
        m = mins.loc[g.index]

        def vw(mask):
            x = g[mask]
            v = x["volume"].sum()
            return float(x["pv"].sum() / v) if v > 0 else np.nan
        out.append({"date": d, "open": float(g["open"].iloc[0]), "vwap30": vw(m < 9 * 60 + 45),
                    "vwap60": vw(m < 10 * 60 + 15), "vwap_day": vw(m >= 0), "close": float(g["close"].iloc[-1]),
                    "volume": int(g["volume"].sum()), "first_bar": g["t"].iloc[0].strftime("%H:%M")})
    return pd.DataFrame(out)


def intraday(vault: Path, since: str = "2024-07-01", log=print) -> int:
    F = pd.read_parquet(io.BytesIO(requests.get(FUT_URL, headers=UA, timeout=180).content), columns=["date", "symbol"])
    F["date"] = pd.to_datetime(F["date"])
    syms = sorted(F.loc[F["date"] >= pd.Timestamp(since), "symbol"].astype(str).unique())
    ids = equity_ids(syms)
    log(f"{len(syms)} stocks in the futures history since {since}; {len(ids)} with an NSE equity id at Dhan "
        f"(missing: {', '.join(sorted(set(syms) - set(ids)))[:300]})")
    dh = Dhan()
    out, failed = [], []
    end = date.today()
    for k, (s, sid) in enumerate(sorted(ids.items())):
        parts = []
        a = pd.Timestamp(since).date()
        try:
            while a <= end:
                b = min(a + timedelta(days=89), end + timedelta(days=1))
                parts.append(_intraday(dh, sid, a, b))
                a = b
        except DhanError as e:
            failed.append(f"{s} ({str(e)[:80]})")
            continue
        sm = summarise(pd.concat(parts, ignore_index=True).drop_duplicates("t") if parts else pd.DataFrame())
        if len(sm):
            out.append(sm.assign(symbol=s))
        if k % 25 == 0:
            log(f"  {k + 1}/{len(ids)} {s}: {len(sm)} sessions")
    R = pd.concat(out, ignore_index=True)
    p = Path(vault) / "research" / "nse_intraday_summary.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    R.to_parquet(p, index=False, compression="zstd")
    log(f"wrote {len(R)} stock-sessions for {R['symbol'].nunique()} stocks; failed {len(failed)}: {'; '.join(failed)[:500]}")
    return 0 if len(R) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["intraday"])
    ap.add_argument("--vault", required=True)
    ap.add_argument("--since", default="2024-07-01")
    a = ap.parse_args(argv)
    return intraday(Path(a.vault), a.since)


if __name__ == "__main__":
    sys.exit(main())
