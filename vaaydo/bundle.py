"""
The data that ships with the repo, so the app starts in seconds on Streamlit Cloud:

    data/opi_inputs.parquet    option-positioning inputs per stock per session
    data/futures.parquet       every stock future's open/close/lot per session (the ledger's prices)
    data/pragyam.parquet       Pragyam's levels per stock per session (stop ratios, range, trend, regime)
    data/sectors.parquet       symbol -> industry (NSE NIFTY 500 list)

All from July 2024 on. `refresh()` appends sessions published since the bundle ends
(run nightly by .github/workflows/refresh.yml). The app fills anything newer itself.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data import bhavcopy

ROOT = Path(__file__).resolve().parent.parent / "data"
START = pd.Timestamp("2024-07-08")
FUT = ROOT / "futures.parquet"
PRG = ROOT / "pragyam.parquet"
SECT = ROOT / "sectors.parquet"
INPUTS = ROOT / "opi_inputs.parquet"


def futures_rows(days) -> pd.DataFrame:
    f = bhavcopy.load_range(list(days), columns=["date", "inst", "symbol", "expiry", "open", "close", "lot"],
                            filters=[("inst", "==", "STF")])
    if f.empty:
        return f
    f = f.drop(columns="inst")
    f["date"] = pd.to_datetime(f["date"])
    f["expiry"] = pd.to_datetime(f["expiry"])
    f["symbol"] = f["symbol"].astype(str)
    for c in ("open", "close"):
        f[c] = f[c].astype("float32")
    f["lot"] = f["lot"].astype("int32")
    return f


def pragyam_rows(symbols, since: pd.Timestamp, until: pd.Timestamp | None = None, history_df=None) -> pd.DataFrame:
    """Pragyam's daily levels, as ratios to the close, for dates in [since, until]."""
    from . import pragyam as PG
    from .data import history
    h = history_df if history_df is not None else history.load(sorted(symbols))
    out = []
    for s, d in h[h["symbol"].isin(symbols)].groupby("symbol"):
        d = d.set_index("date").sort_index()[["open", "high", "low", "close"]]
        if until is not None:
            d = d[d.index <= until]
        d = d[d.index >= since - pd.Timedelta(days=1700)]           # enough warm-up (~1,150 bars)
        if len(d) < 300:
            continue
        R = PG.run(d)
        R = R[R.index >= since]
        c = d["close"].reindex(R.index)
        out.append(pd.DataFrame({
            "date": R.index, "symbol": s, "close": c.values.astype("float32"),
            "stopL_r": (R["stopL"] / c).values.astype("float32"), "stopS_r": (R["stopS"] / c).values.astype("float32"),
            "lo_r": (R["lower"] / c).values.astype("float32"), "up_r": (R["upper"] / c).values.astype("float32"),
            "trend": R["T"].values.astype("float32"), "reg": R["reg"].values.astype("float32"),
            "sig": R["sig"].values.astype("float32")}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sectors() -> pd.DataFrame:
    import io
    import requests
    r = requests.get("https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text))
    return x.rename(columns={"Symbol": "symbol", "Industry": "industry", "Company Name": "name"})[["symbol", "name", "industry"]]


def build_all(progress=print) -> None:
    from . import signal
    from .data import store
    days = [d for d in bhavcopy.available_days() if pd.Timestamp(d) >= START]
    futures_rows(days).to_parquet(FUT, index=False, compression="zstd")
    progress(f"futures: {len(days)} sessions")
    syms = sorted(set(pd.read_parquet(INPUTS)["symbol"]) | set(pd.read_parquet(FUT)["symbol"]))
    pragyam_rows(syms, START).to_parquet(PRG, index=False, compression="zstd")
    progress("pragyam done")
    sectors().to_parquet(SECT, index=False)
    progress("sectors done")


def refresh(progress=print) -> None:
    """Append sessions published after the bundle ends (inputs, futures, Pragyam)."""
    from . import signal
    from .features import day_features
    have = pd.read_parquet(INPUTS)
    have["date"] = pd.to_datetime(have["date"])
    last = have["date"].max().date()
    new_days = []
    d = last + timedelta(days=1)
    while d <= date.today():
        if d.weekday() < 5 and bhavcopy.fetch_day(d) is not None:
            new_days.append(d)
        d += timedelta(days=1)
    if not new_days:
        progress(f"nothing new after {last}")
        return
    rows = []
    for d in new_days:
        f = day_features(d)[["date", "symbol", *signal.INPUTS]]
        for c in signal.INPUTS:
            f[c] = f[c].astype("float32")
        rows.append(f)
    pd.concat([have, *rows], ignore_index=True).drop_duplicates(["date", "symbol"], keep="last") \
        .to_parquet(INPUTS, index=False, compression="zstd")
    F = pd.read_parquet(FUT)
    pd.concat([F, futures_rows(new_days)], ignore_index=True).drop_duplicates(["date", "symbol", "expiry"], keep="last") \
        .to_parquet(FUT, index=False, compression="zstd")
    P = pd.read_parquet(PRG)
    syms = sorted(set(P["symbol"]) | set(pd.read_parquet(FUT)["symbol"]))
    from .data import history
    h = history.load(syms, refresh=True)
    newp = pragyam_rows(syms, pd.Timestamp(new_days[0]), history_df=h)
    pd.concat([P, newp], ignore_index=True).drop_duplicates(["date", "symbol"], keep="last") \
        .to_parquet(PRG, index=False, compression="zstd")
    progress(f"bundle now ends {new_days[-1]}")
    refresh_index(new_days, progress)


def refresh_index(new_days, progress=print) -> None:
    """Append the index desk's readings, participant OI, the F&O ban list and India VIX."""
    from . import banlist, indices as I
    p = ROOT / "index_panel.parquet"
    if p.exists():
        P = pd.read_parquet(p)
        rows = [r for d in new_days for r in I.day(d)]
        if rows:
            pd.concat([P, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["date", "symbol"], keep="last") \
                .to_parquet(p, index=False, compression="zstd")
    q = ROOT / "participant_oi.parquet"
    if q.exists():
        Q = pd.read_parquet(q)
        add = [x for x in (I.participant_day(d) for d in new_days) if x is not None]
        if add:
            pd.concat([Q, *add], ignore_index=True).drop_duplicates(["date", "client"], keep="last").to_parquet(q, index=False)
    b = ROOT / "banlist.parquet"
    if b.exists():
        B = pd.read_parquet(b)
        add = [(pd.Timestamp(d), s) for d in new_days for s in ((banlist.day(d) or []) + [""])]
        pd.concat([B, pd.DataFrame(add, columns=["date", "symbol"])], ignore_index=True).drop_duplicates() \
            .to_parquet(b, index=False)
    br = ROOT / "breadth_nse.parquet"
    if br.exists() and p.exists():
        from . import breadth as BR
        raw = pd.read_parquet(br)
        P = pd.read_parquet(p)
        spot = P[P["symbol"] == "NIFTY"].assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date")["spot"]
        BR.extend_live(raw, spot, [pd.Timestamp(d) for d in new_days]).to_parquet(br, index=False, compression="zstd")
    v = ROOT / "vix.parquet"
    try:
        x = I.yahoo("VIX", start="2018-01-01")
        x.rename("vix").rename_axis("date").reset_index().to_parquet(v, index=False)
    except Exception:
        pass
    progress("index desk data refreshed")
