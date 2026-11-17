"""
Long out-of-sample test of the frozen strategy on 2019 - Jul 2024 (pre-UDiFF archive),
data no design choice ever saw. Rules exactly as frozen in docs/PREREGISTRATION.md (v2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import alpha as A
from . import features as Fe
from .config import DATA_DIR
from .data import history, legacy
from .simulate import Futures, futures_cost

OPI = [("pcr_vol", 1), ("pcr_oi", 1), ("skew", 1), ("cskew", -1), ("cpiv", -1), ("net_opt_flow", 1)]


def opi(P: pd.DataFrame) -> pd.Series:
    s = 0
    for f, sign in OPI:
        r = P.groupby("date")[f].rank(pct=True)
        s = s + (r if sign > 0 else 1 - r)
    return s


def futures_panel_legacy() -> str:
    path = DATA_DIR / "panels" / "futures_ohlc_legacy.parquet"
    frames = []
    for d in legacy.available_days():
        x = pd.read_parquet(legacy._path(d), columns=["date", "inst", "symbol", "expiry", "open", "high", "low", "close", "oi", "contracts", "lot"],
                            filters=[("inst", "==", "STF")])
        frames.append(x)
    F = pd.concat(frames, ignore_index=True).drop(columns="inst")
    F["symbol"] = F["symbol"].astype(str)
    F.to_parquet(path, index=False)
    return str(path)


def corporate_action_days(fut_path: str) -> set:
    """(symbol, date) where the raw near-future return differs from Yahoo's adjusted return by > 15%."""
    F = pd.read_parquet(fut_path)
    F["date"] = pd.to_datetime(F["date"])
    F = F.sort_values(["symbol", "expiry", "date"])
    F["r"] = np.log(F["close"] / F.groupby(["symbol", "expiry"])["close"].shift(1))
    near = F.dropna(subset=["r"]).sort_values("expiry").groupby(["symbol", "date"]).first()["r"]
    h = history.load()
    C = h.pivot(index="date", columns="symbol", values="close")
    ra = np.log(C / C.shift(1)).stack().rename("ra")
    j = near.to_frame().join(ra, how="left")
    bad = j[(j["r"] - j["ra"]).abs() > 0.15]
    return set(bad.index)


def run(start="2019-01-01", end="2024-07-05", decile=0.10, H=10, stop_mult=2.0, processes=3, progress=print):
    fe = Fe.build(processes=processes, legacy=True)
    fe["date"] = pd.to_datetime(fe["date"])
    fe = fe[(fe["date"] >= start) & (fe["date"] <= end) & ~fe["symbol"].isin(A.INDICES)].copy()
    fe = fe.sort_values(["symbol", "date"])
    fe["net_opt_flow"] = fe["put_oi_flow"] - fe["call_oi_flow"]
    fe["opi"] = opi(fe)
    progress(f"features {fe.shape}, {fe['date'].nunique()} days")
    # Pragyam stops from long adjusted history
    h = history.load()
    from . import pragyam as PG
    stops = []
    for s, d in h.groupby("symbol"):
        if s not in set(fe["symbol"]):
            continue
        d = d.set_index("date")[["open", "high", "low", "close"]]
        if len(d) < 300:
            continue
        R = PG.run(d)
        stops.append(pd.DataFrame({"date": R.index, "symbol": s, "stopL": R["stopL"].values / d["close"].values,
                                   "stopS": R["stopS"].values / d["close"].values}))
    st = pd.concat(stops)
    st["date"] = pd.to_datetime(st["date"])
    fe = fe.merge(st, on=["date", "symbol"], how="left")
    fp = futures_panel_legacy()
    cad = corporate_action_days(fp)
    progress(f"corporate-action days flagged: {len(cad)}")
    FU = Futures(fp)
    cost = futures_cost()
    fe["pct"] = fe.groupby("date")["opi"].rank(pct=True)
    sel = fe[(fe["pct"] > 1 - decile) | (fe["pct"] <= decile)].copy()
    sel["side"] = np.where(sel["pct"] > 1 - decile, 1, -1)
    rows = []
    for r in sel.itertuples():
        sr = None
        base = r.stopL if r.side > 0 else r.stopS
        if stop_mult and np.isfinite(base):
            sr = 1 + stop_mult * (base - 1)
        x = FU.trade(r.symbol, r.date, r.side, H, stop_ratio=sr, cost=cost)
        if x is None:
            continue
        i = FU.pos[r.date]
        if any((r.symbol, d) in cad for d in FU.cal[i:i + H + 2]):
            continue
        rows.append(x)
    T = pd.DataFrame(rows)
    return fe, T


def report(fe: pd.DataFrame, T: pd.DataFrame, H=10) -> dict:
    d = T.groupby(["date", "side"])["net"].mean().unstack()
    pair = (d[1] + d[-1]).dropna()
    out = dict(pair_net=pair.mean(), t=A._nw_t(pair.to_numpy(), H), sharpe=pair.mean() / pair.std() * np.sqrt(252 / H),
               long=d[1].mean(), short=d[-1].mean(), trades=len(T), days=T["date"].nunique(),
               stop_rate=(T["exit_reason"] == "stop").mean())
    by_year = pair.groupby(pair.index.year).agg(["mean", "std", "count"])
    by_year["t"] = [A._nw_t(pair[pair.index.year == y].to_numpy(), H) for y in by_year.index]
    out["by_year"] = by_year
    return out


def corporate_action_factors_legacy(fut_path: str) -> dict:
    """{(date, symbol): f} for the pre-UDiFF archive (no lot sizes there): f = raw futures
    return / Yahoo adjusted return on days they disagree by more than 15%."""
    F = pd.read_parquet(fut_path)
    F["date"] = pd.to_datetime(F["date"])
    F = F.sort_values(["symbol", "expiry", "date"])
    F["r"] = np.log(F["close"] / F.groupby(["symbol", "expiry"])["close"].shift(1))
    near = F.dropna(subset=["r"]).sort_values("expiry").groupby(["symbol", "date"]).first()["r"]
    h = history.load()
    C = h.pivot(index="date", columns="symbol", values="close")
    ra = np.log(C / C.shift(1)).stack().rename("ra")
    j = near.to_frame().join(ra, how="left")
    bad = j[(j["r"] - j["ra"]).abs() > 0.15]
    return {(d, s): float(np.exp(r - a)) for (s, d), r, a in zip(bad.index, bad["r"], bad["ra"])}
