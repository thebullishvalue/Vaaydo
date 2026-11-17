"""
The daily F&O feature panel: one row per underlying per session, everything known at
that session's close. Built from the NSE F&O bhavcopy (futures + every option) and the
NSE equity bhavcopy (spot OHLC, VWAP, delivery).

Option features are read on the FRONT expiry: the first expiry at least 7 calendar days
away (expiry-week chains are dominated by gamma and pin effects).

    atm_iv        ATM implied vol, interpolated at the forward from traded OTM options
    atm_iv_next   the same for the next expiry (term structure)
    cpiv          call IV minus put IV at the same strikes near the money (Cremers-Weinbaum):
                  positive = calls bid up relative to puts
    skew          IV of the ~5% OTM put minus ATM IV (Xing-Zhang-Zhao): put demand
    cskew         IV of the ~5% OTM call minus ATM IV
    pcr_vol       put / call contracts traded (all expiries)
    pcr_oi        put / call open interest (all expiries)
    put_oi_flow   put OI added today / put OI yesterday   (call_oi_flow likewise)
    os_ratio      option notional traded / futures notional traded (Johnson-So)
    max_pain_d    log(max-pain strike / forward), front expiry
    wall_put_d    log(heaviest put-OI strike below forward / forward); wall_call_d likewise
    basis         annualised log(near future / spot) minus the risk-free rate: excess carry
    fut_oi        futures OI, all expiries (units); fut_oi_flow = change / previous
    roll_share    share of futures OI outside the near expiry (rollover)
    fut_turnover  futures notional traded (₹)
"""
from __future__ import annotations

from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import pricing
from .config import DATA_DIR, RISK_FREE
from .data import bhavcopy

FRONT_MIN_DTE = 7


def _interp0(k, iv):
    o = np.argsort(k)
    k, iv = k[o], iv[o]
    if len(k) == 0:
        return np.nan
    if k[0] > 0 or k[-1] < 0:
        j = np.argmin(np.abs(k))
        return iv[j] if abs(k[j]) < 0.05 else np.nan
    return float(np.interp(0.0, k, iv))


def _at_moneyness(k, iv, target, tol=0.03):
    if len(k) == 0:
        return np.nan
    j = np.argmin(np.abs(k - target))
    return float(iv[j]) if abs(k[j] - target) <= tol else np.nan


def _max_pain(ex: pd.DataFrame) -> float:
    """Strike at which option writers pay out least at expiry (vectorised)."""
    ks = np.sort(ex["strike"].unique())
    if len(ks) < 3:
        return np.nan
    c = ex[ex["otype"] == "CE"]
    p = ex[ex["otype"] == "PE"]
    S = ks[:, None]
    pay = (np.maximum(S - c["strike"].to_numpy()[None, :], 0) @ c["oi"].to_numpy(dtype=float)
           + np.maximum(p["strike"].to_numpy()[None, :] - S, 0) @ p["oi"].to_numpy(dtype=float))
    return float(ks[int(np.argmin(pay))])


def day_features(d: date, legacy: bool = False) -> pd.DataFrame:
    if legacy:
        from .data import legacy as L
        df = L.load_day(d)
    else:
        df = bhavcopy.load_day(d)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["inst"] = df["inst"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["otype"] = df["otype"].astype(str)
    asof = pd.Timestamp(d)
    fut = df[df["inst"].isin(["STF", "IDF"])]
    opt = df[df["inst"].isin(["STO", "IDO"])].copy()

    # ── futures ──
    alive = fut[fut["expiry"] > asof].sort_values("expiry")
    near = alive.groupby("symbol").first()
    ft = fut.groupby("symbol").agg(fut_oi=("oi", "sum"), fut_oi_chg=("oi_chg", "sum"), fut_contracts=("contracts", "sum"),
                                   lot=("lot", "max"), spot=("spot", "first"))
    ft["fut_close"] = near["close"]
    ft["fut_expiry"] = near["expiry"]
    ft["near_oi"] = near["oi"]
    T_near = ((ft["fut_expiry"] - asof).dt.days.clip(lower=1)) / 365
    ft["basis"] = np.log(ft["fut_close"] / ft["spot"]) / T_near - RISK_FREE
    ft["fut_oi_flow"] = ft["fut_oi_chg"] / (ft["fut_oi"] - ft["fut_oi_chg"]).where(lambda x: x > 0)
    ft["roll_share"] = 1 - ft["near_oi"] / ft["fut_oi"].where(ft["fut_oi"] > 0)
    ft["fut_turnover"] = ft["fut_contracts"] * ft["lot"] * ft["fut_close"]

    # ── options: aggregates over all expiries ──
    o = opt
    agg = o.groupby(["symbol", "otype"]).agg(vol=("contracts", "sum"), oi=("oi", "sum"), oi_chg=("oi_chg", "sum")).unstack()
    oa = pd.DataFrame(index=agg.index)
    oa["pcr_vol"] = agg[("vol", "PE")] / agg[("vol", "CE")].where(agg[("vol", "CE")] > 0)
    oa["pcr_oi"] = agg[("oi", "PE")] / agg[("oi", "CE")].where(agg[("oi", "CE")] > 0)
    for side, nm in (("PE", "put"), ("CE", "call")):
        prev = (agg[("oi", side)] - agg[("oi_chg", side)])
        oa[f"{nm}_oi_flow"] = agg[("oi_chg", side)] / prev.where(prev > 0)
    o_not = (o["contracts"] * o["lot"] * o["spot"]).groupby(o["symbol"]).sum()
    oa["opt_notional"] = o_not
    oa["opt_premium"] = (o["contracts"] * o["lot"] * o["close"]).groupby(o["symbol"]).sum()

    # ── options: front expiry surface ──
    o = o.copy()
    o["dte"] = (o["expiry"] - asof).dt.days
    exps = o[o["dte"] >= FRONT_MIN_DTE].groupby("symbol")["expiry"].apply(lambda s: sorted(s.unique()))
    front = exps.apply(lambda e: e[0]).rename("front")
    nxt = exps.apply(lambda e: e[1] if len(e) > 1 else pd.NaT).rename("next")
    o = o.join(front, on="symbol").join(nxt, on="symbol")
    o = o[(o["expiry"] == o["front"]) | (o["expiry"] == o["next"])]
    # forward per symbol-expiry: future if listed, else put-call parity at the busiest strike
    fwd = fut[["symbol", "expiry", "close"]].rename(columns={"close": "fwd"})
    o = o.merge(fwd, on=["symbol", "expiry"], how="left")
    miss = o["fwd"].isna()
    if miss.any():
        c = o[miss & (o["otype"] == "CE")].set_index(["symbol", "expiry", "strike"])[["close", "contracts"]]
        p = o[miss & (o["otype"] == "PE")].set_index(["symbol", "expiry", "strike"])[["close", "contracts"]]
        b = c.join(p, lsuffix="_c", rsuffix="_p", how="inner")
        b = b[(b["contracts_c"] > 0) & (b["contracts_p"] > 0)].copy()
        if len(b):
            b["act"] = b["contracts_c"] + b["contracts_p"]
            best = b.reset_index().sort_values("act").groupby(["symbol", "expiry"]).last()
            T = ((best.index.get_level_values(1) - asof).days / 365).to_numpy()
            pf = best["strike"] + np.exp(RISK_FREE * T) * (best["close_c"] - best["close_p"])
            o = o.merge(pf.rename("pfwd").reset_index(), on=["symbol", "expiry"], how="left")
            o["fwd"] = o["fwd"].fillna(o["pfwd"])
            o = o.drop(columns="pfwd")
    o["T"] = o["dte"] / 365
    tr = o[(o["contracts"] > 0) & (o["close"] >= 0.1) & o["fwd"].notna()].copy()
    tr["iv"] = pricing.implied_vol(tr["close"].to_numpy(), tr["fwd"].to_numpy(), tr["strike"].to_numpy(),
                                   tr["T"].to_numpy(), (tr["otype"] == "CE").to_numpy())
    tr = tr[tr["iv"].notna()]
    tr["k"] = np.log(tr["strike"] / tr["fwd"])
    otm = tr[((tr["otype"] == "CE") & (tr["k"] >= 0)) | ((tr["otype"] == "PE") & (tr["k"] <= 0))]

    rows = {}
    for (sym, exp), g in otm.groupby(["symbol", "expiry"]):
        r = rows.setdefault(sym, {})
        is_front = exp == g["front"].iloc[0]
        atm = _interp0(g["k"].to_numpy(), g["iv"].to_numpy())
        if not is_front:
            r["atm_iv_next"] = atm
            continue
        r["atm_iv"] = atm
        r["front_dte"] = int(g["dte"].iloc[0])
        r["fwd"] = float(g["fwd"].iloc[0])
        pu = g[g["otype"] == "PE"]
        ca = g[g["otype"] == "CE"]
        r["skew"] = _at_moneyness(pu["k"].to_numpy(), pu["iv"].to_numpy(), -0.05) - atm
        r["cskew"] = _at_moneyness(ca["k"].to_numpy(), ca["iv"].to_numpy(), 0.05) - atm
    # call-put IV spread at identical strikes, front expiry, near the money
    fr = tr[tr["expiry"] == tr["front"]]
    pair = fr[fr["k"].abs() <= 0.10].pivot_table(index=["symbol", "strike"], columns="otype",
                                                  values=["iv", "contracts"], aggfunc="first").dropna()
    if len(pair):
        w = np.minimum(pair[("contracts", "CE")], pair[("contracts", "PE")])
        diff = pair[("iv", "CE")] - pair[("iv", "PE")]
        cp = (diff * w).groupby(level=0).sum() / w.groupby(level=0).sum()
        for sym, v in cp.items():
            rows.setdefault(sym, {})["cpiv"] = float(v)
    # max pain and OI walls, front expiry, all strikes (OI counts even if untraded today)
    fo = o[o["expiry"] == o["front"]]
    for sym, g in fo.groupby("symbol"):
        F = g["fwd"].iloc[0]
        if not np.isfinite(F):
            continue
        r = rows.setdefault(sym, {})
        mp = _max_pain(g)
        r["max_pain_d"] = np.log(mp / F) if np.isfinite(mp) else np.nan
        pu = g[(g["otype"] == "PE") & (g["strike"] <= F)]
        ca = g[(g["otype"] == "CE") & (g["strike"] >= F)]
        if len(pu) and pu["oi"].max() > 0:
            r["wall_put_d"] = np.log(pu.loc[pu["oi"].idxmax(), "strike"] / F)
        if len(ca) and ca["oi"].max() > 0:
            r["wall_call_d"] = np.log(ca.loc[ca["oi"].idxmax(), "strike"] / F)
    sf = pd.DataFrame.from_dict(rows, orient="index")
    out = ft.join(oa, how="left").join(sf, how="left")
    out["os_ratio"] = out["opt_notional"] / out["fut_turnover"].where(out["fut_turnover"] > 0)
    out["term"] = out["atm_iv"] / out["atm_iv_next"]
    out["date"] = asof
    out.index.name = "symbol"
    return out.reset_index()


def _safe(d, legacy=False):
    try:
        return day_features(d, legacy)
    except Exception as e:  # one bad file must not kill the panel
        print("feature day failed", d, e)
        return None


def _safe_legacy(d):
    return _safe(d, True)


def build(days: list[date] | None = None, processes: int = 4, legacy: bool = False) -> pd.DataFrame:
    """Build (or extend) the panel at $VAAYDO_DATA/panels/features[_legacy].parquet."""
    path = DATA_DIR / "panels" / ("features_legacy.parquet" if legacy else "features.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(pd.to_datetime(old["date"]).dt.date) if len(old) else set()
    if days is None:
        from .data import legacy as L
        days = L.available_days() if legacy else bhavcopy.available_days()
    todo = [d for d in days if d not in have]
    if todo:
        with Pool(processes) as pool:
            parts = [p for p in pool.map(_safe_legacy if legacy else _safe, todo, chunksize=4) if p is not None]
        new = pd.concat([old, *parts], ignore_index=True) if parts else old
        new = new.sort_values(["date", "symbol"]).reset_index(drop=True)
        new.to_parquet(path, index=False)
        return new
    return old
