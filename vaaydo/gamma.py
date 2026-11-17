"""
Gamma exposure and writers' hedge flow for any underlying, from one session's bhavcopy.

    gex_dealer   Σ sign·γ·OI·S²·1%, dealers long calls (+) and short puts (−)
    gex_seller   −Σ γ·OI·S²·1%: all open options treated as written (gamma sold)
    hedge_flow   Σ ΔOI·δ·S: what writers of the day's new OI must trade in the future to stay
                 hedged (new calls written → buy; new puts written → sell)

All three are per unit of futures OI notional (S × futures OI), so a stock's numbers compare
with another's. IV for every option comes from its expiry's out-of-the-money smile (traded
strikes), held flat beyond the traded wings. Options more than 60 days out are left out.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import pricing
from .config import DATA_DIR, RISK_FREE

STOCK_GAMMA = DATA_DIR / "panels" / "stock_gamma.parquet"
MAX_DTE = 60


def gamma(F, K, T, sigma):
    sT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / sT
    return np.exp(-RISK_FREE * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (F * sT)


def underlying(opt: pd.DataFrame, fut: pd.DataFrame, asof: pd.Timestamp, S: float) -> dict:
    """opt: live options of one underlying (expiry > asof); fut: its futures rows."""
    o = opt[(opt["expiry"] > asof)].copy()
    if o.empty or not np.isfinite(S) or S <= 0:
        return {}
    o["dte"] = (o["expiry"] - asof).dt.days
    o = o[o["dte"] <= MAX_DTE]
    if o.empty:
        return {}
    o["T"] = o["dte"].clip(lower=1) / 365
    fw = fut.set_index("expiry")["close"].to_dict()
    o["F"] = o["expiry"].map(fw)
    o["F"] = o["F"].fillna(S * np.exp(RISK_FREE * o["T"]))
    o["k"] = np.log(o["strike"] / o["F"])
    is_c = (o["otype"] == "CE").to_numpy()
    tr = (o["contracts"] > 0) & (o["close"] >= 0.05) & ((is_c & (o["k"] >= 0)) | (~is_c & (o["k"] <= 0)))
    t = o[tr]
    iv_t = pricing.implied_vol(t["close"].to_numpy(), t["F"].to_numpy(), t["strike"].to_numpy(), t["T"].to_numpy(),
                               (t["otype"] == "CE").to_numpy())
    t = t.assign(iv=iv_t).dropna(subset=["iv"])
    iv = np.full(len(o), np.nan)
    for e, x in t.groupby("expiry"):
        m = (o["expiry"] == e).to_numpy()
        if len(x) >= 2:
            srt = np.argsort(x["k"].to_numpy())
            iv[m] = np.interp(o.loc[m, "k"].to_numpy(), x["k"].to_numpy()[srt], x["iv"].to_numpy()[srt])
        else:
            iv[m] = x["iv"].iloc[0]
    if np.isfinite(iv).any():
        iv = np.where(np.isfinite(iv), iv, np.nanmedian(iv))
    else:
        return {}
    oi = o["oi"].to_numpy(float)
    base = S * float(fut["oi"].sum())
    if base <= 0:
        return {}
    g = gamma(o["F"].to_numpy(), o["strike"].to_numpy(), o["T"].to_numpy(), iv)
    unit = g * oi * S * S * 0.01
    sign = np.where(is_c, 1.0, -1.0)
    dl = pricing.delta(o["F"].to_numpy(), o["strike"].to_numpy(), o["T"].to_numpy(), iv, is_c)
    return dict(gex_dealer=float((sign * unit).sum() / base), gex_seller=float(-unit.sum() / base),
                hedge_flow=float((o["oi_chg"].to_numpy(float) * dl * S).sum() / base))


def stock_day(d) -> pd.DataFrame:
    from .data import bhavcopy, legacy
    try:
        df = bhavcopy.load_day(d)
    except Exception:
        df = legacy.load_day(d)
    df = df[df["inst"].astype(str).isin(["STO", "STF"])].copy()
    df["inst"] = df["inst"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["otype"] = df["otype"].astype(str)
    df["expiry"] = pd.to_datetime(df["expiry"])
    asof = pd.Timestamp(d)
    rows = []
    for s, g in df.groupby("symbol"):
        fut = g[(g["inst"] == "STF") & (g["expiry"] >= asof) & (g["close"] > 0)].sort_values("expiry")
        if fut.empty:
            continue
        sp = g["spot"].dropna() if "spot" in g else pd.Series(dtype=float)
        sp = sp[sp > 0]
        T0 = max((fut["expiry"].iloc[0] - asof).days, 1) / 365
        S = float(sp.iloc[0]) if len(sp) else float(fut["close"].iloc[0] * np.exp(-RISK_FREE * T0))
        try:
            r = underlying(g[g["inst"] == "STO"], fut, asof, S)
        except Exception:
            r = {}
        if r:
            rows.append({"date": asof, "symbol": s, **r})
    return pd.DataFrame(rows)


def build_stocks(days=None, processes=4) -> pd.DataFrame:
    from .data import bhavcopy, legacy
    if days is None:
        days = sorted(set(legacy.available_days()) | set(bhavcopy.available_days()))
    with Pool(processes) as p:
        G = pd.concat(p.imap(stock_day, days, chunksize=8), ignore_index=True)
    G.to_parquet(STOCK_GAMMA, index=False)
    return G
