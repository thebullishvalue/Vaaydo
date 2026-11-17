"""
The index desk's options book: a defined-risk short-premium position on NIFTY and BANKNIFTY
each expiry cycle (docs/PREREGISTRATION.md, "Index desk, round 2").

    entry     open of the first session with 5 or fewer sessions (and at least 2) to the expiry
    strikes   from the previous close: forward F, ATM IV σ, T; m = F·σ·√T
    condor    sell call ≈ F + 0.75m and put ≈ F − 0.75m; buy call ≈ F + 2m and put ≈ F − 2m
    exit      held to expiry, settled at NSE's final settlement price
    costs     half-spread max(₹0.5, 1%) a leg, STT, exchange, GST, stamp, brokerage

Every price is a real bhavcopy print; P&L is per unit of the index and as % of max loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import indices as I
from . import pricing
from .config import DATA_DIR, RISK_FREE

CHAIN = DATA_DIR / "panels" / "index_chain.parquet"
LOT_GUESS = {"NIFTY": 50, "BANKNIFTY": 25}        # brokerage only, for days the file carries no lot


def _chain_day(d):
    x = I._load(d)
    x = x[x["inst"] == "IDO"]
    if x.empty:
        return x
    f = x.groupby("symbol")["strike"].median()
    x = x[(x["expiry"] - pd.Timestamp(d)).dt.days <= 45]
    x = x[np.abs(np.log(x["strike"] / x["symbol"].map(f))) < 0.25]
    x = x[["symbol", "expiry", "strike", "otype", "open", "close", "settle", "contracts", "oi", "lot"]].copy()
    x["date"] = pd.Timestamp(d)
    return x


def build_chain(days=None, processes=4) -> pd.DataFrame:
    from .data import bhavcopy, legacy
    if days is None:
        days = sorted(set(legacy.available_days()) | set(bhavcopy.available_days()))
    with Pool(processes) as p:
        C = pd.concat(p.map(_chain_day, days, chunksize=16), ignore_index=True)
    for c in ("open", "close", "settle"):
        C[c] = C[c].astype("float32")
    C.to_parquet(CHAIN, index=False)
    return C


def stt_sell(d) -> float:
    return 0.001 if pd.Timestamp(d) >= pd.Timestamp("2024-10-01") else 0.000625


def stt_exercise(d) -> float:
    return 0.00125


@dataclass
class Spec:
    short_m: float = 0.75
    wing_m: float = 2.0
    max_left: int = 5
    min_left: int = 2
    fly: bool = False            # iron fly: short ATM, wings at ±wing_m


def _nearest(strikes: np.ndarray, x: float) -> float:
    return float(strikes[np.argmin(np.abs(strikes - x))])


def cycles(C: pd.DataFrame, P: pd.DataFrame, spec: Spec = Spec()) -> pd.DataFrame:
    """One row per (symbol, expiry): legs, fills, settlement, P&L."""
    cal = pd.DatetimeIndex(sorted(C["date"].unique()))
    spot = P.set_index(["symbol", "date"])["spot"].to_dict()
    pos = {d: i for i, d in enumerate(cal)}
    C = C.set_index(["symbol", "expiry", "date", "strike", "otype"]).sort_index()
    out = []
    for (sym, E), g in C.groupby(level=[0, 1]):
        E = pd.Timestamp(E)
        g = g.droplevel([0, 1])                       # (date, strike, otype)
        if E not in pos:
            continue
        iE = pos[E]
        # entry session: first t with min_left <= iE - i(t) <= max_left
        t_i = iE - spec.max_left
        if t_i < 1:
            continue
        t, prev = cal[t_i], cal[t_i - 1]
        try:
            g0 = g.xs(prev, level=0)       # previous close: strikes
            g1 = g.xs(t, level=0)          # entry open
            gE = g.xs(E, level=0)          # expiry settlement
        except KeyError:
            continue
        ks = np.sort(g0.index.get_level_values(0).unique().to_numpy(dtype=float))
        if len(ks) < 8:
            continue
        # forward by put-call parity at the busiest strike pair, ATM IV
        c0 = g0.xs("CE", level=1)
        p0 = g0.xs("PE", level=1)
        j = c0[["close", "contracts"]].join(p0[["close", "contracts"]], lsuffix="_c", rsuffix="_p", how="inner")
        j = j[(j["contracts_c"] > 0) & (j["contracts_p"] > 0)]
        if j.empty:
            continue
        T = max((E - prev).days, 1) / 365
        k_act = (j["contracts_c"] + j["contracts_p"]).idxmax()
        F = float(k_act + np.exp(RISK_FREE * T) * (j.loc[k_act, "close_c"] - j.loc[k_act, "close_p"]))
        katm = _nearest(j.index.to_numpy(dtype=float), F)
        ivc = pricing.implied_vol(j.loc[katm, "close_c"], F, katm, T, True)
        ivp = pricing.implied_vol(j.loc[katm, "close_p"], F, katm, T, False)
        sig = float(np.nanmean([ivc, ivp]))
        if not np.isfinite(sig):
            continue
        m = F * sig * np.sqrt(T)
        if spec.fly:
            kc_s = kp_s = katm
        else:
            kc_s, kp_s = _nearest(ks, F + spec.short_m * m), _nearest(ks, F - spec.short_m * m)
        kc_l, kp_l = _nearest(ks, F + spec.wing_m * m), _nearest(ks, F - spec.wing_m * m)
        if not (kc_l > kc_s >= kp_s > kp_l) and not (spec.fly and kc_l > katm > kp_l):
            continue
        legs = [("CE", kc_s, -1), ("PE", kp_s, -1), ("CE", kc_l, +1), ("PE", kp_l, +1)]
        fills, ok = [], True
        for ot, k, q in legs:
            try:
                r1 = g1.loc[(k, ot)]
                r0 = g0.loc[(k, ot)]
            except KeyError:
                ok = False
                break
            px = float(r1["open"]) if (r1["contracts"] > 0 and r1["open"] > 0) else float(r0["close"])
            hs = max(0.5, 0.01 * px)
            fills.append(px + hs if q > 0 else max(px - hs, 0.05))
        if not ok:
            continue
        st = gE["settle"].to_numpy(dtype=float)
        st = st[np.isfinite(st) & (st > 0)]
        S_E = float(np.median(st)) if len(st) else float(spot.get((sym, E), np.nan))
        if not np.isfinite(S_E) or S_E <= 0 or abs(np.log(S_E / F)) > 0.3:
            continue
        pay = [max(S_E - k, 0) if ot == "CE" else max(k - S_E, 0) for ot, k, q in legs]
        credit = -sum(q * f for (ot, k, q), f in zip(legs, fills))          # received (positive)
        payout = -sum(q * p for (ot, k, q), p in zip(legs, pay))            # paid at expiry (positive = we pay)
        lot = float(g1["lot"].max()) if g1["lot"].max() and g1["lot"].max() > 0 else LOT_GUESS[sym]
        sold = sum(f for (ot, k, q), f in zip(legs, fills) if q < 0)
        bought = sum(f for (ot, k, q), f in zip(legs, fills) if q > 0)
        exch = 0.00035 * (sold + bought)
        costs = (stt_sell(t) * sold + exch * 1.18 + 0.00003 * bought + 4 * 20 / lot * 1.18
                 + stt_exercise(E) * sum(p for (ot, k, q), p in zip(legs, pay) if q > 0))
        width = max(kc_l - kc_s, kp_s - kp_l)
        max_loss = width - credit + costs
        pnl = credit - payout - costs
        out.append(dict(symbol=sym, expiry=E, entry=t, F=F, iv=sig, m=m, kc_s=kc_s, kp_s=kp_s, kc_l=kc_l, kp_l=kp_l,
                        credit=credit, settle=S_E, payout=payout, costs=costs, pnl=pnl, max_loss=max_loss,
                        ret=pnl / max_loss if max_loss > 0 else np.nan, sessions=iE - t_i, lot=lot))
    return pd.DataFrame(out)
