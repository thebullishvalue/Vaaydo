"""
Option-contract flow: what the day's trading did to every option contract, read from two
consecutive end-of-day files, and aggregated per underlying.

Per contract (traded today, with a real previous close):
    premium up,   OI up    buyers opening      (long build-up)
    premium down, OI up    writers opening     (short build-up)
    premium up,   OI down  writers closing     (short covering)
    premium down, OI down  buyers closing      (long unwinding)
    volume above the open interest it started the day with  -> unusual activity
    volume against the previous session's volume            -> volume surge

Per underlying (lots, normalised by the side's open interest at the start of the day):
    call_buy / call_write / put_buy / put_write      OI opened by each party on each side
    buyer_tilt   (call_buy - put_buy) / OI           option BUYERS leaning bullish
    writer_tilt  (put_write - call_write) / OI       option WRITERS leaning bullish (selling puts)
    unusual_tilt (unusual call volume - unusual put volume) / volume
    surge_call / surge_put                          volume vs previous session
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .data import bhavcopy, legacy

NOISE_PREMIUM = 0.005      # premium must move at least 0.5% to count as up/down
NOISE_OI = 0.0             # any OI change counts (it is exact)
COLS = ["inst", "symbol", "expiry", "strike", "otype", "close", "oi", "oi_chg", "contracts", "lot"]


def _load(d: date, is_legacy: bool) -> pd.DataFrame:
    x = (legacy.load_day(d) if is_legacy else bhavcopy.load_day(d))[COLS + ["date"]]
    x = x[x["inst"].astype(str).isin(["STO", "IDO"])].copy()
    for c in ("symbol", "otype"):
        x[c] = x[c].astype(str)
    x["expiry"] = pd.to_datetime(x["expiry"])
    return x


def contract_flow(today: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Per-contract day reading: build-up class, unusual and surge flags."""
    key = ["symbol", "expiry", "strike", "otype"]
    p = prev[key + ["close", "contracts"]].rename(columns={"close": "prev_close", "contracts": "prev_contracts"})
    x = today.merge(p, on=key, how="left")
    x["oi_start"] = x["oi"] - x["oi_chg"]
    traded = (x["contracts"] > 0) & (x["prev_close"] > 0) & (x["prev_contracts"].fillna(0) > 0)
    dp = (x["close"] / x["prev_close"] - 1).where(traded)
    up, dn = dp > NOISE_PREMIUM, dp < -NOISE_PREMIUM
    add, cut = x["oi_chg"] > NOISE_OI, x["oi_chg"] < -NOISE_OI
    x["flow"] = np.select([up & add, dn & add, up & cut, dn & cut],
                          ["buyers opening", "writers opening", "writers closing", "buyers closing"], "")
    lot = x["lot"].where(x["lot"] > 0, 1)
    x["oi_chg_lots"] = x["oi_chg"] / lot
    x["unusual"] = (x["contracts"] > x["oi_start"] / lot) & (x["oi_start"] / lot >= 50)
    x["premium_chg"] = dp
    return x


def underlying_flow(cf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a contract_flow frame per underlying."""
    lot = cf["lot"].where(cf["lot"] > 0, 1)
    cf = cf.assign(oi_start_lots=cf["oi_start"] / lot, add_lots=cf["oi_chg_lots"].clip(lower=0))
    rows = {}
    for (sym, ot), g in cf.groupby(["symbol", "otype"]):
        r = rows.setdefault(sym, {})
        side = "call" if ot == "CE" else "put"
        r[f"{side}_oi"] = g["oi_start_lots"].sum()
        r[f"{side}_buy"] = g.loc[g["flow"] == "buyers opening", "add_lots"].sum()
        r[f"{side}_write"] = g.loc[g["flow"] == "writers opening", "add_lots"].sum()
        r[f"{side}_vol"] = g["contracts"].sum()
        r[f"{side}_prev_vol"] = g["prev_contracts"].fillna(0).sum()
        r[f"{side}_unusual_vol"] = g.loc[g["unusual"], "contracts"].sum()
    u = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    oi = (u.get("call_oi", 0) + u.get("put_oi", 0)).replace(0, np.nan)
    vol = (u.get("call_vol", 0) + u.get("put_vol", 0)).replace(0, np.nan)
    u["buyer_tilt"] = (u.get("call_buy", 0) - u.get("put_buy", 0)) / oi
    u["writer_tilt"] = (u.get("put_write", 0) - u.get("call_write", 0)) / oi
    u["unusual_tilt"] = (u.get("call_unusual_vol", 0) - u.get("put_unusual_vol", 0)) / vol
    u["surge_call"] = u.get("call_vol", 0) / u.get("call_prev_vol", 0).replace(0, np.nan)
    u["surge_put"] = u.get("put_vol", 0) / u.get("put_prev_vol", 0).replace(0, np.nan)
    u["surge_tilt"] = np.log(u["surge_call"] / u["surge_put"])
    u.index.name = "symbol"
    return u.reset_index()


def day(d: date, prev_d: date, is_legacy: bool) -> pd.DataFrame | None:
    try:
        cf = contract_flow(_load(d, is_legacy), _load(prev_d, is_legacy))
    except FileNotFoundError:
        return None
    u = underlying_flow(cf)
    u["date"] = pd.Timestamp(d)
    return u
