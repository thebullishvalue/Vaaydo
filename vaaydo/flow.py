"""
Positioning: who is opening and closing, read from futures price + open interest and
from the option chain's open interest, computed end-of-day over history so their predictive value can be *measured* (see `backtest.flow_study`) before
the system is allowed to act on them.

    price up,   OI up    long build-up        price down, OI up    short build-up
    price up,   OI down  short covering       price down, OI down  long unwinding

Futures OI is summed across expiries so the monthly rollover is not misread as unwinding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LONG_BUILD, SHORT_BUILD, SHORT_COVER, LONG_UNWIND, NEUTRAL = (
    "Long build-up", "Short build-up", "Short covering", "Long unwinding", "Neutral")
# noise floors: a move smaller than this is not a move
FLOORS = {1: (0.005, 0.02), 5: (0.015, 0.04)}   # window: (|return|, |OI change %|)


def classify(ret: np.ndarray, oi_pct: np.ndarray, window: int = 1) -> np.ndarray:
    pr, oi = FLOORS[window]
    out = np.full(len(ret), NEUTRAL, dtype=object)
    up, dn = ret > pr, ret < -pr
    add, cut = oi_pct > oi, oi_pct < -oi
    out[up & add] = LONG_BUILD
    out[dn & add] = SHORT_BUILD
    out[up & cut] = SHORT_COVER
    out[dn & cut] = LONG_UNWIND
    return out


def history(pos: pd.DataFrame) -> pd.DataFrame:
    """Add 1-day and 5-day build-up states to the positioning panel."""
    p = pos.sort_values(["symbol", "date"]).copy()
    p["oi_pct"] = p["oi_chg"] / p["oi_prev"].where(p["oi_prev"] > 0)
    g = p.groupby("symbol", observed=True)
    p["ret5"] = g["ret"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    oi5_prev = g["oi"].shift(5)
    p["oi5_pct"] = p["oi"] / oi5_prev.where(oi5_prev > 0) - 1
    p["build1"] = classify(p["ret"].fillna(0).to_numpy(), p["oi_pct"].fillna(0).to_numpy(), 1)
    p["build5"] = classify(p["ret5"].fillna(0).to_numpy(), p["oi5_pct"].fillna(0).to_numpy(), 5)
    # signed positioning score: direction of the move, scaled up when positions are opened
    # with it and down when the move is positions closing
    conv = np.where(p["build5"].isin([LONG_BUILD, SHORT_BUILD]), 1.0,
                    np.where(p["build5"].isin([SHORT_COVER, LONG_UNWIND]), 0.5, 0.0))
    p["flow_score"] = np.sign(p["ret5"].fillna(0)) * conv
    return p


def option_walls(ex: pd.DataFrame, spot: float) -> dict:
    """Heaviest put OI at/below spot (support) and call OI at/above spot (resistance)."""
    puts = ex[(ex["otype"] == "PE") & (ex["strike"] <= spot)]
    calls = ex[(ex["otype"] == "CE") & (ex["strike"] >= spot)]
    out = {"support": np.nan, "resistance": np.nan, "pcr_oi": np.nan, "put_oi_added": np.nan,
           "call_oi_added": np.nan}
    if len(puts) and puts["oi"].max() > 0:
        out["support"] = float(puts.loc[puts["oi"].idxmax(), "strike"])
    if len(calls) and calls["oi"].max() > 0:
        out["resistance"] = float(calls.loc[calls["oi"].idxmax(), "strike"])
    c_oi = ex.loc[ex["otype"] == "CE", "oi"].sum()
    p_oi = ex.loc[ex["otype"] == "PE", "oi"].sum()
    out["pcr_oi"] = float(p_oi / c_oi) if c_oi > 0 else np.nan
    out["put_oi_added"] = float(ex.loc[ex["otype"] == "PE", "oi_chg"].sum())
    out["call_oi_added"] = float(ex.loc[ex["otype"] == "CE", "oi_chg"].sum())
    return out
