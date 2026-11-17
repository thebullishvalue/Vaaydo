"""
Options expression of the same signal: a debit vertical in the signal's direction.

    long signal   buy the ATM call, sell the call ~1 sigma(10 sessions) higher   (bull call spread)
    short signal  buy the ATM put,  sell the put  ~1 sigma(10 sessions) lower    (bear put spread)

Monthly expiry with >= 15 sessions left at entry. Both legs must have traded on the signal
day. Entry: signal-day close +/- liquidity half-spread on each leg (end-of-day data has no
next-open option price; this is the honest proxy). Exit: 10 sessions later at NSE settlement
price -/+ half-spread. All statutory option charges. Return on the debit paid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import costs as C
from .data import bhavcopy, legacy


def _day(d, is_legacy):
    x = legacy.load_day(d) if is_legacy else bhavcopy.load_day(d)
    x = x[x["inst"].astype(str) == "STO"].copy()
    for c in ("symbol", "otype"):
        x[c] = x[c].astype(str)
    x["expiry"] = pd.to_datetime(x["expiry"])
    return x


def spread_trade(sym, side, d0, d1, chain0, chain1, fut_close, sig_daily, lot_hint=1):
    """chain0/chain1: option rows for sym on entry/exit day. fut_close: forward at entry."""
    ot = "CE" if side > 0 else "PE"
    c0 = chain0[(chain0["otype"] == ot)]
    exps = sorted(e for e in c0["expiry"].unique() if (e - d0).days >= 21)
    if not exps:
        return None
    e = exps[0]
    c0 = c0[c0["expiry"] == e].set_index("strike")
    F = fut_close
    ks = np.array(sorted(c0.index))
    if len(ks) < 4 or not np.isfinite(F):
        return None
    k_atm = ks[np.argmin(np.abs(ks - F))]
    width = F * sig_daily * np.sqrt(10)
    k_far_t = k_atm + side * max(width, 1e-9)
    k_far = ks[np.argmin(np.abs(ks - k_far_t))]
    if k_far == k_atm:
        return None
    a, b = c0.loc[k_atm], c0.loc[k_far]
    if a["contracts"] <= 0 or b["contracts"] <= 0 or a["close"] <= 0 or b["close"] <= 0:
        return None
    qty = np.array([1.0, -1.0])
    K = np.array([k_atm, k_far])
    V = np.array([a["contracts"], b["contracts"]], dtype=float)
    entry_fill = C.fill(np.array([a["close"], b["close"]]), qty, strike=K, contracts=V)
    debit = float(entry_fill[0] - entry_fill[1])
    if debit <= 0 or debit >= abs(k_far - k_atm):
        return None
    c1 = chain1[(chain1["otype"] == ot) & (chain1["expiry"] == e)].set_index("strike")
    if k_atm not in c1.index or k_far not in c1.index:
        return None
    m = np.array([c1.loc[k_atm, "settle"], c1.loc[k_far, "settle"]], dtype=float)
    if not np.all(np.isfinite(m)):
        return None
    value = float(np.clip(m[0] - m[1], 0, abs(k_far - k_atm)))
    exit_fill = C.fill(m, -qty, strike=K, contracts=V)
    slip = float((m - exit_fill) @ qty) if value > 0 else 0.0
    units = max(1.0, round(1_000_000 / F))          # a lot is ~₹5-15 lakh notional: charges per lot, then per unit
    charges = (C.charges(entry_fill, qty, units) + C.charges(exit_fill, -qty, units)) / units
    pnl = value - abs(slip) - debit - charges
    return dict(symbol=sym, date=d0, side=side, expiry=e, k_atm=k_atm, k_far=k_far, debit=debit, value=value,
                ret_on_debit=pnl / debit, notional_frac=debit / F)
