"""
Trade simulator on real NSE futures prices.

A trade decided at the close of day d:
    contract  the nearest expiry that still has >= h + 3 sessions after entry; otherwise
              the next month (no position is ever carried into its expiry week)
    entry     that contract's OPEN on the next session (d+1)
    stop      Pragyam's stop (20% touch in 10 bars), translated from spot to the contract
              by the entry-day ratio; checked against each day's futures high/low; a gap
              through the stop exits at the open
    target    optional: Pragyam's cone edge
    exit      the stop, the target, or the close of session d+h
    costs     round trip on futures: STT 0.02% on the sell side, exchange 0.00173%, SEBI,
              stamp 0.002% on the buy side, GST, brokerage, plus slippage by liquidity
Returns are on notional (the futures price), per trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR


def futures_cost(notional_turnover_cr: float | None = None) -> float:
    """Round-trip cost as a fraction of notional. Slippage 3 bp/side for liquid contracts,
    up to 8 bp for thin ones (by daily futures turnover in ₹ crore)."""
    statutory = 0.0002 + 2 * 0.0000173 + 0.00002 + 2 * 1e-6
    statutory += 0.18 * (2 * 0.0000173 + 2 * 1e-6)
    if notional_turnover_cr is None or not np.isfinite(notional_turnover_cr):
        slip = 0.0005
    else:
        slip = float(np.clip(0.0008 - 0.00005 * np.log10(max(notional_turnover_cr, 1)) * 10, 0.0003, 0.0008))
    return statutory + 2 * slip


class Futures:
    def __init__(self, path=None):
        F = pd.read_parquet(path or DATA_DIR / "panels" / "futures_ohlc.parquet")
        F["date"] = pd.to_datetime(F["date"])
        F["expiry"] = pd.to_datetime(F["expiry"])
        self.cal = pd.DatetimeIndex(sorted(F["date"].unique()))
        self.pos = {d: i for i, d in enumerate(self.cal)}
        self.c = {}
        for (s, e), g in F.groupby(["symbol", "expiry"]):
            g = g.set_index("date").reindex(self.cal)
            # an untraded contract has open/high/low = 0 in the bhavcopy: that is "no price"
            self.c[(s, e)] = g[["open", "high", "low", "close"]].where(lambda x: x > 0).to_numpy()
        self.exps = F.groupby("symbol")["expiry"].apply(lambda x: sorted(x.unique())).to_dict()

    def contract(self, sym: str, d: pd.Timestamp, h: int):
        i = self.pos.get(d)
        if i is None or i + 1 >= len(self.cal):
            return None
        for e in self.exps.get(sym, []):
            if e <= d:
                continue
            left = int(((self.cal > self.cal[i + 1]) & (self.cal <= e)).sum()) if e <= self.cal[-1] else 10 ** 6
            if left >= h + 3:
                return e
        return None

    def trade(self, sym, d, side, h, stop_ratio=None, target_ratio=None, cost=None):
        """side +1 long / -1 short. stop_ratio, target_ratio: level / spot close at d."""
        e = self.contract(sym, pd.Timestamp(d), h)
        if e is None:
            return None
        a = self.c.get((sym, e))
        i = self.pos[pd.Timestamp(d)]
        if a is None or i + h >= len(self.cal):
            return None
        entry = a[i + 1, 0]
        ref = a[i, 3]                       # contract close at d: translate spot levels by ratio
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(ref):
            return None
        stop = ref * stop_ratio if stop_ratio else None
        tgt = ref * target_ratio if target_ratio else None
        exit_px, why, held = None, "time", h
        for k in range(1, h + 1):
            o, hi, lo, cl = a[i + k]
            if not np.isfinite(cl):
                continue
            if stop is not None and np.isfinite(lo) and np.isfinite(hi):
                hit = lo <= stop if side > 0 else hi >= stop
                if hit:
                    gap = np.isfinite(o) and ((o <= stop) if side > 0 else (o >= stop))
                    exit_px, why, held = (o if (gap and k > 1) else stop), "stop", k
                    break
            if tgt is not None and np.isfinite(lo) and np.isfinite(hi):
                hit = hi >= tgt if side > 0 else lo <= tgt
                if hit:
                    exit_px, why, held = tgt, "target", k
                    break
            exit_px = cl
        if exit_px is None:
            return None
        gross = side * (exit_px / entry - 1)
        c = futures_cost() if cost is None else cost
        return dict(symbol=sym, date=pd.Timestamp(d), side=side, expiry=e, entry=entry, exit=exit_px, exit_reason=why,
                    held=held, gross=gross, net=gross - c, cost=c)
