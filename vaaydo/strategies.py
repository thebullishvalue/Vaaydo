"""
Candidate structures, built only from listed strikes that actually traded.

Defined risk only. Single-stock options in India are physically settled, lot values are
₹5-15 lakh, and a single overnight gap (results, a block deal, a regulatory order) can
move a stock 15-20%. A naked short strangle on one name is a bet you cannot size safely.

    short vol (market IV > forecast):  put credit spread, call credit spread,
                                       iron condor, iron butterfly
    long vol  (forecast > market IV):  long straddle, long strangle
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT, Config
from .vol import Smile


@dataclass
class Leg:
    otype: str        # 'CE' | 'PE'
    strike: float
    qty: int          # +1 buy, -1 sell (lots)
    close: float      # the day's close (reference)
    iv: float
    delta: float
    model: float = float("nan")   # smile-consistent price (see vol._smile_prices)
    contracts: int = 0             # traded on the entry day: drives the spread estimate

    @property
    def entry_ref(self) -> float:
        """Conservative reference: sell at the lower, buy at the higher of close and model."""
        if not self.model == self.model:   # NaN
            return self.close
        return min(self.close, self.model) if self.qty < 0 else max(self.close, self.model)


@dataclass
class Structure:
    kind: str
    vol_side: str     # 'short' | 'long'
    legs: list[Leg] = field(default_factory=list)

    @property
    def label(self) -> str:
        parts = []
        for l in sorted(self.legs, key=lambda l: (l.strike, l.otype)):
            parts.append(f"{'+' if l.qty > 0 else '-'}{abs(l.qty) if abs(l.qty) > 1 else ''}{l.strike:g}{l.otype}")
        return " ".join(parts)

    @property
    def net_delta(self) -> float:
        return float(sum(l.qty * l.delta for l in self.legs))


def _leg(row) -> Leg:
    return Leg(row.otype, float(row.strike), 0, float(row.close), float(row.iv), float(row.delta),
               float(getattr(row, "model_px", float("nan"))), int(getattr(row, "contracts", 0)))


def _with(row, qty) -> Leg:
    l = _leg(row)
    l.qty = qty
    return l


def _nearest_delta(side: pd.DataFrame, target: float) -> pd.Series | None:
    if side.empty:
        return None
    i = (side["delta"].abs() - target).abs().idxmin()
    return side.loc[i]


def candidates(sm: Smile, cfg: Config = DEFAULT, allow_long_vol: bool = True) -> list[Structure]:
    t = sm.table
    t = t[t["tradable"] & t["iv"].notna()]
    F = sm.forward
    puts = t[(t["otype"] == "PE") & (t["strike"] < F)].sort_values("strike", ascending=False).reset_index(drop=True)
    calls = t[(t["otype"] == "CE") & (t["strike"] > F)].sort_values("strike").reset_index(drop=True)
    out: list[Structure] = []
    sel = cfg.selection

    def wing(side: pd.DataFrame, i: int, steps: int):
        j = i + steps
        return side.loc[j] if j < len(side) else None

    pcs, ccs = {}, {}
    for d in sel.short_deltas:
        sp = _nearest_delta(puts, d)
        sc = _nearest_delta(calls, d)
        for w in sel.wing_steps:
            if sp is not None:
                lp = wing(puts, int(sp.name), w)
                if lp is not None:
                    s = Structure("Put credit spread", "short", [_with(sp, -1), _with(lp, +1)])
                    pcs[(d, w)] = s
                    out.append(s)
            if sc is not None:
                lc = wing(calls, int(sc.name), w)
                if lc is not None:
                    s = Structure("Call credit spread", "short", [_with(sc, -1), _with(lc, +1)])
                    ccs[(d, w)] = s
                    out.append(s)
    for key in pcs.keys() & ccs.keys():
        out.append(Structure("Iron condor", "short", pcs[key].legs + ccs[key].legs))

    # ATM: strike nearest the forward with both sides tradable
    both = set(t[t["otype"] == "CE"]["strike"]) & set(t[t["otype"] == "PE"]["strike"])
    if both:
        k = min(both, key=lambda s: abs(s - F))
        c_atm = t[(t["otype"] == "CE") & (t["strike"] == k)].iloc[0]
        p_atm = t[(t["otype"] == "PE") & (t["strike"] == k)].iloc[0]
        for w in sel.wing_steps:
            lp = puts[puts["strike"] < k].reset_index(drop=True)
            lc = calls[calls["strike"] > k].reset_index(drop=True)
            if len(lp) >= w and len(lc) >= w:
                out.append(Structure("Iron butterfly", "short", [
                    _with(c_atm, -1), _with(p_atm, -1), _with(lp.loc[w - 1], +1), _with(lc.loc[w - 1], +1)]))
        if allow_long_vol:
            out.append(Structure("Long straddle", "long", [_with(c_atm, +1), _with(p_atm, +1)]))
            for d in (0.25, 0.35):
                p = _nearest_delta(puts, d)
                c = _nearest_delta(calls, d)
                if p is not None and c is not None:
                    out.append(Structure("Long strangle", "long", [_with(c, +1), _with(p, +1)]))
    # de-duplicate identical leg sets
    seen, uniq = set(), []
    for s in out:
        key = (s.kind, s.label)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq
