"""
What a structure is worth, in rupees per lot, on the forecast distribution.

Entry is at the day's close adjusted for spread (we sell at the bid side, buy at the
offer side). The position is assumed held to the exit window just before expiry (stock
options are physically settled, so VAAYDO always exits before expiry), valued at
intrinsic, with exit spread and charges. Every number here is after costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import costs as C
from .config import DEFAULT, Config
from .forecast import Shocks
from .strategies import Structure


@dataclass
class Evaluation:
    structure: Structure
    lot: int
    entry_net: float          # ₹/lot received (+) or paid (-) at entry, after spread
    max_profit: float         # ₹/lot, after entry costs
    max_loss: float           # ₹/lot, positive number, after entry costs
    ev: float                 # ₹/lot expected P&L after all costs
    ev_stressed: float        # ₹/lot EV with the vol forecast wrong against us
    pop: float                # P(P&L > 0)
    cvar5: float              # ₹/lot mean of the worst 5% outcomes (negative)
    costs: float              # ₹/lot expected round-trip costs
    market_vol: float         # premium-weighted IV of the legs
    breakevens: tuple

    @property
    def ev_to_risk(self) -> float:
        return self.ev / self.max_loss if self.max_loss > 0 else 0.0

    @property
    def ev_to_cvar(self) -> float:
        return self.ev / abs(self.cvar5) if self.cvar5 < 0 else np.inf


def _intrinsic(otype: str, K: float, S: np.ndarray) -> np.ndarray:
    return np.maximum(S - K, 0) if otype == "CE" else np.maximum(K - S, 0)


def payoff(st: Structure, S: np.ndarray) -> np.ndarray:
    out = np.zeros_like(S, dtype=float)
    for l in st.legs:
        out += l.qty * _intrinsic(l.otype, l.strike, S)
    return out


def _grid(st: Structure, forward: float) -> np.ndarray:
    ks = sorted({l.strike for l in st.legs})
    return np.array([1e-6, *ks, forward * 4])


def evaluate(st: Structure, shocks: Shocks, forward: float, vol_forecast: float, lot: int,
             cfg: Config = DEFAULT) -> Evaluation:
    qty = np.array([l.qty for l in st.legs], dtype=float)
    close = np.array([l.entry_ref for l in st.legs])
    K = np.array([l.strike for l in st.legs])
    V = np.array([l.contracts or 1 for l in st.legs], dtype=float)
    fills = C.fill(close, qty, cfg, K, V)
    entry_net = float(-(qty * fills).sum())                       # per unit
    entry_cost = C.charges(fills, qty, lot, cfg)                   # ₹ per lot

    def pnl_paths(vol: float):
        S = shocks.terminal(forward, vol)
        intr = np.stack([_intrinsic(l.otype, l.strike, S) for l in st.legs])   # legs x paths
        value = (qty[:, None] * intr).sum(axis=0)
        exit_slip = (np.abs(qty)[:, None] * np.where(intr > 0, C.half_spread(intr, cfg, K[:, None], V[:, None]), 0.0)).sum(axis=0)
        exit_cost = C.charges(intr.mean(axis=1), -qty, lot, cfg)
        return (value + entry_net - exit_slip) * lot - entry_cost - exit_cost, exit_cost, exit_slip.mean() * lot

    pnl, exit_cost, exit_slip = pnl_paths(vol_forecast)
    s = cfg.selection.forecast_stress
    stressed_vol = vol_forecast * (1 + s if st.vol_side == "short" else 1 - s)
    pnl_s, _, _ = pnl_paths(stressed_vol)

    g = _grid(st, forward)
    pay = payoff(st, g) + entry_net
    # worst case includes closing at the worst point: spread on every leg + exit charges
    at_min = np.where(pay <= pay.min() + 1e-9)[0]
    worst = g[[at_min[np.argmin(np.abs(g[at_min] - forward))]]]   # where you'd actually be closing
    intr_w = np.array([_intrinsic(l.otype, l.strike, worst)[0] for l in st.legs])
    exit_w = float((np.abs(qty) * np.where(intr_w > 0, C.half_spread(intr_w, cfg, K, V), 0)).sum()) * lot \
        + C.charges(intr_w, -qty, lot, cfg)
    max_profit = float(pay.max() * lot - entry_cost)
    max_loss = float(-(pay.min() * lot - entry_cost) + exit_w)
    if st.vol_side == "long":
        max_profit = float(np.percentile(pnl, 99))   # unbounded upside: report a plausible high

    # breakevens from the payoff on a fine grid
    fine = np.linspace(min(g[1:-1]) * 0.7, max(g[1:-1]) * 1.3, 2000)
    pf = payoff(st, fine) + entry_net
    sign = np.sign(pf)
    be = tuple(float(fine[i]) for i in np.where(np.diff(sign) != 0)[0])

    tail = np.sort(pnl)[: max(1, len(pnl) // 20)]
    w = np.abs(close)
    mvol = float(np.nansum([l.iv for l in st.legs] * w) / w.sum()) if w.sum() > 0 else np.nan
    spread_cost = float(np.abs(qty * (fills - close)).sum() * lot)
    return Evaluation(
        structure=st, lot=lot, entry_net=entry_net * lot, max_profit=max_profit, max_loss=max_loss,
        ev=float(pnl.mean()), ev_stressed=float(pnl_s.mean()), pop=float((pnl > 0).mean()),
        cvar5=float(tail.mean()), costs=entry_cost + exit_cost + exit_slip + spread_cost,
        market_vol=mvol, breakevens=be,
    )
