"""
From candidates to a book you could actually hold.

Single-stock option positions are not independent: almost every Indian F&O stock has a
beta near 1 to NIFTY, and one sector can be a fifth of the list. Ranking trades one by
one and taking the top N (what the old engine did) builds a concentrated bet on the index.

Rules, in order:
  1. rank candidates by EV per rupee of max loss
  2. size each to `risk_per_trade` of capital by max loss (never more than that)
  3. skip a name if it would put more than `max_per_cluster` positions in one group of
     highly correlated stocks (groups found from returns, not a sector table)
  4. skip a name if it would push the book's worst NIFTY-gap stress loss past the limit
  5. stop at `max_positions` or `max_total_risk`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT, Config
from .engine import Candidate
from .evaluate import payoff

INDEX = "NIFTY"


@dataclass
class Planned:
    candidate: Candidate
    lots: int
    beta: float

    @property
    def risk(self) -> float:
        return self.candidate.evaluation.max_loss * self.lots

    @property
    def ev(self) -> float:
        return self.candidate.evaluation.ev * self.lots


def betas(rets: pd.DataFrame, window: int = 126) -> pd.Series:
    r = rets.iloc[-window:]
    if INDEX not in r:
        return pd.Series(1.0, index=r.columns)
    m = r[INDEX]
    cov = r.apply(lambda s: s.cov(m))
    return (cov / m.var()).clip(0, 3).fillna(1.0)


def stress_pnl(items: list[Planned], shock: float) -> float:
    """Book P&L at expiry if NIFTY gaps by `shock` and each stock moves beta * shock."""
    tot = 0.0
    for p in items:
        c = p.candidate
        e = c.evaluation
        S = np.array([c.forward * (1 + p.beta * shock)])
        per_unit = payoff(e.structure, S)[0] + e.entry_net / e.lot
        tot += (per_unit * e.lot - e.costs) * p.lots
    return tot


def stress_table(items: list[Planned], cfg: Config = DEFAULT) -> pd.Series:
    return pd.Series({f"{s:+.0%}": stress_pnl(items, s) for s in cfg.risk.stress_shocks})


def construct(cands: list[Candidate], rets: pd.DataFrame, cfg: Config = DEFAULT,
              existing: list[Planned] | None = None) -> list[Planned]:
    risk = cfg.risk
    book: list[Planned] = list(existing or [])
    held = {p.candidate.symbol for p in book}
    corr = rets.iloc[-126:].corr()
    beta = betas(rets)
    ranked = sorted(cands, key=lambda c: c.evaluation.ev_to_risk, reverse=True)
    for c in ranked:
        if len(book) >= risk.max_positions:
            break
        if c.symbol in held:
            continue
        e = c.evaluation
        lots = int(risk.capital * risk.risk_per_trade // e.max_loss) if e.max_loss > 0 else 0
        if lots < 1:
            continue
        if sum(p.risk for p in book) + e.max_loss * lots > risk.max_total_risk * risk.capital:
            continue
        if c.symbol in corr:
            peers = [p for p in book if p.candidate.symbol in corr
                     and corr.loc[c.symbol, p.candidate.symbol] > risk.cluster_corr]
            if len(peers) >= risk.max_per_cluster:
                continue
        trial = book + [Planned(c, lots, float(beta.get(c.symbol, 1.0)))]
        worst = min(stress_pnl(trial, s) for s in risk.stress_shocks)
        if worst < -risk.max_stress_loss * risk.capital:
            continue
        book = trial
        held.add(c.symbol)
    return book


def plan_frame(book: list[Planned]) -> pd.DataFrame:
    rows = []
    for p in book:
        c, e = p.candidate, p.candidate.evaluation
        rows.append(dict(
            symbol=c.symbol, structure=e.structure.kind, legs=e.structure.label, expiry=c.expiry.date(),
            lots=p.lots, lot_size=e.lot, entry=e.entry_net * p.lots, max_profit=e.max_profit * p.lots,
            max_loss=e.max_loss * p.lots, ev=e.ev * p.lots, ev_to_risk=e.ev_to_risk, pop=e.pop,
            cvar5=e.cvar5 * p.lots, atm_iv=c.atm_iv, vol_forecast=c.vol_forecast, beta=p.beta,
            breakevens=", ".join(f"{b:,.0f}" for b in e.breakevens)))
    return pd.DataFrame(rows)
