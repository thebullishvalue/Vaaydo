"""
The playbook: what VAAYDO is allowed to trade, and why.

A rule = a structure (from `research.RULES`) + an optional condition: the name's tercile
on one feature within that day's cross-section (e.g. "Long straddle, ATM IV in the
cheapest third"). The model does not decide what has an edge. Each rule's own record on
real prices does:

    live      net mean return on risk > 0, t >= T_LIVE, over >= MIN_CYCLES prior cycles
    watch     net mean > 0 and t >= T_WATCH: shown, not traded
    rejected  everything else

Status is always computed from cycles strictly before the decision date (walk-forward),
so the playbook's own record (`walk_forward`) is out-of-sample: a rule had to earn its
status before it was traded. With ~65 rules tested, a few will look good by chance;
the walk-forward record is what shows whether they kept working.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = {"atm_iv": "ATM IV", "iv_rv1m": "IV / 1M realised", "skew": "put skew", "chain_contracts": "chain liquidity"}
BUCKETS = {0: "low third", 1: "middle third", 2: "high third"}
T_LIVE, T_WATCH, MIN_CYCLES = 2.5, 1.5, 12
VARIANT = {"monthly": "TP50 + T-3", "weekly": "TP50 + expiry"}


@dataclass(frozen=True)
class Rule:
    structure: str                 # key of research.RULES
    feature: str | None = None
    bucket: int | None = None
    universe: str = "stocks"       # "stocks" | "index" (monthly) | "NIFTY weekly"

    @property
    def name(self) -> str:
        cond = f" · {FEATURES[self.feature]} {BUCKETS[self.bucket]}" if self.feature else ""
        return f"{self.structure}{cond} · {self.universe}"


def add_buckets(h: pd.DataFrame) -> pd.DataFrame:
    """Tercile of each feature within each cycle (same structure, same universe)."""
    h = h.copy()
    if "universe" not in h:
        h["universe"] = np.where(h["index"], "index", "stocks")
    for f in FEATURES:
        h[f"b_{f}"] = h.groupby(["entry", "rule", "universe", "variant", "costs"])[f].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if s.notna().sum() >= 6 else np.nan)
    return h


def rule_rows(h: pd.DataFrame, r: Rule) -> pd.DataFrame:
    x = h[(h["rule"] == r.structure) & (h["universe"] == r.universe)]
    if r.feature:
        x = x[x[f"b_{r.feature}"] == r.bucket]
    return x


def all_rules(h: pd.DataFrame) -> list[Rule]:
    out = []
    for uni in sorted(h["universe"].unique()):
        for s in sorted(h["rule"].unique()):
            out.append(Rule(s, None, None, uni))
            if uni == "stocks":
                for f in FEATURES:
                    for b in BUCKETS:
                        out.append(Rule(s, f, b, uni))
    return out


def per_cycle(h: pd.DataFrame, costs: str = "net") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(mean return on risk, trade count) per cycle x rule, and name -> Rule."""
    hb = h if "b_atm_iv" in h else add_buckets(h)
    hb = hb[(hb["costs"] == costs) & hb["variant"].isin(VARIANT.values())]
    means, counts, rules = {}, {}, {}
    for r in all_rules(hb):
        g = rule_rows(hb, r).groupby("entry")["ret_on_risk"]
        means[r.name], counts[r.name], rules[r.name] = g.mean(), g.size(), r
    return pd.DataFrame(means).sort_index(), pd.DataFrame(counts).sort_index().fillna(0), rules


def _stats(per: pd.Series, n_trades: float) -> dict:
    per = per.dropna()
    n = len(per)
    m = float(per.mean()) if n else np.nan
    sd = float(per.std()) if n > 1 else np.nan
    t = m / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return dict(mean_ror=m, t=t, cycles=n, pos_cycles=float((per > 0).mean()) if n else np.nan, trades=int(n_trades))


def status_of(st: dict) -> str:
    if st["cycles"] >= MIN_CYCLES and st["mean_ror"] > 0 and st["t"] >= T_LIVE:
        return "live"
    if st["cycles"] >= MIN_CYCLES and st["mean_ror"] > 0 and st["t"] >= T_WATCH:
        return "watch"
    return "rejected"


def table(h: pd.DataFrame | None = None, before: pd.Timestamp | None = None, costs: str = "net",
          pc: tuple | None = None) -> pd.DataFrame:
    """Every rule's record (optionally using only cycles before `before`), with status."""
    means, counts, rules = pc or per_cycle(h, costs)
    if before is not None:
        means, counts = means[means.index < before], counts[counts.index < before]
    rows = []
    for name, r in rules.items():
        st = _stats(means[name] if name in means else pd.Series(dtype=float),
                    counts[name].sum() if name in counts else 0)
        rows.append(dict(rule=name, structure=r.structure, feature=r.feature, bucket=r.bucket,
                         universe=r.universe, **st, status=status_of(st), _rule=r))
    cols = ["rule", "structure", "feature", "bucket", "universe", "mean_ror", "t", "cycles", "pos_cycles",
            "trades", "status", "_rule"]
    return pd.DataFrame(rows, columns=cols).sort_values("t", ascending=False, na_position="last")


def walk_forward(h: pd.DataFrame) -> pd.DataFrame:
    """For each cycle: which rules were live using only earlier cycles, and what trading
    exactly those rules returned that cycle (equal weight per rule)."""
    pc = per_cycle(h)
    means = pc[0]
    out = []
    for c in means.index:
        t = table(before=c, pc=pc)
        live = t[t["status"] == "live"]["rule"].tolist()
        rets = [means.loc[c, n] for n in live if np.isfinite(means.loc[c, n])]
        out.append(dict(entry=c, live_rules=len(live), traded_rules=len(rets),
                        ret_on_risk=float(np.mean(rets)) if rets else 0.0, rules=", ".join(live[:4])))
    return pd.DataFrame(out)
