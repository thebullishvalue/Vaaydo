"""
The daily scan. For one trading day, using only information available at that close:

  1. every underlying: expiry, smile (real option prices), realised vol, event flag,
     positioning, and the features the playbook conditions on (with today's terciles)
  2. names that match a LIVE playbook rule -> that rule's structure, valued and sized
  3. names that match a WATCH rule -> watchlist (shown, never traded)

Division of labour, learned the hard way (docs/RESEARCH.md):
  * the playbook decides WHAT to trade, from each rule's out-of-sample record on real prices;
  * the model prices RISK: max loss, P(profit), tail loss, breakevens, stress. Its
    distribution is centred on the futures forward and scaled to the market's ATM IV, the
    best out-of-sample vol forecast found, so its EV is honest: about minus costs unless the
    evidence says otherwise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import flow as F
from . import forecast as FC
from . import strategies as S
from . import vol as V
from .config import DEFAULT, Config
from .data import store
from .evaluate import Evaluation, evaluate

log = logging.getLogger(__name__)
INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}


@dataclass
class Candidate:
    symbol: str
    expiry: pd.Timestamp
    evaluation: Evaluation
    atm_iv: float
    vol_forecast: float
    forward: float
    spot: float
    reason: str = ""          # the playbook rule that selected it ("" = gated in the legacy engine)


@dataclass
class ScanResult:
    asof: pd.Timestamp
    market: pd.DataFrame                              # one row per underlying
    candidates: list[Candidate] = field(default_factory=list)   # from LIVE rules
    watchlist: list[Candidate] = field(default_factory=list)    # from WATCH rules
    rejected: dict = field(default_factory=dict)
    smiles: dict = field(default_factory=dict)


def pick_expiry(expiries, asof: pd.Timestamp, cfg: Config) -> pd.Timestamp | None:
    ok = [e for e in sorted(expiries) if cfg.selection.min_dte <= (e - asof).days <= cfg.selection.max_dte]
    return ok[0] if ok else None


def _bucket(s: pd.Series) -> pd.Series:
    ok = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if ok.sum() >= 6:
        out[ok] = pd.qcut(s[ok].rank(method="first"), 3, labels=False).astype(float)
    return out


def scan(asof: date, cfg: Config = DEFAULT, rules: pd.DataFrame | None = None, n_paths: int | None = None,
         symbols: list[str] | None = None) -> ScanResult:
    """`rules`: playbook.table(...) output (status column decides live/watch)."""
    asof = pd.Timestamp(asof)
    chain = store.chain(asof.date())
    if chain.empty:
        raise ValueError(f"no option data for {asof.date()}")
    rets = store.returns().loc[:asof]
    pos = F.history(store.positioning())
    pos = pos[pos["date"] == asof].set_index("symbol")
    syms = symbols or sorted(chain["symbol"].unique())
    rows, rejected, smiles = [], {}, {}
    for sym in syms:
        ch = chain[chain["symbol"] == sym]
        if ch.empty:
            continue
        exp = pick_expiry(ch["expiry"].unique(), asof, cfg)
        if exp is None:
            rejected[sym] = "no expiry 20-60 days out"
            continue
        ex = ch[ch["expiry"] == exp]
        sm = V.smile(ex, asof, cfg)
        if sm is None:
            rejected[sym] = "no usable smile (too few traded options)"
            continue
        later = sorted(e for e in ch["expiry"].unique() if e > exp)
        nxt = V.smile(ch[ch["expiry"] == later[0]], asof, cfg) if later else None
        term = sm.atm_iv / nxt.atm_iv if nxt is not None and nxt.atm_iv > 0 else np.nan
        r = rets[sym] if sym in rets else pd.Series(dtype=float)
        vf = V.forecast(r, cfg)
        rv1 = V.realised(r, 21)
        prow = pos.loc[sym] if sym in pos.index else None
        smiles[sym] = sm
        rows.append(dict(
            symbol=sym, index=sym in INDICES, expiry=exp, dte=(exp - asof).days, spot=sm.spot, forward=sm.forward,
            lot=int(ex["lot"].iloc[0]), atm_iv=sm.atm_iv, skew=sm.skew, term=term,
            event=bool(np.isfinite(term) and term > cfg.selection.term_inversion_flag),
            n_tradable=sm.n_tradable, chain_contracts=int(ex["contracts"].sum()),
            rv_1m=rv1, iv_rv1m=sm.atm_iv / rv1 if rv1 and np.isfinite(rv1) else np.nan,
            rv_forecast=vf.forecast if vf else np.nan,
            build1=prow["build1"] if prow is not None else None, build5=prow["build5"] if prow is not None else None,
            fut_ret5=prow["ret5"] if prow is not None else np.nan, oi5_pct=prow["oi5_pct"] if prow is not None else np.nan,
            **F.option_walls(ex, sm.spot)))
    market = pd.DataFrame(rows)
    if market.empty:
        return ScanResult(asof, market, rejected=rejected)
    stocks = ~market["index"]
    for f in ("atm_iv", "iv_rv1m", "skew", "chain_contracts"):
        market[f"b_{f}"] = np.nan
        market.loc[stocks, f"b_{f}"] = _bucket(market.loc[stocks, f])

    res = ScanResult(asof, market, rejected=rejected, smiles=smiles)
    if rules is None or rules.empty:
        return res
    pool = np.concatenate([FC.standardised_shocks(rets[s], cfg) for s in list(rets.columns)[:60]])
    from .research import RULES
    for status, bucket in (("live", res.candidates), ("watch", res.watchlist)):
        for _, rr in rules[rules["status"] == status].iterrows():
            rule = rr["_rule"]
            if rule.universe == "NIFTY weekly":
                c = _nifty_weekly(asof, chain, rets, pool, rule, cfg, n_paths)
                if c is not None:
                    c.reason = rr["rule"]
                    bucket.append(c)
                continue
            m = market[market["index"] == (rule.universe == "index")]
            if rule.feature:
                m = m[m[f"b_{rule.feature}"] == rule.bucket]
            kind, delta, width = RULES[rule.structure]
            for _, row in m.iterrows():
                if row["event"] and kind != "Long straddle":
                    continue    # known event priced in: no short premium into it
                c = value(asof, row, smiles[row["symbol"]], kind, delta, width, rets, pool, cfg, n_paths)
                if c is not None:
                    c.reason = rr["rule"]
                    bucket.append(c)
    cheap = market.set_index("symbol")["iv_rv1m"].to_dict()
    res.watchlist.sort(key=lambda c: cheap.get(c.symbol, np.inf))   # cheapest vol first
    for name in ("candidates", "watchlist"):   # one structure per name: the best-evidenced rule wins
        seen, keep = set(), []
        for c in getattr(res, name):
            if c.symbol not in seen:
                seen.add(c.symbol)
                keep.append(c)
        setattr(res, name, keep)
    return res


def value(asof, row, sm: V.Smile, kind: str, delta, width, rets: pd.DataFrame, pool, cfg: Config = DEFAULT,
          n_paths: int | None = None) -> Candidate | None:
    """Build one structure on one name and value it on the market-implied, fat-tailed distribution."""
    from .research import _pick
    st = _pick(S.candidates(sm, cfg, allow_long_vol=True), kind, delta, width)
    if st is None:
        return None
    H = FC.trading_days_between(pd.Timestamp(asof), sm.expiry)
    r = rets[row["symbol"]] if row["symbol"] in rets else pd.Series(dtype=float)
    sh = FC.shocks_for(r, max(H, 1), pool, cfg, n=n_paths)
    if sh is None:
        return None
    e = evaluate(st, sh, sm.forward, sm.atm_iv, int(row["lot"]), cfg)
    if e.max_loss <= 0:
        return None
    return Candidate(row["symbol"], sm.expiry, e, sm.atm_iv, sm.atm_iv, sm.forward, sm.spot)


def _nifty_weekly(asof, chain, rets, pool, rule, cfg, n_paths) -> Candidate | None:
    """The nearest NIFTY expiry at least 2 days out (cash settled: held to expiry)."""
    from .research import RULES
    n = chain[chain["symbol"] == "NIFTY"]
    exps = sorted(e for e in n["expiry"].unique() if (e - asof).days >= 2)
    if not exps:
        return None
    from dataclasses import replace as _rep
    sm = V.smile(n[n["expiry"] == exps[0]], asof, cfg)
    if sm is None:
        return None
    kind, delta, width = RULES[rule.structure]
    row = dict(symbol="NIFTY", lot=int(n["lot"].iloc[0]))
    return value(asof, row, sm, kind, delta, width, rets, pool, cfg, n_paths)


def analyse(asof: date, symbol: str, res: ScanResult, cfg: Config = DEFAULT, n_paths: int = 10000) -> list[Candidate]:
    """Every defined-risk structure on one name, valued the same way (for discretionary use)."""
    sm = res.smiles.get(symbol)
    if sm is None:
        return []
    row = res.market.set_index("symbol").loc[symbol]
    rets = store.returns().loc[:pd.Timestamp(asof)]
    r = rets[symbol] if symbol in rets else pd.Series(dtype=float)
    H = FC.trading_days_between(pd.Timestamp(asof), sm.expiry)
    pool = np.concatenate([FC.standardised_shocks(rets[s], cfg) for s in list(rets.columns)[:60]])
    sh = FC.shocks_for(r, max(H, 1), pool, cfg, n=n_paths)
    out = []
    for st in S.candidates(sm, cfg, allow_long_vol=True):
        e = evaluate(st, sh, sm.forward, sm.atm_iv, int(row["lot"]), cfg)
        if e.max_loss > 0:
            out.append(Candidate(symbol, sm.expiry, e, sm.atm_iv, sm.atm_iv, sm.forward, sm.spot, reason="manual"))
    return sorted(out, key=lambda c: c.evaluation.ev_to_risk, reverse=True)
