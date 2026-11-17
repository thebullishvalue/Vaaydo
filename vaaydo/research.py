"""
Model-free harvest study: what did fixed-rule option structures actually earn on real
NSE prices, after spread and charges, with the system's exit rules?

No forecast, no gates: every liquid name, every monthly cycle, the same structures.
If a structure (or a structure in a state, e.g. "high IV vs its own history") earns
consistently across cycles, that is an edge the live system can exploit. If nothing does,
the system should say "no trade" and the dashboard should say why.

Results are aggregated per cycle first and t-tested across cycles: trades within one
month share one market and are not independent observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import strategies as S
from . import vol as V
from .backtest import MARK_COLS, Management, _index_day, _settle_position, monthly_expiries
from .config import DEFAULT, Config
from .data import bhavcopy, store
from .engine import Candidate
from .evaluate import evaluate
from .forecast import Shocks

INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
_DUMMY = Shocks(np.random.default_rng(0).standard_normal(400), 20, 400, 5)


def _pick(structs: list[S.Structure], kind: str, delta: float | None = None, width: int | None = None):
    """Choose the listed structure of `kind` whose short leg delta is nearest `delta`
    and whose wing width (in strikes, via strike distance rank) is `width`."""
    xs = [s for s in structs if s.kind == kind]
    if not xs:
        return None
    width_of = lambda s: max(l.strike for l in s.legs) - min(l.strike for l in s.legs)
    if delta is None:
        xs.sort(key=width_of)
        return xs[min((width or 1) - 1, len(xs) - 1)]
    def short_delta(s):
        return np.mean([abs(l.delta) for l in s.legs if l.qty < 0])
    best = min(abs(short_delta(s) - delta) for s in xs)
    xs = [s for s in xs if abs(short_delta(s) - delta) <= best + 1e-9]
    xs.sort(key=width_of)
    return xs[min(width - 1, len(xs) - 1)] if width else xs[0]


RULES = {
    "PCS 15Δ": ("Put credit spread", 0.15, 2), "PCS 25Δ": ("Put credit spread", 0.25, 2),
    "CCS 15Δ": ("Call credit spread", 0.15, 2), "CCS 25Δ": ("Call credit spread", 0.25, 2),
    "IC 20Δ": ("Iron condor", 0.20, 2), "Iron fly": ("Iron butterfly", None, 3),
    "Long straddle": ("Long straddle", None, None),
}
VARIANTS = {"TP50 + T-3": Management(), "hold to T-3": Management(take_profit=None, long_take_profit=None)}


def zero_cost(cfg: Config) -> Config:
    from dataclasses import replace
    from .config import Costs
    z = Costs(brokerage_per_order=0, stt_sell_premium=0, exchange_txn=0, sebi_per_crore=0, stamp_buy=0, gst=0)
    return replace(cfg, costs=z, liquidity=replace(cfg.liquidity, tick=0.0, half_spread_frac=0.0))


def index_expiries(symbol: str = "NIFTY") -> list[pd.Timestamp]:
    """Every expiry listed for an index (weeklies included), from the bhavcopy itself."""
    days = bhavcopy.available_days()
    o = bhavcopy.load_range(days[::3], columns=["inst", "symbol", "expiry"],
                            filters=[("symbol", "==", symbol), ("inst", "==", "IDO")])
    return sorted(pd.to_datetime(o["expiry"].unique()))


def harvest(start, end=None, cfg: Config = DEFAULT, entry_lag: int = 1, progress=None,
            expiries: list | None = None, symbols: set | None = None, variants: dict | None = None) -> pd.DataFrame:
    """expiries/symbols default to monthly stock expiries on every name. For index weeklies
    pass `expiries=index_expiries()` and `symbols={"NIFTY"}` (cash settled: hold to expiry)."""
    from dataclasses import replace as _rep
    cal = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in bhavcopy.available_days()))
    all_exps = expiries or monthly_expiries()
    variants = variants or VARIANTS
    if expiries is not None:   # short-dated: no DTE floor
        cfg = _rep(cfg, selection=_rep(cfg.selection, min_dte=1))
    exps = [e for e in all_exps if pd.Timestamp(start) < e <= min(pd.Timestamp(end or cal[-1]), cal[-1])]
    spot_px = store.spot()
    rets = store.returns()
    rows = []
    for i, E in enumerate(exps):
        prev = [x for x in all_exps if x < E]
        if not prev:
            continue
        entry = cal[cal > prev[-1]][entry_lag - 1]
        window = cal[(cal > entry) & (cal <= E)]
        if progress:
            progress(i, len(exps), entry)
        chain = store.chain(entry.date())
        chain = chain[chain["expiry"] == E]
        if symbols:
            chain = chain[chain["symbol"].isin(symbols)]
        days_all = list(window)
        opts = {}
        for d in days_all:
            o = bhavcopy.load_range([d.date()], columns=MARK_COLS,
                                    filters=[("inst", "in", ["STO", "IDO"]), ("expiry", "==", E)])
            opts[d] = _index_day(o)
        for sym, ch in chain.groupby("symbol", observed=True):
            sm = V.smile(ch, entry, cfg)
            if sm is None or sm.n_tradable < 4:
                continue
            r = rets[sym].loc[:entry] if sym in rets else pd.Series(dtype=float)
            rv1 = V.realised(r, 21)
            structs = S.candidates(sm, cfg, allow_long_vol=True)
            lot = int(ch["lot"].iloc[0])
            feats = dict(entry=entry, expiry=E, symbol=sym, index=sym in INDICES, atm_iv=sm.atm_iv, skew=sm.skew,
                         chain_contracts=int(ch["contracts"].sum()), n_tradable=sm.n_tradable,
                         iv_rv1m=sm.atm_iv / rv1 if rv1 and np.isfinite(rv1) else np.nan, lot=lot)
            for name, (kind, delta, width) in RULES.items():
                st = _pick(structs, kind, delta, width)
                if st is None:
                    continue
                for cname, ccfg in (("net", cfg), ("gross", zero_cost(cfg))):
                    ev = evaluate(st, _DUMMY, sm.forward, sm.atm_iv, lot, ccfg)
                    if ev.max_loss <= 0:
                        continue
                    cand = Candidate(sym, E, ev, sm.atm_iv, sm.atm_iv, sm.forward, sm.spot)
                    for vname, mg in variants.items():
                        exit_day = window[-(mg.exit_days_before + 1)]
                        days = [d for d in days_all if d <= exit_day]
                        out = _settle_position(cand, days, opts, spot_px, mg, ccfg)
                        rows.append({**feats, "rule": name, "variant": vname, "costs": cname, "pnl": out["pnl"],
                                     "max_loss": ev.max_loss, "ret_on_risk": out["pnl"] / ev.max_loss,
                                     "credit": ev.entry_net, "realised_vol": out["realised_vol"],
                                     "exit": out["exit_reason"]})
    return pd.DataFrame(rows)


def cycle_t(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean return on risk per cycle, then mean / t-stat across cycles."""
    per = df.groupby(["entry", *by])["ret_on_risk"].mean().reset_index()
    g = per.groupby(by)["ret_on_risk"]
    out = pd.DataFrame({"mean_ror": g.mean(), "sd": g.std(), "cycles": g.count(),
                        "pos_cycles": per.assign(p=per["ret_on_risk"] > 0).groupby(by)["p"].mean()})
    out["t"] = out["mean_ror"] / (out["sd"] / np.sqrt(out["cycles"]))
    n = df.groupby(by).size().rename("trades")
    return out.join(n).reset_index()


def conditional(df: pd.DataFrame, feature: str, q: int = 3) -> pd.DataFrame:
    """Within each cycle, bucket names by `feature`; does the bucket predict return on risk?"""
    d = df.dropna(subset=[feature]).copy()
    d["bucket"] = d.groupby(["entry", "rule", "variant"])[feature].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False) if len(s) >= q else np.nan)
    return cycle_t(d.dropna(subset=["bucket"]), ["rule", "variant", "bucket"])


# ── evidence store ────────────────────────────────────────────────────────────

def evidence_paths():
    from .config import DATA_DIR
    d = DATA_DIR / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / "evidence_monthly.parquet", d / "evidence_nifty_weekly.parquet"


def load_evidence() -> pd.DataFrame:
    """All study rows, tagged by universe: 'stocks' / 'index' (monthly) and 'NIFTY weekly'."""
    pm, pw = evidence_paths()
    frames = []
    if pm.exists():
        m = pd.read_parquet(pm)
        m["universe"] = np.where(m["index"], "index", "stocks")
        frames.append(m)
    if pw.exists():
        w = pd.read_parquet(pw)
        w["universe"] = "NIFTY weekly"
        frames.append(w)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def update_evidence(progress=None) -> pd.DataFrame:
    """Extend both studies with any cycles completed since the last run."""
    from .backtest import Management
    pm, pw = evidence_paths()
    weekly_variants = {"TP50 + expiry": Management(exit_days_before=0),
                       "hold to expiry": Management(take_profit=None, long_take_profit=None, exit_days_before=0)}
    for path, kw in ((pm, {}), (pw, dict(expiries=index_expiries("NIFTY"), symbols={"NIFTY"}, variants=weekly_variants))):
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        start = (old["entry"].max() + pd.Timedelta(days=1)) if len(old) else pd.Timestamp("2024-07-10")
        new = harvest(start, progress=progress, **kw)
        if len(new):
            new = new[new["entry"] > (old["entry"].max() if len(old) else pd.Timestamp(0))]
            pd.concat([old, new], ignore_index=True).to_parquet(path, index=False)
    return load_evidence()
