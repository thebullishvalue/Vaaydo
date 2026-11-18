"""
Inference for time-series signals on one instrument (the NIFTY desk), at the standard the
owner's Nishkarsh system uses:

    evidence(signal, fwd, h)   phase-averaged IC on non-overlapping h-spaced rows, Fisher t,
                               Newey–West(h) t of the daily rank IC, stationary block bootstrap
                               CI, yearly ICs, the label's momentum as a baseline, and the
                               minimum detectable IC (80% power, one-sided 5%)
    accept(e)                  HAC t ≥ 2, CI excluding 0, ≥ 70% of years positive, IC > baseline
    precedent(state, fwd, h)   nearest past states (OAS-shrunk Mahalanobis, Theiler gap h,
                               only realised outcomes): mean analog outcome minus the pool mean
    tranche_backtest(pos, r)   h-tranche book of a [-1, 1] position, net of costs, break-even cost

All causal: nothing at row t uses an outcome not realised by t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def _rank(x):
    return pd.Series(x).rank().to_numpy(float)


def spearman(x, y, min_obs=8) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_obs:
        return np.nan
    ra, rb = _rank(a[m]), _rank(b[m])
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def nw_t(x, lags) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < lags + 5:
        return np.nan
    u = v - v.mean()
    s = u @ u / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    return float(v.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def block_bootstrap_ic(x, y, mean_block, reps=2000, seed=7) -> tuple[float, float]:
    a, b = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 30:
        return np.nan, np.nan
    ra, rb = _rank(a[m]), _rank(b[m])
    rng = np.random.default_rng(seed)
    p = 1 / max(mean_block, 1)
    out = np.empty(reps)
    t = np.arange(n)
    for k in range(reps):
        new = rng.random(n) < p
        new[0] = True
        start = rng.integers(0, n, n)
        sp = np.maximum.accumulate(np.where(new, t, 0))
        idx = (start[sp] + (t - sp)) % n
        A, B = ra[idx] - ra[idx].mean(), rb[idx] - rb[idx].mean()
        out[k] = (A @ B) / np.sqrt((A @ A) * (B @ B))
    lo, hi = np.nanquantile(out, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lo), float(hi)


def mde(n) -> float:
    """Smallest IC detectable with 80% power at one-sided 5%, n independent windows."""
    if n <= 3:
        return np.nan
    return float(np.tanh((stats.norm.ppf(1 - ALPHA) + stats.norm.ppf(0.8)) * np.sqrt(1.06 / (n - 3))))


def evidence(signal: pd.Series, fwd: pd.Series, h: int, momentum: pd.Series | None = None) -> dict:
    df = pd.DataFrame({"s": signal, "r": fwd})
    if momentum is not None:
        df["m"] = momentum
    df = df.dropna(subset=["s", "r"])
    if len(df) < 60:
        return dict(ic=np.nan, n=0)
    ph = [spearman(df["s"].iloc[k::h], df["r"].iloc[k::h]) for k in range(h)]
    ph = [x for x in ph if np.isfinite(x)]
    ic = float(np.mean(ph))
    n = len(df.iloc[::h])
    t = np.arctanh(ic) / np.sqrt(1.06 / (n - 3)) if n > 3 else np.nan
    rs, rr = _rank(df["s"]), _rank(df["r"])
    hac = nw_t(((rs - rs.mean()) / rs.std()) * ((rr - rr.mean()) / rr.std()), h)
    lo, hi = block_bootstrap_ic(df["s"], df["r"], 2 * h)
    yrs = {int(y): spearman(g["s"], g["r"]) for y, g in df.groupby(pd.DatetimeIndex(df.index).year) if len(g) >= 30}
    base = np.nan
    if "m" in df:
        dm = df.dropna(subset=["m"])
        base = float(np.nanmean([spearman(dm["m"].iloc[k::h], dm["r"].iloc[k::h]) for k in range(h)]))
    return dict(ic=ic, n=n, t=t, hac_t=hac, ci_lo=lo, ci_hi=hi, mde=mde(n), phase_min=min(ph), phase_max=max(ph),
                years_pos=float(np.mean([v > 0 for v in yrs.values()])) if yrs else np.nan, n_years=len(yrs),
                baseline=base, yearly=yrs)


def accept(e: dict) -> bool:
    return bool(np.isfinite(e.get("ic", np.nan)) and e["hac_t"] >= 2 and e["ci_lo"] > 0 and e["years_pos"] >= 0.7
                and (not np.isfinite(e["baseline"]) or e["ic"] > e["baseline"]))


# ── precedent ────────────────────────────────────────────────────────────────────────

def oas(S: np.ndarray, n: int) -> np.ndarray:
    """Oracle Approximating Shrinkage toward a scaled identity (Chen, Wiesel, Eldar, Hero 2010)."""
    p = S.shape[0]
    mu = np.trace(S) / p
    a = np.mean(S * S)
    den = (n + 1.0) * (a - mu * mu / p)
    rho = min(max((a + mu * mu) / den, 0.0), 1.0) if den != 0 else 1.0
    return (1 - rho) * S + rho * mu * np.eye(p)


def precedent(state: pd.DataFrame, fwd: pd.Series, h: int, k: int = 10, min_pool: int = 500) -> pd.Series:
    """Read at t = mean h-forward outcome of the k nearest past states minus the pool's mean.
    Candidates: rows i with i + h <= t (outcome realised), all state features finite.
    Distinct episodes: greedy Theiler exclusion of width h on the candidates' positions."""
    X = state.to_numpy(float)
    y = fwd.reindex(state.index).to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    out = np.full(len(X), np.nan)
    for t in range(len(X)):
        if not ok[t]:
            continue
        cand = np.where(ok[: max(t - h + 1, 0)] & np.isfinite(y[: max(t - h + 1, 0)]))[0]
        if len(cand) < min_pool:
            continue
        C = X[cand]
        S = np.cov(C, rowvar=False)
        Si = np.linalg.pinv(oas(S, len(C)))
        d = X[t] - C
        dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, Si, d), 0))
        chosen, times = [], []
        for j in np.argsort(dist):
            if all(abs(cand[j] - u) >= h for u in times):
                chosen.append(j)
                times.append(cand[j])
                if len(chosen) == k:
                    break
        out[t] = y[cand[chosen]].mean() - y[cand].mean()
    return pd.Series(out, index=state.index)


def tranche_backtest(pos: pd.Series, r1: pd.Series, h: int, cost: float = 0.0003) -> dict:
    """pos decided at row t's close in [-1, 1]; r1 = the return earned from t to t+1 by a
    position held over that step. Book = mean of the last h tranches."""
    p = pos.clip(-1, 1).fillna(0)
    book = p.rolling(h, min_periods=1).sum() / h
    trade = (p - p.shift(h).fillna(0)).abs() / h
    g = book * r1
    net = (g - cost * trade).dropna()
    turn = trade.reindex(net.index).sum() / (len(net) / 252)
    ga = g.reindex(net.index).mean() * 252
    return dict(net_sharpe=float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else np.nan,
                gross_ann=float(ga), net_ann=float(net.mean() * 252), turnover=float(turn),
                breakeven=float(ga / turn) if turn > 0 else np.nan, net=net)


def precedent_now(state: pd.DataFrame, fwd: pd.Series, h: int, k: int = 10, min_pool: int = 500) -> dict:
    """Today's precedent read only (the last row), with the analogs it used."""
    X = state.to_numpy(float)
    y = fwd.reindex(state.index).to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    t = len(X) - 1
    if t < 0 or not ok[t]:
        return {}
    cand = np.where(ok[: t - h + 1] & np.isfinite(y[: t - h + 1]))[0]
    if len(cand) < min_pool:
        return {}
    C = X[cand]
    Si = np.linalg.pinv(oas(np.cov(C, rowvar=False), len(C)))
    d = X[t] - C
    dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, Si, d), 0))
    chosen, times = [], []
    for j in np.argsort(dist):
        if all(abs(cand[j] - u) >= h for u in times):
            chosen.append(j)
            times.append(cand[j])
            if len(chosen) == k:
                break
    idx = state.index[cand[chosen]]
    outs = y[cand[chosen]]
    up, base = float((outs > 0).mean()), float((y[cand] > 0).mean())
    p = float(stats.binomtest(int((outs > 0).sum()), len(outs), base).pvalue)
    return dict(read=float(outs.mean() - y[cand].mean()), analogs=list(zip(idx, outs, dist[chosen])), up=up,
                base_up=base, p=p, pool=len(cand))
