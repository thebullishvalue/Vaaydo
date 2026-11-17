"""
Tests of the index desk's registered hypotheses (docs/PREREGISTRATION.md, "Index desk").

    python -m vaaydo.index_research

Outcome: roll-continuous futures return from the open of d+1 to the open of d+1+h, in the
contract the ledger would hold (nearest expiry with more than 3 sessions left).
Statistics: pooled slope of the forward return on the z-scored reading, both indices
stacked, standard error from the per-date score (so NIFTY/BANKNIFTY co-movement is not
counted twice) with Newey-West over h lags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indices as I
from .ledger import Market

PERIODS = {"discovery": ("2019-01-01", "2022-12-31"), "validation": ("2023-01-01", "2024-06-30"),
           "final": ("2024-07-01", "2026-12-31")}


def continuous(F: pd.DataFrame) -> pd.DataFrame:
    """Per (date, symbol): open-to-next-open and close-to-close returns of the held contract."""
    M = Market(F)
    rows = []
    for s in sorted(F["symbol"].unique()):
        cal = M.cal
        for i in range(len(cal) - 1):
            d, n = cal[i], cal[i + 1]
            e = M.active.get((d, s))
            if e is None:
                continue
            o0, o1 = M.px(d, s, e, "open"), M.px(n, s, e, "open")
            c0, c1 = M.px(d, s, e), M.px(n, s, e)
            rows.append((d, s, e, o1 / o0 - 1, c1 / c0 - 1, c0))
    R = pd.DataFrame(rows, columns=["date", "symbol", "contract", "r_oo", "r_cc_next", "close"])
    return R


def zscore(s: pd.Series, win=250, minp=120) -> pd.Series:
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / sd


def nw_var(x: np.ndarray, lags: int) -> float:
    x = x - x.mean()
    n = len(x)
    v = x @ x / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * (x[L:] @ x[:-L]) / n
    return v / n


def pooled_t(df: pd.DataFrame, x: str, y: str, lags: int):
    d = df[["date", x, y]].dropna()
    if len(d) < 60:
        return np.nan, np.nan, len(d)
    xv = d[x] - d[x].mean()
    yv = d[y] - d[y].mean()
    b = (xv * yv).sum() / (xv * xv).sum()
    e = yv - b * xv
    score = (xv * e).groupby(d["date"]).sum().to_numpy()
    T = len(score)
    se = np.sqrt(nw_var(score, lags) * T * T) / (xv * xv).sum()
    return b, b / se, len(d)


def panel() -> pd.DataFrame:
    P = pd.read_parquet(I.PANEL)
    F = pd.read_parquet(I.FUT_PANEL)
    R = continuous(F)
    X = R.merge(P, on=["date", "symbol"], how="left").sort_values(["symbol", "date"])
    g = X.groupby("symbol")
    for h in (1, 5, 10):
        # forward from open d+1 to open d+1+h: r_oo of sessions d+1 .. d+h
        X[f"fwd{h}"] = g["r_oo"].transform(lambda r: np.exp(np.log1p(r).rolling(h).sum().shift(-h)) - 1)
    X["ret_cc"] = g["close"].pct_change()   # same-contract only approx at rolls; used for trend/vol
    X["rv20"] = g["ret_cc"].transform(lambda r: r.rolling(20).std() * np.sqrt(252))
    X["rv_fwd5"] = g["r_cc_next"].transform(lambda r: (r.rolling(5).std().shift(-4)) * np.sqrt(252))
    X["mom60"] = g["close"].transform(lambda c: np.log(c / c.shift(60)))
    X["basis_c"] = X["basis"].where(X["fut_dte"] >= 5)
    # participants (market-wide index positions, same for both indices)
    if I.PART.exists():
        pf = I.participant_features(pd.read_parquet(I.PART)).reset_index()
        X = X.merge(pf, on="date", how="left")
    # India VIX
    try:
        v = I.yahoo("VIX").rename("vix") / 100
        X = X.merge(v.rename_axis("date").reset_index(), on="date", how="left")
    except Exception:
        X["vix"] = np.nan
    X["vrp"] = X["iv_30"] - X["rv20"]
    X = X.sort_values(["symbol", "date"]).reset_index(drop=True)
    return X


READINGS = {  # name: (column, registered sign)
    "H1 trend 60d": ("mom60", +1), "H1b Pragyam trend": ("prg_T", +1), "H4 PCR OI": ("pcr_oi", +1),
    "H5a FII fut long": ("fii_fut_long", +1), "H5b Client fut long": ("client_fut_long", -1),
    "H6a FII opt bias": ("fii_opt_bias", +1), "H6b Client opt bias": ("client_opt_bias", -1),
    "H8 hedge flow": ("hedge_flow", +1), "H9a VRP": ("vrp", +1), "H9b VIX": ("vix", +1),
    "H10 basis": ("basis_c", 0), "H11 skew": ("skew", +1), "H12 breadth": ("breadth", +1),
}


def add_pragyam(X: pd.DataFrame) -> pd.DataFrame:
    from . import pragyam as PG
    import yfinance as yf
    out = []
    for s in ("NIFTY", "BANKNIFTY"):
        h = yf.Ticker(I.YAHOO[s]).history(start="2014-01-01", auto_adjust=False)
        h.index = pd.to_datetime(h.index.date)
        h = h[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
        h = h[(h["high"] > 0) & ~h.index.duplicated()]
        R = PG.run(h)
        out.append(pd.DataFrame({"date": R.index, "symbol": s, "prg_T": R["T"].values, "prg_stopL": (R["stopL"] / h["close"].reindex(R.index)).values,
                                 "prg_stopS": (R["stopS"] / h["close"].reindex(R.index)).values, "prg_reg": R["reg"].values}))
    return X.merge(pd.concat(out), on=["date", "symbol"], how="left")


def add_breadth(X: pd.DataFrame) -> pd.DataFrame:
    """Share of stock futures in long build-up minus short build-up over 5 sessions."""
    from .config import DATA_DIR
    parts = []
    for f in ("futures_ohlc_legacy.parquet", "futures_ohlc.parquet"):
        p = DATA_DIR / "panels" / f
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["date", "symbol", "expiry", "close", "oi"]))
    S = pd.concat(parts, ignore_index=True)
    S["date"] = pd.to_datetime(S["date"])
    S["expiry"] = pd.to_datetime(S["expiry"])
    S = S[S["expiry"] >= S["date"]]
    near = S.sort_values("expiry").groupby(["date", "symbol"]).first()["close"]
    oi = S.groupby(["date", "symbol"])["oi"].sum()
    W = pd.DataFrame({"c": near, "oi": oi}).reset_index().sort_values(["symbol", "date"])
    g = W.groupby("symbol")
    W["dp"] = g["c"].transform(lambda c: np.log(c / c.shift(5)))
    W["doi"] = g["oi"].transform(lambda c: np.log(c / c.shift(5)))
    W = W[np.abs(W["dp"]) < 0.3]                                  # corporate-action days out
    lb = (W["dp"] > 0) & (W["doi"] > 0)
    sb = (W["dp"] < 0) & (W["doi"] > 0)
    b = (lb.astype(float) - sb.astype(float)).groupby(W["date"]).mean().rename("breadth")
    return X.merge(b.reset_index(), on="date", how="left")


def run() -> dict:
    X = add_breadth(add_pragyam(panel()))
    for name, (col, _) in READINGS.items():
        X[f"z_{col}"] = X.groupby("symbol")[col].transform(zscore)
    res = []
    for name, (col, sign) in READINGS.items():
        for h in (1, 5, 10):
            for per, (a, b) in {**PERIODS, "all": ("2019-01-01", "2026-12-31")}.items():
                d = X[(X["date"] >= a) & (X["date"] <= b)]
                beta, t, n = pooled_t(d, f"z_{col}", f"fwd{h}", lags=max(h, 1))
                res.append(dict(reading=name, h=h, period=per, beta_bp=beta * 1e4, t=t, n=n, sign=sign))
    T = pd.DataFrame(res)
    return {"X": X, "tests": T}


def verdicts(T: pd.DataFrame, h=5) -> pd.DataFrame:
    t = T[T["h"] == h].pivot_table(index="reading", columns="period", values="t")
    sign = T.drop_duplicates("reading").set_index("reading")["sign"]
    out = t[["discovery", "validation", "final", "all"]].copy()
    s = sign.reindex(out.index).replace(0, np.nan)
    signed = out.mul(s, axis=0)
    out["accepted"] = (signed[["discovery", "validation", "final"]] >= 1.0).all(axis=1) & (signed["all"] >= 2.0)
    return out.round(2)


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 200)
    print(verdicts(r["tests"]))
    print(verdicts(r["tests"], 1))
    print(verdicts(r["tests"], 10))
    r["X"].to_parquet(I.PANEL.parent / "index_research.parquet", index=False)
