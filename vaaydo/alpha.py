"""
Cross-sectional signal research (protocol: docs/PREREGISTRATION.md).

`panel()` joins, per stock and session (all known at that close):
    F&O features      vaaydo.features (options surface, OI flows, basis, rollover, walls)
    equity features   delivery %, VWAP gap, close location, volume surprise
    price features    reversal / momentum / 52-week high, Pragyam's trend, surprise,
                      risk regime and risk momentum, idiosyncratic vol, beta
and outcomes: next-open -> close h sessions later (h = 1, 5, 10, 20), minus the day's
cross-sectional mean, in units of the stock's trailing 63-day vol.

`ic_table()` gives each feature's daily Spearman IC, averaged over days, with a
Newey-West t (lags = h): dates are the independent unit, never stock-days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import equity, history

DISCOVERY_END = pd.Timestamp("2025-12-31")
HORIZONS = (1, 5, 10, 20)
INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI"}

# feature -> (description, family)
FEATURES = {
    # options
    "atm_iv": ("ATM implied vol", "options"),
    "iv_rv": ("ATM IV / 20d realised vol", "options"),
    "d_iv5": ("5-day change in ATM IV", "options"),
    "cpiv": ("call IV - put IV, same strikes (Cremers-Weinbaum)", "options"),
    "d_cpiv5": ("5-day change in call-put IV spread", "options"),
    "skew": ("5% OTM put IV - ATM IV (Xing-Zhang-Zhao)", "options"),
    "cskew": ("5% OTM call IV - ATM IV", "options"),
    "term": ("front / next ATM IV", "options"),
    "pcr_vol": ("put/call contracts traded", "options"),
    "pcr_oi": ("put/call open interest", "options"),
    "d_pcr_oi5": ("5-day change in put/call OI", "options"),
    "put_oi_flow": ("put OI added today, % of OI", "options"),
    "call_oi_flow": ("call OI added today, % of OI", "options"),
    "net_opt_flow": ("put OI flow - call OI flow", "options"),
    "os_ratio": ("log option notional / futures notional (Johnson-So)", "options"),
    "max_pain_d": ("log max-pain strike / forward", "options"),
    "wall_put_d": ("log put-OI wall / forward", "options"),
    "wall_call_d": ("log call-OI wall / forward", "options"),
    # futures
    "basis": ("annualised futures carry minus risk-free", "futures"),
    "d_basis5": ("5-day change in basis", "futures"),
    "fut_oi_flow": ("futures OI change today, %", "futures"),
    "oi5": ("futures OI change over 5 days, %", "futures"),
    "buildup": ("sign(5d move) x 5d OI change (Order Flow build-up)", "futures"),
    "roll_share": ("share of futures OI beyond the near month", "futures"),
    "turn_z": ("log futures turnover vs its 20-day mean", "futures"),
    # equity
    "deliv_pct": ("delivery % of traded quantity", "equity"),
    "deliv_z": ("delivery % vs its 20-day mean", "equity"),
    "deliv_ret": ("1-day return x delivery surprise", "equity"),
    "vwap_gap": ("log close / VWAP", "equity"),
    "clv": ("close location in the day's range", "equity"),
    "vol_z": ("log volume vs 20-day mean", "equity"),
    # price (Pragyam and classic)
    "rev1": ("1-day return (reversal)", "price"),
    "rev5": ("5-day return", "price"),
    "mom21": ("21-day return", "price"),
    "mom126": ("126-day return skipping 21", "price"),
    "hi52": ("close / 52-week high", "price"),
    "prg_T": ("Pragyam trend (sigma)", "price"),
    "prg_u": ("Pragyam surprise: 10-bar move / forecast", "price"),
    "prg_prk": ("Pragyam risk rank in 500 bars", "price"),
    "prg_slope": ("Pragyam risk momentum", "price"),
    "ivol": ("idiosyncratic vol, 63d", "price"),
    "beta": ("beta to NIFTY, 126d", "price"),
}


def _wide(h: pd.DataFrame, col: str) -> pd.DataFrame:
    return h.pivot(index="date", columns="symbol", values=col)


def price_block(h: pd.DataFrame, run_pragyam: bool = True) -> tuple[pd.DataFrame, dict]:
    """Price features and outcomes from adjusted daily OHLC (long format out)."""
    C, O, H, L = (_wide(h, c) for c in ("close", "open", "high", "low"))
    lc = np.log(C)
    r = lc.diff()
    sd63 = r.rolling(63, min_periods=40).std()
    mkt = r["NIFTY"] if "NIFTY" in r else r.mean(axis=1)
    f = {}
    f["rev1"] = r
    f["rev5"] = lc - lc.shift(5)
    f["mom21"] = lc - lc.shift(21)
    f["mom126"] = lc.shift(21) - lc.shift(126)
    f["hi52"] = C / C.rolling(252, min_periods=200).max()
    cov = r.rolling(126, min_periods=80).cov(mkt)
    beta = cov.div(mkt.rolling(126, min_periods=80).var(), axis=0)
    f["beta"] = beta
    f["ivol"] = (r - beta.mul(mkt, axis=0)).rolling(63, min_periods=40).std()
    f["rv20"] = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    out = {}
    lo = np.log(O)
    for hzn in HORIZONS:
        fwd = lc.shift(-hzn) - lo.shift(-1)                      # next open -> close h later
        x = fwd.sub(fwd.mean(axis=1), axis=0)                    # market-neutral
        out[f"y{hzn}"] = x / (sd63 * np.sqrt(hzn))
        out[f"raw{hzn}"] = fwd                                   # for P&L, not ranking
    if run_pragyam:
        from . import pragyam as PG
        pr = {k: {} for k in ("prg_T", "prg_u", "prg_prk", "prg_slope", "prg_sig", "prg_stopL", "prg_stopS",
                              "prg_upper", "prg_lower", "prg_reg", "prg_E")}
        for s, d in h.groupby("symbol"):
            d = d.set_index("date")[["open", "high", "low", "close"]]
            if len(d) < 300:
                continue
            R = PG.run(d)
            for k, src in (("prg_T", "T"), ("prg_u", "u"), ("prg_prk", "prk"), ("prg_slope", "slope"), ("prg_sig", "sig"),
                           ("prg_stopL", "stopL"), ("prg_stopS", "stopS"), ("prg_upper", "upper"), ("prg_lower", "lower"),
                           ("prg_reg", "reg"), ("prg_E", "E")):
                pr[k][s] = R[src]
        for k, v in pr.items():
            f[k] = pd.DataFrame(v)
    long = pd.concat({k: v.stack() for k, v in {**f, **out}.items()}, axis=1)
    long.index.names = ["date", "symbol"]
    return long.reset_index(), dict(sd63=sd63)


def equity_block(days=None) -> pd.DataFrame:
    e = equity.load(days)
    e = e.sort_values(["symbol", "date"])
    g = e.groupby("symbol")
    e["deliv_z"] = e["deliv_pct"] - g["deliv_pct"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    e["vol_z"] = np.log(e["qty"] / g["qty"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1)))
    e["vwap_gap"] = np.log(e["close"] / e["vwap"])
    rng = (e["high"] - e["low"]).where(lambda x: x > 0)
    e["clv"] = ((e["close"] - e["low"]) / rng - 0.5) * 2
    ret1 = np.log(e["close"] / e["prev_close"])
    e["deliv_ret"] = ret1 * e["deliv_z"]
    return e[["date", "symbol", "deliv_pct", "deliv_z", "deliv_ret", "vwap_gap", "clv", "vol_z", "turnover_lacs"]]


def fno_block() -> pd.DataFrame:
    p = pd.read_parquet(DATA_DIR / "panels" / "features.parquet").sort_values(["symbol", "date"])
    g = p.groupby("symbol")
    p["d_iv5"] = p["atm_iv"] - g["atm_iv"].shift(5)
    p["d_cpiv5"] = p["cpiv"] - g["cpiv"].shift(5)
    p["d_pcr_oi5"] = p["pcr_oi"] - g["pcr_oi"].shift(5)
    p["d_basis5"] = p["basis"] - g["basis"].shift(5)
    p["oi5"] = p["fut_oi"] / g["fut_oi"].shift(5) - 1
    p["net_opt_flow"] = p["put_oi_flow"] - p["call_oi_flow"]
    p["os_ratio"] = np.log(p["os_ratio"].where(p["os_ratio"] > 0))
    p["turn_z"] = np.log(p["fut_turnover"] / g["fut_turnover"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1)))
    return p


def panel(run_pragyam: bool = True) -> pd.DataFrame:
    fno = fno_block()
    syms = sorted(set(fno["symbol"]) | {"NIFTY"})
    h = history.load(syms)
    px, _ = price_block(h, run_pragyam)
    eq = equity_block()
    P = fno.merge(px, on=["date", "symbol"], how="left").merge(eq, on=["date", "symbol"], how="left")
    P["iv_rv"] = P["atm_iv"] / P["rv20"]
    # Order Flow build-up as a number: direction of the 5-day move, scaled by OI opened with it
    P["buildup"] = np.sign(P["rev5"]) * P["oi5"]
    P = P[~P["symbol"].isin(INDICES)]
    return P


def _nw_t(x: np.ndarray, lags: int) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan
    e = x - x.mean()
    v = (e * e).sum()
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] * e[:-L]).sum()
    return float(x.mean() / np.sqrt(v / n / n))


def daily_ic(P: pd.DataFrame, feat: str, h: int, min_names: int = 50) -> pd.Series:
    d = P[["date", feat, f"y{h}"]].dropna()
    d = d[d.groupby("date")[feat].transform("size") >= min_names]
    rk = d.groupby("date")[[feat, f"y{h}"]].rank()
    rk["date"] = d["date"]
    return rk.groupby("date").apply(lambda g: g[feat].corr(g[f"y{h}"]))


def ic_table(P: pd.DataFrame, feats=None, horizons=HORIZONS, period: str = "discovery") -> pd.DataFrame:
    if period == "discovery":
        P = P[P["date"] <= DISCOVERY_END]
    elif period == "holdout":
        P = P[P["date"] > DISCOVERY_END]
    rows = []
    for f in feats or FEATURES:
        if f not in P:
            continue
        for h in horizons:
            ic = daily_ic(P, f, h)
            rows.append(dict(feature=f, family=FEATURES.get(f, ("", "derived"))[1], h=h, ic=ic.mean(),
                             t=_nw_t(ic.to_numpy(), h), days=len(ic), pos=(ic > 0).mean()))
    return pd.DataFrame(rows)


def attach_futures_outcomes(P: pd.DataFrame, futret: pd.DataFrame) -> pd.DataFrame:
    """Replace outcomes with FUTURES returns (next open -> close h, contract chosen to avoid
    expiry), market-neutral and vol-scaled: what a futures position actually earns."""
    P = P.merge(futret, on=["date", "symbol"], how="left")
    h = history.load()
    C = h.pivot(index="date", columns="symbol", values="close")
    sd63 = np.log(C).diff().rolling(63, min_periods=40).std().stack().rename("sd63").reset_index()
    P = P.merge(sd63, on=["date", "symbol"], how="left")
    for hz in (1, 5, 10):
        x = P[f"fraw{hz}"] - P.groupby("date")[f"fraw{hz}"].transform("mean")
        P[f"y{hz}"] = x / (P["sd63"] * np.sqrt(hz))
        P[f"raw{hz}"] = P[f"fraw{hz}"]
    return P
