"""
The index desk's data: one row per index per session, NIFTY and BANKNIFTY, Jan 2019 on.

Everything is read from end-of-day files NSE publishes:

    F&O bhavcopy          index futures (all three months) and the full option chain
    participant OI        index futures and options, long and short, for FII / DII / Pro / Client
    India VIX, spot       Yahoo (^INDIAVIX, ^NSEI, ^NSEBANK); spot also from the bhavcopy since Jul 2024

Futures
    fut_*          near-month open/close, total futures OI and its change
    basis          annualised log(near future / spot) minus the risk-free rate (cost of carry premium)
    rollover       share of futures OI already in the next and far months
    roll_cost      annualised log(next / near) spread

Option chain (OI in units, Black-76 on each expiry's forward)
    pcr_oi, pcr_vol            all expiries
    pcr_oi_front               nearest expiry
    iv_front, iv_30            ATM IV of the nearest expiry with 2+ days, and interpolated to 30 days
    term                       iv_front / iv_30
    skew                       IV at 3% below the forward minus IV at 3% above (front)
    max_pain_d                 log(max pain / forward), nearest expiry
    wall_call_d, wall_put_d    log(strike / forward) of the biggest call OI above and put OI below
    gex_dealer                 Σ γ·OI·S²·1% with dealers long calls, short puts, as % of S·futures OI
    gex_seller                 the same with writers short both (always negative): gamma sold
    flip_d                     log(gamma flip / spot): where dealer-model GEX changes sign
    hedge_flow                 writers' delta hedge of the day's OI change, as % of futures OI notional
    straddle                   ATM straddle of the nearest expiry / spot (the move priced to that expiry)
"""
from __future__ import annotations

from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import pricing
from .config import DATA_DIR, RISK_FREE

INDICES = ("NIFTY", "BANKNIFTY")
YAHOO = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "VIX": "^INDIAVIX"}
PANEL = DATA_DIR / "panels" / "index_panel.parquet"
PART = DATA_DIR / "panels" / "participant_oi.parquet"
PART_URL = "https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv"
HEADERS = {"User-Agent": "vaaydo/6 (end-of-day research)"}
GEX_DTE = 60          # options further out carry little gamma; leaving them out avoids stale far strikes


def _gamma(F, K, T, sigma):
    sT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / sT
    return np.exp(-RISK_FREE * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (F * sT)


def _interp_iv(k, iv, kq):
    o = np.argsort(k)
    return np.interp(kq, k[o], iv[o])


def _iv_at(g, target, tol):
    if len(g) < 2:
        return np.nan
    k, iv = g["k"].to_numpy(), g["iv"].to_numpy()
    if k.min() - tol > target or k.max() + tol < target:
        return np.nan
    return float(_interp_iv(k, iv, target))


def _load(d: date) -> pd.DataFrame:
    from .data import bhavcopy, legacy
    try:
        df = bhavcopy.load_day(d)
    except Exception:
        df = legacy.load_day(d)
    df = df[df["symbol"].astype(str).isin(INDICES) & df["inst"].astype(str).isin(["IDF", "IDO"])].copy()
    df["inst"] = df["inst"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["otype"] = df["otype"].astype(str)
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def day(d: date, spot_hint: dict | None = None) -> list[dict]:
    asof = pd.Timestamp(d)
    df = _load(d)
    out = []
    for sym, g in df.groupby("symbol"):
        r = {"date": asof, "symbol": sym}
        fut = g[(g["inst"] == "IDF") & (g["expiry"] >= asof)].sort_values("expiry")
        if fut.empty:
            continue
        near = fut.iloc[0]
        S = near["spot"] if "spot" in g and np.isfinite(near.get("spot", np.nan)) and near["spot"] > 0 else np.nan
        if not np.isfinite(S) and spot_hint:
            S = spot_hint.get(sym, np.nan)
        Tn = max((near["expiry"] - asof).days, 1) / 365
        if not np.isfinite(S):
            S = near["close"] * np.exp(-RISK_FREE * Tn)
        r.update(spot=float(S), fut_open=float(near["open"]), fut_close=float(near["close"]),
                 fut_expiry=near["expiry"], lot=int(near["lot"]) if near["lot"] else 0,
                 fut_oi=float(fut["oi"].sum()), fut_oi_chg=float(fut["oi_chg"].sum()),
                 fut_contracts=float(fut["contracts"].sum()))
        r["basis"] = np.log(near["close"] / S) / Tn - RISK_FREE
        r["rollover"] = 1 - near["oi"] / fut["oi"].sum() if fut["oi"].sum() > 0 else np.nan
        if len(fut) > 1:
            nx = fut.iloc[1]
            r["roll_cost"] = np.log(nx["close"] / near["close"]) / max((nx["expiry"] - near["expiry"]).days, 1) * 365
        r["fut_dte"] = (near["expiry"] - asof).days

        o = g[(g["inst"] == "IDO") & (g["expiry"] > asof)].copy()      # an expiring chain is dead at the close
        if o.empty:
            out.append(r)
            continue
        ce, pe = o["otype"] == "CE", o["otype"] == "PE"
        r["pcr_oi"] = o.loc[pe, "oi"].sum() / max(o.loc[ce, "oi"].sum(), 1)
        r["pcr_vol"] = o.loc[pe, "contracts"].sum() / max(o.loc[ce, "contracts"].sum(), 1)
        o["dte"] = (o["expiry"] - asof).dt.days
        o["T"] = np.maximum(o["dte"], 0.25) / 365
        # forward per expiry: listed future, else put-call parity at the busiest strike
        fw = {}
        for e, x in o.groupby("expiry"):
            f = fut[fut["expiry"] == e]
            if len(f):
                fw[e] = float(f["close"].iloc[0])
                continue
            c = x[x["otype"] == "CE"].set_index("strike")
            p = x[x["otype"] == "PE"].set_index("strike")
            j = c[["close", "contracts"]].join(p[["close", "contracts"]], lsuffix="_c", rsuffix="_p", how="inner")
            j = j[(j["contracts_c"] > 0) & (j["contracts_p"] > 0)]
            if len(j):
                k = (j["contracts_c"] + j["contracts_p"]).idxmax()
                T = x["T"].iloc[0]
                fw[e] = float(k + np.exp(RISK_FREE * T) * (j.loc[k, "close_c"] - j.loc[k, "close_p"]))
            else:
                fw[e] = S * np.exp(RISK_FREE * x["T"].iloc[0])
        o["F"] = o["expiry"].map(fw)
        o["k"] = np.log(o["strike"] / o["F"])
        tr = o[(o["contracts"] > 0) & (o["close"] >= 0.05)].copy()
        tr["iv"] = pricing.implied_vol(tr["close"].to_numpy(), tr["F"].to_numpy(), tr["strike"].to_numpy(),
                                       tr["T"].to_numpy(), (tr["otype"] == "CE").to_numpy())
        tr = tr[tr["iv"].notna()]
        otm = tr[((tr["otype"] == "CE") & (tr["k"] >= 0)) | ((tr["otype"] == "PE") & (tr["k"] <= 0))]
        smile = {e: x for e, x in otm.groupby("expiry") if len(x) >= 4}
        atm = {e: _iv_at(x, 0.0, 0.01) for e, x in smile.items()}
        atm = {e: v for e, v in atm.items() if np.isfinite(v)}
        exps = sorted(atm)
        front = next((e for e in exps if (e - asof).days >= 2), None)
        if front is not None:
            r["iv_front"] = atm[front]
            r["front_dte"] = (front - asof).days
            r["skew"] = _iv_at(smile[front], -0.03, 0.005) - _iv_at(smile[front], 0.03, 0.005)
        # 30-day ATM IV by interpolating total variance
        tv = [((e - asof).days / 365, atm[e]) for e in exps if (e - asof).days >= 2]
        if len(tv) >= 2:
            Ts = np.array([t for t, _ in tv])
            w = np.array([v * v * t for t, v in tv])
            t30 = 30 / 365
            if Ts.min() <= t30 <= Ts.max():
                r["iv_30"] = float(np.sqrt(np.interp(t30, Ts, w) / t30))
            elif t30 < Ts.min():
                r["iv_30"] = float(tv[0][1])
        if np.isfinite(r.get("iv_front", np.nan)) and np.isfinite(r.get("iv_30", np.nan)):
            r["term"] = r["iv_front"] / r["iv_30"]

        # nearest live expiry: max pain, walls, straddle, PCR
        e0 = o["expiry"].min()
        f0 = o[o["expiry"] == e0]
        F0 = fw[e0]
        r["pcr_oi_front"] = f0.loc[f0["otype"] == "PE", "oi"].sum() / max(f0.loc[f0["otype"] == "CE", "oi"].sum(), 1)
        ks = np.sort(f0["strike"].unique())
        c0, p0 = f0[f0["otype"] == "CE"], f0[f0["otype"] == "PE"]
        if len(ks) >= 3:
            pay = (np.maximum(ks[:, None] - c0["strike"].to_numpy()[None, :], 0) @ c0["oi"].to_numpy(float)
                   + np.maximum(p0["strike"].to_numpy()[None, :] - ks[:, None], 0) @ p0["oi"].to_numpy(float))
            r["max_pain"] = float(ks[int(np.argmin(pay))])
            r["max_pain_d"] = np.log(r["max_pain"] / F0)
        cu, pd_ = c0[c0["strike"] >= F0], p0[p0["strike"] <= F0]
        if len(cu) and cu["oi"].max() > 0:
            r["wall_call"] = float(cu.loc[cu["oi"].idxmax(), "strike"])
            r["wall_call_d"] = np.log(r["wall_call"] / F0)
        if len(pd_) and pd_["oi"].max() > 0:
            r["wall_put"] = float(pd_.loc[pd_["oi"].idxmax(), "strike"])
            r["wall_put_d"] = np.log(r["wall_put"] / F0)
        kk = ks[np.argmin(np.abs(ks - F0))] if len(ks) else np.nan
        cs = c0[c0["strike"] == kk]["close"]
        ps = p0[p0["strike"] == kk]["close"]
        if len(cs) and len(ps):
            r["straddle"] = float((cs.iloc[0] + ps.iloc[0]) / S)
        r["front_expiry"] = e0

        # gamma, delta for every live option within GEX_DTE: IV from its expiry's smile
        gx = o[(o["dte"] <= GEX_DTE) & (o["oi"] > 0)].copy()
        if len(gx) and smile:
            iv = np.full(len(gx), np.nan)
            for e, idx in gx.groupby("expiry").groups.items():
                sm = smile.get(e)
                if sm is not None:
                    kq = gx.loc[idx, "k"].to_numpy()
                    kq = np.clip(kq, sm["k"].min(), sm["k"].max())
                    iv[gx.index.get_indexer(idx)] = _interp_iv(sm["k"].to_numpy(), sm["iv"].to_numpy(), kq)
            fallback = r.get("iv_front", np.nan)
            gx["iv"] = np.where(np.isfinite(iv), iv, fallback)
            gx = gx[np.isfinite(gx["iv"])]
            is_c = (gx["otype"] == "CE").to_numpy()
            sign = np.where(is_c, 1.0, -1.0)
            Fv, Kv, Tv, sv = gx["F"].to_numpy(), gx["strike"].to_numpy(), gx["T"].to_numpy(), gx["iv"].to_numpy()
            oi = gx["oi"].to_numpy(float)
            base = S * fut["oi"].sum()                               # futures OI notional, in index-point units
            gam = _gamma(Fv, Kv, Tv, sv)
            unit = gam * oi * S * S * 0.01                            # Δ(delta notional) per 1% move
            r["gex_dealer"] = float((sign * unit).sum() / base)
            r["gex_seller"] = float(-unit.sum() / base)
            # gamma flip: dealer-model GEX across spot levels ±8%
            grid = S * np.exp(np.linspace(-0.08, 0.08, 81))
            Fg = Fv[None, :] * (grid[:, None] / S)
            gg = _gamma(Fg, Kv[None, :], Tv[None, :], sv[None, :])
            prof = (gg * oi[None, :] * grid[:, None] ** 2 * 0.01 * sign[None, :]).sum(axis=1)
            sc = np.where(np.diff(np.sign(prof)) != 0)[0]
            if len(sc):
                j = sc[np.argmin(np.abs(np.log(grid[sc] / S)))]
                x0, x1, y0, y1 = grid[j], grid[j + 1], prof[j], prof[j + 1]
                flip = x0 - y0 * (x1 - x0) / (y1 - y0)
                r["flip"] = float(flip)
                r["flip_d"] = float(np.log(flip / S))
            # writers' hedge of today's OI change: short new calls -> buy delta, short new puts -> sell delta
            dl = pricing.delta(Fv, Kv, Tv, sv, is_c)
            r["hedge_flow"] = float((gx["oi_chg"].to_numpy(float) * dl * S).sum() / base)
        out.append(r)
    return out


def _safe(args):
    d, hint = args
    try:
        return day(d, hint)
    except Exception as e:                       # one bad file must not kill the panel
        return [{"date": pd.Timestamp(d), "symbol": "ERR", "err": str(e)[:200]}]


def vix_nse(days, session=None) -> pd.Series:
    """India VIX closes for the given sessions from NSE's daily index-close file (the same file the
    breadth panel reads). A file whose own date is not the session asked for is skipped."""
    import io
    import requests
    from .breadth import IND_CLOSE, _get
    s = session or requests.Session()
    out = {}
    for d in days:
        d = pd.Timestamp(d)
        r = _get(IND_CLOSE.format(d=d), s)
        if r is None or not r.text.lstrip().lower().startswith("index name"):
            continue
        x = pd.read_csv(io.StringIO(r.text))
        x.columns = [c.strip() for c in x.columns]
        row = x[x["Index Name"].astype(str).str.strip().str.lower() == "india vix"]
        if row.empty or pd.to_datetime(row["Index Date"].iloc[0], dayfirst=True, errors="coerce") != d:
            continue
        v = pd.to_numeric(str(row["Closing Index Value"].iloc[0]).replace(",", ""), errors="coerce")
        if np.isfinite(v) and v > 0:
            out[d] = float(v)
    return pd.Series(out, dtype=float).sort_index()


def yahoo(sym: str, start="2018-01-01") -> pd.Series:
    """Research only (index_research, the panel build's spot hints): the live desk and the nightly
    refresh read NSE's own files."""
    import yfinance as yf
    h = yf.Ticker(YAHOO[sym]).history(start=start, auto_adjust=False)
    s = h["Close"]
    s.index = pd.to_datetime(s.index.date)
    return s[~s.index.duplicated()]


def build(days=None, processes: int = 4) -> pd.DataFrame:
    from .data import bhavcopy, legacy
    if days is None:
        days = sorted(set(legacy.available_days()) | set(bhavcopy.available_days()))
    spots = {s: yahoo(s) for s in INDICES}
    hints = [{s: float(spots[s].get(pd.Timestamp(d), np.nan)) for s in INDICES} for d in days]
    with Pool(processes) as p:
        rows = [r for rs in p.imap(_safe, zip(days, hints), chunksize=8) for r in rs]
    P = pd.DataFrame(rows)
    bad = P[P["symbol"] == "ERR"]
    if len(bad):
        print(f"{len(bad)} days failed, e.g. {bad.iloc[0].to_dict()}")
    P = P[P["symbol"] != "ERR"].drop(columns=["err"], errors="ignore").sort_values(["symbol", "date"])
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(PANEL, index=False)
    return P


# ── participant-wise OI ─────────────────────────────────────────────────────
PART_COLS = ["fut_idx_long", "fut_idx_short", "fut_stk_long", "fut_stk_short", "opt_idx_call_long",
             "opt_idx_put_long", "opt_idx_call_short", "opt_idx_put_short", "opt_stk_call_long",
             "opt_stk_put_long", "opt_stk_call_short", "opt_stk_put_short", "total_long", "total_short"]


def participant_day(d: date, session=None) -> pd.DataFrame | None:
    import io
    import requests
    s = session or requests.Session()
    r = s.get(PART_URL.format(d=d), headers=HEADERS, timeout=30)
    if r.status_code != 200 or len(r.content) < 200:
        return None
    x = pd.read_csv(io.StringIO(r.text), skiprows=1)
    x.columns = ["client"] + PART_COLS[:len(x.columns) - 1]
    x["client"] = x["client"].astype(str).str.strip()
    x = x[x["client"].isin(["Client", "DII", "FII", "Pro"])].copy()
    for c in PART_COLS:
        x[c] = pd.to_numeric(x[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    x["date"] = pd.Timestamp(d)
    return x


def participants(days=None, progress=None, workers: int = 4) -> pd.DataFrame:
    import time
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from .data import bhavcopy, legacy
    if days is None:
        days = sorted(set(legacy.available_days()) | set(bhavcopy.available_days()))
    have = pd.read_parquet(PART) if PART.exists() else pd.DataFrame(columns=["date"])
    got = set(pd.to_datetime(have["date"]).dt.date) if len(have) else set()
    todo = [d for d in days if d not in got]

    def one(d):
        s = requests.Session()
        for attempt in range(3):
            try:
                return participant_day(d, s)
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    new = []
    with ThreadPoolExecutor(workers) as ex:
        for i, x in enumerate(ex.map(one, todo)):
            if x is not None:
                new.append(x)
            if progress and i % 200 == 0:
                progress(f"participants {i}/{len(todo)}")
    out = pd.concat([have, *new], ignore_index=True) if new else have
    if len(out):
        out["date"] = pd.to_datetime(out["date"])
        out = out.drop_duplicates(["date", "client"]).sort_values(["date", "client"])
        PART.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(PART, index=False)
    return out


def participant_features(Pt: pd.DataFrame) -> pd.DataFrame:
    """Per date: each group's index futures net long share and index options net delta-ish position."""
    w = Pt.pivot_table(index="date", columns="client",
                       values=["fut_idx_long", "fut_idx_short", "opt_idx_call_long", "opt_idx_call_short",
                               "opt_idx_put_long", "opt_idx_put_short"])
    f = pd.DataFrame(index=w.index)
    for g in ("FII", "Client", "Pro", "DII"):
        L, S_ = w[("fut_idx_long", g)], w[("fut_idx_short", g)]
        f[f"{g.lower()}_fut_long"] = L / (L + S_)
        cl, cs = w[("opt_idx_call_long", g)], w[("opt_idx_call_short", g)]
        pl, ps = w[("opt_idx_put_long", g)], w[("opt_idx_put_short", g)]
        # bullish option stance: long calls + short puts minus short calls + long puts, over gross
        f[f"{g.lower()}_opt_bias"] = ((cl - cs) - (pl - ps)) / (cl + cs + pl + ps)
    return f


FUT_PANEL = DATA_DIR / "panels" / "index_futures.parquet"


def _fut_day(d):
    df = _load(d)
    f = df[df["inst"] == "IDF"][["symbol", "expiry", "open", "high", "low", "close", "oi", "contracts", "lot"]].copy()
    f["date"] = pd.Timestamp(d)
    return f


def futures_history(days=None, processes: int = 4) -> pd.DataFrame:
    """Every NIFTY / BANKNIFTY futures contract on every session (the index ledger's prices)."""
    from .data import bhavcopy, legacy
    if days is None:
        days = sorted(set(legacy.available_days()) | set(bhavcopy.available_days()))
    with Pool(processes) as p:
        F = pd.concat(p.map(_fut_day, days, chunksize=16), ignore_index=True)
    F["lot"] = F["lot"].where(F["lot"] > 0)
    F.to_parquet(FUT_PANEL, index=False)
    return F
