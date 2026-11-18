"""
NIFTY breadth and regime, recomputed causally from a NIFTY workbook's raw inputs.

Raw inputs (one row per session, 2006 on): NSE market-wide advance/decline ratio, NIFTY close,
NIFTY PE / PB / DY, rates and macro. Derived columns follow the workbook's own formulas:

    adv_share     A/(A+D) = ad / (ad + 1)
    breadth       10-session EMA of adv_share (α = 2/11)
    rel_breadth   mean(breadth, mean of its 2/3/5/8/13/21-session MAs)
    rel_ad        mean(Δad, mean of Δad's 2/3/5-session MAs)
    rsi           Wilder RSI(14);  bands: 20-session MA ± 2 population σ
    osc / count   running count of +1%/−1% days; count = osc − its 50-session mean
    spread90/200  % above the 90-session MA and the 200-session MA lagged 8 sessions
    pe_dev        PE − fair PE; fair PE from an EXPANDING-window multiple regression on the
                  workbook's selected variables, refit monthly (the workbook fits on all
                  2006–2026 data, which is look-ahead), on PE adjusted for NSE's 31 Mar 2021
                  switch to consolidated earnings
    regime        RSI zone + breadth zone + spread90 zone + pe_dev zone, −4 … +4
    signal_str    (RSI−50)/50 + Bollinger z/2 + ⅓·sign(pe_dev)
    btd / stt     below the lower band with RSI < 40 / above the upper band, RSI > 70, breadth > 0.5
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PE_BREAK = pd.Timestamp("2021-03-31")
# variables the workbook selects for fair PE (selector = 1), COUNT excluded (its changelog removes it)
FAIR_PE_X = ["IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO", "US02Y", "US10Y", "US30Y", "NIFTY50_DY",
             "NIFTY50_PB", "AD_RATIO", "rel_ad", "rel_breadth"]
RAW = ["DATE", "AD_RATIO", "NIFTY", "NIFTY50_PE", "NIFTY50_PB", "NIFTY50_DY", "INDIAVIX", "USDINR", "IN10Y",
       "IN02Y", "IN30Y", "INIRYY", "REPO", "CRR", "US02Y", "US10Y", "US30Y", "US_FED"]


def load_workbook(path) -> pd.DataFrame:
    d = pd.read_excel(path, "NIFTY50")
    d = d[[c for c in RAW if c in d.columns]].dropna(subset=["DATE"]).copy()
    d["DATE"] = pd.to_datetime(d["DATE"])
    return d.sort_values("DATE").drop_duplicates("DATE").reset_index(drop=True)


def fix_prices(d: pd.DataFrame, ref: pd.Series | None) -> pd.DataFrame:
    """Replace NIFTY prints that disagree with NSE closes by more than 0.2%."""
    d = d.copy()
    if ref is not None and len(ref):
        r = ref.reindex(d["DATE"]).to_numpy()
        bad = np.isfinite(r) & (np.abs(d["NIFTY"].to_numpy() / r - 1) > 0.002)
        d.loc[bad, "NIFTY"] = r[bad]
        d["price_fixed"] = bad
    return d


def _wilder_rsi(p: pd.Series, n=14) -> pd.Series:
    ch = p.diff()
    g, l = ch.clip(lower=0), (-ch).clip(lower=0)
    ag = g.rolling(n).mean()
    al = l.rolling(n).mean()
    ag_, al_ = ag.copy(), al.copy()
    for i in range(n + 1, len(p)):
        ag_.iat[i] = (ag_.iat[i - 1] * (n - 1) + g.iat[i]) / n
        al_.iat[i] = (al_.iat[i - 1] * (n - 1) + l.iat[i]) / n
    return 100 - 100 / (1 + ag_ / al_)


def _fair_pe(d: pd.DataFrame, min_obs=500) -> pd.Series:
    """Expanding multiple regression, refit at each month end, applied to the next month.
    Needs the rate and macro inputs; without them (NSE-only history) there is no fair PE."""
    if any(c not in d for c in FAIR_PE_X):
        return pd.Series(np.nan, index=d.index)
    X = d[FAIR_PE_X].ffill()
    y = d["pe_adj"]
    out = pd.Series(np.nan, index=d.index)
    month = d["DATE"].dt.to_period("M")
    for m in month.unique():
        train = (month < m) & X.notna().all(axis=1) & y.notna()
        if train.sum() < min_obs:
            continue
        A = np.column_stack([np.ones(train.sum()), X[train].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, y[train].to_numpy(), rcond=None)
        cur = (month == m) & X.notna().all(axis=1)
        if cur.any():
            out[cur] = np.column_stack([np.ones(cur.sum()), X[cur].to_numpy()]) @ beta
    return out


def features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    p = d["NIFTY"]
    d["adv_share"] = d["AD_RATIO"] / (d["AD_RATIO"] + 1)
    d["breadth"] = d["adv_share"].ewm(alpha=2 / 11, adjust=False).mean()
    mas = pd.concat([d["breadth"].rolling(n).mean() for n in (2, 3, 5, 8, 13, 21)], axis=1).mean(axis=1, skipna=False)
    d["rel_breadth"] = (mas + d["breadth"]) / 2
    dad = d["AD_RATIO"].diff()
    d["rel_ad"] = (dad + pd.concat([dad.rolling(n).mean() for n in (2, 3, 5)], axis=1).mean(axis=1, skipna=False)) / 2
    d["rsi"] = _wilder_rsi(p)
    d["ma20"] = p.rolling(20).mean()
    d["sd20"] = p.rolling(20).std(ddof=0)
    d["lower"], d["upper"] = d["ma20"] - 2 * d["sd20"], d["ma20"] + 2 * d["sd20"]
    chg = p.pct_change(fill_method=None) * 100
    step = np.where(chg > 1.0, 1, np.where(chg < -1.0, -1, 0))
    d["osc"] = np.cumsum(step)
    d["count"] = d["osc"] - pd.Series(d["osc"]).rolling(50).mean()
    d["spread90"] = (p / p.rolling(90).mean() - 1) * 100
    ma200 = p.shift(8).rolling(192).mean()
    d["spread200"] = (p / ma200 - 1) * 100
    d["rvol20"] = chg.rolling(20).std() * np.sqrt(252)
    # PE across the 2021 methodology switch: scale the old series to the new basis
    i = d.index[d["DATE"] == PE_BREAK]
    f = 1.0
    if "NIFTY50_PE" not in d:
        d["NIFTY50_PE"] = np.nan
    if len(i) and i[0] > 0:
        j = i[0]
        pe0, pe1 = d.at[j - 1, "NIFTY50_PE"], d.at[j, "NIFTY50_PE"]
        f = (pe1 / pe0) / (p.iat[j] / p.iat[j - 1])
    d["pe_adj"] = np.where(d["DATE"] < PE_BREAK, d["NIFTY50_PE"] * f, d["NIFTY50_PE"])
    d["pe_break_factor"] = f
    d["fair_pe"] = _fair_pe(d)
    d["pe_dev"] = d["pe_adj"] - d["fair_pe"]
    d["ey_gap"] = 100 / d["pe_adj"] - d["IN10Y"].ffill() if "IN10Y" in d else np.nan
    d["in_yc"] = d["IN10Y"].ffill() - d["IN02Y"].ffill() if "IN02Y" in d else np.nan
    r = d["rsi"]
    rz = np.select([r < 30, r < 40, r > 70, r > 60], [-1, -0.5, 1, 0.5], 0)
    bz = np.select([d["breadth"] < 0.35, d["breadth"] > 0.55], [-1, 1], 0)
    sz = np.select([d["spread90"] < -5, d["spread90"] > 5], [-1, 1], 0)
    pz = np.select([d["pe_dev"] < -3, d["pe_dev"] > 3], [-1, 1], 0)
    d["regime"] = np.where(r.notna(), rz + bz + sz + pz, np.nan)
    z = (p - d["ma20"]) / d["sd20"].replace(0, 1)
    pdv = d["pe_dev"]
    d["signal_str"] = (r - 50) / 50 + z / 2 + np.where(pdv.abs() > 0.01, np.sign(pdv) / 3, pdv)
    d["btd"] = ((p < d["lower"]) & (r < 40)).astype(float)
    d["stt"] = ((p > d["upper"]) & (r > 70) & (d["breadth"] > 0.5)).astype(float)
    d["spread_flag"] = ((d["spread90"] > 1.5) & (d["spread200"] > 1.5)).astype(float)
    return d


def ad_ratio_day(d) -> float | None:
    """NSE market-wide advances / declines from the equity bhavcopy (EQ series)."""
    from .data import equity
    try:
        e = equity.load([d])
    except Exception:
        e = None
    if e is None or not len(e):
        x = equity.fetch_day(d)
        if x is None:
            return None
        e = x
    ch = e["close"] - e["prev_close"]
    a, dcl = int((ch > 0).sum()), int((ch < 0).sum())
    return round(a / dcl, 2) if dcl else None


def extend_live(raw: pd.DataFrame, spot: pd.Series, days) -> pd.DataFrame:
    """Append sessions after the history, from NSE's files: A/D from the equity bhavcopy,
    NIFTY close and P/E, P/B, dividend yield from the daily index-close file (the index
    desk's spot fills the close if that file is late)."""
    have = set(pd.to_datetime(raw["DATE"]))
    todo = [pd.Timestamp(d) for d in days if pd.Timestamp(d) not in have]
    if not todo:
        return raw
    new = nse_history(todo, workers=2)
    if new.empty:
        return raw
    if "NIFTY" not in new:
        new["NIFTY"] = np.nan
    new["NIFTY"] = new["NIFTY"].fillna(new["DATE"].map(spot))
    return pd.concat([raw, new], ignore_index=True).sort_values("DATE").drop_duplicates("DATE", keep="last") \
        .reset_index(drop=True)


# ── NSE sources: everything the desk's breadth needs, from NSE's own archive ────────────
NSE_HEADERS = {"User-Agent": "vaaydo/7 (end-of-day research)"}
CM_OLD = "https://nsearchives.nseindia.com/content/historical/EQUITIES/{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip"
IND_CLOSE = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{d:%d%m%Y}.csv"
UDIFF_FROM = pd.Timestamp("2024-07-08")      # the old CM bhavcopy stops; sec_bhavdata_full carries on


def _get(url, session, tries=3):
    import time
    for k in range(tries):
        try:
            r = session.get(url, headers=NSE_HEADERS, timeout=30)
            return r if r.status_code == 200 else None
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def nse_ad_day(d, session) -> tuple[int, int] | None:
    """(advances, declines) of EQ-series stocks on day d, from NSE's equity bhavcopy."""
    import io
    import zipfile
    t = pd.Timestamp(d)
    if t >= UDIFF_FROM:
        from .data import equity
        e = equity.fetch_day(t.date(), session)
        if e is None or not len(e):
            return None
        ch = e["close"] - e["prev_close"]
    else:
        r = _get(CM_OLD.format(d=t, mon=t.strftime("%b").upper()), session)
        if r is None or r.content[:2] != b"PK":
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        x = pd.read_csv(z.open(z.namelist()[0]))
        if "TIMESTAMP" in x and not (pd.to_datetime(x["TIMESTAMP"].astype(str).str.strip(), format="%d-%b-%Y",
                                                    errors="coerce").dt.date == t.date()).all():
            return None                     # a holiday request answered with another session's file
        x = x[x["SERIES"].astype(str).str.strip() == "EQ"]
        ch = x["CLOSE"] - x["PREVCLOSE"]
    return int((ch > 0).sum()), int((ch < 0).sum())


def nse_index_day(d, session) -> dict | None:
    """NIFTY 50 close, P/E, P/B, dividend yield from NSE's daily index-close file (2013 on)."""
    import io
    r = _get(IND_CLOSE.format(d=pd.Timestamp(d)), session)
    if r is None or not r.text.lstrip().lower().startswith("index name"):
        return None
    x = pd.read_csv(io.StringIO(r.text))
    x.columns = [c.strip() for c in x.columns]
    row = x[x["Index Name"].astype(str).str.strip().str.lower() == "nifty 50"]
    if row.empty:
        return None
    row = row.iloc[0]
    num = lambda v: pd.to_numeric(str(v).replace(",", "").strip(), errors="coerce")
    return {"NIFTY": num(row["Closing Index Value"]), "NIFTY50_PE": num(row["P/E"]),
            "NIFTY50_PB": num(row["P/B"]), "NIFTY50_DY": num(row["Div Yield"])}


def nse_history(days, workers: int = 4, progress=None) -> pd.DataFrame:
    """Raw breadth inputs for the given sessions, entirely from NSE files."""
    from concurrent.futures import ThreadPoolExecutor
    import requests

    def one(d):
        s = requests.Session()
        ad = nse_ad_day(d, s)
        if ad is None or ad[1] == 0:
            return None
        row = {"DATE": pd.Timestamp(d), "ADV": ad[0], "DEC": ad[1], "AD_RATIO": round(ad[0] / ad[1], 2)}
        ix = nse_index_day(d, s) if pd.Timestamp(d) >= pd.Timestamp("2013-01-01") else None
        if ix:
            row.update(ix)
        return row

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, list(days))):
            if r:
                rows.append(r)
            if progress and i % 500 == 0:
                progress(f"{i} days read")
    return pd.DataFrame(rows).sort_values("DATE").reset_index(drop=True)
