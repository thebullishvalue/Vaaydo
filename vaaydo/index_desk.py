"""
The index desk: NIFTY and BANKNIFTY, read every session.

    run(asof)            the panel through asof (bundle + any newer sessions fetched), participants,
                         VIX, today's chain views, and the paper candidates' state
    chain_view(d, sym)   per strike: call/put OI, OI change, dealer gamma; per expiry: ATM IV; smile

What it does NOT do: give index trades. Every directional and short-premium hypothesis was
tested on 2019–2026 and rejected (docs/PREREGISTRATION.md, "Index desk"); two near-misses are
tracked on paper from 18 Nov 2026 and shown as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import indices as I
from . import pricing
from .bundle import ROOT
from .config import RISK_FREE

PANEL_B = ROOT / "index_panel.parquet"
PART_B = ROOT / "participant_oi.parquet"
TESTS_B = ROOT / "index_tests.parquet"
CYCLES_B = ROOT / "index_cycles.parquet"
VIX_B = ROOT / "vix.parquet"
BREADTH_B = ROOT / "breadth_nse.parquet"   # NSE A/D (2006 on) and NIFTY close/PE/PB/DY (2013 on)
WB_TESTS_B = ROOT / "workbook_tests.parquet"
NROUND_B = ROOT / "nifty_combined_tests.parquet"
NTRADES_B = ROOT / "nifty_combined_trades.parquet"
IFUT_B = ROOT / "index_futures.parquet"


def nifty_reads(P: pd.DataFrame, pf: pd.DataFrame, B: pd.DataFrame) -> dict:
    """Today's paper reads of the combined NIFTY candidates (precedent N3, agreement N4)."""
    from . import index_research as IR, inference as INF
    if not IFUT_B.exists() or B is None or B.empty or pf is None or pf.empty:
        return {}
    F = pd.read_parquet(IFUT_B)
    F = F[F["symbol"] == "NIFTY"]
    F["date"] = pd.to_datetime(F["date"])
    R = IR.continuous(F).set_index("date")
    n = P[P["symbol"] == "NIFTY"].set_index("date").sort_index()
    X = n.join(R[["r_oo"]], how="left").join(pf, how="left")
    X = X.join(B.set_index("DATE")[["breadth"]].rename(columns={"breadth": "w_breadth"}), how="left")
    X["fwd5"] = np.exp(np.log1p(X["r_oo"]).rolling(5).sum().shift(-5)) - 1
    z = IR.zscore
    state = pd.DataFrame({"trend": z(np.log(X["spot"] / X["spot"].shift(60))), "breadth": z(X["w_breadth"]),
                          "pcr": z(X["pcr_oi"]), "fii": z(X["fii_opt_bias"]), "iv": z(X["iv_30"]),
                          "gamma": z(X["gex_dealer"])})
    out = {"state": state.iloc[-1].to_dict(), "date": X.index[-1]}
    out["precedent"] = INF.precedent_now(state, X["fwd5"], 5)
    a, b = np.sign(state["fii"].iloc[-1]), np.sign(state["breadth"].iloc[-1])
    out["agreement"] = float(a) if np.isfinite(a) and a == b else 0.0
    return out
PAPER_START = pd.Timestamp("2026-11-18")


@dataclass
class Desk:
    asof: pd.Timestamp
    panel: pd.DataFrame                 # date x symbol readings, 2019 on
    participants: pd.DataFrame          # date x feature (FII/Client/Pro/DII)
    part_raw: pd.DataFrame              # latest participant table
    vix: pd.Series
    views: dict                         # symbol -> chain_view dict
    paper: pd.DataFrame                 # candidate states per symbol
    tests: pd.DataFrame
    cycles: pd.DataFrame
    breadth: pd.DataFrame = field(default_factory=pd.DataFrame)   # market breadth and the workbook's readings
    wb_tests: pd.DataFrame = field(default_factory=pd.DataFrame)
    reads: dict = field(default_factory=dict)                     # combined NIFTY candidates, paper
    ntests: pd.DataFrame = field(default_factory=pd.DataFrame)
    ntrades: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list = field(default_factory=list)


def _bhav_days(since: pd.Timestamp, until: date) -> list[date]:
    from .data import bhavcopy
    out, d = [], since.date() + timedelta(days=1)
    while d <= until:
        if d.weekday() < 5 and bhavcopy.fetch_day(d) is not None:
            out.append(d)
        d += timedelta(days=1)
    return out


def chain_view(d, sym: str) -> dict:
    """Strike profile of the nearest live expiry and of all expiries ≤ 60 days, plus term structure."""
    asof = pd.Timestamp(d)
    df = I._load(d)
    g = df[df["symbol"] == sym]
    fut = g[(g["inst"] == "IDF") & (g["expiry"] >= asof)].sort_values("expiry")
    o = g[(g["inst"] == "IDO") & (g["expiry"] > asof)].copy()
    if o.empty or fut.empty:
        return {}
    sp = g["spot"].dropna()
    sp = sp[sp > 0]
    S = float(sp.iloc[0]) if len(sp) else float(fut["close"].iloc[0])
    o["dte"] = (o["expiry"] - asof).dt.days
    o["T"] = o["dte"].clip(lower=1) / 365
    fw = fut.set_index("expiry")["close"].to_dict()
    o["F"] = o["expiry"].map(fw).fillna(S * np.exp(RISK_FREE * o["T"]))
    o["k"] = np.log(o["strike"] / o["F"])
    is_c = (o["otype"] == "CE").to_numpy()
    tr = (o["contracts"] > 0) & (o["close"] >= 0.05)
    o["iv"] = np.nan
    o.loc[tr, "iv"] = pricing.implied_vol(o.loc[tr, "close"].to_numpy(), o.loc[tr, "F"].to_numpy(), o.loc[tr, "strike"].to_numpy(),
                                          o.loc[tr, "T"].to_numpy(), is_c[tr.to_numpy()])
    otm = o[o["iv"].notna() & (((o["otype"] == "CE") & (o["k"] >= 0)) | ((o["otype"] == "PE") & (o["k"] <= 0)))]
    term = []
    for e, x in otm.groupby("expiry"):
        x = x.sort_values("k")
        if len(x) >= 4 and x["k"].min() < 0 < x["k"].max():
            term.append(dict(expiry=e, dte=int(x["dte"].iloc[0]), atm_iv=float(np.interp(0, x["k"], x["iv"]))))
    term = pd.DataFrame(term)
    e0 = o["expiry"].min()
    near = o[o["expiry"] == e0]
    prof = near.pivot_table(index="strike", columns="otype", values=["oi", "oi_chg", "contracts", "close"], aggfunc="sum").fillna(0)
    prof.columns = [f"{a}_{b.lower()}" for a, b in prof.columns]
    prof = prof.reset_index()
    # dealer gamma per strike, all expiries within 60 days, IV from each expiry's smile
    gx = o[(o["dte"] <= 60) & (o["oi"] > 0)].copy()
    iv = np.full(len(gx), np.nan)
    for e, x in otm.groupby("expiry"):
        m = (gx["expiry"] == e).to_numpy()
        if len(x) >= 2 and m.any():
            x = x.sort_values("k")
            iv[m] = np.interp(gx.loc[m, "k"], x["k"], x["iv"])
    fb = term["atm_iv"].median() if len(term) else 0.15
    gx["iv"] = np.where(np.isfinite(iv), iv, fb)
    gam = I._gamma(gx["F"].to_numpy(), gx["strike"].to_numpy(), gx["T"].to_numpy(), gx["iv"].to_numpy())
    gx["gex"] = np.where(gx["otype"] == "CE", 1, -1) * gam * gx["oi"] * S * S * 0.01 / 1e7     # ₹ Cr per 1% move
    gstrike = gx.groupby("strike")["gex"].sum().rename("gex_cr").reset_index()
    smile = otm[otm["expiry"] == (term["expiry"].iloc[0] if len(term) else e0)][["strike", "otype", "iv", "k"]]
    return dict(spot=S, expiry=e0, profile=prof, gex=gstrike, term=term, smile=smile, lot=int(fut["lot"].max() or 0))


def _bundle_or_store(bundle_path, store_path) -> pd.DataFrame:
    for p in (bundle_path, store_path):
        if p.exists():
            return pd.read_parquet(p)
    return pd.DataFrame()


def breadth_state(B: pd.DataFrame) -> dict:
    """The breadth candidate (paper): long NIFTY when breadth is above its 250-session median."""
    if B is None or B.empty:
        return {}
    med = B["breadth"].rolling(250, min_periods=120).median()
    last = B.iloc[-1]
    return dict(date=last["DATE"], breadth=float(last["breadth"]), median=float(med.iloc[-1]),
                long=bool(last["breadth"] > med.iloc[-1]))


def paper_state(P: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """Today's state of the two candidates tracked on paper (never traded)."""
    rows = []
    for s, d in P.groupby("symbol"):
        d = d.sort_values("date").set_index("date")
        z = None
        if len(pf):
            fb = pf["fii_opt_bias"]
            z = ((fb - fb.rolling(250, min_periods=120).mean()) / fb.rolling(250, min_periods=120).std()).reindex(d.index).iloc[-1]
        ivf = d["iv_front"]
        q90 = ivf.rolling(250, min_periods=120).quantile(0.9).iloc[-1]
        rows.append(dict(symbol=s, fii_opt_z=z, fii_side=("LONG" if z is not None and z > 0 else "SHORT") if z is not None and np.isfinite(z) else "—",
                         iv_front=ivf.iloc[-1], iv_q90=q90, stress=bool(ivf.iloc[-1] >= q90) if np.isfinite(q90) else False))
    return pd.DataFrame(rows)


def run(asof: date | None = None, progress=None) -> Desk:
    say = progress or (lambda p, l: None)
    notes = []
    say(10, "loading index history")
    P = _bundle_or_store(PANEL_B, I.PANEL)
    P["date"] = pd.to_datetime(P["date"])
    until = asof or date.today()
    P = P[P["date"] <= pd.Timestamp(until)]
    last = P["date"].max()
    say(25, "reading new sessions")
    new = _bhav_days(last, until) if pd.Timestamp(until) > last else []
    rows = []
    for d in new:
        try:
            rows += I.day(d)
        except Exception as e:
            notes.append(f"{d}: index chain unreadable ({str(e)[:80]})")
    if rows:
        P = pd.concat([P, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
    P = P.sort_values(["symbol", "date"]).reset_index(drop=True)
    A = P["date"].max()
    say(45, "participant positions")
    Pt = _bundle_or_store(PART_B, I.PART)
    if len(Pt):
        Pt["date"] = pd.to_datetime(Pt["date"])
        Pt = Pt[Pt["date"] <= A]
        have = set(Pt["date"])
        miss = [d.date() for d in sorted(P["date"].unique()) if d not in have and d > Pt["date"].max()]
        for d in miss[-10:]:
            try:
                x = I.participant_day(d)
                if x is not None:
                    Pt = pd.concat([Pt, x], ignore_index=True)
            except Exception:
                pass
    pf = I.participant_features(Pt) if len(Pt) else pd.DataFrame()
    if len(Pt) and Pt["date"].max() < A:
        notes.append(f"Participant positions are through {Pt['date'].max():%d %b}; NSE publishes them in the evening.")
    say(60, "india vix")
    vix = pd.Series(dtype=float)
    if VIX_B.exists():
        vb = pd.read_parquet(VIX_B)
        vix = pd.Series(vb["vix"].values, index=pd.to_datetime(vb["date"]))
    try:
        live = I.yahoo("VIX", start=(vix.index.max() if len(vix) else pd.Timestamp("2018-01-01")).strftime("%Y-%m-%d"))
        vix = pd.concat([vix, live]).groupby(level=0).last()
    except Exception:
        if not len(vix):
            notes.append("India VIX could not be fetched; volatility panels use the chain's own IV.")
    vix = vix[vix.index <= A]
    say(75, "option chains")
    views = {}
    try:
        from .data import bhavcopy
        bhavcopy.fetch_day(A.date())                    # make sure the session's file is on disk
    except Exception:
        pass
    for s in I.INDICES:
        try:
            views[s] = chain_view(A.date(), s)
        except Exception as e:
            notes.append(f"{s} chain view failed: {str(e)[:80]}")
    say(85, "market breadth")
    B = pd.DataFrame()
    if BREADTH_B.exists():
        from . import breadth as BR
        raw = pd.read_parquet(BREADTH_B)
        raw["DATE"] = pd.to_datetime(raw["DATE"])
        raw = raw[raw["DATE"] <= A]
        spot = P[P["symbol"] == "NIFTY"].set_index("date")["spot"]
        newer = [d for d in spot.index if d > raw["DATE"].max()]
        try:
            raw = BR.extend_live(raw, spot, newer[-30:])
        except Exception as e:
            notes.append(f"Breadth could not be extended past {raw['DATE'].max():%d %b}: {str(e)[:80]}")
        B = BR.features(raw)
        if B["DATE"].max() < A:
            notes.append(f"Market breadth is through {B['DATE'].max():%d %b}.")
    say(90, "verdicts")                     # after breadth: the steps run in the order the work does
    tests = pd.read_parquet(TESTS_B) if TESTS_B.exists() else pd.DataFrame()
    cycles = pd.read_parquet(CYCLES_B) if CYCLES_B.exists() else pd.DataFrame()
    part_raw = Pt[Pt["date"] == Pt["date"].max()] if len(Pt) else Pt
    wb = pd.read_parquet(WB_TESTS_B) if WB_TESTS_B.exists() else pd.DataFrame()
    nt = pd.read_parquet(NROUND_B) if NROUND_B.exists() else pd.DataFrame()
    try:
        reads = nifty_reads(P, pf, B)
    except Exception as e:
        reads = {}
        notes.append(f"Combined NIFTY reads unavailable: {str(e)[:80]}")
    return Desk(asof=A, panel=P, participants=pf, part_raw=part_raw, vix=vix, views=views,
                paper=paper_state(P, pf), tests=tests, cycles=cycles, breadth=B, wb_tests=wb, reads=reads,
                ntests=nt, ntrades=pd.read_parquet(NTRADES_B) if NTRADES_B.exists() else pd.DataFrame(), notes=notes)
