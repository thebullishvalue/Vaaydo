"""
VAAYDO — F&O trade screener.                 streamlit run app.py

Pick a universe and a date in the rail, press Run: every NSE F&O stock is screened on
where its options market is positioned, and the trades come out as tickets.
"""
from __future__ import annotations

import html
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

import ui.components as ui
from ui.theme import inject_css, progress_bar, chart_layout, style_axes, chart_color
from vaaydo import curator, signal
from vaaydo.universe import UNIVERSES, constituents

VERSION = "6.2.0"
st.set_page_config(page_title="VAAYDO | F&O Trade Screener", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
inject_css(theme=st.session_state["theme"])
TRACK = "data/track_record.parquet"


# ── formatting ──────────────────────────────────────────────────────────────
def inr(x, sign=False) -> str:
    if x is None or not isinstance(x, (int, float, np.floating)) or not np.isfinite(x):
        return "—"
    s = "−" if x < 0 else ("+" if sign and x > 0 else "")
    x = abs(x)
    if x >= 1e7:
        return f"{s}₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"{s}₹{x / 1e5:.2f} L"
    return f"{s}₹{x:,.0f}"


def pct(x, sign=True, d=1) -> str:
    try:
        return f"{x:+.{d}%}" if sign else f"{x:.{d}%}"
    except (TypeError, ValueError):
        return "—"


def px(x) -> str:
    try:
        return f"{x:,.2f}" if np.isfinite(x) else "—"
    except TypeError:
        return "—"


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d %b %y") if d is not None and not pd.isna(d) else "—"


# ── rail ────────────────────────────────────────────────────────────────────
with st.sidebar:
    ui.render_nav_brand("VAAYDO", "वायदो · F&O Trade Screener")
    st.markdown('<div class="sidebar-title">Universe</div>', unsafe_allow_html=True)
    universe = st.selectbox("Universe", UNIVERSES, key="sb_universe", label_visibility="collapsed")
    st.markdown('<div class="sidebar-title">Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("Mode", ["Single Date", "Historical Range"], key="sb_mode", label_visibility="collapsed")
    today = date.today()
    if mode == "Single Date":
        st.markdown('<div class="sidebar-title">Analysis Date</div>', unsafe_allow_html=True)
        asof = st.date_input("Date", today, min_value=date(2024, 10, 1), max_value=today, key="sb_date",
                             label_visibility="collapsed")
        start = None
    else:
        st.markdown('<div class="sidebar-title">Analysis Range</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", today - timedelta(days=365), min_value=date(2024, 7, 8), max_value=today,
                              key="sb_start", label_visibility="collapsed")
        asof = c2.date_input("End", today, min_value=date(2024, 7, 8), max_value=today, key="sb_end",
                             label_visibility="collapsed")
    st.markdown('<div class="sidebar-title">Sizing (optional)</div>', unsafe_allow_html=True)
    size_on = st.toggle("Size to my capital", value=False, key="sb_size")
    capital, leverage = None, 1.0
    if size_on:
        capital = st.number_input("Capital (₹)", 500_000, 500_000_000, 8_500_000, step=500_000, key="sb_capital")
        leverage = st.select_slider("Gross exposure ÷ capital", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], value=1.0, key="sb_lev")
    run_clicked = st.button("RUN SCREENER" if mode == "Single Date" else "RUN HARVEST", type="primary", width="stretch")
    st.markdown('<div class="sidebar-title">Appearance</div>', unsafe_allow_html=True)
    look = st.radio("Appearance", ["Slate", "Paper"], index=0 if st.session_state["theme"] == "dark" else 1,
                    horizontal=True, key="sb_theme", label_visibility="collapsed")
    want = "dark" if look == "Slate" else "light"
    if want != st.session_state["theme"]:
        st.session_state["theme"] = want
        st.rerun()


@st.cache_resource(show_spinner=False, ttl=6 * 3600, max_entries=6)
def cached_run(asof_key: str, day_key: str):
    return curator.run(None if asof_key == "latest" else date.fromisoformat(asof_key))


@st.cache_data(show_spinner=False)
def track() -> pd.DataFrame:
    return pd.read_parquet(TRACK)


def uni_filter(df: pd.DataFrame, members) -> pd.DataFrame:
    if members is None or df is None or df.empty or "symbol" not in df:
        return df
    return df[df["symbol"].isin(members)]


# ── run ─────────────────────────────────────────────────────────────────────
if run_clicked:
    slot = st.empty()
    key = "latest" if asof >= today else asof.isoformat()
    try:
        progress_bar(slot, 3, "loading the bundle")
        r = cached_run(key, today.isoformat()) if key == "latest" else None
        if r is None:
            r = curator.run(asof, progress=lambda p, label: progress_bar(slot, p, label))
        progress_bar(slot, 100, "analysis complete")
        st.session_state["run"] = r
        st.session_state["run_meta"] = dict(universe=universe, mode=mode, start=start, asof=asof)
        st.session_state["run_error"] = None
    except Exception as e:
        st.session_state["run_error"] = str(e)
    slot.empty()

r: curator.Run | None = st.session_state.get("run")
meta = st.session_state.get("run_meta", {})


def landing():
    st.markdown("""<div class="lede">
  <div class="lede-claim">Every NSE F&amp;O stock, screened each evening on where its options market is positioned —
    and the futures trades that follow from it, as tickets.</div>
  <div class="lede-cta">Pick a universe, a mode and a date in the rail, then <strong>Run</strong>.</div>
</div>""", unsafe_allow_html=True)
    T = track()
    ui.render_section_header("Coverage", icon="layers")
    ui.render_kpi_strip([
        dict(label="Universe", value="~210", subtext="every NSE single-stock F&O underlying, every session"),
        dict(label="Readings per stock", value="30+", subtext="option positioning, option flow, futures flow, levels, delivery, risk"),
        dict(label="History", value="Jul 2024 →", subtext="any session since; research on 2019 – today"),
    ], max_cols=3, key="landing-cov")
    ui.render_section_header("System", icon="cpu")
    cols = st.columns(3, gap="small")
    parts = [
        ("Signal", "Option positioning", "When the options crowd leans hardest to puts on a stock, it has tended to beat the "
         "other F&O stocks over the following weeks; hardest to calls, it has lagged. Ten-session score, ranked across "
         "every F&O stock.", [("Enter", "top / bottom 5%"), ("Hold", "inside 15%"), ("Instrument", "stock futures")]),
        ("Risk", "Pragyam", "Every position carries a trailing stop at 1.5× Pragyam's calibrated stop distance, decided on "
         "the close and executed at the next open, and Pragyam's 80% ten-session range.",
         [("Stop", "trailing, 1.5×"), ("After a stop", "out until reset"), ("Range", "80%, 10 sessions")]),
        ("Book", "Positions", "Long and short in equal rupees. Entries at the next open in the contract active that day, "
         "rolled three sessions before expiry; every leg kept, with its own lot size.",
         [("Balance", "market-neutral"), ("Rolls", "T−3 sessions"), ("Corporate actions", "exchange factor")]),
    ]
    for col, (k, name, body, specs) in zip(cols, parts):
        with col:
            with ui.panel(f"landing-{k}", name, context=k):
                st.markdown(f'<div class="panel-copy">{html.escape(body)}</div><div class="panel-specs">' +
                            "".join(f'<div class="lookback-row"><span class="lbl">{html.escape(a)}</span>'
                                    f'<span class="val">{html.escape(b)}</span></div>' for a, b in specs) + "</div>",
                            unsafe_allow_html=True)
    ui.render_section_header("What a Run Returns", icon="target")
    out = (("Signals", "New ▲ and ▼, exits, and the open book as tickets."),
           ("Book", "Every position leg by leg: entries, rolls, lot changes, P&L, stop."),
           ("Watchlist", "The names closest to firing."),
           ("Flow", "Breadth, sectors, futures build-up, option flow."),
           ("Contracts", "The option-contract scanner."),
           ("Screen", "Every stock with every reading."),
           ("System Data", "Track record, evidence, exports, definitions."))
    st.markdown('<div class="outcome-grid">' + "".join(
        f'<div class="outcome"><div class="o-t">{html.escape(t)}</div><div class="o-d">{html.escape(d)}</div></div>'
        for t, d in out) + "</div>", unsafe_allow_html=True)


def footer():
    st.markdown(f'<div class="app-footer">VAAYDO v{VERSION} · NSE F&amp;O bhavcopy, end of day · '
                f'decision support, not investment advice</div>', unsafe_allow_html=True)


if st.session_state.get("run_error"):
    ui.render_empty_state("The run could not complete", html.escape(st.session_state["run_error"]),
                          eyebrow="Run failed", action_label="Try another date, or run again in a minute")
if r is None:
    landing()
    footer()
    st.stop()

# ── results ─────────────────────────────────────────────────────────────────
members = constituents(meta.get("universe", "All F&O stocks"))
uname = meta.get("universe", "All F&O stocks")
scr = uni_filter(r.screen, members)
pos = uni_filter(r.positions, members)
legs = uni_filter(r.legs, members)
book_pos, book_legs = pos, legs
if size_on and capital:
    sp, sl = curator.sized(r, capital, leverage, members)
    book_pos, book_legs = sp, sl
T = track()
new_l = scr[scr["state"] == "▲ NEW"]
new_s = scr[scr["state"] == "▼ NEW"]
held = book_pos[book_pos["action"] != "EXIT"] if len(book_pos) else book_pos
ui.render_top_bar(target=uname.upper(), status_label=meta.get("mode", ""), status_tone="accent",
                  meta_items=[("As of", r.asof.strftime("%d %b %Y")),
                              ("Signals", f"{len(new_l)} ▲ · {len(new_s)} ▼"),
                              ("Book", f"{int((held['side'] == 'LONG').sum()) if len(held) else 0} L · "
                                       f"{int((held['side'] == 'SHORT').sum()) if len(held) else 0} S")])
notices = []
if meta.get("asof") and pd.Timestamp(meta["asof"]) > r.asof:
    notices.append(dict(kind="info", title="Latest published session",
                        body=f"NSE's file for {meta['asof']:%d %b} is not out yet (it publishes around 18:00 IST); "
                             f"showing {r.asof:%d %b %Y}."))
if members is None and uname != "All F&O stocks":
    notices.append(dict(kind="warning", title="Universe list unavailable",
                        body="The index constituents could not be fetched; showing all F&O stocks."))
if size_on and capital and len(held) < 10:
    k6, k12 = T["k6"], T["k12"]
    sr = lambda x: x.mean() / x.std() * np.sqrt(252)
    notices.append(dict(kind="warning", title="Book too small to be the strategy",
                        body=f"{len(held)} names at {inr(capital)} and {leverage:g}×. Six names backtested at Sharpe "
                             f"{sr(k6):.2f}, twelve at {sr(k12):.2f}: the edge needs enough names for stock-specific "
                             f"noise to wash out."))
for n in r.notes:
    notices.append(dict(kind="info", title="Note", body=html.escape(n)))
ui.render_notice_rail(notices)


def tickets(df: pd.DataFrame, kind: str):
    """Order tickets as the app's institutional table."""
    if df is None or df.empty:
        return
    rows = []
    for _, p in df.iterrows():
        unfilled = kind == "ENTER" or p["entry_px"] is None or not np.isfinite(p["entry_px"])
        rows.append({
            "Symbol": p["symbol"], "Side": ("▲ " if p["side"] == "LONG" else "▼ ") + p["side"],
            "Contract": f"FUT {pd.Timestamp(p['contract']):%d %b %y}" if p["contract"] is not None else "—",
            "Lot": int(p["lot"]) if p["lot"] else "—",
            "Signal": fmt_date(p["signal"]),
            "Entry": "next open" if unfilled else f"{px(p['entry_px'])} · {fmt_date(p['entry_date'])}",
            "Last": px(p["last"]),
            "P&L": "—" if unfilled else inr(p["pnl"], True),
            "Return": "—" if unfilled else pct(p["pnl_pct"]),
            "Stop": f"{px(p['stop'])} ({pct(p['stop_from_last'])})",
            "Risk to stop": inr(p["risk_to_stop"]),
            "Range (80%)": f"{px(p['range_lo'])} – {px(p['range_hi'])}",
            "Legs": int(p["legs"]),
            "Industry": p.get("industry", "—"),
            "Why": p["why"] or "",
        })
    ui.render_data_table(pd.DataFrame(rows), show_index=False, label_col="Symbol", max_height=520)


def leg_table(lg: pd.DataFrame, sym: str):
    x = lg[lg["symbol"] == sym]
    if x.empty:
        return
    t = pd.DataFrame({
        "Leg": x["leg"], "Contract": [f"FUT {pd.Timestamp(c):%d %b %y}" for c in x["contract"]],
        "Entry": [f"{px(a)} · {fmt_date(b)}" for a, b in zip(x["entry_px"], x["entry_date"])],
        "Lot": x["lot"],
        "Exit": [("open" if e is None or pd.isna(e) else f"{px(a)} · {fmt_date(e)}") for a, e in zip(x["exit_px"], x["exit_date"])],
        "Status": x["status"], "P&L": [inr(v, True) for v in x["pnl"]],
        "Adjusted": ["split/bonus factor applied" if a else "" for a in x["adjusted"]]})
    ui.render_data_table(t, show_index=False, max_height=260)


# ── Single Date ─────────────────────────────────────────────────────────────
if meta.get("mode") == "Single Date":
    tabs = st.tabs(["Signals", "Book", "Watchlist", "Flow", "Contracts", "Screen", "System Data"])

    with tabs[0]:
        exits = book_pos[book_pos["action"] == "EXIT"] if len(book_pos) else book_pos
        enters = book_pos[book_pos["action"] == "ENTER"] if len(book_pos) else book_pos
        longs = held[held["side"] == "LONG"] if len(held) else held
        shorts = held[held["side"] == "SHORT"] if len(held) else held
        ready = scr[scr["state"].isin(["READY ▲", "READY ▼"])]
        ui.render_set_board([
            dict(key="▲", title="Long", tone="long", line="options crowd leaning hardest to puts",
                 rate=f"{len(longs)} held", sides=[("▲", len(enters[enters['side'] == 'LONG']) if len(enters) else 0, "new today", "long"),
                                                   ("●", len(longs), "in the book", "neutral")],
                 watch=("Ready ▲", str(int((ready['state'] == 'READY ▲').sum()))), edges=[]),
            dict(key="▼", title="Short", tone="short", line="options crowd leaning hardest to calls",
                 rate=f"{len(shorts)} held", sides=[("▼", len(enters[enters['side'] == 'SHORT']) if len(enters) else 0, "new today", "short"),
                                                    ("●", len(shorts), "in the book", "neutral")],
                 watch=("Ready ▼", str(int((ready['state'] == 'READY ▼').sum()))), edges=[]),
            dict(key="×", title="Exits", tone="neutral", line="leave the book at the next open",
                 rate=f"{len(exits)} today", sides=[("×", int((exits['why'].str.contains('stop')).sum()) if len(exits) else 0, "stops", "short"),
                                                    ("○", int((~exits['why'].str.contains('stop')).sum()) if len(exits) else 0, "signal reset", "neutral")],
                 watch=("Stopped, waiting", str(int((scr['state'] == 'STOPPED').sum()))), edges=[]),
        ])
        ui.render_section_header("Enter at the next open", "New signals: buy ▲ or sell ▼ one lot in the contract shown.",
                                 icon="target")
        if len(enters):
            tickets(enters, "ENTER")
        else:
            ui.render_empty_state("No new signals", "No stock crossed into the top or bottom 5% today.", eyebrow="Signals")
        ui.render_section_header("Exit at the next open", "Close these positions at the next session's open.", icon="arrow-down")
        if len(exits):
            tickets(exits, "EXIT")
        else:
            ui.render_empty_state("No exits", "Every position stays.", eyebrow="Exits")
        ui.render_section_header("Holding", "Open positions, with P&L since entry across every roll, and today's stop.",
                                 icon="layers")
        if len(held):
            tickets(held[held["action"] == "HOLD"], "HOLD")
        if size_on and capital:
            st.markdown(f'<div class="panel-copy">Sized to {inr(capital)} at {leverage:g}× — the same rules, limited to '
                        f'the most extreme names that fit, long and short balanced.</div>', unsafe_allow_html=True)

    with tabs[1]:
        gl = held.loc[held["side"] == "LONG", "notional"].sum() if len(held) else 0
        gs = held.loc[held["side"] == "SHORT", "notional"].sum() if len(held) else 0
        ui.render_kpi_strip([
            dict(label="Positions", value=str(len(held)), subtext=f"{int((held['side'] == 'LONG').sum()) if len(held) else 0} long · "
                                                                   f"{int((held['side'] == 'SHORT').sum()) if len(held) else 0} short"),
            dict(label="Gross exposure", value=inr(gl + gs), subtext=f"net {inr(gl - gs, True)}"),
            dict(label="Open P&L", value=inr(held["pnl"].sum() if len(held) else 0, True), subtext="realised on rolls + open legs",
                 color_class="success" if len(held) and held["pnl"].sum() >= 0 else "danger"),
            dict(label="Risk to stops", value=inr(held["risk_to_stop"].sum() if len(held) else 0), subtext="if every stop hit"),
            dict(label="Margin (est.)", value=inr((gl + gs) * 0.2), subtext="≈ 20% of notional"),
        ], key="book-kpis")
        ui.render_section_header("Positions", "Pick one to see every contract leg it has been held in.", icon="layers")
        if len(held):
            tickets(held, "HOLD")
            sym = st.selectbox("Position", list(held["symbol"]), key="book_sym")
            p = held.set_index("symbol").loc[sym]
            ui.render_section_header(f"{sym} · {p['side']}", f"Signal {fmt_date(p['signal'])} · {int(p['legs'])} leg(s) · "
                                     f"realised {inr(p['realised'], True)} · open {inr(p['open_pnl'], True)}", icon="list")
            leg_table(book_legs, sym)
        ui.render_section_header("By industry", icon="grid")
        if len(held):
            ind = held.groupby(["industry", "side"])["notional"].sum().unstack(fill_value=0)
            ind["Net"] = ind.get("LONG", 0) - ind.get("SHORT", 0)
            ui.render_data_table(ind.map(lambda v: inr(v, True)).reset_index(), show_index=False, label_col="industry", max_height=360)

    with tabs[2]:
        ui.render_section_header("Watchlist", "In the 5–15% band and moving toward an entry, nearest first. Not yet a trade.",
                                 icon="eye")
        w = scr[scr["state"].isin(["READY ▲", "READY ▼"])].copy()
        if len(w):
            w["dist"] = w["to_entry"].abs()
            w = w.sort_values("dist")
            ui.render_data_table(pd.DataFrame({
                "Symbol": w["symbol"], "State": w["state"], "Score rank": [pct(v, False, 0) for v in w["score"]],
                "5-session change": [pct(v, True, 0) for v in w["score_chg5"]], "To entry": [pct(v, False, 0) for v in w["dist"]],
                "Put/call vol": w["pcr_vol"].round(2), "Put skew": [pct(v) for v in w["skew"]], "IV": [pct(v, False, 0) for v in w["atm_iv"]],
                "Futures 5d": w["build5"].fillna("—"), "Industry": w["industry"]}), show_index=False, label_col="Symbol")
        else:
            ui.render_empty_state("Nobody close", "No stock is moving toward an entry.", eyebrow="Watchlist")
        st_ = scr[scr["state"] == "STOPPED"]
        if len(st_):
            ui.render_section_header("Stopped out", "Out until the signal leaves the band; a new signal is a new trade.", icon="alert-triangle")
            ui.render_data_table(st_[["symbol", "score", "industry"]].assign(score=lambda x: x["score"].map(lambda v: pct(v, False, 0))),
                                 show_index=False, label_col="symbol")

    with tabs[3]:
        fl = scr.dropna(subset=["fut_ret1"])
        up, dn = int((fl["fut_ret1"] > 0.005).sum()), int((fl["fut_ret1"] < -0.005).sum())
        ui.render_kpi_strip([
            dict(label="Breadth (futures, day)", value=f"{up} ▲ · {dn} ▼", subtext=f"{len(fl) - up - dn} flat of {len(fl)}"),
            dict(label="Long build-up (day)", value=str(int((fl['build1'] == 'Long build-up').sum())), color_class="success"),
            dict(label="Short build-up (day)", value=str(int((fl['build1'] == 'Short build-up').sum())), color_class="danger"),
            dict(label="Short covering / long unwinding", value=f"{int((fl['build1'] == 'Short covering').sum())} / "
                                                                f"{int((fl['build1'] == 'Long unwinding').sum())}"),
        ], key="flow-kpis")
        ui.render_section_header("Sectors", "Median futures move, build-up and option positioning by industry.", icon="grid")
        g = fl.groupby("industry")
        sect = pd.DataFrame({
            "Names": g.size(), "Median move": g["fut_ret1"].median().map(pct),
            "5-day move": g["fut_ret5"].median().map(pct),
            "Long build-up (5d)": g["build5"].apply(lambda s: int((s == "Long build-up").sum())),
            "Short build-up (5d)": g["build5"].apply(lambda s: int((s == "Short build-up").sum())),
            "Score (median rank)": g["score"].median().map(lambda v: pct(v, False, 0)),
            "In book ▲ / ▼": g["state"].apply(lambda s: f"{int(s.isin(['LONG', '▲ NEW']).sum())} / {int(s.isin(['SHORT', '▼ NEW']).sum())}"),
        }).sort_values("Names", ascending=False).reset_index()
        ui.render_data_table(sect, show_index=False, label_col="industry", max_height=420)
        ui.render_section_header("Futures build-up, five sessions", icon="activity")
        bc = fl["build5"].value_counts()
        fig = go.Figure(go.Bar(x=bc.values, y=bc.index, orientation="h", marker_color=chart_color("accent")))
        fig.update_layout(**chart_layout(height=260, show_legend=False))
        ui.render_chart_panel(fig, "flow-build", units="names")
        ui.render_section_header("Option flow", "Today's option contracts: who opened positions, unusual volume, volume surges. "
                                 "Context: tested and not used as signals.", icon="activity")
        of = scr.dropna(subset=["buyer_tilt"]).copy()
        of = of.reindex(of["surge_tilt"].abs().sort_values(ascending=False).index).head(40)
        ui.render_data_table(pd.DataFrame({
            "Symbol": of["symbol"], "Call buyers − put buyers": [pct(v, True, 2) for v in of["buyer_tilt"]],
            "Put writers − call writers": [pct(v, True, 2) for v in of["writer_tilt"]],
            "Unusual call − put": [pct(v, True, 0) for v in of["unusual_tilt"]],
            "Call vs put volume surge": [f"{np.exp(v):.2f}×" if np.isfinite(v) else "—" for v in of["surge_tilt"]],
            "State": of["state"]}), show_index=False, label_col="Symbol", max_height=420)

    with tabs[4]:
        ui.render_section_header("Option contracts", "Every stock option that traded today, read against its previous session.",
                                 icon="list")
        cf = uni_filter(r.contracts, members)
        if cf is None or cf.empty:
            ui.render_empty_state("No contract data", "The previous session's file was not available.", eyebrow="Contracts")
        else:
            c1, c2, c3 = st.columns(3)
            scan = c1.selectbox("Scan", ["Most active", "Open interest gainers", "Open interest losers", "Buyers opening",
                                         "Writers opening", "Premium gainers", "Premium losers", "Volume above open interest"],
                                key="scan")
            side = c2.selectbox("Side", ["Both", "Calls", "Puts"], key="scan_side")
            stock = c3.selectbox("Underlying", ["All"] + sorted(cf["symbol"].unique()), key="scan_sym")
            x = cf[cf["inst"].astype(str) == "STO"].copy()
            if side != "Both":
                x = x[x["otype"] == ("CE" if side == "Calls" else "PE")]
            if stock != "All":
                x = x[x["symbol"] == stock]
            lot = x["lot"].where(x["lot"] > 0, 1)
            x["value"] = x["contracts"] * lot * x["close"]
            x["oi_lots"] = x["oi"] / lot
            x["oi_pct"] = x["oi_chg"] / x["oi_start"].where(x["oi_start"] > 0)
            x = x[x["contracts"] > 0]
            sorts = {"Most active": ("value", False, None), "Open interest gainers": ("oi_chg_lots", False, x["oi_chg_lots"] > 0),
                     "Open interest losers": ("oi_chg_lots", True, x["oi_chg_lots"] < 0),
                     "Buyers opening": ("oi_chg_lots", False, x["flow"] == "buyers opening"),
                     "Writers opening": ("oi_chg_lots", False, x["flow"] == "writers opening"),
                     "Premium gainers": ("premium_chg", False, x["premium_chg"] > 0),
                     "Premium losers": ("premium_chg", True, x["premium_chg"] < 0),
                     "Volume above open interest": ("contracts", False, x["unusual"])}
            col, asc, mask = sorts[scan]
            if mask is not None:
                x = x[mask]
            x = x.sort_values(col, ascending=asc).head(100)
            ui.render_data_table(pd.DataFrame({
                "Contract": [f"{s} {pd.Timestamp(e):%b} {k:g} {o}" for s, e, k, o in zip(x["symbol"], x["expiry"], x["strike"], x["otype"])],
                "Premium": x["close"].round(2), "Change": [pct(v) for v in x["premium_chg"]],
                "Lots traded": x["contracts"], "Value": [inr(v) for v in x["value"]],
                "OI (lots)": x["oi_lots"].round(0).astype("Int64"), "OI change (lots)": x["oi_chg_lots"].round(0).astype("Int64"),
                "OI change": [pct(v) for v in x["oi_pct"]], "Reading": x["flow"].replace("", "—")}),
                show_index=False, label_col="Contract", max_height=560)

    with tabs[5]:
        ui.render_section_header("Every stock", "The screen, ordered by state then score.", icon="grid")
        ui.render_data_table(pd.DataFrame({
            "Symbol": scr["symbol"], "State": scr["state"], "Score rank": [pct(v, False, 0) for v in scr["score"]],
            "Today's reading": [pct(v, False, 0) for v in scr["opi_today"]], "Put/call vol": scr["pcr_vol"].round(2),
            "Put/call OI": scr["pcr_oi"].round(2), "Put skew": [pct(v) for v in scr["skew"]],
            "Call skew": [pct(v) for v in scr["cskew"]], "IV": [pct(v, False, 0) for v in scr["atm_iv"]],
            "Event": ["⚠" if e else "" for e in scr["event"]], "Futures 1d": scr["build1"].fillna("—"),
            "Futures 5d": scr["build5"].fillna("—"), "Put wall": scr["wall_put"].round(0), "Call wall": scr["wall_call"].round(0),
            "Delivery": [f"{v:.0f}%" if np.isfinite(v) else "—" for v in scr["deliv_pct"]],
            "Risk": scr["risk_regime"].fillna("—"), "Industry": scr["industry"]}),
            show_index=False, label_col="Symbol", max_height=640)

    with tabs[6]:
        sr = lambda x: x.mean() / x.std() * np.sqrt(252)
        ui.render_section_header("Track record, 2019 – today", "Backtest on real NSE futures prices: next-open fills, all "
                                 "charges and slippage, trailing stops, rolls, corporate actions. Returns on gross exposure.",
                                 icon="trending-up")
        fig = go.Figure()
        for col, nm in (("model_30", "model book (~30)"), ("k20", "20 names"), ("k12", "12 names"), ("k6", "6 names")):
            fig.add_trace(go.Scatter(x=T.index, y=(1 + T[col]).cumprod(), mode="lines", name=nm))
        fig.add_vrect(x0="2023-01-01", x1="2024-06-30", fillcolor=chart_color("amber"), opacity=0.08, line_width=0)
        fig.update_layout(**chart_layout(height=340))
        ui.render_chart_panel(fig, "track", units="growth of ₹1")
        summ = []
        for col, nm in (("model_30", "Model book (~30)"), ("k20", "20 names"), ("k12", "12 names"), ("k6", "6 names")):
            x = T[col]
            eq = (1 + x).cumprod()
            yrs = (1 + x).groupby(x.index.year).prod() - 1
            summ.append({"Book": nm, "Sharpe": round(sr(x), 2), "Return / yr": pct(x.mean() * 252), "Volatility": pct(x.std() * np.sqrt(252), False),
                         "Worst drawdown": pct((eq / eq.cummax() - 1).min()), "Losing years": f"{int((yrs < 0).sum())} of {len(yrs)}"})
        ui.render_data_table(pd.DataFrame(summ), show_index=False, label_col="Book")
        ui.render_section_header("Every trade since July 2024", "Closed trades the model book made, from this same engine.",
                                 icon="list")
        tr = uni_filter(r.trades, members)
        if len(tr):
            ui.render_kpi_strip([
                dict(label="Trades", value=f"{len(tr):,}", subtext=f"{int((tr['side'] == 'LONG').sum())} long · {int((tr['side'] == 'SHORT').sum())} short"),
                dict(label="Won", value=pct((tr["ret"] > 0).mean(), False, 0)),
                dict(label="Average trade", value=pct(tr["ret"].mean(), True, 2), subtext="after costs, on notional"),
                dict(label="Median sessions held", value=f"{tr['sessions'].median():.0f}"),
                dict(label="Stopped out", value=pct(tr["why"].str.contains("stop").mean(), False, 0)),
            ], key="trades-kpis")
        ui.render_section_header("Exports", icon="download")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Screen (CSV)", scr.to_csv(index=False).encode(), f"vaaydo_screen_{r.asof:%Y%m%d}.csv", width="stretch")
        c2.download_button("Positions (CSV)", book_pos.to_csv(index=False).encode(), f"vaaydo_positions_{r.asof:%Y%m%d}.csv", width="stretch")
        c3.download_button("Legs (CSV)", book_legs.to_csv(index=False).encode(), f"vaaydo_legs_{r.asof:%Y%m%d}.csv", width="stretch")
        ui.render_section_header("Definitions", icon="info")
        ui.render_kv_table({
            "Score": "10-session mean of the day's option-positioning rank (put/call volume and OI, put skew, call skew, "
                     "call−put IV, net option OI added), ranked across all F&O stocks",
            "▲ / ▼ NEW": "score entered the top / bottom 5% at this close; enter at the next open",
            "Held": "while the score stays in the top / bottom 15%",
            "Stop": "trailing: 1.5 × Pragyam's 20%-touch stop distance from the best close, decided on the close, filled next open",
            "STOPPED": "stopped out; no re-entry until the score leaves the band",
            "READY": "inside the 5–15% band and moving toward an entry over five sessions",
            "Contract": "shown with its expiry date: the nearest expiry with more than 3 sessions left on the fill day (expiries "
                        "moved for holidays are taken as NSE lists them); rolled at the close 3 sessions before expiry",
            "Corporate actions": "splits, bonuses and demergers rescale entry, stop and lot by the exchange's factor",
        }, header_left="Term", header_right="Meaning")

# ── Historical Range ────────────────────────────────────────────────────────
else:
    s0, s1 = pd.Timestamp(meta.get("start") or date(2024, 7, 8)), r.asof
    tr = uni_filter(r.trades, members)
    tr = tr[(pd.to_datetime(tr["signal"]) >= s0) & (pd.to_datetime(tr["signal"]) <= s1)] if len(tr) else tr
    tabs = st.tabs(["Trades", "By month", "Track record"])
    with tabs[0]:
        if tr is None or tr.empty:
            ui.render_empty_state("No closed trades in the range", "Widen the range.", eyebrow="Harvest")
        else:
            ui.render_kpi_strip([
                dict(label="Trades", value=f"{len(tr):,}", subtext=f"{s0:%d %b %y} – {s1:%d %b %y}"),
                dict(label="Won", value=pct((tr["ret"] > 0).mean(), False, 0)),
                dict(label="Average", value=pct(tr["ret"].mean(), True, 2), subtext="per trade, after costs"),
                dict(label="Long / short average", value=f"{pct(tr.loc[tr['side'] == 'LONG', 'ret'].mean(), True, 2)} / "
                                                         f"{pct(tr.loc[tr['side'] == 'SHORT', 'ret'].mean(), True, 2)}"),
                dict(label="Stopped out", value=pct(tr["why"].str.contains("stop").mean(), False, 0)),
            ], key="hr-kpis")
            ui.render_section_header("Every trade", "Signal, fill, exit and the compounded return across rolls, after costs.",
                                     icon="list")
            t = tr.sort_values("signal", ascending=False)
            ui.render_data_table(pd.DataFrame({
                "Symbol": t["symbol"], "Side": t["side"], "Signal": t["signal"].map(fmt_date),
                "Entry": [f"{px(a)} · {fmt_date(b)}" for a, b in zip(t["entry_px"], t["entry"])],
                "Exit": [f"{px(a)} · {fmt_date(b)}" for a, b in zip(t["exit_px"], t["exit"])],
                "Sessions": t["sessions"], "Legs": t["legs"], "Return": t["ret"].map(pct), "Exit reason": t["why"]}),
                show_index=False, label_col="Symbol", max_height=600)
    with tabs[1]:
        if tr is not None and len(tr):
            m = tr.assign(month=pd.to_datetime(tr["signal"]).dt.to_period("M").astype(str)).groupby("month")
            mm = pd.DataFrame({"Trades": m.size(), "Won": m["ret"].apply(lambda x: (x > 0).mean()),
                               "Average": m["ret"].mean()}).reset_index()
            fig = go.Figure(go.Bar(x=mm["month"], y=mm["Average"] * 100,
                                   marker_color=[chart_color("emerald") if v >= 0 else chart_color("rose") for v in mm["Average"]]))
            fig.update_layout(**chart_layout(height=300, show_legend=False))
            ui.render_chart_panel(fig, "hr-month", units="average trade return, %")
            ui.render_data_table(mm.assign(Won=mm["Won"].map(lambda v: pct(v, False, 0)), Average=mm["Average"].map(lambda v: pct(v, True, 2))),
                                 show_index=False, label_col="month")
    with tabs[2]:
        x = T[(T.index >= s0) & (T.index <= s1)]
        fig = go.Figure()
        for col, nm in (("model_30", "model book (~30)"), ("k20", "20 names"), ("k12", "12 names"), ("k6", "6 names")):
            fig.add_trace(go.Scatter(x=x.index, y=(1 + x[col]).cumprod(), mode="lines", name=nm))
        fig.update_layout(**chart_layout(height=340))
        ui.render_chart_panel(fig, "hr-track", units="growth of ₹1 over the range")

footer()
