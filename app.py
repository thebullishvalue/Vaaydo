"""
VAAYDO — F&O trade screener.                 streamlit run app.py

Pick a universe and a date in the rail, press Run: every NSE F&O stock is screened on
where its options market is positioned, and the trades come out as tickets.
"""
from __future__ import annotations

import html
import warnings
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

import ui.components as ui
from ui.theme import inject_css, progress_bar, chart_layout, style_axes, chart_color
from vaaydo import curator, index_desk, signal
import ui.desk as desk_view
from vaaydo.universe import UNIVERSES, constituents

VERSION = "7.3.1"
STOCKS, INDICES = "F&O Stocks", "Indices · NIFTY & BANKNIFTY"
st.set_page_config(page_title="VAAYDO | F&O Trade Screener", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# ── appearance (the family's pattern, as in Sanket) ─────────────────────────
# The choice lives in a DURABLE session key, never a widget key: Streamlit drops the state
# of a widget that a run did not reach, and the switch sits at the foot of the rail, below
# RUN. It is read here, before anything is styled, so chrome and charts agree on one theme
# for the whole run. Slate leads (a screener is a working surface); config.toml's dark
# base agrees with it.
APPEARANCES = ("Slate", "Paper")
_THEME_CHOICE = "theme_choice"


def theme_choice() -> str:
    choice = st.session_state.get(_THEME_CHOICE)
    return choice if choice in APPEARANCES else APPEARANCES[0]


st.session_state["theme"] = "light" if theme_choice() == "Paper" else "dark"
inject_css(theme=st.session_state["theme"])


def render_appearance_control() -> None:
    """The theme switch: the LAST control in the rail, pinned to its foot by the stylesheet."""
    with st.container(key="appearance"):
        st.markdown('<div class="sidebar-title">Appearance</div>', unsafe_allow_html=True)
        mode = st.segmented_control("Appearance", list(APPEARANCES), key="theme_mode",
                                    default=theme_choice(), label_visibility="collapsed",
                                    help="Slate — dark, for working. Paper — light, for reading and print.")
        if mode is not None and mode != theme_choice():
            st.session_state[_THEME_CHOICE] = mode
            st.rerun()
TRACK = "data/track_record.parquet"
TRACK_BOOKS = (("one_lot", "One lot of every name (what VAAYDO trades)"), ("equal_rupees", "Equal rupees in every name (research)"))


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
        if x is None or not np.isfinite(x):
            return "—"
        x = round(float(x), d + 2) + 0.0          # no "-0.0%"
        return f"{x:+.{d}%}" if sign else f"{x:.{d}%}"
    except (TypeError, ValueError):
        return "—"


def px(x) -> str:
    try:
        return f"{x:,.2f}" if np.isfinite(x) else "—"
    except TypeError:
        return "—"


def volpts(x) -> str:
    """An implied-volatility DIFFERENCE (skew, call-put spread) in vol points: 0.021 -> +2.1 pts."""
    try:
        return f"{round(x * 100, 1) + 0.0:+.1f} pts" if np.isfinite(x) else "—"
    except TypeError:
        return "—"


def rank(x) -> str:
    """A cross-sectional percentile rank, one decimal: 94.96% must not read as the 95% entry line."""
    try:
        return f"{x * 100:.1f}%" if np.isfinite(x) else "—"
    except TypeError:
        return "—"


def rank_pts(x, sign=True) -> str:
    """A change or gap in percentile POINTS (not a percent change)."""
    try:
        return (f"{round(x * 100, 1) + 0.0:+.1f} pts" if sign else f"{abs(x) * 100:.1f} pts") if np.isfinite(x) else "—"
    except TypeError:
        return "—"


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d %b %y") if d is not None and not pd.isna(d) else "—"


# ── rail ────────────────────────────────────────────────────────────────────
with st.sidebar:
    ui.render_nav_brand("VAAYDO", "वायदो · F&O Trade Screener")
    st.markdown('<div class="sidebar-title">Universe</div>', unsafe_allow_html=True)
    kind = st.selectbox("Universe", [STOCKS, INDICES], key="sb_kind", label_visibility="collapsed")
    universe = "All F&O stocks"
    if kind == STOCKS:
        st.markdown('<div class="sidebar-title">Filter</div>', unsafe_allow_html=True)
        universe = st.selectbox("Filter", UNIVERSES, key="sb_universe", label_visibility="collapsed")
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
    run_clicked = st.button("RUN DESK" if kind == INDICES else ("RUN SCREENER" if mode == "Single Date" else "RUN HARVEST"),
                            type="primary", width="stretch")
    render_appearance_control()


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=6)
def cached_run(asof_key: str, day_key: str):
    return curator.run(None if asof_key == "latest" else date.fromisoformat(asof_key))


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=4)
def cached_desk(asof_key: str, day_key: str):
    return index_desk.run(None if asof_key == "latest" else date.fromisoformat(asof_key))


@st.cache_data(show_spinner=False)
def track() -> pd.DataFrame:
    return pd.read_parquet(TRACK)


def uni_filter(df: pd.DataFrame, members) -> pd.DataFrame:
    if members is None or df is None or df.empty or "symbol" not in df:
        return df
    return df[df["symbol"].isin(members)]


# ── run ─────────────────────────────────────────────────────────────────────
if run_clicked and kind == INDICES:
    slot = st.empty()
    key = "latest" if asof >= today else asof.isoformat()
    try:
        progress_bar(slot, 5, "reading the index chains")
        dk = cached_desk(key, datetime.now().strftime("%Y-%m-%d %H"))
        progress_bar(slot, 100, "desk ready")
        st.session_state["desk"] = dk
        st.session_state["view"] = INDICES
        st.session_state["desk_meta"] = dict(mode=mode, start=start, asof=asof)
        st.session_state["run_error"] = None
    except Exception as e:
        st.session_state["run_error"] = str(e)
    slot.empty()
elif run_clicked:
    slot = st.empty()
    key = "latest" if asof >= today else asof.isoformat()
    try:
        progress_bar(slot, 3, "loading the bundle")
        r = cached_run(key, datetime.now().strftime("%Y-%m-%d %H")) if key == "latest" else None
        if r is None:
            r = curator.run(asof, progress=lambda p, label: progress_bar(slot, p, label))
        progress_bar(slot, 100, "analysis complete")
        st.session_state["run"] = r
        st.session_state["run_meta"] = dict(universe=universe, mode=mode, start=start, asof=asof)
        st.session_state["view"] = STOCKS
        st.session_state["run_error"] = None
    except Exception as e:
        st.session_state["run_error"] = str(e)
    slot.empty()

r: curator.Run | None = st.session_state.get("run")
meta = st.session_state.get("run_meta", {})


def landing():
    """Cold start, as every member of the family opens: the masthead, the claim and one line on
    what to do, then Coverage, System and What a Run Returns, in the family's components. The
    numbers a trader asks first (book size, money, orders, the record) are the system panels'
    spec rows, computed from the track record rather than typed."""
    ui.render_header("VAAYDO", "F&O Trade Screener · Option Positioning · Stock Futures")
    run_error_state()
    st.markdown("""<div class="lede">
  <div class="lede-claim">Every NSE F&amp;O stock, screened each evening on where its options market is positioned —
    and the futures trades that follow from it, as tickets.</div>
  <div class="lede-cta">Pick a universe, a mode and a date in the rail, then <strong>Run</strong>.</div>
</div>""", unsafe_allow_html=True)
    T = track()
    x = T["one_lot"]
    eq = (1 + x).cumprod()
    yrs = (1 + x).groupby(x.index.year).prod() - 1
    sharpe, ret = x.mean() / x.std() * np.sqrt(252), x.mean() * 252
    worst = (eq / eq.cummax() - 1).min()
    ui.render_section_header("Coverage", icon="layers")
    ui.render_kpi_strip([
        dict(label="F&O Stocks", value="~220", subtext="every NSE single-stock F&O underlying, every session; "
                                                       "index lists filter it"),
        dict(label="Index Desk", value="2", subtext="NIFTY and BANKNIFTY: chain, gamma, volatility, breadth, "
                                                    "participants; a market read, no index trades"),
        dict(label="History", value="Jul 2024 →", subtext="any session since; the research on 2019 – today"),
    ], max_cols=3, key="landing-cov")
    ui.render_section_header("System", icon="cpu")
    cols = st.columns(3, gap="small")
    parts = [
        ("signal", "Option positioning", "Signal",
         "When the options crowd leans hardest to puts on a stock, it has tended to beat the other F&O stocks "
         "over the following weeks; hardest to calls, it has lagged. Ten-session score, ranked across every "
         "F&O stock.",
         [("Enter", "top / bottom 5%"), ("Hold", "inside 15%"), ("Instrument", "stock futures"),
          ("Record, 2019 – today", f"Sharpe {sharpe:.2f} · {ret:.1%} a year")]),
        ("risk", "Pragyam", "Risk",
         "Every position carries a trailing stop at 1.5× Pragyam's calibrated stop distance, decided on the "
         "close and executed at the next open, and Pragyam's 80% ten-session range.",
         [("Stop", "trailing, 1.5×"), ("After a stop", "out until reset"), ("Range", "80%, 10 sessions"),
          ("Worst drawdown", f"{worst:.0%} · {int((yrs < 0).sum())} losing year of {len(yrs)}")]),
        ("book", "Positions", "Book",
         "One lot of every name, long and short, no capital sizing. Entries at the next open in the contract "
         "active that day, rolled three sessions before expiry; every leg kept, with its own lot size.",
         [("Positions", "≈ 30, one lot each"), ("Money", "≈ ₹2 Cr gross · ₹40–50 L margin"),
          ("Each evening", "1–4 orders"), ("Per trade", "≈ 1 in 2 wins · 11 sessions")]),
    ]
    for col, (k, name, kicker, body, specs) in zip(cols, parts):
        with col:
            with ui.panel(f"landing-{k}", name, context=kicker):
                st.markdown(f'<div class="panel-copy">{html.escape(body)}</div><div class="panel-specs">' +
                            "".join(f'<div class="lookback-row"><span class="lbl">{html.escape(a)}</span>'
                                    f'<span class="val">{html.escape(b)}</span></div>' for a, b in specs) + "</div>",
                            unsafe_allow_html=True)
    ui.render_section_header("What a Run Returns", icon="target")
    out = (("Tonight's orders", "New ▲ and ▼, exits and rolls, as tickets: contract, quantity, stop."),
           ("The book", "Every position leg by leg: entries, rolls, lot changes, P&L, stop; exposure by industry."),
           ("The market around it", "The watchlist, breadth and build-up, option flow, the contract scanner and "
                                    "every stock's readings."),
           ("The record", "The track record 2019 – today, every closed trade since July 2024, exports and "
                          "definitions."))
    st.markdown('<div class="outcome-grid">' + "".join(
        f'<div class="outcome"><div class="o-t">{html.escape(t)}</div><div class="o-d">{html.escape(d)}</div></div>'
        for t, d in out) + "</div>", unsafe_allow_html=True)


def footer():
    """The family's footer: product, author, version, the time it was drawn (IST)."""
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    st.markdown(f'<div class="app-footer"><div class="content">© {ist.year} <strong>VAAYDO</strong> &nbsp;·&nbsp; '
                f'@thebullishvalue &nbsp;·&nbsp; v{VERSION} &nbsp;·&nbsp; {ist:%Y-%m-%d %H:%M:%S} IST</div></div>',
                unsafe_allow_html=True)


def run_error_state() -> None:
    """A failed run, said once: under the masthead on the cold start, under the command bar otherwise."""
    if st.session_state.get("run_error"):
        ui.render_empty_state("The run could not complete", html.escape(st.session_state["run_error"]),
                              eyebrow="Run failed", action_label="Try another date, or run again in a minute")


view = st.session_state.get("view")
if view == INDICES and st.session_state.get("desk") is not None:
    run_error_state()
    desk_view.render(st.session_state["desk"], st.session_state.get("desk_meta", {}))
    footer()
    st.stop()
if r is None or view != STOCKS:
    landing()
    footer()
    st.stop()

# ── results ─────────────────────────────────────────────────────────────────
members = constituents(meta.get("universe", "All F&O stocks"))
uname = meta.get("universe", "All F&O stocks")
scr = uni_filter(r.screen, members)
pos = uni_filter(r.positions, members)
legs = uni_filter(r.legs, members)
book_pos, book_legs = pos, legs          # one lot of every position: no capital sizing
T = track()
if meta.get("mode") == "Single Date":
    T = T[T.index <= r.asof]          # a past-dated run shows only what was known that evening
new_l = scr[scr["state"] == "▲ NEW"]
new_s = scr[scr["state"] == "▼ NEW"]
# "held" = the book after the next open: positions kept plus new entries (exits leave at that open)
held = book_pos[book_pos["action"] != "EXIT"] if len(book_pos) else book_pos
RANGE = meta.get("mode") == "Historical Range"
# Every chart and table panel's context line (universe · date · units) reads these, as in
# Sanket: built from the same values as the command bar, so the two cannot disagree.
st.session_state["active_universe"] = uname
st.session_state["active_timeframe"] = (
    f"{pd.Timestamp(meta.get('start') or date(2024, 7, 8)):%d %b %Y} – {r.asof:%d %b %Y}" if RANGE
    else f"{r.asof:%d %b %Y}")
if RANGE:
    ui.render_top_bar(target=uname.upper(), status_label="Historical Range", status_tone="accent",
                      meta_items=[("From", pd.Timestamp(meta.get("start") or date(2024, 7, 8)).strftime("%d %b %Y")),
                                  ("To", r.asof.strftime("%d %b %Y")), ("Size", "one lot each")])
else:
    ui.render_top_bar(target=uname.upper(), status_label=meta.get("mode", ""), status_tone="accent",
                      meta_items=[("As of", r.asof.strftime("%d %b %Y")),
                                  ("Tonight's orders", f"{len(new_l)} buy · {len(new_s)} sell · "
                                                       f"{int((book_pos['action'] == 'EXIT').sum()) if len(book_pos) else 0} exit"),
                                  ("Book after next open", f"{int((held['side'] == 'LONG').sum()) if len(held) else 0} L · "
                                                           f"{int((held['side'] == 'SHORT').sum()) if len(held) else 0} S")])
notices = []
if not RANGE and meta.get("asof") and pd.Timestamp(meta["asof"]) > r.asof:
    notices.append(dict(kind="info", title="Latest published session",
                        body=f"NSE's file for {meta['asof']:%d %b} is not out yet (it publishes around 18:00 IST); "
                             f"showing {r.asof:%d %b %Y}."))
if members is None and uname != "All F&O stocks":
    notices.append(dict(kind="warning", title="Universe list unavailable",
                        body="The index constituents could not be fetched; showing all F&O stocks."))
nb_date, nb = getattr(r, "ban_next", (None, None))
if nb and not RANGE:
    mine = sorted(set(nb) & set(r.positions["symbol"]))
    notices.append(dict(kind="warning" if mine else "info", title=f"F&O ban on {nb_date:%d %b}",
                        body=f"{', '.join(nb)}: no new positions or rolls in these until the ban lifts. "
                             + (f"In VAAYDO's book or orders: {', '.join(mine)} (an entry waits; a roll is deferred)."
                                if mine else "None of them is in VAAYDO's book or tonight's orders.")))
for n in r.notes:
    notices.append(dict(kind="info", title="Note", body=html.escape(n)))
run_error_state()
ui.render_notice_rail(notices)


def traded_as(sym: str, when) -> str:
    """The symbol a past trade actually carried on the exchange, if it has since been renamed."""
    old = getattr(r, "renames", {}).get(sym) if r is not None else None
    if old and when is not None and not pd.isna(when) and pd.Timestamp(when) <= old[1]:
        return f"{sym} (traded as {old[0]})"
    return sym


def fut(c) -> str:
    return f"FUT {pd.Timestamp(c):%d %b %y}" if c is not None and not pd.isna(c) else "—"


def qty_text(p) -> str:
    """Shares held. One lot at entry; if NSE has since revised the contract's lot, the same
    shares are a different number of today's lots (the position itself does not change)."""
    q, now = int(p["lot"] or 0), int(p.get("lot_now") or 0)
    if not q:
        return "—"
    if now and now != q:
        return f"{q:,} (= {q / now:g} lots of {now:,} now)"
    return f"{q:,} (1 lot)"


def order(p, kind: str) -> str:
    """The broker instruction, in plain words: side, shares, contract, how and when."""
    long_ = p["side"] == "LONG"
    q = int(p["lot"] or 0)
    if kind == "ENTER":
        return f"{'BUY' if long_ else 'SELL'} {q:,} · {fut(p['contract'])}"
    if kind == "EXIT":
        return f"{'SELL' if long_ else 'BUY'} {q:,} · {fut(p['contract'])}"
    return ""


def entry_orders(df: pd.DataFrame):
    rows = [{"Symbol": p["symbol"], "Order": order(p, "ENTER"), "Signal": fmt_date(p["signal"]),
             "Last close": px(p["last"]), "Stop after entry": f"{px(p['stop'])} ({pct(p['stop_from_last'])})",
             "Risk to stop": inr(p["risk_to_stop"]), "Range, 10 sessions (80%)": f"{px(p['range_lo'])} – {px(p['range_hi'])}",
             "Industry": p.get("industry", "—")} for _, p in df.iterrows()]
    ui.render_table_panel(pd.DataFrame(rows), "orders-enter", units="market orders at the next open · one lot each",
                          show_index=False, label_col="Symbol", max_height=420)


def exit_orders(df: pd.DataFrame):
    rows = [{"Symbol": p["symbol"], "Order": order(p, "EXIT"),
             "P&L (all legs, before charges)": inr(p["pnl"], True), "Return": pct(p["pnl_pct"]), "Reason": p["why"] or "",
             "Opened": f"{px(p['first_px'])} · {fmt_date(p['entry_date'])} · {fut(p['first_contract'])}"} for _, p in df.iterrows()]
    ui.render_table_panel(pd.DataFrame(rows), "orders-exit", units="market orders at the next open · closes the position",
                          show_index=False, label_col="Symbol", max_height=420,
                          sign_color_cols={"P&L (all legs, before charges)", "Return"})


def roll_orders(df: pd.DataFrame):
    rows = []
    for _, p in df.iterrows():
        long_, q, q2 = p["side"] == "LONG", int(p["lot"] or 0), int(p.get("roll_qty") or 0)
        if str(p["roll_what"]).startswith("roll to"):
            nxt = p["roll_what"].replace("roll to ", "")
            o = (f"{'SELL' if long_ else 'BUY'} {q:,} {fut(p['contract'])} · {'BUY' if long_ else 'SELL'} "
                 f"{q2 or q:,} {nxt}")
            note = "" if not q2 or q2 == q else f"new contract's lot is {q2:,} (NSE revised it)"
        else:
            o = f"{'SELL' if long_ else 'BUY'} {q:,} {fut(p['contract'])}: no later contract is listed"
            note = "stock leaving F&O"
        rows.append({"Symbol": p["symbol"], "Side": p["side"], "Orders": o, "Last": px(p["last"]), "Note": note})
    ui.render_table_panel(pd.DataFrame(rows), "orders-roll", units="both legs at the close",
                          show_index=False, label_col="Symbol", max_height=420)


def trade_kpis(tr: pd.DataFrame, key: str, span: str = ""):
    win, loss = tr.loc[tr["ret"] > 0, "ret"], tr.loc[tr["ret"] <= 0, "ret"]
    rs = tr["pnl_rs"] if "pnl_rs" in tr else pd.Series(dtype=float)
    ui.render_kpi_strip([
        dict(label="Closed trades", value=f"{len(tr):,}", subtext=span or f"{int((tr['side'] == 'LONG').sum())} long · "
                                                                           f"{int((tr['side'] == 'SHORT').sum())} short"),
        dict(label="Rupee P&L, one lot each", value=inr(rs.sum(), True), subtext=f"after charges · {inr(rs.mean(), True)} a trade",
             color_class="success" if rs.sum() >= 0 else "danger"),
        dict(label="Won", value=pct((tr["ret"] > 0).mean(), False, 0),
             subtext=f"average win {pct(win.mean(), True, 1)} · average loss {pct(loss.mean(), True, 1)}"),
        dict(label="Average trade", value=pct(tr["ret"].mean(), True, 2), subtext="on the contract's value, after charges"),
        dict(label="Held / stopped", value=f"{tr['sessions'].median():.0f} sessions",
             subtext=f"median holding · {pct(tr['why'].str.contains('stop').mean(), False, 0)} ended on a stop"),
    ], key=key)


def tickets(df: pd.DataFrame, kind: str):
    """Order tickets. A held position is a chain of contract legs, so a ticket shows what was
    actually traded first (contract, date, exchange price), the contract held now (since its
    last roll, at that roll's price), and the next roll, known in advance from the calendar."""
    if df is None or df.empty:
        return
    rows = []
    for _, p in df.iterrows():
        enter = kind == "ENTER" or p.get("action") == "ENTER"      # a new signal: not filled until the next open
        if enter:
            opened = f"next open · {fut(p['contract'])}"
            now_in = "—"
        else:
            adj = p.get("first_adj")
            opened = f"{px(p['first_px'])} · {fmt_date(p['entry_date'])} · {fut(p['first_contract'])}"
            if adj is not None and np.isfinite(adj) and adj != 1.0:
                opened += f" (= {px(p['first_px'] * adj)} after a ×{adj:.3g} split/bonus)"
            now_in = fut(p["contract"]) if p["legs"] == 1 else \
                f"{fut(p['contract'])} · rolled {fmt_date(p['leg_since'])} at {px(p['leg_px'])}"
        roll = "—" if kind == "EXIT" or p.get("roll_on") is None or pd.isna(p.get("roll_on")) else \
            f"{fmt_date(p['roll_on'])} close · {p['roll_what']}"
        # What a trader acts on first (size, price, P&L, stop), then where it came from.
        rows.append({
            "Symbol": p["symbol"], "Side": ("▲ " if p["side"] == "LONG" else "▼ ") + p["side"],
            "Quantity": qty_text(p),
            "Last": px(p["last"]),
            "P&L (all legs, before charges)": "—" if enter else inr(p["pnl"], True),
            "Return": "—" if enter else pct(p["pnl_pct"]),
            "Stop": f"{px(p['stop'])} ({pct(p['stop_from_last'])})",
            "Risk to stop": inr(p["risk_to_stop"]),
            "Opened": opened,
            "Holding now": now_in,
            "Next roll": roll,
            "Range (80%)": f"{px(p['range_lo'])} – {px(p['range_hi'])}",
            "Signal": fmt_date(p["signal"]),
            "Industry": p.get("industry", "—"),
        })
    ui.render_table_panel(pd.DataFrame(rows), "book-positions", units="one lot each · P&L before charges",
                          show_index=False, label_col="Symbol", max_height=640,
                          sign_color_cols={"P&L (all legs, before charges)", "Return"})


def leg_table(lg: pd.DataFrame, sym: str):
    x = lg[lg["symbol"] == sym]
    if x.empty:
        return
    fills = []
    for a, b, f in zip(x["fill_px"], x["entry_date"], x["adj"]):
        t = f"{px(a)} · {fmt_date(b)}"
        if f is not None and np.isfinite(f) and f != 1.0:
            t += f" (= {px(a * f)} after ×{f:.3g})"
        fills.append(t)
    t = pd.DataFrame({
        "Leg": x["leg"], "Contract": [f"{traded_as(sym, d).replace(sym, '').strip(' ()') or sym} {fut(c)}"
                                      for c, d in zip(x["contract"], x["entry_date"])],
        "Filled": fills, "Quantity": x["lot"],
        "Closed": [("open" if e is None or pd.isna(e) else f"{px(a)} · {fmt_date(e)}") for a, e in zip(x["exit_px"], x["exit_date"])],
        "How": x["status"], "P&L": [inr(v, True) for v in x["pnl"]]})
    ui.render_table_panel(t, "book-legs", context=f"{sym} · contract legs · P&L before charges",
                          show_index=False, max_height=260, sign_color_cols={"P&L"})
    ui.render_note("Each leg is one futures contract: bought or sold at the exchange print shown, closed at its roll "
                   "(the close three sessions before expiry, into the next contract at that same close) or at the exit. "
                   "P&L adds every leg.")


@st.fragment
def position_legs(held: pd.DataFrame, book_legs: pd.DataFrame):
    """Picking a position reruns only this block, so the page stays on the Book tab."""
    order_ = held.assign(_f=(held["action"] != "ENTER").astype(int)).sort_values(["_f", "legs"], ascending=False)
    ui.render_section_header("Contract Legs", "Every futures contract a position has been held in: its fill, its roll "
                             "and its P&L.", icon="list")
    pick, _ = st.columns([1, 2])
    sym = pick.selectbox("Position", list(order_["symbol"]), key="book_sym",
                         format_func=lambda x: f"{x} · {held.set_index('symbol').loc[x, 'side'].lower()}"
                         + (" · enters at the next open" if held.set_index('symbol').loc[x, 'action'] == 'ENTER' else ""))
    p = held.set_index("symbol").loc[sym]
    ui.render_sub_header(f"{sym} · {p['side'].lower()} · signal {fmt_date(p['signal'])} · {int(p['legs'])} "
                         f"leg{'s' if int(p['legs']) != 1 else ''} · realised {inr(p['realised'], True)} · "
                         f"open {inr(p['open_pnl'], True)}")
    leg_table(book_legs, sym)


@st.fragment
def contracts_scanner(cf: pd.DataFrame):
    """Interacting with the scanner reruns only this block, so the page keeps its tab."""
    c1, c2, c3 = st.columns(3)
    scan = c1.selectbox("Scan", ["Most active", "Open interest gainers", "Open interest losers", "Buyers opening",
                                 "Writers opening", "Premium gainers", "Premium losers", "Volume above open interest"],
                        key="scan")
    side = c2.selectbox("Side", ["Both", "Calls", "Puts"], key="scan_side")
    stock = c3.selectbox("Underlying", ["All", "My book & tonight's orders"] + sorted(cf["symbol"].unique()), key="scan_sym")
    x = cf[cf["inst"].astype(str) == "STO"].copy()
    if side != "Both":
        x = x[x["otype"] == ("CE" if side == "Calls" else "PE")]
    if stock == "My book & tonight's orders":
        x = x[x["symbol"].isin(set(book_pos["symbol"]))]
    elif stock != "All":
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
    ui.render_table_panel(pd.DataFrame({
        "Contract": [f"{s} {pd.Timestamp(e):%d %b %y} {k:g} {o}" for s, e, k, o in zip(x["symbol"], x["expiry"], x["strike"], x["otype"])],
        "Premium": x["close"].round(2), "Change": [pct(v) for v in x["premium_chg"]],
        "Lots traded": x["contracts"], "Premium traded": [inr(v) for v in x["value"]],
        "OI (lots)": x["oi_lots"].round(0).astype("Int64"), "OI change (lots)": x["oi_chg_lots"].round(0).astype("Int64"),
        "OI change": [pct(v) for v in x["oi_pct"]], "Who": x["flow"].replace("", "unclear (OI and premium disagree)")}),
        "contracts-scan", context=f"{scan.lower()} · {side.lower()} · " + (f"{len(x)} contracts" if len(x) < 100 else "top 100 contracts"),
        show_index=False, label_col="Contract", max_height=560, sign_color_cols={"Change", "OI change"})


# ── Single Date ─────────────────────────────────────────────────────────────
if meta.get("mode") == "Single Date":
    exits = book_pos[book_pos["action"] == "EXIT"] if len(book_pos) else book_pos
    enters = book_pos[book_pos["action"] == "ENTER"] if len(book_pos) else book_pos
    ready = scr[scr["state"].isin(["READY ▲", "READY ▼"])]
    # A tab that holds a list carries its count, as in Sanket: tonight's orders, the book
    # after the next open, the names near an entry.
    tabs = st.tabs([f"Signals · {len(enters) + len(exits)}", f"Book · {len(held)}", f"Watchlist · {len(ready)}",
                    "Flow", "Contracts", "Screen", "System Data"])

    with tabs[0]:
        longs = held[held["side"] == "LONG"] if len(held) else held
        shorts = held[held["side"] == "SHORT"] if len(held) else held
        hold = held[held["action"] == "HOLD"] if len(held) else held
        n_el = int((enters["side"] == "LONG").sum()) if len(enters) else 0
        n_es = int((enters["side"] == "SHORT").sum()) if len(enters) else 0
        due = held[held["roll_on"].notna()] if len(held) and "roll_on" in held else held.iloc[0:0]
        nxt_roll = due["roll_on"].min() if len(due) else None
        due = due[due["roll_on"] == nxt_roll] if len(due) else due
        ui.render_section_header(
            f"Tonight's Orders · {r.asof:%d %b %Y}",
            f"{len(enters)} to enter and {len(exits)} to exit at the next open"
            + (f", {len(due)} to roll at the {fmt_date(nxt_roll)} close" if len(due) else "")
            + f". {len(hold)} other positions need nothing (Book tab). One lot each.", icon="zap", accent="amber")
        ui.render_set_board([
            dict(key="▲", title="Long", tone="success side", line="options crowd leaning hardest to puts",
                 rate="", sides=[("▲", n_el, "to buy at the open", "long"), ("●", len(longs), "long after the open", "neutral")],
                 watch=("Near entry ▲ (watchlist)", str(int((ready['state'] == 'READY ▲').sum()))), edges=[]),
            dict(key="▼", title="Short", tone="danger side", line="options crowd leaning hardest to calls",
                 rate="", sides=[("▼", n_es, "to sell at the open", "short"), ("●", len(shorts), "short after the open", "neutral")],
                 watch=("Near entry ▼ (watchlist)", str(int((ready['state'] == 'READY ▼').sum()))), edges=[]),
            dict(key="×", title="Exits", tone="neutral side", line="close at the next open",
                 rate="", sides=[("×", int((exits['why'].str.contains('stop')).sum()) if len(exits) else 0, "stops hit", "short"),
                                 ("○", int((~exits['why'].str.contains('stop')).sum()) if len(exits) else 0, "signal reset", "neutral")],
                 watch=("Stopped out, waiting to re-arm", str(int((scr['state'] == 'STOPPED').sum()))), edges=[]),
        ])
        ui.render_section_header("Enter at the Next Open", "New signals. The stop applies from the fill, on closing prices.",
                                 icon="target")
        if len(enters):
            entry_orders(enters)
        else:
            ui.render_empty_state("No new signals", "No stock crossed into the top or bottom 5% today.", eyebrow="Signals")
        ui.render_section_header("Exit at the Next Open", "Close these positions at the next session's open.", icon="arrow-down")
        if len(exits):
            exit_orders(exits)
        else:
            ui.render_empty_state("No exits", "Every position stays.", eyebrow="Exits")
        if len(due):
            ui.render_section_header(f"Roll at the {fmt_date(nxt_roll)} Close",
                                     "Close the expiring contract and open the next one at that session's close (three sessions "
                                     "before expiry, known in advance).", icon="move-horizontal")
            roll_orders(due)

    with tabs[1]:
        gl = held.loc[held["side"] == "LONG", "notional"].sum() if len(held) else 0
        gs = held.loc[held["side"] == "SHORT", "notional"].sum() if len(held) else 0
        ui.render_section_header("The Book", "Held positions plus tomorrow's entries, one lot each, every contract leg kept.",
                                 icon="briefcase")
        ui.render_kpi_strip([
            dict(label="Positions after next open", value=str(len(held)), subtext=f"{int((held['side'] == 'LONG').sum()) if len(held) else 0} long · "
                                                                   f"{int((held['side'] == 'SHORT').sum()) if len(held) else 0} short"),
            dict(label="Gross exposure", value=inr(gl + gs), subtext=f"net {inr(gl - gs, True)}"),
            dict(label="P&L of open positions", value=inr(held["pnl"].sum() if len(held) else 0, True),
                 subtext="since entry, every leg, before charges",
                 color_class="success" if len(held) and held["pnl"].sum() >= 0 else "danger"),
            dict(label="Risk to stops", value=inr(held["risk_to_stop"].sum() if len(held) else 0),
                 subtext="lost from today's close if every stop hit"),
            dict(label="Margin (rough)", value=inr((gl + gs) * 0.2), subtext="≈ 20% of notional; your broker's SPAN differs"),
        ], key="book-kpis")
        ui.render_section_header("Positions After the Next Open", "Entries first, then the positions held longest. Each "
                                 "position's contract legs are below.", icon="layers")
        if len(held):
            tickets(held, "HOLD")
            position_legs(held, book_legs)
            ui.render_section_header("By Industry", "Gross notional by industry, long against short.", icon="grid")
            ind = held.groupby(["industry", "side"])["notional"].sum().unstack(fill_value=0)
            for c in ("LONG", "SHORT"):
                if c not in ind:
                    ind[c] = 0.0
            ind["Net"] = ind["LONG"] - ind["SHORT"]
            ind = ind.reindex(ind["Net"].abs().sort_values(ascending=False).index)
            ui.render_table_panel(pd.DataFrame({"Industry": ind.index, "Long": [inr(v) if v else "—" for v in ind["LONG"]],
                                                "Short": [inr(v) if v else "—" for v in ind["SHORT"]],
                                                "Net (long − short)": [inr(v, True) for v in ind["Net"]]}),
                                  "book-industry", units="gross notional", show_index=False, label_col="Industry",
                                  max_height=360, sign_color_cols={"Net (long − short)"})
        else:
            ui.render_empty_state("No positions", "The book is empty after the next open.", eyebrow="Book")

    with tabs[2]:
        ui.render_section_header("Watchlist", "Not a trade yet. A long enters when its score rank goes above 95.0%, a short "
                                 "when it falls below 5.0%; these are inside 85–95% (or 5–15%) and moved toward that line "
                                 "over the last five sessions. Nearest first.", icon="eye", accent="violet")
        w = ready.copy()
        if len(w):
            w["dist"] = w["to_entry"].abs()
            w = w.sort_values("dist")
            ui.render_table_panel(pd.DataFrame({
                "Symbol": w["symbol"], "Would be": ["long ▲" if x == "READY ▲" else "short ▼" for x in w["state"]],
                "Score rank": [rank(v) for v in w["score"]], "Gap to entry": [rank_pts(v, False) for v in w["dist"]],
                "Rank change, 5 sessions": [rank_pts(v) for v in w["score_chg5"]],
                "Put/call vol": w["pcr_vol"].round(2), "Put skew (vol pts)": [volpts(v) for v in w["skew"]],
                "IV": [pct(v, False, 0) for v in w["atm_iv"]], "Futures, 5 sessions": w["build5"].fillna("—"),
                "Industry": w["industry"]}), "watch-near", units="near an entry · nearest first",
                show_index=False, label_col="Symbol")
        else:
            ui.render_empty_state("Nobody close", "No stock is moving toward an entry.", eyebrow="Watchlist")
        st_ = scr[scr["state"] == "STOPPED"]
        if len(st_):
            ui.render_section_header("Stopped Out, Waiting to Re-arm", "A stopped position does not re-enter while its signal "
                                     "stays strong. It re-arms only after the score leaves the band (below 85% for a long, "
                                     "above 15% for a short); a later signal is then a new trade.", icon="alert-triangle")
            ui.render_table_panel(pd.DataFrame({
                "Symbol": st_["symbol"], "Was": st_["stopped_side"].fillna("—").str.lower(),
                "Stopped on": st_["stopped_on"].map(fmt_date), "Score rank now": [rank(v) for v in st_["score"]],
                "Re-arms when the rank goes": ["below 85%" if sd == "LONG" else ("above 15%" if sd == "SHORT" else "—")
                                               for sd in st_["stopped_side"]],
                "Industry": st_["industry"]}), "watch-stopped", units="stopped out", show_index=False, label_col="Symbol")

    with tabs[3]:
        fl = scr.dropna(subset=["fut_ret1"])
        up, dn = int((fl["fut_ret1"] > 0.005).sum()), int((fl["fut_ret1"] < -0.005).sum())
        ui.render_section_header("Flow", "Market context from the stock futures and options. None of it is a signal: "
                                 "build-up and option flow were tested and did not predict returns.", icon="activity")
        ui.render_kpi_strip([
            dict(label="Stock futures today", value=f"{up} ▲ · {dn} ▼", subtext=f"moved more than 0.5%; {len(fl) - up - dn} flat, of {len(fl)}"),
            dict(label="Long build-up today", value=str(int((fl['build1'] == 'Long build-up').sum())), subtext="price up, OI up",
                 color_class="success"),
            dict(label="Short build-up today", value=str(int((fl['build1'] == 'Short build-up').sum())), subtext="price down, OI up",
                 color_class="danger"),
            dict(label="Short covering / long unwinding today", value=f"{int((fl['build1'] == 'Short covering').sum())} / "
                                                                      f"{int((fl['build1'] == 'Long unwinding').sum())}",
                 subtext="price up, OI down / price down, OI down"),
        ], key="flow-kpis")
        ui.render_section_header("Sectors", "By industry: the median futures move, build-up counts over five sessions, the "
                                 "median score rank (50% is neutral; above it the options crowd leans to puts), and the "
                                 "book's positions.", icon="grid")
        g = fl.groupby("industry")
        sect = pd.DataFrame({
            "Names": g.size(), "Move today": g["fut_ret1"].median().map(pct),
            "Move, 5 sessions": g["fut_ret5"].median().map(pct),
            "Long build-up": g["build5"].apply(lambda s: int((s == "Long build-up").sum())),
            "Short build-up": g["build5"].apply(lambda s: int((s == "Short build-up").sum())),
            "Score rank": g["score"].median().map(rank),
            "Book ▲ / ▼": g["state"].apply(lambda s: f"{int(s.isin(['LONG', '▲ NEW']).sum())} / {int(s.isin(['SHORT', '▼ NEW']).sum())}"),
        }).sort_values("Names", ascending=False).reset_index().rename(columns={"industry": "Industry"})
        ui.render_table_panel(sect, "flow-sectors", units="stock futures by industry", show_index=False,
                              label_col="Industry", max_height=420,
                              sign_color_cols={"Move today", "Move, 5 sessions"})
        ui.render_section_header("Futures Build-Up, Five Sessions", "How many stock futures are in each state over the "
                                 "last five sessions.", icon="activity")
        order_b = ["Long build-up", "Short covering", "Neutral", "Long unwinding", "Short build-up"]
        bc = fl["build5"].value_counts().reindex(order_b).fillna(0)
        colors = {"Long build-up": "emerald", "Short covering": "cyan", "Neutral": "slate", "Long unwinding": "amber", "Short build-up": "rose"}
        fig = go.Figure(go.Bar(x=bc.values, y=bc.index, orientation="h", marker_color=[chart_color(colors[k]) for k in bc.index],
                               hovertemplate="%{y}: %{x:.0f} stock futures<extra></extra>"))
        fig.update_layout(**chart_layout(height=260, show_legend=False))
        style_axes(fig)
        fig.update_yaxes(autorange="reversed")          # read top to bottom in the order listed
        ui.render_chart_panel(fig, "flow-build", units="stock futures by state")
        ui.render_section_header("Option Flow", "Today's stock options, read against yesterday: buyers vs writers opening, "
                                 "unusual volume (volume above open interest) and call-vs-put volume surges. Names in the book "
                                 "or on tonight's orders first, then the largest surges. Context only.", icon="activity")
        of = scr.dropna(subset=["buyer_tilt"]).copy()
        mine = of["state"].isin(["LONG", "SHORT", "▲ NEW", "▼ NEW", "EXIT"])
        of = pd.concat([of[mine], of[~mine].reindex(of[~mine]["surge_tilt"].abs().sort_values(ascending=False).index).head(25)])
        ui.render_table_panel(pd.DataFrame({
            "Symbol": of["symbol"], "In VAAYDO": of["state"].replace({"—": ""}),
            "Call − put buying (% of OI)": [pct(v, True, 2) for v in of["buyer_tilt"]],
            "Put − call writing (% of OI)": [pct(v, True, 2) for v in of["writer_tilt"]],
            "Unusual call − put (% of volume)": [pct(v, True, 0) for v in of["unusual_tilt"]],
            "Call vs put volume surge": [f"{np.exp(v):.2f}×" if np.isfinite(v) else "—" for v in of["surge_tilt"]]}),
            "flow-options", units="stock options against yesterday", show_index=False, label_col="Symbol", max_height=520)

    with tabs[4]:
        ui.render_section_header("Option Contracts", "Every stock option that traded today, read against its previous session.",
                                 icon="list")
        cf = uni_filter(r.contracts, members)
        if cf is None or cf.empty:
            ui.render_empty_state("No contract data", "The previous session's file was not available.", eyebrow="Contracts")
        else:
            contracts_scanner(cf)

    with tabs[5]:
        ui.render_section_header("Every Stock", "Ordered by state, then score. The score rank is the 10-session average of the "
                                 "daily positioning rank (that is what trades); today's rank is the single day. 'Event priced': "
                                 "the nearest expiry's IV is 8%+ above the next one's, usually results due.", icon="grid")
        ui.render_table_panel(pd.DataFrame({
            "Symbol": scr["symbol"], "State": scr["state"], "Score rank (10-session)": [rank(v) for v in scr["score"]],
            "Today's rank (1 session)": [rank(v) if np.isfinite(v) else "—" for v in scr["opi_today"]], "Put/call vol": scr["pcr_vol"].round(2),
            "Put/call OI": scr["pcr_oi"].round(2), "Put skew (vol pts)": [volpts(v) for v in scr["skew"]],
            "Call skew (vol pts)": [volpts(v) for v in scr["cskew"]], "IV": [pct(v, False, 0) for v in scr["atm_iv"]],
            "Event priced": ["yes" if e else "" for e in scr["event"]], "Futures 1d": scr["build1"].fillna("—"),
            "Futures 5d": scr["build5"].fillna("—"), "Put wall": scr["wall_put"].round(0), "Call wall": scr["wall_call"].round(0),
            "Delivery": [f"{v:.0f}%" if np.isfinite(v) else "—" for v in scr["deliv_pct"]],
            "Risk": scr["risk_regime"].fillna("—"), "Industry": scr["industry"]}),
            "screen-all", units=f"{len(scr)} F&O stocks", show_index=False, label_col="Symbol", max_height=640,
            col_precision={"Put wall": 0, "Call wall": 0})
        ui.render_note("A dash in today's rank means that stock's input was missing for the day; its 10-session score "
                       "averages the days it has.")

    with tabs[6]:
        sr = lambda x: x.mean() / x.std() * np.sqrt(252)
        ui.render_section_header(f"Track Record, 2019 – {r.asof:%d %b %Y}", "Backtest on real NSE futures prices: next-open fills, "
                                 "all charges and slippage, trailing stops, rolls, corporate actions, F&O bans, renames. Returns on "
                                 "gross exposure: one lot of every name (lots differ in rupee size, so the book is not exactly "
                                 "balanced), and, for reference, equal rupees in every name. Shaded: the validation period, which the score "
                                 "was not designed on.", icon="trending-up")
        fig = go.Figure()
        for col, nm in TRACK_BOOKS:
            fig.add_trace(go.Scatter(x=T.index, y=(1 + T[col]).cumprod(), mode="lines", name=nm))
        fig.add_vrect(x0="2023-01-01", x1="2024-06-30", fillcolor=chart_color("amber"), opacity=0.08, line_width=0,
                      annotation_text="validation period", annotation_position="top left")
        fig.update_layout(**chart_layout(height=340))
        style_axes(fig)
        ui.render_chart_panel(fig, "track", context=f"2019 – {r.asof:%d %b %Y} · growth of ₹1 of gross exposure")
        summ = []
        for col, nm in TRACK_BOOKS:
            x = T[col]
            eq = (1 + x).cumprod()
            yrs = (1 + x).groupby(x.index.year).prod() - 1
            summ.append({"Book": nm, "Sharpe": round(sr(x), 2), "Return / yr": pct(x.mean() * 252), "Volatility": pct(x.std() * np.sqrt(252), False),
                         "Worst drawdown": pct((eq / eq.cummax() - 1).min()), "Losing years": f"{int((yrs < 0).sum())} of {len(yrs)}"})
        ui.render_table_panel(pd.DataFrame(summ), "track-summary", context=f"2019 – {r.asof:%d %b %Y} · net of costs",
                              show_index=False, label_col="Book")
        ui.render_section_header("Every Closed Trade Since July 2024", "One lot each, from the same engine as tonight's orders, "
                                 "after charges and slippage. Open positions are on the Book tab.", icon="list")
        tr = uni_filter(r.trades, members)
        if len(tr):
            trade_kpis(tr, "trades-kpis")
        ui.render_section_header("Exports", "The screen, the positions and every contract leg, as of this run.", icon="download")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Screen (CSV)", scr.to_csv(index=False).encode(), f"vaaydo_screen_{r.asof:%Y%m%d}.csv", width="stretch")
        c2.download_button("Positions (CSV)", book_pos.to_csv(index=False).encode(), f"vaaydo_positions_{r.asof:%Y%m%d}.csv", width="stretch")
        c3.download_button("Legs (CSV)", book_legs.to_csv(index=False).encode(), f"vaaydo_legs_{r.asof:%Y%m%d}.csv", width="stretch")
        ui.render_section_header("Definitions", icon="info")
        ui.render_kv_table({
            "Score rank": "the 10-session average of the day's option-positioning rank (put/call volume and OI, put skew, "
                          "call skew, call−put IV, net option OI added) across all F&O stocks; 100% leans hardest to puts",
            "Today's rank": "the same positioning rank for one session only (what the score averages)",
            "▲ / ▼ NEW": "score entered the top / bottom 5% at this close; enter at the next open",
            "Held": "while the score stays in the top / bottom 15%",
            "Stop": "trailing: 1.5 × Pragyam's 20%-touch stop distance from the best close, decided on the close, filled next open",
            "STOPPED": "stopped out; no re-entry until the score leaves the band",
            "Near entry (READY)": "rank inside 85–95% (or 5–15%) and moved toward the entry line over five sessions",
            "Quantity": "shares held: one lot at entry; a roll takes the new contract's lot; a split or bonus rescales it; "
                        "NSE revising the lot of a running contract does not change it",
            "Risk to stop": "rupees lost from today's close if the position exits at its stop",
            "Range, 10 sessions (80%)": "Pragyam's calibrated band: the price stayed inside it 80% of the time over 10 sessions",
            "Build-up": "futures price and open interest together: up/up long build-up, down/up short build-up, up/down "
                        "short covering, down/down long unwinding",
            "Event priced": "the nearest expiry's ATM IV is 8%+ above the next expiry's: the options price a known event",
            "Contract": "shown with its expiry date: the nearest expiry with more than 3 sessions left on the fill day (expiries "
                        "moved for holidays are taken as NSE lists them); rolled at the close 3 sessions before expiry",
            "Corporate actions": "splits, bonuses and demergers rescale entry, stop and lot by the exchange's factor",
            "Renamed stocks": "a renamed symbol (ZOMATO → ETERNAL, …) is one continuous position and signal; past legs show "
                              "the name they traded under",
            "F&O ban": "no new position and no roll while a stock is in the ban; a held contract rolls on the first "
                       "unbanned session, or settles at expiry if the ban outlasts it",
            "Leaving F&O": "if NSE lists no next contract when a roll falls due, the position closes at that close",
            "Stop with no Pragyam level": "the last level within 5 sessions, else the day's median stop distance",
            "Book after next open": "positions kept plus tomorrow's new entries; exits leave at that open",
        }, header_left="Term", header_right="Meaning")
        ui.render_note("NSE F&O bhavcopy, end of day. Decision support, not investment advice: futures can lose "
                       "more than the margin posted.")

# ── Historical Range ────────────────────────────────────────────────────────
else:
    s0, s1 = pd.Timestamp(meta.get("start") or date(2024, 7, 8)), r.asof
    tr = uni_filter(r.trades, members)
    # trades REALISED in the range (by exit date): grouping by signal date would leave the newest
    # months with only the trades that already closed, mostly quick stop-outs
    tr = tr[(pd.to_datetime(tr["exit"]) >= s0) & (pd.to_datetime(tr["exit"]) <= s1)] if len(tr) else tr
    still = book_pos[book_pos["action"] == "HOLD"] if len(book_pos) else book_pos
    tabs = st.tabs([f"Trades · {len(tr)}", "By Month", "Track Record"])
    with tabs[0]:
        ui.render_section_header(f"Trades Closed {s0:%d %b %Y} – {s1:%d %b %Y}",
                                 f"By exit date. Still open at {s1:%d %b %Y} and not counted: {len(still)} positions, "
                                 f"{inr(still['pnl'].sum() if len(still) else 0, True)} marked to that close.", icon="list")
        if tr is None or tr.empty:
            ui.render_empty_state("No trades closed in the range", "Widen the range.", eyebrow="Harvest")
        else:
            trade_kpis(tr, "hr-kpis", span=f"closed {s0:%d %b %y} – {s1:%d %b %y}")
            t = tr.sort_values("exit", ascending=False)
            ui.render_table_panel(pd.DataFrame({
                "Symbol": [traded_as(a, b) for a, b in zip(t["symbol"], t["entry"])], "Side": t["side"],
                "Entry": [f"{px(a)} · {fmt_date(b)} · {fut(c)}" for a, b, c in zip(t["entry_px"], t["entry"], t["entry_contract"])],
                "Exit": [f"{px(a)} · {fmt_date(b)} · {fut(c)}" for a, b, c in zip(t["exit_px"], t["exit"], t["exit_contract"])],
                "Sessions": t["sessions"], "Legs": t["legs"], "Return": t["ret"].map(pct),
                "Rupees, one lot": [inr(v, True) for v in t["pnl_rs"]], "Exit reason": t["why"]}),
                "hr-trades", units="fill to exit across rolls · after charges", show_index=False, label_col="Symbol",
                max_height=600, sign_color_cols={"Return", "Rupees, one lot"})
    with tabs[1]:
        ui.render_section_header("P&L Realised Each Month", "Trades closed that month, one lot each, after charges.",
                                 icon="bar-chart")
        if tr is not None and len(tr):
            m = tr.assign(month=pd.to_datetime(tr["exit"]).dt.to_period("M").astype(str)).groupby("month")
            mm = pd.DataFrame({"Trades closed": m.size(), "Won": m["ret"].apply(lambda x: (x > 0).mean()),
                               "Rupees, one lot each": m["pnl_rs"].sum(), "Average trade": m["ret"].mean()}).reset_index()
            fig = go.Figure(go.Bar(x=mm["month"], y=mm["Rupees, one lot each"] / 1e5,
                                   marker_color=[chart_color("emerald") if v >= 0 else chart_color("rose") for v in mm["Rupees, one lot each"]],
                                   hovertemplate="%{x}: ₹%{y:.2f} L<extra></extra>"))
            fig.update_layout(**chart_layout(height=300, show_legend=False))
            style_axes(fig)
            ui.render_chart_panel(fig, "hr-month", units="₹ lakh")
            ui.render_table_panel(pd.DataFrame({"Month": mm["month"], "Trades closed": mm["Trades closed"],
                                                "Won": mm["Won"].map(lambda v: pct(v, False, 0)),
                                                "Rupees, one lot each": mm["Rupees, one lot each"].map(lambda v: inr(v, True)),
                                                "Average trade": mm["Average trade"].map(lambda v: pct(v, True, 2))}),
                                  "hr-month-table", units="by exit month", show_index=False, label_col="Month",
                                  sign_color_cols={"Rupees, one lot each", "Average trade"})
        else:
            ui.render_empty_state("No trades closed in the range", "Widen the range.", eyebrow="By month")
    with tabs[2]:
        ui.render_section_header("The Book Over the Range", "Growth of ₹1 of gross exposure, net of costs: one lot of every "
                                 "name, and equal rupees in every name for reference.", icon="trending-up")
        x = T[(T.index >= s0) & (T.index <= s1)]
        fig = go.Figure()
        for col, nm in TRACK_BOOKS:
            fig.add_trace(go.Scatter(x=x.index, y=(1 + x[col]).cumprod(), mode="lines", name=nm))
        fig.update_layout(**chart_layout(height=340))
        style_axes(fig)
        ui.render_chart_panel(fig, "hr-track", units="growth of ₹1")
        summ = []
        for col, nm in TRACK_BOOKS:
            y = x[col]
            if len(y) > 20:
                eq = (1 + y).cumprod()
                summ.append({"Book": nm, "Sharpe": round(y.mean() / y.std() * np.sqrt(252), 2), "Return / yr": pct(y.mean() * 252),
                             "Return over the range": pct(eq.iloc[-1] - 1), "Worst drawdown": pct((eq / eq.cummax() - 1).min())})
        if summ:
            ui.render_table_panel(pd.DataFrame(summ), "hr-track-summary", units="net of costs", show_index=False,
                                  label_col="Book")

footer()
