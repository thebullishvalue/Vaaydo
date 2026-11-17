"""
VAAYDO — F&O trade curator.            streamlit run app.py

Each evening after NSE publishes (≈ 18:00 IST): tomorrow's futures orders for your capital,
chosen by where the options market is positioned on every F&O stock.
"""
from __future__ import annotations

import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from vaaydo import curator
from vaaydo.data import history

st.set_page_config(page_title="VAAYDO", page_icon="⚡", layout="wide")
GOLD, GREEN, RED, MUTED = "#D9A400", "#16a34a", "#dc2626", "#8a8a8a"
st.markdown(f"""<style>
.block-container {{padding-top: 2rem; max-width: 1500px;}}
.kpi {{border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .7rem 1rem; height: 100%;}}
.kpi .l {{font-size: .68rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED};}}
.kpi .v {{font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums;}}
.kpi .s {{font-size: .74rem; color: {MUTED};}}
.brand {{font-weight: 800; font-size: 1.6rem; color: {GOLD}; letter-spacing: -.02em;}}
.order {{border: 1px solid rgba(128,128,128,.25); border-left: 5px solid var(--c); border-radius: 8px;
        padding: .65rem .9rem; margin-bottom: .55rem;}}
.order .t {{font-weight: 700; font-size: 1.02rem;}} .order .m {{font-size: .84rem; color: {MUTED};}}
.order .n {{font-variant-numeric: tabular-nums; font-size: .9rem;}}
.chip {{display:inline-block; font-size:.72rem; padding:.08rem .45rem; border-radius:10px; margin:.15rem .2rem 0 0;
       border:1px solid rgba(128,128,128,.35);}}
.note {{border-left: 4px solid {GOLD}; padding: .6rem 1rem; background: rgba(217,164,0,.07); border-radius: 6px; font-size:.9rem;}}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>",
                 unsafe_allow_html=True)


def inr(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    s, x = ("−" if x < 0 else ""), abs(x)
    if x >= 1e7:
        return f"{s}₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"{s}₹{x / 1e5:.2f} L"
    return f"{s}₹{x:,.0f}"


def num(x, fmt="{:.2f}"):
    try:
        return fmt.format(x) if np.isfinite(x) else "—"
    except (TypeError, ValueError):
        return "—"


def style(fig, h=320, title=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=55 if title else 20, b=10),
                      title=dict(text=title, y=.97) if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)")
    return fig


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='brand'>VAAYDO</div><div style='color:#8a8a8a;font-size:.8rem'>वायदो · F&O trade curator</div>",
                unsafe_allow_html=True)
    st.write("")
    capital = st.number_input("Trading capital (₹)", 500_000, 500_000_000, 8_500_000, step=500_000,
                              help="Cash set aside for this strategy.")
    leverage = st.slider("Gross exposure ÷ capital", 0.5, 2.0, 1.0, 0.25,
                         help="Long notional + short notional, as a multiple of capital. 1.0× is the tested default. "
                              "Drawdowns scale with it: the 12-name book's worst was −17% at 1×, about −34% at 2×.")
    margin_rate = st.slider("Margin per lot (% of notional)", 12, 30, 20, 1,
                            help="NSE SPAN + exposure margin for stock futures is typically 15–25%.") / 100
    st.divider()
    if st.button("Refresh NSE data", width="stretch",
                 help="Fetch the latest bhavcopy (published around 18:00 IST) and rebuild."):
        st.cache_resource.clear()
        st.rerun()
    st.caption("Data: NSE F&O and equity bhavcopy (end of day); Yahoo daily history for Pragyam's risk levels.")


@st.cache_resource(show_spinner=False, ttl=3 * 3600)
def prepared(day_key: str, margin_rate: float):
    return curator.prepare(margin_rate=margin_rate)


@st.cache_data(show_spinner=False)
def track():
    return pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "track_record.parquet"))


with st.spinner("Reading every F&O option chain, scoring positioning, rebuilding the book… (first run ~1 min)"):
    try:
        cur = prepared(date.today().isoformat(), margin_rate)
    except Exception as e:  # network or NSE hiccup: say so plainly
        st.error(f"Could not build today's curation: {e}")
        st.stop()
yours = curator.personalize(cur, capital, leverage)
acts = cur.actions
ctx = cur.context.set_index("symbol") if len(cur.context) else pd.DataFrame()
held = yours[yours["your_action"] != "EXIT"] if len(yours) else yours
K = len(held)

TR = track()
bins = {6: "k6", 12: "k12", 20: "k20", 30: "model_30"}
kcol = bins[min(bins, key=lambda b: abs(b - max(K, 1)))]


def stats(r: pd.Series, lev: float = 1.0) -> dict:
    r = r * lev
    eq = (1 + r).cumprod()
    return dict(ret=r.mean() * 252, vol=r.std() * np.sqrt(252), sr=r.mean() / r.std() * np.sqrt(252),
                dd=(eq / eq.cummax() - 1).min())


S_you = stats(TR[kcol], leverage)
gl = held.loc[held["side"] == "LONG", "notional"].sum() if K else 0.0
gs = held.loc[held["side"] == "SHORT", "notional"].sum() if K else 0.0

tabs = st.tabs(["Orders", "Book", "Name", "Track record", "How it works"])

# ── Orders ───────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Orders for the session after {cur.asof:%a %d %b %Y}")
    c = st.columns(6)
    kpi(c[0], "Your book", f"{K} names", f"{int((held['side'] == 'LONG').sum()) if K else 0} long · "
                                         f"{int((held['side'] == 'SHORT').sum()) if K else 0} short")
    kpi(c[1], "Gross exposure", inr(gl + gs), f"{(gl + gs) / capital:.2f}× capital · net {inr(gl - gs)}")
    mg = held["margin_est"].sum() if K else 0.0
    kpi(c[2], "Margin (est.)", inr(mg), f"{mg / capital:.0%} of capital")
    n_ord = int((yours["your_action"] != "HOLD").sum()) if len(yours) else 0
    kpi(c[3], "Orders", n_ord, "entries + exits for your book")
    kpi(c[4], "Backtest, this size", f"{S_you['ret']:+.1%}/yr", f"vol {S_you['vol']:.1%} · Sharpe {S_you['sr']:.2f}")
    kpi(c[5], "Worst drawdown (backtest)", f"{S_you['dd']:.1%}",
        f"{kcol.replace('k', '').replace('model_', '')}-name book at {leverage:.2g}×")
    st.write("")
    if K < 8:
        st.markdown(f"<div class='note'><b>Small book.</b> At {inr(capital)} and {leverage:.2g}× your book holds {K} names. "
                    f"Fewer names means more single-stock risk: the 6-name version's worst drawdown was −21% and it lost "
                    f"in 4 of 8 years, against −17% and 1 losing year with 12 names. About ₹85 L at 1× runs 12 names.</div>",
                    unsafe_allow_html=True)
        st.write("")

    def order_card(r, action):
        color = RED if action == "EXIT" else (GREEN if r["side"] == "LONG" else "#e67e22")
        verb = {"EXIT": f"Close {r['side'].lower()}", "ENTER": "Buy" if r["side"] == "LONG" else "Sell", "HOLD": "Hold"}[action]
        cx = ctx.loc[r["symbol"]] if r["symbol"] in ctx.index else None
        chips = []
        if cx is not None:
            if isinstance(cx.get("build5"), str):
                chips.append(f"futures 5d: {cx['build5']}")
            if np.isfinite(cx.get("pcr_vol", np.nan)):
                chips.append(f"put/call vol {cx['pcr_vol']:.2f}")
            if np.isfinite(cx.get("atm_iv", np.nan)):
                chips.append(f"IV {cx['atm_iv']:.0%}")
            if bool(cx.get("event")):
                chips.append("⚠ event priced (IV term inverted)")
            if np.isfinite(cx.get("deliv_pct", np.nan)):
                chips.append(f"delivery {cx['deliv_pct']:.0f}%")
        stop = r.get("stop", np.nan)
        lvl = ""
        if isinstance(stop, float) and np.isfinite(stop):
            lvl = f"stop {stop:,.2f} ({r['stop_dist']:.0%} away)"
            if np.isfinite(r.get("range_lo", np.nan)):
                lvl += f" · 10-session range {r['range_lo']:,.0f}–{r['range_hi']:,.0f}"
        pnl = r.get("pnl_pct", np.nan)
        pnl_s = f" · since entry {pnl:+.1%}" if isinstance(pnl, float) and np.isfinite(pnl) else ""
        st.markdown(f"""<div class='order' style='--c:{color}'>
            <div class='t'>{verb} 1 lot {r['symbol']} FUT {pd.Timestamp(r['contract']):%b %y}</div>
            <div class='n'>{int(r['lot']):,} units · ref ₹{r['ref_price']:,.2f} · notional {inr(r['notional'])} · margin ≈ {inr(r['margin_est'])}{pnl_s}</div>
            <div class='m'>{r.get('why', '')}{' · ' + lvl if lvl else ''}</div>
            <div>{''.join(f"<span class='chip'>{c}</span>" for c in chips)}</div></div>""", unsafe_allow_html=True)

    if len(yours):
        for action, title in (("EXIT", "Exit"), ("ENTER", "Enter"), ("HOLD", "Hold")):
            sub = yours[yours["your_action"] == action]
            if len(sub):
                st.markdown(f"#### {title} ({len(sub)})")
                cols = st.columns(2)
                for i, (_, r) in enumerate(sub.iterrows()):
                    with cols[i % 2]:
                        order_card(r, action)
    else:
        st.info("Your capital does not fit one lot at this leverage. Raise capital or leverage.")
    st.caption("Place orders at the next open; limit orders near the reference price are fine — the edge plays out over "
               "weeks, not ticks. Exits are decided at the close (rule) or by the disaster stop. Not investment advice.")
    with st.expander(f"Model book (all {len(acts)} names, independent of your capital)"):
        if len(acts):
            st.dataframe(acts[["action", "symbol", "side", "contract", "lot", "ref_price", "stop", "stop_dist", "range_lo",
                               "range_hi", "since", "pnl_pct", "score_pct", "why"]], hide_index=True, width="stretch",
                         column_config={"stop_dist": st.column_config.NumberColumn(format="percent"),
                                        "pnl_pct": st.column_config.NumberColumn("since entry", format="percent"),
                                        "score_pct": st.column_config.NumberColumn("score rank", format="percent")})

# ── Book ─────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Your book")
    if K:
        cols = ["your_action", "symbol", "side", "contract", "lot", "ref_price", "notional", "margin_est", "stop", "stop_dist",
                "since", "pnl_pct", "score_pct", "risk_regime"]
        st.dataframe(held[[c for c in cols if c in held]], hide_index=True, width="stretch",
                     column_config={"stop_dist": st.column_config.NumberColumn(format="percent"),
                                    "pnl_pct": st.column_config.NumberColumn("since entry", format="percent"),
                                    "score_pct": st.column_config.NumberColumn("score rank", format="percent"),
                                    "notional": st.column_config.NumberColumn(format="%.0f"),
                                    "margin_est": st.column_config.NumberColumn(format="%.0f")})
        c = st.columns(4)
        kpi(c[0], "Long notional", inr(gl))
        kpi(c[1], "Short notional", inr(gs))
        med_l = held.loc[held["side"] == "LONG", "ref_price"].median() if (held["side"] == "LONG").any() else np.nan
        med_s = held.loc[held["side"] == "SHORT", "ref_price"].median() if (held["side"] == "SHORT").any() else np.nan
        kpi(c[2], "Median price, long / short", f"₹{num(med_l, '{:,.0f}')} / ₹{num(med_s, '{:,.0f}')}",
            "shorts lean to cheap retail favourites: the 2023 risk")
        kpi(c[3], "Names within 5% of stop", int((held["stop_dist"] < 0.05).sum()) if "stop_dist" in held else 0)
    st.subheader("Model book")
    st.caption("Long when a stock's smoothed positioning score is in the top 5% of F&O stocks, held while in the top 15%; "
               "short for the bottom. Your book is the subset that fits your capital.")
    if len(acts):
        mb = acts[acts["action"] != "EXIT"].sort_values("score_pct")
        fig = go.Figure(go.Bar(x=mb["symbol"], y=mb["score_pct"] - 0.5,
                               marker_color=[GREEN if s == "LONG" else RED for s in mb["side"]]))
        fig.update_yaxes(title="score rank − 50%", tickformat=".0%")
        st.plotly_chart(style(fig, 300), width="stretch")

# ── Name ─────────────────────────────────────────────────────────────────────
with tabs[2]:
    names = list(acts["symbol"]) if len(acts) else []
    sym = st.selectbox("Name", names, index=0) if names else None
    if sym:
        r = acts.set_index("symbol").loc[sym]
        cx = ctx.loc[sym] if sym in ctx.index else pd.Series(dtype=float)
        c = st.columns(6)
        kpi(c[0], "Position", r["side"], f"{r['action']} · {r['why']}")
        kpi(c[1], "Score rank", num(r["score_pct"], "{:.0%}"), f"today's positioning {num(r['opi_today'], '{:.0%}')}")
        kpi(c[2], "Put/call volume · OI", f"{num(cx.get('pcr_vol'))} · {num(cx.get('pcr_oi'))}")
        kpi(c[3], "ATM IV", num(cx.get("atm_iv"), "{:.0%}"),
            f"put skew {num(cx.get('skew'), '{:+.1%}')} · call skew {num(cx.get('cskew'), '{:+.1%}')}")
        kpi(c[4], "Futures, 5 sessions", str(cx.get("build5", "—")),
            f"move {num(cx.get('fut_ret5'), '{:+.1%}')} · OI {num(cx.get('fut_oi5'), '{:+.1%}')}")
        kpi(c[5], "Delivery", num(cx.get("deliv_pct"), "{:.0f}%"),
            f"put wall {num(cx.get('wall_put'), '{:,.0f}')} · call wall {num(cx.get('wall_call'), '{:,.0f}')}")
        a, b = st.columns(2)
        with a:
            sc = cur.scores[cur.scores["symbol"] == sym].tail(120)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sc["date"], y=sc["opi_pct"], mode="lines", line=dict(color=MUTED, width=1), name="daily reading"))
            fig.add_trace(go.Scatter(x=sc["date"], y=sc["pct"], mode="lines", line=dict(color=GOLD, width=3), name="score rank"))
            for y, col in ((0.95, GREEN), (0.85, GREEN), (0.15, RED), (0.05, RED)):
                fig.add_hline(y=y, line_dash="dot", line_color=col, opacity=.5)
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(style(fig, 330, "Positioning score: enter beyond 5/95%, hold inside 15/85%"), width="stretch")
        with b:
            try:
                h = history.load([sym])
                h = h[h["symbol"] == sym].tail(130)
                fig = go.Figure(go.Candlestick(x=h["date"], open=h["open"], high=h["high"], low=h["low"], close=h["close"], name=sym))
                if np.isfinite(r["stop"]):
                    k = h["close"].iloc[-1] / r["ref_price"]
                    fig.add_hline(y=r["stop"] * k, line_dash="dash", line_color=RED, annotation_text="disaster stop")
                    if np.isfinite(r["range_lo"]):
                        x0 = pd.Timestamp(h["date"].iloc[-1])
                        fig.add_shape(type="rect", x0=x0, x1=x0 + pd.Timedelta(days=15), y0=r["range_lo"] * k,
                                      y1=r["range_hi"] * k, fillcolor=GOLD, opacity=.25, line_width=0)
                        fig.add_annotation(x=x0 + pd.Timedelta(days=7), y=r["range_hi"] * k, text="next 10 sessions, 80%",
                                           showarrow=False, yshift=10, font=dict(size=10))
                fig.update_layout(xaxis_rangeslider_visible=False)
                st.plotly_chart(style(fig, 330, "Price (adjusted spot) and risk levels"), width="stretch")
            except Exception as e:
                st.caption(f"Price history unavailable: {e}")
        st.markdown("**Today's option-market readings** (the score ranks these across all F&O stocks; higher put/call, "
                    "steeper put skew, flatter call skew = more put-leaning = more bullish for the score)")
        st.dataframe(pd.DataFrame({k: [cx.get(k, np.nan)] for k in ("pcr_vol", "pcr_oi", "skew", "cskew", "cpiv",
                                                                    "put_oi_flow", "call_oi_flow", "atm_iv", "term", "basis")}),
                     hide_index=True, width="stretch")

# ── Track record ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("What this strategy did, 2019 – today")
    st.caption("Backtest on real NSE futures prices: next-open execution, all statutory charges and slippage, disaster stops, "
               "rolls 3 sessions before expiry. Returns on gross exposure (long + short), shown at your leverage.")
    fig = go.Figure()
    for col, nm, w in (("model_30", "model book (~30)", 3), ("k20", "20 names", 1.5), ("k12", "12 names", 1.5), ("k6", "6 names", 1.5)):
        e = (1 + TR[col] * leverage).cumprod()
        fig.add_trace(go.Scatter(x=e.index, y=e, mode="lines", name=nm, line=dict(width=w)))
    fig.add_vrect(x0="2023-01-01", x1="2024-06-30", fillcolor=MUTED, opacity=.08, line_width=0, annotation_text="validation")
    st.plotly_chart(style(fig, 380, f"Growth of ₹1 at {leverage:.2g}× gross"), width="stretch")
    yr = (TR * leverage).groupby(TR.index.year).apply(lambda x: (1 + x).prod() - 1)
    yr = yr.rename(columns={"model_30": "model book", "k20": "20 names", "k12": "12 names", "k6": "6 names"})
    st.dataframe(yr, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in yr.columns})
    st.markdown("""<div class='note'><b>Read this before trading it.</b> The signal was designed on 2024–25 data and then
    confirmed on 2019–2022, which no design choice had seen (IC t ≈ 5). It was flat from 2023 to mid-2024 and is weak so far
    in 2026. Expect a live Sharpe nearer 0.5–0.8 than the backtest's ~1, losing years, and drawdowns of 15–20% of gross
    exposure. Discipline: pause if the book loses more than 1.5× its worst backtest drawdown.</div>""", unsafe_allow_html=True)

# ── How it works ─────────────────────────────────────────────────────────────
with tabs[4]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "METHOD.md")
    st.markdown(open(path).read() if os.path.exists(path) else "See README.md")
