"""
VAAYDO — F&O trade curator.            streamlit run app.py

Each evening after NSE publishes (≈ 18:00 IST): tomorrow's stock-futures orders for your
capital, chosen by where the options market is positioned on every F&O stock.
"""
from __future__ import annotations

import html
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
.block-container {{padding-top: 2rem; max-width: 1560px;}}
.kpi {{border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .65rem .9rem; height: 100%;}}
.kpi .l {{font-size: .66rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED};}}
.kpi .v {{font-size: 1.3rem; font-weight: 700; font-variant-numeric: tabular-nums;}}
.kpi .s {{font-size: .74rem; color: {MUTED};}}
.brand {{font-weight: 800; font-size: 1.6rem; color: {GOLD}; letter-spacing: -.02em;}}
.note {{border-left: 4px solid {GOLD}; padding: .6rem 1rem; background: rgba(217,164,0,.07); border-radius: 6px; font-size:.9rem;}}
.bl {{width: 100%; overflow-x: auto; margin-bottom: .4rem;}}
.bl table {{width: 100%; border-collapse: collapse; font-size: .86rem; font-variant-numeric: tabular-nums;}}
.bl th {{text-align: right; font-weight: 600; font-size: .68rem; text-transform: uppercase; letter-spacing: .05em;
        color: {MUTED}; padding: .45rem .6rem; border-bottom: 1px solid rgba(128,128,128,.35); white-space: nowrap;}}
.bl th.l, .bl td.l {{text-align: left;}}
.bl td {{text-align: right; padding: .5rem .6rem .1rem; white-space: nowrap; vertical-align: top;}}
.bl tr.ctx td {{text-align: left; padding: 0 .6rem .55rem; color: {MUTED}; font-size: .76rem; white-space: normal;
               border-bottom: 1px solid rgba(128,128,128,.18);}}
.bl .sym {{font-weight: 700;}} .bl .sub {{display:block; font-size: .72rem; color: {MUTED};}}
.pill {{display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.04em; padding:.1rem .45rem; border-radius:4px;}}
.pill.long {{background: rgba(22,163,74,.14); color: {GREEN};}}
.pill.short {{background: rgba(220,38,38,.12); color: {RED};}}
.pos {{color: {GREEN};}} .neg {{color: {RED};}}
.sect {{display:flex; justify-content:space-between; align-items:baseline; margin: 1.1rem 0 .2rem;}}
.sect h4 {{margin:0;}} .sect span {{color: {MUTED}; font-size: .85rem;}}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>",
                 unsafe_allow_html=True)


def inr(x, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    s = "−" if x < 0 else ("+" if sign and x > 0 else "")
    x = abs(x)
    if x >= 1e7:
        return f"{s}₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"{s}₹{x / 1e5:.2f} L"
    return f"{s}₹{x:,.0f}"


def px(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:,.2f}"


def pct(x, sign=True) -> str:
    return "—" if x is None or not np.isfinite(x) else (f"{x:+.1%}" if sign else f"{x:.1%}")


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
                         help="Long notional + short notional, as a multiple of capital. 1.0× is the tested default; "
                              "returns, volatility and drawdowns all scale with it.")
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


with st.spinner("Reading every F&O option chain, scoring positioning, replaying the book… (first run ~1–2 min)"):
    try:
        prep = prepared(date.today().isoformat(), margin_rate)
    except Exception as e:  # network or NSE hiccup: say so plainly
        st.error(f"Could not build today's curation: {e}")
        st.stop()
O = curator.orders(prep, capital, leverage)
yours, model = O.yours, O.model
ctx = prep.context.set_index("symbol") if len(prep.context) else pd.DataFrame()
held = yours[yours["action"] != "EXIT"] if len(yours) else yours
K = len(held)

TR = track()
bins = {6: "k6", 12: "k12", 20: "k20", 30: "model_30"}
kcol = bins[min(bins, key=lambda b: abs(b - max(K, 1)))]


def stats(r: pd.Series, lev: float = 1.0) -> dict:
    r = r * lev
    eq = (1 + r).cumprod()
    yrs = (1 + r).groupby(r.index.year).prod() - 1
    return dict(ret=r.mean() * 252, vol=r.std() * np.sqrt(252), sr=r.mean() / r.std() * np.sqrt(252),
                dd=(eq / eq.cummax() - 1).min(), losing=int((yrs < 0).sum()), years=len(yrs))


ST = {c: stats(TR[c], 1.0) for c in TR}
S_you = stats(TR[kcol], leverage)
gl = held.loc[held["side"] == "LONG", "notional"].sum() if K else 0.0
gs = held.loc[held["side"] == "SHORT", "notional"].sum() if K else 0.0


def blotter(df: pd.DataFrame, kind: str) -> str:
    """Order tickets: one aligned row per position, reason and context underneath."""
    head = ("<tr><th class='l'>Instrument</th><th class='l'>Side</th><th>Qty</th><th>Entry</th><th>Last</th>"
            "<th>P&amp;L</th><th>Stop</th><th>Risk to stop</th><th>10-session range</th><th>Margin</th></tr>")
    body = []
    for _, r in df.iterrows():
        side = r["side"].lower()
        if kind == "ENTER":
            entry = "<span class='sub'>at next open</span>"
            pnl = "<span class='sub'>—</span>"
        else:
            entry = f"{px(r['entry_price'])}<span class='sub'>{pd.Timestamp(r['entry_date']):%d %b %y}</span>" \
                if r["entry_date"] is not None else "<span class='sub'>pending</span>"
            cls = "pos" if (r["pnl"] or 0) >= 0 else "neg"
            pnl = f"<span class='{cls}'>{inr(r['pnl'], True)}</span><span class='sub'>{pct(r['pnl_pct'])}</span>"
        stop = f"{px(r['stop'])}<span class='sub'>{pct(r['stop_from_last'])} from last</span>"
        risk = f"{inr(r['risk_to_stop'])}<span class='sub'>{pct(r['risk_pct_capital'], False)} of capital</span>"
        rng = f"{num(r['range_lo'], '{:,.0f}')}–{num(r['range_hi'], '{:,.0f}')}<span class='sub'>80% band</span>"
        qty = f"1 lot<span class='sub'>{int(r['lot']):,} units</span>"
        inst = f"<span class='sym'>{html.escape(r['symbol'])}</span><span class='sub'>FUT {pd.Timestamp(r['contract']):%b %y}</span>"
        body.append(f"<tr><td class='l'>{inst}</td><td class='l'><span class='pill {side}'>{r['side']}</span></td>"
                    f"<td>{qty}</td><td>{entry}</td><td>{px(r['last'])}</td><td>{pnl}</td><td>{stop}</td>"
                    f"<td>{risk}</td><td>{rng}</td><td>{inr(r['margin'])}<span class='sub'>{inr(r['notional'])} notional</span></td></tr>")
        cx = ctx.loc[r["symbol"]] if r["symbol"] in ctx.index else None
        bits = [html.escape(str(r["why"]))]
        if cx is not None:
            if np.isfinite(cx.get("pcr_vol", np.nan)):
                bits.append(f"put/call vol {cx['pcr_vol']:.2f} · OI {num(cx.get('pcr_oi'))}")
            if np.isfinite(cx.get("atm_iv", np.nan)):
                bits.append(f"IV {cx['atm_iv']:.0%}, put skew {num(cx.get('skew'), '{:+.1%}')}")
            if isinstance(cx.get("build5"), str):
                bits.append(f"futures 5d: {cx['build5']}")
            if np.isfinite(cx.get("deliv_pct", np.nan)):
                bits.append(f"delivery {cx['deliv_pct']:.0f}%")
            if bool(cx.get("event")):
                bits.append("⚠ event priced in (IV term inverted)")
        if isinstance(r.get("risk_regime"), str):
            bits.append(f"risk {r['risk_regime']}")
        body.append(f"<tr class='ctx'><td colspan='10'>{' · '.join(bits)}</td></tr>")
    return f"<div class='bl'><table>{head}{''.join(body)}</table></div>"


def section(df, kind, title):
    sub = df[df["action"] == kind]
    if not len(sub):
        return
    tot = []
    if kind != "ENTER" and sub["pnl"].notna().any():
        tot.append(f"P&L {inr(sub['pnl'].sum(), True)}")
    if kind != "EXIT":
        tot.append(f"risk to stops {inr(sub['risk_to_stop'].sum())}")
    tot.append(f"notional {inr(sub['notional'].sum())}")
    st.markdown(f"<div class='sect'><h4>{title} · {len(sub)}</h4><span>{' · '.join(tot)}</span></div>", unsafe_allow_html=True)
    st.markdown(blotter(sub, kind), unsafe_allow_html=True)


tabs = st.tabs(["Orders", "Book", "Name", "Track record", "How it works"])

# ── Orders ───────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Orders for the session after {prep.asof:%a %d %b %Y}")
    c = st.columns(6)
    kpi(c[0], "Your book", f"{K} names", f"{int((held['side'] == 'LONG').sum()) if K else 0} long · "
                                         f"{int((held['side'] == 'SHORT').sum()) if K else 0} short")
    kpi(c[1], "Gross exposure", inr(gl + gs), f"{(gl + gs) / capital:.2f}× capital · net {inr(gl - gs, True)}")
    open_pnl = held["pnl"].sum() if K else 0.0
    kpi(c[2], "Open P&L", inr(open_pnl, True), f"{open_pnl / capital:+.1%} of capital")
    rsk = held["risk_to_stop"].sum() if K else 0.0
    kpi(c[3], "Risk to stops", inr(rsk), f"{rsk / capital:.1%} of capital if every stop hit")
    kpi(c[4], "Backtest at this size", f"{S_you['ret']:+.1%}/yr", f"Sharpe {S_you['sr']:.2f} · vol {S_you['vol']:.1%}")
    kpi(c[5], "Worst backtest drawdown", f"{S_you['dd']:.1%}",
        f"{S_you['losing']} of {S_you['years']} years lost · {kcol.replace('k', '').replace('model_', '')} names, {leverage:.2g}×")
    if K < 10:
        k6, k12 = ST["k6"], ST["k12"]
        st.write("")
        st.markdown(f"<div class='note'><b>Book too small to be the strategy.</b> At {inr(capital)} and {leverage:.2g}× "
                    f"your book holds {K} names. Six names backtested at Sharpe {k6['sr']:.2f} with a worst drawdown of "
                    f"{k6['dd']:.0%}; twelve names at {k12['sr']:.2f} and {k12['dd']:.0%}. The edge comes from holding "
                    f"enough names for stock-specific noise to wash out.</div>", unsafe_allow_html=True)
    if len(yours):
        n_ord = int((yours["action"] != "HOLD").sum())
        if n_ord == 0:
            st.markdown("<div class='sect'><h4>No orders tomorrow</h4><span>hold every position</span></div>",
                        unsafe_allow_html=True)
        section(yours, "EXIT", "Exit at the open")
        section(yours, "ENTER", "Enter at the open")
        section(yours, "HOLD", "Hold")
    else:
        st.info("Your capital does not fit one lot at this leverage. Raise capital or leverage.")
    st.caption("Entries and exits are at the next session's open, as in the backtest. Stops trail the best close: "
               "1.5× Pragyam's 20%-touch distance, checked on the close, exit at the next open; a stopped name "
               "stays out until its signal resets. P&L excludes charges. Not investment advice.")

# ── Book ─────────────────────────────────────────────────────────────────────
with tabs[1]:
    c = st.columns(5)
    kpi(c[0], "Long notional", inr(gl))
    kpi(c[1], "Short notional", inr(gs))
    kpi(c[2], "Margin (est.)", inr(held["margin"].sum() if K else 0), f"{(held['margin'].sum() if K else 0) / capital:.0%} of capital")
    med_l = held.loc[held["side"] == "LONG", "last"].median() if K and (held["side"] == "LONG").any() else np.nan
    med_s = held.loc[held["side"] == "SHORT", "last"].median() if K and (held["side"] == "SHORT").any() else np.nan
    kpi(c[3], "Median price, long / short", f"₹{num(med_l, '{:,.0f}')} / ₹{num(med_s, '{:,.0f}')}",
        "shorts lean to cheap retail favourites")
    kpi(c[4], "Within 5% of stop", int((held["stop_from_last"].abs() < 0.05).sum()) if K else 0, "positions")
    st.markdown("#### Model book")
    st.caption("The full strategy at any capital: long when a stock's smoothed positioning score is in the top 5% of "
               "F&O stocks, held while in the top 15%; short for the bottom. Your book is the subset that fits your capital.")
    if len(model):
        section(model, "EXIT", "Exit at the open")
        section(model, "ENTER", "Enter at the open")
        section(model, "HOLD", "Hold")

# ── Name ─────────────────────────────────────────────────────────────────────
with tabs[2]:
    names = list(model["symbol"]) if len(model) else []
    sym = st.selectbox("Name", names, index=0) if names else None
    if sym:
        r = model.set_index("symbol").loc[sym]
        cx = ctx.loc[sym] if sym in ctx.index else pd.Series(dtype=float)
        c = st.columns(6)
        kpi(c[0], "Position", r["side"], f"{r['action']} · {r['why']}")
        kpi(c[1], "Entry", px(r["entry_price"]),
            f"{pd.Timestamp(r['entry_date']):%d %b %y}" if r["entry_date"] is not None else "at next open")
        kpi(c[2], "P&L per lot", inr(r["pnl"], True), pct(r["pnl_pct"]))
        kpi(c[3], "Stop", px(r["stop"]), f"{pct(r['stop_from_last'])} from last")
        kpi(c[4], "Put/call vol · OI", f"{num(cx.get('pcr_vol'))} · {num(cx.get('pcr_oi'))}",
            f"IV {num(cx.get('atm_iv'), '{:.0%}')} · put skew {num(cx.get('skew'), '{:+.1%}')}")
        kpi(c[5], "Futures, 5 sessions", str(cx.get("build5", "—")),
            f"move {num(cx.get('fut_ret5'), '{:+.1%}')} · OI {num(cx.get('fut_oi5'), '{:+.1%}')}")
        a, b = st.columns(2)
        with a:
            sc = prep.scores[prep.scores["symbol"] == sym].tail(120)
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
                k = h["close"].iloc[-1] / r["last"]              # futures -> adjusted spot scale
                fig = go.Figure(go.Candlestick(x=h["date"], open=h["open"], high=h["high"], low=h["low"], close=h["close"], name=sym))
                if np.isfinite(r["entry_price"]) and r["entry_date"] is not None:
                    fig.add_hline(y=r["entry_price"] * k, line_dash="dot", line_color=MUTED, annotation_text="entry")
                if np.isfinite(r["stop"]):
                    fig.add_hline(y=r["stop"] * k, line_dash="dash", line_color=RED, annotation_text="trailing stop")
                if np.isfinite(r["range_lo"]):
                    x0 = pd.Timestamp(h["date"].iloc[-1])
                    fig.add_shape(type="rect", x0=x0, x1=x0 + pd.Timedelta(days=15), y0=r["range_lo"] * k,
                                  y1=r["range_hi"] * k, fillcolor=GOLD, opacity=.25, line_width=0)
                    fig.add_annotation(x=x0 + pd.Timedelta(days=7), y=r["range_hi"] * k, text="next 10 sessions, 80%",
                                       showarrow=False, yshift=10, font=dict(size=10))
                fig.update_layout(xaxis_rangeslider_visible=False)
                st.plotly_chart(style(fig, 330, "Price (adjusted spot) with entry, stop and range"), width="stretch")
            except Exception as e:
                st.caption(f"Price history unavailable: {e}")
        st.markdown("**Today's option-market readings** — ranked across all F&O stocks; higher put/call, steeper put skew, "
                    "flatter call skew and puts being added all push the score up (bullish).")
        st.dataframe(pd.DataFrame({k_: [cx.get(k_, np.nan)] for k_ in ("pcr_vol", "pcr_oi", "skew", "cskew", "cpiv",
                                                                       "put_oi_flow", "call_oi_flow", "atm_iv", "term")}),
                     hide_index=True, width="stretch")

# ── Track record ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("What this strategy did, 2019 – today")
    st.caption("Backtest on real NSE futures prices: next-open execution, all statutory charges and slippage, trailing "
               "stops decided on the close and executed next open, rolls 3 sessions before expiry. Returns on gross "
               "exposure (long + short), shown at your leverage.")
    fig = go.Figure()
    for col, nm, w in (("model_30", "model book (~30)", 3), ("k20", "20 names", 1.5), ("k12", "12 names", 1.5), ("k6", "6 names", 1.5)):
        e = (1 + TR[col] * leverage).cumprod()
        fig.add_trace(go.Scatter(x=e.index, y=e, mode="lines", name=nm, line=dict(width=w)))
    fig.add_vrect(x0="2023-01-01", x1="2024-06-30", fillcolor=MUTED, opacity=.08, line_width=0, annotation_text="validation")
    st.plotly_chart(style(fig, 380, f"Growth of ₹1 at {leverage:.2g}× gross"), width="stretch")
    summ = pd.DataFrame({nm: stats(TR[c], leverage) for c, nm in (("model_30", "model book"), ("k20", "20 names"),
                                                                  ("k12", "12 names"), ("k6", "6 names"))}).T
    summ = summ.rename(columns={"ret": "return / yr", "vol": "volatility", "sr": "Sharpe", "dd": "worst drawdown",
                                "losing": "losing years", "years": "years"})
    st.dataframe(summ, width="stretch", column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                                       ("return / yr", "volatility", "worst drawdown")} |
                 {"Sharpe": st.column_config.NumberColumn(format="%.2f")})
    yr = (TR * leverage).groupby(TR.index.year).apply(lambda x: (1 + x).prod() - 1)
    yr = yr.rename(columns={"model_30": "model book", "k20": "20 names", "k12": "12 names", "k6": "6 names"})
    st.dataframe(yr, width="stretch", column_config={c: st.column_config.NumberColumn(format="percent") for c in yr.columns})
    st.markdown(f"""<div class='note'><b>Read this before trading it.</b> The signal was designed on 2024–25 data and then
    confirmed on 2019–2022, which no design choice had seen. It was flat from 2023 to mid-2024 and is weak so far in 2026.
    The model book backtested at Sharpe {ST['model_30']['sr']:.2f}; expect less live. Pause if the book loses more than
    1.5× its worst backtest drawdown.</div>""", unsafe_allow_html=True)

# ── How it works ─────────────────────────────────────────────────────────────
with tabs[4]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "METHOD.md")
    st.markdown(open(path).read() if os.path.exists(path) else "See README.md")
