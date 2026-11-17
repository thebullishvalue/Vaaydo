"""
VAAYDO — evidence-gated F&O desk.     streamlit run app.py
"""
from __future__ import annotations

import os
import warnings
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from vaaydo import engine, journal, playbook as P, portfolio, research as R
from vaaydo.config import DEFAULT
from vaaydo.data import bhavcopy, store
from vaaydo.evaluate import payoff

st.set_page_config(page_title="VAAYDO", page_icon="⚡", layout="wide")
GOLD, GREEN, RED, MUTED, BLUE = "#D9A400", "#16a34a", "#dc2626", "#8a8a8a", "#3b82f6"
st.markdown(f"""<style>
.block-container {{padding-top: 2rem; max-width: 1500px;}}
.kpi {{border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .75rem 1rem; height: 100%;}}
.kpi .l {{font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED};}}
.kpi .v {{font-size: 1.4rem; font-weight: 700; font-variant-numeric: tabular-nums;}}
.kpi .s {{font-size: .75rem; color: {MUTED};}}
.brand {{font-weight: 800; font-size: 1.6rem; color: {GOLD}; letter-spacing: -.02em;}}
.verdict {{border-left: 4px solid {GOLD}; padding: .8rem 1.1rem; border-radius: 6px;
           background: rgba(217,164,0,.08); margin: .4rem 0 1rem;}}
.verdict b {{font-size: 1.05rem;}}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def inr(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    s, x = ("−" if x < 0 else ""), abs(x)
    if x >= 1e7:
        return f"{s}₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"{s}₹{x / 1e5:.2f} L"
    return f"{s}₹{x:,.0f}"


def style(fig, h=340, title=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=70 if title else 30, b=10), title=dict(text=title, y=.97),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)")
    return fig


PCT = lambda label=None: st.column_config.NumberColumn(label, format="percent")

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='brand'>VAAYDO</div><div style='color:#8a8a8a;font-size:.8rem'>"
                "वायदो · evidence-gated F&O desk</div>", unsafe_allow_html=True)
    days = bhavcopy.available_days()
    if st.button("Sync NSE data", use_container_width=True, help="Download end-of-day files up to today (after ~18:00 IST)"):
        with st.spinner("Downloading bhavcopies…"):
            bhavcopy.sync(days[-1] if days else bhavcopy.FIRST_UDIFF_DAY, date.today())
        st.cache_data.clear()
        st.rerun()
    if not days:
        st.warning("No data yet. Click **Sync NSE data** (first run: ~2 years, ~400 MB), then run "
                   "`python -m vaaydo research` once to build the evidence.")
        st.stop()
    asof = st.selectbox("Session", days[::-1], format_func=lambda d: d.strftime("%a %d %b %Y"))
    st.divider()
    capital = st.number_input("Capital (₹)", 100_000, 100_000_000, int(DEFAULT.risk.capital), step=100_000)
    rpt = st.slider("Max loss per trade (% of capital)", 0.5, 3.0, DEFAULT.risk.risk_per_trade * 100, 0.25) / 100
    stress = st.slider("Max NIFTY-gap stress loss (% of capital)", 2.0, 20.0, DEFAULT.risk.max_stress_loss * 100, 1.0) / 100
    cfg = replace(DEFAULT, risk=replace(DEFAULT.risk, capital=float(capital), risk_per_trade=rpt, max_stress_loss=stress))
    st.caption(f"NSE F&O bhavcopy · {len(days)} sessions · {days[0]:%b %Y} – {days[-1]:%d %b %Y}")


@st.cache_data(show_spinner=False)
def evidence():
    return R.load_evidence()


@st.cache_data(show_spinner=False)
def rules_asof(d: date):
    h = evidence()
    return P.table(h, before=pd.Timestamp(d)) if len(h) else pd.DataFrame()


@st.cache_data(show_spinner="Reading every chain: smiles, liquidity, positioning, playbook matches…", max_entries=6)
def run_scan(d: date):
    return engine.scan(d, DEFAULT, rules=rules_asof(d), n_paths=8000)


@st.cache_data(show_spinner=False)
def walk():
    h = evidence()
    return P.walk_forward(h) if len(h) else pd.DataFrame()


rules = rules_asof(asof)
res = run_scan(asof)
mkt = res.market
rets = store.returns().loc[:pd.Timestamp(asof)]
book = portfolio.construct(res.candidates, rets, cfg)
plan = portfolio.plan_frame(book)
n_live = int((rules["status"] == "live").sum()) if len(rules) else 0
n_watch = int((rules["status"] == "watch").sum()) if len(rules) else 0

tabs = st.tabs(["Today", "Playbook", "Market", "Name", "Book", "How it works"])

# ── Today ────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Plan for the session after {asof:%a %d %b %Y}")
    if rules.empty:
        st.warning("No evidence yet. Run `python -m vaaydo research` to build it (one-off, ~15 minutes).")
    elif n_live == 0:
        st.markdown(f"<div class='verdict'><b>No trades.</b> None of the {len(rules)} rules in the playbook has a "
                    f"positive, statistically reliable record after costs on real NSE prices "
                    f"(t ≥ {P.T_LIVE} over ≥ {P.MIN_CYCLES} cycles, judged only on data before today). "
                    f"Trading anyway would mean paying spreads and charges for no expected return. "
                    f"{n_watch} rule(s) are on watch below.</div>", unsafe_allow_html=True)
    c = st.columns(6)
    kpi(c[0], "Live rules", n_live, f"{n_watch} on watch · {len(rules)} tested")
    kpi(c[1], "Names read", len(mkt), f"{int((mkt['n_tradable'] >= 4).sum())} with liquid chains")
    kpi(c[2], "Median ATM IV", f"{mkt['atm_iv'].median():.1%}", f"vs 1M realised {mkt['rv_1m'].median():.1%}")
    kpi(c[3], "Event flags", int(mkt["event"].sum()), "near IV > next by 8%: event priced")
    kpi(c[4], "Book max loss", inr(plan["max_loss"].sum()) if len(plan) else "—",
        f"{plan['max_loss'].sum() / capital:.1%} of capital" if len(plan) else "nothing to hold")
    st_tbl = portfolio.stress_table(book, cfg) if book else pd.Series(dtype=float)
    kpi(c[5], "Worst NIFTY-gap stress", inr(st_tbl.min()) if len(st_tbl) else "—", f"limit {stress:.0%} of capital")
    st.write("")
    if len(plan):
        st.markdown("#### Trades")
        st.dataframe(plan, hide_index=True, use_container_width=True, column_config={
            "ev_to_risk": PCT("EV / max loss"), "pop": PCT("P(profit)"), "atm_iv": PCT("ATM IV"),
            "vol_forecast": PCT("vol used")})
        pick = st.multiselect("Record as taken (adds to Book)", plan["symbol"].tolist())
        if st.button("Record", disabled=not pick):
            for p in book:
                if p.candidate.symbol in pick:
                    journal.open_position(p, asof)
            st.success(f"Recorded {len(pick)} position(s).")
    if res.watchlist:
        st.markdown("#### Watchlist: not traded")
        st.caption("Names matching a rule whose record is positive but not yet reliable. Shown so you can see what "
                   "the system is waiting on, not as recommendations. Sorted cheapest volatility first.")
        wl = pd.DataFrame([dict(symbol=c.symbol, structure=c.evaluation.structure.kind, legs=c.evaluation.structure.label,
                                expiry=c.expiry.date(), atm_iv=c.atm_iv,
                                iv_vs_1m=mkt.set_index("symbol").loc[c.symbol, "iv_rv1m"],
                                cost_per_lot=c.evaluation.entry_net, max_loss=c.evaluation.max_loss,
                                rule=c.reason) for c in res.watchlist])
        st.dataframe(wl.head(25), hide_index=True, use_container_width=True,
                     column_config={"atm_iv": PCT("ATM IV"), "iv_vs_1m": st.column_config.NumberColumn("IV / 1M RV", format="%.2f"),
                                    "cost_per_lot": st.column_config.NumberColumn("Entry ₹/lot", format="%.0f"),
                                    "max_loss": st.column_config.NumberColumn("Max loss ₹/lot", format="%.0f")})

# ── Playbook ─────────────────────────────────────────────────────────────────
with tabs[1]:
    h = evidence()
    if h.empty:
        st.info("Run `python -m vaaydo research` to build the evidence.")
    else:
        st.subheader("What has actually worked on real NSE prices")
        st.caption("Every liquid name, every cycle, fixed-rule structures entered at the session close ± estimated spread, "
                   "managed by the system's exit rules, after all statutory charges. Means are taken per cycle first: "
                   "trades in the same month share one market.")
        cyc = h.groupby("universe")["entry"].agg(["nunique", "min", "max"])
        c = st.columns(4)
        kpi(c[0], "Stock monthly cycles", int(cyc.loc["stocks", "nunique"]) if "stocks" in cyc.index else 0)
        kpi(c[1], "NIFTY weekly cycles", int(cyc.loc["NIFTY weekly", "nunique"]) if "NIFTY weekly" in cyc.index else 0)
        kpi(c[2], "Trades studied", f"{len(h[h['costs'] == 'net']) // 2:,}", "per exit variant")
        kpi(c[3], "Rules live today", n_live, f"{n_watch} watch")

        base = h[(h["costs"].isin(["gross", "net"])) & h["variant"].isin(P.VARIANT.values())]
        agg = R.cycle_t(base, ["universe", "rule", "costs"])
        a, b = st.columns(2)
        for col, uni in ((a, "stocks"), (b, "NIFTY weekly")):
            x = agg[agg["universe"] == uni].pivot_table(index="rule", columns="costs", values="mean_ror")
            if x.empty:
                continue
            fig = go.Figure()
            for cst, color in (("gross", MUTED), ("net", GOLD)):
                if cst in x:
                    fig.add_trace(go.Bar(y=x.index, x=x[cst], orientation="h", name=f"{cst} of costs", marker_color=color))
            fig.add_vline(x=0, line_color=MUTED)
            fig.update_xaxes(tickformat=".0%", title="mean return on max loss, per trade")
            col.plotly_chart(style(fig, 360, f"{uni}: return on risk per trade"), use_container_width=True)
        st.caption("Short premium (credit spreads, condors, flies) was flat before costs and negative after, in both "
                   "universes, despite winning 70–90% of trades: the losses are rarer but larger.")

        st.markdown("#### Rule status (judged only on cycles before the selected session)")
        show = rules.drop(columns="_rule")[["rule", "status", "mean_ror", "t", "cycles", "pos_cycles", "trades"]]
        st.dataframe(show, hide_index=True, use_container_width=True, height=360,
                     column_config={"mean_ror": PCT("net return on risk"), "pos_cycles": PCT("cycles positive"),
                                    "t": st.column_config.NumberColumn("t-stat", format="%.2f")})
        wf = walk()
        if len(wf):
            st.markdown("#### The playbook's own out-of-sample record")
            st.caption(f"At each decision date, only rules that were live on earlier data are traded. "
                       f"Rules were live in {(wf['live_rules'] > 0).sum()} of {len(wf)} decision dates; "
                       f"the system stood aside the rest of the time.")
            fig = go.Figure(go.Bar(x=wf["entry"], y=wf["live_rules"], marker_color=GOLD))
            st.plotly_chart(style(fig, 200, "Live rules at each decision date"), use_container_width=True)
        with st.expander("Other studies (positioning, OI walls, vol forecasts)"):
            st.markdown(open("docs/RESEARCH.md").read() if os.path.exists("docs/RESEARCH.md") else "")

# ── Market ───────────────────────────────────────────────────────────────────
with tabs[2]:
    m = mkt.copy()
    a, b = st.columns([3, 2])
    with a:
        fig = go.Figure()
        for flag, color, name in ((False, GOLD, "no event signal"), (True, RED, "event priced (term inversion)")):
            x = m[m["event"] == flag]
            fig.add_trace(go.Scatter(x=x["rv_1m"], y=x["atm_iv"], mode="markers", name=name, text=x["symbol"],
                                     marker=dict(color=color, size=7, opacity=.75),
                                     hovertemplate="%{text}<br>1M realised %{x:.1%}<br>ATM IV %{y:.1%}<extra></extra>"))
        hi = float(np.nanpercentile(np.r_[m["rv_1m"], m["atm_iv"]], 98)) * 1.05
        fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=MUTED, dash="dot"), name="IV = realised"))
        fig.update_xaxes(title="realised vol, last month", tickformat=".0%", range=[0, hi])
        fig.update_yaxes(title="ATM implied vol", tickformat=".0%", range=[0, hi])
        st.plotly_chart(style(fig, 460, "Implied vs recent realised volatility"), use_container_width=True)
    with b:
        bs = m["build5"].value_counts()
        fig = go.Figure(go.Bar(x=bs.values, y=bs.index, orientation="h", marker_color=MUTED))
        st.plotly_chart(style(fig, 230, "5-day futures positioning"), use_container_width=True)
        st.caption("Positioning is context only: 5- and 20-day build-up states showed no predictive power for forward "
                   "returns on this data (all |t| < 1.2). Options OI walls were breached about as often as option "
                   "prices implied.")
        fig = go.Figure(go.Histogram(x=m["atm_iv"] / m["rv_1m"], nbinsx=40, marker_color=GOLD))
        fig.add_vline(x=1, line_dash="dot", line_color=MUTED)
        st.plotly_chart(style(fig, 210, "IV / 1M realised across names"), use_container_width=True)
    cols = ["symbol", "spot", "expiry", "dte", "atm_iv", "rv_1m", "iv_rv1m", "skew", "term", "event", "build5",
            "fut_ret5", "oi5_pct", "support", "resistance", "pcr_oi", "chain_contracts", "n_tradable", "lot"]
    m["expiry"] = pd.to_datetime(m["expiry"]).dt.date
    st.dataframe(m[cols].sort_values("iv_rv1m"), hide_index=True, use_container_width=True,
                 column_config={k: PCT() for k in ("atm_iv", "rv_1m", "skew", "fut_ret5", "oi5_pct")})

# ── Name ─────────────────────────────────────────────────────────────────────
with tabs[3]:
    syms = sorted(res.smiles)
    default = res.watchlist[0].symbol if res.watchlist else ("NIFTY" if "NIFTY" in syms else syms[0])
    sym = st.selectbox("Underlying", syms, index=syms.index(default) if default in syms else 0)
    row = mkt.set_index("symbol").loc[sym]
    sm = res.smiles[sym]
    c = st.columns(6)
    kpi(c[0], "Spot · forward", f"{row['spot']:,.1f}", f"fwd {row['forward']:,.1f} · {row['dte']}d to {row['expiry']:%d %b}")
    kpi(c[1], "ATM IV", f"{row['atm_iv']:.1%}", f"25Δ skew {row['skew']:+.1%}" if np.isfinite(row["skew"]) else "")
    kpi(c[2], "Realised 1M", f"{row['rv_1m']:.1%}" if np.isfinite(row["rv_1m"]) else "—",
        f"IV/RV {row['iv_rv1m']:.2f}" if np.isfinite(row["iv_rv1m"]) else "")
    kpi(c[3], "Positioning 5d", row["build5"] or "—",
        f"fut {row['fut_ret5']:+.1%} · OI {row['oi5_pct']:+.1%}" if np.isfinite(row["fut_ret5"]) and np.isfinite(row["oi5_pct"]) else "")
    kpi(c[4], "Put / call OI wall", f"{row['support']:,.0f} / {row['resistance']:,.0f}"
        if np.isfinite(row["support"]) and np.isfinite(row["resistance"]) else "—", f"PCR (OI) {row['pcr_oi']:.2f}")
    kpi(c[5], "Lot · liquidity", f"{int(row['lot'])}", f"{int(row['chain_contracts']):,} contracts · {int(row['n_tradable'])} tradable")
    a, b = st.columns(2)
    t = sm.table
    otm = t[((t["otype"] == "CE") & (t["strike"] >= sm.forward)) | ((t["otype"] == "PE") & (t["strike"] <= sm.forward))]
    otm = otm[otm["iv"].notna() & (otm["contracts"] > 0)].sort_values("strike")   # untraded closes are stale
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=otm["strike"], y=otm["iv"], mode="markers", marker=dict(color=MUTED, size=5), name="close IV"))
        tr = otm[otm["tradable"]]
        fig.add_trace(go.Scatter(x=tr["strike"], y=tr["iv"], mode="markers", marker=dict(color=GOLD, size=9), name="tradable"))
        mp = otm.copy()
        from vaaydo import pricing
        mp["model_iv"] = pricing.implied_vol(mp["model_px"], sm.forward, mp["strike"], sm.T, (mp["otype"] == "CE").to_numpy())
        fig.add_trace(go.Scatter(x=mp["strike"], y=mp["model_iv"], mode="lines", line=dict(color=BLUE), name="fitted smile"))
        if np.isfinite(row["rv_1m"]):
            fig.add_hline(y=row["rv_1m"], line_dash="dot", line_color=GREEN, annotation_text="1M realised")
        fig.add_vline(x=sm.forward, line_color=MUTED, line_dash="dot")
        fig.update_yaxes(tickformat=".0%", range=[0, max(otm["iv"].quantile(.95) if len(otm) else 1, row["rv_1m"] or 0) * 1.15])
        st.plotly_chart(style(fig, 340, "Volatility smile (traded strikes)"), use_container_width=True)
    with b:
        ex = store.chain(asof, sym)
        ex = ex[ex["expiry"] == row["expiry"]]
        near = ex[(ex["strike"] > sm.forward * 0.85) & (ex["strike"] < sm.forward * 1.15)]
        pv = near.pivot_table(index="strike", columns="otype", values="oi", aggfunc="sum").fillna(0) / row["lot"]
        fig = go.Figure()
        for ot, color, nm in (("PE", GREEN, "put OI (lots)"), ("CE", RED, "call OI (lots)")):
            if ot in pv:
                fig.add_trace(go.Bar(x=pv.index, y=pv[ot], name=nm, marker_color=color, opacity=.8))
        fig.add_vline(x=row["spot"], line_color=GOLD)
        st.plotly_chart(style(fig, 340, "Open interest by strike").update_layout(barmode="overlay"), use_container_width=True)

    st.markdown("#### Every defined-risk structure on this name, valued honestly")
    st.caption("Distribution: centred on the futures price, scaled to the market's ATM IV, with this stock's own "
               "fat tails. On that basis, expected value is roughly minus costs. The table shows what each structure "
               "costs you to hold and what it risks, not a hidden edge.")
    an = engine.analyse(asof, sym, res, cfg)
    if an:
        tbl = pd.DataFrame([dict(structure=c_.evaluation.structure.kind, legs=c_.evaluation.structure.label,
                                 entry=c_.evaluation.entry_net, max_profit=c_.evaluation.max_profit,
                                 max_loss=c_.evaluation.max_loss, ev=c_.evaluation.ev, pop=c_.evaluation.pop,
                                 tail5=c_.evaluation.cvar5, costs=c_.evaluation.costs) for c_ in an])
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=300, column_config={
            "pop": PCT("P(profit)"), **{k: st.column_config.NumberColumn(k, format="%.0f") for k in
                                        ("entry", "max_profit", "max_loss", "ev", "tail5", "costs")}})
        choice = st.selectbox("Payoff", range(len(an)), format_func=lambda i: f"{an[i].evaluation.structure.kind}  {an[i].evaluation.structure.label}")
        e = an[choice].evaluation
        S = np.linspace(sm.forward * 0.8, sm.forward * 1.2, 400)
        fig = go.Figure(go.Scatter(x=S, y=payoff(e.structure, S) * e.lot + e.entry_net, mode="lines",
                                   line=dict(color=GOLD, width=3), fill="tozeroy"))
        fig.add_hline(y=0, line_color=MUTED)
        fig.add_vline(x=sm.spot, line_dash="dot", line_color=MUTED, annotation_text="spot")
        st.plotly_chart(style(fig, 280, "P&L per lot at expiry (before charges)"), use_container_width=True)

# ── Book ─────────────────────────────────────────────────────────────────────
with tabs[4]:
    marks = journal.mark(asof, cfg)
    if marks.empty:
        st.info("No open positions. Trades appear here when you record them from Today.")
    else:
        c = st.columns(3)
        kpi(c[0], "Open P&L", inr(marks["pnl"].sum()), f"{len(marks)} positions")
        kpi(c[1], "Max loss at risk", inr(marks["max_loss"].sum()))
        kpi(c[2], "Actions due", int(marks["action"].str.startswith("CLOSE").sum()))
        st.dataframe(marks, hide_index=True, use_container_width=True)
        with st.form("close"):
            a, b, c2 = st.columns(3)
            pid = a.selectbox("Position", marks["id"], format_func=lambda i: f"#{i} {marks.set_index('id').loc[i, 'symbol']}")
            pnl = b.number_input("Realised P&L ₹", value=float(np.nan_to_num(marks.set_index("id").loc[pid, "pnl"])))
            why = c2.selectbox("Reason", ["target", "expiry window", "stop", "manual"])
            if st.form_submit_button("Close position"):
                journal.close_position(int(pid), pnl, asof, why)
                st.rerun()
    cal = journal.calibration()
    if not cal.empty:
        st.subheader("Closed trades: predicted vs realised")
        c = st.columns(3)
        kpi(c[0], "Realised P&L", inr(cal["pnl"].sum()), f"predicted {inr(cal['ev'].sum())}")
        kpi(c[1], "Win rate", f"{cal['win'].mean():.0%}", f"predicted {cal['pop'].mean():.0%}")
        kpi(c[2], "Return on risk", f"{cal['ret_on_risk'].mean():+.1%}", f"predicted {cal['pred_ev_to_risk'].mean():+.1%}")
        st.dataframe(cal, hide_index=True, use_container_width=True)

# ── How it works ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown(open("docs/METHOD.md").read() if os.path.exists("docs/METHOD.md") else "See README.md")
