"""
The index desk view: NIFTY and BANKNIFTY, read on Graphite.

Rendered by app.py when the Indices universe is run. It shows where the index options and
futures market is positioned, how volatile it is priced, who holds what, and the evidence
for every index trade VAAYDO tested. It gives no index trades: none passed (the Evidence
tab says why), and five candidates are tracked on paper.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui.components as ui
from ui.theme import chart_color, chart_layout


def _pct(x, d=1, sign=True):
    try:
        return f"{x:+.{d}%}" if sign else f"{x:.{d}%}"
    except (TypeError, ValueError):
        return "—"


def _num(x, d=0):
    try:
        return f"{x:,.{d}f}" if np.isfinite(x) else "—"
    except TypeError:
        return "—"


def _last(P: pd.DataFrame, sym: str) -> pd.Series:
    x = P[P["symbol"] == sym].sort_values("date")
    return x.iloc[-1] if len(x) else pd.Series(dtype=object)


def _pctile(s: pd.Series, v) -> float:
    s = s.dropna().tail(250)
    return float((s < v).mean()) if len(s) and np.isfinite(v) else np.nan


def _regime(row) -> tuple[str, str]:
    g = row.get("gex_dealer", np.nan)
    if not np.isfinite(g):
        return "—", "neutral"
    return ("Dealers long gamma · damping", "success") if g > 0 else ("Dealers short gamma · amplifying", "danger")


def _index_card(P, sym, dk):
    r = _last(P, sym)
    h = P[P["symbol"] == sym].sort_values("date")
    prev = h.iloc[-2] if len(h) > 1 else r
    v = dk.views.get(sym, {})
    reg, tone = _regime(r)
    chg = r["spot"] / prev["spot"] - 1 if np.isfinite(prev.get("spot", np.nan)) else np.nan
    em = r.get("straddle", np.nan)
    rows = [
        ("Spot", f"{_num(r['spot'], 2)}  ({_pct(chg, 2)})"),
        ("Near future", f"{_num(r['fut_close'], 2)} · {pd.Timestamp(r['fut_expiry']):%d %b} · carry over risk-free "
                        + (f"{_pct(r.get('basis'), 2)} p.a." if r.get("fut_dte", 0) >= 5 else "— (too near expiry to annualise)")),
        ("Rollover", f"{_pct(r.get('rollover'), 0, False)} of futures OI in later months · {int(r.get('fut_dte', 0))} days to expiry"),
        ("ATM IV", f"{_pct(r.get('iv_front'), 1, False)} front ({int(r.get('front_dte', 0) or 0)}d) · {_pct(r.get('iv_30'), 1, False)} 30-day"),
        ("Priced move", f"±{_pct(em, 2, False)} (±{_num(em * r['spot'], 0)} pts) to {pd.Timestamp(r['front_expiry']):%d %b}"
         if np.isfinite(em) else "—"),
        ("Put/call OI", f"{r.get('pcr_oi', np.nan):.2f} all · {r.get('pcr_oi_front', np.nan):.2f} nearest"),
        ("Max pain", f"{_num(r.get('max_pain'))}  ({_pct(r.get('max_pain_d'), 2)})"),
        ("Call wall / put wall", f"{_num(r.get('wall_call'))} / {_num(r.get('wall_put'))}"),
        ("Gamma flip", f"{_num(r.get('flip'))}  ({_pct(r.get('flip_d'), 2)} from spot)"),
        ("Skew (IV 3% below − 3% above)", f"{r.get('skew', np.nan) * 100:+.1f} vol pts" if np.isfinite(r.get("skew", np.nan)) else "—"),
    ]
    with ui.panel(f"desk-{sym}", sym, context=f"{pd.Timestamp(r['date']):%d %b %Y}", chip=(reg, tone)):
        st.markdown('<div class="panel-specs">' + "".join(
            f'<div class="lookback-row"><span class="lbl">{html.escape(a)}</span><span class="val">{html.escape(b)}</span></div>'
            for a, b in rows) + "</div>", unsafe_allow_html=True)


def _oi_chart(v, sym, r):
    pr = v["profile"]
    S = v["spot"]
    lo, hi = S * 0.94, S * 1.06
    pr = pr[(pr["strike"] >= lo) & (pr["strike"] <= hi)]
    fig = go.Figure()
    fig.add_bar(x=pr["strike"], y=pr.get("oi_ce", 0) / 1e5, name="Call OI", marker_color=chart_color("rose"), opacity=0.85)
    fig.add_bar(x=pr["strike"], y=-pr.get("oi_pe", 0) / 1e5, name="Put OI", marker_color=chart_color("emerald"), opacity=0.85)
    for x, nm, c, pos in ((S, "spot", "accent", "top right"), (r.get("max_pain"), "max pain", "amber", "top left"),
                          (r.get("flip"), "gamma flip", "violet", "bottom left")):
        if x is not None and np.isfinite(x):
            fig.add_vline(x=x, line_dash="dot", line_color=chart_color(c), annotation_text=nm, annotation_position=pos)
    fig.update_layout(**chart_layout(height=380), barmode="relative")
    ui.render_chart_panel(fig, f"oi-{sym}", f"{sym} open interest by strike · {pd.Timestamp(v['expiry']):%d %b} expiry",
                          units="lakh units · calls up, puts down")
    fig2 = go.Figure()
    fig2.add_bar(x=pr["strike"], y=pr.get("oi_chg_ce", 0) / 1e5, name="Call OI change", marker_color=chart_color("rose"))
    fig2.add_bar(x=pr["strike"], y=-pr.get("oi_chg_pe", 0) / 1e5, name="Put OI change", marker_color=chart_color("emerald"))
    fig2.add_vline(x=S, line_dash="dot", line_color=chart_color("accent"))
    fig2.update_layout(**chart_layout(height=280), barmode="relative")
    ui.render_chart_panel(fig2, f"oichg-{sym}", "Today's change in open interest", units="lakh units · calls up, puts down")


def _gex_chart(v, sym, r):
    g = v["gex"]
    S = v["spot"]
    g = g[(g["strike"] >= S * 0.94) & (g["strike"] <= S * 1.06)]
    fig = go.Figure(go.Bar(x=g["strike"], y=g["gex_cr"],
                           marker_color=[chart_color("emerald") if y >= 0 else chart_color("rose") for y in g["gex_cr"]]))
    fig.add_vline(x=S, line_dash="dot", line_color=chart_color("accent"), annotation_text="spot")
    if np.isfinite(r.get("flip", np.nan)):
        fig.add_vline(x=r["flip"], line_dash="dash", line_color=chart_color("violet"), annotation_text="flip")
    fig.update_layout(**chart_layout(height=320, show_legend=False))
    ui.render_chart_panel(fig, f"gex-{sym}", f"{sym} dealer gamma by strike (options ≤ 60 days)",
                          units="₹ Cr of delta per 1% move · dealers long calls, short puts")


def _vol_charts(P, dk, sym, start):
    h = P[(P["symbol"] == sym) & (P["date"] >= start)].sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["iv_30"] * 100, name="30-day ATM IV", line_color=chart_color("accent"))
    if "rv20" in h:
        fig.add_scatter(x=h["date"], y=h["rv20"] * 100, name="20-day realised", line_color=chart_color("amber"))
    else:
        r = np.log(h["spot"]).diff()
        fig.add_scatter(x=h["date"], y=r.rolling(20).std() * np.sqrt(252) * 100, name="20-day realised",
                        line_color=chart_color("amber"))
    fig.update_layout(**chart_layout(height=300))
    ui.render_chart_panel(fig, f"iv-{sym}", f"{sym}: implied against realised", units="% annualised")
    v = dk.views.get(sym, {})
    c1, c2 = st.columns(2)
    if v and len(v.get("term", [])):
        t = v["term"]
        f = go.Figure(go.Scatter(x=t["dte"], y=t["atm_iv"] * 100, mode="lines+markers", line_color=chart_color("accent")))
        f.update_layout(**chart_layout(height=260, show_legend=False))
        with c1:
            ui.render_chart_panel(f, f"term-{sym}", "Term structure", units="ATM IV % by days to expiry")
    if v and len(v.get("smile", [])):
        sm = v["smile"].sort_values("strike")
        f = go.Figure(go.Scatter(x=sm["strike"], y=sm["iv"] * 100, mode="lines+markers", line_color=chart_color("violet")))
        f.add_vline(x=v["spot"], line_dash="dot", line_color=chart_color("accent"))
        f.update_layout(**chart_layout(height=260, show_legend=False))
        with c2:
            ui.render_chart_panel(f, f"smile-{sym}", "Smile, nearest expiry", units="IV % by strike (out-of-the-money side)")


def _participants(dk, start):
    raw = dk.part_raw
    if raw is None or raw.empty:
        ui.render_empty_state("No participant data", "NSE's participant-wise open interest was not available.", eyebrow="Participants")
        return
    t = raw.set_index("client")
    rows = []
    for g in ("FII", "DII", "Pro", "Client"):
        if g not in t.index:
            continue
        x = t.loc[g]
        fl, fs = x["fut_idx_long"], x["fut_idx_short"]
        rows.append({"Participant": g, "Index futures long": _num(fl), "Index futures short": _num(fs),
                     "Net futures": _num(fl - fs), "Long share": _pct(fl / (fl + fs), 1, False),
                     "Calls long − short": _num(x["opt_idx_call_long"] - x["opt_idx_call_short"]),
                     "Puts long − short": _num(x["opt_idx_put_long"] - x["opt_idx_put_short"])})
    ui.render_section_header("Who holds the index", f"NSE participant-wise open interest, contracts, as on {raw['date'].max():%d %b %Y}. "
                             "All index futures and options together.", icon="briefcase")
    ui.render_data_table(pd.DataFrame(rows), show_index=False, label_col="Participant")
    pf = dk.participants
    pf = pf[pf.index >= start]
    fig = go.Figure()
    for g, c in (("fii", "accent"), ("client", "amber"), ("pro", "violet"), ("dii", "emerald")):
        if f"{g}_fut_long" in pf:
            fig.add_scatter(x=pf.index, y=pf[f"{g}_fut_long"] * 100, name=g.upper() if g != "client" else "Client",
                            line_color=chart_color(c))
    fig.add_hline(y=50, line_dash="dot", line_color=chart_color("slate"))
    fig.update_layout(**chart_layout(height=320))
    ui.render_chart_panel(fig, "part-fut", "Index futures: long share by participant", units="% of each group's futures that is long")
    fig = go.Figure()
    for g, c in (("fii", "accent"), ("client", "amber"), ("pro", "violet")):
        fig.add_scatter(x=pf.index, y=pf[f"{g}_opt_bias"] * 100, name=g.upper() if g != "client" else "Client", line_color=chart_color(c))
    fig.add_hline(y=0, line_dash="dot", line_color=chart_color("slate"))
    fig.update_layout(**chart_layout(height=300))
    ui.render_chart_panel(fig, "part-opt", "Index options stance by participant",
                          units="(long calls + short puts − short calls − long puts) / gross, %")


def _futures(P, sym, start):
    h = P[(P["symbol"] == sym) & (P["date"] >= start)].sort_values("date")
    h = h.assign(basis_c=h["basis"].where(h["fut_dte"] >= 5))
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["basis_c"] * 100, name="Near basis over risk-free", line_color=chart_color("accent"))
    fig.add_scatter(x=h["date"], y=(h["roll_cost"] - 0.065) * 100, name="Roll cost over risk-free", line_color=chart_color("amber"))
    fig.add_hline(y=0, line_dash="dot", line_color=chart_color("slate"))
    fig.update_layout(**chart_layout(height=280))
    ui.render_chart_panel(fig, f"basis-{sym}", f"{sym}: cost of carry", units="% a year")
    # rollover now vs the same days-to-expiry in earlier months
    r = h.iloc[-1]
    same = P[(P["symbol"] == sym) & (P["fut_dte"] == r["fut_dte"])].sort_values("date").tail(13)
    fig = go.Figure(go.Bar(x=[f"{pd.Timestamp(e):%b %y}" for e in same["fut_expiry"]], y=same["rollover"] * 100,
                           marker_color=[chart_color("accent") if d == r["date"] else chart_color("slate") for d in same["date"]]))
    fig.update_layout(**chart_layout(height=260, show_legend=False))
    ui.render_chart_panel(fig, f"roll-{sym}", f"Rollover with {int(r['fut_dte'])} days to expiry, this month against the last twelve",
                          units="% of futures OI in later months")
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["fut_oi"] / 1e5, name="Futures OI", line_color=chart_color("violet"))
    fig.update_layout(**chart_layout(height=240, show_legend=False))
    ui.render_chart_panel(fig, f"futoi-{sym}", f"{sym} futures open interest", units="lakh units, all months")


def _combined_reads(dk):
    rd = getattr(dk, "reads", None) or {}
    if not rd:
        return
    ui.render_section_header("NIFTY: combined reads (paper)", "Weak readings combined without fitted weights. Both pass the "
                             "statistical bar on 2019–2026 but lost to simply holding NIFTY in 2023–24, so they are tracked, "
                             "not traded.", icon="layers")
    pr = rd.get("precedent", {})
    stt = rd.get("state", {})
    ag = rd.get("agreement", 0.0)
    ui.render_kpi_strip([
        dict(label="Agreement (FII stance × breadth)", value={1.0: "▲ both up", -1.0: "▼ both down"}.get(ag, "— disagree"),
             subtext=f"FII stance z {stt.get('fii', np.nan):+.2f} · breadth z {stt.get('breadth', np.nan):+.2f}",
             color_class="success" if ag > 0 else ("danger" if ag < 0 else "neutral")),
        dict(label="Precedent, next 5 sessions", value=_pct(pr.get("read"), 2) if pr else "—",
             subtext=(f"{pr['up']:.0%} of 10 analogs rose vs {pr['base_up']:.0%} usually · p {pr['p']:.2f}" if pr else "warming up"),
             color_class="neutral" if not pr or pr["p"] >= 0.05 else ("success" if pr["read"] > 0 else "danger")),
    ], max_cols=2, key="combined-reads")
    if pr and pr.get("analogs"):
        ui.render_data_table(pd.DataFrame([{"Past session": f"{pd.Timestamp(d):%d %b %Y}", "NIFTY futures, next 5 sessions": _pct(o, 2),
                                            "Distance": f"{dist:.2f}"} for d, o, dist in pr["analogs"]]),
                             show_index=False, label_col="Past session", max_height=300)
        st.markdown('<div class="panel-copy">The ten past sessions whose trend, breadth, PCR, FII stance, IV and dealer gamma '
                    'looked most like today\'s (Mahalanobis distance, distinct episodes, outcomes known at the time).</div>',
                    unsafe_allow_html=True)


def _evidence(dk):
    T = dk.tests
    ui.render_section_header("Every index trade VAAYDO tested", "Registered before testing (docs/PREREGISTRATION.md). "
                             "Pooled NIFTY + BANKNIFTY, Newey–West t, 5-session forward futures return from the next open. "
                             "Accepted only with the registered sign at t ≥ 1 in every period and |t| ≥ 2 overall.", icon="shield")
    if len(T):
        x = T[T["h"] == 5].pivot_table(index="reading", columns="period", values="t")
        sg = T.drop_duplicates("reading").set_index("reading")["sign"].reindex(x.index)
        cols = [c for c in ("discovery", "validation", "final", "all") if c in x]
        x = x[cols]
        verdict = []
        for nm, row in x.iterrows():
            s = sg.get(nm, 0)
            if nm.startswith(("H2", "H3")):
                ok = (row[["discovery", "validation", "final"]] * s > 0).all() and row["all"] * s >= 2
                verdict.append("holds · volatility only" if ok else "rejected")
            elif not s:
                verdict.append("exploratory")
            elif (row[["discovery", "validation", "final"]] * s >= 1).all() and row["all"] * s >= 2:
                verdict.append("passes")
            else:
                verdict.append("rejected")
        x = x.round(2)
        x.insert(0, "Registered sign", ["+" if s > 0 else ("−" if s < 0 else "±") for s in sg])
        x["Verdict"] = verdict
        x.columns = [c.title() if c not in ("Registered sign", "Verdict") else c for c in x.columns]
        ui.render_data_table(x.reset_index().rename(columns={"reading": "Reading"}), show_index=False, label_col="Reading")
    ui.render_section_header("Strategies against simply being long", "50/50 NIFTY / BANKNIFTY futures, net of 0.03% a side.",
                             icon="trending-up")
    ui.render_data_table(pd.DataFrame([
        {"Strategy": "S0 long, always (benchmark)", "Discovery": 0.48, "Validation": 0.66, "Final": -0.10, "2019–26": 0.36},
        {"Strategy": "S1 long, scaled by 15% / IV", "Discovery": 0.38, "Validation": 0.79, "Final": -0.30, "2019–26": 0.25},
        {"Strategy": "S2 60-day trend, scaled", "Discovery": 0.27, "Validation": 0.29, "Final": -0.21, "2019–26": 0.14},
    ]), show_index=False, label_col="Strategy")
    st.markdown('<div class="panel-copy">Net Sharpe. Futures earn the index minus carry, so being long has '
                'lost since mid-2024 even as the index rose a little. None of the tested readings or rules beat it in every period.</div>',
                unsafe_allow_html=True)
    C = dk.cycles
    if len(C):
        ui.render_section_header("Selling the volatility premium", "Iron condor (short ±0.75 × the priced move, wings at ±2×) and "
                                 "iron fly, every expiry, real bhavcopy fills at the open, held to NSE's final settlement.",
                                 icon="activity")
        fig = go.Figure()
        for (s, k), g in C.groupby(["symbol", "structure"]):
            g = g.sort_values("expiry")
            fig.add_scatter(x=g["expiry"], y=g["ret"].cumsum() * 100, name=f"{s} {k}", mode="lines")
        fig.add_hline(y=0, line_dash="dot", line_color=chart_color("slate"))
        fig.update_layout(**chart_layout(height=320))
        ui.render_chart_panel(fig, "cyc", "Cumulative P&L per cycle", units="% of max loss, summed")
        summ = C.groupby(["structure", "symbol"]).agg(cycles=("ret", "size"), mean=("ret", "mean"), won=("pnl", lambda x: (x > 0).mean()),
                                                      worst=("ret", "min")).reset_index()
        ui.render_data_table(pd.DataFrame({"Structure": summ["structure"], "Index": summ["symbol"], "Cycles": summ["cycles"],
                                           "Mean per cycle": summ["mean"].map(lambda v: _pct(v, 1)),
                                           "Won": summ["won"].map(lambda v: _pct(v, 0, False)),
                                           "Worst": summ["worst"].map(lambda v: _pct(v, 0))}), show_index=False, label_col="Structure")
    W = getattr(dk, "wb_tests", None)
    if W is not None and len(W):
        ui.render_section_header("NIFTY breadth and regime workbook", "Every column rebuilt causally (the workbook's fair PE "
                                 "was fitted on all of 2006–2026). Spot 2006–2026 in three periods, then NIFTY futures 2019–2026.",
                                 icon="layers")
        w = W.copy()
        for c in ("P1", "P2", "P3", "all", "disc", "vali", "fina", "fut"):
            w[c] = w[c].round(2)
        w["Verdict"] = np.where(w["accept"], "passes", "rejected")
        ui.render_data_table(w.rename(columns={"reading": "Reading", "sign": "Sign", "P1": "2006–12", "P2": "2013–18", "P3": "2019–26",
                                               "all": "Spot all", "disc": "Fut disc", "vali": "Fut val", "fina": "Fut final",
                                               "fut": "Fut all"}).drop(columns=["accept"]), show_index=False, label_col="Reading")
        ui.render_data_table(pd.DataFrame([
            {"Rule": "S0 long, always", "2006–12": 0.34, "2013–18": 0.31, "2019–26": 0.41, "Fut disc": 0.58, "Fut val": 1.15, "Fut final": -0.22},
            {"Rule": "A1 long when regime > 0", "2006–12": 0.85, "2013–18": 0.12, "2019–26": 0.29, "Fut disc": 0.86, "Fut val": 0.33, "Fut final": -0.31},
            {"Rule": "A2 long when breadth > 1-year median (paper)", "2006–12": 0.72, "2013–18": 0.31, "2019–26": 0.90,
             "Fut disc": 1.33, "Fut val": 1.00, "Fut final": -0.37},
        ]), show_index=False, label_col="Rule")
        st.markdown('<div class="panel-copy">Net Sharpe. A2 doubles the long-run spot Sharpe in half the time, but loses to '
                    'plain long on futures in 2023–24 and since mid-2024, so it is tracked on paper. After the workbook\'s '
                    'dip signal, returns were <em>lower</em> than on an average day; after its top signal, higher.</div>',
                    unsafe_allow_html=True)
    N = getattr(dk, "ntests", None)
    if N is not None and len(N):
        ui.render_section_header("NIFTY at Nishkarsh's standard", "Non-overlapping ICs averaged over start offsets, Newey–West t, "
                                 "block-bootstrap CI, years positive, against NIFTY's own momentum, with the smallest IC this "
                                 "history can detect (MDE). Statistical pass: HAC t ≥ 2, CI above 0, ≥ 70% of years, above momentum.",
                                 icon="shield")
        n = N.copy()
        for c in ("IC", "MDE", "HAC t", "mom IC", "IC disc", "IC vali", "IC fina"):
            n[c] = n[c].round(3)
        n["accept"] = np.where(n["accept"], "passes", "—")
        ui.render_data_table(n.rename(columns={"read": "Read", "accept": "Statistical pass", "yrs+": "Years +"}),
                             show_index=False, label_col="Read")
        TR = getattr(dk, "ntrades", None)
        if TR is not None and len(TR):
            ui.render_data_table(TR.rename(columns={"Disc": "2019–22", "Vali": "2023–24H1", "Fina": "2024H2–26", "All": "2019–26"}),
                                 show_index=False, label_col="Trade (h = 5 tranches)")
        st.markdown('<div class="panel-copy">Net Sharpe on NIFTY futures. Every combination loses to plain long in 2023–24. '
                    'They all contain the FII options stance, so they are not independent confirmations. Single readings '
                    'sit far below the detectable IC (0.13 at 5 sessions): unproven, not disproven.</div>', unsafe_allow_html=True)
    ui.render_section_header("Rejected myths", icon="alert-triangle")
    ui.render_kv_table({
        "Max pain pulls the index into expiry": "spot moved toward max pain in 47.5% of 737 expiries",
        "Positive dealer gamma calms the market": "right direction, too weak: realised/implied 0.835 vs 0.887, t −1.8",
        "Writers' hedging pushes the future": "wrong sign in two of three periods",
        "High PCR is bullish": "t 1.6 / −0.8 / 1.2: not reliable",
    }, header_left="Claim", header_right="What 2019–2026 says")


def _breadth(dk, start):
    B = dk.breadth
    if B is None or B.empty:
        ui.render_empty_state("No breadth history", "The breadth inputs are not in this bundle.", eyebrow="Breadth")
        return
    B = B.copy()
    B["median"] = B["breadth"].rolling(250, min_periods=120).median()
    r = B.iloc[-1]
    ui.render_kpi_strip([
        dict(label="Advance / decline", value=f"{r['AD_RATIO']:.2f}", subtext=f"all NSE stocks · {pd.Timestamp(r['DATE']):%d %b}"),
        dict(label="Breadth (10-day EMA)", value=_pct(r["breadth"], 1, False), subtext=f"1-year median {_pct(r['median'], 1, False)}",
             color_class="success" if r["breadth"] > r["median"] else "danger"),
        dict(label="Regime", value=f"{r['regime']:+.1f}", subtext="RSI + breadth + trend zones, −3 … +3"),
        dict(label="RSI 14", value=f"{r['rsi']:.1f}", subtext=f"bands {_num(r['lower'])} – {_num(r['upper'])}"),
        dict(label="NIFTY P/E", value=f"{r['NIFTY50_PE']:.2f}" if np.isfinite(r.get("NIFTY50_PE", np.nan)) else "—",
             subtext=f"P/B {r.get('NIFTY50_PB', np.nan):.2f} · dividend yield {r.get('NIFTY50_DY', np.nan):.2f}%"),
    ], key="breadth-kpi")
    h = B[B["DATE"] >= start]
    fig = go.Figure()
    fig.add_scatter(x=h["DATE"], y=h["breadth"] * 100, name="Breadth (10-day EMA)", line_color=chart_color("accent"))
    fig.add_scatter(x=h["DATE"], y=h["median"] * 100, name="1-year median", line_color=chart_color("amber"), line_dash="dot")
    fig.update_layout(**chart_layout(height=300))
    ui.render_chart_panel(fig, "breadth", "Market breadth: share of NSE stocks advancing", units="%")
    fig = go.Figure()
    fig.add_scatter(x=h["DATE"], y=h["NIFTY"], name="NIFTY", line_color=chart_color("slate"))
    on = h[h["breadth"] > h["median"]]
    fig.add_scatter(x=on["DATE"], y=on["NIFTY"], mode="markers", marker=dict(size=3, color=chart_color("emerald")),
                    name="breadth above median")
    for col, nm, c in (("btd", "dip signal", "amber"), ("stt", "top signal", "rose")):
        ev = h[h[col] > 0]
        fig.add_scatter(x=ev["DATE"], y=ev["NIFTY"], mode="markers", marker=dict(size=8, symbol="diamond", color=chart_color(c)), name=nm)
    fig.update_layout(**chart_layout(height=320))
    ui.render_chart_panel(fig, "breadth-nifty", "NIFTY with the breadth state and the dip / top signals", units="index")
    rows = [("Advancing share today", _pct(r["adv_share"], 1, False)), ("Relative breadth", _pct(r["rel_breadth"], 1, False)),
            ("A/D momentum", f"{r['rel_ad']:+.2f}"), ("Above 90-day MA", _pct(r["spread90"] / 100, 1)),
            ("Above 200-day MA (8d lag)", _pct(r["spread200"] / 100, 1)), ("±1% day count vs 50-day mean", f"{r['count']:+.1f}"),
            ("20-day realised vol", _pct(r["rvol20"] / 100, 1, False)), ("Signal strength", f"{r['signal_str']:+.2f}"),
            ("Earnings yield", _pct(1 / r["NIFTY50_PE"], 2, False) if np.isfinite(r.get("NIFTY50_PE", np.nan)) else "—"),
            ("Dip / top signal today", "dip" if r["btd"] else ("top" if r["stt"] else "none"))]
    ui.render_kv_table(dict(rows), header_left="Reading", header_right="Now")
    st.markdown('<div class="panel-copy">Advance/decline counts every NSE stock (the bhavcopy), so breadth reads the '
                'whole market, small caps included, not NIFTY\'s fifty. Everything here is computed from NSE\'s own daily '
                'files (equity bhavcopy and index closes).</div>', unsafe_allow_html=True)


@st.fragment
def _oi_tab(dk, P):
    sym = st.radio("Index", ["NIFTY", "BANKNIFTY"], horizontal=True, key="desk_sym_oi")
    v = dk.views.get(sym)
    if not v:
        ui.render_empty_state("No chain", "The option chain for this session could not be read.", eyebrow="Gamma & OI")
    else:
        r = _last(P, sym)
        ui.render_kpi_strip([
            dict(label="Spot", value=_num(v["spot"], 2), subtext=f"nearest expiry {pd.Timestamp(v['expiry']):%d %b}"),
            dict(label="Max pain", value=_num(r.get("max_pain")), subtext=_pct(r.get("max_pain_d"), 2) + " from forward"),
            dict(label="Call wall", value=_num(r.get("wall_call")), subtext="largest call OI above"),
            dict(label="Put wall", value=_num(r.get("wall_put")), subtext="largest put OI below"),
            dict(label="Gamma flip", value=_num(r.get("flip")), subtext=_regime(r)[0],
                 color_class="success" if r.get("gex_dealer", 0) > 0 else "danger"),
        ], key=f"oi-kpi-{sym}")
        _oi_chart(v, sym, r)
        _gex_chart(v, sym, r)


@st.fragment
def _vol_tab(dk, P, start):
    sym = st.radio("Index", ["NIFTY", "BANKNIFTY"], horizontal=True, key="desk_sym_vol")
    h = P[P["symbol"] == sym].sort_values("date")
    r = h.iloc[-1]
    rv = np.log(h["spot"]).diff().rolling(20).std().iloc[-1] * np.sqrt(252)
    ui.render_kpi_strip([
        dict(label="Front ATM IV", value=_pct(r.get("iv_front"), 1, False), subtext=f"{int(r.get('front_dte', 0) or 0)} days"),
        dict(label="30-day ATM IV", value=_pct(r.get("iv_30"), 1, False),
             subtext=f"{_pct(_pctile(h['iv_30'], r.get('iv_30')), 0, False)} percentile, 1y"),
        dict(label="20-day realised", value=_pct(rv, 1, False), subtext="spot"),
        dict(label="India VIX", value=_num(dk.vix.iloc[-1], 2) if len(dk.vix) else "—",
             subtext=f"{_pct(_pctile(dk.vix, dk.vix.iloc[-1]), 0, False)} percentile, 1y" if len(dk.vix) else ""),
        dict(label="Priced move to expiry", value=f"±{_pct(r.get('straddle'), 2, False)}",
             subtext=f"±{_num(r.get('straddle', np.nan) * r['spot'])} pts"),
    ], key=f"vol-kpi-{sym}")
    _vol_charts(P.assign(rv20=P.groupby("symbol")["spot"].transform(lambda s: np.log(s).diff().rolling(20).std() * np.sqrt(252))),
                dk, sym, start)
    if len(dk.vix):
        vx = dk.vix[dk.vix.index >= start]
        fig = go.Figure(go.Scatter(x=vx.index, y=vx.values, line_color=chart_color("rose")))
        for q, c in ((0.2, "emerald"), (0.8, "amber")):
            fig.add_hline(y=float(dk.vix.tail(250).quantile(q)), line_dash="dot", line_color=chart_color(c),
                          annotation_text=f"{int(q * 100)}th pct, 1y")
        fig.update_layout(**chart_layout(height=260, show_legend=False))
        ui.render_chart_panel(fig, "vix", "India VIX", units="index")


@st.fragment
def _fut_tab(P, start):
    sym = st.radio("Index", ["NIFTY", "BANKNIFTY"], horizontal=True, key="desk_sym_fut")
    _futures(P, sym, start)


def render(dk, meta: dict):
    P = dk.panel
    start = pd.Timestamp(meta.get("start") or (dk.asof - pd.Timedelta(days=730)))
    n, b = _last(P, "NIFTY"), _last(P, "BANKNIFTY")
    vix = dk.vix.iloc[-1] if len(dk.vix) else np.nan
    ui.render_top_bar(target="INDICES", status_label="NO TRADE · EVIDENCE", status_tone="neutral",
                      meta_items=[("As of", dk.asof.strftime("%d %b %Y")), ("NIFTY", _num(n.get("spot"), 2)),
                                  ("BANKNIFTY", _num(b.get("spot"), 2)), ("India VIX", _num(vix, 2))])
    notices = [dict(kind="info", title="The desk gives no index trades",
                    body="Every directional reading and both premium-selling structures were tested on 2019–2026 and failed "
                         "(Evidence tab). Five candidates (FII options stance, high-IV condor, market breadth, precedent, agreement) are tracked on paper from 18 Nov 2026, never with capital.")]
    notices += [dict(kind="info", title="Note", body=html.escape(x)) for x in dk.notes]
    ui.render_notice_rail(notices)
    tabs = st.tabs(["Desk", "Breadth", "Gamma & OI", "Volatility", "Participants", "Futures & Rollover", "Evidence"])
    with tabs[0]:
        c1, c2 = st.columns(2, gap="small")
        with c1:
            _index_card(P, "NIFTY", dk)
        with c2:
            _index_card(P, "BANKNIFTY", dk)
        ui.render_section_header("On paper only", "Candidates that failed validation, tracked forward without capital. "
                                 "Promoted only if they earn it out of sample.", icon="eye")
        pp = dk.paper
        if len(pp):
            ui.render_data_table(pd.DataFrame({
                "Index": pp["symbol"],
                "FII options stance (z)": pp["fii_opt_z"].map(lambda v: f"{v:+.2f}" if v is not None and np.isfinite(v) else "—"),
                "Paper futures side": pp["fii_side"],
                "Front IV": pp["iv_front"].map(lambda v: _pct(v, 1, False)),
                "Top-decile IV line": pp["iv_q90"].map(lambda v: _pct(v, 1, False)),
                "Paper condor": pp["stress"].map(lambda s: "sell this cycle (IV in top decile)" if s else "stand aside"),
            }), show_index=False, label_col="Index")
        B = dk.breadth
        if B is not None and len(B):
            med = B["breadth"].rolling(250, min_periods=120).median().iloc[-1]
            b = B["breadth"].iloc[-1]
            st.markdown('<div class="panel-copy">Breadth candidate (paper): NIFTY long while market breadth is above its '
                        f'one-year median, flat otherwise. Today breadth {b:.1%} against {med:.1%}: '
                        f'<strong>{"paper long" if b > med else "paper flat"}</strong>.</div>', unsafe_allow_html=True)
        _combined_reads(dk)
        ui.render_section_header("Readings against their own last year", icon="grid")
        rows = []
        for sym in ("NIFTY", "BANKNIFTY"):
            h = P[P["symbol"] == sym].sort_values("date")
            r = h.iloc[-1]
            for nm, col, f in (("Put/call OI", "pcr_oi", lambda v: f"{v:.2f}"), ("30-day IV", "iv_30", lambda v: _pct(v, 1, False)),
                               ("Skew (vol pts)", "skew", lambda v: f"{v * 100:+.1f}"), ("Dealer gamma", "gex_dealer", lambda v: f"{v:+.3f}"),
                               ("Rollover", "rollover", lambda v: _pct(v, 0, False)), ("Term (front / 30d)", "term", lambda v: f"{v:.2f}")):
                v = r.get(col, np.nan)
                rows.append({"Index": sym, "Reading": nm, "Now": f(v) if np.isfinite(v) else "—",
                             "Percentile, 250 sessions": _pct(_pctile(h[col], v), 0, False)})
        ui.render_data_table(pd.DataFrame(rows), show_index=False, label_col="Index")
    with tabs[1]:
        _breadth(dk, start)
    with tabs[2]:
        _oi_tab(dk, P)
    with tabs[3]:
        _vol_tab(dk, P, start)
    with tabs[4]:
        _participants(dk, start)
    with tabs[5]:
        _fut_tab(P, start)
    with tabs[6]:
        _evidence(dk)
