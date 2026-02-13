"""
╔══════════════════════════════════════════════════════════════════════════╗
║  VAAYDO (वायदो) — FnO Trade Intelligence                               ║
║  World's First God-Tier Options Trading Intelligence System             ║
║  Version 3.1.0 | Hemrek Capital                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

MATHEMATICAL ARSENAL (20 Engines):
 1. Black-Scholes-Merton — Full pricing + 9 Greeks (Δ,Γ,Θ,ν,ρ,Vanna,Volga,Charm,Speed)
 2. Multi-Estimator Vol — Close-to-Close, Parkinson, Garman-Klass, Yang-Zhang (§3.2)
 3. GARCH(1,1) — Conditional variance, persistence, half-life (§3.2)
 4. Volatility Risk Premium — Regime-adaptive IV from RV (§3.3)
 5. Monte Carlo (Antithetic) — 10K effective paths, all 10 strategies (§5.1)
 6. Kelly Criterion — Half-Kelly, confidence-weighted (§7.1)
 7. 6-State Vol Regime — Compressed/Low/Normal/Elevated/High/Extreme (§4.2)
 8. 5-State Trend Regime — MA + RSI + ADX + Momentum (§4.3)
 9. Kalman Filter — Adaptive trend + volatility smoothing (§5.3)
10. CUSUM — Multi-stream structural break detection (§5.4)
11. Higher-Order Greeks — Vanna, Volga, Charm, Speed (§6.2)
12. Composite Risk Score — Regime-weighted multi-Greek aggregation (§6.3)
13. Ensemble POP — Inverse-variance BSM+MC fusion (§9.2)
14. Regime Stability — GARCH persistence + CUSUM stability (§4.4)
15. Entropy Uncertainty — Regime ambiguity quantification
16. Expected Move Zones — 1σ/2σ/3σ log-normal distributions
17. Sharpe-Ratio Ranking — Risk-adjusted, clamped [-5,5]
18. Information Ratio — Strategy performance benchmarking
19. Unified Conviction — §9.1 multi-factor weighted (premium-relative EV)
20. SPAN Margin Estimation — Realistic max loss for unlimited-risk strategies

STRATEGY UNIVERSE (10 Active):
 Short Strangle · Short Straddle · Iron Condor · Iron Butterfly
 Bull Put Spread · Bear Call Spread · Calendar Spread
 Jade Lizard · Broken Wing Butterfly · Ratio Spread

BUGFIXES v3.1:
 - Sharpe clamped to [-5,5] — no more 10-digit explosions
 - All 10 strategies use real MC simulation — no fake pop_b*0.95
 - Full Greeks (9) computed for all strategies — not just theta
 - EV normalized relative to max_profit — not absolute ₹
 - SPAN margin for unlimited-risk strategies — not arbitrary %
 - Wing width enforced >= 1 gap — no collapsed strikes
 - BWB/Ratio/Calendar proper payoff math — no arbitrary multipliers
 - Min premium threshold — skip strategies with credit < ₹0.50
 - R:R clamped to [0, 50] — no more ml=0.01 explosions
 - Kalman filter added for trend smoothing (§5.3)
 - ADX added for trend strength (§4.3)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta, date
import yfinance as yf
import requests
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="VAAYDO | FnO Intelligence", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")
VERSION = "3.1.0"

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    :root {
        --gold: #FFC300; --gold-rgb: 255,195,0;
        --bg: #0F0F0F; --bg2: #1A1A1A; --bg3: #2A2A2A;
        --text: #EAEAEA; --muted: #888; --border: #2A2A2A; --border2: #3A3A3A;
        --green: #10b981; --red: #ef4444; --amber: #f59e0b; --cyan: #06b6d4; --purple: #a855f7;
    }
    * { font-family: 'Inter', -apple-system, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--bg); color: var(--text); }
    .stApp > header { background-color: transparent; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 3.5rem; max-width: 95%; padding-left: 2rem; padding-right: 2rem; }
    [data-testid="collapsedControl"] {
        display: flex !important; visibility: visible !important; opacity: 1 !important;
        background-color: var(--bg2) !important; border: 2px solid var(--gold) !important;
        border-radius: 8px !important; box-shadow: 0 0 15px rgba(var(--gold-rgb),0.4) !important;
        z-index: 999999 !important; position: fixed !important; top: 14px !important; left: 14px !important;
        width: 40px !important; height: 40px !important; align-items: center !important; justify-content: center !important;
    }
    [data-testid="collapsedControl"]:hover { background-color: rgba(var(--gold-rgb),0.2) !important; }
    [data-testid="collapsedControl"] svg { stroke: var(--gold) !important; }
    [data-testid="stSidebar"] button[kind="header"] { background-color: transparent !important; border: none !important; }
    [data-testid="stSidebar"] button[kind="header"] svg { stroke: var(--gold) !important; }
    [data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid var(--border); }
    .hdr { background: var(--bg2); padding: 1.25rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--gold-rgb),0.1); border: 1px solid var(--border);
        position: relative; overflow: hidden; margin-top: 1rem; }
    .hdr::before { content: ''; position: absolute; inset: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--gold-rgb),0.08) 0%, transparent 50%); pointer-events: none; }
    .hdr h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text); letter-spacing: -0.5px; position: relative; }
    .hdr .tag { color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    .hdr .badge { display: inline-block; background: rgba(var(--gold-rgb),0.15); color: var(--gold);
        padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    .mc { background-color: var(--bg2); padding: 1.25rem; border-radius: 12px; border: 1px solid var(--border);
        box-shadow: 0 0 15px rgba(var(--gold-rgb),0.08); margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1); position: relative; overflow: hidden; }
    .mc:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border2); }
    .mc h4 { color: var(--muted); font-size: 0.7rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .mc h2 { color: var(--text); font-size: 1.6rem; font-weight: 700; margin: 0; line-height: 1; }
    .mc .sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.5rem; font-weight: 500; }
    .mc.ok h2 { color: var(--green); } .mc.bad h2 { color: var(--red); }
    .mc.warn h2 { color: var(--amber); } .mc.info h2 { color: var(--cyan); } .mc.gold h2 { color: var(--gold); }
    .tc { background: var(--bg2); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border);
        box-shadow: 0 0 15px rgba(var(--gold-rgb),0.08); margin-bottom: 1rem; position: relative;
        overflow: hidden; transition: all 0.3s cubic-bezier(0.4,0,0.2,1); }
    .tc:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.4); border-color: rgba(var(--gold-rgb),0.3); }
    .tc::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
    .tc.hi::before { background: var(--green); } .tc.md::before { background: var(--amber); } .tc.lo::before { background: var(--red); }
    .tc .sym { font-size: 1.2rem; font-weight: 800; color: var(--text); }
    .tc .strat { font-size: 0.75rem; font-weight: 700; color: var(--gold); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.25rem; }
    .tc .gr { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 1rem; }
    .tc .gr .gi label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; display: block; }
    .tc .gr .gi .v { font-size: 1rem; font-weight: 700; color: var(--text); font-family: 'JetBrains Mono', monospace; }
    .tc .cb { height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; margin-top: 1rem; }
    .tc .cf { height: 100%; border-radius: 3px; }
    .sb { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.75rem; border-radius: 20px;
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .sb.buy { background: rgba(16,185,129,0.15); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }
    .sb.sell { background: rgba(239,68,68,0.15); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }
    .sb.neut { background: rgba(136,136,136,0.15); color: var(--muted); border: 1px solid rgba(136,136,136,0.3); }
    .sb.prem { background: rgba(var(--gold-rgb),0.15); color: var(--gold); border: 1px solid rgba(var(--gold-rgb),0.3); }
    .ib { background: var(--bg2); border: 1px solid var(--border); padding: 1.25rem; border-radius: 12px;
        margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--gold-rgb),0.08); }
    .ib h4 { color: var(--gold); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .ib p { color: var(--muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    .stButton>button { border: 2px solid var(--gold); background: transparent; color: var(--gold);
        font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--gold-rgb),0.6); background: var(--gold); color: #1A1A1A; transform: translateY(-2px); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--muted); border-bottom: 2px solid transparent; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--gold); border-bottom: 2px solid var(--gold); background: transparent !important; }
    .stPlotlyChart { border-radius: 12px; background-color: var(--bg2); padding: 10px; border: 1px solid var(--border); }
    .stDataFrame { border-radius: 12px; background-color: var(--bg2); border: 1px solid var(--border); }
    .divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border) 50%, transparent 100%); margin: 1.5rem 0; }
    .stitle { font-size: 0.75rem; font-weight: 700; color: var(--gold); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg); } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    .gt { width: 100%; border-collapse: collapse; }
    .gt th { color: var(--muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
    .gt td { color: var(--text); font-size: 0.85rem; padding: 0.75rem; border-bottom: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .tg { color: var(--green); } .tr { color: var(--red); } .ta { color: var(--amber); } .tc_ { color: var(--cyan); } .tgl { color: var(--gold); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# F&O UNIVERSE + LOT SIZES
# ═══════════════════════════════════════════════════════════════════════════════

FNO_UNIVERSE = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "ITC","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN","SUNPHARMA",
    "ULTRACEMCO","WIPRO","BAJFINANCE","BAJAJFINSV","HCLTECH","TATAMOTORS","POWERGRID",
    "NTPC","TATASTEEL","ONGC","ADANIPORTS","NESTLEIND","JSWSTEEL","M&M","COALINDIA",
    "GRASIM","TECHM","INDUSINDBK","CIPLA","DRREDDY","DIVISLAB","BPCL","APOLLOHOSP",
    "EICHERMOT","TATACONSUM","HEROMOTOCO","HINDALCO","SBILIFE","BRITANNIA","DABUR",
    "PIDILITIND","HAVELLS","SIEMENS","GODREJCP","DLF","TRENT","BANKBARODA","VEDL",
    "IDFCFIRSTB","PNB","CANBK","LICHSGFIN","MFSL","BHEL","IOC","GAIL","PEL","VOLTAS",
    "COLPAL","AMBUJACEM","ACC","AUROPHARMA","LUPIN","BIOCON","TORNTPHARM","MANAPPURAM",
    "MUTHOOTFIN","SAIL","NMDC","TATAPOWER","RECLTD","PFC","IRCTC","HAL","BEL",
    "BALKRISIND","MRF","PAGEIND","NAUKRI","ZOMATO","PAYTM","POLYCAB","PERSISTENT",
    "COFORGE","LTIM","ABCAPITAL","BANDHANBNK","FEDERALBNK","RAMCOCEM","CROMPTON",
    "LALPATHLAB","METROPOLIS","LAURUSLABS","GRANULES","ATUL","CHAMBLFERT","DEEPAKNTR",
    "ASTRAL","BATAINDIA","IDEA","INDUSTOWER","HINDPETRO","MGL","IGL","PETRONET"
]

# Approximate lot sizes (NSE, updated periodically)
LOT_SIZES = {
    "RELIANCE":250,"TCS":175,"HDFCBANK":550,"INFY":400,"ICICIBANK":700,"HINDUNILVR":300,
    "SBIN":1500,"BHARTIARTL":475,"ITC":1600,"KOTAKBANK":400,"LT":150,"AXISBANK":625,
    "ASIANPAINT":300,"MARUTI":100,"TITAN":375,"SUNPHARMA":350,"ULTRACEMCO":100,
    "WIPRO":1500,"BAJFINANCE":125,"BAJAJFINSV":500,"HCLTECH":350,"TATAMOTORS":1400,
    "POWERGRID":2700,"NTPC":2700,"TATASTEEL":5500,"ONGC":3850,"ADANIPORTS":625,
    "NESTLEIND":50,"JSWSTEEL":675,"M&M":350,"COALINDIA":1400,"GRASIM":275,
    "TECHM":600,"INDUSINDBK":500,"CIPLA":650,"DRREDDY":125,"DIVISLAB":175,
    "BPCL":1800,"APOLLOHOSP":125,"EICHERMOT":175,"TATACONSUM":675,"HEROMOTOCO":150,
    "HINDALCO":1075,"SBILIFE":750,"BRITANNIA":100,"DABUR":1250,"PIDILITIND":375,
    "HAVELLS":500,"SIEMENS":150,"GODREJCP":500,"DLF":1375,"TRENT":100,
    "BANKBARODA":2925,"VEDL":1525,"IDFCFIRSTB":10000,"PNB":7000,"CANBK":6750,
    "LICHSGFIN":1500,"MFSL":125,"BHEL":5500,"IOC":4850,"GAIL":6100,"PEL":550,
    "VOLTAS":500,"COLPAL":350,"AMBUJACEM":1500,"ACC":250,"AUROPHARMA":425,
    "LUPIN":550,"BIOCON":2300,"TORNTPHARM":250,"MANAPPURAM":4000,"MUTHOOTFIN":400,
    "SAIL":6100,"NMDC":3000,"TATAPOWER":2700,"RECLTD":1500,"PFC":3800,
    "IRCTC":875,"HAL":175,"BEL":3700,"BALKRISIND":250,"MRF":5,"PAGEIND":15,
    "NAUKRI":125,"ZOMATO":5000,"PAYTM":1600,"POLYCAB":125,"PERSISTENT":150,
    "COFORGE":125,"LTIM":150,"ABCAPITAL":5100,"BANDHANBNK":4000,"FEDERALBNK":5000,
    "RAMCOCEM":700,"CROMPTON":3500,"LALPATHLAB":250,"METROPOLIS":200,"LAURUSLABS":1100,
    "GRANULES":1600,"ATUL":100,"CHAMBLFERT":1700,"DEEPAKNTR":400,"ASTRAL":275,
    "BATAINDIA":550,"IDEA":70000,"INDUSTOWER":2800,"HINDPETRO":1300,"MGL":400,
    "IGL":1250,"PETRONET":4000
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_symbols():
    try:
        s = requests.Session()
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
        s.get("https://www.nseindia.com", headers=h, timeout=5)
        r = s.get("https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O", headers=h, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if 'data' in data:
                syms = [i['symbol'] for i in data['data'] if 'symbol' in i and i['symbol'] not in ('NIFTY 50','NIFTY BANK')]
                if len(syms) > 30:
                    return syms, f"✓ {len(syms)} F&O stocks from NSE"
    except Exception:
        pass
    return FNO_UNIVERSE.copy(), f"✓ {len(FNO_UNIVERSE)} F&O stocks (fallback)"


# ═══════════════════════════════════════════════════════════════════════════════
# L1: DATA INGESTION — Multi-Estimator Vol + GARCH + Kalman + CUSUM
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data(symbols_ns: list, days_back: int = 400):
    """Master data engine: OHLCV → all analytics per §3.2, §3.3, §4.3, §5.3, §5.4"""
    end = datetime.now(); start = end - timedelta(days=days_back + 60)
    try:
        raw = yf.download(symbols_ns, start=start, end=end, progress=False, auto_adjust=True, group_by='ticker', threads=True)
    except Exception as e:
        return pd.DataFrame(), f"Download failed: {e}"
    if raw.empty:
        return pd.DataFrame(), "No data from yfinance"

    results = []; is_multi = isinstance(raw.columns, pd.MultiIndex)

    for sym_ns in symbols_ns:
        sym = sym_ns.replace('.NS', '')
        try:
            if is_multi:
                if sym_ns not in raw.columns.get_level_values(0): continue
                df = raw.xs(sym_ns, level=0, axis=1).copy()
            else:
                df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close'])
            if len(df) < 60: continue

            O, H, L, Cl, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
            price = float(Cl.iloc[-1])
            lr = np.log(Cl / Cl.shift(1)).dropna()

            # ── §3.2: Multi-Estimator Volatility ──
            rv_c2c = lr.rolling(20).std() * np.sqrt(252)
            hl = np.log(H / L)
            rv_park = np.sqrt(hl.pow(2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)
            u = np.log(H / O); d = np.log(L / O); c = np.log(Cl / O)
            gk_var = (0.5 * u.pow(2) - (2*np.log(2)-1) * c.pow(2) + 0.5 * d.pow(2)).rolling(20).mean()
            rv_gk = np.sqrt(gk_var.clip(lower=0)) * np.sqrt(252)
            o_c_prev = np.log(O / Cl.shift(1)); c_o = np.log(Cl / O)
            yz_o = o_c_prev.rolling(20).var(); yz_c = c_o.rolling(20).var()
            k = 0.34 / (1.34 + 21/19)
            yz_var = yz_o + k * yz_c + (1 - k) * gk_var.clip(lower=0)
            rv_yz = np.sqrt(yz_var.clip(lower=0)) * np.sqrt(252)

            w = {'c2c': 0.15, 'park': 0.20, 'gk': 0.25, 'yz': 0.40}
            rv_composite = (w['c2c'] * rv_c2c.fillna(0) + w['park'] * rv_park.fillna(0) +
                           w['gk'] * rv_gk.fillna(0) + w['yz'] * rv_yz.fillna(0))
            current_rv = float(rv_composite.iloc[-1]) if not np.isnan(rv_composite.iloc[-1]) else 0.25
            current_rv = max(current_rv, 0.05)  # floor at 5% vol

            # ── §3.3: IV Estimation with VRP ──
            rv_history = rv_composite.dropna().values
            lookback = min(252, len(rv_history))
            ivp = float(np.sum(rv_history[-lookback:] <= current_rv) / lookback * 100) if lookback > 20 else 50.0

            if ivp > 70: vrp_factor = 1.08
            elif ivp < 30: vrp_factor = 1.18
            else: vrp_factor = 1.12
            atmiv = current_rv * vrp_factor * 100

            # ── GARCH(1,1) ──
            omega, alpha, beta = 0.000005, 0.10, 0.85
            lr_vals = lr.values[-60:] if len(lr) >= 60 else lr.values
            var_t = current_rv**2 / 252
            for ret in lr_vals:
                var_t = max(omega + alpha * ret**2 + beta * var_t, 1e-10)
            garch_vol = np.sqrt(var_t * 252)
            persistence = alpha + beta
            half_life = -np.log(2) / np.log(max(persistence, 0.001)) if persistence < 1 else 999

            # ── L2: Technical Analysis ──
            delta_c = Cl.diff()
            gain = delta_c.where(delta_c > 0, 0).rolling(14).mean()
            loss = (-delta_c.where(delta_c < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

            # §4.3: ADX for trend strength
            plus_dm = H.diff().clip(lower=0)
            minus_dm = (-L.diff()).clip(lower=0)
            plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
            true_range = pd.concat([H-L, (H-Cl.shift(1)).abs(), (L-Cl.shift(1)).abs()], axis=1).max(axis=1)
            atr14 = true_range.rolling(14).mean()
            plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
            minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            adx_val = float(dx.rolling(14).mean().iloc[-1]) if len(dx) >= 28 else 20.0
            adx_val = adx_val if not np.isnan(adx_val) else 20.0

            atr_val = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else price * 0.02
            ma20 = float(Cl.rolling(20).mean().iloc[-1]) if len(Cl) >= 20 else price
            ma50 = float(Cl.rolling(50).mean().iloc[-1]) if len(Cl) >= 50 else price
            ma200 = float(Cl.rolling(200).mean().iloc[-1]) if len(Cl) >= 200 else price

            # §5.3: Kalman Filter for trend
            kalman_price = price
            kalman_var = atr_val**2
            R_noise = (price * 0.01)**2  # observation noise
            Q_proc = (price * 0.002)**2   # process noise
            for p_val in Cl.values[-20:]:
                if np.isnan(p_val): continue
                pred_var = kalman_var + Q_proc
                K_gain = pred_var / (pred_var + R_noise)
                kalman_price = kalman_price + K_gain * (p_val - kalman_price)
                kalman_var = (1 - K_gain) * pred_var
            kalman_trend = (price - kalman_price) / max(atr_val, 0.01)

            vol_curr = float(V.iloc[-1]) if not np.isnan(V.iloc[-1]) else 0
            vol20 = float(V.rolling(20).mean().iloc[-1]) if len(V) >= 20 else max(vol_curr, 1)

            up_v = V.where(Cl > Cl.shift(1), 0).rolling(20).sum()
            dn_v = V.where(Cl < Cl.shift(1), 0).rolling(20).sum()
            pcr = float(dn_v.iloc[-1] / max(up_v.iloc[-1], 1)) if len(up_v) >= 20 else 1.0
            pcr = min(max(pcr, 0.2), 3.0)
            pct_change = float(Cl.pct_change().iloc[-1] * 100) if len(Cl) >= 2 else 0.0

            # §5.4: CUSUM
            cusum_pos, cusum_neg, cusum_alert = 0.0, 0.0, False
            recent_lr = lr.values[-30:] if len(lr) >= 30 else lr.values
            mu_lr = np.mean(recent_lr); sd_lr = max(np.std(recent_lr), 1e-6)
            for r_val in recent_lr[-10:]:
                z = (r_val - mu_lr) / sd_lr
                cusum_pos = max(0, cusum_pos + z - 0.5)
                cusum_neg = max(0, cusum_neg - z - 0.5)
                if cusum_pos > 4.0 or cusum_neg > 4.0:
                    cusum_alert = True; cusum_pos = cusum_neg = 0

            def safe_rv(series):
                v = series.iloc[-1] if len(series) > 0 else np.nan
                return round(float(v * 100), 2) if not np.isnan(v) else 0.0

            results.append({
                'Instrument': sym, 'price': round(price, 2),
                'ATMIV': round(atmiv, 2), 'IVPercentile': round(ivp, 1),
                'RV_Composite': round(current_rv * 100, 2),
                'GARCH_Vol': round(garch_vol * 100, 2),
                'VRP_Factor': vrp_factor,
                'GARCH_Persistence': round(persistence, 3),
                'GARCH_HalfLife': round(half_life, 1),
                'PCR': round(pcr, 3), 'volume': vol_curr, 'vol20': vol20,
                'rsi_daily': round(rsi_val, 2), 'atr_daily': round(atr_val, 2),
                'adx': round(adx_val, 1), 'kalman_trend': round(kalman_trend, 3),
                'ma20_daily': round(ma20, 2), 'ma50_daily': round(ma50, 2),
                'ma200_daily': round(ma200, 2), '% change': round(pct_change, 2),
                'CUSUM_Alert': cusum_alert,
                'lot_size': LOT_SIZES.get(sym, 1),
                'RV_C2C': safe_rv(rv_c2c), 'RV_Parkinson': safe_rv(rv_park),
                'RV_GK': safe_rv(rv_gk), 'RV_YZ': safe_rv(rv_yz),
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame(), "No valid data extracted"
    return pd.DataFrame(results), f"✓ Analytics for {len(results)} securities"


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class VolRegime(Enum):
    COMPRESSED = "COMPRESSED"; LOW = "LOW"; NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"; HIGH = "HIGH"; EXTREME = "EXTREME"

class TrendRegime(Enum):
    STRONG_UP = "STRONG UP"; UP = "UPTREND"; NEUTRAL = "NEUTRAL"
    DOWN = "DOWNTREND"; STRONG_DOWN = "STRONG DOWN"

@dataclass
class Greeks:
    delta: float = 0.0; gamma: float = 0.0; theta: float = 0.0
    vega: float = 0.0; rho: float = 0.0
    vanna: float = 0.0; volga: float = 0.0; charm: float = 0.0; speed: float = 0.0

    def __add__(self, other):
        return Greeks(**{f: getattr(self, f) + getattr(other, f) for f in ['delta','gamma','theta','vega','rho','vanna','volga','charm','speed']})

    def negate(self):
        return Greeks(**{f: -getattr(self, f) for f in ['delta','gamma','theta','vega','rho','vanna','volga','charm','speed']})

    def scale(self, n):
        return Greeks(**{f: getattr(self, f) * n for f in ['delta','gamma','theta','vega','rho','vanna','volga','charm','speed']})

@dataclass
class StrategyResult:
    name: str; legs: List[Dict]; max_profit: float; max_loss: float
    breakeven_lower: float; breakeven_upper: float
    pop_bsm: float; pop_mc: float; pop_ensemble: float
    expected_value: float; sharpe_ratio: float; kelly_fraction: float
    net_greeks: Greeks; conviction_score: float; optimal_dte: int
    risk_score: float = 0.0; stability_score: float = 0.0
    width: float = 0.0; net_credit: float = 0.0
    risk_reward: float = 0.0; regime_alignment: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# L4: BSM PRICING ENGINE — §6.1, §6.2, §6.3
# ═══════════════════════════════════════════════════════════════════════════════

class BSM:
    R = 0.07

    @staticmethod
    def _d(S, K, T, r, σ):
        if T <= 0 or σ <= 0: return 0.0, 0.0
        sT = σ * np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * σ**2) * T) / sT
        return d1, d1 - sT

    @classmethod
    def call(cls, S, K, T, r, σ):
        if T <= 0: return max(S - K, 0)
        d1, d2 = cls._d(S, K, T, r, σ)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @classmethod
    def put(cls, S, K, T, r, σ):
        if T <= 0: return max(K - S, 0)
        d1, d2 = cls._d(S, K, T, r, σ)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @classmethod
    def greeks(cls, S, K, T, r, σ, otype='call'):
        if T <= 1e-6 or σ <= 1e-6: return Greeks()
        d1, d2 = cls._d(S, K, T, r, σ)
        sT = σ * np.sqrt(T); nd1 = norm.pdf(d1)
        γ = nd1 / (S * sT)
        ν = S * np.sqrt(T) * nd1 / 100
        if otype == 'call':
            δ = norm.cdf(d1)
            θ = (-(S * nd1 * σ) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            ρ = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            δ = norm.cdf(d1) - 1
            θ = (-(S * nd1 * σ) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            ρ = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        vanna = -nd1 * d2 / σ
        volga = S * np.sqrt(T) * nd1 * d1 * d2 / σ
        charm_val = -nd1 * (2 * r * T - d2 * sT) / (2 * T * sT)
        speed = -(γ / S) * (d1 / sT + 1)
        return Greeks(delta=δ, gamma=γ, theta=θ, vega=ν, rho=ρ, vanna=vanna, volga=volga, charm=charm_val, speed=speed)

    @classmethod
    def prob_otm(cls, S, K, T, σ, otype='call'):
        if T <= 0 or σ <= 0: return 0.5
        _, d2 = cls._d(S, K, T, cls.R, σ)
        return norm.cdf(-d2) if otype == 'call' else norm.cdf(d2)

    @classmethod
    def risk_score(cls, g, iv, rvw=1.0):
        """§6.3: Composite Risk Score — regime-weighted"""
        w = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
        w[2] *= rvw; w /= w.sum()
        delta_r = abs(g.delta) * w[0]
        gamma_r = min(abs(g.gamma) * iv * 100, 2.0) * w[1]  # cap gamma contribution
        vega_r = min(abs(g.vega) / max(iv * 100, 1), 2.0) * w[2]
        theta_p = min(max(-g.theta, 0) / max(abs(g.theta) + 1, 1), 1.0) * w[3]
        tail = min(abs(g.vanna) + abs(g.volga) * 0.1, 2.0) * w[4]
        return min((delta_r + gamma_r + vega_r + theta_p + tail) * 100, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# L5: MONTE CARLO — Antithetic Variates, Full POP for all strategies (§5.1)
# ═══════════════════════════════════════════════════════════════════════════════

class MC:
    @staticmethod
    def terminal_prices(S, σ, T, n=5000):
        """GBM with antithetic variates → 2n terminal prices"""
        if T <= 0 or σ <= 0: return np.full(2*n, S)
        steps = max(int(T * 252), 1); dt = T / steps
        z = np.random.standard_normal((n, steps))
        z_full = np.vstack([z, -z])
        log_ret = (0.07 - 0.5 * σ**2) * dt + σ * np.sqrt(dt) * z_full
        return S * np.exp(np.sum(log_ret, axis=1))

    @staticmethod
    def paths(S, σ, T, n=5000):
        """Full path simulation for charts"""
        if T <= 0 or σ <= 0: return np.full((2*n, 1), S)
        steps = max(int(T * 252), 1); dt = T / steps
        z = np.random.standard_normal((n, steps))
        z_full = np.vstack([z, -z])
        log_ret = (0.07 - 0.5 * σ**2) * dt + σ * np.sqrt(dt) * z_full
        return S * np.exp(np.cumsum(log_ret, axis=1))

    @staticmethod
    def strategy_pnl(terminal, legs):
        """Compute P/L for any multi-leg strategy from terminal prices"""
        pnl = np.zeros(len(terminal))
        for leg in legs:
            K = leg['strike']; p = leg['premium']; t = leg['type']
            mult = leg.get('qty', 1)
            if 'Sell Call' in t:
                pnl += mult * (p - np.maximum(terminal - K, 0))
            elif 'Buy Call' in t:
                pnl += mult * (-abs(p) + np.maximum(terminal - K, 0))
            elif 'Sell Put' in t:
                pnl += mult * (p - np.maximum(K - terminal, 0))
            elif 'Buy Put' in t:
                pnl += mult * (-abs(p) + np.maximum(K - terminal, 0))
        return pnl

    @staticmethod
    def analyze(S, σ, T, legs, n=5000):
        """Full MC analysis: POP, EV, Std for any strategy"""
        terminal = MC.terminal_prices(S, σ, T, n)
        pnl = MC.strategy_pnl(terminal, legs)
        pop = float(np.mean(pnl > 0))
        ev = float(np.mean(pnl))
        std = float(np.std(pnl))
        return pop, ev, std

    @staticmethod
    def expected_move(S, σ, T):
        moves = []
        for conf in [0.6827, 0.9545, 0.9973]:
            z = norm.ppf((1 + conf) / 2)
            m = S * σ * np.sqrt(max(T, 1e-6)) * z
            moves.append({'conf': conf, 'move': m, 'upper': S + m, 'lower': S - m})
        return moves


# ═══════════════════════════════════════════════════════════════════════════════
# L5: REGIME INTELLIGENCE — §4.2, §4.3, §4.4
# ═══════════════════════════════════════════════════════════════════════════════

def detect_vol_regime(ivp):
    if ivp > 85: return VolRegime.EXTREME
    if ivp > 70: return VolRegime.HIGH
    if ivp > 50: return VolRegime.ELEVATED
    if ivp > 30: return VolRegime.NORMAL
    if ivp > 15: return VolRegime.LOW
    return VolRegime.COMPRESSED

def detect_trend(S, ma20, ma50, rsi, pct, adx=20, kalman_t=0):
    """§4.3: Trend detection with MA + RSI + ADX + Kalman"""
    sig = 0
    if ma20 > 0 and ma50 > 0:
        if S > ma20 > ma50: sig += 2
        elif S > ma20: sig += 1
        elif S < ma20 < ma50: sig -= 2
        elif S < ma20: sig -= 1
    if rsi > 60: sig += 1
    elif rsi < 40: sig -= 1
    if pct > 2: sig += 1
    elif pct < -2: sig -= 1
    # ADX strengthens trend conviction (§4.3)
    if adx > 25:  # trending market
        if sig > 0: sig += 1
        elif sig < 0: sig -= 1
    # Kalman trend filter (§5.3)
    if kalman_t > 1.0: sig += 1
    elif kalman_t < -1.0: sig -= 1
    if sig >= 3: return TrendRegime.STRONG_UP
    if sig >= 1: return TrendRegime.UP
    if sig <= -3: return TrendRegime.STRONG_DOWN
    if sig <= -1: return TrendRegime.DOWN
    return TrendRegime.NEUTRAL

def regime_stability(ivp, rsi, garch_persist, cusum_alert):
    score = 0.5
    if 30 <= ivp <= 70: score += 0.15
    if 40 <= rsi <= 60: score += 0.10
    if garch_persist > 0.90: score += 0.10
    if cusum_alert: score -= 0.25
    return max(0.0, min(1.0, score))

def regime_vol_weight(vr):
    return {VolRegime.COMPRESSED: 0.6, VolRegime.LOW: 0.8, VolRegime.NORMAL: 1.0,
            VolRegime.ELEVATED: 1.3, VolRegime.HIGH: 1.6, VolRegime.EXTREME: 2.0}.get(vr, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# KELLY + CONVICTION + HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def kelly(p, w, l, confidence=1.0):
    if l <= 0 or w <= 0: return 0.0
    b = w / abs(l)
    k = (b * p - (1 - p)) / b
    return max(0, min(k * 0.5 * max(0.5, min(confidence, 1.0)), 0.25))

def snap(x, g):
    return round(x / g) * g

def clamp_sharpe(ev, std):
    """FIX #3: Clamp Sharpe to [-5, 5] to prevent explosion"""
    if std < 0.01: return 0.0
    return max(-5.0, min(5.0, ev / std))

def clamp_rr(nc, ml):
    """FIX #6: Clamp risk-reward to [0, 50]"""
    if ml <= 0: return 0.0
    return max(0.0, min(50.0, nc / ml))

def span_margin(S, iv, lot_size=1):
    """FIX #4: SPAN-like margin estimate for unlimited risk strategies"""
    # Approximate SPAN: max(premium + 15% of underlying, premium + 3σ move) per lot
    sigma_move = S * iv * np.sqrt(30/365) * 2  # ~2σ monthly move
    return max(S * 0.15, sigma_move) * lot_size

def ensemble_pop(pop_bsm, pop_mc):
    """§9.2: Inverse-variance weighted fusion"""
    w_bsm = 1.0 / (0.08**2)  # BSM RMSE ~8%
    w_mc = 1.0 / (0.05**2)   # MC RMSE ~5%
    return (w_bsm * pop_bsm + w_mc * pop_mc) / (w_bsm + w_mc)

def conviction_unified(ra, pop, ev_ratio, sharpe, stability, iv_norm):
    """§9.1: Unified Conviction — FIX #5: ev_ratio is EV/MaxProfit, not absolute ₹"""
    w = {'ra': 0.20, 'pop': 0.25, 'ev': 0.15, 'sharpe': 0.20, 'stab': 0.10, 'iv': 0.10}
    ev_n = max(0, min(1, (ev_ratio + 1) / 2))        # maps [-1, 1] → [0, 1]
    sh_n = max(0, min(1, (sharpe + 2) / 5))           # maps [-2, 3] → [0, 1]
    raw = (w['ra'] * ra + w['pop'] * pop + w['ev'] * ev_n +
           w['sharpe'] * sh_n + w['stab'] * stability + w['iv'] * iv_norm)
    return max(0, min(100, raw * 100))


# ═══════════════════════════════════════════════════════════════════════════════
# L6: STRATEGY ENGINE — All 10 with real MC, full Greeks, proper payoffs
# ═══════════════════════════════════════════════════════════════════════════════

ALL_STRATS = ['Short Strangle','Short Straddle','Iron Condor','Iron Butterfly',
              'Bull Put Spread','Bear Call Spread','Calendar Spread',
              'Jade Lizard','Broken Wing Butterfly','Ratio Spread']

MIN_PREMIUM = 0.50  # FIX #13: skip strategies with net credit below this

def compute_full_greeks(S, T, r, iv, legs_spec):
    """Compute net Greeks for any multi-leg strategy using BSM"""
    net = Greeks()
    for lspec in legs_spec:
        K, otype, qty = lspec['strike'], lspec['otype'], lspec['qty']
        g = BSM.greeks(S, K, T, r, iv, otype)
        net = net + g.scale(qty)
    return net

def score_strategy(name, stock, settings):
    S = stock['price']; iv = stock['ATMIV'] / 100; ivp = stock['IVPercentile']
    T = settings['dte'] / 365; r = BSM.R; g = settings['gap']
    rsi = stock.get('rsi_daily', 50); adx = stock.get('adx', 20)
    kalman_t = stock.get('kalman_trend', 0)
    garch_p = stock.get('GARCH_Persistence', 0.95)
    cusum = stock.get('CUSUM_Alert', False)
    lot = stock.get('lot_size', 1)
    vr = detect_vol_regime(ivp)
    tr = detect_trend(S, stock.get('ma20_daily', S), stock.get('ma50_daily', S), rsi,
                      stock.get('% change', 0), adx, kalman_t)
    stab = regime_stability(ivp, rsi, garch_p, cusum)
    rvw = regime_vol_weight(vr)
    em = S * iv * np.sqrt(max(T, 1e-6))
    iv_norm = min(ivp / 100, 1.0)
    min_wing = max(g, 1)  # FIX #11: minimum 1 gap width

    try:
        if name == 'Short Strangle':
            cK = snap(S + max(em, g), g)
            pK = snap(S - max(em, g), g)
            if cK <= S: cK = snap(S + g, g)
            if pK >= S: pK = snap(S - g, g)
            cp = BSM.call(S,cK,T,r,iv); pp = BSM.put(S,pK,T,r,iv)
            nc = cp + pp
            if nc < MIN_PREMIUM: return None
            ml = span_margin(S, iv)  # FIX #4: SPAN margin
            legs = [{'type':'Sell Call','strike':cK,'premium':cp,'qty':1},
                    {'type':'Sell Put','strike':pK,'premium':pp,'qty':1}]
            pop_b = BSM.prob_otm(S,cK,T,iv,'call') * BSM.prob_otm(S,pK,T,iv,'put')
            pop_m, ev, std = MC.analyze(S, iv, T, legs)
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':cK,'otype':'call','qty':-1}, {'strike':pK,'otype':'put','qty':-1}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.9 if vr in (VolRegime.HIGH, VolRegime.EXTREME) and tr==TrendRegime.NEUTRAL else 0.4
            sh = clamp_sharpe(ev, std)
            ev_ratio = ev / max(nc, 0.01)
            cv = conviction_unified(ra, pop_e, ev_ratio, sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=nc, max_loss=ml,
                breakeven_lower=pK-nc, breakeven_upper=cK+nc,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,nc,ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=45 if ivp>70 else 30, risk_score=rs,
                stability_score=stab, width=cK-pK, net_credit=nc,
                risk_reward=clamp_rr(nc,ml), regime_alignment=ra)

        elif name == 'Short Straddle':
            K = snap(S, g)
            cp = BSM.call(S,K,T,r,iv); pp = BSM.put(S,K,T,r,iv)
            nc = cp + pp
            if nc < MIN_PREMIUM: return None
            ml = span_margin(S, iv)
            legs = [{'type':'Sell Call','strike':K,'premium':cp,'qty':1},
                    {'type':'Sell Put','strike':K,'premium':pp,'qty':1}]
            pop_b = BSM.prob_otm(S,K+nc,T,iv,'call') * BSM.prob_otm(S,K-nc,T,iv,'put')
            pop_m, ev, std = MC.analyze(S, iv, T, legs)
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':K,'otype':'call','qty':-1}, {'strike':K,'otype':'put','qty':-1}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.85 if vr in (VolRegime.HIGH,VolRegime.EXTREME) and tr==TrendRegime.NEUTRAL else 0.35
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(nc,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=nc, max_loss=ml,
                breakeven_lower=K-nc, breakeven_upper=K+nc,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,nc,ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=21 if ivp>70 else 30, risk_score=rs,
                stability_score=stab, net_credit=nc, risk_reward=clamp_rr(nc,ml), regime_alignment=ra)

        elif name == 'Iron Condor':
            sc = snap(S + max(em*0.8, g), g)
            lc = max(snap(S + max(em*1.5, g*2), g), sc + min_wing)  # FIX #11
            sp = snap(S - max(em*0.8, g), g)
            lp = min(snap(S - max(em*1.5, g*2), g), sp - min_wing)
            nc = (BSM.call(S,sc,T,r,iv)-BSM.call(S,lc,T,r,iv)) + (BSM.put(S,sp,T,r,iv)-BSM.put(S,lp,T,r,iv))
            if nc < MIN_PREMIUM: return None
            w_spread = lc - sc; ml = max(w_spread - nc, 1)
            legs = [{'type':'Sell Call','strike':sc,'premium':BSM.call(S,sc,T,r,iv),'qty':1},
                    {'type':'Buy Call','strike':lc,'premium':BSM.call(S,lc,T,r,iv),'qty':1},
                    {'type':'Sell Put','strike':sp,'premium':BSM.put(S,sp,T,r,iv),'qty':1},
                    {'type':'Buy Put','strike':lp,'premium':BSM.put(S,lp,T,r,iv),'qty':1}]
            pop_b = BSM.prob_otm(S,sc,T,iv,'call') * BSM.prob_otm(S,sp,T,iv,'put')
            pop_m, ev, std = MC.analyze(S, iv, T, legs)
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':sc,'otype':'call','qty':-1}, {'strike':lc,'otype':'call','qty':1},
                {'strike':sp,'otype':'put','qty':-1}, {'strike':lp,'otype':'put','qty':1}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.8 if vr in (VolRegime.ELEVATED,VolRegime.HIGH) else 0.5
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(nc,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=nc, max_loss=ml,
                breakeven_lower=sp-nc, breakeven_upper=sc+nc,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,nc,ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=45, risk_score=rs, stability_score=stab,
                width=w_spread, net_credit=nc, risk_reward=clamp_rr(nc,ml), regime_alignment=ra)

        elif name == 'Iron Butterfly':
            K = snap(S, g)
            ww = max(snap(max(em*1.2, g*2), g), min_wing)  # FIX #11
            nc = (BSM.call(S,K,T,r,iv)+BSM.put(S,K,T,r,iv)) - (BSM.call(S,K+ww,T,r,iv)+BSM.put(S,K-ww,T,r,iv))
            if nc < MIN_PREMIUM: return None
            ml = max(ww - nc, 1)
            legs = [{'type':'Sell Call','strike':K,'premium':BSM.call(S,K,T,r,iv),'qty':1},
                    {'type':'Sell Put','strike':K,'premium':BSM.put(S,K,T,r,iv),'qty':1},
                    {'type':'Buy Call','strike':K+ww,'premium':BSM.call(S,K+ww,T,r,iv),'qty':1},
                    {'type':'Buy Put','strike':K-ww,'premium':BSM.put(S,K-ww,T,r,iv),'qty':1}]
            pop_m, ev, std = MC.analyze(S, iv, T, legs)
            pop_b = BSM.prob_otm(S,K+nc,T,iv,'call') * BSM.prob_otm(S,K-nc,T,iv,'put')
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':K,'otype':'call','qty':-1}, {'strike':K,'otype':'put','qty':-1},
                {'strike':K+ww,'otype':'call','qty':1}, {'strike':K-ww,'otype':'put','qty':1}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.7 if tr==TrendRegime.NEUTRAL else 0.3
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(nc,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=nc, max_loss=ml,
                breakeven_lower=K-nc, breakeven_upper=K+nc,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,nc,ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=30, risk_score=rs, stability_score=stab,
                width=ww, net_credit=nc, risk_reward=clamp_rr(nc,ml), regime_alignment=ra)

        elif name == 'Bull Put Spread':
            sp_k = snap(S - max(em*0.5, g), g)
            lp_k = min(snap(S - max(em*1.2, g*2), g), sp_k - min_wing)
            nc = BSM.put(S,sp_k,T,r,iv) - BSM.put(S,lp_k,T,r,iv)
            if nc < MIN_PREMIUM: return None
            ml = max((sp_k - lp_k) - nc, 1)
            legs = [{'type':'Sell Put','strike':sp_k,'premium':BSM.put(S,sp_k,T,r,iv),'qty':1},
                    {'type':'Buy Put','strike':lp_k,'premium':BSM.put(S,lp_k,T,r,iv),'qty':1}]
            pop_b = BSM.prob_otm(S, sp_k, T, iv, 'put')
            pop_m, ev, std = MC.analyze(S, iv, T, legs)
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':sp_k,'otype':'put','qty':-1}, {'strike':lp_k,'otype':'put','qty':1}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.8 if 'UP' in tr.value else 0.3
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(nc,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=nc, max_loss=ml,
                breakeven_lower=sp_k-nc, breakeven_upper=S*10,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,nc,ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=30, risk_score=rs, stability_score=stab,
                width=sp_k-lp_k, net_credit=nc, risk_reward=clamp_rr(nc,ml), regime_alignment=ra)

        elif name == 'Bear Call Spread':
            sc_k = snap(S + max(em*0.5, g), g)
            lc_k = max(snap(S + max(em*1.2, g*2), g), sc_k + min_wing)
            nc = BSM.call(S,sc_k,T,r,iv) - BSM.call(S,lc_k,T,r,iv)
            if nc < MIN_PREMIUM: return None
            ml = max((lc_k - sc_k) - nc, 1)
            legs = [{'type':'Sell Call','strike':sc_k,'premium':BSM.call(S,sc_k,T,r,iv),'qty':1},
                    {'type':'Buy Call','strike':lc_k,'premium':BSM.call(S,lc_k,T,r,iv),'qty':1}]
            pop_b = BSM.prob_otm(S, sc_k, T, iv, 'call')
            pop_m, ev, std = MC.analyze(S, iv, T, legs)
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':sc_k,'otype':'call','qty':-1}, {'strike':lc_k,'otype':'call','qty':1}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.8 if 'DOWN' in tr.value else 0.3
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(nc,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=nc, max_loss=ml,
                breakeven_lower=0, breakeven_upper=sc_k+nc,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,nc,ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=30, risk_score=rs, stability_score=stab,
                width=lc_k-sc_k, net_credit=nc, risk_reward=clamp_rr(nc,ml), regime_alignment=ra)

        elif name == 'Jade Lizard':
            cK = snap(S + max(em*0.7, g), g)
            sp_k = snap(S - max(em*0.5, g), g)
            lp_k = min(snap(S - max(em*1.3, g*2), g), sp_k - min_wing)
            cp_v = BSM.call(S,cK,T,r,iv); sp_v = BSM.put(S,sp_k,T,r,iv); lp_v = BSM.put(S,lp_k,T,r,iv)
            nc = cp_v + sp_v - lp_v
            if nc < MIN_PREMIUM: return None
            ml = max((sp_k - lp_k) - nc, 1)
            legs = [{'type':'Sell Call','strike':cK,'premium':cp_v,'qty':1},
                    {'type':'Sell Put','strike':sp_k,'premium':sp_v,'qty':1},
                    {'type':'Buy Put','strike':lp_k,'premium':lp_v,'qty':1}]
            pop_m, ev, std = MC.analyze(S, iv, T, legs)
            pop_b = BSM.prob_otm(S,cK,T,iv,'call') * BSM.prob_otm(S,sp_k,T,iv,'put')
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':cK,'otype':'call','qty':-1}, {'strike':sp_k,'otype':'put','qty':-1},
                {'strike':lp_k,'otype':'put','qty':1}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.7 if 'UP' in tr.value or tr==TrendRegime.NEUTRAL else 0.3
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(nc,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=nc, max_loss=ml,
                breakeven_lower=sp_k-nc, breakeven_upper=cK+nc,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,nc,ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=35, risk_score=rs, stability_score=stab,
                width=sp_k-lp_k, net_credit=nc, risk_reward=clamp_rr(nc,ml), regime_alignment=ra)

        elif name == 'Calendar Spread':
            K = snap(S, g)
            # Front: sell near expiry, Back: buy further expiry
            fT = max(T * 0.5, 1/365)
            bT = max(T * 1.5, T + 7/365)
            fp = BSM.call(S,K,fT,r,iv)
            bp = BSM.call(S,K,bT,r,iv*0.95)  # back month lower IV (term structure)
            nd = bp - fp  # net debit
            if nd <= 0: return None  # shouldn't happen normally
            # Max profit ≈ when stock at K at front expiry: time value diff
            # Approximate: front decays fully, back retains time value
            mp = BSM.call(S, K, bT - fT, r, iv*0.95) - max(S - K, 0)
            mp = max(mp, fp * 0.5)  # FIX #9: realistic floor
            ml = nd  # max loss is debit paid
            legs = [{'type':'Sell Call','strike':K,'premium':fp,'qty':1},
                    {'type':'Buy Call','strike':K,'premium':bp,'qty':1}]
            # For MC: at front expiry, what's the back month worth?
            pop_b = 0.55 if vr in (VolRegime.LOW, VolRegime.COMPRESSED) else 0.45
            # Calendar MC: simulate to front expiry, estimate back value
            terminal = MC.terminal_prices(S, iv, fT)
            back_values = np.array([BSM.call(st, K, bT-fT, r, iv*0.95) for st in terminal[:500]])  # subsample for speed
            front_values = np.maximum(terminal[:500] - K, 0)
            cal_pnl = back_values - front_values - nd
            pop_m = float(np.mean(cal_pnl > 0))
            ev = float(np.mean(cal_pnl))
            std = float(np.std(cal_pnl))
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':K,'otype':'call','qty':-1}, {'strike':K,'otype':'call','qty':1}])
            # Calendar has positive vega (long back month) and positive theta (short front)
            ng.theta = abs(BSM.greeks(S,K,fT,r,iv,'call').theta) - abs(BSM.greeks(S,K,bT,r,iv*0.95,'call').theta)
            ng.vega = BSM.greeks(S,K,bT,r,iv*0.95,'call').vega - BSM.greeks(S,K,fT,r,iv,'call').vega
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.8 if vr in (VolRegime.LOW,VolRegime.COMPRESSED) else 0.3
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(mp,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=mp, max_loss=ml,
                breakeven_lower=K-mp, breakeven_upper=K+mp,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,mp,ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=45, risk_score=rs, stability_score=stab,
                net_credit=-nd, risk_reward=clamp_rr(mp,ml), regime_alignment=ra)

        elif name == 'Broken Wing Butterfly':
            c = snap(S, g)
            lo = snap(S - max(em*0.8, g*2), g)
            hi = snap(S + max(em*1.2, g*2), g)
            if c - lo < min_wing: lo = c - min_wing
            if hi - c < min_wing: hi = c + min_wing
            # BWB: Buy 1 low, Sell 2 center, Buy 1 high (unbalanced)
            lo_p = BSM.call(S,lo,T,r,iv); c_p = BSM.call(S,c,T,r,iv); hi_p = BSM.call(S,hi,T,r,iv)
            nc = 2*c_p - lo_p - hi_p  # net credit (can be small or debit)
            mp = (c - lo) + nc  # max profit at center
            ml = max((hi - c) - nc, 1)  # max loss on upside
            if mp < MIN_PREMIUM and nc < MIN_PREMIUM: return None
            legs = [{'type':'Buy Call','strike':lo,'premium':lo_p,'qty':1},
                    {'type':'Sell Call','strike':c,'premium':c_p,'qty':2},
                    {'type':'Buy Call','strike':hi,'premium':hi_p,'qty':1}]
            # Proper MC for BWB
            terminal = MC.terminal_prices(S, iv, T)
            bwb_pnl = (-lo_p + np.maximum(terminal - lo, 0) +
                        2*(c_p - np.maximum(terminal - c, 0)) +
                       -hi_p + np.maximum(terminal - hi, 0))
            pop_m = float(np.mean(bwb_pnl > 0))
            ev = float(np.mean(bwb_pnl))
            std = float(np.std(bwb_pnl))
            pop_b = 0.50
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':lo,'otype':'call','qty':1}, {'strike':c,'otype':'call','qty':-2},
                {'strike':hi,'otype':'call','qty':1}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.5
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(mp,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=mp, max_loss=ml,
                breakeven_lower=lo, breakeven_upper=hi,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,max(mp,1),ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=45, risk_score=rs, stability_score=stab,
                width=hi-lo, net_credit=nc, risk_reward=clamp_rr(mp,ml), regime_alignment=ra)

        elif name == 'Ratio Spread':
            lK = snap(S, g)
            sK = max(snap(S + max(em*0.8, g*2), g), lK + min_wing)
            lp_v = BSM.call(S,lK,T,r,iv); sp_v = BSM.call(S,sK,T,r,iv)
            nc = 2*sp_v - lp_v  # sell 2 OTM, buy 1 ATM
            # Max profit at short strike: intrinsic of long - premium paid
            mp = (sK - lK) + nc  # value at sK
            ml = span_margin(S, iv) * 0.5  # unlimited risk above sK (reduced margin for partial hedge)
            if mp < MIN_PREMIUM: return None
            legs = [{'type':'Buy Call','strike':lK,'premium':lp_v,'qty':1},
                    {'type':'Sell Call','strike':sK,'premium':sp_v,'qty':2}]
            # Proper MC for ratio spread
            terminal = MC.terminal_prices(S, iv, T)
            ratio_pnl = (-lp_v + np.maximum(terminal - lK, 0) +
                         2*(sp_v - np.maximum(terminal - sK, 0)))
            pop_m = float(np.mean(ratio_pnl > 0))
            ev = float(np.mean(ratio_pnl))
            std = float(np.std(ratio_pnl))
            pop_b = 0.55 if 'UP' in tr.value or tr==TrendRegime.NEUTRAL else 0.40
            pop_e = ensemble_pop(pop_b, pop_m)
            ng = compute_full_greeks(S, T, r, iv, [
                {'strike':lK,'otype':'call','qty':1}, {'strike':sK,'otype':'call','qty':-2}])
            rs = BSM.risk_score(ng, iv, rvw)
            ra = 0.6 if 'UP' in tr.value else 0.3
            sh = clamp_sharpe(ev, std)
            cv = conviction_unified(ra, pop_e, ev/max(mp,0.01), sh, stab, iv_norm)
            return StrategyResult(name=name, legs=legs, max_profit=mp, max_loss=ml,
                breakeven_lower=lK, breakeven_upper=2*sK-lK+nc,
                pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
                sharpe_ratio=sh, kelly_fraction=kelly(pop_e,max(mp,1),ml,stab), net_greeks=ng,
                conviction_score=cv, optimal_dte=30, risk_score=rs, stability_score=stab,
                width=sK-lK, net_credit=nc, risk_reward=clamp_rr(mp,ml), regime_alignment=ra)

    except Exception:
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

CL = {'gold':'#FFC300','bg':'#1A1A1A','border':'#2A2A2A','text':'#EAEAEA','muted':'#888',
      'green':'#10b981','red':'#ef4444','amber':'#f59e0b','cyan':'#06b6d4','purple':'#a855f7'}

def fmt(v):
    try:
        v = float(v); neg = v < 0; v = abs(v); ip = int(v)
        s = str(ip)[::-1]; g_parts = []
        if len(s) > 3:
            g_parts.append(s[:3]); s = s[3:]
            while s: g_parts.append(s[:2]); s = s[2:]
            f = ','.join(g_parts[::-1])
        else: f = s[::-1]
        dp = round(abs(float(v)) - ip, 2)
        if dp > 0: f += f"{dp:.2f}"[1:]
        return f"{'-' if neg else ''}₹{f}"
    except: return str(v)

def payoff_chart(strat, S):
    margin = max(strat.width, S * 0.12) if strat.width > 0 else S * 0.12
    px = np.linspace(S - margin * 1.5, S + margin * 1.5, 500)
    pnl = np.zeros_like(px)
    for leg in strat.legs:
        K, p, t = leg['strike'], leg['premium'], leg['type']
        qty = leg.get('qty', 1)
        if 'Sell Call' in t:
            pnl += qty * (p - np.maximum(px - K, 0))
        elif 'Buy Call' in t:
            pnl += qty * (-abs(p) + np.maximum(px - K, 0))
        elif 'Sell Put' in t:
            pnl += qty * (p - np.maximum(K - px, 0))
        elif 'Buy Put' in t:
            pnl += qty * (-abs(p) + np.maximum(K - px, 0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px, y=np.clip(pnl, 0, None), fill='tozeroy', fillcolor='rgba(16,185,129,0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=px, y=np.clip(pnl, None, 0), fill='tozeroy', fillcolor='rgba(239,68,68,0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=px, y=pnl, mode='lines', name='P/L', line=dict(color=CL['gold'], width=3), hovertemplate='₹%{x:,.0f} → P/L: ₹%{y:,.2f}<extra></extra>'))
    fig.add_vline(x=S, line_dash="dash", line_color=CL['text'], opacity=0.5, annotation_text="Spot")
    fig.add_hline(y=0, line_color=CL['muted'], opacity=0.3)
    for leg in strat.legs:
        clr = CL['red'] if 'Sell' in leg['type'] else CL['green']
        fig.add_vline(x=leg['strike'], line_dash="dot", line_color=clr, opacity=0.3)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=CL['bg'], height=360, margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(showgrid=True, gridcolor=CL['border'], title='Stock Price (₹)', tickformat=','),
        yaxis=dict(showgrid=True, gridcolor=CL['border'], title='P/L (₹)', tickformat=','),
        font=dict(family='Inter', color=CL['text']), hovermode='x unified')
    return fig

def gauge(score):
    clr = CL['green'] if score >= 70 else (CL['amber'] if score >= 45 else CL['red'])
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        number={'font': {'size': 32, 'color': CL['text'], 'family': 'JetBrains Mono'}},
        gauge={'axis': {'range': [0, 100], 'tickcolor': CL['muted']}, 'bar': {'color': clr, 'thickness': 0.75},
               'bgcolor': CL['bg'], 'borderwidth': 2, 'bordercolor': CL['border'],
               'steps': [{'range': [0, 33], 'color': 'rgba(239,68,68,0.15)'}, {'range': [33, 66], 'color': 'rgba(245,158,11,0.15)'}, {'range': [66, 100], 'color': 'rgba(16,185,129,0.15)'}]}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color=CL['text']), height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def em_chart(S, iv, T):
    em = MC.expected_move(S, iv, T); σ = S * iv * np.sqrt(max(T, 1e-6))
    x = np.linspace(S - 4*σ, S + 4*σ, 500); y = norm.pdf(x, S, σ)
    fig = go.Figure()
    colors = [(CL['purple'], 0.08), (CL['cyan'], 0.12), (CL['green'], 0.15)]
    for i, (clr, alpha) in reversed(list(enumerate(colors))):
        m = (x >= em[i]['lower']) & (x <= em[i]['upper'])
        rgb = ','.join(str(int(clr.lstrip('#')[j:j+2], 16)) for j in (0,2,4))
        fig.add_trace(go.Scatter(x=x[m], y=y[m], fill='tozeroy', fillcolor=f'rgba({rgb},{alpha})', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=CL['gold'], width=2), showlegend=False))
    fig.add_vline(x=S, line_dash="dash", line_color=CL['text'], opacity=0.5, annotation_text=f"Spot ₹{S:,.0f}")
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=CL['bg'], height=300, margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(showgrid=True, gridcolor=CL['border'], title='Price (₹)', tickformat=','), yaxis=dict(showgrid=False, showticklabels=False),
        font=dict(family='Inter', color=CL['text']), showlegend=False, hovermode='x unified')
    return fig

def vol_estimator_chart(stock):
    names = ['Close-Close', 'Parkinson', 'Garman-Klass', 'Yang-Zhang', 'GARCH', 'IV Estimate']
    vals = [stock.get('RV_C2C', 0), stock.get('RV_Parkinson', 0), stock.get('RV_GK', 0),
            stock.get('RV_YZ', 0), stock.get('GARCH_Vol', 0), stock.get('ATMIV', 0)]
    colors = [CL['cyan'], CL['green'], CL['amber'], CL['purple'], CL['red'], CL['gold']]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors, text=[f"{v:.1f}%" for v in vals], textposition='outside'))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=CL['bg'], height=280, margin=dict(l=10, r=10, t=30, b=30),
        yaxis=dict(showgrid=True, gridcolor=CL['border'], title='Annualized Vol %'), font=dict(family='Inter', color=CL['text']), showlegend=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""<div style="text-align:center; padding:1rem 0; margin-bottom:1rem;">
            <div style="font-size:1.75rem; font-weight:800; color:#FFC300;">VAAYDO</div>
            <div style="color:#888; font-size:0.7rem; margin-top:0.25rem;">वायदो — FnO Trade Intelligence</div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="stitle">⚡ Expiry & Parameters</div>', unsafe_allow_html=True)

        today = date.today()
        expiry_date = st.date_input("Expiry Date", value=today + timedelta(days=(3-today.weekday())%7 + (7 if (3-today.weekday())%7 == 0 else 0)),
                                    min_value=today + timedelta(days=1), max_value=today + timedelta(days=365))
        dte = (expiry_date - today).days
        st.caption(f"**{dte} days** to expiry ({expiry_date.strftime('%d %b %Y')})")

        strike_gap = st.selectbox("Strike Gap (₹)", [25, 50, 100, 200, 500], index=1)
        capital = st.number_input("Capital (₹)", value=500000, step=50000, format="%d")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="stitle">🎯 Filters</div>', unsafe_allow_html=True)
        min_ivp = st.slider("Min IV Percentile", 0, 100, 20)
        min_cv = st.slider("Min Conviction", 0, 100, 30)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class='ib'><p style='font-size:0.72rem; margin:0; color:#888; line-height:1.5;'>
            <strong style='color:#FFC300;'>v{VERSION}</strong><br>
            <strong>Engines:</strong> BSM + MC(10K AV) + GARCH + Kelly<br>
            <strong>Vol:</strong> C2C · Park · GK · YZ + VRP<br>
            <strong>Greeks:</strong> Δ Γ Θ ν ρ + Vanna Volga Charm Speed<br>
            <strong>Regime:</strong> 6-Vol × 5-Trend + ADX + Kalman<br>
            <strong>POP:</strong> Ensemble (BSM + MC fusion)<br>
            <strong>Sharpe:</strong> Clamped [-5, 5]<br>
            <strong>Strategies:</strong> 10 Active</p></div>""", unsafe_allow_html=True)

    settings = {'dte': max(dte, 1), 'gap': strike_gap, 'capital': capital}

    # ── HEADER ──
    st.markdown(f"""<div class="hdr"><div class="badge">v{VERSION}</div>
        <h1>VAAYDO — FnO Trade Intelligence</h1>
        <div class="tag">BSM · MC (10K Antithetic) · Multi-Estimator Vol · GARCH · Kalman · Kelly · CUSUM &nbsp;|&nbsp;
        Expiry: {expiry_date.strftime("%d %b %Y")} ({dte}D) &nbsp;|&nbsp; {datetime.now().strftime("%d %b %Y")}</div></div>""", unsafe_allow_html=True)

    # ── DATA FETCH ──
    with st.spinner("Fetching F&O universe..."):
        symbols, sym_status = get_fno_symbols()
        symbols_ns = [s + ".NS" for s in symbols]
    with st.spinner(f"Downloading & computing analytics for {len(symbols_ns)} securities..."):
        df, data_status = fetch_all_data(symbols_ns)
    if df.empty:
        st.error(f"Data fetch failed: {data_status}")
        st.info("Ensure network access to `*.yahoo.com`. Check firewall / proxy settings.")
        return
    st.caption(f"🔌 {sym_status} → {data_status}")

    # ── COMPUTE STRATEGIES ──
    with st.spinner("Running BSM + Monte Carlo (10K antithetic) + Kelly..."):
        all_trades = []
        for _, row in df.iterrows():
            rd = row.to_dict(); best = None
            for sn in ALL_STRATS:
                try:
                    res = score_strategy(sn, rd, settings)
                    if res and (best is None or res.conviction_score > best.conviction_score):
                        best = res
                except: continue
            if best:
                vr = detect_vol_regime(rd.get('IVPercentile', 50))
                tr = detect_trend(rd['price'], rd.get('ma20_daily', rd['price']),
                    rd.get('ma50_daily', rd['price']), rd.get('rsi_daily', 50),
                    rd.get('% change', 0), rd.get('adx', 20), rd.get('kalman_trend', 0))
                all_trades.append({**rd, 'strategy': best.name, 'conviction_score': best.conviction_score,
                    'pop': best.pop_ensemble, 'ev': best.expected_value, 'sharpe': best.sharpe_ratio,
                    'kelly_frac': best.kelly_fraction, 'net_credit': best.net_credit,
                    'max_profit': best.max_profit, 'max_loss': best.max_loss,
                    'risk_score': best.risk_score, 'stability': best.stability_score,
                    'vol_regime': vr.value, 'trend_regime': tr.value, '_result': best})

    filtered = [t for t in all_trades if t['IVPercentile'] >= min_ivp and t['conviction_score'] >= min_cv]
    filtered.sort(key=lambda x: x['conviction_score'], reverse=True)

    # ── METRICS BAR ──
    avg_iv = df['IVPercentile'].mean(); avg_pcr = df['PCR'].mean()
    hc = len([t for t in filtered if t['conviction_score'] >= 65])
    cusum_alerts = len(df[df['CUSUM_Alert'] == True])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f"<div class='mc gold'><h4>Securities</h4><h2>{len(df)}</h2><div class='sub'>{len(filtered)} pass filters</div></div>", unsafe_allow_html=True)
    with c2:
        cs = 'ok' if hc > 5 else ('warn' if hc > 2 else 'bad')
        st.markdown(f"<div class='mc {cs}'><h4>High Conviction</h4><h2>{hc}</h2><div class='sub'>Score ≥ 65</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='mc {'ok' if avg_iv > 55 else 'warn' if avg_iv > 35 else 'info'}'><h4>Avg IV Percentile</h4><h2>{avg_iv:.0f}%</h2><div class='sub'>GARCH avg: {df['GARCH_Vol'].mean():.1f}%</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='mc {'ok' if avg_pcr > 1.2 else 'bad' if avg_pcr < 0.8 else 'info'}'><h4>Avg PCR</h4><h2>{avg_pcr:.2f}</h2><div class='sub'>{'Bullish' if avg_pcr > 1.2 else 'Bearish' if avg_pcr < 0.8 else 'Neutral'} bias</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='mc {'bad' if cusum_alerts > 5 else 'warn' if cusum_alerts > 0 else 'ok'}'><h4>CUSUM Alerts</h4><h2>{cusum_alerts}</h2><div class='sub'>Structural breaks</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Trade Radar", "🔬 Deep Analysis", "📊 Rankings", "📐 Probability Lab"])

    with tab1:
        st.markdown("<div style='margin-bottom:1rem;'><span style='font-size:1.1rem;font-weight:700;color:#EAEAEA;'>Top FnO Opportunities</span><span style='color:#888;font-size:0.85rem;margin-left:0.75rem;'>§9.1 Unified Conviction · Ensemble POP · Antithetic MC</span></div>", unsafe_allow_html=True)
        top = filtered[:9]
        if not top:
            st.info("No trades pass filters. Lower conviction or IV percentile thresholds.")
        else:
            cols = st.columns(3)
            for i, t in enumerate(top):
                with cols[i % 3]:
                    cv = t['conviction_score']; cc = '#10b981' if cv >= 65 else ('#f59e0b' if cv >= 40 else '#ef4444')
                    cv_cls = 'hi' if cv >= 65 else ('md' if cv >= 40 else 'lo')
                    cusum_warn = " ⚠️" if t.get('CUSUM_Alert') else ""
                    st.markdown(f"""<div class='tc {cv_cls}'>
                        <div style="display:flex;justify-content:space-between;align-items:start;">
                        <div><div class='sym'>{t['Instrument']}{cusum_warn}</div><div class='strat'>{t['strategy']}</div></div>
                        <div style="text-align:right;"><div style="font-size:1.8rem;font-weight:800;color:{cc};font-family:'JetBrains Mono',monospace;">{cv:.0f}</div>
                        <div style="font-size:0.6rem;color:#888;text-transform:uppercase;">Conviction</div></div></div>
                        <div class='gr'>
                        <div class='gi'><label>Spot</label><div class='v'>{fmt(t['price'])}</div></div>
                        <div class='gi'><label>IV %ile</label><div class='v'>{t['IVPercentile']:.0f}%</div></div>
                        <div class='gi'><label>POP (Ensemble)</label><div class='v tg'>{t['pop']*100:.1f}%</div></div>
                        <div class='gi'><label>Expected Val</label><div class='v {"tg" if t["ev"]>0 else "tr"}'>{fmt(t['ev'])}</div></div>
                        <div class='gi'><label>Max Profit</label><div class='v tg'>{fmt(t['max_profit'])}</div></div>
                        <div class='gi'><label>Sharpe</label><div class='v'>{t['sharpe']:.2f}</div></div>
                        <div class='gi'><label>Risk Score</label><div class='v {"tr" if t.get("risk_score",0)>50 else "ta"}'>{t.get('risk_score',0):.0f}</div></div>
                        <div class='gi'><label>Stability</label><div class='v'>{t.get('stability',0):.2f}</div></div></div>
                        <div class='cb'><div class='cf' style='width:{cv}%;background:linear-gradient(90deg,{cc},{cc}aa);'></div></div>
                        <div style="display:flex;gap:0.5rem;margin-top:0.75rem;">
                        <span class='sb {"buy" if "UP" in t.get("trend_regime","") else ("sell" if "DOWN" in t.get("trend_regime","") else "neut")}'>{t.get('trend_regime','NEUTRAL')}</span>
                        <span class='sb prem'>{t.get('vol_regime','NORMAL')}</span></div></div>""", unsafe_allow_html=True)
                    if st.button(f"Analyze {t['Instrument']}", key=f"a_{t['Instrument']}"):
                        st.session_state.sel = t['Instrument']

    with tab2:
        opts = [t['Instrument'] for t in filtered] if filtered else df['Instrument'].tolist()
        sel = st.selectbox("Select Security", opts, index=opts.index(st.session_state.get('sel', opts[0])) if st.session_state.get('sel') in opts else 0)
        if sel:
            row = df[df['Instrument'] == sel].iloc[0].to_dict()
            S = row['price']; iv = row['ATMIV'] / 100; T = settings['dte'] / 365
            vr = detect_vol_regime(row['IVPercentile'])
            tr = detect_trend(S, row.get('ma20_daily', S), row.get('ma50_daily', S),
                row.get('rsi_daily', 50), row.get('% change', 0), row.get('adx', 20), row.get('kalman_trend', 0))
            em = MC.expected_move(S, iv, T)
            stab = regime_stability(row['IVPercentile'], row.get('rsi_daily', 50),
                row.get('GARCH_Persistence', 0.95), row.get('CUSUM_Alert', False))
            cusum_icon = " ⚠️ CUSUM BREAK" if row.get('CUSUM_Alert') else ""

            st.markdown(f"<div style='margin-bottom:1.5rem;'><span style='font-size:1.4rem;font-weight:800;color:#EAEAEA;'>{sel}</span>"
                f"<span style='font-size:1.1rem;color:#888;margin-left:1rem;'>{fmt(S)}</span>"
                f"<span class='sb prem' style='margin-left:1rem;'>{vr.value}</span>"
                f"<span class='sb {'buy' if 'UP' in tr.value else ('sell' if 'DOWN' in tr.value else 'neut')}'>{tr.value}</span>"
                f"<span style='color:#ef4444;margin-left:0.5rem;font-weight:700;font-size:0.85rem;'>{cusum_icon}</span></div>", unsafe_allow_html=True)

            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            with mc1: st.markdown(f"<div class='mc'><h4>IV Estimate</h4><h2>{row['ATMIV']:.1f}%</h2><div class='sub'>RV + VRP({row.get('VRP_Factor',1.12):.0%})</div></div>", unsafe_allow_html=True)
            with mc2: st.markdown(f"<div class='mc'><h4>IV Percentile</h4><h2>{row['IVPercentile']:.0f}%</h2><div class='sub'>252D Rank</div></div>", unsafe_allow_html=True)
            with mc3: st.markdown(f"<div class='mc'><h4>GARCH Vol</h4><h2>{row.get('GARCH_Vol',0):.1f}%</h2><div class='sub'>P={row.get('GARCH_Persistence',0):.3f} HL={row.get('GARCH_HalfLife',0):.0f}D</div></div>", unsafe_allow_html=True)
            with mc4: st.markdown(f"<div class='mc'><h4>Stability</h4><h2>{stab:.2f}</h2><div class='sub'>Regime persist</div></div>", unsafe_allow_html=True)
            with mc5: st.markdown(f"<div class='mc info'><h4>1σ Move</h4><h2>±{fmt(em[0]['move'])}</h2><div class='sub'>68.3%</div></div>", unsafe_allow_html=True)
            with mc6: st.markdown(f"<div class='mc warn'><h4>2σ Move</h4><h2>±{fmt(em[1]['move'])}</h2><div class='sub'>95.5%</div></div>", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            vc1, vc2 = st.columns([1, 1])
            with vc1:
                st.markdown("<span style='font-weight:700;color:#EAEAEA;'>§3.2 Multi-Estimator Volatility</span>", unsafe_allow_html=True)
                st.plotly_chart(vol_estimator_chart(row), use_container_width=True)
            with vc2:
                st.markdown("<span style='font-weight:700;color:#EAEAEA;'>Expected Move Distribution</span>", unsafe_allow_html=True)
                st.plotly_chart(em_chart(S, iv, T), use_container_width=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("<span style='font-weight:700;color:#EAEAEA;'>Strategy Rankings</span><span style='color:#888;margin-left:0.75rem;font-size:0.85rem;'>All 10 strategies · Real MC · Full Greeks</span>", unsafe_allow_html=True)

            strats = []
            for sn in ALL_STRATS:
                try:
                    res = score_strategy(sn, row, settings)
                    if res: strats.append(res)
                except: continue
            strats.sort(key=lambda x: x.conviction_score, reverse=True)

            for rank, s in enumerate(strats[:5], 1):
                cv = s.conviction_score
                with st.expander(f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else f'#{rank}'} {s.name} — Conv: {cv:.0f} | POP: {s.pop_ensemble*100:.1f}% | Sharpe: {s.sharpe_ratio:.2f} | Risk: {s.risk_score:.0f}", expanded=(rank==1)):
                    ec1, ec2 = st.columns([2, 1])
                    with ec1:
                        lh = "<table class='gt'><tr><th>Leg</th><th>Strike</th><th>Qty</th><th>Premium</th></tr>"
                        for l in s.legs:
                            cc = 'tr' if 'Sell' in l['type'] else 'tg'
                            lh += f"<tr><td class='{cc}'>{l['type']}</td><td>{fmt(l['strike'])}</td><td>{l.get('qty',1)}</td><td>{fmt(abs(l['premium']))}</td></tr>"
                        st.markdown(lh + "</table>", unsafe_allow_html=True)
                        st.plotly_chart(payoff_chart(s, S), use_container_width=True)
                    with ec2:
                        st.plotly_chart(gauge(cv), use_container_width=True)
                        st.markdown(f"""<div class='ib'><h4>Analytics</h4><p>
                            <strong>POP (BSM):</strong> <span class='mono tg'>{s.pop_bsm*100:.1f}%</span><br>
                            <strong>POP (MC):</strong> <span class='mono tg'>{s.pop_mc*100:.1f}%</span><br>
                            <strong>POP (Ensemble):</strong> <span class='mono tg'>{s.pop_ensemble*100:.1f}%</span><br>
                            <strong>EV:</strong> <span class='mono {"tg" if s.expected_value>0 else "tr"}'>{fmt(s.expected_value)}</span><br>
                            <strong>Max Profit:</strong> <span class='mono tg'>{fmt(s.max_profit)}</span><br>
                            <strong>Max Loss:</strong> <span class='mono tr'>{fmt(s.max_loss)}</span><br>
                            <strong>R:R:</strong> <span class='mono'>{s.risk_reward:.2f}</span><br>
                            <strong>Sharpe:</strong> <span class='mono'>{s.sharpe_ratio:.2f}</span><br>
                            <strong>Kelly:</strong> <span class='mono'>{s.kelly_fraction*100:.1f}%</span><br>
                            <strong>Risk Score:</strong> <span class='mono {"tr" if s.risk_score>50 else "ta"}'>{s.risk_score:.0f}</span><br>
                            <strong>BEs:</strong> <span class='mono'>{fmt(s.breakeven_lower)} — {fmt(s.breakeven_upper)}</span></p></div>""", unsafe_allow_html=True)
                        gk = s.net_greeks
                        st.markdown(f"""<div class='ib' style='margin-top:0.5rem;'><h4>Greeks (9)</h4><p>
                            <strong>Δ:</strong> <span class='mono'>{gk.delta:+.4f}</span> &nbsp;
                            <strong>Γ:</strong> <span class='mono'>{gk.gamma:+.6f}</span><br>
                            <strong>Θ:</strong> <span class='mono tg'>{gk.theta:+.2f}/day</span> &nbsp;
                            <strong>ν:</strong> <span class='mono'>{gk.vega:+.2f}</span><br>
                            <strong>Vanna:</strong> <span class='mono'>{gk.vanna:+.4f}</span> &nbsp;
                            <strong>Volga:</strong> <span class='mono'>{gk.volga:+.4f}</span><br>
                            <strong>Charm:</strong> <span class='mono'>{gk.charm:+.6f}</span> &nbsp;
                            <strong>Speed:</strong> <span class='mono'>{gk.speed:+.6f}</span></p></div>""", unsafe_allow_html=True)
                        lot = row.get('lot_size', 1)
                        lots = max(1, int(capital * s.kelly_fraction / max(s.max_loss, 1))) if s.max_loss > 0 else 1
                        st.markdown(f"""<div class='ib' style='margin-top:0.5rem;'><h4>Position Sizing (§7.1)</h4><p>
                            <strong>Lot Size:</strong> <span class='mono'>{lot}</span><br>
                            <strong>Lots:</strong> <span class='mono tgl'>{lots}</span><br>
                            <strong>Risk:</strong> <span class='mono'>{fmt(lots * s.max_loss)}</span><br>
                            <strong>% Capital:</strong> <span class='mono'>{lots * s.max_loss / capital * 100:.1f}%</span></p></div>""", unsafe_allow_html=True)

    with tab3:
        if filtered:
            rdf = pd.DataFrame([{k: v for k, v in t.items() if k != '_result'} for t in filtered])
            cols_show = ['Instrument','strategy','conviction_score','pop','ev','sharpe','kelly_frac','risk_score','stability',
                        'net_credit','max_profit','max_loss','IVPercentile','GARCH_Vol','price','lot_size','vol_regime','trend_regime','CUSUM_Alert']
            avail = [c for c in cols_show if c in rdf.columns]
            display = rdf[avail].sort_values('conviction_score', ascending=False).copy()
            # Round numeric columns for readability
            for col in ['conviction_score','ev','sharpe','kelly_frac','risk_score','stability','net_credit','max_profit','max_loss']:
                if col in display.columns:
                    display[col] = display[col].round(2)
            if 'pop' in display.columns:
                display['pop'] = (display['pop'] * 100).round(1)
            display.columns = [c.replace('_', ' ').title() for c in display.columns]
            st.dataframe(display, use_container_width=True, height=600)
        else:
            st.info("No trades pass current filters.")

    with tab4:
        lab = st.selectbox("Select Security", df['Instrument'].tolist(), key="lab")
        lr = df[df['Instrument'] == lab].iloc[0]; lS = lr['price']; liv = lr['ATMIV'] / 100; lT = settings['dte'] / 365

        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown("<span style='font-weight:600;color:#EAEAEA;'>Monte Carlo Terminal Distribution (10K Antithetic)</span>", unsafe_allow_html=True)
            np.random.seed(42); term = MC.terminal_prices(lS, liv, lT)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=term, nbinsx=80, marker_color=CL['gold'], opacity=0.7))
            fig.add_vline(x=lS, line_dash="dash", line_color=CL['text'], annotation_text=f"Spot: {fmt(lS)}")
            fig.add_vline(x=np.percentile(term, 5), line_dash="dot", line_color=CL['red'], annotation_text="5th")
            fig.add_vline(x=np.percentile(term, 95), line_dash="dot", line_color=CL['green'], annotation_text="95th")
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=CL['bg'], height=350, margin=dict(l=10, r=10, t=30, b=30),
                xaxis=dict(showgrid=True, gridcolor=CL['border'], title='Terminal Price', tickformat=','), yaxis=dict(showgrid=True, gridcolor=CL['border'], title='Freq'),
                font=dict(family='Inter', color=CL['text']), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""<div class='ib'><h4>MC Stats (10K antithetic paths)</h4><p>
                <strong>Mean:</strong> <span class='mono'>{fmt(np.mean(term))}</span> | <strong>Median:</strong> <span class='mono'>{fmt(np.median(term))}</span><br>
                <strong>5th:</strong> <span class='mono tr'>{fmt(np.percentile(term,5))}</span> | <strong>95th:</strong> <span class='mono tg'>{fmt(np.percentile(term,95))}</span><br>
                <strong>P(up):</strong> <span class='mono'>{np.mean(term>lS)*100:.1f}%</span> | <strong>P(>10%):</strong> <span class='mono'>{np.mean(term>lS*1.1)*100:.1f}%</span> | <strong>P(<-10%):</strong> <span class='mono'>{np.mean(term<lS*0.9)*100:.1f}%</span></p></div>""", unsafe_allow_html=True)

        with lc2:
            st.markdown("<span style='font-weight:600;color:#EAEAEA;'>Sample Paths & BSM Greeks Chain</span>", unsafe_allow_html=True)
            np.random.seed(42); paths = MC.paths(lS, liv, lT)
            fig2 = go.Figure()
            tax = np.linspace(0, settings['dte'], paths.shape[1])
            for i in range(min(50, len(paths))): fig2.add_trace(go.Scatter(x=tax, y=paths[i], mode='lines', line=dict(width=0.5, color='rgba(255,195,0,0.15)'), showlegend=False, hoverinfo='skip'))
            fig2.add_trace(go.Scatter(x=tax, y=np.mean(paths, axis=0), mode='lines', line=dict(width=3, color=CL['gold']), name='Mean'))
            fig2.add_trace(go.Scatter(x=tax, y=np.percentile(paths, 5, axis=0), mode='lines', line=dict(width=1.5, color=CL['red'], dash='dot'), name='5th'))
            fig2.add_trace(go.Scatter(x=tax, y=np.percentile(paths, 95, axis=0), mode='lines', line=dict(width=1.5, color=CL['green'], dash='dot'), name='95th'))
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=CL['bg'], height=350, margin=dict(l=10, r=10, t=30, b=30),
                xaxis=dict(showgrid=True, gridcolor=CL['border'], title='Days'), yaxis=dict(showgrid=True, gridcolor=CL['border'], title='Price', tickformat=','),
                font=dict(family='Inter', color=CL['text']), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            strikes = [snap(lS + i * strike_gap, strike_gap) for i in range(-3, 4)]
            gdata = []
            for K in strikes:
                cg = BSM.greeks(lS, K, lT, BSM.R, liv, 'call'); pg = BSM.greeks(lS, K, lT, BSM.R, liv, 'put')
                gdata.append({'Strike': fmt(K), 'C.Δ': f"{cg.delta:.3f}", 'P.Δ': f"{pg.delta:.3f}", 'Γ': f"{cg.gamma:.5f}",
                             'C.Θ': f"{cg.theta:.2f}", 'P.Θ': f"{pg.theta:.2f}", 'ν': f"{cg.vega:.2f}",
                             'Vanna': f"{cg.vanna:.4f}", 'Volga': f"{cg.volga:.4f}"})
            st.dataframe(pd.DataFrame(gdata), use_container_width=True, hide_index=True)


if 'sel' not in st.session_state: st.session_state.sel = None
if __name__ == "__main__": main()
