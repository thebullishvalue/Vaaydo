"""
╔══════════════════════════════════════════════════════════════════════════╗
║  VAAYDO (वायदो) — FnO Trade Intelligence                              ║
║  Quantitative Options Strategy Screener & Analytics Platform           ║
║  Version 4.0.0 | Hemrek Capital — Adaptive Intelligence Engine         ║
╚══════════════════════════════════════════════════════════════════════════╝

ENGINES (20):
  BSM Pricing (9 Greeks) · Multi-Estimator Vol (C2C/Park/GK/YZ) · GARCH(1,1)
  VRP Regime-Adaptive · MC 10K Antithetic · Kelly (Continuous, Capital-Normalized)
  6-State Vol Regime · 5-State Trend Regime · Kalman Filter · CUSUM Structural Break
  Higher-Order Greeks (Vanna/Volga/Charm/Speed) · Composite Risk Score
  Ensemble POP (BSM+MC Fusion) · Regime Stability · Expected Move Zones
  Sharpe Ranking · SPAN Margin · Unified Conviction (9-Factor)

STRATEGY UNIVERSE (14):
  ▲ BULLISH:  Bull Put Spread · Bull Call Spread · BWB · Ratio Spread
  ▼ BEARISH:  Bear Call Spread · Bear Put Spread
  ◆ NEUTRAL:  Iron Condor · Iron Butterfly · Short Strangle · Short Straddle
              Calendar Spread · Jade Lizard
  ⚡ VOLATILE: Long Straddle · Long Strangle

INTELLIGENCE LAYERS:
  IVP Gate — Credit/debit strategy selection by IV environment
  DTE Gate — Strategy viability per time-to-expiry
  Regime Alignment — Cross-references IV × Trend × Vol for each strategy
  Premium Quality — Credit/risk ratio scoring
  CUSUM Penalty — Structural break kills neutral strats, boosts volatile
  Lot-Adjusted Financials — ₹ P&L, ROM%, Θ/day in real capital terms
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
from adaptive_engine import AdaptiveEngine, STRATEGY_STRUCTURE, AdaptiveGating, FuzzyRegime, SignalSpace
from datetime import datetime, timedelta, date
import yfinance as yf
import requests
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="VAAYDO | FnO Intelligence", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")
VERSION = "4.0.0"

# v4.0: Global engine instance (initialized in main())
_engine = None

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
    .tc.bias-bull::before { background: #10b981; } .tc.bias-bear::before { background: #ef4444; }
    .tc.bias-neut::before { background: #FFC300; } .tc.bias-vol::before { background: #A78BFA; }
    .bias-tag { display:inline-block; font-size:0.6rem; font-weight:700; letter-spacing:0.5px; padding:0.15rem 0.5rem;
        border-radius:3px; text-transform:uppercase; }
    .bias-tag.bull { background:rgba(16,185,129,0.15); color:#10b981; }
    .bias-tag.bear { background:rgba(239,68,68,0.15); color:#ef4444; }
    .bias-tag.neut { background:rgba(255,195,0,0.15); color:#FFC300; }
    .bias-tag.vol { background:rgba(167,139,250,0.15); color:#A78BFA; }
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
    'AARTIIND':650,'ABB':125,'ABBOTINDIA':25,'ABCAPITAL':5400,'ABFRL':2900,
    'ACC':125,'ADANIENT':500,'ADANIGREEN':1800,'ADANIPORTS':625,'ADANIENSOL':400,
    'ALKEM':125,'AMBUJACEM':450,'ANGELONE':350,'APLAPOLLO':125,'APOLLOHOSP':125,
    'APOLLOTYRE':1850,'ASHOKLEY':3500,'ASTRAL':475,'ATUL':150,'AUBANK':500,
    'AUROPHARMA':325,'AXISBANK':600,'BAJAJ-AUTO':75,'BAJAJFINSV':500,'BAJFINANCE':125,
    'BALKRISIND':200,'BANDHANBNK':1800,'BANKBARODA':2700,'BANKINDIA':5400,'BATAINDIA':550,
    'BDL':200,'BEL':3300,'BERGEPAINT':550,'BHARATFORG':375,'BHARTIARTL':463,
    'BHEL':3500,'BIOCON':1500,'BOSCHLTD':25,'BPCL':1800,'BRITANNIA':100,
    'BSE':200,'BSOFT':900,'CAMS':800,'CANFINHOME':850,'CGPOWER':700,
    'CDSL':525,'CHAMBLFERT':950,'CHOLAFIN':625,'CIPLA':650,'COALINDIA':1400,
    'COFORGE':150,'COLPAL':350,'CONCOR':1400,'COROMANDEL':350,'CROMPTON':1000,
    'CUB':1200,'CUMMINSIND':200,'DABUR':1250,'DALBHARAT':100,'DEEPAKNTR':250,
    'DELHIVERY':1350,'DIVISLAB':125,'DIXON':75,'DLF':700,'DRREDDY':125,
    'EICHERMOT':10,'ESCORTS':200,'ETERNAL':1500,'EXIDEIND':1200,'FEDERALBNK':5000,
    'FORTIS':500,'GAIL':1225,'GLENMARK':350,'GMRAIRPORT':6000,'GNFC':500,
    'GODREJCP':500,'GODREJPROP':325,'GRANULES':1600,'GRASIM':275,'GUJGASLTD':500,
    'HAL':150,'HAVELLS':350,'HCLTECH':350,'HDFCAMC':150,'HDFCBANK':550,
    'HDFCLIFE':550,'HEROMOTOCO':150,'HINDALCO':1075,'HINDCOPPER':1075,'HINDPETRO':1350,
    'HINDUNILVR':300,'HUDCO':7000,'ICICIBANK':700,'ICICIGI':200,'ICICIPRULI':600,
    'IDEA':70000,'IDFCFIRSTB':10000,'IEX':3750,'IGL':550,'INDHOTEL':700,
    'INDIANB':850,'INDIGO':300,'INDUSINDBK':300,'INFY':400,'IOC':4850,
    'IPCALAB':350,'IRCTC':575,'IREDA':7500,'IRFC':10000,'ITC':1600,
    'JINDALSTEL':250,'JIOFIN':3500,'JKCEMENT':75,'JSWENERGY':600,'JSWSTEEL':600,
    'JUBLFOOD':1000,'KALYANKJIL':1000,'KAYNES':100,'KEI':75,'KFINTECH':400,
    'KOTAKBANK':400,'KPITTECH':650,'LALPATHLAB':250,'LAURUSLABS':1100,'LICHSGFIN':700,
    'LICI':700,'LODHA':500,'LT':150,'LTF':1500,'LTIM':150,
    'LTTS':275,'LUPIN':550,'M&M':350,'M&MFIN':2500,'MANAPPURAM':2000,
    'MARICO':800,'MARUTI':50,'MAXHEALTH':700,'MCX':300,'METROPOLIS':300,
    'MFSL':350,'MGL':375,'MOTHERSON':5000,'MPHASIS':175,'MRF':5,
    'MUTHOOTFIN':375,'NATIONALUM':1500,'NAUKRI':125,'NAVINFLUOR':175,'NBCC':5000,
    'NCC':850,'NESTLEIND':20,'NHPC':6250,'NMDC':3000,'NTPC':1350,
    'NUVAMA':300,'NYKAA':3200,'OBEROIRLTY':350,'OFSS':75,'OIL':850,
    'ONGC':3575,'PAGEIND':30,'PATANJALI':950,'PERSISTENT':75,'PETRONET':4000,
    'PFC':3000,'PGEL':1050,'PIIND':150,'PNBHOUSING':600,'PNB':7000,
    'POLICYBZR':600,'POLYCAB':200,'POWERGRID':2700,'POWERINDIA':25,'PPLPHARMA':3750,
    'PREMIERENE':600,'PRESTIGE':350,'PVRINOX':325,'RAMCOCEM':500,'RBLBANK':2000,
    'RECLTD':1500,'RELIANCE':250,'RVNL':1700,'SBICARD':500,'SBILIFE':750,
    'SBIN':750,'SHREECEM':25,'SHRIRAMFIN':625,'SIEMENS':75,'SJVN':3000,
    'SONACOMS':750,'SOLARINDS':25,'SRF':125,'SUNPHARMA':700,'SUNTV':500,
    'SUPREMEIND':50,'SUZLON':10000,'SYNGENE':1000,'TATACHEM':500,'TATACOMM':250,
    'TATACONSUM':450,'TATAELXSI':125,'TATAMOTORS':575,'TATAPOWER':2700,'TATASTEEL':850,
    'TATATECH':1200,'TCS':175,'TECHM':300,'TIINDIA':175,'TITAN':375,
    'TMPV':1250,'TORNTPHARM':250,'TORNTPOWER':500,'TRENT':325,'TVSMOTOR':175,
    'UBL':350,'ULTRACEMCO':50,'UNIONBANK':4000,'UNITDSPR':500,'UPL':1300,
    'VBL':550,'VEDL':1550,'VOLTAS':700,'WAAREEENER':100,'WIPRO':1000,
    'YESBANK':20000,'ZOMATO':1000,'ZYDUSLIFE':550,
    'AMBER':50,'ANGELONE':350,'BLUESTARCO':100,'DMART':125,'HDFCAMC':150,
    'INDHOTEL':700,'INOXWIND':3000,'JINDALSTEL':250,'LICI':700,'MANKIND':200,
    'MAZDOCK':200,'PHOENIXLTD':300,'SHREECEM':25,'SHREECEM':25,'TRENT':325,
    'UNOMINDA':400,'360ONE':300,'ADANIENSOL':400,'ASHOKLEY':3500,'GMRAIRPORT':6000,
    'HINDZINC':850,'MARICO':800,'SWIGGY':2500,'INDUSTOWER':2200,'CANBK':2700,'ASIANPAINT':200,'PIDILITIND':375,'SAIL':2600,'PAYTM':1250}

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
    conviction_std: float = 0.0; conviction_ci_lower: float = 0.0; conviction_ci_upper: float = 0.0
    viability: float = 0.0; model_agreement: float = 0.0; pop_std: float = 0.0


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
        if any(np.isnan(x) for x in [S, K, T, σ] if isinstance(x, (int, float))): return 0.0
        if T <= 0: return max(S - K, 0)
        if σ <= 0 or K <= 0 or S <= 0: return max(S - K, 0)
        d1, d2 = cls._d(S, K, T, r, σ)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @classmethod
    def put(cls, S, K, T, r, σ):
        if any(np.isnan(x) for x in [S, K, T, σ] if isinstance(x, (int, float))): return 0.0
        if T <= 0: return max(K - S, 0)
        if σ <= 0 or K <= 0 or S <= 0: return max(K - S, 0)
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
        if T <= 0 or σ <= 0 or np.isnan(S) or np.isnan(σ) or np.isnan(T): return np.full(2*n, max(S, 0) if not np.isnan(S) else 100)
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
    def analyze(S, σ, T, legs, n=5000, sim_vol=None):
        """Full MC analysis: POP, EV, Std for any strategy
        σ = pricing vol (IV) used for premiums
        sim_vol = simulation vol (RV/GARCH) — if IV > sim_vol, selling options has +EV edge
        """
        vol_for_sim = sim_vol if sim_vol is not None else σ
        terminal = MC.terminal_prices(S, vol_for_sim, T, n)
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

def kelly(p, w, l, confidence=1.0, mc_ev=None, mc_std=None):
    """Kelly: continuous f* = (EV/capital) / (σ/capital)² → normalized by max_loss."""
    if mc_ev is not None and mc_std is not None and mc_std > 0.01:
        capital = max(abs(l), abs(w), 1)
        ev_n = mc_ev / capital
        std_n = mc_std / capital
        if std_n > 0.001:
            k = ev_n / (std_n ** 2)
        else:
            k = 0
        return max(0, min(k * 0.5 * max(0.5, min(confidence, 1.0)), 0.25))
    if l <= 0 or w <= 0: return 0.0
    b = w / abs(l)
    k = (b * p - (1 - p)) / b
    return max(0, min(k * 0.5 * max(0.5, min(confidence, 1.0)), 0.25))

def auto_gap(price):
    """NSE actual strike intervals (2024-25 contract specs)"""
    if price <= 50: return 0.5
    if price <= 250: return 2.5
    if price <= 500: return 5
    if price <= 1000: return 10
    if price <= 2500: return 25
    if price <= 5000: return 50
    if price <= 10000: return 100
    if price <= 25000: return 500
    return 500

def snap(x, g):
    return round(x / g) * g

def clamp_sharpe(ev, std, max_loss=None):
    """Sharpe ratio with optional capital normalization for cross-strategy comparison."""
    if std < 0.01 or np.isnan(ev) or np.isnan(std): return 0.0
    # If max_loss provided, normalize to get return-based Sharpe
    if max_loss and max_loss > 0:
        ev_n = ev / max_loss
        std_n = std / max_loss
        if std_n < 0.001: return 0.0
        result = ev_n / std_n
    else:
        result = ev / std
    return max(-5.0, min(5.0, result)) if not np.isnan(result) else 0.0

def clamp_rr(nc, ml):
    """FIX #6: Clamp risk-reward to [0, 50]"""
    if ml <= 0: return 0.0
    return max(0.0, min(50.0, nc / ml))

def span_margin(S, iv, lot_size=1, dte=30):
    """SPAN-like margin: scales with DTE and IV. Lower for near-expiry."""
    # SPAN ≈ max(premium_margin, movement_margin)
    # Movement scales with sqrt(DTE) not sqrt(30)
    t_factor = max(dte, 1) / 365
    sigma_move = S * iv * np.sqrt(t_factor) * 2.5  # 2.5σ move over DTE
    base_margin = S * 0.12  # minimum 12% of spot (NSE floor)
    return max(base_margin, sigma_move) * lot_size

def ensemble_pop(pop_bsm, pop_mc):
    """§9.2: Inverse-variance weighted fusion"""
    w_bsm = 1.0 / (0.08**2)  # BSM RMSE ~8%
    w_mc = 1.0 / (0.05**2)   # MC RMSE ~5%
    result = (w_bsm * pop_bsm + w_mc * pop_mc) / (w_bsm + w_mc)
    return result if not np.isnan(result) else 0.5

def conviction_unified(ra, pop, ev_ratio, sharpe, stability, iv_norm,
                       prem_quality=1.0, dte_fitness=1.0, cusum_penalty=1.0):
    ev_n = max(0, min(1, (ev_ratio + 1) / 2))
    sh_n = max(0, min(1, (sharpe + 2) / 5))
    pq_n = max(0, min(1, prem_quality))
    df_n = max(0, min(1, dte_fitness))
    raw = (0.20 * ra + 0.22 * pop + 0.12 * ev_n + 0.15 * sh_n +
           0.08 * stability + 0.08 * iv_norm + 0.08 * pq_n + 0.07 * df_n)
    result = raw * 100 * cusum_penalty
    return max(0, min(100, result)) if not np.isnan(result) else 30.0




# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY BIAS CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyBias:
    BULLISH = 'BULLISH'
    BEARISH = 'BEARISH'
    NEUTRAL = 'NEUTRAL'
    VOLATILE = 'VOLATILE'

STRATEGY_BIAS = {
    # Bullish — profit when market rises
    'Bull Put Spread': StrategyBias.BULLISH,
    'Bull Call Spread': StrategyBias.BULLISH,
    # Bearish — profit when market falls
    'Bear Call Spread': StrategyBias.BEARISH,
    'Bear Put Spread': StrategyBias.BEARISH,
    # Neutral / Range-bound — profit when market stays still
    'Short Iron Condor': StrategyBias.NEUTRAL,
    'Long Iron Condor': StrategyBias.NEUTRAL,  # still range-bound (debit)
    'Short Iron Butterfly': StrategyBias.NEUTRAL,
    'Long Iron Butterfly': StrategyBias.NEUTRAL,
    'Short Strangle': StrategyBias.NEUTRAL,
    'Short Straddle': StrategyBias.NEUTRAL,
    'Calendar Spread': StrategyBias.NEUTRAL,
    'Jade Lizard': StrategyBias.NEUTRAL,
    # Volatile / Breakout — profit when market moves big
    'Long Straddle': StrategyBias.VOLATILE,
    'Long Strangle': StrategyBias.VOLATILE,
    # Directional complex
    'Broken Wing Butterfly': StrategyBias.BULLISH,
    'Ratio Spread': StrategyBias.BULLISH,
}

BIAS_COLOR = {
    StrategyBias.BULLISH: '#10b981',   # green
    StrategyBias.BEARISH: '#ef4444',   # red
    StrategyBias.NEUTRAL: '#FFC300',   # gold
    StrategyBias.VOLATILE: '#A78BFA',  # purple
}

BIAS_LABEL = {
    StrategyBias.BULLISH: '▲ BULLISH',
    StrategyBias.BEARISH: '▼ BEARISH',
    StrategyBias.NEUTRAL: '◆ NEUTRAL',
    StrategyBias.VOLATILE: '⚡ VOLATILE',
}

# Strategy credit/debit classification
STRATEGY_TYPE = {
    # Credit: receive premium upfront, theta positive, profit from IV crush
    'Short Strangle': 'CREDIT', 'Short Straddle': 'CREDIT',
    'Iron Condor': 'CREDIT', 'Iron Butterfly': 'CREDIT',
    'Bull Put Spread': 'CREDIT', 'Bear Call Spread': 'CREDIT',
    'Jade Lizard': 'CREDIT',
    # Debit: pay premium upfront, theta negative, profit from IV expansion or move
    'Bull Call Spread': 'DEBIT', 'Bear Put Spread': 'DEBIT',
    'Long Straddle': 'DEBIT', 'Long Strangle': 'DEBIT',
    'Calendar Spread': 'DEBIT',
    # Hybrid: can be credit or debit depending on construction
    'Broken Wing Butterfly': 'HYBRID', 'Ratio Spread': 'HYBRID',
}

# IVP gating: skip strategies that make NO sense in current IV environment
# Returns (should_evaluate, iv_multiplier)
# DTE fitness: each strategy has an optimal DTE range
DTE_RANGE = {
    'Short Strangle':  (1, 60),   'Short Straddle':  (1, 45),
    'Iron Condor':     (2, 60),   'Iron Butterfly':  (2, 50),
    'Bull Put Spread': (1, 60),   'Bear Call Spread': (1, 60),
    'Bull Call Spread': (2, 60),  'Bear Put Spread': (2, 60),
    'Long Straddle':   (3, 90),   'Long Strangle':   (3, 90),
    'Calendar Spread': (14, 90),  'Jade Lizard':     (2, 60),
    'Broken Wing Butterfly': (5, 60), 'Ratio Spread': (3, 60),
}

def dte_gate(strategy_name, dte):
    lo, hi = DTE_RANGE.get(strategy_name, (3, 90))
    if dte < lo or dte > hi: return False, 0.0
    mid = (lo + hi) / 2
    span = (hi - lo) / 2
    dist = abs(dte - mid) / max(span, 1)
    fitness = max(0.5, 1.0 - 0.5 * dist * dist)
    return True, fitness

def ivp_gate(strategy_name, ivp):
    stype = STRATEGY_TYPE.get(strategy_name, 'HYBRID')
    if stype == 'CREDIT':
        # Credit strategies: need IV > 25 to be worth selling
        # Below 25: skip entirely. 25-40: evaluate but penalize. 40+: good.
        if ivp < 20: return False, 0.0
        if ivp < 35: return True, 0.5   # weak environment for selling
        if ivp < 55: return True, 0.8   # decent
        return True, 1.0                 # rich premium, ideal
    elif stype == 'DEBIT':
        # Debit strategies: need IV < 75 to avoid overpaying
        # Above 80: skip (too expensive). 60-80: penalize. Below 60: good.
        if ivp > 85: return False, 0.0
        if ivp > 70: return True, 0.5   # expensive options
        if ivp > 50: return True, 0.8   # moderate
        return True, 1.0                 # cheap options, ideal
    else:  # HYBRID
        return True, 0.9                 # always evaluate

def compute_regime_alignment(strategy_name, vr, tr, ivp):
    """Intelligent regime alignment: cross-references IV regime, trend, and strategy type.
    
    Core logic:
    - Credit strategies thrive in HIGH IV + NEUTRAL trend (sell expensive, let theta decay)
    - Debit directional strategies thrive in LOW IV + TRENDING (buy cheap, ride the move)
    - Debit volatile strategies thrive in LOW IV + ANY trend (buy cheap, wait for explosion)
    - Trend-bias must match: bullish strategies need UP trend, bearish need DOWN
    """
    stype = STRATEGY_TYPE.get(strategy_name, 'HYBRID')
    bias = STRATEGY_BIAS.get(strategy_name, StrategyBias.NEUTRAL)
    
    # Base score from IV environment
    iv_score = 0.5
    if stype == 'CREDIT':
        # Higher IV = better for credit (more premium collected, more room for IV crush)
        if ivp >= 70: iv_score = 0.95
        elif ivp >= 55: iv_score = 0.80
        elif ivp >= 40: iv_score = 0.60
        elif ivp >= 25: iv_score = 0.35
        else: iv_score = 0.15
    elif stype == 'DEBIT':
        # Lower IV = better for debit (cheaper options, room for IV expansion)
        if ivp <= 25: iv_score = 0.90
        elif ivp <= 40: iv_score = 0.80
        elif ivp <= 55: iv_score = 0.65
        elif ivp <= 70: iv_score = 0.40
        else: iv_score = 0.20
    else:  # HYBRID
        iv_score = 0.60  # moderate regardless
    
    # Trend alignment score
    trend_score = 0.5
    if bias == StrategyBias.BULLISH:
        if 'STRONG_UP' in tr.value: trend_score = 0.95
        elif 'UP' in tr.value: trend_score = 0.80
        elif tr == TrendRegime.NEUTRAL: trend_score = 0.50
        elif 'DOWN' in tr.value: trend_score = 0.15
    elif bias == StrategyBias.BEARISH:
        if 'STRONG_DOWN' in tr.value: trend_score = 0.95
        elif 'DOWN' in tr.value: trend_score = 0.80
        elif tr == TrendRegime.NEUTRAL: trend_score = 0.50
        elif 'UP' in tr.value: trend_score = 0.15
    elif bias == StrategyBias.NEUTRAL:
        # Neutral strategies love ranging markets
        if tr == TrendRegime.NEUTRAL: trend_score = 0.90
        elif 'STRONG' in tr.value: trend_score = 0.20
        else: trend_score = 0.55
    elif bias == StrategyBias.VOLATILE:
        # Volatile strategies love strong trends or expected breakouts
        if 'STRONG' in tr.value: trend_score = 0.85
        elif tr == TrendRegime.NEUTRAL: trend_score = 0.50  # could break either way
        else: trend_score = 0.60
    
    # Vol regime bonus/penalty
    vol_bonus = 0.0
    if stype == 'CREDIT':
        # Credit loves high vol regime (mean reversion expected)
        if vr in (VolRegime.HIGH, VolRegime.EXTREME): vol_bonus = 0.10
        elif vr == VolRegime.ELEVATED: vol_bonus = 0.05
        elif vr in (VolRegime.LOW, VolRegime.COMPRESSED): vol_bonus = -0.10
    elif stype == 'DEBIT':
        # Debit loves low vol regime (expansion expected)
        if vr in (VolRegime.LOW, VolRegime.COMPRESSED): vol_bonus = 0.10
        elif vr == VolRegime.NORMAL: vol_bonus = 0.05
        elif vr in (VolRegime.HIGH, VolRegime.EXTREME): vol_bonus = -0.10
    
    # Composite: 50% IV weight, 40% trend weight, 10% vol bonus
    ra = 0.50 * iv_score + 0.40 * trend_score + 0.10 * (0.5 + vol_bonus * 5)
    return max(0.05, min(0.95, ra))

def get_bias(strategy_name):
    return STRATEGY_BIAS.get(strategy_name, StrategyBias.NEUTRAL)

def get_bias_color(strategy_name):
    return BIAS_COLOR.get(get_bias(strategy_name), '#FFC300')

def get_bias_label(strategy_name):
    return BIAS_LABEL.get(get_bias(strategy_name), '◆ NEUTRAL')

def get_strategy_type(strategy_name):
    # Handle dynamic Short/Long prefix
    for prefix in ['Short ', 'Long ']:
        base = strategy_name.replace(prefix, '')
        if base in STRATEGY_TYPE:
            return STRATEGY_TYPE[base]
    return STRATEGY_TYPE.get(strategy_name, 'HYBRID')

def get_type_tag(strategy_name):
    t = get_strategy_type(strategy_name)
    if t == 'CREDIT': return 'neut', '₹ CREDIT'
    elif t == 'DEBIT': return 'vol', '₹ DEBIT'
    return 'warn', '₹ HYBRID'


# ═══════════════════════════════════════════════════════════════════════════════
# L6: STRATEGY ENGINE — All 10 with real MC, full Greeks, proper payoffs
# ═══════════════════════════════════════════════════════════════════════════════

ALL_STRATS = ['Short Strangle','Short Straddle','Iron Condor','Iron Butterfly',
              'Bull Put Spread','Bear Call Spread','Bull Call Spread','Bear Put Spread',
              'Long Straddle','Long Strangle',
              'Calendar Spread','Jade Lizard','Broken Wing Butterfly','Ratio Spread']

# MIN_PREMIUM replaced by adaptive: engine.min_premium(price, dte)
# Kept as fallback for when engine not available
MIN_PREMIUM = 0.50

def compute_full_greeks(S, T, r, iv, legs_spec):
    """Compute net Greeks for any multi-leg strategy using BSM"""
    net = Greeks()
    for lspec in legs_spec:
        K, otype, qty = lspec['strike'], lspec['otype'], lspec['qty']
        g = BSM.greeks(S, K, T, r, iv, otype)
        net = net + g.scale(qty)
    return net


def _compute_quality(net_credit, max_loss, max_profit, name, cusum):
    stype = STRATEGY_TYPE.get(name, 'HYBRID')
    bias = STRATEGY_BIAS.get(name, StrategyBias.NEUTRAL)
    ratio = abs(net_credit) / max(abs(max_loss), 1) if stype == 'CREDIT' else abs(max_profit) / max(abs(max_loss), 1)
    # Strategy-type aware thresholds (what's "good" credit/risk varies by structure)
    if name in ('Short Strangle', 'Short Straddle'):
        _pq = min(1.0, ratio / 0.06)     # naked: 6% credit/margin = excellent
    elif name in ('Iron Condor', 'Iron Butterfly', 'Jade Lizard'):
        _pq = min(1.0, ratio / 0.20)     # 4-leg: 20% credit/risk = excellent
    elif stype == 'CREDIT':
        _pq = min(1.0, ratio / 0.25)     # spreads: 25% = excellent
    elif stype == 'DEBIT':
        _pq = min(1.0, ratio / 1.5)      # debit: 150% profit/risk = excellent
    else:
        _pq = min(1.0, ratio / 0.30)
    if cusum:
        if bias == StrategyBias.NEUTRAL: _cp = 0.60
        elif bias == StrategyBias.VOLATILE: _cp = 1.05
        else: _cp = 0.85
    else:
        _cp = 1.0
    return max(0.0, min(1.0, _pq)), max(0.5, min(1.1, _cp))


# ── v4.0 Adaptive scoring helper ──
_engine = None  # Set during main() after calibration

def _adaptive_tail(name, legs, mp, ml, pop_b, pop_m, ev, std, ng, iv, be_lower, be_upper,
                   nc=0, width=0, settings=None, regime=None):
    """Unified scoring tail using AdaptiveEngine — replaces ensemble_pop + conviction_unified + kelly."""
    global _engine
    dte_val = settings.get('dte', 5) if settings else 5
    if _engine and regime:
        pop_mean, pop_std, agreement = _engine.fuse_pop(pop_b, pop_m)
        viability = _engine.compute_viability(name, regime, dte_val)
        sh = clamp_sharpe(ev, std, abs(ml) if ml else None)
        ev_denom = max(abs(nc) if nc else abs(mp), 0.01)
        ev_ratio = ev / ev_denom
        cd = _engine.score(pop_mean, pop_std, ev_ratio, sh, viability, regime, agreement)
        kf = _engine.kelly(pop_mean, pop_std, abs(mp), abs(ml), ev, std)
        rs = BSM.risk_score(ng, iv, 1.0)
        return StrategyResult(name=name, legs=legs, max_profit=mp, max_loss=ml,
            breakeven_lower=be_lower, breakeven_upper=be_upper,
            pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_mean, expected_value=ev,
            sharpe_ratio=sh, kelly_fraction=kf, net_greeks=ng,
            conviction_score=cd.mean, optimal_dte=dte_val, risk_score=rs,
            stability_score=regime.stability, width=width, net_credit=nc,
            risk_reward=clamp_rr(nc, ml), regime_alignment=viability,
            conviction_std=cd.std, conviction_ci_lower=cd.ci_lower,
            conviction_ci_upper=cd.ci_upper, viability=viability,
            model_agreement=agreement, pop_std=pop_std)
    else:
        # Fallback to old pipeline if engine not initialized
        pop_e = ensemble_pop(pop_b, pop_m)
        sh = clamp_sharpe(ev, std, abs(ml) if ml else None)
        ev_denom = max(abs(nc) if nc else abs(mp), 0.01)
        ev_ratio = ev / ev_denom
        kf = kelly(pop_e, abs(mp), abs(ml))
        rs = BSM.risk_score(ng, iv, 1.0) if ng else 0
        cs = conviction_unified(0.5, pop_e, ev_ratio, sh, 0.5, 0.5, 0.5, 0.5)
        return StrategyResult(name=name, legs=legs, max_profit=mp, max_loss=ml,
            breakeven_lower=be_lower, breakeven_upper=be_upper,
            pop_bsm=pop_b, pop_mc=pop_m, pop_ensemble=pop_e, expected_value=ev,
            sharpe_ratio=sh, kelly_fraction=kf, net_greeks=ng,
            conviction_score=cs, optimal_dte=settings.get('dte',5) if settings else 5,
            risk_score=rs, stability_score=0.5, width=width, net_credit=nc,
            risk_reward=clamp_rr(nc, ml), regime_alignment=0.5)

def score_strategy(name, stock, settings, iv_mult=1.0, regime=None):
    S = stock['price']; iv = stock['ATMIV'] / 100; ivp = stock['IVPercentile']
    # NaN guard
    if any(np.isnan(v) for v in [S, iv, ivp] if isinstance(v, (int, float))):
        return None
    if S <= 0 or iv <= 0: return None
    T = settings['dte'] / 365; r = BSM.R; g = auto_gap(S)
    rsi = stock.get('rsi_daily', 50); adx = stock.get('adx', 20)
    kalman_t = stock.get('kalman_trend', 0)
    ivp = stock.get('IVPercentile', 50)
    dte_val = settings['dte']
    # v4.0: DTE gating now handled by adaptive viability in main loop
    _cusum = stock.get('CUSUM_Alert', False)
    garch_p = stock.get('GARCH_Persistence', 0.95)
    cusum = stock.get('CUSUM_Alert', False)
    lot = stock.get('lot_size', 1)
    # VRP: simulate with realized vol, not IV — captures option selling edge
    rv = stock.get('RV_Composite', iv * 100)
    garch_v = stock.get('GARCH_Vol', iv * 100)
    sim_vol = min(iv, max(rv, garch_v) / 100 * 1.05)  # sim_vol ≤ IV, ≥ RV
    # v4.0: regime passed in as parameter, old detection kept for Deep Analysis tab
    vr = detect_vol_regime(ivp)
    tr = detect_trend(S, stock.get('ma20_daily', S), stock.get('ma50_daily', S), rsi,
                      stock.get('% change', 0), adx, kalman_t)
    stab = regime_stability(ivp, rsi, garch_p, cusum)
    rvw = regime_vol_weight(vr)
    em = S * iv * np.sqrt(max(T, 1e-6))
    iv_norm = min(ivp / 100, 1.0)
    min_wing = max(g, 1)  # FIX #11: minimum 1 gap width
    _min_prem = _engine.min_premium(S, settings['dte']) if _engine else MIN_PREMIUM

    # v4.0: Adaptive strike placement via delta targeting
    def _n_gaps(fallback_mult=0.5):
        """Get number of gaps OTM from engine or EM fallback."""
        if _engine and regime:
            td = _engine.target_delta(name, regime)
            return max(1, _engine.strike_gaps(td, S, T, iv, g))
        return max(1, round(em * fallback_mult / g)) if g > 0 else 1

    try:
        if name == 'Short Strangle':
            # Place shorts at ~0.7 EM OTM (balance premium vs safety)
            n_otm = _n_gaps(0.7)  # v4.0: adaptive delta-based placement
            # For short DTE with wide gaps, ensure at least 1 gap OTM
            n_otm = max(n_otm, 1)
            cK = snap(S + n_otm * g, g)
            pK = snap(S - n_otm * g, g)
            # Validate delta-based: if premium too low, move closer
            for _ in range(3):
                cp_test = BSM.call(S,cK,T,r,iv) + BSM.put(S,pK,T,r,iv)
                if cp_test >= _min_prem: break
                cK = max(cK - g, snap(S + g, g))
                pK = min(pK + g, snap(S - g, g))
            if cK <= S: cK = snap(S + g, g)
            if pK >= S: pK = snap(S - g, g)
            cp = BSM.call(S,cK,T,r,iv); pp = BSM.put(S,pK,T,r,iv)
            nc = cp + pp
            if nc < _min_prem: return None
            ml = span_margin(S, iv, dte=settings['dte'])  # SPAN scales with DTE
            legs = [{'type':'Sell Call','strike':cK,'premium':cp,'qty':1},
                    {'type':'Sell Put','strike':pK,'premium':pp,'qty':1}]
            pop_b = max(0, BSM.prob_otm(S,cK,T,iv,'call') + BSM.prob_otm(S,pK,T,iv,'put') - 1)
            pop_m, ev, std = MC.analyze(S, iv, T, legs, sim_vol=sim_vol)
            return _adaptive_tail(name, legs, nc, ml, pop_b, pop_m, ev, std, ng, iv,
                pK-nc, cK+nc, nc=nc, width=cK-pK, settings=settings, regime=regime)

        elif name == 'Short Straddle':
            K = snap(S, g)
            cp = BSM.call(S,K,T,r,iv); pp = BSM.put(S,K,T,r,iv)
            nc = cp + pp
            if nc < _min_prem: return None
            ml = span_margin(S, iv, dte=settings['dte'])
            legs = [{'type':'Sell Call','strike':K,'premium':cp,'qty':1},
                    {'type':'Sell Put','strike':K,'premium':pp,'qty':1}]
            pop_b = max(0, BSM.prob_otm(S,K+nc,T,iv,'call') + BSM.prob_otm(S,K-nc,T,iv,'put') - 1)
            pop_m, ev, std = MC.analyze(S, iv, T, legs, sim_vol=sim_vol)
            return _adaptive_tail(name, legs, nc, ml, pop_b, pop_m, ev, std, ng, iv,
                K-nc, K+nc, nc=nc, width=0, settings=settings, regime=regime)

        elif name == 'Iron Condor':
            # Short strikes: ~0.6 EM OTM (tighter than strangle, wider than BPS)
            n_short = _n_gaps(0.6)  # v4.0: adaptive
            sc = snap(S + n_short * g, g)
            sp = snap(S - n_short * g, g)
            # Wings: 2 gaps beyond shorts
            lc = sc + max(2 * g, min_wing)
            lp = sp - max(2 * g, min_wing)
            # Tighten if premium too low
            for _ in range(3):
                nc_test = (BSM.call(S,sc,T,r,iv)-BSM.call(S,lc,T,r,iv)) + (BSM.put(S,sp,T,r,iv)-BSM.put(S,lp,T,r,iv))
                if nc_test >= _min_prem: break
                sc = max(sc - g, snap(S + g, g))
                sp = min(sp + g, snap(S - g, g))
                lc = sc + max(2 * g, min_wing)
                lp = sp - max(2 * g, min_wing)
            nc = (BSM.call(S,sc,T,r,iv)-BSM.call(S,lc,T,r,iv)) + (BSM.put(S,sp,T,r,iv)-BSM.put(S,lp,T,r,iv))
            if nc < _min_prem: return None
            w_spread = lc - sc; ml = max(w_spread - nc, 1)
            legs = [{'type':'Sell Call','strike':sc,'premium':BSM.call(S,sc,T,r,iv),'qty':1},
                    {'type':'Buy Call','strike':lc,'premium':BSM.call(S,lc,T,r,iv),'qty':1},
                    {'type':'Sell Put','strike':sp,'premium':BSM.put(S,sp,T,r,iv),'qty':1},
                    {'type':'Buy Put','strike':lp,'premium':BSM.put(S,lp,T,r,iv),'qty':1}]
            pop_b = max(0, BSM.prob_otm(S,sc,T,iv,'call') + BSM.prob_otm(S,sp,T,iv,'put') - 1)
            pop_m, ev, std = MC.analyze(S, iv, T, legs, sim_vol=sim_vol)
            return _adaptive_tail(name, legs, nc, ml, pop_b, pop_m, ev, std, ng, iv,
                sp-nc, sc+nc, nc=nc, width=w_spread, settings=settings, regime=regime)

        elif name == 'Iron Butterfly':
            K = snap(S, g)
            # Wings: ~0.8 EM from ATM (wider = more credit captured)
            n_wing = max(2, _n_gaps(0.8))  # v4.0: adaptive
            ww = max(n_wing * g, min_wing)
            nc = (BSM.call(S,K,T,r,iv)+BSM.put(S,K,T,r,iv)) - (BSM.call(S,K+ww,T,r,iv)+BSM.put(S,K-ww,T,r,iv))
            if nc < _min_prem: return None
            ml = max(ww - nc, 1)
            legs = [{'type':'Sell Call','strike':K,'premium':BSM.call(S,K,T,r,iv),'qty':1},
                    {'type':'Sell Put','strike':K,'premium':BSM.put(S,K,T,r,iv),'qty':1},
                    {'type':'Buy Call','strike':K+ww,'premium':BSM.call(S,K+ww,T,r,iv),'qty':1},
                    {'type':'Buy Put','strike':K-ww,'premium':BSM.put(S,K-ww,T,r,iv),'qty':1}]
            pop_m, ev, std = MC.analyze(S, iv, T, legs, sim_vol=sim_vol)
            pop_b = max(0, BSM.prob_otm(S,K+nc,T,iv,'call') + BSM.prob_otm(S,K-nc,T,iv,'put') - 1)
            return _adaptive_tail(name, legs, nc, ml, pop_b, pop_m, ev, std, ng, iv,
                K-nc, K+nc, nc=nc, width=ww, settings=settings, regime=regime)

        elif name == 'Bull Put Spread':
            # Short put: 1 gap OTM, long put: 2 more gaps below
            n_short = _n_gaps(0.5)  # v4.0: adaptive
            sp_k = snap(S - n_short * g, g)
            lp_k = sp_k - max(2 * g, min_wing)
            # Tighten if premium too low
            for _ in range(3):
                nc_test = BSM.put(S,sp_k,T,r,iv) - BSM.put(S,lp_k,T,r,iv)
                if nc_test >= _min_prem: break
                sp_k = min(sp_k + g, snap(S - g, g))
                lp_k = sp_k - max(2 * g, min_wing)
            nc = BSM.put(S,sp_k,T,r,iv) - BSM.put(S,lp_k,T,r,iv)
            if nc < _min_prem: return None
            ml = max((sp_k - lp_k) - nc, 1)
            legs = [{'type':'Sell Put','strike':sp_k,'premium':BSM.put(S,sp_k,T,r,iv),'qty':1},
                    {'type':'Buy Put','strike':lp_k,'premium':BSM.put(S,lp_k,T,r,iv),'qty':1}]
            pop_b = BSM.prob_otm(S, sp_k, T, iv, 'put')
            pop_m, ev, std = MC.analyze(S, iv, T, legs, sim_vol=sim_vol)
            return _adaptive_tail(name, legs, nc, ml, pop_b, pop_m, ev, std, ng, iv,
                sp_k-nc, S*10, nc=nc, width=sp_k-lp_k, settings=settings, regime=regime)

        elif name == 'Bear Call Spread':
            n_short = _n_gaps(0.5)  # v4.0: adaptive
            sc_k = snap(S + n_short * g, g)
            lc_k = sc_k + max(2 * g, min_wing)
            # Tighten if premium too low
            for _ in range(3):
                nc_test = BSM.call(S,sc_k,T,r,iv) - BSM.call(S,lc_k,T,r,iv)
                if nc_test >= _min_prem: break
                sc_k = max(sc_k - g, snap(S + g, g))
                lc_k = sc_k + max(2 * g, min_wing)
            nc = BSM.call(S,sc_k,T,r,iv) - BSM.call(S,lc_k,T,r,iv)
            if nc < _min_prem: return None
            ml = max((lc_k - sc_k) - nc, 1)
            legs = [{'type':'Sell Call','strike':sc_k,'premium':BSM.call(S,sc_k,T,r,iv),'qty':1},
                    {'type':'Buy Call','strike':lc_k,'premium':BSM.call(S,lc_k,T,r,iv),'qty':1}]
            pop_b = BSM.prob_otm(S, sc_k, T, iv, 'call')
            pop_m, ev, std = MC.analyze(S, iv, T, legs, sim_vol=sim_vol)
            return _adaptive_tail(name, legs, nc, ml, pop_b, pop_m, ev, std, ng, iv,
                0, sc_k+nc, nc=nc, width=lc_k-sc_k, settings=settings, regime=regime)

        elif name == 'Bull Call Spread':
            # Debit spread: buy lower call, sell higher call — bullish
            n_long = _n_gaps(0.3)  # v4.0: adaptive
            lc_k = snap(S - n_long * g, g)  # ITM or ATM long call
            if lc_k >= S: lc_k = snap(S, g)
            sc_k = lc_k + max(2 * g, min_wing)  # OTM short call
            lc_p = BSM.call(S,lc_k,T,r,iv); sc_p = BSM.call(S,sc_k,T,r,iv)
            nd = lc_p - sc_p  # net debit
            nc = -nd  # for quality computation
            if nd <= 0: return None
            mp = (sc_k - lc_k) - nd  # max profit = width - debit
            ml = nd  # max loss = debit paid
            if mp < _min_prem: return None
            legs = [{'type':'Buy Call','strike':lc_k,'premium':lc_p,'qty':1},
                    {'type':'Sell Call','strike':sc_k,'premium':sc_p,'qty':1}]
            pop_b = BSM.prob_otm(S, lc_k, T, iv, 'put')  # prob stock > long strike
            pop_m, ev, std = MC.analyze(S, iv, T, [
                {'type':'Buy Call','strike':lc_k,'premium':lc_p,'qty':1},
                {'type':'Sell Call','strike':sc_k,'premium':sc_p,'qty':1}], sim_vol=sim_vol)
            return _adaptive_tail(name, legs, mp, ml, pop_b, pop_m, ev, std, ng, iv,
                lc_k+nd, sc_k, nc=0, width=sc_k-lc_k, settings=settings, regime=regime)

        elif name == 'Bear Put Spread':
            # Debit spread: buy higher put, sell lower put — bearish
            n_long = _n_gaps(0.3)  # v4.0: adaptive
            lp_k = snap(S + n_long * g, g)  # ITM or ATM long put
            if lp_k <= S: lp_k = snap(S, g)
            sp_k = lp_k - max(2 * g, min_wing)  # OTM short put
            lp_p = BSM.put(S,lp_k,T,r,iv); sp_p = BSM.put(S,sp_k,T,r,iv)
            nd = lp_p - sp_p  # net debit
            nc = -nd
            if nd <= 0: return None
            mp = (lp_k - sp_k) - nd
            ml = nd
            if mp < _min_prem: return None
            legs = [{'type':'Buy Put','strike':lp_k,'premium':lp_p,'qty':1},
                    {'type':'Sell Put','strike':sp_k,'premium':sp_p,'qty':1}]
            pop_b = BSM.prob_otm(S, lp_k, T, iv, 'call')  # prob stock < long strike
            pop_m, ev, std = MC.analyze(S, iv, T, [
                {'type':'Buy Put','strike':lp_k,'premium':lp_p,'qty':1},
                {'type':'Sell Put','strike':sp_k,'premium':sp_p,'qty':1}], sim_vol=sim_vol)
            return _adaptive_tail(name, legs, mp, ml, pop_b, pop_m, ev, std, ng, iv,
                sp_k, lp_k-nd, nc=0, width=lp_k-sp_k, settings=settings, regime=regime)

        elif name == 'Long Straddle':
            # Debit: buy ATM call + put — profit on big move either direction
            K = snap(S, g)
            cp = BSM.call(S,K,T,r,iv); pp = BSM.put(S,K,T,r,iv)
            nd = cp + pp  # total debit
            nc = -nd
            if nd <= 0: return None
            ml = nd  # max loss = debit paid
            mp = S * 0.5  # theoretical unlimited (cap for display)
            legs = [{'type':'Buy Call','strike':K,'premium':cp,'qty':1},
                    {'type':'Buy Put','strike':K,'premium':pp,'qty':1}]
            # MC: simulate terminal PnL
            terminal = MC.terminal_prices(S, iv, T)
            pnl = np.maximum(terminal - K, 0) + np.maximum(K - terminal, 0) - nd
            pop_m = float(np.mean(pnl > 0))
            ev = float(np.mean(pnl))
            std = float(np.std(pnl))
            pop_b = min(1.0, (1 - BSM.prob_otm(S,K+nd,T,iv,'call')) + (1 - BSM.prob_otm(S,K-nd,T,iv,'put')))
            return _adaptive_tail(name, legs, mp, ml, pop_b, pop_m, ev, std, ng, iv,
                K-nd, K+nd, nc=0, width=0, settings=settings, regime=regime)

        elif name == 'Long Strangle':
            # Debit: buy OTM call + OTM put — cheaper than straddle, needs bigger move
            n_otm = _n_gaps(1.0)  # v4.0: adaptive
            cK = snap(S + n_otm * g, g)
            pK = snap(S - n_otm * g, g)
            if cK <= S: cK = snap(S + g, g)
            if pK >= S: pK = snap(S - g, g)
            cp = BSM.call(S,cK,T,r,iv); pp = BSM.put(S,pK,T,r,iv)
            nd = cp + pp
            nc = -nd
            if nd <= 0: return None
            ml = nd
            mp = S * 0.5  # theoretical unlimited
            legs = [{'type':'Buy Call','strike':cK,'premium':cp,'qty':1},
                    {'type':'Buy Put','strike':pK,'premium':pp,'qty':1}]
            terminal = MC.terminal_prices(S, iv, T)
            pnl = np.maximum(terminal - cK, 0) + np.maximum(pK - terminal, 0) - nd
            pop_m = float(np.mean(pnl > 0))
            ev = float(np.mean(pnl))
            std = float(np.std(pnl))
            pop_b = min(1.0, (1 - BSM.prob_otm(S,cK+nd,T,iv,'call')) + (1 - BSM.prob_otm(S,pK-nd,T,iv,'put')))
            return _adaptive_tail(name, legs, mp, ml, pop_b, pop_m, ev, std, ng, iv,
                pK-nd, cK+nd, nc=0, width=0, settings=settings, regime=regime)

        elif name == 'Jade Lizard':
            nc_otm = max(2, _n_gaps(0.8))  # v4.0: adaptive
            np_otm = _n_gaps(0.6)  # v4.0: adaptive
            cK = snap(S + nc_otm * g, g)
            sp_k = snap(S - np_otm * g, g)
            lp_k = min(snap(S - (np_otm + max(2, _n_gaps(0.8))) * g, g), sp_k - min_wing)
            cp_v = BSM.call(S,cK,T,r,iv); sp_v = BSM.put(S,sp_k,T,r,iv); lp_v = BSM.put(S,lp_k,T,r,iv)
            nc = cp_v + sp_v - lp_v
            if nc < _min_prem: return None
            ml = max((sp_k - lp_k) - nc, 1)
            legs = [{'type':'Sell Call','strike':cK,'premium':cp_v,'qty':1},
                    {'type':'Sell Put','strike':sp_k,'premium':sp_v,'qty':1},
                    {'type':'Buy Put','strike':lp_k,'premium':lp_v,'qty':1}]
            pop_m, ev, std = MC.analyze(S, iv, T, legs, sim_vol=sim_vol)
            pop_b = max(0, BSM.prob_otm(S,cK,T,iv,'call') + BSM.prob_otm(S,sp_k,T,iv,'put') - 1)
            return _adaptive_tail(name, legs, nc, ml, pop_b, pop_m, ev, std, ng, iv,
                sp_k-nc, cK+nc, nc=nc, width=sp_k-lp_k, settings=settings, regime=regime)

        elif name == 'Calendar Spread':
            K = snap(S, g)
            # Front: sell near expiry, Back: buy further expiry
            fT = max(T * 0.5, 1/365)
            bT = max(T * 1.5, T + 7/365)
            fp = BSM.call(S,K,fT,r,iv)
            bp = BSM.call(S,K,bT,r,iv*0.95)  # back month lower IV (term structure)
            nd = bp - fp  # net debit
            nc = -nd
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
            return _adaptive_tail(name, legs, mp, ml, pop_b, pop_m, ev, std, ng, iv,
                K-mp, K+mp, nc=0, width=0, settings=settings, regime=regime)

        elif name == 'Broken Wing Butterfly':
            c = snap(S, g)
            n_lo = max(2, _n_gaps(0.9))  # v4.0: adaptive
            n_hi = max(3, _n_gaps(1.3))  # v4.0: adaptive
            lo = snap(S - n_lo * g, g)
            hi = snap(S + n_hi * g, g)
            if c - lo < min_wing: lo = c - min_wing
            if hi - c < min_wing: hi = c + min_wing
            # BWB: Buy 1 low, Sell 2 center, Buy 1 high (unbalanced)
            lo_p = BSM.call(S,lo,T,r,iv); c_p = BSM.call(S,c,T,r,iv); hi_p = BSM.call(S,hi,T,r,iv)
            nc = 2*c_p - lo_p - hi_p  # net credit (can be small or debit)
            mp = (c - lo) + nc  # max profit at center
            ml = max((hi - c) - nc, 1)  # max loss on upside
            if mp < _min_prem and nc < _min_prem: return None
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
            return _adaptive_tail(name, legs, mp, ml, pop_b, pop_m, ev, std, ng, iv,
                lo, hi, nc=nc, width=hi-lo, settings=settings, regime=regime)

        elif name == 'Ratio Spread':
            lK = snap(S, g)
            n_otm = max(2, _n_gaps(0.9))  # v4.0: adaptive
            sK = max(snap(S + n_otm * g, g), lK + min_wing)
            lp_v = BSM.call(S,lK,T,r,iv); sp_v = BSM.call(S,sK,T,r,iv)
            nc = 2*sp_v - lp_v  # sell 2 OTM, buy 1 ATM
            # Max profit at short strike: intrinsic of long - premium paid
            mp = (sK - lK) + nc  # value at sK
            ml = span_margin(S, iv) * 0.5  # unlimited risk above sK (reduced margin for partial hedge)
            if mp < _min_prem: return None
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
            return _adaptive_tail(name, legs, mp, ml, pop_b, pop_m, ev, std, ng, iv,
                lK, 2*sK-lK+nc, nc=nc, width=sK-lK, settings=settings, regime=regime)

    except Exception:
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

CL = {'gold':'#FFC300','bg':'#1A1A1A','border':'#2A2A2A','text':'#EAEAEA','muted':'#888',
      'green':'#10b981','red':'#ef4444','amber':'#f59e0b','cyan':'#06b6d4','purple':'#a855f7'}

def fmt(v):
    """Format number in Indian comma style: ₹1,23,456.78"""
    try:
        v = float(v)
        if np.isnan(v) or np.isinf(v): return '₹0.00'
        neg = v < 0; v = abs(v)
        ip = int(v); dp = v - ip
        s = str(ip)
        if len(s) > 3:
            last3 = s[-3:]; rest = s[:-3]
            parts = []
            while rest:
                parts.insert(0, rest[-2:]); rest = rest[:-2]
            parts.append(last3)
            f = ','.join(parts)
        else:
            f = s
        if dp > 0.005: f += f"{dp:.2f}"[1:]
        return f"{'-' if neg else ''}₹{f}"
    except Exception: return str(v)

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

def landing_page():
    """Premium landing page shown before analysis runs"""
    st.markdown(f"""<div class="hdr">
        <h1>VAAYDO — FnO Trade Intelligence</h1>
        <div class="tag">20 Mathematical Engines · 10 Strategy Evaluators · Ensemble Probability Fusion</div></div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)

    # Hero section
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='ib' style='text-align:center; padding:1.5rem;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>📐</div>
            <h4 style='color:#FFC300; margin:0;'>BSM + Monte Carlo</h4>
            <p style='font-size:0.8rem; color:#aaa; margin-top:0.5rem;'>10,000 antithetic paths · Full 9-Greeks · Ensemble POP fusion with inverse-variance weighting</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='ib' style='text-align:center; padding:1.5rem;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🔬</div>
            <h4 style='color:#FFC300; margin:0;'>Multi-Estimator Vol</h4>
            <p style='font-size:0.8rem; color:#aaa; margin-top:0.5rem;'>Yang-Zhang · Garman-Klass · Parkinson · GARCH(1,1) · Volatility Risk Premium</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='ib' style='text-align:center; padding:1.5rem;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🧠</div>
            <h4 style='color:#FFC300; margin:0;'>Regime Intelligence</h4>
            <p style='font-size:0.8rem; color:#aaa; margin-top:0.5rem;'>6-State Vol × 5-State Trend · ADX · Kalman Filter · CUSUM Break Detection</p></div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # Strategy universe — categorized by bias
    st.markdown("""<div class='ib' style='padding:1.5rem;'>
        <h4 style='color:#FFC300; margin:0 0 1rem 0; text-align:center;'>14 Strategy Universe</h4>
        <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:1rem;'>
            <div>
                <div style='font-size:0.7rem;font-weight:700;color:#10b981;margin-bottom:0.4rem;letter-spacing:1px;'>▲ BULLISH</div>
                <div style='display:flex;flex-wrap:wrap;gap:0.3rem;'>
                    <span style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:#10b981;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Bull Put Spread</span>
                    <span style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:#10b981;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Bull Call Spread</span>
                    <span style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:#10b981;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>BWB</span>
                    <span style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:#10b981;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Ratio Spread</span>
                </div>
            </div>
            <div>
                <div style='font-size:0.7rem;font-weight:700;color:#ef4444;margin-bottom:0.4rem;letter-spacing:1px;'>▼ BEARISH</div>
                <div style='display:flex;flex-wrap:wrap;gap:0.3rem;'>
                    <span style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#ef4444;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Bear Call Spread</span>
                    <span style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#ef4444;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Bear Put Spread</span>
                </div>
            </div>
            <div>
                <div style='font-size:0.7rem;font-weight:700;color:#FFC300;margin-bottom:0.4rem;letter-spacing:1px;'>◆ NEUTRAL</div>
                <div style='display:flex;flex-wrap:wrap;gap:0.3rem;'>
                    <span style='background:rgba(255,195,0,0.1);border:1px solid rgba(255,195,0,0.3);color:#FFC300;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Iron Condor</span>
                    <span style='background:rgba(255,195,0,0.1);border:1px solid rgba(255,195,0,0.3);color:#FFC300;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Iron Butterfly</span>
                    <span style='background:rgba(255,195,0,0.1);border:1px solid rgba(255,195,0,0.3);color:#FFC300;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Short Strangle</span>
                    <span style='background:rgba(255,195,0,0.1);border:1px solid rgba(255,195,0,0.3);color:#FFC300;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Short Straddle</span>
                    <span style='background:rgba(255,195,0,0.1);border:1px solid rgba(255,195,0,0.3);color:#FFC300;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Calendar</span>
                    <span style='background:rgba(255,195,0,0.1);border:1px solid rgba(255,195,0,0.3);color:#FFC300;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Jade Lizard</span>
                </div>
            </div>
            <div>
                <div style='font-size:0.7rem;font-weight:700;color:#A78BFA;margin-bottom:0.4rem;letter-spacing:1px;'>⚡ VOLATILE</div>
                <div style='display:flex;flex-wrap:wrap;gap:0.3rem;'>
                    <span style='background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3);color:#A78BFA;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Long Straddle</span>
                    <span style='background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3);color:#A78BFA;padding:0.25rem 0.5rem;border-radius:4px;font-size:0.7rem;'>Long Strangle</span>
                </div>
            </div>
        </div></div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # Engine stack
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("""<div class='ib' style='padding:1.2rem;'>
            <h4 style='color:#FFC300; margin:0 0 0.75rem 0;'>Pricing & Risk Stack</h4>
            <p style='font-size:0.78rem; color:#ccc; line-height:1.7; margin:0;'>
            <span style='color:#A78BFA;'>●</span> Black-Scholes-Merton — Full analytics + 9 Greeks<br>
            <span style='color:#34D399;'>●</span> Monte Carlo — 10K antithetic, generic payoff<br>
            <span style='color:#FBBF24;'>●</span> GARCH(1,1) — Conditional variance forecasting<br>
            <span style='color:#F87171;'>●</span> Kelly Criterion — Half-Kelly, confidence-weighted<br>
            <span style='color:#60A5FA;'>●</span> SPAN Margin — Realistic max-loss estimation</p></div>""", unsafe_allow_html=True)
    with e2:
        st.markdown("""<div class='ib' style='padding:1.2rem;'>
            <h4 style='color:#FFC300; margin:0 0 0.75rem 0;'>Scoring & Selection</h4>
            <p style='font-size:0.78rem; color:#ccc; line-height:1.7; margin:0;'>
            <span style='color:#A78BFA;'>●</span> Ensemble POP — Inverse-variance BSM+MC fusion<br>
            <span style='color:#34D399;'>●</span> 9-Factor Conviction — RA·POP·EV·Sharpe·Stab·IV·PQ·DTE·CUSUM<br>
            <span style='color:#FBBF24;'>●</span> IVP Gate — Credit/debit filtered by IV environment<br>
            <span style='color:#F87171;'>●</span> DTE Gate — Strategy viability per time-to-expiry<br>
            <span style='color:#60A5FA;'>●</span> CUSUM Penalty — Structural break regime detection</p></div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    st.markdown("""<div style='text-align:center; color:#666; font-size:0.75rem;'>
        <p>Select expiry date and click <strong style='color:#FFC300;'>Run Analysis</strong> in the sidebar to begin.</p></div>""", unsafe_allow_html=True)


def main():
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""<div style="text-align:center; padding:1rem 0; margin-bottom:1rem;">
            <div style="font-size:1.75rem; font-weight:800; color:#FFC300;">VAAYDO</div>
            <div style="color:#888; font-size:0.7rem; margin-top:0.25rem;">वायदो — FnO Trade Intelligence</div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="stitle">⚡ Expiry & Parameters</div>', unsafe_allow_html=True)

        today = date.today()
        # Default: next Thursday (NSE weekly expiry) with minimum 3 DTE
        _days_to_thu = (3 - today.weekday()) % 7
        if _days_to_thu < 3: _days_to_thu += 7  # ensure at least 3 DTE
        expiry_date = st.date_input("Expiry Date", value=today + timedelta(days=_days_to_thu),
                                    min_value=today + timedelta(days=1), max_value=today + timedelta(days=365))
        dte = (expiry_date - today).days
        st.caption(f"**{dte} days** to expiry ({expiry_date.strftime('%d %b %Y')})")

        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

        # Single smart button: "Run Analysis" when new/changed expiry, "Refresh Data" when same
        _has_run = st.session_state.get('analysis_run', False)
        _expiry_changed = st.session_state.get('last_expiry') != str(expiry_date)
        _btn_label = "🚀 Run Analysis" if (not _has_run or _expiry_changed) else "🔄 Refresh Data"
        if st.button(_btn_label, use_container_width=True, type="primary"):
            st.session_state.analysis_run = True
            st.session_state.last_expiry = str(expiry_date)
            st.cache_data.clear()
            st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="stitle">🎯 Filters</div>', unsafe_allow_html=True)
        min_ivp = st.slider("Min IV Percentile", 0, 100, 20)
        min_cv = st.slider("Min Conviction", 0, 100, 30)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class='ib'><p style='font-size:0.72rem; margin:0; color:#888; line-height:1.5;'>
            <strong>v4.0 Adaptive Engine</strong><br>
            <strong>Scoring:</strong> Bayesian conviction (certainty-weighted)<br>
            <strong>Regime:</strong> Fuzzy (continuous probability vectors)<br>
            <strong>Gating:</strong> Sigmoid/Beta viability (never binary)<br>
            <strong>Kelly:</strong> Uncertainty-discounted × entropy budget<br>
            <strong>Strikes:</strong> Delta-targeted (not EM multipliers)<br>
            <strong>Meta:</strong> Reflexivity · Entropy Gov · Diversify<br>
            <strong>Strategies:</strong> 14 Active · 4 bias · 3 types (C/D/H)</p></div>""", unsafe_allow_html=True)

    # ── LANDING PAGE (before analysis runs) ──
    if not st.session_state.get('analysis_run', False):
        landing_page()
        return

    settings = {'dte': max(dte, 1)}

    # Track expiry for button label
    st.session_state.last_expiry = str(expiry_date)

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
    st.toast(f"🔌 {sym_status} → {data_status}", icon="✅")

    # ── COMPUTE STRATEGIES ──
    # ── v4.0 ADAPTIVE ENGINE INITIALIZATION ──
    with st.spinner("Calibrating adaptive intelligence engine..."):
        global _engine
        _engine = AdaptiveEngine()
        _engine.calibrate(df)
        _sys_entropy = 0.5

    with st.spinner("Running adaptive scoring: BSM + MC + Bayesian conviction..."):
        all_trades = []
        for _, row in df.iterrows():
            rd = row.to_dict()
            if pd.isna(rd.get('price')) or pd.isna(rd.get('ATMIV')) or rd['price'] <= 0 or rd['ATMIV'] <= 0:
                continue
            # v4.0: Compute fuzzy regime per stock
            regime = _engine.compute_regime(rd)
            best = None; alt = None
            for sn in ALL_STRATS:
                # v4.0: Continuous viability replaces binary gates
                viability = _engine.compute_viability(sn, regime, settings['dte'])
                if viability < 0.05: continue  # near-zero viability = skip (graceful)
                try:
                    res = score_strategy(sn, rd, settings, iv_mult=1.0, regime=regime)
                    if res:
                        if best is None or res.conviction_score > best.conviction_score:
                            alt = best
                            best = res
                        elif alt is None or res.conviction_score > alt.conviction_score:
                            alt = res
                except Exception:
                    continue
            if best:
                vr = detect_vol_regime(rd.get('IVPercentile', 50))
                tr = detect_trend(rd['price'], rd.get('ma20_daily', rd['price']),
                    rd.get('ma50_daily', rd['price']), rd.get('rsi_daily', 50),
                    rd.get('% change', 0), rd.get('adx', 20), rd.get('kalman_trend', 0))
                # Dynamic strategy labeling (credit/debit auto-detection)
                _sname = best.name
                if _sname == 'Iron Condor':
                    _sname = 'Short Iron Condor' if best.net_credit > 0 else 'Long Iron Condor'
                elif _sname == 'Iron Butterfly':
                    _sname = 'Short Iron Butterfly' if best.net_credit > 0 else 'Long Iron Butterfly'
                _bias = get_bias(_sname)
                _lot = rd.get('lot_size', 1)
                _mp_lot = round(best.max_profit * _lot, 2)
                _ml_lot = round(best.max_loss * _lot, 2)
                _ev_lot = best.expected_value * _lot
                _nc_lot = best.net_credit * _lot
                _theta_day = best.net_greeks.theta * _lot if best.net_greeks else 0
                _rom = (_mp_lot / max(_ml_lot, 1)) * 100
                _alt_name = None; _alt_cv = 0
                if alt:
                    _an = alt.name
                    if _an == 'Iron Condor': _an = 'Short IC' if alt.net_credit > 0 else 'Long IC'
                    elif _an == 'Iron Butterfly': _an = 'Short IB' if alt.net_credit > 0 else 'Long IB'
                    _alt_name = _an; _alt_cv = alt.conviction_score
                all_trades.append({**rd, 'strategy': _sname, 'conviction_score': best.conviction_score,
                    'pop': best.pop_ensemble, 'ev': best.expected_value, 'sharpe': best.sharpe_ratio,
                    'kelly_frac': best.kelly_fraction, 'net_credit': best.net_credit,
                    'max_profit': best.max_profit, 'max_loss': best.max_loss,
                    'risk_score': best.risk_score, 'stability': best.stability_score,
                    'vol_regime': vr.value, 'trend_regime': tr.value, '_result': best,
                    'mp_lot': _mp_lot, 'ml_lot': _ml_lot, 'ev_lot': _ev_lot, 'nc_lot': _nc_lot,
                    'theta_day': _theta_day, 'rom_pct': _rom,
                    'alt_strategy': _alt_name, 'alt_conviction': _alt_cv,
                    'bias': get_bias(_sname), 'stype': STRATEGY_TYPE.get(best.name, 'HYBRID'),
                    'conviction_std': getattr(best, 'conviction_std', 0),
                    'conviction_ci_lower': getattr(best, 'conviction_ci_lower', 0),
                    'conviction_ci_upper': getattr(best, 'conviction_ci_upper', 0),
                    'viability': getattr(best, 'viability', 0),
                    'model_agreement': getattr(best, 'model_agreement', 0),
                    'pop_std': getattr(best, 'pop_std', 0),
                    'direction': STRATEGY_STRUCTURE.get(best.name, {}).get('direction', 'NEUTRAL'),
                    'price': rd.get('price', 0)})

    filtered = [t for t in all_trades if t['IVPercentile'] >= min_ivp and t['conviction_score'] >= min_cv]
    filtered.sort(key=lambda x: x['conviction_score'], reverse=True)
    # v4.0: System entropy for governance
    _sys_entropy = _engine.system_entropy() if _engine else 0.5

    # ── METRICS BAR ──
    avg_iv = df['IVPercentile'].mean(); avg_pcr = df['PCR'].mean()
    hc = len([t for t in filtered if t['conviction_score'] >= 65])
    # Market regime: credit vs debit favored
    _credit_count = len([t for t in filtered if get_strategy_type(t['strategy']) == 'CREDIT'])
    _debit_count = len([t for t in filtered if get_strategy_type(t['strategy']) == 'DEBIT'])
    cusum_alerts = len(df[df['CUSUM_Alert'] == True])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f"<div class='mc gold'><h4>Securities</h4><h2>{len(df)}</h2><div class='sub'>{len(filtered)} pass filters</div></div>", unsafe_allow_html=True)
    with c2:
        cs = 'ok' if hc > 5 else ('warn' if hc > 2 else 'bad')
        st.markdown(f"<div class='mc {cs}'><h4>High Conviction</h4><h2>{hc}</h2><div class='sub'>Score ≥ 65</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='mc {'ok' if avg_iv > 55 else 'warn' if avg_iv > 35 else 'info'}'><h4>Avg IV Percentile</h4><h2>{avg_iv:.0f}%</h2><div class='sub'>GARCH avg: {df['GARCH_Vol'].mean():.1f}%</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='mc {'ok' if avg_pcr > 1.2 else 'bad' if avg_pcr < 0.8 else 'info'}'><h4>Avg PCR</h4><h2>{avg_pcr:.2f}</h2><div class='sub'>{'Bullish' if avg_pcr > 1.2 else 'Bearish' if avg_pcr < 0.8 else 'Neutral'} bias</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='mc {'bad' if cusum_alerts > 5 else 'warn' if cusum_alerts > 0 else 'ok'}'><h4>CUSUM Alerts</h4><h2>{cusum_alerts}</h2><div class='sub'>Structural breaks</div></div>", unsafe_allow_html=True)
    _top5 = filtered[:5]
    _total_cap = sum(t.get('ml_lot', 0) for t in _top5) if _top5 else 0
    _total_theta = sum(t.get('theta_day', 0) for t in _top5) if _top5 else 0
    _avg_rom = sum(t.get('rom_pct', 0) for t in _top5) / max(len(_top5), 1) if _top5 else 0

    # v4.0 System Intelligence metrics
    _conf_thresh = _engine.confidence_threshold() if _engine else 35
    _naked_frac = _engine.max_naked_fraction() if _engine else 0.40
    _entropy_cls = 'ok' if _sys_entropy < 0.3 else ('warn' if _sys_entropy < 0.6 else 'bad')
    c_e1, c_e2, c_e3, c_e4 = st.columns(4)
    with c_e1: st.markdown(f"<div class='mc {_entropy_cls}'><h4>System Entropy</h4><h2>{_sys_entropy:.2f}</h2><div class='sub'>{'Low — confident' if _sys_entropy < 0.3 else ('Medium — cautious' if _sys_entropy < 0.6 else 'High — conservative')}</div></div>", unsafe_allow_html=True)
    with c_e2: st.markdown(f"<div class='mc'><h4>Conviction Floor</h4><h2>{_conf_thresh:.0f}</h2><div class='sub'>Adaptive threshold</div></div>", unsafe_allow_html=True)
    with c_e3: st.markdown(f"<div class='mc'><h4>Max Naked</h4><h2>{_naked_frac*100:.0f}%</h2><div class='sub'>Entropy-governed</div></div>", unsafe_allow_html=True)
    with c_e4:
        # Strategy diversity (reflexivity check)
        from collections import Counter
        _sc = Counter(t.get('strategy','') for t in filtered)
        _dc = Counter(t.get('direction','NEUTRAL') for t in filtered)
        _rp = _engine.reflexivity_penalty(dict(_sc), dict(_dc)) if _engine else 1.0
        _rp_cls = 'ok' if _rp > 0.90 else ('warn' if _rp > 0.75 else 'bad')
        st.markdown(f"<div class='mc {_rp_cls}'><h4>Diversity Score</h4><h2>{_rp:.2f}</h2><div class='sub'>{'Diverse' if _rp > 0.90 else ('Moderate' if _rp > 0.75 else 'Concentrated ⚠')}</div></div>", unsafe_allow_html=True)

    c6, c7, c8, c9 = st.columns(4)
    _regime_lbl = 'CREDIT-FAVORED' if avg_iv >= 55 else ('DEBIT-FAVORED' if avg_iv <= 35 else 'MIXED')
    _regime_cls = 'gold' if avg_iv >= 55 else ('info' if avg_iv <= 35 else 'warn')
    _regime_sub = f'Avg IVP {avg_iv:.0f}% · Sell premium' if avg_iv >= 55 else (f'Avg IVP {avg_iv:.0f}% · Buy premium' if avg_iv <= 35 else f'Avg IVP {avg_iv:.0f}% · Both viable')
    with c6: st.markdown(f"<div class='mc {_regime_cls}'><h4>Market Regime</h4><h2 style='font-size:1.2rem;'>{_regime_lbl}</h2><div class='sub'>{_regime_sub}</div></div>", unsafe_allow_html=True)
    with c7: st.markdown(f"<div class='mc'><h4>Strategy Mix</h4><h2 style='font-size:1.2rem;'>{_credit_count}C / {_debit_count}D</h2><div class='sub'>Credit / Debit selected</div></div>", unsafe_allow_html=True)
    with c8: st.markdown(f"<div class='mc {'ok' if _total_theta > 0 else 'bad'}'><h4>Top-5 Θ/Day</h4><h2 style='font-size:1.2rem;'>{fmt(_total_theta)}</h2><div class='sub'>Combined daily theta</div></div>", unsafe_allow_html=True)
    with c9: st.markdown(f"<div class='mc'><h4>Top-5 Capital</h4><h2 style='font-size:1.2rem;'>{fmt(_total_cap)}</h2><div class='sub'>Avg ROM {_avg_rom:.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Trade Radar", "🔬 Deep Analysis", "📊 Rankings", "📐 Probability Lab"])

    with tab1:
        st.markdown("<div style='margin-bottom:1rem;'><span style='font-size:1.1rem;font-weight:700;color:#EAEAEA;'>Top FnO Opportunities</span><span style='color:#888;font-size:0.85rem;margin-left:0.75rem;'>§9.1 Unified Conviction · Ensemble POP · Antithetic MC</span></div>", unsafe_allow_html=True)
        # NaN-safe filter
        filtered = [t for t in filtered if not any(np.isnan(v) for k, v in t.items() 
                    if isinstance(v, (int, float)) and k in ('conviction_score','pop','ev','sharpe','price','ATMIV'))]
        # v4.0: Diversified portfolio selection (not just top 9 by conviction)
        top = _engine.diversify(filtered, 9) if _engine and len(filtered) > 9 else filtered[:9]
        if not top:
            st.info("No trades pass filters. Lower conviction or IV percentile thresholds.")
        else:
            cols = st.columns(3)
            for i, t in enumerate(top):
                with cols[i % 3]:
                    cv = t['conviction_score']; cc = '#10b981' if cv >= 65 else ('#f59e0b' if cv >= 40 else '#ef4444')
                    _b = get_bias(t['strategy'])
                    _bcls = 'bias-bull' if _b==StrategyBias.BULLISH else ('bias-bear' if _b==StrategyBias.BEARISH else ('bias-vol' if _b==StrategyBias.VOLATILE else 'bias-neut'))
                    _btag = 'bull' if _b==StrategyBias.BULLISH else ('bear' if _b==StrategyBias.BEARISH else ('vol' if _b==StrategyBias.VOLATILE else 'neut'))
                    cusum_warn = " ⚠️" if t.get('CUSUM_Alert') else ""
                    st.markdown(f"""<div class='tc {_bcls}'>
                        <div style="display:flex;justify-content:space-between;align-items:start;">
                        <div><div class='sym'>{t['Instrument']}{cusum_warn}</div>
                        <div class='strat'>{t['strategy']}</div>
                        <span class='bias-tag {_btag}'>{get_bias_label(t['strategy'])}</span>
                        <span class='bias-tag {get_type_tag(t["strategy"])[0]}'>{get_type_tag(t["strategy"])[1]}</span></div>
                        <div style="text-align:right;"><div style="font-size:1.8rem;font-weight:800;color:{cc};font-family:'JetBrains Mono',monospace;">{cv:.0f}</div>
                        <div style="font-size:0.58rem;color:#888;">±{t.get('conviction_std',0):.0f} [{t.get('conviction_ci_lower',0):.0f}–{t.get('conviction_ci_upper',0):.0f}]</div></div></div>
                        <div class='gr'>
                        <div class='gi'><label>₹ Profit (lot)</label><div class='v tg'>{fmt(t.get('mp_lot',0))}</div></div>
                        <div class='gi'><label>₹ Risk (lot)</label><div class='v tr'>{fmt(t.get('ml_lot',0))}</div></div>
                        <div class='gi'><label>ROM %</label><div class='v {"tg" if t.get("rom_pct",0)>15 else "ta"}'>{t.get('rom_pct',0):.1f}%</div></div>
                        <div class='gi'><label>POP</label><div class='v tg'>{t['pop']*100:.1f}%</div></div>
                        <div class='gi'><label>Θ/Day ₹</label><div class='v {"tg" if t.get("theta_day",0)>0 else "tr"}'>{fmt(t.get('theta_day',0))}</div></div>
                        <div class='gi'><label>IV %ile</label><div class='v'>{t['IVPercentile']:.0f}%</div></div>
                        <div class='gi'><label>Spot × Lot</label><div class='v'>{fmt(t['price'])} × {t.get("lot_size",1)}</div></div>
                        <div class='gi'><label>Sharpe</label><div class='v'>{t['sharpe']:.2f}</div></div></div>
                        <div class='cb'><div class='cf' style='width:{cv}%;background:linear-gradient(90deg,{cc},{cc}aa);'></div></div>
                        <div style="display:flex;gap:0.5rem;margin-top:0.75rem;">
                        <span class='sb {"buy" if "UP" in t.get("trend_regime","") else ("sell" if "DOWN" in t.get("trend_regime","") else "neut")}'>{t.get('trend_regime','NEUTRAL')}</span>
                        <span class='sb prem'>{t.get('vol_regime','NORMAL')}</span>
                        {"<span style='font-size:0.62rem;color:#666;margin-left:auto;'>Alt: " + str(t.get("alt_strategy","")) + " (" + str(int(t.get("alt_conviction",0))) + ")</span>" if t.get('alt_strategy') else ""}</div></div>""", unsafe_allow_html=True)
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
                st.plotly_chart(vol_estimator_chart(row), width='stretch', key=f'vol_est_{sel}')
            with vc2:
                st.markdown("<span style='font-weight:700;color:#EAEAEA;'>Expected Move Distribution</span>", unsafe_allow_html=True)
                st.plotly_chart(em_chart(S, iv, T), width='stretch', key=f'em_{sel}')

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("<span style='font-weight:700;color:#EAEAEA;'>Strategy Rankings</span><span style='color:#888;margin-left:0.75rem;font-size:0.85rem;'>All 14 strategies · Adaptive scoring · Full Greeks</span>", unsafe_allow_html=True)

            # v4.0: compute regime for this stock
            _da_regime = _engine.compute_regime(row) if _engine else None
            strats = []
            for sn in ALL_STRATS:
                try:
                    res = score_strategy(sn, row, settings, regime=_da_regime)
                    if res: strats.append(res)
                except Exception:
                    continue
            strats.sort(key=lambda x: x.conviction_score, reverse=True)

            for rank, s in enumerate(strats[:5], 1):
                cv = s.conviction_score
                _stype = STRATEGY_TYPE.get(s.name, 'HYBRID')
                _sbias = get_bias_label(s.name)
                with st.expander(f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else f'#{rank}'} {s.name} [{_stype}] — Conv: {cv:.0f} | POP: {s.pop_ensemble*100:.1f}% | Sharpe: {s.sharpe_ratio:.2f} | {_sbias}", expanded=(rank==1)):
                    ec1, ec2 = st.columns([2, 1])
                    with ec1:
                        lh = "<table class='gt'><tr><th>Leg</th><th>Strike</th><th>Qty</th><th>Premium</th></tr>"
                        for l in s.legs:
                            cc = 'tr' if 'Sell' in l['type'] else 'tg'
                            lh += f"<tr><td class='{cc}'>{l['type']}</td><td>{fmt(l['strike'])}</td><td>{l.get('qty',1)}</td><td>{fmt(abs(l['premium']))}</td></tr>"
                        st.markdown(lh + "</table>", unsafe_allow_html=True)
                        st.plotly_chart(payoff_chart(s, S), width='stretch', key=f'payoff_{s.name}_{rank}')
                    with ec2:
                        st.plotly_chart(gauge(cv), width='stretch', key=f'gauge_{s.name}_{rank}')
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
                        _mp_l = s.max_profit * lot; _ml_l = s.max_loss * lot
                        _td = gk.theta * lot; _rom = (_mp_l / max(_ml_l, 1)) * 100
                        st.markdown(f"""<div class='ib' style='margin-top:0.5rem;'><h4>Lot-Adjusted (×{lot})</h4><p>
                            <strong>₹ Profit:</strong> <span class='mono tg'>{fmt(_mp_l)}</span><br>
                            <strong>₹ Risk:</strong> <span class='mono tr'>{fmt(_ml_l)}</span><br>
                            <strong>ROM:</strong> <span class='mono {"tg" if _rom>15 else "ta"}'>{_rom:.1f}%</span><br>
                            <strong>Θ/Day ₹:</strong> <span class='mono {"tg" if _td>0 else "tr"}'>{fmt(_td)}</span><br>
                            <strong>Kelly %:</strong> <span class='mono tgl'>{s.kelly_fraction*100:.1f}%</span><br>
                            <strong>Gap:</strong> <span class='mono'>₹{auto_gap(S):.0f}</span></p></div>""", unsafe_allow_html=True)

    with tab3:
        if filtered:
            rdf = pd.DataFrame([{k: v for k, v in t.items() if k != '_result'} for t in filtered])
            cols_show = ['Instrument','strategy','bias','stype','conviction_score','pop','mp_lot','ml_lot','rom_pct','theta_day',
                        'ev','sharpe','kelly_frac','net_credit','max_profit','max_loss',
                        'IVPercentile','price','lot_size','risk_score','stability',
                        'alt_strategy','alt_conviction','vol_regime','trend_regime','CUSUM_Alert']
            avail = [c for c in cols_show if c in rdf.columns]
            display = rdf[avail].sort_values('conviction_score', ascending=False).copy()
            # Round numeric columns for readability
            for col in ['conviction_score','ev','sharpe','kelly_frac','risk_score','stability','net_credit','max_profit','max_loss','mp_lot','ml_lot','rom_pct','theta_day','alt_conviction']:
                if col in display.columns:
                    display[col] = pd.to_numeric(display[col], errors='coerce').fillna(0).round(2)
            if 'pop' in display.columns:
                display['pop'] = (display['pop'] * 100).round(1)
            display.columns = [c.replace('_', ' ').title() for c in display.columns]
            st.dataframe(display, width='stretch', height=600)
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
            st.plotly_chart(fig, width='stretch', key='mc_hist')
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
            st.plotly_chart(fig2, width='stretch', key='mc_paths')
            lg = auto_gap(lS)
            strikes = [snap(lS + i * lg, lg) for i in range(-3, 4)]
            gdata = []
            for K in strikes:
                cg = BSM.greeks(lS, K, lT, BSM.R, liv, 'call'); pg = BSM.greeks(lS, K, lT, BSM.R, liv, 'put')
                gdata.append({'Strike': fmt(K), 'C.Δ': f"{cg.delta:.3f}", 'P.Δ': f"{pg.delta:.3f}", 'Γ': f"{cg.gamma:.5f}",
                             'C.Θ': f"{cg.theta:.2f}", 'P.Θ': f"{pg.theta:.2f}", 'ν': f"{cg.vega:.2f}",
                             'Vanna': f"{cg.vanna:.4f}", 'Volga': f"{cg.volga:.4f}"})
            st.dataframe(pd.DataFrame(gdata), width='stretch', hide_index=True)


if 'sel' not in st.session_state: st.session_state.sel = None
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'last_expiry' not in st.session_state: st.session_state.last_expiry = None
if __name__ == "__main__": main()
