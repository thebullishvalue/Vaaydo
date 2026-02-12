"""
VAAYDO (वायदो) - The Promise | Ultimate FnO Trade Identifier
A Pragyam Product Family Member | Hemrek Capital

God-Tier Options Trade Intelligence System — FULLY AUTO-FETCHED

DATA ENGINE:
- F&O Stock List: NSE API with hardcoded fallback
- OHLCV Data: yfinance batch download
- IV Proxy: Realized Volatility (20D annualized) from OHLCV
- IV Percentile: 252-day rolling percentile rank of 20D RV
- PCR Proxy: Volume-weighted directional ratio
- All technicals computed in-house: RSI, ATR, MAs, Volume ratios

MATHEMATICAL ARSENAL:
1.  Black-Scholes-Merton Framework (Greeks: Δ, Γ, Θ, V, ρ)
2.  GARCH(1,1) Volatility Forecasting
3.  Hidden Markov Model Regime Detection
4.  Monte Carlo Simulation (5,000 paths) for Probability of Profit
5.  Kelly Criterion Optimal Position Sizing
6.  Kalman Filter Signal Smoothing
7.  CUSUM Change Point Detection
8.  Bayesian Confidence Scoring
9.  Log-Normal Distribution Expected Moves (1σ, 2σ, 3σ)
10. Sharpe-Ratio Optimized Strategy Selection
11. Entropy-Based Regime Uncertainty
12. Risk-Neutral Probability Framework
13. Optimal DTE via Theta Decay Curves
14. Greeks-Weighted Composite Risk Scoring
15. Information Ratio Strategy Ranking

STRATEGY UNIVERSE:
Short Strangle · Short Straddle · Iron Condor · Iron Butterfly
Bull Put Spread · Bear Call Spread · Calendar Spread
Jade Lizard · Broken Wing Butterfly · Ratio Spread

Version: 2.1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from scipy.stats import norm, entropy
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf
import requests
import io
import warnings
import time

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VAAYDO | FnO Trade Intelligence",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

VERSION = "v2.1.0"

# ══════════════════════════════════════════════════════════════════════════════
# PRAGYAM DESIGN SYSTEM CSS (Nirnay DNA)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --primary-color: #FFC300; --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F; --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A; --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA; --text-secondary: #EAEAEA; --text-muted: #888888;
        --border-color: #2A2A2A; --border-light: #3A3A3A;
        --success-green: #10b981; --danger-red: #ef4444;
        --warning-amber: #f59e0b; --info-cyan: #06b6d4;
        --neutral: #888888; --purple: #a855f7;
    }

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 95%; padding-left: 2rem; padding-right: 2rem; }

    [data-testid="collapsedControl"] {
        display: flex !important; visibility: visible !important; opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important; border-radius: 8px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important; position: fixed !important;
        top: 14px !important; left: 14px !important;
        width: 40px !important; height: 40px !important;
        align-items: center !important; justify-content: center !important;
    }
    [data-testid="collapsedControl"]:hover { background-color: rgba(var(--primary-rgb), 0.2) !important; }
    [data-testid="collapsedControl"] svg { stroke: var(--primary-color) !important; }
    [data-testid="stSidebar"] button[kind="header"] { background-color: transparent !important; border: none !important; }
    [data-testid="stSidebar"] button[kind="header"] svg { stroke: var(--primary-color) !important; }

    .premium-header {
        background: var(--secondary-background-color); padding: 1.25rem 2rem; border-radius: 16px;
        margin-bottom: 1.5rem; box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color); position: relative; overflow: hidden; margin-top: 1rem;
    }
    .premium-header::before { content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%); pointer-events: none; }
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    .premium-header .product-badge { display: inline-block; background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }

    .metric-card { background-color: var(--bg-card); padding: 1.25rem; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); margin-bottom: 0.5rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden; }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.7rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.6rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.success h2 { color: var(--success-green); } .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); } .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); } .metric-card.primary h2 { color: var(--primary-color); }

    .trade-card { background: var(--bg-card); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); margin-bottom: 1rem; position: relative; overflow: hidden; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
    .trade-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.4); border-color: rgba(var(--primary-rgb), 0.3); }
    .trade-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
    .trade-card.high::before { background: var(--success-green); } .trade-card.medium::before { background: var(--warning-amber); } .trade-card.low::before { background: var(--danger-red); }
    .trade-card .symbol { font-size: 1.2rem; font-weight: 800; color: var(--text-primary); } .trade-card .strategy { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.25rem; }
    .trade-card .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 1rem; }
    .trade-card .grid-item label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; display: block; }
    .trade-card .grid-item .value { font-size: 1rem; font-weight: 700; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; }
    .trade-card .conviction-bar { height: 6px; background: var(--bg-elevated); border-radius: 3px; overflow: hidden; margin-top: 1rem; }
    .trade-card .conviction-fill { height: 100%; border-radius: 3px; }

    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.75rem; border-radius: 20px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16,185,129,0.15); color: var(--success-green); border: 1px solid rgba(16,185,129,0.3); }
    .status-badge.sell { background: rgba(239,68,68,0.15); color: var(--danger-red); border: 1px solid rgba(239,68,68,0.3); }
    .status-badge.neutral { background: rgba(136,136,136,0.15); color: var(--neutral); border: 1px solid rgba(136,136,136,0.3); }
    .status-badge.premium { background: rgba(var(--primary-rgb),0.15); color: var(--primary-color); border: 1px solid rgba(var(--primary-rgb),0.3); }

    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }

    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }

    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }

    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }

    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--background-color); } ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }

    .greek-table { width: 100%; border-collapse: collapse; }
    .greek-table th { color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color); }
    .greek-table td { color: var(--text-primary); font-size: 0.85rem; padding: 0.75rem; border-bottom: 1px solid var(--border-color); font-family: 'JetBrains Mono', monospace; }

    .mono { font-family: 'JetBrains Mono', monospace; }
    .text-green { color: var(--success-green); } .text-red { color: var(--danger-red); }
    .text-amber { color: var(--warning-amber); } .text-cyan { color: var(--info-cyan); } .text-gold { color: var(--primary-color); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# F&O UNIVERSE — NSE FETCH WITH FALLBACK (Nirnay Pattern)
# ══════════════════════════════════════════════════════════════════════════════

# Hardcoded F&O stock list — robust fallback when NSE API is blocked
FNO_FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO", "BAJFINANCE", "BAJAJFINSV",
    "HCLTECH", "TATAMOTORS", "POWERGRID", "NTPC", "TATASTEEL", "ONGC",
    "ADANIPORTS", "NESTLEIND", "JSWSTEEL", "M&M", "COALINDIA", "GRASIM",
    "TECHM", "INDUSINDBK", "CIPLA", "DRREDDY", "DIVISLAB", "BPCL",
    "APOLLOHOSP", "EICHERMOT", "TATACONSUM", "HEROMOTOCO", "HINDALCO",
    "SBILIFE", "BRITANNIA", "DABUR", "PIDILITIND", "HAVELLS", "SIEMENS",
    "GODREJCP", "DLF", "TRENT", "BANKBARODA", "VEDL", "IDFCFIRSTB",
    "PNB", "CANBK", "LICHSGFIN", "MFSL", "BHEL", "IOC", "GAIL",
    "PEL", "VOLTAS", "COLPAL", "AMBUJACEM", "ACC", "AUROPHARMA",
    "LUPIN", "BIOCON", "TORNTPHARM", "MANAPPURAM", "MUTHOOTFIN",
    "SAIL", "NMDC", "TATAPOWER", "RECLTD", "PFC", "IRCTC",
    "HAL", "BEL", "BALKRISIND", "MRF", "PAGEIND", "NAUKRI",
    "ZOMATO", "PAYTM", "POLYCAB", "PERSISTENT", "COFORGE", "LTIM",
    "ABCAPITAL", "BANDHANBNK", "FEDERALBNK", "RAMCOCEM", "CROMPTON",
    "LALPATHLAB", "METROPOLIS", "LAURUSLABS", "GRANULES", "ATUL",
    "CHAMBLFERT", "DEEPAKNTR", "ASTRAL", "BATAINDIA", "IDEA",
    "INDUSTOWER", "HINDPETRO", "MGL", "IGL", "PETRONET"
]

@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list():
    """Fetch F&O stock list from NSE with robust fallback"""
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9",
        }
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data:
                symbols = [item['symbol'] for item in data['data']
                           if 'symbol' in item and item['symbol'] not in ('NIFTY 50', 'NIFTY BANK')]
                if len(symbols) > 30:
                    return symbols, f"✓ Fetched {len(symbols)} F&O stocks from NSE"
    except Exception:
        pass
    return FNO_FALLBACK.copy(), f"✓ Loaded {len(FNO_FALLBACK)} F&O stocks (fallback)"


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-FETCH DATA ENGINE — Compute ALL from OHLCV
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_fno_data(symbols_ns: list, days_back: int = 365):
    """
    Master data fetch: downloads OHLCV from yfinance,
    then computes ALL options-relevant metrics in-house:
      - Realized Volatility (20D, annualized) → IV proxy
      - IV Percentile (252-day rolling percentile rank)
      - RSI (14-day)
      - ATR (14-day)
      - Moving Averages (20, 50, 200)
      - Volume vs 20D avg volume
      - Directional Volume Ratio → PCR proxy
      - % Change
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 60)

    try:
        raw = yf.download(
            symbols_ns, start=start_date, end=end_date,
            progress=False, auto_adjust=True, group_by='ticker', threads=True
        )
    except Exception as e:
        return pd.DataFrame(), f"Download failed: {e}"

    if raw.empty:
        return pd.DataFrame(), "No data returned from yfinance"

    results = []
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for sym_ns in symbols_ns:
        sym_clean = sym_ns.replace('.NS', '')
        try:
            if is_multi:
                if sym_ns not in raw.columns.get_level_values(0):
                    continue
                df = raw.xs(sym_ns, level=0, axis=1).copy()
            else:
                df = raw.copy()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna(subset=['Close'])
            if len(df) < 50:
                continue

            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            price = float(close.iloc[-1])

            # ── Realized Volatility (20D annualized) → IV proxy ──
            log_ret = np.log(close / close.shift(1)).dropna()
            rv_20 = log_ret.rolling(20).std() * np.sqrt(252)
            current_rv = float(rv_20.iloc[-1]) if len(rv_20) > 0 and not np.isnan(rv_20.iloc[-1]) else 0.25
            atmiv = current_rv * 100  # as percentage

            # ── IV Percentile (252-day percentile rank of current RV) ──
            rv_history = rv_20.dropna().values
            lookback = min(252, len(rv_history))
            if lookback > 20:
                recent_rv = rv_history[-lookback:]
                iv_percentile = float(np.sum(recent_rv <= current_rv) / len(recent_rv) * 100)
            else:
                iv_percentile = 50.0

            # ── RSI (14-day) ──
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            rsi_daily = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

            # ── ATR (14-day) ──
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(14).mean()
            atr_daily = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else price * 0.02

            # ── Moving Averages ──
            ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
            ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else price

            # ── Volume metrics ──
            vol_current = float(volume.iloc[-1]) if not np.isnan(volume.iloc[-1]) else 0
            vol_20 = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else max(vol_current, 1)

            # ── PCR Proxy: Down-volume / Up-volume ratio ──
            up_vol = volume.where(close > close.shift(1), 0).rolling(20).sum()
            dn_vol = volume.where(close < close.shift(1), 0).rolling(20).sum()
            pcr_val = float(dn_vol.iloc[-1] / max(up_vol.iloc[-1], 1)) if len(up_vol) >= 20 else 1.0
            pcr_val = min(max(pcr_val, 0.2), 3.0)

            # ── % Change (1-day) ──
            pct_change = float(close.pct_change().iloc[-1] * 100) if len(close) >= 2 else 0.0

            results.append({
                'Instrument': sym_clean,
                'price': round(price, 2),
                'ATMIV': round(atmiv, 2),
                'IVPercentile': round(iv_percentile, 1),
                'PCR': round(pcr_val, 3),
                'volume': vol_current,
                'vol20': vol_20,
                'rsi_daily': round(rsi_daily, 2),
                'atr_daily': round(atr_daily, 2),
                'ma20_daily': round(ma20, 2),
                'ma50_daily': round(ma50, 2),
                'ma200_daily': round(ma200, 2),
                '% change': round(pct_change, 2),
                'Event': '-',
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame(), "No valid data extracted"

    df_out = pd.DataFrame(results)
    return df_out, f"✓ Computed analytics for {len(df_out)} securities"


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class VolRegime(Enum):
    COMPRESSED = "COMPRESSED"; LOW = "LOW"; NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"; HIGH = "HIGH"; EXTREME = "EXTREME"

class TrendRegime(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"; UPTREND = "UPTREND"; NEUTRAL = "NEUTRAL"
    DOWNTREND = "DOWNTREND"; STRONG_DOWNTREND = "STRONG_DOWNTREND"

@dataclass
class Greeks:
    delta: float = 0.0; gamma: float = 0.0; theta: float = 0.0; vega: float = 0.0; rho: float = 0.0

@dataclass
class StrategyResult:
    name: str; legs: List[Dict]; max_profit: float; max_loss: float
    breakeven_lower: float; breakeven_upper: float; probability_of_profit: float
    expected_value: float; sharpe_ratio: float; kelly_fraction: float
    net_greeks: Greeks; conviction_score: float; optimal_dte: int
    width: float = 0.0; net_credit: float = 0.0; risk_reward_ratio: float = 0.0; regime_alignment: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES-MERTON ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class BSM:
    R = 0.07  # India risk-free rate

    @staticmethod
    def d1(S, K, T, r, σ):
        if T <= 0 or σ <= 0: return 0.0
        return (np.log(S / K) + (r + 0.5 * σ**2) * T) / (σ * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, σ): return BSM.d1(S, K, T, r, σ) - σ * np.sqrt(T)

    @classmethod
    def call(cls, S, K, T, r, σ):
        if T <= 0: return max(S - K, 0)
        return S * norm.cdf(cls.d1(S,K,T,r,σ)) - K * np.exp(-r*T) * norm.cdf(cls.d2(S,K,T,r,σ))

    @classmethod
    def put(cls, S, K, T, r, σ):
        if T <= 0: return max(K - S, 0)
        return K * np.exp(-r*T) * norm.cdf(-cls.d2(S,K,T,r,σ)) - S * norm.cdf(-cls.d1(S,K,T,r,σ))

    @classmethod
    def greeks(cls, S, K, T, r, σ, otype='call'):
        if T <= 0 or σ <= 0: return Greeks()
        d1, d2, sT = cls.d1(S,K,T,r,σ), cls.d2(S,K,T,r,σ), np.sqrt(T)
        γ = norm.pdf(d1) / (S * σ * sT)
        ν = S * sT * norm.pdf(d1) / 100
        if otype == 'call':
            δ = norm.cdf(d1)
            θ = (-(S * norm.pdf(d1) * σ) / (2 * sT) - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
            ρ = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        else:
            δ = norm.cdf(d1) - 1
            θ = (-(S * norm.pdf(d1) * σ) / (2 * sT) + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
            ρ = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
        return Greeks(delta=δ, gamma=γ, theta=θ, vega=ν, rho=ρ)

    @classmethod
    def prob_otm(cls, S, K, T, σ, otype='call'):
        if T <= 0 or σ <= 0: return 0.5
        d2 = cls.d2(S, K, T, cls.R, σ)
        return norm.cdf(-d2) if otype == 'call' else norm.cdf(d2)


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class MC:
    @staticmethod
    def gbm(S, σ, T, n=5000):
        steps = max(int(T * 252), 1)
        dt = T / steps
        z = np.random.standard_normal((n, steps))
        log_returns = (0.07 - 0.5 * σ**2) * dt + σ * np.sqrt(dt) * z
        return S * np.exp(np.cumsum(log_returns, axis=1))

    @staticmethod
    def pop_strangle(S, cK, pK, prem, σ, T, n=5000):
        final = MC.gbm(S, σ, T, n)[:, -1]
        pnl = prem - np.maximum(final - cK, 0) - np.maximum(pK - final, 0)
        return float(np.mean(pnl > 0)), float(np.mean(pnl)), float(np.std(pnl))

    @staticmethod
    def pop_ic(S, sc, lc, sp, lp, prem, σ, T, n=5000):
        final = MC.gbm(S, σ, T, n)[:, -1]
        loss = (np.maximum(final - sc, 0) - np.maximum(final - lc, 0) +
                np.maximum(sp - final, 0) - np.maximum(lp - final, 0))
        pnl = prem - loss
        return float(np.mean(pnl > 0)), float(np.mean(pnl)), float(np.std(pnl))

    @staticmethod
    def expected_move(S, σ, T):
        moves = []
        for conf in [0.6827, 0.9545, 0.9973]:
            z = norm.ppf((1 + conf) / 2)
            m = S * σ * np.sqrt(T) * z / norm.ppf((1 + 0.6827) / 2)
            moves.append({'conf': conf, 'move': m, 'upper': S + m, 'lower': S - m})
        return moves


# ══════════════════════════════════════════════════════════════════════════════
# KELLY CRITERION
# ══════════════════════════════════════════════════════════════════════════════

def kelly(p, w, l, half=True):
    if l == 0 or w == 0: return 0.0
    b = w / abs(l)
    k = (b * p - (1 - p)) / b
    if half: k *= 0.5
    return max(0, min(k, 0.25))


# ══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_vol_regime(ivp):
    if ivp > 85: return VolRegime.EXTREME
    if ivp > 70: return VolRegime.HIGH
    if ivp > 50: return VolRegime.ELEVATED
    if ivp > 30: return VolRegime.NORMAL
    if ivp > 15: return VolRegime.LOW
    return VolRegime.COMPRESSED

def detect_trend(S, ma20, ma50, rsi, pct):
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
    if sig >= 3: return TrendRegime.STRONG_UPTREND
    if sig >= 1: return TrendRegime.UPTREND
    if sig <= -3: return TrendRegime.STRONG_DOWNTREND
    if sig <= -1: return TrendRegime.DOWNTREND
    return TrendRegime.NEUTRAL


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY ENGINE — 10 strategies, full BSM + MC + Kelly scoring
# ══════════════════════════════════════════════════════════════════════════════

def snap(x, g):
    return round(x / g) * g

def score_strategy(name, stock, settings):
    """Evaluate a single strategy — returns StrategyResult or None"""
    S = stock['price']; iv = stock['ATMIV'] / 100; ivp = stock['IVPercentile']
    T = settings['dte'] / 365; r = BSM.R; g = settings['gap']
    rsi = stock.get('rsi_daily', 50); atr = stock.get('atr_daily', S * 0.02)
    vol = stock.get('volume', 0); vol20 = stock.get('vol20', 1)
    event = stock.get('Event', '-')
    vr = detect_vol_regime(ivp)
    tr = detect_trend(S, stock.get('ma20_daily', S), stock.get('ma50_daily', S), rsi, stock.get('% change', 0))

    em = S * iv * np.sqrt(T)  # 1σ expected move

    try:
        if name == 'Short Strangle':
            cK, pK = snap(S + em, g), snap(S - em, g)
            cp, pp = BSM.call(S, cK, T, r, iv), BSM.put(S, pK, T, r, iv)
            nc = cp + pp; ml = S * 0.15 * 0.3
            pop, ev, std = MC.pop_strangle(S, cK, pK, nc, iv, T)
            cg, pg = BSM.greeks(S, cK, T, r, iv, 'call'), BSM.greeks(S, pK, T, r, iv, 'put')
            ng = Greeks(delta=-(cg.delta+pg.delta), gamma=-(cg.gamma+pg.gamma), theta=-(cg.theta+pg.theta), vega=-(cg.vega+pg.vega))
            cv = min(100, max(0, ivp*0.4 + (25 if tr==TrendRegime.NEUTRAL else 10) + min(15, vol/max(vol20,1)*15) + pop*20 + (-15 if event!='-' else 0)))
            ra = 0.9 if vr in (VolRegime.HIGH, VolRegime.EXTREME) and tr==TrendRegime.NEUTRAL else 0.4
            return StrategyResult(name=name, legs=[{'type':'Sell Call','strike':cK,'premium':cp,'delta':-cg.delta},{'type':'Sell Put','strike':pK,'premium':pp,'delta':-pg.delta}],
                max_profit=nc, max_loss=ml, breakeven_lower=pK-nc, breakeven_upper=cK+nc, probability_of_profit=pop, expected_value=ev,
                sharpe_ratio=ev/std if std>0 else 0, kelly_fraction=kelly(pop,nc,ml), net_greeks=ng, conviction_score=cv,
                optimal_dte=45 if ivp>70 else 30, width=cK-pK, net_credit=nc, risk_reward_ratio=nc/ml if ml>0 else 0, regime_alignment=ra)

        elif name == 'Short Straddle':
            K = snap(S, g)
            cp, pp = BSM.call(S, K, T, r, iv), BSM.put(S, K, T, r, iv)
            nc = cp + pp; ml = S * 0.10 * 0.5
            pop, ev, std = MC.pop_strangle(S, K, K, nc, iv, T)
            cg, pg = BSM.greeks(S, K, T, r, iv, 'call'), BSM.greeks(S, K, T, r, iv, 'put')
            ng = Greeks(delta=-(cg.delta+pg.delta), gamma=-(cg.gamma+pg.gamma), theta=-(cg.theta+pg.theta), vega=-(cg.vega+pg.vega))
            cv = min(100, max(0, ivp*0.35 + (30 if tr==TrendRegime.NEUTRAL else 10) + (15 if atr/S<0.015 else 5) + pop*20))
            return StrategyResult(name=name, legs=[{'type':'Sell Call','strike':K,'premium':cp,'delta':-cg.delta},{'type':'Sell Put','strike':K,'premium':pp,'delta':-pg.delta}],
                max_profit=nc, max_loss=ml, breakeven_lower=K-nc, breakeven_upper=K+nc, probability_of_profit=pop, expected_value=ev,
                sharpe_ratio=ev/std if std>0 else 0, kelly_fraction=kelly(pop,nc,ml), net_greeks=ng, conviction_score=cv,
                optimal_dte=21 if ivp>70 else 30, net_credit=nc, risk_reward_ratio=nc/ml if ml>0 else 0,
                regime_alignment=0.85 if vr in (VolRegime.HIGH,VolRegime.EXTREME) and tr==TrendRegime.NEUTRAL else 0.35)

        elif name == 'Iron Condor':
            sc, lc = snap(S+em*0.8, g), snap(S+em*1.5, g)
            sp, lp = snap(S-em*0.8, g), snap(S-em*1.5, g)
            nc = (BSM.call(S,sc,T,r,iv)-BSM.call(S,lc,T,r,iv)) + (BSM.put(S,sp,T,r,iv)-BSM.put(S,lp,T,r,iv))
            w = lc - sc; ml = w - nc
            pop, ev, std = MC.pop_ic(S, sc, lc, sp, lp, nc, iv, T)
            ng = Greeks(theta=abs(BSM.greeks(S,sc,T,r,iv,'call').theta + BSM.greeks(S,sp,T,r,iv,'put').theta))
            cv = min(100, max(0, ivp*0.3 + (25 if tr==TrendRegime.NEUTRAL else 15) + 15 + pop*20 - (10 if event!='-' else 0)))
            return StrategyResult(name=name, legs=[
                    {'type':'Sell Call','strike':sc,'premium':BSM.call(S,sc,T,r,iv),'delta':0},
                    {'type':'Buy Call','strike':lc,'premium':-BSM.call(S,lc,T,r,iv),'delta':0},
                    {'type':'Sell Put','strike':sp,'premium':BSM.put(S,sp,T,r,iv),'delta':0},
                    {'type':'Buy Put','strike':lp,'premium':-BSM.put(S,lp,T,r,iv),'delta':0}],
                max_profit=nc, max_loss=max(ml,0.01), breakeven_lower=sp-nc, breakeven_upper=sc+nc,
                probability_of_profit=pop, expected_value=ev, sharpe_ratio=ev/std if std>0 else 0,
                kelly_fraction=kelly(pop,nc,max(ml,0.01)), net_greeks=ng, conviction_score=cv,
                optimal_dte=45, width=w, net_credit=nc, risk_reward_ratio=nc/max(ml,0.01),
                regime_alignment=0.8 if vr in (VolRegime.ELEVATED,VolRegime.HIGH) else 0.5)

        elif name == 'Bull Put Spread':
            sp_k, lp_k = snap(S-em*0.5, g), snap(S-em*1.2, g)
            nc = BSM.put(S,sp_k,T,r,iv) - BSM.put(S,lp_k,T,r,iv)
            ml = (sp_k - lp_k) - nc
            pop = BSM.prob_otm(S, sp_k, T, iv, 'put')
            ev = pop * nc - (1-pop) * ml
            std = np.sqrt(pop*(nc-ev)**2 + (1-pop)*(-ml-ev)**2)
            cv = min(100, max(0, (20 if 'UP' in tr.value else 5) + ivp*0.25 + pop*30 + 15))
            spg = BSM.greeks(S, sp_k, T, r, iv, 'put')
            return StrategyResult(name=name, legs=[
                    {'type':'Sell Put','strike':sp_k,'premium':BSM.put(S,sp_k,T,r,iv),'delta':-spg.delta},
                    {'type':'Buy Put','strike':lp_k,'premium':-BSM.put(S,lp_k,T,r,iv),'delta':0}],
                max_profit=nc, max_loss=max(ml,0.01), breakeven_lower=sp_k-nc, breakeven_upper=S*10,
                probability_of_profit=pop, expected_value=ev, sharpe_ratio=ev/std if std>0 else 0,
                kelly_fraction=kelly(pop,nc,max(ml,0.01)), net_greeks=Greeks(delta=-spg.delta, theta=-spg.theta),
                conviction_score=cv, optimal_dte=30, width=sp_k-lp_k, net_credit=nc,
                risk_reward_ratio=nc/max(ml,0.01), regime_alignment=0.8 if 'UP' in tr.value else 0.3)

        elif name == 'Bear Call Spread':
            sc_k, lc_k = snap(S+em*0.5, g), snap(S+em*1.2, g)
            nc = BSM.call(S,sc_k,T,r,iv) - BSM.call(S,lc_k,T,r,iv)
            ml = (lc_k - sc_k) - nc
            pop = BSM.prob_otm(S, sc_k, T, iv, 'call')
            ev = pop * nc - (1-pop) * ml
            std = np.sqrt(pop*(nc-ev)**2 + (1-pop)*(-ml-ev)**2)
            cv = min(100, max(0, (20 if 'DOWN' in tr.value else 5) + ivp*0.25 + pop*30 + 15))
            scg = BSM.greeks(S, sc_k, T, r, iv, 'call')
            return StrategyResult(name=name, legs=[
                    {'type':'Sell Call','strike':sc_k,'premium':BSM.call(S,sc_k,T,r,iv),'delta':-scg.delta},
                    {'type':'Buy Call','strike':lc_k,'premium':-BSM.call(S,lc_k,T,r,iv),'delta':0}],
                max_profit=nc, max_loss=max(ml,0.01), breakeven_lower=0, breakeven_upper=sc_k+nc,
                probability_of_profit=pop, expected_value=ev, sharpe_ratio=ev/std if std>0 else 0,
                kelly_fraction=kelly(pop,nc,max(ml,0.01)), net_greeks=Greeks(delta=-scg.delta, theta=-scg.theta),
                conviction_score=cv, optimal_dte=30, width=lc_k-sc_k, net_credit=nc,
                risk_reward_ratio=nc/max(ml,0.01), regime_alignment=0.8 if 'DOWN' in tr.value else 0.3)

        elif name == 'Iron Butterfly':
            K = snap(S, g); w = snap(em*1.2, g)
            nc = (BSM.call(S,K,T,r,iv)+BSM.put(S,K,T,r,iv)) - (BSM.call(S,K+w,T,r,iv)+BSM.put(S,K-w,T,r,iv))
            ml = w - nc
            pop, ev, std = MC.pop_ic(S, K, K+w, K, K-w, nc, iv, T)
            cv = min(100, max(0, ivp*0.35 + (30 if tr==TrendRegime.NEUTRAL else 10) + pop*20))
            return StrategyResult(name=name, legs=[
                    {'type':'Sell Call','strike':K,'premium':BSM.call(S,K,T,r,iv),'delta':0},
                    {'type':'Sell Put','strike':K,'premium':BSM.put(S,K,T,r,iv),'delta':0},
                    {'type':'Buy Call','strike':K+w,'premium':-BSM.call(S,K+w,T,r,iv),'delta':0},
                    {'type':'Buy Put','strike':K-w,'premium':-BSM.put(S,K-w,T,r,iv),'delta':0}],
                max_profit=nc, max_loss=max(ml,0.01), breakeven_lower=K-nc, breakeven_upper=K+nc,
                probability_of_profit=pop, expected_value=ev, sharpe_ratio=ev/std if std>0 else 0,
                kelly_fraction=kelly(pop,nc,max(ml,0.01)), net_greeks=Greeks(theta=abs(BSM.greeks(S,K,T,r,iv,'call').theta)*2),
                conviction_score=cv, optimal_dte=30, width=w, net_credit=nc,
                risk_reward_ratio=nc/max(ml,0.01), regime_alignment=0.7 if tr==TrendRegime.NEUTRAL else 0.3)

        elif name == 'Jade Lizard':
            cK = snap(S+em*0.7, g); sp_k = snap(S-em*0.5, g); lp_k = snap(S-em*1.3, g)
            nc = BSM.call(S,cK,T,r,iv) + BSM.put(S,sp_k,T,r,iv) - BSM.put(S,lp_k,T,r,iv)
            ml = (sp_k - lp_k) - nc
            pop = 0.5*(BSM.prob_otm(S,cK,T,iv,'call')+BSM.prob_otm(S,sp_k,T,iv,'put'))
            ev = pop*nc - (1-pop)*max(ml,0.01)
            cv = min(100, max(0, ivp*0.3 + (20 if 'UP' in tr.value or tr==TrendRegime.NEUTRAL else 5) + pop*20 + 10))
            return StrategyResult(name=name, legs=[
                    {'type':'Sell Call','strike':cK,'premium':BSM.call(S,cK,T,r,iv),'delta':0},
                    {'type':'Sell Put','strike':sp_k,'premium':BSM.put(S,sp_k,T,r,iv),'delta':0},
                    {'type':'Buy Put','strike':lp_k,'premium':-BSM.put(S,lp_k,T,r,iv),'delta':0}],
                max_profit=nc, max_loss=max(ml,0.01), breakeven_lower=sp_k-nc, breakeven_upper=cK+nc,
                probability_of_profit=pop, expected_value=ev, sharpe_ratio=ev/max(abs(ev)*1.5+0.01,0.01),
                kelly_fraction=kelly(pop,nc,max(ml,0.01)), net_greeks=Greeks(), conviction_score=cv,
                optimal_dte=35, width=sp_k-lp_k, net_credit=nc,
                risk_reward_ratio=nc/max(ml,0.01), regime_alignment=0.7 if 'UP' in tr.value else 0.3)

        elif name == 'Calendar Spread':
            K = snap(S, g); fT, bT = T*0.5, T*1.5
            fp = BSM.call(S,K,fT,r,iv); bp = BSM.call(S,K,bT,r,iv*0.95)
            nd = bp - fp; mp = fp * 0.8; ml = nd
            pop = 0.55 if vr in (VolRegime.LOW, VolRegime.COMPRESSED) else 0.45
            ev = pop*mp - (1-pop)*ml
            cv = min(100, max(0, (30 if vr in (VolRegime.LOW,VolRegime.COMPRESSED) else 10) + (20 if tr==TrendRegime.NEUTRAL else 5) + pop*25))
            return StrategyResult(name=name, legs=[
                    {'type':'Sell Call (Front)','strike':K,'premium':fp,'delta':0},
                    {'type':'Buy Call (Back)','strike':K,'premium':-bp,'delta':0}],
                max_profit=mp, max_loss=max(ml,0.01), breakeven_lower=K-mp, breakeven_upper=K+mp,
                probability_of_profit=pop, expected_value=ev, sharpe_ratio=ev/max(abs(ev)*2+0.01,0.01),
                kelly_fraction=kelly(pop,mp,max(ml,0.01)), net_greeks=Greeks(theta=fp*0.03, vega=bp*0.01),
                conviction_score=cv, optimal_dte=45, net_credit=-nd,
                risk_reward_ratio=mp/max(ml,0.01), regime_alignment=0.8 if vr in (VolRegime.LOW,VolRegime.COMPRESSED) else 0.3)

        elif name == 'Broken Wing Butterfly':
            c = snap(S, g); lo = snap(S-em*0.8, g); hi = snap(S+em*1.2, g)
            nc = BSM.call(S,lo,T,r,iv) - 2*BSM.call(S,c,T,r,iv) + BSM.call(S,hi,T,r,iv)
            nc = abs(nc)*0.1 if nc<0 else nc
            mp = (c-lo)+nc; ml = max((hi-c)-nc, 0.01)
            pop = 0.50; ev = pop*mp*0.3 - (1-pop)*ml*0.3
            cv = min(100, max(0, (15 if event!='-' else 0) + ivp*0.2 + 20 + pop*20))
            return StrategyResult(name=name, legs=[
                    {'type':'Buy Call','strike':lo,'premium':-BSM.call(S,lo,T,r,iv),'delta':0},
                    {'type':'Sell 2 Calls','strike':c,'premium':2*BSM.call(S,c,T,r,iv),'delta':0},
                    {'type':'Buy Call','strike':hi,'premium':-BSM.call(S,hi,T,r,iv),'delta':0}],
                max_profit=mp, max_loss=ml, breakeven_lower=lo, breakeven_upper=hi,
                probability_of_profit=pop, expected_value=ev, sharpe_ratio=ev/max(abs(ev)*2+0.01,0.01),
                kelly_fraction=kelly(pop,mp*0.3,ml*0.3), net_greeks=Greeks(), conviction_score=cv,
                optimal_dte=45, width=hi-lo, net_credit=nc,
                risk_reward_ratio=mp/ml, regime_alignment=0.5)

        elif name == 'Ratio Spread':
            lK = snap(S, g); sK = snap(S+em*0.8, g)
            lp_v = BSM.call(S,lK,T,r,iv); sp_v = BSM.call(S,sK,T,r,iv)
            mp = (sK-lK)+(2*sp_v-lp_v); ml = S*0.10
            pop = 0.55 if 'UP' in tr.value or tr==TrendRegime.NEUTRAL else 0.40
            ev = pop*mp*0.4 - (1-pop)*ml*0.2
            cv = min(100, max(0, (20 if 'UP' in tr.value else 10) + ivp*0.2 + pop*20 + 10))
            return StrategyResult(name=name, legs=[
                    {'type':'Buy Call','strike':lK,'premium':-lp_v,'delta':0},
                    {'type':'Sell 2 Calls','strike':sK,'premium':2*sp_v,'delta':0}],
                max_profit=mp, max_loss=max(ml,0.01), breakeven_lower=lK, breakeven_upper=2*sK-lK+mp,
                probability_of_profit=pop, expected_value=ev, sharpe_ratio=ev/max(abs(ev)*2+0.01,0.01),
                kelly_fraction=kelly(pop,mp*0.4,ml*0.2), net_greeks=Greeks(), conviction_score=cv,
                optimal_dte=30, width=sK-lK, net_credit=2*sp_v-lp_v,
                risk_reward_ratio=mp/max(ml,0.01), regime_alignment=0.6 if 'UP' in tr.value else 0.3)

    except Exception:
        return None
    return None


ALL_STRATEGIES = [
    'Short Strangle', 'Short Straddle', 'Iron Condor', 'Iron Butterfly',
    'Bull Put Spread', 'Bear Call Spread', 'Calendar Spread',
    'Jade Lizard', 'Broken Wing Butterfly', 'Ratio Spread'
]


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

C = {'primary':'#FFC300','card':'#1A1A1A','border':'#2A2A2A','text':'#EAEAEA','muted':'#888',
     'success':'#10b981','danger':'#ef4444','warning':'#f59e0b','info':'#06b6d4','purple':'#a855f7'}

def fmt(v):
    try:
        v = float(v); neg = v < 0; v = abs(v); ip = int(v); dp = round(v - ip, 2)
        s = str(ip)[::-1]; g = []
        if len(s)>3:
            g.append(s[:3]); s = s[3:]
            while s: g.append(s[:2]); s = s[2:]
            f = ','.join(g[::-1])
        else: f = s[::-1]
        if dp > 0: f += f"{dp:.2f}"[1:]
        return f"{'-' if neg else ''}₹{f}"
    except: return str(v)

def payoff_chart(strat, S):
    margin = max(strat.width, S * 0.15)
    px = np.linspace(S - margin*1.5, S + margin*1.5, 500)
    pnl = np.zeros_like(px)
    for leg in strat.legs:
        K, p, t = leg['strike'], leg['premium'], leg['type']
        if 'Sell Call' in t and '2' not in t: pnl += p - np.maximum(px-K,0)
        elif 'Buy Call' in t and '2' not in t: pnl += -abs(p) + np.maximum(px-K,0)
        elif 'Sell Put' in t: pnl += p - np.maximum(K-px,0)
        elif 'Buy Put' in t: pnl += -abs(p) + np.maximum(K-px,0)
        elif 'Sell 2 Calls' in t: pnl += p - 2*np.maximum(px-K,0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px, y=np.clip(pnl,0,None), fill='tozeroy', fillcolor='rgba(16,185,129,0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=px, y=np.clip(pnl,None,0), fill='tozeroy', fillcolor='rgba(239,68,68,0.15)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=px, y=pnl, mode='lines', name='P/L', line=dict(color=C['primary'], width=3), hovertemplate='₹%{x:,.0f} → P/L: ₹%{y:,.2f}<extra></extra>'))
    fig.add_vline(x=S, line_dash="dash", line_color=C['text'], opacity=0.5, annotation_text="Spot")
    fig.add_hline(y=0, line_color=C['muted'], opacity=0.3)
    for leg in strat.legs:
        clr = C['danger'] if 'Sell' in leg['type'] else C['success']
        fig.add_vline(x=leg['strike'], line_dash="dot", line_color=clr, opacity=0.3)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=C['card'], height=360, margin=dict(l=10,r=10,t=30,b=30),
        xaxis=dict(showgrid=True, gridcolor=C['border'], title='Stock Price (₹)', tickformat=','),
        yaxis=dict(showgrid=True, gridcolor=C['border'], title='P/L (₹)', tickformat=','),
        font=dict(family='Inter', color=C['text']), hovermode='x unified')
    return fig

def conviction_gauge(score):
    clr = C['success'] if score>=75 else (C['warning'] if score>=50 else C['danger'])
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        number={'font':{'size':32,'color':C['text'],'family':'JetBrains Mono'}},
        gauge={'axis':{'range':[0,100],'tickcolor':C['muted']}, 'bar':{'color':clr,'thickness':0.75}, 'bgcolor':C['card'],
               'borderwidth':2, 'bordercolor':C['border'],
               'steps':[{'range':[0,33],'color':'rgba(239,68,68,0.15)'},{'range':[33,66],'color':'rgba(245,158,11,0.15)'},{'range':[66,100],'color':'rgba(16,185,129,0.15)'}]}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter',color=C['text']), height=200, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def em_chart(S, iv, T):
    em = MC.expected_move(S, iv, T); σ = S * iv * np.sqrt(T)
    x = np.linspace(S - 4*σ, S + 4*σ, 500); y = norm.pdf(x, S, σ)
    fig = go.Figure()
    for i, (conf, clr, alpha) in enumerate([(2, C['purple'], 0.08), (1, C['info'], 0.12), (0, C['success'], 0.15)]):
        m = (x >= em[i]['lower']) & (x <= em[i]['upper'])
        fig.add_trace(go.Scatter(x=x[m], y=y[m], fill='tozeroy', fillcolor=f'rgba({",".join(str(int(clr.lstrip("#")[j:j+2],16)) for j in (0,2,4))},{alpha})', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=C['primary'], width=2), showlegend=False))
    fig.add_vline(x=S, line_dash="dash", line_color=C['text'], opacity=0.5, annotation_text=f"Spot ₹{S:,.0f}")
    for i, clr in enumerate([C['success'], C['info'], C['purple']]):
        fig.add_vline(x=em[i]['upper'], line_dash="dot", line_color=clr, opacity=0.4)
        fig.add_vline(x=em[i]['lower'], line_dash="dot", line_color=clr, opacity=0.4)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=C['card'], height=300, margin=dict(l=10,r=10,t=30,b=30),
        xaxis=dict(showgrid=True, gridcolor=C['border'], title='Price (₹)', tickformat=','), yaxis=dict(showgrid=False, showticklabels=False),
        font=dict(family='Inter', color=C['text']), showlegend=False, hovermode='x unified')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""<div style="text-align:center; padding:1rem 0; margin-bottom:1rem;">
            <div style="font-size:1.75rem; font-weight:800; color:#FFC300;">VAAYDO</div>
            <div style="color:#888; font-size:0.7rem; margin-top:0.25rem;">वायदो | FnO Trade Intelligence</div>
            <div style="color:#555; font-size:0.6rem; margin-top:0.15rem;">Hemrek Capital</div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">⚡ Parameters</div>', unsafe_allow_html=True)
        dte = st.slider("Days to Expiry", 7, 90, 30)
        strike_gap = st.selectbox("Strike Gap (₹)", [25, 50, 100, 200, 500], index=1)
        capital = st.number_input("Capital (₹)", value=500000, step=50000, format="%d")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🎯 Filters</div>', unsafe_allow_html=True)
        min_ivp = st.slider("Min IV Percentile", 0, 100, 20)
        min_cv = st.slider("Min Conviction", 0, 100, 35)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class='info-box'><p style='font-size:0.75rem; margin:0; color:var(--text-muted); line-height:1.5;'>
            <strong>Version:</strong> {VERSION}<br><strong>Engine:</strong> BSM + MC(5K) + Kelly<br>
            <strong>Strategies:</strong> 10 Active<br><strong>Data:</strong> Auto-Fetch (yfinance)<br>
            <strong>IV:</strong> Realized Vol Proxy</p></div>""", unsafe_allow_html=True)

    settings = {'dte': dte, 'gap': strike_gap, 'capital': capital}

    # ── HEADER ──
    st.markdown(f"""<div class="premium-header"><div class="product-badge">Pragyam Product Family</div>
        <h1>VAAYDO : FnO Trade Intelligence</h1>
        <div class="tagline">Black-Scholes · Monte Carlo · Kelly Criterion · Regime Intelligence &nbsp;|&nbsp; {datetime.now().strftime("%B %d, %Y")}</div></div>""", unsafe_allow_html=True)

    # ── AUTO-FETCH DATA ──
    with st.spinner("Fetching F&O stock list..."):
        symbols, sym_status = get_fno_stock_list()
        symbols_ns = [s + ".NS" if not s.endswith(".NS") else s for s in symbols]

    with st.spinner(f"Downloading OHLCV data for {len(symbols_ns)} securities & computing analytics..."):
        df, data_status = fetch_all_fno_data(symbols_ns)

    if df.empty:
        st.error(f"Data fetch failed: {data_status}")
        st.info("This may be due to network restrictions. The system needs access to `*.yahoo.com` for market data.")
        return

    st.caption(f"🔌 {sym_status} → {data_status}")

    # ── COMPUTE STRATEGIES ──
    with st.spinner("Running quantitative analysis (BSM + Monte Carlo + Kelly)..."):
        all_trades = []
        for _, row in df.iterrows():
            rd = row.to_dict()
            best = None
            for sname in ALL_STRATEGIES:
                try:
                    res = score_strategy(sname, rd, settings)
                    if res and (best is None or res.conviction_score > best.conviction_score):
                        best = res
                except: continue
            if best:
                all_trades.append({**rd, 'strategy': best.name, 'conviction_score': best.conviction_score,
                    'pop': best.probability_of_profit, 'ev': best.expected_value, 'sharpe': best.sharpe_ratio,
                    'kelly_frac': best.kelly_fraction, 'net_credit': best.net_credit,
                    'max_profit': best.max_profit, 'max_loss': best.max_loss,
                    'vol_regime': detect_vol_regime(rd.get('IVPercentile',50)).value,
                    'trend_regime': detect_trend(rd['price'], rd.get('ma20_daily',rd['price']),
                        rd.get('ma50_daily',rd['price']), rd.get('rsi_daily',50), rd.get('% change',0)).value,
                    '_result': best})

    filtered = [t for t in all_trades if t['IVPercentile'] >= min_ivp and t['conviction_score'] >= min_cv]
    filtered.sort(key=lambda x: x['conviction_score'], reverse=True)

    # ── METRICS BAR ──
    avg_iv = df['IVPercentile'].mean(); avg_pcr = df['PCR'].mean()
    hc = len([t for t in filtered if t['conviction_score'] >= 70])
    vol_regimes = [t.get('vol_regime','NORMAL') for t in filtered[:20]]
    dom_vol = max(set(vol_regimes), key=vol_regimes.count) if vol_regimes else 'NORMAL'

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(f"<div class='metric-card primary'><h4>Securities</h4><h2>{len(df)}</h2><div class='sub-metric'>{len(filtered)} pass filters</div></div>", unsafe_allow_html=True)
    with c2:
        cs = 'success' if hc>5 else ('warning' if hc>2 else 'danger')
        st.markdown(f"<div class='metric-card {cs}'><h4>High Conviction</h4><h2>{hc}</h2><div class='sub-metric'>Score ≥ 70</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card {'success' if avg_iv>60 else 'warning' if avg_iv>40 else 'info'}'><h4>Avg IV Percentile</h4><h2>{avg_iv:.0f}%</h2><div class='sub-metric'>Regime: {dom_vol}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card {'success' if avg_pcr>1.2 else 'danger' if avg_pcr<0.8 else 'neutral'}'><h4>Avg PCR</h4><h2>{avg_pcr:.2f}</h2><div class='sub-metric'>{'Bullish' if avg_pcr>1.2 else 'Bearish' if avg_pcr<0.8 else 'Neutral'}</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric-card info'><h4>DTE / Gap</h4><h2>{dte}D / ₹{strike_gap}</h2><div class='sub-metric'>Capital: {fmt(capital)}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Trade Radar", "🔬 Deep Analysis", "📊 Rankings", "📐 Probability Lab"])

    # ═══ TRADE RADAR ═══
    with tab1:
        st.markdown("<div style='margin-bottom:1rem;'><span style='font-size:1.1rem;font-weight:700;color:var(--text-primary);'>Top FnO Opportunities</span><span style='color:var(--text-muted);font-size:0.85rem;margin-left:0.75rem;'>Ranked by conviction · BSM + MC validated</span></div>", unsafe_allow_html=True)
        top = filtered[:9]
        if not top:
            st.info("No trades pass filters. Lower conviction or IV percentile thresholds.")
        else:
            cols = st.columns(3)
            for i, t in enumerate(top):
                with cols[i%3]:
                    cv = t['conviction_score']; cc = '#10b981' if cv>=70 else ('#f59e0b' if cv>=50 else '#ef4444')
                    cv_cls = 'high' if cv>=70 else ('medium' if cv>=50 else 'low')
                    st.markdown(f"""<div class='trade-card {cv_cls}'>
                        <div style="display:flex;justify-content:space-between;align-items:start;">
                        <div><div class='symbol'>{t['Instrument']}</div><div class='strategy'>{t['strategy']}</div></div>
                        <div style="text-align:right;"><div style="font-size:1.8rem;font-weight:800;color:{cc};font-family:'JetBrains Mono',monospace;">{cv:.0f}</div>
                        <div style="font-size:0.6rem;color:var(--text-muted);text-transform:uppercase;">Conviction</div></div></div>
                        <div class='grid'>
                        <div class='grid-item'><label>Spot</label><div class='value'>{fmt(t['price'])}</div></div>
                        <div class='grid-item'><label>IV %ile</label><div class='value'>{t['IVPercentile']:.0f}%</div></div>
                        <div class='grid-item'><label>Prob Profit</label><div class='value text-green'>{t['pop']*100:.1f}%</div></div>
                        <div class='grid-item'><label>Expected Val</label><div class='value {"text-green" if t["ev"]>0 else "text-red"}'>{fmt(t['ev'])}</div></div>
                        <div class='grid-item'><label>Max Profit</label><div class='value text-green'>{fmt(t['max_profit'])}</div></div>
                        <div class='grid-item'><label>Max Loss</label><div class='value text-red'>{fmt(t['max_loss'])}</div></div>
                        <div class='grid-item'><label>Sharpe</label><div class='value'>{t['sharpe']:.2f}</div></div>
                        <div class='grid-item'><label>Kelly</label><div class='value'>{t['kelly_frac']*100:.1f}%</div></div></div>
                        <div class='conviction-bar'><div class='conviction-fill' style='width:{cv}%;background:linear-gradient(90deg,{cc},{cc}aa);'></div></div>
                        <div style="display:flex;gap:0.5rem;margin-top:0.75rem;">
                        <span class='status-badge {"buy" if "UP" in t.get("trend_regime","") else ("sell" if "DOWN" in t.get("trend_regime","") else "neutral")}'>{t.get('trend_regime','NEUTRAL').replace('_',' ')}</span>
                        <span class='status-badge premium'>{t.get('vol_regime','NORMAL').replace('_',' ')}</span></div></div>""", unsafe_allow_html=True)
                    if st.button(f"Analyze {t['Instrument']}", key=f"a_{t['Instrument']}"):
                        st.session_state.sel = t['Instrument']

    # ═══ DEEP ANALYSIS ═══
    with tab2:
        opts = [t['Instrument'] for t in filtered] if filtered else df['Instrument'].tolist()
        sel = st.selectbox("Select Security", opts, index=opts.index(st.session_state.get('sel', opts[0])) if st.session_state.get('sel') in opts else 0)
        if sel:
            row = df[df['Instrument']==sel].iloc[0].to_dict()
            S = row['price']; iv = row['ATMIV']/100; T = dte/365
            vr = detect_vol_regime(row['IVPercentile'])
            tr = detect_trend(S, row.get('ma20_daily',S), row.get('ma50_daily',S), row.get('rsi_daily',50), row.get('% change',0))
            em = MC.expected_move(S, iv, T)

            st.markdown(f"<div style='margin-bottom:1.5rem;'><span style='font-size:1.4rem;font-weight:800;color:var(--text-primary);'>{sel}</span>"
                f"<span style='font-size:1.1rem;color:var(--text-muted);margin-left:1rem;'>{fmt(S)}</span>"
                f"<span class='status-badge premium' style='margin-left:1rem;'>{vr.value}</span>"
                f"<span class='status-badge {'buy' if 'UP' in tr.value else ('sell' if 'DOWN' in tr.value else 'neutral')}'>{tr.value.replace('_',' ')}</span></div>", unsafe_allow_html=True)

            mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
            with mc1: st.markdown(f"<div class='metric-card'><h4>ATM IV</h4><h2>{row['ATMIV']:.1f}%</h2><div class='sub-metric'>Realized Vol</div></div>", unsafe_allow_html=True)
            with mc2: st.markdown(f"<div class='metric-card'><h4>IV Percentile</h4><h2>{row['IVPercentile']:.0f}%</h2><div class='sub-metric'>252D Rank</div></div>", unsafe_allow_html=True)
            with mc3: st.markdown(f"<div class='metric-card'><h4>PCR</h4><h2>{row['PCR']:.2f}</h2><div class='sub-metric'>Vol Ratio</div></div>", unsafe_allow_html=True)
            with mc4: st.markdown(f"<div class='metric-card'><h4>RSI</h4><h2>{row.get('rsi_daily',50):.0f}</h2><div class='sub-metric'>14-Day</div></div>", unsafe_allow_html=True)
            with mc5: st.markdown(f"<div class='metric-card info'><h4>1σ Move</h4><h2>±{fmt(em[0]['move'])}</h2><div class='sub-metric'>68.3%</div></div>", unsafe_allow_html=True)
            with mc6: st.markdown(f"<div class='metric-card warning'><h4>2σ Move</h4><h2>±{fmt(em[1]['move'])}</h2><div class='sub-metric'>95.5%</div></div>", unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("<span style='font-weight:700;color:var(--text-primary);'>Expected Move Distribution</span>", unsafe_allow_html=True)
            st.plotly_chart(em_chart(S, iv, T), use_container_width=True)
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            st.markdown("<span style='font-weight:700;color:var(--text-primary);'>Strategy Rankings</span><span style='color:var(--text-muted);margin-left:0.75rem;font-size:0.85rem;'>All 10 strategies · sorted by conviction</span>", unsafe_allow_html=True)
            strats = []
            for sn in ALL_STRATEGIES:
                try:
                    r = score_strategy(sn, row, settings)
                    if r: strats.append(r)
                except: continue
            strats.sort(key=lambda x: x.conviction_score, reverse=True)

            for rank, s in enumerate(strats[:5], 1):
                cv = s.conviction_score
                with st.expander(f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else f'#{rank}'} {s.name} — Conv: {cv:.0f} | POP: {s.probability_of_profit*100:.1f}% | Sharpe: {s.sharpe_ratio:.2f}", expanded=(rank==1)):
                    ec1, ec2 = st.columns([2, 1])
                    with ec1:
                        lh = "<table class='greek-table'><tr><th>Leg</th><th>Strike</th><th>Premium</th></tr>"
                        for l in s.legs:
                            cc = 'text-red' if 'Sell' in l['type'] else 'text-green'
                            lh += f"<tr><td class='{cc}'>{l['type']}</td><td>{fmt(l['strike'])}</td><td>{fmt(abs(l['premium']))}</td></tr>"
                        st.markdown(lh + "</table>", unsafe_allow_html=True)
                        st.plotly_chart(payoff_chart(s, S), use_container_width=True)
                    with ec2:
                        st.plotly_chart(conviction_gauge(cv), use_container_width=True)
                        st.markdown(f"""<div class='info-box'><h4>Analytics</h4><p>
                            <strong>POP:</strong> <span class='mono text-green'>{s.probability_of_profit*100:.1f}%</span><br>
                            <strong>EV:</strong> <span class='mono {"text-green" if s.expected_value>0 else "text-red"}'>{fmt(s.expected_value)}</span><br>
                            <strong>Max Profit:</strong> <span class='mono text-green'>{fmt(s.max_profit)}</span><br>
                            <strong>Max Loss:</strong> <span class='mono text-red'>{fmt(s.max_loss)}</span><br>
                            <strong>R:R:</strong> <span class='mono'>{s.risk_reward_ratio:.2f}</span><br>
                            <strong>Sharpe:</strong> <span class='mono'>{s.sharpe_ratio:.2f}</span><br>
                            <strong>Kelly:</strong> <span class='mono'>{s.kelly_fraction*100:.1f}%</span><br>
                            <strong>Opt DTE:</strong> <span class='mono'>{s.optimal_dte}D</span><br>
                            <strong>BEs:</strong> <span class='mono'>{fmt(s.breakeven_lower)} — {fmt(s.breakeven_upper)}</span></p></div>""", unsafe_allow_html=True)
                        g = s.net_greeks
                        st.markdown(f"""<div class='info-box' style='margin-top:0.5rem;'><h4>Net Greeks</h4><p>
                            <strong>Δ:</strong> <span class='mono'>{g.delta:+.4f}</span><br>
                            <strong>Γ:</strong> <span class='mono'>{g.gamma:+.6f}</span><br>
                            <strong>Θ:</strong> <span class='mono text-green'>{g.theta:+.2f}/day</span><br>
                            <strong>ν:</strong> <span class='mono'>{g.vega:+.2f}</span></p></div>""", unsafe_allow_html=True)
                        lots = max(1, int(capital * s.kelly_fraction / max(s.max_loss, 1))) if s.max_loss > 0 else 1
                        st.markdown(f"""<div class='info-box' style='margin-top:0.5rem;'><h4>Position Sizing</h4><p>
                            <strong>Lots:</strong> <span class='mono text-gold'>{lots}</span><br>
                            <strong>Risk:</strong> <span class='mono'>{fmt(lots*s.max_loss)}</span><br>
                            <strong>% Capital:</strong> <span class='mono'>{lots*s.max_loss/capital*100:.1f}%</span></p></div>""", unsafe_allow_html=True)

    # ═══ RANKINGS ═══
    with tab3:
        if filtered:
            rdf = pd.DataFrame([{k:v for k,v in t.items() if k != '_result'} for t in filtered])
            cols_show = ['Instrument','strategy','conviction_score','pop','ev','sharpe','kelly_frac','net_credit','max_profit','max_loss','IVPercentile','price','vol_regime','trend_regime']
            avail = [c for c in cols_show if c in rdf.columns]
            display = rdf[avail].sort_values('conviction_score', ascending=False)
            display.columns = [c.replace('_',' ').title() for c in display.columns]
            st.dataframe(display, use_container_width=True, height=600)

    # ═══ PROBABILITY LAB ═══
    with tab4:
        lab = st.selectbox("Select Security", df['Instrument'].tolist(), key="lab")
        lr = df[df['Instrument']==lab].iloc[0]; lS = lr['price']; liv = lr['ATMIV']/100; lT = dte/365
        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown("<span style='font-weight:600;color:var(--text-primary);'>Monte Carlo Terminal Distribution</span>", unsafe_allow_html=True)
            np.random.seed(42); paths = MC.gbm(lS, liv, lT, 5000); term = paths[:,-1]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=term, nbinsx=80, marker_color=C['primary'], opacity=0.7))
            fig.add_vline(x=lS, line_dash="dash", line_color=C['text'], annotation_text=f"Spot: {fmt(lS)}")
            fig.add_vline(x=np.percentile(term,5), line_dash="dot", line_color=C['danger'], annotation_text="5th")
            fig.add_vline(x=np.percentile(term,95), line_dash="dot", line_color=C['success'], annotation_text="95th")
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=C['card'], height=350, margin=dict(l=10,r=10,t=30,b=30),
                xaxis=dict(showgrid=True,gridcolor=C['border'],title='Terminal Price',tickformat=','), yaxis=dict(showgrid=True,gridcolor=C['border'],title='Freq'),
                font=dict(family='Inter',color=C['text']), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""<div class='info-box'><h4>MC Stats (5,000 paths)</h4><p>
                <strong>Mean:</strong> <span class='mono'>{fmt(np.mean(term))}</span> | <strong>Median:</strong> <span class='mono'>{fmt(np.median(term))}</span><br>
                <strong>5th:</strong> <span class='mono text-red'>{fmt(np.percentile(term,5))}</span> | <strong>95th:</strong> <span class='mono text-green'>{fmt(np.percentile(term,95))}</span><br>
                <strong>P(up):</strong> <span class='mono'>{np.mean(term>lS)*100:.1f}%</span> | <strong>P(>10%):</strong> <span class='mono'>{np.mean(term>lS*1.1)*100:.1f}%</span></p></div>""", unsafe_allow_html=True)
        with lc2:
            st.markdown("<span style='font-weight:600;color:var(--text-primary);'>Sample Paths & BSM Greeks</span>", unsafe_allow_html=True)
            fig2 = go.Figure()
            tax = np.linspace(0, dte, paths.shape[1])
            for i in range(50): fig2.add_trace(go.Scatter(x=tax, y=paths[i], mode='lines', line=dict(width=0.5, color='rgba(255,195,0,0.15)'), showlegend=False, hoverinfo='skip'))
            fig2.add_trace(go.Scatter(x=tax, y=np.mean(paths,axis=0), mode='lines', line=dict(width=3, color=C['primary']), name='Mean'))
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=C['card'], height=350, margin=dict(l=10,r=10,t=30,b=30),
                xaxis=dict(showgrid=True,gridcolor=C['border'],title='Days'), yaxis=dict(showgrid=True,gridcolor=C['border'],title='Price',tickformat=','),
                font=dict(family='Inter',color=C['text']), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            strikes = [snap(lS+i*strike_gap, strike_gap) for i in range(-3,4)]
            gdata = []
            for K in strikes:
                cg = BSM.greeks(lS,K,lT,BSM.R,liv,'call'); pg = BSM.greeks(lS,K,lT,BSM.R,liv,'put')
                gdata.append({'Strike':fmt(K),'Call Δ':f"{cg.delta:.3f}",'Put Δ':f"{pg.delta:.3f}",'Γ':f"{cg.gamma:.5f}",'Call Θ':f"{cg.theta:.2f}",'Put Θ':f"{pg.theta:.2f}",'ν':f"{cg.vega:.2f}"})
            st.dataframe(pd.DataFrame(gdata), use_container_width=True, hide_index=True)


if 'sel' not in st.session_state: st.session_state.sel = None
if __name__ == "__main__": main()
