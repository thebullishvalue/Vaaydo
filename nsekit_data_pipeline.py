"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  VAAYDO — NseKit Data Pipeline v1.0                                    ║
    ║  Open Source Data Engine: Quotes, History, Option Chains               ║
    ║  Hemrek Capital                                                        ║
    ╚══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
import logging
import hashlib
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
from NseKit import Nse, NseConfig
from joblib import Parallel, delayed
import diskcache
import yfinance as yf

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# § Configuration
NseConfig.max_rps = 2.0
NseConfig.retries = 3

@dataclass
class OptionQuote:
    """Rich option quote with Greeks and Liquidity mapping."""
    instrument_token: int
    tradingsymbol: str
    strike: float
    option_type: str  # CE/PE
    expiry: date
    last_price: float
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    volume: int
    oi: int
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    total_buy_qty: int = 0
    total_sell_qty: int = 0
    bid_ask_spread: float = 0.0
    spread_pct: float = 0.0
    liquidity_score: float = 0.0
    is_liquid: bool = False

@dataclass
class OptionChain:
    symbol: str
    underlying_price: float
    expiry: date
    calls: List[OptionQuote]
    puts: List[OptionQuote]
    timestamp: datetime

class NsekitDataPipeline:
    """
    NSE-direct data pipeline using NseKit.
    Replaces Zerodha Kite Connect as the primary data source.
    """
    
    def __init__(self):
        self.nse = Nse()
        self._lot_sizes = {
            "NIFTY": 50, "BANKNIFTY": 15, "FINNIFTY": 40,
            "RELIANCE": 250, "TCS": 175, "INFY": 400, "HDFCBANK": 550, "ICICIBANK": 700
        }
        # § Persistence Layer
        cache_dir = os.path.join(os.path.expanduser("~"), ".vaaydo_cache")
        self.cache = diskcache.Cache(cache_dir)

    def initialize(self):
        """Pre-fetch or warm up local indices."""
        try:
            status = self.nse.nse_market_status()
            return f"✓ Pipeline Active: NseKit initialized. Market Status: {status}"
        except Exception as e:
            return f"⚠ Pipeline Warning: Connectivity issue. {str(e)}"

    def is_connected(self):
        return True # NseKit is stateless

    def get_lot_size(self, symbol: str) -> int:
        # Heuristic for NSE FnO lot sizes (mostly correct for indices and major stocks)
        if symbol == "NIFTY": return 50
        if symbol == "BANKNIFTY": return 15
        if symbol == "FINNIFTY": return 40
        return self._lot_sizes.get(symbol, 1)

    def get_fno_symbols(self) -> Tuple[List[str], str]:
        # Basic list for now, ideally fetch from NSE
        symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ITC", "BAJFINANCE"]
        return symbols, f"✓ {len(symbols)} benchmark FnO symbols loaded via NseKit"

    def fetch_all_data(self, symbols: List[str], days_back: int = 252) -> Tuple[pd.DataFrame, str]:
        """
        Parallel fetcher for universe-wide analytics.
        Uses yfinance for fast bulk history and NseKit for live context.
        """
        import time as _time
        t0 = _time.time()
        
        # 1. Cache Check
        sym_hash = hashlib.md5("".join(sorted(symbols)).encode()).hexdigest()
        cache_key = f"nse_analytics_{sym_hash}_{datetime.now().strftime('%Y%m%d_%H')}"
        
        cached_df = self.cache.get(cache_key)
        if cached_df is not None:
            return cached_df, f"✓ Analytics loaded from cache ({len(cached_df)} symbols)"

        # 2. Bulk History Fetch (Fastest way for 252 days)
        yf_symbols = [f"{s}.NS" for s in symbols]
        try:
            raw = yf.download(yf_symbols, period="1y", interval="1d", group_by="ticker", progress=False)
        except Exception as e:
            return pd.DataFrame(), f"⚠ Failed to fetch history: {str(e)}"
        
        # 3. Parallel Compute
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self._compute_analytics_single)(sym, raw[f"{sym}.NS"] if f"{sym}.NS" in raw.columns.levels[0] else None)
            for sym in symbols
        )
        results = [r for r in results if r is not None]
        
        df_final = pd.DataFrame(results)
        if not df_final.empty:
            self.cache.set(cache_key, df_final, expire=3600)
            
        return df_final, f"✓ Hybrid Analytics for {len(results)} symbols via NseKit ({_time.time()-t0:.1f}s)"

    def _compute_analytics_single(self, sym: str, df: Optional[pd.DataFrame]) -> Optional[Dict]:
        """Ported analytics engine from legacy pipeline."""
        if df is None or df.empty: return None
        try:
            df = df.dropna()
            if len(df) < 60: return None
            
            Cl = df['Close']
            O, H, L, V = df['Open'], df['High'], df['Low'], df['Volume']
            price = float(Cl.iloc[-1])
            lr = np.log(Cl / Cl.shift(1)).dropna()
            
            # --- VOLATILITY ---
            rv_c2c = lr.rolling(20).std() * np.sqrt(252)
            hl = np.log(H / L)
            rv_park = np.sqrt(hl.pow(2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)
            u = np.log(H / O); d = np.log(L / O); c = np.log(Cl / O)
            gk_var = (0.5 * u.pow(2) - (2*np.log(2)-1) * c.pow(2) + 0.5 * d.pow(2)).rolling(20).mean()
            rv_gk = np.sqrt(gk_var.clip(lower=0)) * np.sqrt(252)
            
            w = {'c2c': 0.15, 'park': 0.20, 'gk': 0.65}
            rv_composite = (w['c2c'] * rv_c2c.fillna(0) + w['park'] * rv_park.fillna(0) + w['gk'] * rv_gk.fillna(0))
            current_rv = float(rv_composite.iloc[-1]) if not np.isnan(rv_composite.iloc[-1]) else 0.25
            
            # --- IV ESTIMATION ---
            ivp = float(np.sum(rv_composite <= current_rv) / len(rv_composite) * 100) if len(rv_composite) > 20 else 50.0
            atmiv = current_rv * 1.12 * 100 
            
            # --- GARCH ---
            omega, alpha, beta = 0.000005, 0.10, 0.85
            var_t = current_rv**2 / 252
            for ret in lr.values[-60:]:
                var_t = max(omega + alpha * ret**2 + beta * var_t, 1e-10)
            garch_vol = np.sqrt(var_t * 252)

            # --- TA ---
            delta_c = Cl.diff()
            gain = delta_c.where(delta_c > 0, 0).rolling(14).mean()
            loss = (-delta_c.where(delta_c < 0, 0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
            rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
            
            ma20 = float(Cl.rolling(20).mean().iloc[-1])
            ma50 = float(Cl.rolling(50).mean().iloc[-1])
            
            return {
                'Instrument': sym, 'price': round(price, 2),
                'ATMIV': round(atmiv, 2), 'IVPercentile': round(ivp, 1),
                'RV_Composite': round(current_rv * 100, 2),
                'GARCH_Vol': round(garch_vol * 100, 2),
                'PCR': 1.0, 'volume': int(V.iloc[-1]),
                'rsi_daily': round(rsi_val, 2), 
                'ma20_daily': round(ma20, 2), 'ma50_daily': round(ma50, 2),
                '% change': round(float(Cl.pct_change().iloc[-1] * 100), 2),
                'lot_size': self.get_lot_size(sym),
                'CUSUM_Alert': False
            }
        except Exception:
            return None

    def get_option_chain(self, symbol: str, expiry: Optional[date] = None) -> Optional[OptionChain]:
        """Fetch and parse NSE option chain using NseKit."""
        try:
            expiry_str = expiry.strftime('%d-%b-%Y').upper() if expiry else None
            raw = self.nse.fno_live_option_chain(symbol, expiry_date=expiry_str)
            
            underlying_price = 0
            if raw and isinstance(raw, list) and len(raw) > 0:
                # NseKit return list of strike info
                # To get spot, we do a quick single stock quote
                underlying_price = float(self.nse.stock_quote(symbol).get('lastPrice', 0))

            calls, puts = [], []
            for item in raw:
                strike = float(item['strikePrice'])
                
                for otype in ['CE', 'PE']:
                    side = item.get(otype)
                    if not side: continue
                    
                    quote = OptionQuote(
                        instrument_token=0,
                        tradingsymbol=f"{symbol}{expiry_str if expiry_str else ''}{strike}{otype}",
                        strike=strike,
                        option_type=otype,
                        expiry=expiry or date.today(),
                        last_price=side.get('lastPrice', 0),
                        bid=side.get('bidprice', side.get('lastPrice', 0)),
                        ask=side.get('askPrice', side.get('lastPrice', 0)),
                        bid_qty=side.get('bidQty', 0),
                        ask_qty=side.get('askQty', 0),
                        volume=side.get('totalTradedVolume', 0),
                        oi=side.get('openInterest', 0),
                        iv=side.get('impliedVolatility', 0),
                        total_buy_qty=side.get('totalBuyQuantity', 0),
                        total_sell_qty=side.get('totalSellQuantity', 0),
                        is_liquid=True # Simplified
                    )
                    
                    if otype == 'CE': calls.append(quote)
                    else: puts.append(quote)
            
            return OptionChain(symbol, underlying_price, expiry or date.today(), calls, puts, datetime.now())
        except Exception as e:
            logger.error(f"NseKit Option Chain failed for {symbol}: {e}")
            return None

    def fetch_strategy_margin(self, legs: List[Dict], spot_price: float = 0) -> Dict[str, float]:
        """Estimated Margin Estimator."""
        total_margin = 0
        premium_flow = 0
        for leg in legs:
            lot_size = self.get_lot_size(leg.get('tradingsymbol', '').split(' ')[0])
            notional = spot_price * lot_size
            if "Buy" in leg['type']: premium_flow -= leg.get('premium', 0) * lot_size
            else:
                total_margin += notional * 0.12 # 12% margin estimate
                premium_flow += leg.get('premium', 0) * lot_size
        if len(legs) > 1: total_margin *= 0.6 # Hedging benefit estimate
        return {
            "total_margin": max(total_margin, 0),
            "option_premium": premium_flow,
            "span_margin": total_margin * 0.8,
            "exposure_margin": total_margin * 0.2
        }

def render_nsekit_info():
    """Information block for Streamlit."""
    import streamlit as st
    st.sidebar.success("📡 Data: NseKit + yfinance")
    st.sidebar.info("Margin values are heuristic estimates.")
