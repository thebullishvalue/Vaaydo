"""
╔══════════════════════════════════════════════════════════════════════════╗
║  VAAYDO — Zerodha Kite Connect Data Pipeline                           ║
║  Real-time options chain, OHLCV, greeks from Kite Connect API          ║
║  Replaces yfinance with production-grade broker data                   ║
║  Hemrek Capital                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Data Flow:
  1. KiteAuth         — Session management (access_token lifecycle)
  2. InstrumentCache  — Instrument master download + F&O filtering
  3. HistoricalData   — OHLCV candles via historical_data endpoint
  4. OptionChainFetch — Live option chain via instruments + quotes
  5. KiteDataPipeline — Orchestrator replacing fetch_all_data()

API Docs: https://kite.trade/docs/connect/v3/
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import hashlib
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# §1  KITE AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_URL = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login"

@dataclass
class KiteSession:
    """Manages Kite Connect API session."""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    request_token: str = ""
    user_id: str = ""
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self):
        # Load from environment or config
        self.api_key = self.api_key or os.environ.get("KITE_API_KEY", "")
        self.api_secret = self.api_secret or os.environ.get("KITE_API_SECRET", "")
        self.access_token = self.access_token or os.environ.get("KITE_ACCESS_TOKEN", "")
        self._setup_headers()

    def _setup_headers(self):
        if self.access_token:
            self._session.headers.update({
                "X-Kite-Version": "3",
                "Authorization": f"token {self.api_key}:{self.access_token}",
                "Content-Type": "application/x-www-form-urlencoded",
            })

    def generate_session(self, request_token: str) -> dict:
        """Exchange request_token for access_token."""
        self.request_token = request_token
        checksum = hashlib.sha256(
            f"{self.api_key}{request_token}{self.api_secret}".encode()
        ).hexdigest()

        resp = self._session.post(
            f"{BASE_URL}/session/token",
            data={
                "api_key": self.api_key,
                "request_token": request_token,
                "checksum": checksum,
            }
        )
        data = resp.json()
        if data.get("status") == "success":
            self.access_token = data["data"]["access_token"]
            self.user_id = data["data"].get("user_id", "")
            self._setup_headers()
            return data["data"]
        raise ConnectionError(f"Kite session failed: {data.get('message', 'Unknown error')}")

    def is_valid(self) -> bool:
        """Check if current session is valid."""
        if not self.access_token or not self.api_key:
            return False
        try:
            resp = self._session.get(f"{BASE_URL}/user/profile")
            return resp.status_code == 200 and resp.json().get("status") == "success"
        except Exception:
            return False

    def get(self, endpoint: str, params: dict = None) -> dict:
        """Authenticated GET request."""
        resp = self._session.get(f"{BASE_URL}{endpoint}", params=params)
        data = resp.json()
        if data.get("status") == "success":
            return data.get("data", {})
        raise ConnectionError(f"Kite API error: {data.get('message', resp.text[:200])}")

    def get_login_url(self) -> str:
        """Get the Kite login URL for OAuth flow."""
        return f"{LOGIN_URL}?api_key={self.api_key}&v=3"


# ═══════════════════════════════════════════════════════════════════════════════
# §2  INSTRUMENT MASTER CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class InstrumentCache:
    """Downloads and caches the Kite instrument master.
    
    The instrument master is ~50MB CSV updated daily.
    Contains all tradable instruments with tokens, lot sizes, tick sizes, etc.
    """

    CACHE_FILE = "/tmp/kite_instruments.csv"
    CACHE_TTL = 6 * 3600  # 6 hours

    def __init__(self, session: KiteSession):
        self.session = session
        self._instruments: Optional[pd.DataFrame] = None
        self._fno_instruments: Optional[pd.DataFrame] = None

    def _is_cache_fresh(self) -> bool:
        if not os.path.exists(self.CACHE_FILE):
            return False
        mtime = os.path.getmtime(self.CACHE_FILE)
        return (time.time() - mtime) < self.CACHE_TTL

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load instrument master (from cache or API)."""
        if not force_refresh and self._instruments is not None:
            return self._instruments

        if not force_refresh and self._is_cache_fresh():
            try:
                self._instruments = pd.read_csv(self.CACHE_FILE)
                return self._instruments
            except Exception:
                pass

        # Download fresh from Kite
        resp = self.session._session.get(f"{BASE_URL}/instruments")
        if resp.status_code != 200:
            raise ConnectionError(f"Instrument download failed: {resp.status_code}")

        # Parse CSV response
        from io import StringIO
        self._instruments = pd.read_csv(StringIO(resp.text))
        self._instruments.to_csv(self.CACHE_FILE, index=False)
        logger.info(f"Loaded {len(self._instruments)} instruments from Kite")
        return self._instruments

    def get_fno_symbols(self) -> Tuple[List[str], str]:
        """Get list of F&O traded equity symbols."""
        df = self.load()
        # NFO segment, type FUT (futures indicate F&O availability)
        fno = df[
            (df['segment'] == 'NFO-FUT') &
            (df['instrument_type'] == 'FUT')
        ]['name'].unique().tolist()

        # Filter to current month expiry to get active F&O stocks
        fno = sorted(set(fno))
        if len(fno) > 30:
            return fno, f"✓ {len(fno)} F&O stocks from Kite Connect"
        return fno, f"✓ {len(fno)} F&O stocks (Kite)"

    def get_lot_sizes(self) -> Dict[str, int]:
        """Extract lot sizes from instrument master."""
        df = self.load()
        fno = df[
            (df['segment'] == 'NFO-FUT') &
            (df['instrument_type'] == 'FUT')
        ].drop_duplicates(subset='name', keep='first')
        return dict(zip(fno['name'], fno['lot_size'].astype(int)))

    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """Get instrument token for a symbol."""
        df = self.load()
        match = df[
            (df['tradingsymbol'] == symbol) &
            (df['exchange'] == exchange) &
            (df['instrument_type'] == 'EQ')
        ]
        if len(match) > 0:
            return int(match.iloc[0]['instrument_token'])
        return None

    def get_option_instruments(self, symbol: str, expiry: Optional[date] = None) -> pd.DataFrame:
        """Get all option instruments for a symbol, optionally filtered by expiry."""
        df = self.load()
        mask = (df['segment'] == 'NFO-OPT') & (df['name'] == symbol)
        if expiry:
            expiry_str = expiry.strftime('%Y-%m-%d')
            mask &= (df['expiry'] == expiry_str)
        opts = df[mask].copy()
        if len(opts) > 0:
            opts['expiry'] = pd.to_datetime(opts['expiry'])
            opts = opts.sort_values(['expiry', 'strike', 'instrument_type'])
        return opts

    def get_next_expiries(self, symbol: str, n: int = 3) -> List[date]:
        """Get next n expiry dates for a symbol."""
        df = self.load()
        opts = df[(df['segment'] == 'NFO-OPT') & (df['name'] == symbol)]
        if opts.empty:
            return []
        expiries = pd.to_datetime(opts['expiry']).unique()
        today = pd.Timestamp.now().normalize()
        future = sorted([e for e in expiries if e >= today])
        return [e.date() for e in future[:n]]

    def get_strike_gap(self, symbol: str, expiry: date) -> float:
        """Determine actual strike gap from instrument master."""
        opts = self.get_option_instruments(symbol, expiry)
        if len(opts) < 2:
            return 0
        strikes = sorted(opts['strike'].unique())
        if len(strikes) < 2:
            return 0
        gaps = np.diff(strikes)
        # Most common gap (mode)
        from collections import Counter
        gap_counts = Counter(gaps)
        return float(gap_counts.most_common(1)[0][0])


# ═══════════════════════════════════════════════════════════════════════════════
# §3  HISTORICAL DATA (OHLCV)
# ═══════════════════════════════════════════════════════════════════════════════

class HistoricalData:
    """Fetch historical OHLCV candles from Kite Connect.
    
    Kite Connect rate limits:
    - 3 requests/second for historical data
    - Daily candles: max 2000 days per request
    - Minute candles: max 60 days per request
    """

    # Kite candle intervals
    INTERVAL_DAY = "day"
    INTERVAL_5MIN = "5minute"
    INTERVAL_15MIN = "15minute"
    INTERVAL_60MIN = "60minute"

    def __init__(self, session: KiteSession, instruments: InstrumentCache):
        self.session = session
        self.instruments = instruments
        self._rate_limiter_last = 0

    def _rate_limit(self):
        """Enforce 3 req/sec limit."""
        now = time.time()
        elapsed = now - self._rate_limiter_last
        if elapsed < 0.35:  # ~3/sec with margin
            time.sleep(0.35 - elapsed)
        self._rate_limiter_last = time.time()

    def fetch_ohlcv(self, symbol: str, days_back: int = 400,
                    interval: str = "day") -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol.
        
        Returns DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        token = self.instruments.get_instrument_token(symbol)
        if token is None:
            logger.warning(f"No instrument token for {symbol}")
            return None

        end = datetime.now()
        start = end - timedelta(days=days_back + 60)

        self._rate_limit()
        try:
            data = self.session.get(
                f"/instruments/historical/{token}/{interval}",
                params={
                    "from": start.strftime("%Y-%m-%d"),
                    "to": end.strftime("%Y-%m-%d"),
                    "oi": 1  # include open interest if available
                }
            )
        except ConnectionError as e:
            logger.warning(f"Historical data failed for {symbol}: {e}")
            return None

        if not data or "candles" not in data:
            return None

        candles = data["candles"]
        if not candles:
            return None

        df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()

        # Drop OI if all zeros
        if (df['OI'] == 0).all():
            df = df.drop(columns=['OI'])

        return df

    def fetch_batch(self, symbols: List[str], days_back: int = 400) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols with rate limiting."""
        results = {}
        for i, sym in enumerate(symbols):
            df = self.fetch_ohlcv(sym, days_back)
            if df is not None and not df.empty:
                results[sym] = df
            if (i + 1) % 50 == 0:
                logger.info(f"Fetched {i+1}/{len(symbols)} symbols")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# §4  LIVE OPTION CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptionQuote:
    """Single option quote from Kite."""
    instrument_token: int
    tradingsymbol: str
    strike: float
    option_type: str  # CE or PE
    expiry: date
    last_price: float
    bid: float
    ask: float
    volume: int
    oi: int
    iv: float  # implied volatility (if available from Kite)
    # Greeks from Kite (if available via Sensibull integration)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

@dataclass
class OptionChain:
    """Complete option chain for a symbol/expiry."""
    symbol: str
    spot_price: float
    expiry: date
    calls: List[OptionQuote] = field(default_factory=list)
    puts: List[OptionQuote] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_atm_strike(self) -> float:
        """Find ATM strike closest to spot."""
        all_strikes = sorted(set(
            [c.strike for c in self.calls] + [p.strike for p in self.puts]
        ))
        if not all_strikes:
            return self.spot_price
        return min(all_strikes, key=lambda x: abs(x - self.spot_price))

    def get_atm_iv(self) -> float:
        """Average IV of ATM call and put."""
        atm = self.get_atm_strike()
        atm_call = next((c for c in self.calls if c.strike == atm), None)
        atm_put = next((p for p in self.puts if p.strike == atm), None)
        ivs = []
        if atm_call and atm_call.iv > 0:
            ivs.append(atm_call.iv)
        if atm_put and atm_put.iv > 0:
            ivs.append(atm_put.iv)
        return np.mean(ivs) if ivs else 0

    def get_iv_surface(self) -> pd.DataFrame:
        """Construct IV surface from chain."""
        rows = []
        for opt in self.calls + self.puts:
            if opt.iv > 0:
                moneyness = opt.strike / self.spot_price
                rows.append({
                    'strike': opt.strike,
                    'type': opt.option_type,
                    'iv': opt.iv,
                    'moneyness': moneyness,
                    'volume': opt.volume,
                    'oi': opt.oi,
                    'bid': opt.bid,
                    'ask': opt.ask,
                })
        return pd.DataFrame(rows)

    def get_pcr(self) -> float:
        """Put-Call Ratio from OI."""
        put_oi = sum(p.oi for p in self.puts)
        call_oi = sum(c.oi for c in self.calls)
        if call_oi == 0:
            return 1.0
        return put_oi / call_oi

    def get_max_pain(self) -> float:
        """Calculate max pain strike."""
        all_strikes = sorted(set(
            [c.strike for c in self.calls] + [p.strike for p in self.puts]
        ))
        if not all_strikes:
            return self.spot_price

        min_pain = float('inf')
        max_pain_strike = all_strikes[0]

        for settlement_price in all_strikes:
            total_pain = 0
            for call in self.calls:
                if settlement_price > call.strike:
                    total_pain += (settlement_price - call.strike) * call.oi
            for put in self.puts:
                if settlement_price < put.strike:
                    total_pain += (put.strike - settlement_price) * put.oi
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = settlement_price

        return max_pain_strike


class OptionChainFetch:
    """Fetch live option chain from Kite Connect.
    
    Kite doesn't have a dedicated option chain API.
    We construct it from instruments + LTP/quote endpoints.
    """

    def __init__(self, session: KiteSession, instruments: InstrumentCache):
        self.session = session
        self.instruments = instruments

    def fetch(self, symbol: str, expiry: date) -> Optional[OptionChain]:
        """Fetch complete option chain for symbol/expiry."""
        # 1. Get option instruments from master
        opts = self.instruments.get_option_instruments(symbol, expiry)
        if opts.empty:
            logger.warning(f"No option instruments for {symbol} expiry {expiry}")
            return None

        # 2. Get spot price
        spot_token = self.instruments.get_instrument_token(symbol)
        spot_price = 0
        if spot_token:
            try:
                quote = self.session.get("/quote", params={"i": f"NSE:{symbol}"})
                if f"NSE:{symbol}" in quote:
                    spot_price = quote[f"NSE:{symbol}"]["last_price"]
            except Exception:
                pass

        if spot_price <= 0:
            logger.warning(f"Could not get spot price for {symbol}")
            return None

        # 3. Fetch quotes for all option instruments (batch: max 500 per request)
        tokens = opts['instrument_token'].astype(int).tolist()
        trading_symbols = opts.set_index('instrument_token')['tradingsymbol'].to_dict()
        strikes = opts.set_index('instrument_token')['strike'].to_dict()
        types = opts.set_index('instrument_token')['instrument_type'].to_dict()

        calls = []
        puts = []

        # Kite quote endpoint accepts up to 500 instruments
        for batch_start in range(0, len(tokens), 500):
            batch_tokens = tokens[batch_start:batch_start + 500]
            instrument_list = [f"NFO:{trading_symbols.get(t, '')}" for t in batch_tokens]

            try:
                time.sleep(0.35)  # rate limit
                quotes = self.session.get("/quote", params={"i": instrument_list})
            except Exception as e:
                logger.warning(f"Quote batch failed: {e}")
                continue

            for key, qdata in quotes.items():
                # Parse instrument token from key
                tsym = key.replace("NFO:", "")
                # Find matching token
                match = opts[opts['tradingsymbol'] == tsym]
                if match.empty:
                    continue

                row = match.iloc[0]
                token = int(row['instrument_token'])
                strike = float(row['strike'])
                otype = str(row['instrument_type'])  # CE or PE

                depth = qdata.get('depth', {})
                best_bid = depth.get('buy', [{}])[0].get('price', 0) if depth.get('buy') else 0
                best_ask = depth.get('sell', [{}])[0].get('price', 0) if depth.get('sell') else 0

                oq = OptionQuote(
                    instrument_token=token,
                    tradingsymbol=tsym,
                    strike=strike,
                    option_type=otype,
                    expiry=expiry,
                    last_price=qdata.get('last_price', 0),
                    bid=best_bid,
                    ask=best_ask,
                    volume=qdata.get('volume', 0),
                    oi=qdata.get('oi', 0),
                    iv=0,  # Kite doesn't directly provide IV
                )

                if otype == 'CE':
                    calls.append(oq)
                else:
                    puts.append(oq)

        chain = OptionChain(
            symbol=symbol,
            spot_price=spot_price,
            expiry=expiry,
            calls=sorted(calls, key=lambda x: x.strike),
            puts=sorted(puts, key=lambda x: x.strike),
        )

        # 4. Compute IV for each option using BSM inversion
        self._compute_chain_iv(chain)

        return chain

    def _compute_chain_iv(self, chain: OptionChain):
        """Compute implied volatility for each option via Newton-Raphson."""
        from scipy.stats import norm

        S = chain.spot_price
        T = max((chain.expiry - date.today()).days / 365.0, 1 / 365)
        r = 0.07  # India risk-free rate

        def bsm_price(S, K, T, r, sigma, otype):
            if sigma <= 0 or T <= 0:
                return max(S - K, 0) if otype == 'CE' else max(K - S, 0)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if otype == 'CE':
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        def bsm_vega(S, K, T, r, sigma):
            if sigma <= 0 or T <= 0:
                return 0
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * np.sqrt(T) * norm.pdf(d1)

        def implied_vol(market_price, K, otype, max_iter=50, tol=1e-5):
            """Newton-Raphson IV solver."""
            if market_price <= 0:
                return 0

            intrinsic = max(S - K, 0) if otype == 'CE' else max(K - S, 0)
            if market_price < intrinsic * 0.95:
                return 0

            sigma = 0.25  # initial guess
            for _ in range(max_iter):
                price = bsm_price(S, K, T, r, sigma, otype)
                v = bsm_vega(S, K, T, r, sigma)
                if v < 1e-10:
                    break
                diff = price - market_price
                if abs(diff) < tol:
                    break
                sigma -= diff / v
                sigma = max(0.01, min(sigma, 5.0))  # bounds
            return sigma if 0.01 < sigma < 5.0 else 0

        for opt in chain.calls:
            # Use mid price for IV computation
            mid = (opt.bid + opt.ask) / 2 if opt.bid > 0 and opt.ask > 0 else opt.last_price
            if mid > 0:
                opt.iv = implied_vol(mid, opt.strike, 'CE')

        for opt in chain.puts:
            mid = (opt.bid + opt.ask) / 2 if opt.bid > 0 and opt.ask > 0 else opt.last_price
            if mid > 0:
                opt.iv = implied_vol(mid, opt.strike, 'PE')


# ═══════════════════════════════════════════════════════════════════════════════
# §5  MASTER DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class KiteDataPipeline:
    """Master orchestrator replacing yfinance-based fetch_all_data().
    
    Provides the same output schema as the original function but
    sourced from Kite Connect with real market data.
    """

    def __init__(self, api_key: str = "", api_secret: str = "", access_token: str = ""):
        self.session = KiteSession(api_key=api_key, api_secret=api_secret, access_token=access_token)
        self.instruments = InstrumentCache(self.session)
        self.historical = HistoricalData(self.session, self.instruments)
        self.options = OptionChainFetch(self.session, self.instruments)
        self._lot_sizes: Dict[str, int] = {}

    def is_connected(self) -> bool:
        """Check if Kite session is active."""
        return self.session.is_valid()

    def initialize(self) -> str:
        """Load instrument master and extract lot sizes."""
        self.instruments.load()
        self._lot_sizes = self.instruments.get_lot_sizes()
        return f"✓ Loaded {len(self._lot_sizes)} F&O instruments"

    def get_fno_symbols(self) -> Tuple[List[str], str]:
        """Get F&O symbol list from Kite instrument master."""
        return self.instruments.get_fno_symbols()

    def get_lot_size(self, symbol: str) -> int:
        """Get lot size for a symbol."""
        return self._lot_sizes.get(symbol, 1)

    def fetch_all_data(self, symbols: List[str], days_back: int = 400) -> Tuple[pd.DataFrame, str]:
        """Main data fetch — drop-in replacement for original fetch_all_data().
        
        Returns:
            (DataFrame with same schema as original, status_message)
        """
        results = []
        ohlcv_data = self.historical.fetch_batch(symbols, days_back)

        for sym, df in ohlcv_data.items():
            try:
                if len(df) < 60:
                    continue

                O, H, L, Cl, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
                price = float(Cl.iloc[-1])
                lr = np.log(Cl / Cl.shift(1)).dropna()

                # §3.2: Multi-Estimator Volatility
                rv_c2c = lr.rolling(20).std() * np.sqrt(252)
                hl = np.log(H / L)
                rv_park = np.sqrt(hl.pow(2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)
                u = np.log(H / O); d = np.log(L / O); c = np.log(Cl / O)
                gk_var = (0.5 * u.pow(2) - (2 * np.log(2) - 1) * c.pow(2) + 0.5 * d.pow(2)).rolling(20).mean()
                rv_gk = np.sqrt(gk_var.clip(lower=0)) * np.sqrt(252)
                o_c_prev = np.log(O / Cl.shift(1)); c_o = np.log(Cl / O)
                yz_o = o_c_prev.rolling(20).var(); yz_c = c_o.rolling(20).var()
                k = 0.34 / (1.34 + 21 / 19)
                yz_var = yz_o + k * yz_c + (1 - k) * gk_var.clip(lower=0)
                rv_yz = np.sqrt(yz_var.clip(lower=0)) * np.sqrt(252)

                w = {'c2c': 0.15, 'park': 0.20, 'gk': 0.25, 'yz': 0.40}
                rv_composite = (w['c2c'] * rv_c2c.fillna(0) + w['park'] * rv_park.fillna(0) +
                                w['gk'] * rv_gk.fillna(0) + w['yz'] * rv_yz.fillna(0))
                current_rv = float(rv_composite.iloc[-1]) if not np.isnan(rv_composite.iloc[-1]) else 0.25
                current_rv = max(current_rv, 0.05)

                # §3.3: IV Estimation with VRP
                rv_history = rv_composite.dropna().values
                lookback = min(252, len(rv_history))
                ivp = float(np.sum(rv_history[-lookback:] <= current_rv) / lookback * 100) if lookback > 20 else 50.0

                if ivp > 70:
                    vrp_factor = 1.08
                elif ivp < 30:
                    vrp_factor = 1.18
                else:
                    vrp_factor = 1.12
                atmiv = current_rv * vrp_factor * 100

                # GARCH(1,1)
                omega, alpha, beta = 0.000005, 0.10, 0.85
                lr_vals = lr.values[-60:] if len(lr) >= 60 else lr.values
                var_t = current_rv ** 2 / 252
                for ret in lr_vals:
                    var_t = max(omega + alpha * ret ** 2 + beta * var_t, 1e-10)
                garch_vol = np.sqrt(var_t * 252)
                persistence = alpha + beta
                half_life = -np.log(2) / np.log(max(persistence, 0.001)) if persistence < 1 else 999

                # Technical Indicators
                delta_c = Cl.diff()
                gain = delta_c.where(delta_c > 0, 0).rolling(14).mean()
                loss = (-delta_c.where(delta_c < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

                # ADX
                plus_dm = H.diff().clip(lower=0)
                minus_dm = (-L.diff()).clip(lower=0)
                plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
                minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
                true_range = pd.concat([H - L, (H - Cl.shift(1)).abs(), (L - Cl.shift(1)).abs()], axis=1).max(axis=1)
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

                # Kalman Filter
                kalman_price = price
                kalman_var = atr_val ** 2
                R_noise = (price * 0.01) ** 2
                Q_proc = (price * 0.002) ** 2
                for p_val in Cl.values[-20:]:
                    if np.isnan(p_val):
                        continue
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

                # CUSUM
                cusum_pos, cusum_neg, cusum_alert = 0.0, 0.0, False
                recent_lr = lr.values[-30:] if len(lr) >= 30 else lr.values
                mu_lr = np.mean(recent_lr)
                sd_lr = max(np.std(recent_lr), 1e-6)
                for r_val in recent_lr[-10:]:
                    z = (r_val - mu_lr) / sd_lr
                    cusum_pos = max(0, cusum_pos + z - 0.5)
                    cusum_neg = max(0, cusum_neg - z - 0.5)
                    if cusum_pos > 4.0 or cusum_neg > 4.0:
                        cusum_alert = True
                        cusum_pos = cusum_neg = 0

                def safe_rv(series):
                    v = series.iloc[-1] if len(series) > 0 else np.nan
                    return round(float(v * 100), 2) if not np.isnan(v) else 0.0

                results.append({
                    'Instrument': sym,
                    'price': round(price, 2),
                    'ATMIV': round(atmiv, 2),
                    'IVPercentile': round(ivp, 1),
                    'RV_Composite': round(current_rv * 100, 2),
                    'GARCH_Vol': round(garch_vol * 100, 2),
                    'VRP_Factor': vrp_factor,
                    'GARCH_Persistence': round(persistence, 3),
                    'GARCH_HalfLife': round(half_life, 1),
                    'PCR': round(pcr, 3),
                    'volume': vol_curr,
                    'vol20': vol20,
                    'rsi_daily': round(rsi_val, 2),
                    'atr_daily': round(atr_val, 2),
                    'adx': round(adx_val, 1),
                    'kalman_trend': round(kalman_trend, 3),
                    'ma20_daily': round(ma20, 2),
                    'ma50_daily': round(ma50, 2),
                    'ma200_daily': round(ma200, 2),
                    '% change': round(pct_change, 2),
                    'CUSUM_Alert': cusum_alert,
                    'lot_size': self.get_lot_size(sym),
                    'RV_C2C': safe_rv(rv_c2c),
                    'RV_Parkinson': safe_rv(rv_park),
                    'RV_GK': safe_rv(rv_gk),
                    'RV_YZ': safe_rv(rv_yz),
                })
            except Exception as e:
                logger.warning(f"Analytics failed for {sym}: {e}")
                continue

        if not results:
            return pd.DataFrame(), "No valid data from Kite Connect"
        return pd.DataFrame(results), f"✓ Analytics for {len(results)} securities via Kite Connect"

    def fetch_option_chain(self, symbol: str, expiry: date) -> Optional[OptionChain]:
        """Fetch live option chain for strategy pricing."""
        return self.options.fetch(symbol, expiry)

    def get_live_option_prices(self, symbol: str, expiry: date,
                                strikes: List[float]) -> Dict[str, Dict[float, float]]:
        """Get live prices for specific strikes (for strategy execution).
        
        Returns:
            {'CE': {strike: mid_price, ...}, 'PE': {strike: mid_price, ...}}
        """
        chain = self.fetch_option_chain(symbol, expiry)
        if chain is None:
            return {'CE': {}, 'PE': {}}

        ce_prices = {}
        pe_prices = {}

        for c in chain.calls:
            if c.strike in strikes:
                mid = (c.bid + c.ask) / 2 if c.bid > 0 and c.ask > 0 else c.last_price
                ce_prices[c.strike] = mid

        for p in chain.puts:
            if p.strike in strikes:
                mid = (p.bid + p.ask) / 2 if p.bid > 0 and p.ask > 0 else p.last_price
                pe_prices[p.strike] = mid

        return {'CE': ce_prices, 'PE': pe_prices}

    def get_real_iv(self, symbol: str, expiry: date) -> Optional[float]:
        """Get real ATM IV from live option chain."""
        chain = self.fetch_option_chain(symbol, expiry)
        if chain is None:
            return None
        iv = chain.get_atm_iv()
        return iv * 100 if iv > 0 else None  # Return as percentage


# ═══════════════════════════════════════════════════════════════════════════════
# §6  STREAMLIT SESSION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def init_kite_session(api_key: str, api_secret: str, access_token: str = "") -> KiteDataPipeline:
    """Initialize Kite pipeline for Streamlit session."""
    pipeline = KiteDataPipeline(api_key=api_key, api_secret=api_secret, access_token=access_token)
    if pipeline.is_connected():
        pipeline.initialize()
    return pipeline


def render_kite_login(sidebar=True):
    """Render Kite Connect login UI in Streamlit sidebar.
    
    Returns (pipeline, is_connected) tuple.
    """
    import streamlit as st

    container = st.sidebar if sidebar else st

    with container:
        st.markdown('<div class="stitle">🔗 Kite Connect</div>', unsafe_allow_html=True)

        # Check existing session
        if 'kite_pipeline' in st.session_state and st.session_state.kite_pipeline.is_connected():
            st.success("✅ Kite Connected")
            return st.session_state.kite_pipeline, True

        # Credential inputs
        api_key = st.text_input("API Key", type="password", key="kite_api_key",
                                value=os.environ.get("KITE_API_KEY", ""))
        api_secret = st.text_input("API Secret", type="password", key="kite_api_secret",
                                   value=os.environ.get("KITE_API_SECRET", ""))

        # Option 1: Direct access token (for already-authenticated sessions)
        access_token = st.text_input("Access Token (if available)", type="password",
                                     key="kite_access_token",
                                     value=os.environ.get("KITE_ACCESS_TOKEN", ""))

        if access_token and api_key:
            if st.button("Connect with Token", type="primary", key="kite_connect_token"):
                pipeline = KiteDataPipeline(api_key, api_secret, access_token)
                if pipeline.is_connected():
                    pipeline.initialize()
                    st.session_state.kite_pipeline = pipeline
                    st.success("✅ Connected to Kite")
                    st.rerun()
                else:
                    st.error("Invalid token. Please re-authenticate.")
            return None, False

        # Option 2: OAuth flow
        if api_key and api_secret:
            session = KiteSession(api_key=api_key, api_secret=api_secret)
            login_url = session.get_login_url()
            st.markdown(f"[🔐 Login to Zerodha]({login_url})")
            request_token = st.text_input("Paste Request Token", key="kite_request_token")
            if request_token and st.button("Generate Session", key="kite_gen_session"):
                try:
                    session.generate_session(request_token)
                    pipeline = KiteDataPipeline(api_key, api_secret, session.access_token)
                    pipeline.initialize()
                    st.session_state.kite_pipeline = pipeline
                    st.success("✅ Session generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Session failed: {e}")

        st.caption("Enter Kite API credentials or set KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN env vars")
        return None, False
