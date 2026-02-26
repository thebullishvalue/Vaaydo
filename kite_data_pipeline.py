"""
╔══════════════════════════════════════════════════════════════════════════╗
║  VAAYDO — Zerodha Kite Connect Data Pipeline v2.0                      ║
║  Source-of-Truth Engine: Quotes, Margins, Depth, Execution Readiness   ║
║  Hemrek Capital                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Data Flow:
  1. KiteAuth         — Session management & automated TOTP login
  2. InstrumentCache  — Master contract database (F&O, Equity)
  3. MarketData       — L2 Quotes (Depth), OHLCV, and OI
  4. MarginCalculator — Native SPAN + Exposure margin fetching
  5. OptionChainFetch — Synced Option Chain + Spot + IV Solver
  6. KiteDataPipeline — Master Orchestrator

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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
import logging
import re
from urllib.parse import urlparse, parse_qs
from kiteconnect import KiteConnect
import pyotp

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# §1  KITE AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_URL = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login"

@dataclass
class KiteSession:
    """Manages Kite Connect API session and headers."""
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
        """Check if current session is valid via user profile."""
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
        raise ConnectionError(f"Kite API GET error: {data.get('message', resp.text[:200])}")

    def post(self, endpoint: str, payload: Any) -> dict:
        """Authenticated POST request."""
        # For POST, we often send JSON, so we update headers temporarily
        headers = self._session.headers.copy()
        headers["Content-Type"] = "application/json"
        
        resp = self._session.post(
            f"{BASE_URL}{endpoint}", 
            data=json.dumps(payload),
            headers=headers
        )
        data = resp.json()
        if data.get("status") == "success":
            return data.get("data", {})
        raise ConnectionError(f"Kite API POST error: {data.get('message', resp.text[:200])}")

    def get_login_url(self) -> str:
        return f"{LOGIN_URL}?api_key={self.api_key}&v=3"


# ═══════════════════════════════════════════════════════════════════════════════
# §2  INSTRUMENT MASTER
# ═══════════════════════════════════════════════════════════════════════════════

class InstrumentCache:
    """Manages the 50MB+ instrument master file with indexed lookups."""

    CACHE_FILE = "kite_instruments.csv"
    CACHE_TTL = 12 * 3600  # 12 hours

    def __init__(self, session: KiteSession):
        self.session = session
        self._instruments: Optional[pd.DataFrame] = None
        # Fast lookup indices — built once on load
        self._token_index: Dict[str, int] = {}       # "NSE:RELIANCE" → token
        self._name_index: Dict[str, pd.DataFrame] = {}  # grouped by name for options

    def _build_indices(self):
        """Build O(1) lookup dicts from the DataFrame — called once after load."""
        df = self._instruments
        # Token index: exchange:tradingsymbol → instrument_token
        self._token_index = {}
        for _, row in df[['exchange', 'tradingsymbol', 'instrument_token']].iterrows():
            key = f"{row['exchange']}:{row['tradingsymbol']}"
            self._token_index[key] = int(row['instrument_token'])
        
        # Pre-group NFO-OPT by name for fast option chain lookups
        nfo_opts = df[df['segment'] == 'NFO-OPT']
        self._name_index = {name: group for name, group in nfo_opts.groupby('name')}

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load instrument master (from cache or API)."""
        if not force_refresh and self._instruments is not None:
            return self._instruments

        # Check local cache
        if not force_refresh and os.path.exists(self.CACHE_FILE):
            mtime = os.path.getmtime(self.CACHE_FILE)
            if (time.time() - mtime) < self.CACHE_TTL:
                try:
                    self._instruments = pd.read_csv(self.CACHE_FILE)
                    self._build_indices()
                    return self._instruments
                except Exception:
                    pass

        # Download fresh
        resp = self.session._session.get(f"{BASE_URL}/instruments")
        if resp.status_code != 200:
            raise ConnectionError(f"Instrument download failed: {resp.status_code}")

        from io import StringIO
        self._instruments = pd.read_csv(StringIO(resp.text))
        self._instruments.to_csv(self.CACHE_FILE, index=False)
        self._build_indices()
        return self._instruments

    def get_fno_symbols(self) -> Tuple[List[str], str]:
        df = self.load()
        fno = df[
            (df['segment'] == 'NFO-FUT') &
            (df['instrument_type'] == 'FUT')
        ]['name'].unique().tolist()
        fno = sorted(set(fno))
        return fno, f"✓ {len(fno)} F&O symbols loaded from Kite"

    def get_lot_sizes(self) -> Dict[str, int]:
        df = self.load()
        fno = df[
            (df['segment'] == 'NFO-FUT') &
            (df['instrument_type'] == 'FUT')
        ].drop_duplicates(subset='name', keep='first')
        return dict(zip(fno['name'], fno['lot_size'].astype(int)))

    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """O(1) token lookup via pre-built index."""
        self.load()  # ensure loaded
        return self._token_index.get(f"{exchange}:{symbol}")
        
    def get_underlying_token(self, symbol: str) -> Optional[int]:
        """Get token for the underlying Spot (NSE/BSE)."""
        return self.get_instrument_token(symbol, "NSE")

    def get_option_instruments(self, symbol: str, expiry: Optional[date] = None) -> pd.DataFrame:
        """Fast option instrument lookup via pre-grouped index."""
        self.load()  # ensure loaded
        opts = self._name_index.get(symbol, pd.DataFrame())
        if opts.empty:
            return opts
        opts = opts.copy()
        if expiry:
            expiry_str = expiry.strftime('%Y-%m-%d')
            opts = opts[opts['expiry'] == expiry_str]
        if len(opts) > 0:
            opts['expiry'] = pd.to_datetime(opts['expiry'])
            opts = opts.sort_values(['expiry', 'strike', 'instrument_type'])
        return opts


# ═══════════════════════════════════════════════════════════════════════════════
# §3  LIVE MARKET DATA (Quotes, Depth, OHLCV)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketDepthItem:
    price: float
    quantity: int
    orders: int

@dataclass
class FullQuote:
    instrument_token: int
    last_price: float
    volume: int
    oi: int
    bids: List[MarketDepthItem]
    asks: List[MarketDepthItem]
    ohlc: Dict[str, float]
    timestamp: datetime
    # New Depth & Liquidity Metrics
    total_buy_quantity: int = 0
    total_sell_quantity: int = 0
    average_price: float = 0.0
    oi_day_high: int = 0
    oi_day_low: int = 0

class MarketData:
    """Handles all data fetching: Historical and Real-time Quotes."""

    def __init__(self, session: KiteSession, instruments: InstrumentCache):
        self.session = session
        self.instruments = instruments
        self._last_req_time = 0
        self._req_lock = __import__('threading').Lock()

    def _rate_limit(self, reqs_per_sec=8):
        """
        Kite allows 10 req/sec. Use 8 to stay safe.
        Thread-safe rate limiter.
        """
        with self._req_lock:
            now = time.time()
            min_interval = 1.0 / reqs_per_sec
            elapsed = now - self._last_req_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_req_time = time.time()

    def fetch_ohlcv_historical(self, symbol: str, days_back: int = 400) -> Optional[pd.DataFrame]:
        """Fetch historical candles."""
        token = self.instruments.get_instrument_token(symbol)
        if not token: return None
        
        end = datetime.now()
        start = end - timedelta(days=days_back + 30)
        
        self._rate_limit()
        try:
            data = self.session.get(
                f"/instruments/historical/{token}/day",
                params={"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"), "oi": 1}
            )
        except Exception as e:
            logger.error(f"History fetch failed for {symbol}: {e}")
            return None

        candles = data.get("candles", [])
        if not candles: return None

        df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df

    def fetch_ohlcv_batch(self, symbols: List[str], days_back: int = 400, max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV for multiple symbols concurrently.
        Uses ThreadPoolExecutor with rate-limiting to stay within Kite's 10 req/sec.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        
        def _fetch_one(sym):
            df = self.fetch_ohlcv_historical(sym, days_back)
            return sym, df
        
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, s): s for s in symbols}
            for fut in as_completed(futures):
                try:
                    sym, df = fut.result()
                    if df is not None and not df.empty:
                        results[sym] = df
                except Exception as e:
                    logger.error(f"Batch fetch error for {futures[fut]}: {e}")
        
        return results

    def fetch_quotes(self, instrument_tokens: List[int]) -> Dict[int, FullQuote]:
        """Fetch full Mode:Full quotes including depth for multiple tokens."""
        if not instrument_tokens: return {}
        
        results = {}
        chunk_size = 500
        
        for i in range(0, len(instrument_tokens), chunk_size):
            chunk = instrument_tokens[i:i+chunk_size]
            try:
                self._rate_limit()
                data = self.session.get("/quote", params={"i": [str(t) for t in chunk]})
                
                for key, q in data.items():
                    token = int(q['instrument_token'])
                    depth = q.get('depth', {})
                    bids = [MarketDepthItem(**b) for b in depth.get('buy', [])]
                    asks = [MarketDepthItem(**a) for a in depth.get('sell', [])]
                    
                    results[token] = FullQuote(
                        instrument_token=token,
                        last_price=q.get('last_price', 0),
                        volume=q.get('volume', 0),
                        oi=q.get('oi', 0),
                        bids=bids,
                        asks=asks,
                        ohlc=q.get('ohlc', {}),
                        timestamp=datetime.fromisoformat(q['timestamp'].replace('Z', '+00:00')) if q.get('timestamp') else datetime.now(),
                        total_buy_quantity=q.get('buy_quantity', 0),
                        total_sell_quantity=q.get('sell_quantity', 0),
                        average_price=q.get('average_price', 0.0),
                        oi_day_high=q.get('oi_day_high', 0),
                        oi_day_low=q.get('oi_day_low', 0)
                    )
            except Exception as e:
                logger.error(f"Quote batch failed: {e}")
                
        return results

    def fetch_quotes_by_symbols(self, symbols: List[str], exchange: str = "NSE") -> Dict[str, FullQuote]:
        """
        Batch-fetch live quotes for multiple symbols in minimal API calls.
        Returns {symbol: FullQuote} mapping.
        """
        # Resolve all tokens at once using O(1) lookups
        sym_token_map = {}
        for sym in symbols:
            token = self.instruments.get_instrument_token(sym, exchange)
            if token:
                sym_token_map[sym] = token
        
        if not sym_token_map:
            return {}
        
        # Single batched fetch (handles chunking internally)
        token_quotes = self.fetch_quotes(list(sym_token_map.values()))
        
        # Map back to symbols
        token_to_sym = {v: k for k, v in sym_token_map.items()}
        return {token_to_sym[t]: q for t, q in token_quotes.items() if t in token_to_sym}


# ═══════════════════════════════════════════════════════════════════════════════
# §4  NATIVE MARGIN CALCULATOR (The "Source of Truth")
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarginResult:
    initial_margin: float
    exposure_margin: float
    total_margin: float
    span_margin: float
    option_premium: float
    leverage_benefit: float

class MarginCalculator:
    """Interfaces with Kite's Order Margins API to get exact capital requirements."""
    
    def __init__(self, session: KiteSession):
        self.session = session

    def fetch_strategy_margin(self, legs: List[Dict], spot_price: float = 0) -> Optional[MarginResult]:
        """
        Calculate margin for a multi-leg strategy using Kite API.
        
        legs: List of dicts with keys:
            - type: 'Buy Call', 'Sell Put', etc.
            - strike: float
            - qty: int
            - premium: float (for premium calculations)
            - instrument_token: int (Optional but recommended)
            - tradingsymbol: str
            - exchange: 'NFO'
        """
        basket = []
        total_premium_credit = 0.0
        
        for leg in legs:
            # Parse transaction type
            txn_type = "BUY" if "Buy" in leg['type'] else "SELL"
            
            # Construct basket item
            item = {
                "exchange": leg.get("exchange", "NFO"),
                "tradingsymbol": leg["tradingsymbol"],
                "transaction_type": txn_type,
                "variety": "regular",
                "product": "NRML", # Overnight strategy
                "order_type": "MARKET",
                "quantity": abs(leg["qty"])
            }
            basket.append(item)
            
            # Track premium flow (approximate based on last price in leg)
            if txn_type == "SELL":
                total_premium_credit += leg.get("premium", 0) * abs(leg["qty"])
            else:
                total_premium_credit -= leg.get("premium", 0) * abs(leg["qty"])

        try:
            # Call Kite Margin API
            response = self.session.post("/margins/basket", basket)
            
            if not response:
                return None
                
            # Extract Compact Margin details
            initial = response.get('initial', {}).get('total', 0)
            exposure = response.get('orders', [{}])[0].get('exposure', 0) # simplified
            
            # Deep parse: Sum up margin components from API response list
            total_initial = sum(item.get('initial', {}).get('total', 0) for item in response)
            total_exposure = sum(item.get('exposure', 0) for item in response)
            total_span = sum(item.get('initial', {}).get('span', 0) for item in response)
            total_req = total_initial + total_exposure
            
            # Benefit calculation (Portfolio margin vs Sum of Naked)
            # This requires a second call with separate legs, but we skip for performance
            
            return MarginResult(
                initial_margin=total_initial,
                exposure_margin=total_exposure,
                total_margin=total_req,
                span_margin=total_span,
                option_premium=total_premium_credit,
                leverage_benefit=0.0 
            )

        except Exception as e:
            logger.error(f"Margin fetch failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# §5  OPTION CHAIN (Fully Synced)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptionQuote:
    """Rich option quote with Greeks and Depth."""
    instrument_token: int
    tradingsymbol: str
    strike: float
    option_type: str  # CE/PE
    expiry: date
    last_price: float
    bid: float
    ask: float
    bid_qty: int  # Top bid quantity for liquidity check
    ask_qty: int
    volume: int
    oi: int
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    # Liquidity & Depth Data
    total_buy_qty: int = 0
    total_sell_qty: int = 0
    bid_ask_spread: float = 0.0
    spread_pct: float = 0.0
    liquidity_score: float = 0.0  # 0 to 100
    is_liquid: bool = False

@dataclass
class OptionChain:
    symbol: str
    underlying_price: float  # Spot or Future
    expiry: date
    calls: List[OptionQuote]
    puts: List[OptionQuote]
    timestamp: datetime

class OptionChainFetch:
    def __init__(self, session: KiteSession, instruments: InstrumentCache, market: MarketData):
        self.session = session
        self.instruments = instruments
        self.market = market

    def fetch(self, symbol: str, expiry: date) -> Optional[OptionChain]:
        # 1. Get Instruments
        opts = self.instruments.get_option_instruments(symbol, expiry)
        if opts.empty: return None

        # 2. Identify Underlying Token (Spot)
        spot_token = self.instruments.get_underlying_token(symbol)
        
        # 3. Collect Tokens to Fetch (Options + Spot)
        tokens = opts['instrument_token'].astype(int).tolist()
        if spot_token: tokens.append(spot_token)
        
        # 4. Fetch All Quotes in Parallel Batch
        quotes = self.market.fetch_quotes(tokens)
        
        # 5. Extract Underlying Price
        underlying_price = 0
        if spot_token and spot_token in quotes:
            underlying_price = quotes[spot_token].last_price
        
        if underlying_price == 0:
            # Fallback to ATM estimate from option pairs if spot missing
            logger.warning(f"Spot price missing for {symbol}, using fallback.")
            underlying_price = 0 # Will handle gracefully later or fail

        # 6. Build Option Objects
        calls, puts = [], []
        
        # Map tokens back to details
        token_map = opts.set_index('instrument_token')[['tradingsymbol', 'strike', 'instrument_type']].to_dict('index')

        for token, q in quotes.items():
            if token == spot_token: continue
            
            details = token_map.get(token)
            if not details: continue
            
            # Use top of book for bid/ask
            best_bid = q.bids[0].price if q.bids else 0
            best_ask = q.asks[0].price if q.asks else 0
            bid_qty = q.bids[0].quantity if q.bids else 0
            ask_qty = q.asks[0].quantity if q.asks else 0
            
            # Fallback to LTP if no depth
            if best_bid == 0: best_bid = q.last_price
            if best_ask == 0: best_ask = q.last_price

            opt = OptionQuote(
                instrument_token=token,
                tradingsymbol=details['tradingsymbol'],
                strike=float(details['strike']),
                option_type=details['instrument_type'],
                expiry=expiry,
                last_price=q.last_price,
                bid=best_bid,
                ask=best_ask,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                volume=q.volume,
                oi=q.oi,
                total_buy_qty=q.total_buy_quantity,
                total_sell_qty=q.total_sell_quantity
            )
            
            if details['instrument_type'] == 'CE':
                calls.append(opt)
            else:
                puts.append(opt)
        
        # 7. Compute IVs (Locally, but using precise live prices)
        chain = OptionChain(symbol, underlying_price, expiry, calls, puts, datetime.now())
        if underlying_price > 0:
            self._compute_greeks(chain)
            self._compute_liquidity(chain)
            
        return chain

    def _compute_liquidity(self, chain: OptionChain):
        """Evaluate and score the liquidity of each strike based on Kite depth and volume."""
        for opt in chain.calls + chain.puts:
            # Calculate Spread
            if opt.bid > 0 and opt.ask > 0:
                opt.bid_ask_spread = round(opt.ask - opt.bid, 2)
                mid_price = (opt.ask + opt.bid) / 2
                opt.spread_pct = round(opt.bid_ask_spread / mid_price, 4) if mid_price > 0 else 0.0
            else:
                opt.bid_ask_spread = 999.0
                opt.spread_pct = 9.99

            # Liquidity Heuristics:
            # High Volume, High OI, Tight Spread, High Total Bids/Asks = Highly Liquid
            score = 0.0
            
            # 1. Spread component (max 40 pts) - < 2% spread is excellent
            if 0 < opt.spread_pct <= 0.02: score += 40
            elif opt.spread_pct <= 0.05: score += 25
            elif opt.spread_pct <= 0.10: score += 10

            # 2. Volume component (max 30 pts)
            if opt.volume > 10000: score += 30
            elif opt.volume > 1000: score += 20
            elif opt.volume > 100: score += 10
            
            # 3. Open Interest component (max 15 pts)
            if opt.oi > 10000: score += 15
            elif opt.oi > 1000: score += 10
            elif opt.oi > 100: score += 5
            
            # 4. Order Book Depth component (max 15 pts)
            if opt.total_buy_qty > 1000 and opt.total_sell_qty > 1000: score += 15
            elif opt.total_buy_qty > 100 and opt.total_sell_qty > 100: score += 7
            
            opt.liquidity_score = score
            
            # Binary viability flag for the engine to easily filter out junk/illiquid strikes
            opt.is_liquid = (
                opt.bid > 0 and 
                opt.ask > 0 and 
                opt.spread_pct <= 0.15 and 
                (opt.volume > 50 or opt.oi > 100)
            )

    def _compute_greeks(self, chain: OptionChain):
        """Compute IV and Greeks using vectorized numpy — no per-option loops."""
        from scipy.stats import norm
        S = chain.underlying_price
        T = max((chain.expiry - date.today()).days / 365.0, 1/365)
        r = 0.07
        sqrtT = np.sqrt(T)

        def _vectorized_iv_greeks(options: list, otype: str):
            """Compute IV + Greeks for all options at once using vectorized bisection."""
            if not options:
                return
            
            strikes = np.array([o.strike for o in options])
            mids = np.array([
                (o.bid + o.ask) / 2 if o.bid > 0 and o.ask > 0 else o.last_price
                for o in options
            ])
            n = len(options)
            
            # Vectorized bisection for IV (20 iterations)
            low = np.full(n, 0.01)
            high = np.full(n, 5.0)
            
            for _ in range(20):
                mid_vol = (low + high) / 2
                d1 = (np.log(S / strikes) + (r + 0.5 * mid_vol**2) * T) / (mid_vol * sqrtT)
                d2 = d1 - mid_vol * sqrtT
                if otype == 'CE':
                    prices = S * norm.cdf(d1) - strikes * np.exp(-r * T) * norm.cdf(d2)
                else:
                    prices = strikes * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
                too_low = prices < mids
                low = np.where(too_low, mid_vol, low)
                high = np.where(~too_low, mid_vol, high)
            
            ivs = (low + high) / 2
            ivs = np.where(mids < 0.05, 0.0, ivs)
            
            # Vectorized Greeks
            d1 = (np.log(S / strikes) + (r + 0.5 * ivs**2) * T) / (ivs * sqrtT + 1e-10)
            d2 = d1 - ivs * sqrtT
            
            if otype == 'CE':
                deltas = norm.cdf(d1)
                thetas = (-S * norm.pdf(d1) * ivs / (2 * sqrtT) - r * strikes * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                deltas = norm.cdf(d1) - 1
                thetas = (-S * norm.pdf(d1) * ivs / (2 * sqrtT) - r * strikes * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            gammas = norm.pdf(d1) / (S * ivs * sqrtT + 1e-10)
            vegas = S * sqrtT * norm.pdf(d1) / 100
            
            # Assign back to option objects
            for i, opt in enumerate(options):
                opt.iv = float(ivs[i])
                if opt.iv > 0:
                    opt.delta = float(deltas[i])
                    opt.gamma = float(gammas[i])
                    opt.theta = float(thetas[i])
                    opt.vega = float(vegas[i])

        _vectorized_iv_greeks(chain.calls, 'CE')
        _vectorized_iv_greeks(chain.puts, 'PE')


# ═══════════════════════════════════════════════════════════════════════════════
# §6  MASTER PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class KiteDataPipeline:
    """
    The Single Source of Truth for Vaaydo.
    Orchestrates authentication, instrument lookup, live data, and margin calculations.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", access_token: str = ""):
        self.session = KiteSession(api_key, api_secret, access_token)
        self.instruments = InstrumentCache(self.session)
        self.market = MarketData(self.session, self.instruments)
        self.margins = MarginCalculator(self.session)
        self.chains = OptionChainFetch(self.session, self.instruments, self.market)
        self._lot_sizes = {}

    def initialize(self):
        """Warm up caches."""
        self.instruments.load()
        self._lot_sizes = self.instruments.get_lot_sizes()
        return f"✓ Pipeline Active: {len(self._lot_sizes)} F&O instruments loaded."

    def is_connected(self):
        return self.session.is_valid()

    def get_fno_symbols(self):
        return self.instruments.get_fno_symbols()
    
    def get_lot_size(self, symbol):
        return self._lot_sizes.get(symbol, 1)

    # ── High Level Data Fetching ──

    def fetch_all_data(self, symbols: List[str], days_back: int = 400) -> Tuple[pd.DataFrame, str]:
        """
        Main analytics entry point — optimized for speed.
        Produces the SAME DataFrame schema as app.py's yfinance fetch_all_data.
        1. Concurrent historical fetches via ThreadPoolExecutor
        2. Single batched live quote call for all symbols
        3. Full analytics computation (RV, GARCH, RSI, ADX, Kalman, CUSUM, PCR)
        """
        import time as _time
        t0 = _time.time()
        
        # 1. Fetch all history concurrently (5 threads, rate-limited internally)
        history = self.market.fetch_ohlcv_batch(symbols, days_back, max_workers=5)
        t1 = _time.time()
        logger.info(f"Historical fetch: {len(history)}/{len(symbols)} symbols in {t1-t0:.1f}s")
        
        if not history:
            return pd.DataFrame(), "No historical data fetched"
        
        # 2. Single batched live quote fetch for ALL symbols at once
        live_quotes = self.market.fetch_quotes_by_symbols(list(history.keys()))
        t2 = _time.time()
        logger.info(f"Live quotes: {len(live_quotes)} symbols in {t2-t1:.1f}s")
        
        # 3. Compute full analytics per symbol (matching yfinance path schema)
        results = []
        for sym, df in history.items():
            try:
                if len(df) < 60:
                    continue
                
                O, H, L, Cl, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
                price = float(Cl.iloc[-1])
                lr = np.log(Cl / Cl.shift(1)).dropna()
                
                # ── Multi-Estimator Volatility ──
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
                current_rv = max(current_rv, 0.05)
                
                # ── IV Estimation with VRP ──
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
                
                # ── Technical Analysis ──
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
                
                # Kalman Filter
                kalman_price = price
                kalman_var = atr_val**2
                R_noise = (price * 0.01)**2
                Q_proc = (price * 0.002)**2
                for p_val in Cl.values[-20:]:
                    if np.isnan(p_val): continue
                    pred_var = kalman_var + Q_proc
                    K_gain = pred_var / (pred_var + R_noise)
                    kalman_price = kalman_price + K_gain * (p_val - kalman_price)
                    kalman_var = (1 - K_gain) * pred_var
                kalman_trend = (price - kalman_price) / max(atr_val, 0.01)
                
                # Use live quote volume if available, else last bar
                live_q = live_quotes.get(sym)
                vol_curr = live_q.volume if live_q else float(V.iloc[-1]) if not np.isnan(V.iloc[-1]) else 0
                vol20 = float(V.rolling(20).mean().iloc[-1]) if len(V) >= 20 else max(vol_curr, 1)
                
                # PCR (volume-based proxy)
                up_v = V.where(Cl > Cl.shift(1), 0).rolling(20).sum()
                dn_v = V.where(Cl < Cl.shift(1), 0).rolling(20).sum()
                pcr = float(dn_v.iloc[-1] / max(up_v.iloc[-1], 1)) if len(up_v) >= 20 else 1.0
                pcr = min(max(pcr, 0.2), 3.0)
                pct_change = float(Cl.pct_change().iloc[-1] * 100) if len(Cl) >= 2 else 0.0
                
                # CUSUM
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
                    return round(float(v) * 100, 2) if not np.isnan(v) else 0.0
                
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
                    'lot_size': self.get_lot_size(sym),
                    'RV_C2C': safe_rv(rv_c2c), 'RV_Parkinson': safe_rv(rv_park),
                    'RV_GK': safe_rv(rv_gk), 'RV_YZ': safe_rv(rv_yz),
                })
            except Exception as e:
                logger.error(f"Analytics calc failed for {sym}: {e}")
                continue

        t3 = _time.time()
        if not results:
            return pd.DataFrame(), "No valid data extracted"
        logger.info(f"Total pipeline: {len(results)} symbols in {t3-t0:.1f}s")
        return pd.DataFrame(results), f"✓ Analytics for {len(results)} symbols via Kite ({t3-t0:.1f}s)"

    # ── Option Chain & Execution Support ──

    def fetch_option_chain(self, symbol: str, expiry: date) -> Optional[OptionChain]:
        """Get live chain with Greeks."""
        return self.chains.fetch(symbol, expiry)

    def calculate_strategy_margin(self, strategy_legs: List[Dict]) -> Optional[MarginResult]:
        """Get Source-of-Truth margins from Kite."""
        return self.margins.fetch_strategy_margin(strategy_legs)

    def place_gtt_batch(self, orders: List[Dict]):
        """
        Execution Readiness: Helper to place batch GTT orders.
        This would be the entry point for auto-execution.
        """
        # Placeholder for execution logic
        pass

# ── Session Helpers ──

def _sanitize_totp_key(raw_key: str) -> str:
    """
    Clean a TOTP secret for pyotp compatibility.
    - Strip whitespace, dashes, and newlines
    - Uppercase (base32 is A-Z, 2-7 only)
    - Pad to multiple of 8 with '='
    - Validate that only base32 chars remain
    """
    key = raw_key.strip().replace(" ", "").replace("-", "").replace("\n", "").upper()
    # Remove any padding first, then re-pad correctly
    key = key.rstrip("=")
    pad = len(key) % 8
    if pad:
        key += "=" * (8 - pad)
    # Validate base32 charset
    import re as _re
    clean = key.rstrip("=")
    if not _re.fullmatch(r'[A-Z2-7]*', clean):
        raise ValueError(
            f"TOTP key contains invalid characters. "
            f"Base32 only allows A-Z and 2-7. "
            f"Check your KITE_TOTP_KEY in secrets — it should be the raw key from "
            f"Zerodha's 'Can't scan? Copy key' during TOTP setup."
        )
    return key


def get_request_token(credentials: dict) -> str:
    """
    Automated Kite Connect login flow:
      1. GET login URL → establish session cookies
      2. POST /api/login → get request_id
      3. POST /api/twofa with pyotp TOTP → complete 2FA
      4. GET login URL + &skip_session=true → follow redirects → parse request_token
      5. Return request_token for session generation
    
    Ref: https://kite.trade/docs/connect/v3/user/
    """
    for field in ("api_key", "username", "password", "totp_key"):
        if not credentials.get(field, "").strip():
            raise ValueError(f"Missing required credential: {field}")

    api_key = credentials["api_key"]
    session = requests.Session()
    
    # Step 1: GET the Kite login page to establish session cookies
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
    session.get(url=login_url)
    
    # Step 2: POST login credentials
    resp = session.post("https://kite.zerodha.com/api/login", data={
        "user_id": credentials["username"],
        "password": credentials["password"]
    })
    login_data = resp.json()
    if login_data.get("status") != "success":
        raise ConnectionError(f"Kite login failed: {login_data.get('message', 'Unknown error')}")
    request_id = login_data["data"]["request_id"]
    
    # Step 3: POST 2FA using pyotp with sanitized TOTP key
    clean_totp_key = _sanitize_totp_key(credentials["totp_key"])
    totp_value = pyotp.TOTP(clean_totp_key).now()
    resp = session.post("https://kite.zerodha.com/api/twofa", data={
        "user_id": credentials["username"],
        "request_id": request_id,
        "twofa_value": totp_value,
        "twofa_type": "totp"
    })
    twofa_data = resp.json()
    if twofa_data.get("status") != "success":
        raise ConnectionError(f"Kite 2FA failed: {twofa_data.get('message', 'Check TOTP key')}")
    
    # Step 4: GET login URL with skip_session → redirects to registered URL with request_token
    redirect_url = login_url + "&skip_session=true"
    resp = session.get(url=redirect_url, allow_redirects=True)
    parsed = parse_qs(urlparse(resp.url).query)
    if "request_token" not in parsed:
        raise ConnectionError(
            f"No request_token in redirect URL. "
            f"Check that your redirect URL is configured in the Kite developer console. "
            f"Got: {resp.url[:120]}"
        )
    return parsed["request_token"][0]

def render_kite_login(sidebar=True):
    import streamlit as st
    
    def _ui():
        st.markdown('<div class="stitle">🔗 Kite Connect v2</div>', unsafe_allow_html=True)
        
        if 'kite_pipeline' in st.session_state and st.session_state.kite_pipeline.is_connected():
            st.success("✅ Connected")
            return st.session_state.kite_pipeline, True

        with st.expander("Credentials", expanded=True):
            # Pull from Streamlit secrets if available, otherwise show text inputs
            _ak_secret = st.secrets.get("KITE_API_KEY", "")
            _ask_secret = st.secrets.get("KITE_API_SECRET", "")
            
            if _ak_secret and _ask_secret:
                ak, ask = _ak_secret, _ask_secret
                st.success("🔑 API credentials loaded from secrets")
            else:
                ak = st.text_input("API Key", type="password")
                ask = st.text_input("API Secret", type="password")
            
            mode = st.radio("Login Method", ["Manual Token", "Auto-Login (TOTP)"])
            
            if mode == "Manual Token":
                if ak:
                    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={ak}"
                    st.markdown(
                        f"**Step 1:** [Open Kite Login]({login_url}) → login → "
                        f"copy `request_token` from the redirect URL"
                    )
                rt = st.text_input("Step 2: Paste Request Token")
                if st.button("Connect"):
                    if not ak or not ask:
                        st.error("API Key and API Secret are required.")
                    elif not rt or len(rt.strip()) < 10:
                        st.error("Request Token must be at least 10 characters. Complete the Kite login flow first to obtain it.")
                    else:
                        try:
                            ks = KiteSession(ak, ask)
                            ks.generate_session(rt.strip())
                            kp = KiteDataPipeline(ak, ask, ks.access_token)
                            kp.initialize()
                            st.session_state.kite_pipeline = kp
                            st.rerun()
                        except Exception as e:
                            st.error(f"Connection failed: {e}")
            else:
                # Pull auto-login creds from secrets if available
                _uid_secret = st.secrets.get("KITE_USER_ID", "")
                _pwd_secret = st.secrets.get("KITE_PASSWORD", "")
                _totp_secret = st.secrets.get("KITE_TOTP_KEY", "")
                
                if _uid_secret and _pwd_secret and _totp_secret:
                    uid, pwd, totp = _uid_secret, _pwd_secret, _totp_secret
                    st.success("🔑 Login credentials loaded from secrets")
                else:
                    uid = st.text_input("User ID", value=_uid_secret)
                    pwd = st.text_input("Password", type="password", value=_pwd_secret)
                    totp = st.text_input("TOTP Key", type="password", value=_totp_secret)
                
                if st.button("Auto Connect"):
                    if not all([ak, ask, uid, pwd, totp]):
                        st.error("All fields are required for Auto-Login.")
                    else:
                        try:
                            rt = get_request_token({"api_key": ak, "username": uid, "password": pwd, "totp_key": totp})
                            ks = KiteSession(ak, ask)
                            ks.generate_session(rt)
                            kp = KiteDataPipeline(ak, ask, ks.access_token)
                            kp.initialize()
                            st.session_state.kite_pipeline = kp
                            st.rerun()
                        except Exception as e:
                            st.error(f"Auto-login failed: {e}")
        return None, False

    if sidebar:
        with st.sidebar: return _ui()
    return _ui()
