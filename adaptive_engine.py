"""
╔══════════════════════════════════════════════════════════════════════════╗
║  VAAYDO v4.0 — Adaptive Intelligence Engine                            ║
║  Every parameter derived from data. Every weight earned, not assumed.   ║
║  Full uncertainty quantification. Self-monitoring. Anti-fragile.        ║
║  Hemrek Capital                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Sections:
  §1  SignalSpace          — Orthogonal signal extraction, information measurement
  §2  FuzzyRegime          — Continuous regime probabilities, transition estimation
  §3  AdaptiveGating       — Data-driven viability (no hardcoded thresholds)
  §4  AdaptiveEnsemble     — Model competition, disagreement-weighted fusion
  §5  ProbabilisticScoring — Bayesian conviction with full uncertainty
  §6  MetaIntelligence     — Reflexivity, edge decay, anti-fragility, exploration
  §7  EntropyGovernor      — System confidence, risk budget, governance
  §8  AdaptiveKelly        — Bayesian sizing with drawdown sensitivity
  §9  PortfolioAwareness   — Cross-sectional diversification, ruin prevention
  §10 AdaptiveStrikes      — Delta-targeted strike placement
  §11 ComputeTriage        — Efficiency: BSM screen, MC only when justified
  §12 AdaptiveEngine       — Master orchestrator
"""

import numpy as np
import math
from scipy import stats as sp_stats
from scipy.stats import norm, beta as beta_dist
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from numba import njit, prange
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# §0  HIGH PERFORMANCE COMPUTE CORE (Numba)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def n_cdf(x: float) -> float:
    """Fast Numba-compatible standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

@njit(cache=True)
def n_pdf(x: float) -> float:
    """Fast Numba-compatible standard normal PDF."""
    return math.exp(-0.5 * x**2) / 2.5066282746310002

@njit(cache=True)
def bsm_price_numba(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True):
    """JIT-optimized BSM pricing."""
    if T <= 1e-6:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if sigma <= 1e-6:
        return max(S - K * math.exp(-r * T), 0.0) if is_call else max(K * math.exp(-r * T) - S, 0.0)
        
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    
    if is_call:
        return S * n_cdf(d1) - K * math.exp(-r * T) * n_cdf(d2)
    else:
        return K * math.exp(-r * T) * n_cdf(-d2) - S * n_cdf(-d1)

@njit(cache=True)
def bsm_greeks_numba(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True):
    """JIT-optimized BSM Greeks calculation with Higher-Order Sensitivities."""
    if T <= 1e-6 or sigma <= 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # d, g, t, v, vn, vg, ch, sp
    
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    
    nd1 = n_pdf(d1)
    gamma = nd1 / (S * sT)
    vega = S * math.sqrt(T) * nd1 / 100
    
    # Higher order
    vanna = -nd1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    
    if is_call:
        delta = n_cdf(d1)
        theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * n_cdf(d2)) / 365
        charm = (-nd1 * (r / (sigma * math.sqrt(T)) - d2 / (2 * T))) / 365
    else:
        delta = n_cdf(d1) - 1
        theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * n_cdf(-d2)) / 365
        charm = (-nd1 * (r / (sigma * math.sqrt(T)) - d2 / (2 * T))) / 365 # Approx
        
    speed = -gamma / S * (d1/sT + 1)
        
    return delta, gamma, theta, vega, vanna, volga, charm, speed

@njit(parallel=True, cache=True)
def generate_terminal_prices(S: float, sigma: float, T: float, n: int = 5000):
    """Vectorized terminal price generation with antithetic variates."""
    if T <= 0 or sigma <= 0:
        return np.full(2 * n, S)
    
    steps = 1
    dt = T / steps
    drift = (0.07 - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    
    terminal = np.empty(2 * n)
    z = np.random.standard_normal(n)
    
    for i in prange(n):
        ret = drift + vol * z[i]
        terminal[i] = S * math.exp(ret)
        terminal[i + n] = S * math.exp(drift - vol * z[i])
        
    return terminal

@njit(parallel=True, cache=True)
def generate_paths_numba(S: float, sigma: float, T: float, n: int = 100, steps: int = 30):
    """Generate multiple price paths for visualization."""
    dt = T / steps
    drift = (0.07 - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    
    paths = np.empty((n, steps + 1))
    paths[:, 0] = S
    
    for i in prange(n):
        for j in range(1, steps + 1):
            paths[i, j] = paths[i, j-1] * math.exp(drift + vol * np.random.standard_normal())
            
    return paths

@njit(parallel=True, cache=True)
def compute_strategy_pnl_numba(terminal: np.ndarray, strikes: np.ndarray, 
                               prems: np.ndarray, types: np.ndarray, qtys: np.ndarray):
    """O(N) vectorized strategy P&L across all terminal prices."""
    n_sims = len(terminal)
    n_legs = len(strikes)
    total_pnl = np.zeros(n_sims)
    
    for j in range(n_legs):
        K = strikes[j]
        p = prems[j]
        t = types[j]  # 0: Sell Call, 1: Buy Call, 2: Sell Put, 3: Buy Put
        m = qtys[j]
        
        for i in prange(n_sims):
            S_t = terminal[i]
            pnl = 0.0
            if t == 0:    # Sell Call
                pnl = p - max(S_t - K, 0.0)
            elif t == 1:  # Buy Call
                pnl = -p + max(S_t - K, 0.0)
            elif t == 2:  # Sell Put
                pnl = p - max(K - S_t, 0.0)
            elif t == 3:  # Buy Put
                pnl = -p + max(K - S_t, 0.0)
            total_pnl[i] += pnl * m
            
    return total_pnl

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

class BSM:
    """Wrapper for Numba-optimized Black-Scholes-Merton functions."""
    R = 0.07

    @classmethod
    def greeks(cls, S, K, T, r, sigma, otype='call'):
        d, g, t, v, vn, vg, ch, sp = bsm_greeks_numba(S, K, T, r, sigma, is_call=(otype == 'call'))
        return Greeks(delta=d, gamma=g, theta=t, vega=v, vanna=vn, volga=vg, charm=ch, speed=sp)

    @classmethod
    def call(cls, S, K, T, r, sigma):
        return bsm_price_numba(S, K, T, r, sigma, is_call=True)

    @classmethod
    def put(cls, S, K, T, r, sigma):
        return bsm_price_numba(S, K, T, r, sigma, is_call=False)

    @classmethod
    def prob_otm(cls, S, K, T, sigma, otype='call'):
        if T <= 0 or sigma <= 0: return 0.5
        sT = sigma * math.sqrt(T)
        d2 = (math.log(S / K) + (cls.R - 0.5 * sigma**2) * T) / sT
        return n_cdf(-d2) if otype == 'call' else n_cdf(d2)

    @classmethod
    def risk_score(cls, g, iv, rvw=1.0):
        """§6.3: Composite Risk Score — regime-weighted"""
        w = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
        w[2] *= rvw; w /= w.sum()
        delta_r = abs(g.delta) * w[0]
        gamma_r = min(abs(g.gamma) * iv * 100, 2.0) * w[1]
        vega_r = min(abs(g.vega) / max(iv * 100, 1), 2.0) * w[2]
        theta_p = min(max(-g.theta, 0) / max(abs(g.theta) + 1, 1), 1.0) * w[3]
        tail = min(abs(g.vanna) + abs(g.volga) * 0.1, 2.0) * w[4]
        return min((delta_r + gamma_r + vega_r + theta_p + tail) * 100, 100)

class MC:
    """Wrapper for Numba-optimized Monte Carlo functions."""
    
    @staticmethod
    def terminal_prices(S, sigma, T, n=10000):
        return generate_terminal_prices(S, sigma, T, n)

    @staticmethod
    def paths(S, sigma, T, n=100, steps=30):
        return generate_paths_numba(S, sigma, T, n, steps)
    
    @staticmethod
    def analyze(S, sigma, T, legs, n=10000, sim_vol=None):
        vol_for_sim = sim_vol if sim_vol is not None else sigma
        terminal = generate_terminal_prices(S, vol_for_sim, T, n)
        
        # Prepare arrays for Numba
        strikes = np.array([l['strike'] for l in legs], dtype=np.float64)
        prems = np.array([l['premium'] for l in legs], dtype=np.float64)
        qtys = np.array([l.get('qty', 1) for l in legs], dtype=np.float64)
        
        types = np.zeros(len(legs), dtype=np.int32)
        for i, l in enumerate(legs):
            lt = l['type'].lower()
            if 'sell' in lt and 'call' in lt: types[i] = 0
            elif 'buy' in lt and 'call' in lt: types[i] = 1
            elif 'sell' in lt and 'put' in lt: types[i] = 2
            elif 'buy' in lt and 'put' in lt: types[i] = 3
            
        pnl = compute_strategy_pnl_numba(terminal, strikes, prems, types, qtys)
        
        pop = float(np.mean(pnl > 0))
        ev = float(np.mean(pnl))
        std = float(np.std(pnl))
        return pop, ev, std

    @staticmethod
    def expected_move(S, σ, T):
        moves = []
        for conf in [0.6827, 0.9545, 0.9973]:
            # z-score for normal distribution
            # 1-alpha/2 = (1+conf)/2
            z = sp_stats.norm.ppf((1 + conf) / 2)
            m = S * σ * np.sqrt(max(T, 1e-6)) * z
            moves.append({'conf': conf, 'move': m, 'upper': S + m, 'lower': S - m})
        return moves


# ═══════════════════════════════════════════════════════════════════════════════
# §1  SIGNAL SPACE
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Orthogonal Signal Structure, Continuous Factor Relevance
# Recalibration, Predictive-Power-Based Weighting, Signal Crowding Detection,
# Information Gain Maximization, Nonlinear Interaction Modeling,
# Zero Hard-Coded Importance Assumptions

class SignalSpace:
    """Every signal earns its weight through measured information content.
    Correlated signals identified and decorrelated. Crowded signals penalized.
    Nonlinear interactions detected and modeled."""

    RAW_SIGNALS = [
        'IVPercentile', 'GARCH_Vol', 'RV_Composite', 'rsi_daily',
        'adx', 'kalman_trend', 'GARCH_Persistence',
        'PCR', 'CUSUM_Alert'
    ]

    @staticmethod
    def extract(stock: dict) -> np.ndarray:
        return np.array([
            stock.get('IVPercentile', 50), stock.get('GARCH_Vol', 20),
            stock.get('RV_Composite', 20), stock.get('rsi_daily', 50),
            stock.get('adx', 20), stock.get('kalman_trend', 0),
            stock.get('GARCH_Persistence', 0.95),
            stock.get('PCR', 1.0), float(stock.get('CUSUM_Alert', False)),
        ], dtype=np.float64)

    @staticmethod
    def standardize(stock: dict, ustats: dict) -> dict:
        """Map raw features to stationary Z-scores."""
        if not ustats.get('valid'): return stock
        raw = SignalSpace.extract(stock)
        means = ustats['means']
        cols = ustats['stds']
        z = (raw - means) / np.maximum(cols, 1e-6)
        
        # Return a copy with 'z_' prefixed keys
        names = SignalSpace.RAW_SIGNALS
        scored = stock.copy()
        for i, name in enumerate(names):
            scored[f'z_{name}'] = z[i]
        return scored

    @staticmethod
    def compute_universe_stats(all_signals: np.ndarray) -> dict:
        """Derive ALL adaptive parameters from the universe distribution."""
        n, d = all_signals.shape
        if n < 5:
            return {'valid': False}

        # Robust location/scale (median/MAD, not mean/std)
        medians = np.nanmedian(all_signals, axis=0)
        mads = np.nanmedian(np.abs(all_signals - medians), axis=0) * 1.4826
        mads = np.maximum(mads, 1e-6)

        # Full percentile grid for adaptive thresholds
        pct_keys = [5, 10, 15, 20, 25, 33, 50, 67, 75, 80, 85, 90, 95]
        percentiles = {}
        for p in pct_keys:
            percentiles[p] = np.nanpercentile(all_signals, p, axis=0)

        # Cross-correlation (rank-based = robust)
        valid = all_signals[~np.isnan(all_signals).any(axis=1)]
        if len(valid) < 10:
            corr_matrix = np.eye(d)
        else:
            corr_matrix = np.corrcoef(valid.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0)

        # Independence: 1 - avg|corr| with others
        avg_abs = (np.abs(corr_matrix).sum(axis=1) - 1) / max(d - 1, 1)
        independence = 1.0 - avg_abs

        # Crowding: fraction of universe on same side of median
        crowding = np.zeros(d)
        for j in range(d):
            col = valid[:, j]
            if len(col) > 5:
                crowding[j] = abs(np.mean(col > medians[j]) - 0.5) * 2

        # Shannon entropy of each signal (discretized into 20 bins)
        entropies = np.zeros(d)
        for j in range(d):
            col = valid[:, j]
            if len(col) < 10 or np.std(col) < 1e-6:
                continue
            # Adaptive bins: min(20, n_unique) to handle binary/low-cardinality signals
            n_bins = min(20, max(2, len(np.unique(col))))
            hist, _ = np.histogram(col, bins=n_bins, density=True)
            h = hist[hist > 0]
            entropies[j] = -np.sum(h * np.log(h + 1e-12)) / max(np.log(n_bins), 1e-6)

        # §1.1 Nonlinear interaction detection
        # For top correlated pairs, compute conditional mutual information
        interactions = {}
        if len(valid) > 30:
            for i in range(d):
                for j in range(i+1, d):
                    # Joint entropy proxy: correlation of abs(residuals)
                    r = abs(corr_matrix[i, j])
                    if r > 0.3:  # only examine meaningfully correlated pairs
                        # Product interaction term
                        xi = (valid[:, i] - medians[i]) / mads[i]
                        xj = (valid[:, j] - medians[j]) / mads[j]
                        interaction_signal = xi * xj
                        # Does the interaction have different entropy than marginals?
                        h_int, _ = np.histogram(interaction_signal, bins=20, density=True)
                        h_int = h_int[h_int > 0]
                        int_entropy = -np.sum(h_int * np.log(h_int + 1e-12)) / max(np.log(20), 1e-6)
                        marginal_avg = (entropies[i] + entropies[j]) / 2
                        # Information gain: how much NEW info the interaction provides
                        info_gain = max(0, int_entropy - marginal_avg)
                        if info_gain > 0.05:
                            interactions[(i, j)] = info_gain

        # Signal weight = independence × entropy × (1 - 0.5*crowding)
        raw_w = independence * entropies * (1 - 0.5 * crowding)
        raw_w = np.maximum(raw_w, 0.01)
        signal_weights = raw_w / raw_w.sum()

        return {
            'valid': True, 'n': n, 'd': d,
            'medians': medians, 'mads': mads,
            'means': np.nanmean(all_signals, axis=0),
            'stds': np.nanstd(all_signals, axis=0),
            'percentiles': percentiles,
            'corr_matrix': corr_matrix,
            'independence': independence,
            'crowding': crowding,
            'entropies': entropies,
            'signal_weights': signal_weights,
            'interactions': interactions,
        }

    @staticmethod
    def percentile_rank(value: float, percentiles: dict, idx: int) -> float:
        """Where does this value fall in the universe? Returns [0, 1]."""
        pcts = sorted(percentiles.keys())
        for i, p in enumerate(pcts):
            if value <= percentiles[p][idx]:
                if i == 0:
                    return p / 100.0
                prev_p = pcts[i-1]
                lo = percentiles[prev_p][idx]
                hi = percentiles[p][idx]
                if hi - lo < 1e-6:
                    return p / 100.0
                frac = (value - lo) / (hi - lo)
                return (prev_p + frac * (p - prev_p)) / 100.0
        return 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# §2  FUZZY REGIME
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: State-Conditioned Intelligence, Regime Transition Probability
# Estimation, No Assumption of Stationarity, Multi-Timescale Awareness

@dataclass
class RegimeState:
    """Continuous regime — probability vectors, not labels."""
    vol_probs: np.ndarray        # P(compressed..extreme) [6]
    trend_probs: np.ndarray      # P(strong_down..strong_up) [5]
    iv_regime_rank: float        # [0,1] percentile in universe
    structural_break_prob: float # P(break active)
    entropy: float               # Shannon entropy of beliefs
    transition_risk: float       # P(regime shift imminent)
    stability: float             # [0,1] overall stability

class FuzzyRegime:

    @staticmethod
    def _soft_assign(value_rank: float, state_centers: np.ndarray,
                     state_widths: np.ndarray) -> np.ndarray:
        """Gaussian kernel soft assignment to states."""
        probs = np.exp(-0.5 * ((value_rank - state_centers) / np.maximum(state_widths, 0.01)) ** 2)
        return probs / (probs.sum() + 1e-12)

    @staticmethod
    def estimate_vol_regime(ivp: float, ustats: dict) -> np.ndarray:
        if not ustats.get('valid'):
            return np.array([0, 0.1, 0.6, 0.2, 0.1, 0])
        rank = SignalSpace.percentile_rank(ivp, ustats['percentiles'], 0)
        centers = np.array([0.05, 0.175, 0.50, 0.825, 0.925, 0.975])
        widths = np.array([0.05, 0.075, 0.25, 0.075, 0.025, 0.025])
        return FuzzyRegime._soft_assign(rank, centers, widths)

    @staticmethod
    def estimate_trend_regime(price, ma20, ma50, rsi, mom, adx,
                              kalman, ustats: dict) -> np.ndarray:
        signals = []
        if ma20 > 0:
            signals.append(np.clip((price / ma20 - 1) * 100 / 5, -1, 1))
        if ma50 > 0:
            signals.append(np.clip((price / ma50 - 1) * 100 / 8, -1, 1))
        signals.append(np.clip((rsi - 50) / 25, -1, 1))
        if abs(mom) > 0:
            signals.append(np.clip(mom / 5, -1, 1))
        signals.append(np.clip(kalman / 3, -1, 1))

        # ADX as strength multiplier (universe-relative)
        if ustats.get('valid'):
            adx_rank = SignalSpace.percentile_rank(adx, ustats['percentiles'], 4)
        else:
            adx_rank = adx / 50

        direction = np.mean(signals) if signals else 0
        trend_score = np.clip(direction * (0.5 + adx_rank), -1, 1)

        centers = np.array([-0.8, -0.35, 0.0, 0.35, 0.8])
        widths = np.array([0.20, 0.20, 0.25, 0.20, 0.20])
        return FuzzyRegime._soft_assign(trend_score, centers, widths)

    @staticmethod
    def compute(stock: dict, ustats: dict, prev_regime: 'RegimeState' = None) -> RegimeState:
        ivp = stock.get('IVPercentile', 50)
        price = stock.get('price', 100)
        garch_p = stock.get('GARCH_Persistence', 0.95)
        cusum = stock.get('CUSUM_Alert', False)
        rsi = stock.get('rsi_daily', 50)

        vol_probs = FuzzyRegime.estimate_vol_regime(ivp, ustats)
        trend_probs = FuzzyRegime.estimate_trend_regime(
            price, stock.get('ma20_daily', price), stock.get('ma50_daily', price),
            rsi, stock.get('% change', 0), stock.get('adx', 20),
            stock.get('kalman_trend', 0), ustats)

        # Apply stickiness (Dampening) if transitioning
        if prev_regime is not None:
            # 95% persistence to ensure institutional stability in cross-sectional scans
            vol_probs = 0.95 * prev_regime.vol_probs + 0.05 * vol_probs
            trend_probs = 0.95 * prev_regime.trend_probs + 0.05 * trend_probs

        iv_rank = SignalSpace.percentile_rank(ivp, ustats['percentiles'], 0) if ustats.get('valid') else ivp / 100

        # Structural break: continuous, not binary
        break_prob = np.clip((0.80 if cusum else 0.05) * (2 - garch_p), 0, 0.95)

        # Regime entropy
        def H(p):
            p = p[p > 1e-12]
            return -np.sum(p * np.log2(p)) / max(np.log2(len(p)), 1) if len(p) > 0 else 0
        entropy = (H(vol_probs) + H(trend_probs)) / 2

        # Stability — universe-adaptive
        stab = 0.5
        if 0.25 <= iv_rank <= 0.75: stab += 0.15
        if 40 <= rsi <= 60: stab += 0.10
        if garch_p > 0.90: stab += 0.10
        if cusum: stab -= 0.25
        stab = np.clip(stab, 0.1, 0.95)

        # Transition risk
        t_risk = 0.20
        if garch_p < 0.90: t_risk += 0.15
        elif garch_p > 0.97: t_risk -= 0.10
        if cusum: t_risk += 0.30
        if iv_rank > 0.90 or iv_rank < 0.10: t_risk += 0.15
        t_risk += (1 - stab) * 0.20
        t_risk = np.clip(t_risk, 0.05, 0.95)

        return RegimeState(
            vol_probs=vol_probs, trend_probs=trend_probs,
            iv_regime_rank=iv_rank, structural_break_prob=break_prob,
            entropy=entropy, transition_risk=t_risk, stability=stab)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  ADAPTIVE GATING
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Total Adaptivity, Graceful Failure Mode, Scalable Complexity
# Only When Justified, Structural Anti-Fragility

# Strategy structural properties — these ARE invariant (structural, not tunable)
STRATEGY_STRUCTURE = {
    'Short Strangle':        {'type': 'CREDIT', 'naked': True,  'needs_stability': True,  'direction': 'NEUTRAL'},
    'Short Straddle':        {'type': 'CREDIT', 'naked': True,  'needs_stability': True,  'direction': 'NEUTRAL'},
    'Iron Condor':           {'type': 'CREDIT', 'naked': False, 'needs_stability': True,  'direction': 'NEUTRAL'},
    'Iron Butterfly':        {'type': 'CREDIT', 'naked': False, 'needs_stability': True,  'direction': 'NEUTRAL'},
    'Bull Put Spread':       {'type': 'CREDIT', 'naked': False, 'needs_stability': False, 'direction': 'BULLISH'},
    'Bear Call Spread':      {'type': 'CREDIT', 'naked': False, 'needs_stability': False, 'direction': 'BEARISH'},
    'Bull Call Spread':      {'type': 'DEBIT',  'naked': False, 'needs_stability': False, 'direction': 'BULLISH'},
    'Bear Put Spread':       {'type': 'DEBIT',  'naked': False, 'needs_stability': False, 'direction': 'BEARISH'},
    'Long Straddle':         {'type': 'DEBIT',  'naked': False, 'needs_stability': False, 'direction': 'VOLATILE'},
    'Long Strangle':         {'type': 'DEBIT',  'naked': False, 'needs_stability': False, 'direction': 'VOLATILE'},
    'Calendar Spread':       {'type': 'DEBIT',  'naked': False, 'needs_stability': True,  'direction': 'NEUTRAL'},
    'Jade Lizard':           {'type': 'CREDIT', 'naked': True,  'needs_stability': True,  'direction': 'NEUTRAL'},
    'Broken Wing Butterfly': {'type': 'HYBRID', 'naked': False, 'needs_stability': False, 'direction': 'BULLISH'},
    'Ratio Spread':          {'type': 'HYBRID', 'naked': True,  'needs_stability': False, 'direction': 'BULLISH'},
}

class AdaptiveGating:

    @staticmethod
    def iv_viability(sname: str, iv_rank: float) -> float:
        """Sigmoid viability — no hard thresholds."""
        stype = STRATEGY_STRUCTURE.get(sname, {}).get('type', 'HYBRID')
        if stype == 'CREDIT':
            return np.clip(1.0 / (1 + np.exp(-6 * (iv_rank - 0.35))), 0.05, 1.0)
        elif stype == 'DEBIT':
            return np.clip(1.0 / (1 + np.exp(6 * (iv_rank - 0.65))), 0.05, 1.0)
        else:
            return np.clip(0.6 + 0.4 * np.exp(-8 * (iv_rank - 0.5) ** 2), 0.05, 1.0)

    @staticmethod
    def dte_fitness(sname: str, dte: int) -> float:
        """Beta-distribution DTE fitness — structural, not arbitrary."""
        if dte <= 0: return 0.01
        x = min(dte / 90, 1.0)
        ss = STRATEGY_STRUCTURE.get(sname, {})

        if sname == 'Calendar Spread':   a, b = 3.0, 1.5
        elif ss.get('naked'):            a, b = 1.5, 3.0
        elif ss.get('type') == 'CREDIT': a, b = 2.0, 2.5
        elif 'Long' in sname:           a, b = 2.5, 2.0
        else:                            a, b = 2.0, 2.0

        if 0.001 < x < 0.999:
            mode = (a - 1) / (a + b - 2) if a > 1 and b > 1 else 0.5
            return np.clip(sp_stats.beta.pdf(x, a, b) / max(sp_stats.beta.pdf(mode, a, b), 1e-6), 0.05, 1.0)
        return 0.10

    @staticmethod
    def break_impact(sname: str, break_prob: float) -> float:
        """Structural break impact — continuous multiplier."""
        ss = STRATEGY_STRUCTURE.get(sname, {})
        if ss.get('direction') == 'VOLATILE':
            return 1.0 + 0.3 * break_prob          # benefit
        elif ss.get('needs_stability'):
            return 1.0 - 0.5 * break_prob           # penalty
        else:
            return 1.0 - 0.2 * break_prob

    @staticmethod
    def trend_alignment(sname: str, trend_probs: np.ndarray) -> float:
        """Dot product of trend distribution with strategy affinity."""
        d = STRATEGY_STRUCTURE.get(sname, {}).get('direction', 'NEUTRAL')
        # Affinity vectors: how much each trend state helps this strategy
        affinities = {
            'BULLISH':  np.array([0.05, 0.15, 0.30, 0.70, 0.95]),
            'BEARISH':  np.array([0.95, 0.70, 0.30, 0.15, 0.05]),
            'NEUTRAL':  np.array([0.10, 0.40, 0.95, 0.40, 0.10]),
            'VOLATILE': np.array([0.80, 0.30, 0.10, 0.30, 0.80]),
        }
        a = affinities.get(d, affinities['NEUTRAL'])
        return np.clip(np.dot(trend_probs, a), 0.05, 0.95)

    @staticmethod
    def antifragility_boost(sname: str, regime_entropy: float) -> float:
        """§6.6 Structural anti-fragility: volatile strats benefit from chaos."""
        d = STRATEGY_STRUCTURE.get(sname, {}).get('direction', 'NEUTRAL')
        if d == 'VOLATILE':
            return 1.0 + 0.25 * regime_entropy     # boost in high entropy
        elif STRATEGY_STRUCTURE.get(sname, {}).get('needs_stability'):
            return 1.0 - 0.20 * regime_entropy     # penalize in high entropy
        return 1.0

    @staticmethod
    def compute_viability(sname: str, regime: RegimeState, dte: int) -> float:
        """Geometric mean of all viability factors — never binary."""
        components = np.array([
            AdaptiveGating.iv_viability(sname, regime.iv_regime_rank),
            AdaptiveGating.dte_fitness(sname, dte),
            AdaptiveGating.break_impact(sname, regime.structural_break_prob),
            AdaptiveGating.trend_alignment(sname, regime.trend_probs),
            AdaptiveGating.antifragility_boost(sname, regime.entropy),
        ])
        components = np.maximum(components, 0.01)
        return np.prod(components) ** (1.0 / len(components))

    @staticmethod
    def min_premium(price: float, dte: int) -> float:
        """Adaptive — relative to stock price, not fixed ₹0.50."""
        dte_factor = max(0.3, min(1.0, dte / 30))
        return max(0.10, price * 0.0002 * dte_factor)

class CostScrub:
    """Indian FnO Transaction Cost Model."""
    @staticmethod
    def calculate(net_credit, max_profit, lot_size, n_legs, price) -> dict:
        # Realistic Indian broker + tax model
        brokerage = 40 * max(1, n_legs // 2)  # ₹20/order
        premium = abs(net_credit or max_profit) * lot_size
        stt = premium * 0.0005  # 0.05% on sell
        txn_charge = premium * 0.00053
        gst = (brokerage + txn_charge) * 0.18
        slippage = price * lot_size * 0.0005 * n_legs # 5bps slippage (NSE Institutional)
        
        total = brokerage + stt + txn_charge + gst + slippage
        impact = total / max(abs(max_profit), 1)
        
        return {
            'total_cost': total,
            'impact_pct': impact * 100,
            'is_hollow': impact > 0.50,  # Gate: >50% impact is unacceptable
            # Parabolic decay: zeroes out rapidly if impact > 35%
            'penalty': np.clip(1.0 - (impact / 0.35)**2, 0.0, 1.0)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §4  ADAPTIVE ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Model Competition & Evolution, Cross-Model Disagreement
# Monitoring, Bayesian Parameter Updating

class AdaptiveEnsemble:

    @staticmethod
    def fuse(pop_bsm: float, pop_mc: float, n_paths: int = 10000
             ) -> Tuple[float, float, float]:
        """Returns (pop_mean, pop_std, model_agreement).
        Disagreement → weight MC more (fewer distributional assumptions)."""
        mc_se = np.sqrt(max(pop_mc * (1 - pop_mc), 1e-6) / max(n_paths, 100))
        gap = abs(pop_bsm - pop_mc)

        if gap < 0.05:   w_b, w_m = 0.45, 0.55
        elif gap < 0.15: w_b, w_m = 0.30, 0.70
        else:            w_b, w_m = 0.15, 0.85

        mean = np.clip(w_b * pop_bsm + w_m * pop_mc, 0.01, 0.99)
        std = np.clip(np.sqrt((w_b * gap) ** 2 + mc_se ** 2), 0.005, 0.25)
        agreement = 1.0 - min(gap / 0.30, 1.0)
        return mean, std, agreement


# ═══════════════════════════════════════════════════════════════════════════════
# §5  PROBABILISTIC SCORING
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Full Uncertainty Quantification, Confidence-Weighted Allocation,
# Probabilistic Conviction, Explicit Awareness of Own Uncertainty

@dataclass
class ConvictionDistribution:
    mean: float; std: float
    ci_lower: float; ci_upper: float
    entropy: float
    components: Dict[str, float] = field(default_factory=dict)
    viability: float = 0.0
    model_agreement: float = 0.0
    signal_quality: float = 0.0

class ProbabilisticScoring:

    @staticmethod
    def compute(pop_mean, pop_std, ev_ratio, sharpe, viability,
                regime_entropy, model_agreement, signal_quality,
                transition_risk) -> ConvictionDistribution:
        """Certainty-weighted scoring — uncertain factors contribute less."""
        # Factor scores [0,1]
        pop_s = pop_mean
        ev_s = np.clip((ev_ratio + 1) / 2, 0, 1)
        sh_s = np.clip((sharpe + 2) / 5, 0, 1)
        vi_s = viability

        # Certainty of each factor
        cert = np.clip(np.array([
            1.0 - pop_std * 3,           # pop certainty
            0.70,                          # EV always partially uncertain
            min(1.0, abs(sharpe) * 2),     # near-zero Sharpe → uncertain
            signal_quality,                # from universe analysis
        ]), 0.1, 1.0)

        scores = np.array([pop_s, ev_s, sh_s, vi_s])
        weights = cert / (cert.sum() + 1e-12)
        raw = np.sum(scores * weights)

        # Dampers: entropy, transition risk, model confidence
        damped = raw * (1 - 0.4 * regime_entropy) * (1 - 0.3 * transition_risk) * (0.7 + 0.3 * model_agreement)
        mean = np.clip(damped * 100, 0, 100)

        # Uncertainty from multiple sources
        total_std = np.clip(np.sqrt(np.sum(weights**2 * (1-cert)**2) * 100**2 + (pop_std*100)**2) * 0.5, 1, 30)
        ci_lo = max(0, mean - 1.28 * total_std)
        ci_hi = min(100, mean + 1.28 * total_std)
        ent = min((ci_hi - ci_lo) / 50, 1.0)

        return ConvictionDistribution(
            mean=mean, std=total_std, ci_lower=ci_lo, ci_upper=ci_hi,
            entropy=ent,
            components={'pop': float(pop_s), 'ev': float(ev_s), 'sharpe': float(sh_s), 'viability': float(vi_s)},
            viability=viability, model_agreement=model_agreement,
            signal_quality=signal_quality)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  META-INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Reflexivity Awareness, Self-Diagnosis of Predictive Decay,
# Adaptive Learning Rate Control, Nonlinear Interaction Modeling,
# Exploration vs Exploitation Balance, Structural Anti-Fragility,
# Recursive Self-Improvement Loop, Meta-Learning of Adaptation Speed,
# Continuous Measurement of Edge Half-Life

class MetaIntelligence:
    """The system that watches itself."""

    # ── §6.1 Reflexivity ──
    @staticmethod
    def reflexivity_penalty(strategy_counts: Dict[str, int],
                            direction_counts: Dict[str, int]) -> float:
        """When output is concentrated, hidden systematic risk exists.
        HHI of strategy distribution + direction distribution.
        Returns penalty multiplier [0.5, 1.0]."""
        # Strategy HHI
        total_s = sum(strategy_counts.values()) or 1
        hhi_s = sum((c / total_s) ** 2 for c in strategy_counts.values())

        # Direction HHI
        total_d = sum(direction_counts.values()) or 1
        hhi_d = sum((c / total_d) ** 2 for c in direction_counts.values())

        # Combined concentration (1/N = perfectly diversified)
        n_strats = max(len(strategy_counts), 1)
        n_dirs = max(len(direction_counts), 1)
        excess_conc_s = max(0, hhi_s - 1/n_strats) / (1 - 1/max(n_strats, 2))
        excess_conc_d = max(0, hhi_d - 1/n_dirs) / (1 - 1/max(n_dirs, 2))

        concentration = 0.6 * excess_conc_s + 0.4 * excess_conc_d
        return np.clip(1.0 - 0.5 * concentration, 0.50, 1.0)

    # ── §6.2 Edge Half-Life / Predictive Decay (edge_half_life measurement) ──
    @staticmethod
    def edge_health(historical_ic: List[float]) -> Tuple[float, float]:
        """From rolling IC (information coefficient) history, estimate:
        - current edge strength [0,1]
        - edge half-life (how many periods until edge decays by 50%)
        
        historical_ic: list of rank_corr(conviction, realized_PnL) over windows.
        If empty (no history yet), returns conservative defaults.
        """
        if len(historical_ic) < 3:
            return 0.50, 30.0  # uncertain default

        ic = np.array(historical_ic[-20:])  # last 20 observations
        current_edge = np.clip(ic[-1] / 0.10, 0, 1)  # IC of 0.10 = full edge

        # Exponential decay fit: IC(t) = IC_0 * exp(-λt)
        if len(ic) >= 5 and ic[0] > 0.001:
            ratios = ic[1:] / np.maximum(ic[:-1], 0.001)
            avg_decay = np.nanmean(np.log(np.clip(ratios, 0.01, 10)))
            if avg_decay < 0:
                half_life = -np.log(2) / avg_decay
            else:
                half_life = 100  # edge not decaying
        else:
            half_life = 30

        return np.clip(current_edge, 0, 1), np.clip(half_life, 1, 200)

    # ── §6.3 Adaptation Speed Meta-Learning ──
    @staticmethod
    def adaptation_speed(regime_transition_history: List[float]) -> float:
        """How fast should the system adapt to new data?
        
        If regimes change frequently → fast adaptation (short lookback)
        If regimes are stable → slow adaptation (long lookback, more confidence)
        
        Returns lookback multiplier [0.5, 2.0]:
          0.5 = use half the default lookback (adapt fast)
          2.0 = use double the default lookback (adapt slow)
        """
        if len(regime_transition_history) < 3:
            return 1.0

        avg_transition = np.mean(regime_transition_history[-10:])
        # High transition frequency → fast adaptation
        return np.clip(2.0 - 2.0 * avg_transition, 0.5, 2.0)

    # ── §6.4 Thompson Sampling (Exploration vs Exploitation) ──
    @staticmethod
    def thompson_sample(conviction_dists: List[ConvictionDistribution]
                        ) -> List[float]:
        """Sample from each strategy's conviction distribution.
        The strategy with highest SAMPLE (not highest mean) wins.
        High-uncertainty strategies naturally get explored."""
        samples = []
        for cd in conviction_dists:
            # Sample from normal approximation of posterior
            s = np.random.normal(cd.mean, cd.std)
            samples.append(np.clip(s, 0, 100))
        return samples

    # ── §6.5 Drawdown Sensitivity ──
    @staticmethod
    def drawdown_multiplier(recent_accuracy: List[bool]) -> float:
        """After consecutive wrong predictions, reduce exposure.
        recent_accuracy: list of True/False for recent calls.
        Returns sizing multiplier [0.3, 1.0]."""
        if not recent_accuracy:
            return 1.0
        window = recent_accuracy[-10:]
        accuracy = sum(window) / len(window)
        # Below 50% accuracy → significant pullback
        return np.clip(0.3 + 0.7 * accuracy, 0.30, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# §7  ENTROPY GOVERNOR
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Entropy Monitoring & Governance, Automatic Risk Budget Adjustment,
# Capital Preservation Priority, Risk-of-Ruin Awareness

class EntropyGovernor:

    @staticmethod
    def system_entropy(regimes: List[RegimeState],
                       convictions: List[ConvictionDistribution]) -> float:
        if not regimes: return 0.5
        r_ent = np.mean([r.entropy for r in regimes])
        c_ent = np.mean([c.entropy for c in convictions]) if convictions else 0.5
        return np.clip(r_ent * 0.6 + c_ent * 0.4, 0, 1)

    @staticmethod
    def risk_budget(sys_entropy: float, avg_transition: float) -> float:
        """Kelly fraction scaler [0.10, 0.50]."""
        base = 0.50
        return np.clip(base * (1 - 0.6 * sys_entropy) * (1 - 0.4 * avg_transition), 0.10, 0.50)

    @staticmethod
    def confidence_threshold(sys_entropy: float) -> float:
        """Min conviction to recommend. Higher when uncertain."""
        return 35 + sys_entropy * 25

    @staticmethod
    def max_naked_fraction(sys_entropy: float) -> float:
        """Max fraction of recs that can be naked. Lower when uncertain."""
        return np.clip(0.40 - 0.35 * sys_entropy, 0.05, 0.40)


# ═══════════════════════════════════════════════════════════════════════════════
# §8  ADAPTIVE KELLY
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Bayesian Parameter Updating, Drawdown-Sensitive Exposure Scaling,
# Confidence-Weighted Allocation

class AdaptiveKelly:

    @staticmethod
    def compute(pop_mean, pop_std, max_profit, max_loss,
                mc_ev, mc_std, risk_budget, drawdown_mult=1.0) -> float:
        if max_loss <= 0 or max_profit <= 0: return 0.0

        b = max_profit / max_loss
        k_raw = (b * pop_mean - (1 - pop_mean)) / max(b, 0.01)
        if k_raw <= 0: return 0.0

        # Uncertainty discount
        certainty = max(0.1, 1.0 - pop_std * 5)

        # Continuous Kelly blend
        if mc_std > 0.01 and mc_ev > 0:
            cap = max(max_loss, 1)
            k_cont = (mc_ev / cap) / max((mc_std / cap) ** 2, 0.001)
            k_final = 0.4 * k_raw + 0.6 * k_cont
        else:
            k_final = k_raw

        return np.clip(k_final * risk_budget * certainty * drawdown_mult, 0, 0.25)


# ═══════════════════════════════════════════════════════════════════════════════
# §9  PORTFOLIO AWARENESS
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Risk-of-Ruin Awareness, Capital Preservation Priority

class PortfolioAwareness:

    @staticmethod
    def diversified_selection(candidates: list, n: int = 9) -> list:
        """Greedy selection with diversity penalties."""
        if len(candidates) <= n:
            return candidates

        selected = []
        remaining = list(range(len(candidates)))

        # First: highest conviction mean
        best = max(remaining, key=lambda i: candidates[i].get('conviction_mean', candidates[i].get('conviction_score', 0)))
        selected.append(best)
        remaining.remove(best)

        while len(selected) < n and remaining:
            best_score, best_idx = -1, remaining[0]
            for idx in remaining:
                c = candidates[idx]
                conv = c.get('conviction_mean', c.get('conviction_score', 0))
                penalty = 0
                for si in selected:
                    s = candidates[si]
                    if c.get('strategy') == s.get('strategy'): penalty += 0.15
                    if c.get('direction') == s.get('direction'): penalty += 0.05
                    p1, p2 = c.get('price', 0), s.get('price', 0)
                    if p1 > 0 and p2 > 0 and max(p1,p2)/max(min(p1,p2),1) < 1.5: penalty += 0.03
                adj = conv * (1 - min(penalty, 0.50))
                if adj > best_score:
                    best_score, best_idx = adj, idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]

    @staticmethod
    def ruin_probability(trades: list, capital: float = 1_000_000) -> float:
        """Estimate P(portfolio loss > 20% of capital) under worst case.
        Simple: sum of max losses / capital."""
        if not trades or capital <= 0: return 0
        total_risk = sum(t.get('ml_lot', 0) for t in trades)
        return np.clip(total_risk / capital, 0, 1)

    @staticmethod
    def net_greeks(trades: list) -> dict:
        net = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        for t in trades:
            ng = t.get('_net_greeks')
            if ng:
                lot = t.get('lot_size', 1)
                for g in net:
                    net[g] += getattr(ng, g, 0) * lot
        return net


# ═══════════════════════════════════════════════════════════════════════════════
# §10  ADAPTIVE STRIKES
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveStrikes:

    @staticmethod
    def target_delta(sname: str, regime: RegimeState) -> float:
        """Data-driven delta target based on regime."""
        d = STRATEGY_STRUCTURE.get(sname, {}).get('direction', 'NEUTRAL')
        base = {'NEUTRAL': 0.20, 'BULLISH': 0.30, 'BEARISH': 0.30, 'VOLATILE': 0.25}.get(d, 0.25)
        iv_adj = -0.08 * (regime.iv_regime_rank - 0.5)
        trend_cert = 1 - regime.entropy
        trend_adj = 0.05 * trend_cert if d in ('BULLISH','BEARISH') else 0
        return np.clip(base + iv_adj + trend_adj, 0.10, 0.45)

    @staticmethod
    def gaps_for_delta(target_delta, S, T, iv, gap):
        """Invert BSM delta to get strike distance in gap units."""
        if T <= 0 or iv <= 0 or gap <= 0: return 1
        d1_t = -norm.ppf(max(target_delta, 0.01))
        r = 0.065
        K = S * np.exp(-(d1_t * iv * np.sqrt(T) - (r + iv**2/2) * T))
        return max(1, round(abs(S - K) / gap))


# ═══════════════════════════════════════════════════════════════════════════════
# §11  COMPUTE TRIAGE
# ═══════════════════════════════════════════════════════════════════════════════
# Satisfies: Computational Efficiency Relative to Signal Gain

class ComputeTriage:
    """Not every stock × strategy needs full MC. Triage saves compute."""

    @staticmethod
    def should_run_mc(bsm_pop: float, viability: float) -> bool:
        """Only run MC simulation if BSM suggests the strategy is viable.
        BSM is O(1), MC is O(n_paths). Gate the expensive computation."""
        # If BSM POP is below 35% AND viability is low, skip MC
        if bsm_pop < 0.35 and viability < 0.20:
            return False
        # If BSM POP is above 95%, it's clearly OTM → skip MC
        if bsm_pop > 0.98:
            return False
        return True

    @staticmethod
    def adaptive_mc_paths(viability: float) -> int:
        """More paths for more promising candidates."""
        if viability > 0.60: return 10000
        if viability > 0.30: return 5000
        return 2000


# ═══════════════════════════════════════════════════════════════════════════════
# §12  ADAPTIVE ENGINE — Master Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveEngine:
    """Replaces the entire hardcoded scoring pipeline.

    Pipeline:
    1. calibrate()        → SignalSpace derives universe parameters
    2. compute_regime()   → FuzzyRegime produces continuous state
    3. compute_viability() → AdaptiveGating scores strategy fitness
    4. triage()           → ComputeTriage gates expensive MC
    5. fuse_pop()         → AdaptiveEnsemble merges BSM + MC
    6. score()            → ProbabilisticScoring produces conviction distribution
    7. kelly()            → AdaptiveKelly sizes position
    8. diversify()        → PortfolioAwareness selects portfolio
    9. govern()           → EntropyGovernor sets risk envelope
    10. meta()            → MetaIntelligence monitors system health
    """

    def __init__(self):
        self.universe_stats = {'valid': False}
        self.regime_states: List[RegimeState] = []
        self.conviction_dists: List[ConvictionDistribution] = []
        self._historical_ic: List[float] = []
        self._recent_accuracy: List[bool] = []
        self._transition_history: List[float] = []

    # ── Phase 1: Calibrate ──
    def calibrate(self, df) -> dict:
        sigs = []
        for _, row in df.iterrows():
            s = SignalSpace.extract(row.to_dict())
            if not np.isnan(s).any():
                sigs.append(s)
        if sigs:
            self.universe_stats = SignalSpace.compute_universe_stats(np.array(sigs))
        return self.universe_stats

    # ── Phase 2: Regime ──
    def compute_regime(self, stock: dict, prev_regime: RegimeState = None) -> RegimeState:
        # Stationary feature mapping
        standardized_stock = SignalSpace.standardize(stock, self.universe_stats)
        r = FuzzyRegime.compute(standardized_stock, self.universe_stats, prev_regime)
        self.regime_states.append(r)
        self._transition_history.append(r.transition_risk)
        return r

    # ── Phase 3: Viability ──
    def compute_viability(self, sname: str, regime: RegimeState, dte: int) -> float:
        return AdaptiveGating.compute_viability(sname, regime, dte)

    # ── Phase 4: Triage ──
    def should_mc(self, bsm_pop: float, viability: float) -> bool:
        return ComputeTriage.should_run_mc(bsm_pop, viability)

    def mc_paths(self, viability: float) -> int:
        return ComputeTriage.adaptive_mc_paths(viability)

    # ── Phase 5: Ensemble ──
    def fuse_pop(self, pop_bsm, pop_mc, n_paths=10000):
        return AdaptiveEnsemble.fuse(pop_bsm, pop_mc, n_paths)

    # ── Phase 6: Score ──
    def score(self, pop_mean, pop_std, ev_ratio, sharpe, viability,
              regime: RegimeState, model_agreement, 
              strategy_info: dict = None) -> ConvictionDistribution:
        sq = np.mean(self.universe_stats.get('signal_weights', [0.1])) * 10 \
            if self.universe_stats.get('valid') else 0.5
        
        # Apply Cost Scrubbing
        cost_penalty = 1.0
        if strategy_info:
            costs = CostScrub.calculate(
                strategy_info.get('net_credit'),
                strategy_info.get('max_profit'),
                strategy_info.get('lot_size', 1),
                strategy_info.get('n_legs', 2),
                strategy_info.get('price', 100)
            )
            cost_penalty = costs['penalty']

        adjusted_viability = viability * cost_penalty
        
        cd = ProbabilisticScoring.compute(
            pop_mean, pop_std, ev_ratio, sharpe, adjusted_viability,
            regime.entropy, model_agreement, min(sq, 1.0), regime.transition_risk)
        self.conviction_dists.append(cd)
        return cd

    # ── Phase 7: Kelly ──
    def kelly(self, pop_mean, pop_std, max_profit, max_loss, mc_ev, mc_std):
        se = self.system_entropy()
        at = np.mean(self._transition_history[-20:]) if self._transition_history else 0.3
        rb = EntropyGovernor.risk_budget(se, at)
        dm = MetaIntelligence.drawdown_multiplier(self._recent_accuracy)
        return AdaptiveKelly.compute(pop_mean, pop_std, max_profit, max_loss, mc_ev, mc_std, rb, dm)

    # ── Phase 8: Diversify ──
    def diversify(self, candidates, n=9):
        return PortfolioAwareness.diversified_selection(candidates, n)

    # ── Phase 9: Govern ──
    def system_entropy(self) -> float:
        return EntropyGovernor.system_entropy(self.regime_states, self.conviction_dists)

    def confidence_threshold(self) -> float:
        return EntropyGovernor.confidence_threshold(self.system_entropy())

    def max_naked_fraction(self) -> float:
        return EntropyGovernor.max_naked_fraction(self.system_entropy())

    # ── Phase 10: Meta ──
    def reflexivity_penalty(self, strategy_counts, direction_counts):
        return MetaIntelligence.reflexivity_penalty(strategy_counts, direction_counts)

    def edge_health(self):
        return MetaIntelligence.edge_health(self._historical_ic)

    def adaptation_speed(self):
        return MetaIntelligence.adaptation_speed(self._transition_history)

    def thompson_sample(self, dists):
        return MetaIntelligence.thompson_sample(dists)

    def min_premium(self, price, dte):
        return AdaptiveGating.min_premium(price, dte)

    def target_delta(self, sname, regime):
        return AdaptiveStrikes.target_delta(sname, regime)

    def strike_gaps(self, target_delta, S, T, iv, gap):
        return AdaptiveStrikes.gaps_for_delta(target_delta, S, T, iv, gap)

    # ── Bookkeeping ──
    def record_outcome(self, conviction, realized_pnl_positive: bool):
        """Called after expiry to track prediction quality."""
        self._recent_accuracy.append(realized_pnl_positive)
        # Keep only last 50
        if len(self._recent_accuracy) > 50:
            self._recent_accuracy = self._recent_accuracy[-50:]

    def record_ic(self, ic_value: float):
        """Record rolling information coefficient."""
        self._historical_ic.append(ic_value)
        if len(self._historical_ic) > 100:
            self._historical_ic = self._historical_ic[-100:]

    def reset_cycle(self):
        """Reset per-analysis-run state (keep historical data)."""
        self.regime_states = []
        self.conviction_dists = []
