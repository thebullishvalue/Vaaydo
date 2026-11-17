"""
Black-76 on the forward.

Pricing off the futures price (not spot + an assumed dividend/rate) means the forward the
market itself trades is used. All functions are vectorised over numpy arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .config import RISK_FREE

SQRT2PI = np.sqrt(2 * np.pi)


def _d1d2(F, K, T, sigma):
    sT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / sT
    return d1, d1 - sT


def price(F, K, T, sigma, is_call, r: float = RISK_FREE):
    F, K, T, sigma, is_call = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, T, sigma, is_call)))
    is_call = is_call.astype(bool)
    disc = np.exp(-r * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1d2(F, K, np.maximum(T, 1e-9), np.maximum(sigma, 1e-9))
        call = disc * (F * ndtr(d1) - K * ndtr(d2))
        put = disc * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def delta(F, K, T, sigma, is_call, r: float = RISK_FREE):
    """Forward delta (sign: + calls, - puts)."""
    F, K, T, sigma, is_call = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, T, sigma, is_call)))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1d2(F, K, np.maximum(T, 1e-9), np.maximum(sigma, 1e-9))
    disc = np.exp(-r * T)
    return np.where(is_call.astype(bool), disc * ndtr(d1), -disc * ndtr(-d1))


def vega(F, K, T, sigma, r: float = RISK_FREE):
    """Price change per 1 vol point (0.01)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, _ = _d1d2(F, K, np.maximum(T, 1e-9), np.maximum(sigma, 1e-9))
    return np.exp(-r * T) * F * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / SQRT2PI / 100


def implied_vol(px, F, K, T, is_call, r: float = RISK_FREE, lo: float = 0.01, hi: float = 3.0, iters: int = 64):
    """Vectorised bisection. NaN where the price is outside no-arbitrage bounds."""
    px, F, K, T, is_call = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (px, F, K, T, is_call)))
    disc = np.exp(-r * T)
    intrinsic = np.where(is_call.astype(bool), np.maximum(F - K, 0), np.maximum(K - F, 0)) * disc
    upper = np.where(is_call.astype(bool), F, K) * disc
    ok = (px > intrinsic + 1e-9) & (px < upper) & (T > 0) & (F > 0) & (K > 0)
    a = np.full(px.shape, lo)
    b = np.full(px.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_high = price(F, K, T, m, is_call, r) > px
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    iv = 0.5 * (a + b)
    iv = np.where(ok & (iv < hi * 0.999) & (iv > lo * 1.001), iv, np.nan)
    return iv
