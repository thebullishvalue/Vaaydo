"""
Volatility: what the market implies (from real option closes) and what we forecast
(from the underlying's own realised history). The gap between the two is the edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import pricing
from .config import DEFAULT, RISK_FREE, TRADING_DAYS, Config


# ── implied ──────────────────────────────────────────────────────────────────

def tradable_mask(df: pd.DataFrame, cfg: Config = DEFAULT) -> pd.Series:
    liq = cfg.liquidity
    lot = df["lot"].clip(lower=1)
    return (df["contracts"] >= liq.min_contracts) & (df["oi"] / lot >= liq.min_oi_lots) & (df["close"] >= liq.min_premium)


def parity_forward(ex: pd.DataFrame, T: float) -> float | None:
    """F = K + e^{rT}(C - P) at the most-traded strike where both sides traded."""
    c = ex[ex["otype"] == "CE"].set_index("strike")
    p = ex[ex["otype"] == "PE"].set_index("strike")
    both = c.join(p, lsuffix="_c", rsuffix="_p", how="inner")
    both = both[(both["contracts_c"] > 0) & (both["contracts_p"] > 0)]
    if both.empty:
        return None
    k = (both["contracts_c"] + both["contracts_p"]).idxmax()
    row = both.loc[k]
    return float(k + np.exp(RISK_FREE * T) * (row["close_c"] - row["close_p"]))


@dataclass
class Smile:
    symbol: str
    expiry: pd.Timestamp
    forward: float
    spot: float
    T: float
    atm_iv: float
    put25_iv: float
    call25_iv: float
    n_tradable: int
    table: pd.DataFrame   # strike, otype, close, iv, delta, tradable, contracts, oi, lot

    @property
    def skew(self) -> float:
        return self.put25_iv - self.call25_iv


def _interp_at(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    o = np.argsort(x)
    x, y = x[o], y[o]
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.nan
    if x0 <= x[0]:
        return float(y[0]) if x[0] - x0 < 0.25 else np.nan
    if x0 >= x[-1]:
        return float(y[-1]) if x0 - x[-1] < 0.25 else np.nan
    return float(np.interp(x0, x, y))


def smile(ex: pd.DataFrame, asof: pd.Timestamp, cfg: Config = DEFAULT) -> Smile | None:
    """One symbol, one expiry, one day of option closes -> implied vol smile."""
    ex = ex.copy()
    expiry = pd.Timestamp(ex["expiry"].iloc[0])
    T = max((expiry - asof).days, 0) / 365.0
    if T <= 0:
        return None
    F = ex["fwd"].iloc[0] if "fwd" in ex and pd.notna(ex["fwd"].iloc[0]) else None
    F = F or parity_forward(ex, T)
    if not F or not np.isfinite(F):
        return None
    is_call = (ex["otype"] == "CE").to_numpy()
    K = ex["strike"].to_numpy()
    ex["iv"] = pricing.implied_vol(ex["close"].to_numpy(), F, K, T, is_call)
    ex["delta"] = pricing.delta(F, K, T, np.nan_to_num(ex["iv"].to_numpy(), nan=0.3), is_call)
    ex["tradable"] = tradable_mask(ex, cfg)
    otm = ((ex["otype"] == "CE") & (ex["strike"] >= F)) | ((ex["otype"] == "PE") & (ex["strike"] <= F))
    # smile reference points use OTM options that actually traded
    ref = ex[otm & (ex["contracts"] > 0) & ex["iv"].notna()]
    if len(ref) < 3:
        return None
    k = np.log(ref["strike"].to_numpy() / F)
    atm = _interp_at(k, ref["iv"].to_numpy(), 0.0)
    puts = ref[ref["otype"] == "PE"]
    calls = ref[ref["otype"] == "CE"]
    p25 = _interp_at(puts["delta"].to_numpy(), puts["iv"].to_numpy(), -0.25) if len(puts) >= 2 else np.nan
    c25 = _interp_at(calls["delta"].to_numpy(), calls["iv"].to_numpy(), 0.25) if len(calls) >= 2 else np.nan
    if not np.isfinite(atm):
        return None
    ex["model_px"] = _smile_prices(ex, ref, F, T, atm)
    return Smile(symbol=str(ex["symbol"].iloc[0]), expiry=expiry, forward=float(F),
                 spot=float(ex["spot"].iloc[0]), T=T, atm_iv=atm, put25_iv=p25, call25_iv=c25,
                 n_tradable=int(ex["tradable"].sum()), table=ex)


def _smile_prices(ex: pd.DataFrame, ref: pd.DataFrame, F: float, T: float, atm: float) -> np.ndarray:
    """Smooth smile -> one consistent price per strike.

    End-of-day closes are not simultaneous (each is its contract's own last-half-hour
    average, and an untraded contract repeats an old close), so structures built from raw
    closes can show credits larger than their width. A weighted quadratic in standardised
    moneyness, fitted to OTM options that traded, gives prices that are consistent with
    each other. Entry is then priced at the worse of close and model, leg by leg.
    """
    x = np.log(ref["strike"].to_numpy() / F) / (atm * np.sqrt(T))
    y = ref["iv"].to_numpy()
    w = np.sqrt(ref["contracts"].clip(lower=1).to_numpy())
    deg = 2 if len(ref) >= 5 else 1
    try:
        c = np.polyfit(x, y, deg, w=w)
    except (np.linalg.LinAlgError, ValueError):
        c = np.array([0.0] * (deg - 0) + [atm])
    xa = np.log(ex["strike"].to_numpy() / F) / (atm * np.sqrt(T))
    xa = np.clip(xa, x.min() - 0.5, x.max() + 0.5)   # no wild extrapolation
    iv = np.clip(np.polyval(c, xa), 0.03, 3.0)
    return pricing.price(F, ex["strike"].to_numpy(), T, iv, (ex["otype"] == "CE").to_numpy())


# ── realised & forecast ──────────────────────────────────────────────────────

def realised(r: pd.Series, window: int) -> float:
    x = r.dropna().iloc[-window:]
    if len(x) < max(10, window // 2):
        return np.nan
    return float(np.sqrt((x ** 2).mean() * TRADING_DAYS))


def ewma_vol(r: pd.Series, lam: float) -> pd.Series:
    """Daily EWMA vol, using information up to and including each day."""
    v = (r.fillna(0) ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(v)


@dataclass
class VolForecast:
    rv_1m: float
    rv_3m: float
    rv_1y: float
    forecast: float          # annualised, for the option's life
    n_obs: int


def forecast(r: pd.Series, cfg: Config = DEFAULT) -> VolForecast | None:
    """HAR-style blend of realised variance over three horizons.

    Volatility mean-reverts: the 1M number carries the recent regime, the 1Y number the
    stock's normal level. Weights are refit by `backtest.fit_forecast_weights`.
    """
    f = cfg.forecast
    x = r.dropna()
    if len(x) < f.min_history:
        return None
    r1, r3, ry = realised(x, 21), realised(x, 63), realised(x, 252)
    if not np.isfinite(ry):
        ry = realised(x, len(x))
    var = f.w_1m * r1 ** 2 + f.w_3m * r3 ** 2 + f.w_1y * ry ** 2
    return VolForecast(r1, r3, ry, float(np.sqrt(var)), len(x))
