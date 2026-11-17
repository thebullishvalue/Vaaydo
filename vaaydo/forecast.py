"""
The forecast distribution of the underlying at expiry.

Filtered historical simulation (FHS): divide each historical daily return by the
volatility prevailing when it happened, which leaves the *shape* of the stock's moves
(fat tails, skew, gap days) without its old volatility level. Bootstrap those shapes in
blocks over the option's life and rescale to the vol forecast.

Two deliberate choices:
  * The distribution is centred on the futures forward: no directional view is assumed.
    Black-Scholes MC with a 7% drift (the old engine) quietly made every bullish trade
    look good.
  * The level comes only from the forecast vol, so "is the option rich?" is a clean
    question: market IV vs our forecast, with tails from the stock's own history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT, TRADING_DAYS, Config
from .vol import ewma_vol


def standardised_shocks(r: pd.Series, cfg: Config = DEFAULT) -> np.ndarray:
    r = r.dropna()
    sig = ewma_vol(r, cfg.forecast.ewma_lambda).shift(1)
    z = (r / sig).replace([np.inf, -np.inf], np.nan).dropna().iloc[20:]   # skip EWMA burn-in
    z = z.to_numpy()
    if len(z) < 30:
        return np.array([])
    z = np.clip(z, -10, 10)
    return (z - z.mean()) / z.std()


class Shocks:
    """Unit-variance terminal shocks for a horizon, shared by every structure on one name
    (common random numbers: candidates are compared on identical scenarios)."""

    def __init__(self, z: np.ndarray, horizon_days: int, n: int, block: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        H = max(int(horizon_days), 1)
        nb = int(np.ceil(H / block))
        starts = rng.integers(0, len(z), size=(n, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(z)
        paths = z[idx].reshape(n, nb * block)[:, :H]
        s = paths.sum(axis=1)
        self.unit = (s - s.mean()) / s.std()   # empirical tails, exactly unit variance
        self.H = H

    def terminal(self, forward: float, vol: float) -> np.ndarray:
        x = vol * np.sqrt(self.H / TRADING_DAYS) * self.unit
        g = np.exp(x)
        return forward * g / g.mean()        # E[S_T] = forward


def shocks_for(r: pd.Series, horizon_days: int, pool: np.ndarray | None = None,
               cfg: Config = DEFAULT, n: int | None = None, seed: int = 7) -> Shocks | None:
    z = standardised_shocks(r, cfg)
    if len(z) < 250 and pool is not None and len(pool):
        z = np.concatenate([z, pool])
    if len(z) < 60:
        return None
    return Shocks(z, horizon_days, n or cfg.forecast.n_paths, cfg.forecast.block, seed)


def trading_days_between(start: pd.Timestamp, end: pd.Timestamp, calendar: pd.DatetimeIndex | None = None) -> int:
    """Trading days in (start, end]. Uses the observed calendar when it covers the span."""
    if calendar is not None and len(calendar) and calendar[-1] >= end:
        return int(((calendar > start) & (calendar <= end)).sum())
    return int(np.busday_count(start.date(), end.date() + pd.Timedelta(days=1)))
