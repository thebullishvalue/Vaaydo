"""Indian F&O option charges and execution haircuts."""
from __future__ import annotations

import numpy as np

from .config import DEFAULT, Config


def spread_frac(contracts, cfg: Config = DEFAULT):
    """Half-spread as a fraction of premium, from how much the strike traded.

    ~ k / sqrt(contracts): 100 contracts -> 3%, 10,000 -> 0.3%. This is an ASSUMPTION
    (end-of-day data has no quotes). Calibrate `spread_k` from your own broker's quotes
    with `costs.calibrate_spread`. Unknown volume falls back to `half_spread_frac`.
    """
    liq = cfg.liquidity
    if contracts is None or liq.half_spread_frac == 0:
        return liq.half_spread_frac
    c = np.maximum(np.asarray(contracts, dtype=float), 1.0)
    return np.clip(liq.spread_k / np.sqrt(c), liq.spread_floor, liq.spread_cap)


def calibrate_spread(quotes) -> float:
    """Fit `spread_k` from real quotes: a DataFrame with bid, ask, contracts (volume in
    lots) for options that traded. Returns the k that best fits half-spread/mid."""
    q = quotes[(quotes["bid"] > 0) & (quotes["ask"] > quotes["bid"]) & (quotes["contracts"] > 0)]
    mid = (q["bid"] + q["ask"]) / 2
    frac = (q["ask"] - q["bid"]) / 2 / mid
    return float(np.median(frac * np.sqrt(q["contracts"])))


def half_spread(premium, cfg: Config = DEFAULT, strike=None, contracts=None):
    """Estimated half bid-ask spread in rupees per unit. Deep in-the-money options quote
    tight in % terms, so it is capped at 0.3% of the strike."""
    liq = cfg.liquidity
    hs = np.maximum(liq.tick, spread_frac(contracts, cfg) * np.asarray(premium, dtype=float))
    if strike is not None and liq.half_spread_frac > 0:
        hs = np.minimum(hs, np.maximum(liq.tick, 0.003 * np.asarray(strike, dtype=float)))
    return hs


def fill(close, qty, cfg: Config = DEFAULT, strike=None, contracts=None):
    """Price we expect to deal at: pay up when buying, give up when selling."""
    close = np.asarray(close, dtype=float)
    return close + np.sign(qty) * half_spread(close, cfg, strike, contracts)


def charges(prices, qtys, units: float, cfg: Config = DEFAULT) -> float:
    """Statutory + broker charges for one set of orders (one side of a trade).

    prices: per-unit premium of each leg; qtys: signed lots (+buy, -sell); units: lot size.
    """
    c = cfg.costs
    prices = np.asarray(prices, dtype=float)
    qtys = np.asarray(qtys, dtype=float)
    turnover = np.abs(prices * qtys) * units
    sell_prem = np.sum(turnover[qtys < 0])
    buy_prem = np.sum(turnover[qtys > 0])
    n_orders = int(np.sum(qtys != 0))
    brokerage = c.brokerage_per_order * n_orders
    exch = c.exchange_txn * turnover.sum()
    sebi = c.sebi_per_crore * turnover.sum() / 1e7
    gst = c.gst * (brokerage + exch + sebi)
    stt = c.stt_sell_premium * sell_prem
    stamp = c.stamp_buy * buy_prem
    return float(brokerage + exch + sebi + gst + stt + stamp)
