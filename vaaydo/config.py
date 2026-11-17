"""
Every tunable number in VAAYDO lives here, with the reason it has the value it has.

These are *choices*, not truths. The previous version claimed to have "zero hardcoded
thresholds"; that was never possible. What is possible is to keep the choices few,
visible, economically motivated, and testable by the backtest. Where a value was fitted
from data, the comment says so and `vaaydo.backtest` can refit it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(os.environ.get("VAAYDO_DATA", Path.home() / ".vaaydo"))

TRADING_DAYS = 252
RISK_FREE = 0.065  # ~ Indian 3M T-bill; only used for discounting in Black-76.


@dataclass(frozen=True)
class Liquidity:
    # A leg is tradable only if it actually traded and carries open interest.
    # Untraded contracts in the bhavcopy carry an exchange-computed *theoretical* close,
    # which is not a price anyone could have dealt at.
    min_contracts: int = 25          # contracts (lots) traded on the day
    min_oi_lots: int = 50            # open interest, in lots
    min_premium: float = 0.50        # ₹; below this, tick size dominates
    # Execution haircut vs. the close when no live bid/ask is available.
    # Half-spread estimate = max(tick, frac * premium). Calibrated loosely to NSE stock
    # option quotes for liquid names (1-3% of premium near the money, wider in the wings).
    tick: float = 0.05
    half_spread_frac: float = 0.025  # used only when a strike's volume is unknown
    spread_k: float = 0.30           # half-spread/premium ~ spread_k / sqrt(contracts traded)
    spread_floor: float = 0.0025
    spread_cap: float = 0.05


@dataclass(frozen=True)
class Selection:
    # Which expiry to trade: the first monthly expiry with at least this many calendar days.
    min_dte: int = 20
    max_dte: int = 60
    # Short-strike deltas searched for credit structures, and wing widths in strikes.
    short_deltas: tuple = (0.15, 0.20, 0.25, 0.30)
    wing_steps: tuple = (1, 2, 3, 4)
    # Long-vol structures are only evaluated when forecast vol exceeds implied by this ratio.
    long_vol_trigger: float = 1.00
    # Margin of safety: EV must remain >= 0 when the vol forecast is wrong *against us*
    # by this fraction (forecast * (1 + s) for short vol, * (1 - s) for long vol).
    forecast_stress: float = 0.15
    # Minimum EV as a fraction of max loss, after costs. Below this the trade is noise.
    min_ev_to_risk: float = 0.03
    # Event proxy: near-month ATM IV above next-month ATM IV by this ratio implies a
    # known event (results, AGM, policy) inside the near month. Short vol is skipped.
    term_inversion_flag: float = 1.08


@dataclass(frozen=True)
class Forecast:
    # Realised-variance forecast = weighted blend of 1M / 3M / 1Y realised variance,
    # a HAR-style estimator. Defaults are refit by `backtest.fit_forecast_weights`.
    w_1m: float = 0.45
    w_3m: float = 0.35
    w_1y: float = 0.20
    ewma_lambda: float = 0.94        # for de-volatising returns in the bootstrap
    n_paths: int = 20000
    block: int = 5                   # block bootstrap length (days): keeps short-run clustering
    min_history: int = 126           # trading days of history required to forecast at all
    max_abs_return: float = 0.20     # daily |log return| beyond this w/ lot change => corp action


@dataclass(frozen=True)
class Risk:
    capital: float = 1_000_000.0
    risk_per_trade: float = 0.015    # max loss of one position as fraction of capital
    max_positions: int = 10
    max_total_risk: float = 0.20     # sum of max losses across the book
    max_per_cluster: int = 2         # positions per return-correlation cluster
    cluster_corr: float = 0.60
    # Portfolio stress: NIFTY gap moves, single stocks move beta * shock (+ idiosyncratic 0).
    stress_shocks: tuple = (-0.10, -0.06, -0.03, 0.03, 0.06, 0.10)
    max_stress_loss: float = 0.08    # worst stress scenario loss as fraction of capital


@dataclass(frozen=True)
class Costs:
    # Indian F&O options charges (verify against your broker's contract note).
    brokerage_per_order: float = 20.0
    stt_sell_premium: float = 0.001          # 0.1% of premium on sell side (from Oct-2024)
    exchange_txn: float = 0.0003503         # NSE options, on premium
    sebi_per_crore: float = 10.0
    stamp_buy: float = 0.00003               # 0.003% of premium on buy side
    gst: float = 0.18                        # on brokerage + exchange + SEBI


@dataclass(frozen=True)
class Config:
    liquidity: Liquidity = field(default_factory=Liquidity)
    selection: Selection = field(default_factory=Selection)
    forecast: Forecast = field(default_factory=Forecast)
    risk: Risk = field(default_factory=Risk)
    costs: Costs = field(default_factory=Costs)


DEFAULT = Config()
