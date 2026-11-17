"""
The book: positions you actually took, marked daily from the bhavcopy, with the same
exit rules the backtest uses, and a record of predicted vs realised for every closed trade.

SQLite file at $VAAYDO_DATA/journal.db. Nothing here talks to a broker: you place orders
yourself and record fills here (the plan's prices are the default).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

from . import costs as C
from .backtest import Management
from .config import DATA_DIR, DEFAULT, Config
from .data import bhavcopy

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  opened TEXT, symbol TEXT, expiry TEXT, kind TEXT, vol_side TEXT, legs TEXT,
  lots INTEGER, lot_size INTEGER, entry_net REAL, max_loss REAL, max_profit REAL,
  ev REAL, pop REAL, atm_iv REAL, vol_forecast REAL,
  status TEXT DEFAULT 'open', closed TEXT, pnl REAL, exit_reason TEXT, note TEXT
);
"""


def _db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DATA_DIR / "journal.db")
    con.executescript(SCHEMA)
    return con


def open_position(planned, opened: date, note: str = "") -> int:
    """Record a `portfolio.Planned` as taken. All money fields are for the whole position."""
    c, e = planned.candidate, planned.candidate.evaluation
    legs = [dict(otype=l.otype, strike=l.strike, qty=l.qty, close=l.close) for l in e.structure.legs]
    with _db() as con:
        cur = con.execute(
            "INSERT INTO positions (opened,symbol,expiry,kind,vol_side,legs,lots,lot_size,entry_net,max_loss,"
            "max_profit,ev,pop,atm_iv,vol_forecast,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(opened), c.symbol, str(c.expiry.date()), e.structure.kind, e.structure.vol_side, json.dumps(legs),
             planned.lots, e.lot, e.entry_net * planned.lots, e.max_loss * planned.lots, e.max_profit * planned.lots,
             e.ev * planned.lots, e.pop, c.atm_iv, c.vol_forecast, note))
        return int(cur.lastrowid)


def positions(status: str | None = None) -> pd.DataFrame:
    with _db() as con:
        q = "SELECT * FROM positions" + (" WHERE status=?" if status else "")
        return pd.read_sql(q, con, params=(status,) if status else ())


def close_position(pid: int, pnl: float, when: date, reason: str = "manual") -> None:
    with _db() as con:
        con.execute("UPDATE positions SET status='closed', closed=?, pnl=?, exit_reason=? WHERE id=?",
                    (str(when), float(pnl), reason, pid))


def mark(asof: date | None = None, cfg: Config = DEFAULT, mgmt: Management = Management()) -> pd.DataFrame:
    """Mark open positions at the latest bhavcopy and say what to do with each."""
    asof = asof or bhavcopy.latest_day()
    book = positions("open")
    if book.empty:
        return book
    day = bhavcopy.load_range([asof], columns=["inst", "symbol", "expiry", "strike", "otype", "close", "settle", "spot", "date"],
                              filters=[("inst", "in", ["STO", "IDO"])])
    from .backtest import expiry_day_marks
    day["mark"] = day["settle"].where(day["settle"] > 0, day["close"])   # mark at NSE settlement
    day["close"] = expiry_day_marks(day)
    day["expiry"] = pd.to_datetime(day["expiry"]).dt.date.astype(str)
    cal = pd.DatetimeIndex([pd.Timestamp(d) for d in bhavcopy.available_days()])
    rows = []
    for _, p in book.iterrows():
        legs = json.loads(p["legs"])
        x = day[(day["symbol"] == p["symbol"]) & (day["expiry"] == p["expiry"])].set_index(["otype", "strike"])
        marks = [x["close"].get((l["otype"], float(l["strike"])), np.nan) for l in legs]
        qty = np.array([l["qty"] for l in legs], dtype=float)
        K = np.array([l["strike"] for l in legs], dtype=float)
        lots, lot = int(p["lots"]), int(p["lot_size"])
        if np.any(~np.isfinite(marks)):
            pnl, action = np.nan, "no mark: check manually"
        else:
            marks = np.array(marks)
            grid = np.r_[1e-6, np.unique(K), K.max() * 4]
            bounds = sum(q * (np.maximum(grid - k, 0) if o == "CE" else np.maximum(k - grid, 0))
                         for q, k, o in zip(qty, K, [l["otype"] for l in legs]))
            held = float(np.clip((qty * marks).sum(), bounds.min(), bounds.max()))
            slip = float((np.abs(qty) * C.half_spread(marks, cfg, K)).sum())
            value = (held - slip) * lot * lots
            pnl = p["entry_net"] + value - C.charges(C.fill(marks, -qty, cfg, K), -qty * lots, lot, cfg)
            exp = pd.Timestamp(p["expiry"])
            left = int(((cal > pd.Timestamp(asof)) & (cal <= exp)).sum())
            left = left if exp <= cal[-1] else int(np.busday_count(asof, exp.date()))
            action = "hold"
            if p["vol_side"] == "short" and mgmt.take_profit and pnl >= mgmt.take_profit * p["max_profit"]:
                action = "CLOSE: profit target"
            elif p["vol_side"] == "long" and mgmt.long_take_profit and pnl >= mgmt.long_take_profit * abs(p["entry_net"]):
                action = "CLOSE: profit target"
            elif left <= mgmt.exit_days_before:
                action = "CLOSE: expiry window (physical settlement)"
        spot = x["spot"].iloc[0] if len(x) else np.nan
        rows.append(dict(id=p["id"], symbol=p["symbol"], kind=p["kind"], expiry=p["expiry"], lots=lots,
                         spot=spot, pnl=pnl, max_profit=p["max_profit"], max_loss=p["max_loss"],
                         pct_of_max=pnl / p["max_profit"] if p["max_profit"] else np.nan, action=action))
    return pd.DataFrame(rows)


def calibration() -> pd.DataFrame:
    """Closed trades: did they do what the model said?"""
    c = positions("closed")
    if c.empty:
        return c
    c["win"] = c["pnl"] > 0
    c["ret_on_risk"] = c["pnl"] / c["max_loss"]
    c["pred_ev_to_risk"] = c["ev"] / c["max_loss"]
    return c
