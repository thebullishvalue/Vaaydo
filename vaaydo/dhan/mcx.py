"""MCX through Dhan: the listed contracts, their end-of-day bars, and the option chains.

Stored per REAL contract (trading symbol, security id, expiry from Dhan's instrument master), not per rank.
Dhan's daily history for a contract id is stitched by the contract's current rank in its chain, so a fetch
never reaches back across an expiry of that commodity: only sessions after its latest expiry are taken.
"""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from .client import MASTER_URL, Dhan

SEG = "MCX_COMM"
IST = timezone(timedelta(hours=5, minutes=30))
#: commodities whose option chains are recorded (every MCX root with listed options, Nov 2026)
CHAIN_ROOTS = ("GOLD", "GOLDM", "SILVER", "SILVERM", "CRUDEOIL", "CRUDEOILM", "NATURALGAS", "NATGASMINI", "COPPER", "ZINC")
CHAIN_EXPIRIES = 3                       # the nearest option expiries recorded (the nearest may expire tomorrow)


def master(session: requests.Session | None = None) -> pd.DataFrame:
    """MCX futures and options from Dhan's public instrument master (one CSV for every exchange)."""
    s = session or requests.Session()
    r = s.get(MASTER_URL, headers={"User-Agent": "vaaydo/7 (end-of-day research)"}, timeout=180)
    r.raise_for_status()
    m = pd.read_csv(io.StringIO(r.text), low_memory=False)
    m = m[(m["SEM_EXM_EXCH_ID"] == "MCX") & m["SEM_INSTRUMENT_NAME"].isin(["FUTCOM", "OPTFUT"])].copy()
    m["root"] = m["SEM_TRADING_SYMBOL"].astype(str).str.split("-").str[0]
    m["expiry"] = pd.to_datetime(m["SEM_EXPIRY_DATE"]).dt.normalize()
    return m


def futures_list(m: pd.DataFrame) -> pd.DataFrame:
    f = m[m["SEM_INSTRUMENT_NAME"] == "FUTCOM"].sort_values(["root", "expiry"]).copy()
    f["rank"] = f.groupby("root").cumcount() + 1
    return f.rename(columns={"SEM_SMST_SECURITY_ID": "security_id", "SEM_TRADING_SYMBOL": "symbol",
                             "SEM_TICK_SIZE": "tick"})[["root", "symbol", "security_id", "expiry", "rank", "tick"]]


def _bars(j: dict) -> pd.DataFrame:
    if not j or "timestamp" not in j:
        return pd.DataFrame()
    d = pd.DataFrame({c: j.get(c) for c in ("timestamp", "open", "high", "low", "close", "volume")})
    d["oi"] = j.get("open_interest")
    d["date"] = pd.to_datetime(d["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    return d.drop(columns="timestamp")


def quotes(dh: Dhan, fl: pd.DataFrame) -> pd.DataFrame:
    """Every listed MCX future's quote (real contract, far months included), dated by its last trade."""
    q = dh.quote(SEG, fl["security_id"])
    rows = []
    for _, c in fl.iterrows():
        x = q.get(str(int(c["security_id"])))
        if not x:
            continue
        o = x.get("ohlc") or {}
        t = pd.to_datetime(x.get("last_trade_time"), format="%d/%m/%Y %H:%M:%S", errors="coerce")
        rows.append({"date": t.normalize() if pd.notna(t) else pd.NaT, "root": c["root"], "symbol": c["symbol"],
                     "security_id": int(c["security_id"]), "expiry": c["expiry"], "rank": int(c["rank"]),
                     "open": o.get("open"), "high": o.get("high"), "low": o.get("low"), "ltp": x.get("last_price"),
                     "avg_price": x.get("average_price"), "volume": x.get("volume"), "oi": x.get("oi"),
                     "lower_circuit": x.get("lower_circuit_limit"), "upper_circuit": x.get("upper_circuit_limit"),
                     "last_trade": t})
    return pd.DataFrame(rows)


def pull_futures(dh: Dhan, fl: pd.DataFrame, q: pd.DataFrame, last_expiry: dict, today: date, lookback: int = 7) -> pd.DataFrame:
    """Official end-of-day bars per real contract.

    Dhan's daily history for a contract id is stitched by the contract's CURRENT rank, so a row is kept only if
    it is provably that contract's:
      * the latest session: its open, high, low and volume equal the contract's own quote for that session;
      * an earlier session: it falls after the commodity's latest expiry (`last_expiry`, learned from the
        vault's own contract record), so the rank, and hence the contract, has not changed since.
    `lookback = 0` keeps the latest verified session only (the first run, before any expiry is known)."""
    out = []
    end = (today + timedelta(days=1)).isoformat()
    qi = q.set_index("symbol") if len(q) else pd.DataFrame()
    for _, c in fl.iterrows():
        if c["symbol"] not in qi.index or pd.isna(qi.loc[c["symbol"], "date"]):
            continue
        x = qi.loc[c["symbol"]]
        S = pd.Timestamp(x["date"])
        b = _bars(dh.daily(c["security_id"], SEG, "FUTCOM", (S - pd.Timedelta(days=max(lookback, 1))).date().isoformat(), end))
        if b.empty or S not in set(b["date"]):
            continue
        r = b[b["date"] == S].iloc[0]
        same = all(np.isfinite([r[k], x[k]]).all() and abs(float(r[k]) - float(x[k])) < 1e-6 for k in ("open", "high", "low")) \
            and abs(float(r["volume"]) - float(x["volume"] or 0)) <= 1
        if not same:
            continue                                        # the history row is another contract's (or not final)
        keep = b[b["date"] == S].assign(verified="quote")
        le = last_expiry.get(c["root"])
        if lookback and le is not None:
            older = b[(b["date"] < S) & (b["date"] > pd.Timestamp(le)) & (b["date"] >= S - pd.Timedelta(days=lookback))]
            keep = pd.concat([older.assign(verified="after-expiry"), keep])
        for k in ("root", "symbol", "security_id", "expiry", "rank"):
            keep[k] = c[k]
        keep["oi_close"] = x["oi"]                          # the quote's end-of-day OI (history's OI is a snapshot)
        keep.loc[keep["date"] != S, "oi_close"] = np.nan
        out.append(keep)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pull_chains(dh: Dhan, fl: pd.DataFrame, session: pd.Timestamp, roots=CHAIN_ROOTS, n_exp: int = CHAIN_EXPIRIES) -> pd.DataFrame:
    """Every strike of the nearest option expiries of each commodity, as Dhan reports it after the close."""
    rows = []
    for root in roots:
        f = fl[fl["root"] == root]
        if f.empty:
            continue
        und = int(f.iloc[0]["security_id"])                       # the front future: its expiry list spans the chain
        try:
            exps = sorted(dh.expiries(und, SEG))[:n_exp]
        except Exception:
            continue
        for e in exps:
            d = dh.chain(und, SEG, e)
            spot = d.get("last_price")
            for k, v in (d.get("oc") or {}).items():
                for side, t in (("ce", "CE"), ("pe", "PE")):
                    x = (v or {}).get(side)
                    if not x:
                        continue
                    g = x.get("greeks") or {}
                    rows.append({"date": session, "root": root, "underlying_id": und, "expiry": pd.Timestamp(e),
                                 "strike": float(k), "type": t, "security_id": x.get("security_id"),
                                 "ltp": x.get("last_price"), "prev_close": x.get("previous_close_price"),
                                 "oi": x.get("oi"), "prev_oi": x.get("previous_oi"), "volume": x.get("volume"),
                                 "prev_volume": x.get("previous_volume"), "iv": x.get("implied_volatility"),
                                 "delta": g.get("delta"), "gamma": g.get("gamma"), "theta": g.get("theta"),
                                 "vega": g.get("vega"), "bid": x.get("top_bid_price"), "ask": x.get("top_ask_price"),
                                 "underlying_px": spot})
    c = pd.DataFrame(rows)
    if len(c):
        for k in ("ltp", "prev_close", "iv", "delta", "gamma", "theta", "vega", "bid", "ask", "underlying_px"):
            c[k] = pd.to_numeric(c[k], errors="coerce").astype("float32")
        for k in ("oi", "prev_oi", "volume", "prev_volume"):
            c[k] = pd.to_numeric(c[k], errors="coerce").fillna(0).astype("int64")
    return c


def now_ist() -> datetime:
    return datetime.now(IST)
