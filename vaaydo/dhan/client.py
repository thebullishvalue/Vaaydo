"""A small, careful DhanHQ v2 client: auth headers, rate limits, retries, and nothing else.

Credentials come from the environment:
    DHAN_ACCESS_TOKEN   the 24-hour access token (in a Claude session the network proxy injects it;
                        set the variable to any placeholder there)
    DHAN_CLIENT_ID      optional: read from /profile when absent
"""
from __future__ import annotations

import json
import os
import threading
import time

import requests

BASE = "https://api.dhan.co/v2"
MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"


class DhanError(RuntimeError):
    pass


class _Pace:
    """At most `rate` calls per second for one class of endpoint (Dhan: data 5/s, option chain 1 per 3 s)."""

    def __init__(self, min_gap: float):
        self.gap, self.last, self.lock = min_gap, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            dt = self.last + self.gap - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            self.last = time.monotonic()


class Dhan:
    def __init__(self, token: str | None = None, client_id: str | None = None, session: requests.Session | None = None):
        self.token = token or os.environ.get("DHAN_ACCESS_TOKEN", "")
        if not self.token:
            raise DhanError("DHAN_ACCESS_TOKEN is not set")
        self.s = session or requests.Session()
        self.s.headers.update({"Accept": "application/json", "Content-Type": "application/json",
                               "access-token": self.token})
        self._data, self._chain, self._quote = _Pace(0.25), _Pace(3.1), _Pace(1.1)
        self.client_id = client_id or os.environ.get("DHAN_CLIENT_ID") or str(self.profile()["dhanClientId"])
        self.s.headers["client-id"] = self.client_id

    # ── plumbing ─────────────────────────────────────────────────────────
    def _call(self, method: str, path: str, body: dict | None = None, pace: _Pace | None = None, tries: int = 4):
        err = None
        for k in range(tries):
            (pace or self._data).wait()
            try:
                r = self.s.request(method, BASE + path, data=json.dumps(body) if body is not None else None, timeout=90)
            except requests.RequestException as e:          # network blip: back off
                err = str(e)
                time.sleep(2 * (k + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                err = f"HTTP {r.status_code}"
                time.sleep(3 * (k + 1))
                continue
            try:
                j = r.json()
            except ValueError:
                raise DhanError(f"{path}: HTTP {r.status_code}, not JSON")
            if r.status_code != 200:
                if isinstance(j, dict) and j.get("errorCode") == "DH-907":   # "no data present": an empty answer
                    return {}
                raise DhanError(f"{path}: HTTP {r.status_code} {str(j)[:200]}")
            return j
        raise DhanError(f"{path}: gave up after {tries} tries ({err})")

    # ── endpoints ────────────────────────────────────────────────────────
    def profile(self) -> dict:
        return self._call("GET", "/profile")

    def daily(self, security_id: int | str, segment: str, instrument: str, start: str, end: str, oi: bool = True) -> dict:
        """Daily candles in [start, end) (Dhan's toDate is exclusive). For a futures contract id Dhan returns the
        series stitched by its rank in the listed chain (see docs/DHAN.md): trim to the contract's own life."""
        return self._call("POST", "/charts/historical", {"securityId": str(security_id), "exchangeSegment": segment,
                                                          "instrument": instrument, "oi": oi,
                                                          "fromDate": start, "toDate": end})

    def quote(self, segment: str, ids) -> dict:
        """Live quote per REAL contract (up to 1000 ids a call): OHLC, last price, volume, OI, last trade time."""
        out = {}
        ids = [int(x) for x in ids]
        for k in range(0, len(ids), 1000):
            j = self._call("POST", "/marketfeed/quote", {segment: ids[k:k + 1000]}, pace=self._quote)
            out.update(((j.get("data") or {}).get(segment)) or {})
        return out

    def expiries(self, underlying: int | str, segment: str) -> list[str]:
        j = self._call("POST", "/optionchain/expirylist", {"UnderlyingScrip": int(underlying), "UnderlyingSeg": segment},
                       pace=self._chain)
        return list(j.get("data") or [])

    def chain(self, underlying: int | str, segment: str, expiry: str) -> dict:
        j = self._call("POST", "/optionchain", {"UnderlyingScrip": int(underlying), "UnderlyingSeg": segment,
                                                 "Expiry": expiry}, pace=self._chain)
        return j.get("data") or {}

    def renew(self) -> str:
        """Dhan's RenewToken: expires THIS token and returns a new one (24 h). Only works while this one is active.
        The caller must store the new token before anything else: the old one is dead once this returns."""
        r = self.s.get(BASE + "/RenewToken", headers={"dhanClientId": self.client_id}, timeout=60)
        try:
            j = r.json()
        except ValueError:
            raise DhanError(f"RenewToken: HTTP {r.status_code}, not JSON")
        if r.status_code != 200:
            raise DhanError(f"RenewToken: HTTP {r.status_code} {str({k: v for k, v in j.items() if 'oken' not in k})[:200]}")
        tok = _find_token(j)
        if not tok:
            raise DhanError(f"RenewToken: no token in the response (keys: {sorted(j)})")
        return tok


def _find_token(j) -> str | None:
    """The new token, wherever the response puts it (Dhan does not document the response body)."""
    if isinstance(j, dict):
        for k in ("accessToken", "access_token", "token", "jwtToken"):
            v = j.get(k)
            if isinstance(v, str) and len(v) > 40:
                return v
        for v in j.values():
            t = _find_token(v)
            if t:
                return t
    return None
