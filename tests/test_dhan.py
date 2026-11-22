"""The Dhan collector keeps only rows provably belonging to the contract they are filed under."""
from datetime import date

import pandas as pd

from vaaydo.dhan import mcx
from vaaydo.dhan.client import _find_token


def _fl():
    return pd.DataFrame({"root": ["CRUDEOIL"], "symbol": ["CRUDEOIL-18Dec2026-FUT"], "security_id": [1],
                         "expiry": [pd.Timestamp("2026-12-18")], "rank": [1], "tick": [100.0]})


def _q(vol=33717):
    return pd.DataFrame({"date": [pd.Timestamp("2026-11-20")], "symbol": ["CRUDEOIL-18Dec2026-FUT"],
                         "open": [7734.0], "high": [7787.0], "low": [7508.0], "volume": [vol], "oi": [8866]})


class FakeDhan:
    """History stitched by rank: 19 Nov is the November contract (expired that day), 20 Nov is December."""
    def daily(self, *a, **k):
        ts = [int(pd.Timestamp(d, tz="Asia/Kolkata").timestamp()) for d in ("2026-11-18", "2026-11-19", "2026-11-20")]
        return {"timestamp": ts, "open": [8010.0, 7728.0, 7734.0], "high": [8033.0, 7933.0, 7787.0],
                "low": [7668.0, 7711.0, 7508.0], "close": [7694.0, 7869.0, 7695.0], "volume": [9, 9, 33716],
                "open_interest": [1, 1, 8456]}


def test_latest_session_kept_only_when_it_matches_the_contracts_own_quote():
    got = mcx.pull_futures(FakeDhan(), _fl(), _q(), {}, date(2026, 11, 22), lookback=0)
    assert list(got["date"]) == [pd.Timestamp("2026-11-20")] and got["close"].iloc[0] == 7695.0
    assert mcx.pull_futures(FakeDhan(), _fl(), _q(vol=50000), {}, date(2026, 11, 22), lookback=0).empty


def test_never_across_an_expiry():
    # the commodity's last expiry was 19 Nov: the 18th and 19th belong to another contract and are dropped
    got = mcx.pull_futures(FakeDhan(), _fl(), _q(), {"CRUDEOIL": pd.Timestamp("2026-11-19")}, date(2026, 11, 22), lookback=7)
    assert list(got["date"]) == [pd.Timestamp("2026-11-20")]
    # with the previous expiry a month back, the earlier sessions are the same contract and are kept
    got = mcx.pull_futures(FakeDhan(), _fl(), _q(), {"CRUDEOIL": pd.Timestamp("2026-10-19")}, date(2026, 11, 22), lookback=7)
    assert len(got) == 3 and set(got["verified"]) == {"quote", "after-expiry"}


def test_token_found_wherever_the_response_puts_it():
    t = "e" * 64
    assert _find_token({"data": {"accessToken": t}}) == t and _find_token({"token": t}) == t
    assert _find_token({"status": "ok"}) is None
