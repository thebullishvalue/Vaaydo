"""Pragyam levels for sessions after the bundle: never from Yahoo when the price history is shipped,
and a level that cannot be computed is said, not hidden."""
import pandas as pd
import pytest

from vaaydo import bundle, curator
from vaaydo.data import history, equity


@pytest.mark.skipif(not bundle.HIST.exists(), reason="no shipped price history")
def test_missing_bar_is_reported_and_yahoo_not_called(monkeypatch):
    def no_yahoo(*a, **k):
        raise AssertionError("Yahoo was called")
    monkeypatch.setattr(history, "download", no_yahoo)
    monkeypatch.setattr(equity, "fetch_day", lambda *a, **k: None)          # NSE's bar unavailable too
    P0 = pd.read_parquet(bundle.PRG, columns=["date", "symbol"])
    end = pd.to_datetime(P0["date"]).max()
    live = set(P0.loc[pd.to_datetime(P0["date"]) == end, "symbol"])
    nxt = end + pd.offsets.BDay(1)
    P, how, warn = curator._bundle_pragyam(live, nxt)
    assert warn and "could not be computed" in how and f"{end:%d %b}" in how
    assert pd.to_datetime(P["date"]).max() == end                           # nothing invented for the new day


def test_inside_the_bundle_needs_nothing(monkeypatch):
    monkeypatch.setattr(history, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Yahoo")))
    end = pd.to_datetime(pd.read_parquet(bundle.PRG, columns=["date"])["date"]).max()
    P, how, warn = curator._bundle_pragyam({"RELIANCE"}, end)
    assert not warn and "from the bundle" in how
