import numpy as np
import pandas as pd

from vaaydo import breadth as BR


def _raw(n=700, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2012-01-02", periods=n)
    p = 5000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    raw = pd.DataFrame({"DATE": d, "AD_RATIO": np.round(np.exp(rng.normal(0, 0.6, n)), 2), "NIFTY": p,
                        "NIFTY50_PE": 20 + rng.normal(0, 1, n).cumsum() * 0.05, "NIFTY50_PB": 3 + rng.normal(0, 0.05, n),
                        "NIFTY50_DY": 1.3 + rng.normal(0, 0.02, n)})
    for c in ("IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO", "US02Y", "US10Y", "US30Y"):
        raw[c] = 5 + rng.normal(0, 0.1, n).cumsum() * 0.05
    return raw


def test_breadth_is_the_10_day_ema_of_the_advancing_share():
    f = BR.features(_raw())
    a = f["AD_RATIO"] / (f["AD_RATIO"] + 1)
    assert np.allclose(f["breadth"], a.ewm(alpha=2 / 11, adjust=False).mean())


def test_no_reading_changes_when_future_sessions_arrive():
    raw = _raw()
    full = BR.features(raw)
    part = BR.features(raw.iloc[:600])
    cols = ["breadth", "rel_breadth", "rel_ad", "rsi", "count", "spread90", "spread200", "fair_pe", "pe_dev", "regime",
            "signal_str", "btd", "stt"]
    a, b = full.iloc[:600][cols], part[cols]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
