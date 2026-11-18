import numpy as np
import pandas as pd

from vaaydo import signal as S


def _inputs(n_sym=60, n_days=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2026-01-01", periods=n_days):
        for i in range(n_sym):
            rows.append(dict(date=d, symbol=f"S{i:02d}", pcr_vol=rng.uniform(.2, 2), pcr_oi=rng.uniform(.3, 1.5),
                             skew=rng.normal(.02, .01), cskew=rng.normal(.02, .01), cpiv=rng.normal(0, .01),
                             put_oi_flow=rng.normal(0, .05), call_oi_flow=rng.normal(0, .05)))
    return pd.DataFrame(rows)


def test_opi_direction():
    day = _inputs(n_days=1)
    day.loc[day.symbol == "S00", ["pcr_vol", "pcr_oi", "skew", "put_oi_flow"]] = [9, 9, .5, .5]
    day.loc[day.symbol == "S00", ["cskew", "cpiv", "call_oi_flow"]] = [-.5, -.5, -.5]
    day.loc[day.symbol == "S01", ["pcr_vol", "pcr_oi", "skew", "put_oi_flow"]] = [0.01, 0.01, -.5, -.5]
    day.loc[day.symbol == "S01", ["cskew", "cpiv", "call_oi_flow"]] = [.5, .5, .5]
    p = S.opi_day_pct(day)
    assert p["S00"] == p.max() and p["S01"] == p.min()     # put-leaning = top, call-leaning = bottom


def test_indices_never_scored():
    day = _inputs(n_days=1)
    day.loc[0, "symbol"] = "NIFTY"
    assert "NIFTY" not in S.opi_day_pct(day).index


def test_replay_hysteresis_and_sides():
    sc = S.scores(_inputs())
    daily = {}
    book, log = S.replay(sc, daily=daily)
    last = sc[sc.date == sc.date.max()].set_index("symbol")["pct"]
    for s, p in book.items():
        assert (p.side == 1 and last[s] > 1 - S.STAY) or (p.side == -1 and last[s] <= S.STAY)
    for s, p in last.items():                                 # every extreme name is in the book
        if p > 1 - S.ENTER:
            assert book.get(s) and book[s].side == 1
        if p <= S.ENTER:
            assert book.get(s) and book[s].side == -1
