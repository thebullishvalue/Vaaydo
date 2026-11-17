import numpy as np
import pandas as pd

from vaaydo import gamma as G
from vaaydo import index_options as O
from vaaydo import pricing


def test_gamma_matches_finite_difference_of_delta():
    F, K, T, s = 100.0, 100.0, 0.05, 0.2
    h = 0.01
    fd = (pricing.delta(F + h, K, T, s, True) - pricing.delta(F - h, K, T, s, True)) / (2 * h)
    assert np.isclose(G.gamma(F, K, T, s), fd, rtol=1e-3)


def test_dealer_gex_sign_follows_call_vs_put_oi():
    asof = pd.Timestamp("2026-11-02")
    exp = pd.Timestamp("2026-11-24")
    fut = pd.DataFrame({"expiry": [exp], "close": [100.0], "oi": [1000.0]})
    rows = []
    for k in (90, 95, 100, 105, 110):
        for ot in ("CE", "PE"):
            px = float(pricing.price(100, k, 22 / 365, 0.25, ot == "CE"))
            rows.append(dict(expiry=exp, strike=float(k), otype=ot, close=px, contracts=10,
                             oi=500.0 if ot == "CE" else 50.0, oi_chg=0.0))
    r = G.underlying(pd.DataFrame(rows), fut, asof, 100.0)
    assert r["gex_dealer"] > 0 and r["gex_seller"] < 0


def _chain(settle):
    cal = pd.bdate_range("2026-11-02", periods=7)
    E = cal[-1]
    F, sig = 100.0, 0.25
    rows = []
    for i, d in enumerate(cal):
        T = max((E - d).days, 1) / 365
        for k in np.arange(80, 121, 1.0):
            for ot in ("CE", "PE"):
                px = float(pricing.price(F, k, T, sig, ot == "CE"))
                st = settle if d == E else px
                rows.append(dict(symbol="NIFTY", expiry=E, strike=k, otype=ot, open=px, close=px, settle=st,
                                 contracts=100, oi=1000, lot=50, date=d))
    P = pd.DataFrame({"symbol": "NIFTY", "date": cal, "spot": F})
    return pd.DataFrame(rows), P


def test_condor_pays_full_credit_when_settling_between_the_shorts():
    C, P = _chain(settle=100.0)
    y = O.cycles(C, P, O.Spec())
    assert len(y) == 1
    r = y.iloc[0]
    assert r["payout"] == 0 and np.isclose(r["pnl"], r["credit"] - r["costs"])
    assert r["kp_l"] < r["kp_s"] < 100 < r["kc_s"] < r["kc_l"]


def test_condor_loss_is_capped_and_missing_settle_falls_back_to_spot():
    C, P = _chain(settle=0.0)            # file carries no settlement: use the index close (100)
    y = O.cycles(C, P, O.Spec())
    assert y.iloc[0]["settle"] == 100.0
    C2, P2 = _chain(settle=130.0)
    r = O.cycles(C2, P2, O.Spec()).iloc[0]
    assert np.isclose(r["ret"], -1.0)    # beyond the call wing: lose exactly the max loss
