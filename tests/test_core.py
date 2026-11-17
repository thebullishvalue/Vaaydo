import numpy as np
import pandas as pd
import pytest

from vaaydo import costs, flow, pricing
from vaaydo.config import DEFAULT
from vaaydo.evaluate import evaluate, payoff
from vaaydo.forecast import Shocks
from vaaydo.strategies import Leg, Structure, candidates
from vaaydo.vol import Smile, smile


def test_put_call_parity():
    F, K, T = 1000.0, 980.0, 30 / 365
    c = pricing.price(F, K, T, 0.3, True)
    p = pricing.price(F, K, T, 0.3, False)
    assert c - p == pytest.approx(np.exp(-0.065 * T) * (F - K), rel=1e-9)


def test_implied_vol_round_trip():
    F = np.full(5, 1500.0)
    K = np.array([1300, 1400, 1500, 1600, 1700.0])
    T = 25 / 365
    sig = np.array([0.42, 0.35, 0.30, 0.28, 0.27])
    is_call = K >= F
    px = pricing.price(F, K, T, sig, is_call)
    iv = pricing.implied_vol(px, F, K, T, is_call)
    assert np.allclose(iv, sig, atol=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    # below intrinsic -> NaN
    assert np.isnan(pricing.implied_vol(5.0, 1000.0, 900.0, 0.1, True))


def test_charges_positive_and_stt_on_sell_only():
    cb = costs.charges([10.0], [+1], 1000)
    cs = costs.charges([10.0], [-1], 1000)
    assert cs - cb == pytest.approx(DEFAULT.costs.stt_sell_premium * 10_000 - DEFAULT.costs.stamp_buy * 10_000, rel=1e-6)


def test_fill_crosses_spread():
    assert costs.fill(10.0, +1) > 10.0 > costs.fill(10.0, -1)


def _spread():
    return Structure("Put credit spread", "short", [Leg("PE", 950, -1, 10.0, 0.3, -0.25, contracts=400),
                                                    Leg("PE", 900, +1, 3.0, 0.32, -0.1, contracts=400)])


def test_payoff_put_credit_spread():
    st = _spread()
    S = np.array([1000, 950, 925, 900, 800.0])
    assert payoff(st, S).tolist() == [0, 0, -25, -50, -50]


def test_evaluation_max_loss_is_width_minus_credit():
    st = _spread()
    z = np.random.default_rng(0).standard_normal(4000)
    sh = Shocks(z, 20, 4000, 5)
    e = evaluate(st, sh, 1000.0, 0.25, lot=100)
    credit = (costs.fill(10.0, -1, contracts=400) - costs.fill(3.0, +1, contracts=400))
    assert e.entry_net == pytest.approx(credit * 100)
    # width - credit, plus entry charges and the cost of closing at the worst point
    assert (50 - credit) * 100 < e.max_loss < (50 - credit) * 100 + 400
    assert 0 < e.pop < 1


def test_shocks_centre_on_forward():
    z = np.random.default_rng(1).standard_t(4, 5000)
    sh = Shocks(z / z.std(), 22, 20000, 5)
    S = sh.terminal(2000.0, 0.3)
    assert S.mean() == pytest.approx(2000.0, rel=1e-9)
    lr = np.log(S / 2000.0)
    assert lr.std() == pytest.approx(0.3 * np.sqrt(22 / 252), rel=0.03)


def test_build_up_classification():
    ret = np.array([0.02, -0.02, 0.02, -0.02, 0.001])
    oi = np.array([0.05, 0.05, -0.05, -0.05, 0.1])
    assert flow.classify(ret, oi, 1).tolist() == [flow.LONG_BUILD, flow.SHORT_BUILD, flow.SHORT_COVER,
                                                  flow.LONG_UNWIND, flow.NEUTRAL]


def _chain(F=1000.0, T_days=30, vol=0.3):
    strikes = np.arange(800, 1201, 20.0)
    rows = []
    for k in strikes:
        for ot in ("CE", "PE"):
            px = float(pricing.price(F, k, T_days / 365, vol + 0.0002 * (1000 - k) / 10, ot == "CE"))
            rows.append(dict(symbol="TEST", expiry=pd.Timestamp("2026-01-30"), strike=k, otype=ot, close=round(px, 2),
                             contracts=500, oi=500_000, lot=500, spot=F, fwd=F, oi_chg=0))
    return pd.DataFrame(rows)


def test_smile_recovers_atm_vol_and_candidates_are_defined_risk():
    ch = _chain()
    sm = smile(ch, pd.Timestamp("2025-12-31"))
    assert sm is not None
    assert sm.atm_iv == pytest.approx(0.30, abs=0.005)
    assert sm.skew > 0       # puts richer by construction
    cs = candidates(sm)
    kinds = {c.kind for c in cs}
    assert {"Put credit spread", "Call credit spread", "Iron condor", "Iron butterfly", "Long straddle"} <= kinds
    for c in cs:
        g = np.linspace(1, 5000, 4000)
        assert np.isfinite(payoff(c, g).min())
        if c.vol_side == "short":
            assert sum(l.qty for l in c.legs) == 0      # every short is covered


def test_spread_shrinks_with_liquidity():
    thin, thick = costs.spread_frac(100), costs.spread_frac(10_000)
    assert thin == pytest.approx(0.03) and thick == pytest.approx(0.003)
    assert costs.half_spread(1000.0, strike=1000.0, contracts=1) <= 3.0   # deep ITM cap: 0.3% of strike


def test_expiry_day_marks_use_intrinsic_against_final_settlement():
    from vaaydo.backtest import expiry_day_marks
    d = pd.Timestamp("2026-11-17")
    o = pd.DataFrame(dict(date=[d, d, d - pd.Timedelta(days=1)], expiry=[d, d, d], strike=[25300.0, 25500.0, 25300.0],
                          otype=["CE", "PE", "CE"], settle=[25387.4, 25387.4, 150.0], mark=[25387.4, 25387.4, 150.0]))
    assert expiry_day_marks(o).round(2).tolist() == [87.4, 112.6, 150.0]
