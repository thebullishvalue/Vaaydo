import numpy as np
import pandas as pd

from vaaydo import inference as INF


def _series(n=1500, beta=0.0, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    s = pd.Series(rng.normal(size=n), index=idx)
    r1 = pd.Series(rng.normal(0, 0.01, n), index=idx) + beta * 0.01 * s.shift(1).fillna(0)
    fwd5 = np.exp(np.log1p(r1).rolling(5).sum().shift(-5)) - 1
    return s, fwd5


def test_noise_is_not_accepted_and_a_planted_signal_is():
    s, f = _series(beta=0.0)
    assert not INF.accept(INF.evidence(s, f, 5))
    s, f = _series(beta=0.6)
    e = INF.evidence(s, f, 5)
    assert e["ic"] > e["mde"] and INF.accept(e)


def test_mde_shrinks_with_more_windows():
    assert INF.mde(1000) < INF.mde(300) < INF.mde(100)


def test_precedent_is_causal():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=700)
    st = pd.DataFrame(rng.normal(size=(700, 3)), index=idx)
    f = pd.Series(rng.normal(size=700), index=idx)
    a = INF.precedent(st, f, 5, min_pool=300)
    b = INF.precedent(st.iloc[:650], f.iloc[:650], 5, min_pool=300)
    assert np.allclose(a.iloc[:650].to_numpy(), b.to_numpy(), equal_nan=True)
    now = INF.precedent_now(st.iloc[:650], f.iloc[:650], 5, min_pool=300)
    assert np.isclose(now["read"], b.iloc[-1])


def test_oas_is_symmetric_positive_definite_and_shrinks_toward_identity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    S = np.cov(X, rowvar=False)
    R = INF.oas(S, 40)
    assert np.allclose(R, R.T) and np.linalg.eigvalsh(R).min() > 0
    assert np.linalg.cond(R) < np.linalg.cond(S)


def test_precedent_never_uses_an_outcome_not_yet_realised():
    # outcome at row i is known at open i+1+h; poison every outcome that would need a future open
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2015-01-01", periods=620)
    st = pd.DataFrame(rng.normal(size=(620, 2)), index=idx)
    f = pd.Series(rng.normal(size=620), index=idx)
    t, h = 619, 5
    clean = INF.precedent_now(st, f, h, min_pool=300)
    g = f.copy()
    g.iloc[t - h:] = 1e6                     # rows whose outcome is not realised at t's close
    assert np.isclose(INF.precedent_now(st, g, h, min_pool=300)["read"], clean["read"])
