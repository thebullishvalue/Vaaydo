import numpy as np
import pandas as pd

from vaaydo import playbook as P


def _evidence(good_from: int = 0, n_cycles: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic study rows: 'Long straddle' pays +10%/trade from cycle `good_from`, the rest lose."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cycles):
        entry = pd.Timestamp("2024-01-01") + pd.DateOffset(months=c)
        for i in range(12):
            for rule, mu in (("Long straddle", 0.10 if c >= good_from else -0.05), ("IC 20Δ", -0.05)):
                rows.append(dict(entry=entry, symbol=f"S{i}", index=False, rule=rule, variant="TP50 + T-3",
                                 costs="net", ret_on_risk=mu + rng.normal(0, 0.05), atm_iv=rng.uniform(.2, .4),
                                 iv_rv1m=rng.uniform(.8, 1.2), skew=rng.normal(0, .02), chain_contracts=rng.integers(100, 9000)))
    return pd.DataFrame(rows)


def test_status_thresholds():
    assert P.status_of(dict(cycles=20, mean_ror=0.05, t=3.0)) == "live"
    assert P.status_of(dict(cycles=20, mean_ror=0.05, t=2.0)) == "watch"
    assert P.status_of(dict(cycles=5, mean_ror=0.05, t=9.0)) == "rejected"      # too little history
    assert P.status_of(dict(cycles=40, mean_ror=-0.01, t=-3.0)) == "rejected"


def test_consistent_winner_goes_live_and_loser_is_rejected():
    t = P.table(_evidence())
    st = t.set_index("rule")["status"]
    assert st["Long straddle · stocks"] == "live"
    assert st["IC 20Δ · stocks"] == "rejected"


def test_walk_forward_never_uses_the_future():
    # the rule only starts working at cycle 20: it must not be live before it has
    # MIN_CYCLES of *prior* good history, and never in cycle 0
    wf = P.walk_forward(_evidence(good_from=20, n_cycles=30))
    assert wf["live_rules"].iloc[:21].sum() == 0
    assert wf["live_rules"].iloc[0] == 0
