import numpy as np
import pandas as pd

from vaaydo import ledger as LG


def _market(paths: dict, expiries=("2026-03-26",), lots=None, ca=None, start="2026-01-01"):
    """paths: {expiry: [closes...]} one symbol X; open = previous close."""
    n = len(next(iter(paths.values())))
    days = list(pd.bdate_range(start, periods=n))
    rows = []
    for e, closes in paths.items():
        for i, d in enumerate(days):
            c = closes[i]
            if c is None:
                continue
            o = closes[i - 1] if i and closes[i - 1] is not None else c
            rows.append(dict(date=d, symbol="X", expiry=pd.Timestamp(e), open=o, close=c,
                             lot=(lots or {}).get((e, i), 10)))
    return LG.Market(pd.DataFrame(rows), ca=ca), days


def _ratios(days, r=0.9):
    return {("X", 1): pd.Series(r, index=days), ("X", -1): pd.Series(2 - r, index=days)}


def test_entry_next_open_and_trailing_stop():
    M, days = _market({"2026-03-26": [100, 100, 110, 120, 125, 124]})
    led = LG.replay(M, days, {d: {"X": 1} for d in days}, _ratios(days))
    p = led.held["X"]
    assert p.signal == days[0] and p.legs[0].entry_date == days[1] and p.legs[0].entry_px == 100
    assert p.best == 125 and np.isclose(p.stop, 125 * (1 + 1.5 * (0.9 - 1)))


def test_stop_decided_on_close_exit_next_open_then_ban():
    M, days = _market({"2026-03-26": [100, 100, 120, 95, 96, 97]})
    model = {d: {"X": 1} for d in days}
    led = LG.replay(M, days[:3], model, _ratios(days))
    assert "X" in led.held                                    # 120: not stopped
    led = LG.replay(M, days[:4], model, _ratios(days))
    assert led.decisions["X"][0] == "EXIT" and "stop" in led.decisions["X"][1]   # known at the close of day 3
    led = LG.replay(M, days, model, _ratios(days))
    assert "X" not in led.held and led.banned.get("X") == 1  # stays out while the signal is on
    p = [p for p, why, d in led.closed if p.symbol == "X"][0]
    assert p.legs[-1].exit_date == days[4] and p.legs[-1].exit_px == 95   # filled at the next open


def test_ban_lifts_when_signal_resets():
    M, days = _market({"2026-03-26": [100, 100, 120, 95, 96, 97, 98, 99]})
    model = {d: {"X": 1} for d in days}
    model[days[5]] = {}
    led = LG.replay(M, days, model, _ratios(days))
    assert "X" in led.held and led.held["X"].signal == days[6]


def test_roll_keeps_each_legs_lot_and_rescales_the_stop():
    # near contract expires on day 6; the next carries a bigger lot (an NSE revision)
    near, far = "2026-01-09", "2026-02-26"
    closes_near = [100, 100, 101, 102, 103, 104, 105]
    closes_far = [101, 101, 102, 103, 104, 105, 106]
    lots = {(far, i): 20 for i in range(7)}
    M, days = _market({near: closes_near, far: closes_far}, lots=lots, start="2026-01-01")
    led = LG.replay(M, days, {d: {"X": 1} for d in days}, _ratios(days))
    p = led.held["X"]
    assert [str(l.contract.date()) for l in p.legs] == [near, far]
    assert p.legs[0].lot == 10 and p.legs[1].lot == 20
    assert p.legs[0].exit_why == "rolled" and p.legs[0].exit_date == p.legs[1].entry_date


def test_corporate_action_rescales_entry_and_keeps_pnl_continuous():
    # 1:2 split on day 3: price halves, lot doubles on the same contract
    M, days = _market({"2026-03-26": [200, 200, 202, 101, 102]},
                      lots={("2026-03-26", i): (10 if i < 3 else 20) for i in range(5)})
    led = LG.replay(M, days, {d: {"X": 1} for d in days}, _ratios(days, 0.5))
    leg = led.held["X"].leg
    assert np.isclose(leg.adj, 0.5) and leg.lot == 20
    assert np.isclose(leg.pnl(1, last=102), (102 - 100) * 20)   # entry 200 x 10 == 100 x 20


def test_contracts_tradable_at_the_latest_date():
    days = pd.bdate_range("2026-11-02", "2026-11-16")
    rows = [dict(date=d, symbol="X", expiry=e, open=100.0, close=100.0, lot=10)
            for d in days for e in (pd.Timestamp("2026-11-23"), pd.Timestamp("2026-12-29"))]
    M = LG.Market(pd.DataFrame(rows))
    assert M.active[(days[-1], "X")] == pd.Timestamp("2026-11-23")     # 5 sessions left > 3
    assert M.fill_contract[(days[-1], "X")] == pd.Timestamp("2026-11-23")


def test_sessions_to_expiry_skip_nse_holidays():
    # 22 Dec 2026: 23, 24, 28, 29 Dec are sessions (25 Dec is a holiday): 4 left, not 5
    cal = pd.bdate_range("2026-12-14", "2026-12-22")
    assert LG._expiry_positions(cal, [pd.Timestamp("2026-12-29")])[0] - (len(cal) - 1) == 4


def test_fno_ban_defers_the_entry_to_the_first_unbanned_session():
    M, days = _market({"2026-03-26": [100, 100, 101, 102, 103, 104]})
    ban = frozenset({(days[1], "X"), (days[2], "X")})
    led = LG.replay(M, days, {d: {"X": 1} for d in days}, _ratios(days), fno_ban=ban)
    p = led.held["X"]
    assert p.legs[0].entry_date == days[3] and p.signal == days[2]   # re-decided at the last banned close


def test_no_next_contract_closes_at_the_roll_close():
    # only one contract ever listed (the stock is leaving F&O): close when the roll falls due
    M, days = _market({"2026-01-09": [100, 100, 101, 102, 103, 104, 105]}, start="2026-01-01")
    led = LG.replay(M, days, {d: {"X": 1} for d in days}, _ratios(days))
    assert "X" not in led.held
    p = [p for p, why, d in led.closed if p.symbol == "X"][0]
    leg = p.legs[-1]
    assert leg.exit_why == "no next contract" and leg.exit_date <= leg.contract
    assert leg.exit_px == M.px(leg.exit_date, "X", leg.contract)


def test_contract_that_stops_printing_closes_at_its_last_print():
    M, days = _market({"2026-03-26": [100, 100, 101, 102, None, None]})
    led = LG.replay(M, days, {d: {"X": 1} for d in days}, _ratios(days))
    p = [p for p, why, d in led.closed if p.symbol == "X"][0]
    assert p.legs[-1].exit_px == 102 and p.legs[-1].exit_date == days[3]
    assert p.legs[-1].exit_why == "contract stopped trading"


def test_simulator_closes_like_the_ledger_when_no_next_contract():
    from vaaydo import book as B
    days = pd.bdate_range("2026-01-01", periods=7)
    rows = [dict(date=d, symbol=f"S{k}", expiry=pd.Timestamp("2026-01-09"), open=100.0 + i, close=100.0 + i + 1, lot=10)
            for k in range(60) for i, d in enumerate(days)]
    P = B.Prices(pd.DataFrame(rows))
    sc = pd.DataFrame([dict(date=d, symbol=f"S{k}", score=float(k)) for d in days for k in range(60)])
    R = B.simulate(sc, P, enter=0.05, stay=0.15)
    names = R["names"].tolist()
    assert names[-1] == "" and any(n for n in names[:-1])     # held, then closed at the due roll
