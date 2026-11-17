import numpy as np
import pandas as pd

from vaaydo import ledger as LG


def _setup(path, side=1):
    dates = list(pd.bdate_range("2026-01-01", periods=len(path)))
    closes = pd.DataFrame({"X": path}, index=dates)
    opens = closes.shift(1).fillna(closes.iloc[0])          # open = previous close
    model = {d: {"X": side} for d in dates}
    ratios = {("X", 1): pd.Series(0.9, index=dates), ("X", -1): pd.Series(1.1, index=dates)}  # 10% Pragyam stop
    return dates, model, closes, opens, ratios


def test_entry_is_next_open_and_stop_trails():
    dates, model, closes, opens, ratios = _setup([100, 100, 110, 120, 125, 124])
    led = LG.replay(dates, model, closes, opens, ratios)
    p = led.held["X"]
    assert p.signal == dates[0] and p.exec == dates[1]       # signal at close d0, filled at d1's open
    assert p.best == 125                                      # trailed up with the best close
    assert np.isclose(p.stop, 125 * (1 + 1.5 * (0.9 - 1)))    # 1.5x the 10% distance below the best


def test_stop_decided_on_close_no_lookahead_and_ban():
    # rises to 120, then closes at 95: below the trailing stop (120 x 0.85 = 102)
    dates, model, closes, opens, ratios = _setup([100, 100, 120, 95, 96, 97])
    out = []
    for i in range(2, len(dates) + 1):
        led = LG.replay(dates[:i], model, closes, opens, ratios)
        out.append(("X" in led.held, "X" in led.exits_today))
    # the stop is only known at the close of day 3 (index 3) -> exit decided that day, not earlier
    assert out[0] == (True, False) and out[1] == (True, False)
    assert out[2] == (False, True)
    # banned while the signal stays on: no re-entry on later days
    assert out[3] == (False, False) and out[4] == (False, False)


def test_ban_lifts_when_signal_resets():
    dates, model, closes, opens, ratios = _setup([100, 100, 120, 95, 96, 97, 98, 99])
    model[dates[5]] = {}                                      # signal leaves the band
    led = LG.replay(dates, model, closes, opens, ratios)
    assert "X" in led.held and led.held["X"].signal == dates[6]   # a new signal is a new trade
