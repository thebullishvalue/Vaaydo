"""Coherence audit (docs/AUDIT.md). Run: VAAYDO_DATA=... PYTHONPATH=. python scripts/audit/asof.py"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from datetime import date
from vaaydo import curator, ledger as LG
full = curator.run()
cal = list(full.market.cal)
test = [d for d in cal if d >= pd.Timestamp('2024-11-01')][::45][:12]
bad = 0
for A in test:
    r = curator.run(A.date())
    leak = [n for n, v in (('cal', r.market.cal.max()), ('scores', r.scores['date'].max()), ('ban', max((d for d, _ in r.ban if d <= A + pd.Timedelta(days=0)), default=A))) if v > A]
    assert r.asof == A, (r.asof, A)
    # full replay's state at A (same rules, all data) vs the as-of run
    led_full = LG.replay(full.market, [d for d in full.dates if d <= A], full.daily, full.ratios, fno_ban=full.ban)
    a = {s: (p.side, [(l.contract, l.entry_date, round(l.entry_px, 4)) for l in p.legs]) for s, p in led_full.held.items() if not p.pending}
    b = {s: (p_side, legs) for s, p_side, legs in [(x, None, None) for x in []]}
    led_a = LG.replay(r.market, r.dates, r.daily, r.ratios, fno_ban=r.ban)
    b = {s: (p.side, [(l.contract, l.entry_date, round(l.entry_px, 4)) for l in p.legs]) for s, p in led_a.held.items() if not p.pending}
    diff = {s for s in set(a) | set(b) if a.get(s) != b.get(s)}
    pa = r.positions
    ok_contract = all(pd.Timestamp(c) > A for c in pa.loc[pa.action != 'EXIT', 'contract'] if c is not None)
    print(A.date(), 'held', len(b), 'diff vs full replay', len(diff), sorted(diff)[:4], '| leaks', leak, '| contracts alive', ok_contract)
    if diff:
        for s in sorted(diff)[:2]:
            print('    full', a.get(s)); print('    asof', b.get(s))
    bad += len(diff) + len(leak) + (not ok_contract)
print('TOTAL problems', bad)
