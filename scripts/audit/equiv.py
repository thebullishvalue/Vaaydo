"""Coherence audit (docs/AUDIT.md). Run: VAAYDO_DATA=... PYTHONPATH=. python scripts/audit/equiv.py"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from vaaydo import curator, ledger as LG, book as B, bundle
r = curator.run()
F = pd.read_parquet(bundle.FUT); F['date'] = pd.to_datetime(F.date)
P = B.Prices(F)
PR = pd.read_parquet(bundle.PRG); PR['date'] = pd.to_datetime(PR.date)
st = PR.rename(columns={'stopL_r': 'stopL', 'stopS_r': 'stopS'})[['date', 'symbol', 'stopL', 'stopS']]
sc = r.scores[['date', 'symbol', 'score']]
R = B.simulate(sc, P, enter=0.05, stay=0.15, stops=st, stop_mult=1.5, trail=True, ban_after_stop=True,
               start=r.dates[0], fno_ban=r.ban)
sim = {d: set(n for n in names.split(',') if n) for d, names in R['names'].items()}
cps = [d for d in r.dates if d in sim][::6]
mism = 0
for d in cps:
    led = LG.replay(r.market, [x for x in r.dates if x <= d], r.daily, r.ratios, fno_ban=r.ban)
    # simulator's book after the fills of day d == ledger's held set after day d (decided at d-1, filled at d)
    # positions FILLED and still held through day d: filled (not pending) in led.held, plus
    # those whose exit was decided at d's close (they leave at d+1's open)
    L = {f"{s}{'+' if p.side > 0 else '-'}" for s, p in led.held.items() if not p.pending}
    L |= {f"{p.symbol}{'+' if p.side > 0 else '-'}" for p, why, dd in led.closed if dd == d and p.legs
          and p.legs[-1].exit_why not in ('no next contract', 'contract stopped trading')}
    S = sim[d]
    if L != S:
        mism += 1
        print(d.date(), 'ledger-only', sorted(L - S)[:5], 'sim-only', sorted(S - L)[:5])
print('checkpoints', len(cps), 'mismatched', mism)
