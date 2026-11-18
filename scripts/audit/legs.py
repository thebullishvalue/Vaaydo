"""Coherence audit (docs/AUDIT.md). Run: VAAYDO_DATA=... PYTHONPATH=. python scripts/audit/legs.py"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from vaaydo import curator, ledger as LG
r = curator.run()
M = r.market
led = LG.replay(M, r.dates, r.daily, r.ratios, fno_ban=r.ban)
cal = M.cal; pos = {d: i for i, d in enumerate(cal)}
F = pd.DataFrame([(d, s, e) for (d, s, e) in M.c.keys()], columns=['date', 'symbol', 'expiry'])
listed = F.groupby(['symbol', 'expiry']).date.agg(['min', 'max'])
positions = [(p, 'held') for p in led.held.values() if not p.pending] + [(p, 'closed') for p, why, d in led.closed]
issues = []; nlegs = 0
def bad(kind, p, i, msg): issues.append(dict(kind=kind, symbol=p.symbol, signal=p.signal.date(), leg=i + 1, msg=msg))
for p, st in positions:
    if not p.legs:
        continue
    # 7. entry = the session after the signal
    if pos.get(p.legs[0].entry_date, -9) != pos.get(p.signal, -99) + 1:
        bad('entry not next session', p, 0, f'signal {p.signal.date()} entry {p.legs[0].entry_date.date()}')
    for i, l in enumerate(p.legs):
        nlegs += 1
        key = (p.symbol, l.contract)
        if key not in listed.index:
            bad('contract never listed', p, i, str(l.contract.date())); continue
        first, last = listed.loc[key]
        end = l.exit_date if l.exit_date is not None else r.asof
        # 1. listed on entry date and every held date
        if l.entry_date < first:
            bad('contract not yet listed at entry', p, i, f'{l.contract.date()} first listed {first.date()}, entered {l.entry_date.date()}')
        held = [d for d in cal if l.entry_date <= d <= end]
        miss = [d for d in held if (d, p.symbol, l.contract) not in M.c]
        if miss:
            bad('no price on a held day', p, i, f'{len(miss)} days e.g. {miss[0].date()}')
        # 4. never past expiry
        if end > l.contract:
            bad('held past expiry', p, i, f'held to {end.date()} expiry {l.contract.date()}')
        # 2. fill price is that day's real open (first leg) / close (roll legs)
        if i == 0:
            o = M.px(l.entry_date, p.symbol, l.contract, 'open')
            ref = o if np.isfinite(o) else M.px(l.entry_date, p.symbol, l.contract)
        else:
            ref = M.px(l.entry_date, p.symbol, l.contract)
        if not np.isclose(l.entry_px, ref, rtol=1e-6):
            bad('entry price not the exchange print', p, i, f'{l.entry_px} vs {ref}')
        # 3. contract = the rule's contract on the fill day
        if i == 0 and M.active.get((l.entry_date, p.symbol)) != l.contract:
            bad('first leg not the active contract', p, i, f'{l.contract.date()} vs active {M.active.get((l.entry_date, p.symbol))}')
        # 6. lot = the contract's lot on the entry day (before any corporate action)
        lot0 = M.lot.get((l.entry_date, p.symbol, l.contract))
        if l.adj == 1.0 and lot0 and l.lot != lot0:
            bad('lot differs from contract lot', p, i, f'{l.lot} vs {lot0}')
        # 5. chain continuity
        if i > 0:
            pr = p.legs[i - 1]
            if pr.exit_date != l.entry_date or pr.exit_why != 'rolled':
                bad('roll chain broken', p, i, f'prev exit {pr.exit_date} {pr.exit_why} / entry {l.entry_date}')
            if not np.isclose(pr.exit_px, M.px(pr.exit_date, p.symbol, pr.contract), rtol=1e-6):
                bad('roll exit not at old contract close', p, i, '')
            if l.contract <= pr.contract:
                bad('roll to an earlier contract', p, i, '')
        # roll timing: a non-final leg must be rolled when <= 3 sessions remain, not earlier
        if l.exit_why == 'rolled':
            left = pos.get(l.contract, len(cal) + 99) - pos[l.exit_date] if l.contract in pos else None
            if left is not None and left > 3:
                bad('rolled early', p, i, f'{left} sessions left')
print(f'positions {len(positions)}  legs {nlegs}  issues {len(issues)}')
if issues:
    I = pd.DataFrame(issues); print(I.kind.value_counts().to_string()); print(I.head(15).to_string())
# the display question: how many tickets show a contract different from the one the shown entry price was paid in
H = r.positions[r.positions.action != 'ENTER']
print('tickets (held/exit):', len(H), ' showing a contract other than the entry leg:', int((H.legs > 1).sum()))
tr = r.trades
print('closed trades', len(tr), ' with no return (NaN):', int(tr['ret'].isna().sum()))
print(tr[tr.why.str.contains('next contract|stopped', na=False)][['symbol','signal','entry','exit','ret','why']].to_string())
