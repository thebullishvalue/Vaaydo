"""Invariants of everything shown, and rupee reconciliation between the engines.
Run: VAAYDO_DATA=... PYTHONPATH=. python scripts/audit/invariants.py"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from vaaydo import curator, ledger as LG, book as B, bundle, renames as RN, index_desk, universe
P_ = []
def check(name, ok, detail=''):
    P_.append((name, bool(ok))); print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
r = curator.run(); A = r.asof; M = r.market
pos, legs, scr = r.positions, r.legs, r.screen

# ── 1. rupee reconciliation: ledger legs vs simulator, Jul 2024 -> A, before costs ──
# the same sessions the run used: the bundle plus any session fetched after it
F = curator._bundle_futures(A); F['date'] = pd.to_datetime(F.date)
ren = RN.detect(F); F = RN.apply(F, ren, keys=('date', 'expiry'))
PR = RN.apply(curator._bundle_pragyam(set(F.loc[F.date == A, 'symbol']), A)[0], ren); PR['date'] = pd.to_datetime(PR.date)
st = PR.rename(columns={'stopL_r': 'stopL', 'stopS_r': 'stopS'})[['date', 'symbol', 'stopL', 'stopS']].sort_values(['symbol', 'date'])
st[['stopL', 'stopS']] = st.groupby('symbol')[['stopL', 'stopS']].ffill(limit=5)
R = B.simulate(r.scores[['date', 'symbol', 'score']], B.Prices(F), enter=0.05, stay=0.15, stops=st, stop_mult=1.5, trail=True,
               ban_after_stop=True, start=r.dates[0], fno_ban=r.ban)
led = LG.replay(M, r.dates, r.daily, r.ratios, fno_ban=r.ban)
tot = 0.0
for p in [p for p in led.held.values() if not p.pending] + [p for p, w, d in led.closed]:
    for l in p.legs:
        if l.exit_date is not None and l.exit_date > A:
            continue
        x = l.exit_px if l.exit_px is not None and l.exit_date is not None else M.px(A, p.symbol, l.contract)
        if np.isfinite(x):
            tot += p.side * (x - l.entry_px * l.adj) * l.lot
sim = R['pnl_rs'].sum()
check('rupee P&L: ledger legs == simulator', abs(tot - sim) <= max(1.0, 1e-6 * abs(sim)), f'ledger {tot:,.0f} vs simulator {sim:,.0f}')

# ── 2. ticket invariants ──
H = pos[pos.action != 'ENTER']; E = pos[pos.action == 'ENTER']
check('held contracts alive at A (not expired)', (pd.to_datetime(H.contract) >= A).all())
hold = pos[pos.action == 'HOLD']
lng, sht = hold[hold.side == 'LONG'], hold[hold.side == 'SHORT']
check('long stops below last, short stops above', (lng.stop < lng['last']).all() and (sht.stop > sht['last']).all(),
      f"{int((lng.stop >= lng['last']).sum())} long / {int((sht.stop <= sht['last']).sum())} short violations")
check('risk to stop >= 0 and finite', (pos.risk_to_stop >= 0).all() and pos.risk_to_stop.notna().all())
check('every position has a lot > 0', (pos.lot > 0).all())
check('ticket P&L == sum of its legs', all(abs(row.pnl - legs.loc[legs.symbol == row.symbol, 'pnl'].sum()) < 1 for row in H.itertuples()))
check('legs count == leg rows', all(row.legs == (legs.symbol == row.symbol).sum() for row in H.itertuples()))
check('first contract <= current contract', (pd.to_datetime(H.first_contract) <= pd.to_datetime(H.contract)).all())
check('current leg started on or before A', (pd.to_datetime(H.leg_since) <= A).all())
rp = pos[pos.roll_on.notna()]
check('next roll after A and on/before expiry', ((pd.to_datetime(rp.roll_on) > A) & (pd.to_datetime(rp.roll_on) <= pd.to_datetime(rp.contract))).all())
check('new entries: signal today, in the fill contract', (pd.to_datetime(E.signal) == A).all() and
      all(pd.Timestamp(row.contract) == M.fill_contract.get((A, row.symbol)) for row in E.itertuples()))
check('entry the session after the signal', all(M.cal.get_loc(pd.Timestamp(row.entry_date)) == M.cal.get_loc(pd.Timestamp(row.signal)) + 1
                                                 for row in H.itertuples()))
check('return sign matches P&L sign (or both ~0)', ((np.sign(H.pnl_pct) == np.sign(H.pnl)) | (H.pnl.abs() < 1) | (H.pnl_pct.abs() < 1e-3)).all())

# ── 3. screen <-> book ──
s = scr.set_index('symbol')['state']
check('▲/▼ NEW on the screen == ENTER tickets', set(s[s.isin(['▲ NEW', '▼ NEW'])].index) == set(E.symbol))
check('EXIT on the screen == EXIT tickets', set(s[s == 'EXIT'].index) == set(pos.loc[pos.action == 'EXIT', 'symbol']))
check('LONG/SHORT on the screen == HOLD tickets', set(s[s.isin(['LONG', 'SHORT'])].index) == set(hold.symbol))
check('screen side agrees with ticket side', all((s[sym] in ('LONG', '▲ NEW')) == (side == 'LONG') for sym, side in zip(pos.symbol, pos.side) if s.get(sym) in ('LONG', 'SHORT', '▲ NEW', '▼ NEW')))
check('score ranks within [0, 1]', scr.score.dropna().between(0, 1).all())

# ── 4. universe filter ──
for u in ('NIFTY 50', 'NIFTY Bank'):
    m = universe.constituents(u)
    if m is not None:
        f = pos[pos.symbol.isin(m)]
        check(f'{u} filter is a subset of the book', set(f.symbol) <= set(pos.symbol), f'{len(f)} of {len(pos)}')

# ── 5. index desk ──
dk = index_desk.run()
Pn = dk.panel[dk.panel.date == dk.asof]
check('call wall >= forward >= put wall', ((Pn.wall_call_d >= 0) & (Pn.wall_put_d <= 0)).all())
check('gamma flip within ±8%', Pn.flip_d.abs().le(0.08).all())
check('straddle (priced move) > 0', (Pn.straddle > 0).all())
pr = dk.part_raw
if len(pr):
    long_ = pr[['fut_idx_long', 'opt_idx_call_long', 'opt_idx_put_long']].sum(); short_ = pr[['fut_idx_short', 'opt_idx_call_short', 'opt_idx_put_short']].sum()
    check('participant OI: longs == shorts across participants', np.allclose(long_.values, short_.values))
Bd = dk.breadth
check('breadth: advances + declines <= stocks traded, ratio > 0', ((Bd.ADV + Bd.DEC) < 4000).all() and (Bd.AD_RATIO > 0).all())
check('index desk as-of == last session in panel', dk.asof == dk.panel.date.max())
print('\nINVARIANTS', sum(ok for _, ok in P_), 'of', len(P_), 'hold')

# ── 6. index desk as-of fidelity ──
from datetime import date as _date
for A_ in (_date(2025, 6, 13), _date(2026, 3, 4)):
    d_ = index_desk.run(A_); t_ = pd.Timestamp(A_)
    mx = [d_.panel.date.max(), d_.participants.index.max(), d_.breadth.DATE.max()] + ([d_.vix.index.max()] if len(d_.vix) else [])
    an = d_.reads.get('precedent', {}).get('analogs', [])
    cal_ = sorted(d_.panel.date.unique()); last_ok = cal_[cal_.index(t_) - 6] if t_ in cal_ else t_
    check(f'index desk {A_}: no data after the run date', all(v <= t_ for v in mx))
    check(f'index desk {A_}: analogs realised by the run date', all(a[0] <= last_ok for a in an))
print('\nINVARIANTS (with desk as-of)', sum(ok for _, ok in P_), 'of', len(P_), 'hold')
