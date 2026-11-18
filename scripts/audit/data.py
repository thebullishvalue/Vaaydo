"""Data-layer audit: every shipped file, cross-checked. Run: VAAYDO_DATA=... PYTHONPATH=. python scripts/audit/data.py"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from vaaydo import bundle
D = bundle.ROOT
out = []
def say(k, v): out.append((k, v)); print(f'{k:60s} {v}')
F = pd.read_parquet(D / 'futures.parquet'); F['date'] = pd.to_datetime(F.date); F['expiry'] = pd.to_datetime(F.expiry)
I = pd.read_parquet(D / 'opi_inputs.parquet'); I['date'] = pd.to_datetime(I.date)
P = pd.read_parquet(D / 'pragyam.parquet'); P['date'] = pd.to_datetime(P.date)
cal_f, cal_i, cal_p = set(F.date), set(I.date), set(P.date)
say('futures sessions / first / last', (len(cal_f), min(cal_f).date(), max(cal_f).date()))
say('opi_inputs sessions missing vs futures', sorted(d.date() for d in cal_f - cal_i)[:8])
say('futures sessions missing vs opi_inputs', sorted(d.date() for d in cal_i - cal_f)[:8])
say('pragyam sessions missing vs futures (last 300)', len([d for d in sorted(cal_f)[-300:] if d not in cal_p]))
say('weekend sessions in futures (special sessions)', sorted(d.date() for d in cal_f if d.weekday() >= 5))
# duplicates
say('dup (date,symbol,expiry) in futures', int(F.duplicated(['date', 'symbol', 'expiry']).sum()))
say('dup (date,symbol) in opi_inputs', int(I.duplicated(['date', 'symbol']).sum()))
say('dup (date,symbol) in pragyam', int(P.duplicated(['date', 'symbol']).sum()))
# prices
say('futures rows open<=0 or close<=0 (untraded)', int(((F.open <= 0) | (F.close <= 0)).sum()))
say('futures rows close NaN', int(F.close.isna().sum()))
say('futures rows lot <= 0', int((F.lot <= 0).sum()))
say('futures rows expiry < date', int((F.expiry < F.date).sum()))
# expiry calendar: contracts per symbol per day
n = F.groupby(['date', 'symbol']).expiry.nunique()
say('symbol-days with != 3 contracts listed (count by n)', n.value_counts().sort_index().to_dict())
# symbol continuity: symbols that vanish and others that appear same day (renames)
first = F.groupby('symbol').date.min(); last = F.groupby('symbol').date.max()
end = F.date.max(); start = F.date.min()
gone = last[last < end]; new = first[first > start]
pairs = [(g, d.date(), [n for n, dn in new.items() if 0 <= (dn - d).days <= 5]) for g, d in gone.items()]
say('symbols that stopped (with symbols that started within 5 days)', [p for p in pairs if p[2]][:12])
say('symbols that stopped (all)', len(gone))
# opi inputs sanity
for c in ['pcr_vol', 'pcr_oi', 'skew', 'cskew', 'cpiv', 'put_oi_flow', 'call_oi_flow']:
    x = I[c]
    say(f'opi {c}: NaN share / p1 / p99', (round(x.isna().mean(), 3), round(x.quantile(.01), 3), round(x.quantile(.99), 3)))
say('opi names per session min / median', (int(I.groupby('date').size().min()), int(I.groupby('date').size().median())))
thin = I.groupby('date').size(); say('sessions with < 50 names', sorted(d.date() for d in thin[thin < 50].index)[:10])
# pragyam ratios
for c in ['stopL_r', 'stopS_r', 'lo_r', 'up_r']:
    x = P[c]; say(f'pragyam {c}: NaN / min / max', (round(x.isna().mean(), 4), round(x.min(), 3), round(x.max(), 3)))
say('pragyam stopL_r >= 1 (stop above price for a long)', int((P.stopL_r >= 1).sum()))
say('pragyam stopS_r <= 1 (stop below price for a short)', int((P.stopS_r <= 1).sum()))
# pragyam close vs futures close (spot vs near future): basis sanity
near = F[F.expiry >= F.date].sort_values('expiry').groupby(['date', 'symbol']).first()['close']
j = P.set_index(['date', 'symbol'])['close'].to_frame('spot').join(near.rename('fut'), how='inner').dropna()
r = np.log(j.fut / j.spot)
say('|log(near fut / pragyam close)| > 5% rows', int((r.abs() > 0.05).sum()))
print(j[r.abs() > 0.05].assign(r=r[r.abs() > 0.05]).sort_values('r').head(8).to_string())
