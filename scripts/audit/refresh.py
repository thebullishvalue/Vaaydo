"""Replay the nightly refresh: drop the last session from a copy of every bundle file, run
bundle.refresh() against the copy, and compare what it rebuilt with what was there."""
import shutil, sys, tempfile, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from vaaydo import bundle
src = bundle.ROOT
tmp = Path(tempfile.mkdtemp()) / 'data'; shutil.copytree(src, tmp)
FILES = {'opi_inputs.parquet': ['date', 'symbol'], 'futures.parquet': ['date', 'symbol', 'expiry'], 'pragyam.parquet': ['date', 'symbol'],
         'index_panel.parquet': ['date', 'symbol'], 'participant_oi.parquet': ['date', 'client'], 'banlist.parquet': ['date', 'symbol'],
         'breadth_nse.parquet': ['DATE'], 'index_futures.parquet': ['date', 'symbol', 'expiry']}
last = pd.to_datetime(pd.read_parquet(tmp / 'opi_inputs.parquet')['date']).max()
orig = {}
for f, k in FILES.items():
    x = pd.read_parquet(tmp / f); dc = k[0]; x[dc] = pd.to_datetime(x[dc])
    orig[f] = x[x[dc] == last]
    x[x[dc] < last].to_parquet(tmp / f, index=False)
bundle.ROOT = tmp
bundle.FUT, bundle.PRG, bundle.SECT, bundle.INPUTS = tmp / 'futures.parquet', tmp / 'pragyam.parquet', tmp / 'sectors.parquet', tmp / 'opi_inputs.parquet'
bundle.refresh()
bad = 0
for f, k in FILES.items():
    x = pd.read_parquet(tmp / f); dc = k[0]; x[dc] = pd.to_datetime(x[dc])
    new = x[x[dc] == last]
    o = orig[f]
    if f == 'pragyam.parquet':        # levels are rebuilt for stocks in F&O now, not for ones that left
        live = set(pd.read_parquet(tmp / 'futures.parquet').pipe(lambda F: F[pd.to_datetime(F.date) == last]).symbol)
        o, new = o[o.symbol.isin(live)], new[new.symbol.isin(live)]
    if len(new) != len(o):
        print(f'{f:24s} rows rebuilt {len(new)} vs original {len(o)}'); bad += 1; continue
    num = [c for c in o.columns if c not in k and pd.api.types.is_numeric_dtype(o[c])]
    j = o.merge(new, on=k, suffixes=('_o', '_n'))
    worst = max((np.nanmax(np.abs(j[c + '_o'].astype(float) - j[c + '_n'].astype(float))) for c in num), default=0)
    print(f'{f:24s} rows {len(new)} match; largest numeric difference {worst:.3g}')
    if f == 'pragyam.parquet':
        for c in ('stopL_r', 'stopS_r', 'lo_r', 'up_r', 'close'):
            dif = (j[c + '_o'].astype(float) - j[c + '_n'].astype(float)).abs()
            print(f'      {c:8s} median |diff| {dif.median():.4g}  max {dif.max():.4g}  worst {j.loc[dif.idxmax(), "symbol"]}')
    bad += worst > 1e-3 and f not in ('pragyam.parquet',)
print('later sessions appended:', {f: int((pd.to_datetime(pd.read_parquet(tmp / f)[k[0]]) > last).sum()) for f, k in FILES.items()})
print('REFRESH PROBLEMS', bad)
