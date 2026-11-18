"""NIFTY combined-reads round (docs/PREREGISTRATION.md, Nishkarsh round). Writes data/nifty_combined_tests.parquet and
data/nifty_combined_trades.parquet. Run: VAAYDO_DATA=... PYTHONPATH=. python scripts/nifty_round.py [breadth parquet]"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vaaydo import inference as INF, index_research as IR
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
BPATH = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'data/breadth_nse.parquet')
from vaaydo.config import DATA_DIR
X=pd.read_parquet(DATA_DIR / 'panels/index_research.parquet')
X=X[X.symbol=='NIFTY'].drop(columns=[c for c in X.columns if c.startswith('z_')]).sort_values('date').set_index('date')
B=pd.read_parquet(BPATH); B['DATE']=pd.to_datetime(B.DATE); B=B.set_index('DATE')
from vaaydo import breadth as BR
if 'breadth' not in B: B=BR.features(B.reset_index()).set_index('DATE')
X=X.join(B[['breadth','rel_breadth','rel_ad','count']].add_prefix('w_'))
X['mkt_breadth']=X.breadth   # stock-futures breadth (H12)
z=lambda s: IR.zscore(s)
REG={'mom60':1,'prg_T':1,'pcr_oi':1,'fii_fut_long':1,'client_fut_long':-1,'fii_opt_bias':1,'client_opt_bias':-1,'hedge_flow':1,
     'vrp':1,'vix':1,'skew':1,'mkt_breadth':1,'w_breadth':1,'w_rel_breadth':1,'w_rel_ad':1,'w_count':1}
Z=pd.DataFrame({c:z(X[c])*s for c,s in REG.items()})
X['mom20']=np.log(X.close/X.close.shift(20))
X['r_next']=X.r_oo.shift(-1)          # decided at close d -> open d+1 .. open d+2
PER={**IR.PERIODS}
def row(name, sig, h):
    e=INF.evidence(sig, X[f'fwd{h}'], h, momentum=X.mom20)
    per={k: INF.evidence(sig[(sig.index>=a)&(sig.index<=b)], X[f'fwd{h}'], h)['ic'] for k,(a,b) in PER.items()}
    return {'read':name,'h':h,'IC':e['ic'],'n':e['n'],'MDE':e['mde'],'HAC t':e['hac_t'],'CI':f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}]",
            'yrs+':f"{e['years_pos']:.0%} of {e['n_years']}",'mom IC':e['baseline'],**{f'IC {k[:4]}':v for k,v in per.items()},'accept':INF.accept(e)}
R=[]
sigs={'N1 market breadth':z(X.w_breadth),'N1 A/D momentum':z(X.w_rel_ad),
      'N1 A2 breadth > median':(X.w_breadth>X.w_breadth.rolling(250,min_periods=120).median()).astype(float),
      'N1 FII options stance':z(X.fii_opt_bias),'N2 composite':Z.mean(axis=1,skipna=True).where(Z.notna().sum(axis=1)>=10)}
state=pd.DataFrame({'trend':z(X.mom60),'breadth':z(X.w_breadth),'pcr':z(X.pcr_oi),'fii':z(X.fii_opt_bias),'iv':z(X.iv_30),'gamma':z(X.gex_dealer)})
for h in (5,10):
    sigs_h=dict(sigs)
    sigs_h['N3 precedent']=INF.precedent(state, X[f'fwd{h}'], h)
    a=np.sign(z(X.fii_opt_bias)); b=np.sign(z(X.w_breadth)); sigs_h['N4 agreement']=a.where(a==b,0).where(a.notna()&b.notna())
    for nm,s in sigs_h.items(): R.append(row(nm,s,h))
    if h==5: P5=sigs_h
T=pd.DataFrame(R); pd.set_option('display.width',260); pd.set_option('display.max_columns',30)
print(T.round(3).to_string(index=False))
print('\nTrades (tranche book h=5, NIFTY futures, 0.03% a side): net Sharpe by period, break-even cost')
def sr(x): return x.mean()/x.std()*np.sqrt(252) if len(x)>20 else np.nan
pos={'always long':pd.Series(1.0,index=X.index),'A2 breadth':P5['N1 A2 breadth > median'],
     'N2 composite':np.sign(P5['N2 composite']),'N3 precedent':np.sign(P5['N3 precedent']),'N4 agreement':P5['N4 agreement']}
TR=[]
for nm,p in pos.items():
    bt=INF.tranche_backtest(p.reindex(X.index), X.r_next, 5); n=bt['net']
    row={'Trade (h = 5 tranches)': nm, **{k[:4].title(): round(sr(n[a:b]),2) for k,(a,b) in {**PER,'all':('2019','2027')}.items()},
         'Break-even': '—' if nm=='always long' else f"{bt['breakeven']*1e4:.0f} bp"}
    TR.append(row); print(row)
T.to_parquet(ROOT / 'data/nifty_combined_tests.parquet', index=False)
pd.DataFrame(TR).to_parquet(ROOT / 'data/nifty_combined_trades.parquet', index=False)
