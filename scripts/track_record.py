"""Regenerate data/track_record.parquet: the stock book 2019 – today on real futures prices.

    VAAYDO_DATA=... python scripts/track_record.py [--no-ban]

Needs the research store (panels/opi_full, futures_ohlc[_legacy], stops_full, results/mem.pkl,
results/ca_extra.pkl, panels/banlist). Returns are on gross exposure (long + short legs / 2).
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from vaaydo import book as B, banlist, renames as RN, signal as SG  # noqa: E402
from vaaydo.config import DATA_DIR  # noqa: E402

mem = pickle.load(open(DATA_DIR / "results/mem.pkl", "rb"))
F1 = pd.read_parquet(DATA_DIR / "panels/futures_ohlc_legacy.parquet")
F2 = pd.read_parquet(DATA_DIR / "panels/futures_ohlc.parquet")
F1["date"], F2["date"] = pd.to_datetime(F1["date"]), pd.to_datetime(F2["date"])
F = pd.concat([F1, F2[F2["date"] > F1["date"].max()]], ignore_index=True)
# symbol renames (ZOMATO -> ETERNAL, MCDOWELL-N -> UNITDSPR, ...): one identity in every layer
ren = RN.detect(F)
print("renames:", ren)
F = RN.apply(F, ren, keys=("date", "expiry"))
inputs = RN.apply(pd.read_parquet(DATA_DIR / "panels/opi_full.parquet")[["date", "symbol", *SG.INPUTS]], ren)
fe = SG.scores(inputs)[["date", "symbol", "score"]]          # smoothing runs across the rename
mem = {K: {d: {ren.get(s, s): v for s, v in m.items()} for d, m in M.items()} for K, M in mem.items()}
ca = {(d, ren.get(s, s)): f for (d, s), f in pickle.load(open(DATA_DIR / "results/ca_extra.pkl", "rb")).items()}
P = B.Prices(F, extra_ca=ca)
st = RN.apply(pd.read_parquet(DATA_DIR / "panels/stops_full.parquet"), ren).sort_values(["symbol", "date"])
st[["stopL", "stopS"]] = st.groupby("symbol")[["stopL", "stopS"]].ffill(limit=5)   # as the live ledger
ban = None if "--no-ban" in sys.argv else frozenset((d, ren.get(s, s)) for d, s in banlist.banned())
kw = dict(stops=st, stop_mult=1.5, trail=True, ban_after_stop=True, start="2019-01-01", fno_ban=ban)
out = {"model_30": B.simulate(fe, P, enter=0.05, stay=0.15, **kw).ret}
for K in (6, 12, 20):
    out[f"k{K}"] = B.simulate(fe, P, membership=mem[K], **kw).ret
T = pd.DataFrame(out) / 2
if "--dry" not in sys.argv:
    T.astype("float32").to_parquet(Path(__file__).resolve().parent.parent / "data/track_record.parquet")
for c in T:
    r = T[c]
    eq = (1 + r).cumprod()
    yrs = (1 + r).groupby(r.index.year).prod() - 1
    print(f"{c:9s} SR {r.mean() / r.std() * np.sqrt(252):.2f} ret {r.mean() * 252:.1%} vol {r.std() * np.sqrt(252):.1%} "
          f"DD {(eq / eq.cummax() - 1).min():.1%} losing {int((yrs < 0).sum())}/{len(yrs)}")
