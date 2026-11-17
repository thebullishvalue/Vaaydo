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
from vaaydo import book as B, banlist  # noqa: E402
from vaaydo.config import DATA_DIR  # noqa: E402

fe = pd.read_parquet(DATA_DIR / "panels/opi_full.parquet")[["date", "symbol", "score"]]
mem = pickle.load(open(DATA_DIR / "results/mem.pkl", "rb"))
F1 = pd.read_parquet(DATA_DIR / "panels/futures_ohlc_legacy.parquet")
F2 = pd.read_parquet(DATA_DIR / "panels/futures_ohlc.parquet")
F2["date"] = pd.to_datetime(F2["date"])
ca = pickle.load(open(DATA_DIR / "results/ca_extra.pkl", "rb"))
P = B.Prices(pd.concat([F1, F2[F2["date"] > pd.to_datetime(F1["date"]).max()]]), extra_ca=ca)
st = pd.read_parquet(DATA_DIR / "panels/stops_full.parquet")
ban = None if "--no-ban" in sys.argv else banlist.banned()
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
