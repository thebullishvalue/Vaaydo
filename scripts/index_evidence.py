"""Regenerate the index desk's shipped evidence and history (needs the research store).

    VAAYDO_DATA=... python scripts/index_evidence.py

data/index_panel.parquet     readings per index per session, 2019 on
data/participant_oi.parquet  NSE participant-wise OI, 2019 on
data/index_tests.parquet     every registered test: reading, horizon, period, t
data/index_cycles.parquet    the registered condor and fly, every cycle, real fills
data/banlist.parquet         the F&O ban list, 2019 on
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vaaydo import banlist, index_options as O, index_research as IR, indices as I  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data"
r = IR.run()
T = r["tests"].copy()
X = r["X"]
X.to_parquet(I.PANEL.parent / "index_research.parquet", index=False)

# volatility tests (H2, H3a) and the strategies, as rows of the same table
rows = []
for p, (a, b) in {**IR.PERIODS, "all": ("2019-01-01", "2026-12-31")}.items():
    d = X[(X["date"] >= a) & (X["date"] <= b)].copy()
    d["y"], d["x"] = d["rv_fwd5"] ** 2, d["iv_30"] ** 2
    rows.append(dict(reading="H2 IV forecasts realised vol", h=5, period=p, t=IR.pooled_t(d, "x", "y", 5)[1], sign=1))
    d["ratio"], d["gpos"] = d["rv_fwd5"] / d["iv_front"], (d["gex_dealer"] > 0).astype(float)
    rows.append(dict(reading="H3a positive gamma damps vol", h=5, period=p, t=IR.pooled_t(d, "gpos", "ratio", 5)[1], sign=-1))
T = pd.concat([T, pd.DataFrame(rows)], ignore_index=True)
T.to_parquet(OUT / "index_tests.parquet", index=False)

C = pd.read_parquet(O.CHAIN)
P = pd.read_parquet(I.PANEL)
cyc = []
for nm, spec in (("condor", O.Spec()), ("fly", O.Spec(fly=True, wing_m=1.5))):
    cyc.append(O.cycles(C, P, spec).assign(structure=nm))
pd.concat(cyc, ignore_index=True).to_parquet(OUT / "index_cycles.parquet", index=False)

keep = [c for c in P.columns if c not in ("err",)]
P[keep].to_parquet(OUT / "index_panel.parquet", index=False, compression="zstd")
shutil.copy(I.PART, OUT / "participant_oi.parquet")
shutil.copy(banlist.PATH, OUT / "banlist.parquet")
print(IR.verdicts(r["tests"]))
