"""Append the latest sessions' OPI inputs to data/opi_inputs.parquet (run after ~18:30 IST).

    python scripts/refresh_bundle.py
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vaaydo import signal  # noqa: E402
from vaaydo.data import bhavcopy  # noqa: E402
from vaaydo.features import day_features  # noqa: E402

bundle = signal.BUNDLED
have = pd.read_parquet(bundle)
have["date"] = pd.to_datetime(have["date"])
last = have["date"].max().date()
new = []
d = last + timedelta(days=1)
while d <= date.today():
    if d.weekday() < 5 and bhavcopy.fetch_day(d) is not None:
        f = day_features(d)[["date", "symbol", *signal.INPUTS]]
        for c in signal.INPUTS:
            f[c] = f[c].astype("float32")
        new.append(f)
        print("added", d, len(f))
    d += timedelta(days=1)
if new:
    out = pd.concat([have, *new], ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
    out.to_parquet(bundle, index=False, compression="zstd")
    print("bundle now ends", out["date"].max().date())
else:
    print("nothing new after", last)
