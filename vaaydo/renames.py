"""
Symbol renames on NSE F&O (ZOMATO -> ETERNAL, GMRINFRA -> GMRAIRPORT, LTIM -> LTM, ...).

On a rename the exchange carries the open contracts over under the new symbol: same expiries,
same lot, continuous price. Without a map, a position would look closed (the old symbol stops
printing) and the stock's 10-session score would restart under the new name.

Detection, from the futures file alone: symbol A's last session is d, symbol B's first is the
next session; B lists exactly A's live expiries with the same lot in each, and B's first open
is within 3% of A's last close. (A stock leaving F&O while another joins at a month end does not
match: the joiner's lots and expiries differ.)

apply() relabels A as B in any frame with a symbol column, so every layer (inputs, scores,
futures, Pragyam levels, ban list, sectors) sees one continuous identity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect(F: pd.DataFrame, tol: float = 0.03) -> dict:
    """{old: new} from a futures frame (date, symbol, expiry, open, close, lot)."""
    F = F.copy()
    F["date"] = pd.to_datetime(F["date"])
    F["expiry"] = pd.to_datetime(F["expiry"])
    cal = np.array(sorted(F["date"].unique()))
    pos = {d: i for i, d in enumerate(cal)}
    span = F.groupby("symbol")["date"].agg(["min", "max"])
    last_day = cal[-1]
    starts = span[span["min"] > cal[0]].groupby("min").groups
    out = {}
    for a, (a0, a1) in span.iterrows():
        if a1 == last_day or pos[a1] + 1 >= len(cal):
            continue
        nd = cal[pos[a1] + 1]
        for b in starts.get(nd, []):
            if b == a:
                continue
            A = F[(F["symbol"] == a) & (F["date"] == a1) & (F["expiry"] > a1)].set_index("expiry")
            B = F[(F["symbol"] == b) & (F["date"] == nd)].set_index("expiry")
            if A.empty or B.empty or set(A.index) != set(B.index):
                continue
            if not (A["lot"].reindex(B.index).values == B["lot"].values).all():
                continue
            a_c = A.sort_index()["close"].iloc[0]
            b_o = B.sort_index()["open"].iloc[0]
            if a_c > 0 and b_o > 0 and abs(b_o / a_c - 1) <= tol:
                out[a] = b
    # chains (A -> B -> C) resolve to the final name
    for a in list(out):
        seen = {a}
        while out[a] in out and out[a] not in seen:
            seen.add(out[a])
            out[a] = out[out[a]]
    return out


def apply(df: pd.DataFrame, renames: dict, col: str = "symbol", keys: tuple = ("date",)) -> pd.DataFrame:
    """Relabel old symbols as their new names. Where both names carry a row for the same key
    (a source that already back-filled history under the new name), the new name's row wins."""
    if df is None or df.empty or not renames or col not in df:
        return df
    df = df.copy()
    was_old = df[col].astype(str).isin(renames.keys())
    df[col] = df[col].astype(str).replace(renames)
    k = [c for c in keys if c in df] + [col]
    if len(k) > 1:
        df = df.assign(_old=was_old).sort_values("_old").drop_duplicates(k, keep="first").drop(columns="_old")
    return df
