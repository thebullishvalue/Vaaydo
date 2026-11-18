"""
One run of the screener, for any session since the bundle starts.

    run(asof) -> Run        every F&O stock's state and readings, the model book with
                            leg-by-leg history, closed trades, and the day's option contracts
    sized(run, capital, leverage) -> (positions, legs)    the same book cut to your capital

States, per stock, at the close of `asof`:
    ▲ NEW / ▼ NEW     the score entered the top / bottom 5% today: enter at the next open
    LONG / SHORT      held (since the signal date shown)
    EXIT              leaves the book at the next open (signal left the band, or stop)
    STOPPED           stopped out; out until its signal leaves the band
    READY ▲ / ▼       in the 5–15% band and moving toward an entry
    —                 nothing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import banlist, bundle, ledger as LG, session, signal
from .data import bhavcopy
from .simulate import futures_cost

STATE_ORDER = {"▲ NEW": 0, "▼ NEW": 1, "EXIT": 2, "LONG": 3, "SHORT": 4, "STOPPED": 5, "READY ▲": 6, "READY ▼": 7, "—": 8}


@dataclass
class Run:
    asof: pd.Timestamp
    screen: pd.DataFrame          # one row per F&O stock
    positions: pd.DataFrame       # one row per open / exiting position (model book)
    legs: pd.DataFrame            # one row per contract leg of every position shown
    trades: pd.DataFrame          # closed trades since the bundle start
    contracts: pd.DataFrame       # the day's option contracts with their flow
    scores: pd.DataFrame
    market: object
    daily: dict
    ext: dict
    ratios: dict
    dates: list
    notes: list = field(default_factory=list)
    ban: frozenset = frozenset()
    ban_next: tuple = (None, None)


def _sessions_up_to(asof: date | None) -> list[date]:
    """Make sure the six sessions up to `asof` (or the latest) are on disk."""
    if asof is None:
        return session.recent_sessions(6)
    s = __import__("requests").Session()
    got, d = [], asof
    while len(got) < 6 and (asof - d).days < 30:
        if d.weekday() < 5 and bhavcopy.fetch_day(d, s) is not None:
            got.append(d)
        d -= timedelta(days=1)
    return sorted(got)


def _bundle_futures(last_needed: pd.Timestamp) -> pd.DataFrame:
    F = pd.read_parquet(bundle.FUT)
    F["date"] = pd.to_datetime(F["date"])
    end = F["date"].max()
    if last_needed > end:
        extra = [d for d in bhavcopy.available_days() if end < pd.Timestamp(d) <= last_needed]
        if extra:
            F = pd.concat([F, bundle.futures_rows(extra)], ignore_index=True)
    return F[F["date"] <= last_needed]


def _bundle_pragyam(symbols, asof: pd.Timestamp) -> pd.DataFrame:
    P = pd.read_parquet(bundle.PRG)
    P["date"] = pd.to_datetime(P["date"])
    if P["date"].max() < asof:                                  # sessions after the bundle: compute them
        try:
            newp = bundle.pragyam_rows(sorted(symbols), P["date"].max() + pd.Timedelta(days=1), until=asof)
            P = pd.concat([P, newp], ignore_index=True)
        except Exception:
            pass
    return P[P["date"] <= asof]


def _trade_return(p: LG.Position, cost: float) -> float:
    r = 1.0
    for l in p.legs:
        if l.exit_px is None or not np.isfinite(l.exit_px) or not np.isfinite(l.entry_px):
            return np.nan
        r *= 1 + p.side * (l.exit_px / (l.entry_px * l.adj) - 1) - cost
    return r - 1


def run(asof: date | None = None, progress=None) -> Run:
    notes = []
    say = progress or (lambda pct, label: None)
    say(5, "fetching nse files")
    days = _sessions_up_to(asof)
    if not days:
        raise RuntimeError("No NSE F&O file found for that date or the 30 days before it.")
    A = pd.Timestamp(days[-1])
    say(25, "scoring positioning")
    sc = signal.scores(signal.extend_inputs(days))
    sc = sc[sc["date"] <= A]
    daily: dict = {}
    signal.replay(sc, daily=daily)
    say(45, "replaying the book")
    F = _bundle_futures(A)
    M = LG.Market(F)
    cal = set(M.cal)
    dates = [d for d in sorted(daily) if d in cal]
    names = sorted(set(sc["symbol"]))
    P = _bundle_pragyam(names, A)
    ratios = {}
    for s, g in P.groupby("symbol"):
        g = g.set_index("date")
        ratios[(s, 1)], ratios[(s, -1)] = g["stopL_r"], g["stopS_r"]
    say(55, "checking the F&O ban list")
    try:
        ban = banlist.live(dates)
        nb_date, nb = banlist.next_session(A) if asof is None or pd.Timestamp(asof) >= A else (None, None)
    except Exception:
        ban, nb_date, nb = banlist.banned(), None, None
    led = LG.replay(M, dates, daily, ratios, fno_ban=ban)
    if nb:
        for s_, (act, why) in list(led.decisions.items()):
            if act == "ENTER" and s_ in nb:
                led.decisions[s_] = ("ENTER", f"in the F&O ban on {nb_date:%d %b}: no fresh position until it lifts")
    ext = {d: (g.set_index("symbol")["pct"] - 0.5).abs().to_dict() for d, g in sc.groupby("date")}

    say(70, "reading option flow")
    # ── per-stock screen ──
    last = sc[sc["date"] == A].set_index("symbol")
    earlier = sorted(sc.loc[sc["date"] < A, "date"].unique())
    prev5 = sc[sc["date"] == earlier[-5]].set_index("symbol")["pct"] if len(earlier) >= 5 else pd.Series(dtype=float)
    plast = P[P["date"] == P["date"].max()].set_index("symbol")
    from .features import day_features
    from . import flowscan as FS
    feats = day_features(A.date()).set_index("symbol")
    try:
        prev_day = [d for d in bhavcopy.available_days() if d < A.date()][-1]
        cf = FS.contract_flow(FS._load(A.date(), False), FS._load(prev_day, False))
        uf = FS.underlying_flow(cf).set_index("symbol")
    except Exception as e:
        cf, uf = pd.DataFrame(), pd.DataFrame()
        notes.append(f"Option flow unavailable: {e}")
    fc = session.futures_context(days).set_index("symbol")
    dl = session.delivery(A.date()).set_index("symbol")
    try:
        sect = pd.read_parquet(bundle.SECT).set_index("symbol")
    except Exception:
        sect = pd.DataFrame(columns=["name", "industry"])

    def g(row, key):
        try:
            v = row.get(key, np.nan)
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    rows = []
    exiting = {p.symbol for p, _, d in led.closed if d == A}
    for s in sorted(set(last.index) | set(led.held) | exiting):
        pct = g(last.loc[s], "pct") if s in last.index else np.nan
        d5 = pct - float(prev5[s]) if s in prev5.index and np.isfinite(pct) else np.nan
        dec = led.decisions.get(s, (None, ""))
        p = led.held.get(s)
        if dec[0] == "ENTER":
            state = "▲ NEW" if p.side > 0 else "▼ NEW"
        elif dec[0] == "EXIT":
            state = "EXIT"
        elif p is not None:
            state = "LONG" if p.side > 0 else "SHORT"
        elif s in led.banned:
            state = "STOPPED"
        elif np.isfinite(pct) and 1 - signal.STAY < pct <= 1 - signal.ENTER and np.isfinite(d5) and d5 > 0:
            state = "READY ▲"
        elif np.isfinite(pct) and signal.ENTER < pct <= signal.STAY and np.isfinite(d5) and d5 < 0:
            state = "READY ▼"
        else:
            state = "—"
        fe = feats.loc[s] if s in feats.index else pd.Series(dtype=float)
        u = uf.loc[s] if len(uf) and s in uf.index else pd.Series(dtype=float)
        c = fc.loc[s] if s in fc.index else pd.Series(dtype=float)
        pr = plast.loc[s] if s in plast.index else pd.Series(dtype=float)
        dv = dl.loc[s] if s in dl.index else pd.Series(dtype=float)
        fwd = g(fe, "fwd")
        reg = g(pr, "reg")
        rows.append(dict(
            symbol=s, name=sect["name"].get(s, "") if len(sect) else "", industry=sect["industry"].get(s, "—") if len(sect) else "—",
            state=state, why=dec[1], score=pct, score_chg5=d5, opi_today=g(last.loc[s], "opi_pct") if s in last.index else np.nan,
            to_entry=(1 - signal.ENTER - pct) if np.isfinite(pct) and pct >= 0.5 else ((pct - signal.ENTER) if np.isfinite(pct) else np.nan),
            spot=g(fe, "spot"), fut=g(fe, "fut_close"), atm_iv=g(fe, "atm_iv"), pcr_vol=g(fe, "pcr_vol"), pcr_oi=g(fe, "pcr_oi"),
            skew=g(fe, "skew"), cskew=g(fe, "cskew"), cpiv=g(fe, "cpiv"), term=g(fe, "term"),
            event=bool(np.isfinite(g(fe, "term")) and g(fe, "term") > 1.08),
            wall_put=fwd * np.exp(g(fe, "wall_put_d")), wall_call=fwd * np.exp(g(fe, "wall_call_d")),
            max_pain=fwd * np.exp(g(fe, "max_pain_d")), basis=g(fe, "basis"),
            fut_ret1=g(c, "fut_ret1"), fut_oi1=g(c, "fut_oi1"), fut_ret5=g(c, "fut_ret5"), fut_oi5=g(c, "fut_oi5"),
            build1=c.get("build1", None), build5=c.get("build5", None),
            buyer_tilt=g(u, "buyer_tilt"), writer_tilt=g(u, "writer_tilt"), unusual_tilt=g(u, "unusual_tilt"), surge_tilt=g(u, "surge_tilt"),
            deliv_pct=g(dv, "deliv_pct"), risk_regime={0: "calm", 1: "normal", 2: "stressed"}.get(int(reg)) if np.isfinite(reg) else None,
            trend=g(pr, "trend"), range_lo_r=g(pr, "lo_r"), range_hi_r=g(pr, "up_r")))
    screen = pd.DataFrame(rows)
    screen["o"] = screen["state"].map(STATE_ORDER)
    screen = screen.sort_values(["o", "score"], ascending=[True, False]).drop(columns="o").reset_index(drop=True)

    say(90, "assembling the screen")
    positions, legs = _positions(M, led, A, screen)
    cost = futures_cost()
    trades = []
    for p, why, d in led.closed:
        if d >= A or not p.legs or p.legs[-1].exit_date is None:
            continue
        trades.append(dict(symbol=p.symbol, side="LONG" if p.side > 0 else "SHORT", signal=p.signal.date(),
                           entry=p.legs[0].entry_date.date(), entry_px=p.legs[0].entry_px,
                           entry_contract=p.legs[0].contract.date(),
                           exit=p.legs[-1].exit_date.date(), exit_px=p.legs[-1].exit_px,
                           exit_contract=p.legs[-1].contract.date(), legs=len(p.legs), why=why,
                           sessions=int(((M.cal >= p.legs[0].entry_date) & (M.cal <= p.legs[-1].exit_date)).sum()),
                           ret=_trade_return(p, cost)))
    return Run(A, screen, positions, legs, pd.DataFrame(trades), cf, sc, M, daily, ext, ratios, dates, notes, ban, (nb_date, nb))


def _roll_plan(M, A, s, contract) -> tuple:
    """(roll date, what happens): the close at which this contract is rolled (3 sessions before
    expiry), known in advance from the calendar, and whether a later contract is listed today."""
    if contract is None:
        return None, ""
    left = M.cal_pos.get(pd.Timestamp(contract), None)
    if left is None:
        return None, ""
    left -= M.cal_pos.get(A, 0)
    k = left - M.roll_before
    if 0 <= k and M.cal_pos[A] + k < len(M.cal):
        when = M.cal[M.cal_pos[A] + k]
    else:
        from .nse_calendar import holidays, session_offset
        when = session_offset(A, max(k, 0), holidays(M.contracts_on.get((A, s), []))) if k > 0 else A
    later = sorted(e for e in M.contracts_on.get((A, s), []) if e > pd.Timestamp(contract))
    what = f"roll to FUT {later[0]:%d %b %y}" if later else "close: no later contract listed (leaving F&O?)"
    return when, what


def _positions(M, led, A, screen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scr = screen.set_index("symbol")
    rows, legrows = [], []
    items = [(s, p, "ENTER" if led.decisions.get(s, ("",))[0] == "ENTER" else "HOLD", "") for s, p in led.held.items()]
    items += [(p.symbol, p, "EXIT", why) for p, why, d in led.closed if d == A]
    for s, p, action, why in items:
        first_contract = first_px = first_adj = leg_since = leg_px = None
        if action == "ENTER":
            contract = M.fill_contract.get((A, s), M.active.get((A, s)))
            last = M.px(A, s, contract) if contract is not None else np.nan
            lot = int(M.lot.get((A, s, contract), 0) or 0) if contract is not None else 0
            stop = last * p.k if (p.k and np.isfinite(last)) else np.nan     # from today's close
            entry_px, entry_date, realised, open_pnl, total, pnl_pct = np.nan, None, np.nan, np.nan, np.nan, np.nan
            first_contract = contract
        else:
            leg = p.leg
            contract, lot = leg.contract, leg.lot
            last = M.px(A, s, leg.contract)
            stop = p.stop if p.stop is not None else np.nan
            entry_px, entry_date = p.legs[0].entry_px * p.legs[0].adj, p.legs[0].entry_date
            # what was actually traded: the first fill's contract and exchange price (before any
            # corporate-action factor), and the current leg's contract, start and price
            first_contract, first_px, first_adj = p.legs[0].contract, p.legs[0].entry_px, p.legs[0].adj
            leg_since, leg_px = leg.entry_date, leg.entry_px
            realised = p.realised()
            open_pnl = p.side * (last - leg.entry_px * leg.adj) * lot if np.isfinite(last) else np.nan
            total = realised + (open_pnl if np.isfinite(open_pnl) else 0.0)
            # compounded return across legs, the backtest's own measure (before charges)
            g = 1.0
            for i, l in enumerate(p.legs):
                xp = l.exit_px if l.exit_px is not None else last
                if xp is None or not np.isfinite(xp):
                    g = np.nan
                    break
                g *= 1 + p.side * (xp / (l.entry_px * l.adj) - 1)
            pnl_pct = g - 1
            for i, l in enumerate(p.legs):
                xp = l.exit_px if l.exit_px is not None else None
                legrows.append(dict(symbol=s, side="LONG" if p.side > 0 else "SHORT", leg=i + 1, contract=l.contract.date(),
                                    entry_date=l.entry_date.date(), entry_px=l.entry_px * l.adj, fill_px=l.entry_px,
                                    adj=l.adj, lot=l.lot,
                                    exit_date=l.exit_date.date() if l.exit_date is not None else None, exit_px=xp,
                                    status=l.exit_why or ("open" if i == len(p.legs) - 1 else ""),
                                    pnl=p.side * ((xp if xp is not None else last) - l.entry_px * l.adj) * l.lot,
                                    adjusted=l.adj != 1.0))
        sr = scr.loc[s] if s in scr.index else pd.Series(dtype=float)
        risk = abs(last - stop) * lot if (np.isfinite(stop) and np.isfinite(last)) else np.nan
        roll_on, roll_what = _roll_plan(M, A, s, contract) if action != "EXIT" else (None, "")
        rows.append(dict(
            action=action, symbol=s, side="LONG" if p.side > 0 else "SHORT", signal=p.signal.date(),
            contract=pd.Timestamp(contract).date() if contract is not None else None, lot=lot, legs=max(len(p.legs), 1),
            entry_date=entry_date.date() if entry_date is not None else None, entry_px=entry_px, last=last,
            realised=realised, open_pnl=open_pnl, pnl=total, pnl_pct=pnl_pct,
            stop=stop, stop_from_last=(stop / last - 1) if np.isfinite(stop) and np.isfinite(last) and last else np.nan,
            risk_to_stop=risk, range_lo=last * sr.get("range_lo_r", np.nan), range_hi=last * sr.get("range_hi_r", np.nan),
            notional=last * lot if np.isfinite(last) else np.nan, why=why or (sr.get("why") or ""),
            first_contract=pd.Timestamp(first_contract).date() if first_contract is not None else None,
            first_px=first_px, first_adj=first_adj,
            leg_since=leg_since.date() if leg_since is not None else None, leg_px=leg_px,
            roll_on=roll_on.date() if roll_on is not None else None, roll_what=roll_what,
            industry=sr.get("industry", "—"), score=sr.get("score", np.nan)))
    pos = pd.DataFrame(rows)
    if len(pos):
        pos["o"] = pos["action"].map({"EXIT": 0, "ENTER": 1, "HOLD": 2})
        pos = pos.sort_values(["o", "side", "symbol"]).drop(columns="o").reset_index(drop=True)
    return pos, pd.DataFrame(legrows)


def sized(r: Run, capital: float, leverage: float = 1.0, universe: frozenset | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The model book cut to your capital (and universe): the same ledger rules,
    capital-limited. The signal stays cross-sectional over every F&O stock; the universe
    only limits which of those names you trade."""
    A = r.asof
    daily = r.daily if universe is None else {d: {s: v for s, v in m.items() if s in universe} for d, m in r.daily.items()}
    notional = {}
    for s in set(r.screen["symbol"]):
        e = r.market.active.get((A, s))
        if e is not None:
            c, lot = r.market.px(A, s, e), r.market.lot.get((A, s, e), 0)
            if np.isfinite(c) and lot:
                notional[s] = c * lot
    led = LG.replay(r.market, r.dates, daily, r.ratios, ext=r.ext, notional=notional, cap=capital * leverage,
                    fno_ban=getattr(r, "ban", None))
    return _positions(r.market, led, A, r.screen)
