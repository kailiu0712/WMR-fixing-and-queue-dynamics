"""Aggressive trade events with pre-trade state and signed price responses (one cache file per day).

Event = one exchange packet (rows up to is_last; all rows share ts_event) containing >= 1
non-snapshot T row. The packet is the finest unit with an observable book state before and
after, so T rows of the same packet (several price levels, occasionally several aggressor
order ids) form one event; T, F and the follow-up C/M rows are never counted separately.
  size  = sum of T sizes (the executed quantity; same units as the verified volume clock);
  q     = +1 if every T row of the packet is B (aggressive buy), -1 if every T row is A, else 0
          (ambiguous: any N-side T row, or both sides) -> excluded from the signed analysis.
Pre-trade state = end of the previous packet, computed by wmr_value.build_labels.entry_features
(validated, unit-tested) with the CONSUMED side as the order side: same_depth = depth being
consumed, hit_flow_h = recent aggressive volume in the trade's direction.
Responses, in ticks, positive = price moved in the trade's direction:
  I0      = q (m+ - m-), m- = mid at the previous packet end, m+ = first valid mid at or after
            the event packet end;
  I_v{v}  = q (m_v - m-), m_v = mid at the last packet whose cumulative traded volume is
            <= volume at the event packet end + v (volume clock; same convention as clocks.py);
  I_e{v}  = same on the event clock, v * EVENT_PER_CONTRACT messages;
  I_w{h}  = same h seconds after the event packet end (wall clock).
A horizon is NaN when it runs past the end of the file, meets a book reset, or takes longer
than MAX_SPAN_S of wall time (data gap). Maker fills of the Phase 3 sample are linked to the
packet of their first own-price F row by order id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wmr_value.build_labels import entry_features, load_day, local_parts, packets, period_of_clock, tick_size
from wmr_value.clocks import cum_clocks, obs_paths
from wmr_value.config import CACHE as OBS_CACHE, L3

from .config import CACHE, EVENT_PER_CONTRACT, MAX_SPAN_S, NS, UNIVERSE, VOL_H, WALL_H


def trade_events(df: pd.DataFrame, pk: dict) -> pd.DataFrame:
    """One row per packet with a T row: size, side, structure counts."""
    act, sd = df.action.to_numpy(), df.side.to_numpy()
    live = ~df.is_snapshot.to_numpy(bool)
    t = np.flatnonzero(live & (act == "T"))
    oid = df.order_id.to_numpy(np.uint64).astype(np.int64)
    g = pd.DataFrame({"pid": pk["pid"][t], "size": df["size"].to_numpy(np.int64)[t], "buy": sd[t] == "B",
                      "sell": sd[t] == "A", "px": df.price.to_numpy()[t],
                      "aid": np.where(oid[t] > 0, oid[t], np.nan)}).groupby("pid")
    ev = g.agg(size=("size", "sum"), n_t=("size", "size"), buy=("buy", "all"), sell=("sell", "all"),
               n_px=("px", "nunique"), n_aggr_id=("aid", "nunique"))
    ev["q"] = np.where(ev.buy, 1, np.where(ev.sell, -1, 0))
    # passive executed quantity: F rows that are not the aggressor's own F (repriced resting order)
    f = np.flatnonzero(live & (act == "F"))
    t_keys = pd.MultiIndex.from_arrays([pk["pid"][t], oid[t]])
    own = pd.MultiIndex.from_arrays([pk["pid"][f], oid[f]]).isin(t_keys)
    fq = pd.Series(df["size"].to_numpy(np.int64)[f][~own]).groupby(pk["pid"][f][~own]).sum()
    ev["passive_qty"] = fq.reindex(ev.index).fillna(0).astype(int).to_numpy()
    ev["reset_in_packet"] = pk["is_reset"][ev.index.to_numpy()] > 0
    return ev.drop(columns=["buy", "sell"]).reset_index()


def raw_mid(df: pd.DataFrame, pk: dict) -> np.ndarray:
    """Mid at each packet end, NaN when the book is not two-sided."""
    ends = np.flatnonzero(df.is_last.to_numpy(bool))
    bid, ask = df.bid_px.to_numpy()[ends], df.ask_px.to_numpy()[ends]
    ok = np.isfinite(bid) & np.isfinite(ask) & (bid < ask)
    return np.where(ok, (bid + ask) / 2, np.nan)


def responses(pk: dict, cums: dict, mid_raw: np.ndarray, p: np.ndarray, q: np.ndarray, tick: float) -> dict:
    mid = pk["mid"]                                       # forward-filled valid mid
    nxt = pd.Series(mid_raw).bfill().to_numpy()           # first valid mid at or after a packet
    resets = np.r_[0, np.cumsum(pk["is_reset"])]
    end_ts = pk["end_ts"]
    m_pre = mid_raw[np.maximum(p - 1, 0)]
    out = {"m_pre": m_pre, "post_mid_delayed": ~np.isfinite(mid_raw[p]),
           "I0": q * (nxt[p] - m_pre) / tick}

    def at(cum, units, name):
        target = cum[p] + units
        idx = np.searchsorted(cum, target, "right") - 1             # last packet within the horizon
        closes = np.minimum(idx + 1, len(cum) - 1)                    # packet that ends the horizon
        span = (end_ts[closes] - end_ts[p]) / NS
        ok = (cum[-1] > target) & (resets[idx + 1] - resets[p + 1] == 0) & (span <= MAX_SPAN_S)
        out[name] = np.where(ok, q * (mid[idx] - m_pre) / tick, np.nan)
        return span

    for v in VOL_H:
        out[f"span_v{v}"] = at(cums["volume"], v, f"I_v{v}")
        at(cums["event"], v * EVENT_PER_CONTRACT, f"I_e{v}")
    for h in WALL_H:
        at(cums["time"], h * NS, f"I_w{h}")
    return out


def maker_links(stem: str, df: pd.DataFrame, pk: dict) -> pd.DataFrame:
    """Phase 3 maker fills (same filters as load_observations) -> packet of their first own-price F."""
    o = pd.read_parquet(OBS_CACHE / f"{stem}_obs.parquet")
    o = o[(o.label == 1) & o.valid_state & o.follow_up_ok & o.mk_30.notna()]
    act, oid = df.action.to_numpy(), df.order_id.to_numpy(np.uint64).astype(np.int64)
    fr = np.flatnonzero(act == "F")
    m = pd.DataFrame({"k": np.arange(len(o)), "oid": o.order_id.to_numpy(np.int64), "add_row": o.add_row.to_numpy(),
                      "price": o.price.to_numpy()}).merge(pd.DataFrame({"oid": oid[fr], "row": fr}), on="oid")
    m = m[(m.row > m.add_row) & (df.price.to_numpy()[m.row] == m.price)]
    first = m.groupby("k").row.min()
    o = o.iloc[first.index.to_numpy()]
    return pd.DataFrame({"order_id": o.order_id.to_numpy(), "pid": pk["pid"][first.to_numpy()],
                         "s": np.where(o.side.to_numpy() == "B", 1, -1), "period_obs": o.period.to_numpy(),
                         "spread_capture": o.spread_capture.to_numpy(), "mk_5_obs": o.mk_5.to_numpy()})


def event_features(df: pd.DataFrame, pk: dict, ev: pd.DataFrame, tick: float) -> pd.DataFrame:
    """Pre-trade state for each event; only rows before the event packet enter (via entry_features)."""
    first_row = pk["starts"][ev.pid.to_numpy()]
    consumed = np.where(ev.q.to_numpy() < 0, "B", "A")    # ambiguous events get "A"; they are excluded later
    f = entry_features(df, first_row, consumed, tick)
    sgnlog = lambda x: np.sign(x) * np.log1p(np.abs(x))
    return pd.DataFrame({
        "log_size": np.log(ev["size"].to_numpy(float)),
        "depth_consumed": f.same_depth, "depth_opposite": f.opp_depth,
        "log_depth_consumed": np.log1p(f.same_depth), "log_depth_opposite": np.log1p(f.opp_depth),
        **{f"flow_{h}s": sgnlog(f[f"hit_flow_{h}s"]) for h in (1, 5, 30)},
        "log_volume_30s": np.log1p(f.volume_30s), "log_cancels_5s": np.log1p(f.cancels_5s),
        "log_vol_30s": np.log1p(f.vol_30s_ticks), "imbalance": f.imbalance, "spread_ticks": f.spread_ticks,
        "valid_state": f.valid_state.to_numpy()})


def build_day(path, tick: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_day(path)
    pk = packets(df, tick)
    ev = trade_events(df, pk)
    day, _, clock, weekday = local_parts(pk["end_ts"][ev.pid.to_numpy()])
    keep = (clock >= UNIVERSE[0]) & (clock < UNIVERSE[1]) & weekday
    ev = ev[keep].reset_index(drop=True)
    ev["day"], ev["clock"] = day[keep], clock[keep]
    ev["period"] = period_of_clock(ev.clock.to_numpy())
    ev["ts"] = pk["end_ts"][ev.pid.to_numpy()]
    ev = pd.concat([ev, event_features(df, pk, ev, tick)], axis=1)
    r = responses(pk, cum_clocks(df, pk), raw_mid(df, pk), ev.pid.to_numpy(), ev.q.to_numpy(float), tick)
    ev = pd.concat([ev, pd.DataFrame(r)], axis=1)
    links = maker_links(path.stem, df, pk)
    links = links[links.pid.isin(ev.pid)]
    links["day"] = ev.day.iloc[0] if len(ev) else ""
    return ev, links


def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    tick = tick_size(L3.parents[1])
    evs, lks = [], []
    for p in obs_paths():
        fe, fl = CACHE / f"{p.stem}_events.parquet", CACHE / f"{p.stem}_links.parquet"
        if not fe.exists():
            print("events", p.stem, flush=True)
            ev, lk = build_day(p, tick)
            ev.to_parquet(fe, index=False)
            lk.to_parquet(fl, index=False)
        evs.append(pd.read_parquet(fe))
        lks.append(pd.read_parquet(fl))
    return pd.concat(evs, ignore_index=True), pd.concat(lks, ignore_index=True)


if __name__ == "__main__":
    ev, lk = load_events()
    print(len(ev), len(lk))
