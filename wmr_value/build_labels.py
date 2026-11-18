"""Observation table: one row per passive order-entry episode at the best bid/ask.

Population: actual historical participant orders. The episode definition and the 5 s fill
label come unchanged from wmr_eda.py (its cached candidates):
  - entry = a new add (A) at the best price after its packet; at most one entry per
    (second, side) to thin overlapping episodes;
  - label 1 = any fill (F) of that order id within 5 s before it exits;
  - exit with label 0 on cancel, on any modify (M, voluntary resize or reprice), or at 5 s;
  - book resets are excluded here.
Everything else is recomputed here from the L3 events:
  - features use only events in packets strictly BEFORE the entry packet, plus the add row
    itself (own size, own queue-ahead at insertion). The cached EDA features use the
    post-packet book and are not used;
  - markouts are anchored at the first fill packet. A resting order fills at its own limit
    price, so the quantity-weighted fill price equals the entry price (checked in tests).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import CACHE, EDA_CACHE, HORIZONS, L3, LABEL_S, NS, PRIMARY_H, UNIVERSE, WMR

sys.path.insert(0, str(L3.parents[1]))
from wmr_eda import local_parts, period_of_clock, tick_size  # noqa: E402

COLS = ["ts_recv", "action", "side", "price", "size", "order_id", "is_last", "is_snapshot",
        "qty_ahead", "orders_ahead", "bid_px", "bid_qty", "ask_px", "ask_qty"]
FEATURES_RAW = ["queue_ahead", "orders_ahead", "size", "same_depth", "opp_depth", "spread_ticks",
                "imbalance", "hit_flow_1s", "hit_flow_5s", "hit_flow_30s", "volume_30s", "adds_5s",
                "cancels_5s", "vol_30s_ticks", "time_since_trade_s", "clock"]


def load_day(path) -> pd.DataFrame:
    df = pq.read_table(path, columns=COLS).to_pandas()
    last = df.is_last.to_numpy(bool)
    if len(df) and not last[-1]:                       # drop a trailing incomplete packet
        df = df.iloc[: np.flatnonzero(last)[-1] + 1]
    return df.reset_index(drop=True)


def packets(df: pd.DataFrame, tick: float) -> dict:
    """Packet ids, packet starts/end times, and forward-filled valid mid at packet ends."""
    last = df.is_last.to_numpy(bool)
    pid = np.r_[0, np.cumsum(last)[:-1]]
    ends = np.flatnonzero(last)
    starts = np.r_[0, ends[:-1] + 1]
    end_ts = df.ts_recv.astype("int64").to_numpy()[ends]
    bid, ask = df.bid_px.to_numpy()[ends], df.ask_px.to_numpy()[ends]
    mid = pd.Series(np.where(np.isfinite(bid) & np.isfinite(ask) & (bid < ask), (bid + ask) / 2, np.nan))
    return {"pid": pid, "starts": starts, "end_ts": end_ts, "row_ts": end_ts[pid],
            "mid": mid.ffill().to_numpy(), "is_reset": np.bincount(pid, df.action.to_numpy() == "R")}


def entry_features(df: pd.DataFrame, add_row: np.ndarray, side: np.ndarray, tick: float) -> pd.DataFrame:
    """Pre-entry features. Uses rows before the entry packet's first row, plus the add row."""
    pk = packets(df, tick)
    act, sd = df.action.to_numpy(), df.side.to_numpy()
    size = df["size"].to_numpy(np.float64)
    s = np.where(side == "B", 1.0, -1.0)
    ps = pk["starts"][pk["pid"][add_row]]               # first row of the entry packet
    r0 = np.maximum(ps - 1, 0)                          # last row of the previous packet
    t = pk["row_ts"][add_row]
    bid, ask = df.bid_px.to_numpy()[r0], df.ask_px.to_numpy()[r0]
    bq, aq = df.bid_qty.to_numpy(np.float64)[r0], df.ask_qty.to_numpy(np.float64)[r0]
    same, opp = np.where(s > 0, bq, aq), np.where(s > 0, aq, bq)

    oid = df.order_id.to_numpy(np.uint64)
    f_keys = pd.MultiIndex.from_arrays([pk["pid"][act == "F"], oid[act == "F"]])
    follow = np.zeros(len(df), bool)                    # C rows that only remove a filled order
    c_rows = np.flatnonzero(act == "C")
    follow[c_rows] = pd.MultiIndex.from_arrays([pk["pid"][c_rows], oid[c_rows]]).isin(f_keys)
    signed_trade = np.where(act == "T", np.where(sd == "B", size, np.where(sd == "A", -size, 0.0)), 0.0)
    snap = df.is_snapshot.to_numpy(bool)

    def past(values: np.ndarray, h: int) -> np.ndarray:
        cum = np.r_[0.0, np.cumsum(values)]
        lo = np.minimum(np.searchsorted(pk["row_ts"], t - h * NS, "left"), ps)
        return cum[ps] - cum[lo]

    out = {"queue_ahead": df.qty_ahead.to_numpy()[add_row].astype(float),
           "orders_ahead": df.orders_ahead.to_numpy()[add_row].astype(float),
           "size": size[add_row], "price": df.price.to_numpy()[add_row],
           "same_depth": same, "opp_depth": opp, "spread_ticks": (ask - bid) / tick,
           "imbalance": (same - opp) / (same + opp)}
    for h in (1, 5, 30):                                # aggressive volume that would hit this side
        out[f"hit_flow_{h}s"] = -s * past(signed_trade, h)
    out["volume_30s"] = past(np.abs(signed_trade), 30)
    out["adds_5s"] = past(np.where((act == "A") & ~snap, size, 0.0), 5)
    out["cancels_5s"] = past(np.where((act == "C") & ~follow, size, 0.0), 5)

    t_rows = np.flatnonzero(act == "T")
    k = np.searchsorted(t_rows, ps, "left") - 1
    since = np.where(k >= 0, (t - pk["row_ts"][t_rows[np.maximum(k, 0)]]) / NS, np.inf)
    out["time_since_trade_s"] = np.minimum(since, 600.0)

    prev_pkt = pk["pid"][add_row] - 1                   # realized mid volatility over 30 s, 1 s grid
    grid = np.stack([np.minimum(np.searchsorted(pk["end_ts"], t - j * NS, "right") - 1, prev_pkt)
                     for j in range(31)])
    m = np.where(grid >= 0, pk["mid"][np.maximum(grid, 0)], np.nan)
    out["vol_30s_ticks"] = np.sqrt(np.nansum(np.diff(m, axis=0) ** 2, axis=0)) / tick
    out["improves"] = s * (out["price"] - np.where(s > 0, bid, ask)) > 0
    out["valid_state"] = (ps > 0) & np.isfinite(bid) & np.isfinite(ask) & (bid < ask) & (bq > 0) & (aq > 0)
    return pd.DataFrame(out)


def outcomes(df: pd.DataFrame, add_row: np.ndarray, side: np.ndarray, label: np.ndarray,
             entry_ns: np.ndarray, tick: float) -> pd.DataFrame:
    """First-fill time/price, filled quantity and markout decomposition for label==1 rows."""
    pk = packets(df, tick)
    act = df.action.to_numpy()
    oid = df.order_id.to_numpy(np.uint64)
    n = len(add_row)
    res = pd.DataFrame({"fill_price": np.nan, "fill_side": "", "fill_time_s": np.nan, "fill_qty_5s": np.nan,
                        "mid_pre_fill": np.nan, **{f"mid_{h}": np.nan for h in HORIZONS},
                        "follow_up_ok": False}, index=range(n))
    hit = np.flatnonzero(label == 1)
    if len(hit) == 0:
        return res
    fr = np.flatnonzero(act == "F")
    fills = pd.DataFrame({"oid": oid[fr], "row": fr})
    want = pd.DataFrame({"k": hit, "oid": oid[add_row[hit]], "add_row": add_row[hit], "entry_ns": entry_ns[hit]})
    m = want.merge(fills, on="oid")
    m = m[(m.row > m.add_row) & (pk["row_ts"][m.row] <= m.entry_ns + LABEL_S * NS)]
    first = m.loc[m.groupby("k").row.idxmin()].set_index("k")
    same_px = df.price.to_numpy()[m.row] == df.price.to_numpy()[m.add_row]
    qty = pd.Series(df["size"].to_numpy(np.float64)[m.row] * same_px).groupby(m.k.to_numpy()).sum()

    k = first.index.to_numpy()
    f = first.row.to_numpy()
    p_f = pk["pid"][f]
    tau = pk["end_ts"][p_f]
    res.loc[k, "fill_price"] = df.price.to_numpy()[f]
    res.loc[k, "fill_side"] = df.side.to_numpy()[f]
    res.loc[k, "fill_time_s"] = (tau - entry_ns[k]) / NS
    res.loc[k, "fill_qty_5s"] = qty.reindex(k).to_numpy()
    res.loc[k, "mid_pre_fill"] = pk["mid"][p_f - 1]
    for h in HORIZONS:
        res.loc[k, f"mid_{h}"] = pk["mid"][np.searchsorted(pk["end_ts"], tau + h * NS, "right") - 1]
    horizon_end = np.searchsorted(pk["end_ts"], tau + max(HORIZONS) * NS, "right")
    resets = np.r_[0, np.cumsum(pk["is_reset"])]
    res.loc[k, "follow_up_ok"] = (tau + max(HORIZONS) * NS <= pk["end_ts"][-1]) & \
                                 (resets[horizon_end] - resets[p_f + 1] == 0)
    return res


def build_day(path, tick: float) -> pd.DataFrame:
    cand = pd.read_parquet(EDA_CACHE / f"{path.stem}_candidates.parquet",
                           columns=["order_id", "entry_ns", "side", "label", "exit_reason", "fill_time_s"])
    _, _, clock, _ = local_parts(cand.entry_ns.to_numpy(np.int64))
    cand = cand[(clock >= UNIVERSE[0]) & (clock < UNIVERSE[1]) & (cand.exit_reason != "book_reset")]
    if cand.empty:
        return pd.DataFrame()
    df = load_day(path)
    pk = packets(df, tick)
    a_rows = np.flatnonzero((df.action.to_numpy() == "A") & ~df.is_snapshot.to_numpy(bool))
    adds = pd.DataFrame({"order_id": df.order_id.to_numpy(np.uint64)[a_rows].astype(np.int64),
                         "entry_ns": pk["row_ts"][a_rows], "add_row": a_rows})
    adds = adds.drop_duplicates(["order_id", "entry_ns"], keep=False)
    x = cand.merge(adds, on=["order_id", "entry_ns"], how="inner").reset_index(drop=True)
    if len(x) != len(cand):
        print(f"  {path.stem}: {len(cand) - len(x)} candidates without a unique add row dropped")
    add_row, side = x.add_row.to_numpy(), x.side.to_numpy()
    feats = entry_features(df, add_row, side, tick)
    outs = outcomes(df, add_row, side, x.label.to_numpy(), x.entry_ns.to_numpy(np.int64), tick)
    out = pd.concat([x.rename(columns={"fill_time_s": "eda_fill_time_s"}), feats, outs], axis=1)
    # Correction to the EDA label: an F at a price other than the resting price is the order
    # itself repriced across the spread (it appears as the T aggressor). That is a voluntary
    # exit that takes liquidity, not a passive fill.
    aggressive = (out.label == 1) & (out.fill_price != out.price)
    out.loc[aggressive, "label"] = 0
    out.loc[aggressive, "exit_reason"] = "aggressive_reprice"
    out.loc[aggressive, list(outs.columns.drop(["fill_side", "follow_up_ok"]))] = np.nan

    day, _, clock, _ = local_parts(out.entry_ns.to_numpy(np.int64))
    out["day"], out["clock"] = day, clock
    out["period"] = period_of_clock(clock)
    out["instrument"] = path.stem[:4]
    s = np.where(out.side == "B", 1.0, -1.0)
    for h in HORIZONS:                                  # ticks; positive = good for the maker
        out[f"mk_{h}"] = s * (out[f"mid_{h}"] - out.fill_price) / tick
        out[f"move_{h}"] = s * (out[f"mid_{h}"] - out.mid_pre_fill) / tick
    out["spread_capture"] = s * (out.mid_pre_fill - out.fill_price) / tick
    end_s = np.where(out.label == 1, out.fill_time_s + PRIMARY_H, LABEL_S)
    out["crosses_boundary"] = (out.period == "wmr") & (out.clock + end_s > WMR[1])
    return out.drop(columns=["mid_pre_fill", *[f"mid_{h}" for h in HORIZONS]])


def day_paths() -> list:
    paths = []
    for day in pd.date_range("2026-07-14", "2026-10-13", freq="D"):
        sym = "6EU6" if day < pd.Timestamp("2026-09-11") else "6EZ6"
        p = L3 / f"{sym}_mbo_{day.date()}.parquet"
        if p.is_file() and pq.ParquetFile(p).metadata.num_rows and \
                (EDA_CACHE / f"{p.stem}_candidates.parquet").is_file():
            paths.append(p)
    return paths


def load_observations(limit_days: int | None = None) -> pd.DataFrame:
    """Build (or read cached) per-day observation tables and keep valid rows."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tick = tick_size(L3.parents[1])
    parts = []
    for p in day_paths()[:limit_days]:
        c = CACHE / f"{p.stem}_obs.parquet"
        if not c.exists():
            print("building", p.stem, flush=True)
            build_day(p, tick).to_parquet(c, index=False)
        parts.append(pd.read_parquet(c))
    obs = pd.concat(parts, ignore_index=True)
    filled = obs.label == 1
    keep = obs.valid_state & (~filled | (obs.follow_up_ok & obs.mk_30.notna()))
    obs = obs[keep].reset_index(drop=True)
    for h in HORIZONS:
        obs[f"Z_{h}"] = np.where(obs.label == 1, obs[f"mk_{h}"], 0.0)
    return obs
