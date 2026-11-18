"""Volume- and event-clock complements to the 5 s fill label and the h-second markouts.

Clocks (per instrument-day, at packet ends):
  time   - ts_recv in ns (reproduces the existing labels; used as a check);
  volume - cumulative traded contracts (all T rows);
  event  - cumulative book messages (non-snapshot A, C, M, T, F rows).
Calibration: horizon H_h = h seconds x the clock's average rate in NORMAL hours on training days,
so in normal hours every clock runs on average at wall-clock speed, and during the fix (more
trading and messaging per second) the volume and event horizons cover less wall time.

Label (same exit rule as the 5 s label): the first F, C or M of the order id after the add
decides; label 1 iff it is an F at the order's own price and fewer than H_5 units elapsed
between the end of the entry packet and the start of the fill packet. Rows whose horizon runs
past the end of the file, or that meet a book reset first, are dropped.
Markout at h: mid at the last packet whose clock is within H_h units of the fill packet's end.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .build_labels import load_day, packets, day_paths
from .config import CACHE, HORIZONS, L3, NS, OUT, PRIMARY_H, UNIVERSE, WMR
from .build_labels import local_parts, period_of_clock, tick_size

CLOCKS = ["time", "volume", "event"]
CCACHE = OUT / "cache_clocks"
MSG = ["A", "C", "M", "T", "F"]


def cum_clocks(df: pd.DataFrame, pk: dict) -> dict[str, np.ndarray]:
    act = df.action.to_numpy()
    ends = np.flatnonzero(df.is_last.to_numpy(bool))
    vol = np.where(act == "T", df["size"].to_numpy(np.int64), 0)
    msg = (~df.is_snapshot.to_numpy(bool) & np.isin(act, MSG)).astype(np.int64)
    return {"time": pk["end_ts"], "volume": np.cumsum(vol)[ends], "event": np.cumsum(msg)[ends]}


def rates(paths: list, train_days: set) -> dict[str, float]:
    """Units per second in normal hours (universe minus WMR and nearby), training days only."""
    normal_s = UNIVERSE[1] - UNIVERSE[0] - 900
    tot = {"volume": 0.0, "event": 0.0}
    n = 0
    for p in paths:
        df = pq.read_table(p, columns=["ts_recv", "action", "size", "is_snapshot", "is_last"]).to_pandas()
        day, _, clock, _ = local_parts(df.ts_recv.astype("int64").to_numpy())
        keep = (clock >= UNIVERSE[0]) & (clock < UNIVERSE[1]) & (period_of_clock(clock) == "normal")
        if not keep.any() or str(pd.Series(day[keep]).mode().iloc[0]) not in train_days:
            continue
        act = df.action.to_numpy()
        tot["volume"] += df["size"].to_numpy(np.int64)[keep & (act == "T")].sum()
        tot["event"] += (keep & ~df.is_snapshot.to_numpy(bool) & np.isin(act, MSG)).sum()
        n += 1
    return {k: v / (n * normal_s) for k, v in tot.items()} | {"n_days": n}


def day_labels(path, tick: float, H: dict) -> pd.DataFrame:
    x = pd.read_parquet(CACHE / f"{path.stem}_obs.parquet",
                        columns=["order_id", "entry_ns", "add_row", "side", "price", "period", "clock"])
    df = load_day(path)
    pk = packets(df, tick)
    act, oid = df.action.to_numpy(), df.order_id.to_numpy(np.uint64).astype(np.int64)
    px, size = df.price.to_numpy(), df["size"].to_numpy(np.float64)
    cums = cum_clocks(df, pk)
    resets = np.r_[0, np.cumsum(pk["is_reset"])]
    n = len(x)
    add_row = x.add_row.to_numpy()
    p_e = pk["pid"][add_row]
    s = np.where(x.side == "B", 1.0, -1.0)

    ev_rows = np.flatnonzero(np.isin(act, ["F", "C", "M"]))
    m = pd.DataFrame({"k": np.arange(n), "oid": x.order_id.to_numpy(np.int64), "add_row": add_row}) \
        .merge(pd.DataFrame({"oid": oid[ev_rows], "row": ev_rows}), on="oid")
    m = m[m.row > m.add_row]
    first = m.groupby("k").row.min()
    exit_row = np.full(n, -1)
    exit_row[first.index] = first.to_numpy()
    has = exit_row >= 0
    xr = np.maximum(exit_row, 0)
    p_x = pk["pid"][xr]
    own_fill = has & (act[xr] == "F") & (px[xr] == x.price.to_numpy())
    mr, mk = m.row.to_numpy(), m.k.to_numpy()
    fm = m[(act[mr] == "F") & (px[mr] == x.price.to_numpy()[mk])]

    out = {"order_id": x.order_id.to_numpy(), "entry_ns": x.entry_ns.to_numpy()}
    for k, cum in cums.items():
        start = cum if k == "time" else np.r_[0, cum[:-1]]          # clock at the start of a packet
        h = {hh: (hh * NS if k == "time" else H[k] * hh) for hh in [*HORIZONS, PRIMARY_H]}
        E = cum[p_e]

        def wall(c0, t0, units, nominal_s):
            """Seconds until the clock has advanced `units` from c0 (time clock: nominal seconds)."""
            if k == "time":
                return np.full(len(c0), float(nominal_s))
            reach = np.minimum(np.searchsorted(cum, c0 + units, "left"), len(cum) - 1)
            return np.maximum(pk["end_ts"][reach] - t0, 0) / NS
        dS = start[p_x] - E
        exit_in = has & (dS <= h[PRIMARY_H])
        label = own_fill & exit_in
        p_h = np.searchsorted(start, E + h[PRIMARY_H], "right") - 1
        p_end = np.where(exit_in, p_x, p_h)
        censored = ~exit_in & (cum[-1] - E < h[PRIMARY_H])
        reset = resets[p_end + 1] - resets[p_e + 1] > 0
        r = {"label": label.astype(int), "drop": censored | reset,
             "wall_label_s": wall(E, x.entry_ns.to_numpy(), h[PRIMARY_H], PRIMARY_H)}
        tau = pk["end_ts"][p_x]
        r["fill_time_s"] = np.where(label, (tau - x.entry_ns.to_numpy()) / NS, np.nan)
        q = fm[(start[pk["pid"][fm.row.to_numpy()]] - E[fm.k.to_numpy()]) <= h[PRIMARY_H]]
        qty = pd.Series(size[q.row.to_numpy()]).groupby(q.k.to_numpy()).sum().reindex(range(n)).to_numpy()
        r["fill_qty_5s"] = np.where(label, qty, np.nan)
        mid_pre = pk["mid"][np.maximum(p_x - 1, 0)]
        r["spread_capture"] = np.where(label, s * (mid_pre - x.price.to_numpy()) / tick, np.nan)
        c_f = cum[p_x]
        for hh in HORIZONS:
            idx = np.searchsorted(cum, c_f + h[hh], "right") - 1
            mid = pk["mid"][idx]
            r[f"mk_{hh}"] = np.where(label, s * (mid - x.price.to_numpy()) / tick, np.nan)
            r[f"move_{hh}"] = np.where(label, s * (mid - mid_pre) / tick, np.nan)
            r[f"wall_mk_{hh}"] = np.where(label, wall(c_f, tau, h[hh], hh), np.nan)
        idx30 = np.searchsorted(cum, c_f + h[max(HORIZONS)], "right") - 1
        r["follow_up_ok"] = label & (c_f + h[max(HORIZONS)] <= cum[-1]) & (resets[idx30 + 1] - resets[p_x + 1] == 0)
        end_s = np.where(label, r["fill_time_s"] + r[f"wall_mk_{PRIMARY_H}"], r["wall_label_s"])
        r["crosses_boundary"] = (x.period.to_numpy() == "wmr") & (x.clock.to_numpy() + end_s > WMR[1])
        out.update({f"{c}__{k}": v for c, v in r.items()})
    return pd.DataFrame(out)


def obs_paths() -> list:
    """Days with a non-empty observation cache (Sundays have none)."""
    return [p for p in day_paths() if pq.ParquetFile(CACHE / f"{p.stem}_obs.parquet").metadata.num_rows]


def build() -> dict:
    CCACHE.mkdir(parents=True, exist_ok=True)
    paths = obs_paths()
    tick = tick_size(L3.parents[1])
    f = CCACHE / "rates.json"
    if not f.exists():
        days = sorted(p.stem[-10:] for p in paths)             # London date = UTC file date for 08-17
        train = set(days[:round(0.6 * len(days))])
        f.write_text(json.dumps(rates(paths, train), indent=1))
    r = json.loads(f.read_text())
    H = {"volume": r["volume"], "event": r["event"]}
    for p in paths:
        c = CCACHE / f"{p.stem}_clocks.parquet"
        if not c.exists():
            print("clocks", p.stem, flush=True)
            day_labels(p, tick, H).to_parquet(c, index=False)
    return r


CLOCK_COLS = ["label", "fill_time_s", "fill_qty_5s", "spread_capture", "follow_up_ok", "crosses_boundary",
              *[f"{c}_{h}" for h in HORIZONS for c in ("mk", "move")]]


def load_clock_obs(k: str) -> pd.DataFrame:
    """Observation table with the label/markout columns of clock k (same filters as load_observations)."""
    parts = []
    for p in obs_paths():
        o = pd.read_parquet(CACHE / f"{p.stem}_obs.parquet")
        c = pd.read_parquet(CCACHE / f"{p.stem}_clocks.parquet")
        assert (o.entry_ns.to_numpy() == c.entry_ns.to_numpy()).all() and (o.order_id.to_numpy() == c.order_id.to_numpy()).all()
        for col in CLOCK_COLS:
            o[col] = c[f"{col}__{k}"].to_numpy()
        o["wall_label_s"] = c[f"wall_label_s__{k}"].to_numpy()
        o[f"wall_mk_{PRIMARY_H}"] = c[f"wall_mk_{PRIMARY_H}__{k}"].to_numpy()
        parts.append(o[~c[f"drop__{k}"].to_numpy()])
    obs = pd.concat(parts, ignore_index=True)
    filled = obs.label == 1
    obs = obs[obs.valid_state & (~filled | (obs.follow_up_ok & obs.mk_30.notna()))].reset_index(drop=True)
    for h in HORIZONS:
        obs[f"Z_{h}"] = np.where(obs.label == 1, obs[f"mk_{h}"], 0.0)
    return obs
