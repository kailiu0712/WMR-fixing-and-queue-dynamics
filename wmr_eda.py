"""Exploratory WMR fixing analysis of the continuous Jul-Oct 2026 6E L3 sample.

Run from the 5257/project directory: python wmr_eda.py
Optional: --limit-days 1 --out-dir <scratch> for a small smoke run.

The script uses the existing Book class to maintain an analytical top-10 view
from processed events; it never downloads data or rebuilds the L3 parquet files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NS = 1_000_000_000
LONDON = "Europe/London"
COLS = ["ts_recv", "ts_event", "action", "side", "price", "size", "order_id",
        "is_last", "is_snapshot", "order_qty", "qty_ahead", "orders_ahead",
        "level_qty", "bid_px", "bid_qty", "ask_px", "ask_qty"]
PERIODS = ["wmr", "nearby", "normal", "same_hour_control"]
PERIOD_COLORS = {"wmr": "#b93c32", "nearby": "#e29b41",
                 "normal": "#527ea6", "same_hour_control": "#73926e"}


def period_of_clock(seconds: np.ndarray) -> np.ndarray:
    """Full five-minute WMR and adjacent five-minute windows, in London time."""
    out = np.full(len(seconds), "normal", dtype="U20")
    out[((seconds >= 15 * 3600 + 52 * 60 + 30) &
         (seconds < 15 * 3600 + 57 * 60 + 30)) |
        ((seconds >= 16 * 3600 + 2 * 60 + 30) &
         (seconds < 16 * 3600 + 7 * 60 + 30))] = "nearby"
    out[(seconds >= 15 * 3600 + 57 * 60 + 30) &
        (seconds < 16 * 3600 + 2 * 60 + 30)] = "wmr"
    return out


def local_parts(ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clock = pd.to_datetime(ns, utc=True).tz_convert(LONDON)
    date = np.asarray(clock.strftime("%Y-%m-%d"))
    minute = np.asarray(clock.hour * 60 + clock.minute, dtype=np.int16)
    seconds = np.asarray(clock.hour * 3600 + clock.minute * 60 + clock.second, dtype=np.int32)
    eligible = np.asarray(clock.day_of_week < 5)
    return date, minute, seconds, eligible


def tick_size(project: Path) -> float:
    defs = pd.read_csv(project / "data_6e" / "6e_definitions.csv")
    rows = defs.loc[defs.raw_symbol.isin(["6EU6", "6EZ6"]), "min_price_increment"]
    if len(rows) != 2 or rows.nunique() != 1:
        raise ValueError("6EU6 and 6EZ6 must have the same tick size in 6e_definitions.csv")
    return float(rows.iloc[0])


def book_snapshot(book, born: dict, when_ns: int, tick: float,
                  prev_refs: dict) -> tuple[dict | None, list[tuple]]:
    bid, bq, bn = book.best("B")
    ask, aq, an = book.best("A")
    if not (np.isfinite(bid) and np.isfinite(ask) and 0 < bid < ask and bq > 0 and aq > 0):
        return None, []
    shapes = []
    depth = {}
    l1_sizes, l1_ages = [], []
    qa_changes = []
    next_refs = {}
    for side in ("B", "A"):
        lv = book.levels[side]
        keys = list(lv.islice(len(lv) - min(10, len(lv)), len(lv))) if side == "B" else list(lv.islice(0, min(10, len(lv))))
        if side == "B":
            keys.reverse()
        quantities = [sum(lv[p].values()) for p in keys]
        counts = [len(lv[p]) for p in keys]
        shapes.extend((side, k + 1, float(quantities[k]), int(counts[k])) for k in range(len(keys)))
        depth[side] = [sum(quantities[:n]) for n in (1, 5, 10)]
        q = lv[keys[0]]
        l1_sizes.extend(q.values())
        l1_ages.extend((when_ns - born[oid]) / NS for oid in q if oid in born and when_ns >= born[oid])
        old = prev_refs.get(side)
        if old and old[1] == keys[0] and old[0] in q:
            ahead, _ = book.queue_ahead(old[0])
            if ahead >= 0:
                qa_changes.append(ahead - old[2])
        oid = next(reversed(q))
        ahead, _ = book.queue_ahead(oid)
        next_refs[side] = (oid, keys[0], ahead)
    prev_refs.clear()
    prev_refs.update(next_refs)
    spread = (ask - bid) / tick
    result = {
        "bid_px": bid, "ask_px": ask, "mid": (bid + ask) / 2,
        "spread_ticks": spread,
        "bid_qty": bq, "ask_qty": aq,
        "depth1": depth["B"][0] + depth["A"][0],
        "depth5": depth["B"][1] + depth["A"][1],
        "depth10": depth["B"][2] + depth["A"][2],
        "imbalance": (bq - aq) / (bq + aq),
        "abs_imbalance": abs((bq - aq) / (bq + aq)),
        "best_order_count": bn + an,
        "median_best_order_size": float(np.median(l1_sizes)) if l1_sizes else np.nan,
        "median_best_order_age_s": float(np.median(l1_ages)) if l1_ages else np.nan,
        "mean_queue_ahead_change": float(np.mean(qa_changes)) if qa_changes else np.nan,
    }
    return result, shapes


def process_day(path: Path, project: Path, tick: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Counter]:
    """Read one UTC day, sample completed book packets at 1 Hz, label touch adds."""
    sys.path.insert(0, str(project))
    from main import Book  # existing, tested L3 event utility

    data = pq.read_table(path, columns=COLS).to_pandas()
    diagnostics = Counter()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), diagnostics
    ts = data.ts_recv.astype("int64").to_numpy()
    event_ts = data.ts_event.astype("int64").to_numpy()
    actions = data.action.astype("str").to_numpy()
    sides = data.side.astype("str").to_numpy()
    prices = data.price.to_numpy(np.float64)
    sizes = data["size"].to_numpy(np.int64)
    oids = data.order_id.to_numpy(np.uint64)
    snapshots = data.is_snapshot.to_numpy(bool)
    packet_ends = np.flatnonzero(data.is_last.to_numpy(bool))
    diagnostics["rows"] = len(data)
    diagnostics["packets"] = len(packet_ends)
    if len(packet_ends) == 0 or packet_ends[-1] != len(data) - 1:
        diagnostics["trailing_incomplete_rows"] = len(data) - (packet_ends[-1] + 1 if len(packet_ends) else 0)

    first_utc = pd.Timestamp(path.stem[-10:], tz="UTC").value
    grid = np.arange(first_utc, first_utc + 86400 * NS, NS, dtype=np.int64)
    day_names, minute, clock_s, eligible = local_parts(grid)
    period = period_of_clock(clock_s)

    book = Book()
    born: dict[int, int] = {}
    pending_fill: dict[int, int] = {}
    cancel_clean = np.zeros(len(data), dtype=np.int32)
    modify_increase = np.zeros(len(data), dtype=np.int32)
    top1_adds = np.zeros(len(data), dtype=np.int32)
    top1_cancels = np.zeros(len(data), dtype=np.int32)
    top1_executions = np.zeros(len(data), dtype=np.int32)
    candidates: list[dict] = []
    active: dict[int, int] = {}
    active_order = deque()
    selected_seconds = set()
    samples, shape_totals = [], defaultdict(lambda: [0.0, 0.0, 0])
    prev_refs: dict = {}
    next_second = 0
    last_packet_ns = -1
    started = False

    def sample_one(g: int) -> None:
        if not eligible[g] or not started or grid[g] - last_packet_ns > 30 * NS:
            return
        snap, shape = book_snapshot(book, born, int(grid[g]), tick, prev_refs)
        if snap is None:
            diagnostics["invalid_or_empty_samples"] += 1
            return
        snap.update(utc_ns=int(grid[g]), london_day=str(day_names[g]), minute=int(minute[g]),
                    period=str(period[g]))
        samples.append(snap)
        for side, level, qty, count in shape:
            agg = shape_totals[(snap["period"], side, level)]
            agg[0] += qty
            agg[1] += count
            agg[2] += 1

    start = 0
    for end in packet_ends:
        packet_ns = int(ts[end])
        while next_second < len(grid) and grid[next_second] < packet_ns:
            sample_one(next_second)
            next_second += 1
        while active_order and candidates[active_order[0]]["entry_ns"] + 5 * NS < packet_ns:
            idx = active_order.popleft()
            oid = candidates[idx]["order_id"]
            if active.get(oid) == idx:
                active.pop(oid, None)
                candidates[idx]["label"] = 0
                candidates[idx]["exit_reason"] = "five_second_horizon"
        packet_adds = []
        for i in range(start, end + 1):
            a, s, p, z, oid = actions[i], sides[i], float(prices[i]), int(sizes[i]), int(oids[i])
            at_best = False
            if a in ("F", "C") and s in ("B", "A") and book.levels[s]:
                at_best = p == book.levels[s].peekitem(-1 if s == "B" else 0)[0]
            if a == "F":
                pending_fill[oid] = pending_fill.get(oid, 0) + z
                diagnostics["fill_contracts"] += z
                if at_best:
                    top1_executions[i] = z
                k = active.pop(oid, None)
                if k is not None and int(ts[i]) <= candidates[k]["entry_ns"] + 5 * NS:
                    candidates[k]["label"] = 1
                    candidates[k]["exit_reason"] = "fill"
                    candidates[k]["fill_time_s"] = (int(ts[i]) - candidates[k]["entry_ns"]) / NS
            elif a == "C":
                matched = min(z, pending_fill.pop(oid, 0))
                cancel_clean[i] = z - matched
                if at_best:
                    top1_cancels[i] = z - matched
                diagnostics["matched_fill_followup_c_contracts"] += matched
                k = active.pop(oid, None)
                if k is not None:
                    candidates[k]["label"] = 0
                    candidates[k]["exit_reason"] = "cancel"
            elif a == "M":
                pending_fill.pop(oid, None)
                old = book.orders.get(oid)
                if old and old[0] == s and old[1] == p and z > old[2]:
                    modify_increase[i] = z - old[2]
                if old and (old[0] != s or old[1] != p or z > old[2]):
                    born[oid] = int(event_ts[i])
                k = active.pop(oid, None)
                if k is not None:
                    candidates[k]["label"] = 0
                    candidates[k]["exit_reason"] = "modify_or_reprice"
            elif a == "A":
                pending_fill.pop(oid, None)
                born[oid] = int(event_ts[i])
                if not snapshots[i] and s in ("B", "A"):
                    packet_adds.append(i)
            elif a == "R":
                pending_fill.clear()
                born.clear()
                for k in active.values():
                    candidates[k]["label"] = 0
                    candidates[k]["exit_reason"] = "book_reset"
                active.clear()
                prev_refs.clear()
            if a in ("A", "C", "M", "R"):
                if a == "C":
                    old = book.orders.get(oid)
                    if old and z >= old[2]:
                        born.pop(oid, None)
                if not book.apply(a, oid, s, p, z):
                    diagnostics["unknown_order_events"] += 1
                if a == "A" and not snapshots[i] and s in ("B", "A") and book.levels[s]:
                    if p == book.levels[s].peekitem(-1 if s == "B" else 0)[0]:
                        top1_adds[i] = z
        started = True
        last_packet_ns = packet_ns
        bid, bq, _ = book.best("B")
        ask, aq, _ = book.best("A")
        if packet_adds and 0 < bid < ask and bq > 0 and aq > 0:
            local = pd.Timestamp(packet_ns, unit="ns", tz="UTC").tz_convert(LONDON)
            if local.weekday() < 5:
                sec_key = packet_ns // NS
                for i in packet_adds:
                    s, p, oid = sides[i], prices[i], int(oids[i])
                    key = (sec_key, s)
                    if key in selected_seconds or p != (bid if s == "B" else ask):
                        continue
                    if oid not in book.orders or book.orders[oid][0] != s or book.orders[oid][1] != p:
                        continue
                    ahead, n_ahead = book.queue_ahead(oid)
                    if ahead < 0:
                        continue
                    selected_seconds.add(key)
                    if oid in active:
                        diagnostics["duplicate_active_candidate_id"] += 1
                        continue
                    total_side = bq if s == "B" else aq
                    entry = {
                        "order_id": oid, "entry_ns": packet_ns, "side": s,
                        "size": book.orders[oid][2], "queue_ahead": ahead,
                        "orders_ahead": n_ahead,
                        "queue_position": ahead / total_side if total_side > 0 else np.nan,
                        "spread_ticks": (ask - bid) / tick,
                        "same_side_depth": total_side,
                        "imbalance": (bq - aq) / (bq + aq),
                        "label": np.nan, "exit_reason": "right_censored",
                        "fill_time_s": np.nan,
                    }
                    k = len(candidates)
                    candidates.append(entry)
                    active[oid] = k
                    active_order.append(k)
        start = end + 1
    while next_second < len(grid):
        sample_one(next_second)
        next_second += 1

    # No label is assigned when an order is still active at the truncated UTC
    # file boundary and fewer than five observable seconds remain.
    end_ns = int(ts[packet_ends[-1]]) if len(packet_ends) else first_utc
    for k in list(active.values()):
        if candidates[k]["entry_ns"] + 5 * NS <= end_ns:
            candidates[k]["label"] = 0
            candidates[k]["exit_reason"] = "five_second_horizon"

    sample_df = pd.DataFrame(samples)
    shape_df = pd.DataFrame([
        {"period": key[0], "side": key[1], "level": key[2],
         "mean_qty": val[0] / val[2], "mean_orders": val[1] / val[2], "sample_seconds": val[2]}
        for key, val in shape_totals.items() if val[2]
    ])
    if sample_df.empty:
        return sample_df, shape_df, pd.DataFrame(), pd.DataFrame(), diagnostics

    add_mask = (actions == "A") & ~snapshots
    add_day, add_minute, _, add_eligible = local_parts(ts[add_mask])
    add_sizes = pd.DataFrame({"london_day": add_day[add_eligible],
                              "minute": add_minute[add_eligible],
                              "new_order_size": sizes[add_mask][add_eligible]})
    add_sizes = add_sizes.groupby(["london_day", "minute"], as_index=False).agg(
        median_new_order_size=("new_order_size", "median"),
        new_order_count=("new_order_size", "size"))

    sec = ts // NS
    flow = pd.DataFrame({
        "utc_ns": sec * NS,
        "adds": np.where((actions == "A") & ~snapshots, sizes, 0),
        "cancels": cancel_clean,
        "executions": np.where(actions == "F", sizes, 0),
        "top1_adds": top1_adds,
        "top1_cancels": top1_cancels,
        "top1_executions": top1_executions,
        "trade_contracts": np.where(actions == "T", sizes, 0),
        "trade_count": (actions == "T").astype(np.int8),
        "modify_increase": modify_increase,
    }).groupby("utc_ns", sort=False).sum().reset_index()
    sample_df = sample_df.merge(flow, on="utc_ns", how="left")
    for c in ["adds", "cancels", "executions", "top1_adds", "top1_cancels", "top1_executions",
              "trade_contracts", "trade_count", "modify_increase"]:
        sample_df[c] = sample_df[c].fillna(0)

    cand = pd.DataFrame(candidates)
    if not cand.empty:
        cdate, cminute, cclock, cvalid = local_parts(cand.entry_ns.to_numpy(np.int64))
        cand["london_day"] = cdate
        cand["minute"] = cminute
        cand["period"] = period_of_clock(cclock)
        cand = cand.loc[cvalid & cand.label.notna()].copy()
        if not cand.empty:
            # Strictly past events. Events at exactly entry_ns are excluded.
            def past_sum(event_mask: np.ndarray, values: np.ndarray, horizon: int) -> np.ndarray:
                et = ts[event_mask]
                cum = np.r_[0, np.cumsum(values[event_mask], dtype=np.float64)]
                ct = cand.entry_ns.to_numpy(np.int64)
                hi = np.searchsorted(et, ct, side="left")
                lo = np.searchsorted(et, ct - horizon * NS, side="left")
                return cum[hi] - cum[lo]
            signed_trade = np.where(sides == "B", sizes, -sizes)
            for horizon in (1, 5, 30):
                cand[f"signed_flow_{horizon}s"] = past_sum(actions == "T", signed_trade, horizon)
            cand["cancel_5s"] = past_sum((actions == "C") & (cancel_clean > 0), cancel_clean, 5)
            # Last completed 1-Hz quote at/before the candidate observation.
            hist = sample_df[["utc_ns", "mid"]].drop_duplicates("utc_ns").sort_values("utc_ns")
            ht = hist.utc_ns.to_numpy(np.int64)
            hm = hist.mid.to_numpy(np.float64)
            hr = np.zeros(len(hm))
            good = np.diff(ht) == NS
            hr[1:] = np.where(good, np.diff(np.log(hm)), 0.0)
            rolling = np.r_[0, np.cumsum(hr * hr)]
            idx = np.searchsorted(ht, cand.entry_ns.to_numpy(np.int64), side="right") - 1
            valid = idx >= 0
            lo = np.searchsorted(ht, cand.entry_ns.to_numpy(np.int64) - 30 * NS, side="left")
            cand["recent_vol_30s_bps"] = np.where(valid, np.sqrt(np.maximum(rolling[np.maximum(idx, 0) + 1] - rolling[lo], 0)) * 1e4, np.nan)
    diagnostics["sample_seconds"] = len(sample_df)
    diagnostics["candidate_orders_labeled"] = len(cand)
    return sample_df, shape_df, cand, add_sizes, diagnostics


def daily_minutes(samples: pd.DataFrame) -> pd.DataFrame:
    x = samples.sort_values("utc_ns").copy()
    x["log_return"] = np.log(x.mid).diff()
    contiguous = (x.utc_ns.diff() == NS) & (x.london_day == x.london_day.shift())
    x.loc[~contiguous, "log_return"] = 0.0
    x["return_sq"] = x.log_return ** 2
    grouped = x.groupby(["london_day", "minute"], observed=True)
    out = grouped.agg(
        sample_seconds=("utc_ns", "size"), spread_ticks=("spread_ticks", "mean"),
        bid_depth=("bid_qty", "mean"), ask_depth=("ask_qty", "mean"),
        depth1=("depth1", "mean"), depth5=("depth5", "mean"), depth10=("depth10", "mean"),
        imbalance=("imbalance", "mean"), abs_imbalance=("abs_imbalance", "mean"),
        best_order_count=("best_order_count", "mean"),
        median_best_order_size=("median_best_order_size", "median"),
        median_order_age_s=("median_best_order_age_s", "median"),
        queue_ahead_change=("mean_queue_ahead_change", "mean"),
        rv_sq=("return_sq", "sum"),
        adds=("adds", "sum"), cancels=("cancels", "sum"),
        executions=("executions", "sum"), trade_contracts=("trade_contracts", "sum"),
        top1_adds=("top1_adds", "sum"), top1_cancels=("top1_cancels", "sum"),
        top1_executions=("top1_executions", "sum"),
        trade_count=("trade_count", "sum"), modify_increase=("modify_increase", "sum"),
    ).reset_index()
    out["rv_bps"] = np.sqrt(out.pop("rv_sq")) * 1e4
    for key in ("adds", "cancels", "executions", "top1_adds", "top1_cancels",
                "top1_executions", "trade_contracts", "modify_increase"):
        out[key + "_per_s"] = out[key] / out.sample_seconds
    return out


def summarize_periods(samples: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    x = samples.copy()
    x["same_hour"] = (x.minute >= 15 * 60 + 30) & (x.minute < 16 * 60 + 30)
    x["same_hour_control"] = x.same_hour & (x.period == "normal")
    metric_cols = ["spread_ticks", "bid_qty", "ask_qty", "depth1", "depth5", "depth10",
                   "imbalance", "abs_imbalance", "best_order_count", "median_best_order_size",
                   "median_best_order_age_s", "adds", "cancels", "executions", "top1_adds",
                   "top1_cancels", "top1_executions", "trade_contracts",
                   "trade_count", "modify_increase", "mean_queue_ahead_change"]
    rows = []
    for period in PERIODS:
        sub = x[x.same_hour_control] if period == "same_hour_control" else x[x.period == period]
        if sub.empty:
            continue
        by_day = sub.groupby("london_day")[metric_cols].mean(numeric_only=True)
        for metric in metric_cols:
            v = by_day[metric].dropna()
            rows.append({"period": period, "metric": metric,
                         "day_equal_mean": float(v.mean()) if len(v) else np.nan,
                         "day_p10": float(v.quantile(.1)) if len(v) else np.nan,
                         "day_p90": float(v.quantile(.9)) if len(v) else np.nan,
                         "n_days": len(v), "n_seconds": len(sub)})
    return pd.DataFrame(rows)


def transitions(samples: pd.DataFrame) -> pd.DataFrame:
    x = samples.sort_values("utc_ns").copy()
    x["spread_state"] = np.select([x.spread_ticks < 1.5, x.spread_ticks < 2.5], ["1 tick", "2 ticks"], default="3+ ticks")
    x["imbalance_state"] = np.select([x.imbalance < -.2, x.imbalance > .2], ["sell-heavy", "buy-heavy"], default="balanced")
    x["move_state"] = np.where(x.mid.diff() > 0, "up", np.where(x.mid.diff() < 0, "down", "flat"))
    x["bid_move_state"] = np.where(x.bid_px.diff() > 0, "up", np.where(x.bid_px.diff() < 0, "down", "flat"))
    x["ask_move_state"] = np.where(x.ask_px.diff() > 0, "up", np.where(x.ask_px.diff() < 0, "down", "flat"))
    contiguous = (x.utc_ns.shift(-1) - x.utc_ns == NS) & (x.london_day == x.london_day.shift(-1))
    rows = []
    for kind, col in [("spread", "spread_state"), ("imbalance", "imbalance_state"),
                      ("mid_move", "move_state"), ("bid_move", "bid_move_state"),
                      ("ask_move", "ask_move_state")]:
        sub = x.loc[contiguous, ["period", col]].copy()
        sub["to_state"] = x[col].shift(-1)[contiguous].to_numpy()
        counts = sub.groupby(["period", col, "to_state"], observed=True).size().rename("n").reset_index()
        counts["probability"] = counts.n / counts.groupby(["period", col]).n.transform("sum")
        counts = counts.rename(columns={col: "from_state"})
        counts["state_kind"] = kind
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def depletion(samples: pd.DataFrame) -> pd.DataFrame:
    x = samples.sort_values("utc_ns").copy()
    x["next_depth1"] = x.depth1.shift(-1)
    contiguous = (x.utc_ns.shift(-1) - x.utc_ns == NS) & (x.london_day == x.london_day.shift(-1))
    x = x.loc[contiguous].copy()
    q = x.depth1.quantile([1/3, 2/3]).to_numpy()
    x["depth_bucket"] = np.select([x.depth1 <= q[0], x.depth1 <= q[1]], ["thin", "medium"], default="thick")
    x["net_depth_change"] = x.next_depth1 - x.depth1
    metrics = ["top1_executions", "top1_cancels", "top1_adds", "net_depth_change", "mean_queue_ahead_change"]
    long = x.melt(id_vars=["period", "depth_bucket"], value_vars=metrics,
                  var_name="metric", value_name="value")
    return long.groupby(["period", "depth_bucket", "metric"], observed=True).agg(
        mean=("value", "mean"), count=("value", "count")).reset_index()


def fill_bins(candidates: pd.DataFrame, feature: str, period: str, side: str, bins: int = 6) -> pd.DataFrame:
    sub = candidates.loc[(candidates.period == period) & (candidates.side == side) & candidates[feature].notna(),
                         ["london_day", feature, "label"]].copy()
    if len(sub) < 50:
        return pd.DataFrame()
    # Fixed pooled cut points for each feature, shared by the period curves.
    values = candidates[feature].dropna()
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame()
    edges[0], edges[-1] = -np.inf, np.inf
    sub["bin"] = pd.cut(sub[feature], edges, labels=False, include_lowest=True)
    daily = sub.groupby(["london_day", "bin"], observed=True).agg(p=("label", "mean"), n=("label", "size"), x=(feature, "median")).reset_index()
    out = daily.groupby("bin", observed=True).agg(probability=("p", "mean"), p10=("p", lambda s: s.quantile(.1)),
                                                  p90=("p", lambda s: s.quantile(.9)),
                                                  x=("x", "median"), n=("n", "sum"), days=("london_day", "nunique")).reset_index()
    out["period"] = period
    out["side"] = side
    out["feature"] = feature
    return out


def feature_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    features = ["queue_ahead", "orders_ahead", "queue_position", "spread_ticks",
                "same_side_depth", "imbalance", "signed_flow_1s", "signed_flow_5s",
                "signed_flow_30s", "cancel_5s", "recent_vol_30s_bps", "size"]
    rows = []
    for period in ("wmr", "nearby", "normal"):
        for side in ("B", "A"):
            subset = candidates[(candidates.period == period) & (candidates.side == side)]
            for feature in features:
                sub = subset[[feature, "label"]].dropna()
                if len(sub) < 30:
                    continue
                r = (sub[feature].rank(method="average").corr(sub.label.rank(method="average"))
                     if sub[feature].nunique() > 1 else np.nan)
                rows.append({"period": period, "side": side, "feature": feature,
                             "n_orders": len(sub), "n_days": subset.london_day.nunique(),
                             "fill_probability": sub.label.mean(), "spearman": r,
                             "median_feature": sub[feature].median()})
    return pd.DataFrame(rows)


def plot_all(out: Path, minutes: pd.DataFrame, samples: pd.DataFrame,
             shape: pd.DataFrame, trans: pd.DataFrame, dep: pd.DataFrame,
             candidates: pd.DataFrame, curves: pd.DataFrame) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160,
                         "axes.grid": True, "grid.alpha": .25, "font.size": 9})
    profile = minutes.groupby("minute").agg({c: ["mean", lambda s: s.quantile(.1), lambda s: s.quantile(.9)]
                                                for c in ["spread_ticks", "rv_bps", "bid_depth", "ask_depth", "depth5",
                                                          "imbalance", "abs_imbalance", "adds_per_s", "cancels_per_s",
                                                          "trade_contracts_per_s", "trade_count", "median_new_order_size"]})
    profile.columns = [f"{a}_{'mean' if b == 'mean' else 'p10' if b == '<lambda_0>' else 'p90'}" for a,b in profile.columns]
    profile = profile.reset_index()
    x = profile.minute / 60
    panels = [("spread_ticks", "Spread (ticks)"), ("rv_bps", "1-min realized volatility (bps)"),
              ("bid_depth", "Best bid depth (contracts)"), ("ask_depth", "Best ask depth (contracts)"),
              ("depth5", "Total top-5 depth (contracts)"), ("imbalance", "Mean book imbalance"),
              ("abs_imbalance", "Mean absolute imbalance"), ("adds_per_s", "Added contracts/s"),
              ("cancels_per_s", "Cancelled contracts/s"), ("trade_contracts_per_s", "Traded contracts/s"),
              ("trade_count", "Trade messages/min"), ("median_new_order_size", "Median new limit-order size (contracts)")]
    def draw_profile(name: str, zoom: bool) -> None:
        fig, axes = plt.subplots(4, 3, figsize=(16, 12), constrained_layout=True)
        for ax, (col, label) in zip(axes.flat, panels):
            ax.plot(x, profile[f"{col}_mean"], color="#24558b", lw=1)
            ax.fill_between(x, profile[f"{col}_p10"], profile[f"{col}_p90"], color="#24558b", alpha=.16,
                            label="10th–90th day percentile")
            ax.axvspan(15 + 57.5/60, 16 + 2.5/60, color="#e8ae5d", alpha=.3)
            ax.axvline(16, color="#b93c32", ls="--", lw=1)
            ax.set_title(label)
            ax.set_xlabel("London local clock (hours)")
            if zoom:
                ax.set_xlim(15.5, 16.5)
            else:
                ax.set_xlim(0, 24)
        fig.suptitle(f"6E pooled intraday profile, {minutes.london_day.nunique()} London weekdays; per-day minute values", fontsize=14)
        fig.savefig(figdir / name)
        plt.close(fig)
    draw_profile("intraday_profile.png", False)
    draw_profile("wmr_zoom.png", True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, col, label in zip(axes.flat, ["spread_ticks", "depth1", "depth5", "depth10", "imbalance", "abs_imbalance"],
                              ["Spread (ticks)", "Top-1 depth (contracts)", "Top-5 depth (contracts)",
                               "Top-10 depth (contracts)", "Signed imbalance", "Absolute imbalance"]):
        for period in ("wmr", "nearby", "normal"):
            y = samples.loc[samples.period == period, col].dropna().to_numpy()
            if len(y):
                ax.hist(y, bins=50, density=True, histtype="step", lw=1.5,
                        color=PERIOD_COLORS[period], label=f"{period} (n={len(y):,} s)")
        if col.startswith("depth"):
            ax.set_xscale("log")
            ax.set_xlim(left=1)
        ax.set_title(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.savefig(figdir / "spread_depth_distributions.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, (metric, ylabel) in enumerate((("mean_qty", "Mean resting quantity (contracts)"),
                                            ("mean_orders", "Mean resting order count"))):
        for ax, side in zip(axes[row], ("B", "A")):
            for period in ("wmr", "nearby", "normal"):
                sub = shape[(shape.side == side) & (shape.period == period)].sort_values("level")
                if len(sub):
                    ax.plot(sub.level, sub[metric], "o-", label=period, color=PERIOD_COLORS[period])
            ax.set_title("Bid" if side == "B" else "Ask")
            ax.set_xlabel("Price level from best")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(1, 11))
            ax.legend()
    fig.savefig(figdir / "book_shape.png")
    plt.close(fig)

    fig, axes = plt.subplots(5, 3, figsize=(13, 17), constrained_layout=True)
    state_sets = {"spread": ["1 tick", "2 ticks", "3+ ticks"],
                  "imbalance": ["sell-heavy", "balanced", "buy-heavy"],
                  "mid_move": ["down", "flat", "up"],
                  "bid_move": ["down", "flat", "up"],
                  "ask_move": ["down", "flat", "up"]}
    for row, kind in enumerate(state_sets):
        states = state_sets[kind]
        for col, period in enumerate(("wmr", "nearby", "normal")):
            ax = axes[row, col]
            sub = trans[(trans.state_kind == kind) & (trans.period == period)]
            mat = sub.pivot(index="from_state", columns="to_state", values="probability").reindex(index=states, columns=states)
            mat.loc[mat.notna().any(axis=1)] = mat.loc[mat.notna().any(axis=1)].fillna(0)
            cmap = plt.get_cmap("Blues").copy()
            cmap.set_bad("#dddddd")
            im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(float)), vmin=0, vmax=1, cmap=cmap)
            ax.set_xticks(range(3), states, rotation=35, ha="right")
            ax.set_yticks(range(3), states)
            ax.set_title(f"{kind}: {period}")
            for (i,j), v in np.ndenumerate(mat.to_numpy()):
                ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "—", ha="center", va="center",
                        color="white" if v>.6 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.55, label="Next-second probability")
    fig.savefig(figdir / "state_transitions.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, period in zip(axes, ("wmr", "nearby", "normal")):
        sub = dep[dep.period == period]
        for metric in ("top1_executions", "top1_cancels", "top1_adds", "net_depth_change"):
            y = sub[sub.metric == metric].set_index("depth_bucket").reindex(["thin", "medium", "thick"])["mean"]
            ax.plot([1,2,3], y, "o-", label=metric)
        ax.set_xticks([1,2,3], ["thin", "medium", "thick"])
        ax.set_xlabel("Starting top-1 depth bucket")
        ax.set_ylabel("Best-level contracts per second")
        ax.set_title(period)
        ax.legend(fontsize=7)
    fig.savefig(figdir / "queue_depletion.png")
    plt.close(fig)

    if not candidates.empty and not curves.empty:
        features = ["queue_ahead", "orders_ahead", "queue_position", "spread_ticks",
                    "same_side_depth", "imbalance", "signed_flow_5s", "cancel_5s", "recent_vol_30s_bps", "size"]
        fig, axes = plt.subplots(5, 4, figsize=(22, 18), constrained_layout=True)
        for j, feature in enumerate(features):
            for side_idx, side in enumerate(("B", "A")):
                ax = axes.flat[2*j + side_idx]
                for period in ("wmr", "nearby", "normal"):
                    sub = curves[(curves.feature == feature) & (curves.period == period) &
                                 (curves.side == side)].sort_values("bin")
                    if sub.empty:
                        continue
                    ax.plot(sub.x, sub.probability, "o-", color=PERIOD_COLORS[period], label=f"{period} (n={sub.n.sum():,})")
                    ax.fill_between(sub.x.to_numpy(float), sub.p10.to_numpy(float), sub.p90.to_numpy(float),
                                    color=PERIOD_COLORS[period], alpha=.12)
                ax.set_title(f"{feature.replace('_', ' ')} | {'bid' if side == 'B' else 'ask'}")
                ax.set_xlabel("Feature value (bin median)")
                ax.set_ylabel("Fill within 5 s before exit")
                ax.set_ylim(0, 1)
                ax.legend(fontsize=7)
        fig.savefig(figdir / "fill_probability_curves.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
        for period in ("wmr", "nearby", "normal"):
            v = candidates.loc[(candidates.period == period) & (candidates.label == 1), "fill_time_s"].dropna().sort_values().to_numpy()
            if len(v):
                ax.step(v, np.arange(1, len(v)+1)/len(v), where="post", label=f"{period} (n={len(v):,})", color=PERIOD_COLORS[period])
        ax.set(xlabel="Seconds from observed add to first fill", ylabel="CDF among filled orders", xlim=(0,5), ylim=(0,1))
        ax.legend()
        fig.savefig(figdir / "fill_time_distribution.png")
        plt.close(fig)


def report(out: Path, day_diag: list[dict], minutes: pd.DataFrame, samples: pd.DataFrame,
           period_table: pd.DataFrame, candidates: pd.DataFrame, feat: pd.DataFrame) -> None:
    def value(period, metric):
        z = period_table[(period_table.period == period) & (period_table.metric == metric)]
        return float(z.day_equal_mean.iloc[0]) if len(z) else np.nan
    def change(metric):
        a, b = value("wmr", metric), value("nearby", metric)
        return f"{a:.3g} vs {b:.3g} ({(a/b-1)*100:+.1f}%)" if np.isfinite(a) and np.isfinite(b) and b else "insufficient"
    fill_rates = candidates.groupby("period").label.agg(["mean", "size"]) if not candidates.empty else pd.DataFrame()
    lines = [
        "# 6E WMR fixing window: exploratory L3 analysis", "",
        "This is descriptive analysis of the processed Databento 6E MBO files, 2026-07-14 through 2026-10-13 UTC. The sample uses 6EU6 through 2026-09-10 and 6EZ6 from 2026-09-11, following CME's FX quarterly roll date. All clocks are converted from UTC to `Europe/London` with timezone-aware conversion. The WMR period is 15:57:30–16:02:30 London time; nearby controls are the adjacent five-minute windows.", "",
        "## Five observations", "",
        f"1. Spread during WMR versus nearby: {change('spread_ticks')} ticks (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `spread_depth_distributions.png`).",
        f"2. Total top-five displayed depth: {change('depth5')} contracts (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `book_shape.png`).",
        f"3. Executed contracts per sampled second: {change('executions')} (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `queue_depletion.png`).",
        f"4. Cancelled contracts per sampled second: {change('cancels')} (Table `period_comparison.csv`; Figure `queue_depletion.png`).",
        f"5. Absolute best-quote imbalance: {change('abs_imbalance')} (Table `period_comparison.csv`; Figure `wmr_zoom.png`).", "",
        "## Intraday patterns", "",
        f"`intraday_metrics.csv` contains one row per London day and minute for {minutes.london_day.nunique()} weekdays, including partial London days at the file boundaries; {samples.loc[samples.period == 'wmr', 'london_day'].nunique()} weekdays contribute to WMR. Each plotted point in `intraday_profile.png` and `wmr_zoom.png` is the equal-day mean of that minute's value, with 10th–90th day percentile bands. Realized volatility is the square root of summed squared adjacent one-second log-mid returns, in basis points. The median new limit-order size uses `A` event sizes within each day-minute, excluding snapshot rows.", "",
        "## Queue structure and dynamics", "",
        f"Best-level resting order count during WMR versus nearby: {change('best_order_count')}; median best-order age: {change('median_best_order_age_s')} seconds; new displayed quantity per second: {change('adds')}; and mean change in the surviving best-queue reference's volume ahead: {value('wmr', 'mean_queue_ahead_change'):.3g} versus {value('nearby', 'mean_queue_ahead_change'):.3g} contracts (`period_comparison.csv`). Compare the conditional mechanisms in `queue_depletion.png` and the order-count shape in `book_shape.png`.", "",
        "`book_shape.png` and `spread_depth_distributions.png` use one-second completed book states. `state_transitions.png` shows empirical one-second transition probabilities, and `transition_matrices.csv` gives counts. `queue_depletion.png` conditions best-price execution (`F`), best-price cancellation (`C` after subtracting matched fill follow-ups), best-price additions (`A`), and top-of-book net depth change on starting depth. `queue_depletion.csv` provides the underlying numbers. The reference order for queue-ahead change is the last order at the best price in the prior second, retained only if the same order is still at that best price in the next sample.", "",
        "## Fill dynamics", "",
    ]
    if fill_rates.empty:
        lines.append("No valid order-level labels were available; fill figures and feature table were skipped.")
    else:
        for p in ("wmr", "nearby", "normal"):
            if p in fill_rates.index:
                lines.append(f"- {p}: {fill_rates.loc[p, 'mean']:.3f} observed first-fill probability among {int(fill_rates.loc[p, 'size']):,} selected best-quote additions (`fill_probability_curves.png`).")
        for p in ("wmr", "nearby", "normal"):
            for side in ("B", "A"):
                sub = feat[(feat.period == p) & (feat.side == side)].dropna(subset=["spearman"])
                if not sub.empty:
                    top = sub.loc[sub.spearman.abs().idxmax()]
                    lines.append(f"- {p} {side}: strongest displayed single-feature rank association is `{top.feature}` (Spearman {top.spearman:+.3f}, {int(top.n_orders):,} orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).")
        lines.extend(["", "Candidates are the first eligible new best-quote order per side per second, observed after its complete event packet. Label 1 means an `F` event for the same order ID within five seconds and before an exit; partial fills count. `C`, `M`, or book reset before the first `F` ends the observed episode with label 0. Orders still active after five observed seconds receive label 0. Orders still active near a truncated file boundary are excluded. `queue_position` is visible volume ahead divided by total same-side best-level volume. `fill_time_distribution.png` is a CDF **among filled candidates**. `feature_fill_summary.csv` gives descriptive Spearman associations by side and period; `fill_probability_bins.csv` gives day-equal binned curves, day percentile bands, and observation counts. These labels are for observed orders under their owners' cancellation behavior, not hypothetical orders held for five seconds."])
    lines.extend(["", "## Next-step hypotheses", "",
        "1. At the same London clock time, WMR changes the conditional execution hazard after controlling for visible queue-ahead quantity and spread.",
        "2. WMR changes whether a thin best queue is depleted by execution or cancellation, at a fixed starting depth bucket.",
        "3. Queue-ahead quantity has different fill-probability calibration during WMR than in nearby controls, separately for bid and ask orders.", "",
        "## Limits and checks", "",
        f"The input has {sum(d.get('rows', 0) for d in day_diag):,} processed rows across {len(day_diag)} UTC files; Saturday files are empty and Sunday sessions are omitted from weekday comparisons. Day-level percentile bands describe variation across dates and are not confidence intervals. Message streams within a day are dependent. A `T` row gives aggressive trade quantity while `F` records passive execution; the two are never summed as separate trade volume. The post-fill `C`/`M` book update is not counted as another execution. Residual `C` quantity after a matched `F` is classified as cancellation; unmatched `M` reductions are not assigned to execution or cancellation. One-second states are sampled only after complete packets and are omitted if stale for more than 30 seconds. Book depth beyond the best uses the existing `Book` utility on the processed event stream; this is an analytical view, not a regenerated L3 dataset. The first eligible add per side per second limits message-rate weighting but does not remove traders' selection or cancellation bias.", "",
        f"Unknown-order event references observed by analytical replay: {sum(d.get('unknown_order_events',0) for d in day_diag):,}. Trailing incomplete packet rows: {sum(d.get('trailing_incomplete_rows',0) for d in day_diag):,}. Manual event-sequence checks found a trade `T` paired with passive `F` fills and follow-up `C` reductions, and a partial `F` followed by `M` of the remaining order; the accounting counts the `F` once in each case. Inspect `diagnostics.csv` before relying on results. Time-of-day sensitivity is in `period_comparison.csv` under `same_hour_control` (15:30–16:30 excluding WMR and nearby). No causal or statistical-significance claim follows from these descriptive plots.", "",
        "## Reproduce", "", "From the `project` directory, run with its virtual environment: `.venv\\Scripts\\python.exe wmr_eda.py` (or `python wmr_eda.py` after activating `.venv`). Optional `--limit-days 1` provides a smoke run. Completed daily replay results are cached in `daily_cache_v1/` for restart.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--limit-days", type=int)
    args = parser.parse_args()
    project = args.project_root.resolve()
    out = (args.out_dir or project / "outputs" / "wmr_eda").resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    tick = tick_size(project)
    l3 = project / "data_6e" / "l3"
    paths = []
    for day in pd.date_range("2026-07-14", "2026-10-13", freq="D"):
        symbol = "6EU6" if day < pd.Timestamp("2026-09-11") else "6EZ6"
        paths.append(l3 / f"{symbol}_mbo_{day.date()}.parquet")
    if not args.limit_days:
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} L3 UTC days: {missing[:5]}")
    if args.limit_days:
        paths = [p for p in paths if p.is_file() and pq.ParquetFile(p).metadata.num_rows][0:args.limit_days]
    if not paths:
        raise FileNotFoundError("No processed 6E L3 files found")
    minute_frames, sample_frames, shape_frames, candidate_frames, add_size_frames, diagnostics = [], [], [], [], [], []
    cache_dir = out / "daily_cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        cache = [cache_dir / f"{path.stem}_{kind}.parquet" for kind in
                 ("samples", "shape", "candidates", "add_sizes")]
        cache_diag = cache_dir / f"{path.stem}_diagnostics.json"
        if all(p.exists() for p in cache) and cache_diag.exists():
            samples, shape, candidates, add_sizes = [pd.read_parquet(p) for p in cache]
            diag = Counter(json.loads(cache_diag.read_text(encoding="utf-8")))
        else:
            samples, shape, candidates, add_sizes, diag = process_day(path, project, tick)
            for frame, cache_path in zip((samples, shape, candidates, add_sizes), cache):
                frame.to_parquet(cache_path, index=False)
            cache_diag.write_text(json.dumps(dict(diag)), encoding="utf-8")
        diag["file"] = path.name
        diagnostics.append(dict(diag))
        print(path.name, "samples", len(samples), "candidates", len(candidates), flush=True)
        # Arrow strings keep the three-month pooled run within memory without
        # changing groupby semantics or any numeric precision.
        for frame in (samples, candidates):
            for name in ("period", "london_day", "side", "exit_reason"):
                if name in frame:
                    frame[name] = frame[name].astype("string[pyarrow]")
        if not samples.empty:
            samples["minute"] = samples.minute.astype("int16")
        if not samples.empty:
            minute_frames.append(daily_minutes(samples))
            sample_frames.append(samples)
            shape_frames.append(shape)
        if not candidates.empty:
            candidate_frames.append(candidates)
        if not add_sizes.empty:
            add_size_frames.append(add_sizes)
    if not sample_frames:
        raise ValueError("No eligible completed one-second book states")
    samples = pd.concat(sample_frames, ignore_index=True)
    sample_frames.clear()
    minutes = pd.concat(minute_frames, ignore_index=True)
    minute_frames.clear()
    # UTC-day files can each contain part of the same London-local day/minute.
    minutes = minutes.groupby(["london_day", "minute"], as_index=False).agg({
        **{c: "sum" for c in ["sample_seconds", "adds", "cancels", "executions", "top1_adds", "top1_cancels", "top1_executions", "trade_contracts", "trade_count", "modify_increase"]},
        **{c: "mean" for c in ["spread_ticks", "bid_depth", "ask_depth", "depth1", "depth5", "depth10", "imbalance", "abs_imbalance", "best_order_count", "median_best_order_size", "median_order_age_s", "queue_ahead_change", "rv_bps"]},
    })
    for key in ("adds", "cancels", "executions", "top1_adds", "top1_cancels",
                "top1_executions", "trade_contracts", "modify_increase"):
        minutes[key + "_per_s"] = minutes[key] / minutes.sample_seconds
    if add_size_frames:
        add_sizes = pd.concat(add_size_frames, ignore_index=True)
        minutes = minutes.merge(add_sizes, on=["london_day", "minute"], how="left")
    shape_parts = pd.concat(shape_frames, ignore_index=True)
    shape_frames.clear()
    shape_parts["weighted_qty"] = shape_parts.mean_qty * shape_parts.sample_seconds
    shape_parts["weighted_orders"] = shape_parts.mean_orders * shape_parts.sample_seconds
    shape = shape_parts.groupby(["period", "side", "level"], as_index=False).agg(
        weighted_qty=("weighted_qty", "sum"),
        weighted_orders=("weighted_orders", "sum"),
        sample_seconds=("sample_seconds", "sum"))
    shape["mean_qty"] = shape.weighted_qty / shape.sample_seconds
    shape["mean_orders"] = shape.weighted_orders / shape.sample_seconds
    shape = shape.drop(columns=["weighted_qty", "weighted_orders"])
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    candidate_frames.clear()
    add_size_frames.clear()
    period_table = summarize_periods(samples, minutes)
    trans = transitions(samples)
    dep = depletion(samples)
    curves_list = [fill_bins(candidates, feature, period, side) for feature in
                   ["queue_ahead", "orders_ahead", "queue_position", "spread_ticks", "same_side_depth",
                    "imbalance", "signed_flow_5s", "cancel_5s", "recent_vol_30s_bps", "size"]
                   for period in ("wmr", "nearby", "normal")
                   for side in ("B", "A")] if not candidates.empty else []
    curves = pd.concat([x for x in curves_list if not x.empty], ignore_index=True) if any(not x.empty for x in curves_list) else pd.DataFrame()
    feat = feature_summary(candidates) if not candidates.empty else pd.DataFrame()
    for name, frame in [("intraday_metrics", minutes), ("period_comparison", period_table),
                        ("transition_matrices", trans), ("queue_depletion", dep),
                        ("feature_fill_summary", feat), ("fill_probability_bins", curves),
                        ("book_shape", shape), ("diagnostics", pd.DataFrame(diagnostics))]:
        frame.to_csv(out / "tables" / f"{name}.csv", index=False)
    plot_all(out, minutes, samples, shape, trans, dep, candidates, curves)
    report(out, diagnostics, minutes, samples, period_table, candidates, feat)
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()
