"""First-pass tests: is the WMR fix qualitatively different from an equally busy market?

Run from the project directory: .venv/Scripts/python.exe first_pass.py
Optional: --limit-days 2 for a smoke run.

Tests (day is the unit of inference; WMR minus control, paired by London day):
  A  activity mix      top-of-book executed/added and cancelled/added (from wmr_eda cache)
  B  toxicity          passive-fill markouts (maker P&L marked to mid) at several horizons
  C  reversal          correlation of in-window and post-window returns vs placebo windows
  D  flow persistence  z^2 = (net signed volume)^2 / sum(size^2) of aggressive orders
  E  liquidity fade    share of remaining hit-side best depth cancelled right after a trade
  F  market maker      maker P&L when recent trade flow is against vs not against the fill
Controls: nearby windows, US-hours windows (13:30, 14:30, 15:00 London), all normal
seconds, and normal seconds reweighted to WMR's trailing-60s activity distribution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from wmr_eda import NS, local_parts, tick_size

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "outputs" / "first_pass"
CACHE = OUT / "cache"
EDA_CACHE = PROJECT / "outputs" / "wmr_eda" / "daily_cache_v1"
HORIZONS = [0, 1, 5, 30, 60, 300]          # seconds after the fill packet
FADE_MS = [10, 100, 1000]
WINDOWS = {"wmr": [57450], "nearby": [57150, 57750],            # London clock seconds
           "us_event": [48450, 52050, 53850]}                    # 13:27:30, 14:27:30, 14:57:30
CONTROLS = ["nearby", "us_event", "normal", "normal_matched"]
PLACEBO_STARTS = np.arange(3600, 74701, 150)                    # 01:00 .. 20:45 London
RNG = np.random.default_rng(0)


def period_of(clock: np.ndarray) -> np.ndarray:
    out = np.full(len(clock), "normal", dtype="U8")
    for name, starts in WINDOWS.items():
        for s in starts:
            out[(clock >= s) & (clock < s + 300)] = name
    return out


def trailing(t_events: np.ndarray, values: np.ndarray, t: np.ndarray, horizon_ns: int) -> np.ndarray:
    """Sum of values with event time in [t - horizon, t)."""
    cum = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    return cum[np.searchsorted(t_events, t, "left")] - cum[np.searchsorted(t_events, t - horizon_ns, "left")]


def process_day(path: Path, tick: float) -> dict[str, pd.DataFrame]:
    cols = ["ts_recv", "action", "side", "price", "size", "order_id", "is_last",
            "is_snapshot", "bid_px", "bid_qty", "ask_px", "ask_qty"]
    d = pq.read_table(path, columns=cols).to_pandas()
    ts = d.ts_recv.astype("int64").to_numpy()
    act = d.action.astype(str).to_numpy()
    side = d.side.astype(str).to_numpy()
    px = d.price.to_numpy(np.float64)
    sz = d["size"].to_numpy(np.int64)
    oid = d.order_id.to_numpy(np.uint64)
    live = ~d.is_snapshot.to_numpy(bool)
    last = d.is_last.to_numpy(bool)
    ends = np.flatnonzero(last)
    pid = np.r_[0, np.cumsum(last)[:-1]]
    live &= pid < len(ends)                                   # drop a trailing incomplete packet
    end_ts = ts[ends]
    bid = d.bid_px.to_numpy(np.float64)[ends]
    ask = d.ask_px.to_numpy(np.float64)[ends]
    bq = d.bid_qty.to_numpy(np.int64)[ends]
    aq = d.ask_qty.to_numpy(np.int64)[ends]
    ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    mid = pd.Series(np.where(ok, (bid + ask) / 2, np.nan)).ffill().to_numpy()

    first_utc = pd.Timestamp(path.stem[-10:], tz="UTC").value
    grid = first_utc + np.arange(86400, dtype=np.int64) * NS
    day_names, _, clock_s, eligible = local_parts(grid)
    period_s = period_of(clock_s)

    def sec(t):
        return np.clip((t - first_utc) // NS, 0, 86399)

    def mid_at(t):
        j = np.searchsorted(end_ts, t, "right") - 1
        out = np.where(j >= 0, mid[np.maximum(j, 0)], np.nan)
        return np.where(t <= end_ts[-1], out, np.nan)

    # Aggressive trades: T side is the aggressor (B buy, A sell).
    is_t = live & (act == "T") & np.isin(side, ["B", "A"])
    t_ts, t_sign, t_sz = ts[is_t], np.where(side[is_t] == "B", 1, -1), sz[is_t]

    # B/F: passive fills. F side is the resting order's side.
    is_f = live & (act == "F") & np.isin(side, ["B", "A"])
    k = pid[is_f]
    f_ts, f_px = end_ts[k], px[is_f]
    s = np.where(side[is_f] == "B", 1, -1)
    fills = pd.DataFrame({"size": sz[is_f], "s": s})
    fills["half_spread"] = s * (np.where(k > 0, mid[np.maximum(k - 1, 0)], np.nan) - f_px) / tick
    for h in HORIZONS:
        fills[f"mk_{h}"] = s * (mid_at(f_ts + h * NS) - f_px) / tick
    raw_ts = ts[is_f]
    fills["act60"] = trailing(t_ts, t_sz, raw_ts, 60 * NS)
    fills["flow_vs_maker"] = s * trailing(t_ts, t_sign * t_sz, raw_ts, 5 * NS)
    i = sec(raw_ts)
    fills["day"], fills["clock"], fills["period"], fills["ok"] = day_names[i], clock_s[i], period_s[i], eligible[i]
    fills = fills[fills.ok].drop(columns="ok")

    # E: cancellations at (side, price), excluding fill follow-ups in the same packet.
    is_c = live & (act == "C") & np.isin(side, ["B", "A"])
    fkeys = pd.MultiIndex.from_arrays([pid[is_f], oid[is_f]])
    follow = pd.MultiIndex.from_arrays([pid[is_c], oid[is_c]]).isin(fkeys)
    c_idx = np.flatnonzero(is_c)[~follow]
    key = lambda side_arr, price: (np.where(side_arr == "B", 0, 1) * 50000 + np.rint(price / tick)).astype(np.int64)
    comb = key(side[c_idx], px[c_idx]) * 10**11 + (ts[c_idx] - first_utc) // 1000
    order = np.argsort(comb, kind="stable")
    comb = comb[order]
    ccum = np.r_[0, np.cumsum(sz[c_idx][order])]

    def cancelled(side_arr, price, t, ms):
        base = key(side_arr, price) * 10**11 + (t - first_utc) // 1000
        return ccum[np.searchsorted(comb, base + ms * 1000, "right")] - ccum[np.searchsorted(comb, base, "right")]

    # Trade packets: last T row in each packet; keep if the hit level survives.
    tp = pd.DataFrame({"pid": pid[is_t], "row": np.flatnonzero(is_t)}).groupby("pid").row.last()
    rows, kk = tp.to_numpy(), tp.index.to_numpy()
    trade_end_ts = end_ts[kk]                                  # every trade packet, before filtering
    hit = np.where(side[rows] == "A", "B", "A")
    best = np.where(hit == "B", bid[kk], ask[kk])
    q = np.where(hit == "B", bq[kk], aq[kk])
    keep = ok[kk] & (np.rint(best / tick) == np.rint(px[rows] / tick)) & (q > 0)
    rows, kk, hit, best, q = rows[keep], kk[keep], hit[keep], best[keep], q[keep]
    t0 = end_ts[kk]
    fades = pd.DataFrame({"act60": trailing(t_ts, t_sz, t0, 60 * NS), "q": q})
    nxt = np.searchsorted(trade_end_ts, t0, "right")
    fades["next_trade_ms"] = np.where(nxt < len(trade_end_ts),
                                      (trade_end_ts[np.minimum(nxt, len(trade_end_ts) - 1)] - t0) / 1e6, np.inf)
    for ms in FADE_MS:
        fades[f"fade_{ms}"] = np.minimum(1.0, cancelled(hit, best, t0, ms) / q)
    i = sec(t0)
    fades["day"], fades["period"], fades["ok"] = day_names[i], period_s[i], eligible[i]
    fades = fades[fades.ok].drop(columns="ok")

    # Unconditional baseline: same measure from each live 1-Hz grid second, both sides.
    j = np.searchsorted(end_ts, grid, "right") - 1
    g_ok = eligible & (j >= 0) & (grid - end_ts[np.maximum(j, 0)] <= 30 * NS)
    g_ok &= ok[np.maximum(j, 0)]
    g, jj = grid[g_ok], j[g_ok]
    base_parts = []
    for sd, price, qty in (("B", bid[jj], bq[jj]), ("A", ask[jj], aq[jj])):
        part = pd.DataFrame({"act60": trailing(t_ts, t_sz, g, 60 * NS), "q": qty})
        nxt = np.searchsorted(trade_end_ts, g, "right")
        part["next_trade_ms"] = np.where(nxt < len(trade_end_ts),
                                         (trade_end_ts[np.minimum(nxt, len(trade_end_ts) - 1)] - g) / 1e6, np.inf)
        sd_arr = np.full(len(g), sd)
        for ms in FADE_MS:
            part[f"fade_{ms}"] = np.minimum(1.0, cancelled(sd_arr, price, g, ms) / np.maximum(qty, 1))
        part["day"], part["period"] = day_names[g_ok], period_s[g_ok]
        base_parts.append(part)
    base = pd.concat(base_parts, ignore_index=True)

    # C/D: windows on a London-clock grid. Aggressive orders = T rows grouped by packet.
    agg = pd.DataFrame({"pid": pid[is_t], "sv": t_sign * t_sz, "v": t_sz, "t": t_ts}).groupby("pid").agg(
        sv=("sv", "sum"), v=("v", "sum"), t=("t", "first"))
    a_t = agg.t.to_numpy()
    cum_s = np.r_[0.0, np.cumsum(agg.sv.to_numpy(np.float64))]
    cum_q = np.r_[0.0, np.cumsum(agg.v.to_numpy(np.float64) ** 2)]
    cum_v = np.r_[0.0, np.cumsum(agg.v.to_numpy(np.float64))]
    win = []
    pos = {c: n for n, c in enumerate(clock_s)}
    for start in PLACEBO_STARTS:
        if start not in pos or start + 900 not in pos or not eligible[pos[start]]:
            continue
        a = grid[pos[start]]
        lm = np.log(mid_at(np.array([a - 600 * NS, a, a + 300 * NS, a + 900 * NS])))
        lo, hi = np.searchsorted(a_t, [a, a + 300 * NS])
        win.append({"day": day_names[pos[start]], "start": int(start),
                    "r_pre": (lm[1] - lm[0]) * 1e4, "r_in": (lm[2] - lm[1]) * 1e4, "r_post": (lm[3] - lm[2]) * 1e4,
                    "net": cum_s[hi] - cum_s[lo], "sumsq": cum_q[hi] - cum_q[lo], "volume": cum_v[hi] - cum_v[lo],
                    "orders": hi - lo})
    return {"fills": fills, "fades": fades, "base": base, "windows": pd.DataFrame(win)}


# ---------- statistics ----------

def bucket(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    return df.assign(bucket=np.digitize(df.act60.to_numpy(), edges))


def day_table(df: pd.DataFrame, num: pd.Series, den: pd.Series | None = None) -> pd.DataFrame:
    """Per-day ratio sum(num)/sum(den) for each period, plus normal reweighted to WMR's activity buckets."""
    x = pd.DataFrame({"day": df.day.to_numpy(), "period": df.period.to_numpy(), "bucket": df.bucket.to_numpy(),
                      "num": np.asarray(num, float), "den": np.ones(len(df)) if den is None else np.asarray(den, float)})
    x = x[np.isfinite(x.num) & np.isfinite(x.den)]
    tot = x.groupby(["day", "period"]).agg(num=("num", "sum"), den=("den", "sum"))
    out = (tot.num / tot.den).unstack("period")
    b = x.groupby(["day", "period", "bucket"]).agg(num=("num", "sum"), den=("den", "sum"), n=("num", "size"))
    w = b.xs("wmr", level="period").n.rename("w")
    nm = b.xs("normal", level="period")
    nm = nm[nm.n >= 5].join(w, how="inner")
    m = nm.assign(a=nm.w * nm.num / nm.n, c=nm.w * nm.den / nm.n).groupby("day")[["a", "c"]].sum()
    out["normal_matched"] = m.a / m.c
    return out


def paired(tab: pd.DataFrame, metric: str, n_boot: int = 4000) -> pd.DataFrame:
    rows = []
    for ctrl in CONTROLS:
        if ctrl not in tab or "wmr" not in tab:
            continue
        d = (tab.wmr - tab[ctrl]).dropna().to_numpy()
        if len(d) < 10:
            continue
        boot = RNG.choice(d, (n_boot, len(d))).mean(1)
        flips = RNG.choice([-1.0, 1.0], (20000, len(d)))
        null = np.abs((flips * d).mean(1))
        rows.append({"metric": metric, "control": ctrl, "wmr": tab.wmr.mean(), "control_mean": tab[ctrl].mean(),
                     "diff": d.mean(), "ci_lo": np.quantile(boot, .025), "ci_hi": np.quantile(boot, .975),
                     "p_signflip": (1 + (null >= abs(d.mean())).sum()) / (1 + len(null)),
                     "share_days_positive": (d > 0).mean(), "n_days": len(d)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int)
    args = ap.parse_args()
    tick = tick_size(PROJECT)
    for sub in ("cache", "tables", "figures"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    paths = []
    for day in pd.date_range("2026-07-14", "2026-10-13", freq="D"):
        sym = "6EU6" if day < pd.Timestamp("2026-09-11") else "6EZ6"
        p = PROJECT / "data_6e" / "l3" / f"{sym}_mbo_{day.date()}.parquet"
        if p.is_file() and pq.ParquetFile(p).metadata.num_rows:
            paths.append(p)
    paths = paths[:args.limit_days] if args.limit_days else paths

    parts = {k: [] for k in ("fills", "fades", "base", "windows")}
    for p in paths:
        cached = {k: CACHE / f"{p.stem}_{k}.parquet" for k in parts}
        if all(c.exists() for c in cached.values()):
            res = {k: pd.read_parquet(c) for k, c in cached.items()}
        else:
            res = process_day(p, tick)
            for k, frame in res.items():
                frame.to_parquet(cached[k], index=False)
        for k in parts:
            parts[k].append(res[k])
        print(p.name, {k: len(v) for k, v in res.items()}, flush=True)
    fills, fades, base, windows = (pd.concat(parts[k], ignore_index=True) for k in ("fills", "fades", "base", "windows"))

    # A: activity mix from the existing 1-Hz cache.
    smp = pd.concat([pd.read_parquet(EDA_CACHE / f"{p.stem}_samples.parquet",
                                     columns=["utc_ns", "london_day", "top1_adds", "top1_cancels",
                                              "top1_executions", "trade_contracts"]) for p in paths], ignore_index=True)
    smp = smp.sort_values("utc_ns").reset_index(drop=True)
    _, _, clock, _ = local_parts(smp.utc_ns.to_numpy(np.int64))
    smp["period"], smp["day"] = period_of(clock), smp.london_day
    t = smp.utc_ns.to_numpy(np.int64)
    smp["act60"] = trailing(t, smp.trade_contracts.to_numpy(float), t + NS, 60 * NS)

    edges = np.quantile(smp.loc[smp.period == "wmr", "act60"], [.2, .4, .6, .8])
    smp, fills, fades, base = (bucket(x, edges) for x in (smp, fills, fades, base))
    results, tables = [], {}

    def run(name, df, num, den=None):
        tab = day_table(df, num, den)
        tables[name] = tab
        results.append(paired(tab, name))
        return tab

    run("A_top1_exec_per_add", smp, smp.top1_executions, smp.top1_adds)
    run("A_top1_cancel_per_add", smp, smp.top1_cancels, smp.top1_adds)
    for h in HORIZONS:
        run(f"B_markout_{h}s", fills, fills["size"] * fills[f"mk_{h}"], fills["size"])
    run("B_half_spread", fills, fills["size"] * fills.half_spread, fills["size"])
    run("B_adverse_60s", fills, fills["size"] * (fills.mk_60 - fills.half_spread), fills["size"])
    for ms in FADE_MS:
        ft = run(f"E_fade_after_trade_{ms}ms", fades, fades[f"fade_{ms}"])
        bt = run(f"E_fade_baseline_{ms}ms", base, base[f"fade_{ms}"])
        ex = ft - bt
        tables[f"E_fade_excess_{ms}ms"] = ex
        results.append(paired(ex, f"E_fade_excess_{ms}ms"))
    against = fills.flow_vs_maker < 0
    for h in (5, 60):
        a = run(f"F_pnl_{h}s_flow_against", fills[against], fills["size"][against] * fills[f"mk_{h}"][against], fills["size"][against])
        na = run(f"F_pnl_{h}s_flow_not_against", fills[~against], fills["size"][~against] * fills[f"mk_{h}"][~against], fills["size"][~against])
        tables[f"F_gap_{h}s"] = na - a
        results.append(paired(na - a, f"F_gap_{h}s (not-against minus against)"))
    res = pd.concat(results, ignore_index=True)

    # C: reversal; D: flow persistence. Fix vs every placebo window start.
    w = windows.dropna(subset=["r_in", "r_post"]).copy()
    w["z2"] = np.where(w.sumsq > 0, w.net ** 2 / w.sumsq, np.nan)
    per_start = w.groupby("start").apply(lambda g: pd.Series({
        "corr_in_post": g.r_in.corr(g.r_post), "corr_pre_in": g.r_pre.corr(g.r_in),
        "slope_post_on_in": np.polyfit(g.r_in, g.r_post, 1)[0] if len(g) > 5 else np.nan,
        "mean_z2": g.z2.mean(), "median_z2": g.z2.median(), "mean_volume": g.volume.mean(), "n_days": len(g)}))
    per_start["clock"] = pd.to_datetime(per_start.index, unit="s").strftime("%H:%M:%S")
    fix = per_start.loc[57450]
    plac = per_start.drop(index=[57150, 57450, 57750], errors="ignore")
    cd_rows = []
    for col, tail in (("corr_in_post", "low"), ("corr_pre_in", "low"), ("slope_post_on_in", "low"),
                      ("mean_z2", "high"), ("median_z2", "high")):
        v = plac[col].dropna()
        rank_p = ((v <= fix[col]).mean() if tail == "low" else (v >= fix[col]).mean())
        cd_rows.append({"metric": col, "fix": fix[col], "placebo_median": v.median(),
                        "placebo_p05": v.quantile(.05), "placebo_p95": v.quantile(.95),
                        "placebo_rank_p": rank_p, "n_placebo": len(v)})
    # Day-level permutation for the fix correlation and a busy-window z2 comparison.
    fw = w[w.start == 57450]
    null = np.array([np.corrcoef(fw.r_in, RNG.permutation(fw.r_post.to_numpy()))[0, 1] for _ in range(20000)])
    obs = fw.r_in.corr(fw.r_post)
    cd_rows.append({"metric": "corr_in_post_permutation_p", "fix": obs, "placebo_rank_p": (np.abs(null) >= abs(obs)).mean()})
    vol_cut = fw.volume.quantile(.25)
    busy = w[(~w.start.isin([57150, 57450, 57750])) & (w.volume >= vol_cut)]
    zd = fw.set_index("day").z2 - busy.groupby("day").z2.mean()
    zd = zd.dropna().to_numpy()
    flips = RNG.choice([-1.0, 1.0], (20000, len(zd)))
    cd_rows.append({"metric": "z2_fix_minus_busy_placebo (paired by day)", "fix": zd.mean(),
                    "placebo_rank_p": (np.abs((flips * zd).mean(1)) >= abs(zd.mean())).mean(),
                    "n_placebo": len(busy)})
    cd = pd.DataFrame(cd_rows)

    res.to_csv(OUT / "tables" / "paired_tests.csv", index=False)
    cd.to_csv(OUT / "tables" / "window_tests.csv", index=False)
    per_start.to_csv(OUT / "tables" / "window_by_start.csv")
    w[w.start.isin([57150, 57450, 57750])].to_csv(OUT / "tables" / "fix_window_days.csv", index=False)
    for name, tab in tables.items():
        tab.to_csv(OUT / "tables" / f"day_{name}.csv")

    # Figures.
    colors = {"wmr": "#b93c32", "nearby": "#e29b41", "us_event": "#8a5fb0", "normal": "#527ea6", "normal_matched": "#2f8f6b"}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for c in ["wmr", "nearby", "us_event", "normal", "normal_matched"]:
        m = [tables[f"B_markout_{h}s"][c].mean() for h in HORIZONS]
        e = [1.96 * tables[f"B_markout_{h}s"][c].std() / np.sqrt(tables[f"B_markout_{h}s"][c].count()) for h in HORIZONS]
        ax.errorbar(range(len(HORIZONS)), m, yerr=e, marker="o", capsize=3, label=c, color=colors[c])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(HORIZONS)), [f"{h}s" for h in HORIZONS])
    ax.set(xlabel="Horizon after passive fill", ylabel="Maker P&L, ticks per contract (day mean ±95%)",
           title="Passive-fill markouts by period")
    ax.legend()
    fig.savefig(OUT / "figures" / "markouts.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, col, lab in ((axes[0], "corr_in_post", "corr(in-window return, next-10-min return)"),
                         (axes[1], "mean_z2", "mean z² of aggressive signed volume (1 = random signs)")):
        ax.plot(per_start.index / 3600, per_start[col], lw=.8, color="#527ea6")
        for s0, c in ((57450, "wmr"), (57150, "nearby"), (57750, "nearby"), (48450, "us_event"), (52050, "us_event"), (53850, "us_event")):
            if s0 in per_start.index:
                ax.scatter(s0 / 3600, per_start.loc[s0, col], color=colors[c], zorder=3, s=30)
        ax.set(xlabel="Window start, London clock (hours)", title=lab)
    axes[0].axhline(0, color="k", lw=.8)
    axes[1].axhline(1, color="k", lw=.8)
    fig.savefig(OUT / "figures" / "window_placebos.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for n, c in enumerate(["wmr", "nearby", "us_event", "normal", "normal_matched"]):
        for j, ms in enumerate(FADE_MS):
            tab = tables[f"E_fade_excess_{ms}ms"][c].dropna()
            ax.bar(j + n * .16 - .32, tab.mean(), .15, yerr=1.96 * tab.std() / np.sqrt(len(tab)),
                   color=colors[c], label=c if j == 0 else None, capsize=2)
    ax.set_xticks(range(len(FADE_MS)), [f"{ms} ms" for ms in FADE_MS])
    ax.set(ylabel="Excess share of hit-side depth cancelled", title="Liquidity fade after a trade (minus unconditional)")
    ax.legend()
    fig.savefig(OUT / "figures" / "fade.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(res.round(4).to_string())
    print(cd.round(4).to_string())
    print("activity bucket edges (trailing 60s contracts):", edges)
    print("fills by period:", fills.groupby("period").size().to_dict())


if __name__ == "__main__":
    main()
