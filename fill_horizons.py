"""Fill-rate horizon variants and exit decomposition for the WMR EDA candidates.

Reuses the candidate orders cached by wmr_eda.py (selection, features, period)
and looks up each order's later F/C/M events in data_6e/l3; no book replay.
Run from the project directory: python fill_horizons.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NS = 1_000_000_000
HORIZONS = [1, 2, 5, 10, 30]
OUTCOMES = ["fill", "cancel", "modify", "resting_at_touch", "resting_behind"]
PERIODS = ["wmr", "nearby", "normal"]
QA_BINS = [-1, 0, 10, 30, np.inf]
QA_LABELS = ["0", "1-10", "11-30", "31+"]
OUT = Path("outputs/wmr_eda")
CAND_COLS = ["order_id", "entry_ns", "side", "queue_ahead", "label", "london_day", "period"]


def day_outcomes(cand_path: Path, l3_path: Path) -> pd.DataFrame:
    cand = pd.read_parquet(cand_path, columns=CAND_COLS)
    oids = cand.order_id.astype("uint64").unique().tolist()
    ev = pq.read_table(l3_path, columns=["ts_recv", "action", "order_id", "price"],
                       filters=[("order_id", "in", oids)]).to_pandas()
    ev["ts"] = ev.ts_recv.astype("int64")
    ev["action"] = ev.action.astype(str)
    ev["order_id"] = ev.order_id.astype("int64")
    m = cand[["order_id", "entry_ns"]].merge(ev, on="order_id")
    # The candidate is created after its add packet ends, so only later records count.
    after = m[(m.ts >= m.entry_ns) & m.action.isin(["F", "C", "M"])]
    first = after.groupby(["order_id", "entry_ns", "action"]).ts.min().unstack()
    first = first.reindex(columns=["F", "C", "M"])
    price = m[(m.action == "A") & (m.ts <= m.entry_ns)].groupby(["order_id", "entry_ns"]).price.last()
    cand = cand.join(first, on=["order_id", "entry_ns"]).join(price, on=["order_id", "entry_ns"])
    for a in ["F", "C", "M"]:
        cand[a] = (cand[a] - cand.entry_ns) / NS
    cand["exit_s"] = cand[["F", "C", "M"]].min(axis=1)
    cand["exit_kind"] = cand[["F", "C", "M"]].fillna(np.inf).idxmin(axis=1).map({"F": "fill", "C": "cancel", "M": "modify"})

    book = pq.read_table(l3_path, columns=["ts_recv", "is_last", "bid_px", "ask_px"]).to_pandas()
    book = book[book.is_last]
    bt = book.ts_recv.astype("int64").to_numpy()
    last_s = (bt[-1] - cand.entry_ns) / NS
    rows = []
    for h in HORIZONS:
        out = np.where(cand.exit_s <= h, cand.exit_kind, "resting")
        i = np.searchsorted(bt, cand.entry_ns + h * NS, side="right") - 1
        touch = np.where(cand.side == "B", book.bid_px.to_numpy()[i], book.ask_px.to_numpy()[i])
        out = np.where(out == "resting",
                       np.where(np.isclose(touch, cand.price), "resting_at_touch", "resting_behind"), out)
        # Orders still resting when the UTC file ends cannot be labelled at this horizon.
        keep = (cand.exit_s <= h) | (last_s >= h)
        rows.append(cand.loc[keep].assign(horizon=h, outcome=out[keep.to_numpy()]))
    return pd.concat(rows)


def main() -> None:
    days = []
    for cand_path in sorted((OUT / "daily_cache_v1").glob("*_candidates.parquet")):
        l3_path = Path("data_6e/l3") / cand_path.name.replace("_candidates", "")
        if pq.read_metadata(cand_path).num_rows == 0:
            continue
        days.append(day_outcomes(cand_path, l3_path))
        print("done", l3_path.name)
    df = pd.concat(days, ignore_index=True)
    df = df[df.period.isin(PERIODS)]

    # Self-check: the rebuilt 5 s label must reproduce the cached wmr_eda.py label.
    five = df[df.horizon == 5]
    agree = ((five.outcome == "fill") == (five.label == 1)).mean()
    print(f"5 s label agreement with cache: {agree:.5%} of {len(five):,}")
    assert agree > 0.999, agree

    tables = OUT / "tables"
    shares = (df.groupby(["period", "horizon"]).outcome.value_counts(normalize=True)
              .unstack().reindex(columns=OUTCOMES).fillna(0))
    shares["n"] = df.groupby(["period", "horizon"]).size()
    daily = (df.assign(fill=df.outcome == "fill").groupby(["period", "horizon", "london_day"]).fill.mean()
             .groupby(["period", "horizon"]).agg(fill_day_mean="mean",
                                                 fill_day_p10=lambda x: x.quantile(0.1),
                                                 fill_day_p90=lambda x: x.quantile(0.9)))
    shares = shares.join(daily)
    shares.round(4).to_csv(tables / "horizon_outcomes.csv")

    df["qa_bucket"] = pd.cut(df.queue_ahead, QA_BINS, labels=QA_LABELS)
    by_q = (df.assign(fill=df.outcome == "fill")
            .groupby(["horizon", "qa_bucket", "period"], observed=True).fill.agg(["mean", "size"])
            .unstack("period"))
    by_q.round(4).to_csv(tables / "horizon_fill_by_queue.csv")

    c = df[df.horizon == 30]
    lat = c.groupby("period").apply(lambda g: pd.Series({
        # Only cancels that end the episode; a C after a full fill is the fill's cleanup.
        **{f"cancel_within_{t}s": ((g.exit_kind == "cancel") & (g.exit_s <= t)).mean()
           for t in (0.01, 0.1, 0.5, 1, 5)},
        "median_cancel_s": g.loc[g.outcome == "cancel", "C"].median(),
        "median_fill_s": g.loc[g.outcome == "fill", "F"].median(),
        "n": len(g)}), include_groups=False)
    lat.round(4).to_csv(tables / "cancel_latency.csv")

    print(shares.round(3).to_string())
    print(by_q.round(3).to_string())
    print(lat.round(3).to_string())

    colors = {"fill": "#2f6f4e", "cancel": "#b93c32", "modify": "#e29b41",
              "resting_at_touch": "#527ea6", "resting_behind": "#a9bfd6"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(HORIZONS))
    for j, p in enumerate(PERIODS):
        bottom = np.zeros(len(HORIZONS))
        for o in OUTCOMES:
            v = shares.loc[p].reindex(HORIZONS)[o].to_numpy()
            axes[0].bar(x + (j - 1) * 0.27, v, 0.25, bottom=bottom, color=colors[o],
                        label=o if j == 0 else None, edgecolor="white", linewidth=0.5)
            bottom += v
    axes[0].set_xticks(x, [f"{h}s\nW  N  O" for h in HORIZONS])
    axes[0].set_ylabel("share of best-quote additions")
    axes[0].set_title("Outcome by horizon (W=wmr, N=nearby, O=normal)")
    axes[0].set_ylim(0, 1.12)
    axes[0].legend(fontsize=8, loc="upper center", ncol=5)
    for p, col in zip(PERIODS, ["#b93c32", "#e29b41", "#527ea6"]):
        s = shares.loc[p].reindex(HORIZONS)
        axes[1].plot(HORIZONS, s.fill, "o-", color=col, label=f"{p} (n={int(s.n.iloc[0]):,})")
        axes[1].fill_between(HORIZONS, s.fill_day_p10, s.fill_day_p90, color=col, alpha=0.12)
    axes[1].set_xscale("log")
    axes[1].set_xticks(HORIZONS, [str(h) for h in HORIZONS])
    axes[1].set_xlabel("horizon (s)")
    axes[1].set_ylabel("fill before exit")
    axes[1].set_title("Fill rate by horizon; bands = 10th-90th day percentile")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "horizon_outcomes.png", dpi=150)


if __name__ == "__main__":
    main()
