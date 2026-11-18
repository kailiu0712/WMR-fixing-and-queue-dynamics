"""Compare the saved 22-fixing-day baseline with the expanded 6E EDA.

Run from project after wmr_eda.py has finished. Keeps the previous period
comparison in outputs/wmr_eda_22day_baseline/period_comparison.csv.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "wmr_eda"
BASE = ROOT / "outputs" / "wmr_eda_22day_baseline"
METRICS = ["spread_ticks", "depth1", "depth5", "adds", "cancels",
           "executions", "abs_imbalance", "best_order_count"]


def rate_table(end_date: str | None) -> pd.DataFrame:
    parts = []
    for path in sorted((OUT / "daily_cache_v1").glob("*_candidates.parquet")):
        if pq.ParquetFile(path).metadata.num_rows == 0:
            continue
        day = path.stem.split("_mbo_")[-1][:10]
        if end_date is not None and day > end_date:
            continue
        frame = pd.read_parquet(path, columns=["period", "label"])
        if not frame.empty:
            parts.append(frame.dropna(subset=["label"]))
    if not parts:
        return pd.DataFrame(columns=["period", "rate", "orders"])
    return pd.concat(parts).groupby("period").label.agg(rate="mean", orders="size").reset_index()


def main() -> None:
    old = pd.read_csv(BASE / "period_comparison.csv").set_index(["period", "metric"])
    new = pd.read_csv(OUT / "tables" / "period_comparison.csv").set_index(["period", "metric"])
    rows = []
    for metric in METRICS:
        for period in ("wmr", "nearby"):
            a = float(old.loc[(period, metric), "day_equal_mean"])
            b = float(new.loc[(period, metric), "day_equal_mean"])
            rows.append(dict(metric=metric, period=period, baseline_22d=a,
                             expanded=b, change_pct=100 * (b / a - 1),
                             baseline_days=int(old.loc[(period, metric), "n_days"]),
                             expanded_days=int(new.loc[(period, metric), "n_days"])))
    changes = pd.DataFrame(rows)
    changes.to_csv(OUT / "tables" / "comparison_22_vs_expanded.csv", index=False)
    new_rates = rate_table(None)
    # The baseline is the September 14-October 13 slice of the shared cache.
    baseline_parts = []
    for path in sorted((OUT / "daily_cache_v1").glob("6EZ6_mbo_*_candidates.parquet")):
        if pq.ParquetFile(path).metadata.num_rows == 0:
            continue
        day = path.stem.split("_mbo_")[-1][:10]
        if "2026-09-14" <= day <= "2026-10-13":
            f = pd.read_parquet(path, columns=["period", "label"])
            if not f.empty:
                baseline_parts.append(f.dropna(subset=["label"]))
    old_rates = pd.concat(baseline_parts).groupby("period").label.agg(rate="mean", orders="size").reset_index()
    rates = old_rates.merge(new_rates, on="period", suffixes=("_22d", "_expanded"))
    rates["change_pp"] = 100 * (rates.rate_expanded - rates.rate_22d)
    rates.to_csv(OUT / "tables" / "fill_rate_comparison.csv", index=False)
    before = pd.read_csv(BASE / "diagnostics.csv")
    after = pd.read_csv(OUT / "tables" / "diagnostics.csv")
    def val(df, col):
        return int(df[col].fillna(0).sum()) if col in df else 0
    lines = ["# 22-day versus expanded 6E WMR EDA", "",
             "The expanded UTC series covers 2026-07-14 through 2026-10-13. "
             "It uses 6EU6 through 2026-09-10 and 6EZ6 from 2026-09-11. "
             "The 22-day baseline covers 2026-09-14 through 2026-10-13.", "",
             f"Processed UTC files: {len(before)} to {len(after)}; "
             f"MBO rows: {val(before, 'rows'):,} to {val(after, 'rows'):,}; "
             f"WMR fixing days: {int(old.loc[('wmr', 'executions'), 'n_days'])} to "
             f"{int(new.loc[('wmr', 'executions'), 'n_days'])}.", "",
             "## Main changes", "",
             f"Absolute WMR depth and activity shifted after adding July–August: top-five "
             f"depth went from {old.loc[('wmr','depth5'),'day_equal_mean']:,.0f} to "
             f"{new.loc[('wmr','depth5'),'day_equal_mean']:,.0f} contracts, executed "
             f"volume from {old.loc[('wmr','executions'),'day_equal_mean']:.2f} to "
             f"{new.loc[('wmr','executions'),'day_equal_mean']:.2f} contracts/s, "
             f"additions from {old.loc[('wmr','adds'),'day_equal_mean']:.1f} to "
             f"{new.loc[('wmr','adds'),'day_equal_mean']:.1f} contracts/s, and "
             f"cancellations from {old.loc[('wmr','cancels'),'day_equal_mean']:.1f} to "
             f"{new.loc[('wmr','cancels'),'day_equal_mean']:.1f} contracts/s.", "",
             "Relative WMR-versus-nearby contrasts are listed below; execution, "
             "addition, and cancellation uplifts strengthened, while the top-five "
             "depth uplift eased. The spread contrast stayed small.", "",
             f"Observed five-second first-fill probability changed from "
             f"{rates.set_index('period').loc['wmr','rate_22d']:.1%} to "
             f"{rates.set_index('period').loc['wmr','rate_expanded']:.1%} in WMR and "
             f"from {rates.set_index('period').loc['nearby','rate_22d']:.1%} to "
             f"{rates.set_index('period').loc['nearby','rate_expanded']:.1%} nearby.", "",
             "## Day-equal means", "",
             "| Metric | Window | 22 days | Expanded | Change |", "|---|---|---:|---:|---:|"]
    for r in changes.itertuples():
        lines.append(f"| {r.metric} | {r.period} | {r.baseline_22d:.3f} | "
                     f"{r.expanded:.3f} | {r.change_pct:+.1f}% |")
    lines += ["", "## WMR relative to nearby", "",
              "| Metric | 22-day uplift | Expanded uplift |", "|---|---:|---:|"]
    for metric in METRICS:
        b = changes[changes.metric == metric].set_index("period")
        old_u = 100 * (b.loc["wmr", "baseline_22d"] / b.loc["nearby", "baseline_22d"] - 1)
        new_u = 100 * (b.loc["wmr", "expanded"] / b.loc["nearby", "expanded"] - 1)
        lines.append(f"| {metric} | {old_u:+.1f}% | {new_u:+.1f}% |")
    lines += ["", "## Observed five-second first-fill rates", "",
              "| Period | 22 days | Expanded | Change |", "|---|---:|---:|---:|"]
    for r in rates.itertuples():
        lines.append(f"| {r.period} | {r.rate_22d:.1%} ({r.orders_22d:,}) | "
                     f"{r.rate_expanded:.1%} ({r.orders_expanded:,}) | {r.change_pp:+.2f} pp |")
    lines += ["", "## Replay diagnostics", "",
              f"Unknown-order references: {val(before, 'unknown_order_events'):,} to "
              f"{val(after, 'unknown_order_events'):,}.",
              f"Trailing incomplete packet rows: {val(before, 'trailing_incomplete_rows'):,} to "
              f"{val(after, 'trailing_incomplete_rows'):,}.", "",
              "These comparisons are descriptive. The baseline is contained in the expanded sample; "
              "the change is a sample-composition change, not a paired treatment effect. "
              "Observed-order fill labels reflect owners' cancellations and repricings."]
    (OUT / "comparison_22_vs_expanded.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved comparison and fill-rate tables")


if __name__ == "__main__":
    main()
