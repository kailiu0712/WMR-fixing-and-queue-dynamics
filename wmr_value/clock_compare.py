"""Collect the headline numbers of the 5 s (time), volume- and event-clock runs side by side.

Reads the tables written by run.py / robustness.py (time) and clock_run.py (volume, event).
Writes outputs/wmr_value/tables/clock_comparison.csv.
Run from project/: .venv/Scripts/python.exe -m wmr_value.clock_compare
"""
import pandas as pd

from .config import OUT, TABLES

CLOCKS = ["time", "volume", "event"]


def tab(k: str, name: str) -> pd.DataFrame:
    return pd.read_csv((TABLES if k == "time" else OUT / f"clock_{k}" / "tables") / f"{name}.csv")


def rows(k: str) -> list[dict]:
    out = []

    def add(section, item, est, lo=None, hi=None, p=None):
        out.append({"clock": k, "section": section, "item": item, "est": est, "lo": lo, "hi": hi, "p": p})

    d = tab(k, "A1_descriptive").query("side == 'both'").set_index("period")
    wall = None if k == "time" else tab(k, "clock_sample").set_index("period")
    for per in ("wmr", "nearby", "normal"):
        r = d.loc[per]
        add("level", f"{per} fill rate", r.fill_rate)
        add("level", f"{per} fills", r.n_fills)
        for c in ("spread_capture_mean", "move5_mean", "mk5_mean", "mk5_neg_share", "Z5_mean", "fill_time_median"):
            add("level", f"{per} {c}", r[c])
        add("level", f"{per} label horizon, median wall s", 5.0 if wall is None else wall.loc[per, "wall_label_median"])
        add("level", f"{per} markout horizon, median wall s", 5.0 if wall is None else wall.loc[per, "wall_mk5_median"])

    t = tab(k, "A1_tests").query("sample == 'both'")
    for m in ("fill_rate", "spread_capture_given_fill", "move_5s_given_fill", "markout_5s_given_fill",
              "neg_markout_5s_share_of_fills", "Z_5s_per_opportunity", "markout_30s_given_fill"):
        for ctrl in ("normal", "nearby"):
            r = t[(t.metric == m) & (t.control == ctrl)].iloc[0]
            add("A1 day-paired WMR - control", f"{m} vs {ctrl}", r["mean"], r.ci_lo, r.ci_hi, r.p)

    t = tab(k, "A2_matched")
    for s in ("queue x imbalance x flow x spread x side", "walk-forward M0 fill-probability decile x side"):
        for m in ("fill_rate", "markout_5s_given_fill", "Z_5s_per_opportunity"):
            for ctrl in ("normal", "nearby"):
                q = t[(t.strata == s) & (t.metric == m) & (t.control == ctrl)]
                if len(q):
                    r = q.iloc[0]
                    add("A2 matched WMR - control", f"{s} | {m} vs {ctrl} (cov {r.coverage:.2f})",
                        r["mean"], r.ci_lo, r.ci_hi, r.p)

    f = tab(k, "B_fill_metrics").set_index(["group", "model"])
    for g in ("wmr", "normal"):
        for m in ("M0", "M1"):
            add("B fill model", f"{g} {m} realized - predicted fill", f.loc[(g, m), "prevalence"] - f.loc[(g, m), "mean_pred"])
            add("B fill model", f"{g} {m} AUC", f.loc[(g, m), "auc"])
    b = tab(k, "B_day_tests")
    r = b[(b.period == "wmr") & (b.metric == "brier_M0_minus_M1")].iloc[0]
    add("B fill model", "wmr Brier M0 - M1 (test days)", r["mean"], r.ci_lo, r.ci_hi, r.p)
    mk = tab(k, "B_markout_metrics").query("h == 5 and model == 'R0'").set_index("group")
    for g in ("wmr", "normal"):
        add("B markout model", f"{g} R0 corr(pred, realized)", mk.loc[g, "corr"])

    w = tab(k, "R1_wmr_day_tests").query("eval == 'walk_forward' and model == 'wmr_only'")
    for m in ("brier_M1_minus_m", "corr5_m_minus_R1"):
        r = w[w.metric == m].iloc[0]
        add("R1 WMR-only vs pooled M1 (walk-forward)", m, r["mean"], r.ci_lo, r.ci_hi, r.p)

    e = tab(k, "R2_equal_n")
    for _, r in e[e.metric.isin(["mk_corr", "decile_spearman", "auc"])].iterrows():
        add("R2 equal size", f"{r.metric} {r.model} {r['eval']}: WMR {r.wmr:.3f} vs normal median {r.normal_eqn_median:.3f}",
            r.share_draws_below_wmr)

    c = tab(k, "C_paired").query("eval == 'test_days_within_period_budget' and frac == 0.2 and cost == 0")
    for g in ("wmr", "normal"):
        for cmp in ("P3_joint - P0_all", "P3_joint - P1_fill"):
            r = c[(c.group == g) & (c.comparison == cmp)].iloc[0]
            add("C policy (top 20%, within-period budget)", f"{g} {cmp}", r["mean"], r.ci_lo, r.ci_hi, r.p)
    return out


def main() -> None:
    long = pd.DataFrame([r for k in CLOCKS for r in rows(k)])
    long.to_csv(TABLES / "clock_comparison.csv", index=False)
    wide = long.pivot_table(index=["section", "item"], columns="clock", values=["est", "p"], sort=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 95):
        print(wide.round(4).to_string())


if __name__ == "__main__":
    main()
