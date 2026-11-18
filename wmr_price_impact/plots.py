"""Figures for the price-impact report; reads only the tables written by analyze_results.py.
Run from project/: .venv/Scripts/python.exe -m wmr_price_impact.plots
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGURES, TABLES, VOL_H, _r  # noqa: E402

C = {"wmr": "#c0392b", "A": "#d98a1c", "B": "#2a6fb5", "green": "#2e9461", "ink": "#222222", "muted": "#777777"}
NAME = {"A": "nearby windows", "B": "normal hours"}
XV = ["0+", *[str(v) for v in VOL_H]]
VCOLS = ["I0", *[f"I_v{v}" for v in VOL_H]]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "axes.labelcolor": C["ink"], "legend.frameon": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": .6})


def eff(sample="test", comp="B", clock="volume", side="both", K=5, cal=None):
    e = pd.read_csv(TABLES / "effects.csv")
    cal = cal or json.loads((TABLES / "design.json").read_text())["caliper"]
    return e[(e["sample"] == sample) & (e.comparison == comp) & (e.clock == clock) & (e.side == side) & (e.K == K)
             & np.isclose(e.caliper, cal)].set_index("col")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_balance():
    b = pd.read_csv(TABLES / "balance_smd.csv")
    d = pd.read_csv(TABLES / "match_diagnostics.csv").query("sample == 'test' and K == 5")
    cal = json.loads((TABLES / "design.json").read_text())["caliper"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, comp in zip(axes, "AB"):
        s = b[(b["sample"] == "test") & (b.comparison == comp) & (b.K == 5) & np.isclose(b.caliper, cal)].iloc[::-1]
        y = np.arange(len(s))
        ax.axvspan(-.1, .1, color="#eeeeee", zorder=0)
        ax.hlines(y, s.smd_before, s.smd_after, color="#bbbbbb", lw=1)
        ax.plot(s.smd_before, y, "o", mfc="white", mec=C["muted"], ms=5, label="before matching")
        ax.plot(s.smd_after, y, "o", color=C[comp], ms=5, label="after matching")
        ax.set_yticks(y, s.feature)
        frac = d[(d.comparison == comp) & np.isclose(d.caliper, cal)].matched_frac.iloc[0]
        ax.set_title(f"WMR vs {NAME[comp]}\n{frac:.0%} of WMR test-day trades matched")
        ax.set_xlabel("standardized mean difference (WMR - control)")
        ax.axvline(0, color=C["ink"], lw=.6)
    axes[0].legend(loc="lower right", fontsize=8)
    save(fig, "fig1_balance.png")


def band(ax, x, r, prefix, color, label, marker="o", dx=0.0):
    ax.fill_between(x + dx, r[f"{prefix}_lo"], r[f"{prefix}_hi"], color=color, alpha=.15, lw=0)
    ax.plot(x + dx, r[prefix], marker=marker, color=color, lw=2, ms=5, label=label)


def fig2_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = np.arange(len(VCOLS))
    for ax, comp in zip(axes, "AB"):
        r = eff(comp=comp).loc[VCOLS]
        band(ax, x, r, "wmr", C["wmr"], "WMR (matched trades)")
        band(ax, x, r, "ctrl", C[comp], f"matched {NAME[comp]}", marker="s")
        ax.set_xticks(x, XV)
        ax.set_xlabel("contracts traded after the event (volume clock)")
        ax.set_title(f"vs {NAME[comp]} (test days)")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("signed price response I(v), ticks")
    save(fig, "fig2_response_curves.png")


def diff_panel(ax, x, r, color, label, marker="o", dx=0.0, hollow=False):
    ax.errorbar(x + dx, r["diff"], yerr=[r["diff"] - r.lo, r.hi - r["diff"]], fmt=marker, color=color, ms=5,
                mfc="white" if hollow else color, capsize=2, lw=1.2, label=label)


def fig3_diffs():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = np.arange(len(VCOLS))
    for ax, comp in zip(axes, "AB"):
        ax.axhline(0, color=C["ink"], lw=.8)
        diff_panel(ax, x, eff(comp=comp).loc[VCOLS], C[comp], "test days (primary)", dx=-.1)
        diff_panel(ax, x, eff(sample="all", comp=comp).loc[VCOLS], C[comp], "all 66 days", "s", .1, True)
        ax.set_xticks(x, XV)
        ax.set_xlabel("contracts traded after the event (volume clock)")
        ax.set_title(f"WMR - {NAME[comp]}")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("matched difference ΔI(v), ticks")
    save(fig, "fig3_impact_difference.png")


def fig4_decomposition():
    items = [("I0", "I(0+)\nimmediate"), ("R_v25", "R(25)\nlater"), ("R_v100", "R(100)\nlater"),
             ("I_v100", "I(100)\ncumulative")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = np.arange(len(items))
    for ax, comp in zip(axes, "AB"):
        r = eff(comp=comp).loc[[c for c, _ in items]]
        for k, (who, col, dx) in enumerate([("wmr", C["wmr"], -.18), ("ctrl", C[comp], .18)]):
            ax.bar(x + dx, r[who], .34, color=col, label="WMR" if who == "wmr" else f"matched {NAME[comp]}")
            ax.errorbar(x + dx, r[who], yerr=[r[who] - r[f"{who}_lo"], r[f"{who}_hi"] - r[who]], fmt="none",
                        color=C["ink"], lw=1, capsize=2)
        ax.axhline(0, color=C["ink"], lw=.8)
        ax.set_xticks(x, [l for _, l in items])
        ax.set_title(f"vs {NAME[comp]} (test days)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("ticks (in the trade's direction)")
    save(fig, "fig4_decomposition.png")


def fig5_sides():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = np.arange(len(VCOLS))
    for ax, sample in zip(axes, ("test", "all")):
        ax.axhline(0, color=C["ink"], lw=.8)
        for side, m, dx, col in (("buy", "^", -.1, C["B"]), ("sell", "v", .1, C["ink"])):
            r = eff(sample=sample, comp="B", side=side).loc[VCOLS]
            diff_panel(ax, x, r, col, f"aggressive {side}s (n={int(r.n_trades.iloc[0]):,})", m, dx)
        ax.set_xticks(x, XV)
        ax.set_xlabel("contracts traded after the event (volume clock)")
        ax.set_title("WMR - normal hours, " + ("test days" if sample == "test" else "all 66 days"))
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("matched difference ΔI(v), ticks")
    save(fig, "fig5_sides.png")


def fig6_bridge():
    b = pd.read_csv(TABLES / "bridge.csv")
    cols = ["I0", "I_w5", *[f"I_v{v}" for v in VOL_H]]
    labels = ["0+", "5 s", *[f"{v} c" for v in VOL_H]]
    q = {"(1) all aggressive trades, KNN-matched": ("(1) all trades, matched", C["B"], "o"),
         "(2) trades that filled a maker order, KNN-matched": ("(2) maker-linked trades, matched", C["green"], "s"),
         "(4) historical maker markout of those fills": ("(4) historical maker markout", C["wmr"], "D")}
    panels = [("test", "both", "(a) test days, both sides"), ("all", "both", "(b) all 66 days, both sides"),
              ("all", "buy", "(c) all days, aggressive buys"), ("all", "sell", "(d) all days, aggressive sells")]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharey=True)
    x = np.arange(len(cols))
    for ax, (sample, side, title) in zip(axes.ravel(), panels):
        ax.axhline(0, color=C["ink"], lw=.8)
        for k, (label, (name, col, m)) in enumerate(q.items()):
            if sample == "test" and name.startswith("(2)"):   # 157 matched trades on 13 days: uninformative
                continue
            r = b[(b["sample"] == sample) & (b.side == side) & (b.quantity == name)].set_index("col").loc[cols]
            diff_panel(ax, x, r, col, label, m, (k - 1) * .16)
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("horizon (s = wall seconds, c = contracts)")
    for ax in axes[:, 0]:
        ax.set_ylabel("WMR - normal, maker view (ticks)")
    axes[0, 1].legend(fontsize=7.5, loc="upper right")
    ax = axes[0, 0]
    ax.set_ylim(-0.25, 0.45)
    save(fig, "fig6_bridge.png")


def fig7_clocks():
    per_s = _r["volume"]                                   # contracts per normal-hours second (training days)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, comp in zip(axes, "AB"):
        ax.axhline(0, color=C["ink"], lw=.8)
        for clock, cols, xs, col, m in (
                ("volume", [f"I_v{v}" for v in VOL_H], np.array(VOL_H) / per_s, C[comp], "o"),
                ("event", [f"I_e{v}" for v in VOL_H], np.array(VOL_H) / per_s * 1.04, C["green"], "s"),
                ("wall", ["I_w1", "I_w5", "I_w10"], np.array([1, 5, 10]) * 0.96, C["ink"], "^")):
            r = eff(comp=comp, clock=clock).loc[cols]
            diff_panel(ax, xs, r, col, f"{clock} clock", m)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"])
        ax.minorticks_off()
        ax.set_xlabel("horizon, normal-hours-equivalent seconds")
        ax.set_title(f"WMR - {NAME[comp]} (test days)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("matched difference ΔI, ticks")
    save(fig, "fig7_clocks.png")


def main():
    for f in (fig1_balance, fig2_curves, fig3_diffs, fig4_decomposition, fig5_sides, fig6_bridge, fig7_clocks):
        f()


if __name__ == "__main__":
    main()
