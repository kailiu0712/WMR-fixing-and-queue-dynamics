"""Addendum PDF: the main results under volume and event clocks (complement to the 5 s clock).

Reads tables from the 5 s run (outputs/wmr_value/tables) and from clock_run.py
(outputs/wmr_value/clock_<k>/tables). Writes outputs/wmr_value/report_clocks.pdf.
Run from project/: .venv/Scripts/python.exe -m wmr_value.clock_pdf
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate

from .clock_compare import tab
from .config import COLORS, FIGURES, OUT, PRIMARY_H
from .report_pdf import INK, MUTED, W, fonts, styles, summary_box, table

CLK = ["time", "volume", "event"]
CNAME = {"time": "5 s clock", "volume": "Volume clock", "event": "Event clock"}
STRATA = {"book": "queue x imbalance x flow x spread x side", "pfill": "walk-forward M0 fill-probability decile x side"}


def a1(k, metric, ctrl="normal"):
    t = tab(k, "A1_tests").query("sample == 'both' and metric == @metric and control == @ctrl")
    return t.iloc[0]


def a2(k, strata, metric, ctrl="normal"):
    t = tab(k, "A2_matched").query("strata == @strata and metric == @metric and control == @ctrl")
    return t.iloc[0]


def desc(k):
    return tab(k, "A1_descriptive").query("side == 'both'").set_index("period")


def wall(k):
    if k == "time":
        return pd.DataFrame({"wall_label_median": 5.0, "wall_mk5_median": 5.0}, index=["wmr", "nearby", "normal"])
    return tab(k, "clock_sample").set_index("period")


def r2(k):
    return tab(k, "R2_equal_n")


def cpair(k, group, comparison, frac=0.2):
    t = tab(k, "C_paired").query("eval == 'test_days_within_period_budget' and cost == 0 and group == @group "
                                 "and frac == @frac and comparison == @comparison")
    return t.iloc[0]


def ci(r, scale=1.0, fmt="+.3f", p=True):
    s = f"{r['mean'] * scale:{fmt}} [{r.ci_lo * scale:{fmt[1:]}}, {r.ci_hi * scale:{fmt[1:]}}]"
    if p and r.p >= 0.001:
        s += f" (p = {r.p:.2f})" if r.p >= 0.01 else f" (p = {r.p:.3f})"
    return s.replace("-", "−")


def pv(p):
    return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}"


# ------------------------------------------------------------------------------------------ figure
def figure() -> str:
    plt.rcParams.update({"font.family": "Calibri", "font.size": 8, "axes.titlesize": 8.6, "axes.labelsize": 8,
                         "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#888888", "axes.linewidth": .6})
    fig, ax = plt.subplots(1, 4, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    xs = range(len(CLK))
    lab = ["5 s", "Volume", "Event"]

    a = ax[0]                                          # (a) wall-clock length of the fill horizon
    for j, per in enumerate(("wmr", "nearby", "normal")):
        y = [wall(k).loc[per, "wall_label_median"] for k in CLK]
        a.plot([x + (j - 1) * .17 for x in xs], y, "o", ms=5, color=COLORS[per], mec="white", mew=.6,
               label={"wmr": "WMR", "nearby": "Nearby", "normal": "Normal"}[per])
    a.set(xticks=list(xs), xticklabels=lab, ylim=(0, 10.5), ylabel="seconds (median)",
          title="(a) Wall time of the 5 s-equiv. horizon")
    a.legend(frameon=False, fontsize=7, loc="upper left", handletextpad=.2, borderaxespad=.1)

    for a, metric, scale, title, unit in ((ax[1], "fill_rate", 100, "(b) WMR − normal: fill rate", "points"),
                                          (ax[2], f"markout_{PRIMARY_H}s_given_fill", 1, "(c) WMR − normal: markout",
                                           "ticks per fill")):
        for x, k in zip(xs, CLK):
            r = a1(k, metric)
            m = a2(k, STRATA["book"], metric)
            for dx, rr, face in ((-.12, r, INK), (.12, m, "white")):
                a.errorbar(x + dx, rr["mean"] * scale, yerr=[[(rr["mean"] - rr.ci_lo) * scale], [(rr.ci_hi - rr["mean"]) * scale]],
                           fmt="o", ms=4.5, color=INK, mfc=face, mew=.9, elinewidth=.9, capsize=0)
        a.axhline(0, color="#999999", lw=.6)
        a.set(xticks=list(xs), xticklabels=lab, title=title, ylabel=unit)
        a.set_xlim(-.5, 2.5)
    ax[1].set_ylim(0, 8.2)
    ax[2].set_ylim(0, .145)
    ax[1].plot([], [], "o", color=INK, ms=4.5, label="day-paired")
    ax[1].plot([], [], "o", color=INK, mfc="white", ms=4.5, label="book-state matched")
    ax[1].legend(frameon=False, fontsize=7, loc="upper right", handletextpad=.2, borderaxespad=.1)

    a = ax[3]                                          # (d) markout correlation at equal size
    for x, k in zip(xs, CLK):
        for dx, ev in ((-.13, "test_days"), (.13, "walk_forward")):
            r = r2(k).query("model == 'M0' and metric == 'mk_corr' and eval == @ev").iloc[0]
            a.plot([x + dx] * 2, [r["normal_eqn_p2.5"], r["normal_eqn_p97.5"]], color=COLORS["normal"], lw=3.2,
                   alpha=.45, solid_capstyle="round", label="Normal, equal size: 95%" if (x, dx) == (0, -.13) else None)
            a.plot(x + dx, r.wmr, "o", ms=4.8, color=COLORS["wmr"], mec="white", mew=.6,
                   label="WMR" if (x, dx) == (0, -.13) else None)
            a.annotate(f"{r.share_draws_below_wmr:.0%}", (x + dx, r["normal_eqn_p97.5"]), xytext=(0, 2),
                       textcoords="offset points", ha="center", fontsize=6.6, color=MUTED)
    a.axhline(0, color="#999999", lw=.6)
    a.set(xticks=list(xs), xticklabels=lab, xlim=(-.5, 2.5), ylim=(-.12, .3), ylabel="corr(predicted, realized)",
          title="(d) Markout corr., equal size")
    a.legend(frameon=False, fontsize=6.8, loc="upper left", handletextpad=.3, borderaxespad=.1, ncol=1)
    fig.tight_layout(pad=.3, w_pad=.8)
    path = FIGURES / "report_clocks_fig.png"
    fig.savefig(path, dpi=300); plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------------------------- PDF
def build(fig: str) -> None:
    fonts()
    st = styles()
    P = lambda t, s="body": Paragraph(t, st[s])
    rates = json.loads((OUT / "cache_clocks" / "rates.json").read_text())
    d = {k: desc(k) for k in CLK}
    w = {k: wall(k) for k in CLK}
    hdr = ["", *[CNAME[k] for k in CLK]]
    unit = ["One 5 s-equivalent = ", "5 seconds", f"{rates['volume'] * 5:.1f} contracts traded",
            f"{rates['event'] * 5:.0f} book messages"]
    t1 = [hdr, unit,
          ["Median wall time of the fill horizon, WMR / normal",
           *[f"{w[k].loc['wmr', 'wall_label_median']:.1f} / {w[k].loc['normal', 'wall_label_median']:.1f} s" for k in CLK]],
          ["Fill rate, WMR / normal", *[f"{d[k].loc['wmr', 'fill_rate']:.1%} / {d[k].loc['normal', 'fill_rate']:.1%}" for k in CLK]],
          ["<b>WMR − normal fill rate</b> (points), day-paired", *[ci(a1(k, "fill_rate"), 100, "+.1f") for k in CLK]],
          ["   … matched on book state", *[ci(a2(k, STRATA["book"], "fill_rate"), 100, "+.1f") for k in CLK]],
          ["   … matched on predicted fill probability", *[ci(a2(k, STRATA["pfill"], "fill_rate"), 100, "+.1f") for k in CLK]],
          ["Markout per fill, WMR / normal (ticks)",
           *[f"{d[k].loc['wmr', 'mk5_mean']:+.3f} / {d[k].loc['normal', 'mk5_mean']:+.3f}".replace("-", "−") for k in CLK]],
          ["<b>WMR − normal markout</b> (ticks), day-paired", *[ci(a1(k, "markout_5s_given_fill")) for k in CLK]],
          ["   … matched on book state", *[ci(a2(k, STRATA["book"], "markout_5s_given_fill")) for k in CLK]],
          ["   … spread capture part", *[ci(a1(k, "spread_capture_given_fill")) for k in CLK]],
          ["   … post-fill move part", *[ci(a1(k, "move_5s_given_fill")) for k in CLK]],
          ["WMR − normal payoff Z per opportunity", *[ci(a1(k, "Z_5s_per_opportunity"), 1, "+.4f") for k in CLK]]]

    def fb(k):
        f = tab(k, "B_fill_metrics").set_index(["group", "model"])
        return f.loc[("wmr", "M0"), "prevalence"] - f.loc[("wmr", "M0"), "mean_pred"]

    def bday(k):
        b = tab(k, "B_day_tests").query("period == 'wmr' and metric == 'brier_M0_minus_M1'").iloc[0]
        return f"{b['mean']:+.4f} (p = {b.p:.3f})"

    def r1(k):
        b = tab(k, "R1_wmr_day_tests").query("eval == 'walk_forward' and model == 'wmr_only' and "
                                             "metric == 'brier_M1_minus_m'").iloc[0]
        return f"{b['mean']:+.4f} (p = {b.p:.2f})".replace("-", "−")

    def mkc(k):
        m = tab(k, "B_markout_metrics").query("h == 5 and model == 'R0'").set_index("group")
        return f"{m.loc['wmr', 'corr']:.3f} / {m.loc['normal', 'corr']:.3f}"

    def share(k, metric):
        s = r2(k).query("metric == @metric").share_draws_below_wmr
        return f"{s.min():.0%} – {s.max():.0%}"

    def cp(k, g, c):
        r = cpair(k, g, c)
        return f"{r['mean']:+.4f} ({pv(r.p)})".replace("-", "−")

    t2 = [hdr,
          ["WMR fills minus ordinary-model prediction (test days)", *[f"{fb(k) * 100:+.1f} points" for k in CLK]],
          ["Brier gain from WMR terms, WMR test days", *[bday(k) for k in CLK]],
          ["WMR-only model vs pooled, Brier gain, walk-forward", *[r1(k) for k in CLK]],
          ["Markout corr(pred., realized), test fills, WMR / normal", *[mkc(k) for k in CLK]],
          ["Equal size: normal samples below WMR, markout corr.", *[share(k, "mk_corr") for k in CLK]],
          ["Equal size: normal samples below WMR, decile Spearman", *[share(k, "decile_spearman") for k in CLK]],
          ["Inside WMR, top 20%: ordinary joint − fill-only", *[cp(k, "wmr", "P3_joint - P1_fill") for k in CLK]],
          ["Inside WMR, top 20%: WMR-aware joint − ordinary joint", *[cp(k, "wmr", "P4_joint_wmr - P3_joint") for k in CLK]],
          ["Normal hours, top 20%: ordinary joint − fill-only", *[cp(k, "normal", "P3_joint - P1_fill") for k in CLK]]]

    v, e = a1("volume", "fill_rate"), a1("event", "fill_rate")
    t0 = a1("time", "fill_rate")
    shrink_v, shrink_e = 1 - v["mean"] / t0["mean"], 1 - e["mean"] / t0["mean"]
    story = [
        P("Volume and event clocks: which WMR results are about time, and which are about trading", "title"),
        P("Addendum to the Phase 3 summary. Same 66 days, opportunities, features, models and tests; only the clock "
          "that defines the fill horizon and the markout horizon changes.", "sub"),
        summary_box(
            "<b>What changes.</b> During the fix the market trades and messages several times faster per second, so "
            "a 5 s window holds much more activity. Measured per unit of trading instead of per second, "
            f"<b>most of WMR's fill advantage disappears</b> ({shrink_v:.0%} smaller on a volume clock, {shrink_e:.0%} "
            "on an event clock): WMR orders fill more mainly because more trading arrives per second. "
            "<b>The adverse-selection result survives every clock</b>: WMR fills still earn +0.076 to +0.099 ticks more "
            "than normal-hours fills, all in a smaller post-fill move, and still after matching on book state. "
            "<b>Finding 3 does not survive</b>: on volume and event clocks, WMR markouts are about as predictable "
            "as normal-hours markouts at equal sample size. The 5 s gap reflects more price-moving activity packed "
            "into 5 s at the fix, not a book that is less informative. One decision result sharpens: inside WMR, a "
            "markout model that knows it is the fix becomes necessary for the joint score to beat fill-only ranking.",
            st),
        P("1. Design", "h"),
        P("<b>Clocks.</b> The <i>volume clock</i> counts contracts traded (all trade messages) and the <i>event "
          "clock</i> counts book messages (non-snapshot adds, cancels, modifies, trades and fills). Each is scaled so "
          "that, in normal hours on the 40 training days, it advances on average at wall-clock speed: "
          f"one second equals {rates['volume']:.2f} contracts or {rates['event']:.1f} messages. A 5 s-equivalent "
          "horizon is 12.1 contracts or 149 messages. "
          "<b>Label:</b> unchanged rule (the first fill, cancel or modify of the order decides). A fill counts if it "
          "starts before the clock has advanced one 5 s-equivalent from entry. "
          "<b>Markout:</b> the mid once the clock has advanced <i>h</i>-equivalents past the fill packet. "
          "<b>Check:</b> the same code run with the wall clock reproduces the original 5 s table exactly "
          "(all 2,726,066 opportunities, every label and markout). Every model, split, "
          "hyperparameter search and test was re-run unchanged; no opportunity is lost to censoring."),
        P("2. Finding 1 under other clocks: \"fill more\" is mostly speed; \"lose less\" is not", "h"),
        table(t1, [2.55, 1.5, 1.5, 1.55], st),
        P("Table 1. WMR minus normal hours, 66 days, day-bootstrap 95% CI; p &lt; 0.001 unless shown. Normal-hours "
          "volume is skewed, so the median normal-hours horizon on the volume clock is longer than its mean of 5 s.",
          "cap"),
        P("The fix trades 4.9 times as many contracts per second as normal hours (13.8 vs 2.8) and sends 2.6 times "
          "as many messages. Per contract traded, WMR's fill advantage falls from 6.4 to "
          f"{v['mean'] * 100:.1f} points, and to an insignificant +0.4 points (p = "
          f"{a2('volume', STRATA['pfill'], 'fill_rate').p:.2f}) at equal predicted fill probability; the event clock "
          "lies in between. <b>The markout advantage is robust</b>: +0.086 and +0.076 ticks on the new clocks "
          "(+0.077 to +0.106 matched on book state), again with equal spread capture and a smaller adverse move. "
          "WMR's share of fills with negative markouts is 6 points lower (4 on the 5 s clock), and the gain against the "
          "adjacent windows is unchanged (+0.087 and +0.069, p &lt; 0.001). Adverse selection per unit of trading is "
          "genuinely lower at the fix."),
        KeepTogether([Image(fig, width=W * inch, height=2.2 * inch), P("Figure 1. (a) Median wall time over which each clock advances one 5 s-equivalent after entry. "
          "(b, c) WMR minus normal: day-paired (filled) and reweighted to WMR's book-state mix (hollow), 95% CIs. "
          "(d) Correlation of the ordinary markout model with realized markouts: WMR vs 500 equal-size normal-hours "
          "samples (bar = 95% range; number = share of samples below WMR). Left of each pair: test days; right: "
          "walk-forward.", "cap")]),
        P("3. Findings 2 and 3, and the decision results", "h"),
        table(t2, [2.75, 1.45, 1.45, 1.45], st),
        P("Table 2. Model and selection results by clock. Brier gains are positive when the second model is better. "
          "Equal-size shares cover the four variants (two models × test days or walk-forward). Selection results are "
          "test-day gross payoff per kept opportunity, with the budget set within each period.", "cap"),
        P("<b>Finding 2 holds, with a smaller level gap.</b> The ordinary model under-predicts WMR fills by 1.9 points "
          "on both new clocks, against 3.3 points on the 5 s clock, because part of the 5 s miss was just more "
          "trading per second. WMR terms still improve WMR calibration (p ≈ 0.01–0.02). Models trained on WMR alone "
          "still add nothing (p = 0.08 and 0.77, from a Brier difference of at most 0.0003)."),
        P("<b>Finding 3 is a 5 s-clock effect.</b> On the volume clock the ordinary markout model is as correlated with "
          "realized WMR markouts as with normal-hours ones (0.098 vs 0.101). At equal size, 46% to 92% of normal "
          "samples do worse than WMR, so WMR sits mid-distribution. The event "
          "clock gives no consistent gap either: 10% to 49% for correlation, and 2% to 87% for decile monotonicity. "
          "A reading consistent with all three clocks: the visible book predicts the price move over the next "
          "dozen contracts about equally well in both regimes. A 5 s window at the fix holds about five times as "
          "much trading, so the predictable part is diluted by more later moves. We withdraw the claim that "
          "post-fill moves at the fix are intrinsically harder to predict from the book."),
        P("<b>Decision results.</b> Inside WMR, the ordinary joint score (fill probability × ordinary markout "
          "prediction) loses to fill-only ranking on the new clocks (significant on the event clock). Adding WMR "
          "terms to the markout model recovers it, by +0.016 and +0.017 ticks per kept opportunity (p = 0.015 and "
          "0.007), where the 5 s clock showed no gain. The mechanism is the level, not the ranking: the ordinary "
          "markout model is too pessimistic at the fix (bias −0.06 to −0.09 ticks), which pushes out the orders most "
          "likely to fill. In normal hours the joint score beats fill-only ranking on every clock (p ≤ 0.011)."),
        P("4. What this changes in the story", "h"),
        P("(1) \"Orders fill twice as often at the fix\" should read: they fill more often <i>per second</i>, mostly "
          "because the market trades faster; per contract traded the advantage is small. "
          "(2) \"Less adversely selected\" is the robust core result: about 0.08–0.10 ticks per fill on every clock, "
          "against both controls, and after matching on book state. "
          "(3) Finding 3 should be dropped from the main story or stated as a 5 s-clock fact. "
          "(4) The regime indicator matters more than the 5 s run suggested. It is needed to fix the markout level, "
          "and on activity clocks that level fix is what lets a joint score work inside WMR. "
          "<b>Limits:</b> the calibration (normal-hours mean rate, training days) is one reasonable choice among "
          "several. The event clock counts all book messages, including cancellations unrelated to the touch. "
          "Pre-entry features are still measured in seconds. The 30 s-equivalent results are not used, as before."),
    ]
    out = OUT / "report_clocks.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=.7 * inch, rightMargin=.7 * inch,
                            topMargin=.6 * inch, bottomMargin=.6 * inch, title="WMR results under volume and event clocks",
                            author="")

    def footer(c, dd):
        c.setFont("Cal", 8); c.setFillColor(colors.HexColor(MUTED))
        c.drawRightString(letter[0] - .7 * inch, .38 * inch, f"{dd.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(out)


def main() -> None:
    build(figure())


if __name__ == "__main__":
    main()
