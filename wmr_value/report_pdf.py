"""3-page PDF summary of Phase 3 (outputs/wmr_value/report_3page.pdf).

Uses only saved tables plus the observation cache (for the per-period CIs in Figure 1).
Run after wmr_value.run and wmr_value.robustness:  .venv/Scripts/python.exe -m wmr_value.report_pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .build_labels import load_observations
from .config import COLORS, FIGURES, OUT, PERIODS, TABLES
from .stats import ratio_ci

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e3e3e3"
NAME = {"wmr": "WMR", "nearby": "Nearby", "normal": "Normal"}
W = 7.1  # figure width, inches

plt.rcParams.update({"font.family": "Calibri", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": .6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5})


def pct_log(ax, axis: str, ticks: list) -> None:
    """Log axis labelled in plain percentages (Calibri has no mathtext superscripts)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    a.set_minor_formatter(NullFormatter())


def fig1(obs: pd.DataFrame) -> str:
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.7]})
    for i, per in enumerate(PERIODS):
        s = obs[obs.period == per]
        d = s.assign(mk=s.mk_5.fillna(0) * s.label).groupby("day").agg(f=("label", "sum"), n=("label", "size"),
                                                                       mk=("mk", "sum"))
        v, lo, hi = ratio_ci(d.f, d.n)
        ax[0].bar(i, v * 100, .6, color=COLORS[per], yerr=[[(v - lo) * 100], [(hi - v) * 100]], capsize=3,
                  error_kw={"lw": 1, "ecolor": INK})
        ax[0].text(i, hi * 100 + .5, f"{v * 100:.1f}%", ha="center", fontsize=7.5, color=INK)
        m, mlo, mhi = ratio_ci(d.mk, d.f)
        ax[1].errorbar(i, m, yerr=[[m - mlo], [mhi - m]], fmt="o", color=COLORS[per], ms=6, capsize=3, lw=1.5)
        ax[1].text(i + .2, m, f"{m:+.3f}", va="center", fontsize=7.5, color=INK,
                   bbox={"fc": "white", "ec": "none", "pad": .6})
    for a in ax[:2]:
        a.set_xticks(range(3), [NAME[p] for p in PERIODS])
        a.grid(axis="x", visible=False)
    ax[0].set(title="(a) Filled within 5 s", ylabel="% of opportunities", ylim=(0, 16))
    ax[1].axhline(0, color=MUTED, lw=.8)
    ax[1].set(title="(b) 5 s markout per filled order", ylabel="ticks", xlim=(-.5, 2.7))

    f = pd.read_csv(TABLES / "R4_frontier_within_period.csv")
    for per in ("normal", "wmr"):
        s = f[f.period == per].sort_values("bin")
        ax[2].errorbar(s.mean_score * 100, s.mk, yerr=[s.mk - s.mk_lo, s.mk_hi - s.mk], fmt="o-", ms=4, lw=1.5,
                       capsize=2, elinewidth=.8, color=COLORS[per])
        last = s.iloc[-1]
        ax[2].text(last.mean_score * 100 * 1.12, last.mk + (.06 if per == "wmr" else -.07), NAME[per],
                   color=COLORS[per], va="center", fontsize=8, fontweight="bold")
    ax[2].axhline(0, color=MUTED, lw=.8)
    pct_log(ax[2], "x", [1, 2, 5, 10, 20, 50])
    ax[2].set(title="(c) Markout at the same predicted fill probability", ylabel="5 s markout per fill (ticks)",
              xlabel="out-of-sample predicted fill probability (decile mean)", ylim=(-.35, .5), xlim=(.45, 95))
    fig.tight_layout(w_pad=1.2)
    path = FIGURES / "report_fig1.png"
    fig.savefig(path, dpi=220); plt.close(fig)
    return str(path)


def fig3() -> str:
    """Realized Z per kept opportunity, within-period budgets, test days (from C_policies)."""
    c = pd.read_csv(TABLES / "C_policies.csv").query("cost == 0 and budget == 'within_period'")
    pol = {"P1_fill": ("Fill probability only", "#9a9a9a", "s"), "P3_joint": ("Joint, ordinary features", INK, "o"),
           "P4_joint_wmr": ("Joint, + WMR terms", "#7b5ea7", "D")}
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.95))
    for a, g in zip(ax, ("wmr", "normal")):
        s = c[c.group == g]
        base = s[s.policy == "P0_all"].iloc[0]
        a.axhline(base.Z_per_accepted, color=COLORS[g], lw=1.2, ls="--")
        a.text(.5, base.Z_per_accepted, "keep all", color=COLORS[g], va="bottom", ha="center", fontsize=7.5)
        for k, (p, (lab, col, mk)) in enumerate(pol.items()):
            r = s[s.policy == p].sort_values("target_frac")
            xs = np.arange(3) + (k - 1) * .18
            a.errorbar(xs, r.Z_per_accepted, yerr=[r.Z_per_accepted - r.Z_lo, r.Z_hi - r.Z_per_accepted], fmt=mk,
                       ms=5, color=col, capsize=2, lw=1.2, label=lab)
        a.axhline(0, color=MUTED, lw=.6)
        a.set_xticks(range(3), ["top 10%", "top 20%", "top 30%"])
        a.grid(axis="x", visible=False)
        a.set(title=f"({'a' if g == 'wmr' else 'b'}) " + ("Inside the WMR window" if g == "wmr" else "Inside normal hours"),
              ylabel="realized Z per kept opportunity (ticks)" if g == "wmr" else None)
    ax[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout(w_pad=2)
    path = FIGURES / "report_fig3.png"
    fig.savefig(path, dpi=220); plt.close(fig)
    return str(path)


def fig2() -> str:
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1, 1.15, 1.25]})
    f = pd.read_csv(TABLES / "R4_frontier_within_period.csv")
    for per in ("normal", "wmr"):
        s = f[f.period == per].sort_values("bin")
        ax[0].plot(s.mean_score * 100, (s.fill_rate - s.mean_score) * 100, "o-", ms=4, lw=1.5, color=COLORS[per],
                   label=NAME[per])
    ax[0].axhline(0, color=MUTED, lw=.8, ls="--")
    pct_log(ax[0], "x", [1, 3, 10, 30])
    ax[0].set(title="(a) Fill: realized − predicted", xlabel="predicted fill (decile mean)",
              ylabel="realized − predicted (% points)", ylim=(-2, 10))
    ax[0].legend(loc="upper left")

    q = pd.read_csv(TABLES / "R4_markout_deciles_within_period.csv")
    for per in ("normal", "wmr"):
        s = q[(q.model == "M0") & (q.period == per)].sort_values("bin")
        ax[1].errorbar(s.mean_score, s.mk, yerr=[s.mk - s.mk_lo, s.mk_hi - s.mk], fmt="o-", ms=4, lw=1.5, capsize=2,
                       elinewidth=.8, color=COLORS[per], label=f"{NAME[per]} ({s.n_fills.sum() / 1e3:.1f}k fills)")
    ax[1].plot([-.11, .13], [-.11, .13], color=MUTED, lw=.8, ls="--")
    ax[1].axhline(0, color=MUTED, lw=.6)
    ax[1].set(title="(b) Markout: ordinary model, test days", xlabel="predicted 5 s markout (ticks, decile mean)",
              ylabel="realized 5 s markout (ticks)", ylim=(-.42, .48))
    ax[1].legend(loc="lower right", fontsize=6.8, handlelength=1.2)

    r = pd.read_csv(TABLES / "R2_equal_n.csv")
    r = r[r.metric == "mk_corr"]
    rows = [("test_days", "M0", "Ordinary, test"), ("test_days", "M1", "+WMR terms, test"),
            ("walk_forward", "M0", "Ordinary, walk-fwd"), ("walk_forward", "M1", "+WMR terms, walk-fwd")]
    for k, (ev, m, lab) in enumerate(rows):
        x = r[(r["eval"] == ev) & (r.model == m)].iloc[0]
        y = len(rows) - 1 - k
        ax[2].plot([x["normal_eqn_p2.5"], x["normal_eqn_p97.5"]], [y, y], color=COLORS["normal"], lw=5, alpha=.35,
                   solid_capstyle="round", label="Normal, equal size: 95% range" if k == 0 else None)
        ax[2].plot(x.normal_eqn_median, y, "|", color=COLORS["normal"], ms=10, mew=2)
        ax[2].plot(x.wmr, y, "o", color=COLORS["wmr"], ms=6, label="WMR" if k == 0 else None)
        ax[2].text(.165, y, f"{x.share_draws_below_wmr:.0%}", va="center", fontsize=7.5, color=INK)
    ax[2].text(.165, 3.4, "below WMR", fontsize=6.5, color=MUTED, va="bottom")
    ax[2].axvline(0, color=MUTED, lw=.6)
    ax[2].set_yticks(range(len(rows)), [lab for *_, lab in rows][::-1])
    ax[2].set(xlim=(-.06, .22), ylim=(-1.9, 3.8), title="(c) Markout correlation at equal size",
              xlabel="corr(predicted, realized 5 s markout)")
    ax[2].grid(axis="y", visible=False)
    ax[2].legend(loc="lower left", fontsize=6.8, ncol=1, handlelength=1.2)
    fig.tight_layout(w_pad=1.0)
    path = FIGURES / "report_fig2.png"
    fig.savefig(path, dpi=220); plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------------------------- PDF
def fonts() -> None:
    for name, file in (("Cal", "calibri.ttf"), ("Cal-B", "calibrib.ttf"), ("Cal-I", "calibrii.ttf"),
                       ("Cal-BI", "calibriz.ttf")):
        pdfmetrics.registerFont(TTFont(name, f"C:/Windows/Fonts/{file}"))
    addMapping("Cal", 0, 0, "Cal"); addMapping("Cal", 1, 0, "Cal-B")
    addMapping("Cal", 0, 1, "Cal-I"); addMapping("Cal", 1, 1, "Cal-BI")


def styles() -> dict:
    base = ParagraphStyle("body", fontName="Cal", fontSize=9.8, leading=12.2, textColor=INK, alignment=TA_JUSTIFY,
                          spaceAfter=3.5)
    return {"body": base,
            "title": ParagraphStyle("t", parent=base, fontName="Cal-B", fontSize=15.5, leading=18.5, alignment=0,
                                    spaceAfter=2),
            "sub": ParagraphStyle("s", parent=base, fontSize=9, textColor=MUTED, alignment=0, spaceAfter=6),
            "h": ParagraphStyle("h", parent=base, fontName="Cal-B", fontSize=11, leading=13, alignment=0,
                                spaceBefore=5, spaceAfter=2.5, keepWithNext=1),
            "box": ParagraphStyle("b", parent=base, fontSize=9.8, leading=12.4),
            "cap": ParagraphStyle("c", parent=base, fontSize=8.4, leading=10.2, textColor="#444444", spaceAfter=5),
            "bul": ParagraphStyle("u", parent=base, leftIndent=10, bulletIndent=1, spaceAfter=1.8),
            "cell": ParagraphStyle("cell", parent=base, fontSize=8.4, leading=10, alignment=0, spaceAfter=0),
            "cellr": ParagraphStyle("cellr", parent=base, fontSize=8.4, leading=10, alignment=2, spaceAfter=0)}


def table(data, widths, st, header_rows=1, bold_first_col=False) -> Table:
    cells = [[Paragraph(str(c), st["cell"] if j == 0 else st["cellr"]) for j, c in enumerate(r)] for r in data]
    t = Table(cells, colWidths=[w * inch for w in widths], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), .8, colors.HexColor(INK)),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), .5, colors.HexColor(MUTED)),
        ("LINEBELOW", (0, -1), (-1, -1), .8, colors.HexColor(INK)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def summary_box(text: str, st: dict) -> Table:
    t = Table([[Paragraph(text, st["box"])]], colWidths=[W * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f1ec")),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(COLORS["wmr"])),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def build(f1: str, f2: str, f3: str) -> None:
    fonts()
    st = styles()
    P = lambda t, s="body": Paragraph(t, st[s])
    B = lambda t: Paragraph(t, st["bul"], bulletText="•")
    img = lambda p, h: Image(p, width=W * inch, height=h * inch)

    d = pd.read_csv(TABLES / "A1_descriptive.csv").query("side == 'both'").set_index("period")
    t1 = [["", "WMR", "Nearby (±5 min)", "Normal hours"],
          ["Opportunities (66 days)", *[f"{int(d.loc[p, 'n']):,}" for p in PERIODS]],
          ["Filled within 5 s", *[f"{d.loc[p, 'fill_rate']:.1%}" for p in PERIODS]],
          ["Spread capture per fill, s(m<sub>pre</sub> − p)", *[f"{d.loc[p, 'spread_capture_mean']:+.3f}" for p in PERIODS]],
          ["Mid move 5 s after fill, s(m<sub>τ+5</sub> − m<sub>pre</sub>)", *[f"{d.loc[p, 'move5_mean']:+.3f}" for p in PERIODS]],
          ["<b>5 s markout per fill</b>", *[f"<b>{d.loc[p, 'mk5_mean']:+.3f}</b>" for p in PERIODS]],
          ["Fills with negative 5 s markout", *[f"{d.loc[p, 'mk5_neg_share']:.0%}" for p in PERIODS]],
          ["Payoff Z<sub>5</sub> per opportunity", *[f"{d.loc[p, 'Z5_mean']:+.4f}" for p in PERIODS]]]

    story = [
        P("Passive liquidity at the WMR fix: orders fill more and lose less, but are no easier to pick", "title"),
        P("CME Euro FX futures (6E), Databento MBO, 66 trading days (14 Jul – 13 Oct 2026). Phase 3 summary.", "sub"),
        summary_box(
            "<b>The story.</b> In the five-minute window around the 4 pm London WMR fix (15:57:30–16:02:30), "
            "passive orders that join the best bid or offer fill about twice as often as in normal liquid hours "
            "(13.0% vs 6.6% within 5 s). They are also <b>less adversely selected</b>: +0.07 ticks per filled order "
            "after 5 s, against −0.04 in normal hours. "
            "The whole gain comes from a smaller adverse price move after the fill. The spread captured is the same. "
            "The result survives conditioning on queue position, imbalance, recent flow, spread, side and "
            "predicted fill probability. "
            "Predictive models see the fix only as a <b>shift in level</b>. Ordinary order-book features "
            "under-predict WMR fill rates by about 3 points, and a WMR indicator fixes that. But neither "
            "WMR-specific terms nor models trained on WMR data alone rank opportunities <i>within</i> the window "
            "any better. Post-fill price moves during the fix are, if anything, <b>harder</b> to predict from the "
            "book than in normal hours, even at equal sample size. "
            "In short: the fix tells a liquidity provider that adverse selection is lower, not which orders "
            "to choose. The gross edge, about 0.1 tick per fill, is small next to plausible trading costs.", st),
        P("1. Data and design", "h"),
        P("An <b>opportunity</b> is a new limit order that joins the best bid or offer, taking at most one per second "
          "per side, between 08:00 and 17:00 London time. There are 2.73M opportunities in total. Its label <i>Y</i> = 1 "
          "if the order fills at its own price within 5 s, before any cancel or modify. Orders repriced across the "
          "spread are treated as exits, not fills. "
          "<b>Markout</b> at horizon <i>h</i> is <i>M<sub>h</sub></i> = <i>s</i>(<i>m</i><sub>τ+h</sub> − <i>p</i>) "
          "in ticks of 0.00005 ($6.25), measured from the first fill τ. The side sign <i>s</i> is +1 for a bid, so a "
          "positive markout means the price moved the maker's way. "
          "Markout splits exactly into spread capture plus the subsequent mid move. The payoff per opportunity is "
          "<i>Z</i> = <i>Y</i>·<i>M</i><sub>5</sub>. "
          "Features use only messages completed before the entry packet; a unit test scrambles later messages "
          "and checks that features do not change. "
          "<b>The trading day is the unit of inference.</b> Every test uses day-paired differences, a day-bootstrap "
          "95% CI and a sign-flip p-value. "
          "Models are L2 logistic regression (fill) and Ridge (markout given fill). They use a chronological split "
          "of 40 train, 13 validation and 13 untouched test days, plus an expanding walk-forward from day 20. "
          "All outcomes are those of real participants' orders under their own cancellation decisions. "
          "They describe an observed policy, not what a new quote would earn."),
        P("2. Finding 1: WMR orders fill more and are less adversely selected", "h"),
        table(t1, [2.75, 1.4, 1.5, 1.45], st),
        P("Table 1. Pooled per-period means; markouts in ticks. Spread capture is identical across periods; "
          "the WMR advantage is all in the post-fill move.", "cap"),
        img(f1, 2.4),
        P("Figure 1. (a, b) Pooled ratios with day-bootstrap 95% CIs. (c) Opportunities grouped into deciles of "
          "the walk-forward predicted fill probability (ordinary features), with deciles cut separately in each period.",
          "cap"),
        P("<b>Day-paired differences (WMR minus control), 66 days.</b> Against normal hours: fill rate +6.4 points "
          "[5.4, 7.5] and 5 s markout +0.099 ticks [0.073, 0.124]. Against the adjacent windows: +3.6 points "
          "[2.8, 4.4] and +0.079 ticks [0.045, 0.111]. All have p &lt; 0.001; the markout is better on 79% and 71% "
          "of days respectively. The 1 s horizon gives the same picture (+0.098 vs normal), and leave-one-day-out "
          "estimates stay within 0.074–0.083 ticks against the adjacent windows."),
        P("<b>Not a composition effect.</b> We reweight normal-hours orders to WMR's mix of queue-ahead × imbalance × "
          "5 s flow × spread × side, separately for each day (94% of WMR fills have a match). WMR still fills "
          "6.5 points more and earns +0.106 ticks [0.079, 0.132] more per fill. "
          "Matching on the out-of-sample predicted fill probability instead gives +0.092 [0.062, 0.123] against "
          "normal and +0.070 [0.017, 0.121] against the adjacent windows (p = 0.012). "
          "Figure 1c shows the same: at every predicted fill level, WMR fills earn more. WMR also reaches fill "
          "rates (54% in its top decile) that ordinary orders seldom see. "
          "<b>The fix shifts the fill–markout trade-off up and to the right; it does not trade one for the other.</b>"),
        P("3. Finding 2: the regime is predictable as a level, not as a ranking", "h"),
        img(f2, 2.5),
        P("Figure 2. (a) Realized minus predicted fill rate for the ordinary fill model, walk-forward over 46 days, with deciles cut within each "
          "period. WMR fills beat the prediction from about 4% predicted fill upward, by up to 8 points; normal hours "
          "stay within about 1 point. (b) Realized vs predicted 5 s markout on held-out test "
          "fills, with day-bootstrap CIs. (c) The same markout correlation computed on 500 normal-hours samples. "
          "Each sample is built from whole 5-minute windows to WMR's per-day fill count. The bar is the 95% range; "
          "the tick is the median; the number is the share of samples below WMR.", "cap"),
        table([["WMR test opportunities (13 days, 7,144 orders, 14.4% filled)", "Mean predicted", "Calibration intercept",
                "Brier", "AUC"],
               ["Ordinary features (pooled)", "11.1%", "0.40", "0.0927", "0.820"],
               ["+ WMR indicator and 3 interactions (pooled)", "13.8%", "0.06", "0.0909", "0.818"],
               ["Same features, trained on WMR rows only", "15.1%", "−0.08", "0.0907", "0.824"]],
              [3.35, 0.95, 1.1, 0.75, 0.95], st),
        P("Table 2. Fill-probability models on held-out WMR opportunities. A calibration intercept of 0 is perfect; "
          "a positive intercept means the model under-predicts.", "cap"),
        P("The ordinary model under-predicts WMR fills on 12 of 13 test days (p = 0.001) and under-predicts WMR "
          "markouts by 0.11 ticks (p = 0.006). Adding WMR terms removes the fill bias and halves the markout bias. Its Brier gain is real "
          "but small: +0.0018 per day on test days and +0.0011 [0.0005, 0.0018] across 46 walk-forward days. "
          "It also loses significance once orders are thinned to one per 10 s (p = 0.19), so it is best read as a "
          "calibration correction. "
          "<b>Discrimination does not improve.</b> AUC is unchanged. A model trained only on WMR data (22,409 "
          "orders) does no better than the pooled one. Walk-forward, the WMR-only model differs from the pooled "
          "one by −0.0001 in Brier score [−0.0006, 0.0004], −0.0006 in AUC, and its markout error is 0.002 ticks larger; none "
          "is significant. So training on mostly normal-hours data does not hide any WMR structure that these "
          "features could capture."),
        P("4. Finding 3 (new): markouts during the fix are harder to predict", "h"),
        P("On pooled deciles, the predicted-vs-realized markout curve looked flat for WMR and steep for normal hours. "
          "Part of that is sample size: normal hours have 44× as many test fills. Shrunk to WMR's size, normal-hours "
          "decile monotonicity (Spearman) falls from 0.96 to a median of 0.64. "
          "<b>But sample size is not the whole story.</b> Against equal-size normal-hours samples that keep the same "
          "within-day clustering (Figure 2c), WMR's markout correlation is in the lower tail in all four variants: "
          "1.6% to 16% of the samples are worse. Its decile monotonicity is lower still: 0% to 27% of samples are "
          "worse, and only 3% are worse with the ordinary model walk-forward. "
          "Fill ranking shows no consistent gap. "
          "<i>Interpretation:</i> post-fill moves at the fix are driven less by the visible book and more by "
          "fixing flow, which pre-entry features do not see. The evidence is consistent but not overwhelming, "
          "because the tail shares vary by model."),
        P("5. Decision implications (observational selection of historical orders)", "h"),
        P("Each rule keeps the top 20% of opportunities by its score, with thresholds set on validation days, and is "
          "scored on test-day realized <i>Z</i> (gross, ticks per kept opportunity). Every difference is day-paired."),
        table([["Comparison (top 20% kept, 13 test days)", "Difference in Z", "95% CI", "p", "Days > 0"],
               ["All periods: joint score (fill prob. × markout) vs keep all", "+0.0075", "[0.0053, 0.0097]", "&lt;0.001", "13/13"],
               ["All periods: joint score vs fill probability only", "+0.0030", "[0.0005, 0.0050]", "0.027", "12/13"],
               ["Inside WMR: joint score vs keep everything", "+0.038", "[0.017, 0.060]", "0.007", "10/13"],
               ["Inside WMR: WMR-aware joint score vs ordinary joint score", "+0.007", "[−0.006, 0.022]", "0.36", "7/13"],
               ["Inside WMR: WMR-only model vs WMR-aware pooled model", "−0.010", "[−0.023, 0.004]", "0.21", "6/13"]],
              [3.3, 1.0, 1.15, 0.65, 0.65], st),
        P("Table 3. Selecting among historical opportunities; inside WMR, the 20% budget is set within the window.",
          "cap"),
        img(f3, 1.95),
        P("Figure 3. Realized gross payoff per kept opportunity on test days, with thresholds set within each period; "
          "day-bootstrap 95% CIs. The dashed line keeps every opportunity.", "cap"),
        P("Ordinary book features already pick better orders, and they do so inside WMR too, where the top fifth "
          "earns about 0.04–0.06 ticks per opportunity against 0.017 for keeping everything. "
          "Consistent with Finding 3, the markout half of the joint score adds nothing inside WMR. Ranking by fill "
          "probability alone does as well there (0.054 vs 0.042 at 20%; not significant), while in normal hours the "
          "joint score beats fill-only ranking at the 20% and 30% budgets. "
          "Knowing it is the fix adds nothing to this ranking. The only WMR gain comes from moving the budget into "
          "the window: with a single global threshold, the WMR-aware score keeps 92% of WMR opportunities "
          "(small but consistent walk-forward gains of +0.0002 to +0.0011 overall, p ≤ 0.04). "
          "<b>Costs</b> are illustrative, not verified. At 0.1 tick per fill, WMR's net payoff (−0.004 per "
          "opportunity) is still better than both controls (p &lt; 0.01). At 0.3 tick it is worse than normal "
          "hours (−0.030 vs −0.022), because WMR fills more often and each fill now loses money. "
          "None of this is a counterfactual P&amp;L: a rule that \"keeps\" an order its owner cancelled has no "
          "observed outcome."),
        P("6. Robustness and limits", "h"),
        B("<b>Imbalance across periods</b> (WMR is 1.3% of rows): every comparison is per day and per period. "
          "Weighting days by their precision instead of equally moves the main differences by less than 0.01 tick. "
          "Deciles and selection budgets are cut within each period, and Section 3 adds WMR-only models and "
          "equal-size comparisons. Rare fills keep their natural rate (no resampling) and are scored by Brier "
          "score and calibration."),
        B("<b>Sampling and labels:</b> the one-per-second cap keeps a similar share of best-price adds in each period "
          "(7.2% WMR, 6.9% nearby, 10.1% normal). The 2–3% of raw \"fills\" that were aggressive reprices are "
          "relabeled, so all remaining fills are at the order's own price. Dropping orders whose 5 s outcome runs "
          "past 16:02:30 changes nothing."),
        B("<b>Left out as fragile:</b> the 30 s horizon, which depends on side and weakens to p = 0.10 under "
          "precision weighting; and fine-cell matching against the adjacent windows, where only 34% of WMR fills "
          "find a match."),
        B("<b>Limits:</b> results reflect participants' own placement and cancellation choices, the same bias as "
          "observing others' resting orders (Albers et al., 2025). The sample is 66 days of summer and early autumn, "
          "with only 13 untouched test days. It covers CME futures, not the spot venues behind the benchmark. "
          "Trader identities, inventories and fees are unobserved."),
        P("7. Conclusion", "h"),
        P("The WMR fix changes the economics of passive liquidity provision in 6E in a consistent direction. Resting "
          "orders fill more often and suffer about 0.1 tick less adverse selection per fill, at the same book state. "
          "That change is predictable as a regime-level shift, which a single indicator captures. It is not "
          "predictable as a ranking: during the fix, the visible book says less about which fills will be toxic. "
          "The defensible claim is a <b>regime-level difference in adverse selection</b>, not a tradable "
          "within-window signal, and its size is small relative to realistic costs."),
    ]
    out = OUT / "report_3page.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=.7 * inch, rightMargin=.7 * inch,
                            topMargin=.6 * inch, bottomMargin=.6 * inch, title="Passive liquidity at the WMR fix",
                            author="")

    def footer(c, d):
        c.setFont("Cal", 8); c.setFillColor(colors.HexColor(MUTED))
        c.drawRightString(letter[0] - .7 * inch, .38 * inch, f"{d.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(out)


def main() -> None:
    obs = load_observations()
    build(fig1(obs), fig2(), fig3())


if __name__ == "__main__":
    main()
