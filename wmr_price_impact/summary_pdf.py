"""Short PDF summary of the whole story (outputs/WMR_summary.pdf).

Stage 1 (wmr_value): fill rate and markout at the WMR fix, on 5 s / volume / event clocks.
Stage 2 (wmr_price_impact): KNN-matched aggressive-trade price impact and the maker-markout bridge.
Reads only saved tables. Run from project/: .venv/Scripts/python.exe -m wmr_price_impact.summary_pdf
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

from wmr_value.config import COLORS, OUT as VALUE_OUT, PROJECT, TABLES as VALUE_TABLES
from wmr_value.report_pdf import INK, MUTED, W, fonts, styles, summary_box, table

from .config import FIGURES, TABLES

OUT_PDF = PROJECT / "outputs" / "WMR_summary.pdf"
CLOCKS = [("time", "5 s clock"), ("volume", "volume clock"), ("event", "event clock")]


def stage1_fig() -> str:
    c = pd.read_csv(VALUE_TABLES / "clock_comparison.csv")

    def get(clock, starts):
        r = c[(c.clock == clock) & c["item"].str.startswith(starts)].iloc[0]
        return r.est, r.lo, r.hi

    fig, ax = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    x = np.arange(3)
    series = [
        (ax[0], "fill_rate vs normal", "unmatched (day-paired)", 100),
        (ax[0], "walk-forward M0 fill-probability decile x side | fill_rate vs normal", "same predicted fill prob.", 100),
        (ax[1], "markout_5s_given_fill vs normal", "unmatched (day-paired)", 1),
        (ax[1], "queue x imbalance x flow x spread x side | markout_5s_given_fill vs normal", "same book state", 1)]
    for k, (a, item, lab, scale) in enumerate(series):
        hollow = k % 2 == 1
        v = np.array([get(cl, item) for cl, _ in CLOCKS]) * scale
        a.errorbar(x + (.12 if hollow else -.12), v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o",
                   ms=5, capsize=2.5, lw=1.2, color=COLORS["wmr"], mfc="white" if hollow else COLORS["wmr"], label=lab)
    for a in ax[:2]:
        a.axhline(0, color=MUTED, lw=.8)
        a.set_xticks(x, [n.replace(" clock", "") for _, n in CLOCKS])
        a.grid(axis="x", visible=False)
    ax[0].legend(loc="upper right", fontsize=7)
    ax[1].legend(loc="lower left", fontsize=7)
    ax[0].set(title="(a) Fill-rate gap, WMR − normal", ylabel="percentage points", ylim=(-1.5, 8))
    ax[1].set(title="(b) Markout gap per fill, WMR − normal", ylabel="ticks", ylim=(-.02, .15))
    for i, (cl, _) in enumerate(CLOCKS):
        for per, dx in (("wmr", -.17), ("normal", .17)):
            w = 5.0 if cl == "time" else pd.read_csv(VALUE_OUT / f"clock_{cl}" / "tables" / "clock_sample.csv") \
                .set_index("period").loc[per, "wall_label_median"]
            ax[2].bar(i + dx, w, .32, color=COLORS[per], label={"wmr": "WMR", "normal": "normal hours"}[per] if i == 0 else None)
            ax[2].text(i + dx, w + .15, f"{w:.1f}", ha="center", fontsize=7, color=INK)
    ax[2].set_xticks(x, [n.replace(" clock", "") for _, n in CLOCKS])
    ax[2].grid(axis="x", visible=False)
    ax[2].set(title="(c) Wall time of the 5 s-equivalent horizon", ylabel="median seconds", ylim=(0, 10.5))
    ax[2].legend(loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    path = FIGURES / "summary_stage1.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def stage2_fig() -> str:
    e = pd.read_csv(TABLES / "effects.csv")
    raw = pd.read_csv(TABLES / "raw_effects.csv")
    br = pd.read_csv(TABLES / "bridge.csv")
    cal = json.loads((TABLES / "design.json").read_text())["caliper"]
    V = ["I0", "I_v5", "I_v10", "I_v25", "I_v50", "I_v100"]
    XL = ["0+", "5", "10", "25", "50", "100"]
    m = e[(e["sample"] == "test") & (e.comparison == "B") & (e.clock == "volume") & (e.side == "both") & (e.K == 5)
          & np.isclose(e.caliper, cal)].set_index("col").loc[V]
    r = raw[(raw["sample"] == "test") & (raw.comparison == "B")].set_index("col").loc[V]
    x = np.arange(len(V))
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for who, col, lab, mk in (("wmr", COLORS["wmr"], "WMR", "o"), ("ctrl", COLORS["normal"], "matched normal", "s")):
        ax[0].fill_between(x, m[f"{who}_lo"], m[f"{who}_hi"], color=col, alpha=.15, lw=0)
        ax[0].plot(x, m[who], marker=mk, color=col, lw=1.8, ms=4, label=lab)
    ax[0].set(title="(a) Price response after comparable trades", ylabel="ticks in the trade's direction")
    ax[0].legend(loc="lower right", fontsize=7)
    for d, lab, hollow, dx in ((r, "unmatched", True, -.12), (m, "KNN-matched", False, .12)):
        ax[1].errorbar(x + dx, d["diff"], yerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt="o", ms=4.5, capsize=2,
                       lw=1.1, color=INK if hollow else COLORS["wmr"], mfc="white" if hollow else COLORS["wmr"], label=lab)
    ax[1].axhline(0, color=MUTED, lw=.8)
    ax[1].set(title="(b) WMR − normal impact gap", ylabel="ticks", ylim=(-.38, .05), yticks=[-.3, -.2, -.1, 0])
    ax[1].legend(loc="lower right", fontsize=7)
    for a in ax[:2]:
        a.set_xticks(x, XL)
        a.set_xlabel("contracts traded after the trade")
    cols, labels = ["I_w5", "I_v10", "I_v25", "I_v100"], ["5 s", "10 c", "25 c", "100 c"]
    xb = np.arange(len(cols))
    for k, (q, lab, col, mk) in enumerate((("(1) all trades, matched", "all trades, matched", COLORS["normal"], "o"),
                                           ("(4) historical maker markout", "historical maker fills", COLORS["wmr"], "D"))):
        d = br[(br["sample"] == "all") & (br.side == "both") & (br.quantity == q)].set_index("col").loc[cols]
        ax[2].errorbar(xb + (k - .5) * .24, d["diff"], yerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt=mk, ms=4.5,
                       capsize=2, lw=1.1, color=col, label=lab)
    ax[2].axhline(0, color=MUTED, lw=.8)
    ax[2].set_xticks(xb, labels)
    ax[2].set(title="(c) WMR advantage for the passive side", ylabel="ticks (maker view)", ylim=(-.02, .14),
              xlabel="horizon (s = seconds, c = contracts)")
    ax[2].legend(loc="upper right", fontsize=7)
    for a in ax:
        a.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    path = FIGURES / "summary_stage2.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def build(f1: str, f2: str) -> None:
    fonts()
    st = styles()
    P = lambda t, s="body": Paragraph(t, st[s])
    B = lambda t: Paragraph(t, st["bul"], bulletText="•")
    img = lambda p, h: Image(p, width=W * inch, height=h * inch)
    story = [
        P("Passive liquidity at the WMR fix: what changes, and why", "title"),
        P("CME Euro FX futures (6E), Level-3 order book, 66 trading days (Jul–Oct 2026). WMR window = 15:57:30–16:02:30 "
          "London; controls = adjacent 5-minute windows (nearby) and other liquid hours 08:00–17:00 (normal).", "sub"),
        summary_box(
            "<b>Stage 1.</b> At the fix, passive orders at the touch fill about twice as often per second and earn "
            "<b>0.08–0.10 tick better markouts</b> per fill. The fill advantage is mostly speed (it shrinks by 68% when "
            "time is measured in traded contracts); the markout advantage survives every clock and book-state "
            "matching, and comes entirely from a smaller adverse move after the fill.<br/>"
            "<b>Stage 2.</b> Matching each WMR aggressive trade to comparable normal-hours trades (KNN on the "
            "pre-trade book) shows that <b>immediate impact is the same</b>; what is smaller at WMR is the "
            "<b>continued drift in the trade's direction</b> afterwards (about half, −0.12 tick after 100 contracts). "
            "This general effect explains a quarter to a half of the maker advantage at short horizons; the rest sits "
            "in the particular trades that fill resting orders.<br/>"
            "<b>Reading.</b> Consistent with fix flow that trades to minimise <i>tracking error</i> to the benchmark "
            "rather than to profit from information: it moves the book as much as any trade, but predicts less about "
            "where the price goes next.", st),
        Spacer(1, 6),

        P("Stage 1 — How the WMR window differs for a passive order", "h"),
        P("<b>Unit.</b> Real participant limit orders joining the best bid or ask (one per second per side; 2.7 million "
          "opportunities). <b>Outcomes.</b> Fill within the horizon, and the 5-second <b>markout</b> per fill = spread "
          "captured − adverse mid move, in ticks (positive = good for the passive side). <b>Clocks.</b> Besides 5 "
          "wall-clock seconds, horizons are measured in traded contracts (volume clock) and book messages (event "
          "clock), each scaled to equal 5 s on average in normal hours. Inference uses the trading day as the unit "
          "(day-paired differences, day bootstrap)."),
        KeepTogether([img(f1, 2.3 * W / 7.1),
                      P("<b>Figure 1.</b> (a) WMR − normal fill-rate gap; filled: raw day-paired, hollow: at the same "
                        "out-of-sample predicted fill probability. (b) Markout gap per fill; hollow: same queue, "
                        "imbalance, flow, spread and side. 95% day-bootstrap CIs. (c) WMR trades 4.9× the contracts "
                        "per second, so its 5 s-equivalent horizon lasts only ~2 s.", "cap")]),
        B("<b>Fill rate:</b> 13.0% vs 6.6% on the 5 s clock (+6.4 pts). Per unit of trading the gap is +2.0 pts "
          "(volume) and +3.8 (event); at equal predicted fill probability it vanishes on the volume clock "
          "(+0.4, p = 0.20). <b>WMR fills faster mainly because more trades arrive per second.</b>"),
        B("<b>Markout:</b> +0.098 tick on the 5 s clock, +0.086 (volume), +0.076 (event); +0.077 to +0.106 after "
          "book-state matching; all p &lt; 0.001. Spread captured is identical (−0.005 to −0.008), so the "
          "advantage is <b>a smaller adverse price move after the fill</b> (+0.08 to +0.11). This is the "
          "phenomenon Stage 2 explains."),
        Spacer(1, 4),

        P("Stage 2 — Mechanism: price impact of comparable aggressive trades (KNN)", "h"),
        P("A maker's markout is exactly <b>spread captured − the signed price impact of the aggressive trade that "
          "filled it</b> (verified to 1e-9 on every linked fill). So we study impact on <b>all</b> 1.25 million "
          "aggressive trades, not only those that hit our historical orders. <b>Event</b> = one exchange packet with "
          "trade prints (no double counting of trade, fill and book-update rows). <b>Matching:</b> each WMR trade "
          "gets its 5 nearest normal-hours trades in standardized pre-trade features (size, depth on both sides, "
          "signed flow 1/5/30 s, trading intensity, cancellations, volatility), exact on side, spread and intensity "
          "decile, within a caliper chosen on training days for balance (all |SMD| ≤ 0.10). Main tests use the 13 "
          "held-out test days; responses are on the volume clock."),
        KeepTogether([img(f2, 2.45 * W / 7.1),
                      P("<b>Figure 2.</b> (a) Matched WMR and normal-hours trades, test days: same immediate impact, "
                        "then less drift at WMR. (b) Unmatched vs matched gap: the book state explains the raw "
                        "difference at 0+ and most of it later. (c) WMR advantage for the passive side, all days: "
                        "the general effect (all trades, matched) vs the historical maker-markout advantage; "
                        "95% day-bootstrap CIs.", "cap")]),
        table([["WMR − matched normal (test days)", "immediate I(0+)", "later drift R(100)", "total I(100)"],
               ["unmatched", "−0.086 ***", "−0.086 ***", "−0.173 ***"],
               ["KNN-matched (37% of WMR trades)", "−0.006 (p = 0.37)", "−0.110 (p = 0.02)", "−0.116 (p = 0.01)"],
               ["… all 66 days (60% matched)", "−0.005 (p = 0.01)","−0.046 (p = 0.004)", "−0.051 (p = 0.003)"]],
              [2.55, 1.5, 1.5, 1.5], st),
        P("Ticks in the trade's direction; *** p &lt; 0.001. Robust to K = 1/10, a looser caliper, event and wall "
          "clocks and dropping trades at the window edge.", "cap"),
        B("<b>No smaller immediate impact.</b> Raw WMR trades move the mid 0.09 tick less on impact, but only because "
          "the book is deeper at the fix (fewer trades clear the level). Matched on the book, the gap is zero."),
        B("<b>Less continuation, not reversal.</b> After comparable trades, WMR prices keep drifting in the trade's "
          "direction, but about half as much (0.12 vs 0.23 tick over 100 contracts). The effect is concentrated "
          "in aggressive sells."),
        B("<b>Only part of the maker advantage.</b> The general effect equals about half of the maker advantage at "
          "5 s (0.049 vs 0.092) and a quarter to a third at 10–25 contracts (0.017–0.019 vs 0.055–0.066); only by "
          "100 contracts do they match. Trades that actually filled our historical orders show a larger matched "
          "advantage (≈0.06, imprecise): the rest reflects <b>which resting orders get filled</b>."),
        Spacer(1, 4),

        P("Economic interpretation — tracking error vs profit", "h"),
        P("Two motives coexist at the fix. A <b>benchmark trader</b> (asset manager rebalancing, dealer hedging "
          "client fixing orders) is judged on <b>tracking error</b> to the fix rate: it must trade inside the "
          "window whatever it expects prices to do. A <b>profit-seeking trader</b> trades only when it expects "
          "the price to move its way. The evidence lines up with a larger share of the first type at WMR:"),
        table([["", "Tracking-error (benchmark) flow", "Profit-seeking (informed) flow", "What we see at WMR"],
               ["Why it trades now", "the window, not a price view", "a price view", "4.9× the volume per second"],
               ["Immediate impact", "consumes the book like any trade", "same", "same, once the book is matched"],
               ["Later drift in its direction", "little: carries little information", "continues: information",
                "about half as large"],
               ["Passive side's markout", "better (less adverse selection)", "worse", "+0.08–0.10 tick"]],
              [1.55, 1.9, 1.75, 1.9], dict(st, cellr=st["cell"])),
        Spacer(1, 4),
        B("For <b>liquidity providers</b> the fix is cheaper to serve: the same immediate price concession, but less "
          "adverse drift afterwards. The deeper book is the visible response — more resting liquidity is posted "
          "into the window."),
        B("For <b>benchmark traders</b> the trade-off runs the other way: by concentrating flow to hit the fix, they "
          "accept paying the spread and impact to makers in exchange for low tracking error, not for profit."),
        B("<b>Limits.</b> Trader identities are not observed, so this is an interpretation consistent with the data, "
          "not an identified mechanism. Matched results cover the less intense part of WMR (37% of WMR trades vs "
          "normal hours, 11% vs nearby windows); the nearby comparison has the same sign but is mostly not "
          "significant; 13 test days."),
    ]
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter, leftMargin=.7 * inch, rightMargin=.7 * inch,
                            topMargin=.6 * inch, bottomMargin=.6 * inch, title="Passive liquidity at the WMR fix",
                            author="")

    def footer(c, d):
        c.setFont("Cal", 8)
        c.setFillColor(colors.HexColor(MUTED))
        c.drawRightString(letter[0] - .7 * inch, .38 * inch, f"{d.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT_PDF)


def main() -> None:
    build(stage1_fig(), stage2_fig())


if __name__ == "__main__":
    main()
