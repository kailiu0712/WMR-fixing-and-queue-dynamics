"""Generate the three-page full-sample WMR EDA brief.

Run after wmr_eda.py. Inputs are the full-sample period/feature CSVs and figures.
Requires pandas, Pillow, and reportlab in the invoking Python environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


W, H = 612, 792
NAVY = colors.HexColor("#17324c")
TEAL = colors.HexColor("#147a89")
GRAY = colors.HexColor("#52616d")
LIGHT = colors.HexColor("#eaf0f4")
PALE = colors.HexColor("#f5f8fa")
PALE_TEAL = colors.HexColor("#e8f4f4")


def setup_font() -> str:
    for name, path in (("DejaVu", "C:/Windows/Fonts/DejaVuSans.ttf"),
                       ("Arial", "C:/Windows/Fonts/arial.ttf")):
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(name, path))
            return name
    return "Helvetica"


def para(c, content: str, x: float, top: float, width: float, *,
         font: str, size: float = 8.7, leading: float = 12.4,
         color=GRAY) -> float:
    style = ParagraphStyle("p", fontName=font, fontSize=size, leading=leading,
                           textColor=color)
    p = Paragraph(content, style)
    _, height = p.wrap(width, 1000)
    p.drawOn(c, x, top - height)
    return top - height


def header(c, title: str, subtitle: str, page: int, font: str):
    c.setFillColor(NAVY)
    c.rect(0, H - 81, W, 81, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(font, 18.5)
    c.drawString(38, H - 43, title)
    c.setFont(font, 9)
    c.setFillColor(colors.HexColor("#cddde5"))
    c.drawString(39, H - 62, subtitle)
    c.setStrokeColor(colors.HexColor("#cfdae1"))
    c.line(38, 39, W - 38, 39)
    c.setFont(font, 7.3)
    c.setFillColor(GRAY)
    c.drawString(38, 25, "Team 8  |  ORIE 5257  |  CME 6E MBO exploratory analysis")
    c.drawRightString(W - 38, 25, str(page))


def section(c, label: str, top: float, font: str) -> float:
    c.setFillColor(TEAL)
    c.setFont(font, 11)
    c.drawString(39, top - 11, label)
    return top - 23


def draw_image(c, path: Path, x: float, top: float, max_w: float,
               max_h: float) -> float:
    with Image.open(path) as img:
        iw, ih = img.size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    c.drawImage(ImageReader(str(path)), x + (max_w - w) / 2, top - h,
                width=w, height=h)
    return top - h


def fmt(metric: str, value: float) -> str:
    if metric in ("spread_ticks", "abs_imbalance"):
        return f"{value:.3f}"
    if metric == "median_best_order_age_s":
        return f"{value:.1f}"
    return f"{value:,.1f}"


def observed_fill_rates(features: pd.DataFrame) -> dict[str, tuple[float, int]]:
    rows = features.loc[features.feature == "queue_ahead"]
    result = {}
    for period, group in rows.groupby("period"):
        n = int(group.n_orders.sum())
        result[period] = (float((group.fill_probability * group.n_orders).sum() / n), n)
    return result


def report(project: Path, dest: Path) -> None:
    out = project / "outputs" / "wmr_eda"
    period = pd.read_csv(out / "tables" / "period_comparison.csv")
    features = pd.read_csv(out / "tables" / "feature_fill_summary.csv")
    diag = pd.read_csv(out / "tables" / "diagnostics.csv")
    period = period.set_index(["metric", "period"])
    feat = features.set_index(["feature", "side", "period"])
    rates = observed_fill_rates(features)
    def metric(name: str, window: str) -> float:
        return float(period.loc[(name, window), "day_equal_mean"])
    def rho(name: str, side: str, window: str) -> float:
        return float(feat.loc[(name, side, window), "spearman"])
    n_days = int(period.loc[("executions", "wmr"), "n_days"])
    rows = int(diag.rows.sum())
    unknown = int(diag.get("unknown_order_events", pd.Series(0, index=diag.index)).fillna(0).sum())
    font = setup_font()
    c = canvas.Canvas(str(dest), pagesize=(W, H))
    c.setTitle("WMR fixing and CME 6E queue dynamics: full-sample EDA")
    c.setAuthor("Team 8")

    # Page 1: current-sample results only.
    header(c, "WMR fixing and queue dynamics",
           "Full three-month sample  |  14 Jul–13 Oct 2026", 1, font)
    top = 697
    top = para(c,
        f"<b>{n_days} London fixing weekdays, {rows:,} MBO events.</b> The 92-UTC-day series "
        "uses 6EU6 through 10 September and 6EZ6 from 11 September. The fixing window is "
        "15:57:30–16:02:30 London time; nearby controls are the adjacent five-minute windows.",
        39, top, 534, font=font, size=9.2, leading=14) - 18
    top = section(c, "Market state and flow at the fixing", top, font)
    columns = [("Measure", 49), ("WMR", 290), ("Nearby", 380),
               ("Normal", 471), ("WMR / nearby", 561)]
    c.setFillColor(LIGHT)
    c.roundRect(39, top - 27, 534, 27, 5, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont(font, 8)
    for label, x in columns:
        if x == 49:
            c.drawString(x, top - 18, label)
        else:
            c.drawRightString(x, top - 18, label)
    y = top - 27
    metric_labels = [
        ("executions", "Executed contracts / s"),
        ("adds", "Added contracts / s"),
        ("cancels", "Cancelled contracts / s"),
        ("depth5", "Top-five depth, contracts"),
        ("best_order_count", "Best-level order count"),
        ("median_best_order_age_s", "Median best-order age, s"),
        ("spread_ticks", "Best spread, ticks"),
    ]
    for i, (name, label) in enumerate(metric_labels):
        if i % 2:
            c.setFillColor(PALE)
            c.rect(39, y - 26, 534, 26, fill=1, stroke=0)
        wmr, near, normal = (metric(name, p) for p in ("wmr", "nearby", "normal"))
        c.setFillColor(NAVY)
        c.setFont(font, 8.2)
        c.drawString(49, y - 17, label)
        for x, v in ((290, fmt(name, wmr)), (380, fmt(name, near)),
                     (471, fmt(name, normal)), (561, f"{wmr/near:.2f}×")):
            c.drawRightString(x, y - 17, v)
        y -= 26
    top = y - 19
    top = para(c,
        "<b>Reading the contrast.</b> WMR executes 2.16 times the nearby volume per second "
        "while displaying 20% more top-five depth. Additions and cancellations are also faster, "
        "and best-level orders are younger. The queue is more active even though more quantity "
        "is visible; a near-one-tick spread alone does not capture that turnover.",
        39, top, 534, font=font, size=8.8, leading=12.8) - 17
    top = section(c, "Observed first-fill probability within five seconds", top, font)
    boxes = [("WMR", "wmr", TEAL), ("Nearby", "nearby", NAVY),
             ("Normal", "normal", GRAY)]
    for i, (label, period_name, color) in enumerate(boxes):
        x = 39 + i * 181
        c.setFillColor(PALE_TEAL if i == 0 else LIGHT)
        c.roundRect(x, top - 75, 171, 75, 6, fill=1, stroke=0)
        c.setFillColor(color)
        c.setFont(font, 9)
        c.drawString(x + 12, top - 19, label)
        rate, n = rates[period_name]
        c.setFont(font, 17)
        c.drawString(x + 12, top - 44, f"{rate:.1%}")
        c.setFillColor(GRAY)
        c.setFont(font, 7.5)
        c.drawString(x + 12, top - 62, f"{n:,} selected orders")
    top -= 91
    top = para(c,
        "A candidate is the first eligible new best-quote order per side per second after its "
        "completed event packet. The label is any recorded fill before cancellation/repricing "
        "within five seconds. This is an observed-order outcome under historical exit behavior.",
        39, top, 534, font=font, size=8.4, leading=11.8) - 12
    para(c,
        f"<b>Replay check:</b> {unknown} references to unknown orders. All flow rates are "
        "day-equal means of one-second observations; the normal group spans other eligible "
        "weekday periods and has a different clock-time mix.",
        39, top, 534, font=font, size=8.2, leading=11.6)
    c.showPage()

    # Page 2: intraday dynamics and depletion mechanism.
    header(c, "Intraday activity and queue turnover",
           "Equal-day minute profile and conditional best-queue changes", 2, font)
    top = 699
    top = section(c, "Around the London 16:00 fixing", top, font)
    top = draw_image(c, out / "figures" / "wmr_zoom.png", 42, top, 528, 373) - 6
    top = para(c,
        "The shaded interval is the fixing window. Bands show 10th–90th percentiles across "
        "days, not confidence intervals. Execution, addition and cancellation rates spike "
        "near 16:00 while displayed depth also rises.",
        43, top, 526, font=font, size=8, leading=11.2) - 14
    top = section(c, "Best-price flow conditional on starting depth", top, font)
    top = draw_image(c, out / "figures" / "queue_depletion.png", 42, top,
                     528, 163) - 5
    para(c,
        "The execution series counts passive F quantity once; matched post-fill C updates "
        "are removed from cancellations. Higher WMR turnover is therefore present in both "
        "trading and order withdrawal, rather than an accounting duplication.",
        43, top, 526, font=font, size=8, leading=11)
    c.showPage()

    # Page 3: all measured univariate feature correlations, interpretation, next tests.
    header(c, "Queue-state associations with fills",
           "Spearman correlation by side: normal versus WMR", 3, font)
    top = 699
    top = para(c,
        "Each entry is Spearman ρ between a feature at the observed order's completed add "
        "packet and its five-second first-fill label. Negative queue correlations mean "
        "orders farther back are less likely to fill before exit.",
        39, top, 534, font=font, size=8.7, leading=12.4) - 13
    top = section(c, "Feature–fill correlation (predictability proxy)", top, font)
    c.setFillColor(LIGHT)
    c.roundRect(39, top - 35, 534, 35, 5, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont(font, 7.7)
    c.drawString(49, top - 21, "Feature")
    for title, x in (("Bid normal", 291), ("Bid WMR", 374),
                     ("Ask normal", 466), ("Ask WMR", 561)):
        c.drawRightString(x, top - 21, title)
    y = top - 35
    feature_labels = [
        ("queue_ahead", "Queue volume ahead"),
        ("orders_ahead", "Orders ahead"),
        ("queue_position", "Relative queue position"),
        ("same_side_depth", "Same-side best depth"),
        ("imbalance", "Best-quote imbalance"),
        ("size", "Own order size"),
        ("spread_ticks", "Spread"),
        ("cancel_5s", "Prior cancellations, 5 s"),
        ("recent_vol_30s_bps", "Prior volatility, 30 s"),
        ("signed_flow_1s", "Signed flow, 1 s"),
        ("signed_flow_5s", "Signed flow, 5 s"),
        ("signed_flow_30s", "Signed flow, 30 s"),
    ]
    for i, (name, label) in enumerate(feature_labels):
        if i % 2:
            c.setFillColor(PALE)
            c.rect(39, y - 20, 534, 20, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont(font, 8)
        c.drawString(49, y - 13.5, label)
        for x, side, period_name in ((291, "B", "normal"), (374, "B", "wmr"),
                                     (466, "A", "normal"), (561, "A", "wmr")):
            value = rho(name, side, period_name)
            c.setFillColor(TEAL if period_name == "wmr" else NAVY)
            c.drawRightString(x, y - 13.5, f"{value:+.3f}")
        y -= 20
    top = y - 12
    top = para(c,
        "Sample sizes: WMR bid 18,203 / ask 18,385; normal bid 2,538,400 / ask 2,530,867. "
        "These are univariate, in-sample associations, not out-of-sample model scores or "
        "significance tests; the different fill base rates also affect correlation magnitudes.",
        39, top, 534, font=font, size=7.9, leading=10.8) - 14
    top = section(c, "Microstructure and economic interpretation", top, font)
    top = para(c,
        "Queue volume and order count ahead become more negatively associated with filling "
        "during WMR (bid ρ −0.217 to −0.309; ask −0.220 to −0.293). A quote behind more "
        "resting quantity needs more depletion before it can execute; concentrated fixing "
        "flow may make that position more consequential. Positive bid-minus-ask depth "
        "imbalance accompanies fewer bid fills and more ask fills, consistent with "
        "buy pressure reaching the ask; imbalance is also tied to depth, so this is "
        "not a separate causal effect. Spread and short signed-flow correlations remain small, while "
        "volatility's association is weaker in WMR.",
        39, top, 534, font=font, size=8.1, leading=11.4) - 11
    top = para(c,
        "Benchmark-related spot-FX flow can cluster near fixes (Krohn et al., 2021), "
        "but these futures correlations do not identify that mechanism. A higher passive "
        "fill rate also need not improve market-maker economics if fills carry adverse "
        "markouts or inventory risk.",
        39, top, 534, font=font, size=8.1, leading=11.4) - 13
    top = section(c, "Implication for the next test", top, font)
    top = para(c,
        "Prioritize queue-ahead and imbalance interactions, while letting held-out scores "
        "decide whether weak univariate signals add conditional value. Rebuild features "
        "strictly before arrival; validate packet-end book states against MBP-1 and freeze "
        "roll/expiry exclusions. Compare B1 with WMR-interaction M1 on later days using "
        "Brier score, day-block intervals, matched placebos, and post-fill markouts.",
        39, top, 534, font=font, size=8.1, leading=11.4) - 8
    c.setFont(font, 6.8)
    c.setFillColor(GRAY)
    c.drawString(39, max(top - 7, 47),
                 "Context: Krohn, Mueller & Whelan (2021), Bank of Canada SWP 2021-48.")
    c.linkURL("https://www.bankofcanada.ca/2021/10/staff-working-paper-2021-48/",
              (39, max(top - 12, 43), 535, max(top + 3, 58)), relative=0)
    c.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path,
                        default=Path(r"D:\lkh\Cornell\26 Fall\5257\project"))
    parser.add_argument("--out", type=Path,
                        default=Path("wmr_eda_proposal_brief.pdf"))
    args = parser.parse_args()
    report(args.project, args.out)
    print(args.out.resolve())
