# 6EZ6 WMR fixing window: exploratory L3 analysis

This is descriptive analysis of the processed Databento 6EZ6 MBO files. All clocks are converted from UTC to `Europe/London` with timezone-aware conversion. The WMR period is 15:57:30–16:02:30 London time; nearby controls are the adjacent five-minute windows.

## Five observations

1. Spread during WMR versus nearby: 1.12 vs 1.13 (-1.4%) ticks (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `spread_depth_distributions.png`).
2. Total top-five displayed depth: 1.56e+03 vs 1.26e+03 (+24.2%) contracts (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `book_shape.png`).
3. Executed contracts per sampled second: 14.7 vs 7.02 (+109.0%) (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `queue_depletion.png`).
4. Cancelled contracts per sampled second: 113 vs 90.4 (+25.3%) (Table `period_comparison.csv`; Figure `queue_depletion.png`).
5. Absolute best-quote imbalance: 0.399 vs 0.421 (-5.2%) (Table `period_comparison.csv`; Figure `wmr_zoom.png`).

## Intraday patterns

`intraday_metrics.csv` contains one row per London day and minute for 23 weekdays, including partial London days at the file boundaries; 22 weekdays contribute to WMR. Each plotted point in `intraday_profile.png` and `wmr_zoom.png` is the equal-day mean of that minute's value, with 10th–90th day percentile bands. Realized volatility is the square root of summed squared adjacent one-second log-mid returns, in basis points. The median new limit-order size uses `A` event sizes within each day-minute, excluding snapshot rows.

## Queue structure and dynamics

Best-level resting order count during WMR versus nearby: 48.5 vs 40.5 (+19.8%); median best-order age: 13.8 vs 20.1 (-31.3%) seconds; new displayed quantity per second: 151 vs 107 (+42.0%); and mean change in the surviving best-queue reference's volume ahead: -9.63 versus -6.44 contracts (`period_comparison.csv`). Compare the conditional mechanisms in `queue_depletion.png` and the order-count shape in `book_shape.png`.

`book_shape.png` and `spread_depth_distributions.png` use one-second completed book states. `state_transitions.png` shows empirical one-second transition probabilities, and `transition_matrices.csv` gives counts. `queue_depletion.png` conditions best-price execution (`F`), best-price cancellation (`C` after subtracting matched fill follow-ups), best-price additions (`A`), and top-of-book net depth change on starting depth. `queue_depletion.csv` provides the underlying numbers. The reference order for queue-ahead change is the last order at the best price in the prior second, retained only if the same order is still at that best price in the next sample.

## Fill dynamics

- wmr: 0.127 observed first-fill probability among 12,078 selected best-quote additions (`fill_probability_curves.png`).
- nearby: 0.098 observed first-fill probability among 21,552 selected best-quote additions (`fill_probability_curves.png`).
- normal: 0.064 observed first-fill probability among 1,775,327 selected best-quote additions (`fill_probability_curves.png`).
- wmr B: strongest displayed single-feature rank association is `orders_ahead` (Spearman -0.349, 6,000 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- wmr A: strongest displayed single-feature rank association is `orders_ahead` (Spearman -0.295, 6,078 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- nearby B: strongest displayed single-feature rank association is `same_side_depth` (Spearman -0.241, 10,815 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- nearby A: strongest displayed single-feature rank association is `same_side_depth` (Spearman -0.235, 10,737 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- normal B: strongest displayed single-feature rank association is `queue_ahead` (Spearman -0.220, 886,742 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- normal A: strongest displayed single-feature rank association is `queue_ahead` (Spearman -0.219, 888,585 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).

Candidates are the first eligible new best-quote order per side per second, observed after its complete event packet. Label 1 means an `F` event for the same order ID within five seconds and before an exit; partial fills count. `C`, `M`, or book reset before the first `F` ends the observed episode with label 0. Orders still active after five observed seconds receive label 0. Orders still active near a truncated file boundary are excluded. `queue_position` is visible volume ahead divided by total same-side best-level volume. `fill_time_distribution.png` is a CDF **among filled candidates**. `feature_fill_summary.csv` gives descriptive Spearman associations by side and period; `fill_probability_bins.csv` gives day-equal binned curves, day percentile bands, and observation counts. These labels are for observed orders under their owners' cancellation behavior, not hypothetical orders held for five seconds.

## Next-step hypotheses

1. At the same London clock time, WMR changes the conditional execution hazard after controlling for visible queue-ahead quantity and spread.
2. WMR changes whether a thin best queue is depleted by execution or cancellation, at a fixed starting depth bucket.
3. Queue-ahead quantity has different fill-probability calibration during WMR than in nearby controls, separately for bid and ask orders.

## Limits and checks

The input has 47,088,610 processed rows across 30 UTC files; Saturday files are empty and Sunday sessions are omitted from weekday comparisons. Day-level percentile bands describe variation across dates and are not confidence intervals. Message streams within a day are dependent. A `T` row gives aggressive trade quantity while `F` records passive execution; the two are never summed as separate trade volume. The post-fill `C`/`M` book update is not counted as another execution. Residual `C` quantity after a matched `F` is classified as cancellation; unmatched `M` reductions are not assigned to execution or cancellation. One-second states are sampled only after complete packets and are omitted if stale for more than 30 seconds. Book depth beyond the best uses the existing `Book` utility on the processed event stream; this is an analytical view, not a regenerated L3 dataset. The first eligible add per side per second limits message-rate weighting but does not remove traders' selection or cancellation bias.

Unknown-order event references observed by analytical replay: 0. Trailing incomplete packet rows: 0. Manual event-sequence checks found a trade `T` paired with passive `F` fills and follow-up `C` reductions, and a partial `F` followed by `M` of the remaining order; the accounting counts the `F` once in each case. Inspect `diagnostics.csv` before relying on results. Time-of-day sensitivity is in `period_comparison.csv` under `same_hour_control` (15:30–16:30 excluding WMR and nearby). No causal or statistical-significance claim follows from these descriptive plots.

## Reproduce

From the `project` directory, run with its virtual environment: `.venv\Scripts\python.exe wmr_eda.py` (or `python wmr_eda.py` after activating `.venv`). Optional `--limit-days 1` provides a smoke run. Completed daily replay results are cached in `daily_cache_v1/` for restart.
