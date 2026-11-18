# 6E WMR fixing window: exploratory L3 analysis

This is descriptive analysis of the processed Databento 6E MBO files, 2026-07-14 through 2026-10-13 UTC. The sample uses 6EU6 through 2026-09-10 and 6EZ6 from 2026-09-11, following CME's FX quarterly roll date. All clocks are converted from UTC to `Europe/London` with timezone-aware conversion. The WMR period is 15:57:30–16:02:30 London time; nearby controls are the adjacent five-minute windows.

## Five observations

1. Spread during WMR versus nearby: 1.1 vs 1.13 (-2.0%) ticks (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `spread_depth_distributions.png`).
2. Total top-five displayed depth: 1.34e+03 vs 1.12e+03 (+20.1%) contracts (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `book_shape.png`).
3. Executed contracts per sampled second: 13.9 vs 6.44 (+116.4%) (Table `period_comparison.csv`; Figures `wmr_zoom.png`, `queue_depletion.png`).
4. Cancelled contracts per sampled second: 102 vs 77.6 (+31.1%) (Table `period_comparison.csv`; Figure `queue_depletion.png`).
5. Absolute best-quote imbalance: 0.392 vs 0.419 (-6.4%) (Table `period_comparison.csv`; Figure `wmr_zoom.png`).

## Intraday patterns

`intraday_metrics.csv` contains one row per London day and minute for 67 weekdays, including partial London days at the file boundaries; 66 weekdays contribute to WMR. Each plotted point in `intraday_profile.png` and `wmr_zoom.png` is the equal-day mean of that minute's value, with 10th–90th day percentile bands. Realized volatility is the square root of summed squared adjacent one-second log-mid returns, in basis points. The median new limit-order size uses `A` event sizes within each day-minute, excluding snapshot rows.

## Queue structure and dynamics

Best-level resting order count during WMR versus nearby: 44.2 vs 37 (+19.6%); median best-order age: 13.9 vs 18.7 (-25.9%) seconds; new displayed quantity per second: 133 vs 90.5 (+47.4%); and mean change in the surviving best-queue reference's volume ahead: -8.54 versus -5.51 contracts (`period_comparison.csv`). Compare the conditional mechanisms in `queue_depletion.png` and the order-count shape in `book_shape.png`.

`book_shape.png` and `spread_depth_distributions.png` use one-second completed book states. `state_transitions.png` shows empirical one-second transition probabilities, and `transition_matrices.csv` gives counts. `queue_depletion.png` conditions best-price execution (`F`), best-price cancellation (`C` after subtracting matched fill follow-ups), best-price additions (`A`), and top-of-book net depth change on starting depth. `queue_depletion.csv` provides the underlying numbers. The reference order for queue-ahead change is the last order at the best price in the prior second, retained only if the same order is still at that best price in the next sample.

## Fill dynamics

- wmr: 0.134 observed first-fill probability among 36,588 selected best-quote additions (`fill_probability_curves.png`).
- nearby: 0.097 observed first-fill probability among 65,463 selected best-quote additions (`fill_probability_curves.png`).
- normal: 0.057 observed first-fill probability among 5,069,267 selected best-quote additions (`fill_probability_curves.png`).
- wmr B: strongest displayed single-feature rank association is `same_side_depth` (Spearman -0.309, 18,203 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- wmr A: strongest displayed single-feature rank association is `orders_ahead` (Spearman -0.300, 18,385 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- nearby B: strongest displayed single-feature rank association is `same_side_depth` (Spearman -0.235, 32,813 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- nearby A: strongest displayed single-feature rank association is `orders_ahead` (Spearman -0.246, 32,650 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- normal B: strongest displayed single-feature rank association is `queue_ahead` (Spearman -0.217, 2,538,400 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).
- normal A: strongest displayed single-feature rank association is `queue_ahead` (Spearman -0.220, 2,530,867 orders; `feature_fill_summary.csv`, `fill_probability_curves.png`).

Candidates are the first eligible new best-quote order per side per second, observed after its complete event packet. Label 1 means an `F` event for the same order ID within five seconds and before an exit; partial fills count. `C`, `M`, or book reset before the first `F` ends the observed episode with label 0. Orders still active after five observed seconds receive label 0. Orders still active near a truncated file boundary are excluded. `queue_position` is visible volume ahead divided by total same-side best-level volume. `fill_time_distribution.png` is a CDF **among filled candidates**. `feature_fill_summary.csv` gives descriptive Spearman associations by side and period; `fill_probability_bins.csv` gives day-equal binned curves, day percentile bands, and observation counts. These labels are for observed orders under their owners' cancellation behavior, not hypothetical orders held for five seconds.

## Next-step hypotheses

1. At the same London clock time, WMR changes the conditional execution hazard after controlling for visible queue-ahead quantity and spread.
2. WMR changes whether a thin best queue is depleted by execution or cancellation, at a fixed starting depth bucket.
3. Queue-ahead quantity has different fill-probability calibration during WMR than in nearby controls, separately for bid and ask orders.

## Limits and checks

The input has 122,903,237 processed rows across 92 UTC files; Saturday files are empty and Sunday sessions are omitted from weekday comparisons. Day-level percentile bands describe variation across dates and are not confidence intervals. Message streams within a day are dependent. A `T` row gives aggressive trade quantity while `F` records passive execution; the two are never summed as separate trade volume. The post-fill `C`/`M` book update is not counted as another execution. Residual `C` quantity after a matched `F` is classified as cancellation; unmatched `M` reductions are not assigned to execution or cancellation. One-second states are sampled only after complete packets and are omitted if stale for more than 30 seconds. Book depth beyond the best uses the existing `Book` utility on the processed event stream; this is an analytical view, not a regenerated L3 dataset. The first eligible add per side per second limits message-rate weighting but does not remove traders' selection or cancellation bias.

Unknown-order event references observed by analytical replay: 0. Trailing incomplete packet rows: 0. Manual event-sequence checks found a trade `T` paired with passive `F` fills and follow-up `C` reductions, and a partial `F` followed by `M` of the remaining order; the accounting counts the `F` once in each case. Inspect `diagnostics.csv` before relying on results. Time-of-day sensitivity is in `period_comparison.csv` under `same_hour_control` (15:30–16:30 excluding WMR and nearby). No causal or statistical-significance claim follows from these descriptive plots.

## Reproduce

From the `project` directory, run with its virtual environment: `.venv\Scripts\python.exe wmr_eda.py` (or `python wmr_eda.py` after activating `.venv`). Optional `--limit-days 1` provides a smoke run. Completed daily replay results are cached in `daily_cache_v1/` for restart.
