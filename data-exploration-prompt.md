# Task: WMR Fixing vs. Normal Period — L3 Order Book Exploratory Analysis

## Objective

Conduct a rigorous but lightweight microstructure EDA on **one month of CME Euro FX (6E) L3 data**.

**Data location:** `project/data_6e/l3`

The data is already processed into L3 format. **Do not rebuild the ingestion or order book reconstruction pipeline.** Inspect the existing schema and reuse the processed data.

Our research question is:

**Does the WMR 4 PM London fixing window exhibit different order book structure, queue dynamics, and 5-second fill probabilities compared with ordinary periods?**

Use the fixing window **15:57:30–16:02:30 Europe/London**, converting timestamps with timezone-aware daylight-saving handling.

Compare three groups:
- **WMR:** The five-minute fixing window.
- **Nearby controls:** Comparable five-minute windows before and after WMR, excluding the fixing window.
- **Normal:** Other eligible trading periods, with time-of-day controls where appropriate.

This phase is exploratory, not causal. Prioritize clear plots, correct event semantics, and reproducible calculations over complex modeling.

## 1. Intraday market structure — highest priority

Produce a **pooled one-day time-series profile**, averaging each metric across all available trading days by London-local clock time.

Use 1-minute bins (and optional 5-minute smoothing). Include a vertical line at 16:00 and shade the five-minute WMR window.

Generate a multi-panel figure showing:

| Metric | Definition |
|---|---|
| Spread | Mean quoted spread in ticks |
| Volatility | Realized volatility from midprice returns |
| Depth | Mean best bid/ask depth and total top-5 depth |
| Imbalance | \((Q_b-Q_a)/(Q_b+Q_a)\), with mean and preferably absolute imbalance |
| Add rate | Added contracts per second |
| Cancel rate | Cancelled contracts per second |
| Trade rate | Executed contracts per second |
| Trade count | Number of trades per minute |
| Order size | Median new limit-order size |

For each metric, calculate the per-minute value **within each trading day first**, then average across days. Show variation across days using percentile bands or confidence intervals.

Avoid averaging raw messages directly across days, which would overweight unusually active dates.

**Deliverables:**
- `intraday_profile.png`: Full pooled trading-day profile.
- `wmr_zoom.png`: Enlarged view around 16:00, preferably ±30 minutes.

These should reveal whether WMR produces unusual volume, volatility, spread, depth, or order-flow intensity compared with ordinary intraday patterns.

## 2. Static order book structure

Compare WMR, nearby controls, and normal periods.

Analyze:

- Spread distribution (ticks).
- Top-1, top-5, and top-10 depth distributions.
- Order book shape: average resting quantity at levels 1–10 from the best bid/ask.
- Queue composition: number of orders per level, average/median order size, and order age where available.
- Order book imbalance distributions.

Generate readable distribution plots and a bid/ask book-depth profile.

**Important:** Sample book states at regular time intervals, such as once per second, to avoid overrepresenting high-message-rate periods.

## 3. Queue dynamics and state transitions

Study how the book evolves over short horizons.

At 1-second sampling intervals, compute changes in:

- Best bid/ask depth.
- Queue-ahead volume where available.
- Spread state (1 tick, 2 ticks, 3+ ticks).
- Imbalance state (sell-heavy, balanced, buy-heavy).
- Best-price movements.

Estimate simple conditional transition probabilities:

\[
P(S_{t+1}=j\mid S_t=i,\text{period})
\]

Create transition heatmaps separately for WMR and controls.

Also examine **queue depletion mechanisms**:

- Contracts removed by executions.
- Contracts removed by cancellations.
- New limit-order replenishment.
- Net depth change per second.

Compare these quantities conditional on starting depth buckets.

Keep the transition state space small and interpretable. Avoid sophisticated Markov models at this stage.

**Critical correctness:** Respect the existing L3 event semantics. Do not double-count trade/fill/cancel messages associated with the same execution. Distinguish actual executions from cancellations and ambiguous book reductions.

## 4. Fill probability and fill-time analysis

If the processed data supports exact order-ID lifecycle tracking, use observed passive orders arriving at the best bid or ask.

Define:

\[
Y_i=\mathbf{1}\{\text{any execution within 5 seconds before exit}\}.
\]

Use the original proposal's convention: partial fills count, while touching a price without an execution does not.

Analyze fill probability by:

- Queue-ahead volume and queue position percentile.
- Order size.
- Spread.
- Same-side depth and book imbalance.
- Recent signed market-order flow (1s / 5s / 30s).
- Recent volatility.
- WMR vs. control period.
- Bid vs. ask separately.

Produce:

- Binned fill-probability curves with observation counts and day-level uncertainty.
- Fill-time cumulative distribution or survival curves.
- A simple feature–fill association table and visual summary.

**Important:** Cancellation and repricing can censor observations or act as competing exit events. Make the primary label consistent with the proposal's observed execution-before-exit definition; distinguish it from a hypothetical order held for five full seconds. Do not claim they are equivalent.

If exact order-level execution tracking is unavailable, skip this section and explain why. Do not invent fill labels from price touches or aggregate depth changes.

## 5. Single-feature predictive analysis

For each candidate feature, assess its relationship with the 5-second fill indicator.

Use:
- Binned empirical fill probabilities, rather than correlation alone.
- Spearman association or point-biserial correlation as a secondary summary.
- Separate WMR and normal-period curves.
- Simple univariate logistic regressions if useful.

Prioritize these features:

1. Queue-ahead contracts.
2. Queue-ahead order count.
3. Queue position percentile.
4. Spread.
5. Same-side best-level depth.
6. Book imbalance.
7. Recent signed trade flow.
8. Recent cancellation intensity.
9. Recent volatility.
10. Own order size.

For key features, compare the conditional relationship with fill probability across periods, not just their marginal distributions.

Avoid claiming that a feature is useful because its pooled correlation is large: differences in time of day, side, and market activity can confound such relationships.

## 6. Implementation requirements

Keep the project lightweight and readable.

- Use Python, pandas/Polars or NumPy, and matplotlib.
- Reuse existing L3 processing utilities.
- Write small functions for metric calculation, grouping, and plotting.
- Cache intermediate daily summaries where useful.
- Process data day by day if memory is constrained.
- Use a common plotting style with labeled axes, units, sample sizes, and WMR markers.
- Maintain chronological, event-time correctness; features must use information available at the observation time.
- Handle zero-volume and empty-book states explicitly.
- Check that volume units, order counts, and event classifications are consistent.
- Never interpret messages-per-second as contracts-per-second.

Do not implement complex neural networks or a large modeling pipeline.

## 7. Output

Save results under `project/outputs/wmr_eda/`:

```text
wmr_eda/
├── figures/
│   ├── intraday_profile.png
│   ├── wmr_zoom.png
│   ├── book_shape.png
│   ├── spread_depth_distributions.png
│   ├── state_transitions.png
│   ├── queue_depletion.png
│   ├── fill_probability_curves.png
│   └── fill_time_distribution.png
├── tables/
│   ├── intraday_metrics.csv
│   ├── period_comparison.csv
│   ├── transition_matrices.csv
│   └── feature_fill_summary.csv
└── report.md
```

The final `report.md` should contain:

1. **Executive summary:** Five most important observations.
2. **Intraday patterns:** What changes around WMR relative to the surrounding clock time?
3. **Queue structure:** What changes in depth, queue composition, and replenishment?
4. **Fill dynamics:** Which features are most predictive of execution, and do their relationships differ during WMR?
5. **Next-step hypotheses:** Three testable questions for the subsequent fill-probability modeling phase.
6. **Limitations:** Data coverage, dependence across days, event ambiguity, and any skipped analysis.

Each important conclusion must reference a specific figure or table. Do not infer significance from visual differences alone.

## 8. Execution order

Implement in this sequence:

**Phase A — Mandatory:** Intraday pooled profiles, WMR zoom, spread/depth/book shape, and add/cancel/trade intensities.

**Phase B — Mandatory:** Queue changes, depletion/replenishment, and simple transition matrices.

**Phase C — Conditional:** Observed order-level 5-second fill labels, fill-time distributions, and univariate feature analysis, only if exact order lifecycle reconstruction is supported.

Before completing, validate metrics on a small number of manually inspected event sequences, and ensure all figures are readable.

**Success criterion:** Produce an interpretable microstructure report identifying whether WMR changes the distribution of observable order book states, the way queues evolve conditional on those states, and—if measurable—the relationship between queue state and passive-order fill probability.