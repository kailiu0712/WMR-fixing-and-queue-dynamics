# 22-day versus expanded 6E WMR EDA

The expanded UTC series covers 2026-07-14 through 2026-10-13. It uses 6EU6 through 2026-09-10 and 6EZ6 from 2026-09-11. The 22-day baseline covers 2026-09-14 through 2026-10-13.

Processed UTC files: 30 to 92; MBO rows: 47,088,610 to 122,903,237; WMR fixing days: 22 to 66.

## Main changes

Absolute WMR depth and activity shifted after adding July–August: top-five depth went from 1,561 to 1,342 contracts, executed volume from 14.68 to 13.93 contracts/s, additions from 151.5 to 133.5 contracts/s, and cancellations from 113.2 to 101.7 contracts/s.

Relative WMR-versus-nearby contrasts are listed below; execution, addition, and cancellation uplifts strengthened, while the top-five depth uplift eased. The spread contrast stayed small.

Observed five-second first-fill probability changed from 12.7% to 13.4% in WMR and from 9.8% to 9.7% nearby.

## Day-equal means

| Metric | Window | 22 days | Expanded | Change |
|---|---|---:|---:|---:|
| spread_ticks | wmr | 1.118 | 1.103 | -1.3% |
| spread_ticks | nearby | 1.133 | 1.126 | -0.6% |
| depth1 | wmr | 182.403 | 149.032 | -18.3% |
| depth1 | nearby | 140.709 | 114.776 | -18.4% |
| depth5 | wmr | 1561.453 | 1342.347 | -14.0% |
| depth5 | nearby | 1256.988 | 1117.463 | -11.1% |
| adds | wmr | 151.494 | 133.473 | -11.9% |
| adds | nearby | 106.707 | 90.541 | -15.1% |
| cancels | wmr | 113.238 | 101.717 | -10.2% |
| cancels | nearby | 90.354 | 77.599 | -14.1% |
| executions | wmr | 14.680 | 13.934 | -5.1% |
| executions | nearby | 7.023 | 6.440 | -8.3% |
| abs_imbalance | wmr | 0.399 | 0.392 | -1.8% |
| abs_imbalance | nearby | 0.421 | 0.419 | -0.5% |
| best_order_count | wmr | 48.471 | 44.211 | -8.8% |
| best_order_count | nearby | 40.467 | 36.952 | -8.7% |

## WMR relative to nearby

| Metric | 22-day uplift | Expanded uplift |
|---|---:|---:|
| spread_ticks | -1.4% | -2.0% |
| depth1 | +29.6% | +29.8% |
| depth5 | +24.2% | +20.1% |
| adds | +42.0% | +47.4% |
| cancels | +25.3% | +31.1% |
| executions | +109.0% | +116.4% |
| abs_imbalance | -5.2% | -6.4% |
| best_order_count | +19.8% | +19.6% |

## Observed five-second first-fill rates

| Period | 22 days | Expanded | Change |
|---|---:|---:|---:|
| nearby | 9.8% (21,552) | 9.7% (65,463) | -0.06 pp |
| normal | 6.4% (1,775,327) | 5.7% (5,069,267) | -0.69 pp |
| wmr | 12.7% (12,078) | 13.4% (36,588) | +0.68 pp |

## Replay diagnostics

Unknown-order references: 0 to 0.
Trailing incomplete packet rows: 0 to 0.

These comparisons are descriptive. The baseline is contained in the expanded sample; the change is a sample-composition change, not a paired treatment effect. Observed-order fill labels reflect owners' cancellations and repricings.
