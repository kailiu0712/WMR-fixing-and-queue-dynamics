# WMR fix: passive fill probability, adverse selection, and predictive decision value (CME 6E)

**Code:** `project/wmr_value/` (`run.py` reproduces everything in about 5 minutes once the observation cache exists).
**Tests:** `wmr_value/tests` (4 tests: features ignore post-entry messages, WMR window in London time, markout decomposition and passive-fill identity, chronological split).
**Outputs:** tables in `tables/`, figures F01–F10 in `figures/`, source mapping in `literature_mapping.md`. The section 6 imbalance checks come from `python -m wmr_value.robustness` (tables `R1_*`–`R4_*`, figures R04, R07; about 4 minutes). The section 7 clock complement comes from `python -m wmr_value.clock_run` (about 30 minutes; outputs in `clock_volume/`, `clock_event/`), `clock_compare` (`tables/clock_comparison.csv`) and `clock_pdf` (`report_clocks.pdf`).

## Answer in brief

During the WMR window, historical passive orders at the touch are **both more likely to fill and less adversely selected** than comparable orders in the adjacent windows or in normal liquid hours. This holds after conditioning on queue position, imbalance, recent flow, spread and side. The improvement comes from a smaller adverse mid-price move after the fill, not from capturing more spread.

The effect is small in absolute terms: about +0.1 tick per filled contract, roughly $0.6. A simple WMR-aware model predicts it out of sample only as a **level shift**: it corrects a 3-point under-prediction of WMR fill rates. It does **not** rank opportunities *within* the window any better, and neither does a model trained on WMR rows alone (section 6). Ordinary order-book features rank WMR *fills* about as well as normal-hours fills. They rank WMR *markouts* worse than normal-hours markouts even at equal sample size. **Clock caveat (section 7):** measured per contract traded or per book message instead of per second, most of the fill advantage disappears and the markout-ranking gap vanishes; the markout advantage (less adverse selection) survives every clock.

All results describe historical participant orders under their own cancellation decisions. They are not the outcomes a new quote would have achieved.

## Data and estimand

- **Sample:** CME 6E (6EU6, then 6EZ6 from 2026-09-11), 66 London weekdays, 2026-07-14 to 2026-10-13.
- **Universe:** 08:00–17:00 London.
- **WMR window:** 15:57:30–16:02:30 London, converted with time-zone awareness and checked by a unit test.
- **Nearby control:** the two adjacent 5-minute windows.
- **Normal:** all other seconds in the universe.

| Period | Entries | Fills | Days |
|---|---|---|---|
| WMR | 36,588 | 4,766 | 66 |
| Nearby | 65,463 | 6,234 | 66 |
| Normal | 2,624,015 | 172,089 | 66 |

**Observation.** The episode definition is unchanged from the proposal pipeline (`wmr_eda.py`):
- an actual new limit order added at the best bid or ask;
- at most one entry per second per side, to thin overlapping episodes;
- label = any passive execution within 5 s before the order exits;
- a cancel, any modify, or reaching 5 s ends the episode with label 0;
- episodes ending in a book reset are excluded.

**Two corrections to the inherited pipeline:**
1. **Features use only completed packets before the entry packet**, plus the order's own size and queue-ahead at insertion. The proposal's cached features used the book after the entry packet, which can include messages that arrived after the order.
2. **Aggressive executions removed from the fill label.** About 3% of the proposal's "fills" (2.5% in WMR) were participants repricing a resting order *across the spread*. The order id shows up as the aggressor in a T record, then gets an F at a different price. These are taking liquidity, not passive fills; they are now voluntary exits with label 0. After the fix, every fill executes at the order's limit price (unit-tested).

**Markout.** M_h = s(m_{τ+h} − p), in ticks (1 tick = 0.00005 = $6.25), with τ the first-fill packet. It splits into:
- spread capture = s(m_pre − p);
- post-fill mid move = s(m_{τ+h} − m_pre).

**Payoff.** Z_h = Y·(M_h − c), with Z = 0 when unfilled. Costs are **not verified**: c = 0 is the gross case, and c = 0.1 and 0.3 ticks are illustrative.

**Inference.** The trading day is the unit. WMR minus control is computed per day, then tested with a sign-flip permutation p-value and a day-bootstrap 95% CI. Orders are never resampled individually.

## 1. Research question and contribution

The earlier first pass established two things:
- WMR is busier;
- liquidity at the touch fades about half as much after trades.

This phase asks whether that difference changes the **economics of passive liquidity provision**: fill probability, post-fill markout and their trade-off. It also asks whether the change is predictable out of sample, and whether predicting it improves a simple selection objective.

The work is an **extension** of the cited literature, not a replication of any paper.

## 2. Literature comparison

| Finding | Paper | Comparable? | Relationship | Verdict |
|---|---|---|---|---|
| Fill probability and post-fill markout move together across regimes (WMR better on both) | L1 (fill vs return dilemma) | No: L1 uses experimental orders with no cancellation policy; ours are historical orders | Adaptation to a different dataset and market | Between regimes there is **no dilemma**: WMR is better on both axes. Within normal hours the conditional markout is negative in deciles 1–8 and positive only in the top predicted-fill decile (front-of-queue / price-improving entries). That is in line with L1's front-of-queue result, but it is not a test of their trade-off. |
| Queue-ahead lowers fill and worsens markout (A2 strata) | L2 | Partly | New empirical extension | Consistent with queue position mattering for adverse selection. In normal hours, entries with zero queue-ahead earn +0.038 ticks; those with more than 40 contracts ahead earn −0.102 ticks. |
| Lower post-fill adverse move inside the fix window | L3, L4 | No | Not directly comparable; motivation only | We test seconds-scale markouts in one futures contract. Their return patterns are hours-scale, in spot, over long samples. We make no reversal claim; the first pass found none in 6E. |
| Results survive conditioning on book state | L5 | Partly | Motivation for the conditioning design | WMR effects persist after matching on queue, imbalance, flow, spread, side and activity. |

## 3. Experiment A: economic findings

**Unconditional, paired by day over 66 days** (`A1_tests.csv`, F01–F03):

| | WMR | Nearby | Normal | WMR − nearby | WMR − normal |
|---|---|---|---|---|---|
| 5 s fill frequency | 12.9% | 9.3% | 6.5% | +3.6 pts [2.8, 4.4] | +6.4 pts [5.4, 7.5] |
| 5 s markout given fill (ticks) | +0.066 | −0.013 | −0.033 | +0.079 [0.045, 0.111] | +0.099 [0.073, 0.124] |
| Share of fills with 5 s markout < 0 | 43.7% | 48.2% | 47.4% | −4.5 pts | −3.7 pts |
| Spread capture (ticks) | 0.504 | 0.510 | 0.512 | −0.007 | −0.008 |
| Post-fill mid move, 5 s (ticks) | −0.438 | −0.523 | −0.544 | +0.085 | +0.106 |
| Z_5 per opportunity (ticks) | +0.0086 | −0.0002 | −0.0022 | +0.0088 | +0.0108 |

- **Significance:** every difference above has p < 0.001.
- **Spread capture vs mid move:** spread capture is essentially identical across periods (WMR is even slightly lower). The WMR advantage is **entirely a smaller adverse mid move after the fill**: less adverse selection, not better prices.
- **Robustness:** the 1 s horizon gives the same picture (+0.084 / +0.098). Leave-one-day-out estimates stay within 0.074–0.083 ticks (vs nearby). Excluding orders whose outcome crosses 16:02:30 changes nothing (+0.081 / +0.101).
- **Side and horizon:**
  - At 5 s the effect is larger on the ask side (+0.11 / +0.13) than on the bid side (+0.06 with p = 0.10 / +0.07 with p = 0.01).
  - At 30 s it splits by side: ask +0.28 / +0.24; bid −0.08 / −0.03, not significant. The pooled 30 s effect (+0.09; p = 0.047 nearby, 0.003 normal) is therefore direction-dependent and probably reflects price drift in the sample's fix days. We do not interpret it further.

**Conditional on book state** (`A2_matched.csv`): control cells are reweighted to WMR's mix of queue-ahead × imbalance × 5 s flow direction × spread × side, per day, on common support.

| Comparison | Fill | 5 s markout given fill |
|---|---|---|
| WMR − normal | +6.5 pts [5.6, 7.5], 97% of days | +0.106 [0.079, 0.132], p < 0.001, 94% of WMR fills matched |
| WMR − nearby | +3.9 pts [3.1, 4.7] | +0.058 [−0.013, 0.125], **p = 0.13** |

- **The nearby markout comparison is under-powered.** Only 34% of WMR fills fall in cells where the nearby windows have enough fills.
- **Adding 30 s traded volume as a stratifier:**
  - vs normal: fill +5.0 pts, markout +0.083, both p < 0.001;
  - vs nearby: markout +0.041, p = 0.40.
- **Matching instead on the out-of-sample predicted fill decile** (46 walk-forward days):
  - WMR fills **more often than predicted**: +2.8 pts vs normal, +1.6 pts vs nearby;
  - and still has a better markout: +0.092 [0.062, 0.123] vs normal, +0.070 [0.017, 0.121] vs nearby (p = 0.012).

**Frontier (F04).** The WMR curve lies above the normal curve in every predicted-fill decile with enough fills. It also reaches higher realized fill rates in the top deciles (42% realized vs 35% predicted). So WMR shifts the observed fill–markout relation **up and to the right**. There is no evidence of a worse trade-off.

**Costs.**
- At c = 0.1 tick, WMR's net Z is −0.004, still better than both controls (p < 0.01).
- At c = 0.3 tick, WMR is **worse** than normal (−0.030 vs −0.022). It fills more, and each fill loses money.

The gross edge is about 0.07 ticks (about $0.41) per filled contract. Whether that survives depends entirely on a participant's fees, which we did not verify.

## 4. Experiment B: predictive findings

- **Models:**
  - M0 / R0: L2 logistic regression / Ridge on 20 pre-entry features.
  - M1 / R1: adds a WMR indicator and three interactions (× log queue-ahead, × imbalance, × 5 s flow).
- **Split:** chronological by day. 40 train days (Jul 14–Sep 7), 13 validation days (Sep 8–24), 13 untouched test days (Sep 25–Oct 13, all 6EZ6).
- **Hyperparameters:** chosen on validation only. Logistic regression picked the weakest penalty in the grid (C = 1), as expected with about 1.7M training rows.

**Fill (test days; `B_fill_metrics.csv`, F05, F06).** The test set has 7,144 WMR opportunities with 14.4% prevalence.

| | Predicted fill rate | Calibration intercept | Brier score |
|---|---|---|---|
| M0 | 11.1% | 0.40 | 0.0927 |
| M1 | 13.8% | 0.06 | 0.0909 |

- **M0 under-predicts WMR fills** on 12 of 13 test days (p = 0.001). M1 removes the bias, and its calibration slope is about 1.0.
- **The Brier gain is +0.0018 per day** [0.0006, 0.0030], about 2% relative, on 10/13 days (p = 0.017). Walk-forward over 46 days gives +0.0011 [0.0005, 0.0018] (F10).
- **It is fragile:** on a time-spaced subsample (at most one entry per side per 10 s) the gain is +0.0020 [−0.0006, 0.0049], p = 0.19.
- **No ranking gain:** AUC does not improve (0.820 vs 0.818). Normal hours are unaffected. The nearby windows get slightly *worse* log loss (−0.0008 per day), because M1 shifts the shared coefficients a little.
- **Interpretation:** the WMR gain is a **level effect**. The WMR intercept is stable across training halves (0.053 / 0.049 standardized). The interactions are small, and the queue-ahead interaction is unstable (−0.017 vs −0.004).

**Markout given fill (`B_markout_metrics.csv`, F07).**
- **Barely predictable:** test correlation is 0.13 at 1 s, 0.06 at 5 s and 0.03 at 30 s, with a 5 s MAE of 0.78 ticks.
- **R0 under-predicts WMR markouts** by 0.11 ticks (p = 0.006, 11 of 13 days). R1 halves that bias, but its MAE gain is negligible (+0.0055 ticks, p = 0.09).
- **Ranking within WMR is weaker than in normal hours,** and not only because WMR has fewer fills. F07 cuts deciles on the pooled scores, which leaves WMR bins very uneven. R07 redraws it with deciles cut within each period, and section 6 compares WMR with normal-hours samples of the same size.
- **Payoff Z:** neither the two-stage V̂ = p̂·μ̂ nor a direct Ridge on Z beats a constant in mean squared error (0.11843 vs 0.11846). Payoff per opportunity is dominated by noise.

## 5. Experiment C: decision implications

Each policy accepts the top 10/20/30% of opportunities by its score. Thresholds come from validation days and are evaluated on the same held-out test observations. Gross Z_5, c = 0 (`C_policies.csv`, `C_paired.csv`, F08, F09).

- **Scores rank opportunities usefully.**
  - P3 (joint p̂·μ̂) beats accept-all by +0.011 / +0.0075 / +0.0056 ticks per accepted opportunity (p < 0.001, at least 12/13 days).
  - P3 beats fill-only P1 at the 20% and 30% budgets (+0.003, +0.0043; p = 0.03, 0.009).
  - P3 is roughly tied with markout-only P2.
- **P4 (WMR-aware) vs P3 with one global threshold:**
  - +0.0008 per accepted opportunity on test days (p = 0.10); +0.0011 walk-forward (41 days, p = 0.002).
  - The whole gain comes from **reallocating the budget toward WMR**: P4 accepts 89–95% of WMR opportunities, while P3 accepts 29–47%.
- **P4 vs P3 within WMR** (thresholds set inside the period, matched budget): −0.009, +0.007, +0.010 (p = 0.42, 0.36, 0.08). **No reliable improvement.**
  - Inside WMR, the ordinary-feature scores already lift Z from 0.017 (accept-all) to about 0.07 at the top 10%: 49% fill rate, +0.15 ticks per fill.
  - F08 shows the same pattern by decile. In P3's top WMR value decile, realized Z is 0.10 ticks against a predicted 0.044: ordinary features rank WMR correctly but under-state its level. P4's top-decile level is closer (0.086 realized vs 0.058 predicted), but its ordering within WMR is no better.
- **With costs:**
  - At c ≥ 0.1 tick almost every predicted V̂ is negative, so ranking degenerates into avoiding fills. Every policy has negative realized Z.
  - The best WMR decile (+0.15 ticks per fill) would clear a 0.1-tick cost but not 0.3.

**What this does not establish.** These are rankings of **historically observed** orders, scored with their realized outcomes under the participants' own cancellation decisions. A policy that "keeps" an order its owner cancelled has no observed outcome. A new order would change queue positions and possibly behaviour. None of this is a live or counterfactual market-making P&L. The optional FIFO-replay counterfactual (C4) was not built, because the required safeguards would need a new simulation engine.

## 6. Robustness to period imbalance

WMR is 1.3% of observations (36,588 of 2.73M). Fills are left at their natural prevalence; there is no resampling or class weighting. Fill models are judged by Brier score, log loss, calibration and PR-AUC.

The period imbalance affects the three experiments differently:
- **Experiment A** works per day and per period, so the imbalance costs precision, not bias.
- **Experiment B** is exposed: about 96% of the pooled training loss comes from normal-hours rows.
- **Experiment C** was already re-run with thresholds set within each period.

Four checks (`wmr_value/robustness.py`):

**R1. WMR-specific models** (`R1_wmr_models.csv`, `R1_wmr_day_tests.csv`, `R1_wmr_policy.csv`). The same feature set is fitted on WMR rows only (22,409 training rows, 2,996 fills), and on WMR plus nearby rows. Hyperparameters are chosen on WMR validation rows (C = 0.1, alpha = 1). Each is compared with the pooled M1/R1 on WMR rows. Positive differences mean the local model is better.

| WMR-only vs pooled M1/R1 | Test days (13) | Walk-forward (46 days) |
|---|---|---|
| Brier score | +0.0002 [−0.0006, 0.0009], p = 0.69 | −0.0001 [−0.0006, 0.0004], p = 0.84 |
| AUC | +0.001, p = 0.58 | −0.0006, p = 0.62 |
| 5 s markout MAE (ticks) | −0.0008, p = 0.68 | −0.0019, p = 0.26 |

- The WMR + nearby model gives the same picture.
- Selection within WMR is no better either. WMR-only p̂·μ̂ minus P4, at the top 10/20/30%: +0.0015, −0.0095, −0.0077 (p = 0.92, 0.21, 0.04).
- **Pooled training is not hiding WMR structure that these features and models could pick up.** The limit is the features, not the shared coefficients.

**R2. Equal-size evaluation** (`R2_equal_n.csv`). Normal-hours samples are built from whole random 5-minute windows on each day until they reach WMR's row count (for fill metrics) or fill count (for markout metrics) on that day. There are 500 draws, using the same pooled models.

| Pooled model R1, walk-forward (3,229 WMR fills) | WMR | Normal, equal size: median [95% range] | Draws below WMR |
|---|---|---|---|
| Markout correlation | 0.011 | 0.064 [0.013, 0.104] | 1.6% |
| Monotonicity of markout deciles (Spearman) | 0.21 | 0.81 [0.40, 0.96] | 0% |

- **Markouts.** The low WMR position repeats on every variant: R0 walk-forward, correlation 0.033 (13% of draws below) and Spearman 0.42 (3%); R1 test days, 4% and 0.4%. A WMR-only model reaches correlation 0.027 and Spearman 0.52 walk-forward, still below the normal median.
  - So F07's flatter WMR curve is **partly** a sample-size effect: at WMR's test size, normal hours' decile Spearman drops from 0.96 to a median of 0.64.
  - It is **not only** that. The markout signal in these features is genuinely weaker during the fix.
- **Fills.** There is no consistent gap. WMR AUC is above the normal equal-size range on test days (0.818 vs [0.762, 0.806]) and slightly below it walk-forward (0.797 vs [0.800, 0.824]).

**R3. Precision-weighted day aggregation** (`R3_weighted_day_tests.csv`). Each day is weighted by 1 / (1/n_WMR + 1/n_control), where n is the day's row count or fill count, instead of weighting days equally.
- The fill-rate, 1 s and 5 s markout, and Z_5 differences move by less than 0.01 tick and stay at p < 0.001.
- The 30 s markout vs nearby moves from +0.092 to +0.080 and weakens from p = 0.047 to p = 0.097.
- No single day carries more than 3.8% of the weight.

**R4. Within-period deciles** (`R4_*.csv`, figures R04, R07).
- Redrawn this way, F04 still shows the WMR markout above normal at every predicted-fill level.
- WMR's realized fill rate exceeds M0's prediction in every decile above about 4% predicted fill, by up to 8 points: 53.5% vs 45.3% in the top decile. The two lowest deciles are slightly below (−0.3 and −0.2 points).

**Sampling rule.** The one-entry-per-second-per-side cap keeps a similar share of best-price adds in each period: 7.2% WMR, 6.9% nearby, 10.1% normal (checked on 5 days). It does not thin WMR differently from the controls.

## 7. Volume and event clocks (complement to the 5 s clock)

**Design** (`clocks.py`, `clock_run.py`):
- **Volume clock:** contracts traded (all T messages). **Event clock:** book messages (non-snapshot A, C, M, T, F).
- **Calibration:** each clock advances on average at wall-clock speed in normal hours on the 40 training days: 1 s = 2.42 contracts = 29.7 messages, so a 5 s-equivalent is 12.1 contracts or 149 messages.
- **Label:** same exit rule; a fill counts if its packet starts before the clock has advanced one 5 s-equivalent from entry. **Markout at h:** the mid once the clock has advanced h-equivalents past the fill packet.
- **Check:** with the wall clock, the same code reproduces the 5 s observation table exactly on all 2,726,066 rows (labels, markouts, spread capture, fill times, flags). No row is censored on either new clock.
- The fix trades 4.9× as many contracts per second as normal hours (13.8 vs 2.8) and sends 2.6× as many messages. Median wall time of the 5 s-equivalent fill horizon, WMR vs normal: 2.2 s vs 9.1 s (volume), 2.1 s vs 5.2 s (event).

**Finding 1, fill rate: mostly clock speed.** WMR minus normal fill rate, day-paired:

| | 5 s | Volume | Event |
|---|---|---|---|
| Day-paired | +6.4 pts | +2.0 [1.4, 2.8] | +3.8 [3.0, 4.6] |
| Book-state matched | +6.5 | +2.5 | +4.1 |
| Matched on predicted fill | +2.8 | +0.4 (p = 0.20) | +1.6 |

**Finding 1, markout: robust.**
- WMR minus normal markout per fill: +0.099 (5 s), +0.086 [0.064, 0.107] (volume), +0.076 [0.051, 0.100] (event). All p < 0.001.
- Book-state matched: +0.106 / +0.077 / +0.079.
- Spread capture is equal on every clock (within 0.008 ticks); the gain is all in the post-fill move.
- Against the adjacent windows: +0.087 (volume) and +0.069 (event).
- The negative-markout share is 6 points lower in WMR on the new clocks (4 points on the 5 s clock).
- Precision weighting changes none of this.

**Finding 2: holds, with a smaller level gap.**
- M0 under-predicts WMR fills by 1.9 points on both new clocks (3.3 on the 5 s clock).
- WMR terms still improve WMR Brier score (p = 0.019 and 0.012).
- WMR-only models still add nothing (walk-forward p = 0.08 and 0.77).

**The markout-ranking gap is a 5 s-clock effect.**
- R0 correlation with realized markouts, WMR vs normal: 0.098 vs 0.101 (volume), 0.075 vs 0.115 (event), against 0.028 vs 0.059 on the 5 s clock.
- At equal size, the share of normal samples below WMR:
  - markout correlation: 46–92% (volume), 10–49% (event), against 1.6–16% on the 5 s clock;
  - decile Spearman: 23–37% (volume), 2–87% (event).
- Reading: the book predicts the move over the next dozen contracts about equally well in both regimes. A 5 s window at the fix simply holds about five times as much trading.
- The earlier claim that WMR markouts are intrinsically harder to predict is withdrawn.

**Decision implications.** Inside WMR at the 20% budget:
- The ordinary joint score P3 loses to fill-only P1: −0.009 (volume, p = 0.11) and −0.014 (event, p = 0.010).
- The WMR-aware P4 recovers +0.016 (p = 0.015) and +0.017 (p = 0.007) over P3, where the 5 s clock showed no gain.
- The cause is R0's pessimistic WMR markout level (bias −0.06 to −0.09 ticks), which pushes out high-fill orders.
- In normal hours P3 beats P1 on every clock (p ≤ 0.011).

**Caveats:**
- The calibration (normal-hours mean rate) is one reasonable choice among several.
- The event clock counts all messages, including cancellations away from the touch.
- Features are still measured in seconds.

## 8. Limitations

- **Selection bias from historical policy:**
  - fill labels and markouts are conditional on participants' placement and cancellation decisions (L1's "signal bias");
  - markout regressions train on filled orders only.
- **Sample size:**
  - 66 fixing days in all; only 13 untouched test days, so WMR-specific test comparisons have low power;
  - 3 months, two contract months, summer time only.
- **Market:** CME futures, not the spot venues the WMR benchmark uses.
- **Not observed:** trader identities, inventories, client fixing exposure.
- **Dependence:** order observations are strongly correlated within days. All inference is day-level. The time-spaced check weakens the fill-calibration result.
- **Costs and latency:** costs are not verified, and latency and queue-priority effects on a new order are ignored.
- **Episode definition:** inherited conventions (any modify ends the episode; one entry per second per side). Markouts are anchored at the first fill packet.

## 9. Conclusion

**Does the WMR regime meaningfully change the economics of passive liquidity provision?**

Yes, in direction, and modestly in size:
- historical passive orders at the touch fill more often and are less adversely selected during WMR;
- this holds at the same queue, imbalance, flow, spread and predicted fill probability;
- the gain comes from smaller adverse price moves after fills, not from spread capture.

This fits the first-pass finding that liquidity fades less after trades at the fix: makers who expect fix flow to be uninformed are, on average, right.

**Is the change predictable enough to improve a practical decision metric?**

Only weakly:
- a WMR indicator corrects a systematic under-prediction of fill probability (better calibration, no better discrimination);
- WMR-aware selection helps only by shifting more of the budget into the window;
- it does not rank opportunities better within the window, and neither does a model trained on WMR alone; markouts during the fix are less predictable from book features than in normal hours;
- the gross edge (about 0.1 tick per fill) is small relative to plausible trading costs.

The defensible claim is a **regime-level difference in adverse selection**, not a tradable signal.
