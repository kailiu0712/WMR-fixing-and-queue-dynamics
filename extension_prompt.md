# Coding Agent Task: Phase 3 — WMR Liquidity Provision, Fill Quality, and Predictive Decision Value

## 1. Research objective

Extend our existing CME Euro FX (6E) market microstructure project with three connected experiments:

**A. Economic Evidence — Fill Probability vs. Adverse Selection**

Does the WMR fixing window offer a different trade-off between passive-order execution probability and post-fill returns?

**B. Machine Learning — Can We Predict These Differences?**

Do WMR-specific features improve out-of-sample fill-probability calibration and conditional maker-markout prediction beyond ordinary queue and order-flow features?

**C. Decision Value — Can These Predictions Improve Quote Selection?**

Can combining fill-probability and post-fill-return forecasts identify historically more favorable opportunities to provide liquidity?

The intended research narrative is:

1. WMR exhibits unusually concentrated trading activity.
2. Our earlier analysis suggests that liquidity fades less after trading shocks during WMR.
3. We investigate whether this difference creates economically meaningful opportunities for passive liquidity provision, whether these opportunities are predictable, and whether the predictions improve a clearly defined decision objective.

**Do not assume Step 3 is true.** Lower fade may coexist with more toxic executions, and higher fill probability may coincide with worse post-fill returns.

Our goal is a scientifically defensible result, including a meaningful negative result if WMR-specific modeling adds little value.

### Existing project and data

Processed L3 data:

`project/data_6e/l3`

Reuse the existing order-book reconstruction, event processing, fill-label generation, fade analysis, and EDA code wherever correct.

Do not recreate the ingestion pipeline or refactor working code unnecessarily.

Start by inspecting the existing code and outputs to determine which metrics and labels can be reused safely.

Use the official WMR window:

**15:57:30–16:02:30, Europe/London**, with timezone-aware conversion.

Use nearby non-WMR windows as primary controls and broader liquid trading hours as secondary comparisons.

---

## 2. Literature: verifiable sources and precise relationship to our experiments

Read the relevant sections of the following papers before implementation.

Create a short `literature_mapping.md` identifying exactly what each paper establishes, what our project borrows, and how our experiment differs.

### [L1] Albers, Cucuringu, Howison & Shestopaloff (2025, revised November 2025)

**The Market Maker's Dilemma: Navigating the Fill Probability vs. Post-Fill Returns Trade-Off**

Primary source: https://arxiv.org/abs/2502.18625

Full text: https://arxiv.org/html/2502.18625v2

**Relevant sections:** Introduction, Sections 3–7, especially the distinction between fill likelihood and post-fill return, queue position, cancellation-policy selection bias, and the economic performance of maker orders.

**Direct connection to our work:**

- Experiment A examines the same broad economic trade-off: fill probability versus post-fill return.
- Experiment B uses interpretable models to predict the components of that trade-off.
- Experiment C evaluates whether those predictions can improve an order-selection objective.

**Important distinction: This is NOT a replication.**

The paper studies experimental maker orders on Binance BTC perpetual futures, with a controlled placement/cancellation approach. We study historical participant orders on CME 6E during the WMR fix.

Historical orders introduce submission-policy and cancellation-policy selection bias. Our fill labels are not directly equivalent to theirs.

**Research question:** Does the fill–post-fill-return trade-off documented in the paper also appear in 6E, and does its shape differ specifically around WMR?

Do not assume the paper's results transfer unchanged across venues or instruments.

### [L2] Garriott, van Kervel & Zoican (2025)

**Queuing and Inventories in Limit Order Markets**

Journal of Financial Markets, 75, 100982.

DOI: https://doi.org/10.1016/j.finmar.2025.100982

**Connection:**

Queue position affects adverse selection, inventory management, and the economics of liquidity provision.

Our results should examine whether queue-ahead and imbalance influence both fill probability and maker markout, rather than treating queue position as merely an execution-speed feature.

**Relationship:** Theory-guided empirical extension, not an exact test or reproduction of their equilibrium model.

We cannot observe trader inventories or infer participants' private objectives.

### [L3] Krohn, Mueller & Whelan (2024)

**Foreign Exchange Fixings and Returns around the Clock**

Journal of Finance, 79(1), 541–578.

DOI: https://doi.org/10.1111/jofi.13306

Working paper: https://www.bankofcanada.ca/2021/10/staff-working-paper-2021-48/

**Connection:**

Documents systematic FX price movements and reversals around benchmark fixings, with evidence consistent with inventory and order-flow mechanisms.

This motivates the possibility that maker markouts and short-term price responses behave differently near WMR.

**Relationship:** Economic motivation and cross-market extension.

Their primary setting concerns spot FX, broader currencies, and long historical periods. Our setting is a single month of CME EUR/USD futures L3 data.

Do not claim that our results reproduce their return-reversal findings unless an explicitly comparable return test is performed.

### [L4] Muhle-Karbe, Oomen & Weber (2025)

**Hedging of Fixing Exposure**

Mathematical Finance, 35(4), 818–840.

DOI: https://doi.org/10.1111/mafi.12464

**Connection:**

Develops a theoretical model of execution timing and transient/permanent market impact when dealers hedge benchmark fixing exposures.

It motivates why fixing-related demand and execution incentives may affect short-term liquidity and price dynamics.

**Relationship:** Conceptual economic motivation only.

Our experiments do not observe client fixing exposures or estimate their optimal hedging strategies. We are not replicating this model.

### [L5] Huang, Lehalle & Rosenbaum (2015)

**Simulating and Analyzing Order Book Data: The Queue-Reactive Model**

Journal of the American Statistical Association, 110(509), 107–122.

DOI: https://doi.org/10.1080/01621459.2014.982278

**Connection:**

Order-flow event intensities depend on the prevailing order-book state.

This supports conditioning on queue depth, imbalance, and recent flow when comparing WMR with ordinary periods.

**Relationship:** Motivation for the feature representation and conditional analysis, not an attempt to estimate their full queue-reactive model.

### [L6] Official technical and benchmark specifications

Databento MBO:
https://databento.com/docs/schemas/mbo

CME GLBX.MDP3:
https://databento.com/docs/knowledge-base/datasets/glbx-mdp3

Order tracking:
https://databento.com/docs/examples/order-book/order-tracking

WMR methodology:
https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/wmr-fx-methodology.pdf

Use these to confirm event semantics, order priority, trading quantities, timestamps, and benchmark-window definitions.

**Literature reporting rule:** For every substantive comparison in the final report, explicitly classify the relationship as:

- Direct conceptual test of a published result;
- Adaptation to a different dataset or market;
- New empirical extension motivated by the literature;
- Or a result that is not directly comparable.

Never claim a literature result has been replicated merely because our plots look similar.

---

## 3. Common experimental protocol

Use one consistent observation definition across Experiments A–C.

### 3.1 Observations

Primary unit: an actual passive order-entry episode at the best bid or ask.

If a modification loses priority and creates a new queue-entry episode, handle it consistently with CME rules.

Features must reflect the observable book state **before the entry event**.

Use the same eligible observations, historical order lifecycle convention, and five-second label as the existing proposal wherever possible.

Avoid duplicate or overlapping episodes from the same order without a documented reason.

Record:

- Trading date and instrument.
- Entry timestamp.
- Side: passive bid or passive ask.
- Entry price and own order size.
- Queue-ahead contracts and order count.
- Same-side/opposite-side depth.
- Spread, imbalance, and recent order-flow metrics.
- WMR indicator and time relative to fixing.
- Whether the observation has valid follow-up data.

### 3.2 Fill target

Primary target:

\[
Y_i^{fill}
=
\mathbf 1\{
\text{any passive execution within 5 s before exit}
\}.
\]

Partial fills count.

An order cancelled or repriced before execution has outcome zero under the project's historical observed-execution-before-exit definition.

A valid unfilled order at the five-second deadline also has outcome zero.

Exclude observations without adequate follow-up due to feed gaps, resets, or data boundaries.

**Crucial:** This target is the fill probability under actual participants' historical placement and cancellation decisions. It does not represent the probability that an arbitrary new quote would fill if held for five seconds.

Keep this distinction throughout the report.

### 3.3 Maker markout

For a filled order, define the signed markout at horizon \(h\):

\[
M_{i,h}
=
s_i\left(m_{\tau_i+h}-p_i^{fill}\right),
\]

where:

- \(s_i=+1\) for a passive buy;
- \(s_i=-1\) for a passive sell;
- \(\tau_i\) is the fill timestamp;
- \(p_i^{fill}\) is the execution price;
- \(m_t\) is the market midprice.

Positive markout benefits the liquidity provider.

Use **1, 5, and 30 seconds after fill**, with five seconds as the primary horizon.

Report values in ticks, and optionally USD per contract after checking the contract specifications.

Decompose markout into:

\[
M_{i,h}
=
s_i(m_{\tau_i}-p_i^{fill})
+
s_i(m_{\tau_i+h}-m_{\tau_i}).
\]

The first component represents the immediate execution-price advantage relative to mid; the second measures subsequent signed midprice movement.

This prevents spread capture from being mistaken for a lack of adverse selection.

Handle partial fills consistently. Prefer quantity-weighted execution prices and report any restriction to first partial fills.

Do not use future midprices to construct explanatory features.

### 3.4 Observation windows

Compare:

1. WMR fixing window.
2. Nearby five-minute non-WMR control windows.
3. Broad normal trading periods as secondary evidence.

Use the entry timestamp to classify the observation.

If a WMR order fills or reaches its markout horizon after 16:02:30, retain it if follow-up is valid, but flag that the outcome crosses the window boundary.

Perform a sensitivity check excluding boundary-crossing outcomes.

Do not redefine WMR membership using information only known after the order was submitted.

---

## 4. Experiment A — Economic evidence: Fill probability vs. maker markout

### Research question

**Is the relationship between fill probability and post-fill price performance different during WMR?**

This is our central economic experiment.

We are inspired by [L1] but test a new market and a new conditioning variable: benchmark-fixing regime.

### A1. Start with descriptive evidence

For WMR and normal periods, separately by passive bid and ask, calculate:

- Historical five-second fill frequency.
- Time-to-first-fill distribution.
- Mean and median signed maker markout conditional on fill.
- Fraction of fills with negative markout.
- Distribution of signed post-fill midprice movement.
- Optional net markout after a clearly documented fee assumption.

Show results at 1-, 5-, and 30-second horizons.

Do not hide unfavorable results using only mean statistics.

### A2. Control for observable queue state

At minimum stratify by:

- Queue-ahead volume.
- Best-level imbalance.
- Recent signed aggressive order flow.
- Spread.
- Side.

Use a small number of bins and report sample sizes.

The main comparison is **conditional WMR vs. normal behavior**, not merely unconditional averages.

For example, determine whether passive bid orders with similar queue-ahead and imbalance have different fill frequencies or markouts in WMR.

### A3. Construct a fill–markout frontier

Construct an empirical plot with:

- X-axis: estimated five-second fill probability.
- Y-axis: realized average maker markout conditional on fill.

Use bins defined by observable pre-entry characteristics or strictly out-of-sample predicted fill probabilities from Experiment B.

Never use realized fill outcomes to construct the predictor bins.

Draw separate WMR and normal curves, with day-level uncertainty where feasible.

Mark each point with sample count or include counts in an accompanying table.

Interpret carefully:

- A higher conditional markout at comparable fill probability is consistent with more attractive historical passive-execution opportunities.
- A higher fill probability accompanied by more adverse markout may indicate a worse trade-off.
- A higher unconditional fill rate alone is not evidence of better market-making economics.

Do not describe the curves as causal or structural frontiers. They are observational conditional summaries.

### A4. Measure a simple opportunity-level value

Define:

\[
Z_{i,h}
=
Y_i^{fill}
\left(
M_{i,h}-c
\right),
\]

where \(c\) is a documented per-execution cost in matching units.

For unfilled historical orders, set \(Z=0\) under this simplified metric.

If costs cannot be verified, use gross markout and explicitly report that it excludes fees.

Estimate:

\[
E[Z_h\mid X_t,\text{period}].
\]

This is the **observed-policy expected fill-weighted markout**, not a complete market-making PnL estimate.

Also report fill counts and adverse-markout risk; do not judge opportunities solely by this mean.

### Main outputs

- Fill probability vs. signed maker markout frontier.
- Maker markout distributions by regime and side.
- Spread-capture versus subsequent-midprice-movement decomposition.
- Conditional WMR-normal differences with uncertainty.

---

## 5. Experiment B — Simple, interpretable machine learning

### Research question

**Do WMR-specific terms improve out-of-sample fill and markout predictions once ordinary order-book state is controlled for?**

The key point is incremental predictive value.

Different unconditional distributions do not necessarily imply different conditional relationships.

### B1. Feature set

Use only information observable at order entry.

Start with:

- Side.
- Queue-ahead volume.
- Queue-ahead order count.
- Own order size.
- Same-side best depth.
- Opposite-side best depth.
- Spread.
- Order-book imbalance.
- Recent signed aggressive volume over 1/5/30 seconds.
- Recent add and non-trade cancellation rates.
- Recent midprice volatility.
- Time since last trade.
- Continuous time-of-day controls.
- WMR indicator.

Keep the feature set relatively small and interpretable.

Reuse already computed features rather than calculating redundant proxies.

For order flow, distinguish a passive bid from a passive ask, using consistent direction conventions.

### B2. Allowed models only

Do **not** use gradient boosting, random forests, XGBoost, LightGBM, CatBoost, transformers, KANs, or complex deep learning architectures.

**Fill prediction:**

- M0: Regularized logistic regression using ordinary features.
- M1: M0 plus WMR indicator and a small number of pre-specified WMR interactions.
- M2: Optional shallow decision tree, maximum depth approximately 3–4.

Suggested interactions:

\[
WMR\times QueueAhead
\]

\[
WMR\times Imbalance
\]

\[
WMR\times RecentAggressiveFlow.
\]

**Markout prediction:**

- R0: Ridge/linear regression using ordinary features, fitted on filled observations.
- R1: R0 plus WMR indicator and the same small set of interactions.

Optionally compare a small MLP if linear models demonstrably miss important nonlinear relationships.

Suggested maximum architecture:

`Input → 16 hidden units → 8 hidden units → Output`

Use weight decay and early stopping.

Do not add an MLP merely to make the project look more sophisticated.

### B3. Two distinct targets

**Target 1:**

\[
p(x)=P(Y^{fill}=1\mid x).
\]

Use logistic regression.

**Target 2:**

\[
\mu_h(x)
=
E[M_h\mid Y^{fill}=1,x].
\]

Use Ridge regression on orders with observed fills and valid markouts.

This second regression is conditional on execution, so its training sample is selected by the historical fill mechanism.

Do not interpret it as the expected return from an arbitrary submitted quote.

Combine predictions:

\[
\widehat V_h(x)
=
\widehat p(x)
\left(
\widehat\mu_h(x)-c
\right).
\]

As an optional diagnostic, fit a direct Ridge model to the full-sample target \(Z_{i,h}\) and compare its out-of-sample predictive accuracy with the two-stage product.

Do not create unnecessary multi-task training infrastructure.

### B4. Evaluation metrics

**For fill classification:**

Primary:

- Brier score.
- Log loss.
- Calibration curves.

Secondary:

- ROC AUC.
- PR AUC.
- Calibration intercept and slope.

Report positive-label prevalence and sample sizes.

Always evaluate WMR and normal periods separately, using the same frozen model.

Report performance improvements against the same baseline on the same dates and observations.

**For conditional markout regression:**

- MAE and RMSE in ticks.
- Bias: mean predicted minus mean actual markout.
- Correlation between predicted and realized markout.
- Predicted-versus-realized markout by quantile.
- Day-level mean prediction bias.

Measure whether WMR-specific features reduce systematic prediction error rather than simply improving pooled performance.

**For combined expected value:**

Evaluate:

\[
\widehat V_h(x)
\]

against the realized observed-policy payoff \(Z_{i,h}\).

Report prediction calibration by score quantile and realized payoff by predicted-value bucket.

Do not equate statistical prediction accuracy with actionable strategy profitability.

### B5. Chronological evaluation

Use complete trading days as units.

Preferred:

- Approximately 60% training days.
- 20% validation days.
- 20% untouched test days.

Maintain chronological order, with a small temporal gap when necessary.

If the available month produces too few fixing days in validation or test, explicitly state that statistical power is limited.

An expanding-window walk-forward evaluation may be added as a robustness check, but the final held-out test should remain untouched during model development.

Use validation only for:

- Model complexity selection.
- Regularization.
- Optional probability calibration.
- Decision thresholds for Experiment C.

Never select hyperparameters based on the final test.

---

## 6. Experiment C — Does prediction improve quote selection?

### Research question

**Can a predicted fill-weighted markout score identify historically more favorable passive-liquidity opportunities than simple decision rules?**

This is the practical-implication experiment.

It must not be presented as a production-ready trading strategy.

### C1. Define simple decision rules

Evaluate a small number of pre-specified policies.

**P0: Fixed baseline**

Accept every eligible historical opportunity in the evaluation universe.

**P1: Fill-only selection**

Accept opportunities with high predicted fill probability.

**P2: Markout-only selection**

Accept opportunities with high predicted conditional maker markout.

**P3: Joint-value selection**

Accept opportunities with high:

\[
\hat V_h
=
\hat p(\hat\mu_h-c).
\]

**P4: WMR-aware joint-value selection**

Same decision rule, but using the models with WMR-specific terms.

This comparison tests whether information about the WMR regime improves economically motivated selection.

### C2. Fair comparison

Select thresholds on validation days only.

For comparability, evaluate policies at the same acceptance fraction or opportunity budget.

For example, compare the top 10%, 20%, and 30% of eligible historical opportunities according to each score.

Keep bid and ask opportunities clearly labeled.

Report:

- Number of accepted opportunities.
- Observed fill frequency.
- Conditional mean maker markout.
- Realized fill-weighted gross markout per selected opportunity.
- Negative-markout frequency.
- Per-day variability.
- Performance separately during WMR and normal periods.

Use the exact same held-out observations for all policies.

### C3. Critical interpretation: observational policy evaluation

These are **historical opportunity-selection experiments**.

We only observe outcomes for actual orders placed by market participants, under their historical cancellation decisions.

If our hypothetical policy would keep an order that the historical participant cancelled, its counterfactual result is unknown.

Likewise, new hypothetical orders may change queue positions, book dynamics, and market reactions.

Therefore:

- Do not claim that P3 or P4 would have achieved the reported returns if deployed live.
- Do not replace unknown outcomes with fabricated fills.
- Do not infer the value of holding an historically cancelled order from subsequent price movement alone.
- Label all results as observational, conditional on the historical order population and its behavior.

We may claim that a model **ranks historically observed opportunities by realized outcome**, if the held-out evaluation supports that claim.

We may not automatically claim that it establishes an implementable optimal quoting or cancellation strategy.

### C4. Optional counterfactual extension

Only if existing project code already supports reliable FIFO queue replay:

Simulate a small, zero-market-impact passive order joining the back of the best-price queue, with configurable latency and a fixed maximum resting time.

Track its hypothetical fill under subsequent observed order-book evolution.

This could reduce historical order-selection bias, but introduces simulation assumptions.

Required safeguards:

- Correct FIFO order priority.
- Correct treatment of trades, fills, and book reductions.
- No double counting of executions.
- Correct handling of queue-ahead cancellations.
- Sensitivity to latency and order size.
- Explicit treatment of implied or hidden liquidity uncertainty.
- No assumption that the added order changes future order flow.
- Conservative fill inference where the event stream is ambiguous.

If these safeguards cannot be established without substantial engineering, **skip the counterfactual extension**.

Do not build a new event-driven backtesting engine for this project.

---

## 7. Critical safeguards against high-frequency research errors

These requirements are mandatory.

### 7.1 Look-ahead bias and feature leakage

All inputs must be known before the hypothetical quoting or historical order-entry decision.

In particular:

- No future trade flow or volume in rolling features.
- No forward-filled state generated using later messages.
- No future realized volatility.
- No normalization using the full day's information.
- No markout, fill-time, or outcome-derived feature.
- No features computed from an event-close state that includes messages after the order's arrival.
- No using full-WMR-window net flow to classify a decision made earlier in that window.

Use backward-looking rolling windows with explicit time boundaries.

Implement a test verifying that changing post-entry messages cannot affect an observation's features.

### 7.2 Event semantics

Verify Databento's CME-specific MBO normalization.

Trade and Fill records must not independently reduce the book when accompanying cancellation/book-update records already account for the reduction.

Use completed-event states where appropriate, respecting `F_LAST`.

Preserve exchange event ordering and documented modification-priority rules.

Match passive fills by actual order ID and distinguish executions from voluntary cancellations.

Do not infer fills from quote touches.

### 7.3 Sample selection and label bias

A historical fill label reflects actual order placement and cancellation policy.

For each dataset, state clearly whether it represents:

- Actual historical resting orders;
- Filled orders only;
- All eligible order-entry opportunities;
- Or independently simulated quotes.

Do not treat these as interchangeable populations.

Markout regression is trained on filled orders, so it is subject to selection.

Report the conditional nature of the estimand rather than treating it as an unbiased forecast of hypothetical quote returns.

### 7.4 Class imbalance

Check fill-label prevalence separately for WMR, controls, and each trading day.

Avoid arbitrary oversampling of positive fills.

Prefer initially training on the natural class distribution.

If class weighting or resampling becomes necessary, explain the effect on predicted probabilities and recalibrate using validation data only.

Never evaluate calibration on an artificially balanced test set.

Do not use classification accuracy as the primary metric.

### 7.5 Distribution shift and model bias

Report performance separately by:

- WMR versus normal.
- Bid versus ask.
- Queue-ahead quantile.
- Spread or imbalance regime.
- Trading day.

Check whether models systematically overpredict or underpredict WMR fills.

Compare predicted versus realized maker markouts by group.

Inspect coefficients and important WMR interactions for stability.

A positive pooled result must not conceal poor performance in the fixing window.

### 7.6 Dependence and statistical inference

One month may contain only around 20 fixing events.

The number of independent days matters more for regime-level inference than the number of order observations.

Use paired day-level performance differences and day-block bootstrap confidence intervals where appropriate.

Do not bootstrap individual orders as though they were independent.

Do not present a tiny p-value based on millions of correlated orders as decisive evidence.

Report uncertainty across dates and identify whether a few unusually active days dominate the outcome.

### 7.7 Overlapping outcomes

Multiple orders can share the same order-book state and overlapping five-second outcome windows.

This can create high dependence and make effective sample size appear much larger than it is.

Use date-level splitting and inference.

Avoid random row-wise train/test splitting.

For important findings, repeat the evaluation with a modest, time-spaced subsample of opportunities to check stability.

### 7.8 Model comparison and overfitting

Use the same candidate observations, features, splitting rules, and evaluation periods for comparable models.

Limit hyperparameter searches.

Report both performance improvement and model complexity.

If a simple logistic regression performs as well as the neural network, prefer logistic regression.

Do not optimize dozens of feature combinations on the test set.

---

## 8. Required figures

Aim for approximately 8–10 high-quality figures rather than a large collection of repetitive charts.

| Figure | Content |
|---|---|
| F01 | WMR vs. normal: fill rates, conditional markouts, and observed payoff |
| F02 | Signed maker markout distributions at 1/5/30 seconds |
| F03 | Spread capture vs. post-fill adverse-selection component |
| F04 | Fill-probability vs. conditional-markout frontier |
| F05 | Logistic model calibration curves, separately for WMR and normal |
| F06 | Model comparison: Brier score and markout error with uncertainty |
| F07 | Actual vs. predicted markout by prediction quantile |
| F08 | Realized observed-policy payoff vs. predicted-value quantile |
| F09 | Policy comparison at matched acceptance fractions |
| F10 | WMR-specific model's incremental value across trading days |

All figures should:

- Have clear axes, units, legends, and sample sizes.
- Use consistent colors for WMR and normal.
- Distinguish observed results from model predictions.
- Show uncertainty where meaningful.
- Report negative results just as clearly as positive ones.
- Avoid implying causal effects or live profitability.

Every important figure should have a short caption explaining what question it answers and what conclusion is and is not justified.

---

## 9. Implementation requirements

Use:

- Python.
- NumPy and Pandas/Polars.
- Scikit-learn.
- Matplotlib.
- PyTorch only if the optional small MLP is justified.

Do not use boosting, ensemble forests, KANs, transformers, reinforcement learning, or complex backtesting frameworks.

Prefer existing project conventions.

Suggested logical modules:

```text
wmr_liquidity_value/
├── config.py
├── build_labels.py
├── experiment_a.py
├── train_models.py
├── experiment_c.py
├── plots.py
├── run.py
├── tests/
└── outputs/
    ├── figures/
    ├── tables/
    ├── models/
    ├── literature_mapping.md
    └── report.md
```

Adapt these names to the existing project rather than introducing unnecessary duplication.

### Engineering principles

- Reuse already verified L3 and fade-analysis code.
- Write small, well-named functions.
- Avoid excessive helper abstractions or large class hierarchies.
- Centralize horizons, windows, and model parameters.
- Cache compact observation-level features instead of replaying L3 data for every experiment.
- Process data incrementally when needed.
- Keep runtime reasonable on an ordinary laptop.
- Use fixed random seeds.
- Do not generate long documentation for trivial functions.
- Add tests for important correctness properties, not boilerplate.

Do not sacrifice correctness to achieve a performance improvement.

---

## 10. Execution order

### Stage 1 — Literature and data audit

Read the references, inspect existing code, confirm available features and historical fill labels, and document the estimand.

Verify whether partial fills and post-fill midprices can be measured reliably.

### Stage 2 — Experiment A

Compute the descriptive economic metrics and the fill–markout frontier.

If the results are weak or contradictory, report them accurately.

Do not assume a WMR advantage.

### Stage 3 — Experiment B

Train the simple baseline and WMR-aware models.

Evaluate on chronologically held-out days.

Focus on incremental calibration and prediction quality, not raw classification accuracy.

### Stage 4 — Experiment C

Evaluate historical opportunity-selection rules using only validation-selected thresholds and test-set predictions.

Make the observational nature explicit.

### Stage 5 — Robustness and interpretation

Run a compact set of essential checks:

- Bid versus ask.
- Nearby controls.
- Different markout horizons.
- Boundary-crossing outcomes.
- Day-level stability.
- Sensitivity to queue-ahead and imbalance.
- Optional fee assumptions.

Avoid turning the project into a large specification search.

---

## 11. Final report

Create a clear `report.md` with the following structure.

### 1. Research question and contribution

Explain how our work moves from WMR activity and lower fade to the economics of passive liquidity provision.

State explicitly that our research is an extension, not a direct replication of the cited papers.

### 2. Literature comparison

For each central finding:

- What does the relevant paper establish?
- What is our corresponding experiment?
- Are the experimental conditions comparable?
- Does our result align with the paper, differ from it, or remain inconclusive?
- What new conclusion, if any, follows from the WMR setting?

### 3. Experiment A: Economic findings

Does WMR change fill probability, post-fill return, or their joint trade-off?

Discuss the distinction between spread capture and subsequent adverse selection.

### 4. Experiment B: Predictive findings

Do simple WMR-aware ML models improve calibration and conditional markout prediction?

Which features and interactions matter, and are their estimated relationships stable?

### 5. Experiment C: Decision implications

Can predicted fill-weighted markout improve historical opportunity selection?

What changes in accepted trade frequency, markout quality, and downside risk?

State carefully what the analysis does not establish about counterfactual market-making profitability.

### 6. Limitations

Explicitly address:

- Historical order-policy selection bias.
- One-month sample and limited number of fixing days.
- Spot benchmark versus futures market.
- Missing trader identities and inventory.
- Correlated order observations.
- Cost and latency assumptions.
- Unobserved counterfactual order outcomes.

### 7. Final conclusion

Provide a concise answer to:

**Does the WMR fixing regime meaningfully change the economics of passive liquidity provision, and is that change predictable enough to improve a practical decision metric?**

Different valid conclusions are possible:

- WMR improves the observed fill–markout trade-off.
- WMR changes the trade-off but not necessarily favorably.
- WMR affects the unconditional market state, but ordinary observable features already explain its conditional fill behavior.
- WMR-aware predictions add value.
- WMR-specific modeling offers little reliable incremental benefit.

Do not force positive results.

---

## 12. Success criteria

The implementation is successful if it produces:

1. A correct and interpretable empirical comparison of fill probability and maker markout during WMR and normal periods.
2. A fair out-of-sample comparison of simple ML models, including a WMR-aware specification.
3. An honest evaluation of the incremental economic value of combining fill and markout predictions.
4. Readable, publication-quality figures.
5. Clear identification of how the findings relate to the cited literature.
6. Reliable code with explicit protection against look-ahead bias, leakage, selection bias, and class-imbalance errors.
7. A defensible conclusion that distinguishes observed predictive relationships from implementable trading profitability.

**Primary principle: Prefer a small number of trustworthy experiments with strong economic interpretation over complex models or speculative explanations that the available data cannot identify.**