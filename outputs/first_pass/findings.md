# First pass: is the WMR fix qualitatively different from an equally busy market?

Sample: 66 London weekdays, 2026-07-14 to 2026-10-13, 6EU6 then 6EZ6. Code: `first_pass.py`, `first_pass_checks.py`.

Inference: the day is the unit. For each test, WMR minus control is computed per day, then tested with a paired sign-flip permutation test (20,000 flips) and a day-bootstrap 95% CI.

Controls:
- `nearby`: the adjacent 5-minute windows.
- `us_event`: 5-minute windows at 13:30, 14:30 and 15:00 London.
- `normal`: all other seconds.
- `normal_matched`: normal seconds reweighted to WMR's distribution of trailing-60 s traded contracts.

## Results

| Idea | Test | WMR | Matched normal | p | Verdict |
|---|---|---|---|---|---|
| Activity mix | best-price executed / added | 0.153 | 0.141 | 0.012 | Mostly explained by activity |
| Activity mix | best-price cancelled / added | 0.916 | 0.952 | <0.001 | Small residual difference |
| Toxicity | maker P&L 5 s after fill, ticks/contract (winsorized) | 0.104 | 0.047 | <0.001 | **About 2x better at WMR** |
| Toxicity | maker P&L 60 s after fill, ticks/contract (winsorized) | 0.145 | 0.065 | 0.025 | Better at WMR |
| Reversal | corr(in-window return, next 10 min) | 0.04 | placebo median 0.00 | 0.74 | **No reversal in 6E** |
| Flow persistence | mean z² of signed aggressive volume | 2.52 | busiest placebo windows | 0.46 | Highest of all 472 windows, but not significant vs busy windows |
| Liquidity fade | excess share of hit-side depth cancelled within 100 ms (matched on activity and depth, no further trade) | 0.133 | 0.276 | <0.001 | **About half; lower on 66/66 days** |
| Market maker | P&L gap, flow-against vs not (5 s) | — | — | 0.07–0.9 | Pulling quotes on adverse flow doesn't help, in any period |

The nearby windows (0.444) and US-hours windows (0.416) fade like normal time on the same measure. The low fade is specific to the WMR window.

## Story candidate

At the fix, liquidity at the best price stays put when hit. It fades about half as much as in equally busy normal moments, even at matched depth. Fills against it are less adversely selected. This is a difference in how liquidity providers behave relative to the flow, not just more activity.

Two explanations remain open:
1. The same market makers fade less because they expect fix flow to be uninformed.
2. A different group supplies liquidity at the fix: patient benchmark executors resting passive orders.

## Caveats

- Maker P&L is from real fills, before fees, ignoring queue position and latency.
- The fade measure counts cancels at the hit price, which includes some repricing.
- Activity matching uses one trailing-activity variable.
- With ~10 headline tests, the fade, 5 s markout and cancel/add results survive a Holm correction; the 60 s markout does not.
