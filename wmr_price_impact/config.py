"""Settings for the KNN-matched aggressive-trade price-impact experiment.

Every choice here is fixed before looking at price responses: horizon grid from training-day
trade sizes and activity rates, features and caliper rule from training-day features only.
"""
import json

from wmr_value.config import NS, OUT as VALUE_OUT, PROJECT, SEED, SPLIT, UNIVERSE, WMR  # noqa: F401

OUT = PROJECT / "outputs" / "wmr_price_impact"
CACHE, TABLES, FIGURES = (OUT / s for s in ("cache", "tables", "figures"))

# Response horizons. 0+ = end of the event packet. Volume clock (primary): further contracts
# traded after the event packet. Training days: median event 2 contracts, p90 ~10, p99 ~40;
# normal-hours rate 2.42 contracts/s, so 100 contracts ~ 41 s in normal hours.
VOL_H = [5, 10, 25, 50, 100]
WALL_H = [1, 5, 10]                                  # seconds (secondary)
_r = json.loads((VALUE_OUT / "cache_clocks" / "rates.json").read_text())   # training-day rates
EVENT_PER_CONTRACT = _r["event"] / _r["volume"]      # event-clock units for the same normal-hours duration
MAX_SPAN_S = 300                                     # a horizon taking longer than this counts as a data gap

# Matching. Distance features, in this order; a feature is dropped if |corr| > CORR_MAX with an
# earlier one on training days. Exact strata: aggressor side, spread category, and decile of
# recent trade intensity (log_volume_30s; cutpoints = deciles of training-day WMR trades).
# Controls are pooled over the days of the same split (train / val / test).
# Design revision, made on training-day FEATURES only, before any response was read: same-day
# matching (first design) could not balance trade intensity at any useful coverage (SMD of
# log_volume_30s 0.24-0.69 at calipers 1-3); pooled + intensity-decile strata met |SMD| <= 0.10.
FEATURES = ["log_size", "log_depth_consumed", "log_depth_opposite", "flow_1s", "flow_5s", "flow_30s",
            "log_volume_30s", "log_cancels_5s", "log_vol_30s"]
DIAG_ONLY = ["imbalance", "spread_ticks"]
CORR_MAX = 0.9
K_MAIN, K_ROBUST = 5, [1, 10]
INTENSITY_BINS = 10
CALIPERS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]            # standardized Euclidean distance
SMD_MAX = 0.10                                       # caliper rule: largest caliper with all |SMD| <= this on training days
CONTROLS = {"A": "nearby", "B": "normal"}
N_BOOT = 2000
