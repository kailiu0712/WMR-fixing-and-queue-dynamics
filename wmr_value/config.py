"""Central settings for the Phase 3 experiments (A: economics, B: models, C: selection)."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
L3 = PROJECT / "data_6e" / "l3"
EDA_CACHE = PROJECT / "outputs" / "wmr_eda" / "daily_cache_v1"   # labels from wmr_eda.py
OUT = PROJECT / "outputs" / "wmr_value"
CACHE, TABLES, FIGURES, MODELS = (OUT / s for s in ("cache", "tables", "figures", "models"))

NS = 1_000_000_000
LABEL_S = 5                         # fill label horizon, seconds (same as the proposal)
HORIZONS = [1, 5, 30]               # markout horizons after the fill packet, seconds
PRIMARY_H = 5
FLOW_S = [1, 5, 30]                 # backward-looking order-flow windows, seconds

# London clock seconds. WMR 15:57:30-16:02:30; nearby = the adjacent 5-minute windows
# (same definition as wmr_eda.period_of_clock). Universe = liquid hours 08:00-17:00 London.
WMR = (57450, 57750)
UNIVERSE = (8 * 3600, 17 * 3600)
PERIODS = ["wmr", "nearby", "normal"]
COLORS = {"wmr": "#b93c32", "nearby": "#e29b41", "normal": "#527ea6"}

SPLIT = (0.6, 0.2)                  # train / validation share of days; rest is test
WALK_START, WALK_BLOCK = 20, 5      # walk-forward: first prediction day index, block length
LOGIT_C = [1e-3, 1e-2, 1e-1, 1.0]   # validation grid, inverse L2 strength
RIDGE_ALPHA = [1.0, 1e2, 1e4, 1e6]
COSTS = [0.0, 0.1, 0.3]             # ticks per contract; illustrative only, fees NOT verified
ACCEPT = [0.1, 0.2, 0.3]            # acceptance fractions for Experiment C
SEED = 0
