"""Re-run Experiments A, B, C and the robustness checks with volume- and event-clock labels.

Same code and procedures as the 5 s run; only the label/markout columns change (see clocks.py).
Outputs go to outputs/wmr_value/clock_<k>/{tables,figures,models}.
Run from project/: .venv/Scripts/python.exe -m wmr_value.clock_run
"""
import time

import numpy as np
import pandas as pd

from . import experiment_a, experiment_c, robustness, train_models
from .build_labels import load_observations
from .clocks import CLOCK_COLS, build, load_clock_obs
from .config import OUT
from .run import DROP


def check_time_clock() -> None:
    """The time clock must reproduce the existing 5 s observation table exactly."""
    a, b = load_observations(), load_clock_obs("time")
    assert len(a) == len(b) and (a.entry_ns.to_numpy() == b.entry_ns.to_numpy()).all()
    assert (a.label.to_numpy() == b.label.to_numpy()).all()
    f = a.label.to_numpy() == 1                     # follow_up_ok only matters for fills
    for c in CLOCK_COLS + ["Z_5"]:
        x, y = a[c].to_numpy(float)[f], b[c].to_numpy(float)[f]
        assert np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(x[~np.isnan(x)], y[~np.isnan(y)]), c
    print(f"time clock reproduces the 5 s table: {len(a):,} rows", flush=True)


def main() -> None:
    t0 = time.time()
    print("rates (units per normal-hours second, training days):", build(), flush=True)
    check_time_clock()
    for k in ("volume", "event"):
        d = OUT / f"clock_{k}"
        paths = {"TABLES": d / "tables", "FIGURES": d / "figures", "MODELS": d / "models"}
        for v in paths.values():
            v.mkdir(parents=True, exist_ok=True)
        for mod in (train_models, experiment_a, experiment_c, robustness):
            for name, v in paths.items():
                if hasattr(mod, name):
                    setattr(mod, name, v)
        robustness.load_observations = lambda k=k: load_clock_obs(k)
        obs = load_clock_obs(k)
        wall = obs.groupby("period").agg(n=("label", "size"), fill_rate=("label", "mean"),
                                         wall_label_median=("wall_label_s", "median"),
                                         wall_mk5_median=("wall_mk_5", "median"))
        wall.to_csv(paths["TABLES"] / "clock_sample.csv")
        print(f"[{k}] {len(obs):,} rows\n{wall}", flush=True)
        obs = obs.drop(columns=DROP + ["wall_label_s", "wall_mk_5"])
        preds, wf, split = train_models.run(obs)
        experiment_a.run(obs, wf)
        experiment_c.run(obs, preds, wf, split)
        print(f"[{k}] A, B, C done ({time.time() - t0:.0f}s)", flush=True)
        robustness.main()
        print(f"[{k}] robustness done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
