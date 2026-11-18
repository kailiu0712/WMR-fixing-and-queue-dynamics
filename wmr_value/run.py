"""Phase 3 pipeline. Run from project/: .venv/Scripts/python.exe -m wmr_value.run"""
import time

from . import experiment_a, experiment_c, train_models
from .build_labels import load_observations
from .config import FIGURES, MODELS, TABLES

DROP = ["order_id", "add_row", "eda_fill_time_s", "valid_state", "follow_up_ok", "fill_price", "price"]


def main() -> None:
    for d in (TABLES, FIGURES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obs = load_observations().drop(columns=DROP)
    print(f"observations: {len(obs):,} rows, {obs.day.nunique()} days ({time.time() - t0:.0f}s)", flush=True)
    obs.groupby(["period", "side"]).agg(n=("label", "size"), fill_rate=("label", "mean"),
                                        days=("day", "nunique")).to_csv(TABLES / "sample_sizes.csv")
    obs.groupby(["day", "period"]).label.agg(["size", "mean"]).unstack().to_csv(TABLES / "prevalence_by_day.csv")
    preds, wf, split = train_models.run(obs)
    print(f"Experiment B done ({time.time() - t0:.0f}s)", flush=True)
    experiment_a.run(obs, wf)
    print(f"Experiment A done ({time.time() - t0:.0f}s)", flush=True)
    experiment_c.run(obs, preds, wf, split)
    print(f"Experiment C done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
