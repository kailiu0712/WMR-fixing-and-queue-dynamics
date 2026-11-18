"""Correctness checks. Run from project/: .venv/Scripts/python.exe -m pytest wmr_value/tests -q"""
import numpy as np
import pandas as pd
import pytest

from wmr_value.build_labels import (CACHE, L3, build_day, day_paths, entry_features, load_day,
                                    local_parts, period_of_clock, tick_size)
from wmr_value.config import HORIZONS

TICK = tick_size(L3.parents[1])


@pytest.fixture(scope="module")
def day():
    return load_day(day_paths()[1])


def test_features_ignore_post_entry_messages(day):
    """Scrambling every message after the add row must not change that entry's features."""
    rng = np.random.default_rng(0)
    adds = np.flatnonzero((day.action.to_numpy() == "A") & ~day.is_snapshot.to_numpy(bool))
    for add_row in rng.choice(adds[len(adds) // 4: 3 * len(adds) // 4], 3, replace=False):
        side = day.side.to_numpy()[[add_row]]
        before = entry_features(day, np.array([add_row]), side, TICK)
        bad = day.copy()
        tail = slice(add_row + 1, None)
        for col in ["action", "side", "price", "size", "order_id", "qty_ahead", "bid_px", "bid_qty",
                    "ask_px", "ask_qty"]:
            vals = bad[col].to_numpy().copy()
            vals[tail] = rng.permutation(vals[tail])
            bad[col] = vals
        after = entry_features(bad, np.array([add_row]), side, TICK)
        pd.testing.assert_frame_equal(before, after)


def test_wmr_window_is_london_time():
    utc = pd.to_datetime(["2026-07-15 14:57:30", "2026-07-15 15:02:30",
                          "2026-12-15 15:57:30", "2026-12-15 16:02:30"], utc=True)
    _, _, clock, _ = local_parts(utc.as_unit("ns").asi8)
    assert list(period_of_clock(clock)) == ["wmr", "nearby", "wmr", "nearby"]


def test_markout_decomposition_and_passive_fills():
    obs = pd.read_parquet(CACHE / f"{day_paths()[1].stem}_obs.parquet")
    f = obs[obs.label == 1]
    assert len(f) > 100
    assert (f.fill_price == f.price).all()                   # resting orders fill at their limit
    assert (f.fill_side.isin(["N"]) | (f.fill_side == f.side)).all()
    assert (f.fill_time_s <= 5.0).all() and (f.fill_time_s >= 0).all()   # 0 = filled in its entry packet
    for h in HORIZONS:
        np.testing.assert_allclose(f[f"mk_{h}"], f.spread_capture + f[f"move_{h}"], atol=1e-6)
    assert obs.loc[obs.label == 0, "mk_5"].isna().all()


def test_day_split_is_chronological_and_disjoint():
    from wmr_value.train_models import split_days
    days = [f"2026-08-{d:02d}" for d in range(1, 31)]
    s = split_days(pd.DataFrame({"day": days[::-1]}))
    assert s["train"][-1] < s["val"][0] and s["val"][-1] < s["test"][0]
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == len(days)
