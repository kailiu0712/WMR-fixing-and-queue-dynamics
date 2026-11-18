"""Correctness checks. Run from project/: .venv/Scripts/python.exe -m pytest wmr_price_impact/tests -q"""
import numpy as np
import pandas as pd
import pytest

from wmr_value.build_labels import L3, load_day, packets, tick_size
from wmr_value.clocks import cum_clocks, obs_paths
from wmr_price_impact.build_trade_events import (event_features, maker_links, raw_mid, responses, trade_events)
from wmr_price_impact.config import FEATURES, VOL_H
from wmr_price_impact.matching import knn_pairs, strata_key

TICK = tick_size(L3.parents[1])
PATH = obs_paths()[1]


@pytest.fixture(scope="module")
def day():
    df = load_day(PATH)
    pk = packets(df, TICK)
    ev = trade_events(df, pk)
    ev = pd.concat([ev, event_features(df, pk, ev, TICK)], axis=1)
    r = responses(pk, cum_clocks(df, pk), raw_mid(df, pk), ev.pid.to_numpy(), ev.q.to_numpy(float), TICK)
    return df, pk, pd.concat([ev, pd.DataFrame(r)], axis=1)


def test_one_event_per_packet_and_no_double_counting(day):
    df, pk, ev = day
    t = (df.action == "T") & ~df.is_snapshot
    assert ev.pid.is_unique
    assert set(ev.pid) == set(pk["pid"][t.to_numpy()])                 # F/C/M rows never create events
    assert ev["size"].sum() == df.loc[t, "size"].sum()                 # executed quantity preserved, counted once
    assert (ev.size >= ev.n_t).all()


def test_aggressor_side_semantics(day):
    df, pk, ev = day
    e = ev[ev.q != 0]
    ends = np.flatnonzero(df.is_last.to_numpy(bool))
    pre_bid, pre_ask = df.bid_px.to_numpy()[ends][e.pid - 1], df.ask_px.to_numpy()[ends][e.pid - 1]
    t = df[(df.action == "T") & ~df.is_snapshot].assign(pid=lambda d: pk["pid"][d.index])
    lo, hi = t.groupby("pid").price.min()[e.pid].to_numpy(), t.groupby("pid").price.max()[e.pid].to_numpy()
    buy = e.q.to_numpy() > 0
    assert np.mean(lo[buy] >= pre_ask[buy] - 1e-9) > 0.99                # buyers lift the ask
    assert np.mean(hi[~buy] <= pre_bid[~buy] + 1e-9) > 0.99              # sellers hit the bid
    assert (ev.q == 0).mean() < 0.05


def test_mid_and_tick_conversion(day):
    df, pk, ev = day
    e = ev[ev.valid_state & (ev.q != 0)].sample(200, random_state=0)
    prev_end = pk["starts"][e.pid] - 1                                   # last row before the event packet
    mid = (df.bid_px.to_numpy()[prev_end] + df.ask_px.to_numpy()[prev_end]) / 2
    np.testing.assert_allclose(e.m_pre, mid)
    for c in ["I0", *[f"I_v{v}" for v in VOL_H], "I_w5"]:
        x = ev[c].dropna().to_numpy() * 2                                # mids sit on half ticks
        np.testing.assert_allclose(x, np.round(x), atol=1e-6)
    assert (ev.loc[ev.q != 0, "I0"].dropna() >= 0).mean() > 0.999        # a trade does not move the mid against itself


def test_volume_horizon_boundaries(day):
    df, pk, ev = day
    vol = cum_clocks(df, pk)["volume"]
    e = ev[ev.valid_state & (ev.q != 0) & ev.I_v25.notna()].sample(200, random_state=1)
    for _, r in e.iterrows():
        target = vol[r.pid] + 25
        idx = np.flatnonzero(vol <= target).max()
        assert idx >= r.pid and vol[idx + 1] > target                   # last packet within, next one crosses
        assert r.I_v25 == pytest.approx(r.q * (pk["mid"][idx] - r.m_pre) / TICK)
    assert (np.diff(vol) >= 0).all()


def test_features_ignore_future_messages(day):
    """Scrambling everything after the event packet (prices, flow, book) leaves its features unchanged."""
    df, pk, ev = day
    rng = np.random.default_rng(0)
    sel = ev[(ev.q != 0) & ev.valid_state].iloc[[len(ev) // 3, len(ev) // 2, 2 * len(ev) // 3]]
    for _, r in sel.iterrows():
        end = pk["starts"][r.pid + 1]                                   # first row after the event packet
        bad = df.copy()
        for col in ["action", "side", "price", "size", "order_id", "qty_ahead", "bid_px", "bid_qty", "ask_px", "ask_qty"]:
            v = bad[col].to_numpy().copy()
            v[end:] = rng.permutation(v[end:])
            bad[col] = v
        one = ev.loc[[r.name]].reset_index(drop=True)
        a = event_features(df, packets(df, TICK), one, TICK)
        b = event_features(bad, packets(bad, TICK), one, TICK)
        pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])


def test_maker_markout_identity(day):
    """Maker markout = spread capture - signed aggressive impact, exactly, for linked fills."""
    df, pk, ev = day
    m = maker_links(PATH.stem, df, pk).merge(ev[["pid", "q", "I_w5"]], on="pid")
    m = m[m.q != 0]
    assert len(m) > 100 and (m.s == -m.q).all()                         # the passive side is the consumed side
    np.testing.assert_allclose(m.mk_5_obs, m.spread_capture - m.I_w5, atol=1e-9)


def test_matching_stays_in_universe():
    rng = np.random.default_rng(0)
    n = 3000
    ev = pd.DataFrame({"q": rng.choice([-1, 1], n), "wide": rng.random(n) < .1, "ib": rng.integers(0, 10, n),
                       "period": rng.choice(["wmr", "nearby", "normal"], n, p=[.1, .2, .7])})
    Z = rng.normal(size=(n, 4))
    key = strata_key(ev)
    treat, ctrl = (ev.period == "wmr").to_numpy(), (ev.period == "normal").to_numpy()
    p = knn_pairs(key, treat, ctrl, Z, 5, 2.0)
    assert treat[p.t].all() and ctrl[p.c].all()
    for col in ("q", "wide", "ib"):                                       # exact strata: side, spread, intensity bin
        assert (ev[col].to_numpy()[p.t] == ev[col].to_numpy()[p.c]).all()
    assert (p.dist <= 2.0).all()
    np.testing.assert_allclose(p.groupby("t").w.sum(), 1.0)
    np.testing.assert_allclose(np.linalg.norm(Z[p.t] - Z[p.c], axis=1), p.dist, atol=1e-9)
