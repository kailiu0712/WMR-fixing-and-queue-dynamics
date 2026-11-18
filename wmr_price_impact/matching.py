"""KNN matching of WMR aggressive trades to control-period trades with similar pre-trade state.

Exact strata: aggressor side x spread category (1 tick / wider) x recent-intensity decile;
controls are pooled across the days of the evaluation split (the caller passes split masks).
Control reuse across days is handled in inference by a two-sided day bootstrap (resampling the
days of WMR trades and of their controls together). Within a stratum: standardized
Euclidean distance on FEATURES (scaling fitted on training days), K nearest controls with
replacement, controls beyond the caliper dropped, each kept control weighted 1/m (m = kept
controls of that WMR trade). Only pre-trade columns enter; responses are never read here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import CALIPERS, CORR_MAX, FEATURES, INTENSITY_BINS, SMD_MAX

STRATA = ["q", "wide", "ib"]


def choose_features(train: pd.DataFrame) -> list[str]:
    """Drop a feature whose |corr| with an earlier kept feature exceeds CORR_MAX (training days)."""
    c = train[FEATURES].corr().abs()
    keep = []
    for f in FEATURES:
        if all(c.loc[f, k] <= CORR_MAX for k in keep):
            keep.append(f)
    return keep


def fit_scaler(train: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = train[feats].to_numpy(float)
    return x.mean(0), x.std(0)


def intensity_cuts(train_wmr: pd.Series) -> np.ndarray:
    return np.quantile(train_wmr.to_numpy(float), np.linspace(0, 1, INTENSITY_BINS + 1)[1:-1])


def strata_key(ev: pd.DataFrame) -> np.ndarray:
    return (ev.q.to_numpy() > 0) + 2 * ev.wide.to_numpy().astype(int) + 4 * ev.ib.to_numpy()


def knn_pairs(key: np.ndarray, treat: np.ndarray, ctrl: np.ndarray, Z: np.ndarray, K: int,
              caliper: float) -> pd.DataFrame:
    """Rows (t, c, dist, w): event row indices; w sums to 1 over the kept controls of each matched t."""
    out = []
    ti, ci = np.flatnonzero(treat), np.flatnonzero(ctrl)
    cg = pd.Series(ci).groupby(key[ci]).apply(np.asarray)
    for g, t in pd.Series(ti).groupby(key[ti]):
        if g not in cg.index:
            continue
        c = cg[g]
        t = t.to_numpy()
        nn = NearestNeighbors(n_neighbors=min(K, len(c))).fit(Z[c])
        d, j = nn.kneighbors(Z[t])
        out.append(pd.DataFrame({"t": np.repeat(t, d.shape[1]), "c": c[j.ravel()], "dist": d.ravel()}))
    if not out:
        return pd.DataFrame(columns=["t", "c", "dist", "w"])
    p = pd.concat(out, ignore_index=True)
    p = p[p.dist <= caliper].reset_index(drop=True)
    p["w"] = 1.0 / p.groupby("t").t.transform("size")
    return p


def smd(ev: pd.DataFrame, treat: np.ndarray, ctrl: np.ndarray, pairs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Standardized mean differences before (all eligible) and after matching (pre-matching SD)."""
    rows = []
    t_m = np.unique(pairs.t.to_numpy())
    for f in cols:
        x = ev[f].to_numpy(float)
        sd = np.sqrt((x[treat].var() + x[ctrl].var()) / 2)
        after_c = np.bincount(pairs.t, pairs.w * x[pairs.c], minlength=len(x))[t_m].mean()
        rows.append({"feature": f, "treat_mean": x[treat].mean(), "ctrl_mean": x[ctrl].mean(),
                     "smd_before": (x[treat].mean() - x[ctrl].mean()) / sd,
                     "treat_mean_matched": x[t_m].mean(), "ctrl_mean_matched": after_c,
                     "smd_after": (x[t_m].mean() - after_c) / sd})
    return pd.DataFrame(rows)


def choose_caliper(ev: pd.DataFrame, key: np.ndarray, treat: np.ndarray, ctrls: dict, Z: np.ndarray, K: int,
                   feats: list[str]) -> tuple[float, pd.DataFrame]:
    """Largest caliper whose matched samples have all |SMD| <= SMD_MAX for every comparison."""
    rows = []
    for cal in CALIPERS:
        for name, ctrl in ctrls.items():
            p = knn_pairs(key, treat, ctrl, Z, K, cal)
            s = smd(ev, treat, ctrl, p, feats)
            rows.append({"caliper": cal, "comparison": name, "matched_frac": p.t.nunique() / treat.sum(),
                         "max_abs_smd": s.smd_after.abs().max()})
    grid = pd.DataFrame(rows)
    ok = grid.groupby("caliper").max_abs_smd.max() <= SMD_MAX
    chosen = ok[ok].index.max() if ok.any() else min(CALIPERS)
    return float(chosen), grid
