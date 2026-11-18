"""KNN-matched price responses: WMR vs nearby (A) and vs normal hours (B).

Order of operations (design frozen before any response is read):
  1. sample filters and exclusion counts;
  2. training days only: feature list (correlation rule), scaler, intensity-decile cutpoints,
     caliper (balance rule), K = 5;
  3. held-out test days (primary) and all 66 days (supplementary, labelled as such): matched
     differences; validation days: balance checks only;
  4. robustness (K, caliper, clocks, side, WMR-boundary trades) and the maker-markout bridge.
Inference: the trading day is the unit. Matched estimates are means of per-day means; CIs and
p-values come from a two-sided day bootstrap that resamples the days of the split and applies
the draw to WMR trades and to their matched controls (so control reuse across days is
resampled too; no re-matching inside the bootstrap). Unmatched same-day comparisons use
wmr_value.stats.day_diff as in Phase 3.
Run from project/: .venv/Scripts/python.exe -m wmr_price_impact.analyze_results
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wmr_value.stats import day_diff
from wmr_value.train_models import split_days

from .build_trade_events import load_events
from .config import (CALIPERS, CONTROLS, DIAG_ONLY, FEATURES, K_MAIN, K_ROBUST, N_BOOT, SEED, TABLES, VOL_H,
                     WALL_H, WMR)
from .matching import choose_caliper, choose_features, fit_scaler, intensity_cuts, knn_pairs, smd, strata_key

CLOCK_COLS = {"volume": ["I0", *[f"I_v{v}" for v in VOL_H]],
              "event": ["I0", *[f"I_e{v}" for v in VOL_H]],
              "wall": ["I0", *[f"I_w{h}" for h in WALL_H]]}
SIZE_BINS = [0, 1, 4, 9, 24, np.inf]
SIZE_LABELS = ["1", "2-4", "5-9", "10-24", "25+"]


def prepare(ev: pd.DataFrame, lk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ex = ev.groupby("period").agg(n_events=("q", "size"), ambiguous_side=("q", lambda q: (q == 0).mean()),
                                  invalid_state=("valid_state", lambda v: (~v).mean()),
                                  reset_in_packet=("reset_in_packet", "mean"),
                                  post_mid_delayed=("post_mid_delayed", "mean"),
                                  multi_aggressor_id=("n_aggr_id", lambda n: (n > 1).mean()),
                                  multi_price=("n_px", lambda n: (n > 1).mean()),
                                  t_equals_passive=("size", lambda s: (s == ev.loc[s.index, "passive_qty"]).mean()))
    keep = (ev.q != 0) & ev.valid_state & ~ev.reset_in_packet & np.isfinite(ev[FEATURES]).all(axis=1)
    ex["kept"] = keep.groupby(ev.period).mean()
    ev = ev[keep].reset_index(drop=True)
    ev["wide"] = ev.spread_ticks > 1.5
    for cols in CLOCK_COLS.values():
        for c in cols[1:]:
            ev["R" + c[1:]] = ev[c] - ev.I0
    ev["moved0"] = (ev.I0 > 0).astype(float)
    # maker fills of the Phase 3 sample in these events (the passive side is the consumed side)
    m = lk.merge(ev[["day", "pid", "q", "period"]].reset_index(), on=["day", "pid"], how="left")
    lk_stats = pd.Series({"maker_fills": len(lk), "in_kept_event": m["index"].notna().mean(),
                          "side_consistent": (m.s == -m.q)[m["index"].notna()].mean()})
    m = m[m["index"].notna() & (m.s == -m.q)].astype({"index": int})
    ev["n_maker"] = np.bincount(m["index"], minlength=len(ev))
    return ev, m, pd.concat([ex.T, lk_stats.to_frame("all")], axis=0)


class Boot:
    """Day multiplicities for one evaluation split: row 0 = point estimate (all ones)."""

    def __init__(self, days: np.ndarray, ev_day: pd.Series):
        rng = np.random.default_rng(SEED)
        self.days = np.asarray(days)
        n = len(self.days)
        self.code = ev_day.map({d: i for i, d in enumerate(self.days)}).fillna(-1).astype(int).to_numpy()
        self.mult = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), N_BOOT)])


def effect(ev: pd.DataFrame, pairs: pd.DataFrame, col: str, boot: Boot, side: int | None = None) -> dict:
    """WMR mean, matched-control mean and their difference (means of per-day means), with
    two-sided day-bootstrap CIs and p-value."""
    y = ev[col].to_numpy(float)
    t, c, w = pairs.t.to_numpy(), pairs.c.to_numpy(), pairs.w.to_numpy()
    ok = np.isfinite(y[t]) & np.isfinite(y[c])
    if side is not None:
        ok &= ev.q.to_numpy()[t] == side
    t, c, w = t[ok], c[ok], w[ok]
    if len(t) == 0:
        return {"n_trades": 0}
    dcode = boot.code
    tu, ti = np.unique(t, return_inverse=True)
    td, cd = dcode[tu], dcode[c]
    yt, yc = y[tu], y[c]
    nd = len(boot.days)
    res = []
    for m in boot.mult:
        wc = w * m[cd]
        den = np.bincount(ti, wc, minlength=len(tu))
        v = den > 0
        ctrl = np.bincount(ti, wc * yc, minlength=len(tu))[v] / den[v]
        n_d = np.bincount(td[v], minlength=nd)
        has = n_d > 0
        wm = np.bincount(td[v], yt[v], minlength=nd)[has] / n_d[has]
        cm = np.bincount(td[v], ctrl, minlength=nd)[has] / n_d[has]
        md = m[has]
        res.append([(md * wm).sum() / md.sum(), (md * cm).sum() / md.sum(), (md * (wm - cm)).sum() / md.sum()])
        if len(res) == 1:
            share_neg, n_days = ((wm - cm) < 0).mean(), int(has.sum())
    res = np.array(res)
    b = res[1:]
    q = lambda k: np.nanquantile(b[:, k], [.025, .975])
    p = min(1.0, 2 * min((b[:, 2] <= 0).mean(), (b[:, 2] >= 0).mean()))
    return {"n_trades": len(tu), "n_days": n_days, "wmr": res[0, 0], "wmr_lo": q(0)[0], "wmr_hi": q(0)[1],
            "ctrl": res[0, 1], "ctrl_lo": q(1)[0], "ctrl_hi": q(1)[1], "diff": res[0, 2], "lo": q(2)[0],
            "hi": q(2)[1], "p": p, "share_days_neg": share_neg}


def raw_effect(ev: pd.DataFrame, treat: np.ndarray, ctrl: np.ndarray, col: str) -> dict:
    """Unmatched, same-day: mean over WMR trades minus mean over control trades, per day."""
    y = pd.Series(ev[col].to_numpy(float))
    d = pd.DataFrame({"wmr": y[treat].groupby(ev.day[treat]).mean(), "ctrl": y[ctrl].groupby(ev.day[ctrl]).mean()}).dropna()
    s = day_diff(d.wmr - d.ctrl)
    return {"wmr": d.wmr.mean(), "ctrl": d.ctrl.mean(), "diff": s["mean"], "lo": s["ci_lo"], "hi": s["ci_hi"],
            "p": s["p"], "n_days": s["n_days"]}


def match_diagnostics(ev, treat, ctrl, pairs) -> dict:
    t_m = pairs.t.nunique()
    load = pairs.groupby("c").w.sum()
    same_day = ev.day.to_numpy()[pairs.t] == ev.day.to_numpy()[pairs.c]
    return {"n_wmr": int(treat.sum()), "matched": t_m, "matched_frac": t_m / treat.sum(),
            "dist_p50": pairs.dist.median(), "dist_p90": pairs.dist.quantile(.9),
            "controls_per_matched": len(pairs) / max(t_m, 1), "unique_controls": pairs.c.nunique(),
            "control_slots": len(pairs), "max_control_weight_share": load.max() / load.sum(),
            "ctrl_days": pd.Series(ev.day.to_numpy()[pairs.c.unique()]).nunique(),
            "same_day_share": (pairs.w * same_day).sum() / pairs.w.sum(), "n_ctrl_pool": int(ctrl.sum())}


BRIDGE_COLS = ["I0", "I_w5", *[f"I_v{v}" for v in VOL_H]]


def bridge(ev, links, key, treat, ctrl, Z, caliper, sample, boot) -> list[dict]:
    """Maker perspective (positive = better for the passive side at WMR), WMR vs normal hours.
    (1) all aggressive trades, KNN-matched: -dI;
    (2) trades that filled a Phase 3 maker order, KNN-matched among linked trades: -dI;
    (3) the same linked trades weighted by maker fills, unmatched same-day: -dI;
    (4) historical maker markout of those fills, unmatched same-day: dM = dSC - dI (exact identity);
    (5) spread capture difference dSC."""
    linked = ev.n_maker.to_numpy() > 0
    p_all = knn_pairs(key, treat, ctrl, Z, K_MAIN, caliper)
    p_lnk = knn_pairs(key, treat & linked, ctrl & linked, Z, K_MAIN, caliper)
    f = links[links["index"].isin(np.flatnonzero(treat | ctrl))].copy()
    f["grp"] = np.where(ev.period.to_numpy()[f["index"]] == "wmr", "wmr", "ctrl")
    f["q"] = ev.q.to_numpy()[f["index"]]
    out = []
    for side, q in {"both": None, "buy": 1, "sell": -1}.items():     # aggressor side (maker side = opposite)
        for c in BRIDGE_COLS:
            f["I"] = ev[c].to_numpy(float)[f["index"]]
            f["M"] = f.spread_capture - f.I
            ff = f.dropna(subset=["I"])
            ff = ff if q is None else ff[ff.q == q]
            g = ff.groupby(["day", "grp"])[["I", "M", "spread_capture"]].mean().unstack("grp").dropna()
            base = {"sample": sample, "side": side, "col": c}
            for name, p in {"(1) all trades, matched": p_all, "(2) maker-linked trades, matched": p_lnk}.items():
                s = effect(ev, p, c, boot, q)
                out.append({**base, "quantity": name, "diff": -s["diff"], "lo": -s["hi"], "hi": -s["lo"],
                            "p": s["p"], "n_days": s["n_days"], "n": s["n_trades"]})
            for name, d in {"(3) maker-linked trades, fill-weighted": -(g["I"].wmr - g["I"].ctrl),
                            "(4) historical maker markout": g["M"].wmr - g["M"].ctrl,
                            "(5) spread capture": g["spread_capture"].wmr - g["spread_capture"].ctrl}.items():
                s = day_diff(d)
                out.append({**base, "quantity": name, "diff": s["mean"], "lo": s["ci_lo"], "hi": s["ci_hi"],
                            "p": s["p"], "n_days": s["n_days"], "n": int((ff.grp == "wmr").sum())})
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    ev, lk = load_events()
    ev, links, excl = prepare(ev, lk)
    excl.to_csv(TABLES / "exclusions.csv")
    split = split_days(ev)
    ev["split"] = ev.day.map({d: k for k, ds in split.items() for d in ds})
    print({k: len(v) for k, v in split.items()}, f"{len(ev):,} events", flush=True)
    ev.groupby(["day", "split", "period", "q"]).size().unstack(["period", "q"]).to_csv(TABLES / "sample_sizes_by_day.csv")
    ev.groupby(["split", "period", "q"]).agg(n=("I0", "size"), days=("day", "nunique"),
                                             size_mean=("size", "mean")).to_csv(TABLES / "sample_sizes.csv")
    cols = list(dict.fromkeys(c for cs in CLOCK_COLS.values() for c in cs))
    ev.groupby("period")[cols].agg(lambda x: x.notna().mean()).T.to_csv(TABLES / "coverage.csv")
    ev.groupby("period")[[f"span_v{v}" for v in VOL_H]].median().T.to_csv(TABLES / "volume_horizon_wall_seconds.csv")

    # ---- design on training days only (no response column is read) ----
    tr = ev[ev.split == "train"]
    feats = choose_features(tr)
    tr[FEATURES].corr().to_csv(TABLES / "train_feature_corr.csv")
    mu, sd = fit_scaler(tr, feats)
    cuts = intensity_cuts(tr.loc[tr.period == "wmr", "log_volume_30s"])
    ev["ib"] = np.digitize(ev.log_volume_30s, cuts)
    Z = (ev[feats].to_numpy(float) - mu) / sd
    key = strata_key(ev)
    wmr = (ev.period == "wmr").to_numpy()
    in_split = {s: (ev.split == s).to_numpy() for s in ("train", "val", "test")} | {"all": np.ones(len(ev), bool)}
    ctrl_of = {c: (ev.period == p).to_numpy() for c, p in CONTROLS.items()}
    trn = in_split["train"]
    caliper, grid = choose_caliper(ev, key, wmr & trn, {c: m & trn for c, m in ctrl_of.items()}, Z, K_MAIN, feats)
    grid.to_csv(TABLES / "caliper_grid_train.csv", index=False)
    loose = CALIPERS[CALIPERS.index(caliper) + 1]
    design = {"features": feats, "dropped": [f for f in FEATURES if f not in feats], "caliper": caliper,
              "caliper_sensitivity": loose, "K": K_MAIN, "intensity_cuts": cuts.round(3).tolist(),
              "split_days": {k: [v[0], v[-1], len(v)] for k, v in split.items()}}
    (TABLES / "design.json").write_text(json.dumps(design, indent=1))
    print("design:", design, flush=True)

    # ---- matching, diagnostics, effects ----
    eff, raw, diag, bal, sup, brg = [], [], [], [], [], []
    for sample in ("test", "val", "all"):
        S = in_split[sample]
        boot = Boot(sorted(ev.day[S].unique()), ev.day)
        for comp, cmask in ctrl_of.items():
            treat, ctrl = wmr & S, cmask & S
            for K, cal in [(K_MAIN, caliper), *[(k, caliper) for k in K_ROBUST], (K_MAIN, loose)]:
                pairs = knn_pairs(key, treat, ctrl, Z, K, cal)
                main = K == K_MAIN and cal == caliper
                diag.append({"sample": sample, "comparison": comp, "K": K, "caliper": cal,
                             **match_diagnostics(ev, treat, ctrl, pairs)})
                if main or sample == "val":
                    bal.append(smd(ev, treat, ctrl, pairs, feats + DIAG_ONLY)
                               .assign(sample=sample, comparison=comp, K=K, caliper=cal))
                if sample == "val":                    # validation: quality checks only, no responses
                    continue
                sides = {"both": None, "buy": 1, "sell": -1} if main else {"both": None}
                clocks = CLOCK_COLS if main else {"volume": CLOCK_COLS["volume"]}
                for clock, cs in clocks.items():
                    for c in cs + ["R" + c[1:] for c in cs[1:]] + (["moved0"] if clock == "volume" else []):
                        for side, q in sides.items():
                            eff.append({"sample": sample, "comparison": comp, "K": K, "caliper": cal, "side": side,
                                        "clock": clock, "col": c, **effect(ev, pairs, c, boot, q)})
                if not main:
                    continue
                b = pd.cut(ev["size"], SIZE_BINS, labels=SIZE_LABELS)
                matched = np.zeros(len(ev), bool)
                matched[pairs.t.unique()] = True
                for name, grp in {"size": b, "intensity_decile": ev.ib}.items():
                    sup.append(pd.DataFrame({"n_wmr": pd.Series(treat).groupby(grp.to_numpy()).sum(),
                                             "matched_frac": pd.Series(matched[treat]).groupby(grp.to_numpy()[treat]).mean()})
                               .rename_axis("bin").reset_index().assign(by=name, sample=sample, comparison=comp))
                for c in CLOCK_COLS["volume"] + ["R" + c[1:] for c in CLOCK_COLS["volume"][1:]] + ["moved0"]:
                    raw.append({"sample": sample, "comparison": comp, "col": c, **raw_effect(ev, treat, ctrl, c)})
                inner = treat & (ev.clock.to_numpy() < WMR[1] - 30)   # drop WMR trades in the last 30 s
                pb = knn_pairs(key, inner, ctrl, Z, K, cal)
                for c in CLOCK_COLS["volume"]:
                    eff.append({"sample": sample, "comparison": comp, "K": K, "caliper": cal, "side": "both",
                                "clock": "volume_no_last30s", "col": c, **effect(ev, pb, c, boot)})
                if comp == "B":
                    brg += bridge(ev, links, key, treat, ctrl, Z, cal, sample, boot)
        print("done", sample, flush=True)
    pd.DataFrame(eff).to_csv(TABLES / "effects.csv", index=False)
    pd.DataFrame(raw).to_csv(TABLES / "raw_effects.csv", index=False)
    pd.DataFrame(diag).to_csv(TABLES / "match_diagnostics.csv", index=False)
    pd.concat(bal).to_csv(TABLES / "balance_smd.csv", index=False)
    pd.concat(sup).to_csv(TABLES / "support.csv", index=False)
    pd.DataFrame(brg).to_csv(TABLES / "bridge.csv", index=False)


if __name__ == "__main__":
    main()
