"""Robustness to period imbalance (WMR is ~1.3% of observations).

R1  WMR-specific models: fill / markout models fitted on WMR (or WMR + nearby) rows only, compared
    within WMR with the pooled M1/R1 on the same test days, in walk-forward, and as a selection rule.
R2  Equal-size evaluation: pooled-model metrics on normal-hours subsamples built from whole 5-minute
    windows up to WMR's per-day size, so precision is not mistaken for predictability.
R3  Precision-weighted day aggregation of the Experiment A WMR-minus-control tests.
R4  Within-period decile cut points for the frontier (F04) and the markout calibration (F07).
Run from project/: .venv/Scripts/python.exe -m wmr_value.robustness
"""
from __future__ import annotations

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import experiment_a
from .build_labels import load_observations
from .config import (ACCEPT, COLORS, FIGURES, LOGIT_C, MODELS, PERIODS, PRIMARY_H, RIDGE_ALPHA, SEED, TABLES,
                     WALK_BLOCK, WALK_START)
from .run import DROP
from .stats import day_diff, ratio_ci
from .train_models import (brier, design, fill_metrics, fit_logit, fit_ridge, logloss, markout_metrics, split_days,
                           walk_forward)

H = PRIMARY_H
LOCAL = {"wmr_only": ["wmr"], "wmr_nearby": ["wmr", "nearby"]}
MODELS_CMP = ["M0", "M1", "wmr_only", "wmr_nearby"]      # pooled M0/M1 (with R0/R1) vs local models


def pooled_preds(obs: pd.DataFrame) -> pd.DataFrame:
    with open(MODELS / "models.pkl", "rb") as fh:
        m = pickle.load(fh)
    X0, X1 = design(obs, False), design(obs, True)
    return pd.DataFrame({"p_M0": m["M0"].predict_proba(X0)[:, 1], "p_M1": m["M1"].predict_proba(X1)[:, 1],
                         "mu_M0": m[f"R0_{H}"].predict(X0), "mu_M1": m[f"R1_{H}"].predict(X1)}, index=obs.index)


# ---------------------------------------------------------------- R1 WMR-specific models
def fit_local(sub: pd.DataFrame, train_days, hp: dict | None = None, val_days=None) -> tuple[pd.DataFrame, dict]:
    """Fit on WMR (or WMR + nearby) rows of train_days, predict every WMR row.
    hp=None: choose C / alpha on WMR validation rows only."""
    w = (sub.period == "wmr").to_numpy()
    y = sub.label.to_numpy(float)
    f = y == 1
    t = sub[f"mk_{H}"].to_numpy()
    tr = sub.day.isin(train_days).to_numpy()
    va = sub.day.isin(val_days or []).to_numpy() & w
    out, chosen = {}, {}
    for name, pers in LOCAL.items():
        g = tr & sub.period.isin(pers).to_numpy()
        X = design(sub, len(pers) > 1)
        if hp is None:
            C = min(LOGIT_C, key=lambda C: logloss(y[va], fit_logit(X[g], y[g], C).predict_proba(X[va])[:, 1]))
            a = min(RIDGE_ALPHA, key=lambda a: np.mean((fit_ridge(X[g & f], t[g & f], a).predict(X[va & f])
                                                        - t[va & f]) ** 2))
        else:
            C, a = hp[name]
        out[f"p_{name}"] = fit_logit(X[g], y[g], C).predict_proba(X[w])[:, 1]
        out[f"mu_{name}"] = fit_ridge(X[g & f], t[g & f], a).predict(X[w])
        chosen[name] = (C, a)
    return pd.DataFrame(out, index=sub.index[w]), chosen


def walk_local(sub: pd.DataFrame, days: list, hp: dict) -> pd.DataFrame:
    out = []
    for start in range(WALK_START, len(days), WALK_BLOCK):
        p, _ = fit_local(sub, days[:start], hp)
        out.append(p[sub.loc[p.index, "day"].isin(days[start:start + WALK_BLOCK]).to_numpy()])
    return pd.concat(out)


def compare_wmr(x: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """x: WMR rows with p_<model>, mu_<model>. Pooled metrics and day-paired differences vs pooled M1/R1."""
    f = x.label == 1
    rows = []
    for m in MODELS_CMP:
        mk = markout_metrics(x.loc[f, f"mk_{H}"].to_numpy(), x.loc[f, f"mu_{m}"].to_numpy())
        rows.append({"eval": tag, "model": m, **fill_metrics(x.label.to_numpy(float), x[f"p_{m}"].to_numpy()),
                     **{f"mk{H}_{k}": v for k, v in mk.items()}})
    g, gf = x.groupby("day"), x[f].groupby("day")
    drows = []
    for m in MODELS_CMP:
        if m == "M1":
            continue
        diffs = {   # every difference is oriented so that > 0 means model m beats pooled M1/R1
            "brier_M1_minus_m": g.apply(lambda d: brier(d.label, d.p_M1) - brier(d.label, d[f"p_{m}"])),
            "logloss_M1_minus_m": g.apply(lambda d: logloss(d.label, d.p_M1) - logloss(d.label, d[f"p_{m}"])),
            "auc_m_minus_M1": g.apply(lambda d: roc_auc_score(d.label, d[f"p_{m}"]) - roc_auc_score(d.label, d.p_M1)),
            f"mae{H}_R1_minus_m": gf.apply(lambda d: np.mean(np.abs(d.mu_M1 - d[f"mk_{H}"]))
                                           - np.mean(np.abs(d[f"mu_{m}"] - d[f"mk_{H}"]))),
            f"corr{H}_m_minus_R1": gf.apply(lambda d: np.corrcoef(d[f"mu_{m}"], d[f"mk_{H}"])[0, 1]
                                            - np.corrcoef(d.mu_M1, d[f"mk_{H}"])[0, 1]),
        }
        drows += [{"eval": tag, "model": m, "metric": k, **day_diff(v)} for k, v in diffs.items()]
    return pd.DataFrame(rows), pd.DataFrame(drows)


def policy_wmr(va: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """Within-WMR selection at c = 0: accept top fraction of p * mu, threshold from WMR validation rows."""
    z, day = te[f"Z_{H}"].to_numpy(), te.day.to_numpy()
    daily = {}
    for m in MODELS_CMP:
        sv, st = (va[f"p_{m}"] * va[f"mu_{m}"]).to_numpy(), (te[f"p_{m}"] * te[f"mu_{m}"]).to_numpy()
        for fr in ACCEPT:
            a = st >= np.quantile(sv, 1 - fr)
            daily[(m, fr)] = pd.Series(z[a]).groupby(day[a]).mean()
    rows = []
    for fr in ACCEPT:
        for m in MODELS_CMP:
            rows.append({"frac": fr, "model": m, "Z_per_accepted_daymean": daily[(m, fr)].mean(),
                         **{f"vs_M1_{k}": v for k, v in day_diff(daily[(m, fr)] - daily[("M1", fr)]).items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- R2 equal-size evaluation
def fill_stats(s: pd.DataFrame, p: str) -> dict:
    y, q = s.label.to_numpy(float), s[p].to_numpy()
    return {"auc": roc_auc_score(y, q), "pr_auc_lift": average_precision_score(y, q) / y.mean(),
            "brier_skill": 1 - brier(y, q) / (y.mean() * (1 - y.mean()))}


def mk_stats(s: pd.DataFrame, mu: str) -> dict:
    f = s[s.label == 1]
    m, r = f[mu].to_numpy(), f[f"mk_{H}"].to_numpy()
    dec = pd.qcut(pd.Series(m).rank(method="first"), 10, labels=False).to_numpy()
    c = pd.DataFrame({"m": m, "r": r}).groupby(dec).mean()
    return {"mk_corr": np.corrcoef(m, r)[0, 1], "mk_slope": np.polyfit(m, r, 1)[0],
            "decile_spearman": c.m.corr(c.r, method="spearman"), "top_minus_bottom_decile": c.r.iloc[-1] - c.r.iloc[0]}


def equal_n(te: pd.DataFrame, m: str, n_draws: int = 500) -> pd.DataFrame:
    """Normal-hours subsamples made of whole random 5-min windows per day, added until the day's
    row count (fill metrics) or fill count (markout metrics) reaches WMR's count on that day.
    m: pooled model pair, "M0" (M0/R0) or "M1" (M1/R1)."""
    rng = np.random.default_rng(SEED)
    wm = te[te.period == "wmr"]
    nm = te[te.period == "normal"].reset_index(drop=True)
    wins: dict = {}
    for (d, _), idx in nm.groupby([nm.day, nm.clock // 300]).indices.items():
        wins.setdefault(d, []).append(idx)
    lab = nm.label.to_numpy()

    def draw(target: pd.Series, count) -> np.ndarray:
        out = []
        for d, lst in wins.items():
            tot = 0
            for i in rng.permutation(len(lst)):
                out.append(lst[i])
                tot += count(lst[i])
                if tot >= target[d]:
                    break
        return np.concatenate(out)

    rows_t, fills_t = wm.groupby("day").size(), wm.groupby("day").label.sum()
    draws = [{**fill_stats(nm.iloc[draw(rows_t, len)], f"p_{m}"),
              **mk_stats(nm.iloc[draw(fills_t, lambda i: lab[i].sum())], f"mu_{m}")} for _ in range(n_draws)]
    draws = pd.DataFrame(draws)
    full_w = {**fill_stats(wm, f"p_{m}"), **mk_stats(wm, f"mu_{m}")}
    full_n = {**fill_stats(nm, f"p_{m}"), **mk_stats(nm, f"mu_{m}")}
    local_w = {**fill_stats(wm, "p_wmr_only"), **mk_stats(wm, "mu_wmr_only")}
    return pd.DataFrame([{"model": m, "metric": k, "wmr": full_w[k], "wmr_local_wmr_only": local_w[k],
                          "normal_full": full_n[k], "normal_eqn_median": draws[k].median(),
                          "normal_eqn_p2.5": draws[k].quantile(.025), "normal_eqn_p97.5": draws[k].quantile(.975),
                          "share_draws_below_wmr": (draws[k] < full_w[k]).mean(), "n_draws": n_draws,
                          "wmr_rows": len(wm), "wmr_fills": int(wm.label.sum())} for k in draws.columns])


# ---------------------------------------------------------------- R3 precision-weighted day aggregation
def weighted_day_diff(d: pd.Series, w: pd.Series, n_boot: int = 4000, n_flip: int = 20000) -> dict:
    ok = d.notna() & w.notna() & (w > 0)
    d, w = d[ok].to_numpy(float), w[ok].to_numpy(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = (w[idx] * d[idx]).sum(1) / w[idx].sum(1)
    mean = (w * d).sum() / w.sum()
    null = np.abs((rng.choice([-1.0, 1.0], (n_flip, len(d))) * w * d).sum(1) / w.sum())
    return {"w_mean": mean, "w_ci_lo": np.quantile(boot, .025), "w_ci_hi": np.quantile(boot, .975),
            "w_p": (1 + (null >= abs(mean)).sum()) / (1 + n_flip), "max_day_weight_share": w.max() / w.sum()}


def weighted_a1(obs: pd.DataFrame) -> pd.DataFrame:
    """Day weight = 1 / (1/n_wmr + 1/n_ctrl), n = the metric's denominator on that day (rows or fills):
    the inverse variance of a difference of two means under equal per-unit variance."""
    keep = ["fill_rate", "markout_1s_given_fill", f"markout_{H}s_given_fill", "markout_30s_given_fill",
            f"move_{H}s_given_fill", f"Z_{H}s_per_opportunity", f"Z_{H}s_net_c0.1"]
    mets = experiment_a.metrics(obs)
    rows = []
    for name in keep:
        num, den = mets[name]
        t = experiment_a.per_day(obs, num, den)
        n = pd.DataFrame({"day": obs.day, "period": obs.period, "den": np.asarray(den, float)}) \
            .groupby(["day", "period"]).den.sum().unstack("period")
        for ctrl in ("nearby", "normal"):
            d = t.wmr - t[ctrl]
            w = 1 / (1 / n.wmr.where(n.wmr > 0) + 1 / n[ctrl].where(n[ctrl] > 0))
            rows.append({"metric": name, "control": ctrl, **day_diff(d), **weighted_day_diff(d, w)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- R4 within-period deciles
def binned(x: pd.DataFrame, score: str, groupcol: str = "period") -> pd.DataFrame:
    """Deciles of `score` cut within each period; per-bin day-bootstrap CIs."""
    x = x.assign(bin=x.groupby(groupcol)[score].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)))
    rows = []
    for (per, b), s in x.groupby([groupcol, "bin"]):
        d = s.groupby("day").agg(mk=(f"mk_{H}", lambda v: np.nansum(v)), fills=("label", "sum"),
                                 n=("label", "size"), z=(f"Z_{H}", "sum"))
        mk, lo, hi = ratio_ci(d.mk, d.fills)
        z, zlo, zhi = ratio_ci(d.z, d.n)
        rows.append({"period": per, "bin": b, "n": len(s), "n_fills": int(s.label.sum()), "mean_score": s[score].mean(),
                     "fill_rate": s.label.mean(), "mk": mk, "mk_lo": lo, "mk_hi": hi, "Z": z, "Z_lo": zlo, "Z_hi": zhi})
    return pd.DataFrame(rows)


def plots(front: pd.DataFrame, mkdec: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for per in PERIODS:
        s = front[front.period == per].sort_values("bin")
        axes[0].errorbar(s.mean_score, s.mk, yerr=[s.mk - s.mk_lo, s.mk_hi - s.mk], fmt="o-", ms=4, capsize=2,
                         color=COLORS[per], label=f"{per} (fills/bin ~{int(s.n_fills.median()):,})")
        axes[1].errorbar(s.mean_score, s.Z, yerr=[s.Z - s.Z_lo, s.Z_hi - s.Z], fmt="o-", ms=4, capsize=2,
                         color=COLORS[per], label=per)
    for ax in axes:
        ax.axhline(0, color="grey", lw=.8)
        ax.set_xscale("log")
        ax.set_xlabel("walk-forward predicted fill prob. (M0, within-period decile mean, log)")
    axes[0].set_ylabel(f"realized {H} s markout given fill (ticks)")
    axes[1].set_ylabel(f"observed payoff Z_{H} per opportunity (ticks)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("conditional markout", fontsize=10)
    axes[1].set_title("fill-weighted markout", fontsize=10)
    fig.suptitle("R04  F04 redrawn with deciles cut within each period (day-bootstrap 95% CI)")
    fig.tight_layout(); fig.savefig(FIGURES / "R04_frontier_within_period.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    titles = {"M1": "pooled R1 (all periods)", "M0": "pooled R0 (all periods)", "wmr_only": "WMR-only Ridge (WMR fills)"}
    for ax, m in zip(axes, ("M0", "M1", "wmr_only")):
        q = mkdec[m]
        for per in PERIODS:
            s = q[q.period == per].sort_values("bin")
            if len(s):
                ax.errorbar(s.mean_score, s.mk, yerr=[s.mk - s.mk_lo, s.mk_hi - s.mk], fmt="o-", ms=4, capsize=2,
                            color=COLORS[per], label=f"{per} (fills={s.n_fills.sum():,})")
        lo, hi = q.mean_score.min(), q.mean_score.max()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=.8, ls="--")
        ax.axhline(0, color="grey", lw=.5)
        ax.set(title=titles[m], xlabel=f"predicted {H} s markout (ticks, within-period decile mean)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("realized mean markout (ticks), day-bootstrap 95% CI")
    fig.suptitle("R07  F07 redrawn: markout deciles cut within each period, held-out test-day fills")
    fig.tight_layout(); fig.savefig(FIGURES / "R07_markout_deciles_within_period.png", dpi=150); plt.close(fig)


def main() -> None:
    obs = load_observations().drop(columns=DROP)
    days = sorted(obs.day.unique())
    split = split_days(obs)
    preds = pooled_preds(obs)
    sub = obs[obs.period.isin(["wmr", "nearby"])]

    # R1 test days
    loc, hp = fit_local(sub, split["train"], val_days=split["val"])
    print("local hyperparameters (C, alpha):", hp, flush=True)
    x = obs.loc[loc.index].join(preds).join(loc)
    te_w, va_w = x[x.day.isin(split["test"])], x[x.day.isin(split["val"])]
    m_te, d_te = compare_wmr(te_w, "test_days")
    pol = policy_wmr(va_w, te_w)

    # R1 walk-forward (pooled refits use the chosen hyperparameters, as in Experiment B)
    chosen = json.loads((MODELS / "chosen_hyperparameters.json").read_text())
    wf = walk_forward(obs, chosen)
    wl = walk_local(sub, days, hp)
    xw = obs.loc[wl.index].join(wf.rename(columns={"wf_p_M0": "p_M0", "wf_p_M1": "p_M1",
                                                   "wf_mu_R0": "mu_M0", "wf_mu_R1": "mu_M1"})).join(wl)
    m_wf, d_wf = compare_wmr(xw, "walk_forward")
    pd.concat([m_te, m_wf]).assign(hp=lambda t: t.model.map({k: str(v) for k, v in hp.items()})) \
        .to_csv(TABLES / "R1_wmr_models.csv", index=False)
    pd.concat([d_te, d_wf]).to_csv(TABLES / "R1_wmr_day_tests.csv", index=False)
    pol.to_csv(TABLES / "R1_wmr_policy.csv", index=False)
    print("R1 done", flush=True)

    # R2 equal-size evaluation: frozen models on test days, and walk-forward predictions on 46 days
    te = obs[obs.day.isin(split["test"])].join(preds).join(loc)
    wfx = obs.loc[wf.index].join(wf.rename(columns={"wf_p_M0": "p_M0", "wf_p_M1": "p_M1",
                                                    "wf_mu_R0": "mu_M0", "wf_mu_R1": "mu_M1"})).join(wl)
    pd.concat([equal_n(x, m).assign(eval=tag) for tag, x in (("test_days", te), ("walk_forward", wfx))
               for m in ("M0", "M1")]).to_csv(TABLES / "R2_equal_n.csv", index=False)
    print("R2 done", flush=True)

    # R3 precision-weighted A1 tests
    weighted_a1(obs).to_csv(TABLES / "R3_weighted_day_tests.csv", index=False)
    print("R3 done", flush=True)

    # R4 within-period deciles
    front = binned(obs.loc[wf.index].assign(p=wf.wf_p_M0), "p")
    front.to_csv(TABLES / "R4_frontier_within_period.csv", index=False)
    tf = te[te.label == 1]
    mkdec = {m: binned(tf if m != "wmr_only" else tf[tf.period == "wmr"], f"mu_{m}") for m in ("M0", "M1", "wmr_only")}
    pd.concat([v.assign(model=k) for k, v in mkdec.items()]).to_csv(TABLES / "R4_markout_deciles_within_period.csv",
                                                                    index=False)
    plots(front, mkdec)
    print("R4 done", flush=True)


if __name__ == "__main__":
    main()
