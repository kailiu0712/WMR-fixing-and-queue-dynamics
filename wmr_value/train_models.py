"""Experiment B: does adding WMR terms improve out-of-sample fill and markout prediction?

M0/M1: L2 logistic regression for P(fill within 5 s), without/with WMR terms.
R0/R1: Ridge for E[markout_h | filled], fitted on filled orders only (a selected sample).
Days are split chronologically: train / validation (hyperparameters only) / untouched test.
A walk-forward (expanding window, past days only) supplies out-of-sample predictions for
the Experiment A frontier and for the day-by-day comparison in F10.
"""
from __future__ import annotations

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import (COLORS, FIGURES, HORIZONS, LOGIT_C, MODELS, PRIMARY_H, RIDGE_ALPHA, SPLIT, TABLES,
                     WALK_BLOCK, WALK_START)
from .stats import day_diff

WMR_TERMS = ["wmr", "wmr_x_log_queue_ahead", "wmr_x_imbalance", "wmr_x_hit_flow_5s"]


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def design(obs: pd.DataFrame, wmr_terms: bool) -> pd.DataFrame:
    """Small, interpretable feature set; every input is known before the entry packet."""
    hours = obs.clock.to_numpy() / 86400 * 2 * np.pi
    X = pd.DataFrame({
        "bid": (obs.side == "B").astype(float),
        "log_queue_ahead": np.log1p(obs.queue_ahead), "log_orders_ahead": np.log1p(obs.orders_ahead),
        "log_size": np.log(obs["size"]),
        "log_same_depth": np.log1p(obs.same_depth), "log_opp_depth": np.log1p(obs.opp_depth),
        "spread": obs.spread_ticks.clip(upper=5), "imbalance": obs.imbalance,
        "hit_flow_1s": slog(obs.hit_flow_1s), "hit_flow_5s": slog(obs.hit_flow_5s),
        "hit_flow_30s": slog(obs.hit_flow_30s), "log_volume_30s": np.log1p(obs.volume_30s),
        "log_adds_5s": np.log1p(obs.adds_5s), "log_cancels_5s": np.log1p(obs.cancels_5s),
        "log_vol_30s": np.log1p(obs.vol_30s_ticks), "log_since_trade": np.log1p(obs.time_since_trade_s),
        "tod_sin1": np.sin(hours), "tod_cos1": np.cos(hours),
        "tod_sin2": np.sin(2 * hours), "tod_cos2": np.cos(2 * hours),
    }, index=obs.index)
    if wmr_terms:
        w = (obs.period == "wmr").astype(float)
        X["wmr"] = w
        X["wmr_x_log_queue_ahead"] = w * X.log_queue_ahead
        X["wmr_x_imbalance"] = w * X.imbalance
        X["wmr_x_hit_flow_5s"] = w * X.hit_flow_5s
    return X


def split_days(obs: pd.DataFrame) -> dict[str, list]:
    days = sorted(obs.day.unique())
    a, b = round(SPLIT[0] * len(days)), round((SPLIT[0] + SPLIT[1]) * len(days))
    return {"train": days[:a], "val": days[a:b], "test": days[b:]}


def fit_logit(X, y, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500)).fit(X, y)


def fit_ridge(X, y, alpha):
    lo, hi = np.quantile(y, [.01, .99])          # winsorize the training target only
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, np.clip(y, lo, hi))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_int_slope(y, p) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope."""
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    a = 0.0
    for _ in range(50):                            # Newton on the intercept-only offset model
        q = 1 / (1 + np.exp(-(a + lp)))
        a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
    slope = LogisticRegression(C=1e6, max_iter=500).fit(lp[:, None], y).coef_[0, 0] if 0 < y.mean() < 1 else np.nan
    return a, slope


def fill_metrics(y, p) -> dict:
    out = {"n": len(y), "prevalence": y.mean(), "mean_pred": p.mean(), "brier": brier(y, p),
           "logloss": logloss(y, p)}
    ok = 0 < y.mean() < 1
    out["auc"] = roc_auc_score(y, p) if ok else np.nan
    out["pr_auc"] = average_precision_score(y, p) if ok else np.nan
    out["cal_intercept"], out["cal_slope"] = calibration_int_slope(y, p) if ok else (np.nan, np.nan)
    return out


def markout_metrics(y, m) -> dict:
    return {"n": len(y), "mae": np.mean(np.abs(m - y)), "rmse": np.sqrt(np.mean((m - y) ** 2)),
            "bias": m.mean() - y.mean(), "corr": np.corrcoef(m, y)[0, 1] if len(y) > 2 else np.nan}


def select_and_fit(obs: pd.DataFrame, split: dict) -> tuple[dict, dict, pd.DataFrame]:
    """Choose C / alpha on validation days, refit on train only, predict every row."""
    tr, va = obs.day.isin(split["train"]), obs.day.isin(split["val"])
    filled = obs.label == 1
    y = obs.label.to_numpy(float)
    preds = pd.DataFrame(index=obs.index)
    chosen, models = {}, {}
    for name, wmr in (("M0", False), ("M1", True)):
        X = design(obs, wmr)
        scores = {C: logloss(y[va], fit_logit(X[tr], y[tr], C).predict_proba(X[va])[:, 1]) for C in LOGIT_C}
        C = min(scores, key=scores.get)
        models[name] = fit_logit(X[tr], y[tr], C)
        preds[f"p_{name}"] = models[name].predict_proba(X)[:, 1]
        chosen[name] = {"C": C, "val_logloss": scores}
    for name, wmr in (("R0", False), ("R1", True)):
        X = design(obs, wmr)
        for h in HORIZONS:
            t = obs[f"mk_{h}"]
            scores = {a: float(np.mean((fit_ridge(X[tr & filled], t[tr & filled], a).predict(X[va & filled])
                                        - t[va & filled]) ** 2)) for a in RIDGE_ALPHA}
            a = min(scores, key=scores.get)
            models[f"{name}_{h}"] = fit_ridge(X[tr & filled], t[tr & filled], a)
            preds[f"mu_{name}_{h}"] = models[f"{name}_{h}"].predict(X)
            chosen[f"{name}_{h}"] = {"alpha": a, "val_mse": scores}
    # Optional diagnostic: direct Ridge on the full-sample payoff Z (no fill/markout split).
    X = design(obs, False)
    t = obs[f"Z_{PRIMARY_H}"]
    scores = {a: float(np.mean((fit_ridge(X[tr], t[tr], a).predict(X[va]) - t[va]) ** 2)) for a in RIDGE_ALPHA}
    a = min(scores, key=scores.get)
    models["D0"] = fit_ridge(X[tr], t[tr], a)
    preds["v_D0"] = models["D0"].predict(X)
    chosen["D0"] = {"alpha": a, "val_mse": scores}
    return models, chosen, preds


def walk_forward(obs: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    """Expanding-window predictions: each block of days uses models trained on earlier days only."""
    days = sorted(obs.day.unique())
    y = obs.label.to_numpy(float)
    filled = obs.label == 1
    X0, X1 = design(obs, False), design(obs, True)
    t = obs[f"mk_{PRIMARY_H}"]
    out = []
    for start in range(WALK_START, len(days), WALK_BLOCK):
        tr = obs.day.isin(days[:start])
        te = obs.day.isin(days[start:start + WALK_BLOCK])
        part = pd.DataFrame(index=obs.index[te])
        part["wf_p_M0"] = fit_logit(X0[tr], y[tr], chosen["M0"]["C"]).predict_proba(X0[te])[:, 1]
        part["wf_p_M1"] = fit_logit(X1[tr], y[tr], chosen["M1"]["C"]).predict_proba(X1[te])[:, 1]
        part["wf_mu_R0"] = fit_ridge(X0[tr & filled], t[tr & filled], chosen[f"R0_{PRIMARY_H}"]["alpha"]).predict(X0[te])
        part["wf_mu_R1"] = fit_ridge(X1[tr & filled], t[tr & filled], chosen[f"R1_{PRIMARY_H}"]["alpha"]).predict(X1[te])
        part["wf_block"] = start
        out.append(part)
        print(f"  walk-forward block starting day {start} ({days[start]}) done", flush=True)
    return pd.concat(out)


def evaluate(obs: pd.DataFrame, preds: pd.DataFrame, split: dict) -> dict[str, pd.DataFrame]:
    te = obs.day.isin(split["test"])
    x = obs[te].join(preds)
    rows, mrows, qrows, drows = [], [], [], []
    groups = {"all": x, **{p: x[x.period == p] for p in ("wmr", "nearby", "normal")},
              "bid": x[x.side == "B"], "ask": x[x.side == "A"],
              "wmr_no_boundary": x[(x.period == "wmr") & ~x.crosses_boundary]}
    qa = pd.qcut(x.queue_ahead, 4, duplicates="drop")
    for i, b in enumerate(qa.cat.categories):
        groups[f"queue_q{i + 1}"] = x[qa == b]
    sp = np.where(x.spread_ticks > 1, "spread_wide", "spread_1tick")
    for g in ("spread_1tick", "spread_wide"):
        groups[g] = x[sp == g]
    for g, sub in groups.items():
        for m in ("M0", "M1"):
            rows.append({"group": g, "model": m, **fill_metrics(sub.label.to_numpy(float), sub[f"p_{m}"].to_numpy())})
        f = sub[sub.label == 1]
        for h in HORIZONS:
            for m in ("R0", "R1"):
                if len(f) > 10:
                    mrows.append({"group": g, "model": m, "h": h,
                                  **markout_metrics(f[f"mk_{h}"].to_numpy(), f[f"mu_{m}_{h}"].to_numpy())})
    # predicted vs realized markout by prediction decile (pooled cut points), primary horizon
    f = x[x.label == 1]
    for m in ("R0", "R1"):
        dec = pd.qcut(f[f"mu_{m}_{PRIMARY_H}"], 10, labels=False, duplicates="drop")
        for (per, d), s in f.groupby([f.period, dec]):
            qrows.append({"model": m, "period": per, "decile": d, "n": len(s),
                          "pred": s[f"mu_{m}_{PRIMARY_H}"].mean(), "real": s[f"mk_{PRIMARY_H}"].mean()})
    # day-level incremental value of WMR terms on the test days
    for per in ("wmr", "nearby", "normal", "all"):
        s = x if per == "all" else x[x.period == per]
        g = s.groupby("day")
        db = g.apply(lambda d: brier(d.label, d.p_M0) - brier(d.label, d.p_M1))
        dl = g.apply(lambda d: logloss(d.label, d.p_M0) - logloss(d.label, d.p_M1))
        sf = s[s.label == 1].groupby("day")
        dm = sf.apply(lambda d: np.mean(np.abs(d[f"mu_R0_{PRIMARY_H}"] - d[f"mk_{PRIMARY_H}"]))
                      - np.mean(np.abs(d[f"mu_R1_{PRIMARY_H}"] - d[f"mk_{PRIMARY_H}"])))
        bias = sf.apply(lambda d: (d[f"mu_R0_{PRIMARY_H}"] - d[f"mk_{PRIMARY_H}"]).mean())
        fb = g.apply(lambda d: (d.p_M0 - d.label).mean())
        for metric, v in (("brier_M0_minus_M1", db), ("logloss_M0_minus_M1", dl),
                          (f"mae{PRIMARY_H}_R0_minus_R1", dm), (f"bias{PRIMARY_H}_R0", bias),
                          ("fill_bias_M0", fb)):
            drows.append({"period": per, "metric": metric, **day_diff(v)})
    # time-spaced subsample: at most one opportunity per side per 10 s
    sub = x.assign(slot=x.clock // 10).drop_duplicates(["day", "side", "slot"])
    for per in ("wmr", "normal"):
        s = sub[sub.period == per]
        v = s.groupby("day").apply(lambda d: brier(d.label, d.p_M0) - brier(d.label, d.p_M1))
        drows.append({"period": per, "metric": "brier_M0_minus_M1_spaced10s", **day_diff(v)})
    return {"B_fill_metrics": pd.DataFrame(rows), "B_markout_metrics": pd.DataFrame(mrows),
            "B_markout_deciles": pd.DataFrame(qrows), "B_day_tests": pd.DataFrame(drows)}


def coefficients(models: dict) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        est = m[-1]
        coef = est.coef_.ravel()
        names = m[0].feature_names_in_
        rows += [{"model": name, "feature": f, "std_coef": c} for f, c in zip(names, coef)]
    return pd.DataFrame(rows)


def coef_stability(obs: pd.DataFrame, split: dict, chosen: dict) -> pd.DataFrame:
    """WMR-term coefficients refitted on each half of the training days."""
    tr_days = split["train"]
    halves = {"first_half": tr_days[: len(tr_days) // 2], "second_half": tr_days[len(tr_days) // 2:],
              "all_train": tr_days}
    X = design(obs, True)
    rows = []
    for name, days in halves.items():
        m = obs.day.isin(days)
        f = m & (obs.label == 1)
        lg = fit_logit(X[m], obs.label[m], chosen["M1"]["C"])
        rg = fit_ridge(X[f], obs[f"mk_{PRIMARY_H}"][f], chosen[f"R1_{PRIMARY_H}"]["alpha"])
        for model, est in (("M1", lg), ("R1", rg)):
            coef = pd.Series(est[-1].coef_.ravel(), X.columns)
            rows += [{"fit": name, "model": model, "feature": t, "std_coef": coef[t]} for t in WMR_TERMS]
    return pd.DataFrame(rows)


def plots(obs: pd.DataFrame, preds: pd.DataFrame, split: dict, tabs: dict, wf: pd.DataFrame) -> None:
    te = obs.day.isin(split["test"])
    x = obs[te].join(preds)
    # F05 calibration
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, m in zip(axes, ("M0", "M1")):
        for per in ("wmr", "nearby", "normal"):
            s = x[x.period == per]
            b = pd.qcut(s[f"p_{m}"], 10, duplicates="drop")
            g = s.groupby(b, observed=True).agg(p=(f"p_{m}", "mean"), y=("label", "mean"), n=("label", "size"))
            ax.plot(g.p, g.y, "o-", color=COLORS[per], ms=4, label=f"{per} (n={len(s):,}, {s.day.nunique()} days)")
        lim = max(0.6, ax.get_xlim()[1])
        ax.plot([0, lim], [0, lim], color="grey", lw=.8, ls="--")
        ax.set(title=f"{m}: {'ordinary features' if m == 'M0' else '+ WMR terms'}", xlabel="predicted 5 s fill probability (decile mean)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("observed 5 s fill frequency")
    fig.suptitle("F05  Fill-probability calibration on held-out test days (observed policy)")
    fig.tight_layout(); fig.savefig(FIGURES / "F05_calibration.png", dpi=150); plt.close(fig)

    # F06 model comparison with day-level CIs
    dt = tabs["B_day_tests"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in ((axes[0], "brier_M0_minus_M1", "Brier(M0) - Brier(M1)\n>0: WMR terms help"),
                              (axes[1], f"mae{PRIMARY_H}_R0_minus_R1", f"MAE(R0) - MAE(R1), {PRIMARY_H} s markout, ticks\n>0: WMR terms help")):
        s = dt[(dt.metric == metric) & dt.period.isin(["wmr", "nearby", "normal"])].set_index("period").loc[["wmr", "nearby", "normal"]]
        ax.errorbar(range(3), s["mean"], yerr=[s["mean"] - s.ci_lo, s.ci_hi - s["mean"]], fmt="o",
                    color="k", capsize=4)
        for i, (per, r) in enumerate(s.iterrows()):
            ax.scatter(i, r["mean"], color=COLORS[per], s=60, zorder=3)
            ax.annotate(f"p={r.p:.2f}\n{int(r.n_days)} days", (i, r.ci_hi), textcoords="offset points",
                        xytext=(6, 0), fontsize=8)
        ax.axhline(0, color="grey", lw=.8)
        ax.set_xticks(range(3), s.index)
        ax.set_title(title, fontsize=10)
    fig.suptitle("F06  Incremental out-of-sample value of WMR terms (test days, day-bootstrap 95% CI)")
    fig.tight_layout(); fig.savefig(FIGURES / "F06_model_comparison.png", dpi=150); plt.close(fig)

    # F07 predicted vs realized markout
    q = tabs["B_markout_deciles"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, m in zip(axes, ("R0", "R1")):
        for per in ("wmr", "nearby", "normal"):
            s = q[(q.model == m) & (q.period == per)]
            k = s[s.n >= 30]                             # bins with >= 30 fills only
            ax.plot(k.pred, k.real, "o-", color=COLORS[per], ms=4, label=f"{per} (fills={s.n.sum():,})")
        lo, hi = q.pred.min(), q.pred.max()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=.8, ls="--")
        ax.set(title=m, xlabel=f"predicted {PRIMARY_H} s markout given fill (ticks, decile mean)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("realized mean markout (ticks)")
    fig.suptitle("F07  Conditional markout: predicted vs realized on test-day fills (pooled deciles, bins >= 30 fills)")
    fig.tight_layout(); fig.savefig(FIGURES / "F07_markout_deciles.png", dpi=150); plt.close(fig)

    # F10 walk-forward incremental value per day
    w = obs.loc[wf.index].join(wf)
    rows = []
    for (day, per), d in w[w.period.isin(["wmr", "normal"])].groupby(["day", "period"]):
        rows.append({"day": day, "period": per, "d": brier(d.label, d.wf_p_M0) - brier(d.label, d.wf_p_M1)})
    r = pd.DataFrame(rows).pivot(index="day", columns="period", values="d")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xs = np.arange(len(r))
    ax.bar(xs - .2, r.wmr * 1e3, .4, color=COLORS["wmr"], label="WMR window")
    ax.bar(xs + .2, r.normal * 1e3, .4, color=COLORS["normal"], label="normal hours")
    ax.axhline(0, color="grey", lw=.8)
    st_w, st_n = day_diff(r.wmr), day_diff(r.normal)
    ax.set_title(f"F10  Walk-forward Brier(M0) - Brier(M1), x1e3, per day (>0: WMR terms help)\n"
                 f"WMR mean {st_w['mean']*1e3:.2f} [{st_w['ci_lo']*1e3:.2f}, {st_w['ci_hi']*1e3:.2f}], "
                 f"better on {st_w['share_pos']:.0%} of {st_w['n_days']} days;  normal mean {st_n['mean']*1e3:.3f}", fontsize=10)
    ax.set_xticks(xs[::5], [d[5:] for d in r.index[::5]], fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "F10_walkforward_days.png", dpi=150); plt.close(fig)


def run(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    split = split_days(obs)
    print("split:", {k: (v[0], v[-1], len(v)) for k, v in split.items()}, flush=True)
    models, chosen, preds = select_and_fit(obs, split)
    print("chosen:", {k: v.get("C", v.get("alpha")) for k, v in chosen.items()}, flush=True)
    with open(MODELS / "models.pkl", "wb") as fh:
        pickle.dump(models, fh)
    (MODELS / "chosen_hyperparameters.json").write_text(json.dumps(
        {k: {kk: (vv if not isinstance(vv, dict) else {str(a): b for a, b in vv.items()}) for kk, vv in v.items()}
         for k, v in chosen.items()}, indent=1))
    (MODELS / "split.json").write_text(json.dumps(split, indent=1))
    tabs = evaluate(obs, preds, split)
    tabs["B_coefficients"] = coefficients(models)
    tabs["B_coef_stability"] = coef_stability(obs, split, chosen)
    wf = walk_forward(obs, chosen)
    for k, v in tabs.items():
        v.to_csv(TABLES / f"{k}.csv", index=False)
    plots(obs, preds, split, tabs, wf)
    return preds, wf, split
