"""Experiment A: fill probability vs maker markout, WMR vs nearby windows vs normal liquid hours.

Population: actual historical passive order entries at the touch (observed placement and
cancellation policy). Fill rates are NOT the fill probability of an arbitrary new quote.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import COLORS, COSTS, FIGURES, HORIZONS, PERIODS, PRIMARY_H, TABLES
from .stats import day_diff, ratio_ci

H = PRIMARY_H


def per_day(obs: pd.DataFrame, num, den) -> pd.DataFrame:
    """Per-day ratio sum(num)/sum(den), one column per period."""
    x = pd.DataFrame({"day": obs.day, "period": obs.period, "num": np.nan_to_num(np.asarray(num, float)),
                      "den": np.asarray(den, float)})
    g = x.groupby(["day", "period"])[["num", "den"]].sum()
    return (g.num / g.den.where(g.den > 0)).unstack("period")


def metrics(obs: pd.DataFrame) -> dict[str, tuple]:
    """name -> (numerator, denominator); per-day value = sum(num)/sum(den)."""
    y = obs.label.to_numpy(float)
    one = np.ones(len(obs))
    out = {"fill_rate": (y, one)}
    for h in HORIZONS:
        out[f"markout_{h}s_given_fill"] = (obs[f"mk_{h}"].fillna(0) * y, y)
        out[f"neg_markout_{h}s_share_of_fills"] = ((obs[f"mk_{h}"] < 0) * y, y)
        out[f"move_{h}s_given_fill"] = (obs[f"move_{h}"].fillna(0) * y, y)
        out[f"Z_{h}s_per_opportunity"] = (obs[f"Z_{h}"], one)
    out["spread_capture_given_fill"] = (obs.spread_capture.fillna(0) * y, y)
    for c in COSTS[1:]:
        out[f"Z_{H}s_net_c{c}"] = (y * (obs[f"mk_{H}"].fillna(0) - c), one)
    return out


def a1_descriptive(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sides = {"both": obs, "bid": obs[obs.side == "B"], "ask": obs[obs.side == "A"]}
    for side, s in sides.items():
        for per in PERIODS + ["wmr_no_boundary"]:
            p = s[(s.period == "wmr") & ~s.crosses_boundary] if per == "wmr_no_boundary" else s[s.period == per]
            f = p[p.label == 1]
            r = {"side": side, "period": per, "n_days": p.day.nunique(), "n": len(p), "n_fills": len(f),
                 "fill_rate": p.label.mean(), "fill_time_p25": f.fill_time_s.quantile(.25),
                 "fill_time_median": f.fill_time_s.median(), "fill_time_p75": f.fill_time_s.quantile(.75),
                 "spread_capture_mean": f.spread_capture.mean(), "fill_qty_mean": f.fill_qty_5s.mean()}
            for h in HORIZONS:
                r[f"mk{h}_mean"] = f[f"mk_{h}"].mean()
                r[f"mk{h}_median"] = f[f"mk_{h}"].median()
                r[f"mk{h}_neg_share"] = (f[f"mk_{h}"] < 0).mean()
                r[f"move{h}_mean"] = f[f"move_{h}"].mean()
                r[f"Z{h}_mean"] = p[f"Z_{h}"].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def a1_tests(obs: pd.DataFrame) -> pd.DataFrame:
    """Day-paired WMR minus control, for each metric, side, and boundary sensitivity."""
    rows = []
    variants = {"both": obs, "bid": obs[obs.side == "B"], "ask": obs[obs.side == "A"],
                "both_no_boundary": obs[~obs.crosses_boundary]}
    for var, s in variants.items():
        for name, (num, den) in metrics(s).items():
            t = per_day(s, num, den)
            for ctrl in ("nearby", "normal"):
                d = (t.wmr - t[ctrl]).dropna()
                st = day_diff(d)
                loo = [(d.sum() - v) / (len(d) - 1) for v in d] if len(d) > 2 else [np.nan]
                rows.append({"sample": var, "metric": name, "control": ctrl, "wmr": t.wmr.mean(),
                             "control_mean": t[ctrl].mean(), **st,
                             "loo_min": np.min(loo), "loo_max": np.max(loo)})
    return pd.DataFrame(rows)


def cells(obs: pd.DataFrame, activity: bool = False) -> np.ndarray:
    """Coarse strata of the pre-entry state: queue ahead x imbalance x recent flow x spread x side."""
    qa = np.digitize(obs.queue_ahead, [1, 11, 41])
    imb = np.digitize(obs.imbalance, np.quantile(obs.imbalance, [1 / 3, 2 / 3]))
    flow = np.sign(obs.hit_flow_5s).astype(int) + 1
    code = (((qa * 3 + imb) * 3 + flow) * 2 + (obs.spread_ticks > 1)) * 2 + (obs.side == "B")
    if activity:
        code = code * 3 + np.digitize(obs.volume_30s, np.quantile(obs.volume_30s, [1 / 3, 2 / 3]))
    return np.asarray(code)


def matched(obs: pd.DataFrame, num, den, cell, ctrl: str, min_n: int = 5) -> pd.DataFrame:
    """Per day: WMR mean and control mean reweighted to WMR's cell mix, on common support."""
    x = pd.DataFrame({"day": obs.day, "period": obs.period, "cell": cell,
                      "num": np.nan_to_num(np.asarray(num, float)), "den": np.asarray(den, float)})
    x = x[x.period.isin(["wmr", ctrl])]
    g = x.groupby(["day", "cell", "period"])[["num", "den"]].sum().unstack("period").fillna(0)
    g = g[g[("den", "wmr")] > 0]
    cov = g[("den", ctrl)] >= min_n
    w = g[("den", "wmr")]
    ctrl_mean = g[("num", ctrl)] / g[("den", ctrl)].where(cov)
    day = g.index.get_level_values("day")
    out = pd.DataFrame({
        "wmr": g[("num", "wmr")].where(cov, 0).groupby(day).sum() / w.where(cov, 0).groupby(day).sum(),
        "ctrl": (w * ctrl_mean).where(cov, 0).groupby(day).sum() / w.where(cov, 0).groupby(day).sum(),
        "coverage": w.where(cov, 0).groupby(day).sum() / w.groupby(day).sum()})
    return out


def a2_matched(obs: pd.DataFrame, wf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = obs.label.to_numpy(float)
    one = np.ones(len(obs))
    targets = {"fill_rate": (y, one),
               f"markout_{H}s_given_fill": (obs[f"mk_{H}"].fillna(0) * y, y),
               "markout_1s_given_fill": (obs["mk_1"].fillna(0) * y, y),
               "markout_30s_given_fill": (obs["mk_30"].fillna(0) * y, y),
               f"move_{H}s_given_fill": (obs[f"move_{H}"].fillna(0) * y, y),
               f"Z_{H}s_per_opportunity": (obs[f"Z_{H}"], one)}
    p_bin = pd.Series(-1, index=obs.index)
    p_bin.loc[wf.index] = pd.qcut(wf.wf_p_M0, 10, labels=False)
    strata = {"queue x imbalance x flow x spread x side": cells(obs),
              "... + 30 s traded volume": cells(obs, activity=True),
              "walk-forward M0 fill-probability decile x side": (p_bin * 2 + (obs.side == "B")).to_numpy()}
    for sname, cell in strata.items():
        keep = (p_bin >= 0).to_numpy() if sname.startswith("walk") else np.ones(len(obs), bool)
        for tname, (num, den) in targets.items():
            for ctrl in ("nearby", "normal"):
                m = matched(obs[keep], np.asarray(num)[keep], np.asarray(den)[keep], cell[keep], ctrl,
                            min_n=5 if ctrl == "normal" else 3)
                rows.append({"strata": sname, "metric": tname, "control": ctrl, "wmr": m.wmr.mean(),
                             "control_matched": m.ctrl.mean(), "coverage": m.coverage.mean(),
                             **day_diff(m.wmr - m.ctrl)})
    return pd.DataFrame(rows)


def a2_strata(obs: pd.DataFrame) -> pd.DataFrame:
    """One-variable stratified summaries with sample sizes (descriptive)."""
    bins = {"queue_ahead": pd.cut(obs.queue_ahead, [-1, 0, 10, 40, np.inf], labels=["0", "1-10", "11-40", ">40"]),
            "imbalance": pd.qcut(obs.imbalance, 3, labels=["low", "mid", "high"]),
            "hit_flow_5s": pd.Series(np.sign(obs.hit_flow_5s), index=obs.index).map({-1: "away", 0: "none", 1: "toward"}),
            "spread": np.where(obs.spread_ticks > 1, ">1 tick", "1 tick"),
            "side": obs.side.map({"B": "bid", "A": "ask"})}
    rows = []
    for var, b in bins.items():
        for (lvl, per), s in obs.groupby([np.asarray(b), obs.period]):
            f = s[s.label == 1]
            rows.append({"variable": var, "bin": lvl, "period": per, "n": len(s), "n_days": s.day.nunique(),
                         "n_fills": len(f), "fill_rate": s.label.mean(), f"mk{H}_given_fill": f[f"mk_{H}"].mean(),
                         f"Z{H}": s[f"Z_{H}"].mean()})
    return pd.DataFrame(rows)


def a3_frontier(obs: pd.DataFrame, wf: pd.DataFrame) -> pd.DataFrame:
    x = obs.loc[wf.index].assign(p=wf.wf_p_M0)
    x["bin"] = pd.qcut(x.p, 10, labels=False)
    rows = []
    for (per, b), s in x.groupby(["period", "bin"]):
        d = s.groupby("day").agg(mk=(f"mk_{H}", lambda v: np.nansum(v)), fills=("label", "sum"),
                                 n=("label", "size"), z=(f"Z_{H}", "sum"))
        mk, lo, hi = ratio_ci(d.mk, d.fills)
        fr, flo, fhi = ratio_ci(d.fills, d.n)
        rows.append({"period": per, "bin": b, "n": len(s), "n_fills": int(s.label.sum()), "n_days": len(d),
                     "mean_pred_fill": s.p.mean(), "fill_rate": fr, "fill_lo": flo, "fill_hi": fhi,
                     f"mk{H}_given_fill": mk, "mk_lo": lo, "mk_hi": hi, f"Z{H}": s[f"Z_{H}"].mean()})
    return pd.DataFrame(rows)


def plots(obs: pd.DataFrame, front: pd.DataFrame) -> None:
    # F01 headline bars
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (title, numf, denf) in zip(axes, [
            ("5 s fill frequency", lambda s: s.label, lambda s: np.ones(len(s))),
            (f"{H} s markout given fill (ticks)", lambda s: s[f"mk_{H}"].fillna(0) * s.label, lambda s: s.label),
            (f"payoff Z = Y x markout_{H}s (ticks/opportunity)", lambda s: s[f"Z_{H}"], lambda s: np.ones(len(s)))]):
        for i, per in enumerate(PERIODS):
            s = obs[obs.period == per]
            d = pd.DataFrame({"n": numf(s), "d": denf(s), "day": s.day}).groupby("day").sum()
            v, lo, hi = ratio_ci(d.n, d.d)
            ax.bar(i, v, color=COLORS[per], yerr=[[v - lo], [hi - v]], capsize=4)
        ax.set_xticks(range(3), [f"{p}\nn={(obs.period == p).sum():,}" for p in PERIODS], fontsize=8)
        ax.axhline(0, color="grey", lw=.8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("F01  Historical passive entries at the touch, 66 days: fill, markout, observed payoff (day-bootstrap 95% CI)")
    fig.tight_layout(); fig.savefig(FIGURES / "F01_headline.png", dpi=150); plt.close(fig)

    # F02 markout distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        for per in PERIODS:
            v = obs.loc[(obs.period == per) & (obs.label == 1), f"mk_{h}"].clip(-4, 4)
            share = v.round(1).value_counts(normalize=True).sort_index()
            ax.plot(share.index, share.values, "o-", ms=3, color=COLORS[per],
                    label=f"{per}: mean {v.mean():+.3f}, <0 {np.mean(v < 0):.0%}")
        ax.axvline(0, color="grey", lw=.8)
        ax.set(title=f"{h} s after fill", xlabel="maker markout, ticks (clipped at +/-4)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("share of fills")
    fig.suptitle("F02  Distribution of signed maker markout on filled historical entries")
    fig.tight_layout(); fig.savefig(FIGURES / "F02_markout_distributions.png", dpi=150); plt.close(fig)

    # F03 spread capture vs subsequent mid move
    fig, ax = plt.subplots(figsize=(9, 4))
    width = .25
    for i, per in enumerate(PERIODS):
        f = obs[(obs.period == per) & (obs.label == 1)]
        d = f.groupby("day")
        sc = ratio_ci(d.spread_capture.sum(), d.size())
        ax.bar(-1 + i * width, sc[0], width, color=COLORS[per], yerr=[[sc[0] - sc[1]], [sc[2] - sc[0]]], capsize=3,
               label=per)
        for j, h in enumerate(HORIZONS):
            mv = ratio_ci(d[f"move_{h}"].sum(), d.size())
            tot = ratio_ci(d[f"mk_{h}"].sum(), d.size())
            ax.bar(j + i * width, mv[0], width, color=COLORS[per], alpha=.55,
                   yerr=[[mv[0] - mv[1]], [mv[2] - mv[0]]], capsize=3)
            ax.plot(j + i * width, tot[0], "k_", ms=14, mew=2)
    ax.axhline(0, color="grey", lw=.8)
    ax.set_xticks([-1 + width] + [j + width for j in range(3)],
                  ["spread capture\ns(m_pre - p)"] + [f"mid move {h} s\ns(m_t+h - m_pre)" for h in HORIZONS])
    ax.set_ylabel("ticks per filled contract")
    ax.set_title("F03  Markout = spread capture + post-fill mid move (black dash = total markout)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "F03_decomposition.png", dpi=150); plt.close(fig)

    # F04 fill-markout frontier
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for per in PERIODS:
        s = front[front.period == per].sort_values("bin")
        k = s[s.n_fills >= 30]                          # markout shown only where >= 30 fills
        axes[0].errorbar(k.mean_pred_fill, k[f"mk{H}_given_fill"],
                         yerr=[k[f"mk{H}_given_fill"] - k.mk_lo, k.mk_hi - k[f"mk{H}_given_fill"]],
                         fmt="o-", ms=4, capsize=2, color=COLORS[per], label=f"{per} (fills={s.n_fills.sum():,})")
        axes[1].plot(s.mean_pred_fill, s[f"Z{H}"], "o-", ms=4, color=COLORS[per], label=per)
        if per == "wmr":
            for _, r in k.iterrows():
                axes[0].annotate(str(r.n_fills), (r.mean_pred_fill, r.mk_hi), fontsize=7, color=COLORS["wmr"],
                                 ha="center", va="bottom")
    for ax in axes:
        ax.axhline(0, color="grey", lw=.8)
        ax.set_xscale("log")
        ax.set_xlabel("walk-forward predicted 5 s fill probability (M0 decile mean, log scale)")
    axes[0].set_ylabel(f"realized {H} s markout given fill (ticks)")
    axes[1].set_ylabel(f"observed payoff Z_{H} per opportunity (ticks)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("conditional markout, bins with >= 30 fills (numbers: WMR fills)", fontsize=10)
    axes[1].set_title("fill-weighted markout", fontsize=10)
    fig.suptitle("F04  Fill-probability vs post-fill-markout, by out-of-sample predicted fill decile (observational)")
    fig.tight_layout(); fig.savefig(FIGURES / "F04_frontier.png", dpi=150); plt.close(fig)


def run(obs: pd.DataFrame, wf: pd.DataFrame) -> None:
    tabs = {"A1_descriptive": a1_descriptive(obs), "A1_tests": a1_tests(obs),
            "A2_matched": a2_matched(obs, wf), "A2_strata": a2_strata(obs), "A3_frontier": a3_frontier(obs, wf)}
    for k, v in tabs.items():
        v.to_csv(TABLES / f"{k}.csv", index=False)
    plots(obs, tabs["A3_frontier"])
