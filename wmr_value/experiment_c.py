"""Experiment C: does a predicted fill-weighted markout rank historical opportunities better?

Observational only: each policy selects among historically observed order entries and is
scored with their realized outcomes under the participants' own cancellation decisions.
Nothing here is a counterfactual P&L for orders we would have placed.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ACCEPT, COLORS, COSTS, FIGURES, PRIMARY_H, TABLES
from .stats import day_diff, ratio_ci

H = PRIMARY_H
POLICIES = ["P0_all", "P1_fill", "P2_markout", "P3_joint", "P4_joint_wmr"]


def scores(p0, mu0, p1, mu1, c: float) -> dict[str, np.ndarray]:
    return {"P0_all": np.zeros(len(p0)), "P1_fill": p0, "P2_markout": mu0,
            "P3_joint": p0 * (mu0 - c), "P4_joint_wmr": p1 * (mu1 - c)}


def summarize(x: pd.DataFrame, accepted: np.ndarray, z: np.ndarray) -> dict:
    a = x[accepted]
    za = z[accepted]
    f = a.label == 1
    d = pd.DataFrame({"day": a.day, "z": za, "one": 1.0}).groupby("day").sum()
    v, lo, hi = ratio_ci(d.z, d.one) if len(d) else (np.nan,) * 3
    return {"n_accepted": len(a), "share_accepted": len(a) / max(len(x), 1), "fill_freq": f.mean(),
            "markout_given_fill": a.loc[f, f"mk_{H}"].mean(), "Z_per_accepted": v, "Z_lo": lo, "Z_hi": hi,
            "neg_markout_share_of_fills": (a.loc[f, f"mk_{H}"] < 0).mean(), "Z_negative_share": (za < 0).mean(),
            "day_sd_of_Z": (d.z / d.one).std(), "n_days": len(d)}


def accept(sv: dict, st: dict, va_period=None, te_period=None) -> dict:
    """Accept the top fraction by score; thresholds from validation days only.
    With periods given, thresholds are set within each period (matched budget inside WMR)."""
    acc = {("P0_all", 1.0): np.ones(len(st["P0_all"]), bool)}
    for p in POLICIES[1:]:
        for f in ACCEPT:
            if va_period is None:
                acc[(p, f)] = st[p] >= np.quantile(sv[p], 1 - f)
                continue
            a = np.zeros(len(st[p]), bool)
            for per in np.unique(te_period):
                a[te_period == per] = st[p][te_period == per] >= np.quantile(sv[p][va_period == per], 1 - f)
            acc[(p, f)] = a
    return acc


def policy_table(x: pd.DataFrame, acc: dict, z: np.ndarray) -> pd.DataFrame:
    rows = []
    groups = {"all": np.ones(len(x), bool), **{p: (x.period == p).to_numpy() for p in ("wmr", "nearby", "normal")},
              "bid": (x.side == "B").to_numpy(), "ask": (x.side == "A").to_numpy()}
    for (pol, frac), a in acc.items():
        for g, m in groups.items():
            rows.append({"policy": pol, "target_frac": frac, "group": g, **summarize(x[m], a[m], z[m])})
    return pd.DataFrame(rows)


def paired_policies(x: pd.DataFrame, accepted_by: dict, z: np.ndarray, tag: str) -> pd.DataFrame:
    """Per-day mean Z per accepted opportunity, paired across policies."""
    pairs = [("P3_joint", "P0_all"), ("P3_joint", "P1_fill"), ("P3_joint", "P2_markout"), ("P4_joint_wmr", "P3_joint")]
    rows = []
    for g in ("all", "wmr", "normal"):
        m = np.ones(len(x), bool) if g == "all" else (x.period == g).to_numpy()
        for frac in ACCEPT:
            for a, b in pairs:
                def daily(pol):
                    acc = accepted_by[(pol, 1.0 if pol == "P0_all" else frac)] & m
                    return pd.Series(z[acc]).groupby(x.day.to_numpy()[acc]).mean()
                d = (daily(a) - daily(b)).dropna()
                rows.append({"eval": tag, "group": g, "frac": frac, "comparison": f"{a} - {b}", **day_diff(d)})
    return pd.DataFrame(rows)


def run(obs: pd.DataFrame, preds: pd.DataFrame, wf: pd.DataFrame, split: dict) -> None:
    x = obs.join(preds)
    va, te = x[x.day.isin(split["val"])], x[x.day.isin(split["test"])].reset_index(drop=True)
    tabs, paired = [], []
    for c in COSTS:
        z = (te.label * (te[f"mk_{H}"].fillna(0) - c)).to_numpy()
        sv = scores(va.p_M0.to_numpy(), va[f"mu_R0_{H}"].to_numpy(), va.p_M1.to_numpy(), va[f"mu_R1_{H}"].to_numpy(), c)
        st = scores(te.p_M0.to_numpy(), te[f"mu_R0_{H}"].to_numpy(), te.p_M1.to_numpy(), te[f"mu_R1_{H}"].to_numpy(), c)
        acc = accept(sv, st)
        t = policy_table(te, acc, z)
        tabs.append(t.assign(cost=c, budget="global"))
        paired.append(paired_policies(te, acc, z, "test_days").assign(cost=c))
        if c == 0.0:
            main_tab = t
            acc_w = accept(sv, st, va.period.to_numpy(), te.period.to_numpy())
            within_tab = policy_table(te, acc_w, z)
            tabs.append(within_tab.assign(cost=c, budget="within_period"))
            paired.append(paired_policies(te, acc_w, z, "test_days_within_period_budget").assign(cost=c))
            v_rows = []                                  # predicted value vs realized payoff
            for pol in ("P3_joint", "P4_joint_wmr"):            # deciles within each period
                dec = pd.Series(st[pol]).groupby(te.period.to_numpy()).transform(
                    lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)).to_numpy()
                for (per, d), s in te.assign(v=st[pol], z=z).groupby([te.period, dec]):
                    v_rows.append({"policy": pol, "period": per, "decile": d, "n": len(s),
                                   "pred_V": s.v.mean(), "real_Z": s.z.mean()})
            vdec = pd.DataFrame(v_rows)
            mse = {k: float(np.mean((v - z) ** 2)) for k, v in
                   {"P3 two-stage": st["P3_joint"], "P4 two-stage": st["P4_joint_wmr"],
                    "D0 direct ridge": te.v_D0.to_numpy(), "constant (val mean)": np.full(len(z), va[f"Z_{H}"].mean())}.items()}
            pd.Series(mse, name="test_mse_vs_Z").to_csv(TABLES / "C_value_mse.csv")

    # Walk-forward robustness: thresholds from the previous block's out-of-sample scores (past only).
    w = obs.loc[wf.index].join(wf).reset_index(drop=True)
    z = w[f"Z_{H}"].to_numpy()
    s_all = scores(w.wf_p_M0.to_numpy(), w.wf_mu_R0.to_numpy(), w.wf_p_M1.to_numpy(), w.wf_mu_R1.to_numpy(), 0.0)
    blocks = sorted(w.wf_block.unique())
    acc = {(p, f): np.zeros(len(w), bool) for p in POLICIES[1:] for f in ACCEPT}
    acc[("P0_all", 1.0)] = np.ones(len(w), bool)
    keep = np.zeros(len(w), bool)
    for prev, cur in zip(blocks[:-1], blocks[1:]):
        pm, cm = (w.wf_block == prev).to_numpy(), (w.wf_block == cur).to_numpy()
        keep |= cm
        for p in POLICIES[1:]:
            for f in ACCEPT:
                acc[(p, f)][cm] = s_all[p][cm] >= np.quantile(s_all[p][pm], 1 - f)
    wk = w[keep].reset_index(drop=True)
    paired.append(paired_policies(wk, {k: v[keep] for k, v in acc.items()}, z[keep], "walk_forward_days").assign(cost=0.0))
    pd.concat(tabs).to_csv(TABLES / "C_policies.csv", index=False)
    pd.concat(paired).to_csv(TABLES / "C_paired.csv", index=False)
    vdec.to_csv(TABLES / "C_value_deciles.csv", index=False)
    plots(main_tab, within_tab, vdec)


def plots(t: pd.DataFrame, tw: pd.DataFrame, vdec: pd.DataFrame) -> None:
    # F08 realized payoff by predicted-value decile
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, pol in zip(axes, ("P3_joint", "P4_joint_wmr")):
        for per in ("wmr", "nearby", "normal"):
            s = vdec[(vdec.policy == pol) & (vdec.period == per)]
            ax.plot(s.decile + 1, s.real_Z, "o-", ms=4, color=COLORS[per], label=f"{per} realized (n={s.n.sum():,})")
            ax.plot(s.decile + 1, s.pred_V, ":", color=COLORS[per], lw=1)
        ax.axhline(0, color="grey", lw=.8)
        ax.set(title=f"{pol}  (dotted = predicted V)", xlabel="decile of predicted V = p(mu - c) within period, test days")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"realized Z_{H} per opportunity (ticks)")
    fig.suptitle("F08  Realized observed-policy payoff vs predicted value (held-out test days, c = 0)")
    fig.tight_layout(); fig.savefig(FIGURES / "F08_value_deciles.png", dpi=150); plt.close(fig)

    # F09 policy comparison at matched acceptance fractions
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pols = POLICIES[1:]
    colors = dict(zip(pols, ["#8c8c8c", "#6a9fb5", "#2b5d8a", "#b93c32"]))
    for ax, g in zip(axes, ("all", "wmr", "normal")):
        s = t[t.group == g] if g == "all" else tw[tw.group == g]    # WMR/normal: within-period budget
        base = s[s.policy == "P0_all"].iloc[0]
        ax.axhline(base.Z_per_accepted, color="k", lw=1, ls="--", label="P0 accept all")
        for k, pol in enumerate(pols):
            r = s[s.policy == pol].sort_values("target_frac")
            xs = np.arange(len(ACCEPT)) + (k - 1.5) * .18
            ax.errorbar(xs, r.Z_per_accepted, yerr=[r.Z_per_accepted - r.Z_lo, r.Z_hi - r.Z_per_accepted],
                        fmt="o", color=colors[pol], capsize=3, label=pol)
            for xx, (_, rr) in zip(xs, r.iterrows()):
                ax.annotate(f"{rr.n_accepted:,}", (xx, rr.Z_lo), fontsize=6, ha="center", va="top")
        ax.set_xticks(range(len(ACCEPT)), [f"top {int(f*100)}%" for f in ACCEPT])
        ax.set_title(f"{g} opportunities" + (" (one global threshold)" if g == "all" else " (threshold set within period)"),
                     fontsize=10)
        ax.axhline(0, color="grey", lw=.6)
    axes[0].set_ylabel(f"realized Z_{H} per accepted opportunity (ticks)")
    axes[0].legend(fontsize=7)
    fig.suptitle("F09  Historical opportunity selection at validation-chosen thresholds (test days; numbers = accepted)")
    fig.tight_layout(); fig.savefig(FIGURES / "F09_policies.png", dpi=150); plt.close(fig)
