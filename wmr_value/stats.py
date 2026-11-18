"""Day-level inference: the trading day is the unit; orders are never resampled individually."""
import numpy as np
import pandas as pd

from .config import SEED


def day_diff(d, n_boot: int = 4000, n_flip: int = 20000) -> dict:
    """Mean of per-day values (e.g. paired differences), day-bootstrap 95% CI, sign-flip p."""
    d = np.asarray(pd.Series(d).dropna(), float)
    if len(d) < 3:
        return {"mean": d.mean() if len(d) else np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p": np.nan, "share_pos": np.nan, "n_days": len(d)}
    rng = np.random.default_rng(SEED)
    boot = rng.choice(d, (n_boot, len(d))).mean(1)
    null = np.abs((rng.choice([-1.0, 1.0], (n_flip, len(d))) * d).mean(1))
    return {"mean": d.mean(), "ci_lo": np.quantile(boot, .025), "ci_hi": np.quantile(boot, .975),
            "p": (1 + (null >= abs(d.mean())).sum()) / (1 + n_flip), "share_pos": (d > 0).mean(),
            "n_days": len(d)}


def ratio_ci(num: pd.Series, den: pd.Series, n_boot: int = 4000) -> tuple[float, float, float]:
    """Pooled ratio sum(num)/sum(den) with a day-bootstrap CI (inputs are per-day sums)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    b = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return num.sum() / den.sum(), np.quantile(b, .025), np.quantile(b, .975)
