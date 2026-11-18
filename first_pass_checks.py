"""Robustness checks for the first_pass.py headline results. Reads its cache only.

Run after first_pass.py: .venv/Scripts/python.exe first_pass_checks.py
  B  markouts winsorized at the pooled 1st/99th percentiles
  E  fade matched on activity AND hit-side depth, and restricted to trades with no
     further trade within the horizon (so depth cannot be consumed before it is cancelled)
"""
import numpy as np
import pandas as pd

from first_pass import CACHE, FADE_MS, OUT, day_table, paired


def load(kind: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted(CACHE.glob(f"*_{kind}.parquet"))], ignore_index=True)


def buckets(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Joint quintile cells of `cols`, with cut points from WMR rows."""
    code = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        edges = np.quantile(df.loc[df.period == "wmr", c], [.2, .4, .6, .8])
        code = code * 5 + np.digitize(df[c].to_numpy(), edges)
    return df.assign(bucket=code)


def main() -> None:
    rows = []
    fills = buckets(load("fills"), ["act60"])
    for h in (1, 5, 30, 60):
        lo, hi = fills[f"mk_{h}"].quantile([.01, .99])
        mk = fills[f"mk_{h}"].clip(lo, hi)
        rows.append(paired(day_table(fills, fills["size"] * mk, fills["size"]), f"B_markout_{h}s_winsorized"))
        rows.append(paired(day_table(fills, mk), f"B_markout_{h}s_winsorized_per_fill"))

    fades, base = load("fades"), load("base")
    for ms in FADE_MS:
        col = f"fade_{ms}"
        for label, f, b in (("act+depth", fades, base),
                            ("act+depth, no trade within horizon",
                             fades[fades.next_trade_ms > ms], base[base.next_trade_ms > ms])):
            f2, b2 = buckets(f, ["act60", "q"]), buckets(b, ["act60", "q"])
            ft, bt = day_table(f2, f2[col]), day_table(b2, b2[col])
            rows.append(paired(ft, f"E_fade_after_trade_{ms}ms [{label}]"))
            rows.append(paired(ft - bt, f"E_fade_excess_{ms}ms [{label}]"))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT / "tables" / "robustness.csv", index=False)
    pd.set_option("display.width", 220)
    print(res[res.control.isin(["normal_matched", "nearby", "us_event"])].round(4).to_string())
    share = (fades.next_trade_ms > 100).groupby(fades.period).mean()
    print("share of trades with no further trade within 100 ms:", share.round(3).to_dict())
    print("mean hit-side depth after trade:", fades.groupby("period").q.mean().round(1).to_dict())


if __name__ == "__main__":
    main()
