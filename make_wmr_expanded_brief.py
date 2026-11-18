"""Compatibility entry point for the full three-month WMR EDA PDF brief."""
from pathlib import Path
import argparse

from make_wmr_full_brief import report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/wmr_eda/wmr_eda_proposal_brief.pdf"))
    args = parser.parse_args()
    report(args.project, args.out)
    print(args.out.resolve())
