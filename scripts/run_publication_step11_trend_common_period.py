from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_data import DEFAULT_PUBLICATION_PANEL_PATH, load_publication_panel
from strategies.publication_step11_trend_ablation import (
    export_step11_common_period_result,
    frozen_capstone_trend_folds,
    run_publication_walk_forward_on_explicit_folds,
)

DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
DEFAULT_BENCHMARK_PATH = DEFAULT_RESULTS_DIR / "publication_external_benchmark.csv"
DEFAULT_RISK_FREE_PATH = DEFAULT_RESULTS_DIR / "publication_risk_free.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 11.1.1 on the exact seven frozen-capstone folds.")
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PUBLICATION_PANEL_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--risk-free-path", type=Path, default=DEFAULT_RISK_FREE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def _load_return_frame(path: Path, required_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    required = {"trade_date", required_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    if frame["trade_date"].duplicated().any():
        raise ValueError(f"{path} contains duplicate trade_date values.")
    return frame.loc[:, ["trade_date", required_column]].sort_values("trade_date")


def main() -> int:
    args = parse_args()
    prices = load_publication_panel(args.panel_path)
    benchmark = _load_return_frame(args.benchmark_path, "benchmark_return")
    risk_free = _load_return_frame(args.risk_free_path, "risk_free_return")
    result = run_publication_walk_forward_on_explicit_folds(
        strategy_name="trend_following",
        prices=prices,
        benchmark_returns=benchmark,
        risk_free_returns=risk_free,
        folds=frozen_capstone_trend_folds(),
    )
    paths = export_step11_common_period_result(
        result, args.output_dir, panel_path=args.panel_path,
        benchmark_path=args.benchmark_path, risk_free_path=args.risk_free_path,
    )
    nav = pd.to_numeric(result.fold_summary["net_excess_nav_difference"], errors="coerce").dropna()
    print("step=11.1.1")
    print("analysis_role=diagnostic_ablation")
    print(f"folds={len(result.fold_table)}")
    print(f"mean_net_excess_nav_difference={nav.mean():.12f}")
    print(f"positive_folds={(nav > 0).sum()}/{len(nav)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
