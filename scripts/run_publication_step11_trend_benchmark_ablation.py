from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_data import DEFAULT_PUBLICATION_PANEL_PATH, load_publication_panel
from strategies.publication_step11_trend_benchmark_ablation import (
    export_step11_benchmark_ablation,
    run_step11_benchmark_ablation,
)

DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
DEFAULT_XJOA_PATH = DEFAULT_RESULTS_DIR / "publication_external_benchmark.csv"
DEFAULT_RISK_FREE_PATH = DEFAULT_RESULTS_DIR / "publication_risk_free.csv"


def _load_return_frame(path: Path, required_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    required = {"trade_date", required_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    if frame["trade_date"].duplicated().any():
        raise ValueError(f"{path} contains duplicate trade_date values.")
    return frame.loc[:, ["trade_date", required_column]].sort_values("trade_date")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 11.1.3 XJOA versus STW benchmark ablation.")
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PUBLICATION_PANEL_PATH)
    parser.add_argument("--xjoa-path", type=Path, default=DEFAULT_XJOA_PATH)
    parser.add_argument(
        "--stw-path", type=Path, required=True,
        help="CSV containing the frozen STW benchmark proxy returns used by the diagnostic."
    )
    parser.add_argument("--risk-free-path", type=Path, default=DEFAULT_RISK_FREE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prices = load_publication_panel(args.panel_path)
    xjoa = _load_return_frame(args.xjoa_path, "benchmark_return")
    stw = _load_return_frame(args.stw_path, "benchmark_return")
    risk_free = _load_return_frame(args.risk_free_path, "risk_free_return")
    xjoa_result, stw_result = run_step11_benchmark_ablation(
        prices=prices, xjoa_benchmark_returns=xjoa,
        stw_benchmark_returns=stw, risk_free_returns=risk_free,
    )
    paths = export_step11_benchmark_ablation(
        xjoa_result, stw_result, args.output_dir,
        xjoa_path=args.xjoa_path, stw_path=args.stw_path,
    )
    xjoa_nav = pd.to_numeric(xjoa_result.fold_summary["net_excess_nav_difference"], errors="coerce").dropna()
    stw_nav = pd.to_numeric(stw_result.fold_summary["net_excess_nav_difference"], errors="coerce").dropna()
    comparison = pd.read_csv(paths["comparison"])
    print("step=11.1.3")
    print("analysis_role=diagnostic_ablation")
    print(f"folds={len(comparison)}")
    print(f"xjoa_mean_net_excess_nav_difference={xjoa_nav.mean():.12f}")
    print(f"stw_mean_net_excess_nav_difference={stw_nav.mean():.12f}")
    print(f"mean_benchmark_effect_nav_difference={(stw_nav.mean() - xjoa_nav.mean()):.12f}")
    print(f"parameter_selection_changes={int(comparison['parameter_selection_changed'].sum())}/{len(comparison)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
