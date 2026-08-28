from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_data import DEFAULT_PUBLICATION_PANEL_PATH, load_publication_panel
from strategies.publication_step11_trend_universe_ablation import (
    REFERENCE_DATE,
    export_step11_universe_ablation,
    run_step11_universe_ablation,
)

DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
DEFAULT_BENCHMARK_PATH = DEFAULT_RESULTS_DIR / "publication_external_benchmark.csv"
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
    parser = argparse.ArgumentParser(description="Run Step 11.1.2 PIT versus retrospective-current membership ablation.")
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PUBLICATION_PANEL_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--risk-free-path", type=Path, default=DEFAULT_RISK_FREE_PATH)
    parser.add_argument("--reference-date", type=pd.Timestamp, default=REFERENCE_DATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prices = load_publication_panel(args.panel_path)
    benchmark = _load_return_frame(args.benchmark_path, "benchmark_return")
    risk_free = _load_return_frame(args.risk_free_path, "risk_free_return")
    pit_result, retrospective_result, selection = run_step11_universe_ablation(
        prices=prices, benchmark_returns=benchmark, risk_free_returns=risk_free,
        reference_date=args.reference_date,
    )
    paths = export_step11_universe_ablation(
        pit_result, retrospective_result, selection, args.output_dir
    )
    pit = pd.to_numeric(pit_result.fold_summary["net_excess_nav_difference"], errors="coerce").dropna()
    retro = pd.to_numeric(retrospective_result.fold_summary["net_excess_nav_difference"], errors="coerce").dropna()
    print("step=11.1.2")
    print("analysis_role=diagnostic_ablation")
    print(f"reference_date={selection.reference_date.date()}")
    print(f"reference_universe_asset_ids={len(selection.selected_asset_ids)}")
    print(f"pit_mean_net_excess_nav_difference={pit.mean():.12f}")
    print(f"retrospective_mean_net_excess_nav_difference={retro.mean():.12f}")
    print(f"mean_universe_effect_nav_difference={(retro.mean() - pit.mean()):.12f}")
    print("frozen_yahoo_vs_norgate_security_coverage=unresolved_contributor")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
