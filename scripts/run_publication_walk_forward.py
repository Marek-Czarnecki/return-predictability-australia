from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_data import DEFAULT_PUBLICATION_PANEL_PATH, load_publication_panel
from strategies.publication_pairs import run_publication_pairs_walk_forward
from strategies.publication_walk_forward import run_publication_walk_forward

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
DEFAULT_BENCHMARK_PATH = DEFAULT_OUTPUT_DIR / "publication_external_benchmark.csv"
DEFAULT_RISK_FREE_PATH = DEFAULT_OUTPUT_DIR / "publication_risk_free.csv"
SUPPORTED_STRATEGIES = ("trend_following", "mean_reversion", "pairs_trading")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the publication-specific point-in-time walk-forward experiment "
            "for a controlled strategy rerun."
        )
    )
    parser.add_argument("strategy", choices=SUPPORTED_STRATEGIES)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PUBLICATION_PANEL_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--risk-free-path", type=Path, default=DEFAULT_RISK_FREE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-folds", type=int, default=None)
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


def export_publication_walk_forward_result(
    strategy_name: str,
    result,
    output_dir: Path,
    *,
    risk_free_path: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"publication_{strategy_name}_walk_forward"
    paths = {
        "folds": output_dir / f"{prefix}_folds.csv",
        "daily": output_dir / f"{prefix}_daily_returns.csv",
        "summary": output_dir / f"{prefix}_summary.csv",
        "liquidity": output_dir / f"{prefix}_liquidity_diagnostics.csv",
        "metadata": output_dir / f"{prefix}_metadata.json",
    }
    pair_diagnostics = getattr(result, "pair_diagnostics", None)
    if pair_diagnostics is not None:
        paths["pairs"] = output_dir / f"{prefix}_selected_pairs.csv"

    result.fold_table.to_csv(paths["folds"], index=False)
    result.fold_daily_results.to_csv(paths["daily"], index=False)
    result.fold_summary.to_csv(paths["summary"], index=False)
    result.liquidity_diagnostics.to_csv(paths["liquidity"], index=False)
    if pair_diagnostics is not None:
        pair_diagnostics.to_csv(paths["pairs"], index=False)

    daily = result.fold_daily_results
    metadata = {
        "strategy_name": strategy_name,
        "fold_count": int(len(result.fold_table)),
        "daily_row_count": int(len(daily)),
        "evaluation_start": daily["trade_date"].min().isoformat() if not daily.empty else None,
        "evaluation_end": daily["trade_date"].max().isoformat() if not daily.empty else None,
        "benchmark_observed_count": int(daily["benchmark_observed"].sum()) if not daily.empty else 0,
        "benchmark_missing_count": int((~daily["benchmark_observed"]).sum()) if not daily.empty else 0,
        "net_return_null_count": int(daily["net_return"].isna().sum()) if not daily.empty else 0,
        "benchmark_return_null_count": int(daily["benchmark_return"].isna().sum()) if not daily.empty else 0,
        "excess_return_null_count": int(daily["excess_return"].isna().sum()) if not daily.empty else 0,
        "identity_col": "asset_id",
        "eligibility_col": "eligible_to_trade",
        "benchmark_missing_policy": "exclude_from_objective_and_excess_metrics",
        "risk_free_path": str(risk_free_path) if risk_free_path is not None else None,
        "risk_free_use": "summary_risk_adjusted_metrics_only",
        "parameter_selection_affected_by_risk_free": False,
        "metric_semantics": {
            "total_net_excess_return": "legacy_arithmetic_sum_of_daily_excess_returns",
            "preferred_relative_metric": "net_excess_nav_difference",
        },
    }
    if strategy_name == "pairs_trading":
        metadata.update(
            {
                "formation_history_convention": "all_observable_prices_within_formation_window_for_formation_end_eligible_assets",
                "capital_normalization": "gross_exposure_normalized_by_1_plus_abs_hedge_ratio",
                "pair_cost_application": "normalized_two_leg_weighted_tier_cost",
                "borrow_financing": "excluded_disclosed_limitation",
                "selected_pair_rows": int(len(pair_diagnostics)),
            }
        )
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def main() -> int:
    args = parse_args()
    prices = load_publication_panel(args.panel_path)
    benchmark = _load_return_frame(args.benchmark_path, "benchmark_return")
    risk_free = _load_return_frame(args.risk_free_path, "risk_free_return")

    if args.strategy == "pairs_trading":
        result = run_publication_pairs_walk_forward(
            prices=prices,
            benchmark_returns=benchmark,
            risk_free_returns=risk_free,
            max_folds=args.max_folds,
        )
    else:
        result = run_publication_walk_forward(
            strategy_name=args.strategy,
            prices=prices,
            benchmark_returns=benchmark,
            risk_free_returns=risk_free,
            max_folds=args.max_folds,
        )
    paths = export_publication_walk_forward_result(
        strategy_name=args.strategy,
        result=result,
        output_dir=args.output_dir,
        risk_free_path=args.risk_free_path,
    )

    print(f"strategy={args.strategy}")
    print(f"folds={len(result.fold_table)}")
    print(f"daily_rows={len(result.fold_daily_results)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
