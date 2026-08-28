from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .publication_step11_trend_ablation import (
    FROZEN_CAPSTONE_COMMIT,
    frozen_capstone_trend_folds,
    run_publication_walk_forward_on_explicit_folds,
)


STEP11_BENCHMARK_ABLATION_DIMENSION = "xjoa_vs_stw_benchmark"
CONTROL_BENCHMARK = "XJOA_total_return_index"
ABLATION_BENCHMARK = "STW_total_return_etf"


def run_step11_benchmark_ablation(
    prices: pd.DataFrame,
    xjoa_benchmark_returns: pd.DataFrame,
    stw_benchmark_returns: pd.DataFrame,
    risk_free_returns: pd.DataFrame | None = None,
):
    """Run Step 11.1.3 on the exact frozen seven-fold calendar.

    Both arms use identical Norgate prices, point-in-time membership, publication
    transaction costs, strategy rules, parameter grid and execution semantics. The
    supplied benchmark is changed in both formation scoring and evaluation summary.
    """
    folds = frozen_capstone_trend_folds()
    xjoa_result = run_publication_walk_forward_on_explicit_folds(
        strategy_name="trend_following",
        prices=prices,
        benchmark_returns=xjoa_benchmark_returns,
        risk_free_returns=risk_free_returns,
        folds=folds,
    )
    stw_result = run_publication_walk_forward_on_explicit_folds(
        strategy_name="trend_following",
        prices=prices,
        benchmark_returns=stw_benchmark_returns,
        risk_free_returns=risk_free_returns,
        folds=folds,
    )
    return xjoa_result, stw_result


def build_benchmark_fold_comparison(xjoa_result, stw_result) -> pd.DataFrame:
    xjoa = xjoa_result.fold_summary.loc[:, [
        "fold_id",
        "net_excess_nav_difference",
        "total_net_excess_return",
        "chosen_parameters",
        "benchmark_observation_count",
    ]].copy()
    xjoa = xjoa.rename(columns={
        "net_excess_nav_difference": "xjoa_net_excess_nav_difference",
        "total_net_excess_return": "xjoa_legacy_excess",
        "chosen_parameters": "xjoa_chosen_parameters",
        "benchmark_observation_count": "xjoa_benchmark_observation_count",
    })

    stw = stw_result.fold_summary.loc[:, [
        "fold_id",
        "net_excess_nav_difference",
        "total_net_excess_return",
        "chosen_parameters",
        "benchmark_observation_count",
    ]].copy()
    stw = stw.rename(columns={
        "net_excess_nav_difference": "stw_net_excess_nav_difference",
        "total_net_excess_return": "stw_legacy_excess",
        "chosen_parameters": "stw_chosen_parameters",
        "benchmark_observation_count": "stw_benchmark_observation_count",
    })

    comparison = xjoa.merge(stw, on="fold_id", how="inner", validate="one_to_one")
    comparison["benchmark_effect_nav_difference"] = (
        comparison["stw_net_excess_nav_difference"]
        - comparison["xjoa_net_excess_nav_difference"]
    )
    comparison["parameter_selection_changed"] = (
        comparison["xjoa_chosen_parameters"] != comparison["stw_chosen_parameters"]
    )
    return comparison


def export_step11_benchmark_ablation(
    xjoa_result,
    stw_result,
    output_dir: Path,
    *,
    xjoa_path: Path | None = None,
    stw_path: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "publication_step11_trend_benchmark_ablation"
    paths = {
        "comparison": output_dir / f"{prefix}_comparison.csv",
        "xjoa_summary": output_dir / f"{prefix}_xjoa_summary.csv",
        "stw_summary": output_dir / f"{prefix}_stw_summary.csv",
        "metadata": output_dir / f"{prefix}_metadata.json",
    }

    comparison = build_benchmark_fold_comparison(xjoa_result, stw_result)
    comparison.to_csv(paths["comparison"], index=False)
    xjoa_result.fold_summary.to_csv(paths["xjoa_summary"], index=False)
    stw_result.fold_summary.to_csv(paths["stw_summary"], index=False)

    xjoa_nav = pd.to_numeric(comparison["xjoa_net_excess_nav_difference"], errors="coerce")
    stw_nav = pd.to_numeric(comparison["stw_net_excess_nav_difference"], errors="coerce")
    effect = pd.to_numeric(comparison["benchmark_effect_nav_difference"], errors="coerce")

    metadata = {
        "step": "11.1.3",
        "analysis_role": "diagnostic_ablation",
        "confirmatory": False,
        "strategy_name": "trend_following",
        "ablation_dimension": STEP11_BENCHMARK_ABLATION_DIMENSION,
        "control_benchmark": CONTROL_BENCHMARK,
        "ablation_benchmark": ABLATION_BENCHMARK,
        "frozen_capstone_commit": FROZEN_CAPSTONE_COMMIT,
        "fold_schedule_source": "frozen_capstone",
        "fold_count": int(len(comparison)),
        "prices_changed_between_arms": False,
        "point_in_time_universe_changed_between_arms": False,
        "strategy_rule_changed": False,
        "parameter_grid_changed": False,
        "cost_framework_changed": False,
        "execution_semantics_changed": False,
        "benchmark_changed_in_formation_selection": True,
        "benchmark_changed_in_evaluation": True,
        "not_part_of_primary_holm_family": True,
        "xjoa_mean_net_excess_nav_difference": float(xjoa_nav.mean()),
        "xjoa_median_net_excess_nav_difference": float(xjoa_nav.median()),
        "stw_mean_net_excess_nav_difference": float(stw_nav.mean()),
        "stw_median_net_excess_nav_difference": float(stw_nav.median()),
        "mean_benchmark_effect_nav_difference": float(effect.mean()),
        "xjoa_positive_fold_count": int((xjoa_nav > 0).sum()),
        "stw_positive_fold_count": int((stw_nav > 0).sum()),
        "parameter_selection_change_count": int(comparison["parameter_selection_changed"].sum()),
        "xjoa_path": str(xjoa_path) if xjoa_path is not None else None,
        "stw_path": str(stw_path) if stw_path is not None else None,
        "interpretation_rule": (
            "Attribute only the controlled effect of replacing the publication XJOA total-return "
            "benchmark with the frozen STW ETF benchmark on the same point-in-time Norgate trend "
            "experiment. Assess economic magnitude; do not treat this diagnostic as a new "
            "confirmatory hypothesis."
        ),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths
