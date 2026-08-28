from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .publication_data import prepare_publication_strategy_panel
from .publication_step11_trend_ablation import (
    FROZEN_CAPSTONE_COMMIT,
    frozen_capstone_trend_folds,
    run_publication_walk_forward_on_explicit_folds,
)
from .publication_walk_forward import (
    PUBLICATION_STRATEGY_DEFINITIONS,
    PublicationWalkForwardResult,
    _build_daily_results,
    _build_summary,
    _coerce_return_series,
    _expand_parameter_grid,
    _filter_daily_window,
    _format_parameters,
    _select_best_parameters,
    _slice_prices_with_history,
)
from .publication_walk_forward_costs import build_publication_fold_cost_context
from .walk_forward import WalkForwardFold


STEP11_COST_ABLATION_DIMENSION = "publication_base_costs_vs_zero_transaction_costs"
CONTROL_COST_TREATMENT = "publication_liquidity_tiered_base"
ABLATION_COST_TREATMENT = "zero_transaction_costs"
ZERO_TURNOVER_COST_BPS = 0.0


def run_publication_walk_forward_on_explicit_folds_with_flat_cost(
    strategy_name: str,
    prices: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    *,
    turnover_cost_bps: float,
    risk_free_returns: pd.Series | pd.DataFrame | None = None,
    benchmark_col: str = "benchmark_return",
    risk_free_col: str = "risk_free_return",
) -> PublicationWalkForwardResult:
    """Run explicit publication folds with one flat turnover-cost override.

    The override is applied during both formation parameter selection and evaluation.
    All other publication mechanics, including point-in-time eligibility, formation-only
    liquidity context, parameter grid, execution timing and benchmark treatment, are
    retained. This helper is diagnostic-only and does not change the Step 8 runner.
    """
    if strategy_name not in PUBLICATION_STRATEGY_DEFINITIONS:
        raise KeyError(f"Unsupported publication strategy: {strategy_name}")
    if turnover_cost_bps < 0:
        raise ValueError("turnover_cost_bps must be non-negative.")
    explicit_folds = tuple(folds)
    if not explicit_folds:
        raise ValueError("At least one explicit walk-forward fold is required.")

    definition = PUBLICATION_STRATEGY_DEFINITIONS[strategy_name]
    prepared_prices = prepare_publication_strategy_panel(
        prices,
        min_history=definition.min_history,
    )
    benchmark_series = _coerce_return_series(benchmark_returns, benchmark_col)
    risk_free_series = (
        _coerce_return_series(risk_free_returns, risk_free_col)
        if risk_free_returns is not None
        else None
    )
    candidates = _expand_parameter_grid(definition.parameter_grid)

    fold_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    liquidity_frames: list[pd.DataFrame] = []

    for fold in explicit_folds:
        formation_prices = _slice_prices_with_history(
            prepared_prices,
            through=fold.formation_end,
        )
        evaluation_prices = _slice_prices_with_history(
            prepared_prices,
            through=fold.evaluation_end,
        )
        cost_context = build_publication_fold_cost_context(
            prices=prepared_prices,
            fold=fold,
            evaluation_prices=evaluation_prices,
            identity_col="asset_id",
        )
        liquidity_frames.append(cost_context.liquidity_diagnostics)

        fixed_kwargs = {
            "identity_col": "asset_id",
            "eligibility_col": "eligible_to_trade",
            "liquidity_tier_map": cost_context.liquidity_tier_map,
            "turnover_cost_bps": float(turnover_cost_bps),
        }
        best_candidate, best_metrics = _select_best_parameters(
            runner=definition.runner,
            formation_prices=formation_prices,
            benchmark_series=benchmark_series,
            candidates=candidates,
            formation_start=fold.formation_start,
            formation_end=fold.formation_end,
            fixed_kwargs=fixed_kwargs,
        )

        evaluation_kwargs = dict(best_candidate)
        evaluation_kwargs.update(fixed_kwargs)
        evaluation_run = definition.runner(evaluation_prices, **evaluation_kwargs)
        evaluation_daily = _build_daily_results(
            fold=fold,
            daily_results=evaluation_run.daily_results,
            benchmark_series=benchmark_series,
        )
        evaluation_daily = _filter_daily_window(
            evaluation_daily,
            fold.evaluation_start,
            fold.evaluation_end,
        )
        daily_frames.append(evaluation_daily)
        summary_frames.append(
            _build_summary(
                strategy_name=strategy_name,
                fold=fold,
                daily_results=evaluation_daily,
                parameters=best_candidate,
                risk_free_series=risk_free_series,
            )
        )
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "formation_start": fold.formation_start,
                "formation_end": fold.formation_end,
                "evaluation_start": fold.evaluation_start,
                "evaluation_end": fold.evaluation_end,
                "strategy_name": strategy_name,
                "identity_col": "asset_id",
                "eligibility_col": "eligible_to_trade",
                "selection_objective": "net_excess_nav_vs_benchmark",
                "benchmark_missing_policy": "exclude_from_objective_and_excess_metrics",
                "turnover_cost_bps_override": float(turnover_cost_bps),
                "chosen_parameters": _format_parameters(best_candidate),
                "formation_objective_value": best_metrics["objective_value"],
                "formation_total_net_excess_return": best_metrics[
                    "total_net_excess_return"
                ],
                "formation_average_turnover": best_metrics["average_turnover"],
                "formation_benchmark_observation_count": best_metrics[
                    "benchmark_observation_count"
                ],
            }
        )

    return PublicationWalkForwardResult(
        fold_table=pd.DataFrame(fold_rows),
        fold_daily_results=pd.concat(daily_frames, ignore_index=True),
        fold_summary=pd.concat(summary_frames, ignore_index=True),
        liquidity_diagnostics=pd.concat(liquidity_frames, ignore_index=True),
    )


def run_step11_cost_ablation(
    prices: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    risk_free_returns: pd.DataFrame | None = None,
):
    """Run Step 11.1.4 on the exact frozen seven-fold calendar."""
    folds = frozen_capstone_trend_folds()
    base_result = run_publication_walk_forward_on_explicit_folds(
        strategy_name="trend_following",
        prices=prices,
        benchmark_returns=benchmark_returns,
        risk_free_returns=risk_free_returns,
        folds=folds,
    )
    zero_result = run_publication_walk_forward_on_explicit_folds_with_flat_cost(
        strategy_name="trend_following",
        prices=prices,
        benchmark_returns=benchmark_returns,
        risk_free_returns=risk_free_returns,
        folds=folds,
        turnover_cost_bps=ZERO_TURNOVER_COST_BPS,
    )
    return base_result, zero_result


def build_cost_fold_comparison(base_result, zero_result) -> pd.DataFrame:
    base = base_result.fold_summary.loc[:, [
        "fold_id",
        "net_excess_nav_difference",
        "total_net_excess_return",
        "average_turnover",
    ]].copy()
    base = base.rename(columns={
        "net_excess_nav_difference": "base_cost_net_excess_nav_difference",
        "total_net_excess_return": "base_cost_legacy_excess",
        "average_turnover": "base_cost_average_turnover",
    })
    base_parameters = base_result.fold_table.loc[:, ["fold_id", "chosen_parameters"]].rename(
        columns={"chosen_parameters": "base_cost_chosen_parameters"}
    )
    base = base.merge(base_parameters, on="fold_id", how="left", validate="one_to_one")

    zero = zero_result.fold_summary.loc[:, [
        "fold_id",
        "net_excess_nav_difference",
        "total_net_excess_return",
        "average_turnover",
    ]].copy()
    zero = zero.rename(columns={
        "net_excess_nav_difference": "zero_cost_net_excess_nav_difference",
        "total_net_excess_return": "zero_cost_legacy_excess",
        "average_turnover": "zero_cost_average_turnover",
    })
    zero_parameters = zero_result.fold_table.loc[:, ["fold_id", "chosen_parameters"]].rename(
        columns={"chosen_parameters": "zero_cost_chosen_parameters"}
    )
    zero = zero.merge(zero_parameters, on="fold_id", how="left", validate="one_to_one")

    comparison = base.merge(zero, on="fold_id", how="inner", validate="one_to_one")
    comparison["cost_effect_nav_difference"] = (
        comparison["zero_cost_net_excess_nav_difference"]
        - comparison["base_cost_net_excess_nav_difference"]
    )
    comparison["parameter_selection_changed"] = (
        comparison["base_cost_chosen_parameters"]
        != comparison["zero_cost_chosen_parameters"]
    )
    return comparison


def export_step11_cost_ablation(
    base_result,
    zero_result,
    output_dir: Path,
    *,
    benchmark_path: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "publication_step11_trend_cost_ablation"
    paths = {
        "comparison": output_dir / f"{prefix}_comparison.csv",
        "base_summary": output_dir / f"{prefix}_base_summary.csv",
        "zero_summary": output_dir / f"{prefix}_zero_summary.csv",
        "metadata": output_dir / f"{prefix}_metadata.json",
    }

    comparison = build_cost_fold_comparison(base_result, zero_result)
    comparison.to_csv(paths["comparison"], index=False)
    base_result.fold_summary.to_csv(paths["base_summary"], index=False)
    zero_result.fold_summary.to_csv(paths["zero_summary"], index=False)

    base_nav = pd.to_numeric(
        comparison["base_cost_net_excess_nav_difference"], errors="coerce"
    )
    zero_nav = pd.to_numeric(
        comparison["zero_cost_net_excess_nav_difference"], errors="coerce"
    )
    effect = pd.to_numeric(comparison["cost_effect_nav_difference"], errors="coerce")

    metadata = {
        "step": "11.1.4",
        "analysis_role": "diagnostic_ablation",
        "confirmatory": False,
        "strategy_name": "trend_following",
        "ablation_dimension": STEP11_COST_ABLATION_DIMENSION,
        "control_cost_treatment": CONTROL_COST_TREATMENT,
        "ablation_cost_treatment": ABLATION_COST_TREATMENT,
        "zero_turnover_cost_bps": ZERO_TURNOVER_COST_BPS,
        "frozen_capstone_commit": FROZEN_CAPSTONE_COMMIT,
        "fold_schedule_source": "frozen_capstone",
        "fold_count": int(len(comparison)),
        "prices_changed_between_arms": False,
        "point_in_time_universe_changed_between_arms": False,
        "benchmark_changed_between_arms": False,
        "strategy_rule_changed": False,
        "parameter_grid_changed": False,
        "execution_semantics_changed": False,
        "transaction_cost_treatment_changed_in_formation_selection": True,
        "transaction_cost_treatment_changed_in_evaluation": True,
        "liquidity_tier_construction_retained": True,
        "not_part_of_primary_holm_family": True,
        "base_mean_net_excess_nav_difference": float(base_nav.mean()),
        "base_median_net_excess_nav_difference": float(base_nav.median()),
        "zero_mean_net_excess_nav_difference": float(zero_nav.mean()),
        "zero_median_net_excess_nav_difference": float(zero_nav.median()),
        "mean_cost_effect_nav_difference": float(effect.mean()),
        "base_positive_fold_count": int((base_nav > 0).sum()),
        "zero_positive_fold_count": int((zero_nav > 0).sum()),
        "parameter_selection_change_count": int(
            comparison["parameter_selection_changed"].sum()
        ),
        "benchmark_path": str(benchmark_path) if benchmark_path is not None else None,
        "interpretation_rule": (
            "Attribute only the controlled effect of removing publication transaction costs on "
            "the same point-in-time Norgate/XJOA trend experiment. Zero costs are applied in "
            "both formation parameter selection and evaluation. Assess economic magnitude; do "
            "not treat this diagnostic as a new confirmatory hypothesis."
        ),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths
