from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .publication_data import prepare_publication_strategy_panel
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


FROZEN_CAPSTONE_COMMIT = "349fc00b087d404ba11bb9f04e9fe8ba7ad58ed6"
STEP11_ABLATION_DIMENSION = "evaluation_sample_period"


def frozen_capstone_trend_folds() -> tuple[WalkForwardFold, ...]:
    """Return the exact seven trend fold dates used by the frozen capstone."""
    rows = (
        ("fold_01", "2016-07-25", "2019-07-24", "2019-07-25", "2020-07-24"),
        ("fold_02", "2017-07-25", "2020-07-24", "2020-07-27", "2021-07-23"),
        ("fold_03", "2018-07-25", "2021-07-23", "2021-07-26", "2022-07-22"),
        ("fold_04", "2019-07-25", "2022-07-22", "2022-07-25", "2023-07-24"),
        ("fold_05", "2020-07-27", "2023-07-24", "2023-07-25", "2024-07-24"),
        ("fold_06", "2021-07-26", "2024-07-24", "2024-07-25", "2025-07-24"),
        ("fold_07", "2022-07-25", "2025-07-24", "2025-07-25", "2026-07-20"),
    )
    return tuple(
        WalkForwardFold(
            fold_id=fold_id,
            formation_start=pd.Timestamp(formation_start),
            formation_end=pd.Timestamp(formation_end),
            evaluation_start=pd.Timestamp(evaluation_start),
            evaluation_end=pd.Timestamp(evaluation_end),
        )
        for fold_id, formation_start, formation_end, evaluation_start, evaluation_end in rows
    )


def run_publication_walk_forward_on_explicit_folds(
    strategy_name: str,
    prices: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    risk_free_returns: pd.Series | pd.DataFrame | None = None,
    *,
    benchmark_col: str = "benchmark_return",
    risk_free_col: str = "risk_free_return",
) -> PublicationWalkForwardResult:
    """Run the frozen publication method on an explicitly supplied fold schedule.

    This diagnostic path intentionally leaves the frozen Step 8 runner unchanged. It
    reuses the publication preparation, parameter selection, point-in-time eligibility,
    liquidity-cost context, evaluation, and summary helpers, changing only the fold
    schedule supplied to the experiment.
    """
    if strategy_name not in PUBLICATION_STRATEGY_DEFINITIONS:
        raise KeyError(f"Unsupported publication strategy: {strategy_name}")
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


def export_step11_common_period_result(
    result: PublicationWalkForwardResult,
    output_dir: Path,
    *,
    panel_path: Path | None = None,
    benchmark_path: Path | None = None,
    risk_free_path: Path | None = None,
) -> dict[str, Path]:
    """Export dedicated Step 11.1.1 artifacts without touching Step 8 evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "publication_step11_trend_common_period"
    paths = {
        "folds": output_dir / f"{prefix}_folds.csv",
        "daily": output_dir / f"{prefix}_daily_returns.csv",
        "summary": output_dir / f"{prefix}_summary.csv",
        "liquidity": output_dir / f"{prefix}_liquidity_diagnostics.csv",
        "metadata": output_dir / f"{prefix}_metadata.json",
    }

    result.fold_table.to_csv(paths["folds"], index=False)
    result.fold_daily_results.to_csv(paths["daily"], index=False)
    result.fold_summary.to_csv(paths["summary"], index=False)
    result.liquidity_diagnostics.to_csv(paths["liquidity"], index=False)

    nav_difference = pd.to_numeric(
        result.fold_summary["net_excess_nav_difference"], errors="coerce"
    ).dropna()
    metadata = {
        "step": "11.1.1",
        "analysis_role": "diagnostic_ablation",
        "confirmatory": False,
        "strategy_name": "trend_following",
        "ablation_dimension": STEP11_ABLATION_DIMENSION,
        "causal_scope": "sample_extension_and_regime_coverage",
        "fold_schedule_source": "frozen_capstone",
        "frozen_capstone_commit": FROZEN_CAPSTONE_COMMIT,
        "parameter_grid_changed": False,
        "strategy_rule_changed": False,
        "publication_execution_semantics_changed": False,
        "not_part_of_primary_holm_family": True,
        "full_pre_window_history_retained": True,
        "fold_count": int(len(result.fold_table)),
        "evaluation_start": result.fold_table["evaluation_start"].min().isoformat(),
        "evaluation_end": result.fold_table["evaluation_end"].max().isoformat(),
        "mean_net_excess_nav_difference": float(nav_difference.mean()),
        "median_net_excess_nav_difference": float(nav_difference.median()),
        "positive_fold_count": int((nav_difference > 0.0).sum()),
        "positive_fold_fraction": float((nav_difference > 0.0).mean()),
        "panel_path": str(panel_path) if panel_path is not None else None,
        "benchmark_path": str(benchmark_path) if benchmark_path is not None else None,
        "risk_free_path": str(risk_free_path) if risk_free_path is not None else None,
        "interpretation_rule": (
            "Assess economic magnitude relative to the frozen and full-publication trend effects; "
            "do not treat this diagnostic as a new confirmatory hypothesis."
        ),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths
