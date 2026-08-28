from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import pandas as pd

from .mean_reversion import run_mean_reversion_strategy
from .metrics import compute_nav, summarize_return_stream
from .publication_data import prepare_publication_strategy_panel
from .publication_walk_forward_costs import build_publication_fold_cost_context
from .trend_following import run_trend_following_strategy
from .walk_forward import WalkForwardFold, generate_walk_forward_folds


@dataclass(frozen=True)
class PublicationStrategyDefinition:
    runner: Callable[..., object]
    min_history: int
    parameter_grid: dict[str, list[object]]


@dataclass
class PublicationWalkForwardResult:
    fold_table: pd.DataFrame
    fold_daily_results: pd.DataFrame
    fold_summary: pd.DataFrame
    liquidity_diagnostics: pd.DataFrame


PUBLICATION_STRATEGY_DEFINITIONS: dict[str, PublicationStrategyDefinition] = {
    "trend_following": PublicationStrategyDefinition(
        runner=run_trend_following_strategy,
        min_history=220,
        parameter_grid={
            "fast_window": [50, 100],
            "slow_window": [150, 200],
            "min_history": [220],
            "cost_scenario": ["base"],
        },
    ),
    "mean_reversion": PublicationStrategyDefinition(
        runner=run_mean_reversion_strategy,
        min_history=60,
        parameter_grid={
            "lookback_window": [10, 20],
            "entry_z": [-1.5, -2.0],
            "exit_z": [-0.25, -0.5],
            "min_history": [60],
            "cost_scenario": ["base"],
        },
    ),
}


def run_publication_walk_forward(
    strategy_name: str,
    prices: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    risk_free_returns: pd.Series | pd.DataFrame | None = None,
    *,
    benchmark_col: str = "benchmark_return",
    risk_free_col: str = "risk_free_return",
    formation_years: int = 3,
    evaluation_years: int = 1,
    step_years: int = 1,
    max_folds: int | None = None,
) -> PublicationWalkForwardResult:
    """Run the publication long-only walk-forward experiment.

    This path is deliberately separate from the frozen capstone walk-forward runner.
    It uses permanent ``asset_id`` identity, point-in-time eligibility, a different
    ex-ante liquidity/cost map for every fold, and benchmark alignment that never
    replaces missing benchmark observations with zero returns.
    """
    if strategy_name not in PUBLICATION_STRATEGY_DEFINITIONS:
        raise KeyError(f"Unsupported publication strategy: {strategy_name}")
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be at least 1 when supplied.")

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

    folds = generate_walk_forward_folds(
        prepared_prices,
        formation_years=formation_years,
        evaluation_years=evaluation_years,
        step_years=step_years,
    )
    if max_folds is not None:
        folds = folds[:max_folds]

    if not folds:
        return PublicationWalkForwardResult(
            fold_table=pd.DataFrame(),
            fold_daily_results=pd.DataFrame(),
            fold_summary=pd.DataFrame(),
            liquidity_diagnostics=pd.DataFrame(),
        )

    candidates = _expand_parameter_grid(definition.parameter_grid)
    fold_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    liquidity_frames: list[pd.DataFrame] = []

    for fold in folds:
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


def _expand_parameter_grid(parameter_grid: dict[str, list[object]]) -> list[dict[str, object]]:
    parameter_names = list(parameter_grid)
    combinations = product(*(parameter_grid[name] for name in parameter_names))
    return [dict(zip(parameter_names, values)) for values in combinations]


def _coerce_return_series(
    returns: pd.Series | pd.DataFrame,
    column: str,
) -> pd.Series:
    if isinstance(returns, pd.Series):
        series = returns.copy()
    else:
        if "trade_date" not in returns.columns:
            raise ValueError("Return frame must include trade_date.")
        if column not in returns.columns:
            raise KeyError(f"Return column not found: {column}")
        frame = returns.loc[:, ["trade_date", column]].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        if frame["trade_date"].duplicated().any():
            raise ValueError("Return frame contains duplicate trade dates.")
        series = frame.set_index("trade_date")[column]
    series.index = pd.to_datetime(series.index)
    if series.index.duplicated().any():
        raise ValueError("Return series contains duplicate trade dates.")
    return pd.to_numeric(series.sort_index(), errors="coerce").rename(column)


def _slice_prices_with_history(prices: pd.DataFrame, *, through: pd.Timestamp) -> pd.DataFrame:
    return prices.loc[prices["trade_date"] <= through].copy()


def _slice_prices(
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (prices["trade_date"] >= start) & (prices["trade_date"] <= end)
    return prices.loc[mask].copy()


def _select_best_parameters(
    *,
    runner: Callable[..., object],
    formation_prices: pd.DataFrame,
    benchmark_series: pd.Series,
    candidates: list[dict[str, object]],
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    fixed_kwargs: dict[str, object],
) -> tuple[dict[str, object], dict[str, float]]:
    best_candidate: dict[str, object] | None = None
    best_metrics: dict[str, float] | None = None

    for candidate in candidates:
        run_kwargs = dict(candidate)
        run_kwargs.update(fixed_kwargs)
        run = runner(formation_prices, **run_kwargs)
        daily = _build_daily_results(
            fold=WalkForwardFold(
                fold_id="formation",
                formation_start=formation_start,
                formation_end=formation_end,
                evaluation_start=formation_start,
                evaluation_end=formation_end,
            ),
            daily_results=run.daily_results,
            benchmark_series=benchmark_series,
        )
        daily = _filter_daily_window(daily, formation_start, formation_end)
        metrics = _score_candidate(daily)
        if best_metrics is None or _candidate_is_better(metrics, best_metrics):
            best_candidate = candidate
            best_metrics = metrics

    if best_candidate is None or best_metrics is None:
        raise ValueError("No valid publication parameter candidate was evaluated.")
    return best_candidate, best_metrics


def _build_daily_results(
    *,
    fold: WalkForwardFold,
    daily_results: pd.DataFrame,
    benchmark_series: pd.Series,
) -> pd.DataFrame:
    frame = daily_results.copy()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "trade_date"
    frame["benchmark_return"] = benchmark_series.reindex(frame.index)
    frame["benchmark_observed"] = frame["benchmark_return"].notna()
    frame["excess_return"] = frame["net_return"] - frame["benchmark_return"]
    frame["benchmark_nav"] = compute_nav(frame["benchmark_return"])
    frame["excess_nav"] = compute_nav(frame["excess_return"])
    frame = frame.reset_index()
    frame.insert(0, "fold_id", fold.fold_id)
    return frame


def _filter_daily_window(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (frame["trade_date"] >= start) & (frame["trade_date"] <= end)
    return frame.loc[mask].reset_index(drop=True)


def _score_candidate(daily_results: pd.DataFrame) -> dict[str, float]:
    comparable = daily_results.loc[daily_results["benchmark_observed"]].copy()
    if comparable.empty:
        raise ValueError("No benchmark-aligned observations are available for formation scoring.")
    strategy_nav = compute_nav(comparable["net_return"])
    benchmark_nav = compute_nav(comparable["benchmark_return"])
    return {
        "objective_value": float(strategy_nav.iloc[-1] - benchmark_nav.iloc[-1]),
        "total_net_excess_return": float(comparable["excess_return"].sum()),
        "average_turnover": float(comparable["turnover"].mean()),
        "benchmark_observation_count": int(len(comparable)),
    }


def _candidate_is_better(
    candidate: dict[str, float],
    incumbent: dict[str, float],
) -> bool:
    if candidate["objective_value"] != incumbent["objective_value"]:
        return candidate["objective_value"] > incumbent["objective_value"]
    return candidate["average_turnover"] < incumbent["average_turnover"]


def _build_summary(
    *,
    strategy_name: str,
    fold: WalkForwardFold,
    daily_results: pd.DataFrame,
    parameters: dict[str, object],
    risk_free_series: pd.Series | None,
) -> pd.DataFrame:
    dated = daily_results.set_index("trade_date").sort_index()
    comparable = dated.loc[dated["benchmark_observed"].astype(bool)].copy()
    if comparable.empty:
        raise ValueError("No benchmark-aligned observations are available for evaluation summary.")
    strategy_nav = compute_nav(comparable["net_return"])
    benchmark_nav = compute_nav(comparable["benchmark_return"])
    compounded_excess_nav = compute_nav(comparable["excess_return"])
    summary = summarize_return_stream(
        strategy_name,
        dated["net_return"],
        turnover=dated["turnover"],
        risk_free_returns=risk_free_series,
        extra_fields={
            "fold_id": fold.fold_id,
            "window_label": "evaluation",
            "benchmark_observation_count": int(dated["benchmark_observed"].sum()),
            "benchmark_missing_count": int((~dated["benchmark_observed"]).sum()),
            "total_net_excess_return": float(dated["excess_return"].sum(skipna=True)),
            "total_net_excess_return_label": "legacy_arithmetic_sum_of_daily_excess_returns",
            "net_excess_nav_difference": float(strategy_nav.iloc[-1] - benchmark_nav.iloc[-1]),
            "net_excess_nav_difference_label": "preferred_strategy_total_return_minus_benchmark_total_return",
            "compounded_daily_excess_stream": float(compounded_excess_nav.iloc[-1] - 1.0),
            "chosen_parameters": _format_parameters(parameters),
        },
    )
    return summary


def _format_parameters(parameters: dict[str, object]) -> str:
    return "|".join(f"{key}={parameters[key]}" for key in sorted(parameters))
