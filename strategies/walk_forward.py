from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import pandas as pd

from .mean_reversion import run_mean_reversion_strategy
from .metrics import compute_nav, summarize_return_stream
from .pairs_trading import run_pairs_trading_strategy
from .trend_following import run_trend_following_strategy


StrategyCallable = Callable[..., Any]


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    fold_table: pd.DataFrame
    fold_daily_results: pd.DataFrame
    fold_summary: pd.DataFrame
    fold_diagnostics: pd.DataFrame


@dataclass(frozen=True)
class StrategyDefinition:
    runner: StrategyCallable
    parameter_grid: dict[str, list[object]]


DEFAULT_STRATEGY_DEFINITIONS: dict[str, StrategyDefinition] = {
    "trend_following": StrategyDefinition(
        runner=run_trend_following_strategy,
        parameter_grid={
            "fast_window": [50, 100],
            "slow_window": [150, 200],
            "min_history": [220],
            "cost_scenario": ["base"],
        },
    ),
    "mean_reversion": StrategyDefinition(
        runner=run_mean_reversion_strategy,
        parameter_grid={
            "lookback_window": [10, 20],
            "entry_z": [-1.5, -2.0],
            "exit_z": [-0.25, -0.5],
            "min_history": [60],
            "cost_scenario": ["base"],
        },
    ),
    "pairs_trading": StrategyDefinition(
        runner=run_pairs_trading_strategy,
        parameter_grid={
            "top_liquid_tickers": [20],
            "top_pair_count": [5],
            "spread_window": [10, 20],
            "entry_z": [1.5, 2.0],
            "exit_z": [0.25, 0.5],
            "cost_scenario": ["base"],
        },
    ),
}


def generate_walk_forward_folds(
    prices: pd.DataFrame,
    formation_years: int = 3,
    evaluation_years: int = 1,
    step_years: int = 1,
) -> list[WalkForwardFold]:
    trade_dates = pd.Index(pd.to_datetime(prices["trade_date"])).sort_values().unique()
    if trade_dates.empty:
        return []

    folds: list[WalkForwardFold] = []
    fold_start = pd.Timestamp(trade_dates[0])
    final_trade_date = pd.Timestamp(trade_dates[-1])
    fold_number = 1

    while True:
        formation_end_exclusive = fold_start + pd.DateOffset(years=formation_years)
        evaluation_end_exclusive = formation_end_exclusive + pd.DateOffset(
            years=evaluation_years
        )
        formation_dates = trade_dates[
            (trade_dates >= fold_start) & (trade_dates < formation_end_exclusive)
        ]
        evaluation_dates = trade_dates[
            (trade_dates >= formation_end_exclusive)
            & (trade_dates < evaluation_end_exclusive)
        ]
        if len(formation_dates) == 0 or len(evaluation_dates) == 0:
            break

        folds.append(
            WalkForwardFold(
                fold_id=f"fold_{fold_number:02d}",
                formation_start=pd.Timestamp(formation_dates[0]),
                formation_end=pd.Timestamp(formation_dates[-1]),
                evaluation_start=pd.Timestamp(evaluation_dates[0]),
                evaluation_end=pd.Timestamp(evaluation_dates[-1]),
            )
        )
        fold_number += 1
        fold_start = fold_start + pd.DateOffset(years=step_years)
        if fold_start > final_trade_date:
            break

    return folds


def expand_parameter_grid(parameter_grid: dict[str, list[object]]) -> list[dict[str, object]]:
    if not parameter_grid:
        return [{}]
    parameter_names = list(parameter_grid.keys())
    combinations = product(*(parameter_grid[name] for name in parameter_names))
    return [
        dict(zip(parameter_names, combination))
        for combination in combinations
    ]


def parse_chosen_parameters(parameter_string: str | object) -> dict[str, object]:
    if parameter_string is None or pd.isna(parameter_string):
        return {}
    text = str(parameter_string).strip()
    if not text:
        return {}

    parameters: dict[str, object] = {}
    for field in text.split("|"):
        key, separator, raw_value = field.partition("=")
        if separator != "=":
            raise ValueError(f"Malformed chosen-parameter field: {field}")
        parameters[key] = _parse_parameter_value(raw_value)
    return parameters


def prepare_strategy_evaluation_panel(
    strategy_name: str,
    prices: pd.DataFrame,
    fold: WalkForwardFold,
    parameters: dict[str, object] | None = None,
    strategy_definitions: dict[str, StrategyDefinition] | None = None,
    strategy_run_kwargs: dict[str, object] | None = None,
) -> tuple[object, pd.DataFrame]:
    definitions = strategy_definitions or DEFAULT_STRATEGY_DEFINITIONS
    if strategy_name not in definitions:
        raise KeyError(f"Unknown strategy_name: {strategy_name}")

    definition = definitions[strategy_name]
    sample_start = pd.Timestamp(pd.to_datetime(prices["trade_date"]).min())
    uses_history_carry = _strategy_uses_date_specific_history(definition.parameter_grid)
    evaluation_prices = _slice_prices(
        prices,
        sample_start if uses_history_carry else fold.evaluation_start,
        fold.evaluation_end,
    )
    run_kwargs = dict(parameters or {})
    run_kwargs.update(strategy_run_kwargs or {})
    if strategy_name == "pairs_trading":
        formation_prices = _slice_prices(
            prices,
            sample_start if uses_history_carry else fold.formation_start,
            fold.formation_end,
        )
        run_kwargs["formation_prices"] = formation_prices
    evaluation_run = definition.runner(evaluation_prices, **run_kwargs)
    evaluation_panel = getattr(evaluation_run, "panel", evaluation_prices).copy()
    evaluation_panel["trade_date"] = pd.to_datetime(evaluation_panel["trade_date"])
    if uses_history_carry:
        evaluation_panel = _filter_panel_window(
            evaluation_panel,
            start_date=fold.evaluation_start,
            end_date=fold.evaluation_end,
        )
    return evaluation_run, evaluation_panel.reset_index(drop=True)


def run_walk_forward_optimization(
    strategy_name: str,
    prices: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
    risk_free_returns: pd.Series | pd.DataFrame | None = None,
    benchmark_col: str = "benchmark_return",
    risk_free_col: str = "risk_free_return",
    formation_years: int = 3,
    evaluation_years: int = 1,
    step_years: int = 1,
    strategy_definitions: dict[str, StrategyDefinition] | None = None,
    strategy_run_kwargs: dict[str, object] | None = None,
) -> WalkForwardResult:
    definitions = strategy_definitions or DEFAULT_STRATEGY_DEFINITIONS
    if strategy_name not in definitions:
        raise KeyError(f"Unknown strategy_name: {strategy_name}")

    benchmark_series = _coerce_benchmark_series(benchmark_returns, benchmark_col)
    sample_start = pd.Timestamp(pd.to_datetime(prices["trade_date"]).min())
    folds = generate_walk_forward_folds(
        prices,
        formation_years=formation_years,
        evaluation_years=evaluation_years,
        step_years=step_years,
    )
    if not folds:
        return WalkForwardResult(
            fold_table=pd.DataFrame(),
            fold_daily_results=pd.DataFrame(),
            fold_summary=pd.DataFrame(),
            fold_diagnostics=pd.DataFrame(),
        )

    definition = definitions[strategy_name]
    uses_history_carry = _strategy_uses_date_specific_history(definition.parameter_grid)
    candidates = expand_parameter_grid(definition.parameter_grid)
    fold_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []

    for fold in folds:
        formation_prices = _slice_prices(
            prices,
            sample_start if uses_history_carry else fold.formation_start,
            fold.formation_end,
        )
        evaluation_prices = _slice_prices(
            prices,
            sample_start if uses_history_carry else fold.evaluation_start,
            fold.evaluation_end,
        )
        best_candidate, best_metrics = _select_best_parameters(
            definition.runner,
            formation_prices,
            benchmark_series,
            candidates,
            formation_start=fold.formation_start,
            formation_end=fold.formation_end,
            strategy_run_kwargs=strategy_run_kwargs,
        )
        evaluation_kwargs = dict(best_candidate)
        evaluation_kwargs.update(strategy_run_kwargs or {})
        if strategy_name == "pairs_trading":
            evaluation_kwargs["formation_prices"] = formation_prices
        evaluation_run = definition.runner(evaluation_prices, **evaluation_kwargs)
        if strategy_name == "pairs_trading":
            diagnostics = _build_pairs_fold_diagnostics(
                fold=fold,
                strategy_name=strategy_name,
                parameters=best_candidate,
                selected_pairs=getattr(evaluation_run, "selected_pairs", pd.DataFrame()),
            )
            if not diagnostics.empty:
                diagnostics_frames.append(diagnostics)
        evaluation_daily = _build_fold_daily_results(
            fold.fold_id,
            "evaluation",
            evaluation_run.daily_results,
            benchmark_series,
        )
        if uses_history_carry:
            evaluation_daily = _filter_daily_results_window(
                evaluation_daily,
                start_date=fold.evaluation_start,
                end_date=fold.evaluation_end,
            )
        summary_frames.append(
            _build_fold_summary(
                strategy_name=strategy_name,
                fold_id=fold.fold_id,
                window_label="evaluation",
                daily_results=evaluation_daily,
                parameters=best_candidate,
                risk_free_returns=risk_free_returns,
                risk_free_col=risk_free_col,
            )
        )
        daily_frames.append(evaluation_daily)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "formation_start": fold.formation_start,
                "formation_end": fold.formation_end,
                "evaluation_start": fold.evaluation_start,
                "evaluation_end": fold.evaluation_end,
                "strategy_name": strategy_name,
                "selection_objective": "net_excess_return_vs_benchmark",
                "selection_turnover_role": "secondary_diagnostic",
                "chosen_parameters": _format_parameters(best_candidate),
                "formation_objective_value": best_metrics["objective_value"],
                "formation_total_net_excess_return": best_metrics[
                    "total_net_excess_return"
                ],
                "formation_average_turnover": best_metrics["average_turnover"],
            }
        )

    fold_table = pd.DataFrame(fold_rows)
    fold_daily_results = (
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    )
    fold_summary = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )
    fold_diagnostics = (
        pd.concat(diagnostics_frames, ignore_index=True)
        if diagnostics_frames
        else pd.DataFrame()
    )
    return WalkForwardResult(
        fold_table=fold_table,
        fold_daily_results=fold_daily_results,
        fold_summary=fold_summary,
        fold_diagnostics=fold_diagnostics,
    )


def _coerce_benchmark_series(
    benchmark_returns: pd.Series | pd.DataFrame, benchmark_col: str
) -> pd.Series:
    if isinstance(benchmark_returns, pd.Series):
        benchmark_series = benchmark_returns.copy()
    else:
        if benchmark_col not in benchmark_returns.columns:
            raise KeyError(f"Benchmark column not found: {benchmark_col}")
        benchmark_series = benchmark_returns.set_index("trade_date")[benchmark_col]
    benchmark_series.index = pd.to_datetime(benchmark_series.index)
    benchmark_series = benchmark_series.sort_index().rename(benchmark_col)
    return benchmark_series


def _slice_prices(
    prices: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    mask = (prices["trade_date"] >= start_date) & (prices["trade_date"] <= end_date)
    return prices.loc[mask].copy()


def _select_best_parameters(
    runner: StrategyCallable,
    formation_prices: pd.DataFrame,
    benchmark_series: pd.Series,
    candidates: list[dict[str, object]],
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    strategy_run_kwargs: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, float]]:
    best_candidate: dict[str, object] | None = None
    best_metrics: dict[str, float] | None = None
    for candidate in candidates:
        run_kwargs = dict(candidate)
        run_kwargs.update(strategy_run_kwargs or {})
        run = runner(formation_prices, **run_kwargs)
        candidate_daily = _build_fold_daily_results(
            fold_id="formation",
            window_label="formation",
            daily_results=run.daily_results,
            benchmark_series=benchmark_series,
        )
        if "min_history" in candidate:
            candidate_daily = _filter_daily_results_window(
                candidate_daily,
                start_date=formation_start,
                end_date=formation_end,
            )
        metrics = _score_fold(candidate_daily)
        if best_metrics is None or _candidate_is_better(metrics, best_metrics):
            best_candidate = candidate
            best_metrics = metrics

    if best_candidate is None or best_metrics is None:
        raise ValueError("No valid parameter candidates were evaluated.")
    return best_candidate, best_metrics


def _build_fold_daily_results(
    fold_id: str,
    window_label: str,
    daily_results: pd.DataFrame,
    benchmark_series: pd.Series,
) -> pd.DataFrame:
    frame = daily_results.copy()
    frame.index = pd.to_datetime(frame.index)
    aligned_benchmark = benchmark_series.reindex(frame.index).fillna(0.0)
    frame["benchmark_return"] = aligned_benchmark
    frame["excess_return"] = frame["net_return"] - frame["benchmark_return"]
    frame["benchmark_nav"] = compute_nav(frame["benchmark_return"])
    frame["excess_nav"] = compute_nav(frame["excess_return"])
    frame = frame.reset_index().rename(columns={"index": "trade_date"})
    frame.insert(0, "window_label", window_label)
    frame.insert(0, "fold_id", fold_id)
    return frame


def _score_fold(daily_results: pd.DataFrame) -> dict[str, float]:
    strategy_nav = compute_nav(daily_results["net_return"])
    benchmark_nav = compute_nav(daily_results["benchmark_return"])
    objective_value = strategy_nav.iloc[-1] - benchmark_nav.iloc[-1]
    return {
        "objective_value": float(objective_value),
        "total_net_excess_return": float(daily_results["excess_return"].sum()),
        "average_turnover": float(daily_results["turnover"].mean()),
    }


def _candidate_is_better(
    candidate_metrics: dict[str, float], incumbent_metrics: dict[str, float]
) -> bool:
    if candidate_metrics["objective_value"] != incumbent_metrics["objective_value"]:
        return candidate_metrics["objective_value"] > incumbent_metrics["objective_value"]
    return candidate_metrics["average_turnover"] < incumbent_metrics["average_turnover"]


def _build_fold_summary(
    strategy_name: str,
    fold_id: str,
    window_label: str,
    daily_results: pd.DataFrame,
    parameters: dict[str, object],
    risk_free_returns: pd.Series | pd.DataFrame | None,
    risk_free_col: str,
) -> pd.DataFrame:
    dated_daily_results = daily_results.copy()
    dated_daily_results["trade_date"] = pd.to_datetime(dated_daily_results["trade_date"])
    dated_daily_results = dated_daily_results.set_index("trade_date").sort_index()
    summary = summarize_return_stream(
        strategy_name,
        dated_daily_results["net_return"],
        turnover=dated_daily_results["turnover"],
        risk_free_returns=risk_free_returns,
        risk_free_col=risk_free_col,
        extra_fields={
            "fold_id": fold_id,
            "window_label": window_label,
            "selection_objective": "net_excess_return_vs_benchmark",
            "total_net_excess_return": dated_daily_results["excess_return"].sum(),
            "absolute_return_label": "absolute_return",
            "benchmark_relative_return_label": "net_excess_return_vs_benchmark",
            "benchmark_total_return": (
                compute_nav(dated_daily_results["benchmark_return"]).iloc[-1] - 1
            ),
            "chosen_parameters": _format_parameters(parameters),
        },
    )
    return summary


def _format_parameters(parameters: dict[str, object]) -> str:
    return "|".join(f"{name}={value}" for name, value in sorted(parameters.items()))


def _build_pairs_fold_diagnostics(
    fold: WalkForwardFold,
    strategy_name: str,
    parameters: dict[str, object],
    selected_pairs: pd.DataFrame,
) -> pd.DataFrame:
    if selected_pairs.empty:
        return pd.DataFrame()

    diagnostics = selected_pairs.copy().reset_index(drop=True)
    diagnostics.insert(0, "pair_selection_rank", diagnostics.index + 1)
    diagnostics.insert(0, "chosen_parameters", _format_parameters(parameters))
    diagnostics.insert(0, "strategy_name", strategy_name)
    diagnostics.insert(0, "evaluation_end", fold.evaluation_end)
    diagnostics.insert(0, "evaluation_start", fold.evaluation_start)
    diagnostics.insert(0, "formation_end", fold.formation_end)
    diagnostics.insert(0, "formation_start", fold.formation_start)
    diagnostics.insert(0, "fold_id", fold.fold_id)
    diagnostics["used_fallback_candidate_generation"] = diagnostics["candidate_source"].eq(
        "sector_fallback"
    )
    return diagnostics


def _filter_daily_results_window(
    daily_results: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    mask = (daily_results["trade_date"] >= start_date) & (
        daily_results["trade_date"] <= end_date
    )
    return daily_results.loc[mask].reset_index(drop=True)


def _filter_panel_window(
    panel: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    mask = (panel["trade_date"] >= start_date) & (panel["trade_date"] <= end_date)
    return panel.loc[mask].copy()


def _strategy_uses_date_specific_history(
    parameter_grid: dict[str, list[object]],
) -> bool:
    return "min_history" in parameter_grid


def _parse_parameter_value(raw_value: str) -> object:
    text = str(raw_value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(character in text for character in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text
