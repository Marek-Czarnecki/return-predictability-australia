from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .costs import apply_turnover_costs, resolve_cost_scenario
from .metrics import compute_nav, summarize_return_stream
from .pairs_trading import (
    _build_pair_position,
    _build_strategy_daily,
    _coerce_selected_pairs,
    _compute_log_spread,
    select_pairs_in_window,
)
from .publication_data import prepare_publication_strategy_panel
from .publication_walk_forward import (
    _build_daily_results,
    _build_summary,
    _candidate_is_better,
    _coerce_return_series,
    _filter_daily_window,
    _format_parameters,
    _score_candidate,
)
from .publication_walk_forward_costs import build_publication_fold_cost_context
from .walk_forward import generate_walk_forward_folds


PAIR_FORMATION_HISTORY_CONVENTION = (
    "all_observable_prices_within_formation_window_for_formation_end_eligible_assets"
)
PAIR_CAPITAL_NORMALIZATION = "gross_exposure_normalized_by_1_plus_abs_hedge_ratio"
PAIR_COST_APPLICATION = "normalized_two_leg_weighted_tier_cost"
PAIR_BORROW_FINANCING = "excluded_disclosed_limitation"


@dataclass
class PublicationPairsWalkForwardResult:
    fold_table: pd.DataFrame
    fold_daily_results: pd.DataFrame
    fold_summary: pd.DataFrame
    liquidity_diagnostics: pd.DataFrame
    pair_diagnostics: pd.DataFrame


PAIR_PARAMETER_GRID = {
    "spread_window": [10, 20],
    "entry_z": [1.5, 2.0],
    "exit_z": [0.25, 0.5],
    "cost_scenario": ["base"],
}


def run_publication_pairs_walk_forward(
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
    top_liquid_tickers: int = 20,
    top_pair_count: int = 5,
    engle_granger_pvalue_threshold: float = 0.05,
    sector_map: pd.DataFrame | None = None,
    sector_map_path: Path | None = None,
) -> PublicationPairsWalkForwardResult:
    """Run the controlled publication pairs experiment.

    Formation convention A is explicit: candidate identities must be eligible at
    formation end, but pair fitting may use all observed prices for those
    identities inside the three-year formation window, including observations
    before index membership. Evaluation requires both legs to be eligible on the
    signal date. Gross returns and transaction costs are normalized to one unit
    of gross capital. Borrow and financing costs are intentionally excluded.

    ``sector_map`` / ``sector_map_path`` are portability inputs only. Supplying
    either makes the sector-classification dependency explicit; omitting both
    preserves the frozen default lookup used by ``select_pairs_in_window``.
    """
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be at least 1 when supplied.")

    prepared = prepare_publication_strategy_panel(prices, min_history=1)
    benchmark_series = _coerce_return_series(benchmark_returns, benchmark_col)
    risk_free_series = (
        _coerce_return_series(risk_free_returns, risk_free_col)
        if risk_free_returns is not None
        else None
    )
    folds = generate_walk_forward_folds(
        prepared,
        formation_years=formation_years,
        evaluation_years=evaluation_years,
        step_years=step_years,
    )
    if max_folds is not None:
        folds = folds[:max_folds]

    fold_rows = []
    daily_frames = []
    summary_frames = []
    liquidity_frames = []
    pair_frames = []
    candidates = _expand_grid(PAIR_PARAMETER_GRID)

    for fold in folds:
        formation = _slice(prepared, fold.formation_start, fold.formation_end)
        evaluation = _slice(prepared, fold.evaluation_start, fold.evaluation_end)
        coverage = pd.concat([formation, evaluation], ignore_index=True)
        cost_context = build_publication_fold_cost_context(
            prices=prepared,
            fold=fold,
            evaluation_prices=coverage,
            identity_col="asset_id",
        )
        liquidity_frames.append(cost_context.liquidity_diagnostics)

        _, selected_pairs = select_pairs_in_window(
            formation,
            top_liquid_tickers=top_liquid_tickers,
            top_pair_count=top_pair_count,
            sector_map=sector_map,
            sector_map_path=sector_map_path,
            engle_granger_pvalue_threshold=engle_granger_pvalue_threshold,
            liquidity_tier_map=cost_context.liquidity_tier_map,
            identity_col="asset_id",
            eligibility_col="eligible_to_trade",
            label_col="ticker_code",
        )

        best_candidate = None
        best_metrics = None
        for candidate in candidates:
            formation_daily = _run_pair_daily(
                formation,
                selected_pairs,
                **candidate,
            )
            formation_daily = _ensure_daily_calendar(
                formation_daily,
                benchmark_series,
                fold.formation_start,
                fold.formation_end,
            )
            scored = _build_daily_results(
                fold=fold,
                daily_results=formation_daily,
                benchmark_series=benchmark_series,
            )
            scored = _filter_daily_window(scored, fold.formation_start, fold.formation_end)
            metrics = _score_candidate(scored)
            if best_metrics is None or _candidate_is_better(metrics, best_metrics):
                best_candidate = candidate
                best_metrics = metrics

        if best_candidate is None or best_metrics is None:
            raise ValueError(f"No valid pairs candidate for {fold.fold_id}.")

        evaluation_daily = _run_pair_daily(
            evaluation,
            selected_pairs,
            **best_candidate,
        )
        evaluation_daily = _ensure_daily_calendar(
            evaluation_daily,
            benchmark_series,
            fold.evaluation_start,
            fold.evaluation_end,
        )
        evaluation_daily = _build_daily_results(
            fold=fold,
            daily_results=evaluation_daily,
            benchmark_series=benchmark_series,
        )
        evaluation_daily = _filter_daily_window(
            evaluation_daily, fold.evaluation_start, fold.evaluation_end
        )
        daily_frames.append(evaluation_daily)
        summary = _build_summary(
            strategy_name="pairs_trading",
            fold=fold,
            daily_results=evaluation_daily,
            parameters=best_candidate,
            risk_free_series=risk_free_series,
        )
        summary["formation_history_convention"] = PAIR_FORMATION_HISTORY_CONVENTION
        summary["capital_normalization"] = PAIR_CAPITAL_NORMALIZATION
        summary["pair_cost_application"] = PAIR_COST_APPLICATION
        summary["borrow_financing"] = PAIR_BORROW_FINANCING
        summary["selected_pair_count"] = int(len(selected_pairs))
        summary_frames.append(summary)

        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "formation_start": fold.formation_start,
                "formation_end": fold.formation_end,
                "evaluation_start": fold.evaluation_start,
                "evaluation_end": fold.evaluation_end,
                "strategy_name": "pairs_trading",
                "identity_col": "asset_id",
                "eligibility_col": "eligible_to_trade",
                "selection_objective": "net_excess_nav_vs_benchmark",
                "benchmark_missing_policy": "exclude_from_objective_and_excess_metrics",
                "chosen_parameters": _format_parameters(best_candidate),
                "formation_objective_value": best_metrics["objective_value"],
                "formation_total_net_excess_return": best_metrics["total_net_excess_return"],
                "formation_average_turnover": best_metrics["average_turnover"],
                "formation_benchmark_observation_count": best_metrics["benchmark_observation_count"],
                "selected_pair_count": int(len(selected_pairs)),
                "formation_history_convention": PAIR_FORMATION_HISTORY_CONVENTION,
                "capital_normalization": PAIR_CAPITAL_NORMALIZATION,
                "pair_cost_application": PAIR_COST_APPLICATION,
                "borrow_financing": PAIR_BORROW_FINANCING,
            }
        )
        if not selected_pairs.empty:
            diag = selected_pairs.copy()
            diag.insert(0, "fold_id", fold.fold_id)
            diag["chosen_parameters"] = _format_parameters(best_candidate)
            pair_frames.append(diag)

    return PublicationPairsWalkForwardResult(
        fold_table=pd.DataFrame(fold_rows),
        fold_daily_results=pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        fold_summary=pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
        liquidity_diagnostics=pd.concat(liquidity_frames, ignore_index=True) if liquidity_frames else pd.DataFrame(),
        pair_diagnostics=pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(),
    )


def build_publication_pair_return_panel(
    trading_prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    spread_window: int,
    entry_z: float,
    exit_z: float,
    cost_scenario: str = "base",
) -> pd.DataFrame:
    if selected_pairs.empty:
        return pd.DataFrame()
    selected_pairs = _coerce_selected_pairs(selected_pairs)
    prices = trading_prices.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    price_matrix = prices.pivot(index="trade_date", columns="asset_id", values="adj_close").sort_index()
    return_matrix = price_matrix.pct_change(fill_method=None)
    eligibility = (
        prices.pivot(index="trade_date", columns="asset_id", values="eligible_to_trade")
        .sort_index()
        .fillna(False)
        .astype(bool)
    )
    scenario = resolve_cost_scenario(cost_scenario)
    frames = []

    for _, pair in selected_pairs.iterrows():
        left = pair["left_identity"]
        right = pair["right_identity"]
        if left not in price_matrix.columns or right not in price_matrix.columns:
            continue
        hedge = float(pair["hedge_ratio"])
        denominator = 1.0 + abs(hedge)
        left_weight = 1.0 / denominator
        right_weight = abs(hedge) / denominator
        spread = _compute_log_spread(
            price_matrix[left], price_matrix[right], hedge_ratio=hedge, intercept=float(pair["intercept"])
        )
        spread_mean = spread.rolling(spread_window).mean()
        spread_std = spread.rolling(spread_window).std().replace(0.0, np.nan)
        spread_z = ((spread - spread_mean) / spread_std).replace([np.inf, -np.inf], np.nan)
        position = _build_pair_position(spread_z, entry_z=entry_z, exit_z=exit_z)
        pair_eligible = (eligibility[left] & eligibility[right]).reindex(position.index).fillna(False)
        position = position.where(pair_eligible, 0.0)
        previous_position = position.shift(1).fillna(0.0)
        raw_spread_return = return_matrix[left] - hedge * return_matrix[right]
        gross_return = previous_position * (raw_spread_return / denominator)
        gross_return = gross_return.where(previous_position.ne(0.0), 0.0)
        turnover = position.diff().abs().fillna(0.0)
        left_bps = scenario.cost_bps_for_tier(str(pair["left_liquidity_tier"]))
        right_bps = scenario.cost_bps_for_tier(str(pair["right_liquidity_tier"]))
        effective_bps = left_weight * left_bps + right_weight * right_bps
        net_return = apply_turnover_costs(gross_return, turnover, turnover_cost_bps=effective_bps)
        short_exposure = pd.Series(0.0, index=position.index, dtype=float)
        short_exposure.loc[position > 0] = right_weight
        short_exposure.loc[position < 0] = left_weight
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": position.index,
                    "pair_id": pair["pair_id"],
                    "left_identity": left,
                    "right_identity": right,
                    "left_ticker": pair["left_ticker"],
                    "right_ticker": pair["right_ticker"],
                    "pair_gross_return": gross_return.values,
                    "pair_net_return": net_return.values,
                    "pair_turnover": turnover.values,
                    "pair_turnover_cost_bps": effective_bps,
                    "left_liquidity_tier": pair["left_liquidity_tier"],
                    "right_liquidity_tier": pair["right_liquidity_tier"],
                    "spread": spread.values,
                    "spread_z": spread_z.values,
                    "pair_position": position.values,
                    "pair_short_exposure": short_exposure.values,
                    "gross_exposure_denominator": denominator,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_pair_daily(prices: pd.DataFrame, selected_pairs: pd.DataFrame, **parameters) -> pd.DataFrame:
    panel = build_publication_pair_return_panel(
        prices,
        selected_pairs,
        spread_window=int(parameters["spread_window"]),
        entry_z=float(parameters["entry_z"]),
        exit_z=float(parameters["exit_z"]),
        cost_scenario=str(parameters["cost_scenario"]),
    )
    return _build_strategy_daily(panel)


def _ensure_daily_calendar(
    daily: pd.DataFrame,
    benchmark_series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    calendar = benchmark_series.loc[(benchmark_series.index >= start) & (benchmark_series.index <= end)].index
    frame = daily.reindex(calendar).copy() if not daily.empty else pd.DataFrame(index=calendar)
    for column in ("gross_return", "net_return", "turnover", "average_turnover", "short_exposure", "effective_cost_bps"):
        if column not in frame.columns:
            frame[column] = 0.0
    no_strategy_obs = frame["net_return"].isna() & frame["gross_return"].isna()
    frame.loc[no_strategy_obs, ["gross_return", "net_return", "turnover", "average_turnover", "short_exposure", "effective_cost_bps"]] = 0.0
    frame["nav"] = compute_nav(frame["net_return"])
    frame.index.name = "trade_date"
    return frame


def _slice(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (prices["trade_date"] >= start) & (prices["trade_date"] <= end)
    return prices.loc[mask].copy()


def _expand_grid(grid: dict[str, list[object]]) -> list[dict[str, object]]:
    names = list(grid)
    return [dict(zip(names, values)) for values in product(*(grid[name] for name in names))]
