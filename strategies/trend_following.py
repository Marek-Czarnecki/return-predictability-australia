from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import run_equal_weight_long_only_backtest
from .bias_review import apply_date_specific_minimum_history_rule
from .costs import DEFAULT_COST_SCENARIO_NAME, summarize_cost_model_inputs
from .metrics import summarize_return_stream


@dataclass
class StrategyRun:
    panel: pd.DataFrame
    daily_results: pd.DataFrame
    summary: pd.DataFrame


def run_trend_following_strategy(
    prices: pd.DataFrame,
    fast_window: int = 50,
    slow_window: int = 200,
    min_history: int = 220,
    turnover_cost_bps: float | None = None,
    cost_scenario: str = DEFAULT_COST_SCENARIO_NAME,
    benchmark_returns: dict[str, pd.Series | pd.DataFrame] | None = None,
    risk_free_returns: pd.Series | pd.DataFrame | None = None,
    identity_col: str = "ticker_code",
    eligibility_col: str | None = None,
    liquidity_tier_map: pd.Series | None = None,
) -> StrategyRun:
    if eligibility_col is None:
        strategy_prices = apply_date_specific_minimum_history_rule(
            prices,
            min_history=min_history,
            ticker_col=identity_col,
        )
        resolved_eligibility_col = "eligible_flag"
    else:
        required_columns = {identity_col, "trade_date", eligibility_col}
        missing_columns = required_columns.difference(prices.columns)
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Publication trend panel is missing required columns: " f"{missing_list}"
            )
        strategy_prices = prices.copy()
        strategy_prices["trade_date"] = pd.to_datetime(strategy_prices["trade_date"])
        strategy_prices = strategy_prices.sort_values(
            [identity_col, "trade_date"]
        ).reset_index(drop=True)
        resolved_eligibility_col = eligibility_col

    strategy_prices["ma_fast"] = (
        strategy_prices.groupby(identity_col, observed=True)["adj_close"]
        .transform(lambda series: series.rolling(fast_window).mean())
    )
    strategy_prices["ma_slow"] = (
        strategy_prices.groupby(identity_col, observed=True)["adj_close"]
        .transform(lambda series: series.rolling(slow_window).mean())
    )
    strategy_prices["signal"] = (
        strategy_prices[resolved_eligibility_col].astype(bool)
        & strategy_prices["ma_fast"].notna()
        & strategy_prices["ma_slow"].notna()
        & (strategy_prices["ma_fast"] > strategy_prices["ma_slow"])
    ).astype(float)

    results = run_equal_weight_long_only_backtest(
        strategy_prices,
        signal_col="signal",
        return_col="daily_return",
        turnover_cost_bps=turnover_cost_bps,
        cost_scenario=cost_scenario,
        liquidity_tier_map=liquidity_tier_map,
        identity_col=identity_col,
    )
    summary = summarize_return_stream(
        "trend_following_ma_crossover",
        results["net_return"],
        turnover=results["turnover"],
        benchmark_returns=benchmark_returns,
        risk_free_returns=risk_free_returns,
        extra_fields={
            "fast_window": fast_window,
            "slow_window": slow_window,
            "min_history": min_history,
            "identity_col": identity_col,
            "eligibility_col": resolved_eligibility_col,
            **summarize_cost_model_inputs(
                cost_scenario=cost_scenario,
                turnover_cost_bps=turnover_cost_bps,
            ),
        },
    )
    return StrategyRun(panel=strategy_prices, daily_results=results, summary=summary)
