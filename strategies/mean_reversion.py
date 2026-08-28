from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run_equal_weight_long_only_backtest
from .bias_review import apply_date_specific_minimum_history_rule
from .costs import DEFAULT_COST_SCENARIO_NAME, summarize_cost_model_inputs
from .metrics import summarize_return_stream
from .trend_following import StrategyRun


def run_mean_reversion_strategy(
    prices,
    lookback_window: int = 20,
    entry_z: float = -2.0,
    exit_z: float = -0.5,
    min_history: int = 60,
    turnover_cost_bps: float | None = None,
    cost_scenario: str = DEFAULT_COST_SCENARIO_NAME,
    benchmark_returns: dict[str, np.ndarray | object] | None = None,
    risk_free_returns: np.ndarray | object | None = None,
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
                "Publication mean-reversion panel is missing required columns: "
                f"{missing_list}"
            )
        strategy_prices = prices.copy()
        strategy_prices["trade_date"] = np.array(
            strategy_prices["trade_date"], dtype="datetime64[ns]"
        )
        strategy_prices = strategy_prices.sort_values(
            [identity_col, "trade_date"]
        ).reset_index(drop=True)
        resolved_eligibility_col = eligibility_col

    rolling_mean = strategy_prices.groupby(identity_col, observed=True)[
        "daily_return"
    ].transform(lambda series: series.rolling(lookback_window).mean())
    rolling_std = strategy_prices.groupby(identity_col, observed=True)[
        "daily_return"
    ].transform(lambda series: series.rolling(lookback_window).std())
    strategy_prices["return_zscore"] = (
        strategy_prices["daily_return"] - rolling_mean
    ) / rolling_std

    if eligibility_col is None:
        entry_flag = (strategy_prices["history_count"] >= min_history) & (
            strategy_prices["return_zscore"] <= entry_z
        )
    else:
        entry_flag = strategy_prices[resolved_eligibility_col].astype(bool) & (
            strategy_prices["return_zscore"] <= entry_z
        )
    exit_flag = strategy_prices["return_zscore"] >= exit_z

    strategy_prices["signal"] = np.nan
    strategy_prices.loc[entry_flag, "signal"] = 1.0
    strategy_prices.loc[exit_flag, "signal"] = 0.0
    strategy_prices["signal"] = (
        strategy_prices.groupby(identity_col, observed=True)["signal"]
        .ffill()
        .fillna(0.0)
    )
    if eligibility_col is not None:
        strategy_prices.loc[
            ~strategy_prices[resolved_eligibility_col].astype(bool), "signal"
        ] = 0.0

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
        "mean_reversion_return_zscore",
        results["net_return"],
        turnover=results["turnover"],
        benchmark_returns=benchmark_returns,
        risk_free_returns=risk_free_returns,
        extra_fields={
            "lookback_window": lookback_window,
            "entry_z": entry_z,
            "exit_z": exit_z,
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
