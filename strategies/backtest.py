from __future__ import annotations

import pandas as pd

from .costs import (
    build_liquidity_tier_map,
    build_tier_turnover_frame,
    build_ticker_cost_bps_map,
    calculate_turnover_costs_from_tier_turnover,
)
from .metrics import compute_nav


def build_benchmark_table(
    prices: pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    rba_cash_rate_tri: pd.Series | pd.DataFrame | None = None,
    return_col: str = "daily_return",
    benchmark_col: str = "benchmark_return",
    risk_free_tri_col: str = "rba_cash_rate_tri",
    risk_free_return_col: str = "risk_free_return",
    identity_col: str = "ticker_code",
) -> pd.DataFrame:
    benchmark_panel = (
        prices.pivot(index="trade_date", columns=identity_col, values=return_col)
        .sort_index()
    )
    equal_weight_benchmark = benchmark_panel.mean(axis=1, skipna=True).rename(
        "equal_weight_return"
    )
    benchmark_summary = pd.DataFrame(
        {
            "equal_weight_return": equal_weight_benchmark,
            "equal_weight_nav": compute_nav(equal_weight_benchmark),
        }
    )
    if benchmark_returns is not None:
        benchmark_series = _coerce_benchmark_series(
            benchmark_returns, benchmark_col
        ).reindex(
            benchmark_summary.index
        )
        benchmark_summary[benchmark_col] = benchmark_series
        benchmark_summary["benchmark_nav"] = compute_nav(
            benchmark_summary[benchmark_col]
        )
    if rba_cash_rate_tri is not None:
        tri_series = _coerce_benchmark_series(
            rba_cash_rate_tri, risk_free_tri_col
        ).reindex(benchmark_summary.index, method="ffill")
        benchmark_summary[risk_free_tri_col] = tri_series
        benchmark_summary[risk_free_return_col] = build_risk_free_returns_from_tri(
            tri_series
        )
    benchmark_summary.index.name = "trade_date"
    return benchmark_summary


def build_equal_weight_benchmark(
    prices: pd.DataFrame,
    return_col: str = "daily_return",
    identity_col: str = "ticker_code",
) -> pd.DataFrame:
    return build_benchmark_table(
        prices,
        return_col=return_col,
        identity_col=identity_col,
    )


def run_equal_weight_long_only_backtest(
    prices: pd.DataFrame,
    signal_col: str = "signal",
    return_col: str = "daily_return",
    turnover_cost_bps: float | None = None,
    cost_scenario: str | None = "base",
    liquidity_tier_map: pd.Series | None = None,
    identity_col: str = "ticker_code",
) -> pd.DataFrame:
    required_columns = {"trade_date", identity_col, signal_col, return_col}
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prices are missing required backtest columns: {missing_list}")

    signal_matrix = (
        prices.pivot(index="trade_date", columns=identity_col, values=signal_col)
        .sort_index()
        .fillna(0.0)
    )
    return_matrix = (
        prices.pivot(index="trade_date", columns=identity_col, values=return_col)
        .sort_index()
        .fillna(0.0)
    )
    weights = signal_matrix.div(signal_matrix.sum(axis=1), axis=0).fillna(0.0)
    gross_returns = (weights.shift(1).fillna(0.0) * return_matrix).sum(axis=1)
    turnover_matrix = weights.diff().abs().fillna(0.0)
    turnover = turnover_matrix.sum(axis=1).fillna(0.0)
    resolved_liquidity_tier_map = (
        liquidity_tier_map
        if liquidity_tier_map is not None
        else build_liquidity_tier_map(prices, identity_col=identity_col)
    )
    tier_turnover = build_tier_turnover_frame(turnover_matrix, resolved_liquidity_tier_map)
    turnover_cost = calculate_turnover_costs_from_tier_turnover(
        tier_turnover,
        cost_scenario=cost_scenario,
        turnover_cost_bps=turnover_cost_bps,
    )
    ticker_cost_bps = build_ticker_cost_bps_map(
        turnover_matrix.columns,
        cost_scenario=cost_scenario,
        liquidity_tier_map=resolved_liquidity_tier_map,
        turnover_cost_bps=turnover_cost_bps,
    )
    effective_cost_bps = pd.Series(pd.NA, index=turnover.index, dtype="Float64")
    non_zero_turnover = turnover > 0
    effective_cost_bps.loc[non_zero_turnover] = (
        turnover_cost.loc[non_zero_turnover] / turnover.loc[non_zero_turnover] * 10_000.0
    )
    net_returns = gross_returns - turnover_cost

    results = pd.DataFrame(
        {
            "gross_return": gross_returns,
            "turnover": turnover,
            "turnover_cost": turnover_cost,
            "net_return": net_returns,
            "active_ticker_count": signal_matrix.sum(axis=1),
            "effective_cost_bps": effective_cost_bps.astype(float),
        }
    )
    for column_name in tier_turnover.columns:
        results[column_name] = tier_turnover[column_name]
    results["nav"] = compute_nav(results["net_return"])
    results.index.name = "trade_date"
    results.attrs["ticker_cost_bps"] = ticker_cost_bps.to_dict()
    results.attrs["liquidity_tier_map"] = pd.Series(resolved_liquidity_tier_map).to_dict()
    results.attrs["identity_col"] = identity_col
    return results


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
    return benchmark_series.sort_index().rename(benchmark_col)


def build_risk_free_returns_from_tri(tri_series: pd.Series) -> pd.Series:
    aligned_tri = tri_series.copy()
    aligned_tri.index = pd.to_datetime(aligned_tri.index)
    aligned_tri = aligned_tri.sort_index().ffill()
    risk_free_returns = aligned_tri.pct_change(fill_method=None).fillna(0.0)
    return risk_free_returns.rename("risk_free_return")
