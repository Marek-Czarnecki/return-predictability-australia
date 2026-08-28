from __future__ import annotations

import numpy as np
import pandas as pd


def compute_nav(return_series: pd.Series) -> pd.Series:
    return (1 + return_series.fillna(0.0)).cumprod()


def compute_max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    return (nav / nav.cummax() - 1).min()


def summarize_return_stream(
    strategy_name: str,
    net_returns: pd.Series,
    turnover: pd.Series | None = None,
    extra_fields: dict[str, object] | None = None,
    benchmark_returns: dict[str, pd.Series | pd.DataFrame] | None = None,
    risk_free_returns: pd.Series | pd.DataFrame | None = None,
    risk_free_col: str = "risk_free_return",
) -> pd.DataFrame:
    trading_days = 252
    aligned_returns = net_returns.copy()
    aligned_returns.index = pd.to_datetime(aligned_returns.index)
    years = max(len(aligned_returns), 1) / trading_days
    nav = compute_nav(aligned_returns)
    volatility = aligned_returns.std()
    legacy_sharpe_like = (
        (aligned_returns.mean() / volatility) * np.sqrt(trading_days)
        if pd.notna(volatility) and volatility != 0
        else np.nan
    )
    aligned_risk_free = None
    excess_returns = None
    excess_volatility = np.nan
    if risk_free_returns is not None:
        aligned_risk_free = _coerce_benchmark_series(
            risk_free_returns, risk_free_col
        ).reindex(aligned_returns.index, method="ffill")
        excess_returns = aligned_returns - aligned_risk_free.fillna(0.0)
        excess_volatility = excess_returns.std()
    summary = {
        "strategy": strategy_name,
        "annualized_return": aligned_returns.mean() * trading_days,
        "annualized_volatility": volatility * np.sqrt(trading_days),
        "sharpe_ratio": (
            (excess_returns.mean() / excess_volatility) * np.sqrt(trading_days)
            if excess_returns is not None
            and pd.notna(excess_volatility)
            and excess_volatility != 0
            else np.nan
        ),
        "cagr_like": nav.iloc[-1] ** (1 / years) - 1 if len(nav) else np.nan,
        "max_drawdown": compute_max_drawdown(nav),
        "average_turnover": turnover.mean() if turnover is not None else np.nan,
        "absolute_total_return": nav.iloc[-1] - 1 if len(nav) else np.nan,
        "absolute_return_label": "absolute_return",
        "risk_adjusted_performance_label": (
            "sharpe_ratio_excess_return_vs_rba_cash_rate_tri"
            if risk_free_returns is not None
            else "sharpe_ratio_not_computed_missing_risk_free_input"
        ),
        "legacy_sharpe_like": legacy_sharpe_like,
        "legacy_sharpe_like_label": "legacy_descriptive_without_risk_free_adjustment",
    }
    if risk_free_returns is not None:
        risk_free_nav = compute_nav(aligned_risk_free.fillna(0.0))
        excess_nav = compute_nav(excess_returns)
        summary["risk_free_total_return"] = (
            risk_free_nav.iloc[-1] - 1 if len(risk_free_nav) else np.nan
        )
        summary["excess_total_return_vs_risk_free"] = (
            excess_nav.iloc[-1] - 1 if len(excess_nav) else np.nan
        )
        summary["risk_free_benchmark_label"] = "rba_cash_rate_tri"
    if benchmark_returns:
        for benchmark_name, benchmark_input in benchmark_returns.items():
            benchmark_series = _coerce_benchmark_series(
                benchmark_input, f"{benchmark_name}_return"
            ).reindex(aligned_returns.index)
            benchmark_nav = compute_nav(benchmark_series)
            excess_returns = aligned_returns - benchmark_series.fillna(0.0)
            excess_nav = compute_nav(excess_returns)
            summary[f"benchmark_total_return_{benchmark_name}"] = (
                benchmark_nav.iloc[-1] - 1 if len(benchmark_nav) else np.nan
            )
            summary[f"net_excess_total_return_vs_{benchmark_name}"] = (
                excess_nav.iloc[-1] - 1 if len(excess_nav) else np.nan
            )
        summary["benchmark_relative_return_label"] = "net_excess_return_vs_benchmark"
    if extra_fields:
        summary.update(extra_fields)
    return pd.DataFrame([summary])


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
