from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import bootstrap_mean_confidence_interval, sign_flip_mean_p_value


TAX_LOSS_YEAR_ROBUSTNESS_COLUMNS = [
    "analysis_level",
    "focal_year",
    "omitted_year",
    "year_count",
    "matched_observation_count",
    "mean_return_difference",
    "median_return_difference",
    "total_return_difference",
    "share_of_total_effect",
    "p_value",
    "ci_lower_95",
    "ci_upper_95",
    "interpretation_note",
]


def window_return(series: pd.Series, center_date: pd.Timestamp, radius: int) -> float:
    if center_date not in series.index:
        return np.nan
    loc = series.index.get_loc(center_date)
    start = max(loc - radius, 0)
    end = min(loc + radius, len(series.index) - 1)
    window = series.iloc[start : end + 1].fillna(0.0)
    return (1 + window).prod() - 1


def _resolve_benchmark_series(
    benchmark_returns: pd.DataFrame | pd.Series | None,
) -> tuple[pd.Series | None, str | None]:
    if benchmark_returns is None:
        return None, None

    if isinstance(benchmark_returns, pd.Series):
        series = benchmark_returns.copy()
        series.index = pd.to_datetime(series.index)
        return series.sort_index(), series.name or "benchmark_return"

    if "trade_date" not in benchmark_returns.columns:
        raise ValueError("Benchmark returns must include a trade_date column.")

    value_columns = [
        column
        for column in (
            "benchmark_return",
            "stw_return",
            "a200_return",
            "equal_weight_return",
            "daily_return",
            "return",
        )
        if column in benchmark_returns.columns
    ]
    if not value_columns:
        raise ValueError(
            "Benchmark returns must include one of: "
            "benchmark_return, stw_return, a200_return, equal_weight_return, daily_return, return."
        )

    benchmark_col = value_columns[0]
    series = benchmark_returns.loc[:, ["trade_date", benchmark_col]].copy()
    series["trade_date"] = pd.to_datetime(series["trade_date"])
    benchmark_series = (
        series.dropna(subset=["trade_date"])
        .drop_duplicates(subset=["trade_date"], keep="last")
        .set_index("trade_date")[benchmark_col]
        .sort_index()
    )
    return benchmark_series, benchmark_col


def run_tax_loss_selling_event_study(
    prices: pd.DataFrame,
    benchmark_returns: pd.DataFrame | pd.Series | None = None,
    lookback_days: int = 252,
    window_radius: int = 10,
    control_shift_days: int = 60,
    selection_gap_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_prices = prices.copy()
    benchmark_series, benchmark_column = _resolve_benchmark_series(benchmark_returns)
    strategy_prices["trailing_12m_return"] = (
        strategy_prices.groupby("ticker_code", observed=True)["adj_close"]
        .transform(lambda series: series / series.shift(lookback_days) - 1)
    )
    calendar = pd.Index(sorted(strategy_prices["trade_date"].unique()))
    selection_offset_days = window_radius + selection_gap_days

    records = []
    for year in sorted(strategy_prices["trade_date"].dt.year.unique()):
        event_anchor = pd.Timestamp(year=year, month=6, day=30)
        eligible_dates = calendar[calendar <= event_anchor]
        if len(eligible_dates) == 0:
            continue
        event_date = eligible_dates[-1]
        event_loc = calendar.get_loc(event_date)
        if event_loc < selection_offset_days:
            continue

        selection_date = calendar[event_loc - selection_offset_days]
        control_loc = max(calendar.get_loc(event_date) - control_shift_days, 0)
        control_date = calendar[control_loc]

        snapshot = strategy_prices.loc[
            strategy_prices["trade_date"] == selection_date,
            ["ticker_code", "trailing_12m_return"],
        ].dropna()
        if snapshot.empty:
            continue

        cutoff = snapshot["trailing_12m_return"].quantile(0.10)
        selected_snapshot = snapshot.loc[snapshot["trailing_12m_return"] <= cutoff]

        for ticker_code, selection_return in selected_snapshot.itertuples(index=False):
            ticker_returns = (
                strategy_prices.loc[
                    strategy_prices["ticker_code"] == ticker_code,
                    ["trade_date", "daily_return"],
                ]
                .set_index("trade_date")
                .sort_index()["daily_return"]
            )
            event_window_return = window_return(ticker_returns, event_date, window_radius)
            control_window_return = window_return(
                ticker_returns, control_date, window_radius
            )
            benchmark_event_window_return = (
                window_return(benchmark_series, event_date, window_radius)
                if benchmark_series is not None
                else np.nan
            )
            benchmark_control_window_return = (
                window_return(benchmark_series, control_date, window_radius)
                if benchmark_series is not None
                else np.nan
            )
            abnormal_event_window_return = (
                event_window_return - benchmark_event_window_return
                if pd.notna(benchmark_event_window_return)
                else np.nan
            )
            abnormal_control_window_return = (
                control_window_return - benchmark_control_window_return
                if pd.notna(benchmark_control_window_return)
                else np.nan
            )
            records.append(
                {
                    "year": year,
                    "ticker_code": ticker_code,
                    "selection_date": selection_date,
                    "event_date": event_date,
                    "control_date": control_date,
                    "selection_trailing_12m_return": selection_return,
                    "event_window_return": event_window_return,
                    "control_window_return": control_window_return,
                    "return_difference": event_window_return - control_window_return,
                    "benchmark_event_window_return": benchmark_event_window_return,
                    "benchmark_control_window_return": benchmark_control_window_return,
                    "abnormal_event_window_return": abnormal_event_window_return,
                    "abnormal_control_window_return": abnormal_control_window_return,
                    "abnormal_return_difference": (
                        abnormal_event_window_return - abnormal_control_window_return
                        if pd.notna(abnormal_event_window_return)
                        and pd.notna(abnormal_control_window_return)
                        else np.nan
                    ),
                }
            )

    event_study_columns = [
        "year",
        "ticker_code",
        "selection_date",
        "event_date",
        "control_date",
        "selection_trailing_12m_return",
        "event_window_return",
        "control_window_return",
        "return_difference",
        "benchmark_event_window_return",
        "benchmark_control_window_return",
        "abnormal_event_window_return",
        "abnormal_control_window_return",
        "abnormal_return_difference",
    ]
    event_study = pd.DataFrame(records, columns=event_study_columns)
    summary = pd.DataFrame(
        [
            {
                "strategy": "tax_loss_selling_event_study",
                "lookback_days": lookback_days,
                "window_radius_days": window_radius,
                "control_shift_days": control_shift_days,
                "selection_gap_days": selection_gap_days,
                "selection_offset_days": selection_offset_days,
                "benchmark_column": benchmark_column,
                "benchmark_adjustment_applied": benchmark_series is not None,
                "matched_observation_count": len(event_study),
                "mean_event_window_return": event_study["event_window_return"].mean(),
                "mean_control_window_return": event_study[
                    "control_window_return"
                ].mean(),
                "mean_return_difference": event_study["return_difference"].mean(),
                "mean_benchmark_event_window_return": event_study[
                    "benchmark_event_window_return"
                ].mean(),
                "mean_benchmark_control_window_return": event_study[
                    "benchmark_control_window_return"
                ].mean(),
                "mean_abnormal_event_window_return": event_study[
                    "abnormal_event_window_return"
                ].mean(),
                "mean_abnormal_control_window_return": event_study[
                    "abnormal_control_window_return"
                ].mean(),
                "mean_abnormal_return_difference": event_study[
                    "abnormal_return_difference"
                ].mean(),
                "median_abnormal_return_difference": event_study[
                    "abnormal_return_difference"
                ].median(),
                "positive_abnormal_difference_fraction": event_study[
                    "abnormal_return_difference"
                ]
                .gt(0)
                .mean(),
                "median_return_difference": event_study["return_difference"].median(),
                "positive_difference_fraction": event_study["return_difference"]
                .gt(0)
                .mean(),
            }
        ]
    )
    return event_study, summary


def build_tax_loss_year_robustness(
    event_study: pd.DataFrame,
    year_col: str = "year",
    difference_col: str = "return_difference",
) -> pd.DataFrame:
    required_columns = {year_col, difference_col}
    missing_columns = required_columns.difference(event_study.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Tax-loss event study is missing required columns for year robustness: {missing_list}"
        )

    working = event_study.loc[:, [year_col, difference_col]].copy()
    working[year_col] = pd.to_numeric(working[year_col], errors="coerce")
    working[difference_col] = pd.to_numeric(working[difference_col], errors="coerce")
    working = working.dropna(subset=[year_col, difference_col]).copy()
    if working.empty:
        return pd.DataFrame(columns=TAX_LOSS_YEAR_ROBUSTNESS_COLUMNS)

    working[year_col] = working[year_col].astype(int)
    overall_mean = float(working[difference_col].mean())
    overall_median = float(working[difference_col].median())
    overall_sum = float(working[difference_col].sum())

    year_summary = (
        working.groupby(year_col, observed=True)[difference_col]
        .agg(["count", "mean", "median", "sum"])
        .reset_index()
        .rename(
            columns={
                year_col: "focal_year",
                "count": "matched_observation_count",
                "mean": "mean_return_difference",
                "median": "median_return_difference",
                "sum": "total_return_difference",
            }
        )
        .sort_values("focal_year")
        .reset_index(drop=True)
    )
    year_summary["year_count"] = len(year_summary)
    if abs(overall_sum) > 1e-15:
        year_summary["share_of_total_effect"] = (
            year_summary["total_return_difference"] / overall_sum
        )
    else:
        year_summary["share_of_total_effect"] = np.nan
    year_summary["analysis_level"] = "year_summary"
    year_summary["omitted_year"] = pd.NA
    year_summary["p_value"] = np.nan
    year_summary["ci_lower_95"] = np.nan
    year_summary["ci_upper_95"] = np.nan
    year_summary["interpretation_note"] = (
        "Per-year average event-minus-control return difference."
    )

    year_means = year_summary["mean_return_difference"].to_numpy(dtype=float)
    ci_lower, ci_upper = bootstrap_mean_confidence_interval(year_means)
    clustered_p_value = sign_flip_mean_p_value(year_means, alternative="greater")
    overall_rows = [
        {
            "analysis_level": "overall_event_mean",
            "focal_year": pd.NA,
            "omitted_year": pd.NA,
            "year_count": int(len(year_summary)),
            "matched_observation_count": int(len(working)),
            "mean_return_difference": overall_mean,
            "median_return_difference": overall_median,
            "total_return_difference": overall_sum,
            "share_of_total_effect": 1.0,
            "p_value": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "interpretation_note": (
                "All ticker-event observations pooled with equal weight per event."
            ),
        },
        {
            "analysis_level": "year_clustered_sign_flip",
            "focal_year": pd.NA,
            "omitted_year": pd.NA,
            "year_count": int(len(year_summary)),
            "matched_observation_count": int(len(year_summary)),
            "mean_return_difference": float(year_means.mean()),
            "median_return_difference": float(np.median(year_means)),
            "total_return_difference": float(year_means.sum()),
            "share_of_total_effect": np.nan,
            "p_value": clustered_p_value,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "interpretation_note": (
                "Exact year-level sign-flip inference on annual mean differences; "
                "each year receives equal weight."
            ),
        },
    ]

    leave_one_year_out_rows = []
    for focal_year in year_summary["focal_year"].tolist():
        leave_one_out = working.loc[working[year_col] != focal_year, difference_col]
        leave_one_year_out_rows.append(
            {
                "analysis_level": "leave_one_year_out",
                "focal_year": pd.NA,
                "omitted_year": int(focal_year),
                "year_count": int(len(year_summary) - 1),
                "matched_observation_count": int(len(leave_one_out)),
                "mean_return_difference": float(leave_one_out.mean()),
                "median_return_difference": float(leave_one_out.median()),
                "total_return_difference": float(leave_one_out.sum()),
                "share_of_total_effect": np.nan,
                "p_value": np.nan,
                "ci_lower_95": np.nan,
                "ci_upper_95": np.nan,
                "interpretation_note": (
                    "Pooled event-level mean after excluding one calendar year."
                ),
            }
        )

    robustness = pd.concat(
        [
            pd.DataFrame(overall_rows),
            year_summary,
            pd.DataFrame(leave_one_year_out_rows),
        ],
        ignore_index=True,
    )
    return robustness.loc[:, TAX_LOSS_YEAR_ROBUSTNESS_COLUMNS]
