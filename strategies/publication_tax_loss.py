from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import basis_points_to_rate, resolve_cost_scenario
from .publication_costs import build_publication_fold_liquidity_tiers
from .tax_loss_selling import build_tax_loss_year_robustness


TAX_LOSS_IDENTITY_COL = "asset_id"
TAX_LOSS_MEMBERSHIP_COL = "member_of_universe"
TAX_LOSS_LOOKBACK_DAYS = 252
TAX_LOSS_WINDOW_RADIUS = 10
TAX_LOSS_CONTROL_SHIFT_DAYS = 60
TAX_LOSS_SELECTION_GAP_DAYS = 1
TAX_LOSS_SELECTION_QUANTILE = 0.10
TAX_LOSS_COST_SCENARIO = "base"
TAX_LOSS_COST_APPLICATION = "symmetric_round_trip_cost_event_and_control"
TAX_LOSS_MISSING_WINDOW_POLICY = "require_complete_security_and_benchmark_windows"


@dataclass
class PublicationTaxLossResult:
    event_study: pd.DataFrame
    summary: pd.DataFrame
    year_robustness: pd.DataFrame
    liquidity_diagnostics: pd.DataFrame


def run_publication_tax_loss_event_study(
    prices: pd.DataFrame,
    benchmark_returns: pd.DataFrame | pd.Series,
    *,
    lookback_days: int = TAX_LOSS_LOOKBACK_DAYS,
    window_radius: int = TAX_LOSS_WINDOW_RADIUS,
    control_shift_days: int = TAX_LOSS_CONTROL_SHIFT_DAYS,
    selection_gap_days: int = TAX_LOSS_SELECTION_GAP_DAYS,
    selection_quantile: float = TAX_LOSS_SELECTION_QUANTILE,
    cost_scenario: str = TAX_LOSS_COST_SCENARIO,
    max_years: int | None = None,
) -> PublicationTaxLossResult:
    required = {
        "asset_id",
        "ticker_code",
        "trade_date",
        "adj_close",
        "daily_return",
        "dollar_volume",
        "member_of_universe",
    }
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(
            "Publication tax-loss panel is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if not 0 < selection_quantile < 1:
        raise ValueError("selection_quantile must be between zero and one.")
    if max_years is not None and max_years < 1:
        raise ValueError("max_years must be at least 1 when supplied.")

    frame = prices.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values(["asset_id", "trade_date"]).reset_index(drop=True)
    frame["trailing_12m_return"] = (
        frame.groupby("asset_id", observed=True)["adj_close"]
        .transform(lambda series: series / series.shift(lookback_days) - 1.0)
    )

    benchmark = _coerce_benchmark_series(benchmark_returns)
    calendar = pd.DatetimeIndex(sorted(frame["trade_date"].dropna().unique()))
    selection_offset = window_radius + selection_gap_days
    scenario = resolve_cost_scenario(cost_scenario)

    records: list[dict[str, object]] = []
    liquidity_frames: list[pd.DataFrame] = []
    years = sorted(frame["trade_date"].dt.year.unique())
    if max_years is not None:
        years = years[:max_years]

    for year in years:
        event_anchor = pd.Timestamp(year=int(year), month=6, day=30)
        eligible_dates = calendar[calendar <= event_anchor]
        if len(eligible_dates) == 0:
            continue
        event_date = eligible_dates[-1]
        event_loc = calendar.get_loc(event_date)
        control_loc = event_loc - control_shift_days
        selection_loc = event_loc - selection_offset
        if (
            selection_loc < 0
            or control_loc - window_radius < 0
            or event_loc + window_radius >= len(calendar)
        ):
            continue

        selection_date = calendar[selection_loc]
        control_date = calendar[control_loc]
        event_window_dates = calendar[event_loc - window_radius : event_loc + window_radius + 1]
        control_window_dates = calendar[
            control_loc - window_radius : control_loc + window_radius + 1
        ]

        snapshot = frame.loc[
            (frame["trade_date"] == selection_date)
            & frame["member_of_universe"].fillna(False).astype(bool),
            ["asset_id", "ticker_code", "trailing_12m_return"],
        ].dropna(subset=["asset_id", "trailing_12m_return"])
        if snapshot.empty:
            continue

        cutoff = snapshot["trailing_12m_return"].quantile(selection_quantile)
        selected = snapshot.loc[snapshot["trailing_12m_return"] <= cutoff].copy()
        if selected.empty:
            continue

        formation_start = selection_date - pd.DateOffset(years=3)
        liquidity = build_publication_fold_liquidity_tiers(
            prices=frame,
            formation_start=formation_start,
            formation_end=selection_date,
            evaluation_identities=selected["asset_id"],
            identity_col="asset_id",
        )
        diagnostics = liquidity.diagnostics.copy()
        diagnostics.insert(0, "year", int(year))
        diagnostics.insert(1, "selection_date", selection_date)
        liquidity_frames.append(diagnostics)

        benchmark_event = _strict_window_return(benchmark, event_window_dates)
        benchmark_control = _strict_window_return(benchmark, control_window_dates)

        for row in selected.itertuples(index=False):
            asset_id = row.asset_id
            ticker_code = row.ticker_code
            asset_returns = (
                frame.loc[frame["asset_id"] == asset_id, ["trade_date", "daily_return"]]
                .drop_duplicates("trade_date", keep="last")
                .set_index("trade_date")["daily_return"]
                .sort_index()
            )
            event_return = _strict_window_return(asset_returns, event_window_dates)
            control_return = _strict_window_return(asset_returns, control_window_dates)
            tier = str(liquidity.tier_map.loc[asset_id])
            cost_bps = scenario.cost_bps_for_tier(tier)
            round_trip_cost = scenario.tax_loss_trade_legs * basis_points_to_rate(cost_bps)

            net_event = event_return - round_trip_cost if pd.notna(event_return) else np.nan
            net_control = control_return - round_trip_cost if pd.notna(control_return) else np.nan
            abnormal_event = (
                net_event - benchmark_event
                if pd.notna(net_event) and pd.notna(benchmark_event)
                else np.nan
            )
            abnormal_control = (
                net_control - benchmark_control
                if pd.notna(net_control) and pd.notna(benchmark_control)
                else np.nan
            )

            records.append(
                {
                    "year": int(year),
                    "asset_id": asset_id,
                    "ticker_code": ticker_code,
                    "selection_date": selection_date,
                    "event_date": event_date,
                    "control_date": control_date,
                    "selection_trailing_12m_return": float(row.trailing_12m_return),
                    "selection_cutoff": float(cutoff),
                    "liquidity_tier": tier,
                    "turnover_cost_bps_per_leg": float(cost_bps),
                    "round_trip_cost_rate": float(round_trip_cost),
                    "event_window_return": event_return,
                    "control_window_return": control_return,
                    "return_difference": (
                        event_return - control_return
                        if pd.notna(event_return) and pd.notna(control_return)
                        else np.nan
                    ),
                    "net_event_window_return": net_event,
                    "net_control_window_return": net_control,
                    "net_return_difference": (
                        net_event - net_control
                        if pd.notna(net_event) and pd.notna(net_control)
                        else np.nan
                    ),
                    "benchmark_event_window_return": benchmark_event,
                    "benchmark_control_window_return": benchmark_control,
                    "abnormal_net_event_window_return": abnormal_event,
                    "abnormal_net_control_window_return": abnormal_control,
                    "abnormal_net_return_difference": (
                        abnormal_event - abnormal_control
                        if pd.notna(abnormal_event) and pd.notna(abnormal_control)
                        else np.nan
                    ),
                    "complete_event_window": bool(pd.notna(event_return) and pd.notna(benchmark_event)),
                    "complete_control_window": bool(pd.notna(control_return) and pd.notna(benchmark_control)),
                }
            )

    event_study = pd.DataFrame(records)
    complete = event_study.dropna(subset=["net_return_difference"]).copy() if not event_study.empty else event_study.copy()
    robustness = build_tax_loss_year_robustness(
        complete,
        year_col="year",
        difference_col="net_return_difference",
    ) if not complete.empty else pd.DataFrame()
    year_count = int(complete["year"].nunique()) if not complete.empty else 0
    summary = pd.DataFrame(
        [
            {
                "strategy": "publication_tax_loss_selling_event_study",
                "identity_col": TAX_LOSS_IDENTITY_COL,
                "membership_col": TAX_LOSS_MEMBERSHIP_COL,
                "lookback_days": lookback_days,
                "window_radius_days": window_radius,
                "control_shift_days": control_shift_days,
                "selection_gap_days": selection_gap_days,
                "selection_quantile": selection_quantile,
                "cost_scenario": scenario.name,
                "cost_application": TAX_LOSS_COST_APPLICATION,
                "missing_window_policy": TAX_LOSS_MISSING_WINDOW_POLICY,
                "event_observation_count": int(len(event_study)),
                "complete_matched_observation_count": int(len(complete)),
                "year_count": year_count,
                "mean_event_window_return": complete["event_window_return"].mean() if not complete.empty else np.nan,
                "mean_control_window_return": complete["control_window_return"].mean() if not complete.empty else np.nan,
                "mean_return_difference": complete["return_difference"].mean() if not complete.empty else np.nan,
                "mean_net_event_window_return": complete["net_event_window_return"].mean() if not complete.empty else np.nan,
                "mean_net_control_window_return": complete["net_control_window_return"].mean() if not complete.empty else np.nan,
                "mean_net_return_difference": complete["net_return_difference"].mean() if not complete.empty else np.nan,
                "mean_abnormal_net_event_window_return": complete["abnormal_net_event_window_return"].mean() if not complete.empty else np.nan,
                "mean_abnormal_net_control_window_return": complete["abnormal_net_control_window_return"].mean() if not complete.empty else np.nan,
                "mean_abnormal_net_return_difference": complete["abnormal_net_return_difference"].mean() if not complete.empty else np.nan,
                "positive_net_difference_fraction": complete["net_return_difference"].gt(0).mean() if not complete.empty else np.nan,
            }
        ]
    )
    liquidity_diagnostics = (
        pd.concat(liquidity_frames, ignore_index=True) if liquidity_frames else pd.DataFrame()
    )
    return PublicationTaxLossResult(
        event_study=event_study,
        summary=summary,
        year_robustness=robustness,
        liquidity_diagnostics=liquidity_diagnostics,
    )


def _coerce_benchmark_series(benchmark_returns: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(benchmark_returns, pd.Series):
        series = benchmark_returns.copy()
    else:
        required = {"trade_date", "benchmark_return"}
        missing = required.difference(benchmark_returns.columns)
        if missing:
            raise ValueError("Benchmark returns are missing required columns: " + ", ".join(sorted(missing)))
        series = benchmark_returns.loc[:, ["trade_date", "benchmark_return"]].copy()
        series["trade_date"] = pd.to_datetime(series["trade_date"])
        if series["trade_date"].duplicated().any():
            raise ValueError("Benchmark returns contain duplicate trade dates.")
        series = series.set_index("trade_date")["benchmark_return"]
    series.index = pd.to_datetime(series.index)
    if series.index.duplicated().any():
        raise ValueError("Benchmark return series contains duplicate trade dates.")
    return pd.to_numeric(series.sort_index(), errors="coerce")


def _strict_window_return(series: pd.Series, window_dates: pd.DatetimeIndex) -> float:
    aligned = pd.to_numeric(series.reindex(window_dates), errors="coerce")
    if len(aligned) != len(window_dates) or aligned.isna().any():
        return np.nan
    return float((1.0 + aligned).prod() - 1.0)
