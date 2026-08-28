from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .costs import (
    DEFAULT_COST_SCENARIO_NAME,
    apply_turnover_costs,
    build_liquidity_tier_map,
    resolve_cost_scenario,
    summarize_cost_model_inputs,
)
from .metrics import compute_nav, summarize_return_stream
from .sector_liquidity import load_sector_map


DEFAULT_SECTOR_MAP_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "asx_ticker_sector_map.csv"
)


@dataclass
class PairsTradingRun:
    pair_table: pd.DataFrame
    selected_pairs: pd.DataFrame
    pair_return_panel: pd.DataFrame
    daily_results: pd.DataFrame
    summary: pd.DataFrame


def run_pairs_trading_strategy(
    prices: pd.DataFrame,
    top_liquid_tickers: int = 20,
    top_pair_count: int = 5,
    spread_window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    turnover_cost_bps: float | None = None,
    cost_scenario: str = DEFAULT_COST_SCENARIO_NAME,
    formation_prices: pd.DataFrame | None = None,
    sector_map: pd.DataFrame | None = None,
    sector_map_path: Path | None = None,
    engle_granger_pvalue_threshold: float = 0.05,
    selected_pairs: pd.DataFrame | None = None,
    benchmark_returns: dict[str, pd.Series | pd.DataFrame] | None = None,
    risk_free_returns: pd.Series | pd.DataFrame | None = None,
    identity_col: str = "ticker_code",
    eligibility_col: str | None = None,
    label_col: str = "ticker_code",
) -> PairsTradingRun:
    trading_prices = prices.copy()
    formation_panel = (
        formation_prices.copy() if formation_prices is not None else trading_prices.copy()
    )

    liquidity_tier_map = build_liquidity_tier_map(
        formation_panel,
        identity_col=identity_col,
    )
    resolved_sector_map = _resolve_sector_map(sector_map, sector_map_path)
    if selected_pairs is None:
        pair_table, selected_pairs = select_pairs_in_window(
            formation_panel,
            top_liquid_tickers=top_liquid_tickers,
            top_pair_count=top_pair_count,
            sector_map=resolved_sector_map,
            engle_granger_pvalue_threshold=engle_granger_pvalue_threshold,
            liquidity_tier_map=liquidity_tier_map,
            identity_col=identity_col,
            eligibility_col=eligibility_col,
            label_col=label_col,
        )
    else:
        pair_table = selected_pairs.copy()
        selected_pairs = _coerce_selected_pairs(selected_pairs)

    pair_return_panel = build_pair_return_panel(
        trading_prices=trading_prices,
        selected_pairs=selected_pairs,
        spread_window=spread_window,
        entry_z=entry_z,
        exit_z=exit_z,
        turnover_cost_bps=turnover_cost_bps,
        cost_scenario=cost_scenario,
        identity_col=identity_col,
        eligibility_col=eligibility_col,
    )

    strategy_daily = _build_strategy_daily(pair_return_panel)
    summary = summarize_return_stream(
        "pairs_trading_cointegration_zscore",
        strategy_daily["net_return"],
        turnover=strategy_daily["average_turnover"],
        benchmark_returns=benchmark_returns,
        risk_free_returns=risk_free_returns,
        extra_fields={
            "top_liquid_tickers": top_liquid_tickers,
            "top_pair_count": top_pair_count,
            "spread_window": spread_window,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "formation_method": "sector_first_engle_granger_with_fit_ranking",
            "pair_cost_application": "two_leg_average_tier_cost",
            "identity_col": identity_col,
            "eligibility_col": eligibility_col or "none",
            **summarize_cost_model_inputs(
                cost_scenario=cost_scenario,
                turnover_cost_bps=turnover_cost_bps,
            ),
        },
    )
    return PairsTradingRun(
        pair_table=pair_table,
        selected_pairs=selected_pairs,
        pair_return_panel=pair_return_panel,
        daily_results=strategy_daily,
        summary=summary,
    )


def select_pairs_in_window(
    prices: pd.DataFrame,
    top_liquid_tickers: int,
    top_pair_count: int,
    sector_map: pd.DataFrame | None = None,
    sector_map_path: Path | None = None,
    engle_granger_pvalue_threshold: float = 0.05,
    liquidity_tier_map: pd.Series | None = None,
    identity_col: str = "ticker_code",
    eligibility_col: str | None = None,
    label_col: str = "ticker_code",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {identity_col, "trade_date", "dollar_volume", "adj_close"}
    if eligibility_col is not None:
        required_columns.add(eligibility_col)
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Pair formation panel is missing required columns: {missing_list}")

    resolved_sector_map = _resolve_sector_map(sector_map, sector_map_path)
    resolved_liquidity_tier_map = (
        liquidity_tier_map
        if liquidity_tier_map is not None
        else build_liquidity_tier_map(prices, identity_col=identity_col)
    )

    candidate_source = prices.copy()
    if eligibility_col is not None:
        latest_rows = (
            candidate_source.sort_values([identity_col, "trade_date"])
            .groupby(identity_col, observed=True)
            .tail(1)
        )
        eligible_identities = latest_rows.loc[
            latest_rows[eligibility_col].astype(bool), identity_col
        ]
        candidate_source = candidate_source.loc[
            candidate_source[identity_col].isin(eligible_identities)
        ].copy()

    liquidity_ranking = (
        candidate_source.groupby(identity_col, observed=True)["dollar_volume"]
        .median()
        .sort_values(ascending=False)
    )
    candidate_identities = liquidity_ranking.head(top_liquid_tickers).index.tolist()
    price_matrix = (
        prices.loc[prices[identity_col].isin(candidate_identities)]
        .pivot(index="trade_date", columns=identity_col, values="adj_close")
        .sort_index()
    )
    identity_label_map = _build_identity_label_map(
        prices,
        candidate_identities,
        identity_col=identity_col,
        label_col=label_col,
    )
    sector_lookup = _build_sector_lookup(
        candidate_identities,
        resolved_sector_map,
        identity_label_map=identity_label_map,
        identity_col=identity_col,
        label_col=label_col,
    )
    liquidity_lookup = pd.Series(resolved_liquidity_tier_map).reindex(
        candidate_identities
    ).to_dict()
    candidate_pairs = _generate_candidate_pairs(
        candidate_identities,
        sector_lookup,
        liquidity_lookup,
        identity_label_map,
        top_pair_count=top_pair_count,
    )

    pair_rows = []
    for pair in candidate_pairs:
        metrics = _fit_pair(
            price_matrix,
            pair["left_identity"],
            pair["right_identity"],
        )
        if metrics is None:
            continue
        pair_rows.append(
            {
                **pair,
                **metrics,
                "passed_cointegration": metrics["cointegration_pvalue"]
                <= engle_granger_pvalue_threshold,
            }
        )

    pair_table = pd.DataFrame(pair_rows)
    if pair_table.empty:
        return pair_table, pair_table.copy()

    pair_table = pair_table.sort_values(
        ["passed_cointegration", "fit_statistic", "cointegration_pvalue"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    selected_pairs = (
        pair_table.loc[pair_table["passed_cointegration"]]
        .head(top_pair_count)
        .reset_index(drop=True)
    )
    return pair_table, selected_pairs


def build_pair_return_panel(
    trading_prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    spread_window: int,
    entry_z: float,
    exit_z: float,
    turnover_cost_bps: float | None,
    cost_scenario: str,
    identity_col: str = "ticker_code",
    eligibility_col: str | None = None,
) -> pd.DataFrame:
    if selected_pairs.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "pair_id",
                "left_identity",
                "right_identity",
                "left_ticker",
                "right_ticker",
                "pair_gross_return",
                "pair_net_return",
                "pair_turnover",
                "spread",
                "spread_z",
                "pair_position",
                "pair_short_exposure",
            ]
        )

    required_columns = {identity_col, "trade_date", "adj_close"}
    if eligibility_col is not None:
        required_columns.add(eligibility_col)
    missing_columns = required_columns.difference(trading_prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Pair trading panel is missing required columns: {missing_list}")

    selected_pairs = _coerce_selected_pairs(selected_pairs)
    price_matrix = (
        trading_prices.pivot(index="trade_date", columns=identity_col, values="adj_close")
        .sort_index()
    )
    return_matrix = price_matrix.pct_change(fill_method=None).fillna(0.0)
    eligibility_matrix = None
    if eligibility_col is not None:
        eligibility_matrix = (
            trading_prices.pivot(
                index="trade_date",
                columns=identity_col,
                values=eligibility_col,
            )
            .sort_index()
            .fillna(False)
            .astype(bool)
        )
    resolved_cost_scenario = resolve_cost_scenario(cost_scenario)

    pair_returns = []
    for _, pair in selected_pairs.iterrows():
        left_identity = pair["left_identity"]
        right_identity = pair["right_identity"]
        left_series = price_matrix[left_identity]
        right_series = price_matrix[right_identity]
        spread = _compute_log_spread(
            left_series,
            right_series,
            hedge_ratio=float(pair["hedge_ratio"]),
            intercept=float(pair["intercept"]),
        )
        spread_mean = spread.rolling(spread_window).mean()
        spread_std = spread.rolling(spread_window).std().replace(0.0, np.nan)
        spread_z = ((spread - spread_mean) / spread_std).replace([np.inf, -np.inf], np.nan)

        pair_position = _build_pair_position(spread_z, entry_z=entry_z, exit_z=exit_z)
        if eligibility_matrix is not None:
            pair_eligible = (
                eligibility_matrix[left_identity]
                & eligibility_matrix[right_identity]
            ).reindex(pair_position.index).fillna(False)
            pair_position = pair_position.where(pair_eligible, 0.0)

        left_return = return_matrix[left_identity]
        right_return = return_matrix[right_identity]
        gross_pair_return = pair_position.shift(1).fillna(0.0) * (
            left_return - float(pair["hedge_ratio"]) * right_return
        )
        normalized_right_weight = abs(float(pair["hedge_ratio"])) / (
            1.0 + abs(float(pair["hedge_ratio"]))
        )
        normalized_left_weight = 1.0 / (1.0 + abs(float(pair["hedge_ratio"])))
        pair_short_exposure = pd.Series(0.0, index=pair_position.index, dtype=float)
        pair_short_exposure.loc[pair_position > 0] = normalized_right_weight
        pair_short_exposure.loc[pair_position < 0] = normalized_left_weight
        pair_turnover = pair_position.diff().abs().fillna(0.0)
        if turnover_cost_bps is not None:
            pair_cost_bps = float(turnover_cost_bps)
        else:
            pair_cost_bps = (
                resolved_cost_scenario.cost_bps_for_tier(str(pair["left_liquidity_tier"]))
                + resolved_cost_scenario.cost_bps_for_tier(str(pair["right_liquidity_tier"]))
            )
        pair_net_return = apply_turnover_costs(
            gross_pair_return,
            pair_turnover,
            turnover_cost_bps=pair_cost_bps,
        )
        pair_returns.append(
            pd.DataFrame(
                {
                    "trade_date": pair_net_return.index,
                    "pair_id": pair["pair_id"],
                    "left_identity": left_identity,
                    "right_identity": right_identity,
                    "left_ticker": pair["left_ticker"],
                    "right_ticker": pair["right_ticker"],
                    "pair_gross_return": gross_pair_return.values,
                    "pair_net_return": pair_net_return.values,
                    "pair_turnover": pair_turnover.values,
                    "pair_turnover_cost_bps": pair_cost_bps,
                    "left_liquidity_tier": pair["left_liquidity_tier"],
                    "right_liquidity_tier": pair["right_liquidity_tier"],
                    "spread": spread.values,
                    "spread_z": spread_z.values,
                    "pair_position": pair_position.values,
                    "pair_short_exposure": pair_short_exposure.values,
                }
            )
        )

    return pd.concat(pair_returns, ignore_index=True)


def _resolve_sector_map(
    sector_map: pd.DataFrame | None, sector_map_path: Path | None
) -> pd.DataFrame:
    if sector_map is not None:
        return sector_map.copy()
    resolved_path = sector_map_path or DEFAULT_SECTOR_MAP_PATH
    return load_sector_map(resolved_path)


def _build_identity_label_map(
    prices: pd.DataFrame,
    candidate_identities: list[object],
    identity_col: str,
    label_col: str,
) -> dict[object, str]:
    if label_col not in prices.columns:
        return {identity: str(identity) for identity in candidate_identities}
    labels = (
        prices.loc[prices[identity_col].isin(candidate_identities)]
        .sort_values([identity_col, "trade_date"])
        .dropna(subset=[label_col])
        .groupby(identity_col, observed=True)[label_col]
        .last()
    )
    return {
        identity: str(labels.get(identity, identity))
        for identity in candidate_identities
    }


def _build_sector_lookup(
    candidate_identities: list[object],
    sector_map: pd.DataFrame,
    identity_label_map: dict[object, str],
    identity_col: str,
    label_col: str,
) -> dict[object, str]:
    working_map = sector_map.copy()
    working_map["sector"] = working_map["sector"].astype("string").str.strip()

    if identity_col in working_map.columns:
        lookup = working_map.drop_duplicates(identity_col).set_index(identity_col)["sector"]
        return {
            identity: str(lookup.get(identity, "Unmapped"))
            if pd.notna(lookup.get(identity))
            else "Unmapped"
            for identity in candidate_identities
        }

    if label_col in working_map.columns:
        working_map[label_col] = working_map[label_col].astype("string").str.strip()
        lookup = working_map.drop_duplicates(label_col).set_index(label_col)["sector"]
        return {
            identity: str(lookup.get(identity_label_map[identity], "Unmapped"))
            if pd.notna(lookup.get(identity_label_map[identity]))
            else "Unmapped"
            for identity in candidate_identities
        }

    return {identity: "Unmapped" for identity in candidate_identities}


def _generate_candidate_pairs(
    candidate_identities: list[object],
    sector_lookup: dict[object, str],
    liquidity_lookup: dict[object, str],
    identity_label_map: dict[object, str],
    top_pair_count: int,
) -> list[dict[str, object]]:
    same_sector_pairs: list[dict[str, object]] = []
    fallback_pairs: list[dict[str, object]] = []
    for left_identity, right_identity in combinations(candidate_identities, 2):
        left_sector = sector_lookup.get(left_identity, "Unmapped")
        right_sector = sector_lookup.get(right_identity, "Unmapped")
        pair = {
            "pair_id": f"{left_identity}_{right_identity}",
            "left_identity": left_identity,
            "right_identity": right_identity,
            "left_ticker": identity_label_map.get(left_identity, str(left_identity)),
            "right_ticker": identity_label_map.get(right_identity, str(right_identity)),
            "left_sector": left_sector,
            "right_sector": right_sector,
            "left_liquidity_tier": liquidity_lookup.get(left_identity, "lower"),
            "right_liquidity_tier": liquidity_lookup.get(right_identity, "lower"),
        }
        if left_sector == right_sector and left_sector != "Unmapped":
            same_sector_pairs.append({**pair, "candidate_source": "same_sector"})
        else:
            fallback_pairs.append({**pair, "candidate_source": "sector_fallback"})

    if len(same_sector_pairs) >= top_pair_count:
        return same_sector_pairs
    return same_sector_pairs + fallback_pairs


def _fit_pair(
    price_matrix: pd.DataFrame, left_identity: object, right_identity: object
) -> dict[str, float] | None:
    pair_frame = price_matrix[[left_identity, right_identity]].dropna()
    if len(pair_frame) < 10:
        return None

    left_log = np.log(pair_frame[left_identity].astype(float))
    right_log = np.log(pair_frame[right_identity].astype(float))
    intercept, hedge_ratio = _estimate_hedge_ratio(left_log, right_log)
    spread = left_log - (intercept + hedge_ratio * right_log)
    spread_volatility = float(spread.std(ddof=1))
    if spread.isna().all():
        return None
    if np.isclose(spread_volatility, 0.0):
        fit_statistic, cointegration_pvalue = float("-inf"), 0.0
    else:
        fit_statistic, cointegration_pvalue = _residual_stationarity_test(spread)
    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "fit_statistic": fit_statistic,
        "cointegration_pvalue": cointegration_pvalue,
        "spread_volatility": spread_volatility,
        "formation_observations": float(len(pair_frame)),
    }


def _estimate_hedge_ratio(
    left_log: pd.Series, right_log: pd.Series
) -> tuple[float, float]:
    x = right_log.to_numpy(dtype=float)
    y = left_log.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coefficients[0])
    hedge_ratio = float(coefficients[1])
    return intercept, hedge_ratio


def _residual_stationarity_test(spread: pd.Series) -> tuple[float, float]:
    residuals = spread.dropna().to_numpy(dtype=float)
    lagged = residuals[:-1]
    delta = np.diff(residuals)
    if len(delta) < 3:
        return np.nan, np.nan

    denominator = float(np.dot(lagged, lagged))
    if denominator == 0.0:
        return np.nan, np.nan

    phi_hat = float(np.dot(lagged, delta) / denominator)
    fitted = phi_hat * lagged
    errors = delta - fitted
    dof = len(delta) - 1
    if dof <= 0:
        return np.nan, np.nan

    sigma2 = float(np.dot(errors, errors) / dof)
    standard_error = np.sqrt(sigma2 / denominator)
    if standard_error == 0.0:
        return np.nan, np.nan

    t_stat = phi_hat / standard_error
    pvalue = float(stats.t.cdf(t_stat, df=dof))
    return float(t_stat), pvalue


def _compute_log_spread(
    left_series: pd.Series,
    right_series: pd.Series,
    hedge_ratio: float,
    intercept: float,
) -> pd.Series:
    return np.log(left_series.astype(float)) - (
        intercept + hedge_ratio * np.log(right_series.astype(float))
    )


def _build_pair_position(
    spread_z: pd.Series, entry_z: float, exit_z: float
) -> pd.Series:
    long_left_signal = spread_z.le(-entry_z).astype(float)
    short_left_signal = spread_z.ge(entry_z).astype(float)
    flat_signal = spread_z.abs().le(exit_z).astype(float)

    pair_position = pd.Series(np.nan, index=spread_z.index, dtype=float)
    pair_position.loc[long_left_signal.eq(1.0)] = 1.0
    pair_position.loc[short_left_signal.eq(1.0)] = -1.0
    pair_position.loc[flat_signal.eq(1.0)] = 0.0
    return pair_position.ffill().fillna(0.0)


def _build_strategy_daily(pair_return_panel: pd.DataFrame) -> pd.DataFrame:
    if pair_return_panel.empty:
        return pd.DataFrame(
            columns=[
                "gross_return",
                "net_return",
                "turnover",
                "average_turnover",
                "short_exposure",
                "effective_cost_bps",
                "nav",
            ]
        )

    strategy_daily = (
        pair_return_panel.groupby("trade_date", observed=True)
        .agg(
            gross_return=("pair_gross_return", "mean"),
            net_return=("pair_net_return", "mean"),
            turnover=("pair_turnover", "mean"),
            average_turnover=("pair_turnover", "mean"),
            short_exposure=("pair_short_exposure", "mean"),
            effective_cost_bps=("pair_turnover_cost_bps", "mean"),
        )
        .sort_index()
    )
    strategy_daily["nav"] = compute_nav(strategy_daily["net_return"])
    return strategy_daily


def _coerce_selected_pairs(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "pair_id",
        "left_ticker",
        "right_ticker",
        "hedge_ratio",
        "intercept",
    }
    missing_columns = required_columns.difference(selected_pairs.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Selected pairs are missing required columns: {missing_list}")
    coerced = selected_pairs.copy().reset_index(drop=True)
    if "left_identity" not in coerced.columns:
        coerced["left_identity"] = coerced["left_ticker"]
    if "right_identity" not in coerced.columns:
        coerced["right_identity"] = coerced["right_ticker"]
    for column_name, default_value in [
        ("left_liquidity_tier", "lower"),
        ("right_liquidity_tier", "lower"),
    ]:
        if column_name not in coerced.columns:
            coerced[column_name] = default_value
    return coerced
