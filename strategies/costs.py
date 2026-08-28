from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


LIQUIDITY_TIER_ORDER = ("high", "medium", "lower")
DEFAULT_HIGH_LIQUIDITY_CUTOFF = 0.30
DEFAULT_MEDIUM_LIQUIDITY_CUTOFF = 0.70


@dataclass(frozen=True)
class CostScenarioDefinition:
    name: str
    turnover_cost_bps_by_tier: dict[str, float]
    tax_loss_trade_legs: int = 2
    tax_loss_adjust_control_leg: bool = True
    borrow_costs_modeled: bool = False
    financing_costs_modeled: bool = False
    exclusions_disclosure: str = (
        "turnover_costs_only_liquidity_tiered_borrow_and_financing_excluded"
    )

    def cost_bps_for_tier(self, liquidity_tier: str) -> float:
        tier_key = str(liquidity_tier).strip().lower()
        if tier_key not in self.turnover_cost_bps_by_tier:
            raise KeyError(f"Unsupported liquidity_tier: {liquidity_tier}")
        return float(self.turnover_cost_bps_by_tier[tier_key])


DEFAULT_COST_SCENARIOS: dict[str, CostScenarioDefinition] = {
    "low": CostScenarioDefinition(
        name="low",
        turnover_cost_bps_by_tier={"high": 5.0, "medium": 10.0, "lower": 20.0},
    ),
    "base": CostScenarioDefinition(
        name="base",
        turnover_cost_bps_by_tier={"high": 10.0, "medium": 20.0, "lower": 35.0},
    ),
    "high": CostScenarioDefinition(
        name="high",
        turnover_cost_bps_by_tier={"high": 20.0, "medium": 35.0, "lower": 50.0},
    ),
}
DEFAULT_COST_SCENARIO_NAME = "base"


def basis_points_to_rate(basis_points: float) -> float:
    return float(basis_points) / 10_000.0


def resolve_cost_scenario(
    cost_scenario: str | CostScenarioDefinition | None = None,
) -> CostScenarioDefinition:
    resolved = cost_scenario or DEFAULT_COST_SCENARIO_NAME
    if isinstance(resolved, CostScenarioDefinition):
        return resolved
    if resolved not in DEFAULT_COST_SCENARIOS:
        raise KeyError(f"Unknown cost_scenario: {resolved}")
    return DEFAULT_COST_SCENARIOS[resolved]


def build_liquidity_tier_map(
    prices: pd.DataFrame,
    high_liquidity_cutoff: float = DEFAULT_HIGH_LIQUIDITY_CUTOFF,
    medium_liquidity_cutoff: float = DEFAULT_MEDIUM_LIQUIDITY_CUTOFF,
    identity_col: str = "ticker_code",
) -> pd.Series:
    if identity_col not in prices.columns or "dollar_volume" not in prices.columns:
        raise ValueError(
            f"Prices must include '{identity_col}' and 'dollar_volume' to derive liquidity tiers."
        )
    if not 0 < high_liquidity_cutoff < medium_liquidity_cutoff <= 1:
        raise ValueError("Liquidity cutoffs must satisfy 0 < high < medium <= 1.")

    liquidity = (
        prices.groupby(identity_col, observed=True)["dollar_volume"]
        .median()
        .sort_values(ascending=False)
        .rename("median_dollar_volume")
        .to_frame()
    )
    liquidity["liquidity_rank"] = np.arange(1, len(liquidity) + 1)
    liquidity["liquidity_percentile"] = liquidity["liquidity_rank"] / len(liquidity)
    liquidity["liquidity_tier"] = np.select(
        [
            liquidity["liquidity_percentile"] <= high_liquidity_cutoff,
            liquidity["liquidity_percentile"] <= medium_liquidity_cutoff,
        ],
        ["high", "medium"],
        default="lower",
    )
    return liquidity["liquidity_tier"].astype("string")


def build_ticker_cost_bps_map(
    tickers: pd.Index | list[object],
    cost_scenario: str | CostScenarioDefinition | None = None,
    liquidity_tier_map: pd.Series | Mapping[object, str] | None = None,
    turnover_cost_bps: float | None = None,
) -> pd.Series:
    ticker_index = pd.Index(tickers)
    if turnover_cost_bps is not None:
        return pd.Series(float(turnover_cost_bps), index=ticker_index, dtype=float)

    scenario = resolve_cost_scenario(cost_scenario)
    if liquidity_tier_map is None:
        raise ValueError(
            "liquidity_tier_map is required when turnover_cost_bps is not provided."
        )
    tier_series = pd.Series(liquidity_tier_map, dtype="string").reindex(ticker_index)
    if tier_series.isna().any():
        missing_tickers = ", ".join(tier_series.index[tier_series.isna()].astype(str))
        raise ValueError(f"Missing liquidity tiers for identities: {missing_tickers}")
    return tier_series.map(scenario.cost_bps_for_tier).astype(float)


def build_tier_turnover_frame(
    turnover_matrix: pd.DataFrame,
    liquidity_tier_map: pd.Series | Mapping[object, str],
) -> pd.DataFrame:
    if turnover_matrix.empty:
        return pd.DataFrame(index=turnover_matrix.index)

    tier_series = pd.Series(liquidity_tier_map, dtype="string").reindex(
        turnover_matrix.columns
    )
    if tier_series.isna().any():
        missing_tickers = ", ".join(tier_series.index[tier_series.isna()].astype(str))
        raise ValueError(f"Missing liquidity tiers for identities: {missing_tickers}")

    tier_turnover = {}
    for tier in LIQUIDITY_TIER_ORDER:
        tier_columns = tier_series.index[tier_series == tier]
        if len(tier_columns) == 0:
            tier_turnover[f"turnover_{tier}"] = pd.Series(
                0.0, index=turnover_matrix.index, dtype=float
            )
        else:
            tier_turnover[f"turnover_{tier}"] = turnover_matrix.loc[
                :, tier_columns
            ].sum(axis=1)
    return pd.DataFrame(tier_turnover, index=turnover_matrix.index)


def calculate_turnover_costs_from_tier_turnover(
    tier_turnover: pd.DataFrame,
    cost_scenario: str | CostScenarioDefinition | None = None,
    turnover_cost_bps: float | None = None,
) -> pd.Series:
    if tier_turnover.empty:
        return pd.Series(dtype=float)

    if turnover_cost_bps is not None:
        total_turnover = tier_turnover.sum(axis=1)
        return total_turnover * basis_points_to_rate(turnover_cost_bps)

    scenario = resolve_cost_scenario(cost_scenario)
    turnover_cost = pd.Series(0.0, index=tier_turnover.index, dtype=float)
    for tier in LIQUIDITY_TIER_ORDER:
        column_name = f"turnover_{tier}"
        if column_name not in tier_turnover.columns:
            continue
        turnover_cost = turnover_cost + (
            tier_turnover[column_name].fillna(0.0)
            * basis_points_to_rate(scenario.cost_bps_for_tier(tier))
        )
    return turnover_cost


def apply_turnover_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    turnover_cost_bps: float | None = None,
    turnover_cost_rate: float | pd.Series | None = None,
) -> pd.Series:
    if turnover_cost_rate is None:
        if turnover_cost_bps is None:
            raise ValueError(
                "Either turnover_cost_bps or turnover_cost_rate must be provided."
            )
        turnover_cost_rate = basis_points_to_rate(turnover_cost_bps)
    return gross_returns - turnover.fillna(0.0) * turnover_cost_rate


def summarize_cost_model_inputs(
    cost_scenario: str | CostScenarioDefinition | None = None,
    turnover_cost_bps: float | None = None,
) -> dict[str, object]:
    if turnover_cost_bps is not None:
        return {
            "cost_model_mode": "flat_turnover_cost_legacy",
            "cost_scenario": "legacy_flat",
            "turnover_cost_bps": float(turnover_cost_bps),
            "cost_disclosure": "flat_turnover_cost_only_legacy_borrow_and_financing_excluded",
            "borrow_costs_modeled": False,
            "financing_costs_modeled": False,
        }

    scenario = resolve_cost_scenario(cost_scenario)
    summary = {
        "cost_model_mode": "liquidity_tiered_scenario",
        "cost_scenario": scenario.name,
        "turnover_cost_bps": np.nan,
        "cost_disclosure": scenario.exclusions_disclosure,
        "borrow_costs_modeled": scenario.borrow_costs_modeled,
        "financing_costs_modeled": scenario.financing_costs_modeled,
        "tax_loss_trade_legs": scenario.tax_loss_trade_legs,
        "tax_loss_adjust_control_leg": scenario.tax_loss_adjust_control_leg,
    }
    for tier in LIQUIDITY_TIER_ORDER:
        summary[f"turnover_cost_bps_{tier}"] = scenario.cost_bps_for_tier(tier)
    return summary
