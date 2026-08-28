from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .costs import (
    DEFAULT_HIGH_LIQUIDITY_CUTOFF,
    DEFAULT_MEDIUM_LIQUIDITY_CUTOFF,
)


DEFAULT_MIN_LIQUIDITY_OBSERVATIONS = 60
CONSERVATIVE_FALLBACK_TIER = "lower"


@dataclass(frozen=True)
class PublicationLiquidityTierResult:
    tier_map: pd.Series
    diagnostics: pd.DataFrame


def build_publication_fold_liquidity_tiers(
    prices: pd.DataFrame,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    evaluation_identities: Iterable[object] | None = None,
    identity_col: str = "asset_id",
    date_col: str = "trade_date",
    dollar_volume_col: str = "dollar_volume",
    membership_col: str = "member_of_universe",
    min_liquidity_observations: int = DEFAULT_MIN_LIQUIDITY_OBSERVATIONS,
    high_liquidity_cutoff: float = DEFAULT_HIGH_LIQUIDITY_CUTOFF,
    medium_liquidity_cutoff: float = DEFAULT_MEDIUM_LIQUIDITY_CUTOFF,
) -> PublicationLiquidityTierResult:
    """Build ex-ante liquidity tiers for one publication walk-forward fold."""
    required_columns = {
        identity_col,
        date_col,
        dollar_volume_col,
        membership_col,
    }
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Publication liquidity panel is missing required columns: "
            f"{missing_list}"
        )
    if min_liquidity_observations <= 0:
        raise ValueError("min_liquidity_observations must be positive.")
    if not 0 < high_liquidity_cutoff < medium_liquidity_cutoff <= 1:
        raise ValueError("Liquidity cutoffs must satisfy 0 < high < medium <= 1.")

    frame = prices.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    formation_start = pd.Timestamp(formation_start)
    formation_end = pd.Timestamp(formation_end)
    if formation_end < formation_start:
        raise ValueError("formation_end must be on or after formation_start.")

    formation = frame.loc[
        (frame[date_col] >= formation_start) & (frame[date_col] <= formation_end)
    ].copy()
    if formation.empty:
        raise ValueError("No observations are available in the formation window.")

    dollar_volume = pd.to_numeric(formation[dollar_volume_col], errors="coerce")
    if (dollar_volume.dropna() < 0).any():
        raise ValueError("dollar_volume must be non-negative where observed.")
    formation[dollar_volume_col] = dollar_volume

    formation_end_members = pd.Index(
        formation.loc[
            (formation[date_col] == formation_end)
            & formation[membership_col].fillna(False).astype(bool),
            identity_col,
        ].dropna().unique()
    )

    valid_observation = formation[dollar_volume_col].notna()
    liquidity = (
        formation.loc[valid_observation]
        .groupby(identity_col, observed=True)[dollar_volume_col]
        .agg(
            liquidity_observation_count="size",
            median_dollar_volume="median",
        )
    )
    liquidity = liquidity.reindex(formation_end_members)
    liquidity["formation_end_member"] = True
    liquidity["sufficient_liquidity_history"] = (
        liquidity["liquidity_observation_count"].fillna(0)
        >= min_liquidity_observations
    )

    ranked = liquidity.loc[liquidity["sufficient_liquidity_history"]].copy()
    ranked = ranked.sort_values(
        ["median_dollar_volume"],
        ascending=[False],
        kind="mergesort",
    )
    if not ranked.empty:
        ranked["liquidity_rank"] = np.arange(1, len(ranked) + 1)
        ranked["liquidity_percentile"] = ranked["liquidity_rank"] / len(ranked)
        ranked["liquidity_tier"] = np.select(
            [
                ranked["liquidity_percentile"] <= high_liquidity_cutoff,
                ranked["liquidity_percentile"] <= medium_liquidity_cutoff,
            ],
            ["high", "medium"],
            default=CONSERVATIVE_FALLBACK_TIER,
        )
    else:
        ranked["liquidity_rank"] = pd.Series(dtype=float)
        ranked["liquidity_percentile"] = pd.Series(dtype=float)
        ranked["liquidity_tier"] = pd.Series(dtype="string")

    diagnostics = liquidity.join(
        ranked[["liquidity_rank", "liquidity_percentile", "liquidity_tier"]],
        how="left",
    )
    diagnostics["liquidity_observation_count"] = (
        diagnostics["liquidity_observation_count"].fillna(0).astype(int)
    )
    diagnostics["liquidity_tier"] = diagnostics["liquidity_tier"].fillna(
        CONSERVATIVE_FALLBACK_TIER
    )
    diagnostics["tier_assignment_reason"] = np.where(
        diagnostics["sufficient_liquidity_history"],
        "formation_window_rank",
        "insufficient_history_conservative_lower",
    )

    if evaluation_identities is None:
        evaluation_index = pd.Index(frame[identity_col].dropna().unique())
    else:
        evaluation_index = pd.Index(list(evaluation_identities)).dropna().unique()

    tier_map = pd.Series(
        CONSERVATIVE_FALLBACK_TIER,
        index=evaluation_index,
        dtype="string",
        name="liquidity_tier",
    )
    ranked_tiers = diagnostics["liquidity_tier"]
    common_identities = tier_map.index.intersection(ranked_tiers.index)
    tier_map.loc[common_identities] = ranked_tiers.reindex(common_identities).astype(
        "string"
    )

    diagnostics.index.name = identity_col
    diagnostics = diagnostics.reset_index()
    diagnostics["formation_start"] = formation_start
    diagnostics["formation_end"] = formation_end
    diagnostics["min_liquidity_observations"] = min_liquidity_observations

    return PublicationLiquidityTierResult(
        tier_map=tier_map,
        diagnostics=diagnostics,
    )
