from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .publication_costs import (
    DEFAULT_MIN_LIQUIDITY_OBSERVATIONS,
    PublicationLiquidityTierResult,
    build_publication_fold_liquidity_tiers,
)
from .walk_forward import WalkForwardFold


@dataclass(frozen=True)
class PublicationFoldCostContext:
    fold_id: str
    liquidity_tier_map: pd.Series
    liquidity_diagnostics: pd.DataFrame


def build_publication_fold_cost_context(
    prices: pd.DataFrame,
    fold: WalkForwardFold,
    evaluation_prices: pd.DataFrame | None = None,
    identity_col: str = "asset_id",
    min_liquidity_observations: int = DEFAULT_MIN_LIQUIDITY_OBSERVATIONS,
) -> PublicationFoldCostContext:
    """Build the frozen ex-ante cost context for one publication fold.

    The liquidity ranking is formed only from the fold's formation window and
    formation-end membership. The resulting map covers every identity observed
    in the supplied evaluation panel (or the full input panel if an evaluation
    panel is not supplied), assigning conservative lower-tier costs to identities
    that were not rankable at formation end.
    """
    source = evaluation_prices if evaluation_prices is not None else prices
    if identity_col not in source.columns:
        raise ValueError(f"Evaluation prices are missing identity column: {identity_col}")

    evaluation_identities = pd.Index(source[identity_col].dropna().unique())
    result: PublicationLiquidityTierResult = build_publication_fold_liquidity_tiers(
        prices=prices,
        formation_start=fold.formation_start,
        formation_end=fold.formation_end,
        evaluation_identities=evaluation_identities,
        identity_col=identity_col,
        min_liquidity_observations=min_liquidity_observations,
    )

    diagnostics = result.diagnostics.copy()
    diagnostics.insert(0, "fold_id", fold.fold_id)
    return PublicationFoldCostContext(
        fold_id=fold.fold_id,
        liquidity_tier_map=result.tier_map,
        liquidity_diagnostics=diagnostics,
    )


def publication_strategy_cost_kwargs(
    context: PublicationFoldCostContext,
) -> dict[str, object]:
    """Return fixed execution kwargs for publication long-only strategies."""
    return {"liquidity_tier_map": context.liquidity_tier_map}
