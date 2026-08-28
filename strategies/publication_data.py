from __future__ import annotations

from pathlib import Path

import pandas as pd

from .publication_eligibility import (
    NEXT_SESSION_AFTER_CLOSE,
    annotate_publication_eligibility,
)


DEFAULT_PUBLICATION_PANEL_PATH = Path("data/licensed/asx200_point_in_time_panel.parquet")


def load_publication_panel(
    publication_panel_path: Path = DEFAULT_PUBLICATION_PANEL_PATH,
) -> pd.DataFrame:
    prices = pd.read_parquet(publication_panel_path).copy()
    required_columns = {
        "asset_id",
        "trade_date",
        "adj_close",
        "daily_return",
        "dollar_volume",
        "member_of_universe",
    }
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Publication panel is missing required strategy columns: " f"{missing_list}"
        )

    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.sort_values(["asset_id", "trade_date"]).reset_index(drop=True)
    return prices


def prepare_publication_strategy_panel(
    prices: pd.DataFrame,
    min_history: int,
    timing_convention: str = NEXT_SESSION_AFTER_CLOSE,
) -> pd.DataFrame:
    return annotate_publication_eligibility(
        prices,
        min_history=min_history,
        identity_col="asset_id",
        membership_col="member_of_universe",
        date_col="trade_date",
        timing_convention=timing_convention,
        filter_eligible=False,
    )
