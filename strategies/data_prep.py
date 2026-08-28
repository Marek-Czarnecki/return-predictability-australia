from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    clean_panel_path: Path
    results_root: Path


def find_project_root(start_path: Path) -> Path:
    """Locate the public reproducibility repository without assuming notebooks exist."""
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "data").exists() and (candidate / "strategies").exists():
            return candidate
    raise FileNotFoundError("Could not locate the public reproducibility project root.")


def get_project_paths(start_path: Path | None = None) -> ProjectPaths:
    """Return public-safe licensed-input and generated-output paths.

    This is a portability-only adaptation of the frozen capstone helper. The licensed
    point-in-time panel is never committed and is expected under data/licensed/.
    """
    root = find_project_root((start_path or Path.cwd()).resolve())
    clean_panel_path = root / "data" / "licensed" / "asx200_point_in_time_panel.parquet"
    results_root = root / "data" / "generated" / "publication_results"
    results_root.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(project_root=root, clean_panel_path=clean_panel_path, results_root=results_root)


def load_clean_panel(clean_panel_path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(clean_panel_path).copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    sort_identity = "asset_id" if "asset_id" in prices.columns else "ticker_code"
    prices = prices.sort_values([sort_identity, "trade_date"]).reset_index(drop=True)
    if "ticker_code" in prices.columns:
        prices["ticker_code"] = prices["ticker_code"].astype("string")
    if "adj_close" not in prices.columns and "adj close" in prices.columns:
        prices["adj_close"] = pd.to_numeric(prices["adj close"], errors="coerce")
    if "daily_return" not in prices.columns:
        prices["daily_return"] = prices.groupby(sort_identity, observed=True)["adj_close"].pct_change(fill_method=None)
    if "dollar_volume" not in prices.columns:
        prices["dollar_volume"] = prices["adj_close"] * pd.to_numeric(prices["volume"], errors="coerce")
    return prices


def summarize_dataset(prices: pd.DataFrame) -> dict[str, object]:
    identity_col = "asset_id" if "asset_id" in prices.columns else "ticker_code"
    return {
        "row_count": len(prices),
        "ticker_count": prices[identity_col].nunique(),
        "start_date": prices["trade_date"].min(),
        "end_date": prices["trade_date"].max(),
    }


def build_ticker_liquidity(
    prices: pd.DataFrame, high_liquidity_cutoff: float = 0.30
) -> pd.DataFrame:
    ticker_liquidity = (
        prices.groupby("ticker_code", observed=True)
        .agg(
            first_date=("trade_date", "min"),
            last_date=("trade_date", "max"),
            row_count=("trade_date", "size"),
            median_dollar_volume=("dollar_volume", "median"),
            mean_dollar_volume=("dollar_volume", "mean"),
        )
        .sort_values("median_dollar_volume", ascending=False)
    )
    ticker_liquidity["liquidity_rank"] = np.arange(1, len(ticker_liquidity) + 1)
    ticker_liquidity["liquidity_percentile"] = ticker_liquidity["liquidity_rank"] / len(ticker_liquidity)
    ticker_liquidity["high_liquidity_flag"] = ticker_liquidity["liquidity_percentile"] <= high_liquidity_cutoff
    return ticker_liquidity
