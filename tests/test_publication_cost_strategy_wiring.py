from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategies.mean_reversion import run_mean_reversion_strategy
from strategies.trend_following import run_trend_following_strategy


class PublicationCostStrategyWiringTests(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        rows = []
        for asset_id, closes in [(1, [10, 11, 12, 13, 14, 15]), (2, [20, 19, 18, 17, 18, 19])]:
            close_series = pd.Series(closes, dtype=float)
            for i, trade_date in enumerate(dates):
                rows.append(
                    {
                        "asset_id": asset_id,
                        "trade_date": trade_date,
                        "adj_close": close_series.iloc[i],
                        "daily_return": close_series.pct_change(fill_method=None).iloc[i],
                        "dollar_volume": 1_000_000.0,
                        "eligible_to_trade": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_trend_uses_supplied_publication_liquidity_tier_map(self):
        tier_map = pd.Series({1: "high", 2: "lower"}, dtype="string")
        run = run_trend_following_strategy(
            self._panel(),
            fast_window=1,
            slow_window=2,
            min_history=2,
            cost_scenario="base",
            identity_col="asset_id",
            eligibility_col="eligible_to_trade",
            liquidity_tier_map=tier_map,
        )

        self.assertEqual(run.daily_results.attrs["liquidity_tier_map"], {1: "high", 2: "lower"})
        self.assertEqual(run.daily_results.attrs["identity_col"], "asset_id")

    def test_mean_reversion_uses_supplied_publication_liquidity_tier_map(self):
        tier_map = pd.Series({1: "medium", 2: "lower"}, dtype="string")
        run = run_mean_reversion_strategy(
            self._panel(),
            lookback_window=2,
            entry_z=-0.1,
            exit_z=0.1,
            min_history=2,
            cost_scenario="base",
            identity_col="asset_id",
            eligibility_col="eligible_to_trade",
            liquidity_tier_map=tier_map,
        )

        self.assertEqual(run.daily_results.attrs["liquidity_tier_map"], {1: "medium", 2: "lower"})
        self.assertEqual(run.daily_results.attrs["identity_col"], "asset_id")


if __name__ == "__main__":
    unittest.main()
