import unittest

import pandas as pd

from strategies.publication_walk_forward_costs import (
    build_publication_fold_cost_context,
    publication_strategy_cost_kwargs,
)
from strategies.walk_forward import WalkForwardFold


class PublicationWalkForwardCostTests(unittest.TestCase):
    def _prices(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=80, freq="D")
        rows = []
        for asset_id, dollar_volume in [(1, 100.0), (2, 50.0), (3, 10.0)]:
            for trade_date in dates:
                rows.append(
                    {
                        "asset_id": asset_id,
                        "trade_date": trade_date,
                        "dollar_volume": dollar_volume,
                        "member_of_universe": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_context_uses_fold_formation_window_and_covers_evaluation_identities(self):
        prices = self._prices()
        fold = WalkForwardFold(
            fold_id="fold_01",
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_start=pd.Timestamp("2020-03-11"),
            evaluation_end=pd.Timestamp("2020-03-20"),
        )
        evaluation = pd.DataFrame(
            {
                "asset_id": [1, 2, 3, 99],
                "trade_date": [pd.Timestamp("2020-03-11")] * 4,
            }
        )

        context = build_publication_fold_cost_context(
            prices=prices,
            fold=fold,
            evaluation_prices=evaluation,
            min_liquidity_observations=60,
        )

        self.assertEqual(context.fold_id, "fold_01")
        self.assertEqual(context.liquidity_tier_map.loc[99], "lower")
        self.assertTrue(context.liquidity_diagnostics["fold_id"].eq("fold_01").all())
        self.assertEqual(
            publication_strategy_cost_kwargs(context)["liquidity_tier_map"].to_dict(),
            context.liquidity_tier_map.to_dict(),
        )

    def test_future_liquidity_cannot_change_fold_cost_context(self):
        prices = self._prices()
        fold = WalkForwardFold(
            fold_id="fold_01",
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_start=pd.Timestamp("2020-03-11"),
            evaluation_end=pd.Timestamp("2020-03-20"),
        )
        evaluation = prices.loc[
            prices["trade_date"].between(fold.evaluation_start, fold.evaluation_end)
        ].copy()

        baseline = build_publication_fold_cost_context(
            prices=prices,
            fold=fold,
            evaluation_prices=evaluation,
        ).liquidity_tier_map

        future = pd.DataFrame(
            {
                "asset_id": [3] * 10,
                "trade_date": pd.date_range("2020-04-01", periods=10, freq="D"),
                "dollar_volume": [10_000_000.0] * 10,
                "member_of_universe": [True] * 10,
            }
        )
        altered = build_publication_fold_cost_context(
            prices=pd.concat([prices, future], ignore_index=True),
            fold=fold,
            evaluation_prices=evaluation,
        ).liquidity_tier_map

        pd.testing.assert_series_equal(baseline, altered)


if __name__ == "__main__":
    unittest.main()
