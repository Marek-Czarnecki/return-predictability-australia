from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategies.publication_walk_forward import (
    _coerce_return_series,
    run_publication_walk_forward,
)
from strategies.publication_walk_forward_costs import (
    build_publication_fold_cost_context,
)
from strategies.walk_forward import WalkForwardFold


class PublicationWalkForwardTests(unittest.TestCase):
    @staticmethod
    def _prices() -> pd.DataFrame:
        dates = pd.bdate_range("2000-03-31", "2004-06-30")
        frames = []
        for asset_id, slope, dollar_volume in (
            (101, 0.0005, 10_000_000.0),
            (202, 0.0002, 4_000_000.0),
            (303, -0.0001, 1_000_000.0),
        ):
            steps = np.arange(len(dates), dtype=float)
            adj_close = 100.0 * np.exp(slope * steps)
            frame = pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "ticker_code": f"T{asset_id}",
                    "trade_date": dates,
                    "adj_close": adj_close,
                    "dollar_volume": dollar_volume,
                    "member_of_universe": True,
                }
            )
            frame["daily_return"] = frame["adj_close"].pct_change(fill_method=None)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _benchmark(prices: pd.DataFrame) -> pd.DataFrame:
        dates = pd.Index(sorted(prices["trade_date"].unique()))
        returns = pd.Series(0.0001, index=dates, dtype=float)
        returns.iloc[0] = np.nan
        return pd.DataFrame(
            {
                "trade_date": dates,
                "benchmark_return": returns.to_numpy(),
            }
        )

    def test_return_series_preserves_missing_benchmark_value(self):
        frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2000-03-31", "2000-04-03"]),
                "benchmark_return": [np.nan, 0.01],
            }
        )
        series = _coerce_return_series(frame, "benchmark_return")
        self.assertTrue(pd.isna(series.iloc[0]))
        self.assertEqual(series.iloc[1], 0.01)

    def test_one_fold_trend_smoke_uses_publication_contract(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)

        result = run_publication_walk_forward(
            strategy_name="trend_following",
            prices=prices,
            benchmark_returns=benchmark,
            max_folds=1,
        )

        self.assertEqual(len(result.fold_table), 1)
        fold = result.fold_table.iloc[0]
        self.assertEqual(fold["strategy_name"], "trend_following")
        self.assertEqual(fold["identity_col"], "asset_id")
        self.assertEqual(fold["eligibility_col"], "eligible_to_trade")
        self.assertEqual(
            fold["benchmark_missing_policy"],
            "exclude_from_objective_and_excess_metrics",
        )

        daily = result.fold_daily_results
        self.assertFalse(daily.empty)
        self.assertTrue(daily["benchmark_observed"].all())
        self.assertFalse(daily["benchmark_return"].isna().any())
        self.assertEqual(daily["fold_id"].nunique(), 1)

        self.assertFalse(result.fold_summary.empty)
        self.assertFalse(result.liquidity_diagnostics.empty)
        self.assertEqual(result.liquidity_diagnostics["fold_id"].nunique(), 1)

    def test_cost_map_covers_identity_seen_in_formation_but_not_evaluation_window(self):
        prices = self._prices()
        dates = pd.bdate_range("2000-03-31", "2002-12-31")
        steps = np.arange(len(dates), dtype=float)
        disappearing = pd.DataFrame(
            {
                "asset_id": 404,
                "ticker_code": "T404",
                "trade_date": dates,
                "adj_close": 80.0 * np.exp(0.0003 * steps),
                "dollar_volume": 2_000_000.0,
                "member_of_universe": True,
            }
        )
        disappearing["daily_return"] = disappearing["adj_close"].pct_change(
            fill_method=None
        )
        prices = pd.concat([prices, disappearing], ignore_index=True)

        fold = WalkForwardFold(
            fold_id="fold_01",
            formation_start=pd.Timestamp("2000-03-31"),
            formation_end=pd.Timestamp("2003-03-31"),
            evaluation_start=pd.Timestamp("2003-04-01"),
            evaluation_end=pd.Timestamp("2004-03-31"),
        )
        all_history_through_evaluation = prices.loc[
            prices["trade_date"] <= fold.evaluation_end
        ].copy()
        context = build_publication_fold_cost_context(
            prices=prices,
            fold=fold,
            evaluation_prices=all_history_through_evaluation,
            identity_col="asset_id",
        )

        self.assertIn(404, context.liquidity_tier_map.index)
        self.assertEqual(context.liquidity_tier_map.loc[404], "lower")
        self.assertNotIn(404, context.liquidity_diagnostics["asset_id"].tolist())

    def test_one_fold_mean_reversion_smoke_uses_locked_grid(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)

        result = run_publication_walk_forward(
            strategy_name="mean_reversion",
            prices=prices,
            benchmark_returns=benchmark,
            max_folds=1,
        )

        self.assertEqual(len(result.fold_table), 1)
        chosen = str(result.fold_table.iloc[0]["chosen_parameters"])
        self.assertIn("min_history=60", chosen)
        self.assertIn("cost_scenario=base", chosen)
        self.assertFalse(result.fold_daily_results.empty)
        self.assertFalse(result.fold_summary.empty)

    def test_rejects_unsupported_publication_strategy(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        with self.assertRaises(KeyError):
            run_publication_walk_forward(
                strategy_name="pairs_trading",
                prices=prices,
                benchmark_returns=benchmark,
                max_folds=1,
            )


if __name__ == "__main__":
    unittest.main()
