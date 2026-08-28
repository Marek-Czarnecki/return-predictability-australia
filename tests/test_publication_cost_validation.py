import unittest

import pandas as pd

from strategies.costs import build_liquidity_tier_map
from strategies.publication_walk_forward_costs import build_publication_fold_cost_context
from strategies.trend_following import run_trend_following_strategy
from strategies.walk_forward import WalkForwardFold


class PublicationCostValidationTests(unittest.TestCase):
    def _publication_panel(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        rows = []
        for asset_id, ticker, dollar_volume in [
            (1, "AAA", 1_000_000.0),
            (2, "BBB", 500_000.0),
            (3, "CCC", 100_000.0),
        ]:
            close = 100.0
            for i, trade_date in enumerate(dates):
                close = close * (1.001 if asset_id == 1 else 1.0005)
                rows.append(
                    {
                        "asset_id": asset_id,
                        "ticker_code": ticker,
                        "trade_date": trade_date,
                        "adj_close": close,
                        "daily_return": 0.001 if i > 0 else float("nan"),
                        "dollar_volume": dollar_volume,
                        "member_of_universe": True,
                        "eligible_to_trade": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_post_formation_liquidity_cannot_change_fold_tiers(self):
        prices = self._publication_panel()
        fold = WalkForwardFold(
            fold_id="fold_01",
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_start=pd.Timestamp("2020-03-11"),
            evaluation_end=pd.Timestamp("2020-04-09"),
        )
        evaluation = prices.loc[
            prices["trade_date"].between(fold.evaluation_start, fold.evaluation_end)
        ].copy()

        baseline = build_publication_fold_cost_context(
            prices=prices,
            fold=fold,
            evaluation_prices=evaluation,
        ).liquidity_tier_map

        altered = prices.copy()
        altered.loc[
            (altered["asset_id"] == 3)
            & (altered["trade_date"] > fold.formation_end),
            "dollar_volume",
        ] = 1_000_000_000.0
        perturbed = build_publication_fold_cost_context(
            prices=altered,
            fold=fold,
            evaluation_prices=evaluation,
        ).liquidity_tier_map

        pd.testing.assert_series_equal(baseline, perturbed)

    def test_flat_cost_override_is_independent_of_liquidity_tier_map(self):
        prices = self._publication_panel()
        evaluation = prices.loc[prices["trade_date"] >= "2020-03-11"].copy()
        map_a = pd.Series({1: "high", 2: "medium", 3: "lower"}, dtype="string")
        map_b = pd.Series({1: "lower", 2: "lower", 3: "high"}, dtype="string")

        run_a = run_trend_following_strategy(
            evaluation,
            fast_window=2,
            slow_window=3,
            min_history=3,
            turnover_cost_bps=25.0,
            identity_col="asset_id",
            eligibility_col="eligible_to_trade",
            liquidity_tier_map=map_a,
        )
        run_b = run_trend_following_strategy(
            evaluation,
            fast_window=2,
            slow_window=3,
            min_history=3,
            turnover_cost_bps=25.0,
            identity_col="asset_id",
            eligibility_col="eligible_to_trade",
            liquidity_tier_map=map_b,
        )

        pd.testing.assert_series_equal(
            run_a.daily_results["turnover_cost"],
            run_b.daily_results["turnover_cost"],
        )
        pd.testing.assert_series_equal(
            run_a.daily_results["net_return"],
            run_b.daily_results["net_return"],
        )

    def test_default_strategy_cost_path_matches_explicit_frozen_global_tier_map(self):
        prices = self._publication_panel().rename(columns={"asset_id": "publication_asset_id"})
        global_map = build_liquidity_tier_map(prices, identity_col="ticker_code")

        default_run = run_trend_following_strategy(
            prices,
            fast_window=2,
            slow_window=3,
            min_history=3,
            cost_scenario="base",
            identity_col="ticker_code",
            eligibility_col="eligible_to_trade",
        )
        explicit_run = run_trend_following_strategy(
            prices,
            fast_window=2,
            slow_window=3,
            min_history=3,
            cost_scenario="base",
            identity_col="ticker_code",
            eligibility_col="eligible_to_trade",
            liquidity_tier_map=global_map,
        )

        pd.testing.assert_frame_equal(
            default_run.daily_results,
            explicit_run.daily_results,
        )


if __name__ == "__main__":
    unittest.main()
