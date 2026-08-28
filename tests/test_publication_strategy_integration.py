from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from strategies.mean_reversion import run_mean_reversion_strategy
from strategies.pairs_trading import build_pair_return_panel
from strategies.publication_data import prepare_publication_strategy_panel
from strategies.trend_following import run_trend_following_strategy
from strategies.walk_forward import StrategyDefinition, run_walk_forward_optimization


class PublicationStrategyIntegrationTests(unittest.TestCase):
    def test_pre_membership_history_can_make_first_member_date_immediately_eligible(self):
        prices = _single_asset_panel(
            asset_id=101,
            ticker_code="AAA",
            closes=[10, 11, 12, 13, 14],
            membership=[0, 0, 0, 1, 1],
        )

        prepared = prepare_publication_strategy_panel(prices, min_history=3)
        first_member = prepared.loc[prepared["member_of_universe"].eq(1)].iloc[0]

        self.assertEqual(int(first_member["history_observation_count"]), 4)
        self.assertTrue(bool(first_member["has_min_history"]))
        self.assertTrue(bool(first_member["eligible_to_trade"]))

    def test_reentry_retains_prior_history_and_can_resume_immediately(self):
        prices = _single_asset_panel(
            asset_id=202,
            ticker_code="BBB",
            closes=[20, 21, 22, 23, 24, 25],
            membership=[0, 1, 1, 0, 0, 1],
        )

        prepared = prepare_publication_strategy_panel(prices, min_history=2)
        reentry = prepared.iloc[-1]

        self.assertEqual(int(reentry["history_observation_count"]), 6)
        self.assertTrue(bool(reentry["has_min_history"]))
        self.assertTrue(bool(reentry["eligible_to_trade"]))

    def test_trend_groups_and_backtests_by_asset_id_not_reused_ticker(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        prices = pd.concat(
            [
                _asset_rows(1, "REUSE", dates, [10, 11, 12, 13, 14]),
                _asset_rows(2, "REUSE", dates, [20, 20, 20, 20, 20]),
            ],
            ignore_index=True,
        )
        prices["eligible_to_trade"] = True

        run = run_trend_following_strategy(
            prices,
            fast_window=1,
            slow_window=2,
            min_history=2,
            turnover_cost_bps=0.0,
            identity_col="asset_id",
            eligibility_col="eligible_to_trade",
        )

        asset_one = run.panel.loc[run.panel["asset_id"].eq(1), "signal"]
        asset_two = run.panel.loc[run.panel["asset_id"].eq(2), "signal"]

        self.assertGreater(float(asset_one.sum()), 0.0)
        self.assertEqual(float(asset_two.sum()), 0.0)
        self.assertEqual(run.summary.iloc[0]["identity_col"], "asset_id")

    def test_mean_reversion_forces_signal_flat_when_membership_eligibility_ends(self):
        dates = pd.date_range("2024-02-01", periods=6, freq="D")
        prices = pd.DataFrame(
            {
                "asset_id": [303] * 6,
                "ticker_code": ["CCC"] * 6,
                "trade_date": dates,
                "adj_close": [100, 101, 96, 92, 93, 94],
                "daily_return": [np.nan, 0.01, -0.05, -0.04, 0.01, 0.01],
                "dollar_volume": [1_000_000.0] * 6,
                "eligible_to_trade": [True, True, True, True, False, True],
            }
        )

        run = run_mean_reversion_strategy(
            prices,
            lookback_window=2,
            entry_z=-0.5,
            exit_z=99.0,
            min_history=2,
            turnover_cost_bps=0.0,
            identity_col="asset_id",
            eligibility_col="eligible_to_trade",
        )

        panel = run.panel.set_index("trade_date")
        self.assertEqual(float(panel.loc[dates[3], "signal"]), 1.0)
        self.assertEqual(float(panel.loc[dates[4], "signal"]), 0.0)

    def test_pairs_require_both_assets_to_be_eligible_on_signal_date(self):
        dates = pd.date_range("2024-03-01", periods=6, freq="D")
        left = _asset_rows(401, "LEFT", dates, [100, 100, 100, 80, 80, 80])
        right = _asset_rows(402, "RIGHT", dates, [100, 100, 100, 100, 100, 100])
        left["eligible_to_trade"] = True
        right["eligible_to_trade"] = [True, True, True, True, False, False]
        trading_prices = pd.concat([left, right], ignore_index=True)
        selected_pairs = pd.DataFrame(
            [
                {
                    "pair_id": "401_402",
                    "left_identity": 401,
                    "right_identity": 402,
                    "left_ticker": "LEFT",
                    "right_ticker": "RIGHT",
                    "hedge_ratio": 1.0,
                    "intercept": 0.0,
                    "left_liquidity_tier": "high",
                    "right_liquidity_tier": "high",
                }
            ]
        )

        panel = build_pair_return_panel(
            trading_prices=trading_prices,
            selected_pairs=selected_pairs,
            spread_window=2,
            entry_z=0.5,
            exit_z=0.1,
            turnover_cost_bps=0.0,
            cost_scenario="base",
            identity_col="asset_id",
            eligibility_col="eligible_to_trade",
        ).set_index("trade_date")

        self.assertNotEqual(float(panel.loc[dates[3], "pair_position"]), 0.0)
        self.assertEqual(float(panel.loc[dates[4], "pair_position"]), 0.0)
        self.assertEqual(float(panel.loc[dates[5], "pair_position"]), 0.0)

    def test_ticker_reuse_does_not_merge_history_between_asset_ids(self):
        dates = pd.date_range("2024-04-01", periods=3, freq="D")
        prices = pd.concat(
            [
                _asset_rows(501, "SAME", dates, [10, 11, 12], membership=[1, 1, 1]),
                _asset_rows(502, "SAME", dates, [30, 31, 32], membership=[1, 1, 1]),
            ],
            ignore_index=True,
        )

        prepared = prepare_publication_strategy_panel(prices, min_history=2)
        histories = prepared.groupby("asset_id")["history_observation_count"].apply(list)

        self.assertEqual(histories.loc[501], [1, 2, 3])
        self.assertEqual(histories.loc[502], [1, 2, 3])

    def test_walk_forward_passes_publication_semantics_without_optimizing_them(self):
        calls: list[tuple[str, str | None]] = []

        def runner(
            prices: pd.DataFrame,
            alpha: int = 1,
            identity_col: str = "ticker_code",
            eligibility_col: str | None = None,
        ) -> SimpleNamespace:
            calls.append((identity_col, eligibility_col))
            index = pd.DatetimeIndex(pd.to_datetime(prices["trade_date"]))
            daily = pd.DataFrame(
                {
                    "net_return": np.zeros(len(index), dtype=float),
                    "turnover": np.zeros(len(index), dtype=float),
                },
                index=index,
            )
            daily.index.name = "trade_date"
            return SimpleNamespace(daily_results=daily, panel=prices.copy())

        dates = pd.date_range("2020-01-31", "2025-12-31", freq="M")
        prices = pd.DataFrame(
            {
                "trade_date": dates,
                "asset_id": [601] * len(dates),
                "ticker_code": ["DDD"] * len(dates),
                "eligible_to_trade": [True] * len(dates),
            }
        )
        benchmark = pd.Series(0.0, index=dates, name="benchmark_return")
        definitions = {
            "publication_dummy": StrategyDefinition(
                runner=runner,
                parameter_grid={"alpha": [1]},
            )
        }

        result = run_walk_forward_optimization(
            strategy_name="publication_dummy",
            prices=prices,
            benchmark_returns=benchmark,
            strategy_definitions=definitions,
            strategy_run_kwargs={
                "identity_col": "asset_id",
                "eligibility_col": "eligible_to_trade",
            },
        )

        self.assertFalse(result.fold_table.empty)
        self.assertTrue(result.fold_table["chosen_parameters"].eq("alpha=1").all())
        self.assertTrue(calls)
        self.assertTrue(all(call == ("asset_id", "eligible_to_trade") for call in calls))


def _single_asset_panel(
    asset_id: int,
    ticker_code: str,
    closes: list[float],
    membership: list[int],
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return _asset_rows(
        asset_id,
        ticker_code,
        dates,
        closes,
        membership=membership,
    )


def _asset_rows(
    asset_id: int,
    ticker_code: str,
    dates: pd.DatetimeIndex,
    closes: list[float],
    membership: list[int] | None = None,
) -> pd.DataFrame:
    close_series = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "asset_id": [asset_id] * len(dates),
            "ticker_code": [ticker_code] * len(dates),
            "trade_date": dates,
            "adj_close": close_series.to_numpy(),
            "daily_return": close_series.pct_change(fill_method=None).to_numpy(),
            "dollar_volume": [1_000_000.0] * len(dates),
            "member_of_universe": membership if membership is not None else [1] * len(dates),
        }
    )


if __name__ == "__main__":
    unittest.main()
