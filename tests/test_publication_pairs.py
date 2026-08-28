from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategies.publication_pairs import (
    PAIR_BORROW_FINANCING,
    PAIR_CAPITAL_NORMALIZATION,
    PAIR_COST_APPLICATION,
    PAIR_FORMATION_HISTORY_CONVENTION,
    build_publication_pair_return_panel,
    run_publication_pairs_walk_forward,
)


class PublicationPairsTests(unittest.TestCase):
    def test_pair_returns_are_gross_exposure_normalized_and_use_weighted_two_leg_cost(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        left_close = pd.Series([100.0, 100.0, 100.0, 80.0, 88.0, 88.0])
        right_close = pd.Series([100.0] * 6)
        prices = pd.concat(
            [
                pd.DataFrame(
                    {
                        "asset_id": 1,
                        "ticker_code": "LEFT",
                        "trade_date": dates,
                        "adj_close": left_close,
                        "eligible_to_trade": True,
                    }
                ),
                pd.DataFrame(
                    {
                        "asset_id": 2,
                        "ticker_code": "RIGHT",
                        "trade_date": dates,
                        "adj_close": right_close,
                        "eligible_to_trade": True,
                    }
                ),
            ],
            ignore_index=True,
        )
        selected = pd.DataFrame(
            [
                {
                    "pair_id": "1_2",
                    "left_identity": 1,
                    "right_identity": 2,
                    "left_ticker": "LEFT",
                    "right_ticker": "RIGHT",
                    "hedge_ratio": 1.0,
                    "intercept": 0.0,
                    "left_liquidity_tier": "high",
                    "right_liquidity_tier": "lower",
                }
            ]
        )

        panel = build_publication_pair_return_panel(
            prices,
            selected,
            spread_window=2,
            entry_z=0.5,
            exit_z=0.1,
            cost_scenario="base",
        ).set_index("trade_date")

        self.assertEqual(float(panel.loc[dates[4], "gross_exposure_denominator"]), 2.0)
        self.assertAlmostEqual(float(panel.loc[dates[4], "pair_gross_return"]), 0.05, places=10)
        self.assertAlmostEqual(float(panel.loc[dates[4], "pair_turnover_cost_bps"]), 22.5, places=10)

    def test_one_fold_pairs_uses_formation_convention_a_and_locked_method_labels(self):
        dates = pd.bdate_range("2000-03-31", "2004-06-30")
        steps = np.arange(len(dates), dtype=float)
        frames = []
        for asset_id, ticker, scale in ((11, "AAA", 1.0), (22, "BBB", 1.02)):
            close = scale * 100.0 * np.exp(0.0003 * steps)
            membership = np.ones(len(dates), dtype=bool)
            if asset_id == 22:
                membership[:200] = False
            frame = pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "ticker_code": ticker,
                    "trade_date": dates,
                    "adj_close": close,
                    "daily_return": pd.Series(close).pct_change(fill_method=None).to_numpy(),
                    "dollar_volume": 5_000_000.0 if asset_id == 11 else 4_000_000.0,
                    "member_of_universe": membership,
                }
            )
            frames.append(frame)
        prices = pd.concat(frames, ignore_index=True)
        benchmark = pd.DataFrame(
            {
                "trade_date": dates,
                "benchmark_return": np.r_[np.nan, np.repeat(0.0001, len(dates) - 1)],
            }
        )

        result = run_publication_pairs_walk_forward(
            prices,
            benchmark,
            max_folds=1,
            top_liquid_tickers=2,
            top_pair_count=1,
        )

        self.assertEqual(len(result.fold_table), 1)
        fold = result.fold_table.iloc[0]
        self.assertEqual(fold["formation_history_convention"], PAIR_FORMATION_HISTORY_CONVENTION)
        self.assertEqual(fold["capital_normalization"], PAIR_CAPITAL_NORMALIZATION)
        self.assertEqual(fold["pair_cost_application"], PAIR_COST_APPLICATION)
        self.assertEqual(fold["borrow_financing"], PAIR_BORROW_FINANCING)
        self.assertEqual(int(fold["selected_pair_count"]), 1)
        self.assertFalse(result.pair_diagnostics.empty)
        formation_obs = float(result.pair_diagnostics.iloc[0]["formation_observations"])
        simultaneous_member_obs = int(
            prices.loc[
                prices["trade_date"].between(fold["formation_start"], fold["formation_end"])
                & prices["asset_id"].eq(22)
                & prices["member_of_universe"],
                "trade_date",
            ].nunique()
        )
        self.assertGreater(formation_obs, simultaneous_member_obs)
        self.assertFalse(result.fold_daily_results.empty)
        self.assertFalse(result.fold_summary.empty)


if __name__ == "__main__":
    unittest.main()
