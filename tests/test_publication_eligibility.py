from __future__ import annotations

import unittest

import pandas as pd

from strategies.backtest import run_equal_weight_long_only_backtest
from strategies.publication_eligibility import (
    NEXT_SESSION_AFTER_CLOSE,
    annotate_publication_eligibility,
)


def make_publication_eligibility_panel() -> pd.DataFrame:
    rows = [
        {"asset_id": 1001, "ticker_code": "AAA", "trade_date": "2026-01-01", "member_of_universe": 0, "adj_close": 10.0, "daily_return": 0.00, "dollar_volume": 1000.0},
        {"asset_id": 1001, "ticker_code": "AAA", "trade_date": "2026-01-02", "member_of_universe": 0, "adj_close": 10.5, "daily_return": 0.01, "dollar_volume": 1000.0},
        {"asset_id": 1001, "ticker_code": "AAA", "trade_date": "2026-01-03", "member_of_universe": 1, "adj_close": 11.0, "daily_return": 0.02, "dollar_volume": 1000.0},
        {"asset_id": 1001, "ticker_code": "AAA", "trade_date": "2026-01-04", "member_of_universe": 1, "adj_close": 11.2, "daily_return": 0.01, "dollar_volume": 1000.0},
        {"asset_id": 1001, "ticker_code": "AAA", "trade_date": "2026-01-05", "member_of_universe": 0, "adj_close": 11.1, "daily_return": -0.01, "dollar_volume": 1000.0},
        {"asset_id": 1001, "ticker_code": "AAA", "trade_date": "2026-01-06", "member_of_universe": 1, "adj_close": 11.4, "daily_return": 0.03, "dollar_volume": 1000.0},
        {"asset_id": 2002, "ticker_code": "BBB", "trade_date": "2026-01-01", "member_of_universe": 0, "adj_close": 20.0, "daily_return": 0.00, "dollar_volume": 2000.0},
        {"asset_id": 2002, "ticker_code": "BBB", "trade_date": "2026-01-02", "member_of_universe": 1, "adj_close": 20.2, "daily_return": 0.01, "dollar_volume": 2000.0},
        {"asset_id": 2002, "ticker_code": "BBB", "trade_date": "2026-01-03", "member_of_universe": 1, "adj_close": 20.5, "daily_return": 0.02, "dollar_volume": 2000.0},
        {"asset_id": 3003, "ticker_code": "AAA", "trade_date": "2026-01-01", "member_of_universe": 1, "adj_close": 30.0, "daily_return": 0.00, "dollar_volume": 3000.0},
        {"asset_id": 3003, "ticker_code": "CCC", "trade_date": "2026-01-02", "member_of_universe": 1, "adj_close": 30.3, "daily_return": 0.01, "dollar_volume": 3000.0},
        {"asset_id": 3003, "ticker_code": "CCC", "trade_date": "2026-01-03", "member_of_universe": 1, "adj_close": 30.6, "daily_return": 0.01, "dollar_volume": 3000.0},
    ]
    panel = pd.DataFrame(rows)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    return panel


class PublicationEligibilityTests(unittest.TestCase):
    def test_pre_membership_history_counts_toward_immediate_eligibility(self) -> None:
        panel = make_publication_eligibility_panel()
        annotated = annotate_publication_eligibility(panel, min_history=3)
        asset = annotated.loc[annotated["asset_id"] == 1001].reset_index(drop=True)
        self.assertEqual(len(asset), 6)
        self.assertEqual(asset.loc[2, "history_observation_count"], 3)
        self.assertTrue(bool(asset.loc[2, "has_min_history"]))
        self.assertTrue(bool(asset.loc[2, "eligible_to_trade"]))

    def test_membership_entry_without_enough_history_delays_eligibility(self) -> None:
        panel = make_publication_eligibility_panel()
        annotated = annotate_publication_eligibility(panel, min_history=3)
        asset = annotated.loc[annotated["asset_id"] == 2002].reset_index(drop=True)
        self.assertFalse(bool(asset.loc[1, "eligible_to_trade"]))
        self.assertTrue(bool(asset.loc[2, "eligible_to_trade"]))

    def test_membership_exit_and_reentry_update_eligibility_without_dropping_history(self) -> None:
        panel = make_publication_eligibility_panel()
        annotated = annotate_publication_eligibility(panel, min_history=3)
        asset = annotated.loc[annotated["asset_id"] == 1001].reset_index(drop=True)
        self.assertFalse(bool(asset.loc[4, "eligible_to_trade"]))
        self.assertEqual(asset.loc[5, "history_observation_count"], 6)
        self.assertTrue(bool(asset.loc[5, "eligible_to_trade"]))

    def test_asset_id_identity_overrides_shared_or_changed_ticker_strings(self) -> None:
        panel = make_publication_eligibility_panel()
        annotated = annotate_publication_eligibility(panel, min_history=2)
        shared_ticker_asset = annotated.loc[annotated["asset_id"] == 3003].reset_index(drop=True)
        self.assertEqual(shared_ticker_asset.loc[1, "ticker_code"], "CCC")
        self.assertEqual(shared_ticker_asset.loc[1, "history_observation_count"], 2)
        self.assertTrue(bool(shared_ticker_asset.loc[1, "eligible_to_trade"]))
        other_asset = annotated.loc[annotated["asset_id"] == 1001].reset_index(drop=True)
        self.assertEqual(other_asset.loc[1, "history_observation_count"], 2)
        self.assertFalse(bool(other_asset.loc[1, "eligible_to_trade"]))

    def test_next_session_timing_matches_backtest_position_shift(self) -> None:
        panel = pd.DataFrame(
            [
                {"asset_id": 1, "ticker_code": "AAA", "trade_date": "2026-01-01", "member_of_universe": 0, "adj_close": 10.0, "daily_return": 0.00, "dollar_volume": 1000.0},
                {"asset_id": 1, "ticker_code": "AAA", "trade_date": "2026-01-02", "member_of_universe": 1, "adj_close": 10.5, "daily_return": 0.10, "dollar_volume": 1000.0},
                {"asset_id": 1, "ticker_code": "AAA", "trade_date": "2026-01-03", "member_of_universe": 1, "adj_close": 10.8, "daily_return": 0.20, "dollar_volume": 1000.0},
            ]
        )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        annotated = annotate_publication_eligibility(
            panel,
            min_history=2,
            timing_convention=NEXT_SESSION_AFTER_CLOSE,
        )
        annotated["signal"] = annotated["eligible_to_trade"].astype(float)
        results = run_equal_weight_long_only_backtest(
            annotated,
            signal_col="signal",
            return_col="daily_return",
            turnover_cost_bps=10.0,
        )
        self.assertEqual(annotated.attrs["timing_convention"], NEXT_SESSION_AFTER_CLOSE)
        self.assertTrue(bool(annotated.loc[1, "eligible_to_trade"]))
        self.assertAlmostEqual(results.loc[pd.Timestamp("2026-01-02"), "gross_return"], 0.0)
        self.assertAlmostEqual(results.loc[pd.Timestamp("2026-01-03"), "gross_return"], 0.20)


if __name__ == "__main__":
    unittest.main()
