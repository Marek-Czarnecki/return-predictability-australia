from __future__ import annotations

import unittest

import pandas as pd

from strategies.publication_benchmarks import (
    XJOA_ASSET_ID,
    XJOA_SYMBOL,
    build_external_benchmark_returns,
    build_point_in_time_equal_weight_benchmark,
)


class PublicationBenchmarkTests(unittest.TestCase):
    def make_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["2026-01-05", "2026-01-02", "2026-01-06"],
                "asset_id": [XJOA_ASSET_ID] * 3,
                "symbol": [XJOA_SYMBOL] * 3,
                "close": [102.0, 100.0, 101.0],
            }
        )

    def make_publication_panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2026-01-02",
                        "2026-01-02",
                        "2026-01-05",
                        "2026-01-05",
                        "2026-01-06",
                        "2026-01-06",
                    ]
                ),
                "asset_id": [1, 2, 1, 2, 1, 2],
                "daily_return": [None, None, 0.10, 0.20, 0.30, 0.40],
                "member_of_universe": [True, False, True, True, False, True],
            }
        )

    def test_build_external_benchmark_returns_calculates_close_to_close_returns(self) -> None:
        result = build_external_benchmark_returns(self.make_frame())
        self.assertEqual(
            list(result["trade_date"]),
            list(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])),
        )
        self.assertAlmostEqual(result.loc[1, "benchmark_return"], 0.02)
        self.assertAlmostEqual(result.loc[2, "benchmark_return"], (101.0 / 102.0) - 1.0)

    def test_first_benchmark_return_remains_nan(self) -> None:
        result = build_external_benchmark_returns(self.make_frame())
        self.assertTrue(pd.isna(result.loc[0, "benchmark_return"]))

    def test_duplicate_trade_dates_are_rejected(self) -> None:
        frame = self.make_frame()
        frame.loc[2, "trade_date"] = frame.loc[1, "trade_date"]
        with self.assertRaisesRegex(ValueError, "duplicate trade dates"):
            build_external_benchmark_returns(frame)

    def test_null_close_is_rejected(self) -> None:
        frame = self.make_frame()
        frame.loc[1, "close"] = None
        with self.assertRaisesRegex(ValueError, "null or non-numeric close"):
            build_external_benchmark_returns(frame)

    def test_non_numeric_close_is_rejected(self) -> None:
        frame = self.make_frame()
        frame["close"] = frame["close"].astype(object)
        frame.loc[1, "close"] = "bad"
        with self.assertRaisesRegex(ValueError, "null or non-numeric close"):
            build_external_benchmark_returns(frame)

    def test_non_positive_close_is_rejected(self) -> None:
        frame = self.make_frame()
        frame.loc[1, "close"] = 0.0
        with self.assertRaisesRegex(ValueError, "non-positive close"):
            build_external_benchmark_returns(frame)

    def test_wrong_asset_id_is_rejected(self) -> None:
        frame = self.make_frame()
        frame.loc[1, "asset_id"] = 999999
        with self.assertRaisesRegex(ValueError, "asset_id"):
            build_external_benchmark_returns(frame)

    def test_correct_xjoa_metadata_is_accepted(self) -> None:
        result = build_external_benchmark_returns(self.make_frame())
        self.assertEqual(
            list(result.columns),
            ["trade_date", "benchmark_level", "benchmark_return"],
        )
        self.assertEqual(len(result), 3)

    def test_equal_weight_benchmark_uses_previous_session_membership(self) -> None:
        result = build_point_in_time_equal_weight_benchmark(
            self.make_publication_panel()
        ).set_index("trade_date")
        self.assertTrue(pd.isna(result.loc[pd.Timestamp("2026-01-02"), "equal_weight_return"]))
        self.assertAlmostEqual(result.loc[pd.Timestamp("2026-01-05"), "equal_weight_return"], 0.10)
        self.assertAlmostEqual(result.loc[pd.Timestamp("2026-01-06"), "equal_weight_return"], 0.35)
        self.assertEqual(result.loc[pd.Timestamp("2026-01-05"), "member_count"], 1)
        self.assertEqual(result.loc[pd.Timestamp("2026-01-06"), "member_count"], 2)

    def test_equal_weight_benchmark_reweights_observable_member_returns(self) -> None:
        frame = self.make_publication_panel()
        frame.loc[(frame["trade_date"] == pd.Timestamp("2026-01-05")) & (frame["asset_id"] == 2), "member_of_universe"] = False
        frame.loc[(frame["trade_date"] == pd.Timestamp("2026-01-02")), "member_of_universe"] = True
        frame.loc[(frame["trade_date"] == pd.Timestamp("2026-01-05")) & (frame["asset_id"] == 2), "daily_return"] = None
        result = build_point_in_time_equal_weight_benchmark(frame).set_index("trade_date")
        row = result.loc[pd.Timestamp("2026-01-05")]
        self.assertAlmostEqual(row["equal_weight_return"], 0.10)
        self.assertEqual(row["member_count"], 2)
        self.assertEqual(row["observable_member_return_count"], 1)
        self.assertEqual(row["missing_member_return_count"], 1)
        self.assertAlmostEqual(row["missing_member_return_fraction"], 0.5)

    def test_equal_weight_benchmark_does_not_use_strategy_eligibility(self) -> None:
        frame = self.make_publication_panel()
        frame["eligible_to_trade"] = False
        result = build_point_in_time_equal_weight_benchmark(frame).set_index("trade_date")
        self.assertAlmostEqual(result.loc[pd.Timestamp("2026-01-05"), "equal_weight_return"], 0.10)

    def test_equal_weight_benchmark_rejects_duplicate_identity_date_rows(self) -> None:
        frame = pd.concat([self.make_publication_panel(), self.make_publication_panel().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate identity/trade_date"):
            build_point_in_time_equal_weight_benchmark(frame)

    def test_equal_weight_benchmark_rejects_invalid_membership_values(self) -> None:
        frame = self.make_publication_panel()
        frame["member_of_universe"] = frame["member_of_universe"].astype(object)
        frame.loc[0, "member_of_universe"] = "yes"
        with self.assertRaisesRegex(ValueError, "membership values"):
            build_point_in_time_equal_weight_benchmark(frame)

    def test_equal_weight_benchmark_keeps_missing_return_when_no_expected_members(self) -> None:
        frame = self.make_publication_panel()
        frame.loc[frame["trade_date"] == pd.Timestamp("2026-01-02"), "member_of_universe"] = False
        result = build_point_in_time_equal_weight_benchmark(frame).set_index("trade_date")
        row = result.loc[pd.Timestamp("2026-01-05")]
        self.assertTrue(pd.isna(row["equal_weight_return"]))
        self.assertEqual(row["member_count"], 0)
        self.assertTrue(pd.isna(row["missing_member_return_fraction"]))


if __name__ == "__main__":
    unittest.main()
