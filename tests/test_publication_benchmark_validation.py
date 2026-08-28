from __future__ import annotations

import unittest

import pandas as pd

from strategies.publication_benchmark_validation import (
    XEW_ASSET_ID,
    XEW_SYMBOL,
    XJO_ASSET_ID,
    XJO_SYMBOL,
    compare_return_series,
    validate_publication_benchmarks,
)


class PublicationBenchmarkValidationTests(unittest.TestCase):
    def make_derived(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        external = pd.DataFrame(
            {
                "trade_date": dates,
                "benchmark_level": [100.0, 101.0, 102.0],
                "benchmark_return": [float("nan"), 0.01, 102.0 / 101.0 - 1.0],
            }
        )
        equal_weight = pd.DataFrame(
            {
                "trade_date": dates,
                "equal_weight_return": [float("nan"), 0.008, 0.012],
                "member_count": [0, 200, 200],
                "observable_member_return_count": [0, 199, 200],
                "missing_member_return_count": [0, 1, 0],
                "missing_member_return_fraction": [float("nan"), 0.005, 0.0],
            }
        )
        return external, equal_weight

    def make_reference(
        self,
        *,
        asset_id: int,
        symbol: str,
        levels: list[float],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-05", "2026-01-06"],
                "asset_id": [asset_id] * 3,
                "symbol": [symbol] * 3,
                "close": levels,
            }
        )

    def test_compare_return_series_reports_overlap_and_difference_metrics(self) -> None:
        left = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2026-01-02", "2026-01-05"]), "a": [0.01, 0.02]}
        )
        right = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2026-01-02", "2026-01-05"]), "b": [0.011, 0.018]}
        )
        result = compare_return_series(left, right, left_return_col="a", right_return_col="b")
        self.assertEqual(result["overlap_return_count"], 2)
        self.assertAlmostEqual(result["mean_absolute_return_difference"], 0.0015)

    def test_validation_accepts_structurally_valid_inputs(self) -> None:
        external, equal_weight = self.make_derived()
        xew = self.make_reference(
            asset_id=XEW_ASSET_ID,
            symbol=XEW_SYMBOL,
            levels=[100.0, 100.8, 102.0096],
        )
        xjo = self.make_reference(
            asset_id=XJO_ASSET_ID,
            symbol=XJO_SYMBOL,
            levels=[100.0, 100.9, 101.8],
        )
        result = validate_publication_benchmarks(
            external_benchmark=external,
            equal_weight_benchmark=equal_weight,
            xew_source=xew,
            xjo_source=xjo,
        )
        self.assertEqual(result["structural_validation"]["status"], "passed")
        self.assertEqual(result["xew_validation_reference"]["overlap_return_count"], 2)
        self.assertEqual(result["xjo_price_index_reference"]["overlap_return_count"], 2)

    def test_mismatched_derived_dates_are_rejected(self) -> None:
        external, equal_weight = self.make_derived()
        equal_weight.loc[2, "trade_date"] = pd.Timestamp("2026-01-07")
        xew = self.make_reference(
            asset_id=XEW_ASSET_ID, symbol=XEW_SYMBOL, levels=[100.0, 101.0, 102.0]
        )
        xjo = self.make_reference(
            asset_id=XJO_ASSET_ID, symbol=XJO_SYMBOL, levels=[100.0, 101.0, 102.0]
        )
        with self.assertRaisesRegex(ValueError, "identical dates"):
            validate_publication_benchmarks(
                external_benchmark=external,
                equal_weight_benchmark=equal_weight,
                xew_source=xew,
                xjo_source=xjo,
            )

    def test_wrong_xew_identity_is_rejected(self) -> None:
        external, equal_weight = self.make_derived()
        xew = self.make_reference(asset_id=999, symbol=XEW_SYMBOL, levels=[100.0, 101.0, 102.0])
        xjo = self.make_reference(
            asset_id=XJO_ASSET_ID, symbol=XJO_SYMBOL, levels=[100.0, 101.0, 102.0]
        )
        with self.assertRaisesRegex(ValueError, "asset_id"):
            validate_publication_benchmarks(
                external_benchmark=external,
                equal_weight_benchmark=equal_weight,
                xew_source=xew,
                xjo_source=xjo,
            )


if __name__ == "__main__":
    unittest.main()
