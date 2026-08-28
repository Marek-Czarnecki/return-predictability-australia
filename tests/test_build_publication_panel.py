from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(TESTS_ROOT))

import scripts.build_publication_panel as publication_builder
from scripts.build_publication_panel import (
    build_validation_payload,
    build_publication_panel,
    transform_publication_panel,
    validate_raw_panel,
    validate_transformed_panel,
)
from strategy_test_utils import make_publication_raw_panel


class BuildPublicationPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_counts = {
            "row_count": 4,
            "asset_count": 2,
            "member_rows": 2,
            "non_member_rows": 2,
            "active_assets": 1,
            "delisted_assets": 1,
        }
        self.baseline_patch = patch.object(
            publication_builder, "BASELINE_COUNTS", self.baseline_counts
        )
        self.end_date_patch = patch.object(
            publication_builder, "BASELINE_END_DATE", pd.Timestamp("2026-08-27")
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.validation_path = (
            Path(self.temp_dir.name) / "publication_results" / "publication_panel_validation.json"
        )
        self.baseline_patch.start()
        self.end_date_patch.start()

    def tearDown(self) -> None:
        self.end_date_patch.stop()
        self.baseline_patch.stop()
        self.temp_dir.cleanup()

    def test_transform_publication_panel_uses_asset_identity_and_turnover(self) -> None:
        raw = make_publication_raw_panel()
        transformed = transform_publication_panel(raw)
        self.assertEqual(list(transformed["asset_id"]), [1001, 1001, 1002, 1002])
        self.assertTrue(transformed["adj_close"].equals(transformed["close"]))
        self.assertTrue(transformed["dollar_volume"].equals(transformed["turnover"]))
        self.assertTrue(pd.isna(transformed.loc[0, "daily_return"]))
        self.assertAlmostEqual(transformed.loc[1, "daily_return"], 0.1)
        self.assertTrue(pd.isna(transformed.loc[2, "daily_return"]))
        self.assertAlmostEqual(transformed.loc[3, "daily_return"], -0.1)

    def test_validate_transformed_panel_allows_only_first_row_return_nulls(self) -> None:
        raw = make_publication_raw_panel()
        source_summary = validate_raw_panel(raw)
        transformed = transform_publication_panel(raw)
        summary = validate_transformed_panel(raw, transformed, source_summary)
        self.assertEqual(summary["row_count"], self.baseline_counts["row_count"])
        self.assertEqual(summary["max_date"], pd.Timestamp("2026-08-27"))

    def test_build_publication_panel_writes_expected_parquet(self) -> None:
        raw = make_publication_raw_panel()
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_path = temp_dir / "publication_raw.csv"
            output_path = temp_dir / "exchange=ASX" / "asx200_point_in_time_panel.parquet"
            raw.to_csv(input_path, index=False)
            summary = build_publication_panel(
                input_path,
                output_path,
                validation_output_path=self.validation_path,
            )
            built = pd.read_parquet(output_path)
            validation = json.loads(self.validation_path.read_text())
        self.assertEqual(summary["output_path"], str(output_path))
        self.assertEqual(len(built), len(raw))
        self.assertIn("daily_return", built.columns)
        self.assertIn("dollar_volume", built.columns)
        self.assertEqual(summary["validation_path"], str(self.validation_path))
        self.assertEqual(validation["input_filename"], "publication_raw.csv")
        self.assertNotIn("/Users/", json.dumps(validation))
        self.assertEqual(validation["row_count"], 4)
        self.assertEqual(validation["asset_count"], 2)
        self.assertEqual(validation["vendor_symbol_count"], 1)
        self.assertEqual(validation["member_row_count"], 2)
        self.assertEqual(validation["nonmember_row_count"], 2)
        self.assertEqual(validation["membership_values"], [0, 1])
        self.assertEqual(validation["duplicate_asset_date_rows"], 0)
        self.assertEqual(validation["active_asset_count"], 1)
        self.assertEqual(validation["delisted_asset_count"], 1)
        self.assertEqual(validation["daily_return_null_count"], 2)
        self.assertEqual(validation["nonfirst_daily_return_null_count"], 0)
        self.assertEqual(validation["adj_close_close_mismatch_count"], 0)
        self.assertEqual(validation["dollar_volume_turnover_mismatch_count"], 0)
        self.assertEqual(validation["output_parquet_path"], "asx200_point_in_time_panel.parquet")
        self.assertEqual(validation["price_adjustment_convention"], "Norgate TotalReturn default; adj_close = close")
        self.assertEqual(validation["liquidity_convention"], "dollar_volume = turnover")
        self.assertEqual(validation["identity_convention"], "asset_id")

    def test_build_validation_payload_uses_repo_relative_output_path(self) -> None:
        raw = make_publication_raw_panel()
        source_summary = validate_raw_panel(raw)
        transformed = transform_publication_panel(raw)
        output_summary = validate_transformed_panel(raw, transformed, source_summary)
        payload = build_validation_payload(
            Path("norgate_asx200_publication_raw.csv"),
            publication_builder.DEFAULT_OUTPUT_PATH,
            transformed,
            output_summary,
        )
        self.assertEqual(
            payload["output_parquet_path"],
            "data/licensed/asx200_point_in_time_panel.parquet",
        )

    def test_build_publication_panel_sanitizes_external_output_path(self) -> None:
        raw = make_publication_raw_panel()
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_path = temp_dir / "publication_raw.csv"
            output_path = temp_dir / "private" / "custom_panel.parquet"
            raw.to_csv(input_path, index=False)
            build_publication_panel(
                input_path,
                output_path,
                validation_output_path=self.validation_path,
            )
            validation = json.loads(self.validation_path.read_text())
        self.assertEqual(validation["output_parquet_path"], "custom_panel.parquet")


if __name__ == "__main__":
    unittest.main()
