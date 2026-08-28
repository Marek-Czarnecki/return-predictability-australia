from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_publication_benchmarks import (
    BUILD_SUMMARY_FILENAME,
    EQUAL_WEIGHT_BENCHMARK_FILENAME,
    EXTERNAL_BENCHMARK_FILENAME,
    build_publication_benchmark_artifacts,
    write_publication_benchmark_artifacts,
)
from strategies.publication_benchmarks import XJOA_ASSET_ID, XJOA_SYMBOL


class PublicationBenchmarkBuildTests(unittest.TestCase):
    def make_xjoa_source(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-05", "2026-01-06"],
                "asset_id": [XJOA_ASSET_ID] * 3,
                "symbol": [XJOA_SYMBOL] * 3,
                "close": [100.0, 102.0, 101.0],
            }
        )

    def make_publication_panel(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2026-01-02",
                        "2026-01-05",
                        "2026-01-06",
                        "2026-01-02",
                        "2026-01-05",
                        "2026-01-06",
                    ]
                ),
                "asset_id": [1, 1, 1, 2, 2, 2],
                "daily_return": [None, 0.02, -0.01, None, 0.04, 0.03],
                "member_of_universe": [1, 1, 1, 1, 1, 1],
            }
        )

    def test_build_artifacts_produces_expected_columns_and_summary(self) -> None:
        external, equal_weight, summary = build_publication_benchmark_artifacts(
            xjoa_source=self.make_xjoa_source(),
            publication_panel=self.make_publication_panel(),
        )

        self.assertEqual(
            list(external.columns),
            ["trade_date", "benchmark_level", "benchmark_return", "benchmark_nav"],
        )
        self.assertEqual(
            list(equal_weight.columns),
            [
                "trade_date",
                "equal_weight_return",
                "equal_weight_nav",
                "member_count",
                "observable_member_return_count",
                "missing_member_return_count",
                "missing_member_return_fraction",
            ],
        )
        self.assertEqual(summary["external_benchmark"]["row_count"], 3)
        self.assertEqual(summary["equal_weight_benchmark"]["row_count"], 3)
        self.assertEqual(summary["external_benchmark"]["null_return_count"], 1)
        self.assertEqual(summary["equal_weight_benchmark"]["null_return_count"], 1)

    def test_write_artifacts_uses_locked_filenames(self) -> None:
        external, equal_weight, summary = build_publication_benchmark_artifacts(
            xjoa_source=self.make_xjoa_source(),
            publication_panel=self.make_publication_panel(),
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = write_publication_benchmark_artifacts(
                external=external,
                equal_weight=equal_weight,
                summary=summary,
                output_dir=output_dir,
            )

            self.assertEqual(paths["external_benchmark"].name, EXTERNAL_BENCHMARK_FILENAME)
            self.assertEqual(
                paths["equal_weight_benchmark"].name,
                EQUAL_WEIGHT_BENCHMARK_FILENAME,
            )
            self.assertEqual(paths["build_summary"].name, BUILD_SUMMARY_FILENAME)
            self.assertTrue(paths["external_benchmark"].exists())
            self.assertTrue(paths["equal_weight_benchmark"].exists())
            self.assertTrue(paths["build_summary"].exists())

    def test_build_summary_is_deterministic_for_same_inputs(self) -> None:
        _, _, first_summary = build_publication_benchmark_artifacts(
            xjoa_source=self.make_xjoa_source(),
            publication_panel=self.make_publication_panel(),
        )
        _, _, second_summary = build_publication_benchmark_artifacts(
            xjoa_source=self.make_xjoa_source(),
            publication_panel=self.make_publication_panel(),
        )

        self.assertEqual(first_summary, second_summary)


if __name__ == "__main__":
    unittest.main()
