from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_publication_walk_forward import export_publication_walk_forward_result
from strategies.publication_walk_forward import PublicationWalkForwardResult


class RunPublicationWalkForwardScriptTests(unittest.TestCase):
    def test_exports_expected_publication_artifacts_and_metadata(self):
        fold_table = pd.DataFrame(
            [
                {
                    "fold_id": "fold_01",
                    "strategy_name": "trend_following",
                }
            ]
        )
        daily = pd.DataFrame(
            {
                "fold_id": ["fold_01", "fold_01"],
                "trade_date": pd.to_datetime(["2003-03-31", "2003-04-01"]),
                "net_return": [0.01, 0.02],
                "benchmark_return": [0.005, 0.006],
                "benchmark_observed": [True, True],
                "excess_return": [0.005, 0.014],
            }
        )
        summary = pd.DataFrame([{"fold_id": "fold_01", "annualized_return": 0.10}])
        liquidity = pd.DataFrame(
            [{"fold_id": "fold_01", "asset_id": 101, "liquidity_tier": "high"}]
        )
        result = PublicationWalkForwardResult(
            fold_table=fold_table,
            fold_daily_results=daily,
            fold_summary=summary,
            liquidity_diagnostics=liquidity,
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = export_publication_walk_forward_result(
                "trend_following",
                result,
                Path(tmp),
            )

            self.assertEqual(set(paths), {"folds", "daily", "summary", "liquidity", "metadata"})
            for path in paths.values():
                self.assertTrue(path.exists())

            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["strategy_name"], "trend_following")
            self.assertEqual(metadata["fold_count"], 1)
            self.assertEqual(metadata["daily_row_count"], 2)
            self.assertEqual(metadata["benchmark_missing_count"], 0)
            self.assertEqual(metadata["net_return_null_count"], 0)
            self.assertEqual(metadata["identity_col"], "asset_id")
            self.assertEqual(metadata["eligibility_col"], "eligible_to_trade")


if __name__ == "__main__":
    unittest.main()
