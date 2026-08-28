from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.publication_validation import (
    build_risk_free_coverage_audit,
    build_walk_forward_metric_audit,
)


class PublicationValidationTests(unittest.TestCase):
    def test_metric_audit_distinguishes_sum_from_compounded_relative_return(self):
        daily = pd.DataFrame(
            {
                "fold_id": ["fold_01", "fold_01"],
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "net_return": [0.10, -0.05],
                "benchmark_return": [0.02, 0.01],
                "excess_return": [0.08, -0.06],
                "benchmark_observed": [True, True],
            }
        )
        summary = pd.DataFrame(
            [{"fold_id": "fold_01", "total_net_excess_return": 0.02}]
        )

        result = build_walk_forward_metric_audit("trend_following", daily, summary)
        row = result.iloc[0]
        self.assertAlmostEqual(row["sum_daily_excess_return"], 0.02)
        self.assertTrue(row["legacy_metric_matches_sum_daily_excess"])
        self.assertAlmostEqual(row["strategy_total_return"], 0.045)
        self.assertAlmostEqual(row["benchmark_total_return"], 0.0302)
        self.assertAlmostEqual(row["net_excess_nav_difference"], 0.0148)
        self.assertNotAlmostEqual(
            row["sum_daily_excess_return"], row["net_excess_nav_difference"]
        )

    def test_risk_free_coverage_reports_incomplete_start(self):
        metrics = pd.DataFrame(
            {
                "evaluation_start": pd.to_datetime(["2003-03-31"]),
                "evaluation_end": pd.to_datetime(["2026-08-27"]),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rf.csv"
            pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2011-01-04", "2026-08-27"]),
                    "risk_free_return": [0.0, 0.0001],
                }
            ).to_csv(path, index=False)
            result = build_risk_free_coverage_audit(metrics, path)
        self.assertEqual(result.iloc[0]["coverage_status"], "incomplete")

    def test_risk_free_coverage_reports_complete(self):
        metrics = pd.DataFrame(
            {
                "evaluation_start": pd.to_datetime(["2003-03-31"]),
                "evaluation_end": pd.to_datetime(["2026-08-27"]),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rf.csv"
            pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2000-01-03", "2026-08-27"]),
                    "risk_free_return": [0.0, 0.0001],
                }
            ).to_csv(path, index=False)
            result = build_risk_free_coverage_audit(metrics, path)
        self.assertEqual(result.iloc[0]["coverage_status"], "complete")


if __name__ == "__main__":
    unittest.main()
