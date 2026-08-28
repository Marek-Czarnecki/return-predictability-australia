from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategies.publication_step11_trend_attribution import (
    build_trend_attribution_table,
    export_trend_attribution,
)


class PublicationStep11TrendAttributionTests(unittest.TestCase):
    def _write_inputs(self, root: Path):
        comparison = root / "comparison.csv"
        pd.DataFrame([
            {
                "strategy_family": "trend_following",
                "frozen_reconstructed_nav_difference_mean": 0.10276428548304055,
                "publication_nav_difference_mean": 0.0008204009098994785,
            }
        ]).to_csv(comparison, index=False)

        payloads = {
            "common.json": {
                "step": "11.1.1",
                "mean_net_excess_nav_difference": -0.041322384896077646,
                "positive_fold_count": 1,
            },
            "universe.json": {
                "step": "11.1.2",
                "mean_universe_effect_nav_difference": 0.15305148515250488,
                "pit_mean_net_excess_nav_difference": -0.041322384896077646,
                "retrospective_mean_net_excess_nav_difference": 0.11172910025642722,
                "pit_positive_fold_count": 1,
                "retrospective_positive_fold_count": 6,
                "parameter_selection_changed_fold_count": 5,
            },
            "benchmark.json": {
                "step": "11.1.3",
                "mean_benchmark_effect_nav_difference": -0.001412579004500203,
                "xjoa_mean_net_excess_nav_difference": -0.041322384896077646,
                "stw_mean_net_excess_nav_difference": -0.04273496390057785,
                "xjoa_positive_fold_count": 1,
                "stw_positive_fold_count": 1,
                "parameter_selection_change_count": 0,
            },
            "cost.json": {
                "step": "11.1.4",
                "mean_cost_effect_nav_difference": 0.015027500322725864,
                "base_mean_net_excess_nav_difference": -0.041322384896077646,
                "zero_mean_net_excess_nav_difference": -0.02629488457335178,
                "base_positive_fold_count": 1,
                "zero_positive_fold_count": 1,
                "parameter_selection_change_count": 1,
            },
        }
        paths = {}
        for name, payload in payloads.items():
            path = root / name
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths[name] = path
        return comparison, paths

    def test_builds_locked_attribution_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison, paths = self._write_inputs(root)
            table, metadata = build_trend_attribution_table(
                comparison,
                paths["common.json"],
                paths["universe.json"],
                paths["benchmark.json"],
                paths["cost.json"],
            )
        classes = table.set_index("design_change")["attribution_class"].to_dict()
        self.assertEqual(classes["sample_period_and_fold_calendar"], "not_explanatory")
        self.assertEqual(classes["point_in_time_membership"], "directly_demonstrated")
        self.assertEqual(classes["benchmark_choice"], "not_explanatory")
        self.assertEqual(classes["transaction_costs"], "directly_demonstrated")
        self.assertEqual(classes["vendor_and_security_coverage"], "unresolved_contributor")
        self.assertFalse(metadata["confirmatory"])

    def test_effect_magnitudes_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison, paths = self._write_inputs(root)
            table, metadata = build_trend_attribution_table(
                comparison,
                paths["common.json"],
                paths["universe.json"],
                paths["benchmark.json"],
                paths["cost.json"],
            )
        indexed = table.set_index("design_change")
        self.assertAlmostEqual(float(indexed.loc["point_in_time_membership", "controlled_effect_nav_difference"]), 0.15305148515250488)
        self.assertAlmostEqual(float(indexed.loc["transaction_costs", "controlled_effect_nav_difference"]), 0.015027500322725864)
        self.assertAlmostEqual(metadata["sample_extension_effect_full_minus_common_period"], 0.04214278580597712)

    def test_statement_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison, paths = self._write_inputs(root)
            _, metadata = build_trend_attribution_table(
                comparison,
                paths["common.json"],
                paths["universe.json"],
                paths["benchmark.json"],
                paths["cost.json"],
            )
        statement = metadata["attribution_statement"]
        self.assertIn("major identified contributor", statement)
        self.assertIn("security coverage remains unresolved", statement)
        self.assertIn("rather than a claim that it fully explains every difference", statement)

    def test_export_writes_two_evidence_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison, paths = self._write_inputs(root)
            table, metadata = build_trend_attribution_table(
                comparison,
                paths["common.json"],
                paths["universe.json"],
                paths["benchmark.json"],
                paths["cost.json"],
            )
            output = root / "out"
            exported = export_trend_attribution(table, metadata, output)
            self.assertTrue(exported["attribution"].exists())
            self.assertTrue(exported["metadata"].exists())
            reread = pd.read_csv(exported["attribution"])
            self.assertEqual(len(reread), 7)


if __name__ == "__main__":
    unittest.main()
