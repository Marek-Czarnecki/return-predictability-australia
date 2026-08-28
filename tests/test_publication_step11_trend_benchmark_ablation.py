from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.publication_step11_trend_benchmark_ablation import (
    build_benchmark_fold_comparison,
    export_step11_benchmark_ablation,
    run_step11_benchmark_ablation,
)
from strategies.publication_step11_trend_ablation import run_publication_walk_forward_on_explicit_folds
from strategies.walk_forward import WalkForwardFold


class PublicationStep11TrendBenchmarkAblationTests(unittest.TestCase):
    @staticmethod
    def _prices() -> pd.DataFrame:
        dates = pd.bdate_range("2015-01-02", "2026-07-20")
        frames = []
        for asset_id, slope in ((101, 0.0004), (202, 0.0002), (303, -0.0001)):
            steps = np.arange(len(dates), dtype=float)
            frame = pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "ticker_code": f"T{asset_id}",
                    "trade_date": dates,
                    "adj_close": 100.0 * np.exp(slope * steps),
                    "dollar_volume": 10_000_000.0 - asset_id,
                    "member_of_universe": True,
                }
            )
            frame["daily_return"] = frame["adj_close"].pct_change(fill_method=None)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _benchmark(prices: pd.DataFrame, daily_return: float) -> pd.DataFrame:
        dates = pd.Index(sorted(prices["trade_date"].unique()))
        values = pd.Series(daily_return, index=dates)
        return pd.DataFrame({"trade_date": dates, "benchmark_return": values.to_numpy()})

    @staticmethod
    def _fold() -> WalkForwardFold:
        return WalkForwardFold(
            fold_id="diagnostic_01",
            formation_start=pd.Timestamp("2016-07-25"),
            formation_end=pd.Timestamp("2019-07-24"),
            evaluation_start=pd.Timestamp("2019-07-25"),
            evaluation_end=pd.Timestamp("2020-07-24"),
        )

    def test_comparison_records_benchmark_effect_and_parameter_change(self):
        prices = self._prices()
        xjoa = self._benchmark(prices, 0.0002)
        stw = self._benchmark(prices, 0.0001)
        fold = self._fold()
        xjoa_result = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, xjoa, [fold]
        )
        stw_result = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, stw, [fold]
        )
        comparison = build_benchmark_fold_comparison(xjoa_result, stw_result)
        self.assertEqual(len(comparison), 1)
        self.assertIn("benchmark_effect_nav_difference", comparison.columns)
        self.assertIn("parameter_selection_changed", comparison.columns)
        expected = (
            comparison.loc[0, "stw_net_excess_nav_difference"]
            - comparison.loc[0, "xjoa_net_excess_nav_difference"]
        )
        self.assertAlmostEqual(comparison.loc[0, "benchmark_effect_nav_difference"], expected)

    def test_benchmark_change_is_applied_to_formation_and_evaluation(self):
        prices = self._prices()
        xjoa = self._benchmark(prices, 0.0003)
        stw = self._benchmark(prices, 0.0)
        fold = self._fold()
        xjoa_result = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, xjoa, [fold]
        )
        stw_result = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, stw, [fold]
        )
        self.assertNotEqual(
            xjoa_result.fold_table.loc[0, "formation_objective_value"],
            stw_result.fold_table.loc[0, "formation_objective_value"],
        )
        self.assertNotEqual(
            xjoa_result.fold_summary.loc[0, "net_excess_nav_difference"],
            stw_result.fold_summary.loc[0, "net_excess_nav_difference"],
        )

    def test_export_records_diagnostic_contract(self):
        prices = self._prices()
        xjoa = self._benchmark(prices, 0.0002)
        stw = self._benchmark(prices, 0.0001)
        fold = self._fold()
        xjoa_result = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, xjoa, [fold]
        )
        stw_result = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, stw, [fold]
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_step11_benchmark_ablation(
                xjoa_result,
                stw_result,
                Path(tmp),
                xjoa_path=Path("xjoa.csv"),
                stw_path=Path("stw.csv"),
            )
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            comparison = pd.read_csv(paths["comparison"])
        self.assertEqual(metadata["step"], "11.1.3")
        self.assertEqual(metadata["analysis_role"], "diagnostic_ablation")
        self.assertFalse(metadata["confirmatory"])
        self.assertTrue(metadata["benchmark_changed_in_formation_selection"])
        self.assertTrue(metadata["benchmark_changed_in_evaluation"])
        self.assertTrue(metadata["not_part_of_primary_holm_family"])
        self.assertIn("parameter_selection_changed", comparison.columns)

    def test_public_runner_returns_two_seven_fold_results(self):
        prices = self._prices()
        xjoa = self._benchmark(prices, 0.0002)
        stw = self._benchmark(prices, 0.0001)
        xjoa_result, stw_result = run_step11_benchmark_ablation(prices, xjoa, stw)
        self.assertEqual(len(xjoa_result.fold_summary), 7)
        self.assertEqual(len(stw_result.fold_summary), 7)
        self.assertEqual(
            xjoa_result.fold_summary["fold_id"].tolist(),
            stw_result.fold_summary["fold_id"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
