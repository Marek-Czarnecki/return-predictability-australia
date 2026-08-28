from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.publication_step11_trend_cost_ablation import (
    build_cost_fold_comparison,
    export_step11_cost_ablation,
    run_publication_walk_forward_on_explicit_folds_with_flat_cost,
)
from strategies.publication_step11_trend_ablation import run_publication_walk_forward_on_explicit_folds
from strategies.walk_forward import WalkForwardFold


class PublicationStep11TrendCostAblationTests(unittest.TestCase):
    @staticmethod
    def _prices() -> pd.DataFrame:
        dates = pd.bdate_range("2015-01-02", "2020-12-31")
        frames = []
        for asset_id, slope, volume in (
            (101, 0.0006, 20_000_000.0),
            (202, 0.0003, 10_000_000.0),
            (303, -0.0001, 5_000_000.0),
        ):
            steps = np.arange(len(dates), dtype=float)
            frame = pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "ticker_code": f"T{asset_id}",
                    "trade_date": dates,
                    "adj_close": 100.0 * np.exp(slope * steps),
                    "dollar_volume": volume,
                    "member_of_universe": True,
                }
            )
            frame["daily_return"] = frame["adj_close"].pct_change(fill_method=None)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _benchmark(prices: pd.DataFrame) -> pd.DataFrame:
        dates = pd.Index(sorted(prices["trade_date"].unique()))
        return pd.DataFrame(
            {"trade_date": dates, "benchmark_return": np.full(len(dates), 0.0002)}
        )

    @staticmethod
    def _fold() -> WalkForwardFold:
        return WalkForwardFold(
            fold_id="diagnostic_01",
            formation_start=pd.Timestamp("2016-07-25"),
            formation_end=pd.Timestamp("2019-07-24"),
            evaluation_start=pd.Timestamp("2019-07-25"),
            evaluation_end=pd.Timestamp("2020-07-24"),
        )

    def test_zero_cost_override_removes_transaction_costs(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        result = run_publication_walk_forward_on_explicit_folds_with_flat_cost(
            "trend_following",
            prices,
            benchmark,
            [self._fold()],
            turnover_cost_bps=0.0,
        )
        daily = result.fold_daily_results
        self.assertTrue(np.allclose(daily["gross_return"], daily["net_return"], equal_nan=True))

    def test_cost_comparison_records_effect_and_parameter_change(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        fold = self._fold()
        base = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, benchmark, [fold]
        )
        zero = run_publication_walk_forward_on_explicit_folds_with_flat_cost(
            "trend_following", prices, benchmark, [fold], turnover_cost_bps=0.0
        )
        comparison = build_cost_fold_comparison(base, zero)
        self.assertIn("cost_effect_nav_difference", comparison.columns)
        self.assertIn("parameter_selection_changed", comparison.columns)
        expected = (
            comparison.loc[0, "zero_cost_net_excess_nav_difference"]
            - comparison.loc[0, "base_cost_net_excess_nav_difference"]
        )
        self.assertAlmostEqual(comparison.loc[0, "cost_effect_nav_difference"], expected)

    def test_zero_cost_is_applied_during_formation_and_evaluation(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        fold = self._fold()
        base = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, benchmark, [fold]
        )
        zero = run_publication_walk_forward_on_explicit_folds_with_flat_cost(
            "trend_following", prices, benchmark, [fold], turnover_cost_bps=0.0
        )
        self.assertGreaterEqual(
            zero.fold_table.loc[0, "formation_objective_value"],
            base.fold_table.loc[0, "formation_objective_value"],
        )
        self.assertGreaterEqual(
            zero.fold_summary.loc[0, "net_excess_nav_difference"],
            base.fold_summary.loc[0, "net_excess_nav_difference"],
        )

    def test_export_records_diagnostic_contract(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        fold = self._fold()
        base = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, benchmark, [fold]
        )
        zero = run_publication_walk_forward_on_explicit_folds_with_flat_cost(
            "trend_following", prices, benchmark, [fold], turnover_cost_bps=0.0
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_step11_cost_ablation(
                base, zero, Path(tmp), benchmark_path=Path("xjoa.csv")
            )
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            comparison = pd.read_csv(paths["comparison"])
        self.assertEqual(metadata["step"], "11.1.4")
        self.assertEqual(metadata["analysis_role"], "diagnostic_ablation")
        self.assertFalse(metadata["confirmatory"])
        self.assertTrue(metadata["transaction_cost_treatment_changed_in_formation_selection"])
        self.assertTrue(metadata["transaction_cost_treatment_changed_in_evaluation"])
        self.assertTrue(metadata["not_part_of_primary_holm_family"])
        self.assertIn("cost_effect_nav_difference", comparison.columns)


if __name__ == "__main__":
    unittest.main()
