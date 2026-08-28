from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.publication_step11_trend_ablation import (
    FROZEN_CAPSTONE_COMMIT,
    export_step11_common_period_result,
    frozen_capstone_trend_folds,
    run_publication_walk_forward_on_explicit_folds,
)
from strategies.walk_forward import WalkForwardFold


class PublicationStep11TrendAblationTests(unittest.TestCase):
    @staticmethod
    def _prices() -> pd.DataFrame:
        dates = pd.bdate_range("2000-03-31", "2004-06-30")
        frames = []
        for asset_id, slope, dollar_volume in (
            (101, 0.0005, 10_000_000.0),
            (202, 0.0002, 4_000_000.0),
            (303, -0.0001, 1_000_000.0),
        ):
            steps = np.arange(len(dates), dtype=float)
            adj_close = 100.0 * np.exp(slope * steps)
            frame = pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "ticker_code": f"T{asset_id}",
                    "trade_date": dates,
                    "adj_close": adj_close,
                    "dollar_volume": dollar_volume,
                    "member_of_universe": True,
                }
            )
            frame["daily_return"] = frame["adj_close"].pct_change(fill_method=None)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _benchmark(prices: pd.DataFrame) -> pd.DataFrame:
        dates = pd.Index(sorted(prices["trade_date"].unique()))
        returns = pd.Series(0.0001, index=dates, dtype=float)
        returns.iloc[0] = np.nan
        return pd.DataFrame(
            {
                "trade_date": dates,
                "benchmark_return": returns.to_numpy(),
            }
        )

    def test_frozen_schedule_matches_committed_capstone_windows(self):
        folds = frozen_capstone_trend_folds()
        self.assertEqual(len(folds), 7)
        self.assertEqual(folds[0].formation_start, pd.Timestamp("2016-07-25"))
        self.assertEqual(folds[0].evaluation_start, pd.Timestamp("2019-07-25"))
        self.assertEqual(folds[0].evaluation_end, pd.Timestamp("2020-07-24"))
        self.assertEqual(folds[-1].formation_start, pd.Timestamp("2022-07-25"))
        self.assertEqual(folds[-1].evaluation_start, pd.Timestamp("2025-07-25"))
        self.assertEqual(folds[-1].evaluation_end, pd.Timestamp("2026-07-20"))

    def test_explicit_fold_runner_preserves_publication_contract(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        fold = WalkForwardFold(
            fold_id="diagnostic_01",
            formation_start=pd.Timestamp("2000-03-31"),
            formation_end=pd.Timestamp("2003-03-31"),
            evaluation_start=pd.Timestamp("2003-04-01"),
            evaluation_end=pd.Timestamp("2004-03-31"),
        )

        result = run_publication_walk_forward_on_explicit_folds(
            strategy_name="trend_following",
            prices=prices,
            benchmark_returns=benchmark,
            folds=[fold],
        )

        self.assertEqual(len(result.fold_table), 1)
        actual = result.fold_table.iloc[0]
        self.assertEqual(actual["fold_id"], "diagnostic_01")
        self.assertEqual(actual["formation_start"], fold.formation_start)
        self.assertEqual(actual["evaluation_start"], fold.evaluation_start)
        self.assertEqual(actual["evaluation_end"], fold.evaluation_end)
        self.assertEqual(actual["identity_col"], "asset_id")
        self.assertEqual(actual["eligibility_col"], "eligible_to_trade")
        self.assertEqual(actual["selection_objective"], "net_excess_nav_vs_benchmark")
        chosen = str(actual["chosen_parameters"])
        self.assertIn("cost_scenario=base", chosen)
        self.assertIn("min_history=220", chosen)
        self.assertFalse(result.fold_summary.empty)
        self.assertIn("net_excess_nav_difference", result.fold_summary.columns)

    def test_export_is_diagnostic_and_uses_dedicated_step11_names(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        fold = WalkForwardFold(
            fold_id="diagnostic_01",
            formation_start=pd.Timestamp("2000-03-31"),
            formation_end=pd.Timestamp("2003-03-31"),
            evaluation_start=pd.Timestamp("2003-04-01"),
            evaluation_end=pd.Timestamp("2004-03-31"),
        )
        result = run_publication_walk_forward_on_explicit_folds(
            strategy_name="trend_following",
            prices=prices,
            benchmark_returns=benchmark,
            folds=[fold],
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = export_step11_common_period_result(result, Path(tmp))
            self.assertTrue(paths["summary"].name.startswith("publication_step11_"))
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        self.assertEqual(metadata["step"], "11.1.1")
        self.assertEqual(metadata["analysis_role"], "diagnostic_ablation")
        self.assertFalse(metadata["confirmatory"])
        self.assertFalse(metadata["parameter_grid_changed"])
        self.assertFalse(metadata["strategy_rule_changed"])
        self.assertTrue(metadata["not_part_of_primary_holm_family"])
        self.assertTrue(metadata["full_pre_window_history_retained"])
        self.assertEqual(metadata["frozen_capstone_commit"], FROZEN_CAPSTONE_COMMIT)

    def test_rejects_empty_explicit_fold_schedule(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        with self.assertRaises(ValueError):
            run_publication_walk_forward_on_explicit_folds(
                strategy_name="trend_following",
                prices=prices,
                benchmark_returns=benchmark,
                folds=[],
            )


if __name__ == "__main__":
    unittest.main()
