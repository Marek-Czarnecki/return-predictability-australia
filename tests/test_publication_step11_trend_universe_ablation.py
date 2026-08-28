from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.publication_step11_trend_universe_ablation import (
    build_retrospective_reference_universe_panel,
    export_step11_universe_ablation,
    select_norgate_reference_date_universe,
)
from strategies.publication_step11_trend_ablation import (
    run_publication_walk_forward_on_explicit_folds,
)
from strategies.walk_forward import WalkForwardFold


class PublicationStep11TrendUniverseAblationTests(unittest.TestCase):
    @staticmethod
    def _prices() -> pd.DataFrame:
        dates = pd.bdate_range("2000-03-31", "2004-06-30")
        frames = []
        specs = (
            (101, "AAA", 0.0005, True),
            (202, "BBB", 0.0002, True),
            (303, "CCC", -0.0001, False),
        )
        for asset_id, ticker, slope, final_member in specs:
            steps = np.arange(len(dates), dtype=float)
            member = np.zeros(len(dates), dtype=bool)
            if final_member:
                member[len(dates) // 2 :] = True
            frame = pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "ticker_code": ticker,
                    "vendor_symbol": f"{ticker}.au",
                    "security_name": f"Security {ticker}",
                    "trade_date": dates,
                    "adj_close": 100.0 * np.exp(slope * steps),
                    "dollar_volume": 5_000_000.0,
                    "member_of_universe": member,
                }
            )
            frame["daily_return"] = frame["adj_close"].pct_change(fill_method=None)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _benchmark(prices: pd.DataFrame) -> pd.DataFrame:
        dates = pd.Index(sorted(prices["trade_date"].unique()))
        values = pd.Series(0.0001, index=dates)
        values.iloc[0] = np.nan
        return pd.DataFrame(
            {"trade_date": dates, "benchmark_return": values.to_numpy()}
        )

    def test_reference_date_selection_uses_norgate_membership_only(self):
        prices = self._prices()
        selection = select_norgate_reference_date_universe(
            prices,
            reference_date=pd.Timestamp("2004-06-30"),
        )
        self.assertEqual(set(selection.selected_asset_ids), {101, 202})
        self.assertEqual(selection.reference_date, pd.Timestamp("2004-06-30"))
        self.assertEqual(len(selection.reference_table), 2)
        self.assertIn("security_name", selection.reference_table.columns)

    def test_reference_date_selection_rejects_missing_snapshot_date(self):
        prices = self._prices()
        with self.assertRaises(ValueError):
            select_norgate_reference_date_universe(
                prices,
                reference_date=pd.Timestamp("2004-07-03"),
            )

    def test_retrospective_panel_changes_membership_only(self):
        prices = self._prices()
        retro = build_retrospective_reference_universe_panel(prices, [101, 202])
        self.assertTrue(
            retro.loc[
                retro["asset_id"].isin([101, 202]), "member_of_universe"
            ].all()
        )
        self.assertFalse(
            retro.loc[retro["asset_id"] == 303, "member_of_universe"].any()
        )
        pd.testing.assert_series_equal(prices["adj_close"], retro["adj_close"])
        pd.testing.assert_series_equal(prices["daily_return"], retro["daily_return"])
        pd.testing.assert_series_equal(prices["ticker_code"], retro["ticker_code"])

    def test_export_records_norgate_internal_design_and_unresolved_vendor_coverage(self):
        prices = self._prices()
        benchmark = self._benchmark(prices)
        fold = WalkForwardFold(
            fold_id="diagnostic_01",
            formation_start=pd.Timestamp("2000-03-31"),
            formation_end=pd.Timestamp("2003-03-31"),
            evaluation_start=pd.Timestamp("2003-04-01"),
            evaluation_end=pd.Timestamp("2004-03-31"),
        )
        pit = run_publication_walk_forward_on_explicit_folds(
            "trend_following", prices, benchmark, [fold]
        )
        selection = select_norgate_reference_date_universe(
            prices,
            reference_date=pd.Timestamp("2004-06-30"),
        )
        retro_prices = build_retrospective_reference_universe_panel(
            prices, selection.selected_asset_ids
        )
        retro = run_publication_walk_forward_on_explicit_folds(
            "trend_following", retro_prices, benchmark, [fold]
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = export_step11_universe_ablation(
                pit, retro, selection, Path(tmp)
            )
            metadata = json.loads(
                paths["metadata"].read_text(encoding="utf-8")
            )
            comparison = pd.read_csv(paths["comparison"])
            reference_universe = pd.read_csv(paths["reference_universe"])

        self.assertEqual(metadata["step"], "11.1.2")
        self.assertEqual(metadata["analysis_role"], "diagnostic_ablation")
        self.assertFalse(metadata["confirmatory"])
        self.assertTrue(metadata["not_part_of_primary_holm_family"])
        self.assertFalse(metadata["frozen_yahoo_ticker_reconciliation_used"])
        self.assertEqual(
            metadata["frozen_yahoo_vs_norgate_security_coverage"],
            "unresolved_contributor_not_part_of_this_ablation",
        )
        self.assertIn("universe_effect_nav_difference", comparison.columns)
        self.assertIn("parameter_selection_changed", comparison.columns)
        self.assertEqual(len(reference_universe), 2)


if __name__ == "__main__":
    unittest.main()
