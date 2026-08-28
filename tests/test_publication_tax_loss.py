from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_publication_tax_loss import export_publication_tax_loss_result
from strategies.publication_tax_loss import (
    PublicationTaxLossResult,
    _strict_window_return,
    run_publication_tax_loss_event_study,
)


class PublicationTaxLossTests(unittest.TestCase):
    def test_strict_window_return_rejects_missing_observation(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        series = pd.Series([0.01, np.nan, 0.02], index=dates)
        self.assertTrue(np.isnan(_strict_window_return(series, dates)))

    def test_selection_uses_asset_id_history_and_point_in_time_membership(self):
        dates = pd.bdate_range("2020-01-01", periods=900)
        rows = []
        for asset_id, ticker, start, member_on_selection in [
            (101, "AAA", 100.0, True),
            (202, "BBB", 100.0, False),
            (303, "CCC", 100.0, True),
        ]:
            close = np.linspace(start, start * 0.5 if asset_id == 101 else start * 1.2, len(dates))
            member = np.ones(len(dates), dtype=bool)
            if not member_on_selection:
                member[:] = False
            frame = pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "ticker_code": ticker,
                    "trade_date": dates,
                    "adj_close": close,
                    "daily_return": pd.Series(close).pct_change(fill_method=None).to_numpy(),
                    "dollar_volume": 1_000_000.0 + asset_id,
                    "member_of_universe": member,
                }
            )
            rows.append(frame)
        prices = pd.concat(rows, ignore_index=True)
        benchmark = pd.DataFrame(
            {"trade_date": dates, "benchmark_return": np.zeros(len(dates), dtype=float)}
        )

        result = run_publication_tax_loss_event_study(prices, benchmark)

        if not result.event_study.empty:
            self.assertFalse(result.event_study["asset_id"].eq(202).any())
            self.assertTrue(result.event_study["asset_id"].isin([101, 303]).all())
            self.assertTrue(result.event_study["complete_event_window"].all())
            self.assertTrue(result.event_study["complete_control_window"].all())

    def test_symmetric_cost_leaves_event_control_difference_unchanged(self):
        dates = pd.bdate_range("2019-01-01", periods=1000)
        close_a = np.linspace(200.0, 80.0, len(dates))
        close_b = np.linspace(100.0, 130.0, len(dates))
        prices = pd.concat(
            [
                _asset_frame(1, "AAA", dates, close_a, 5_000_000.0),
                _asset_frame(2, "BBB", dates, close_b, 1_000_000.0),
            ],
            ignore_index=True,
        )
        benchmark = pd.DataFrame(
            {"trade_date": dates, "benchmark_return": np.zeros(len(dates), dtype=float)}
        )

        result = run_publication_tax_loss_event_study(prices, benchmark)
        complete = result.event_study.dropna(
            subset=["return_difference", "net_return_difference"]
        )
        self.assertFalse(complete.empty)
        np.testing.assert_allclose(
            complete["return_difference"].to_numpy(dtype=float),
            complete["net_return_difference"].to_numpy(dtype=float),
            atol=1e-12,
        )

    def test_export_writes_expected_artifacts(self):
        result = PublicationTaxLossResult(
            event_study=pd.DataFrame([{"year": 2024, "asset_id": 1}]),
            summary=pd.DataFrame(
                [{"complete_matched_observation_count": 1, "year_count": 1}]
            ),
            year_robustness=pd.DataFrame([{"analysis_level": "overall_event_mean"}]),
            liquidity_diagnostics=pd.DataFrame([{"year": 2024, "asset_id": 1}]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_publication_tax_loss_result(result, Path(tmp))
            self.assertEqual(
                set(paths), {"events", "summary", "robustness", "liquidity", "metadata"}
            )
            self.assertTrue(all(path.exists() for path in paths.values()))


def _asset_frame(
    asset_id: int,
    ticker: str,
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    dollar_volume: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": asset_id,
            "ticker_code": ticker,
            "trade_date": dates,
            "adj_close": close,
            "daily_return": pd.Series(close).pct_change(fill_method=None).to_numpy(),
            "dollar_volume": dollar_volume,
            "member_of_universe": True,
        }
    )


if __name__ == "__main__":
    unittest.main()
