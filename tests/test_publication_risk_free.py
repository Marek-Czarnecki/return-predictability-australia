from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.publication_risk_free import (
    build_publication_risk_free,
    normalize_cash_rate_schedule,
    validate_overlap_with_existing_tri,
)


class PublicationRiskFreeTests(unittest.TestCase):
    def test_weekend_accrual_uses_calendar_day_gap(self):
        schedule = pd.DataFrame(
            {
                "Effective Date": ["01 Jan 2020"],
                "Cash rate target %": [3.65],
            }
        )
        dates = pd.to_datetime(["2020-01-03", "2020-01-06"])
        result = build_publication_risk_free(dates, schedule)
        daily = 0.0365 / 365.0
        expected = (1.0 + daily) ** 3 - 1.0
        self.assertAlmostEqual(result.loc[1, "risk_free_return"], expected, places=14)

    def test_effective_date_applies_without_extra_shift(self):
        schedule = pd.DataFrame(
            {
                "Effective Date": ["01 Jan 2020", "06 Jan 2020"],
                "Cash rate target %": [3.65, 7.30],
            }
        )
        dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
        result = build_publication_risk_free(dates, schedule)
        expected_weekend = (1.0 + 0.0365 / 365.0) ** 2 * (1.0 + 0.073 / 365.0) - 1.0
        self.assertAlmostEqual(result.loc[1, "risk_free_return"], expected_weekend, places=14)
        self.assertAlmostEqual(result.loc[2, "risk_free_return"], 0.073 / 365.0, places=14)

    def test_schedule_normalization(self):
        raw = pd.DataFrame(
            {
                "Effective Date": ["05 Mar 2003", "02 Apr 2003"],
                "Change% points": [0.0, 0.0],
                "Cash rate target %": [4.75, 4.75],
            }
        )
        result = normalize_cash_rate_schedule(raw)
        self.assertEqual(list(result.columns), ["effective_date", "cash_rate_target_percent"])
        self.assertEqual(float(result.iloc[0]["cash_rate_target_percent"]), 4.75)

    def test_overlap_validation(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        publication = pd.DataFrame(
            {"trade_date": dates, "risk_free_return": [0.0, 0.001, 0.002]}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.csv"
            pd.DataFrame(
                {"trade_date": dates, "risk_free_return": [0.0, 0.001, 0.002001]}
            ).to_csv(path, index=False)
            audit = validate_overlap_with_existing_tri(publication, path)
        self.assertEqual(int(audit.iloc[0]["overlap_count"]), 3)
        self.assertAlmostEqual(float(audit.iloc[0]["max_abs_return_difference"]), 0.000001)


if __name__ == "__main__":
    unittest.main()
