from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "evidence"


class V10MechanismEvidenceTests(unittest.TestCase):
    def test_exact_fold_decomposition_and_headline_means(self):
        frame = pd.read_csv(EVIDENCE / "publication_trend_2x2_decomposition.csv")
        self.assertEqual(len(frame), 7)

        a = frame["a_pit_pitparams_nav_difference"].astype(float)
        b = frame["b_retro_retroparams_nav_difference"].astype(float)
        total = frame["total_universe_treatment_effect_nav_difference"].astype(float)
        universe = frame["shapley_universe_component_nav_difference"].astype(float)
        parameter = frame["shapley_parameter_selection_component_nav_difference"].astype(float)

        np.testing.assert_allclose(total, b - a, atol=1e-12)
        np.testing.assert_allclose(universe + parameter, total, atol=1e-12)
        self.assertAlmostEqual(float(total.mean()), 0.15305148515250488, places=12)
        self.assertAlmostEqual(float(universe.mean()), 0.15170128640783137, places=12)
        self.assertAlmostEqual(float(parameter.mean()), 0.0013501987446734587, places=12)
        self.assertEqual(int((universe > 0).sum()), 7)
        self.assertEqual(int(frame["parameter_selection_changed"].astype(str).str.lower().eq("true").sum()), 5)

    def test_public_concentration_summary_contains_only_aggregate_contract(self):
        summary = json.loads((EVIDENCE / "publication_trend_concentration_summary.json").read_text())
        self.assertEqual(summary["asset_count"], 718)
        self.assertEqual(summary["asset_count_to_50pct_absolute_share"], 28)
        self.assertEqual(summary["asset_count_to_80pct_absolute_share"], 81)
        self.assertAlmostEqual(summary["absolute_contribution_hhi"], 0.012697823907628782, places=12)
        self.assertNotIn("assets", summary)
        self.assertNotIn("asset_ids", summary)


if __name__ == "__main__":
    unittest.main()
