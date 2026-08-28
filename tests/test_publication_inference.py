from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategies.publication_inference import (
    PRIMARY_TAX_LOSS_METRIC,
    PRIMARY_WALK_FORWARD_METRIC,
    WALK_FORWARD_STRATEGIES,
    build_publication_primary_inference,
)


class PublicationInferenceTests(unittest.TestCase):
    def test_builds_four_hypotheses_and_uses_locked_primary_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for strategy in WALK_FORWARD_STRATEGIES:
                pd.DataFrame(
                    {
                        "strategy": [strategy] * 3,
                        "fold_id": ["f1", "f2", "f3"],
                        "window_label": ["evaluation"] * 3,
                        PRIMARY_WALK_FORWARD_METRIC: [0.01, 0.02, -0.005],
                    }
                ).to_csv(root / f"publication_{strategy}_walk_forward_summary.csv", index=False)

            pd.DataFrame(
                {
                    "year": [2020, 2020, 2021, 2021],
                    "net_return_difference": [-0.9, -0.9, -0.9, -0.9],
                    PRIMARY_TAX_LOSS_METRIC: [0.10, 0.20, 0.30, 0.10],
                }
            ).to_csv(root / "publication_tax_loss_selling_event_study.csv", index=False)

            result = build_publication_primary_inference(root)
            primary = result.primary_inference.set_index("analysis_key")

            self.assertEqual(len(primary), 4)
            self.assertEqual(set(primary.index), set(WALK_FORWARD_STRATEGIES) | {"tax_loss_selling"})
            self.assertTrue((primary["multiple_testing_method"] == "holm").all())
            self.assertEqual(int(primary.loc["tax_loss_selling", "sample_size"]), 2)
            self.assertAlmostEqual(float(primary.loc["tax_loss_selling", "effect_estimate"]), 0.175)
            self.assertEqual(
                primary.loc["tax_loss_selling", "source_artifact"],
                "publication_tax_loss_selling_event_study.csv",
            )

    def test_tax_loss_primary_is_benchmark_adjusted_not_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for strategy in WALK_FORWARD_STRATEGIES:
                pd.DataFrame(
                    {
                        "strategy": [strategy],
                        "fold_id": ["f1"],
                        "window_label": ["evaluation"],
                        PRIMARY_WALK_FORWARD_METRIC: [0.0],
                    }
                ).to_csv(root / f"publication_{strategy}_walk_forward_summary.csv", index=False)

            pd.DataFrame(
                {
                    "year": [2020, 2021],
                    "net_return_difference": [-1.0, -1.0],
                    PRIMARY_TAX_LOSS_METRIC: [0.2, 0.4],
                }
            ).to_csv(root / "publication_tax_loss_selling_event_study.csv", index=False)

            result = build_publication_primary_inference(root)
            tax = result.primary_inference.loc[
                result.primary_inference["analysis_key"] == "tax_loss_selling"
            ].iloc[0]
            self.assertAlmostEqual(float(tax["effect_estimate"]), 0.3)
            self.assertEqual(tax["sample_unit"], "calendar_years")


if __name__ == "__main__":
    unittest.main()
