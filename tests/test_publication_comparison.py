from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategies.publication_comparison import build_publication_comparison


class PublicationComparisonTests(unittest.TestCase):
    def _build_fixture(self, frozen: Path, publication: Path) -> None:
        frozen.mkdir(parents=True, exist_ok=True)
        publication.mkdir(parents=True, exist_ok=True)

        for strategy, frozen_effect, publication_effect in (
            ("trend_following", 0.10, 0.01),
            ("mean_reversion", -0.40, -0.50),
            ("pairs_trading", -0.15, -0.10),
        ):
            pd.DataFrame(
                {
                    "fold_id": [f"fold_{index:02d}" for index in range(1, 8)],
                    "absolute_total_return": [frozen_effect + 0.05] * 7,
                    "benchmark_total_return": [0.05] * 7,
                    "total_net_excess_return": [frozen_effect] * 7,
                }
            ).to_csv(frozen / f"{strategy}_walk_forward_summary.csv", index=False)

            # Match the real publication summary schema: publication summaries carry the
            # preferred net_excess_nav_difference directly and do not carry
            # benchmark_total_return as a summary column.
            pd.DataFrame(
                {
                    "fold_id": [f"fold_{index:02d}" for index in range(1, 25)],
                    "absolute_total_return": [publication_effect + 0.05] * 24,
                    "total_net_excess_return": [publication_effect - 0.01] * 24,
                    "net_excess_nav_difference": [publication_effect] * 24,
                }
            ).to_csv(
                publication / f"publication_{strategy}_walk_forward_summary.csv",
                index=False,
            )

        pd.DataFrame(
            {
                "analysis_key": [
                    "trend_following",
                    "mean_reversion",
                    "pairs_trading",
                    "tax_loss_event_minus_control",
                ],
                "effect_estimate": [0.10, -0.40, -0.15, 0.04],
                "effect_unit": ["folds", "folds", "folds", "ticker_events"],
                "sample_size": [7, 7, 7, 169],
                "p_value": [0.01, 1.0, 1.0, 0.003],
                "adjusted_p_value": [0.04, 1.0, 1.0, 0.012],
                "reject_null_0_05": [True, False, False, True],
            }
        ).to_csv(frozen / "statistical_inference_summary.csv", index=False)

        pd.DataFrame(
            {
                "mean_return_difference": [0.04],
                "mean_abnormal_return_difference": [0.05],
            }
        ).to_csv(frozen / "tax_loss_selling_summary.csv", index=False)
        pd.DataFrame(
            {
                "analysis_level": ["year_clustered_sign_flip"],
                "mean_return_difference": [0.043],
                "p_value": [0.143],
            }
        ).to_csv(frozen / "tax_loss_selling_year_robustness.csv", index=False)

        pd.DataFrame(
            {
                "analysis_key": [
                    "trend_following",
                    "mean_reversion",
                    "pairs_trading",
                    "tax_loss_selling",
                ],
                "effect_estimate": [0.01, -0.50, -0.10, 0.025],
                "primary_metric": [
                    "net_excess_nav_difference",
                    "net_excess_nav_difference",
                    "net_excess_nav_difference",
                    "abnormal_net_return_difference",
                ],
                "sample_unit": [
                    "evaluation_folds",
                    "evaluation_folds",
                    "evaluation_folds",
                    "calendar_years",
                ],
                "sample_size": [24, 24, 24, 26],
                "p_value": [0.48, 1.0, 0.99, 0.17],
                "adjusted_p_value": [1.0, 1.0, 1.0, 0.69],
                "reject_null_0_05": [False, False, False, False],
            }
        ).to_csv(publication / "publication_primary_inference.csv", index=False)

        pd.DataFrame(
            {
                "mean_return_difference": [0.012],
                "mean_abnormal_net_return_difference": [0.03],
            }
        ).to_csv(publication / "publication_tax_loss_selling_summary.csv", index=False)

        (publication / "publication_step8_evidence_metadata.json").write_text(
            json.dumps({"status": "frozen", "step": "8", "artifact_count": 40}),
            encoding="utf-8",
        )

    def test_builds_four_strategy_comparison_and_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            publication = root / "publication"
            self._build_fixture(frozen, publication)

            result = build_publication_comparison(frozen, publication)

            self.assertEqual(len(result.comparison), 4)
            self.assertEqual(result.metadata["status"], "complete")
            self.assertEqual(result.metadata["comparison_strategy_count"], 4)
            self.assertGreaterEqual(len(result.attribution), 10)

            trend = result.comparison.loc[
                result.comparison["strategy_family"] == "trend_following"
            ].iloc[0]
            self.assertAlmostEqual(
                trend["frozen_reconstructed_nav_difference_mean"], 0.10
            )
            self.assertAlmostEqual(trend["publication_nav_difference_mean"], 0.01)
            self.assertEqual(trend["comparison_class"], "derived_comparable")
            self.assertEqual(
                trend["evidence_strength_change"], "supported_to_unsupported"
            )

            mean_reversion = result.comparison.loc[
                result.comparison["strategy_family"] == "mean_reversion"
            ].iloc[0]
            self.assertEqual(
                mean_reversion["evidence_strength_change"],
                "unsupported_to_unsupported",
            )

            tax = result.comparison.loc[
                result.comparison["strategy_family"] == "tax_loss_selling"
            ].iloc[0]
            self.assertEqual(tax["comparison_class"], "context_only")
            self.assertEqual(tax["publication_inferential_unit"], "calendar_years")
            self.assertEqual(
                tax["evidence_strength_change"], "supported_to_unsupported"
            )

    def test_rejects_unfrozen_step8_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            publication = root / "publication"
            self._build_fixture(frozen, publication)
            (publication / "publication_step8_evidence_metadata.json").write_text(
                json.dumps({"status": "complete", "step": "8"}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                build_publication_comparison(frozen, publication)

    def test_rejects_wrong_fold_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            publication = root / "publication"
            self._build_fixture(frozen, publication)
            path = frozen / "trend_following_walk_forward_summary.csv"
            frame = pd.read_csv(path).iloc[:-1]
            frame.to_csv(path, index=False)

            with self.assertRaises(ValueError):
                build_publication_comparison(frozen, publication)


if __name__ == "__main__":
    unittest.main()
