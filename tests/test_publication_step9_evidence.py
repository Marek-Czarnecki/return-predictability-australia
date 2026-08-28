from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategies.publication_comparison import (
    ATTRIBUTION_NAME,
    COMPARISON_NAME,
    METADATA_NAME,
)
from strategies.publication_step9_evidence import (
    validate_and_freeze_publication_step9_evidence,
)


class PublicationStep9EvidenceTests(unittest.TestCase):
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

        step8_artifact = publication / "step8_dummy.csv"
        step8_artifact.write_text("x\n", encoding="utf-8")
        import hashlib
        digest = hashlib.sha256(step8_artifact.read_bytes()).hexdigest()
        pd.DataFrame(
            [{"artifact": "step8_dummy.csv", "bytes": step8_artifact.stat().st_size, "sha256": digest}]
        ).to_csv(publication / "publication_step8_evidence_manifest.csv", index=False)
        (publication / "publication_step8_evidence_metadata.json").write_text(
            json.dumps({"status": "frozen", "step": "8", "artifact_count": 1}),
            encoding="utf-8",
        )

        from strategies.publication_comparison import build_publication_comparison
        result = build_publication_comparison(frozen, publication)
        result.comparison.to_csv(publication / COMPARISON_NAME, index=False)
        result.attribution.to_csv(publication / ATTRIBUTION_NAME, index=False)
        (publication / METADATA_NAME).write_text(
            json.dumps(result.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def test_validates_and_freezes_step9_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            publication = root / "publication"
            self._build_fixture(frozen, publication)

            result = validate_and_freeze_publication_step9_evidence(frozen, publication)
            self.assertEqual(result.metadata["status"], "frozen")
            self.assertEqual(result.metadata["step"], "9")
            self.assertEqual(result.metadata["comparison_strategy_count"], 4)
            self.assertEqual(result.metadata["step8_hash_validation_status"], "passed")
            self.assertEqual(len(result.manifest), 3)
            self.assertTrue(result.manifest["sha256"].str.len().eq(64).all())

    def test_rejects_modified_step8_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            publication = root / "publication"
            self._build_fixture(frozen, publication)
            (publication / "step8_dummy.csv").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_and_freeze_publication_step9_evidence(frozen, publication)

    def test_rejects_tampered_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            publication = root / "publication"
            self._build_fixture(frozen, publication)
            path = publication / COMPARISON_NAME
            frame = pd.read_csv(path)
            frame.loc[frame["strategy_family"] == "trend_following", "publication_primary_value"] = 0.5
            frame.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                validate_and_freeze_publication_step9_evidence(frozen, publication)


if __name__ == "__main__":
    unittest.main()
