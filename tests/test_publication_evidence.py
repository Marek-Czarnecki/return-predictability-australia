from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategies.publication_evidence import (
    CRITICAL_STEP8_ARTIFACTS,
    freeze_publication_step8_evidence,
)


class PublicationEvidenceTests(unittest.TestCase):
    def _build_fixture(self, root: Path) -> None:
        for name in CRITICAL_STEP8_ARTIFACTS:
            path = root / name
            if name == "publication_validation_metadata.json":
                path.write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "check_count": 26,
                            "failed_check_count": 0,
                            "risk_free_coverage_status": "complete",
                        }
                    ),
                    encoding="utf-8",
                )
            elif name == "publication_primary_inference_metadata.json":
                path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "primary_hypothesis_count": 4,
                            "multiple_testing_method": "holm",
                            "reject_count_after_holm_0_05": 0,
                            "walk_forward_primary_metric": "net_excess_nav_difference",
                            "tax_loss_primary_metric": "abnormal_net_return_difference",
                        }
                    ),
                    encoding="utf-8",
                )
            elif name == "publication_primary_inference.csv":
                pd.DataFrame(
                    {
                        "analysis_key": ["trend", "mean", "pairs", "tax"],
                        "adjusted_p_value": [1.0, 1.0, 1.0, 0.7],
                    }
                ).to_csv(path, index=False)
            elif path.suffix == ".json":
                path.write_text("{}", encoding="utf-8")
            else:
                path.write_text("x\n", encoding="utf-8")

    def test_freeze_requires_passed_validation_and_complete_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            result = freeze_publication_step8_evidence(root)
            self.assertEqual(result.metadata["status"], "frozen")
            self.assertEqual(result.metadata["validation_check_count"], 26)
            self.assertEqual(result.metadata["primary_hypothesis_count"], 4)
            self.assertEqual(result.metadata["multiple_testing_method"], "holm")
            self.assertGreaterEqual(len(result.manifest), len(CRITICAL_STEP8_ARTIFACTS))
            self.assertTrue(result.manifest["sha256"].str.len().eq(64).all())

    def test_freeze_rejects_failed_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            metadata_path = root / "publication_validation_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["status"] = "blocked"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                freeze_publication_step8_evidence(root)

    def test_freeze_rejects_missing_critical_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            (root / "publication_primary_inference.csv").unlink()
            with self.assertRaises(ValueError):
                freeze_publication_step8_evidence(root)


if __name__ == "__main__":
    unittest.main()
