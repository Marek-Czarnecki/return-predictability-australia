from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategies.publication_step12_freeze import (
    STEP12_FREEZE_ARTIFACTS,
    validate_and_freeze_publication_step12_evidence,
)


class PublicationStep12FreezeTests(unittest.TestCase):
    def _fixture(self, root: Path):
        primary = pd.DataFrame(
            [
                {"strategy_family": name, "reject_after_holm_0_05": False}
                for name in (
                    "trend_following",
                    "mean_reversion",
                    "pairs_trading",
                    "tax_loss_selling",
                )
            ]
        )
        primary.to_csv(root / "publication_final_primary_results.csv", index=False)

        metadata = {
            "status": "built_not_frozen",
            "raw_licensed_data_excluded": True,
            "empirical_results_recomputed": False,
            "primary_hypothesis_count": 4,
            "holm_reject_count": 0,
        }
        (root / "publication_final_evidence_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

        evidence_names = []
        for i in range(16):
            name = f"evidence_{i:02d}.txt"
            path = root / name
            path.write_text(f"evidence {i}\n", encoding="utf-8")
            evidence_names.append(name)

        import hashlib

        rows = []
        for name in evidence_names:
            path = root / name
            rows.append(
                {
                    "artifact": name,
                    "source_step": "8" if name != evidence_names[-1] else "11.1.6",
                    "artifact_role": "primary_result",
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        manifest = pd.DataFrame(rows)
        manifest.to_csv(root / "publication_final_evidence_manifest.csv", index=False)

        validation_metadata = {
            "status": "passed",
            "validation_check_count": 10,
            "validation_failed_check_count": 0,
            "upstream_integrity_validation_status": "passed",
            "final_primary_reconciliation_status": "passed",
            "final_metadata_reconciliation_status": "passed",
            "final_manifest_reconciliation_status": "passed",
            "trend_attribution_boundary_status": "passed",
            "empirical_results_recomputed": False,
        }
        return validation_metadata

    def test_builds_frozen_step12_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation_metadata = self._fixture(root)
            fake_validation = type("Validation", (), {"metadata": validation_metadata})()
            with patch(
                "strategies.publication_step12_freeze.validate_publication_step12_scientific_invariants",
                return_value=fake_validation,
            ):
                result = validate_and_freeze_publication_step12_evidence(root)

            self.assertEqual(result.metadata["status"], "frozen")
            self.assertEqual(result.metadata["step"], "12.7")
            self.assertEqual(result.metadata["artifact_count"], 3)
            self.assertEqual(result.metadata["definitive_final_evidence_artifact_count"], 16)
            self.assertEqual(result.metadata["holm_reject_count"], 0)
            self.assertFalse(result.metadata["empirical_results_recomputed"])
            self.assertEqual(tuple(result.manifest["artifact"]), STEP12_FREEZE_ARTIFACTS)

    def test_rejects_failed_step12_6_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation_metadata = self._fixture(root)
            validation_metadata["validation_failed_check_count"] = 1
            fake_validation = type("Validation", (), {"metadata": validation_metadata})()
            with patch(
                "strategies.publication_step12_freeze.validate_publication_step12_scientific_invariants",
                return_value=fake_validation,
            ):
                with self.assertRaisesRegex(ValueError, "failed scientific invariant"):
                    validate_and_freeze_publication_step12_evidence(root)

    def test_rejects_changed_definitive_manifest_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation_metadata = self._fixture(root)
            manifest = pd.read_csv(root / "publication_final_evidence_manifest.csv")
            manifest.loc[0, "sha256"] = "0" * 64
            manifest.to_csv(root / "publication_final_evidence_manifest.csv", index=False)
            fake_validation = type("Validation", (), {"metadata": validation_metadata})()
            with patch(
                "strategies.publication_step12_freeze.validate_publication_step12_scientific_invariants",
                return_value=fake_validation,
            ):
                with self.assertRaisesRegex(ValueError, "hash changed"):
                    validate_and_freeze_publication_step12_evidence(root)


if __name__ == "__main__":
    unittest.main()
