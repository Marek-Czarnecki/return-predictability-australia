from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategies.publication_step12_evidence import (
    validate_publication_step12_upstream_evidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicationStep12UpstreamEvidenceTests(unittest.TestCase):
    def _write_layer(
        self,
        root: Path,
        *,
        manifest_name: str,
        metadata_name: str,
        step: str,
        artifact_count: int,
        metadata_extra: dict[str, object] | None = None,
    ) -> None:
        rows = []
        for index in range(artifact_count):
            name = f"{step.replace('.', '_')}_artifact_{index:02d}.txt"
            path = root / name
            path.write_text(f"fixture-{step}-{index}\n", encoding="utf-8")
            rows.append(
                {
                    "artifact": name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        pd.DataFrame(rows).to_csv(root / manifest_name, index=False)
        metadata = {
            "status": "frozen",
            "step": step,
            "artifact_count": artifact_count,
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        (root / metadata_name).write_text(json.dumps(metadata), encoding="utf-8")

    def _build_fixture(self, root: Path) -> None:
        self._write_layer(
            root,
            manifest_name="publication_step8_evidence_manifest.csv",
            metadata_name="publication_step8_evidence_metadata.json",
            step="8",
            artifact_count=40,
            metadata_extra={
                "validation_status": "passed",
                "validation_failed_check_count": 0,
                "primary_hypothesis_count": 4,
                "multiple_testing_method": "holm",
                "reject_count_after_holm_0_05": 0,
            },
        )
        self._write_layer(
            root,
            manifest_name="publication_step9_evidence_manifest.csv",
            metadata_name="publication_step9_evidence_metadata.json",
            step="9",
            artifact_count=3,
            metadata_extra={"step8_hash_validation_status": "passed"},
        )
        self._write_layer(
            root,
            manifest_name="publication_step11_trend_evidence_manifest.csv",
            metadata_name="publication_step11_trend_evidence_metadata.json",
            step="11.1.6",
            artifact_count=21,
            metadata_extra={"step9_frozen_hash_validation_status": "passed"},
        )

    def test_validates_all_three_upstream_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            result = validate_publication_step12_upstream_evidence(root)
            self.assertEqual(result.metadata["status"], "passed")
            self.assertEqual(result.metadata["step"], "12.2")
            self.assertEqual(result.metadata["upstream_layer_count"], 3)
            self.assertEqual(result.metadata["upstream_artifact_count"], 64)
            self.assertFalse(result.metadata["empirical_results_recomputed"])
            self.assertEqual(result.summary["status"].tolist(), ["passed", "passed", "passed"])

    def test_rejects_changed_frozen_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            (root / "8_artifact_00.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
                validate_publication_step12_upstream_evidence(root)

    def test_rejects_broken_dependency_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            path = root / "publication_step11_trend_evidence_metadata.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["step9_frozen_hash_validation_status"] = "failed"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Step 9 frozen-hash"):
                validate_publication_step12_upstream_evidence(root)

    def test_rejects_changed_step8_scientific_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            path = root / "publication_step8_evidence_metadata.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["primary_hypothesis_count"] = 5
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hypothesis count"):
                validate_publication_step12_upstream_evidence(root)


if __name__ == "__main__":
    unittest.main()
