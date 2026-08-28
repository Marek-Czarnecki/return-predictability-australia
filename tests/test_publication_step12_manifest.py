from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategies.publication_step12_evidence import (
    FINAL_EVIDENCE_ARTIFACTS,
    FINAL_METADATA_NAME,
    FINAL_PRIMARY_RESULTS_NAME,
    build_publication_final_evidence_manifest,
)


class PublicationStep12ManifestTests(unittest.TestCase):
    def _build_fixture(self, root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
        expected_primary = pd.DataFrame(
            [{"strategy_family": "trend_following", "effect_estimate": 0.0008204009098994785}]
        )
        expected_primary.to_csv(root / FINAL_PRIMARY_RESULTS_NAME, index=False)

        expected_metadata = {
            "status": "built_not_frozen",
            "step": "12.4",
            "raw_licensed_data_excluded": True,
        }
        (root / FINAL_METADATA_NAME).write_text(
            json.dumps(expected_metadata, sort_keys=True), encoding="utf-8"
        )

        for name, _, _ in FINAL_EVIDENCE_ARTIFACTS:
            path = root / name
            if path.exists():
                continue
            path.write_text(f"fixture for {name}\n", encoding="utf-8")
        return expected_primary, expected_metadata

    def _patch_builders(self, expected_primary: pd.DataFrame, expected_metadata: dict[str, object]):
        return (
            patch(
                "strategies.publication_step12_evidence.validate_publication_step12_upstream_evidence"
            ),
            patch(
                "strategies.publication_step12_evidence.build_publication_final_primary_results",
                return_value=expected_primary,
            ),
            patch(
                "strategies.publication_step12_evidence.build_publication_final_evidence_metadata",
                return_value=expected_metadata,
            ),
        )

    def test_builds_compact_16_artifact_manifest_with_all_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_primary, expected_metadata = self._build_fixture(root)
            upstream_patch, primary_patch, metadata_patch = self._patch_builders(
                expected_primary, expected_metadata
            )
            with upstream_patch, primary_patch, metadata_patch:
                manifest = build_publication_final_evidence_manifest(root)

            self.assertEqual(len(manifest), 16)
            self.assertEqual(
                list(manifest.columns),
                ["artifact", "source_step", "artifact_role", "bytes", "sha256"],
            )
            self.assertEqual(manifest["artifact"].nunique(), 16)
            self.assertEqual(
                set(manifest["artifact_role"]),
                {
                    "primary_result",
                    "comparison_evidence",
                    "diagnostic_evidence",
                    "methodology_validation",
                    "provenance",
                    "upstream_freeze",
                },
            )
            self.assertTrue((manifest["bytes"] > 0).all())
            self.assertTrue(manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all())

    def test_rejects_stale_final_primary_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_primary, expected_metadata = self._build_fixture(root)
            observed = pd.read_csv(root / FINAL_PRIMARY_RESULTS_NAME)
            observed.loc[0, "effect_estimate"] = 0.25
            observed.to_csv(root / FINAL_PRIMARY_RESULTS_NAME, index=False)
            upstream_patch, primary_patch, metadata_patch = self._patch_builders(
                expected_primary, expected_metadata
            )
            with upstream_patch, primary_patch, metadata_patch:
                with self.assertRaises(AssertionError):
                    build_publication_final_evidence_manifest(root)

    def test_rejects_stale_final_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_primary, expected_metadata = self._build_fixture(root)
            stale = dict(expected_metadata)
            stale["step"] = "changed"
            (root / FINAL_METADATA_NAME).write_text(json.dumps(stale), encoding="utf-8")
            upstream_patch, primary_patch, metadata_patch = self._patch_builders(
                expected_primary, expected_metadata
            )
            with upstream_patch, primary_patch, metadata_patch:
                with self.assertRaisesRegex(ValueError, "metadata is stale"):
                    build_publication_final_evidence_manifest(root)

    def test_rejects_metadata_that_claims_raw_licensed_data_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_primary, expected_metadata = self._build_fixture(root)
            expected_metadata = dict(expected_metadata)
            expected_metadata["raw_licensed_data_excluded"] = False
            (root / FINAL_METADATA_NAME).write_text(
                json.dumps(expected_metadata), encoding="utf-8"
            )
            upstream_patch, primary_patch, metadata_patch = self._patch_builders(
                expected_primary, expected_metadata
            )
            with upstream_patch, primary_patch, metadata_patch:
                with self.assertRaisesRegex(ValueError, "exclude raw licensed data"):
                    build_publication_final_evidence_manifest(root)


if __name__ == "__main__":
    unittest.main()
