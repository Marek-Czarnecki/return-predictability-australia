from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategies.publication_step12_evidence import (
    build_publication_final_primary_results,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicationStep12PrimaryResultsTests(unittest.TestCase):
    def _write_layer(self, root: Path, step: str, count: int, manifest: str, metadata: str, extra: dict[str, object]) -> None:
        rows = []
        for i in range(count):
            name = f"fixture_{step.replace('.', '_')}_{i}.txt"
            path = root / name
            path.write_text(f"{step}-{i}\n", encoding="utf-8")
            rows.append({"artifact": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
        pd.DataFrame(rows).to_csv(root / manifest, index=False)
        payload = {"status": "frozen", "step": step, "artifact_count": count, **extra}
        (root / metadata).write_text(json.dumps(payload), encoding="utf-8")

    def _fixture(self, root: Path) -> None:
        self._write_layer(root, "8", 40, "publication_step8_evidence_manifest.csv", "publication_step8_evidence_metadata.json", {
            "validation_status": "passed", "validation_failed_check_count": 0,
            "primary_hypothesis_count": 4, "multiple_testing_method": "holm",
            "reject_count_after_holm_0_05": 0,
        })
        self._write_layer(root, "9", 3, "publication_step9_evidence_manifest.csv", "publication_step9_evidence_metadata.json", {
            "step8_hash_validation_status": "passed",
        })
        self._write_layer(root, "11.1.6", 21, "publication_step11_trend_evidence_manifest.csv", "publication_step11_trend_evidence_metadata.json", {
            "step9_frozen_hash_validation_status": "passed",
        })

        strategies = ["trend_following", "mean_reversion", "pairs_trading", "tax_loss_selling"]
        effects = [0.0008204009098994785, -0.5555920906637198, -0.10683994834016741, 0.024918808045132232]
        samples = [24, 24, 24, 26]
        units = ["evaluation_folds", "evaluation_folds", "evaluation_folds", "calendar_years"]
        metrics = ["net_excess_nav_difference", "net_excess_nav_difference", "net_excess_nav_difference", "abnormal_net_return_difference"]
        rows = []
        for s, e, n, u, m in zip(strategies, effects, samples, units, metrics):
            rows.append({
                "research_question": "publication_primary", "analysis_key": s, "test_label": s,
                "governance": "confirmatory", "claim_scope": "strategy_level_confirmatory_transferability",
                "inference_method": "year_clustered_sign_flip" if s == "tax_loss_selling" else "walk_forward_fold_sign_flip",
                "effect_estimate": e, "effect_unit": "annual_mean_abnormal_event_minus_control_return" if s == "tax_loss_selling" else "evaluation_fold_nav_difference",
                "null_value": 0.0, "alternative": "greater", "ci_lower_95": e - 0.01, "ci_upper_95": e + 0.01,
                "p_value": 0.5, "adjusted_p_value": 1.0 if s != "tax_loss_selling" else 0.6932,
                "multiple_testing_family": "confirmatory_primary_family", "multiple_testing_method": "holm",
                "reject_null_0_05": False, "sample_size": n, "claim_label": "confirmatory_not_supported_after_holm",
                "limitation_note": "fixture", "source_artifact": f"{s}.csv", "primary_metric": m, "sample_unit": u,
            })
        pd.DataFrame(rows).to_csv(root / "publication_primary_inference.csv", index=False)

        changes = ["supported_to_unsupported", "unsupported_to_unsupported", "unsupported_to_unsupported", "supported_to_unsupported"]
        pd.DataFrame({
            "strategy_family": strategies,
            "evidence_strength_change": changes,
            "comparability_note": ["fixture"] * 4,
        }).to_csv(root / "publication_corrected_vs_frozen_comparison.csv", index=False)

    def test_builds_exact_four_row_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            table = build_publication_final_primary_results(root)
            self.assertEqual(table["strategy_family"].tolist(), [
                "trend_following", "mean_reversion", "pairs_trading", "tax_loss_selling"
            ])
            self.assertEqual(len(table), 4)
            self.assertEqual(int(table["reject_after_holm_0_05"].sum()), 0)
            self.assertEqual(table.loc[3, "sample_unit"], "calendar_years")
            self.assertAlmostEqual(table.loc[0, "effect_estimate"], 0.0008204009098994785)

    def test_rejects_non_holm_primary_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            path = root / "publication_primary_inference.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "multiple_testing_method"] = "fdr"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Holm-controlled"):
                build_publication_final_primary_results(root)

    def test_rejects_missing_step9_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            path = root / "publication_corrected_vs_frozen_comparison.csv"
            frame = pd.read_csv(path).iloc[:3]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly four"):
                build_publication_final_primary_results(root)


if __name__ == "__main__":
    unittest.main()
