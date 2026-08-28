from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.refine_publication_step11_common_period_metadata import refine_metadata
from strategies.publication_step11_trend_evidence import (
    STEP11_ARTIFACTS,
    validate_and_freeze_publication_step11_trend_evidence,
)


FROZEN_COMMIT = "349fc00b087d404ba11bb9f04e9fe8ba7ad58ed6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicationStep11TrendEvidenceTests(unittest.TestCase):
    def _build_fixture(self, root: Path) -> None:
        for name in STEP11_ARTIFACTS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")

        common = {
            "step": "11.1.1", "analysis_role": "diagnostic_ablation", "confirmatory": False,
            "frozen_capstone_commit": FROZEN_COMMIT, "fold_count": 7,
            "causal_scope": "sample_period_and_fold_calendar",
            "mean_net_excess_nav_difference": -0.041322384896077646,
            "positive_fold_count": 1,
        }
        universe = {
            "step": "11.1.2", "analysis_role": "diagnostic_ablation", "confirmatory": False,
            "frozen_capstone_commit": FROZEN_COMMIT, "fold_count": 7,
            "pit_mean_net_excess_nav_difference": -0.041322384896077646,
            "retrospective_mean_net_excess_nav_difference": 0.11172910025642722,
            "mean_universe_effect_nav_difference": 0.15305148515250488,
            "reference_universe_asset_count": 200,
            "pit_positive_fold_count": 1, "retrospective_positive_fold_count": 6,
        }
        benchmark = {
            "step": "11.1.3", "analysis_role": "diagnostic_ablation", "confirmatory": False,
            "frozen_capstone_commit": FROZEN_COMMIT, "fold_count": 7,
            "xjoa_mean_net_excess_nav_difference": -0.041322384896077646,
            "stw_mean_net_excess_nav_difference": -0.04273496390057785,
            "mean_benchmark_effect_nav_difference": -0.001412579004500203,
            "parameter_selection_change_count": 0,
        }
        cost = {
            "step": "11.1.4", "analysis_role": "diagnostic_ablation", "confirmatory": False,
            "frozen_capstone_commit": FROZEN_COMMIT, "fold_count": 7,
            "base_mean_net_excess_nav_difference": -0.041322384896077646,
            "zero_mean_net_excess_nav_difference": -0.02629488457335178,
            "mean_cost_effect_nav_difference": 0.015027500322725864,
            "parameter_selection_change_count": 1,
        }
        attribution_meta = {
            "step": "11.1.5", "confirmatory": False,
            "attribution_statement": (
                "Retrospective-current membership is the major identified contributor; "
                "security coverage remains unresolved and the evidence does not fully explain every difference."
            ),
        }
        for name, payload in (
            ("publication_step11_trend_common_period_metadata.json", common),
            ("publication_step11_trend_universe_ablation_metadata.json", universe),
            ("publication_step11_trend_benchmark_ablation_metadata.json", benchmark),
            ("publication_step11_trend_cost_ablation_metadata.json", cost),
            ("publication_step11_trend_attribution_metadata.json", attribution_meta),
        ):
            (root / name).write_text(json.dumps(payload), encoding="utf-8")

        expected_classes = [
            ("metric_semantics", "not_explanatory"),
            ("risk_free_treatment", "not_explanatory"),
            ("sample_period_and_fold_calendar", "not_explanatory"),
            ("point_in_time_membership", "directly_demonstrated"),
            ("benchmark_choice", "not_explanatory"),
            ("transaction_costs", "directly_demonstrated"),
            ("vendor_and_security_coverage", "unresolved_contributor"),
        ]
        pd.DataFrame(expected_classes, columns=["design_change", "attribution_class"]).to_csv(
            root / "publication_step11_trend_attribution.csv", index=False
        )
        pd.DataFrame([{
            "strategy_family": "trend_following",
            "frozen_reconstructed_nav_difference_mean": 0.10276428548304055,
            "publication_nav_difference_mean": 0.0008204009098994785,
        }]).to_csv(root / "publication_corrected_vs_frozen_comparison.csv", index=False)

        step9_metadata = root / "publication_step9_evidence_metadata.json"
        step9_metadata.write_text(json.dumps({"status": "frozen", "step": "9"}), encoding="utf-8")
        step9_sources = [
            "publication_corrected_vs_frozen_comparison.csv",
            "publication_step11_trend_attribution.csv",
        ]
        rows = []
        for name in step9_sources:
            path = root / name
            rows.append({"artifact": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
        pd.DataFrame(rows).to_csv(root / "publication_step9_evidence_manifest.csv", index=False)

    def test_refine_common_period_metadata_is_empirical_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.json"
            path.write_text(json.dumps({
                "step": "11.1.1", "analysis_role": "diagnostic_ablation", "confirmatory": False,
                "mean_net_excess_nav_difference": -0.041322384896077646,
                "causal_scope": "sample_extension_and_regime_coverage",
            }), encoding="utf-8")
            result = refine_metadata(path)
            self.assertEqual(result["causal_scope"], "sample_period_and_fold_calendar")
            self.assertAlmostEqual(result["mean_net_excess_nav_difference"], -0.041322384896077646)

    def test_validates_and_builds_21_artifact_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            result = validate_and_freeze_publication_step11_trend_evidence(root)
            self.assertEqual(len(result.manifest), 21)
            self.assertEqual(result.metadata["status"], "frozen")
            self.assertEqual(result.metadata["step"], "11.1.6")
            self.assertEqual(result.metadata["major_identified_contributor"], "point_in_time_membership_vs_retrospective_current_membership")

    def test_rejects_stale_common_period_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            path = root / "publication_step11_trend_common_period_metadata.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["causal_scope"] = "sample_extension_and_regime_coverage"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "causal_scope"):
                validate_and_freeze_publication_step11_trend_evidence(root)

    def test_rejects_changed_control_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            path = root / "publication_step11_trend_cost_ablation_metadata.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["base_mean_net_excess_nav_difference"] = -0.02
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base-cost control"):
                validate_and_freeze_publication_step11_trend_evidence(root)


if __name__ == "__main__":
    unittest.main()
