from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from strategies.publication_step12_validation import (
    EXPECTED_ATTRIBUTION_STATEMENT,
    validate_publication_step12_scientific_invariants,
)


class PublicationStep12InvariantTests(unittest.TestCase):
    def _build_fixture(self, root: Path) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
        primary = pd.DataFrame(
            [
                {
                    "strategy_family": "trend_following",
                    "primary_metric": "net_excess_nav_difference",
                    "effect_estimate": 0.0008204009098994785,
                    "effect_unit": "evaluation_fold_nav_difference",
                    "ci_lower_95": -0.031459583391562795,
                    "ci_upper_95": 0.03087051700012456,
                    "raw_p_value": 0.4843,
                    "holm_p_value": 1.0,
                    "reject_after_holm_0_05": False,
                    "sample_size": 24,
                    "sample_unit": "evaluation_folds",
                    "inference_method": "walk_forward_fold_sign_flip",
                    "claim_scope": "strategy_level_confirmatory_transferability",
                    "frozen_evidence_change": "supported_to_unsupported",
                    "publication_conclusion": "confirmatory_not_supported_after_holm",
                    "source_artifact": "publication_trend_following_walk_forward_summary.csv",
                    "comparability_note": "fixture",
                },
                {
                    "strategy_family": "mean_reversion",
                    "primary_metric": "net_excess_nav_difference",
                    "effect_estimate": -0.5555920906637198,
                    "effect_unit": "evaluation_fold_nav_difference",
                    "ci_lower_95": -0.6101901294618491,
                    "ci_upper_95": -0.4998745427235552,
                    "raw_p_value": 1.0,
                    "holm_p_value": 1.0,
                    "reject_after_holm_0_05": False,
                    "sample_size": 24,
                    "sample_unit": "evaluation_folds",
                    "inference_method": "walk_forward_fold_sign_flip",
                    "claim_scope": "strategy_level_confirmatory_transferability",
                    "frozen_evidence_change": "unsupported_to_unsupported",
                    "publication_conclusion": "confirmatory_not_supported_after_holm",
                    "source_artifact": "publication_mean_reversion_walk_forward_summary.csv",
                    "comparability_note": "fixture",
                },
                {
                    "strategy_family": "pairs_trading",
                    "primary_metric": "net_excess_nav_difference",
                    "effect_estimate": -0.10683994834016741,
                    "effect_unit": "evaluation_fold_nav_difference",
                    "ci_lower_95": -0.17666741867773816,
                    "ci_upper_95": -0.030207222406029174,
                    "raw_p_value": 0.99435,
                    "holm_p_value": 1.0,
                    "reject_after_holm_0_05": False,
                    "sample_size": 24,
                    "sample_unit": "evaluation_folds",
                    "inference_method": "walk_forward_fold_sign_flip",
                    "claim_scope": "strategy_level_confirmatory_transferability",
                    "frozen_evidence_change": "unsupported_to_unsupported",
                    "publication_conclusion": "confirmatory_not_supported_after_holm",
                    "source_artifact": "publication_pairs_trading_walk_forward_summary.csv",
                    "comparability_note": "fixture",
                },
                {
                    "strategy_family": "tax_loss_selling",
                    "primary_metric": "abnormal_net_return_difference",
                    "effect_estimate": 0.024918808045132232,
                    "effect_unit": "annual_mean_abnormal_event_minus_control_return",
                    "ci_lower_95": -0.02787318500659601,
                    "ci_upper_95": 0.07108805201608011,
                    "raw_p_value": 0.1733,
                    "holm_p_value": 0.6932,
                    "reject_after_holm_0_05": False,
                    "sample_size": 26,
                    "sample_unit": "calendar_years",
                    "inference_method": "year_clustered_sign_flip",
                    "claim_scope": "strategy_level_confirmatory_transferability",
                    "frozen_evidence_change": "supported_to_unsupported",
                    "publication_conclusion": "confirmatory_not_supported_after_holm",
                    "source_artifact": "publication_tax_loss_selling_event_study.csv",
                    "comparability_note": "fixture",
                },
            ]
        )
        primary.to_csv(root / "publication_final_primary_results.csv", index=False)

        metadata = {
            "status": "built_not_frozen",
            "multiple_testing_method": "holm",
            "primary_hypothesis_count": 4,
            "canonical_benchmark": "$XJOA.au",
            "canonical_benchmark_asset_id": 203461,
            "pit_execution_convention": "next_session_after_close",
            "identity_convention": "asset_id",
            "membership_convention": "member_of_universe",
            "liquidity_information_set": "formation_window_only",
            "liquidity_fallback_tier": "lower",
            "liquidity_tier_cutoffs": {"high": 0.3, "medium": 0.7},
            "base_turnover_cost_bps_by_tier": {"high": 10.0, "medium": 20.0, "lower": 35.0},
            "risk_free_affects_parameter_selection": False,
            "risk_free_affects_strategy_returns": False,
            "risk_free_role": "Sharpe and risk-adjusted publication metrics only",
            "trend_attribution_is_diagnostic_not_confirmatory": True,
            "trend_major_identified_contributor": "point_in_time_membership_vs_retrospective_current_membership",
            "trend_modest_identified_contributor": "transaction_costs",
            "trend_unresolved_contributor": "vendor_and_security_coverage",
            "empirical_results_recomputed": False,
            "raw_licensed_data_excluded": True,
        }
        (root / "publication_final_evidence_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

        manifest = pd.DataFrame(
            [
                {
                    "artifact": f"artifact_{index:02d}.txt",
                    "source_step": "11.1.5" if index == 15 else "8",
                    "artifact_role": "primary_result",
                    "bytes": index + 1,
                    "sha256": f"{index:064x}",
                }
                for index in range(16)
            ]
        )
        manifest.to_csv(root / "publication_final_evidence_manifest.csv", index=False)

        attribution = {
            "confirmatory": False,
            "not_part_of_primary_holm_family": True,
            "attribution_statement": EXPECTED_ATTRIBUTION_STATEMENT,
            "interpretation_rule": (
                "Treat the table as diagnostic attribution across already-locked design changes. "
                "Do not sum ablation effects as if they were independent, and do not claim that "
                "the identified survivorship-membership effect fully reconciles the frozen and publication implementations."
            ),
        }
        (root / "publication_step11_trend_attribution_metadata.json").write_text(
            json.dumps(attribution), encoding="utf-8"
        )
        return primary, metadata, manifest

    def _validate_with_patches(
        self,
        root: Path,
        primary: pd.DataFrame,
        metadata: dict[str, object],
        manifest: pd.DataFrame,
    ):
        with (
            patch("strategies.publication_step12_validation.validate_publication_step12_upstream_evidence"),
            patch(
                "strategies.publication_step12_validation.build_publication_final_primary_results",
                return_value=primary,
            ),
            patch(
                "strategies.publication_step12_validation.build_publication_final_evidence_metadata",
                return_value=metadata,
            ),
            patch(
                "strategies.publication_step12_validation.build_publication_final_evidence_manifest",
                return_value=manifest,
            ),
        ):
            return validate_publication_step12_scientific_invariants(root)

    def test_validates_locked_scientific_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary, metadata, manifest = self._build_fixture(root)
            result = self._validate_with_patches(root, primary, metadata, manifest)
            self.assertEqual(result.metadata["status"], "passed")
            self.assertEqual(result.metadata["step"], "12.6")
            self.assertEqual(result.metadata["validation_check_count"], 10)
            self.assertEqual(result.metadata["validation_failed_check_count"], 0)

    def test_rejects_tax_loss_inferential_unit_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary, metadata, manifest = self._build_fixture(root)
            observed = pd.read_csv(root / "publication_final_primary_results.csv")
            observed.loc[observed["strategy_family"] == "tax_loss_selling", "sample_unit"] = "ticker_events"
            observed.to_csv(root / "publication_final_primary_results.csv", index=False)
            with self.assertRaises((AssertionError, ValueError)):
                self._validate_with_patches(root, primary, metadata, manifest)

    def test_rejects_benchmark_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary, metadata, manifest = self._build_fixture(root)
            changed = dict(metadata)
            changed["canonical_benchmark"] = "$XJO.au"
            (root / "publication_final_evidence_metadata.json").write_text(
                json.dumps(changed), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                self._validate_with_patches(root, primary, metadata, manifest)

    def test_rejects_strengthened_trend_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary, metadata, manifest = self._build_fixture(root)
            path = root / "publication_step11_trend_attribution_metadata.json"
            attribution = json.loads(path.read_text(encoding="utf-8"))
            attribution["attribution_statement"] = "Survivorship bias fully explains the frozen trend result."
            path.write_text(json.dumps(attribution), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trend_attribution_scope"):
                self._validate_with_patches(root, primary, metadata, manifest)


if __name__ == "__main__":
    unittest.main()
