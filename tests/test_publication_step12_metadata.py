from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from strategies.publication_step12_evidence import build_publication_final_evidence_metadata


class PublicationStep12MetadataTests(unittest.TestCase):
    def _build_fixture(self, root: Path) -> None:
        inference_rows = []
        comparison_rows = []
        for name, unit, n in (
            ("trend_following", "evaluation_folds", 24),
            ("mean_reversion", "evaluation_folds", 24),
            ("pairs_trading", "evaluation_folds", 24),
            ("tax_loss_selling", "calendar_years", 26),
        ):
            inference_rows.append({
                "analysis_key": name,
                "governance": "confirmatory",
                "multiple_testing_method": "holm",
                "reject_null_0_05": False,
                "primary_metric": "abnormal_net_return_difference" if name == "tax_loss_selling" else "net_excess_nav_difference",
                "effect_estimate": 0.01,
                "effect_unit": "unit",
                "ci_lower_95": -0.01,
                "ci_upper_95": 0.02,
                "p_value": 0.5,
                "adjusted_p_value": 1.0,
                "sample_size": n,
                "sample_unit": unit,
                "inference_method": "test",
                "claim_scope": "scope",
                "claim_label": "not_supported",
                "source_artifact": "source.csv",
            })
            comparison_rows.append({
                "strategy_family": name,
                "evidence_strength_change": "unsupported_to_unsupported",
                "comparability_note": "note",
            })
        pd.DataFrame(inference_rows).to_csv(root / "publication_primary_inference.csv", index=False)
        pd.DataFrame(comparison_rows).to_csv(root / "publication_corrected_vs_frozen_comparison.csv", index=False)

        payloads = {
            "publication_step11_trend_evidence_metadata.json": {
                "major_identified_contributor": "point_in_time_membership_vs_retrospective_current_membership",
                "modest_identified_contributor": "transaction_costs",
                "unresolved_contributor": "vendor_and_security_coverage",
                "not_part_of_primary_holm_family": True,
            },
            "publication_eligibility_validation.json": {
                "timing_convention": "next_session_after_close",
                "identity_convention": "asset_id",
                "membership_convention": "member_of_universe",
            },
            "publication_benchmark_build_summary.json": {
                "external_benchmark": {
                    "source_symbol": "$XJOA.au",
                    "source_asset_id": 203461,
                    "start_date": "2000-03-31",
                    "end_date": "2026-08-27",
                }
            },
            "publication_liquidity_cost_validation.json": {
                "liquidity_information_set": "formation_window_only",
                "fallback_tier": "lower",
                "tier_cutoffs": {"high": 0.3, "medium": 0.7},
            },
            "publication_risk_free_metadata.json": {
                "source": "Reserve Bank of Australia cash rate target history",
                "construction": "target_rate_percent / 365 compounded over actual calendar-day gaps",
                "use": "Sharpe and risk-adjusted publication metrics only",
                "parameter_selection_affected": False,
                "strategy_returns_affected": False,
            },
        }
        for name, payload in payloads.items():
            (root / name).write_text(json.dumps(payload), encoding="utf-8")

    def _upstream(self):
        return SimpleNamespace(metadata={"upstream_artifact_count": 64})

    def test_builds_locked_provenance_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            with patch(
                "strategies.publication_step12_evidence.validate_publication_step12_upstream_evidence",
                return_value=self._upstream(),
            ):
                metadata = build_publication_final_evidence_metadata(root)
            self.assertEqual(metadata["step"], "12.4")
            self.assertEqual(metadata["status"], "built_not_frozen")
            self.assertEqual(metadata["canonical_benchmark"], "$XJOA.au")
            self.assertEqual(metadata["pit_execution_convention"], "next_session_after_close")
            self.assertEqual(metadata["tax_loss_primary_inferential_unit"], "calendar_years")
            self.assertEqual(metadata["holm_reject_count"], 0)
            self.assertEqual(metadata["base_turnover_cost_bps_by_tier"], {"high": 10.0, "medium": 20.0, "lower": 35.0})
            self.assertFalse(metadata["empirical_results_recomputed"])

    def test_rejects_changed_canonical_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            path = root / "publication_benchmark_build_summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["external_benchmark"]["source_symbol"] = "$XJO.au"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "strategies.publication_step12_evidence.validate_publication_step12_upstream_evidence",
                return_value=self._upstream(),
            ):
                with self.assertRaisesRegex(ValueError, "benchmark changed"):
                    build_publication_final_evidence_metadata(root)

    def test_rejects_tax_loss_inferential_unit_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            inference = pd.read_csv(root / "publication_primary_inference.csv")
            inference.loc[inference["analysis_key"] == "tax_loss_selling", "sample_unit"] = "ticker_events"
            inference.to_csv(root / "publication_primary_inference.csv", index=False)
            with patch(
                "strategies.publication_step12_evidence.validate_publication_step12_upstream_evidence",
                return_value=self._upstream(),
            ):
                with self.assertRaisesRegex(ValueError, "Tax-loss primary inferential unit"):
                    build_publication_final_evidence_metadata(root)


if __name__ == "__main__":
    unittest.main()
