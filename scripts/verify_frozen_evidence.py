from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "evidence"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    primary = pd.read_csv(require(EVIDENCE / "publication_final_primary_results.csv"))
    inference = pd.read_csv(require(EVIDENCE / "publication_primary_inference.csv"))
    comparison = pd.read_csv(require(EVIDENCE / "publication_corrected_vs_frozen_comparison.csv"))
    attribution = pd.read_csv(require(EVIDENCE / "publication_step11_trend_attribution.csv"))
    metadata = json.loads(require(EVIDENCE / "publication_final_evidence_metadata.json").read_text())

    expected_strategies = {
        "trend_following",
        "mean_reversion",
        "pairs_trading",
        "tax_loss_selling",
    }

    assert len(primary) == 4
    assert set(primary["strategy_family"]) == expected_strategies
    assert primary["reject_after_holm_0_05"].astype(str).str.lower().eq("false").all()

    assert len(inference) == 4
    assert set(inference["analysis_key"]) == expected_strategies
    assert inference["multiple_testing_method"].eq("holm").all()
    assert pd.to_numeric(inference["adjusted_p_value"], errors="raise").ge(0).all()

    assert set(comparison["strategy_family"]) == expected_strategies
    trend_change = comparison.loc[
        comparison["strategy_family"].eq("trend_following"), "evidence_strength_change"
    ].iloc[0]
    tax_change = comparison.loc[
        comparison["strategy_family"].eq("tax_loss_selling"), "evidence_strength_change"
    ].iloc[0]
    assert trend_change == "supported_to_unsupported"
    assert tax_change == "supported_to_unsupported"

    membership = attribution.loc[attribution["design_change"].eq("point_in_time_membership")].iloc[0]
    costs = attribution.loc[attribution["design_change"].eq("transaction_costs")].iloc[0]
    coverage = attribution.loc[attribution["design_change"].eq("vendor_and_security_coverage")].iloc[0]
    assert membership["attribution_class"] == "directly_demonstrated"
    assert membership["magnitude_role"] == "major_contributor"
    assert abs(float(membership["controlled_effect_nav_difference"]) - 0.15305148515250488) < 1e-12
    assert costs["magnitude_role"] == "modest_contributor"
    assert coverage["attribution_class"] == "unresolved_contributor"

    assert metadata["primary_hypothesis_count"] == 4
    assert metadata["holm_reject_count"] == 0
    assert metadata["raw_licensed_data_excluded"] is True
    assert metadata["trend_attribution_is_diagnostic_not_confirmatory"] is True

    print("PASS: public frozen-evidence scientific invariants verified")
    print("confirmatory_hypotheses=4 holm_rejections=0")
    print("trend_major_contributor=point_in_time_membership_vs_retrospective_current_membership")
    print("licensed_raw_data_present=false (by package design; see DATA_AVAILABILITY.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
