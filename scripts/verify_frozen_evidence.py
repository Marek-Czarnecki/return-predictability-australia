from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "evidence"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    mechanism = pd.read_csv(require(EVIDENCE / "publication_trend_2x2_decomposition.csv"))
    concentration = json.loads(require(EVIDENCE / "publication_trend_concentration_summary.json").read_text())

    assert len(mechanism) == 7
    required_cols = {
        "fold_id",
        "a_pit_pitparams_nav_difference",
        "b_retro_retroparams_nav_difference",
        "c_retro_pitparams_nav_difference",
        "d_pit_retroparams_nav_difference",
        "total_universe_treatment_effect_nav_difference",
        "shapley_universe_component_nav_difference",
        "shapley_parameter_selection_component_nav_difference",
        "parameter_selection_changed",
    }
    assert required_cols.issubset(mechanism.columns)

    a = mechanism["a_pit_pitparams_nav_difference"].astype(float)
    b = mechanism["b_retro_retroparams_nav_difference"].astype(float)
    total = mechanism["total_universe_treatment_effect_nav_difference"].astype(float)
    universe = mechanism["shapley_universe_component_nav_difference"].astype(float)
    parameter = mechanism["shapley_parameter_selection_component_nav_difference"].astype(float)

    assert np.allclose(total.to_numpy(), (b - a).to_numpy(), atol=1e-12)
    assert np.allclose((universe + parameter).to_numpy(), total.to_numpy(), atol=1e-12)
    assert abs(float(total.mean()) - 0.15305148515250488) < 1e-12
    assert abs(float(universe.mean()) - 0.15170128640783137) < 1e-12
    assert abs(float(parameter.mean()) - 0.0013501987446734587) < 1e-12
    assert int((universe > 0).sum()) == 7
    changed = mechanism["parameter_selection_changed"].astype(str).str.lower().eq("true")
    assert int(changed.sum()) == 5

    assert concentration["asset_count"] == 718
    assert concentration["fold_count"] == 7
    assert concentration["asset_count_to_50pct_absolute_share"] == 28
    assert concentration["asset_count_to_80pct_absolute_share"] == 81
    assert abs(float(concentration["top_1_absolute_share"]) - 0.0437085158220576) < 1e-12
    assert abs(float(concentration["top_5_absolute_share"]) - 0.15302333182940622) < 1e-12
    assert abs(float(concentration["top_10_absolute_share"]) - 0.24558295546218692) < 1e-12
    assert abs(float(concentration["absolute_contribution_hhi"]) - 0.012697823907628782) < 1e-12

    primary = pd.read_csv(require(EVIDENCE / "publication_final_primary_results.csv"))
    inference = pd.read_csv(require(EVIDENCE / "publication_primary_inference.csv"))
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
    assert metadata["primary_hypothesis_count"] == 4
    assert metadata["holm_reject_count"] == 0
    assert metadata["raw_licensed_data_excluded"] is True

    print("PASS: public manuscript-evidence invariants verified")
    print("mechanism_folds=7 exact_decomposition=true")
    print("mean_total_effect=0.153051485153")
    print("mean_universe_component=0.151701286408")
    print("mean_parameter_component=0.001350198745")
    print("positive_universe_folds=7 parameter_changed_folds=5")
    print("concentration_assets=718 top10_absolute_share=0.245582955462")
    print("confirmatory_hypotheses=4 holm_rejections=0")
    print("licensed_raw_and_security_level_contribution_data_present=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
