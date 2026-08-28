from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from .publication_step12_evidence import FINAL_MANIFEST_NAME, FINAL_METADATA_NAME, FINAL_PRIMARY_RESULTS_NAME, STEP11_ATTRIBUTION_METADATA_NAME, build_publication_final_evidence_manifest, build_publication_final_evidence_metadata, build_publication_final_primary_results, validate_publication_step12_upstream_evidence

EXPECTED_STRATEGIES={"trend_following","mean_reversion","pairs_trading","tax_loss_selling"}
EXPECTED_DAILY_STRATEGIES={"trend_following","mean_reversion","pairs_trading"}
EXPECTED_ATTRIBUTION_STATEMENT=("The trend-result collapse is not explained by metric semantics, risk-free treatment, sample extension, or benchmark choice. Within a common Norgate design and the frozen seven-fold calendar, retrospective application of the later constituent set raises mean benchmark-relative NAV performance by 0.1531, from -0.0413 to 0.1117, directly demonstrating that retrospective-current membership is the major identified contributor. Removing transaction costs adds 0.0150 but leaves mean performance negative, so costs are a modest contributor. Yahoo-versus-Norgate security coverage remains unresolved; the evidence therefore supports a bounded attribution to the survivorship-membership mechanism rather than a claim that it fully explains every difference from the frozen capstone.")

@dataclass(frozen=True)
class PublicationStep12InvariantValidation:
    checks: pd.DataFrame
    metadata: dict[str,object]

def validate_publication_step12_scientific_invariants(results_root:Path)->PublicationStep12InvariantValidation:
    results_root=Path(results_root); validate_publication_step12_upstream_evidence(results_root)
    required=(FINAL_PRIMARY_RESULTS_NAME,FINAL_METADATA_NAME,FINAL_MANIFEST_NAME,STEP11_ATTRIBUTION_METADATA_NAME); missing=[name for name in required if not (results_root/name).is_file()]
    if missing: raise ValueError("Missing Step 12.6 evidence files: "+", ".join(sorted(missing)))
    observed_primary=pd.read_csv(results_root/FINAL_PRIMARY_RESULTS_NAME); expected_primary=build_publication_final_primary_results(results_root); pd.testing.assert_frame_equal(observed_primary,expected_primary,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    observed_metadata=_load_json(results_root/FINAL_METADATA_NAME); expected_metadata=build_publication_final_evidence_metadata(results_root)
    if observed_metadata!=expected_metadata: raise ValueError("Step 12.4 metadata differs from the locked provenance builder output.")
    observed_manifest=pd.read_csv(results_root/FINAL_MANIFEST_NAME); expected_manifest=build_publication_final_evidence_manifest(results_root); pd.testing.assert_frame_equal(observed_manifest,expected_manifest,check_dtype=False,check_exact=True)
    attribution=_load_json(results_root/STEP11_ATTRIBUTION_METADATA_NAME); checks=[]
    _check(checks,"confirmatory_family",len(observed_primary)==4 and set(observed_primary["strategy_family"].astype(str))==EXPECTED_STRATEGIES and int(observed_primary["reject_after_holm_0_05"].astype(bool).sum())==0 and observed_metadata.get("multiple_testing_method")=="holm" and int(observed_metadata.get("primary_hypothesis_count",-1))==4,"Four strategy hypotheses remain one Holm-controlled confirmatory family with 0/4 rejections.")
    daily=observed_primary.loc[observed_primary["strategy_family"].isin(EXPECTED_DAILY_STRATEGIES)]
    _check(checks,"daily_strategy_estimand",len(daily)==3 and (daily["primary_metric"]=="net_excess_nav_difference").all() and (daily["sample_unit"]=="evaluation_folds").all() and (daily["sample_size"].astype(int)==24).all(),"Trend, mean reversion and pairs retain net_excess_nav_difference over 24 evaluation folds.")
    tax=observed_primary.loc[observed_primary["strategy_family"]=="tax_loss_selling"]
    _check(checks,"tax_loss_inferential_unit",len(tax)==1 and tax.iloc[0]["primary_metric"]=="abnormal_net_return_difference" and tax.iloc[0]["sample_unit"]=="calendar_years" and int(tax.iloc[0]["sample_size"])==26,"Tax-loss primary inference remains benchmark-adjusted net event-minus-control return over 26 calendar years.")
    _check(checks,"benchmark_and_pit",observed_metadata.get("canonical_benchmark")=="$XJOA.au" and int(observed_metadata.get("canonical_benchmark_asset_id",-1))==203461 and observed_metadata.get("pit_execution_convention")=="next_session_after_close" and observed_metadata.get("identity_convention")=="asset_id" and observed_metadata.get("membership_convention")=="member_of_universe","Canonical benchmark and point-in-time execution/identity conventions remain locked.")
    _check(checks,"liquidity_and_costs",observed_metadata.get("liquidity_information_set")=="formation_window_only" and observed_metadata.get("liquidity_fallback_tier")=="lower" and observed_metadata.get("liquidity_tier_cutoffs")=={"high":0.3,"medium":0.7} and observed_metadata.get("base_turnover_cost_bps_by_tier")=={"high":10.0,"medium":20.0,"lower":35.0},"Formation-only liquidity tiers and 10/20/35 bps base costs remain locked.")
    _check(checks,"risk_free_role",observed_metadata.get("risk_free_affects_parameter_selection") is False and observed_metadata.get("risk_free_affects_strategy_returns") is False and observed_metadata.get("risk_free_role")=="Sharpe and risk-adjusted publication metrics only","Risk-free treatment remains descriptive/risk-adjusted only and does not affect strategy selection or returns.")
    _check(checks,"diagnostic_boundary",observed_metadata.get("trend_attribution_is_diagnostic_not_confirmatory") is True and attribution.get("confirmatory") is False and attribution.get("not_part_of_primary_holm_family") is True,"Step 11 trend attribution remains diagnostic and outside the primary Holm family.")
    _check(checks,"trend_attribution_classes",observed_metadata.get("trend_major_identified_contributor")=="point_in_time_membership_vs_retrospective_current_membership" and observed_metadata.get("trend_modest_identified_contributor")=="transaction_costs" and observed_metadata.get("trend_unresolved_contributor")=="vendor_and_security_coverage","Trend attribution remains major=PIT membership, modest=costs, unresolved=vendor/security coverage.")
    _check(checks,"trend_attribution_scope",attribution.get("attribution_statement")==EXPECTED_ATTRIBUTION_STATEMENT and "Do not sum ablation effects as if they were independent" in str(attribution.get("interpretation_rule","")) and "do not claim that the identified survivorship-membership effect fully reconciles" in str(attribution.get("interpretation_rule","")),"Trend attribution preserves the bounded statement and explicitly prohibits additive or full-reconciliation claims.")
    _check(checks,"freeze_preconditions",observed_metadata.get("status")=="built_not_frozen" and observed_metadata.get("empirical_results_recomputed") is False and observed_metadata.get("raw_licensed_data_excluded") is True and len(observed_manifest)==16,"Step 12 remains pre-freeze, uses no empirical recomputation, excludes raw licensed data, and retains the 16-artifact manifest.")
    frame=pd.DataFrame(checks,columns=["check","status","detail"]); failed=frame.loc[frame["status"]!="passed","check"].tolist()
    if failed: raise ValueError("Step 12.6 scientific invariant validation failed: "+", ".join(failed))
    metadata={"status":"passed","step":"12.6","step_label":"validate_final_scientific_and_attribution_invariants","validation_check_count":int(len(frame)),"validation_failed_check_count":0,"primary_hypothesis_count":4,"holm_reject_count":0,"upstream_integrity_validation_status":"passed","final_primary_reconciliation_status":"passed","final_metadata_reconciliation_status":"passed","final_manifest_reconciliation_status":"passed","trend_attribution_boundary_status":"passed","empirical_results_recomputed":False}
    return PublicationStep12InvariantValidation(frame,metadata)

def _check(checks,name,condition,detail): checks.append({"check":name,"status":"passed" if condition else "failed","detail":detail})
def _load_json(path): return json.loads(path.read_text(encoding="utf-8"))
