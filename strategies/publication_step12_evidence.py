from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .costs import DEFAULT_COST_SCENARIOS

STEP8_MANIFEST_NAME = "publication_step8_evidence_manifest.csv"
STEP8_METADATA_NAME = "publication_step8_evidence_metadata.json"
STEP9_MANIFEST_NAME = "publication_step9_evidence_manifest.csv"
STEP9_METADATA_NAME = "publication_step9_evidence_metadata.json"
STEP11_MANIFEST_NAME = "publication_step11_trend_evidence_manifest.csv"
STEP11_METADATA_NAME = "publication_step11_trend_evidence_metadata.json"
FINAL_PRIMARY_RESULTS_NAME = "publication_final_primary_results.csv"
FINAL_METADATA_NAME = "publication_final_evidence_metadata.json"
FINAL_MANIFEST_NAME = "publication_final_evidence_manifest.csv"
PRIMARY_INFERENCE_NAME = "publication_primary_inference.csv"
COMPARISON_NAME = "publication_corrected_vs_frozen_comparison.csv"
STEP11_ATTRIBUTION_NAME = "publication_step11_trend_attribution.csv"
STEP11_ATTRIBUTION_METADATA_NAME = "publication_step11_trend_attribution_metadata.json"
ELIGIBILITY_VALIDATION_NAME = "publication_eligibility_validation.json"
BENCHMARK_BUILD_SUMMARY_NAME = "publication_benchmark_build_summary.json"
LIQUIDITY_VALIDATION_NAME = "publication_liquidity_cost_validation.json"
RISK_FREE_METADATA_NAME = "publication_risk_free_metadata.json"
FROZEN_CAPSTONE_COMMIT = "349fc00b087d404ba11bb9f04e9fe8ba7ad58ed6"
PUBLICATION_RESEARCH_QUESTION = (
    "Do established return-predictability effects survive transfer to Australian equities "
    "under one common out-of-sample, multiplicity-controlled and implementation-aware standard?"
)
EXPECTED_PRIMARY_STRATEGIES = (
    "trend_following",
    "mean_reversion",
    "pairs_trading",
    "tax_loss_selling",
)
FINAL_EVIDENCE_ARTIFACTS = (
    (FINAL_PRIMARY_RESULTS_NAME, "12.3", "primary_result"),
    (FINAL_METADATA_NAME, "12.4", "provenance"),
    (PRIMARY_INFERENCE_NAME, "8", "primary_result"),
    (STEP8_MANIFEST_NAME, "8", "upstream_freeze"),
    (STEP8_METADATA_NAME, "8", "upstream_freeze"),
    (BENCHMARK_BUILD_SUMMARY_NAME, "8", "methodology_validation"),
    (ELIGIBILITY_VALIDATION_NAME, "8", "methodology_validation"),
    (LIQUIDITY_VALIDATION_NAME, "8", "methodology_validation"),
    (RISK_FREE_METADATA_NAME, "8", "methodology_validation"),
    (COMPARISON_NAME, "9", "comparison_evidence"),
    (STEP9_MANIFEST_NAME, "9", "upstream_freeze"),
    (STEP9_METADATA_NAME, "9", "upstream_freeze"),
    (STEP11_ATTRIBUTION_NAME, "11.1.5", "diagnostic_evidence"),
    (STEP11_ATTRIBUTION_METADATA_NAME, "11.1.5", "diagnostic_evidence"),
    (STEP11_MANIFEST_NAME, "11.1.6", "upstream_freeze"),
    (STEP11_METADATA_NAME, "11.1.6", "upstream_freeze"),
)

@dataclass(frozen=True)
class PublicationStep12UpstreamValidation:
    summary: pd.DataFrame
    metadata: dict[str, object]

def validate_publication_step12_upstream_evidence(results_root: Path) -> PublicationStep12UpstreamValidation:
    results_root = Path(results_root)
    step8 = _validate_frozen_layer(results_root, manifest_name=STEP8_MANIFEST_NAME, metadata_name=STEP8_METADATA_NAME, expected_step="8", expected_artifact_count=40)
    step9 = _validate_frozen_layer(results_root, manifest_name=STEP9_MANIFEST_NAME, metadata_name=STEP9_METADATA_NAME, expected_step="9", expected_artifact_count=3)
    step11 = _validate_frozen_layer(results_root, manifest_name=STEP11_MANIFEST_NAME, metadata_name=STEP11_METADATA_NAME, expected_step="11.1.6", expected_artifact_count=21)
    _validate_step8_contract(step8["metadata"]); _validate_dependency_chain(step9["metadata"], step11["metadata"])
    summary = pd.DataFrame([
        {"source_step":"8","manifest":STEP8_MANIFEST_NAME,"metadata":STEP8_METADATA_NAME,"artifact_count":step8["artifact_count"],"status":"passed"},
        {"source_step":"9","manifest":STEP9_MANIFEST_NAME,"metadata":STEP9_METADATA_NAME,"artifact_count":step9["artifact_count"],"status":"passed"},
        {"source_step":"11.1.6","manifest":STEP11_MANIFEST_NAME,"metadata":STEP11_METADATA_NAME,"artifact_count":step11["artifact_count"],"status":"passed"},
    ], columns=["source_step","manifest","metadata","artifact_count","status"])
    metadata = {"step":"12.2","step_label":"validate_upstream_frozen_evidence","status":"passed","validation_role":"integrity_only_no_empirical_recomputation","upstream_layer_count":3,"upstream_artifact_count":int(step8["artifact_count"]+step9["artifact_count"]+step11["artifact_count"]),"step8_hash_validation_status":"passed","step9_hash_validation_status":"passed","step11_hash_validation_status":"passed","dependency_chain_validation_status":"passed","empirical_results_recomputed":False}
    return PublicationStep12UpstreamValidation(summary, metadata)

def build_publication_final_primary_results(results_root: Path) -> pd.DataFrame:
    results_root=Path(results_root); validate_publication_step12_upstream_evidence(results_root); _require_files(results_root,(PRIMARY_INFERENCE_NAME,COMPARISON_NAME))
    inference=pd.read_csv(results_root/PRIMARY_INFERENCE_NAME); comparison=pd.read_csv(results_root/COMPARISON_NAME)
    if len(inference)!=4 or inference["analysis_key"].nunique()!=4: raise ValueError("Frozen primary inference must contain exactly four unique strategy rows.")
    observed=tuple(inference["analysis_key"].astype(str).tolist())
    if set(observed)!=set(EXPECTED_PRIMARY_STRATEGIES): raise ValueError(f"Frozen primary inference strategy set changed: {observed}")
    if not (inference["governance"].astype(str)=="confirmatory").all(): raise ValueError("All final primary-result rows must remain confirmatory.")
    if not (inference["multiple_testing_method"].astype(str).str.lower()=="holm").all(): raise ValueError("Final primary-result rows must remain in the Holm-controlled family.")
    if int(inference["reject_null_0_05"].astype(bool).sum())!=0: raise ValueError("Frozen publication primary inference no longer has 0/4 Holm rejections.")
    if len(comparison)!=4 or comparison["strategy_family"].nunique()!=4: raise ValueError("Frozen Step 9 comparison must contain exactly four unique strategy rows.")
    if set(comparison["strategy_family"].astype(str))!=set(EXPECTED_PRIMARY_STRATEGIES): raise ValueError("Frozen Step 9 comparison strategy set changed.")
    comparison_fields=comparison[["strategy_family","evidence_strength_change","comparability_note"]].rename(columns={"evidence_strength_change":"frozen_evidence_change"})
    table=inference.rename(columns={"analysis_key":"strategy_family","p_value":"raw_p_value","adjusted_p_value":"holm_p_value","reject_null_0_05":"reject_after_holm_0_05","claim_label":"publication_conclusion"}).merge(comparison_fields,on="strategy_family",how="left",validate="one_to_one")
    if table["frozen_evidence_change"].isna().any(): raise ValueError("Step 12.3 could not reconcile every primary result to Step 9 evidence.")
    columns=["strategy_family","primary_metric","effect_estimate","effect_unit","ci_lower_95","ci_upper_95","raw_p_value","holm_p_value","reject_after_holm_0_05","sample_size","sample_unit","inference_method","claim_scope","frozen_evidence_change","publication_conclusion","source_artifact","comparability_note"]
    table=table.loc[:,columns].copy(); order={name:index for index,name in enumerate(EXPECTED_PRIMARY_STRATEGIES)}; table["_order"]=table["strategy_family"].map(order); return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)

def build_publication_final_evidence_metadata(results_root: Path) -> dict[str, object]:
    results_root=Path(results_root); upstream=validate_publication_step12_upstream_evidence(results_root); primary=build_publication_final_primary_results(results_root)
    _require_files(results_root,(STEP11_METADATA_NAME,ELIGIBILITY_VALIDATION_NAME,BENCHMARK_BUILD_SUMMARY_NAME,LIQUIDITY_VALIDATION_NAME,RISK_FREE_METADATA_NAME))
    step11=_load_json(results_root/STEP11_METADATA_NAME); eligibility=_load_json(results_root/ELIGIBILITY_VALIDATION_NAME); benchmark=_load_json(results_root/BENCHMARK_BUILD_SUMMARY_NAME); liquidity=_load_json(results_root/LIQUIDITY_VALIDATION_NAME); risk_free=_load_json(results_root/RISK_FREE_METADATA_NAME)
    external=benchmark.get("external_benchmark",{})
    if external.get("source_symbol")!="$XJOA.au": raise ValueError("Canonical publication benchmark changed from $XJOA.au.")
    if eligibility.get("timing_convention")!="next_session_after_close": raise ValueError("PIT execution convention changed from next_session_after_close.")
    if eligibility.get("identity_convention")!="asset_id": raise ValueError("Publication identity convention changed from asset_id.")
    if liquidity.get("liquidity_information_set")!="formation_window_only": raise ValueError("Liquidity information set changed from formation_window_only.")
    if liquidity.get("fallback_tier")!="lower": raise ValueError("Liquidity fallback tier changed from lower.")
    if risk_free.get("parameter_selection_affected") is not False or risk_free.get("strategy_returns_affected") is not False: raise ValueError("Risk-free treatment is no longer descriptive-only.")
    base_costs=DEFAULT_COST_SCENARIOS["base"].turnover_cost_bps_by_tier; tax_row=primary.loc[primary["strategy_family"]=="tax_loss_selling"].iloc[0]
    if tax_row["sample_unit"]!="calendar_years" or int(tax_row["sample_size"])!=26: raise ValueError("Tax-loss primary inferential unit changed from 26 calendar years.")
    return {"status":"built_not_frozen","step":"12.4","step_label":"build_final_publication_provenance_metadata","research_question":PUBLICATION_RESEARCH_QUESTION,"frozen_capstone_commit":FROZEN_CAPSTONE_COMMIT,"publication_branch":"publication-extension","upstream_step8_status":"validated_frozen","upstream_step9_status":"validated_frozen","upstream_step11_status":"validated_frozen","upstream_hash_validation_status":"passed","upstream_artifact_count":upstream.metadata["upstream_artifact_count"],"primary_hypothesis_count":4,"multiple_testing_method":"holm","holm_reject_count":int(primary["reject_after_holm_0_05"].astype(bool).sum()),"primary_results_row_count":int(len(primary)),"canonical_benchmark":external["source_symbol"],"canonical_benchmark_asset_id":int(external["source_asset_id"]),"publication_panel_start_date":external["start_date"],"publication_panel_end_date":external["end_date"],"pit_execution_convention":eligibility["timing_convention"],"identity_convention":eligibility["identity_convention"],"membership_convention":eligibility["membership_convention"],"liquidity_information_set":liquidity["liquidity_information_set"],"liquidity_tier_cutoffs":liquidity["tier_cutoffs"],"liquidity_fallback_tier":liquidity["fallback_tier"],"base_turnover_cost_bps_by_tier":{"high":float(base_costs["high"]),"medium":float(base_costs["medium"]),"lower":float(base_costs["lower"])},"preferred_daily_metric":"net_excess_nav_difference","tax_loss_primary_metric":"abnormal_net_return_difference","tax_loss_primary_inferential_unit":"calendar_years","tax_loss_primary_sample_size":26,"risk_free_source":risk_free["source"],"risk_free_construction":risk_free["construction"],"risk_free_role":risk_free["use"],"risk_free_affects_parameter_selection":bool(risk_free["parameter_selection_affected"]),"risk_free_affects_strategy_returns":bool(risk_free["strategy_returns_affected"]),"trend_major_identified_contributor":step11["major_identified_contributor"],"trend_modest_identified_contributor":step11["modest_identified_contributor"],"trend_unresolved_contributor":step11["unresolved_contributor"],"trend_attribution_is_diagnostic_not_confirmatory":bool(step11["not_part_of_primary_holm_family"]),"confirmatory_vs_diagnostic_boundary":"Four Step 8 strategy hypotheses form the confirmatory Holm family; Step 11 trend attribution is diagnostic only.","raw_licensed_data_excluded":True,"empirical_results_recomputed":False,"manifest_hash_algorithm":"sha256","scope_note":"Step 12.4 records provenance and locked methodological conventions only. It does not rerun, retune, or reinterpret empirical results, and it does not freeze Step 12 until 12.7."}

def build_publication_final_evidence_manifest(results_root: Path) -> pd.DataFrame:
    results_root=Path(results_root); validate_publication_step12_upstream_evidence(results_root); _require_files(results_root,tuple(name for name,_,_ in FINAL_EVIDENCE_ARTIFACTS))
    expected_primary=build_publication_final_primary_results(results_root); observed_primary=pd.read_csv(results_root/FINAL_PRIMARY_RESULTS_NAME); pd.testing.assert_frame_equal(observed_primary,expected_primary,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    expected_metadata=build_publication_final_evidence_metadata(results_root); observed_metadata=_load_json(results_root/FINAL_METADATA_NAME)
    if observed_metadata!=expected_metadata: raise ValueError("Step 12.4 final evidence metadata is stale or differs from the locked builder output.")
    if observed_metadata.get("status")!="built_not_frozen": raise ValueError("Step 12.5 expects Step 12.4 metadata to remain built_not_frozen before final freeze.")
    if observed_metadata.get("raw_licensed_data_excluded") is not True: raise ValueError("Final evidence metadata must explicitly exclude raw licensed data.")
    rows=[]
    for artifact,source_step,artifact_role in FINAL_EVIDENCE_ARTIFACTS:
        path=results_root/artifact; rows.append({"artifact":artifact,"source_step":source_step,"artifact_role":artifact_role,"bytes":int(path.stat().st_size),"sha256":_sha256(path)})
    manifest=pd.DataFrame(rows,columns=["artifact","source_step","artifact_role","bytes","sha256"])
    if manifest["artifact"].duplicated().any(): raise ValueError("Final evidence manifest contains duplicate artifact names.")
    if len(manifest)!=16: raise ValueError(f"Final evidence manifest has {len(manifest)} rows; expected 16.")
    required_roles={"primary_result","comparison_evidence","diagnostic_evidence","methodology_validation","provenance","upstream_freeze"}
    if set(manifest["artifact_role"])!=required_roles: raise ValueError("Final evidence manifest artifact-role coverage changed from the locked contract.")
    return manifest

def _validate_frozen_layer(results_root: Path, *, manifest_name: str, metadata_name: str, expected_step: str, expected_artifact_count: int) -> dict[str,object]:
    manifest_path=results_root/manifest_name; metadata_path=results_root/metadata_name; _require_files(results_root,(manifest_name,metadata_name)); metadata=_load_json(metadata_path)
    if metadata.get("status")!="frozen" or str(metadata.get("step"))!=expected_step: raise ValueError(f"Upstream evidence layer {expected_step} is not frozen with the expected step identifier.")
    manifest=pd.read_csv(manifest_path); required_columns={"artifact","bytes","sha256"}; missing_columns=sorted(required_columns-set(manifest.columns))
    if missing_columns: raise ValueError(f"Upstream Step {expected_step} manifest missing columns: "+", ".join(missing_columns))
    if len(manifest)!=expected_artifact_count: raise ValueError(f"Upstream Step {expected_step} manifest has {len(manifest)} artifacts; expected {expected_artifact_count}.")
    if manifest["artifact"].astype(str).duplicated().any(): raise ValueError(f"Upstream Step {expected_step} manifest contains duplicate artifacts.")
    if int(metadata.get("artifact_count",-1))!=expected_artifact_count: raise ValueError(f"Upstream Step {expected_step} metadata artifact_count does not match the frozen contract.")
    for row in manifest.itertuples(index=False):
        artifact=str(row.artifact); path=results_root/artifact
        if not path.is_file(): raise ValueError(f"Upstream Step {expected_step} frozen artifact missing: {artifact}")
        if int(path.stat().st_size)!=int(row.bytes): raise ValueError(f"Upstream Step {expected_step} frozen artifact size changed: {artifact}")
        if _sha256(path)!=str(row.sha256): raise ValueError(f"Upstream Step {expected_step} frozen artifact hash changed: {artifact}")
    return {"metadata":metadata,"artifact_count":int(len(manifest))}

def _validate_step8_contract(metadata:dict[str,object])->None:
    if metadata.get("validation_status")!="passed": raise ValueError("Step 8 frozen validation status is not passed.")
    if int(metadata.get("validation_failed_check_count",-1))!=0: raise ValueError("Step 8 frozen evidence contains failed validation checks.")
    if int(metadata.get("primary_hypothesis_count",-1))!=4: raise ValueError("Step 8 primary hypothesis count changed from four.")
    if str(metadata.get("multiple_testing_method","")).lower()!="holm": raise ValueError("Step 8 multiple-testing method changed from Holm.")
    if int(metadata.get("reject_count_after_holm_0_05",-1))!=0: raise ValueError("Step 8 Holm rejection count changed from zero.")
def _validate_dependency_chain(step9_metadata,step11_metadata):
    if step9_metadata.get("step8_hash_validation_status")!="passed": raise ValueError("Step 9 does not record a passed Step 8 frozen-hash validation.")
    if step11_metadata.get("step9_frozen_hash_validation_status")!="passed": raise ValueError("Step 11 does not record a passed Step 9 frozen-hash validation.")
def _require_files(root,names):
    missing=[name for name in names if not (root/name).is_file()]
    if missing: raise ValueError("Missing upstream frozen evidence files: "+", ".join(sorted(missing)))
def _load_json(path): return json.loads(path.read_text(encoding="utf-8"))
def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
