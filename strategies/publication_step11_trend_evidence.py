from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FROZEN_CAPSTONE_COMMIT = "349fc00b087d404ba11bb9f04e9fe8ba7ad58ed6"
STEP11_MANIFEST_NAME = "publication_step11_trend_evidence_manifest.csv"
STEP11_METADATA_NAME = "publication_step11_trend_evidence_metadata.json"
EXPECTED_COMMON_PERIOD_MEAN = -0.041322384896077646
EXPECTED_FROZEN_RECONSTRUCTED_MEAN = 0.10276428548304055
EXPECTED_FULL_PUBLICATION_MEAN = 0.0008204009098994785

STEP11_ARTIFACTS = (
    "publication_step11_trend_common_period_folds.csv",
    "publication_step11_trend_common_period_daily_returns.csv",
    "publication_step11_trend_common_period_summary.csv",
    "publication_step11_trend_common_period_liquidity_diagnostics.csv",
    "publication_step11_trend_common_period_metadata.json",
    "publication_step11_trend_universe_ablation_comparison.csv",
    "publication_step11_trend_universe_ablation_norgate_reference_universe.csv",
    "publication_step11_trend_universe_ablation_pit_summary.csv",
    "publication_step11_trend_universe_ablation_retrospective_summary.csv",
    "publication_step11_trend_universe_ablation_metadata.json",
    "publication_step11_trend_universe_mapping_diagnostics.csv",
    "publication_step11_trend_benchmark_ablation_comparison.csv",
    "publication_step11_trend_benchmark_ablation_xjoa_summary.csv",
    "publication_step11_trend_benchmark_ablation_stw_summary.csv",
    "publication_step11_trend_benchmark_ablation_metadata.json",
    "publication_step11_trend_cost_ablation_comparison.csv",
    "publication_step11_trend_cost_ablation_base_summary.csv",
    "publication_step11_trend_cost_ablation_zero_summary.csv",
    "publication_step11_trend_cost_ablation_metadata.json",
    "publication_step11_trend_attribution.csv",
    "publication_step11_trend_attribution_metadata.json",
)

@dataclass(frozen=True)
class PublicationStep11TrendEvidence:
    manifest: pd.DataFrame
    metadata: dict[str, object]

def validate_and_freeze_publication_step11_trend_evidence(results_root: Path) -> PublicationStep11TrendEvidence:
    results_root = Path(results_root)
    _require_files(results_root, STEP11_ARTIFACTS)
    _validate_step9_manifest(results_root)
    common = _load_json(results_root / "publication_step11_trend_common_period_metadata.json")
    universe = _load_json(results_root / "publication_step11_trend_universe_ablation_metadata.json")
    benchmark = _load_json(results_root / "publication_step11_trend_benchmark_ablation_metadata.json")
    cost = _load_json(results_root / "publication_step11_trend_cost_ablation_metadata.json")
    attribution_meta = _load_json(results_root / "publication_step11_trend_attribution_metadata.json")
    attribution = pd.read_csv(results_root / "publication_step11_trend_attribution.csv")
    comparison = pd.read_csv(results_root / "publication_corrected_vs_frozen_comparison.csv")
    _validate_common_period(common); _validate_universe(universe, common); _validate_benchmark(benchmark, common); _validate_cost(cost, common); _validate_attribution(attribution, attribution_meta); _validate_step9_trend_anchor(comparison)
    rows=[]
    for name in STEP11_ARTIFACTS:
        path=results_root/name
        rows.append({"artifact":name,"bytes":int(path.stat().st_size),"sha256":_sha256(path)})
    manifest=pd.DataFrame(rows,columns=["artifact","bytes","sha256"])
    metadata={
        "status":"frozen","step":"11.1.6","step_label":"validate_and_freeze_step11_trend_evidence","strategy_name":"trend_following","frozen_capstone_commit":FROZEN_CAPSTONE_COMMIT,"artifact_count":int(len(manifest)),"manifest_hash_algorithm":"sha256","step9_frozen_hash_validation_status":"passed","required_artifact_validation_status":"passed","fold_reconciliation_status":"passed","control_value_reconciliation_status":"passed","attribution_classification_validation_status":"passed","common_period_mean_net_excess_nav_difference":float(common["mean_net_excess_nav_difference"]),"frozen_reconstructed_nav_difference_mean":EXPECTED_FROZEN_RECONSTRUCTED_MEAN,"full_publication_mean_net_excess_nav_difference":EXPECTED_FULL_PUBLICATION_MEAN,"major_identified_contributor":"point_in_time_membership_vs_retrospective_current_membership","modest_identified_contributor":"transaction_costs","unresolved_contributor":"vendor_and_security_coverage","confirmatory":False,"not_part_of_primary_holm_family":True,"scope_note":"Freezes the Step 11.1 trend-collapse diagnostic evidence only. It does not alter the frozen capstone, Step 8 publication evidence, Step 9 comparison evidence, or create new confirmatory hypotheses."
    }
    return PublicationStep11TrendEvidence(manifest,metadata)

def _validate_step9_manifest(results_root: Path) -> None:
    manifest_path=results_root/"publication_step9_evidence_manifest.csv"; metadata_path=results_root/"publication_step9_evidence_metadata.json"; _require_files(results_root,(manifest_path.name,metadata_path.name)); metadata=_load_json(metadata_path)
    if metadata.get("status")!="frozen" or str(metadata.get("step"))!="9": raise ValueError("Step 9 evidence metadata is not frozen.")
    manifest=pd.read_csv(manifest_path); required={"artifact","bytes","sha256"}
    if not required.issubset(manifest.columns): raise ValueError("Step 9 evidence manifest is missing required columns.")
    for row in manifest.itertuples(index=False):
        path=results_root/str(row.artifact)
        if not path.is_file(): raise ValueError(f"Step 9 frozen artifact missing: {row.artifact}")
        if int(path.stat().st_size)!=int(row.bytes) or _sha256(path)!=str(row.sha256): raise ValueError(f"Step 9 frozen artifact changed: {row.artifact}")

def _validate_common_period(meta):
    _validate_diagnostic_metadata(meta,"11.1.1")
    if int(meta.get("fold_count",-1))!=7: raise ValueError("Step 11.1.1 must contain seven frozen-calendar folds.")
    if meta.get("causal_scope")!="sample_period_and_fold_calendar": raise ValueError("Step 11.1.1 causal_scope must be sample_period_and_fold_calendar before freeze.")
    _assert_close(float(meta["mean_net_excess_nav_difference"]),EXPECTED_COMMON_PERIOD_MEAN,"11.1.1 common-period mean")
    if int(meta.get("positive_fold_count",-1))!=1: raise ValueError("Step 11.1.1 positive-fold count must be 1/7.")

def _validate_universe(meta,common):
    _validate_diagnostic_metadata(meta,"11.1.2")
    if int(meta.get("fold_count",-1))!=7: raise ValueError("Step 11.1.2 must contain seven folds.")
    if int(meta.get("reference_universe_asset_count",-1))!=200: raise ValueError("Step 11.1.2 reference universe must contain 200 Norgate asset_ids.")
    _assert_close(float(meta["pit_mean_net_excess_nav_difference"]),float(common["mean_net_excess_nav_difference"]),"11.1.2 PIT control")
    effect=float(meta["retrospective_mean_net_excess_nav_difference"])-float(meta["pit_mean_net_excess_nav_difference"]); _assert_close(float(meta["mean_universe_effect_nav_difference"]),effect,"11.1.2 universe effect")
    if int(meta.get("pit_positive_fold_count",-1))!=1 or int(meta.get("retrospective_positive_fold_count",-1))!=6: raise ValueError("Step 11.1.2 positive-fold reconciliation must be PIT 1/7 and retrospective 6/7.")

def _validate_benchmark(meta,common):
    _validate_diagnostic_metadata(meta,"11.1.3"); _assert_close(float(meta["xjoa_mean_net_excess_nav_difference"]),float(common["mean_net_excess_nav_difference"]),"11.1.3 XJOA control"); effect=float(meta["stw_mean_net_excess_nav_difference"])-float(meta["xjoa_mean_net_excess_nav_difference"]); _assert_close(float(meta["mean_benchmark_effect_nav_difference"]),effect,"11.1.3 benchmark effect")
    if int(meta.get("parameter_selection_change_count",-1))!=0: raise ValueError("Step 11.1.3 parameter-selection change count must be 0/7.")

def _validate_cost(meta,common):
    _validate_diagnostic_metadata(meta,"11.1.4"); _assert_close(float(meta["base_mean_net_excess_nav_difference"]),float(common["mean_net_excess_nav_difference"]),"11.1.4 base-cost control"); effect=float(meta["zero_mean_net_excess_nav_difference"])-float(meta["base_mean_net_excess_nav_difference"]); _assert_close(float(meta["mean_cost_effect_nav_difference"]),effect,"11.1.4 cost effect")
    if int(meta.get("parameter_selection_change_count",-1))!=1: raise ValueError("Step 11.1.4 parameter-selection change count must be 1/7.")

def _validate_attribution(table,meta):
    if str(meta.get("step"))!="11.1.5" or meta.get("confirmatory") is not False: raise ValueError("Step 11.1.5 attribution metadata contract is invalid.")
    if len(table)!=7: raise ValueError("Step 11.1.5 attribution table must contain seven rows.")
    classes=table.set_index("design_change")["attribution_class"].astype(str).to_dict(); expected={"metric_semantics":"not_explanatory","risk_free_treatment":"not_explanatory","sample_period_and_fold_calendar":"not_explanatory","point_in_time_membership":"directly_demonstrated","benchmark_choice":"not_explanatory","transaction_costs":"directly_demonstrated","vendor_and_security_coverage":"unresolved_contributor"}
    if classes!=expected: raise ValueError(f"Step 11.1.5 attribution classifications changed: {classes}")
    statement=str(meta.get("attribution_statement","")); statement_lower=statement.lower()
    if "major identified contributor" not in statement or "security coverage remains unresolved" not in statement: raise ValueError("Step 11.1.5 bounded attribution statement is incomplete.")
    search_from=0
    while True:
        index=statement_lower.find("fully explains",search_from)
        if index<0: break
        context=statement_lower[max(0,index-64):index]; bounded_prefixes=("rather than a claim that it ","do not claim that it ","cannot claim that it ","not claim that it ")
        if not any(context.endswith(prefix) for prefix in bounded_prefixes): raise ValueError("Step 11.1.5 attribution statement overclaims causal completeness.")
        search_from=index+len("fully explains")

def _validate_step9_trend_anchor(comparison):
    row=comparison.loc[comparison["strategy_family"]=="trend_following"]
    if len(row)!=1: raise ValueError("Step 9 comparison must contain exactly one trend_following row.")
    record=row.iloc[0]; _assert_close(float(record["frozen_reconstructed_nav_difference_mean"]),EXPECTED_FROZEN_RECONSTRUCTED_MEAN,"frozen trend anchor"); _assert_close(float(record["publication_nav_difference_mean"]),EXPECTED_FULL_PUBLICATION_MEAN,"publication trend anchor")

def _validate_diagnostic_metadata(meta,step):
    if str(meta.get("step"))!=step: raise ValueError(f"Expected Step {step} metadata.")
    if meta.get("analysis_role")!="diagnostic_ablation" or meta.get("confirmatory") is not False: raise ValueError(f"Step {step} must remain a non-confirmatory diagnostic ablation.")
    if meta.get("frozen_capstone_commit")!=FROZEN_CAPSTONE_COMMIT: raise ValueError(f"Step {step} references the wrong frozen capstone commit.")
    if int(meta.get("fold_count",-1))!=7: raise ValueError(f"Step {step} must contain seven frozen-calendar folds.")

def _assert_close(observed,expected,label,tolerance=1e-12):
    if abs(observed-expected)>tolerance: raise ValueError(f"{label} changed: observed={observed}, expected={expected}")
def _require_files(root,names):
    missing=[name for name in names if not (root/name).is_file()]
    if missing: raise ValueError("Missing Step 11 trend evidence inputs: "+", ".join(sorted(missing)))
def _load_json(path): return json.loads(path.read_text(encoding="utf-8"))
def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
