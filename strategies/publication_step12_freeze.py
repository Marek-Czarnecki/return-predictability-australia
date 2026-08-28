from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .publication_step12_evidence import FINAL_MANIFEST_NAME, FINAL_METADATA_NAME, FINAL_PRIMARY_RESULTS_NAME, FROZEN_CAPSTONE_COMMIT
from .publication_step12_validation import validate_publication_step12_scientific_invariants

STEP12_FREEZE_MANIFEST_NAME = "publication_step12_evidence_manifest.csv"
STEP12_FREEZE_METADATA_NAME = "publication_step12_evidence_metadata.json"
STEP12_FREEZE_ARTIFACTS = (FINAL_PRIMARY_RESULTS_NAME, FINAL_METADATA_NAME, FINAL_MANIFEST_NAME)

@dataclass(frozen=True)
class PublicationStep12Freeze:
    manifest: pd.DataFrame
    metadata: dict[str, object]

def validate_and_freeze_publication_step12_evidence(results_root: Path) -> PublicationStep12Freeze:
    results_root=Path(results_root); validation=validate_publication_step12_scientific_invariants(results_root)
    if validation.metadata.get("status")!="passed": raise ValueError("Step 12.6 scientific invariant validation has not passed.")
    if int(validation.metadata.get("validation_failed_check_count",-1))!=0: raise ValueError("Step 12.6 contains failed scientific invariant checks.")
    if validation.metadata.get("empirical_results_recomputed") is not False: raise ValueError("Step 12.6 unexpectedly recomputed empirical results.")
    _require_files(results_root,STEP12_FREEZE_ARTIFACTS)
    final_metadata=_load_json(results_root/FINAL_METADATA_NAME); final_manifest=pd.read_csv(results_root/FINAL_MANIFEST_NAME); final_primary=pd.read_csv(results_root/FINAL_PRIMARY_RESULTS_NAME)
    if final_metadata.get("status")!="built_not_frozen": raise ValueError("Step 12.4 metadata must remain built_not_frozen before the Step 12.7 freeze layer is created.")
    if final_metadata.get("raw_licensed_data_excluded") is not True: raise ValueError("Final evidence metadata must exclude raw licensed data.")
    if final_metadata.get("empirical_results_recomputed") is not False: raise ValueError("Final evidence metadata indicates empirical recomputation.")
    if int(final_metadata.get("primary_hypothesis_count",-1))!=4: raise ValueError("Final evidence metadata no longer records four primary hypotheses.")
    if int(final_metadata.get("holm_reject_count",-1))!=0: raise ValueError("Final evidence metadata no longer records 0/4 Holm rejections.")
    if len(final_primary)!=4: raise ValueError("Final primary results must contain exactly four rows at freeze.")
    if int(final_primary["reject_after_holm_0_05"].astype(bool).sum())!=0: raise ValueError("Final primary results no longer contain 0/4 Holm rejections.")
    if len(final_manifest)!=16: raise ValueError("Definitive final evidence manifest must contain 16 artifacts at freeze.")
    _validate_final_manifest_hashes(results_root,final_manifest)
    rows=[]
    for artifact in STEP12_FREEZE_ARTIFACTS:
        path=results_root/artifact; rows.append({"artifact":artifact,"bytes":int(path.stat().st_size),"sha256":_sha256(path)})
    manifest=pd.DataFrame(rows,columns=["artifact","bytes","sha256"])
    metadata={"status":"frozen","step":"12.7","step_label":"freeze_final_publication_evidence","frozen_capstone_commit":FROZEN_CAPSTONE_COMMIT,"publication_branch":"publication-extension","artifact_count":int(len(manifest)),"definitive_final_evidence_artifact_count":int(len(final_manifest)),"primary_hypothesis_count":4,"holm_reject_count":0,"step12_6_validation_status":"passed","step12_6_validation_check_count":int(validation.metadata["validation_check_count"]),"step12_6_failed_check_count":int(validation.metadata["validation_failed_check_count"]),"upstream_integrity_validation_status":validation.metadata["upstream_integrity_validation_status"],"final_primary_reconciliation_status":validation.metadata["final_primary_reconciliation_status"],"final_metadata_reconciliation_status":validation.metadata["final_metadata_reconciliation_status"],"final_manifest_reconciliation_status":validation.metadata["final_manifest_reconciliation_status"],"trend_attribution_boundary_status":validation.metadata["trend_attribution_boundary_status"],"manifest_hash_algorithm":"sha256","raw_licensed_data_excluded":True,"empirical_results_recomputed":False,"confirmatory_result":"0_of_4_supported_after_holm","scope_note":"Freezes the Step 12 publication evidence state for manuscript use. The freeze records validated hashes of the final primary-results table, provenance metadata, and definitive final evidence manifest. It does not rerun or reinterpret empirical analyses."}
    return PublicationStep12Freeze(manifest,metadata)

def _validate_final_manifest_hashes(results_root,manifest):
    required={"artifact","bytes","sha256"}
    if not required.issubset(manifest.columns): raise ValueError("Definitive final evidence manifest is missing required hash columns.")
    if manifest["artifact"].astype(str).duplicated().any(): raise ValueError("Definitive final evidence manifest contains duplicate artifacts.")
    for row in manifest.itertuples(index=False):
        path=results_root/str(row.artifact)
        if not path.is_file(): raise ValueError(f"Definitive final evidence artifact missing: {row.artifact}")
        if int(path.stat().st_size)!=int(row.bytes): raise ValueError(f"Definitive final evidence artifact size changed: {row.artifact}")
        if _sha256(path)!=str(row.sha256): raise ValueError(f"Definitive final evidence artifact hash changed: {row.artifact}")
def _require_files(root,names):
    missing=[name for name in names if not (root/name).is_file()]
    if missing: raise ValueError("Missing Step 12 freeze inputs: "+", ".join(sorted(missing)))
def _load_json(path): return json.loads(path.read_text(encoding="utf-8"))
def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
