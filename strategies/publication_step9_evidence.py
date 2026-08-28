from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from strategies.publication_comparison import ATTRIBUTION_NAME, COMPARISON_NAME, FROZEN_CAPSTONE_COMMIT, METADATA_NAME, build_publication_comparison

STEP9_MANIFEST_NAME = "publication_step9_evidence_manifest.csv"
STEP9_METADATA_NAME = "publication_step9_evidence_metadata.json"
STEP9_ARTIFACTS = (COMPARISON_NAME, ATTRIBUTION_NAME, METADATA_NAME)
EXPECTED_STRATEGIES = ("trend_following", "mean_reversion", "pairs_trading", "tax_loss_selling")
EXPECTED_EVIDENCE_STRENGTH = {
    "trend_following": "supported_to_unsupported",
    "mean_reversion": "unsupported_to_unsupported",
    "pairs_trading": "unsupported_to_unsupported",
    "tax_loss_selling": "supported_to_unsupported",
}

@dataclass(frozen=True)
class PublicationStep9Evidence:
    manifest: pd.DataFrame
    metadata: dict[str, object]

def validate_and_freeze_publication_step9_evidence(frozen_results_root: Path, publication_results_root: Path) -> PublicationStep9Evidence:
    frozen_results_root=Path(frozen_results_root); publication_results_root=Path(publication_results_root)
    _require_files(publication_results_root,STEP9_ARTIFACTS); _validate_step8_manifest(publication_results_root)
    stored_comparison=pd.read_csv(publication_results_root/COMPARISON_NAME); stored_attribution=pd.read_csv(publication_results_root/ATTRIBUTION_NAME); stored_metadata=_load_json(publication_results_root/METADATA_NAME)
    _validate_stored_step9_outputs(stored_comparison,stored_attribution,stored_metadata)
    rebuilt=build_publication_comparison(frozen_results_root,publication_results_root)
    assert_frame_equal(stored_comparison,rebuilt.comparison,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    assert_frame_equal(stored_attribution,rebuilt.attribution,check_dtype=False,check_exact=True)
    if stored_metadata!=rebuilt.metadata: raise ValueError("Stored Step 9 comparison metadata does not match a fresh rebuild.")
    rows=[]
    for name in STEP9_ARTIFACTS:
        path=publication_results_root/name; rows.append({"artifact":name,"bytes":int(path.stat().st_size),"sha256":_sha256(path)})
    manifest=pd.DataFrame(rows,columns=["artifact","bytes","sha256"])
    metadata={"status":"frozen","step":"9","step_label":"corrected_vs_frozen_comparison","frozen_capstone_commit":FROZEN_CAPSTONE_COMMIT,"artifact_count":int(len(manifest)),"comparison_strategy_count":int(len(stored_comparison)),"attribution_row_count":int(len(stored_attribution)),"step8_hash_validation_status":"passed","comparison_rebuild_validation_status":"passed","expected_evidence_strength_status":"passed","manifest_hash_algorithm":"sha256","scope_note":"Freezes Step 9 corrected-vs-frozen comparison evidence only. It does not start Step 10 publication-worthiness assessment or alter the frozen capstone/Step 8 evidence."}
    return PublicationStep9Evidence(manifest,metadata)

def _validate_step8_manifest(results_root: Path) -> None:
    manifest_path=results_root/"publication_step8_evidence_manifest.csv"; metadata_path=results_root/"publication_step8_evidence_metadata.json"; _require_files(results_root,(manifest_path.name,metadata_path.name)); metadata=_load_json(metadata_path)
    if metadata.get("status")!="frozen" or str(metadata.get("step"))!="8": raise ValueError("Step 8 evidence metadata is not frozen.")
    manifest=pd.read_csv(manifest_path); required={"artifact","bytes","sha256"}; missing=sorted(required-set(manifest.columns))
    if missing: raise ValueError("Step 8 evidence manifest missing columns: "+", ".join(missing))
    for row in manifest.itertuples(index=False):
        path=results_root/str(row.artifact)
        if not path.is_file(): raise ValueError(f"Step 8 frozen artifact missing: {row.artifact}")
        if int(path.stat().st_size)!=int(row.bytes) or _sha256(path)!=str(row.sha256): raise ValueError(f"Step 8 frozen artifact changed: {row.artifact}")

def _validate_stored_step9_outputs(comparison,attribution,metadata):
    if metadata.get("status")!="complete" or str(metadata.get("step"))!="9.6": raise ValueError("Step 9.6 comparison metadata is not complete.")
    if metadata.get("frozen_capstone_commit")!=FROZEN_CAPSTONE_COMMIT: raise ValueError("Step 9.6 metadata references the wrong frozen capstone commit.")
    if len(comparison)!=4 or comparison["strategy_family"].nunique()!=4: raise ValueError("Step 9 comparison must contain exactly four strategy rows.")
    if set(comparison["strategy_family"].astype(str))!=set(EXPECTED_STRATEGIES): raise ValueError("Step 9 comparison strategy set is incorrect.")
    if "evidence_strength_change" not in comparison.columns: raise ValueError("Step 9 comparison is missing evidence_strength_change.")
    for strategy,expected in EXPECTED_EVIDENCE_STRENGTH.items():
        observed=str(comparison.loc[comparison["strategy_family"]==strategy].iloc[0]["evidence_strength_change"])
        if observed!=expected: raise ValueError(f"Unexpected evidence_strength_change for {strategy}: {observed}; expected {expected}.")
    allowed={"directly_demonstrated","plausible_contributor","not_explanatory"}
    if not set(attribution["attribution_class"].astype(str)).issubset(allowed): raise ValueError("Step 9 attribution contains an unsupported attribution class.")

def _require_files(root,names):
    missing=[name for name in names if not (root/name).is_file()]
    if missing: raise ValueError("Missing Step 9 evidence inputs: "+", ".join(sorted(missing)))
def _load_json(path): return json.loads(path.read_text(encoding="utf-8"))
def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
