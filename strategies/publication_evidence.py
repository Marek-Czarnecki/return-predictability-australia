from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


STEP8_MANIFEST_NAME = "publication_step8_evidence_manifest.csv"
STEP8_METADATA_NAME = "publication_step8_evidence_metadata.json"

CRITICAL_STEP8_ARTIFACTS = (
    "publication_external_benchmark.csv",
    "publication_benchmark_validation.json",
    "publication_trend_following_walk_forward_summary.csv",
    "publication_mean_reversion_walk_forward_summary.csv",
    "publication_pairs_trading_walk_forward_summary.csv",
    "publication_tax_loss_selling_event_study.csv",
    "publication_tax_loss_selling_summary.csv",
    "publication_tax_loss_selling_year_robustness.csv",
    "publication_risk_free.csv",
    "publication_risk_free_metadata.json",
    "publication_validation_checks.csv",
    "publication_validation_metadata.json",
    "publication_validation_metric_audit.csv",
    "publication_validation_risk_free_coverage.csv",
    "publication_primary_inference.csv",
    "publication_tax_loss_primary_year_effects.csv",
    "publication_primary_inference_metadata.json",
)


@dataclass(frozen=True)
class PublicationStep8Evidence:
    manifest: pd.DataFrame
    metadata: dict[str, object]


def freeze_publication_step8_evidence(results_root: Path) -> PublicationStep8Evidence:
    results_root = Path(results_root)
    if not results_root.exists():
        raise ValueError(f"Publication results directory does not exist: {results_root}")

    missing = [name for name in CRITICAL_STEP8_ARTIFACTS if not (results_root / name).is_file()]
    if missing:
        raise ValueError("Missing critical Step 8 artifacts: " + ", ".join(sorted(missing)))

    validation_metadata = _load_json(results_root / "publication_validation_metadata.json")
    if validation_metadata.get("status") != "passed":
        raise ValueError("Publication validation status is not passed.")
    if int(validation_metadata.get("failed_check_count", -1)) != 0:
        raise ValueError("Publication validation metadata reports failed checks.")
    if validation_metadata.get("risk_free_coverage_status") != "complete":
        raise ValueError("Publication risk-free coverage is not complete.")

    inference_metadata = _load_json(results_root / "publication_primary_inference_metadata.json")
    if inference_metadata.get("status") != "complete":
        raise ValueError("Publication primary inference status is not complete.")
    if int(inference_metadata.get("primary_hypothesis_count", -1)) != 4:
        raise ValueError("Publication primary inference must contain four hypotheses.")
    if inference_metadata.get("multiple_testing_method") != "holm":
        raise ValueError("Publication primary inference is not using Holm adjustment.")

    excluded = {STEP8_MANIFEST_NAME, STEP8_METADATA_NAME}
    files = sorted(
        path for path in results_root.iterdir()
        if path.is_file() and path.name not in excluded
    )
    rows = [
        {
            "artifact": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in files
    ]
    manifest = pd.DataFrame(rows, columns=["artifact", "bytes", "sha256"])

    primary = pd.read_csv(results_root / "publication_primary_inference.csv")
    if len(primary) != 4:
        raise ValueError("Publication primary inference CSV must contain four rows.")
    adjusted = pd.to_numeric(primary["adjusted_p_value"], errors="coerce")
    if adjusted.isna().any():
        raise ValueError("Publication primary inference has missing Holm-adjusted p-values.")

    metadata: dict[str, object] = {
        "status": "frozen",
        "step": "8",
        "step_label": "controlled_historical_reruns",
        "artifact_count": int(len(manifest)),
        "validation_status": "passed",
        "validation_check_count": int(validation_metadata.get("check_count", 0)),
        "validation_failed_check_count": 0,
        "risk_free_coverage_status": "complete",
        "primary_hypothesis_count": 4,
        "multiple_testing_method": "holm",
        "reject_count_after_holm_0_05": int(
            inference_metadata.get("reject_count_after_holm_0_05", 0)
        ),
        "walk_forward_primary_metric": inference_metadata.get("walk_forward_primary_metric"),
        "tax_loss_primary_metric": inference_metadata.get("tax_loss_primary_metric"),
        "manifest_hash_algorithm": "sha256",
        "scope_note": (
            "Freezes the Step 8 publication_results evidence set only. Later Step 12 may freeze "
            "the final publication evidence after corrected-vs-frozen comparison and any approved robustness work."
        ),
    }
    return PublicationStep8Evidence(manifest=manifest, metadata=metadata)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
