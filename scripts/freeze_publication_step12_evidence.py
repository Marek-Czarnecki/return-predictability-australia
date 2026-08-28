from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_step12_freeze import (
    STEP12_FREEZE_MANIFEST_NAME,
    STEP12_FREEZE_METADATA_NAME,
    validate_and_freeze_publication_step12_evidence,
)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and freeze final Step 12 publication evidence.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    result = validate_and_freeze_publication_step12_evidence(args.results_root)

    manifest_path = args.results_root / STEP12_FREEZE_MANIFEST_NAME
    metadata_path = args.results_root / STEP12_FREEZE_METADATA_NAME
    result.manifest.to_csv(manifest_path, index=False)
    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("status=frozen")
    print("step=12.7")
    print(f"freeze_artifacts={result.metadata['artifact_count']}")
    print(f"definitive_final_evidence_artifacts={result.metadata['definitive_final_evidence_artifact_count']}")
    print(f"step12_6_checks={result.metadata['step12_6_validation_check_count']}")
    print(f"step12_6_failed_checks={result.metadata['step12_6_failed_check_count']}")
    print(f"holm_reject_count={result.metadata['holm_reject_count']}")
    print("empirical_results_recomputed=false")
    print(f"manifest={manifest_path}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
