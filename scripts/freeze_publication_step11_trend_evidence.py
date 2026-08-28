from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_step11_trend_evidence import (
    STEP11_MANIFEST_NAME,
    STEP11_METADATA_NAME,
    validate_and_freeze_publication_step11_trend_evidence,
)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze Step 11.1 trend-collapse diagnostic evidence."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_and_freeze_publication_step11_trend_evidence(args.results_root)
    manifest_path = args.results_root / STEP11_MANIFEST_NAME
    metadata_path = args.results_root / STEP11_METADATA_NAME
    result.manifest.to_csv(manifest_path, index=False)
    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("status=frozen")
    print("step=11.1.6")
    print(f"artifacts={len(result.manifest)}")
    print("step9_frozen_hash_validation=passed")
    print("required_artifact_validation=passed")
    print("fold_reconciliation=passed")
    print("control_value_reconciliation=passed")
    print("attribution_classification_validation=passed")
    print(f"manifest={manifest_path}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
