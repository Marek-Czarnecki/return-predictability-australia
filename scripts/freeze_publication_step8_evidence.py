from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_evidence import (
    STEP8_MANIFEST_NAME,
    STEP8_METADATA_NAME,
    freeze_publication_step8_evidence,
)


DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the complete Step 8 publication evidence set."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = freeze_publication_step8_evidence(args.results_root)

    manifest_path = args.results_root / STEP8_MANIFEST_NAME
    metadata_path = args.results_root / STEP8_METADATA_NAME
    result.manifest.to_csv(manifest_path, index=False)
    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("status=frozen")
    print("step=8")
    print(f"artifacts={len(result.manifest)}")
    print(f"validation_checks={result.metadata['validation_check_count']}")
    print(f"primary_hypotheses={result.metadata['primary_hypothesis_count']}")
    print(f"holm_rejects={result.metadata['reject_count_after_holm_0_05']}")
    print(f"manifest={manifest_path}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
