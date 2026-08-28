from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_step12_evidence import (
    FINAL_METADATA_NAME,
    build_publication_final_evidence_metadata,
)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step 12.4 publication provenance/methodological metadata."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = build_publication_final_evidence_metadata(args.results_root)
    output_path = args.results_root / FINAL_METADATA_NAME
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print("status=complete")
    print("step=12.4")
    print(f"primary_hypotheses={metadata['primary_hypothesis_count']}")
    print(f"holm_reject_count={metadata['holm_reject_count']}")
    print(f"upstream_artifacts={metadata['upstream_artifact_count']}")
    print("empirical_results_recomputed=false")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
