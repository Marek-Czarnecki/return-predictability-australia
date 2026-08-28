from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_step12_evidence import (
    FINAL_MANIFEST_NAME,
    build_publication_final_evidence_manifest,
)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the compact Step 12.5 final publication evidence manifest."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_publication_final_evidence_manifest(args.results_root)
    output_path = args.results_root / FINAL_MANIFEST_NAME
    manifest.to_csv(output_path, index=False)
    print("status=complete")
    print("step=12.5")
    print(f"artifacts={len(manifest)}")
    print(f"roles={manifest['artifact_role'].nunique()}")
    print("upstream_integrity_validation=passed")
    print("final_primary_reconciliation=passed")
    print("final_metadata_reconciliation=passed")
    print("raw_licensed_data_excluded=true")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
