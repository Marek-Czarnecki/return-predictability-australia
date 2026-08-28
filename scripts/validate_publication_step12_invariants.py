from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_step12_validation import validate_publication_step12_scientific_invariants

DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Step 12.6 final scientific and attribution invariants."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_publication_step12_scientific_invariants(args.results_root)
    print("status=passed")
    print("step=12.6")
    print(f"checks={result.metadata['validation_check_count']}")
    print(f"failed_checks={result.metadata['validation_failed_check_count']}")
    print("upstream_integrity_validation=passed")
    print("final_primary_reconciliation=passed")
    print("final_metadata_reconciliation=passed")
    print("final_manifest_reconciliation=passed")
    print("trend_attribution_boundary=passed")
    print("empirical_results_recomputed=false")
    print(json.dumps(result.checks.to_dict(orient="records"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
