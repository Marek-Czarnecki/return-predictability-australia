from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_step12_evidence import validate_publication_step12_upstream_evidence

DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen Step 8, Step 9, and Step 11 evidence for Step 12.2."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_publication_step12_upstream_evidence(args.results_root)
    print("status=passed")
    print("step=12.2")
    print(f"upstream_layers={result.metadata['upstream_layer_count']}")
    print(f"upstream_artifacts={result.metadata['upstream_artifact_count']}")
    print("step8_hash_validation=passed")
    print("step9_hash_validation=passed")
    print("step11_hash_validation=passed")
    print("dependency_chain_validation=passed")
    print("empirical_results_recomputed=false")
    print(json.dumps(result.summary.to_dict(orient="records"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
