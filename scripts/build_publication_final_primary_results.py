from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_step12_evidence import (
    FINAL_PRIMARY_RESULTS_NAME,
    build_publication_final_primary_results,
)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Step 12.3 four-row final publication primary-results table from frozen evidence."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = build_publication_final_primary_results(args.results_root)
    output_path = args.results_root / FINAL_PRIMARY_RESULTS_NAME
    table.to_csv(output_path, index=False)
    print("status=complete")
    print("step=12.3")
    print(f"rows={len(table)}")
    print("empirical_results_recomputed=false")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
