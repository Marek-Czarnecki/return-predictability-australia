from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_comparison import (
    ATTRIBUTION_NAME,
    COMPARISON_NAME,
    METADATA_NAME,
    build_publication_comparison,
)


DEFAULT_PUBLICATION_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Step 9 corrected-vs-frozen publication comparison artifacts."
    )
    parser.add_argument(
        "--frozen-results-root",
        type=Path,
        required=True,
        help=(
            "Directory containing the frozen original-capstone strategy result artifacts. "
            "These source artifacts are not redistributed by this publication repository."
        ),
    )
    parser.add_argument(
        "--publication-results-root",
        type=Path,
        default=DEFAULT_PUBLICATION_RESULTS_ROOT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_publication_comparison(
        args.frozen_results_root,
        args.publication_results_root,
    )

    comparison_path = args.publication_results_root / COMPARISON_NAME
    attribution_path = args.publication_results_root / ATTRIBUTION_NAME
    metadata_path = args.publication_results_root / METADATA_NAME

    result.comparison.to_csv(comparison_path, index=False)
    result.attribution.to_csv(attribution_path, index=False)
    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("status=complete")
    print("step=9.6")
    print(f"strategies={len(result.comparison)}")
    print(f"attribution_rows={len(result.attribution)}")
    print(f"comparison={comparison_path}")
    print(f"attribution={attribution_path}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
