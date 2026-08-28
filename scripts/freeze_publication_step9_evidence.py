from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_comparison import FROZEN_CAPSTONE_COMMIT
from strategies.publication_step9_evidence import (
    STEP9_MANIFEST_NAME,
    STEP9_METADATA_NAME,
    validate_and_freeze_publication_step9_evidence,
)


DEFAULT_PUBLICATION_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"

FROZEN_SOURCE_PATHS = (
    "data/processed/strategy_results/trend_following_walk_forward_summary.csv",
    "data/processed/strategy_results/mean_reversion_walk_forward_summary.csv",
    "data/processed/strategy_results/pairs_trading_walk_forward_summary.csv",
    "data/processed/strategy_results/tax_loss_selling_summary.csv",
    "data/processed/strategy_results/tax_loss_selling_year_robustness.csv",
    "data/processed/strategy_results/statistical_inference_summary.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze Step 9 corrected-vs-frozen comparison evidence."
    )
    parser.add_argument(
        "--frozen-results-root",
        type=Path,
        required=True,
        help="Directory containing the frozen original-capstone comparison inputs.",
    )
    parser.add_argument(
        "--publication-results-root",
        type=Path,
        default=DEFAULT_PUBLICATION_RESULTS_ROOT,
    )
    parser.add_argument(
        "--frozen-capstone-repo",
        type=Path,
        default=None,
        help=(
            "Optional checkout of the original capstone repository. When supplied, the script "
            "also verifies that the six comparison source artifacts match the frozen capstone commit."
        ),
    )
    return parser.parse_args()


def _validate_frozen_sources_unchanged(repo_root: Path) -> None:
    command = [
        "git",
        "diff",
        "--quiet",
        FROZEN_CAPSTONE_COMMIT,
        "--",
        *FROZEN_SOURCE_PATHS,
    ]
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode == 1:
        raise ValueError("Frozen capstone comparison source artifacts differ from the frozen commit.")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Unable to validate frozen capstone source artifacts; git diff returned {completed.returncode}."
        )


def main() -> int:
    args = parse_args()
    if args.frozen_capstone_repo is not None:
        _validate_frozen_sources_unchanged(args.frozen_capstone_repo)

    result = validate_and_freeze_publication_step9_evidence(
        args.frozen_results_root,
        args.publication_results_root,
    )

    manifest_path = args.publication_results_root / STEP9_MANIFEST_NAME
    metadata_path = args.publication_results_root / STEP9_METADATA_NAME
    result.manifest.to_csv(manifest_path, index=False)
    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("status=frozen")
    print("step=9")
    print(f"artifacts={len(result.manifest)}")
    print(
        "frozen_sources_unchanged="
        + ("passed" if args.frozen_capstone_repo is not None else "not_checked_no_source_repo_supplied")
    )
    print("step8_hash_validation=passed")
    print("comparison_rebuild_validation=passed")
    print("evidence_strength_validation=passed")
    print(f"manifest={manifest_path}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
