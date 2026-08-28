from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_data import DEFAULT_PUBLICATION_PANEL_PATH
from strategies.publication_eligibility import (
    NEXT_SESSION_AFTER_CLOSE,
    build_publication_eligibility_validation,
    write_publication_eligibility_validation,
)

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "data"
    / "generated"
    / "publication_results"
    / "publication_eligibility_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate point-in-time membership, identity, minimum-history and execution eligibility."
    )
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PUBLICATION_PANEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prices = pd.read_parquet(args.panel_path)
    payload = build_publication_eligibility_validation(
        prices,
        min_history_values=[220, 60],
        timing_convention=NEXT_SESSION_AFTER_CLOSE,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_publication_eligibility_validation(args.output_path, payload)
    print(args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
