from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_risk_free import (
    RBA_CASH_RATE_URL,
    build_publication_risk_free,
    load_rba_cash_rate_schedule,
    validate_overlap_with_existing_tri,
)


DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
DEFAULT_BENCHMARK_PATH = DEFAULT_RESULTS_DIR / "publication_external_benchmark.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "publication_risk_free.csv"
DEFAULT_VALIDATION_PATH = DEFAULT_RESULTS_DIR / "publication_risk_free_overlap_validation.csv"
DEFAULT_METADATA_PATH = DEFAULT_RESULTS_DIR / "publication_risk_free_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the publication risk-free return series from official RBA cash-rate target history."
    )
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument(
        "--existing-risk-free-path",
        type=Path,
        default=None,
        help="Optional prior risk-free TRI CSV used only for overlap validation.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validation-path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = pd.read_csv(args.benchmark_path, parse_dates=["trade_date"])
    if benchmark["trade_date"].duplicated().any():
        raise ValueError("Publication benchmark contains duplicate trade dates.")

    schedule = load_rba_cash_rate_schedule()
    result = build_publication_risk_free(benchmark["trade_date"], schedule)
    overlap = validate_overlap_with_existing_tri(result, args.existing_risk_free_path)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.validation_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_path, index=False)
    overlap.to_csv(args.validation_path, index=False)

    metadata = {
        "source": "Reserve Bank of Australia cash rate target history",
        "source_url": RBA_CASH_RATE_URL,
        "calendar_start": result["trade_date"].min().isoformat() if not result.empty else None,
        "calendar_end": result["trade_date"].max().isoformat() if not result.empty else None,
        "row_count": int(len(result)),
        "construction": "target_rate_percent / 365 compounded over actual calendar-day gaps",
        "first_return_policy": "zero_only_on_first_output_observation",
        "parameter_selection_affected": False,
        "strategy_returns_affected": False,
        "use": "Sharpe and risk-adjusted publication metrics only",
        "existing_tri_overlap": overlap.iloc[0].to_dict(),
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"rows={len(result)}")
    print(f"date_range={result['trade_date'].min().date()}..{result['trade_date'].max().date()}")
    print(f"overlap_count={int(overlap.iloc[0]['overlap_count'])}")
    print(f"overlap_max_abs_difference={overlap.iloc[0]['max_abs_return_difference']}")
    print(f"output={args.output_path}")
    print(f"validation={args.validation_path}")
    print(f"metadata={args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
