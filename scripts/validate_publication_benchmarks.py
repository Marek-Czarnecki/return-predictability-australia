from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_benchmark_validation import validate_publication_benchmarks

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
DEFAULT_EXTERNAL_BENCHMARK = DEFAULT_OUTPUT_DIR / "publication_external_benchmark.csv"
DEFAULT_EQUAL_WEIGHT_BENCHMARK = DEFAULT_OUTPUT_DIR / "publication_equal_weight_benchmark.csv"
VALIDATION_FILENAME = "publication_benchmark_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate publication benchmark artifacts against licensed Norgate reference indices."
    )
    parser.add_argument("--xew-source", type=Path, required=True)
    parser.add_argument("--xjo-source", type=Path, required=True)
    parser.add_argument("--external-benchmark", type=Path, default=DEFAULT_EXTERNAL_BENCHMARK)
    parser.add_argument("--equal-weight-benchmark", type=Path, default=DEFAULT_EQUAL_WEIGHT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def run_validation(
    *,
    external_benchmark_path: Path,
    equal_weight_benchmark_path: Path,
    xew_path: Path,
    xjo_path: Path,
    output_path: Path,
) -> dict[str, object]:
    external = read_csv_required(external_benchmark_path, "External benchmark artifact")
    equal_weight = read_csv_required(equal_weight_benchmark_path, "Equal-weight benchmark artifact")
    xew = read_csv_required(xew_path, "Norgate XEW reference")
    xjo = read_csv_required(xjo_path, "Norgate XJO reference")

    payload = validate_publication_benchmarks(
        external_benchmark=external,
        equal_weight_benchmark=equal_weight,
        xew_source=xew,
        xjo_source=xjo,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    output_path = args.output_dir / VALIDATION_FILENAME
    payload = run_validation(
        external_benchmark_path=args.external_benchmark,
        equal_weight_benchmark_path=args.equal_weight_benchmark,
        xew_path=args.xew_source,
        xjo_path=args.xjo_source,
        output_path=output_path,
    )

    structural = payload["structural_validation"]
    xew = payload["xew_validation_reference"]
    xjo = payload["xjo_price_index_reference"]
    print(output_path)
    print(
        f"structural_status={structural['status']} rows={structural['row_count']} "
        f"date_range={structural['start_date']}..{structural['end_date']}"
    )
    print(
        "xew_overlap="
        f"{xew['overlap_return_count']} correlation={xew['correlation']:.6f} "
        f"mae={xew['mean_absolute_return_difference']:.8f}"
    )
    print(
        "xjo_overlap="
        f"{xjo['overlap_return_count']} correlation={xjo['correlation']:.6f} "
        f"mean_difference={xjo['mean_return_difference']:.8f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
