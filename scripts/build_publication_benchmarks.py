from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.metrics import compute_nav
from strategies.publication_benchmarks import (
    XJOA_ASSET_ID,
    XJOA_SYMBOL,
    build_external_benchmark_returns,
    build_point_in_time_equal_weight_benchmark,
)


DEFAULT_PUBLICATION_PANEL_PATH = REPO_ROOT / "data" / "licensed" / "asx200_point_in_time_panel.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
EXTERNAL_BENCHMARK_FILENAME = "publication_external_benchmark.csv"
EQUAL_WEIGHT_BENCHMARK_FILENAME = "publication_equal_weight_benchmark.csv"
BUILD_SUMMARY_FILENAME = "publication_benchmark_build_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build derived publication benchmark artifacts from a lawfully obtained Norgate "
            "XJOA source file and the licensed point-in-time ASX 200 publication panel."
        )
    )
    parser.add_argument(
        "--xjoa-source",
        type=Path,
        required=True,
        help="Path to the licensed Norgate XJOA total-return-index CSV.",
    )
    parser.add_argument(
        "--publication-panel",
        type=Path,
        default=DEFAULT_PUBLICATION_PANEL_PATH,
        help="Path to the licensed publication point-in-time parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated publication benchmark artifacts.",
    )
    return parser.parse_args()


def load_xjoa_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Norgate XJOA source file not found: {path}")
    return pd.read_csv(path)


def load_publication_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Publication panel not found: {path}")
    return pd.read_parquet(path)


def build_publication_benchmark_artifacts(
    *,
    xjoa_source: pd.DataFrame,
    publication_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    external = build_external_benchmark_returns(
        xjoa_source,
        expected_asset_id=XJOA_ASSET_ID,
        expected_symbol=XJOA_SYMBOL,
    )
    external["benchmark_nav"] = compute_nav(external["benchmark_return"])
    external = external[
        ["trade_date", "benchmark_level", "benchmark_return", "benchmark_nav"]
    ]

    equal_weight = build_point_in_time_equal_weight_benchmark(publication_panel)

    summary = {
        "external_benchmark": {
            "source_symbol": XJOA_SYMBOL,
            "source_asset_id": XJOA_ASSET_ID,
            "row_count": int(len(external)),
            "start_date": external["trade_date"].min().date().isoformat(),
            "end_date": external["trade_date"].max().date().isoformat(),
            "duplicate_trade_dates": int(external["trade_date"].duplicated().sum()),
            "null_return_count": int(external["benchmark_return"].isna().sum()),
        },
        "equal_weight_benchmark": {
            "row_count": int(len(equal_weight)),
            "start_date": equal_weight["trade_date"].min().date().isoformat(),
            "end_date": equal_weight["trade_date"].max().date().isoformat(),
            "duplicate_trade_dates": int(equal_weight["trade_date"].duplicated().sum()),
            "null_return_count": int(equal_weight["equal_weight_return"].isna().sum()),
            "member_count_min": int(equal_weight["member_count"].min()),
            "member_count_median": float(equal_weight["member_count"].median()),
            "member_count_max": int(equal_weight["member_count"].max()),
            "missing_member_return_count_max": int(
                equal_weight["missing_member_return_count"].max()
            ),
            "missing_member_return_fraction_mean": float(
                equal_weight["missing_member_return_fraction"].mean()
            ),
            "missing_member_return_fraction_max": float(
                equal_weight["missing_member_return_fraction"].max()
            ),
        },
        "membership_timing_convention": (
            "member_of_universe observed on session t determines equal-weight holdings "
            "earning returns on the next observed market session"
        ),
        "missing_return_convention": (
            "no zero imputation; equal weights are renormalized across observable returns "
            "for expected members"
        ),
    }
    return external, equal_weight, summary


def write_publication_benchmark_artifacts(
    *,
    external: pd.DataFrame,
    equal_weight: pd.DataFrame,
    summary: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    external_path = output_dir / EXTERNAL_BENCHMARK_FILENAME
    equal_weight_path = output_dir / EQUAL_WEIGHT_BENCHMARK_FILENAME
    summary_path = output_dir / BUILD_SUMMARY_FILENAME

    external.to_csv(external_path, index=False, date_format="%Y-%m-%d")
    equal_weight.to_csv(equal_weight_path, index=False, date_format="%Y-%m-%d")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return {
        "external_benchmark": external_path,
        "equal_weight_benchmark": equal_weight_path,
        "build_summary": summary_path,
    }


def main() -> int:
    args = parse_args()
    xjoa_source = load_xjoa_source(args.xjoa_source)
    publication_panel = load_publication_panel(args.publication_panel)
    external, equal_weight, summary = build_publication_benchmark_artifacts(
        xjoa_source=xjoa_source,
        publication_panel=publication_panel,
    )
    paths = write_publication_benchmark_artifacts(
        external=external,
        equal_weight=equal_weight,
        summary=summary,
        output_dir=args.output_dir,
    )

    print(paths["external_benchmark"])
    print(paths["equal_weight_benchmark"])
    print(paths["build_summary"])
    print(
        "external_rows="
        f"{summary['external_benchmark']['row_count']} "
        "equal_weight_rows="
        f"{summary['equal_weight_benchmark']['row_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
