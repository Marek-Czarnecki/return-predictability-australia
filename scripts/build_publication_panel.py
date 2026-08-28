from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "licensed" / "asx200_point_in_time_panel.parquet"
DEFAULT_VALIDATION_OUTPUT_PATH = (
    REPO_ROOT / "data" / "generated" / "publication_results" / "publication_panel_validation.json"
)
REQUIRED_COLUMNS = [
    "asset_id",
    "ticker_code",
    "vendor_symbol",
    "security_name",
    "source_database",
    "delisted_flag",
    "trade_date",
    "member_of_universe",
    "open",
    "high",
    "low",
    "close",
    "unadjusted_close",
    "dividend",
    "volume",
    "turnover",
]
BASELINE_START_DATE = pd.Timestamp("2000-03-31")
BASELINE_END_DATE = pd.Timestamp("2026-08-27")
BASELINE_COUNTS = {
    "row_count": 2_462_415,
    "asset_count": 718,
    "member_rows": 1_332_282,
    "non_member_rows": 1_130_133,
    "active_assets": 315,
    "delisted_assets": 403,
}
PRICE_ADJUSTMENT_CONVENTION = "Norgate TotalReturn default; adj_close = close"
LIQUIDITY_CONVENTION = "dollar_volume = turnover"
IDENTITY_CONVENTION = "asset_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the publication-ready ASX point-in-time parquet from a lawfully obtained "
            "Norgate CSV. Licensed inputs and the derived licensed panel remain outside version control."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the licensed Norgate publication raw CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path for the licensed publication panel.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT_PATH,
        help="Output path for the non-row-level panel validation JSON.",
    )
    return parser.parse_args()


def read_raw_panel(input_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        input_path,
        parse_dates=["trade_date"],
        dtype={
            "asset_id": "int64",
            "ticker_code": "string",
            "vendor_symbol": "string",
            "security_name": "string",
            "source_database": "string",
            "member_of_universe": "int8",
        },
    )
    raw["delisted_flag"] = raw["delisted_flag"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    return raw


def compute_summary(frame: pd.DataFrame) -> dict[str, object]:
    member_mask = frame["member_of_universe"] == 1
    delisted_mask = frame["delisted_flag"].astype(bool)
    return {
        "row_count": int(len(frame)),
        "asset_count": int(frame["asset_id"].nunique()),
        "min_date": pd.Timestamp(frame["trade_date"].min()),
        "max_date": pd.Timestamp(frame["trade_date"].max()),
        "duplicate_keys": int(frame.duplicated(["asset_id", "trade_date"]).sum()),
        "membership_values": set(frame["member_of_universe"].dropna().unique().tolist()),
        "member_rows": int(member_mask.sum()),
        "non_member_rows": int((~member_mask).sum()),
        "active_assets": int(frame.loc[~delisted_mask, "asset_id"].nunique()),
        "delisted_assets": int(frame.loc[delisted_mask, "asset_id"].nunique()),
        "null_cells": int(frame[REQUIRED_COLUMNS].isna().sum().sum()),
    }


def validate_columns(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Raw CSV is missing required columns: {missing_columns}")


def validate_raw_panel(frame: pd.DataFrame) -> dict[str, object]:
    validate_columns(frame)
    summary = compute_summary(frame)
    if summary["min_date"] != BASELINE_START_DATE:
        raise ValueError(
            "Raw CSV start date changed unexpectedly: "
            f"{summary['min_date'].date()} != {BASELINE_START_DATE.date()}"
        )
    if summary["max_date"] < BASELINE_END_DATE:
        raise ValueError(
            "Raw CSV end date predates the validated baseline: "
            f"{summary['max_date'].date()} < {BASELINE_END_DATE.date()}"
        )
    if summary["duplicate_keys"] != 0:
        raise ValueError("Raw CSV contains duplicate asset_id + trade_date rows.")
    if summary["membership_values"] != {0, 1}:
        raise ValueError(
            "Raw CSV member_of_universe values must be exactly {0, 1}: "
            f"{summary['membership_values']}"
        )
    if summary["null_cells"] != 0:
        raise ValueError("Raw CSV contains nulls in required columns.")
    if summary["asset_count"] != BASELINE_COUNTS["asset_count"]:
        raise ValueError(
            "Raw CSV asset count changed unexpectedly: "
            f"{summary['asset_count']} != {BASELINE_COUNTS['asset_count']}"
        )

    exact_snapshot = summary["max_date"] == BASELINE_END_DATE
    for key, expected in BASELINE_COUNTS.items():
        observed = summary[key]
        if exact_snapshot and observed != expected:
            raise ValueError(f"Raw CSV {key} mismatch: {observed} != {expected}")
        if not exact_snapshot and observed < expected:
            raise ValueError(
                f"Raw CSV {key} regressed below validated baseline: {observed} < {expected}"
            )
    return summary


def transform_publication_panel(frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.loc[:, REQUIRED_COLUMNS].copy()
    panel["delisted_flag"] = panel["delisted_flag"].astype(bool)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["asset_id"] = pd.to_numeric(panel["asset_id"], errors="raise").astype("int64")
    panel["member_of_universe"] = pd.to_numeric(
        panel["member_of_universe"], errors="raise"
    ).astype("int8")
    for column in [
        "open",
        "high",
        "low",
        "close",
        "unadjusted_close",
        "dividend",
        "volume",
        "turnover",
    ]:
        panel[column] = pd.to_numeric(panel[column], errors="raise")

    panel = panel.sort_values(["asset_id", "trade_date"]).reset_index(drop=True)
    panel["adj_close"] = panel["close"]
    panel["daily_return"] = panel.groupby("asset_id", observed=True)["adj_close"].pct_change()
    panel["dollar_volume"] = panel["turnover"]
    return panel


def validate_transformed_panel(
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    source_summary: dict[str, object],
) -> dict[str, object]:
    summary = compute_summary(transformed)
    checks = {
        "row count": summary["row_count"] == source_summary["row_count"],
        "asset count": summary["asset_count"] == source_summary["asset_count"],
        "start date": summary["min_date"] == source_summary["min_date"],
        "end date": summary["max_date"] == source_summary["max_date"],
        "member rows": summary["member_rows"] == source_summary["member_rows"],
        "non-member rows": summary["non_member_rows"] == source_summary["non_member_rows"],
        "active assets": summary["active_assets"] == source_summary["active_assets"],
        "delisted assets": summary["delisted_assets"] == source_summary["delisted_assets"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Transformed panel validation failed: " + ", ".join(failed))
    if summary["duplicate_keys"] != 0:
        raise ValueError("Transformed panel contains duplicate asset_id + trade_date rows.")
    if summary["membership_values"] != {0, 1}:
        raise ValueError("Transformed panel member_of_universe values changed.")
    if not transformed["adj_close"].equals(transformed["close"]):
        raise ValueError("adj_close must match close for every row.")
    if not transformed["dollar_volume"].equals(transformed["turnover"]):
        raise ValueError("dollar_volume must match turnover for every row.")

    first_rows = transformed.groupby("asset_id", observed=True).cumcount() == 0
    expected_nulls = int(first_rows.sum())
    actual_nulls = int(transformed["daily_return"].isna().sum())
    if actual_nulls != expected_nulls or transformed.loc[~first_rows, "daily_return"].isna().any():
        raise ValueError(
            "daily_return nulls must only occur on the first observation of each asset_id."
        )
    if transformed["asset_id"].is_monotonic_increasing is False:
        raise ValueError("Transformed panel is not sorted by asset_id.")
    if not transformed.groupby("asset_id", observed=True)["trade_date"].is_monotonic_increasing.all():
        raise ValueError("Transformed panel is not sorted by trade_date within asset_id.")
    if len(transformed) != len(source):
        raise ValueError("Transformed panel lost rows during conversion.")
    return summary


def build_validation_payload(
    input_path: Path,
    output_path: Path,
    transformed: pd.DataFrame,
    output_summary: dict[str, object],
) -> dict[str, object]:
    first_rows = transformed.groupby("asset_id", observed=True).cumcount() == 0
    try:
        safe_output_path = str(output_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        safe_output_path = str(Path(output_path).name)
    return {
        "input_filename": input_path.name,
        "row_count": output_summary["row_count"],
        "asset_count": output_summary["asset_count"],
        "vendor_symbol_count": int(transformed["vendor_symbol"].nunique()),
        "start_date": output_summary["min_date"].date().isoformat(),
        "end_date": output_summary["max_date"].date().isoformat(),
        "member_row_count": output_summary["member_rows"],
        "nonmember_row_count": output_summary["non_member_rows"],
        "membership_values": sorted(output_summary["membership_values"]),
        "duplicate_asset_date_rows": output_summary["duplicate_keys"],
        "active_asset_count": output_summary["active_assets"],
        "delisted_asset_count": output_summary["delisted_assets"],
        "daily_return_null_count": int(transformed["daily_return"].isna().sum()),
        "nonfirst_daily_return_null_count": int(
            transformed.loc[~first_rows, "daily_return"].isna().sum()
        ),
        "adj_close_close_mismatch_count": int(
            (transformed["adj_close"] != transformed["close"]).sum()
        ),
        "dollar_volume_turnover_mismatch_count": int(
            (transformed["dollar_volume"] != transformed["turnover"]).sum()
        ),
        "output_parquet_path": safe_output_path,
        "price_adjustment_convention": PRICE_ADJUSTMENT_CONVENTION,
        "liquidity_convention": LIQUIDITY_CONVENTION,
        "identity_convention": IDENTITY_CONVENTION,
    }


def build_publication_panel(
    input_path: Path,
    output_path: Path,
    validation_output_path: Path = DEFAULT_VALIDATION_OUTPUT_PATH,
) -> dict[str, object]:
    raw = read_raw_panel(input_path)
    raw_summary = validate_raw_panel(raw)
    transformed = transform_publication_panel(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_parquet(output_path, index=False)

    reread = pd.read_parquet(output_path)
    output_summary = validate_transformed_panel(raw, reread, raw_summary)
    validation_payload = build_validation_payload(input_path, output_path, reread, output_summary)
    validation_output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_output_path.write_text(json.dumps(validation_payload, indent=2) + "\n")
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_count": output_summary["row_count"],
        "asset_count": output_summary["asset_count"],
        "start_date": output_summary["min_date"].date().isoformat(),
        "end_date": output_summary["max_date"].date().isoformat(),
        "validation_path": str(validation_output_path),
    }


def main() -> int:
    args = parse_args()
    summary = build_publication_panel(args.input, args.output, args.validation_output)
    print(summary["output_path"])
    print(
        f"rows={summary['row_count']} assets={summary['asset_count']} "
        f"date_range={summary['start_date']}..{summary['end_date']}"
    )
    print(f"validation={summary['validation_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
