from __future__ import annotations

import numpy as np
import pandas as pd

from .publication_benchmarks import build_external_benchmark_returns


XEW_ASSET_ID = 2312765
XEW_SYMBOL = "$XEW.au"
XJO_ASSET_ID = 14702
XJO_SYMBOL = "$XJO.au"


def build_reference_index_returns(
    frame: pd.DataFrame,
    *,
    expected_asset_id: int,
    expected_symbol: str,
    return_name: str,
) -> pd.DataFrame:
    """Convert a validated Norgate reference index level series to daily returns."""
    result = build_external_benchmark_returns(
        frame,
        expected_asset_id=expected_asset_id,
        expected_symbol=expected_symbol,
    ).rename(columns={"benchmark_level": "index_level", "benchmark_return": return_name})
    return result[["trade_date", "index_level", return_name]]


def compare_return_series(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_return_col: str,
    right_return_col: str,
) -> dict[str, object]:
    """Summarize overlap and differences between two observed daily return series."""
    merged = left[["trade_date", left_return_col]].merge(
        right[["trade_date", right_return_col]],
        on="trade_date",
        how="inner",
        validate="one_to_one",
    )
    usable = merged.dropna(subset=[left_return_col, right_return_col]).copy()
    if usable.empty:
        raise ValueError("Benchmark comparison has no overlapping non-null return observations.")

    difference = usable[left_return_col] - usable[right_return_col]
    correlation = usable[left_return_col].corr(usable[right_return_col])
    return {
        "overlap_start_date": usable["trade_date"].min().date().isoformat(),
        "overlap_end_date": usable["trade_date"].max().date().isoformat(),
        "overlap_return_count": int(len(usable)),
        "correlation": float(correlation) if pd.notna(correlation) else None,
        "mean_return_difference": float(difference.mean()),
        "mean_absolute_return_difference": float(difference.abs().mean()),
        "root_mean_squared_return_difference": float(np.sqrt(np.mean(np.square(difference)))),
        "max_absolute_return_difference": float(difference.abs().max()),
    }


def validate_publication_benchmarks(
    *,
    external_benchmark: pd.DataFrame,
    equal_weight_benchmark: pd.DataFrame,
    xew_source: pd.DataFrame,
    xjo_source: pd.DataFrame,
) -> dict[str, object]:
    """Build structural and reference-series diagnostics for publication benchmarks."""
    required_external = {"trade_date", "benchmark_level", "benchmark_return"}
    required_equal_weight = {
        "trade_date",
        "equal_weight_return",
        "member_count",
        "observable_member_return_count",
        "missing_member_return_count",
        "missing_member_return_fraction",
    }
    if missing := required_external.difference(external_benchmark.columns):
        raise ValueError(f"External benchmark artifact missing columns: {sorted(missing)}")
    if missing := required_equal_weight.difference(equal_weight_benchmark.columns):
        raise ValueError(f"Equal-weight benchmark artifact missing columns: {sorted(missing)}")

    external = external_benchmark.copy()
    equal_weight = equal_weight_benchmark.copy()
    for frame in (external, equal_weight):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        if frame["trade_date"].isna().any():
            raise ValueError("Derived benchmark artifact contains invalid trade dates.")
        if frame["trade_date"].duplicated().any():
            raise ValueError("Derived benchmark artifact contains duplicate trade dates.")
        frame.sort_values("trade_date", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    if len(external) != len(equal_weight):
        raise ValueError("External and equal-weight benchmark artifacts have different row counts.")
    if not external["trade_date"].equals(equal_weight["trade_date"]):
        raise ValueError("External and equal-weight benchmark artifacts do not share identical dates.")
    if external["benchmark_return"].isna().sum() != 1:
        raise ValueError("External benchmark must contain exactly one initial null return.")
    if equal_weight["equal_weight_return"].isna().sum() != 1:
        raise ValueError("Equal-weight benchmark must contain exactly one initial null return.")
    if (external["benchmark_level"] <= 0).any():
        raise ValueError("External benchmark contains non-positive index levels.")

    xew = build_reference_index_returns(
        xew_source,
        expected_asset_id=XEW_ASSET_ID,
        expected_symbol=XEW_SYMBOL,
        return_name="xew_return",
    )
    xjo = build_reference_index_returns(
        xjo_source,
        expected_asset_id=XJO_ASSET_ID,
        expected_symbol=XJO_SYMBOL,
        return_name="xjo_return",
    )

    xew_comparison = compare_return_series(
        equal_weight,
        xew,
        left_return_col="equal_weight_return",
        right_return_col="xew_return",
    )
    xjoa_xjo_comparison = compare_return_series(
        external,
        xjo,
        left_return_col="benchmark_return",
        right_return_col="xjo_return",
    )

    return {
        "structural_validation": {
            "status": "passed",
            "row_count": int(len(external)),
            "start_date": external["trade_date"].min().date().isoformat(),
            "end_date": external["trade_date"].max().date().isoformat(),
            "identical_external_and_equal_weight_dates": True,
            "external_null_return_count": int(external["benchmark_return"].isna().sum()),
            "equal_weight_null_return_count": int(equal_weight["equal_weight_return"].isna().sum()),
        },
        "equal_weight_missing_return_diagnostics": {
            "member_count_median": float(equal_weight["member_count"].median()),
            "member_count_min": int(equal_weight["member_count"].min()),
            "member_count_max": int(equal_weight["member_count"].max()),
            "missing_member_return_count_max": int(equal_weight["missing_member_return_count"].max()),
            "missing_member_return_fraction_mean": float(equal_weight["missing_member_return_fraction"].mean()),
            "missing_member_return_fraction_max": float(equal_weight["missing_member_return_fraction"].max()),
        },
        "xew_validation_reference": xew_comparison,
        "xjo_price_index_reference": xjoa_xjo_comparison,
        "interpretation_note": (
            "$XEW is validation-only and exact equality is not expected because official index "
            "rebalancing/maintenance can differ from the reconstructed point-in-time equal-weight "
            "series. $XJO is price-index reference only; $XJOA remains the canonical total-return benchmark."
        ),
    }
