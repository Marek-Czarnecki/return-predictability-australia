from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


NEXT_SESSION_AFTER_CLOSE = "next_session_after_close"
SAME_SESSION = "same_session"
SUPPORTED_TIMING_CONVENTIONS = {
    NEXT_SESSION_AFTER_CLOSE: (
        "Eligibility is evaluated on the signal date and is intended to govern "
        "next-session execution, matching backtests that apply positions with a one-day lag."
    ),
    SAME_SESSION: (
        "Eligibility is evaluated on the signal date and is intended to govern "
        "same-session execution when the caller can justify same-day information availability."
    ),
}
DEFAULT_PUBLICATION_VALIDATION_PATH = Path(
    "data/evidence/publication_eligibility_validation.json"
)


def annotate_publication_eligibility(
    prices: pd.DataFrame,
    min_history: int,
    identity_col: str = "asset_id",
    membership_col: str = "member_of_universe",
    date_col: str = "trade_date",
    timing_convention: str = NEXT_SESSION_AFTER_CLOSE,
    filter_eligible: bool = False,
) -> pd.DataFrame:
    if min_history <= 0:
        raise ValueError("min_history must be positive.")
    if timing_convention not in SUPPORTED_TIMING_CONVENTIONS:
        raise ValueError(
            "Unsupported timing_convention: "
            f"{timing_convention}. Expected one of {sorted(SUPPORTED_TIMING_CONVENTIONS)}."
        )

    required_columns = {identity_col, membership_col, date_col}
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Price panel is missing required columns: {missing_list}")

    annotated = prices.copy()
    annotated[date_col] = pd.to_datetime(annotated[date_col])
    annotated = annotated.sort_values([identity_col, date_col]).reset_index(drop=True)
    annotated["history_observation_count"] = (
        annotated.groupby(identity_col, observed=True).cumcount() + 1
    )
    annotated["has_min_history"] = annotated["history_observation_count"] >= int(
        min_history
    )
    annotated["is_index_member"] = (
        pd.to_numeric(annotated[membership_col], errors="raise").astype("int8") == 1
    )
    annotated["eligible_to_trade"] = (
        annotated["has_min_history"] & annotated["is_index_member"]
    )
    annotated.attrs["timing_convention"] = timing_convention
    annotated.attrs["timing_note"] = SUPPORTED_TIMING_CONVENTIONS[timing_convention]
    annotated.attrs["identity_col"] = identity_col
    annotated.attrs["membership_col"] = membership_col
    annotated.attrs["min_history"] = int(min_history)
    if filter_eligible:
        return annotated.loc[annotated["eligible_to_trade"]].reset_index(drop=True)
    return annotated


def summarize_publication_eligibility(
    annotated: pd.DataFrame,
    identity_col: str = "asset_id",
    membership_col: str = "member_of_universe",
    date_col: str = "trade_date",
) -> dict[str, object]:
    required_columns = {
        identity_col,
        membership_col,
        date_col,
        "history_observation_count",
        "has_min_history",
        "is_index_member",
        "eligible_to_trade",
    }
    missing_columns = required_columns.difference(annotated.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Annotated publication panel is missing required columns: " f"{missing_list}"
        )

    working = annotated.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    working = working.sort_values([identity_col, date_col]).reset_index(drop=True)

    prior_membership = working.groupby(identity_col, observed=True)[
        "is_index_member"
    ].shift(1, fill_value=False)
    membership_entry = working["is_index_member"] & ~prior_membership
    membership_exit = ~working["is_index_member"] & prior_membership

    first_membership = (
        working.loc[membership_entry, [identity_col, date_col, "eligible_to_trade"]]
        .groupby(identity_col, observed=True)
        .first()
    )
    first_eligible = (
        working.loc[working["eligible_to_trade"], [identity_col, date_col]]
        .groupby(identity_col, observed=True)[date_col]
        .min()
    )
    eligible_assets = first_eligible.index
    first_eligible_summary = _summarize_first_eligible_dates(first_eligible)
    reentry_counts = (
        working.loc[membership_entry]
        .groupby(identity_col, observed=True)
        .size()
        .sub(1)
        .clip(lower=0)
    )

    return {
        "timing_convention": annotated.attrs.get(
            "timing_convention", NEXT_SESSION_AFTER_CLOSE
        ),
        "timing_note": annotated.attrs.get("timing_note", ""),
        "identity_convention": annotated.attrs.get("identity_col", identity_col),
        "membership_convention": annotated.attrs.get(
            "membership_col", membership_col
        ),
        "min_history": int(annotated.attrs.get("min_history", 0)),
        "row_count": int(len(working)),
        "asset_count": int(working[identity_col].nunique()),
        "eligible_asset_date_rows": int(working["eligible_to_trade"].sum()),
        "member_insufficient_history_rows": int(
            (working["is_index_member"] & ~working["has_min_history"]).sum()
        ),
        "eligible_asset_count": int(len(eligible_assets)),
        "first_eligible_date_summary": first_eligible_summary,
        "assets_eligible_immediately_on_first_membership": int(
            first_membership["eligible_to_trade"].sum()
        )
        if not first_membership.empty
        else 0,
        "assets_requiring_additional_history_after_first_membership": int(
            (~first_membership["eligible_to_trade"]).sum()
        )
        if not first_membership.empty
        else 0,
        "membership_entry_transitions": int(membership_entry.sum()),
        "membership_exit_transitions": int(membership_exit.sum()),
        "assets_with_reentry": int((reentry_counts > 0).sum()) if not reentry_counts.empty else 0,
        "total_reentries": int(reentry_counts.sum()) if not reentry_counts.empty else 0,
        "max_reentries_per_asset": int(reentry_counts.max()) if not reentry_counts.empty else 0,
    }


def build_publication_eligibility_validation(
    prices: pd.DataFrame,
    min_history_values: list[int] | tuple[int, ...],
    identity_col: str = "asset_id",
    membership_col: str = "member_of_universe",
    date_col: str = "trade_date",
    timing_convention: str = NEXT_SESSION_AFTER_CLOSE,
) -> dict[str, object]:
    validation: dict[str, object] = {
        "source_panel_path": "data/licensed/asx200_point_in_time_panel.parquet",
        "timing_convention": timing_convention,
        "timing_note": SUPPORTED_TIMING_CONVENTIONS[timing_convention],
        "identity_convention": identity_col,
        "membership_convention": membership_col,
    }
    for min_history in min_history_values:
        annotated = annotate_publication_eligibility(
            prices,
            min_history=min_history,
            identity_col=identity_col,
            membership_col=membership_col,
            date_col=date_col,
            timing_convention=timing_convention,
        )
        validation[f"min_history_{int(min_history)}"] = summarize_publication_eligibility(
            annotated,
            identity_col=identity_col,
            membership_col=membership_col,
            date_col=date_col,
        )
    return validation


def write_publication_eligibility_validation(
    output_path: Path, payload: dict[str, object]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _summarize_first_eligible_dates(first_eligible: pd.Series) -> dict[str, object]:
    if first_eligible.empty:
        return {
            "asset_count": 0,
            "earliest": None,
            "median": None,
            "latest": None,
        }

    ordered_dates = pd.Series(pd.to_datetime(first_eligible).sort_values().reset_index(drop=True))
    return {
        "asset_count": int(len(ordered_dates)),
        "earliest": ordered_dates.iloc[0].date().isoformat(),
        "median": ordered_dates.iloc[len(ordered_dates) // 2].date().isoformat(),
        "latest": ordered_dates.iloc[-1].date().isoformat(),
    }
