from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .publication_step11_trend_ablation import (
    FROZEN_CAPSTONE_COMMIT,
    frozen_capstone_trend_folds,
    run_publication_walk_forward_on_explicit_folds,
)


REFERENCE_DATE = pd.Timestamp("2026-07-20")
STEP11_UNIVERSE_ABLATION_DIMENSION = "point_in_time_vs_norgate_reference_date_retrospective_universe"


@dataclass(frozen=True)
class ReferenceUniverseSelection:
    reference_date: pd.Timestamp
    selected_asset_ids: tuple[object, ...]
    reference_table: pd.DataFrame


def select_norgate_reference_date_universe(
    prices: pd.DataFrame,
    *,
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> ReferenceUniverseSelection:
    required = {"asset_id", "trade_date", "member_of_universe"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(
            "Publication panel is missing columns required for reference-universe selection: "
            + ", ".join(sorted(missing))
        )

    working = prices.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"])
    reference_date = pd.Timestamp(reference_date)
    reference_rows = working.loc[working["trade_date"] == reference_date].copy()
    if reference_rows.empty:
        raise ValueError(
            f"Publication panel contains no rows on reference date {reference_date.date()}."
        )

    membership = pd.to_numeric(reference_rows["member_of_universe"], errors="raise").astype(int)
    members = reference_rows.loc[membership == 1].copy()
    if members.empty:
        raise ValueError(
            f"Publication panel contains no ASX 200 members on {reference_date.date()}."
        )
    if members["asset_id"].duplicated().any():
        raise ValueError("Reference-date universe contains duplicate asset_id rows.")

    optional = [
        column
        for column in ("ticker_code", "vendor_symbol", "security_name", "delisted_flag")
        if column in members.columns
    ]
    reference_table = members.loc[:, ["asset_id", "trade_date", *optional]].copy()
    reference_table = reference_table.sort_values("asset_id").reset_index(drop=True)
    selected_asset_ids = tuple(reference_table["asset_id"].tolist())
    return ReferenceUniverseSelection(reference_date, selected_asset_ids, reference_table)


def build_retrospective_reference_universe_panel(
    prices: pd.DataFrame,
    selected_asset_ids: Iterable[object],
) -> pd.DataFrame:
    selected = set(selected_asset_ids)
    if not selected:
        raise ValueError("At least one selected asset_id is required.")
    required = {"asset_id", "member_of_universe"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(
            "Publication panel is missing columns required for retrospective membership: "
            + ", ".join(sorted(missing))
        )
    panel = prices.copy()
    panel["member_of_universe"] = panel["asset_id"].isin(selected)
    return panel


def build_fold_comparison(pit_result, retrospective_result) -> pd.DataFrame:
    pit = pit_result.fold_summary.loc[:, ["fold_id", "net_excess_nav_difference", "total_net_excess_return", "chosen_parameters"]].copy()
    pit = pit.rename(columns={"net_excess_nav_difference": "pit_net_excess_nav_difference", "total_net_excess_return": "pit_legacy_excess", "chosen_parameters": "pit_chosen_parameters"})
    retro = retrospective_result.fold_summary.loc[:, ["fold_id", "net_excess_nav_difference", "total_net_excess_return", "chosen_parameters"]].copy()
    retro = retro.rename(columns={"net_excess_nav_difference": "retrospective_net_excess_nav_difference", "total_net_excess_return": "retrospective_legacy_excess", "chosen_parameters": "retrospective_chosen_parameters"})
    comparison = pit.merge(retro, on="fold_id", how="inner", validate="one_to_one")
    comparison["universe_effect_nav_difference"] = comparison["retrospective_net_excess_nav_difference"] - comparison["pit_net_excess_nav_difference"]
    comparison["parameter_selection_changed"] = comparison["pit_chosen_parameters"] != comparison["retrospective_chosen_parameters"]
    return comparison


def export_step11_universe_ablation(
    pit_result,
    retrospective_result,
    selection: ReferenceUniverseSelection,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "publication_step11_trend_universe_ablation"
    paths = {
        "comparison": output_dir / f"{prefix}_comparison.csv",
        "reference_universe": output_dir / f"{prefix}_norgate_reference_universe.csv",
        "pit_summary": output_dir / f"{prefix}_pit_summary.csv",
        "retrospective_summary": output_dir / f"{prefix}_retrospective_summary.csv",
        "metadata": output_dir / f"{prefix}_metadata.json",
    }
    comparison = build_fold_comparison(pit_result, retrospective_result)
    comparison.to_csv(paths["comparison"], index=False)
    selection.reference_table.to_csv(paths["reference_universe"], index=False)
    pit_result.fold_summary.to_csv(paths["pit_summary"], index=False)
    retrospective_result.fold_summary.to_csv(paths["retrospective_summary"], index=False)

    pit_nav = pd.to_numeric(comparison["pit_net_excess_nav_difference"], errors="coerce")
    retro_nav = pd.to_numeric(comparison["retrospective_net_excess_nav_difference"], errors="coerce")
    effect = pd.to_numeric(comparison["universe_effect_nav_difference"], errors="coerce")
    metadata = {
        "step": "11.1.2",
        "analysis_role": "diagnostic_ablation",
        "confirmatory": False,
        "ablation_dimension": STEP11_UNIVERSE_ABLATION_DIMENSION,
        "causal_scope": "membership_timing_within_common_norgate_security_coverage",
        "frozen_capstone_commit": FROZEN_CAPSTONE_COMMIT,
        "reference_date": selection.reference_date.date().isoformat(),
        "reference_universe_definition": "Norgate asset_ids with member_of_universe=1 on reference_date",
        "reference_universe_asset_count": int(len(selection.selected_asset_ids)),
        "retrospective_arm_definition": "Reference-date Norgate ASX 200 asset_ids treated as members throughout each asset's available Norgate history; all other Norgate assets treated as non-members.",
        "frozen_yahoo_ticker_reconciliation_used": False,
        "frozen_yahoo_vs_norgate_security_coverage": "unresolved_contributor_not_part_of_this_ablation",
        "fold_schedule_source": "frozen_capstone",
        "fold_count": int(len(comparison)),
        "strategy_rule_changed": False,
        "parameter_grid_changed": False,
        "benchmark_changed": False,
        "cost_framework_changed": False,
        "price_source_changed_between_arms": False,
        "identity_convention": "asset_id",
        "not_part_of_primary_holm_family": True,
        "pit_mean_net_excess_nav_difference": float(pit_nav.mean()),
        "retrospective_mean_net_excess_nav_difference": float(retro_nav.mean()),
        "mean_universe_effect_nav_difference": float(effect.mean()),
        "median_universe_effect_nav_difference": float(effect.median()),
        "pit_positive_fold_count": int((pit_nav > 0).sum()),
        "retrospective_positive_fold_count": int((retro_nav > 0).sum()),
        "parameter_selection_changed_fold_count": int(comparison["parameter_selection_changed"].sum()),
        "interpretation_rule": "Attribute only the controlled difference between genuine point-in-time membership and retrospective application of the Norgate reference-date member set. This isolates the survivorship-membership mechanism within common Norgate security coverage and does not reproduce or explain Yahoo-versus-Norgate security-universe differences.",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def run_step11_universe_ablation(
    prices: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    risk_free_returns: pd.DataFrame,
    *,
    reference_date: pd.Timestamp = REFERENCE_DATE,
):
    selection = select_norgate_reference_date_universe(prices, reference_date=reference_date)
    folds = frozen_capstone_trend_folds()
    pit_result = run_publication_walk_forward_on_explicit_folds(
        strategy_name="trend_following", prices=prices, benchmark_returns=benchmark_returns, risk_free_returns=risk_free_returns, folds=folds
    )
    retrospective_prices = build_retrospective_reference_universe_panel(prices, selection.selected_asset_ids)
    retrospective_result = run_publication_walk_forward_on_explicit_folds(
        strategy_name="trend_following", prices=retrospective_prices, benchmark_returns=benchmark_returns, risk_free_returns=risk_free_returns, folds=folds
    )
    return pit_result, retrospective_result, selection
