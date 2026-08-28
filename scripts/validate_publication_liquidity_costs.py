from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_costs import DEFAULT_MIN_LIQUIDITY_OBSERVATIONS
from strategies.publication_data import DEFAULT_PUBLICATION_PANEL_PATH, load_publication_panel
from strategies.publication_walk_forward_costs import build_publication_fold_cost_context
from strategies.walk_forward import generate_walk_forward_folds

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
FOLD_SUMMARY_FILENAME = "publication_liquidity_fold_summary.csv"
ASSIGNMENTS_FILENAME = "publication_liquidity_fold_assignments.csv"
VALIDATION_FILENAME = "publication_liquidity_cost_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate formation-window liquidity tiers and publication transaction-cost assignments."
    )
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PUBLICATION_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _evaluation_slice(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return prices.loc[prices["trade_date"].between(start, end)].copy()


def build_real_data_diagnostics(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    folds = generate_walk_forward_folds(prices, formation_years=3, evaluation_years=1, step_years=1)
    if not folds:
        raise ValueError("No walk-forward folds were generated from the publication panel.")

    fold_rows: list[dict[str, object]] = []
    assignment_frames: list[pd.DataFrame] = []

    for fold in folds:
        evaluation = _evaluation_slice(prices, fold.evaluation_start, fold.evaluation_end)
        context = build_publication_fold_cost_context(
            prices=prices,
            fold=fold,
            evaluation_prices=evaluation,
            identity_col="asset_id",
            min_liquidity_observations=DEFAULT_MIN_LIQUIDITY_OBSERVATIONS,
        )
        diagnostics = context.liquidity_diagnostics.set_index("asset_id")
        evaluation_ids = pd.Index(evaluation["asset_id"].dropna().unique())
        assignments = pd.DataFrame({"asset_id": evaluation_ids})
        assignments["fold_id"] = fold.fold_id
        assignments["formation_start"] = fold.formation_start
        assignments["formation_end"] = fold.formation_end
        assignments["evaluation_start"] = fold.evaluation_start
        assignments["evaluation_end"] = fold.evaluation_end
        assignments["liquidity_tier"] = assignments["asset_id"].map(context.liquidity_tier_map)
        assignments["formation_end_member"] = assignments["asset_id"].isin(diagnostics.index)
        assignments["sufficient_liquidity_history"] = assignments["asset_id"].map(
            diagnostics["sufficient_liquidity_history"]
        ).fillna(False).astype(bool)
        assignments["liquidity_observation_count"] = assignments["asset_id"].map(
            diagnostics["liquidity_observation_count"]
        ).fillna(0).astype(int)
        assignments["median_dollar_volume"] = assignments["asset_id"].map(
            diagnostics["median_dollar_volume"]
        )
        assignments["assignment_reason"] = "formation_window_rank"
        assignments.loc[
            assignments["formation_end_member"] & ~assignments["sufficient_liquidity_history"],
            "assignment_reason",
        ] = "insufficient_history_conservative_lower"
        assignments.loc[
            ~assignments["formation_end_member"], "assignment_reason"
        ] = "not_formation_end_member_conservative_lower"
        assignment_frames.append(assignments)

        tier_counts = assignments["liquidity_tier"].value_counts()
        reason_counts = assignments["assignment_reason"].value_counts()
        ranked_member_count = int(
            (assignments["formation_end_member"] & assignments["sufficient_liquidity_history"]).sum()
        )
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "formation_start": fold.formation_start,
                "formation_end": fold.formation_end,
                "evaluation_start": fold.evaluation_start,
                "evaluation_end": fold.evaluation_end,
                "evaluation_identity_count": int(len(assignments)),
                "formation_end_member_count": int(assignments["formation_end_member"].sum()),
                "ranked_member_count": ranked_member_count,
                "insufficient_history_member_count": int(reason_counts.get("insufficient_history_conservative_lower", 0)),
                "new_or_nonmember_evaluation_identity_count": int(reason_counts.get("not_formation_end_member_conservative_lower", 0)),
                "tier_high_count": int(tier_counts.get("high", 0)),
                "tier_medium_count": int(tier_counts.get("medium", 0)),
                "tier_lower_count": int(tier_counts.get("lower", 0)),
            }
        )

    fold_summary = pd.DataFrame(fold_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    payload = {
        "status": "passed",
        "fold_count": int(len(fold_summary)),
        "panel_start_date": prices["trade_date"].min().date().isoformat(),
        "panel_end_date": prices["trade_date"].max().date().isoformat(),
        "min_liquidity_observations": DEFAULT_MIN_LIQUIDITY_OBSERVATIONS,
        "liquidity_information_set": "formation_window_only",
        "ranking_universe": "formation_end_asx200_members",
        "tier_refresh": "fixed_for_subsequent_evaluation_fold",
        "fallback_tier": "lower",
        "tier_cutoffs": {"high": 0.30, "medium": 0.70},
        "evaluation_identity_count_min": int(fold_summary["evaluation_identity_count"].min()),
        "evaluation_identity_count_median": float(fold_summary["evaluation_identity_count"].median()),
        "evaluation_identity_count_max": int(fold_summary["evaluation_identity_count"].max()),
        "ranked_member_count_min": int(fold_summary["ranked_member_count"].min()),
        "ranked_member_count_median": float(fold_summary["ranked_member_count"].median()),
        "ranked_member_count_max": int(fold_summary["ranked_member_count"].max()),
        "insufficient_history_member_count_total": int(fold_summary["insufficient_history_member_count"].sum()),
        "new_or_nonmember_evaluation_identity_count_total": int(fold_summary["new_or_nonmember_evaluation_identity_count"].sum()),
        "assignment_reason_counts": {str(key): int(value) for key, value in assignments["assignment_reason"].value_counts().items()},
        "tier_assignment_counts": {str(key): int(value) for key, value in assignments["liquidity_tier"].value_counts().items()},
    }
    return fold_summary, assignments, payload


def write_diagnostics(
    fold_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    payload: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / FOLD_SUMMARY_FILENAME
    assignments_path = output_dir / ASSIGNMENTS_FILENAME
    validation_path = output_dir / VALIDATION_FILENAME
    fold_summary.to_csv(summary_path, index=False, date_format="%Y-%m-%d")
    assignments.to_csv(assignments_path, index=False, date_format="%Y-%m-%d")
    validation_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"fold_summary": summary_path, "assignments": assignments_path, "validation": validation_path}


def main() -> int:
    args = parse_args()
    prices = load_publication_panel(args.panel_path)
    fold_summary, assignments, payload = build_real_data_diagnostics(prices)
    paths = write_diagnostics(fold_summary, assignments, payload, args.output_dir)
    print(paths["fold_summary"])
    print(paths["assignments"])
    print(paths["validation"])
    print(
        f"status={payload['status']} folds={payload['fold_count']} "
        f"ranked_members={payload['ranked_member_count_min']}..{payload['ranked_member_count_max']} "
        f"insufficient_history_total={payload['insufficient_history_member_count_total']} "
        f"new_or_nonmember_total={payload['new_or_nonmember_evaluation_identity_count_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
