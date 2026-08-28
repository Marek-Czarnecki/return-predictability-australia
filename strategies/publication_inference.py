from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.inference import (
    CONFIRMATORY_FAMILY_ID,
    CONFIRMATORY_GOVERNANCE,
    HOLM_ADJUSTMENT_METHOD,
    apply_confirmatory_holm_adjustment,
    bootstrap_mean_confidence_interval,
    sign_flip_mean_p_value,
)


WALK_FORWARD_STRATEGIES = ("trend_following", "mean_reversion", "pairs_trading")
PRIMARY_WALK_FORWARD_METRIC = "net_excess_nav_difference"
PRIMARY_TAX_LOSS_METRIC = "abnormal_net_return_difference"
PRIMARY_ALTERNATIVE = "greater"


@dataclass(frozen=True)
class PublicationInferenceResult:
    primary_inference: pd.DataFrame
    tax_loss_year_effects: pd.DataFrame


def build_publication_primary_inference(results_root: Path) -> PublicationInferenceResult:
    results_root = Path(results_root)
    rows: list[dict[str, object]] = []

    for strategy_name in WALK_FORWARD_STRATEGIES:
        path = results_root / f"publication_{strategy_name}_walk_forward_summary.csv"
        summary = pd.read_csv(path)
        if PRIMARY_WALK_FORWARD_METRIC not in summary.columns:
            raise ValueError(
                f"{path.name} is missing primary metric {PRIMARY_WALK_FORWARD_METRIC}."
            )
        if "window_label" in summary.columns:
            summary = summary.loc[
                summary["window_label"].astype("string").eq("evaluation")
            ].copy()
        values = pd.to_numeric(summary[PRIMARY_WALK_FORWARD_METRIC], errors="coerce").dropna()
        rows.append(
            _primary_row(
                strategy_name=strategy_name,
                values=values,
                effect_unit="evaluation_fold_nav_difference",
                inference_method="walk_forward_fold_sign_flip",
                source_artifact=path.name,
                primary_metric=PRIMARY_WALK_FORWARD_METRIC,
                sample_unit="evaluation_folds",
                limitation_note=(
                    "Fold-level confirmatory inference on out-of-sample strategy total return minus "
                    "benchmark total return. The test addresses positive average benchmark-relative "
                    "performance across evaluation folds; it is not a causal claim."
                ),
            )
        )

    event_path = results_root / "publication_tax_loss_selling_event_study.csv"
    events = pd.read_csv(event_path)
    required = {"year", PRIMARY_TAX_LOSS_METRIC}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(
            f"{event_path.name} is missing required columns: {', '.join(sorted(missing))}"
        )
    complete = events.loc[
        pd.to_numeric(events[PRIMARY_TAX_LOSS_METRIC], errors="coerce").notna()
    ].copy()
    complete[PRIMARY_TAX_LOSS_METRIC] = pd.to_numeric(
        complete[PRIMARY_TAX_LOSS_METRIC], errors="coerce"
    )
    year_effects = (
        complete.groupby("year", observed=True)[PRIMARY_TAX_LOSS_METRIC]
        .agg(
            [
                ("mean_abnormal_net_return_difference", "mean"),
                ("matched_observation_count", "size"),
            ]
        )
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )
    tax_values = year_effects["mean_abnormal_net_return_difference"]
    rows.append(
        _primary_row(
            strategy_name="tax_loss_selling",
            values=tax_values,
            effect_unit="annual_mean_abnormal_event_minus_control_return",
            inference_method="year_clustered_sign_flip",
            source_artifact=event_path.name,
            primary_metric=PRIMARY_TAX_LOSS_METRIC,
            sample_unit="calendar_years",
            limitation_note=(
                "Primary tax-loss inference uses equal-weight calendar-year means of benchmark-adjusted "
                "net event-minus-control returns to respect within-year clustering. Raw event-minus-control "
                "returns remain secondary descriptive evidence."
            ),
        )
    )

    raw_primary = pd.DataFrame(rows)
    metadata = raw_primary.loc[:, ["analysis_key", "primary_metric", "sample_unit"]].copy()
    primary = apply_confirmatory_holm_adjustment(raw_primary)
    primary = primary.merge(metadata, on="analysis_key", how="left", validate="one_to_one")

    return PublicationInferenceResult(
        primary_inference=primary,
        tax_loss_year_effects=year_effects,
    )


def _primary_row(
    *,
    strategy_name: str,
    values: pd.Series,
    effect_unit: str,
    inference_method: str,
    source_artifact: str,
    primary_metric: str,
    sample_unit: str,
    limitation_note: str,
) -> dict[str, object]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        effect = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        p_value = np.nan
    else:
        effect = float(clean.mean())
        ci_lower, ci_upper = bootstrap_mean_confidence_interval(clean)
        p_value = sign_flip_mean_p_value(clean, alternative=PRIMARY_ALTERNATIVE)

    return {
        "research_question": "publication_primary",
        "analysis_key": strategy_name,
        "test_label": f"{strategy_name} primary publication hypothesis",
        "governance": CONFIRMATORY_GOVERNANCE,
        "claim_scope": "strategy_level_confirmatory_transferability",
        "inference_method": inference_method,
        "effect_estimate": effect,
        "effect_unit": effect_unit,
        "null_value": 0.0,
        "alternative": PRIMARY_ALTERNATIVE,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "p_value": p_value,
        "adjusted_p_value": np.nan,
        "multiple_testing_family": CONFIRMATORY_FAMILY_ID,
        "multiple_testing_method": HOLM_ADJUSTMENT_METHOD,
        "reject_null_0_05": pd.NA,
        "sample_size": int(clean.size),
        "claim_label": pd.NA,
        "limitation_note": limitation_note,
        "source_artifact": source_artifact,
        "primary_metric": primary_metric,
        "sample_unit": sample_unit,
    }
