from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FROZEN_CAPSTONE_COMMIT = "349fc00b087d404ba11bb9f04e9fe8ba7ad58ed6"
COMPARISON_NAME = "publication_corrected_vs_frozen_comparison.csv"
ATTRIBUTION_NAME = "publication_corrected_vs_frozen_attribution.csv"
METADATA_NAME = "publication_corrected_vs_frozen_metadata.json"

DAILY_STRATEGIES = (
    "trend_following",
    "mean_reversion",
    "pairs_trading",
)


@dataclass(frozen=True)
class PublicationComparison:
    comparison: pd.DataFrame
    attribution: pd.DataFrame
    metadata: dict[str, object]


def build_publication_comparison(
    frozen_results_root: Path,
    publication_results_root: Path,
) -> PublicationComparison:
    frozen_results_root = Path(frozen_results_root)
    publication_results_root = Path(publication_results_root)

    _require_files(
        frozen_results_root,
        [
            "trend_following_walk_forward_summary.csv",
            "mean_reversion_walk_forward_summary.csv",
            "pairs_trading_walk_forward_summary.csv",
            "tax_loss_selling_summary.csv",
            "tax_loss_selling_year_robustness.csv",
            "statistical_inference_summary.csv",
        ],
        "frozen",
    )
    _require_files(
        publication_results_root,
        [
            "publication_trend_following_walk_forward_summary.csv",
            "publication_mean_reversion_walk_forward_summary.csv",
            "publication_pairs_trading_walk_forward_summary.csv",
            "publication_tax_loss_selling_summary.csv",
            "publication_primary_inference.csv",
            "publication_step8_evidence_metadata.json",
        ],
        "publication",
    )

    step8_metadata = _load_json(
        publication_results_root / "publication_step8_evidence_metadata.json"
    )
    if step8_metadata.get("status") != "frozen" or str(step8_metadata.get("step")) != "8":
        raise ValueError("Publication Step 8 evidence is not frozen.")

    frozen_inference = pd.read_csv(frozen_results_root / "statistical_inference_summary.csv")
    publication_inference = pd.read_csv(
        publication_results_root / "publication_primary_inference.csv"
    )

    rows: list[dict[str, object]] = []
    for strategy in DAILY_STRATEGIES:
        frozen_summary = pd.read_csv(
            frozen_results_root / f"{strategy}_walk_forward_summary.csv"
        )
        publication_summary = pd.read_csv(
            publication_results_root / f"publication_{strategy}_walk_forward_summary.csv"
        )
        _validate_daily_summary(
            frozen_summary,
            expected_folds=7,
            label=f"frozen {strategy}",
            publication=False,
        )
        _validate_daily_summary(
            publication_summary,
            expected_folds=24,
            label=f"publication {strategy}",
            publication=True,
        )

        frozen_primary = _one_row(frozen_inference, "analysis_key", strategy)
        publication_primary = _one_row(publication_inference, "analysis_key", strategy)

        frozen_nav = pd.to_numeric(
            frozen_summary["absolute_total_return"], errors="raise"
        ) - pd.to_numeric(frozen_summary["benchmark_total_return"], errors="raise")
        publication_nav = pd.to_numeric(
            publication_summary["net_excess_nav_difference"], errors="raise"
        )

        frozen_reject = bool(frozen_primary["reject_null_0_05"])
        publication_reject = bool(publication_primary["reject_null_0_05"])

        rows.append(
            {
                "strategy_family": strategy,
                "comparison_class": "derived_comparable",
                "frozen_primary_metric": "total_net_excess_return",
                "frozen_primary_value": float(frozen_primary["effect_estimate"]),
                "frozen_legacy_excess_mean": float(
                    pd.to_numeric(frozen_summary["total_net_excess_return"], errors="raise").mean()
                ),
                "frozen_reconstructed_nav_difference_mean": float(frozen_nav.mean()),
                "frozen_inferential_unit": str(frozen_primary["effect_unit"]),
                "frozen_sample_size": int(frozen_primary["sample_size"]),
                "frozen_raw_p_value": float(frozen_primary["p_value"]),
                "frozen_adjusted_p_value": float(frozen_primary["adjusted_p_value"]),
                "frozen_reject_null_0_05": frozen_reject,
                "publication_legacy_excess_mean": float(
                    pd.to_numeric(publication_summary["total_net_excess_return"], errors="raise").mean()
                ),
                "publication_primary_metric": str(publication_primary["primary_metric"]),
                "publication_primary_value": float(publication_primary["effect_estimate"]),
                "publication_nav_difference_mean": float(publication_nav.mean()),
                "publication_inferential_unit": str(publication_primary["sample_unit"]),
                "publication_sample_size": int(publication_primary["sample_size"]),
                "publication_raw_p_value": float(publication_primary["p_value"]),
                "publication_adjusted_p_value": float(publication_primary["adjusted_p_value"]),
                "publication_reject_null_0_05": publication_reject,
                "economic_direction_change": _direction_change(
                    float(frozen_nav.mean()), float(publication_nav.mean())
                ),
                "evidence_strength_change": _evidence_strength_change(
                    frozen_reject, publication_reject
                ),
                "comparability_note": (
                    "Frozen preferred-comparison value is reconstructed as mean fold "
                    "absolute_total_return minus benchmark_total_return; publication value "
                    "uses the explicitly preferred net_excess_nav_difference."
                ),
            }
        )

    rows.append(
        _build_tax_loss_row(
            frozen_results_root,
            publication_results_root,
            frozen_inference,
            publication_inference,
        )
    )

    comparison = pd.DataFrame(rows)
    attribution = _build_attribution()
    metadata: dict[str, object] = {
        "status": "complete",
        "step": "9.6",
        "step_label": "consolidated_corrected_vs_frozen_comparison",
        "frozen_capstone_commit": FROZEN_CAPSTONE_COMMIT,
        "publication_step8_status": step8_metadata.get("status"),
        "publication_step8_artifact_count": step8_metadata.get("artifact_count"),
        "comparison_strategy_count": int(len(comparison)),
        "attribution_row_count": int(len(attribution)),
        "comparison_classes": sorted(comparison["comparison_class"].unique().tolist()),
        "causal_attribution_policy": (
            "Do not assign causal responsibility to simultaneous publication-design changes "
            "without an isolated counterfactual. Use directly_demonstrated, plausible_contributor, "
            "or not_explanatory classifications."
        ),
        "trend_key_result": (
            "Trend deterioration persists under both legacy arithmetic-excess semantics and "
            "the comparable NAV-difference metric; metric cleanup does not explain the collapse."
        ),
        "tax_loss_key_result": (
            "The positive tax-loss point estimate persists, but confirmatory support disappears "
            "when inference is based on independent calendar-year means rather than pooled ticker-events."
        ),
    }
    return PublicationComparison(
        comparison=comparison,
        attribution=attribution,
        metadata=metadata,
    )


def _build_tax_loss_row(
    frozen_root: Path,
    publication_root: Path,
    frozen_inference: pd.DataFrame,
    publication_inference: pd.DataFrame,
) -> dict[str, object]:
    frozen_summary = pd.read_csv(frozen_root / "tax_loss_selling_summary.csv").iloc[0]
    frozen_year = pd.read_csv(frozen_root / "tax_loss_selling_year_robustness.csv")
    publication_summary = pd.read_csv(
        publication_root / "publication_tax_loss_selling_summary.csv"
    ).iloc[0]

    frozen_primary = _one_row(
        frozen_inference, "analysis_key", "tax_loss_event_minus_control"
    )
    publication_primary = _one_row(
        publication_inference, "analysis_key", "tax_loss_selling"
    )
    frozen_year_primary = _one_row(
        frozen_year, "analysis_level", "year_clustered_sign_flip"
    )

    frozen_reject = bool(frozen_primary["reject_null_0_05"])
    publication_reject = bool(publication_primary["reject_null_0_05"])

    return {
        "strategy_family": "tax_loss_selling",
        "comparison_class": "context_only",
        "frozen_primary_metric": "mean_return_difference",
        "frozen_primary_value": float(frozen_primary["effect_estimate"]),
        "frozen_legacy_excess_mean": float(frozen_summary["mean_return_difference"]),
        "frozen_reconstructed_nav_difference_mean": float(
            frozen_summary["mean_abnormal_return_difference"]
        ),
        "frozen_inferential_unit": str(frozen_primary["effect_unit"]),
        "frozen_sample_size": int(frozen_primary["sample_size"]),
        "frozen_raw_p_value": float(frozen_primary["p_value"]),
        "frozen_adjusted_p_value": float(frozen_primary["adjusted_p_value"]),
        "frozen_reject_null_0_05": frozen_reject,
        "publication_legacy_excess_mean": float(publication_summary["mean_return_difference"]),
        "publication_primary_metric": str(publication_primary["primary_metric"]),
        "publication_primary_value": float(publication_primary["effect_estimate"]),
        "publication_nav_difference_mean": float(
            publication_summary["mean_abnormal_net_return_difference"]
        ),
        "publication_inferential_unit": str(publication_primary["sample_unit"]),
        "publication_sample_size": int(publication_primary["sample_size"]),
        "publication_raw_p_value": float(publication_primary["p_value"]),
        "publication_adjusted_p_value": float(publication_primary["adjusted_p_value"]),
        "publication_reject_null_0_05": publication_reject,
        "economic_direction_change": _direction_change(
            float(frozen_summary["mean_abnormal_return_difference"]),
            float(publication_summary["mean_abnormal_net_return_difference"]),
        ),
        "evidence_strength_change": _evidence_strength_change(
            frozen_reject, publication_reject
        ),
        "comparability_note": (
            "Context-only: frozen primary inference pooled ticker-events, whereas publication "
            "primary inference uses equal-weight calendar-year means of benchmark-adjusted net "
            "event-minus-control returns. Frozen year-level robustness effect="
            f"{float(frozen_year_primary['mean_return_difference']):.12g}, "
            f"p={float(frozen_year_primary['p_value']):.12g}."
        ),
    }


def _build_attribution() -> pd.DataFrame:
    rows = [
        ("trend_following", "metric_semantics", "not_explanatory", "Deterioration appears under both legacy arithmetic excess and comparable NAV difference."),
        ("trend_following", "risk_free_correction", "not_explanatory", "Risk-free series is used for risk-adjusted summaries and not the primary benchmark-relative selection/effect metric."),
        ("trend_following", "sample_extension", "plausible_contributor", "OOS evidence expands from 7 to 24 folds and materially broadens regime coverage; no isolated same-universe counterfactual was run."),
        ("trend_following", "point_in_time_universe", "plausible_contributor", "Historical membership replaces retrospective current survivors; no isolated same-period counterfactual was run."),
        ("trend_following", "benchmark_change", "plausible_contributor", "STW is replaced by XJOA TRI; the benchmark change was not isolated from the other publication corrections."),
        ("trend_following", "liquidity_aware_costs", "plausible_contributor", "Formation-only liquidity tiers and implementation-aware costs may affect magnitude but were not isolated."),
        ("mean_reversion", "overall_design_change", "directly_demonstrated", "The strategy remains strongly negative under both frozen and publication designs; corrections change magnitude, not conclusion."),
        ("pairs_trading", "capital_normalization", "directly_demonstrated", "Publication implementation corrects the gross-exposure denominator; its individual effect cannot be separated from simultaneous design changes."),
        ("pairs_trading", "overall_design_change", "directly_demonstrated", "Pairs remains negative and unsupported after PIT eligibility, capital normalization and two-leg cost corrections."),
        ("tax_loss_selling", "inferential_unit", "directly_demonstrated", "Frozen pooled ticker-event significance was already contradicted by non-significant year-level robustness; publication makes calendar years primary and expands to 26 years."),
        ("tax_loss_selling", "economic_effect", "directly_demonstrated", "The benchmark-adjusted net point estimate remains positive; the publication result is unsupported rather than sign-reversed."),
        ("tax_loss_selling", "benchmark_cost_missing_data_redesign", "plausible_contributor", "XJOA abnormal adjustment, symmetric costs and strict complete-window rules affect magnitude but are not isolated individually."),
    ]
    return pd.DataFrame(
        rows,
        columns=["strategy_family", "design_change", "attribution_class", "evidence_note"],
    )


def _direction_change(frozen_value: float, publication_value: float) -> str:
    if frozen_value > 0 and publication_value <= 0:
        return "positive_to_non_positive"
    if frozen_value < 0 and publication_value >= 0:
        return "negative_to_non_negative"
    if frozen_value > 0 and publication_value > 0:
        return "positive_to_positive"
    if frozen_value < 0 and publication_value < 0:
        return "negative_to_negative"
    return "contains_zero"


def _evidence_strength_change(frozen_reject: bool, publication_reject: bool) -> str:
    if frozen_reject and not publication_reject:
        return "supported_to_unsupported"
    if not frozen_reject and publication_reject:
        return "unsupported_to_supported"
    if frozen_reject and publication_reject:
        return "supported_to_supported"
    return "unsupported_to_unsupported"


def _validate_daily_summary(
    frame: pd.DataFrame,
    expected_folds: int,
    label: str,
    publication: bool,
) -> None:
    required = {
        "fold_id",
        "absolute_total_return",
        "total_net_excess_return",
    }
    if publication:
        required.add("net_excess_nav_difference")
    else:
        required.add("benchmark_total_return")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} summary missing columns: {', '.join(missing)}")
    if len(frame) != expected_folds:
        raise ValueError(f"{label} summary has {len(frame)} rows; expected {expected_folds}.")
    if frame["fold_id"].nunique() != expected_folds:
        raise ValueError(f"{label} summary does not contain {expected_folds} unique folds.")


def _one_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    selected = frame.loc[frame[column].astype(str) == value]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one row where {column}={value}; found {len(selected)}.")
    return selected.iloc[0]


def _require_files(root: Path, names: list[str], label: str) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing {label} comparison inputs: " + ", ".join(sorted(missing)))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
