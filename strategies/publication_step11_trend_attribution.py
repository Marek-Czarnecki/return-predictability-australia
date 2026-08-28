from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STEP = "11.1.5"
FROZEN_CAPSTONE_COMMIT = "349fc00b087d404ba11bb9f04e9fe8ba7ad58ed6"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trend_row(comparison_path: Path) -> pd.Series:
    frame = pd.read_csv(comparison_path)
    rows = frame.loc[frame["strategy_family"] == "trend_following"]
    if len(rows) != 1:
        raise ValueError("Expected exactly one trend_following row in corrected-vs-frozen comparison.")
    return rows.iloc[0]


def build_trend_attribution_table(
    corrected_vs_frozen_path: Path,
    common_period_metadata_path: Path,
    universe_metadata_path: Path,
    benchmark_metadata_path: Path,
    cost_metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frozen = _trend_row(corrected_vs_frozen_path)
    common = _load_json(common_period_metadata_path)
    universe = _load_json(universe_metadata_path)
    benchmark = _load_json(benchmark_metadata_path)
    cost = _load_json(cost_metadata_path)

    expected_steps = {
        common.get("step"): "11.1.1",
        universe.get("step"): "11.1.2",
        benchmark.get("step"): "11.1.3",
        cost.get("step"): "11.1.4",
    }
    if any(observed != expected for observed, expected in expected_steps.items()):
        raise ValueError(f"Unexpected Step 11 metadata inputs: {expected_steps}")

    control_mean = float(common["mean_net_excess_nav_difference"])
    publication_mean = float(frozen["publication_nav_difference_mean"])
    frozen_nav = float(frozen["frozen_reconstructed_nav_difference_mean"])
    sample_extension_effect = publication_mean - control_mean

    rows = [
        {
            "design_change": "metric_semantics",
            "attribution_class": "not_explanatory",
            "magnitude_role": "none",
            "controlled_effect_nav_difference": pd.NA,
            "control_mean_nav_difference": frozen_nav,
            "ablation_mean_nav_difference": publication_mean,
            "positive_fold_control": pd.NA,
            "positive_fold_ablation": pd.NA,
            "parameter_selection_change_count": pd.NA,
            "evidence_source": "step9_corrected_vs_frozen",
            "evidence_note": (
                "The frozen trend deterioration appears under both legacy arithmetic excess and "
                "reconstructed NAV-difference semantics; metric redefinition does not explain the collapse."
            ),
        },
        {
            "design_change": "risk_free_treatment",
            "attribution_class": "not_explanatory",
            "magnitude_role": "none",
            "controlled_effect_nav_difference": pd.NA,
            "control_mean_nav_difference": pd.NA,
            "ablation_mean_nav_difference": pd.NA,
            "positive_fold_control": pd.NA,
            "positive_fold_ablation": pd.NA,
            "parameter_selection_change_count": pd.NA,
            "evidence_source": "step9_method_semantics",
            "evidence_note": (
                "Risk-free returns are used for risk-adjusted summaries, not the primary benchmark-relative "
                "selection objective or net-excess-NAV effect metric."
            ),
        },
        {
            "design_change": "sample_period_and_fold_calendar",
            "attribution_class": "not_explanatory",
            "magnitude_role": "opposes_collapse",
            "controlled_effect_nav_difference": sample_extension_effect,
            "control_mean_nav_difference": control_mean,
            "ablation_mean_nav_difference": publication_mean,
            "positive_fold_control": int(common["positive_fold_count"]),
            "positive_fold_ablation": pd.NA,
            "parameter_selection_change_count": pd.NA,
            "evidence_source": "step11.1.1",
            "evidence_note": (
                "The publication-standard result is already negative on the exact frozen seven evaluation periods. "
                "Extending to the full 24-fold publication sample raises, rather than causes the loss of, mean performance."
            ),
        },
        {
            "design_change": "point_in_time_membership",
            "attribution_class": "directly_demonstrated",
            "magnitude_role": "major_contributor",
            "controlled_effect_nav_difference": float(universe["mean_universe_effect_nav_difference"]),
            "control_mean_nav_difference": float(universe["pit_mean_net_excess_nav_difference"]),
            "ablation_mean_nav_difference": float(universe["retrospective_mean_net_excess_nav_difference"]),
            "positive_fold_control": int(universe["pit_positive_fold_count"]),
            "positive_fold_ablation": int(universe["retrospective_positive_fold_count"]),
            "parameter_selection_change_count": int(universe["parameter_selection_changed_fold_count"]),
            "evidence_source": "step11.1.2",
            "evidence_note": (
                "Applying the Norgate 2026-07-20 constituent set retrospectively changes the same-vendor, same-period "
                "trend result from negative to strongly positive. This demonstrates a major survivorship-membership effect. "
                "The publication cost framework is unchanged, although liquidity-tier assignments may change endogenously "
                "because the membership universe itself changes."
            ),
        },
        {
            "design_change": "benchmark_choice",
            "attribution_class": "not_explanatory",
            "magnitude_role": "immaterial",
            "controlled_effect_nav_difference": float(benchmark["mean_benchmark_effect_nav_difference"]),
            "control_mean_nav_difference": float(benchmark["xjoa_mean_net_excess_nav_difference"]),
            "ablation_mean_nav_difference": float(benchmark["stw_mean_net_excess_nav_difference"]),
            "positive_fold_control": int(benchmark["xjoa_positive_fold_count"]),
            "positive_fold_ablation": int(benchmark["stw_positive_fold_count"]),
            "parameter_selection_change_count": int(benchmark["parameter_selection_change_count"]),
            "evidence_source": "step11.1.3",
            "evidence_note": (
                "Replacing XJOA with STW in both formation selection and evaluation changes mean performance only marginally, "
                "leaves positive-fold incidence unchanged, and changes parameter selection in no folds."
            ),
        },
        {
            "design_change": "transaction_costs",
            "attribution_class": "directly_demonstrated",
            "magnitude_role": "modest_contributor",
            "controlled_effect_nav_difference": float(cost["mean_cost_effect_nav_difference"]),
            "control_mean_nav_difference": float(cost["base_mean_net_excess_nav_difference"]),
            "ablation_mean_nav_difference": float(cost["zero_mean_net_excess_nav_difference"]),
            "positive_fold_control": int(cost["base_positive_fold_count"]),
            "positive_fold_ablation": int(cost["zero_positive_fold_count"]),
            "parameter_selection_change_count": int(cost["parameter_selection_change_count"]),
            "evidence_source": "step11.1.4",
            "evidence_note": (
                "Removing all transaction costs improves mean performance but leaves it negative and leaves positive-fold "
                "incidence at 1/7. Costs are therefore a modest contributor, not an explanation for the sign reversal."
            ),
        },
        {
            "design_change": "vendor_and_security_coverage",
            "attribution_class": "unresolved_contributor",
            "magnitude_role": "unresolved",
            "controlled_effect_nav_difference": pd.NA,
            "control_mean_nav_difference": pd.NA,
            "ablation_mean_nav_difference": pd.NA,
            "positive_fold_control": pd.NA,
            "positive_fold_ablation": pd.NA,
            "parameter_selection_change_count": pd.NA,
            "evidence_source": "step11.1.2_mapping_diagnostic",
            "evidence_note": (
                "The frozen Yahoo current-constituent set could not be reconciled one-for-one to the Norgate panel without "
                "unsupported identity substitutions. Vendor/security coverage therefore remains unresolved and is not "
                "attributed to the point-in-time membership ablation."
            ),
        },
    ]
    table = pd.DataFrame(rows)

    statement = (
        "The trend-result collapse is not explained by metric semantics, risk-free treatment, sample extension, or benchmark choice. "
        "Within a common Norgate design and the frozen seven-fold calendar, retrospective application of the later constituent set "
        f"raises mean benchmark-relative NAV performance by {float(universe['mean_universe_effect_nav_difference']):.4f}, from "
        f"{float(universe['pit_mean_net_excess_nav_difference']):.4f} to {float(universe['retrospective_mean_net_excess_nav_difference']):.4f}, "
        "directly demonstrating that retrospective-current membership is the major identified contributor. Removing transaction costs "
        f"adds {float(cost['mean_cost_effect_nav_difference']):.4f} but leaves mean performance negative, so costs are a modest contributor. "
        "Yahoo-versus-Norgate security coverage remains unresolved; the evidence therefore supports a bounded attribution to the "
        "survivorship-membership mechanism rather than a claim that it fully explains every difference from the frozen capstone."
    )

    metadata = {
        "step": STEP,
        "analysis_role": "consolidated_diagnostic_attribution",
        "confirmatory": False,
        "strategy_name": "trend_following",
        "frozen_capstone_commit": FROZEN_CAPSTONE_COMMIT,
        "preferred_metric": "net_excess_nav_difference",
        "common_period_control_mean": control_mean,
        "frozen_reconstructed_nav_difference_mean": frozen_nav,
        "full_publication_mean": publication_mean,
        "sample_extension_effect_full_minus_common_period": sample_extension_effect,
        "major_identified_contributor": "point_in_time_membership_vs_retrospective_current_membership",
        "modest_identified_contributor": "transaction_costs",
        "unresolved_contributor": "vendor_and_security_coverage",
        "not_part_of_primary_holm_family": True,
        "attribution_statement": statement,
        "interpretation_rule": (
            "Treat the table as diagnostic attribution across already-locked design changes. Do not sum ablation effects as if they "
            "were independent, and do not claim that the identified survivorship-membership effect fully reconciles the frozen and "
            "publication implementations."
        ),
    }
    return table, metadata


def export_trend_attribution(table: pd.DataFrame, metadata: dict[str, object], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "attribution": output_dir / "publication_step11_trend_attribution.csv",
        "metadata": output_dir / "publication_step11_trend_attribution_metadata.json",
    }
    table.to_csv(paths["attribution"], index=False)
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return paths
