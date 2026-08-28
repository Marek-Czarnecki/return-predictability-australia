from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import erf, sqrt

import numpy as np
import pandas as pd


CONFIRMATORY_GOVERNANCE = "confirmatory"
DESCRIPTIVE_GOVERNANCE = "descriptive"
HOLM_ADJUSTMENT_METHOD = "holm"
NO_ADJUSTMENT_METHOD = "not_applicable_descriptive"
CONFIRMATORY_FAMILY_ID = "confirmatory_primary_family"

RQ1_INFERENCE_METHOD = "walk_forward_fold_sign_flip"
RQ2_INFERENCE_METHOD = "paired_event_control_sign_flip"
RQ3_INFERENCE_METHOD = "clustered_descriptive_association"
RQ4_INFERENCE_METHOD = "descriptive_sensitivity_summary"

INFERENCE_OUTPUT_COLUMNS = [
    "research_question",
    "analysis_key",
    "test_label",
    "governance",
    "claim_scope",
    "inference_method",
    "effect_estimate",
    "effect_unit",
    "null_value",
    "alternative",
    "ci_lower_95",
    "ci_upper_95",
    "p_value",
    "adjusted_p_value",
    "multiple_testing_family",
    "multiple_testing_method",
    "reject_null_0_05",
    "sample_size",
    "claim_label",
    "limitation_note",
    "source_artifact",
]

RQ1_SMALL_SAMPLE_COLUMNS = [
    "strategy",
    "row_type",
    "fold_id",
    "omitted_fold_id",
    "fold_count_used",
    "omitted_fold_effect",
    "effect_estimate",
    "effect_shift_vs_full_sample",
    "ci_lower_95",
    "ci_upper_95",
    "p_value",
    "p_value_shift_vs_full_sample",
    "positive_fold_count",
    "non_positive_fold_count",
    "positive_fold_share",
    "all_folds_positive",
    "worst_fold_id",
    "worst_fold_effect",
    "best_fold_id",
    "best_fold_effect",
]


@dataclass(frozen=True)
class InferenceResultSpec:
    research_question: str
    analysis_key: str
    test_label: str
    governance: str
    claim_scope: str
    inference_method: str
    source_artifact: str
    alternative: str = "greater"
    null_value: float = 0.0
    multiple_testing_family: str | None = None
    multiple_testing_method: str = NO_ADJUSTMENT_METHOD


def build_rq1_walk_forward_inference(
    fold_summary: pd.DataFrame,
    metric_col: str = "total_net_excess_return",
    strategy_col: str = "strategy",
    fold_id_col: str = "fold_id",
) -> pd.DataFrame:
    required_columns = {strategy_col, metric_col}
    missing_columns = required_columns.difference(fold_summary.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"RQ1 fold summary is missing required columns: {missing_list}")

    working = fold_summary.copy()
    if "window_label" in working.columns:
        working = working.loc[working["window_label"].astype("string").eq("evaluation")].copy()

    rows: list[dict[str, object]] = []
    for strategy_name, strategy_frame in working.groupby(strategy_col, observed=True):
        metric = pd.to_numeric(strategy_frame[metric_col], errors="coerce").dropna()
        spec = InferenceResultSpec(
            research_question="RQ1",
            analysis_key=str(strategy_name),
            test_label=(
                f"{strategy_name} walk-forward mean net excess return vs benchmark across evaluation folds"
            ),
            governance=CONFIRMATORY_GOVERNANCE,
            claim_scope="strategy_level_confirmatory_performance",
            inference_method=RQ1_INFERENCE_METHOD,
            source_artifact="*_walk_forward_summary.csv",
            multiple_testing_family=CONFIRMATORY_FAMILY_ID,
            multiple_testing_method=HOLM_ADJUSTMENT_METHOD,
        )
        limitation_note = (
            "Fold-level confirmatory inference on evaluation-fold outcomes. "
            "Interpret as evidence about average out-of-sample benchmark-relative performance, "
            "not as a causal claim."
        )
        rows.append(
            _build_scalar_inference_row(
                metric,
                spec=spec,
                limitation_note=limitation_note,
                sample_unit_label="folds",
            )
        )
    return _coerce_inference_frame(rows)


def build_rq1_small_sample_sensitivity(
    fold_summary: pd.DataFrame,
    metric_col: str = "total_net_excess_return",
    strategy_col: str = "strategy",
    fold_id_col: str = "fold_id",
) -> pd.DataFrame:
    required_columns = {strategy_col, metric_col, fold_id_col}
    missing_columns = required_columns.difference(fold_summary.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "RQ1 small-sample sensitivity is missing required columns: "
            f"{missing_list}"
        )

    working = fold_summary.copy()
    if "window_label" in working.columns:
        working = working.loc[working["window_label"].astype("string").eq("evaluation")].copy()

    rows: list[dict[str, object]] = []
    for strategy_name, strategy_frame in working.groupby(strategy_col, observed=True):
        strategy_rows = strategy_frame.loc[
            pd.to_numeric(strategy_frame[metric_col], errors="coerce").notna()
        ].copy()
        if strategy_rows.empty:
            continue
        strategy_rows[metric_col] = pd.to_numeric(strategy_rows[metric_col], errors="coerce")
        strategy_rows = strategy_rows.sort_values(fold_id_col).reset_index(drop=True)
        full_values = strategy_rows[metric_col].to_numpy(dtype=float)
        full_stats = _rq1_small_sample_row(
            strategy_name=str(strategy_name),
            row_type="full_sample",
            fold_id=None,
            omitted_fold_id=None,
            values=full_values,
            omitted_fold_effect=np.nan,
            full_sample_effect=float(full_values.mean()),
            full_sample_p_value=sign_flip_mean_p_value(full_values, alternative="greater"),
            full_frame=strategy_rows,
            fold_id_col=fold_id_col,
            metric_col=metric_col,
        )
        rows.append(full_stats)
        for _, omitted_row in strategy_rows.iterrows():
            leave_one_out = strategy_rows.loc[
                strategy_rows[fold_id_col] != omitted_row[fold_id_col]
            ].copy()
            leave_values = leave_one_out[metric_col].to_numpy(dtype=float)
            rows.append(
                _rq1_small_sample_row(
                    strategy_name=str(strategy_name),
                    row_type="leave_one_out",
                    fold_id=str(omitted_row[fold_id_col]),
                    omitted_fold_id=str(omitted_row[fold_id_col]),
                    values=leave_values,
                    omitted_fold_effect=float(omitted_row[metric_col]),
                    full_sample_effect=full_stats["effect_estimate"],
                    full_sample_p_value=full_stats["p_value"],
                    full_frame=leave_one_out,
                    fold_id_col=fold_id_col,
                    metric_col=metric_col,
                )
            )
    return _coerce_rq1_small_sample_frame(rows)


def build_rq2_tax_loss_inference(
    event_study: pd.DataFrame,
    difference_col: str = "return_difference",
) -> pd.DataFrame:
    if difference_col not in event_study.columns:
        raise ValueError(f"RQ2 event study is missing required column: {difference_col}")

    differences = pd.to_numeric(event_study[difference_col], errors="coerce").dropna()
    spec = InferenceResultSpec(
        research_question="RQ2",
        analysis_key="tax_loss_event_minus_control",
        test_label="Tax-loss event-window return minus paired control-window return",
        governance=CONFIRMATORY_GOVERNANCE,
        claim_scope="strategy_level_confirmatory_event_study",
        inference_method=RQ2_INFERENCE_METHOD,
        source_artifact="tax_loss_selling_event_study.csv",
        multiple_testing_family=CONFIRMATORY_FAMILY_ID,
        multiple_testing_method=HOLM_ADJUSTMENT_METHOD,
    )
    limitation_note = (
        "Paired confirmatory inference on ticker-event return differences. "
        "Interpret as event-versus-control association evidence under the fixed event-study design, "
        "not as a causal estimate."
    )
    row = _build_scalar_inference_row(
        differences,
        spec=spec,
        limitation_note=limitation_note,
        sample_unit_label="ticker_events",
    )
    return _coerce_inference_frame([row])


def build_rq3_descriptive_inference_summary(model_results: pd.DataFrame) -> pd.DataFrame:
    if model_results.empty:
        return _coerce_inference_frame(
            [
                {
                    "research_question": "RQ3",
                    "analysis_key": "rq3_model_unavailable",
                    "test_label": "RQ3 clustered descriptive association model unavailable",
                    "governance": DESCRIPTIVE_GOVERNANCE,
                    "claim_scope": "descriptive_association_only",
                    "inference_method": RQ3_INFERENCE_METHOD,
                    "effect_estimate": np.nan,
                    "effect_unit": "coefficient",
                    "null_value": 0.0,
                    "alternative": "two-sided",
                    "ci_lower_95": np.nan,
                    "ci_upper_95": np.nan,
                    "p_value": np.nan,
                    "adjusted_p_value": np.nan,
                    "multiple_testing_family": pd.NA,
                    "multiple_testing_method": NO_ADJUSTMENT_METHOD,
                    "reject_null_0_05": pd.NA,
                    "sample_size": 0,
                    "claim_label": "descriptive_unavailable",
                    "limitation_note": "RQ3 descriptive association output is not yet available.",
                    "source_artifact": "sector_liquidity_model_results.csv",
                }
            ]
        )

    if _rq3_model_unavailable(model_results):
        first_row = model_results.iloc[0]
        return _coerce_inference_frame(
            [
                {
                    "research_question": "RQ3",
                    "analysis_key": "rq3_model_unavailable",
                    "test_label": "RQ3 clustered descriptive association model unavailable",
                    "governance": DESCRIPTIVE_GOVERNANCE,
                    "claim_scope": "descriptive_association_only",
                    "inference_method": RQ3_INFERENCE_METHOD,
                    "effect_estimate": np.nan,
                    "effect_unit": "coefficient",
                    "null_value": 0.0,
                    "alternative": "two-sided",
                    "ci_lower_95": np.nan,
                    "ci_upper_95": np.nan,
                    "p_value": np.nan,
                    "adjusted_p_value": np.nan,
                    "multiple_testing_family": pd.NA,
                    "multiple_testing_method": NO_ADJUSTMENT_METHOD,
                    "reject_null_0_05": pd.NA,
                    "sample_size": first_row.get("sample_size", 0),
                    "claim_label": "descriptive_unavailable",
                    "limitation_note": str(
                        first_row.get(
                            "limitation_note",
                            "RQ3 descriptive association output is not yet available.",
                        )
                    ),
                    "source_artifact": "sector_liquidity_model_results.csv",
                }
            ]
        )

    rows: list[dict[str, object]] = []
    for record in model_results.to_dict("records"):
        term = str(record.get("term", "unknown_term"))
        limitation_note = str(record.get("limitation_note", "")).strip()
        limitation_note = (
            f"{limitation_note} No causal interpretation is supported."
            if limitation_note
            else "No causal interpretation is supported."
        )
        rows.append(
            {
                "research_question": "RQ3",
                "analysis_key": term,
                "test_label": f"RQ3 descriptive clustered association term: {term}",
                "governance": DESCRIPTIVE_GOVERNANCE,
                "claim_scope": "descriptive_association_only",
                "inference_method": RQ3_INFERENCE_METHOD,
                "effect_estimate": record.get("coefficient", np.nan),
                "effect_unit": "coefficient",
                "null_value": 0.0,
                "alternative": "two-sided",
                "ci_lower_95": record.get("lower_ci_95", np.nan),
                "ci_upper_95": record.get("upper_ci_95", np.nan),
                "p_value": record.get("p_value", np.nan),
                "adjusted_p_value": np.nan,
                "multiple_testing_family": pd.NA,
                "multiple_testing_method": NO_ADJUSTMENT_METHOD,
                "reject_null_0_05": pd.NA,
                "sample_size": record.get("sample_size", np.nan),
                "claim_label": "descriptive_association",
                "limitation_note": limitation_note,
                "source_artifact": "sector_liquidity_model_results.csv",
            }
        )
    return _coerce_inference_frame(rows)


def _rq3_model_unavailable(model_results: pd.DataFrame) -> bool:
    if model_results.empty:
        return True
    first_term = str(model_results.iloc[0].get("term", ""))
    return first_term in {
        "no_data",
        "insufficient_sample",
        "missing_cluster_identifier",
        "insufficient_clusters",
    }


def build_rq4_descriptive_inference_summary(
    scenario_results: pd.DataFrame | None = None,
    grid_results: pd.DataFrame | None = None,
    break_even: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sample_size = 0
    if scenario_results is not None and not scenario_results.empty:
        sample_size = max(sample_size, int(len(scenario_results)))
    if grid_results is not None and not grid_results.empty:
        sample_size = max(sample_size, int(len(grid_results)))
    if break_even is not None and not break_even.empty:
        sample_size = max(sample_size, int(len(break_even)))

    row = {
        "research_question": "RQ4",
        "analysis_key": "transaction_cost_sensitivity",
        "test_label": "Transaction-cost sensitivity and break-even evidence",
        "governance": DESCRIPTIVE_GOVERNANCE,
        "claim_scope": "descriptive_sensitivity_only",
        "inference_method": RQ4_INFERENCE_METHOD,
        "effect_estimate": np.nan,
        "effect_unit": "not_applicable",
        "null_value": np.nan,
        "alternative": "not_applicable",
        "ci_lower_95": np.nan,
        "ci_upper_95": np.nan,
        "p_value": np.nan,
        "adjusted_p_value": np.nan,
        "multiple_testing_family": pd.NA,
        "multiple_testing_method": NO_ADJUSTMENT_METHOD,
        "reject_null_0_05": pd.NA,
        "sample_size": sample_size,
        "claim_label": "descriptive_sensitivity",
        "limitation_note": (
            "RQ4 remains descriptive sensitivity-analysis evidence under the frozen specification. "
            "No confirmatory multiple-testing adjustment is applied."
        ),
        "source_artifact": (
            "transaction_cost_scenario_results.csv|transaction_cost_grid_results.csv|"
            "transaction_cost_break_even.csv"
        ),
    }
    return _coerce_inference_frame([row])


def apply_confirmatory_holm_adjustment(inference_results: pd.DataFrame) -> pd.DataFrame:
    if inference_results.empty:
        return _coerce_inference_frame([])

    adjusted = inference_results.copy()
    if "adjusted_p_value" not in adjusted.columns:
        adjusted["adjusted_p_value"] = np.nan
    if "reject_null_0_05" not in adjusted.columns:
        adjusted["reject_null_0_05"] = pd.NA
    if "claim_label" not in adjusted.columns:
        adjusted["claim_label"] = pd.NA

    family_mask = adjusted["governance"].astype("string").eq(CONFIRMATORY_GOVERNANCE)
    family_mask &= adjusted["multiple_testing_family"].astype("string").eq(
        CONFIRMATORY_FAMILY_ID
    )
    family = adjusted.loc[family_mask].copy()
    if family.empty:
        descriptive_mask = ~family_mask
        adjusted.loc[descriptive_mask, "claim_label"] = adjusted.loc[
            descriptive_mask, "claim_label"
        ].fillna("descriptive_only")
        return _coerce_inference_frame(adjusted.to_dict("records"))

    raw_p = pd.to_numeric(family["p_value"], errors="coerce")
    valid_mask = raw_p.notna()
    adjusted_values = pd.Series(np.nan, index=family.index, dtype=float)
    reject_flags = pd.Series(pd.NA, index=family.index, dtype="boolean")
    claim_labels = pd.Series(pd.NA, index=family.index, dtype="string")

    if valid_mask.any():
        holm_values = _holm_adjust(raw_p.loc[valid_mask])
        adjusted_values.loc[holm_values.index] = holm_values
        reject_flags.loc[holm_values.index] = holm_values <= 0.05
        claim_labels.loc[holm_values.index] = np.where(
            holm_values <= 0.05,
            "confirmatory_supported_after_holm",
            "confirmatory_not_supported_after_holm",
        )

    missing_mask = ~valid_mask
    if missing_mask.any():
        claim_labels.loc[family.index[missing_mask]] = "confirmatory_inference_incomplete"

    adjusted.loc[family.index, "adjusted_p_value"] = adjusted_values
    adjusted.loc[family.index, "reject_null_0_05"] = reject_flags
    adjusted.loc[family.index, "claim_label"] = claim_labels

    descriptive_mask = ~family_mask
    adjusted.loc[descriptive_mask, "claim_label"] = adjusted.loc[
        descriptive_mask, "claim_label"
    ].fillna("descriptive_only")
    adjusted.loc[descriptive_mask, "reject_null_0_05"] = pd.NA
    adjusted.loc[descriptive_mask, "adjusted_p_value"] = np.nan
    return _coerce_inference_frame(adjusted.to_dict("records"))


def build_statistical_inference_summary(
    rq1_results: pd.DataFrame | None = None,
    rq2_results: pd.DataFrame | None = None,
    rq3_results: pd.DataFrame | None = None,
    rq4_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = [
        frame
        for frame in [rq1_results, rq2_results, rq3_results, rq4_results]
        if frame is not None and not frame.empty
    ]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return apply_confirmatory_holm_adjustment(combined)


def _build_scalar_inference_row(
    values: pd.Series,
    spec: InferenceResultSpec,
    limitation_note: str,
    sample_unit_label: str,
) -> dict[str, object]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return {
            "research_question": spec.research_question,
            "analysis_key": spec.analysis_key,
            "test_label": spec.test_label,
            "governance": spec.governance,
            "claim_scope": spec.claim_scope,
            "inference_method": spec.inference_method,
            "effect_estimate": np.nan,
            "effect_unit": sample_unit_label,
            "null_value": spec.null_value,
            "alternative": spec.alternative,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "p_value": np.nan,
            "adjusted_p_value": np.nan,
            "multiple_testing_family": spec.multiple_testing_family,
            "multiple_testing_method": spec.multiple_testing_method,
            "reject_null_0_05": pd.NA,
            "sample_size": 0,
            "claim_label": "confirmatory_inference_incomplete"
            if spec.governance == CONFIRMATORY_GOVERNANCE
            else "descriptive_unavailable",
            "limitation_note": f"No valid observations were available. {limitation_note}",
            "source_artifact": spec.source_artifact,
        }

    ci_lower, ci_upper = bootstrap_mean_confidence_interval(clean.to_numpy(dtype=float))
    p_value = sign_flip_mean_p_value(clean.to_numpy(dtype=float), alternative=spec.alternative)
    return {
        "research_question": spec.research_question,
        "analysis_key": spec.analysis_key,
        "test_label": spec.test_label,
        "governance": spec.governance,
        "claim_scope": spec.claim_scope,
        "inference_method": spec.inference_method,
        "effect_estimate": float(clean.mean()),
        "effect_unit": sample_unit_label,
        "null_value": spec.null_value,
        "alternative": spec.alternative,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "p_value": p_value,
        "adjusted_p_value": np.nan,
        "multiple_testing_family": spec.multiple_testing_family,
        "multiple_testing_method": spec.multiple_testing_method,
        "reject_null_0_05": pd.NA,
        "sample_size": int(clean.shape[0]),
        "claim_label": pd.NA,
        "limitation_note": limitation_note,
        "source_artifact": spec.source_artifact,
    }


def _rq1_small_sample_row(
    strategy_name: str,
    row_type: str,
    fold_id: str | None,
    omitted_fold_id: str | None,
    values: np.ndarray,
    omitted_fold_effect: float,
    full_sample_effect: float,
    full_sample_p_value: float,
    full_frame: pd.DataFrame,
    fold_id_col: str,
    metric_col: str,
) -> dict[str, object]:
    if values.size == 0:
        ci_lower = np.nan
        ci_upper = np.nan
        p_value = np.nan
        effect_estimate = np.nan
    else:
        ci_lower, ci_upper = bootstrap_mean_confidence_interval(values)
        p_value = sign_flip_mean_p_value(values, alternative="greater")
        effect_estimate = float(values.mean())

    positive_count = int((values > 0.0).sum())
    non_positive_count = int(values.size - positive_count)
    if full_frame.empty:
        worst_fold_id = np.nan
        worst_fold_effect = np.nan
        best_fold_id = np.nan
        best_fold_effect = np.nan
    else:
        worst_index = full_frame[metric_col].idxmin()
        best_index = full_frame[metric_col].idxmax()
        worst_fold_id = str(full_frame.loc[worst_index, fold_id_col])
        worst_fold_effect = float(full_frame.loc[worst_index, metric_col])
        best_fold_id = str(full_frame.loc[best_index, fold_id_col])
        best_fold_effect = float(full_frame.loc[best_index, metric_col])

    return {
        "strategy": strategy_name,
        "row_type": row_type,
        "fold_id": fold_id if fold_id is not None else pd.NA,
        "omitted_fold_id": omitted_fold_id if omitted_fold_id is not None else pd.NA,
        "fold_count_used": int(values.size),
        "omitted_fold_effect": omitted_fold_effect,
        "effect_estimate": effect_estimate,
        "effect_shift_vs_full_sample": effect_estimate - full_sample_effect
        if not np.isnan(effect_estimate)
        else np.nan,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "p_value": p_value,
        "p_value_shift_vs_full_sample": p_value - full_sample_p_value
        if not np.isnan(p_value)
        else np.nan,
        "positive_fold_count": positive_count,
        "non_positive_fold_count": non_positive_count,
        "positive_fold_share": float(positive_count / values.size) if values.size else np.nan,
        "all_folds_positive": bool(values.size > 0 and positive_count == values.size),
        "worst_fold_id": worst_fold_id,
        "worst_fold_effect": worst_fold_effect,
        "best_fold_id": best_fold_id,
        "best_fold_effect": best_fold_effect,
    }


def bootstrap_mean_confidence_interval(
    values: np.ndarray, alpha: float = 0.05, bootstrap_reps: int = 10000
) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(bootstrap_reps, values.size), replace=True)
    sample_means = samples.mean(axis=1)
    lower = float(np.quantile(sample_means, alpha / 2))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2))
    return lower, upper


def sign_flip_mean_p_value(
    values: np.ndarray,
    alternative: str = "greater",
    exact_max_size: int = 16,
    monte_carlo_draws: int = 20000,
) -> float:
    if values.size == 0:
        return np.nan
    observed = float(values.mean())
    if values.size <= exact_max_size:
        sign_matrix = np.array(list(product([-1.0, 1.0], repeat=values.size)), dtype=float)
        permuted = (sign_matrix * values).mean(axis=1)
    else:
        rng = np.random.default_rng(0)
        sign_matrix = rng.choice(
            np.array([-1.0, 1.0], dtype=float),
            size=(monte_carlo_draws, values.size),
            replace=True,
        )
        permuted = (sign_matrix * values).mean(axis=1)
    if alternative == "greater":
        return float(np.mean(permuted >= observed - 1e-15))
    if alternative == "less":
        return float(np.mean(permuted <= observed + 1e-15))
    if alternative == "two-sided":
        return float(np.mean(np.abs(permuted) >= abs(observed) - 1e-15))
    raise ValueError(f"Unsupported alternative: {alternative}")


def _bootstrap_confidence_interval(
    values: np.ndarray, alpha: float = 0.05, bootstrap_reps: int = 10000
) -> tuple[float, float]:
    return bootstrap_mean_confidence_interval(
        values,
        alpha=alpha,
        bootstrap_reps=bootstrap_reps,
    )


def _sign_flip_p_value(
    values: np.ndarray,
    alternative: str = "greater",
    exact_max_size: int = 16,
    monte_carlo_draws: int = 20000,
) -> float:
    return sign_flip_mean_p_value(
        values,
        alternative=alternative,
        exact_max_size=exact_max_size,
        monte_carlo_draws=monte_carlo_draws,
    )


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    ordered = p_values.sort_values(kind="mergesort")
    m = len(ordered)
    adjusted_ordered = pd.Series(index=ordered.index, dtype=float)
    running_max = 0.0
    for position, (index, value) in enumerate(ordered.items(), start=1):
        adjusted_value = min(1.0, (m - position + 1) * float(value))
        running_max = max(running_max, adjusted_value)
        adjusted_ordered.loc[index] = running_max
    return adjusted_ordered.reindex(p_values.index)


def _coerce_inference_frame(rows: list[dict[str, object]] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy() if not isinstance(rows, pd.DataFrame) else rows.copy()
    for column in INFERENCE_OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, INFERENCE_OUTPUT_COLUMNS]


def _coerce_rq1_small_sample_frame(
    rows: list[dict[str, object]] | pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy() if not isinstance(rows, pd.DataFrame) else rows.copy()
    for column in RQ1_SMALL_SAMPLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, RQ1_SMALL_SAMPLE_COLUMNS]
