from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from .costs import build_ticker_cost_bps_map
from .costs import basis_points_to_rate

if TYPE_CHECKING:
    from .walk_forward import WalkForwardFold


@dataclass(frozen=True)
class SectorMapValidation:
    sector_map: pd.DataFrame
    coverage: pd.DataFrame
    mapped_ticker_count: int
    unmatched_ticker_count: int
    duplicate_mapping_count: int
    missing_sector_count: int
    classification_treatment: str
    grouping_design: str
    classification_date_min: str | None
    classification_date_max: str | None


@dataclass(frozen=True)
class SectorLiquidityArtifacts:
    ticker_table: pd.DataFrame
    group_results: pd.DataFrame
    model_results: pd.DataFrame
    summary: pd.DataFrame
    coverage: pd.DataFrame
    rq3_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)
    leave_one_fold_out: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


CURRENT_ONLY_SECTOR_TREATMENT = "current_only"
FINAL_SECTOR_GROUPING_DESIGN = "Financials / Resources / Other"
FINAL_SECTOR_BUCKETS = ("Financials", "Resources", "Other")
CURRENT_ONLY_SECTOR_LIMITATION_NOTE = (
    "Sector classification is current-only from config/asx_ticker_sector_map.csv; "
    "no point-in-time sector history is modeled."
)
RQ3_INFERENCE_MODEL_TYPE = "rq3_clustered_association_ols"


def load_sector_map(path: Path) -> pd.DataFrame:
    sector_map = pd.read_csv(path).copy()
    required_columns = {"ticker_code", "sector"}
    missing_columns = required_columns.difference(sector_map.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Sector map is missing required columns: {missing_list}")
    optional_columns = [
        column
        for column in ["classification_date", "status", "source", "industry", "company_name"]
        if column in sector_map.columns
    ]
    sector_map = sector_map.loc[:, ["ticker_code", "sector", *optional_columns]]
    sector_map["ticker_code"] = sector_map["ticker_code"].astype("string").str.strip()
    sector_map["sector"] = sector_map["sector"].astype("string")
    return sector_map


def validate_sector_map(
    sector_map: pd.DataFrame, ticker_universe: pd.Series | pd.Index
) -> SectorMapValidation:
    working_map = sector_map.copy()
    working_map["ticker_code"] = working_map["ticker_code"].astype("string").str.strip()
    working_map["sector"] = working_map["sector"].astype("string")

    duplicate_mask = working_map["ticker_code"].duplicated(keep=False)
    duplicate_mapping_count = int(duplicate_mask.sum())
    if duplicate_mapping_count:
        duplicate_codes = working_map.loc[duplicate_mask, "ticker_code"].tolist()
        duplicate_list = ", ".join(sorted(set(str(code) for code in duplicate_codes)))
        raise ValueError(f"Duplicate sector mappings found for: {duplicate_list}")

    working_map["sector"] = working_map["sector"].str.strip()
    missing_sector_mask = working_map["sector"].isna() | working_map["sector"].eq("")
    missing_sector_count = int(missing_sector_mask.sum())
    classification_dates = pd.Series(dtype="string")
    if "classification_date" in working_map.columns:
        classification_dates = (
            working_map["classification_date"]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )

    ticker_index = pd.Index(pd.Series(ticker_universe, dtype="string").dropna().unique())
    coverage = pd.DataFrame({"ticker_code": ticker_index.astype("string")})
    mapped_sector = working_map.set_index("ticker_code")["sector"]
    coverage["sector"] = coverage["ticker_code"].map(mapped_sector)
    coverage["mapped_flag"] = coverage["sector"].notna() & coverage["sector"].ne("")
    coverage["unmatched_flag"] = ~coverage["mapped_flag"]

    return SectorMapValidation(
        sector_map=working_map,
        coverage=coverage,
        mapped_ticker_count=int(coverage["mapped_flag"].sum()),
        unmatched_ticker_count=int(coverage["unmatched_flag"].sum()),
        duplicate_mapping_count=duplicate_mapping_count,
        missing_sector_count=missing_sector_count,
        classification_treatment=CURRENT_ONLY_SECTOR_TREATMENT,
        grouping_design=FINAL_SECTOR_GROUPING_DESIGN,
        classification_date_min=(
            str(classification_dates.min()) if not classification_dates.empty else None
        ),
        classification_date_max=(
            str(classification_dates.max()) if not classification_dates.empty else None
        ),
    )


def build_liquidity_tiers(
    ticker_liquidity: pd.DataFrame,
    high_liquidity_cutoff: float = 0.30,
    medium_liquidity_cutoff: float = 0.70,
) -> pd.DataFrame:
    if "liquidity_percentile" not in ticker_liquidity.columns:
        raise ValueError("Ticker liquidity input must include 'liquidity_percentile'.")
    if not 0 < high_liquidity_cutoff < medium_liquidity_cutoff <= 1:
        raise ValueError(
            "Liquidity cutoffs must satisfy 0 < high < medium <= 1."
        )

    liquidity = ticker_liquidity.copy()
    conditions = [
        liquidity["liquidity_percentile"] <= high_liquidity_cutoff,
        liquidity["liquidity_percentile"] <= medium_liquidity_cutoff,
    ]
    categories = np.select(conditions, ["high", "medium"], default="lower")
    liquidity["liquidity_tier"] = pd.Categorical(
        categories,
        categories=["high", "medium", "lower"],
        ordered=True,
    )
    return liquidity


def summarize_liquidity_tiers(liquidity_table: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"liquidity_tier", "median_dollar_volume", "mean_dollar_volume"}
    missing_columns = required_columns.difference(liquidity_table.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Liquidity table is missing required columns: {missing_list}"
        )

    summary = (
        liquidity_table.groupby("liquidity_tier", observed=True)
        .agg(
            ticker_count=("liquidity_tier", "size"),
            median_dollar_volume=("median_dollar_volume", "median"),
            mean_dollar_volume=("mean_dollar_volume", "mean"),
            min_percentile=("liquidity_percentile", "min"),
            max_percentile=("liquidity_percentile", "max"),
        )
        .reset_index()
    )
    return summary


def build_long_only_attribution_panel(
    strategy_panel: pd.DataFrame,
    signal_col: str = "signal",
    return_col: str = "daily_return",
    turnover_cost_bps: float = 0.0,
    cost_scenario: str | None = None,
    liquidity_tier_map: pd.Series | dict[str, str] | None = None,
) -> pd.DataFrame:
    required_columns = {"trade_date", "ticker_code", signal_col, return_col}
    missing_columns = required_columns.difference(strategy_panel.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Strategy panel is missing required columns: {missing_list}"
        )

    signal_matrix = (
        strategy_panel.pivot(index="trade_date", columns="ticker_code", values=signal_col)
        .sort_index()
        .fillna(0.0)
    )
    return_matrix = (
        strategy_panel.pivot(index="trade_date", columns="ticker_code", values=return_col)
        .sort_index()
        .fillna(0.0)
    )
    weights = signal_matrix.div(signal_matrix.sum(axis=1), axis=0).fillna(0.0)
    lagged_weights = weights.shift(1).fillna(0.0)
    weight_change = weights.diff().fillna(0.0)
    turnover_component = weight_change.abs()
    resolved_turnover_cost_bps = _normalize_turnover_cost_bps(turnover_cost_bps)
    ticker_cost_bps = build_ticker_cost_bps_map(
        signal_matrix.columns,
        cost_scenario=cost_scenario,
        liquidity_tier_map=liquidity_tier_map,
        turnover_cost_bps=resolved_turnover_cost_bps,
    )

    long_weights = weights.stack(dropna=False).rename("weight").reset_index()
    long_weights["lagged_weight"] = (
        lagged_weights.stack(dropna=False).rename("lagged_weight").reset_index(drop=True)
    )
    long_weights["turnover_component"] = (
        turnover_component.stack(dropna=False)
        .rename("turnover_component")
        .reset_index(drop=True)
    )
    long_weights["asset_return"] = (
        return_matrix.stack(dropna=False).rename("asset_return").reset_index(drop=True)
    )
    long_weights["gross_contribution"] = (
        long_weights["lagged_weight"] * long_weights["asset_return"]
    )
    long_weights["turnover_cost_bps"] = (
        long_weights["ticker_code"].astype("string").map(ticker_cost_bps).astype(float)
    )
    long_weights["net_contribution"] = long_weights["gross_contribution"] - (
        long_weights["turnover_component"]
        * long_weights["turnover_cost_bps"].map(basis_points_to_rate)
    )
    return long_weights


def build_ticker_level_outcomes(
    attribution_panel: pd.DataFrame, strategy_family: str
) -> pd.DataFrame:
    required_columns = {
        "ticker_code",
        "trade_date",
        "gross_contribution",
        "net_contribution",
        "turnover_component",
    }
    missing_columns = required_columns.difference(attribution_panel.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Attribution panel is missing required columns: {missing_list}"
        )

    outcomes = (
        attribution_panel.groupby("ticker_code", observed=True)
        .agg(
            observation_count=("trade_date", "nunique"),
            mean_gross_contribution=("gross_contribution", "mean"),
            mean_net_contribution=("net_contribution", "mean"),
            volatility=("net_contribution", "std"),
            cumulative_net_contribution=("net_contribution", "sum"),
            average_turnover=("turnover_component", "mean"),
        )
        .reset_index()
    )
    outcomes["strategy_family"] = strategy_family
    return outcomes


def summarize_group_performance(
    outcomes: pd.DataFrame,
    group_col: str,
    performance_col: str,
    turnover_col: str | None = None,
) -> pd.DataFrame:
    required_columns = {"strategy_family", group_col, performance_col}
    missing_columns = required_columns.difference(outcomes.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Outcome table is missing required columns: {missing_list}")

    grouped = (
        outcomes.groupby(["strategy_family", group_col], observed=True)
        .agg(
            observation_count=("ticker_code", "size"),
            mean_return=(performance_col, "mean"),
            median_return=(performance_col, "median"),
            volatility=(performance_col, "std"),
            cumulative_performance=(performance_col, "sum"),
        )
        .reset_index()
    )
    if turnover_col and turnover_col in outcomes.columns:
        turnover_summary = (
            outcomes.groupby(["strategy_family", group_col], observed=True)[turnover_col]
            .mean()
            .rename("average_turnover")
            .reset_index()
        )
        grouped = grouped.merge(
            turnover_summary,
            on=["strategy_family", group_col],
            how="left",
        )
    else:
        grouped["average_turnover"] = np.nan
    return grouped


def build_tax_loss_group_outcomes(
    event_study: pd.DataFrame,
    ticker_metadata: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    required_columns = {
        "year",
        "ticker_code",
        "event_window_return",
        "control_window_return",
        "return_difference",
    }
    missing_columns = required_columns.difference(event_study.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Event study is missing required columns: {missing_list}"
        )

    merged = event_study.merge(ticker_metadata, on="ticker_code", how="left")
    grouped = (
        merged.groupby(group_col, observed=True)
        .agg(
            observation_count=("ticker_code", "size"),
            mean_return=("return_difference", "mean"),
            median_return=("return_difference", "median"),
            volatility=("return_difference", "std"),
            cumulative_performance=("return_difference", "sum"),
        )
        .reset_index()
    )
    grouped["strategy_family"] = "tax_loss_selling"
    grouped["average_turnover"] = np.nan
    return grouped[
        [
            "strategy_family",
            group_col,
            "observation_count",
            "mean_return",
            "median_return",
            "volatility",
            "cumulative_performance",
            "average_turnover",
        ]
    ]


def classify_sector_bucket(
    sector_series: pd.Series,
    financials_labels: tuple[str, ...] = ("Financials",),
    resources_labels: tuple[str, ...] = ("Materials", "Energy"),
) -> pd.Categorical:
    sector_text = sector_series.astype("string")
    bucket = np.where(
        sector_text.isin(financials_labels),
        "Financials",
        np.where(sector_text.isin(resources_labels), "Resources", "Other"),
    )
    return pd.Categorical(
        bucket,
        categories=list(FINAL_SECTOR_BUCKETS),
        ordered=False,
    )

def build_sector_liquidity_artifacts(
    prices: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
    tax_event_study: pd.DataFrame | None = None,
    walk_forward_results: dict[str, object] | None = None,
) -> SectorLiquidityArtifacts:
    from .data_prep import build_ticker_liquidity
    from .tax_loss_selling import run_tax_loss_selling_event_study

    ticker_liquidity = build_ticker_liquidity(prices)
    liquidity_with_tiers = build_liquidity_tiers(ticker_liquidity.reset_index())
    liquidity_lookup = liquidity_with_tiers[
        [
            "ticker_code",
            "liquidity_tier",
            "median_dollar_volume",
            "mean_dollar_volume",
            "liquidity_percentile",
        ]
    ].copy()
    liquidity_tier_map = (
        liquidity_lookup.set_index("ticker_code")["liquidity_tier"].astype("string")
    )

    sector_validation: SectorMapValidation | None = None
    coverage_table = pd.DataFrame()
    if sector_map is not None and not sector_map.empty:
        sector_validation = validate_sector_map(sector_map, liquidity_lookup["ticker_code"])
        coverage_table = sector_validation.coverage.copy()

    resolved_walk_forward_results = walk_forward_results or _build_default_rq3_walk_forward_results(
        prices
    )
    strategy_ticker_outcomes: list[pd.DataFrame] = []
    reconciliation_frames: list[pd.DataFrame] = []
    strategy_group_results: list[pd.DataFrame] = []
    for strategy_family in ("trend_following", "mean_reversion"):
        result = resolved_walk_forward_results[strategy_family]
        strategy_panel, reconciliation = build_fold_aware_ticker_outcomes(
            prices=prices,
            fold_table=result.fold_table,
            committed_daily_results=result.fold_daily_results,
            strategy_family=strategy_family,
            liquidity_tier_map=liquidity_tier_map,
        )
        enriched_panel = strategy_panel.merge(liquidity_lookup, on="ticker_code", how="left")
        strategy_ticker_outcomes.append(enriched_panel)
        reconciliation_frames.append(reconciliation)
        strategy_group_results.append(
            summarize_group_performance(
                enriched_panel,
                group_col="liquidity_tier",
                performance_col="mean_net_contribution",
                turnover_col="average_turnover",
            )
        )

    resolved_tax_event_study = (
        tax_event_study.copy()
        if tax_event_study is not None
        else run_tax_loss_selling_event_study(prices)[0]
    )
    tax_liquidity_results = build_tax_loss_group_outcomes(
        resolved_tax_event_study,
        liquidity_lookup[["ticker_code", "liquidity_tier"]],
        "liquidity_tier",
    )
    strategy_group_results.append(tax_liquidity_results)

    liquidity_group_results = pd.concat(strategy_group_results, ignore_index=True)
    rq3_panel = pd.concat(strategy_ticker_outcomes, ignore_index=True)
    reconciliation_table = (
        pd.concat(reconciliation_frames, ignore_index=True)
        if reconciliation_frames
        else pd.DataFrame()
    )
    if not reconciliation_table.empty and not reconciliation_table["reconciliation_passed"].all():
        failed = reconciliation_table.loc[
            ~reconciliation_table["reconciliation_passed"], ["strategy_family", "fold_id"]
        ]
        raise ValueError(
            "RQ3 fold attribution reconciliation failed for: "
            + ", ".join(
                f"{row.strategy_family}:{row.fold_id}" for row in failed.itertuples(index=False)
            )
        )

    sector_group_results = pd.DataFrame()
    if sector_validation is not None:
        sector_lookup = sector_validation.sector_map.copy()
        sector_lookup["sector_bucket"] = classify_sector_bucket(sector_lookup["sector"])

        sector_ticker_outcomes = rq3_panel.merge(
            sector_lookup[["ticker_code", "sector", "sector_bucket"]],
            on="ticker_code",
            how="left",
        )
        sector_group_results = summarize_group_performance(
            sector_ticker_outcomes,
            group_col="sector_bucket",
            performance_col="mean_net_contribution",
            turnover_col="average_turnover",
        )
        tax_sector_results = build_tax_loss_group_outcomes(
            resolved_tax_event_study,
            sector_lookup[["ticker_code", "sector_bucket"]],
            "sector_bucket",
        )
        sector_group_results = pd.concat(
            [sector_group_results, tax_sector_results], ignore_index=True
        )

    model_results = pd.DataFrame()
    model_note = "insufficient_strategy_attribution"
    leave_one_fold_out = pd.DataFrame()
    robustness_summary = pd.DataFrame()
    export_payload = rq3_panel.copy()
    if sector_validation is not None and not rq3_panel.empty:
        model_input = prepare_model_dataset(
            rq3_panel,
            sector_validation.sector_map,
        )
        model_results = fit_rq3_association_model(model_input)
        leave_one_fold_out = build_rq3_leave_one_fold_out(model_input, model_results)
        robustness_summary = summarize_rq3_robustness(leave_one_fold_out, model_results)
        model_note = str(model_results.loc[0, "limitation_note"])
        export_payload = model_input.copy()

    summary = _build_sector_liquidity_summary(
        liquidity_group_results=liquidity_group_results,
        sector_group_results=sector_group_results,
        sector_validation=sector_validation,
        cleaned_ticker_count=int(liquidity_lookup["ticker_code"].nunique()),
        model_results=model_results,
        model_note=model_note,
        rq3_panel=export_payload,
    )

    combined_group_results = (
        pd.concat([liquidity_group_results, sector_group_results], ignore_index=True)
        if not sector_group_results.empty
        else liquidity_group_results
    )
    return SectorLiquidityArtifacts(
        ticker_table=export_payload,
        group_results=combined_group_results,
        model_results=model_results,
        summary=summary,
        coverage=coverage_table,
        rq3_panel=export_payload,
        reconciliation=reconciliation_table,
        leave_one_fold_out=leave_one_fold_out,
        robustness_summary=robustness_summary,
    )


def fit_preliminary_exposure_model(
    model_data: pd.DataFrame,
    dependent_col: str = "mean_net_contribution",
    liquidity_col: str = "liquidity_tier",
) -> pd.DataFrame:
    return fit_rq3_association_model(
        model_data,
        dependent_col=dependent_col,
        liquidity_col=liquidity_col,
        covariance_design="ticker_only",
    )


def fit_rq3_association_model(
    model_data: pd.DataFrame,
    dependent_col: str = "mean_net_contribution",
    liquidity_col: str = "liquidity_tier",
    covariance_design: str = "two_way",
) -> pd.DataFrame:
    required_columns = {dependent_col, liquidity_col, "sector_bucket", "strategy_family"}
    if covariance_design == "two_way":
        required_columns.update({"ticker_code", "fold_id"})
    elif covariance_design == "ticker_only":
        required_columns.add("ticker_code")
    missing_columns = required_columns.difference(model_data.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Model data is missing required columns: {missing_list}")

    working = model_data.dropna(subset=[dependent_col, liquidity_col]).copy()
    if working.empty:
        return pd.DataFrame(
            [
                {
                    "term": "no_data",
                    "coefficient": np.nan,
                    "standard_error": np.nan,
                    "test_statistic": np.nan,
                    "p_value": np.nan,
                    "lower_ci_95": np.nan,
                    "upper_ci_95": np.nan,
                    "sample_size": 0,
                    "ticker_cluster_count": 0,
                    "fold_cluster_count": 0,
                    "intersection_cluster_count": 0,
                    "cluster_unit": "unavailable",
                    "covariance_design": covariance_design,
                    "finite_sample_correction": "not_applicable",
                    "parameter_count": 0,
                    "inference_design": "no_valid_observations",
                    "model_type": RQ3_INFERENCE_MODEL_TYPE,
                    "adjusted_r_squared": np.nan,
                    "limitation_note": "No valid observations were available for modelling.",
                }
            ]
        )

    design, working = _build_rq3_design_matrix(working, liquidity_col=liquidity_col)
    y = working[dependent_col].astype(float).to_numpy()
    x = design.astype(float).to_numpy()
    sample_size, parameter_count = x.shape
    if sample_size <= parameter_count:
        limitation_note = (
            "Sample size is too small relative to the parameter count for a stable regression."
        )
        return pd.DataFrame(
            [
                {
                    "term": "insufficient_sample",
                    "coefficient": np.nan,
                    "standard_error": np.nan,
                    "test_statistic": np.nan,
                    "p_value": np.nan,
                    "lower_ci_95": np.nan,
                    "upper_ci_95": np.nan,
                    "sample_size": sample_size,
                    "ticker_cluster_count": 0,
                    "fold_cluster_count": 0,
                    "intersection_cluster_count": 0,
                    "cluster_unit": "unavailable",
                    "covariance_design": covariance_design,
                    "finite_sample_correction": "not_applicable",
                    "parameter_count": parameter_count,
                    "inference_design": "insufficient_sample",
                    "model_type": RQ3_INFERENCE_MODEL_TYPE,
                    "adjusted_r_squared": np.nan,
                    "limitation_note": limitation_note,
                }
            ]
        )

    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residuals = y - fitted
    dof = sample_size - parameter_count
    sigma2 = float((residuals @ residuals) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    total_ss = float(((y - y.mean()) ** 2).sum())
    residual_ss = float((residuals**2).sum())
    r_squared = 1.0 - residual_ss / total_ss if total_ss else np.nan
    adjusted_r_squared = (
        1.0 - (1.0 - r_squared) * (sample_size - 1) / dof if total_ss and dof > 0 else np.nan
    )

    ticker_clusters = (
        working["ticker_code"].astype("string").str.strip().replace("", pd.NA)
        if "ticker_code" in working.columns
        else pd.Series(pd.NA, index=working.index, dtype="string")
    )
    fold_clusters = (
        working["fold_id"].astype("string").str.strip().replace("", pd.NA)
        if "fold_id" in working.columns
        else pd.Series(pd.NA, index=working.index, dtype="string")
    )
    if covariance_design == "two_way":
        if ticker_clusters.dropna().empty or fold_clusters.dropna().empty:
            limitation_note = (
                "RQ3 inference requires ticker_code and fold_id for genuine two-way clustering. "
                f"{CURRENT_ONLY_SECTOR_LIMITATION_NOTE}"
            )
            return _rq3_unavailable_frame(
                term="missing_cluster_identifier",
                sample_size=sample_size,
                adjusted_r_squared=adjusted_r_squared,
                limitation_note=limitation_note,
                covariance_design=covariance_design,
                parameter_count=parameter_count,
            )
        ticker_cluster_count = int(ticker_clusters.nunique())
        fold_cluster_count = int(fold_clusters.nunique())
        intersection_clusters = (
            ticker_clusters.fillna("__missing_ticker__")
            + "||"
            + fold_clusters.fillna("__missing_fold__")
        )
        intersection_cluster_count = int(intersection_clusters.nunique())
        if min(ticker_cluster_count, fold_cluster_count, intersection_cluster_count) <= 1:
            limitation_note = (
                "RQ3 two-way clustered inference needs at least two ticker, fold, and intersection clusters. "
                f"{CURRENT_ONLY_SECTOR_LIMITATION_NOTE}"
            )
            return _rq3_unavailable_frame(
                term="insufficient_clusters",
                sample_size=sample_size,
                adjusted_r_squared=adjusted_r_squared,
                limitation_note=limitation_note,
                covariance_design=covariance_design,
                parameter_count=parameter_count,
                ticker_cluster_count=ticker_cluster_count,
                fold_cluster_count=fold_cluster_count,
                intersection_cluster_count=intersection_cluster_count,
            )
        covariance = _compute_two_way_cluster_robust_covariance(
            x,
            residuals,
            ticker_clusters=ticker_clusters,
            fold_clusters=fold_clusters,
        )
        cluster_unit = "ticker_code|fold_id"
        inference_design = "two_way_clustered_fold_aware"
        finite_sample_correction = (
            "Cameron-Gelbach-Miller with componentwise G/(G-1)*((N-1)/(N-K)) correction"
        )
    else:
        if ticker_clusters.dropna().empty:
            limitation_note = (
                "RQ3 ticker-clustered inference requires ticker_code as the cluster identifier. "
                f"{CURRENT_ONLY_SECTOR_LIMITATION_NOTE}"
            )
            return _rq3_unavailable_frame(
                term="missing_cluster_identifier",
                sample_size=sample_size,
                adjusted_r_squared=adjusted_r_squared,
                limitation_note=limitation_note,
                covariance_design=covariance_design,
                parameter_count=parameter_count,
            )
        ticker_cluster_count = int(ticker_clusters.nunique())
        fold_cluster_count = 0
        intersection_cluster_count = 0
        if ticker_cluster_count <= 1:
            limitation_note = (
                "RQ3 ticker-clustered inference needs at least two non-empty ticker clusters. "
                f"{CURRENT_ONLY_SECTOR_LIMITATION_NOTE}"
            )
            return _rq3_unavailable_frame(
                term="insufficient_clusters",
                sample_size=sample_size,
                adjusted_r_squared=adjusted_r_squared,
                limitation_note=limitation_note,
                covariance_design=covariance_design,
                parameter_count=parameter_count,
                ticker_cluster_count=ticker_cluster_count,
            )
        covariance = _compute_cluster_robust_covariance(x, residuals, ticker_clusters)
        cluster_unit = "ticker_code"
        inference_design = "ticker_clustered"
        finite_sample_correction = "one_way G/(G-1)*((N-1)/(N-K)) correction"

    if covariance is None:
        limitation_note = (
            "RQ3 covariance estimation failed to produce a usable clustered covariance matrix. "
            f"{CURRENT_ONLY_SECTOR_LIMITATION_NOTE}"
        )
        return _rq3_unavailable_frame(
            term="missing_cluster_identifier",
            sample_size=sample_size,
            adjusted_r_squared=adjusted_r_squared,
            limitation_note=limitation_note,
            covariance_design=covariance_design,
            parameter_count=parameter_count,
            ticker_cluster_count=ticker_cluster_count,
            fold_cluster_count=fold_cluster_count,
            intersection_cluster_count=intersection_cluster_count,
        )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_scores = np.divide(
        beta,
        standard_errors,
        out=np.full_like(beta, np.nan, dtype=float),
        where=standard_errors > 0,
    )
    p_values = np.array([_two_sided_normal_p_value(value) for value in z_scores])

    limitation_note = (
        "RQ3 pooled fold-aware association model. Interpret coefficients as descriptive "
        "associations, not causal effects. Sector classification is current-only from "
        "config/asx_ticker_sector_map.csv; no point-in-time sector history is modeled. "
        "Only 7 fold clusters are available, so clustered uncertainty estimates have limited "
        "small-cluster precision. Pairs trading and tax-loss selling are excluded from the "
        "primary regression because their outcome units are structurally non-comparable."
    )
    rows = []
    for term, coefficient, standard_error, z_score, p_value in zip(
        design.columns, beta, standard_errors, z_scores, p_values
    ):
        rows.append(
            {
                "term": term,
                "coefficient": float(coefficient),
                "standard_error": float(standard_error),
                "test_statistic": float(z_score) if not np.isnan(z_score) else np.nan,
                "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                "lower_ci_95": float(coefficient - 1.96 * standard_error),
                "upper_ci_95": float(coefficient + 1.96 * standard_error),
                "sample_size": sample_size,
                "ticker_cluster_count": ticker_cluster_count,
                "fold_cluster_count": fold_cluster_count,
                "intersection_cluster_count": intersection_cluster_count,
                "cluster_unit": cluster_unit,
                "covariance_design": covariance_design,
                "finite_sample_correction": finite_sample_correction,
                "parameter_count": parameter_count,
                "inference_design": inference_design,
                "model_type": RQ3_INFERENCE_MODEL_TYPE,
                "adjusted_r_squared": adjusted_r_squared,
                "limitation_note": limitation_note,
            }
        )
    return pd.DataFrame(rows)


def _normalize_turnover_cost_bps(turnover_cost_bps: float | None) -> float | None:
    if turnover_cost_bps is None or pd.isna(turnover_cost_bps):
        return None
    return float(turnover_cost_bps)


def _build_sector_liquidity_summary(
    liquidity_group_results: pd.DataFrame,
    sector_group_results: pd.DataFrame,
    sector_validation: SectorMapValidation | None,
    cleaned_ticker_count: int,
    model_results: pd.DataFrame,
    model_note: str,
    rq3_panel: pd.DataFrame,
) -> pd.DataFrame:
    high_liquidity = liquidity_group_results.loc[
        liquidity_group_results["liquidity_tier"] == "high", "mean_return"
    ]
    remaining_liquidity = liquidity_group_results.loc[
        liquidity_group_results["liquidity_tier"].isin(["medium", "lower"]), "mean_return"
    ]
    financials_association = (
        sector_group_results.loc[
            sector_group_results.get("sector_bucket", pd.Series(dtype="object"))
            == "Financials",
            "mean_return",
        ]
        if not sector_group_results.empty
        else pd.Series(dtype="float64")
    )
    resources_association = (
        sector_group_results.loc[
            sector_group_results.get("sector_bucket", pd.Series(dtype="object"))
            == "Resources",
            "mean_return",
        ]
        if not sector_group_results.empty
        else pd.Series(dtype="float64")
    )
    model_observation_count = 0 if model_results.empty else int(model_results.loc[0, "sample_size"])
    strategy_counts = (
        rq3_panel["strategy_family"].value_counts().sort_index().to_dict()
        if not rq3_panel.empty and "strategy_family" in rq3_panel.columns
        else {}
    )
    fold_count = int(rq3_panel["fold_id"].nunique()) if "fold_id" in rq3_panel.columns else 0
    rows_by_fold = (
        rq3_panel.groupby("fold_id", observed=True).size().sort_index().to_dict()
        if not rq3_panel.empty and "fold_id" in rq3_panel.columns
        else {}
    )
    missing_sector_rows = (
        int(rq3_panel["sector"].isna().sum()) if "sector" in rq3_panel.columns else 0
    )
    missing_liquidity_rows = (
        int(rq3_panel["liquidity_tier"].isna().sum())
        if "liquidity_tier" in rq3_panel.columns
        else 0
    )

    if sector_validation is None:
        status = "liquidity_only_sector_mapping_required"
    elif _rq3_model_is_populated(model_results):
        status = "complete_rq3_inference_upgrade"
    else:
        status = "rq3_model_unavailable"

    return pd.DataFrame(
        [
            {
                "mapped_ticker_coverage": (
                    0.0
                    if sector_validation is None
                    else sector_validation.mapped_ticker_count / cleaned_ticker_count
                ),
                "classification_treatment": (
                    CURRENT_ONLY_SECTOR_TREATMENT
                    if sector_validation is None
                    else sector_validation.classification_treatment
                ),
                "grouping_design": (
                    FINAL_SECTOR_GROUPING_DESIGN
                    if sector_validation is None
                    else sector_validation.grouping_design
                ),
                "high_liquidity_performance": (
                    high_liquidity.mean() if not high_liquidity.empty else pd.NA
                ),
                "remaining_universe_performance": (
                    remaining_liquidity.mean() if not remaining_liquidity.empty else pd.NA
                ),
                "financials_association": (
                    financials_association.mean()
                    if not financials_association.empty
                    else pd.NA
                ),
                "resources_association": (
                    resources_association.mean()
                    if not resources_association.empty
                    else pd.NA
                ),
                "model_observation_count": model_observation_count,
                "rq3_panel_row_count": int(len(rq3_panel)),
                "rq3_unique_ticker_count": int(rq3_panel["ticker_code"].nunique())
                if "ticker_code" in rq3_panel.columns
                else 0,
                "rq3_fold_count": fold_count,
                "rows_by_strategy": str(strategy_counts),
                "rows_by_fold": str(rows_by_fold),
                "missing_sector_rows": missing_sector_rows,
                "missing_liquidity_rows": missing_liquidity_rows,
                "status": status,
                "limitation_note": model_note,
            }
        ]
    )


def _rq3_model_is_populated(model_results: pd.DataFrame) -> bool:
    if model_results.empty:
        return False
    first_term = str(model_results.loc[0, "term"])
    if first_term in {
        "no_data",
        "insufficient_sample",
        "missing_cluster_identifier",
        "insufficient_clusters",
    }:
        return False
    return model_results["coefficient"].notna().any()


def _compute_cluster_robust_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    cluster_values: pd.Series,
) -> np.ndarray:
    xtx_inv = np.linalg.pinv(x.T @ x)
    cluster_codes = pd.Series(cluster_values, dtype="string").fillna("__missing_cluster__")
    unique_clusters = pd.Index(cluster_codes.unique())
    cluster_count = len(unique_clusters)
    sample_size, parameter_count = x.shape

    meat = np.zeros((parameter_count, parameter_count), dtype=float)
    for cluster in unique_clusters:
        cluster_mask = cluster_codes.eq(cluster).to_numpy()
        x_cluster = x[cluster_mask]
        residual_cluster = residuals[cluster_mask]
        score = x_cluster.T @ residual_cluster
        meat += np.outer(score, score)

    small_sample_scale = (cluster_count / (cluster_count - 1)) * (
        (sample_size - 1) / (sample_size - parameter_count)
    )
    return small_sample_scale * (xtx_inv @ meat @ xtx_inv)


def _compute_two_way_cluster_robust_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    ticker_clusters: pd.Series,
    fold_clusters: pd.Series,
) -> np.ndarray:
    intersection_clusters = (
        ticker_clusters.astype("string").fillna("__missing_ticker__")
        + "||"
        + fold_clusters.astype("string").fillna("__missing_fold__")
    )
    return (
        _compute_cluster_robust_covariance(x, residuals, ticker_clusters)
        + _compute_cluster_robust_covariance(x, residuals, fold_clusters)
        - _compute_cluster_robust_covariance(x, residuals, intersection_clusters)
    )


def _build_default_rq3_walk_forward_results(prices: pd.DataFrame) -> dict[str, object]:
    from .walk_forward import run_walk_forward_optimization

    zero_benchmark = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(prices["trade_date"]).sort_values().unique(),
            "benchmark_return": 0.0,
        }
    )
    return {
        strategy_family: run_walk_forward_optimization(
            strategy_name=strategy_family,
            prices=prices,
            benchmark_returns=zero_benchmark,
        )
        for strategy_family in ("trend_following", "mean_reversion")
    }


def build_fold_aware_ticker_outcomes(
    prices: pd.DataFrame,
    fold_table: pd.DataFrame,
    committed_daily_results: pd.DataFrame,
    strategy_family: str,
    liquidity_tier_map: pd.Series | dict[str, str] | None = None,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-8,
    timing_logger: Callable[[str, dict[str, object]], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .walk_forward import WalkForwardFold, parse_chosen_parameters, prepare_strategy_evaluation_panel
    from .costs import build_liquidity_tier_map

    required_columns = {
        "fold_id",
        "formation_start",
        "formation_end",
        "evaluation_start",
        "evaluation_end",
        "chosen_parameters",
    }
    missing_columns = required_columns.difference(fold_table.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fold table is missing required columns: {missing_list}")

    outcome_frames: list[pd.DataFrame] = []
    reconciliation_rows: list[dict[str, object]] = []
    committed_daily = committed_daily_results.copy()
    strategy_start_time = perf_counter()
    resolved_liquidity_tier_map = (
        liquidity_tier_map
        if liquidity_tier_map is not None
        else build_liquidity_tier_map(prices)
    )
    if not committed_daily.empty:
        committed_daily["trade_date"] = pd.to_datetime(committed_daily["trade_date"])

    for fold_row in fold_table.to_dict("records"):
        fold_start_time = perf_counter()
        fold = WalkForwardFold(
            fold_id=str(fold_row["fold_id"]),
            formation_start=pd.Timestamp(fold_row["formation_start"]),
            formation_end=pd.Timestamp(fold_row["formation_end"]),
            evaluation_start=pd.Timestamp(fold_row["evaluation_start"]),
            evaluation_end=pd.Timestamp(fold_row["evaluation_end"]),
        )
        parameters = parse_chosen_parameters(fold_row["chosen_parameters"])
        execution_start_time = perf_counter()
        evaluation_run, evaluation_panel = prepare_strategy_evaluation_panel(
            strategy_name=strategy_family,
            prices=prices,
            fold=fold,
            parameters=parameters,
        )
        execution_elapsed = perf_counter() - execution_start_time
        full_panel = getattr(evaluation_run, "panel", evaluation_panel).copy()
        full_panel["trade_date"] = pd.to_datetime(full_panel["trade_date"])
        evaluation_daily_results = getattr(evaluation_run, "daily_results", pd.DataFrame())
        execution_liquidity_tier_map = resolved_liquidity_tier_map
        if hasattr(evaluation_daily_results, "attrs"):
            execution_liquidity_tier_map = (
                evaluation_daily_results.attrs.get("liquidity_tier_map")
                or resolved_liquidity_tier_map
            )
        attribution_start_time = perf_counter()
        attribution = build_long_only_attribution_panel(
            full_panel,
            signal_col="signal",
            return_col="daily_return",
            turnover_cost_bps=np.nan,
            cost_scenario=str(parameters.get("cost_scenario", "base")),
            liquidity_tier_map=execution_liquidity_tier_map,
        )
        attribution = attribution.loc[
            (attribution["trade_date"] >= fold.evaluation_start)
            & (attribution["trade_date"] <= fold.evaluation_end)
        ].reset_index(drop=True)
        attribution_elapsed = perf_counter() - attribution_start_time
        aggregation_start_time = perf_counter()
        fold_outcomes = build_ticker_level_outcomes(attribution, strategy_family)
        participation = (
            attribution.groupby("ticker_code", observed=True)
            .agg(
                active_observation_count=("lagged_weight", lambda s: int(np.count_nonzero(s > 0))),
                mean_portfolio_weight=("lagged_weight", "mean"),
            )
            .reset_index()
        )
        fold_outcomes = fold_outcomes.merge(participation, on="ticker_code", how="left")
        fold_outcomes["participation_rate"] = np.divide(
            fold_outcomes["active_observation_count"],
            fold_outcomes["observation_count"],
            out=np.zeros(len(fold_outcomes), dtype=float),
            where=fold_outcomes["observation_count"].to_numpy(dtype=float) > 0,
        )
        fold_outcomes["fold_id"] = fold.fold_id
        fold_outcomes["evaluation_start"] = fold.evaluation_start
        fold_outcomes["evaluation_end"] = fold.evaluation_end
        fold_outcomes["chosen_parameters"] = str(fold_row["chosen_parameters"])
        outcome_frames.append(fold_outcomes)
        aggregation_elapsed = perf_counter() - aggregation_start_time

        reconciliation_start_time = perf_counter()
        reconstructed_daily = (
            attribution.groupby("trade_date", observed=True)
            .agg(
                reconstructed_net_return=("net_contribution", "sum"),
                reconstructed_gross_return=("gross_contribution", "sum"),
            )
            .reset_index()
        )
        fold_committed_daily = committed_daily.loc[
            committed_daily["fold_id"].astype("string").eq(fold.fold_id)
        ].copy()
        comparison = fold_committed_daily.merge(
            reconstructed_daily,
            on="trade_date",
            how="outer",
        ).sort_values("trade_date")
        comparison["net_return"] = comparison["net_return"].fillna(0.0)
        comparison["reconstructed_net_return"] = comparison["reconstructed_net_return"].fillna(0.0)
        comparison["absolute_daily_difference"] = (
            comparison["net_return"] - comparison["reconstructed_net_return"]
        ).abs()
        committed_value = float(comparison["net_return"].sum())
        reconstructed_value = float(comparison["reconstructed_net_return"].sum())
        absolute_difference = abs(committed_value - reconstructed_value)
        relative_difference = absolute_difference / max(abs(committed_value), 1e-12)
        max_daily_abs_difference = float(comparison["absolute_daily_difference"].max())
        reconciliation_passed = bool(
            (absolute_difference <= absolute_tolerance or relative_difference <= relative_tolerance)
            and max_daily_abs_difference <= absolute_tolerance
        )
        reconciliation_rows.append(
            {
                "strategy_family": strategy_family,
                "fold_id": fold.fold_id,
                "evaluation_start": fold.evaluation_start,
                "evaluation_end": fold.evaluation_end,
                "committed_value": committed_value,
                "reconstructed_value": reconstructed_value,
                "absolute_difference": absolute_difference,
                "relative_difference": relative_difference,
                "max_daily_abs_difference": max_daily_abs_difference,
                "observation_count": int(len(comparison)),
                "reconciliation_status": "pass" if reconciliation_passed else "fail",
                "reconciliation_passed": reconciliation_passed,
            }
        )
        reconciliation_elapsed = perf_counter() - reconciliation_start_time
        fold_elapsed = perf_counter() - fold_start_time
        if timing_logger is not None:
            timing_logger(
                "rq3_fold_timing",
                {
                    "strategy_family": strategy_family,
                    "fold_id": fold.fold_id,
                    "input_rows": int(len(full_panel)),
                    "evaluation_rows": int(len(evaluation_panel)),
                    "attributed_daily_rows": int(len(attribution)),
                    "ticker_rows": int(len(fold_outcomes)),
                    "execution_seconds": execution_elapsed,
                    "attribution_seconds": attribution_elapsed,
                    "aggregation_seconds": aggregation_elapsed,
                    "reconciliation_seconds": reconciliation_elapsed,
                    "fold_seconds": fold_elapsed,
                },
            )

    total_elapsed = perf_counter() - strategy_start_time
    if timing_logger is not None:
        timing_logger(
            "rq3_strategy_timing",
            {
                "strategy_family": strategy_family,
                "fold_count": int(len(outcome_frames)),
                "strategy_seconds": total_elapsed,
            },
        )

    return (
        pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame(),
        pd.DataFrame(reconciliation_rows),
    )


def prepare_model_dataset(
    ticker_outcomes: pd.DataFrame, sector_map: pd.DataFrame | None = None
) -> pd.DataFrame:
    model_data = ticker_outcomes.copy()
    if sector_map is not None and not sector_map.empty:
        model_data = model_data.merge(sector_map, on="ticker_code", how="left")
        model_data["sector_bucket"] = classify_sector_bucket(model_data["sector"])
    else:
        model_data["sector"] = pd.NA
        model_data["sector_bucket"] = pd.Categorical(
            ["Other"] * len(model_data), categories=list(FINAL_SECTOR_BUCKETS)
        )
    model_data["liquidity_tier"] = pd.Categorical(
        model_data["liquidity_tier"],
        categories=["high", "medium", "lower"],
        ordered=True,
    )
    model_data["financials_exposure"] = model_data["sector_bucket"].astype("string").eq(
        "Financials"
    )
    model_data["resources_exposure"] = model_data["sector_bucket"].astype("string").eq(
        "Resources"
    )
    return model_data


def build_rq3_leave_one_fold_out(
    model_data: pd.DataFrame,
    full_model_results: pd.DataFrame,
) -> pd.DataFrame:
    if model_data.empty or "fold_id" not in model_data.columns:
        return pd.DataFrame()
    if not _rq3_model_is_populated(full_model_results):
        return pd.DataFrame()

    full_coefficients = full_model_results.set_index("term")["coefficient"]
    rows: list[dict[str, object]] = []
    for omitted_fold_id in sorted(model_data["fold_id"].astype("string").unique()):
        reduced_data = model_data.loc[
            model_data["fold_id"].astype("string") != omitted_fold_id
        ].copy()
        reduced_results = fit_rq3_association_model(reduced_data)
        if not _rq3_model_is_populated(reduced_results):
            continue
        reduced_coefficients = reduced_results.set_index("term")["coefficient"]
        for term, coefficient in reduced_coefficients.items():
            full_value = float(full_coefficients.get(term, np.nan))
            rows.append(
                {
                    "omitted_fold_id": omitted_fold_id,
                    "term": term,
                    "coefficient": float(coefficient),
                    "full_model_coefficient": full_value,
                    "coefficient_shift": float(coefficient - full_value),
                    "sign_changed_vs_full": bool(
                        np.sign(coefficient) != np.sign(full_value)
                        and not np.isclose(full_value, 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_rq3_robustness(
    leave_one_fold_out: pd.DataFrame,
    full_model_results: pd.DataFrame,
) -> pd.DataFrame:
    if leave_one_fold_out.empty or not _rq3_model_is_populated(full_model_results):
        return pd.DataFrame()
    full_coefficients = full_model_results.set_index("term")["coefficient"]
    summary = (
        leave_one_fold_out.groupby("term", observed=True)
        .agg(
            leave_one_fold_count=("omitted_fold_id", "nunique"),
            min_coefficient=("coefficient", "min"),
            max_coefficient=("coefficient", "max"),
            sign_change_count=("sign_changed_vs_full", "sum"),
            max_absolute_shift=("coefficient_shift", lambda s: float(np.abs(s).max())),
        )
        .reset_index()
    )
    summary["full_model_coefficient"] = summary["term"].map(full_coefficients).astype(float)
    summary["any_sign_change"] = summary["sign_change_count"].gt(0)
    return summary


def _build_rq3_design_matrix(
    working: pd.DataFrame,
    liquidity_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    design_working = working.reset_index(drop=True).copy()
    design_working["financials_exposure"] = design_working["sector_bucket"].eq(
        "Financials"
    ).astype(float)
    design_working["resources_exposure"] = design_working["sector_bucket"].eq(
        "Resources"
    ).astype(float)

    liquidity_categories = pd.Categorical(
        design_working[liquidity_col],
        categories=["high", "medium", "lower"],
        ordered=True,
    )
    liquidity_dummies = pd.get_dummies(
        liquidity_categories,
        prefix="liquidity",
        drop_first=True,
    ).astype(float)
    for column in ("liquidity_medium", "liquidity_lower"):
        if column not in liquidity_dummies.columns:
            liquidity_dummies[column] = 0.0

    strategy_categories = pd.Categorical(
        design_working["strategy_family"],
        categories=["mean_reversion", "trend_following"],
        ordered=False,
    )
    strategy_dummies = pd.get_dummies(
        strategy_categories,
        prefix="strategy",
        drop_first=True,
    ).astype(float)
    if "strategy_trend_following" not in strategy_dummies.columns:
        strategy_dummies["strategy_trend_following"] = 0.0

    design = pd.concat(
        [
            pd.Series(1.0, index=design_working.index, name="intercept"),
            design_working[["financials_exposure", "resources_exposure"]].astype(float),
            liquidity_dummies[["liquidity_medium", "liquidity_lower"]],
            strategy_dummies[["strategy_trend_following"]],
        ],
        axis=1,
    )
    return design, design_working


def _rq3_unavailable_frame(
    term: str,
    sample_size: int,
    adjusted_r_squared: float,
    limitation_note: str,
    covariance_design: str,
    parameter_count: int,
    ticker_cluster_count: int = 0,
    fold_cluster_count: int = 0,
    intersection_cluster_count: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term": term,
                "coefficient": np.nan,
                "standard_error": np.nan,
                "test_statistic": np.nan,
                "p_value": np.nan,
                "lower_ci_95": np.nan,
                "upper_ci_95": np.nan,
                "sample_size": sample_size,
                "ticker_cluster_count": ticker_cluster_count,
                "fold_cluster_count": fold_cluster_count,
                "intersection_cluster_count": intersection_cluster_count,
                "cluster_unit": "unavailable",
                "covariance_design": covariance_design,
                "finite_sample_correction": "not_applicable",
                "parameter_count": parameter_count,
                "inference_design": "cluster_identifier_required",
                "model_type": RQ3_INFERENCE_MODEL_TYPE,
                "adjusted_r_squared": adjusted_r_squared,
                "limitation_note": limitation_note,
            }
        ]
    )


def _two_sided_normal_p_value(test_statistic: float) -> float:
    if np.isnan(test_statistic):
        return np.nan
    tail_probability = 1.0 - 0.5 * (1.0 + erf(abs(float(test_statistic)) / sqrt(2.0)))
    return float(2.0 * tail_probability)
