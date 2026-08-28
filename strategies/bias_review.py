from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from .sector_liquidity import CURRENT_ONLY_SECTOR_LIMITATION_NOTE


CURRENT_UNIVERSE_TREATMENT = "current_constituent_retrospective"
CURRENT_UNIVERSE_LIMITATION_NOTE = (
    "Universe membership is treated as current-constituent retrospective; "
    "no point-in-time historical membership or delisting archive is modeled."
)
DATE_SPECIFIC_ELIGIBILITY_NOTE = (
    "Eligibility is date-specific: a ticker becomes eligible only after accumulating "
    "the required observed history in the full cleaned panel, not separately within "
    "each reporting or walk-forward window."
)


def apply_date_specific_minimum_history_rule(
    prices: pd.DataFrame,
    min_history: int,
    ticker_col: str = "ticker_code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    if min_history <= 0:
        raise ValueError("min_history must be positive.")

    required_columns = {ticker_col, date_col}
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Price panel is missing required columns: {missing_list}")

    ordered = prices.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values([ticker_col, date_col]).reset_index(drop=True)
    ordered["history_count"] = (
        ordered.groupby(ticker_col, observed=True).cumcount() + 1
    )
    ordered["minimum_history_required"] = int(min_history)
    ordered["history_shortfall"] = (
        min_history - ordered["history_count"]
    ).clip(lower=0)
    ordered["eligible_flag"] = ordered["history_count"] >= min_history
    ordered["first_trade_date"] = ordered.groupby(ticker_col, observed=True)[
        date_col
    ].transform("min")

    first_eligible_dates = (
        ordered.loc[ordered["eligible_flag"], [ticker_col, date_col]]
        .groupby(ticker_col, observed=True)[date_col]
        .min()
    )
    ordered["first_eligible_trade_date"] = ordered[ticker_col].map(first_eligible_dates)
    return ordered


def build_eligibility_review(
    prices: pd.DataFrame,
    strategy_name: str,
    min_history: int,
    ticker_col: str = "ticker_code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    annotated = apply_date_specific_minimum_history_rule(
        prices,
        min_history=min_history,
        ticker_col=ticker_col,
        date_col=date_col,
    )
    annotated["calendar_year"] = annotated[date_col].dt.year
    first_eligible = (
        annotated.loc[annotated["eligible_flag"], [ticker_col, "first_eligible_trade_date"]]
        .drop_duplicates()
        .assign(calendar_year=lambda frame: frame["first_eligible_trade_date"].dt.year)
    )
    newly_eligible = (
        first_eligible.groupby("calendar_year", observed=True)[ticker_col]
        .nunique()
        .rename("newly_eligible_tickers")
    )

    review = (
        annotated.groupby("calendar_year", observed=True)
        .agg(
            total_name_dates=(ticker_col, "size"),
            eligible_name_dates=("eligible_flag", "sum"),
            total_tickers_seen=(ticker_col, "nunique"),
            eligible_tickers_seen=(
                ticker_col,
                lambda values: int(
                    annotated.loc[values.index, [ticker_col, "eligible_flag"]]
                    .drop_duplicates()
                    .loc[lambda frame: frame["eligible_flag"], ticker_col]
                    .nunique()
                ),
            ),
            median_history_count=("history_count", "median"),
        )
        .reset_index()
    )
    review["strategy_name"] = strategy_name
    review["min_history"] = int(min_history)
    review["eligible_name_date_share"] = (
        review["eligible_name_dates"] / review["total_name_dates"]
    ).fillna(0.0)
    review["newly_eligible_tickers"] = (
        review["calendar_year"].map(newly_eligible).fillna(0).astype(int)
    )
    review["eligibility_rule_note"] = DATE_SPECIFIC_ELIGIBILITY_NOTE
    return review[
        [
            "strategy_name",
            "min_history",
            "calendar_year",
            "total_name_dates",
            "eligible_name_dates",
            "eligible_name_date_share",
            "total_tickers_seen",
            "eligible_tickers_seen",
            "newly_eligible_tickers",
            "median_history_count",
            "eligibility_rule_note",
        ]
    ]


def build_survivorship_bias_summary(
    prices: pd.DataFrame,
    ticker_col: str = "ticker_code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    required_columns = {ticker_col, date_col}
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Price panel is missing required columns: {missing_list}")

    working = prices.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    sample_start = pd.Timestamp(working[date_col].min())
    sample_end = pd.Timestamp(working[date_col].max())
    ticker_dates = (
        working.groupby(ticker_col, observed=True)[date_col]
        .agg(first_trade_date="min", last_trade_date="max")
        .reset_index()
    )
    total_tickers = int(len(ticker_dates))
    starting_tickers = int((ticker_dates["first_trade_date"] == sample_start).sum())
    ending_tickers = int((ticker_dates["last_trade_date"] == sample_end).sum())
    later_entry_tickers = int((ticker_dates["first_trade_date"] > sample_start).sum())
    full_history_tickers = int(
        (
            (ticker_dates["first_trade_date"] == sample_start)
            & (ticker_dates["last_trade_date"] == sample_end)
        ).sum()
    )
    ticker_lookup = ticker_dates.set_index(ticker_col)
    working["later_entry_flag"] = working[ticker_col].map(
        ticker_lookup["first_trade_date"] > sample_start
    )

    summary = pd.DataFrame(
        [
            {
                "sample_start": sample_start,
                "sample_end": sample_end,
                "total_tickers": total_tickers,
                "starting_tickers": starting_tickers,
                "ending_tickers": ending_tickers,
                "later_entry_tickers": later_entry_tickers,
                "later_entry_share_tickers": (
                    later_entry_tickers / total_tickers if total_tickers else np.nan
                ),
                "later_entry_observation_share": float(
                    working["later_entry_flag"].mean()
                )
                if not working.empty
                else np.nan,
                "full_history_tickers": full_history_tickers,
                "full_history_share_tickers": (
                    full_history_tickers / total_tickers if total_tickers else np.nan
                ),
                "universe_treatment": CURRENT_UNIVERSE_TREATMENT,
                "sector_classification_treatment": "current_only",
                "limitation_note": (
                    f"{CURRENT_UNIVERSE_LIMITATION_NOTE} "
                    f"{CURRENT_ONLY_SECTOR_LIMITATION_NOTE}"
                ),
                "interpretation_consequence": (
                    "Final claims should be read as conditional on the surviving current "
                    "universe and current-only sector labels, not as point-in-time index "
                    "membership evidence."
                ),
            }
        ]
    )
    return summary


def build_concentration_review(
    ticker_outcomes: pd.DataFrame | None = None,
    fold_summary: pd.DataFrame | None = None,
    event_study: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if ticker_outcomes is not None and not ticker_outcomes.empty:
        frames.append(
            _summarize_concentration(
                ticker_outcomes,
                analysis_key_col="strategy_family",
                entity_col="ticker_code",
                value_col="cumulative_net_contribution",
                concentration_dimension="names",
                source_artifact="sector_liquidity_ticker_table.csv",
            )
        )
    if fold_summary is not None and not fold_summary.empty:
        working = fold_summary.copy()
        if "window_label" in working.columns:
            working = working.loc[
                working["window_label"].astype("string").eq("evaluation")
            ].copy()
        if not working.empty:
            frames.append(
                _summarize_concentration(
                    working,
                    analysis_key_col="strategy",
                    entity_col="fold_id",
                    value_col="total_net_excess_return",
                    concentration_dimension="folds",
                    source_artifact="*_walk_forward_summary.csv",
                )
            )
    if event_study is not None and not event_study.empty:
        yearly = (
            event_study.groupby("year", observed=True)["return_difference"]
            .sum()
            .rename("yearly_return_difference")
            .reset_index()
        )
        yearly["analysis_key"] = "tax_loss_selling"
        frames.append(
            _summarize_concentration(
                yearly,
                analysis_key_col="analysis_key",
                entity_col="year",
                value_col="yearly_return_difference",
                concentration_dimension="years",
                source_artifact="tax_loss_selling_event_study.csv",
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "analysis_key",
                "concentration_dimension",
                "entity_count",
                "top_entity",
                "top_entity_absolute_share",
                "top_3_absolute_share",
                "effective_entity_count",
                "concentration_flag",
                "interpretation_consequence",
                "source_artifact",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_bias_review_summary(
    prices: pd.DataFrame,
    min_history_by_strategy: dict[str, int] | None = None,
    ticker_outcomes: pd.DataFrame | None = None,
    fold_summary: pd.DataFrame | None = None,
    event_study: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "review_component": "survivorship_current_universe",
            "status": "implemented",
            "evidence_artifact": "bias_review_survivorship.csv",
            "key_result": CURRENT_UNIVERSE_TREATMENT,
            "limitation_note": CURRENT_UNIVERSE_LIMITATION_NOTE,
            "interpretation_consequence": (
                "Confirmatory and descriptive results remain conditional on the current "
                "constituent retrospective universe."
            ),
        }
    )
    if min_history_by_strategy:
        for strategy_name, min_history in sorted(min_history_by_strategy.items()):
            rows.append(
                {
                    "review_component": f"eligibility_{strategy_name}",
                    "status": "implemented",
                    "evidence_artifact": "bias_review_eligibility.csv",
                    "key_result": f"min_history={min_history}",
                    "limitation_note": DATE_SPECIFIC_ELIGIBILITY_NOTE,
                    "interpretation_consequence": (
                        "Early sample windows before a ticker accumulates the required "
                        "history are excluded from investable interpretation."
                    ),
                }
            )

    concentration = build_concentration_review(
        ticker_outcomes=ticker_outcomes,
        fold_summary=fold_summary,
        event_study=event_study,
    )
    if concentration.empty:
        rows.append(
            {
                "review_component": "concentration_review",
                "status": "pending_artifact",
                "evidence_artifact": "bias_review_concentration.csv",
                "key_result": "No name/year/fold inputs supplied.",
                "limitation_note": (
                    "Concentration review requires ticker-, year-, or fold-level result "
                    "artifacts."
                ),
                "interpretation_consequence": (
                    "Do not describe robustness across names, years, or folds until the "
                    "concentration artifact is populated."
                ),
            }
        )
    else:
        for record in concentration.to_dict("records"):
            rows.append(
                {
                    "review_component": (
                        f"concentration_{record['analysis_key']}_"
                        f"{record['concentration_dimension']}"
                    ),
                    "status": "implemented",
                    "evidence_artifact": "bias_review_concentration.csv",
                    "key_result": (
                        f"top_entity_share={record['top_entity_absolute_share']:.3f}; "
                        f"top_3_share={record['top_3_absolute_share']:.3f}; "
                        f"flag={record['concentration_flag']}"
                    ),
                    "limitation_note": (
                        "Concentration diagnostics are descriptive and indicate dependence "
                        "on a small subset of names, years, or folds when flagged."
                    ),
                    "interpretation_consequence": record["interpretation_consequence"],
                }
            )

    survivorship = build_survivorship_bias_summary(prices)
    if not survivorship.empty:
        rows[0]["key_result"] = (
            f"later_entry_share_tickers="
            f"{survivorship.loc[0, 'later_entry_share_tickers']:.3f}; "
            f"full_history_share_tickers="
            f"{survivorship.loc[0, 'full_history_share_tickers']:.3f}"
        )
    return pd.DataFrame(rows)


def _summarize_concentration(
    frame: pd.DataFrame,
    analysis_key_col: str,
    entity_col: str,
    value_col: str,
    concentration_dimension: str,
    source_artifact: str,
) -> pd.DataFrame:
    required_columns = {analysis_key_col, entity_col, value_col}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Concentration input is missing required columns: {missing_list}")

    rows: list[dict[str, object]] = []
    for analysis_key, subset in frame.groupby(analysis_key_col, observed=True):
        values = pd.to_numeric(subset[value_col], errors="coerce").dropna()
        entity_lookup = subset.loc[values.index, entity_col].astype("string").reset_index(
            drop=True
        )
        abs_values = values.abs().reset_index(drop=True)
        total_abs = float(abs_values.sum())
        if total_abs == 0.0 or abs_values.empty:
            top_entity_share = 0.0
            top_3_share = 0.0
            top_entity = pd.NA
            effective_entity_count = np.nan
            concentration_flag = "no_signal"
        else:
            ordering = abs_values.sort_values(ascending=False)
            top_entity_idx = int(ordering.index[0])
            top_entity = entity_lookup.iloc[top_entity_idx]
            shares = ordering / total_abs
            top_entity_share = float(shares.iloc[0])
            top_3_share = float(shares.head(3).sum())
            effective_entity_count = float(1.0 / np.square(shares).sum())
            concentration_flag = _classify_concentration(
                top_entity_share, top_3_share, effective_entity_count
            )
        rows.append(
            {
                "analysis_key": str(analysis_key),
                "concentration_dimension": concentration_dimension,
                "entity_count": int(len(abs_values)),
                "top_entity": top_entity,
                "top_entity_absolute_share": top_entity_share,
                "top_3_absolute_share": top_3_share,
                "effective_entity_count": effective_entity_count,
                "concentration_flag": concentration_flag,
                "interpretation_consequence": _interpret_concentration(
                    concentration_dimension, concentration_flag
                ),
                "source_artifact": source_artifact,
            }
        )
    return pd.DataFrame(rows)


def _classify_concentration(
    top_entity_share: float,
    top_3_share: float,
    effective_entity_count: float,
) -> str:
    if (
        top_entity_share >= 0.50
        or top_3_share >= 0.80
        or (
            not np.isnan(effective_entity_count)
            and effective_entity_count <= sqrt(3.0)
        )
    ):
        return "high"
    if (
        top_entity_share >= 0.35
        or top_3_share >= 0.65
        or (
            not np.isnan(effective_entity_count)
            and effective_entity_count <= 3.0
        )
    ):
        return "moderate"
    return "diffuse"


def _interpret_concentration(
    concentration_dimension: str, concentration_flag: str
) -> str:
    if concentration_flag == "high":
        return (
            f"Results are materially concentrated across {concentration_dimension}; "
            "final claims should avoid broad robustness language."
        )
    if concentration_flag == "moderate":
        return (
            f"Results show meaningful dependence on a subset of {concentration_dimension}; "
            "final claims should emphasize partial rather than uniform robustness."
        )
    if concentration_flag == "no_signal":
        return "No measurable contribution signal was available for this concentration review."
    return (
        f"No dominant concentration across {concentration_dimension} was detected in the "
        "available artifact."
    )
