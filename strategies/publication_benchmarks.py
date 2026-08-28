from __future__ import annotations

import pandas as pd

from .metrics import compute_nav


XJOA_ASSET_ID = 203461
XJOA_SYMBOL = "$XJOA.au"


def build_external_benchmark_returns(
    frame: pd.DataFrame,
    *,
    expected_asset_id: int = XJOA_ASSET_ID,
    expected_symbol: str = XJOA_SYMBOL,
    validate_metadata: bool = True,
) -> pd.DataFrame:
    """Build close-to-close returns from an official benchmark index level series."""
    required_columns = {"trade_date", "close"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "External benchmark input is missing required columns: " f"{missing_list}"
        )
    benchmark = frame.copy()
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"], errors="coerce")
    if benchmark["trade_date"].isna().any():
        raise ValueError("External benchmark input contains invalid trade_date values.")
    benchmark = benchmark.sort_values("trade_date").reset_index(drop=True)
    if benchmark["trade_date"].duplicated().any():
        raise ValueError("External benchmark input contains duplicate trade dates.")
    close = pd.to_numeric(benchmark["close"], errors="coerce")
    if close.isna().any():
        raise ValueError("External benchmark input contains null or non-numeric close values.")
    if (close <= 0).any():
        raise ValueError("External benchmark input contains non-positive close values.")
    if validate_metadata:
        if "asset_id" in benchmark.columns:
            asset_ids = pd.to_numeric(benchmark["asset_id"], errors="coerce")
            if asset_ids.isna().any() or not asset_ids.eq(expected_asset_id).all():
                raise ValueError(
                    "External benchmark asset_id does not match expected publication benchmark "
                    f"asset_id {expected_asset_id}."
                )
        if "symbol" in benchmark.columns:
            symbols = benchmark["symbol"].astype("string")
            if symbols.isna().any() or not symbols.eq(expected_symbol).all():
                raise ValueError(
                    "External benchmark symbol does not match expected publication benchmark "
                    f"symbol {expected_symbol}."
                )
    result = pd.DataFrame(
        {"trade_date": benchmark["trade_date"], "benchmark_level": close.astype(float)}
    )
    result["benchmark_return"] = result["benchmark_level"].pct_change(fill_method=None)
    return result


def build_point_in_time_equal_weight_benchmark(
    frame: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    identity_col: str = "asset_id",
    return_col: str = "daily_return",
    membership_col: str = "member_of_universe",
) -> pd.DataFrame:
    """Build the publication point-in-time equal-weight ASX 200 benchmark."""
    required_columns = {date_col, identity_col, return_col, membership_col}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Publication benchmark input is missing required columns: " f"{missing_list}"
        )
    panel = frame[[date_col, identity_col, return_col, membership_col]].copy()
    panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
    if panel[date_col].isna().any():
        raise ValueError("Publication benchmark input contains invalid trade_date values.")
    if panel[identity_col].isna().any():
        raise ValueError("Publication benchmark input contains null asset identities.")
    if panel.duplicated([identity_col, date_col]).any():
        raise ValueError("Publication benchmark input contains duplicate identity/trade_date rows.")
    original_returns = panel[return_col]
    numeric_returns = pd.to_numeric(original_returns, errors="coerce")
    invalid_returns = original_returns.notna() & numeric_returns.isna()
    if invalid_returns.any():
        raise ValueError("Publication benchmark input contains non-numeric daily_return values.")
    panel[return_col] = numeric_returns.astype(float)
    membership = panel[membership_col]
    if membership.isna().any():
        raise ValueError("Publication benchmark input contains null membership values.")
    valid_membership = membership.isin([True, False, 1, 0])
    if not valid_membership.all():
        raise ValueError("Publication benchmark membership values must be boolean or 0/1.")
    panel[membership_col] = membership.astype(bool)
    panel = panel.sort_values([date_col, identity_col]).reset_index(drop=True)
    market_dates = pd.Index(panel[date_col].drop_duplicates().sort_values(), name=date_col)
    result = pd.DataFrame({date_col: market_dates})
    if len(market_dates) == 0:
        result["equal_weight_return"] = pd.Series(dtype=float)
        result["equal_weight_nav"] = pd.Series(dtype=float)
        result["member_count"] = pd.Series(dtype="int64")
        result["observable_member_return_count"] = pd.Series(dtype="int64")
        result["missing_member_return_count"] = pd.Series(dtype="int64")
        result["missing_member_return_fraction"] = pd.Series(dtype=float)
        return result
    next_session = dict(zip(market_dates[:-1], market_dates[1:]))
    expected_holdings = panel.loc[panel[membership_col], [date_col, identity_col]].copy()
    expected_holdings[date_col] = expected_holdings[date_col].map(next_session)
    expected_holdings = expected_holdings.dropna(subset=[date_col])
    current_returns = panel[[date_col, identity_col, return_col]]
    holding_returns = expected_holdings.merge(
        current_returns, on=[date_col, identity_col], how="left", validate="one_to_one"
    )
    if holding_returns.empty:
        diagnostics = pd.DataFrame(index=market_dates)
    else:
        diagnostics = holding_returns.groupby(date_col, observed=True).agg(
            equal_weight_return=(return_col, "mean"),
            member_count=(identity_col, "size"),
            observable_member_return_count=(return_col, "count"),
        )
        diagnostics["missing_member_return_count"] = (
            diagnostics["member_count"] - diagnostics["observable_member_return_count"]
        )
        diagnostics["missing_member_return_fraction"] = (
            diagnostics["missing_member_return_count"] / diagnostics["member_count"]
        )
    result = result.set_index(date_col)
    result = result.join(diagnostics, how="left")
    for count_col in (
        "member_count", "observable_member_return_count", "missing_member_return_count"
    ):
        if count_col not in result.columns:
            result[count_col] = 0
        result[count_col] = result[count_col].fillna(0).astype(int)
    if "equal_weight_return" not in result.columns:
        result["equal_weight_return"] = float("nan")
    if "missing_member_return_fraction" not in result.columns:
        result["missing_member_return_fraction"] = float("nan")
    no_expected_members = result["member_count"].eq(0)
    result.loc[no_expected_members, "missing_member_return_fraction"] = float("nan")
    result["equal_weight_nav"] = compute_nav(result["equal_weight_return"])
    return result.reset_index()[[
        date_col, "equal_weight_return", "equal_weight_nav", "member_count",
        "observable_member_return_count", "missing_member_return_count",
        "missing_member_return_fraction",
    ]]
