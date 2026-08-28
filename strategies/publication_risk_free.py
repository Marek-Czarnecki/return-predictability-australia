from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import Request, urlopen
import csv

import numpy as np
import pandas as pd


RBA_CASH_RATE_URL = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
RBA_HISTORICAL_CASH_RATE_URL = (
    "https://www.rba.gov.au/statistics/tables/xls-hist/f01dhist.xls"
)
RBA_CASH_RATE_SERIES_ID = "FIRMMCRTD"
PUBLICATION_RISK_FREE_LABEL = "rba_cash_rate_target_calendar_day_accrual"


@dataclass(frozen=True)
class PublicationRiskFreeResult:
    returns: pd.DataFrame
    overlap_validation: pd.DataFrame


def load_rba_cash_rate_schedule(
    current_url: str = RBA_CASH_RATE_URL,
    historical_url: str = RBA_HISTORICAL_CASH_RATE_URL,
) -> pd.DataFrame:
    current_request = Request(current_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(current_request, timeout=30) as response:
        current_text = response.read().decode("utf-8-sig")
    current = parse_rba_f1_cash_rate_csv(current_text)

    historical_request = Request(historical_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(historical_request, timeout=30) as response:
        historical_bytes = response.read()
    historical = parse_rba_f1_cash_rate_xls(historical_bytes)

    combined = pd.concat([historical, current], ignore_index=True)
    combined = combined.drop_duplicates("effective_date", keep="last").sort_values("effective_date").reset_index(drop=True)
    changed = combined["cash_rate_target_percent"].ne(combined["cash_rate_target_percent"].shift(1))
    return combined.loc[changed].reset_index(drop=True)


def parse_rba_f1_cash_rate_csv(text: str) -> pd.DataFrame:
    rows = list(csv.reader(StringIO(text)))
    if not rows:
        raise ValueError("RBA F1 CSV is empty.")
    series_row_index = None
    series_col_index = None
    for row_index, row in enumerate(rows):
        if not row:
            continue
        if str(row[0]).strip().lower() == "series id":
            series_row_index = row_index
            for column_index, value in enumerate(row):
                if str(value).strip() == RBA_CASH_RATE_SERIES_ID:
                    series_col_index = column_index
                    break
            break
    if series_row_index is None:
        raise ValueError("RBA F1 CSV does not contain a Series ID row.")
    if series_col_index is None:
        raise ValueError(f"RBA F1 CSV does not contain cash-rate target series {RBA_CASH_RATE_SERIES_ID}.")
    observations = []
    for row in rows[series_row_index + 1:]:
        if len(row) <= series_col_index:
            continue
        trade_date = pd.to_datetime(row[0], errors="coerce", dayfirst=True)
        rate = pd.to_numeric(row[series_col_index], errors="coerce")
        if pd.isna(trade_date) or pd.isna(rate):
            continue
        observations.append({"effective_date": pd.Timestamp(trade_date), "cash_rate_target_percent": float(rate)})
    result = pd.DataFrame(observations)
    if result.empty:
        raise ValueError("No valid cash-rate target observations were found in RBA F1 CSV.")
    return _collapse_daily_target_observations(result)


def parse_rba_f1_cash_rate_xls(content: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_excel(BytesIO(content), header=None, engine="xlrd")
    except ImportError as exc:
        raise ImportError("Reading the official RBA historical F1 .xls file requires xlrd. Install it with: python -m pip install xlrd") from exc
    matches = np.argwhere(raw.astype(str).to_numpy() == RBA_CASH_RATE_SERIES_ID)
    if len(matches) == 0:
        raise ValueError(f"RBA historical F1 workbook does not contain {RBA_CASH_RATE_SERIES_ID}.")
    series_row_index, series_col_index = (int(value) for value in matches[0])
    observations = []
    for row_index in range(series_row_index + 1, len(raw)):
        trade_date = pd.to_datetime(raw.iat[row_index, 0], errors="coerce", dayfirst=True)
        rate = pd.to_numeric(raw.iat[row_index, series_col_index], errors="coerce")
        if pd.isna(trade_date) or pd.isna(rate):
            continue
        observations.append({"effective_date": pd.Timestamp(trade_date), "cash_rate_target_percent": float(rate)})
    result = pd.DataFrame(observations)
    if result.empty:
        raise ValueError("No valid cash-rate target observations were found in RBA historical F1 workbook.")
    return _collapse_daily_target_observations(result)


def _collapse_daily_target_observations(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("effective_date").drop_duplicates("effective_date", keep="last").reset_index(drop=True)
    changed = result["cash_rate_target_percent"].ne(result["cash_rate_target_percent"].shift(1))
    return result.loc[changed].reset_index(drop=True)


def normalize_cash_rate_schedule(frame: pd.DataFrame) -> pd.DataFrame:
    if {"effective_date", "cash_rate_target_percent"}.issubset(frame.columns):
        result = frame.loc[:, ["effective_date", "cash_rate_target_percent"]].copy()
    else:
        def _flatten(column: object) -> str:
            if isinstance(column, tuple):
                return " ".join(str(part) for part in column if str(part) != "nan").strip()
            return str(column).strip()
        columns = {_flatten(col).lower(): col for col in frame.columns}
        date_col = next((original for key, original in columns.items() if "effective date" in key), None)
        rate_col = next((original for key, original in columns.items() if "cash rate target" in key and "%" in key), None)
        if rate_col is None:
            rate_col = next((original for key, original in columns.items() if "cash rate target" in key), None)
        if date_col is None or rate_col is None:
            raise ValueError("RBA cash-rate table does not contain effective date and cash-rate target columns.")
        result = frame.loc[:, [date_col, rate_col]].copy()
        result.columns = ["effective_date", "cash_rate_target_percent"]
    result["effective_date"] = pd.to_datetime(result["effective_date"], errors="coerce", dayfirst=True)
    result["cash_rate_target_percent"] = pd.to_numeric(result["cash_rate_target_percent"], errors="coerce")
    result = result.dropna(subset=["effective_date", "cash_rate_target_percent"]).drop_duplicates("effective_date", keep="last").sort_values("effective_date").reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid RBA cash-rate target observations were found.")
    return result


def build_publication_risk_free(trading_dates, cash_rate_schedule: pd.DataFrame, *, initial_tri: float = 100.0) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_dates))).dropna().unique().sort_values()
    if dates.empty:
        return pd.DataFrame(columns=["trade_date", "cash_rate_target_percent", "risk_free_return", "rba_cash_rate_tri"])
    schedule = normalize_cash_rate_schedule(cash_rate_schedule)
    if schedule["effective_date"].min() > dates[0]:
        raise ValueError("Cash-rate schedule does not begin before the publication calendar.")
    calendar = pd.date_range(dates[0], dates[-1], freq="D")
    schedule_series = schedule.set_index("effective_date")["cash_rate_target_percent"].sort_index()
    target = schedule_series.reindex(calendar, method="ffill")
    if target.isna().any():
        raise ValueError("Cash-rate target could not be resolved for every calendar day.")
    calendar_factor = 1.0 + (target / 100.0 / 365.0)
    returns = [0.0]
    tri = [float(initial_tri)]
    targets = [float(target.loc[dates[0]])]
    for previous, current in zip(dates[:-1], dates[1:]):
        interval_days = pd.date_range(previous + pd.Timedelta(days=1), current, freq="D")
        interval_return = float(calendar_factor.reindex(interval_days).prod() - 1.0)
        returns.append(interval_return)
        tri.append(tri[-1] * (1.0 + interval_return))
        targets.append(float(target.loc[current]))
    return pd.DataFrame({"trade_date": dates, "cash_rate_target_percent": targets, "risk_free_return": returns, "rba_cash_rate_tri": tri, "source": "Reserve Bank of Australia Statistical Table F1 cash rate target", "source_url": RBA_CASH_RATE_URL, "construction": PUBLICATION_RISK_FREE_LABEL})


def validate_overlap_with_existing_tri(publication_risk_free: pd.DataFrame, existing_tri_path: Path | None) -> pd.DataFrame:
    if existing_tri_path is None or not Path(existing_tri_path).exists():
        return pd.DataFrame([{"overlap_count": 0, "max_abs_return_difference": np.nan, "mean_abs_return_difference": np.nan}])
    existing = pd.read_csv(existing_tri_path, parse_dates=["trade_date"])
    if "risk_free_return" not in existing.columns:
        raise ValueError("Existing RBA TRI file must contain risk_free_return.")
    merged = publication_risk_free.loc[:, ["trade_date", "risk_free_return"]].merge(existing.loc[:, ["trade_date", "risk_free_return"]], on="trade_date", how="inner", suffixes=("_publication", "_existing"))
    if merged.empty:
        return pd.DataFrame([{"overlap_count": 0, "max_abs_return_difference": np.nan, "mean_abs_return_difference": np.nan}])
    difference = (merged["risk_free_return_publication"] - merged["risk_free_return_existing"]).abs()
    return pd.DataFrame([{"overlap_count": int(len(merged)), "max_abs_return_difference": float(difference.max()), "mean_abs_return_difference": float(difference.mean())}])
