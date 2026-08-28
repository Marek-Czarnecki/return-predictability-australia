from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd


def make_processed_panel() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    price_map = {
        "AAA": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "BBB": [15.0, 14.0, 13.0, 12.0, 11.0, 10.0],
        "CCC": [8.0, 8.2, 8.1, 8.3, 8.4, 8.5],
    }
    volume_map = {
        "AAA": [100, 110, 120, 130, 140, 150],
        "BBB": [200, 210, 220, 230, 240, 250],
        "CCC": [90, 95, 100, 105, 110, 115],
    }

    rows = []
    for ticker_code, prices in price_map.items():
        for trade_date, adj_close, volume in zip(dates, prices, volume_map[ticker_code]):
            rows.append(
                {
                    "ticker_code": ticker_code,
                    "trade_date": trade_date,
                    "adj_close": adj_close,
                    "volume": volume,
                }
            )

    panel = pd.DataFrame(rows).sort_values(["ticker_code", "trade_date"]).reset_index(
        drop=True
    )
    panel["daily_return"] = panel.groupby("ticker_code", observed=True)[
        "adj_close"
    ].pct_change()
    panel["dollar_volume"] = panel["adj_close"] * panel["volume"]
    return panel


def make_raw_panel_for_parquet() -> pd.DataFrame:
    panel = make_processed_panel().copy()
    panel["dataset_id"] = "market_ohlcv_daily_v2"
    panel["exchange"] = "ASX"
    panel["currency"] = "AUD"
    panel["ticker"] = panel["ticker_code"]
    panel["vendor_symbol"] = panel["ticker_code"] + ".AX"
    panel["adj close"] = panel["adj_close"]
    panel["open"] = panel["adj_close"] * 0.99
    panel["high"] = panel["adj_close"] * 1.01
    panel["low"] = panel["adj_close"] * 0.98
    panel["close"] = panel["adj_close"]
    panel["source_file"] = "synthetic.csv"
    panel["ohlc_valid_flag"] = True
    panel["ohlc_issue_type"] = "none"
    panel["open_clean"] = panel["open"]
    panel["close_clean"] = panel["close"]
    panel["high_clean"] = panel["high"]
    panel["low_clean"] = panel["low"]
    ordered_columns = [
        "dataset_id",
        "exchange",
        "currency",
        "ticker",
        "vendor_symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "adj close",
        "volume",
        "ticker_code",
        "source_file",
        "ohlc_valid_flag",
        "ohlc_issue_type",
        "open_clean",
        "close_clean",
        "high_clean",
        "low_clean",
    ]
    return panel[ordered_columns]


def make_publication_raw_panel() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "asset_id": 1001,
                "ticker_code": "AAA",
                "vendor_symbol": "AAA.au",
                "security_name": "Asset AAA",
                "source_database": "AU Equities",
                "delisted_flag": False,
                "trade_date": "2000-03-31",
                "member_of_universe": 1,
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.0,
                "unadjusted_close": 12.0,
                "dividend": 0.0,
                "volume": 100.0,
                "turnover": 1_000.0,
            },
            {
                "asset_id": 1002,
                "ticker_code": "AAA",
                "vendor_symbol": "AAA.au",
                "security_name": "Asset AAA Replacement",
                "source_database": "AU Equities",
                "delisted_flag": True,
                "trade_date": "2000-03-31",
                "member_of_universe": 0,
                "open": 20.0,
                "high": 20.5,
                "low": 19.8,
                "close": 20.0,
                "unadjusted_close": 21.0,
                "dividend": 0.0,
                "volume": 200.0,
                "turnover": 2_000.0,
            },
            {
                "asset_id": 1001,
                "ticker_code": "AAA",
                "vendor_symbol": "AAA.au",
                "security_name": "Asset AAA",
                "source_database": "AU Equities",
                "delisted_flag": False,
                "trade_date": "2026-08-27",
                "member_of_universe": 1,
                "open": 11.0,
                "high": 11.1,
                "low": 10.7,
                "close": 11.0,
                "unadjusted_close": 13.2,
                "dividend": 0.0,
                "volume": 110.0,
                "turnover": 1_210.0,
            },
            {
                "asset_id": 1002,
                "ticker_code": "AAA",
                "vendor_symbol": "AAA.au",
                "security_name": "Asset AAA Replacement",
                "source_database": "AU Equities",
                "delisted_flag": True,
                "trade_date": "2026-08-27",
                "member_of_universe": 0,
                "open": 18.0,
                "high": 18.3,
                "low": 17.7,
                "close": 18.0,
                "unadjusted_close": 18.9,
                "dividend": 0.0,
                "volume": 210.0,
                "turnover": 3_780.0,
            },
        ]
    )
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["asset_id"] = raw["asset_id"].astype("int64")
    raw["ticker_code"] = raw["ticker_code"].astype("string")
    raw["vendor_symbol"] = raw["vendor_symbol"].astype("string")
    raw["security_name"] = raw["security_name"].astype("string")
    raw["source_database"] = raw["source_database"].astype("string")
    raw["member_of_universe"] = raw["member_of_universe"].astype("int8")
    raw["delisted_flag"] = raw["delisted_flag"].astype(bool)
    return raw


def make_mean_reversion_panel() -> pd.DataFrame:
    dates = pd.date_range("2026-02-01", periods=6, freq="D")
    price_map = {
        "AAA": [10.0, 10.0, 9.0, 8.0, 9.0, 10.0],
        "BBB": [10.0, 10.1, 10.0, 10.1, 10.0, 10.1],
        "CCC": [9.0, 9.2, 9.4, 9.3, 9.2, 9.1],
    }
    rows = []
    for ticker_code, prices in price_map.items():
        for idx, (trade_date, adj_close) in enumerate(zip(dates, prices), start=1):
            rows.append(
                {
                    "ticker_code": ticker_code,
                    "trade_date": trade_date,
                    "adj_close": adj_close,
                    "volume": 100 + idx,
                }
            )
    panel = pd.DataFrame(rows).sort_values(["ticker_code", "trade_date"]).reset_index(
        drop=True
    )
    panel["daily_return"] = panel.groupby("ticker_code", observed=True)[
        "adj_close"
    ].pct_change()
    panel["dollar_volume"] = panel["adj_close"] * panel["volume"]
    return panel


def make_pairs_panel() -> pd.DataFrame:
    dates = pd.date_range("2026-03-01", periods=24, freq="D")
    price_map = {
        "AAA": [100.0 + idx * 1.5 for idx in range(24)],
        "AAB": [80.0 + idx * 1.2 for idx in range(24)],
        "BBC": [60.0 + idx * 0.6 + (0.8 if idx % 2 == 0 else -0.4) for idx in range(24)],
        "BBD": [50.0 + idx * 0.5 + (0.6 if idx % 3 == 0 else -0.2) for idx in range(24)],
        "CCE": [40.0 + idx * 0.9 + (0.5 if idx % 4 == 0 else -0.1) for idx in range(24)],
    }
    volume_map = {"AAA": 1400, "AAB": 1350, "BBC": 900, "BBD": 850, "CCE": 800}
    rows = []
    for ticker_code, prices in price_map.items():
        for idx, (trade_date, adj_close) in enumerate(zip(dates, prices), start=1):
            rows.append(
                {
                    "ticker_code": ticker_code,
                    "trade_date": trade_date,
                    "adj_close": adj_close,
                    "volume": volume_map[ticker_code] + idx * 5,
                }
            )
    panel = pd.DataFrame(rows).sort_values(["ticker_code", "trade_date"]).reset_index(
        drop=True
    )
    panel["daily_return"] = panel.groupby("ticker_code", observed=True)[
        "adj_close"
    ].pct_change()
    panel["dollar_volume"] = panel["adj_close"] * panel["volume"]
    return panel


def make_pairs_sector_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker_code": "AAA", "sector": "Financials"},
            {"ticker_code": "AAB", "sector": "Financials"},
            {"ticker_code": "BBC", "sector": "Materials"},
            {"ticker_code": "BBD", "sector": "Materials"},
            {"ticker_code": "CCE", "sector": "Industrials"},
        ]
    )


def make_tax_loss_panel() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2025-06-24",
            "2025-06-25",
            "2025-06-26",
            "2025-06-27",
            "2025-06-30",
            "2025-07-01",
            "2025-07-02",
            "2025-07-03",
        ]
    )
    price_map = {
        "AAA": [10.0, 9.0, 8.0, 7.0, 6.5, 7.0, 7.3, 7.5],
        "BBB": [10.0, 10.2, 10.4, 10.5, 10.7, 10.8, 10.9, 11.0],
        "CCC": [8.0, 8.0, 8.1, 8.1, 8.2, 8.2, 8.3, 8.3],
        "DDD": [12.0, 11.9, 11.8, 11.7, 11.6, 11.7, 11.8, 11.9],
        "EEE": [7.0, 7.1, 7.2, 7.1, 7.2, 7.3, 7.4, 7.5],
    }
    rows = []
    for ticker_code, prices in price_map.items():
        for idx, (trade_date, adj_close) in enumerate(zip(dates, prices), start=1):
            rows.append(
                {
                    "ticker_code": ticker_code,
                    "trade_date": trade_date,
                    "adj_close": adj_close,
                    "volume": 1000 + idx,
                }
            )
    panel = pd.DataFrame(rows).sort_values(["ticker_code", "trade_date"]).reset_index(
        drop=True
    )
    panel["daily_return"] = panel.groupby("ticker_code", observed=True)[
        "adj_close"
    ].pct_change()
    panel["dollar_volume"] = panel["adj_close"] * panel["volume"]
    return panel


def make_temp_project_root() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temp_dir = tempfile.TemporaryDirectory()
    root = Path(temp_dir.name)
    (root / "data" / "licensed").mkdir(parents=True)
    (root / "data" / "generated" / "publication_results").mkdir(parents=True)
    return temp_dir, root
