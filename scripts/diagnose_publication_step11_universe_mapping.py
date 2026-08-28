from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_data import DEFAULT_PUBLICATION_PANEL_PATH, load_publication_panel

FROZEN_UNIVERSE_REFERENCE_DATE = pd.Timestamp("2026-07-20")
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "data" / "generated" / "publication_results" /
    "publication_step11_trend_universe_mapping_diagnostics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose unresolved frozen-ticker to Norgate asset_id mappings for Step 11.1.2."
    )
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PUBLICATION_PANEL_PATH)
    parser.add_argument(
        "--frozen-config-path", type=Path, required=True,
        help="Path to the frozen original-capstone asx_data_request.json containing ticker_list."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _norm_symbol(value: object) -> str:
    text = str(value).strip().upper()
    for suffix in (".AU", ".AX"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def load_frozen_capstone_tickers(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tickers = payload.get("ticker_list")
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("Frozen config must contain a non-empty ticker_list.")
    return tuple(_norm_symbol(value) for value in tickers)


def map_frozen_tickers_to_asset_ids(
    prices: pd.DataFrame, frozen_tickers: tuple[str, ...]
) -> tuple[dict[str, object], tuple[str, ...]]:
    working = prices.loc[:, ["asset_id", "ticker_code", "vendor_symbol"]].drop_duplicates().copy()
    working["ticker_norm"] = working["ticker_code"].map(_norm_symbol)
    working["vendor_norm"] = working["vendor_symbol"].map(_norm_symbol)

    mapping: dict[str, object] = {}
    unmatched: list[str] = []
    for ticker in frozen_tickers:
        matches = working.loc[
            (working["ticker_norm"] == ticker) | (working["vendor_norm"] == ticker),
            "asset_id",
        ].dropna().unique()
        if len(matches) == 1:
            mapping[ticker] = matches[0]
        else:
            unmatched.append(ticker)
    return mapping, tuple(unmatched)


def build_diagnostics(prices: pd.DataFrame, frozen_tickers: tuple[str, ...]) -> pd.DataFrame:
    mapping, unmatched = map_frozen_tickers_to_asset_ids(prices, frozen_tickers)
    working = prices.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"])
    reference_rows = (
        working.loc[working["trade_date"] <= FROZEN_UNIVERSE_REFERENCE_DATE]
        .sort_values(["asset_id", "trade_date"])
        .groupby("asset_id", observed=True)
        .tail(1)
    )
    reference_members = reference_rows.loc[
        pd.to_numeric(reference_rows["member_of_universe"], errors="coerce").fillna(0).astype(int) == 1
    ].copy()
    matched_asset_ids = set(mapping.values())
    candidate_members = reference_members.loc[~reference_members["asset_id"].isin(matched_asset_ids)].copy()

    rows: list[dict[str, object]] = []
    for ticker in sorted(unmatched):
        prefix_candidates = working.loc[
            working["ticker_code"].astype(str).str.upper().str.startswith(ticker)
            | working["vendor_symbol"].astype(str).str.upper().str.startswith(ticker)
        ].sort_values(["asset_id", "trade_date"])
        for _, row in prefix_candidates.groupby("asset_id", observed=True).tail(1).iterrows():
            rows.append({
                "record_type": "prefix_candidate",
                "frozen_ticker": ticker,
                "asset_id": row.get("asset_id"),
                "ticker_code": row.get("ticker_code"),
                "vendor_symbol": row.get("vendor_symbol"),
                "security_name": row.get("security_name"),
                "trade_date": row.get("trade_date"),
                "member_of_universe": row.get("member_of_universe"),
            })

    for _, row in candidate_members.sort_values(["ticker_code", "asset_id"]).iterrows():
        rows.append({
            "record_type": "unmatched_reference_member",
            "frozen_ticker": None,
            "asset_id": row.get("asset_id"),
            "ticker_code": row.get("ticker_code"),
            "vendor_symbol": row.get("vendor_symbol"),
            "security_name": row.get("security_name"),
            "trade_date": row.get("trade_date"),
            "member_of_universe": row.get("member_of_universe"),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    prices = load_publication_panel(args.panel_path)
    frozen_tickers = load_frozen_capstone_tickers(args.frozen_config_path)
    mapping, unmatched = map_frozen_tickers_to_asset_ids(prices, frozen_tickers)
    diagnostics = build_diagnostics(prices, frozen_tickers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.output, index=False)

    print("step=11.1.2-mapping-diagnostics")
    print("analysis_role=diagnostic_only")
    print(f"frozen_tickers={len(frozen_tickers)}")
    print(f"mapped_asset_ids={len(set(mapping.values()))}")
    print(f"unmatched_tickers={len(unmatched)}")
    print("unmatched=" + ",".join(unmatched))
    print(f"diagnostics={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
