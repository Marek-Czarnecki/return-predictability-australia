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
from strategies.publication_tax_loss import run_publication_tax_loss_event_study

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "generated" / "publication_results"
DEFAULT_BENCHMARK_PATH = DEFAULT_OUTPUT_DIR / "publication_external_benchmark.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the publication point-in-time tax-loss-selling event study."
    )
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PUBLICATION_PANEL_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-years", type=int, default=None)
    return parser.parse_args()


def _load_benchmark(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    required = {"trade_date", "benchmark_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    if frame["trade_date"].duplicated().any():
        raise ValueError(f"{path} contains duplicate trade dates.")
    return frame.loc[:, ["trade_date", "benchmark_return"]].sort_values("trade_date")


def export_publication_tax_loss_result(result, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "publication_tax_loss_selling"
    paths = {
        "events": output_dir / f"{prefix}_event_study.csv",
        "summary": output_dir / f"{prefix}_summary.csv",
        "robustness": output_dir / f"{prefix}_year_robustness.csv",
        "liquidity": output_dir / f"{prefix}_liquidity_diagnostics.csv",
        "metadata": output_dir / f"{prefix}_metadata.json",
    }
    result.event_study.to_csv(paths["events"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.year_robustness.to_csv(paths["robustness"], index=False)
    result.liquidity_diagnostics.to_csv(paths["liquidity"], index=False)

    summary = result.summary.iloc[0].to_dict() if not result.summary.empty else {}
    metadata = {
        "strategy_name": "tax_loss_selling",
        "identity_col": "asset_id",
        "membership_selection_policy": "point_in_time_member_on_selection_date",
        "history_policy": "all_observable_asset_id_history_for_252_observation_trailing_return",
        "benchmark": "XJOA_total_return_index",
        "cost_application": "symmetric_round_trip_cost_event_and_control",
        "missing_window_policy": "require_complete_security_and_benchmark_windows",
        "event_observation_count": int(len(result.event_study)),
        "complete_matched_observation_count": int(summary.get("complete_matched_observation_count", 0)),
        "year_count": int(summary.get("year_count", 0)),
        "liquidity_diagnostic_rows": int(len(result.liquidity_diagnostics)),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def main() -> int:
    args = parse_args()
    prices = load_publication_panel(args.panel_path)
    benchmark = _load_benchmark(args.benchmark_path)
    result = run_publication_tax_loss_event_study(prices, benchmark, max_years=args.max_years)
    paths = export_publication_tax_loss_result(result, args.output_dir)

    print("strategy=tax_loss_selling")
    print(f"events={len(result.event_study)}")
    print(f"years={int(result.summary.iloc[0]['year_count']) if not result.summary.empty else 0}")
    print("complete_matched=" f"{int(result.summary.iloc[0]['complete_matched_observation_count']) if not result.summary.empty else 0}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
