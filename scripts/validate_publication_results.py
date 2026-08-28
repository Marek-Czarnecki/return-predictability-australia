from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_validation import validate_publication_results


DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"
DEFAULT_RISK_FREE_PATH = DEFAULT_RESULTS_ROOT / "publication_risk_free.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit publication result structure, metric semantics and risk-free coverage."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--risk-free-path", type=Path, default=DEFAULT_RISK_FREE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_publication_results(args.results_root, args.risk_free_path)

    metric_path = args.results_root / "publication_validation_metric_audit.csv"
    checks_path = args.results_root / "publication_validation_checks.csv"
    risk_free_path = args.results_root / "publication_validation_risk_free_coverage.csv"
    metadata_path = args.results_root / "publication_validation_metadata.json"

    result.strategy_metrics.to_csv(metric_path, index=False)
    result.checks.to_csv(checks_path, index=False)
    result.risk_free_coverage.to_csv(risk_free_path, index=False)

    failed = result.checks.loc[result.checks["status"] != "passed"]
    metadata = {
        "status": "passed" if failed.empty else "blocked",
        "check_count": int(len(result.checks)),
        "failed_check_count": int(len(failed)),
        "failed_checks": failed["check"].tolist(),
        "metric_semantics": {
            "legacy_total_net_excess_return": "arithmetic_sum_of_daily_excess_returns",
            "preferred_relative_metric": "net_excess_nav_difference",
        },
        "risk_free_coverage_status": result.risk_free_coverage.iloc[0]["coverage_status"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"status={metadata['status']}")
    print(f"checks={len(result.checks)}")
    print(f"failed_checks={len(failed)}")
    for check in failed["check"].tolist():
        print(f"failed={check}")
    print(f"metric_audit={metric_path}")
    print(f"checks_output={checks_path}")
    print(f"risk_free_coverage={risk_free_path}")
    print(f"metadata={metadata_path}")
    return 0 if failed.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
