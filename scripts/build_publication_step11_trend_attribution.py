from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_step11_trend_attribution import (
    build_trend_attribution_table,
    export_trend_attribution,
)

RESULTS = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 11.1.5 consolidated trend attribution evidence.")
    parser.add_argument("--corrected-vs-frozen", type=Path, default=RESULTS / "publication_corrected_vs_frozen_comparison.csv")
    parser.add_argument("--common-period-metadata", type=Path, default=RESULTS / "publication_step11_trend_common_period_metadata.json")
    parser.add_argument("--universe-metadata", type=Path, default=RESULTS / "publication_step11_trend_universe_ablation_metadata.json")
    parser.add_argument("--benchmark-metadata", type=Path, default=RESULTS / "publication_step11_trend_benchmark_ablation_metadata.json")
    parser.add_argument("--cost-metadata", type=Path, default=RESULTS / "publication_step11_trend_cost_ablation_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table, metadata = build_trend_attribution_table(
        args.corrected_vs_frozen,
        args.common_period_metadata,
        args.universe_metadata,
        args.benchmark_metadata,
        args.cost_metadata,
    )
    paths = export_trend_attribution(table, metadata, args.output_dir)
    print("step=11.1.5")
    print("analysis_role=consolidated_diagnostic_attribution")
    print(f"rows={len(table)}")
    print(f"major_identified_contributor={metadata['major_identified_contributor']}")
    print(f"modest_identified_contributor={metadata['modest_identified_contributor']}")
    print(f"unresolved_contributor={metadata['unresolved_contributor']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    print("attribution_statement=")
    print(metadata["attribution_statement"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
