from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = (
    REPO_ROOT / "data" / "generated" / "publication_results" /
    "publication_step11_trend_common_period_metadata.json"
)
EXPECTED_MEAN = -0.041322384896077646


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine Step 11.1.1 metadata scope without changing any empirical result."
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    return parser.parse_args()


def refine_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("step")) != "11.1.1":
        raise ValueError("Expected Step 11.1.1 metadata.")
    if payload.get("analysis_role") != "diagnostic_ablation" or payload.get("confirmatory") is not False:
        raise ValueError("Step 11.1.1 metadata contract changed unexpectedly.")
    observed_mean = float(payload.get("mean_net_excess_nav_difference"))
    if abs(observed_mean - EXPECTED_MEAN) > 1e-12:
        raise ValueError("Step 11.1.1 empirical mean changed; metadata-only refinement aborted.")

    payload["causal_scope"] = "sample_period_and_fold_calendar"
    payload["scope_refinement_reason"] = (
        "The exact frozen-capstone schedule changes both the evaluation sample period and the "
        "annual fold anchor relative to the full publication run; the safe inference is that "
        "sample extension is not required for the collapse, not that calendar choice has no effect."
    )
    payload["interpretation_rule"] = (
        "The publication-standard trend result is already negative on the exact frozen seven-fold "
        "calendar. Therefore adding older publication folds is not required to explain the loss of "
        "the frozen positive result. Do not claim that sample period or fold calendar has no effect."
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    payload = refine_metadata(args.metadata_path)
    print("step=11.1.1")
    print("change=metadata_scope_refinement_only")
    print(f"causal_scope={payload['causal_scope']}")
    print(f"mean_net_excess_nav_difference={float(payload['mean_net_excess_nav_difference']):.12f}")
    print(f"metadata={args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
