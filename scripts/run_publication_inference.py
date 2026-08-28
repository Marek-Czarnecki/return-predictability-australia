from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.publication_inference import build_publication_primary_inference

DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "generated" / "publication_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the four pre-defined publication confirmatory tests and Holm adjustment."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    result = build_publication_primary_inference(args.results_root)
    primary_path = args.results_root / "publication_primary_inference.csv"
    tax_year_path = args.results_root / "publication_tax_loss_primary_year_effects.csv"
    metadata_path = args.results_root / "publication_primary_inference_metadata.json"

    result.primary_inference.to_csv(primary_path, index=False)
    result.tax_loss_year_effects.to_csv(tax_year_path, index=False)

    primary = result.primary_inference
    metadata = {
        "status": "complete",
        "primary_hypothesis_count": int(len(primary)),
        "multiple_testing_family": "confirmatory_primary_family",
        "multiple_testing_method": "holm",
        "alternative": "greater",
        "walk_forward_primary_metric": "net_excess_nav_difference",
        "walk_forward_sample_unit": "evaluation_folds",
        "tax_loss_primary_metric": "abnormal_net_return_difference",
        "tax_loss_sample_unit": "calendar_years_equal_weight",
        "tax_loss_raw_return_difference_role": "secondary_descriptive",
        "reject_count_after_holm_0_05": int(primary["reject_null_0_05"].fillna(False).sum()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"primary_hypotheses={len(primary)}")
    for row in primary.itertuples(index=False):
        print(
            f"strategy={row.analysis_key} effect={row.effect_estimate:.10f} "
            f"p={row.p_value:.6f} holm={row.adjusted_p_value:.6f} "
            f"reject={row.reject_null_0_05} n={row.sample_size}"
        )
    print(f"primary_output={primary_path}")
    print(f"tax_year_output={tax_year_path}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
