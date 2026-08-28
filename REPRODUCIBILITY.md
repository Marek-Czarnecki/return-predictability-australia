# Reproducibility

This repository supports two reproducibility levels.

## 1. Evidence-level reproduction without licensed source data

Using the public files under `data/evidence/`, a reader can verify the frozen aggregate evidence and reproduce the manuscript-facing tables, figures, and stated confirmatory/diagnostic conclusions without access to the licensed security-level market panel.

This level includes:

- the four primary confirmatory results and Holm-adjusted inference;
- frozen-capstone versus publication comparisons;
- aggregate trend diagnostic attribution;
- fold-level publication summaries where publicly redistributable;
- year-level tax-loss evidence; and
- validation/provenance metadata.

Create a Python environment and install the curated public dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Verify the manuscript-facing scientific invariants from the public evidence package:

```bash
python scripts/verify_frozen_evidence.py
```

Reproduce the public manuscript tables and figures from redistributable frozen evidence:

```bash
python scripts/reproduce_tables_figures.py
```

Generated outputs are written under `tables/` and `figures/`.

## 2. Full empirical rerun with equivalent licensed data

A full rerun of the publication experiments requires lawful access to equivalent Norgate Data Australia Stocks Platinum history, including historical point-in-time ASX 200 membership. The source data must be transformed into the publication panel expected by the research code.

The public portability convention expects the licensed panel at:

`data/licensed/asx200_point_in_time_panel.parquet`

or at an explicitly supplied alternative path where supported by the relevant script.

The required panel contains, at minimum:

- `asset_id`
- `trade_date`
- `adj_close`
- `daily_return`
- `dollar_volume`
- `member_of_universe`

The historical opportunity set must be reconstructed point in time. Replacing it with a retrospective/current constituent set changes the scientific design and is specifically the subject of one of the paper's diagnostic analyses.

## Script catalogue and intended use

The public package contains two evidence-level wrapper scripts that do not require licensed market data, followed by the publication execution, diagnostic, freezing, and validation scripts used for a full empirical rerun. Scripts copied from the private development repository are reviewed for public-relative paths before release; their calculations are not changed during portability work.

### Evidence-level wrappers

| Script | Purpose | Licensed panel required? |
|---|---|---:|
| `scripts/verify_frozen_evidence.py` | Check the public frozen evidence package against the manuscript's scientific invariants, including the four-hypothesis confirmatory family and the Step 11 diagnostic boundary. | No |
| `scripts/reproduce_tables_figures.py` | Generate manuscript-facing tables and figures from redistributable aggregate evidence only. | No |

### Full-rerun publication scripts

| Script | Role in the research workflow |
|---|---|
| `build_publication_panel.py` | Build the standardized point-in-time publication panel from lawfully obtained source data. |
| `build_publication_benchmarks.py` | Construct the publication benchmark return series used by the strategy experiments. |
| `validate_publication_benchmarks.py` | Validate benchmark structure and external-comparison properties. |
| `validate_publication_eligibility.py` | Validate point-in-time membership, permanent identity, minimum-history, and execution-eligibility rules. |
| `validate_publication_liquidity_costs.py` | Validate formation-window liquidity tiers and the ex-ante transaction-cost schedule. |
| `build_publication_risk_free.py` | Construct the RBA cash-rate risk-free series with calendar-day accrual. |
| `run_publication_walk_forward.py` | Run the trend-following and mean-reversion publication walk-forward experiments. |
| `run_publication_tax_loss.py` | Run the publication tax-loss event study and year-level robustness outputs. |
| `run_publication_inference.py` | Build the four pre-defined confirmatory tests and apply Holm family-wise multiplicity control. |
| `validate_publication_results.py` | Validate the completed Step 8 publication result set and metric semantics. |
| `freeze_publication_step8_evidence.py` | Freeze and hash the controlled historical-rerun evidence package. |
| `build_publication_comparison.py` | Compare the frozen capstone evidence with the publication-standard results. |
| `freeze_publication_step9_evidence.py` | Freeze the corrected-versus-frozen comparison evidence. |
| `run_publication_step11_trend_common_period.py` | Re-run trend following on the common seven-fold calendar for diagnostic attribution. |
| `refine_publication_step11_common_period_metadata.py` | Finalize the metadata describing the common-period diagnostic run. |
| `diagnose_publication_step11_universe_mapping.py` | Diagnose legacy Yahoo-to-Norgate security coverage and mapping gaps; diagnostic only. |
| `run_publication_step11_trend_universe_ablation.py` | Compare point-in-time membership with retrospective-current membership on the same Norgate data and period. |
| `run_publication_step11_trend_benchmark_ablation.py` | Test whether benchmark choice materially explains the trend discrepancy. |
| `run_publication_step11_trend_cost_ablation.py` | Test the contribution of transaction costs to the trend discrepancy. |
| `build_publication_step11_trend_attribution.py` | Consolidate the Step 11 diagnostic results into the bounded attribution used by the manuscript. |
| `freeze_publication_step11_trend_evidence.py` | Freeze and hash the Step 11 diagnostic evidence set. |
| `build_publication_final_primary_results.py` | Assemble the definitive manuscript-facing primary-results table from already-frozen evidence. |
| `build_publication_final_evidence_metadata.py` | Assemble provenance and methodological metadata for the final evidence contract. |
| `build_publication_final_evidence_manifest.py` | Build the definitive final artifact manifest and hashes. |
| `validate_publication_step12_upstream.py` | Check that all required upstream Step 8, Step 9, and Step 11 evidence is present and valid. |
| `validate_publication_step12_invariants.py` | Check final scientific invariants, including the four confirmatory hypotheses and bounded diagnostic claims. |
| `freeze_publication_step12_evidence.py` | Freeze the definitive publication evidence package used by the manuscript. |

The Step 11 scripts are **diagnostic**, not additional confirmatory hypothesis tests. They are used to explain why the earlier trend result changes and must not be pooled with the four confirmatory tests or treated as an independent causal decomposition.

For a full rerun, the intended sequence is: construct and validate data/benchmark/cost/risk-free inputs; run the three walk-forward strategy families and tax-loss study; run confirmatory inference and Step 8 validation; freeze Step 8; build/freeze the historical comparison; run and freeze the Step 11 diagnostics; then construct, validate, and freeze the Step 12 final evidence package. `verify_frozen_evidence.py` and `reproduce_tables_figures.py` operate after that freeze and do not alter empirical results.

## Evidence freeze

The authoritative empirical manuscript evidence was frozen at private development commit:

`c835cd89e45a46b0b82356ef6b6d40334971da39`

The public package does not recompute or modify those frozen results during transfer. Portability edits to publication code, where necessary, are restricted to non-scientific matters such as paths, documentation, and public data-location handling.

The public verification script checks the key scientific invariants rather than attempting to recreate licensed source observations. Full source-data hash equivalence cannot be verified from the public repository because the licensed market panel is intentionally excluded.
