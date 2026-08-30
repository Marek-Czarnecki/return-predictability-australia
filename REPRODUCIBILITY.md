# Reproducibility

This repository supports two reproducibility levels. The first is designed so that a reviewer can verify the article's principal aggregate claims without access to licensed source data. The second supports a full empirical rerun for researchers with lawful access to equivalent Norgate Data.

## 1. Evidence-level reproduction without licensed source data

Create a Python environment and install the curated public dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Verify the manuscript-facing scientific invariants:

```bash
python scripts/verify_frozen_evidence.py
```

The verifier checks, among other things:

- all seven A/B/C/D mechanism rows;
- the exact identity `universe component + parameter-selection component = B - A` for every fold;
- mean total effect `0.153051485153`;
- mean universe/composition component `0.151701286408`;
- mean parameter-selection component `0.001350198745`;
- positive universe component in all seven folds;
- parameter-selection changes in five of seven folds;
- the aggregate cross-sectional concentration statistics;
- the existing four-test Holm confirmatory family and zero Holm rejections; and
- the licensing boundary excluding raw and security-level Norgate data.

Reproduce reviewer-facing tables and figures from the redistributable evidence package:

```bash
python scripts/reproduce_tables_figures.py
```

Generated outputs are written under `tables/` and `figures/`.

### Public evidence supporting the principal contribution

`data/evidence/publication_trend_2x2_decomposition.csv` contains the seven matched fold outcomes:

- A = point-in-time universe + point-in-time-selected parameters;
- B = retrospective universe + retrospective-selected parameters;
- C = retrospective universe + point-in-time-selected parameters;
- D = point-in-time universe + retrospective-selected parameters.

The public file is sufficient to recompute the exact two-factor decomposition and the manuscript's headline means.

`data/evidence/publication_trend_concentration_summary.json` contains only aggregate concentration statistics. Security-level contribution and ranking rows are intentionally excluded.

### Technical boundary on concentration evidence

The fold-level A/B/C/D Shapley decomposition is exact for compounded terminal benchmark-relative NAV difference.

The security-level accounting underlying the concentration statistics reconciles to the **arithmetic daily net-return stream**, not compounded terminal NAV. The concentration result therefore answers whether the universe/composition effect is concentrated across securities; it is not a linear security-level decomposition of terminal NAV.

## 2. Full empirical rerun with equivalent licensed data

A full rerun requires lawful access to equivalent Norgate Data Australia Stocks Platinum history, including historical point-in-time ASX 200 membership. The source data must be transformed into the publication panel expected by the research code.

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

The historical opportunity set must be reconstructed point in time. Replacing it with a retrospective/current constituent set changes the scientific design and is the treatment studied in the article's central mechanism experiment.

### Pairs-classification input

The frozen pairs implementation also uses a current-only ticker-to-sector classification map during formation-stage candidate generation. In the development repository this input was `config/asx_ticker_sector_map.csv`, derived from Yahoo Finance/yfinance classifications dated 2026-07-24, with one manually verified classification. A full pairs rerun must supply an equivalent map with at least `ticker_code` and `sector` columns or disclose any substitution.

### Historical-comparison inputs

Some legacy comparison scripts require the frozen original-capstone result directory because those source files are not part of the publication public-file whitelist. Those comparisons are supporting historical context and are not the article's exact two-factor mechanism decomposition.

## Script catalogue and intended use

### Evidence-level wrappers

| Script | Purpose | Licensed panel required? |
|---|---|---:|
| `scripts/verify_frozen_evidence.py` | Verify the central mechanism decomposition, concentration aggregates, confirmatory family, and licensing boundary. | No |
| `scripts/reproduce_tables_figures.py` | Generate reviewer-facing mechanism, concentration, confirmatory, and supporting diagnostic displays from public aggregate evidence. | No |

### Full-rerun publication scripts

The repository also contains the publication panel, benchmark, eligibility, liquidity-cost, risk-free, walk-forward, tax-loss, inference, validation, historical-comparison, and diagnostic scripts ported from the private development workspace. Their scientific calculations are not changed during public portability work.

The older files and scripts that use development labels such as "Step 9" or "Step 11" preserve their historical implementation names. In the revised manuscript project plan, **Step 9** refers to the later post-rejection two-factor mechanism and concentration analysis. This documentation mapping prevents the historical development labels from being mistaken for the current manuscript task numbering.

All full-rerun outputs default to `data/generated/publication_results/`, which is gitignored. Licensed market inputs remain under `data/licensed/`, also gitignored. The public `data/evidence/` directory is reserved for frozen redistributable evidence and is not overwritten by full-rerun wrappers.

## Evidence provenance

The broader confirmatory and earlier diagnostic package was frozen at private development commit:

`c835cd89e45a46b0b82356ef6b6d40334971da39`

The revised article's post-rejection mechanism decomposition and concentration analysis derives from private `publication-extension` commit:

`7b4bec52bcdf94691b2206c2049cfa6d69ba526e`

Public portability edits are restricted to documentation, paths, aggregate evidence packaging, verification, and reviewer-facing reproduction. They do not alter the frozen empirical calculations.
