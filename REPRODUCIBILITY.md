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

Public reproduction scripts will operate only on these redistributable files.

## 2. Full empirical rerun with equivalent licensed data

A full rerun of the publication experiments requires lawful access to equivalent Norgate Data Australia Stocks Platinum history, including historical point-in-time ASX 200 membership. The source data must be transformed into the publication panel expected by the research code.

The required panel contains, at minimum:

- `asset_id`
- `trade_date`
- `adj_close`
- `daily_return`
- `dollar_volume`
- `member_of_universe`

The historical opportunity set must be reconstructed point in time. Replacing it with a retrospective/current constituent set changes the scientific design and is specifically the subject of one of the paper's diagnostic analyses.

## Evidence freeze

The authoritative empirical manuscript evidence was frozen at private development commit:

`c835cd89e45a46b0b82356ef6b6d40334971da39`

The public package does not recompute or modify those frozen results during transfer. Portability edits to publication code, where necessary, are restricted to non-scientific matters such as paths, documentation, and public data-location handling.
