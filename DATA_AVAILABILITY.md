# Data availability

The empirical analysis uses licensed Norgate Data Australia Stocks Platinum data, including historical point-in-time ASX 200 membership and security-level market history. Those licensed source data cannot be redistributed through this repository.

Accordingly, this repository does **not** contain:

- the raw Norgate extraction;
- the point-in-time ASX 200 security-level panel;
- historical constituent-level datasets that could substitute for licensed membership access;
- row-level security-price histories derived from Norgate;
- security-by-fold contribution files from the concentration analysis; or
- security-level concentration rankings derived from those contribution files.

## Public aggregate evidence

The repository does provide redistributable aggregate evidence sufficient to inspect and recompute the revised article's principal claims without exposing security-level licensed data. In particular:

- `data/evidence/publication_trend_2x2_decomposition.csv` contains the seven matched A/B/C/D fold outcomes and exact universe/composition and parameter-selection components;
- `data/evidence/publication_trend_concentration_summary.json` contains aggregate concentration statistics only, with no security identifiers or security-level contributions;
- the existing publication evidence files provide the four pre-defined confirmatory results, fold-level summaries, year-level tax-loss evidence, benchmark validation, and provenance metadata.

The A/B/C/D decomposition is exact at the fold-level compounded terminal NAV estimand. The concentration statistics are based on security-level contributions that reconcile to the arithmetic daily net-return stream; they must not be interpreted as a linear security-level decomposition of compounded terminal NAV.

A full empirical rerun requires lawful access to equivalent Norgate data and reconstruction of the publication panel according to the documented schema and point-in-time membership rules.

## Provenance

The broader frozen confirmatory evidence derives from private development commit:

`c835cd89e45a46b0b82356ef6b6d40334971da39`

The revised manuscript's post-rejection mechanism decomposition and concentration analysis derives from private `publication-extension` commit:

`7b4bec52bcdf94691b2206c2049cfa6d69ba526e`

Public-only documentation and reproduction wrappers may be newer; they do not modify those frozen empirical results.
