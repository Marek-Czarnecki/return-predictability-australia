# Do return-predictability effects survive transfer? Evidence from Australian equities

This repository is the public reproducibility companion for an empirical finance study examining whether four established return-predictability mechanisms—trend following, mean reversion, pairs trading, and tax-loss selling—survive transfer to Australian equities under a common point-in-time, out-of-sample, implementation-aware, and multiplicity-controlled research design.

## Research design

The publication analysis uses historical point-in-time ASX 200 membership, walk-forward out-of-sample evaluation, ex-ante transaction-cost assumptions, dependence-aware inference, and Holm family-wise multiplicity control across four pre-defined confirmatory tests.

The principal confirmatory result is that none of the four pre-defined effects receives support after Holm adjustment under the publication-standard design. A separate diagnostic analysis shows that retrospective-current constituent membership is the major directly demonstrated contributor to the change in the earlier trend-following result. Diagnostic ablations are not part of the confirmatory Holm family and are not interpreted as an additive causal decomposition.

## Repository purpose

This repository is deliberately narrower than the private development workspace. It is intended to provide:

- frozen aggregate evidence supporting the manuscript;
- code implementing the publication research design;
- tests for publication-specific scientific invariants;
- scripts to reproduce manuscript tables and figures from redistributable evidence; and
- documentation of the boundary created by licensed source data.

## Data availability

The underlying security-level market and historical constituent data are licensed from Norgate Data and cannot be redistributed here. The repository therefore does not contain the raw Norgate extraction, the point-in-time ASX 200 panel, or other row-level licensed datasets that could substitute for access to the licensed source.

Publicly redistributable aggregate evidence and validation artifacts are provided where they are sufficient to reproduce the reported manuscript tables, figures, and inferential conclusions. See `DATA_AVAILABILITY.md` and `REPRODUCIBILITY.md` for the precise boundary.

## Evidence provenance

The empirical manuscript evidence was frozen in the private development repository at commit:

`c835cd89e45a46b0b82356ef6b6d40334971da39`

Files transferred into this repository are drawn from that frozen state unless explicitly documented as public-only reproducibility material.

## Status

The repository is being assembled as the publication-facing reproducibility package. The manuscript itself is not included in the initial repository package.
