# Historical universe contamination in an adaptive trend-following backtest

This repository is the public reproducibility companion for the empirical study **Historical universe contamination in an adaptive trend-following backtest: Evidence from Australian equities**.

## Principal contribution

The article's main contribution is a controlled same-vendor experiment that compares genuine point-in-time ASX 200 membership with a retrospectively applied later constituent universe in the same seven-fold adaptive trend-following design.

Retrospective universe construction raises mean benchmark-relative terminal NAV performance by **0.15305** (about **15.3 percentage points**) per evaluation fold. An exact symmetric two-factor decomposition attributes **0.15170** (about **15.17 points**) to the direct universe/composition channel and **0.00135** (about **0.14 points**) to endogenous parameter reselection, even though selected parameters change in five of seven folds. The universe component is positive in all seven matched folds. Cross-sectional concentration diagnostics indicate that the composition effect is broad rather than driven by a small number of extreme survivors.

The novelty claim is deliberately bounded. The experiment isolates the constituent-membership mechanism within common Norgate security coverage. It does **not** claim that survivorship bias is newly discovered, that historical-universe bias has never affected Australian momentum research, or that the controlled Norgate experiment fully explains the earlier Yahoo-based capstone result.

## Supporting confirmatory study

The broader publication design evaluates four pre-defined return-predictability mechanisms - trend following, mean reversion, pairs trading, and tax-loss selling - using historical point-in-time ASX 200 membership, walk-forward out-of-sample evaluation, ex-ante transaction-cost assumptions, dependence-aware inference, and Holm family-wise multiplicity control.

None of the four pre-defined mechanisms receives Holm-adjusted confirmatory support under the publication-standard design. This is important supporting context, but it is not the headline identity of the article.

## Reproducibility scope

This repository is deliberately narrower than the private analytical development workspace. It provides:

- redistributable aggregate evidence for the seven-fold A/B/C/D mechanism experiment;
- aggregate concentration statistics supporting the cross-sectional claim;
- frozen aggregate evidence for the four-test confirmatory family;
- verification code for the manuscript's principal scientific invariants;
- scripts that reproduce reviewer-facing tables and figures from public evidence; and
- documentation of the boundary created by licensed Norgate Data.

The underlying security-level market and historical constituent data are licensed from Norgate Data and are not redistributed here. Security-level contribution files used for the concentration diagnosis are also withheld pending explicit redistribution/licensing clearance.

See `DATA_AVAILABILITY.md`, `REPRODUCIBILITY.md`, and `docs/PUBLICATION_EVIDENCE.md` for the precise evidence boundary and reproduction paths.

## Evidence provenance

Two frozen analytical states are relevant:

- confirmatory publication evidence and the earlier diagnostic attribution package: private development commit `c835cd89e45a46b0b82356ef6b6d40334971da39`;
- the post-rejection mechanism decomposition and concentration analysis supporting the revised manuscript contribution: private `publication-extension` commit `7b4bec52bcdf94691b2206c2049cfa6d69ba526e`.

Public-only portability, documentation, and aggregate-reproduction files may be newer; they do not alter the frozen empirical results.

## Status

This repository is the publication-facing reproducibility package. The manuscript itself is maintained separately and is not distributed here.
