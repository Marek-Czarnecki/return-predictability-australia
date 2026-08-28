# Publication evidence classes

The public evidence package preserves three distinct evidence classes used in the manuscript.

## A. Confirmatory publication evidence

The primary question is whether four pre-defined return-predictability mechanisms survive transfer to Australian equities under one common empirical standard. The confirmatory family contains exactly four tests: trend following, mean reversion, pairs trading, and tax-loss selling. Holm family-wise multiplicity control is applied across this four-test family.

The definitive manuscript-facing confirmatory results are recorded in `data/evidence/publication_final_primary_results.csv`.

## B. Historical comparison evidence

Frozen-capstone versus publication comparisons are used to establish where evidentiary conclusions changed and where they did not. These comparisons are not a fifth confirmatory hypothesis. For tax-loss selling, the frozen pooled-event and publication calendar-year inferential units differ, so their p-values are contextual rather than like-for-like comparisons.

## C. Diagnostic attribution evidence

The Step 11 trend analyses diagnose why the earlier trend result changes. These analyses are outside the four-test Holm family. The major directly demonstrated contributor is the use of retrospective-current constituent membership rather than point-in-time membership in a same-vendor, common-period comparison. Transaction costs are a smaller directly demonstrated contributor. Benchmark choice, metric semantics, risk-free treatment, and sample extension do not explain the deterioration. Vendor/security coverage remains unresolved.

The diagnostic ablations are not additive causal components and must not be summed into a complete decomposition.

## Provenance

The empirical evidence freeze is private development commit:

`c835cd89e45a46b0b82356ef6b6d40334971da39`

The public repository exists to expose redistributable frozen evidence and reproducibility logic without redistributing licensed Norgate security-level data.
