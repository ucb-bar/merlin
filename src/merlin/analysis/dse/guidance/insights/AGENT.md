# AGENT.md - src/merlin/analysis/dse/guidance/insights

These owners consume explicit case-study artifacts and produce source-traced
facts, design questions, analytical envelopes and reports. Keep evidence tiers,
unknowns, caveats, output schemas and ordering unchanged during refactoring.
They issue no numerical acceptance policy, device qualification or timing claim.

`insight_mining.mine` composes the analyses. Readers and plot renderers call the
defining analysis owner. Keep report formatting beside its analysis; cross-analysis
summaries and ordered artifact emission have their own consumers. The optional
gap-audit fallback imports the pipeline only when it must construct a bundle.
Generated outputs belong under the caller-selected run directory, never inputs.
