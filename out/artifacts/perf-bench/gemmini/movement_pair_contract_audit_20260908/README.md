# Frozen movement controlled-pair audit

Conclusion: **REFUSED**. The six existing runs cannot populate the controlled-pair feature
calibration contract without new execution. No controlled observation or feature receipt was
created.

The content-addressed audit is `audit_receipt.json` (canonical receipt digest
`c8e74b9f35750b3290a562ff763705dea7fafd8ab77ffd38c55005db12595a24`). The standard feature preparer independently returns
`incomplete` in `physical_movement_preparation.json` and produces no calibration.

The decisive gaps are: no predeclared controls; only one run has a distinct static emitted movement
count; no executed-command measurement; no physical-interface byte measurement; no warm-predecessor
proof; no original receipt binding source, target, package, emitted artifacts, and console; and a
currently inconsistent package `SHA256SUMS` receipt.

All activity in this audit was host-only reading and hashing of frozen bytes. No target, simulator,
model, or layer execution occurred.
