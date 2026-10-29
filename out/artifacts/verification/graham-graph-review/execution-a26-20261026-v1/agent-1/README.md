# Admission source review

This agent reviewed the ten Python projection files changed between the original
review's **58dbc7c00** source cut and **c62f81a8a**. The changes wire in conservative
input admission and bind the helper into capture/source identities. Frozen Lean
semantics and statements were unchanged.

The independent Python 3.12/xDSL 0.68.0 subset passed **32/32 controls**, with no
failures or skips. It checked the transferred indirect-call/resource fixtures,
metadata refusal before parsing, nested resources/symbols, signatures, explicit
direct-call modes, repeated parser state and positive cases. Exact loaded-source
pins and raw outputs are in [the result](ADMISSION_SUBSET_XDSL_0.68.0.json).

[CHANGED_SOURCE_REVIEW.json](CHANGED_SOURCE_REVIEW.json) records the per-file
observations and remaining trust boundaries; [SUPPORTED_SCOPE.csv](SUPPORTED_SCOPE.csv)
states the supported variants and assumptions. The original 81-case runner and
producer closure were not transferred, so that replay is unavailable. No new
implementation bug was demonstrated within this bounded review.

The first own harness failed because it rejected Python's current-process
`ctypes.PyDLL(None)` bootstrap. Its source/failure is retained. The successor
permits that bootstrap and continues refusing named library loads and child
launches; the completed run observed none of those prohibited actions.

This report supplies agent review and admission-test evidence. It does not
establish raw MLIR semantics, the host matcher, call-body correspondence, native
fidelity, graph qualification, human approval or compiler enforcement. No compiler
implementation was edited, and no owned job remains. [MANIFEST.json](MANIFEST.json)
binds the 19 original run files; this publication README is an additional index.
