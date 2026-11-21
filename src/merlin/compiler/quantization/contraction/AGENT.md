# src/merlin/compiler/quantization/contraction

This package owns source-bound contraction precision: intake, exact numerical
records, checked logical-value conversion, static operand preparation, explicit
tile/whole-contraction transformation, and finite-float encoder construction.
Keep the diagnostic transformation distinct from equivalence rules and preserve
all source, policy, arithmetic, and target facts required by existing callers.

`contract.py` and `values.py` hold the numerical records and checked conversion.
`intake.py` recognizes source-bound matrix operations; `transform.py` constructs
explicit diagnostic tiles. `scalar.py` and `batched.py` lower semantic contraction
requests; `constants.py` prepares declared static inputs. `codegen.py` constructs
the existing checked integer-only finite-float C encoder.

Semantic graph/model and Linalg bridge types remain explicit dependencies in
`semantic_compiler`; graph precision does not establish compilation equivalence
or application quality. Native snapshot source accounting must include these
actual implementation owners as well as any compatibility entrypoints.

Historical `semantic_compiler` imports alias these exact modules for external
Atlas callers and old pickle lookups. Current records identify their canonical
owner; do not relabel `__module__`, copy algorithms, or add forwarding wrappers.
