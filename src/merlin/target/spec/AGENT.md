# src/merlin/target/spec

Reuse existing v1 numerical, ISA and typed computation records. This adapter
composes records; it does not execute Python authoring callbacks or establish
numerical/RTL equivalence. Refuse unknown fields and unsupported projections.
Imports are explicit paths pinned to exact source bytes. Inspectable expansion
and original authored identities must remain distinct.

`semantic_sources.py` owns exact-source composition and the existing typed
scalar-body and linalg records; `instructions.py` binds instruction models to
selected software/facts/source identities, compares the described type/rank axes
and validates frozen normalized records. `expansion.py` selects, expands and
normalizes a compact spec. These are structural descriptions with explicit
unknowns, not executable mathematical specifications or RTL equivalence proofs.
The historical instruction_semantics module is an exact alias to instructions;
public callable pickle spellings remain compatible. Reexported pure helpers
resolve globals in their defining modules. Compact and startup source receipts
must observe the actual owners as well as any used compatibility file.
