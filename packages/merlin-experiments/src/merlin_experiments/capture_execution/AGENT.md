# packages/merlin-experiments/src/merlin_experiments/capture_execution

This package owns the independent, fresh capture execution boundary. The static
issuer runs only static ELF programs inside a private bubblewrap namespace. The
sealed Model2MLIR CPU runner (`sealed_m2m`, schema `merlin.sealed_m2m_cpu.v2`) is
admitted as a Phase 0 issuer by an operator policy decision, only for runs that
were preselected (`phase0/capture_selection.py`) and replayed in a fresh sandbox;
the admission gate lives in `phase0/capture_execution_attestation.py` and records
the accepted residuals (unsigned receipt, copied venv rather than a pinned
dependency closure). Do not widen it to another runner or schema without a new
reviewed decision. Replay the fixed sandbox policy and compare every input/output
byte before admitting a receipt. Keep this package separate from Phase 0 source hashing.
