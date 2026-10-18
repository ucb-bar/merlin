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

`m2m_origin.py` owns the narrow Git-origin inspection and staged Phase 0
runtime-receipt binding used by checkpoint-free sealed M2M captures. Its Git
revision is only an origin hint; copied package bytes remain execution authority.
`runtime_rehydrate.py` owns offline recovery of a new source venv from an
independently selected, fully verified runtime CAS. It records a new byte
identity and excludes stale editable origins explicitly; it never restores an
old venv path or promotes historical capture evidence.
Shared runtime hard links change inode ctime/link counts while preserving bytes.
Inventory never accepts an unstable hash: a link-count-only transition permits
at most two rehashes, and the final read must keep full device/inode/size/mtime/
mode/owner/ctime identity stable. Content, mode or ownership changes still fail;
selected tree SHA checks remain authoritative. Persistent link churn is unavailable.

Failed sealed child processes retain their complete stdout/stderr and exit status
beside the output directory, outside the capture whose bytes are attested.
The error includes both stream tails; warnings cannot hide a worker's actual
integerization refusal.

Runtime CAS builders keep exclusive publication locks; independent full inventories
and hard-link snapshots use shared reader locks. A reader must still hash every
selected byte and require stable final identity, including the existing bounded
rehash for link-count-only ctime transitions. Never replace verification with a
completion-marker check. Persistent churn and any content/mode/owner drift refuse.
