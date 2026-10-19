# packages/merlin-experiments/src/merlin_experiments/capture_execution

Explicit `native_integer_v1` retains the original integral frontend and unchanged
source graph. Refuse recipes, staging, tolerances, floating intermediate types,
missing live-outs and changed integral ABI; preserve default float/int8-recipe
commands. The existing preselection and independent isolated replay remain
authoritative. Pin the selected verifier bytes; arithmetic/target correspondence,
private exclusion, compiler correctness and complete cohort admission are separate.

`readonly_runtime` is an explicit cache-hit-only mount route. Never open locks,
write, copy or hardlink in the source CAS. V2 defaults to16 absent leaves/1MiB;
explicit serialized limits may select at most64 leaves/16MiB. Compose only explicitly
selected absent leaves through read-only mounts and empty task-owned placeholders;
verify the complete effective namespace against the original selected inventory.
Absent flat leaf directories require existing CAS ancestors and pinned parent modes.
Do not replace existing members, waive missing aliases/dependencies, or repair the
source store. Preserve the default writable-CAS owner and its historical behavior.

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
