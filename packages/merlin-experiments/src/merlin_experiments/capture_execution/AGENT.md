# packages/merlin-experiments/src/merlin_experiments/capture_execution

Explicit `native_integer_v1` retains the original integral frontend and unchanged
source graph. Refuse recipes, staging, tolerances, floating intermediate types,
missing live-outs and changed integral ABI; preserve default float/int8-recipe
commands. Checkpoint-free v3 sessions may explicitly select this same mode with empty
extra-input/environment maps. Every stage must contain unchanged signed native
integer work; never admit a float/untransformed or host-only stage in that mode.
Legacy v3 checkpoint/recipe rules stay unchanged. The complete original session
roster, receipt, numeric ABI and shared source-state checks remain mandatory.
The existing preselection and independent isolated replay remain
authoritative. Pin the selected verifier bytes; arithmetic/target correspondence,
private exclusion, compiler correctness and complete cohort admission are separate.

`sealed_m2m_validation.py` holds the source and materialization checks delegated
by the issuer. Its exact source digest is embedded in `sealed_m2m.py` and checked
before delegation, so the selected issuer identity also binds this helper.
Changes require a new issuer identity; existing historical-issuer sets and
native-integer/read-only-runtime admission rules are unchanged. The original
issuer functions retain their callable signatures and callback patch points.

`readonly_runtime` is an explicit cache-hit-only mount route. Never open locks,
write, copy or hardlink in the source CAS. V2 defaults to16 absent leaves/1MiB;
explicit serialized limits may select at most64 leaves/16MiB. Compose only explicitly
selected absent leaves through read-only mounts and empty task-owned placeholders;
verify the complete effective namespace against the original selected inventory.
Absent flat leaf directories require existing CAS ancestors and pinned parent modes.
V3 requires explicit `overlay_version=3` and adds bounded nested absent directories;
derive canonical, owned directory modes from every selected source leaf, limiting
created directories to the selected leaf ceiling (at most64). Reopen the complete
CAS, leaves and parent modes before returning. V1/V2 interpretation stays unchanged.
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

Failed sealed child processes retain their complete stdout/stderr and exit status
beside the output directory, outside the capture whose bytes are attested.
The error includes both stream tails; warnings cannot hide a worker's actual
integerization refusal.

Runtime CAS builders keep exclusive publication locks; independent full inventories
and hard-link snapshots use shared reader locks. A reader must still hash every
selected byte and require stable final identity, including the existing bounded
rehash for link-count-only ctime transitions. Never replace verification with a
completion-marker check. Persistent churn and any content/mode/owner drift refuse.
A second per-entry inventory barrier permits concurrent inventories while
excluding hard-link creation during their stable reads. Acquire publication
ownership before this barrier; it grants no discounted or marker-only verification.
