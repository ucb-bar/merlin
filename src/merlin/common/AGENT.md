# AGENT.md — src/merlin/common

## Purpose

Shared utilities: schemas, IO, source/resource paths, storage and access identities.
`dense_float_ir` adapts xDSL dense float IO without converting hex bit patterns
through Python floats. Its printer retains exact dense bytes, including NaN
payloads and signed zeros. Grammar and shape checks remain xDSL-owned; this
serialization boundary grants no numerical equivalence or arithmetic rewrite.
`owned_children` is the original shared Linux pidfd/subreaper cleanup primitive.
Only a dedicated child becomes a subreaper; real waitpid/ECHILD closes its tree.
The experiments import remains a reexport and explicit source rosters must pin
this actual core implementation. No descendant snapshot grants cleanup.
`owned_process` applies that owner to captured external package commands. A fresh
isolated Python guardian watches its caller's pidfd and reaps direct/adopted
children before success or timeout returns, including session-escaping children.
Only the guardian forks; concurrent caller threads never run Python after fork.
Unknown cleanup retains the guardian and cannot establish job completion.
`cpu_pool` applies explicitly selected launch-thread affinity and optional exclusive
per-CPU subprocess slots, restoring caller affinity. It chooses no role or CPU;
trusted callers own pool selection, timeout budgets and process supervision.
An explicit shared compute roster and slot directory provide one namespace for
build, simulation and bounded preparation steps. Same-process/thread nested steps
reuse one active lease; incompatible nested selections refuse. Preparation worker
affinity never holds a capsule-wide slot across child or RPC waits.
The pool is divided only by lease priority (`cpu_priority`: AUTHOR 1, GRADE 2,
PREPARATION 3, AUXILIARY 4; `MERLIN_CPU_PRIORITY` carries it to children). A
cross-process `waiters/` queue in the slot directory serves more urgent classes
first and FIFO within a class; no CPU is reserved for a class, and dead or
stopped waiters never block. Tickets record eligible CPUs; matching across the
complete queue preserves earlier feasible waiters while letting unrelated free
CPUs run, including flexible-head/singleton combinations. Legacy empty tickets
remain conservative. Fresh post-flock checks re-prove admission before use.
Unset engine caps default to `compute_capacity()`.
The common access registry contains only target-neutral identities; the experiments
sandbox extends it with mandatory packaged historical target denials before launch.
`frozen_imports` provides stdlib-only, process-local source import isolation for
trusted bootstraps. Callers own snapshot seals, invocation records and grading.
`source_membership` inventories ordinary Python source files for existing run
receipts; it owns no seal, target policy or optional-distribution dependency.
`ir_audit` owns opt-in exact-byte named-stage inspection records. Lowering callers
own stage serialization and explicit destination and sidecar selection; these observations
are not compiler certificates or a replacement for frozen-source seals.
Caller-produced compact stage views are explicitly non-executable and bind the exact
parent content hash; printer/framework dependencies remain outside this common utility.
Large xDSL dense inspection payloads are stored as exact raw bytes under content-addressed
`tensors/` files within one audit. Stage descriptors bind producer element types and shapes;
identical bytes share storage across stages. Completion rechecks generated tensor hashes.
This is inspection storage, not safetensors conversion or an executable reconstruction ABI.
`compile_trace` is the user-facing debugging switch over the same recording points: a request
(in `MERLIN_COMPILE_TRACE`, so child processes see it) names stages to dump and one to stop after.
Pipelines declare their stage names beside the code that records them; never list them here. Only
the session's own process raises the stop (`StopAfterStage`, a `BaseException`), so a stop is
never swallowed as a per-group failure.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Workstream-specific logic (this is shared infrastructure only).
- Generated artifacts (write those to `runs/` or `artifacts/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
- Shared helpers (schema load/validate, yaml, llm summary) are real and dependency-light.
- CAS-backed snapshots share file modes. Before deleting a read-only bundle, use
  its snapshot cleanup owner; bare TemporaryDirectory cleanup can chmod shared
  files after an unlink failure and invalidate other sealed consumers.
- Content-store copy fallbacks keep public files read-only (0444 or executable
  0555), including cross-filesystem and disabled-store copies. Explicit
  permission-preserving copies bypass CAS and keep only the source's read/execute
  bits; they must never widen owner-only access or overwrite a shared inode.
  Destination symlinks refuse; callers own destination paths, existing regular
  files and directory modes.
- Frozen imports never fall through to a live owner or unchecked bytecode. This
  provenance boundary is not a Python sandbox and does not propagate to subprocesses
  without an explicit bootstrap. Keep experiment-specific launch policy out of core.
Selected interpreter bootstraps may use `frozen_imports.activate` with
`expose_roots_to_path=False`: protected namespaces and resources use their pinned
roots without adding an outer interpreter's dependency directory to `sys.path`.
Split protected packages still merge only their declared ordinary source roots.
The caller owns existing search paths and the selected interpreter's dependencies;
this option grants no dependency, sandbox or compilation authority.

`strict_json` rejects ambiguous keys, non-finite numbers and oversized authority
records before consumers interpret them; it does not confer provenance or admission.
`jsonio.strict_json_equal` compares closed JSON records without Python bool/int,
float/int, or tuple/list aliases; callers still own source and receipt identity.

`invocation_record` retains exact file/tool/source pins and complete stdout/stderr
at actually invoked process or Python-call boundaries. Interrupted observations
remain unavailable. Caller-owned destinations stay outside invoked packages;
callers own dependency completeness, sandbox routing and semantic stage lift.
The observer confers no correctness, stage applicability or admission.
`complete_streams` seals the caller's incrementally captured raw files without
loading the full trace into RAM. The caller still owns actual bounded execution,
cleanup and stream completion; the method cannot issue those observations.
`replayable_text` feeds existing line consumers from complete bounded raw UTF-8
files without materializing the whole text. Each exhausted pass checks original
bytes and a fresh reopened hash; it preserves Python splitlines semantics across
chunk boundaries. The caller supplies its original raw byte ceiling and owns
execution, decoder, source and admission authority. No compression exemption,
sampling, persistent proof cache or partial-pass receipt is supplied.
Actual subprocess runs freeze the selected effective environment and bind its
complete process-byte mapping by digest and key roster without recording values.
An explicit `owned_process_tree` selection uses the same frozen command and
environment through the dedicated child guardian, pinning both cleanup sources.
`require_environment` compares an explicit protected expected mapping; legacy
receipts cannot supply that identity. A matching environment is not a runtime,
dependency-closure, secret-management or sandbox qualification. Python call
observations do not claim a child process environment.
`record_artifact_reuse` writes a distinct `artifact_reuse.json` byte receipt,
joined to the actual original completed process and exact restored products.
It has no new process argv, return code, invocation timing or repeat; ordinary
`verify` refuses it. `verify_artifact_reuse` reopens byte provenance only and
does not issue a live cache or closed runtime owner. The original build owner
must authenticate actual namespace adoption before selecting reuse.

`selected_pin_replay` issues an explicit bounded same-thread/same-process call
owner for selected canonical file content. It binds the actual Thread identity
and issuing process ID, preventing recycled thread IDs and fork inheritance.
Invocation verification may reuse that content
only in executable/dependency roles; complete pre/post rereads remain mandatory,
including exceptional exits. Every unselected input, product and record stays
fresh. Actual observation/run/completion always hash independently. There is no
ambient selection, mtime inference or reuse across calls, nested owners, threads or
processes.
This controls replay cost and grants no source, stage or runtime authority.

`provenance_lost` records explicitly declared unrecoverable artifact identities.
Pin and artifact loaders reject redeclaring those identities as live; build products
use ordinary checkout-relative paths and are hashed independently of git status.

`storage_ops` refreshes cached lsof listings after a verified move copy and at
each peer/store duplicate replacement boundary. Census may reuse its original
listing; a destructive operation may not. Live fuser checks and injected checkers
retain their holders interface. A refused late check preserves the original
name and removes only the operation's temporary staged link.

`execution_deadline` carries one explicit monotonic wall budget across selected
ordinary stages. Passing a parent never resets it; absent selection changes no
legacy caller behavior. It owns no correctness, runtime or hardware timer authority.

`pinned_files` reopens explicitly selected regular files and creates exclusive
owner-private streamed snapshots with exact source/output hashes. Callers own
protected parent directories, resource locking, complete input membership and
actual consumer correspondence. A file snapshot is never a compiler, hardware,
runtime, numerical or timing qualification. Failed attempts are retained by the
caller and return no snapshot selection; mode bits alone do not prove immutability.

`bundle_snapshot` owns generic original-input custody: V4 archived declarations,
canonical frozen source mappings, payload digest/counts and separately bound
marker bytes. It has no experiment dependency and does not decide candidate
visibility. The experiment freezer delegates payload hashing; its independent
mount/answer checks remain mandatory. A verified input is not an execution or
semantic receipt, and host-only inputs do not become public grants.

Inside one `verified_snapshot_scope`, Linux watches of every file and directory
may reuse the names/types/sizes roster. Watches include individual inodes so an
external hard-link resize invalidates the shortcut. Mount-table changes, events
(including overflow), unavailable watches or closed custody restore complete
scanning. Payload hashing and selected read-lease checks remain independent at
both operation boundaries; this shortcut never authenticates payload bytes.
An original executor may select its issued `SharedEntry` with
`shared_entry_scope`; implicit selection stays on the issuing process/thread
and each nested operation receives a distinct verification token. Every final
boundary independently reads/hashes payloads or validates their live read
leases. A kernel-watched roster may avoid the repeated directory walk at that
boundary; any event during the read repeats the complete scan and payload
verification. Independent consumer reads clear both entry and roster reuse.
Explicit bounded `MERLIN_SNAPSHOT_LEASE_MIN_BYTES` and
`MERLIN_SNAPSHOT_LEASE_MAX_FILES` select cache admission, never replace a live
kernel read lease. Cached path checks pin an O_PATH lookup and compare its
kernel-resolved ordinary name and inode; replaced, indirect or deleted names
refuse. Descriptor headroom is reserved at cache admission without a quadratic
per-member descriptor enumeration.
`read_lease_digest.reuse` batches hits for a closed pin set that one process
reverifies repeatedly. It returns only members admitted earlier by `digest`,
after rechecking every member's O_PATH name, identity and lease, then every lease
again with one custodian check. A failed member declines so the caller's ordinary
per-file hit reaches the existing refusal; `required=True` refuses directly. It
never admits, hashes or trusts timestamps. An explicit `minimum_bytes` admission
threshold changes only which files may be leased, never capacity or custody.
