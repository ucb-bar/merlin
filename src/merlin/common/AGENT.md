# AGENT.md — src/merlin/common

## Purpose

Shared utilities: schemas, IO, source/resource paths, storage and access identities.
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
Actual subprocess runs freeze the selected effective environment and bind its
complete process-byte mapping by digest and key roster without recording values.
`require_environment` compares an explicit protected expected mapping; legacy
receipts cannot supply that identity. A matching environment is not a runtime,
dependency-closure, secret-management or sandbox qualification. Python call
observations do not claim a child process environment.

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
