# src/merlin/common/process

## Purpose

Owned child processes: who owns a child's lifetime, what was invoked and with which bytes,
and which CPUs a step may use.

- `ownership.py` — the shared Linux pidfd/subreaper cleanup primitive, owned command capture
  through an isolated guardian, and the explicit wall-clock deadline.
- `invocation_records.py` — records of invoked processes and Python calls, selected pin
  replay, `run_checked`, and bubblewrap argv transport.
- `cpu_pool.py` — CPU leases and the priority queue.

## Invariants

- Only a dedicated child becomes a subreaper; real waitpid/ECHILD closes its tree. Explicit
  source rosters must pin this actual core implementation. No descendant snapshot grants
  cleanup. Owned command capture applies that owner to captured external package commands:
  a fresh isolated Python guardian (`ownership.py` run as a script, stdlib-only) watches its
  caller's pidfd and reaps direct/adopted children before success or timeout returns,
  including session-escaping children. Only the guardian forks; concurrent caller threads
  never run Python after fork. Unknown cleanup retains the guardian and cannot establish job
  completion. Keep `ownership.py` free of non-stdlib imports: the guardian runs it with `-I`.
- The execution deadline carries one explicit monotonic wall budget across selected ordinary
  stages. Passing a parent never resets it; absent selection changes no legacy caller
  behavior. It owns no correctness, runtime or hardware timer authority.
- `invocation_records` retains exact file/tool/source pins and complete stdout/stderr at
  actually invoked process or Python-call boundaries. Interrupted observations remain
  unavailable. Caller-owned destinations stay outside invoked packages; callers own
  dependency completeness, sandbox routing and semantic stage lift. The observer confers no
  correctness, stage applicability or admission. `complete_streams` seals the caller's
  incrementally captured raw files without loading the full trace into RAM. The caller still
  owns actual bounded execution, cleanup and stream completion; the method cannot issue those
  observations. Actual subprocess runs freeze the selected effective environment and bind its
  complete process-byte mapping by digest and key roster without recording values. An
  explicit `owned_process_tree` selection uses the same frozen command and environment
  through the dedicated child guardian, pinning its cleanup source. `require_environment`
  compares an explicit protected expected mapping; legacy receipts cannot supply that
  identity. A matching environment is not a runtime, dependency-closure, secret-management or
  sandbox qualification. Python call observations do not claim a child process environment.
  `record_artifact_reuse` writes a distinct `artifact_reuse.json` byte receipt, joined to the
  actual original completed process and exact restored products. It has no new process argv,
  return code, invocation timing or repeat; ordinary `verify` refuses it.
  `verify_artifact_reuse` reopens byte provenance only and does not issue a live cache or
  closed runtime owner. The original build owner must authenticate actual namespace adoption
  before selecting reuse.
- Selected pin replay issues an explicit bounded same-thread/same-process call owner for
  selected canonical file content. It binds the actual Thread identity and issuing process
  ID, preventing recycled thread IDs and fork inheritance. Invocation verification may reuse
  that content only in executable/dependency roles; complete pre/post rereads remain
  mandatory, including exceptional exits. Every unselected input, product and record stays
  fresh. Actual observation/run/completion always hash independently. There is no ambient
  selection, mtime inference or reuse across calls, nested owners, threads or processes.
  This controls replay cost and grants no source, stage or runtime authority.
- `cpu_pool` applies explicitly selected launch-thread affinity and optional exclusive
  per-CPU subprocess slots, restoring caller affinity. It chooses no role or CPU; trusted
  callers own pool selection, timeout budgets and process supervision. An explicit shared
  compute roster and slot directory provide one namespace for build, simulation and bounded
  preparation steps. Same-process/thread nested steps reuse one active lease; incompatible
  nested selections refuse. Preparation worker affinity never holds a capsule-wide slot
  across child or RPC waits. The pool is divided only by lease priority (`cpu_priority`:
  AUTHOR 1, GRADE 2, PREPARATION 3, AUXILIARY 4, VERIFICATION 5; `MERLIN_CPU_PRIORITY` carries
  it to children). A cross-process `waiters/` queue in the slot directory serves more urgent
  classes first and FIFO within a class; no CPU is reserved for a class, and dead or stopped
  waiters never block. Tickets record eligible CPUs; matching across the complete queue
  preserves earlier feasible waiters while letting unrelated free CPUs run, including
  flexible-head/singleton combinations. Legacy empty tickets remain conservative. Fresh
  post-flock checks re-prove admission before use. Unset engine caps default to
  `compute_capacity()`.
