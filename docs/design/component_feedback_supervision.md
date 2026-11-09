---
title: Component feedback transport and supervision
kind: design
status: current
owner: experiments
last_verified: 2026-10-08
related: [component_independent_feedback, component_phase2_workflow]
code_refs:
  - src/merlin_experiments/phase2/feedback_protocol.py
  - src/merlin_experiments/phase2/feedback_guardian.py
  - src/merlin_experiments/phase2/supervised_feedback.py
  - src/merlin_experiments/execution/owned_children.py
  - src/merlin_experiments/phase2/component_workflow.py
---

# Component feedback transport and supervision

The normal component workflow runs analytical and CCA callbacks under a separate
invocation guardian. Forking preserves the live issued evaluator objects. These
objects are never reconstructed from JSON, import strings or a reference compiler.
The selected compiler commands still require their original admitted inner sandbox.

## Closed result protocol

The worker serializes plain scalars, lists, tuples, maps, `CycleInterval`, and the
existing analytical result envelope. Unsupported objects refuse in the worker;
no arbitrary reconstruction or pickle result crosses into the parent. Encoding
shares the callback budget. Reception caps actual file bytes, then bounds syntax
nesting and structural units before parsing. The decoder additionally limits
nodes, depth, integer size, finite numbers and string size.

The parent checks its absolute deadline before reception, during bounded walks,
after decoding and after cleanup. A file published while the parent is paused
cannot authorize a late result. These checks reject late completion; they do not
make local filesystem calls, JSON parsing or parent scheduling interruptible.
A hard OS wall deadline remains UNKNOWN.

## Persistent invocation child ownership

The newly forked guardian becomes a Linux child subreaper before creating the
callback worker. It monitors the exact parent and worker with pidfds, retaining
ownership of adopted double forks and separate sessions. The shared
`execution.owned_children` primitive terminates direct/adopted children through
pidfds and actually reaps them with `waitpid`. ECHILD closes the owned tree.
A `/proc` direct-child roster locates unreaped children; that roster alone never
proves cleanup. This extraction preserves the existing native guardian behavior.

The guardian has its own deadline and parent-loss observation. Cancellation and
completion use a bounded private socket. Cleanup completion is acknowledged
before socket closure, preserving the final packet even if cancellation arrives
while cleanup is finishing. Parent admission additionally requires the actual
retained guardian product and actual guardian reaping.

The callback guardian is forked directly and reaped through its pidfd and
`waitpid`. A multiprocessing pipe sentinel can stay open in forked descendants
after the guardian dies; pipe EOF is therefore not its exit/ownership evidence.

The proved scope is this invocation's direct and adopted descendants. Processes
submitted to external services are outside that tree and remain UNKNOWN. Actual
host probes support subreapers and pidfds; user/PID namespace creation refused.
These controls do not qualify namespace isolation, hostile same-user signaling,
host/service failure, or interruptible parent reception.

## Parent resource lease

For normal analytical feedback, the controller acquires the existing declared
engine lease before creating the guardian. The guardian retains a custodial copy
until ECHILD; the callback worker closes its copy before running any callback.
A process-local exact delegation prevents the analytical evaluator from acquiring
that same lease again. Its own descendants cannot inherit that scheduling role.
Direct diagnostic evaluator calls retain their older worker-owned lease scope.

On normal completion the parent closes its lease only after the exact cleanup
product and guardian reaping. If the coordinator dies, the guardian's descriptor
keeps the lock held through actual cleanup. If cleanup is unresolved, the guardian
retains child ownership and its custodial lock while retrying; a live parent
retains its handles as a private quarantine. No snapshot or supplied PASS file
permits release. No general release/recovery capability is issued by this slice.
A lease excludes only callers honoring that selected lock, not all machine work.

## Retained evidence and controls

Private lifecycle v2 records bounded reception, worker and guardian identities,
actual exit/reaping, cleanup product SHA, declared lease release, elapsed time,
failure and owning source pins. Guardian products remain beside result or partial
bytes. Historical v1 receipts retain their UNKNOWN cleanup/release meaning.
File pins alone do not authenticate arbitrary loaded interpreter/source closure.

Actual controls exercise a double fork in a new session, stable descendant pidfd
observation, lease exclusion and absence of the lease descriptor in the worker,
coordinator SIGKILL with the custodial lease held until cleanup, stalled callbacks,
excessive results, unsupported reconstruction, and late coordinator resumption.
The baseline reproduction returned while its orphan was alive and inherited lock
remained busy; a disposable test subreaper killed and reaped only that owned orphan.
No shared user processes were signalled.

Candidate SHA checks and private mutation/refusal evidence remain in place. There
is no automatic restoration: concurrent candidate ownership and restoration are
separate unresolved contracts. This lifecycle changes no prompts, cohorts,
statistical gates, accuracy thresholds or physical measurement authority.
