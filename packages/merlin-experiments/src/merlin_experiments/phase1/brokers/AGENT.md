# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase1/brokers

ISA, CCA and synchronous/asynchronous feedback channel loops are owned here. These
are trusted host implementations, never candidate grants. Public standalone
clients remain under `merlin.benchharness` and are staged by the tool registry.

ISA invocation requires an explicit descriptor/work root or InvocationContext.
Imports and help must not select a target or mutate environment/search paths.
BrokerCtx owns per-invocation lazy model callbacks; schedule contracts reload on use.
Keep debug output redaction delegated to the existing program-oracle implementation.

Preserve request/response formats, error text, restart and STOP/parent-death behavior.
`selfcheck_sources` owns which tree a self-check grades: `submission/` or an author-side
`variants/<name>/`, opened without following symlinks. Only `submission/` is official; a variant
response is labelled `official: false` and never reaches promotion, freezing or foreground markers.
Grading, redaction, promotion and enqueue-time source recovery belong to `phase1.feedback`.
Never rediscover an explicitly supplied public root. Promotion has a separate policy-root
input, preserving descriptor-defined coverage rather than silently narrowing it to a QA subset.
Do not import native launchers or imply full-controller/hardware qualification.

The selfcheck dispatcher supervises the original request deadline even while its
handler prepares frozen inputs before entering its child poll loop. Reap only that
owned process tree before refilling the lane; preserve existing durable completion.
Conservative suite budgets cannot extend the selected campaign clock. Timeout
records describe incomplete work, never numerical failure or completed coverage.

In explicit `snapshot_external_v1`, selfcheck brokers do not construct the legacy
common-pass proof service; those proof actions remain unavailable. Model-check
children accept the separately reviewed full compiler selection only with the
original bundle/context and a valid staged source tree. This never grants legacy
proof authority or changes the external numerical evaluator.
