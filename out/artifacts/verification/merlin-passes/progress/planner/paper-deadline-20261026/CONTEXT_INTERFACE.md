# Shared context decisions — implementation and review in progress

This records coordinator decisions for Q/F/W and the independent reviewer.
It does NOT authorize native execution or qualify any new graph source.

F owns `lean_graph_original_image_context*.py`; Q owns gate consumption; W owns
actual-after queries. Independent reviewer owns source/data findings only.
Root retains concrete new-context review/authorization. Historical contexts,
queries, expected pins, failures and qualification capsules remain immutable.

## Intended API

`Context.from_review` consumes exact reviewed metadata/copy/basic-probe/trace
references, an explicit module-to-source-reference map and a private workspace.
`check_query(module, query_bytes, final_root=..., support_roots=...)` returns a
process-issued `CheckedQuery` binding actual query, source/import/object closure,
exact final-root type, transitive standard-three audit, context and raw evidence.
`revalidate(CheckedQuery)` must recheck the SAME live result and complete bindings
immediately before gate admission. Caller dictionaries/JSON/callbacks do not
manufacture proof authority. This API is source under development, not accepted
native behavior.

## Source and runtime bindings

Use only the independent private ORIGINAL images delivered at
`out/build/graph-paper-deadline/original-runtime-candidate-v1/toolchain`.
Every consumed SDK/runtime object and sidecar must already have an ORIGINAL pin
in the reviewed selection. No new hashes, expanded SDK selection, installed-SDK
reads, inherited failed-context guards or fallback lookup. Project source and
fresh compiled object closure must be exact and stable across import reporting,
compilation, checking, audit and final revalidation. Bind actual commands,
environment, pre/postchecks and all claimed evidence by content.

Compiled-only STANDARD modules are an explicit new policy. Actual `--deps`
objects/sidecars require existing original pins. Hypothetical canonical
`--src-deps` .lean paths are module-name cross-checks, NOT source-byte evidence;
never open an unselected source. Project source/object correspondence remains
exact. Unresolvable/unconfirmed reporter output stops the context. Pinned stdlib
binary delivery, native Lean loader/kernel, OS and host association remain trust.

## Pure declaration query and separate audit

Q/W must emit a NEW query version whose compilation produces empty stdout.
In-source `#print axioms`/other output commands are not part of that compilation
contract. Context separately constructs and binds exact final-root type output
and transitive standard-three axiom audit. Do not remove the audit, accept
arbitrary output, or strip commands from/rewrite historical retained queries.
Record the new query/source identity and preserve earlier candidate versions.

The independent reviewer found that Q's original query embedded `#print axioms`
while Context rejected any compile output. Q/F are resolving that concrete
integration blocker; W has the same consumer contract. No positive native test
has yet demonstrated the revised integration.

Some FROZEN dependency libraries themselves contain `#print axioms`. Preserve
their exact sources. A separately explicit `compile_audit_roots` map may name
only those frozen project dependency modules, with exact source refs and root
lists bound BEFORE launch. Consume the WHOLE compile stdout with the frozen
exact-root/std3 parser; no warnings/arbitrary output, observed-output learning
or dynamic root discovery. Generated query modules must be disjoint from this
map and retain empty compile stdout. Test name-collision and bypass cases.
This is a source-reviewed design exception for library audit commands, not a
native authorization or a modification of a historical evidence policy.

## Failure classification

Recognized ordinary proof rejection is separate from infrastructure failure.
Do not classify any source-position `error:` as a proof refusal. The independent
reviewer reproduced interpreter/load/heartbeat/OOM-style diagnostics accepted by
the earlier broad classifier in SDK-free probes. F owns the correction and
regressions; reviewer rechecks the exact new cut, including mixed buffers.

Unknown diagnostics, resource exhaustion, mismatched images/associations,
signals, timeouts and unreviewed runtime observations permanently stop that
context and retain its first failure. No unchanged retry, policy widening or
promotion to semantic-negative evidence. A corrected source policy gets a NEW
identity; earlier failures stay visible.
