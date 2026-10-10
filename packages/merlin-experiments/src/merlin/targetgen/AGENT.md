# Trusted experiment evaluation

Contains evaluation implementations, not generic compiler package invocation.
Package/runtime primitives and stable error identities live in core package_runtime.
The common.access registry must name every withheld implementation before moving
it. Preserve grading, failure-plane, AET record, and monkeypatch behavior.
No competing targetgen/__init__.py: core extends that package path.
`group_capsules` owns corpus writing/promotion and grading; it reexports the exact
deterministic entry builder from core `group_capsule_entries`. `store_probe` owns
measured backend capacity sweeps. Neither operation is compiler-core metadata.
Golden evaluation keeps local compatibility adapters for provenance queries:
`capsule_golden._load_golden_yaml` and `capsule_golden.golden_source` overrides
must still affect subsequent queries. Shared metadata policy lives in core
`golden_provenance`; direct reexports would change the legacy globals lookup.
Production codegen smoke kernels and prerequisites belong to selected OOT support
backends. `capsule_runner.codegen_smoke` only dispatches and validates tri-state
evidence: broken providers refuse admission; missing coverage is never a pass.
Simulator identity and ISA class must not select a target-specific compiler.
Prompt queries select an already-loaded evaluator's callbacks without importing
the optional evaluator; its adapters must delegate to pure queries, not selectors.
`numeric_falsifiability.audit_outputs` checks the existing constant candidates
against already selected outputs. Strict callers must report unavailable,
non-finite or unsizeable oracle outputs as unmeasured, never as an assessed pass.
The public CI gate may explicitly allow a partial audit; it never changes a
numeric policy, writes an answer key or grants full corpus numerical coverage.
The integer capsule golden must refuse noninteger operand formats when no
matching independent golden was selected. Integer surrogate stimuli cannot
establish fidelity or falsifiability for a declared floating program.
Only typed `UnavailableGolden` refusals may pass an explicitly partial audit.
Malformed stored outputs or archives and evaluator failures stay hard failures.
Candidate mandatory-tier re-audits inherit the trusted invocation's explicit
ReadbackPolicy from the model context and selected oracle adapters. Default and
B64 reopen text; selected binary readback reopens raw bytes and checks the
complete declared value roster before ordinary numerical comparison. A result
record, console filename or apparent byte format cannot select the transport.
Decoding full values grants no build, source, dispatch or hardware authority.

The selected native component route spends one explicit wall deadline across
package build, all ordinary lowering commands, source checks, object/link,
execution, original output comparison and result publication. Failed attempt
evidence is retained after expiry; it cannot become a completed result.
Boundary checks do not preempt arbitrary in-process callbacks or confer any
semantic, physical or performance authority.

`native_component_inputs` owns the shared error identity and ordered tensor
binder reexported by `native_component_execution`; both import orders must work.
An explicitly selected original member supplies exact typed input storage and
the original full-output comparator without entering legacy golden dispatch.
Reopen that live owner before and after ordinary lowering/execution and keep
reference details in private evidence; caller-visible original failures contain
no expected values or private paths. Opaque pointer arity does not prove flat
tensor storage or a ranked memref descriptor bridge.
