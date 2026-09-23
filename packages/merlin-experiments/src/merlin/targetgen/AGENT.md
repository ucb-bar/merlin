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
