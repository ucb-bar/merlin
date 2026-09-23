# AGENT.md — src/merlin/integrations

Owns upstream discovery/import/process adapters, not upstream compiler semantics.
Keep PyTorch/model2MLIR capture separate from compiler Python bindings and ModeLIR
execution. Import availability checks must not mutate process-wide paths permanently.

`model2mlir` owns capture checkout aliases and the shared framework-capture interpreter.
Keep per-workload capture.toml selection and compiler-only Python policy separate.
Optional quant dialect imports retain class identity; conflicting preloaded checkouts
are unavailable, never purged or reloaded to simulate multi-checkout isolation.

`modelir` owns scoped imports and the single artifact cwd lock. Retained oracle imports,
new-module eviction for discovery, and serialized artifact execution are distinct policies.
Preserve configured-root resolution inside the artifact lock; never move numerical/oracle
semantics here. This is not multi-checkout or noncooperating-thread isolation.

`specir.importable` scopes checkout paths while retaining normal module identities.
Discovery remains caller-owned: capture has its sibling fallback, Phase 0 may use an
installed package. Explicit roots reject conflicting cached owners, never reload them.
Phase-0 startup receipts and golden-cache identity include this adapter's source bytes;
that does not establish a commitment to every upstream SpecIR source dependency.
