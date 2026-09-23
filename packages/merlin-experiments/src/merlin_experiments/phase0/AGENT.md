# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase0

Profiles declare tests; sweeps expand them; writer constructs and validates capsules;
numerics owns the independent golden engines. Generation orchestrates those owners,
and provenance records emitted inputs without exposing hidden member names.

Keep numerical function bodies and ordering stable during structural work. Inputs
are external: never bundle profiles, holdouts, or generated goldens in this package.
All runs require an explicit output destination; never default writes to the
descriptor's source corpus. Both checkout and installed runs require explicit recipe
inputs or a legacy profiles directory; the former in-tree recipe default is retired.
Installed runs also require an explicit descriptor.

`load_profile` / `generate_target` support explicit `recipe`, `performance_template`,
`synth_profile`, `smt_profile`, and `hidden_profile` paths. Recipe mode requires the shared
template and never discovers siblings; it cannot be mixed with legacy `profiles_root`.
Missing optional declared sidecars mean omission. Public readers use `include_holdouts=False`,
which must not even stat the hidden path. Merge order remains public, shared performance,
synthesis, SMT, hidden. Frozen orchestration binds these inputs and optional-file absence;
the loader does not invent another seal or target-discovery registry.

The cache and experiment runner must bind the complete package source closure, not
only the CLI or numerical entry function. No private-helper compatibility facade:
tests and maintenance callers import and patch the authoritative owner.
