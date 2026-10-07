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

Capture-derived stages share ONE grouping (`compute_groups` + `group_command`, stated by core
`group_capsule_entries`) and ONE binding (`group_capsule_entries.group_binding`, the corpus binding
with a derived, never literal, accumulator). `model_forms` derives functional MF forms and
`form_perf` the form-perf scope, both at `corpus derive` from the declared ITERATION captures only;
held-out models (claim and evaluation-only, `claim_boundary.held_out_models`) are refused by name.
`group_forms.write_group_capsule` is the only path a group capsule reaches disk. `instruction_roles`
derives the role taxonomy and resolves the experiment's `prohibited_instruction_roles`; it declares,
it does not enforce. Form-perf members, their coverage and the claim-model statistic never read a
claim model's capture before the Phase-1 freeze, and never write a capsule from one.
