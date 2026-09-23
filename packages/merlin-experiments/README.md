# Merlin experiments

Owns versioned phase definitions and process orchestration (`merlin_experiments`),
and trusted evaluation and agent isolation (the stable `merlin.targetgen` shared
namespace). Core owns package invocation, schemas, and declarative engine policy;
this distribution owns graders, independent golden evaluation, and the sandbox.
Transcript accounting and the AET bridge also live here under their stable
`experiment_tokens` and `aet_bridge` imports. The former targetgen Claude CLI
slot was uncalled duplicate transport and is retired; Phase 1 owns agent launch.
Descriptor-owned post-search evaluation is exposed as `merlin-evaluation-cohort`.
Native reduced-source A/B execution lives in `merlin.perf.source_program_pair_provider`;
its exact worker grants and answer-mask enforcement remain experiment-owned.
Compute-group corpus writing, promotion and evaluation (`merlin.targetgen.group_capsules`)
and measured capacity sweeps (`merlin.targetgen.store_probe`) also live here.
Core compiler diagnostics use the deterministic `group_capsule_entries` builder;
the research module reexports that same function without copying its implementation.
Numerical falsifiability audits and answer-bearing RTL replay construction also belong
here (`numeric_falsifiability` and `rtl.gen_rocc_replay` under the stable targetgen
namespace). Their output can contain private expected values and remains host-only.

`merlin experiment list` starts at the checked-in experiment catalog. Inspect and
preflight never launch agents. Run and resume invoke the existing phase engines;
their scoring, AET records and certification policies remain authoritative.
The optional package does not introduce a second telemetry store.

Existing grader imports remain their canonical module identities, not duplicated
wrappers. Core extends the namespace; this wheel never replaces its initializer.
Source relocation remains subject to the core access registry and the same
fail-closed candidate import and filesystem-mask checks.

Phase 0 emits a run-owned corpus. `merlin experiment corpus prepare`, `inspect`,
and explicit operator `seal` provide the reviewed handoff described in
[`experiments/README.md`](../../experiments/README.md). A phase-1 definition must
select both the sealed descriptor and its `corpus_seal` input. Combined phase 0/1
selection remains refused: automation cannot supply the human review. Phase 2 catalog templates require operator
inputs and are not runnable examples with fabricated compiler certificates.

The Phase 0 engine lives in `merlin_experiments.phase0`, partitioned into profile
loading, sweep expansion, independent numerics, caching, writing and provenance.
Installed invocation needs explicit external profiles, descriptor and output:

```sh
python -m merlin_experiments.phase0 --target PROFILE \
  --profiles-root /absolute/profiles --descriptor /absolute/target_experiment.yaml \
  --output-root /absolute/run/capsules
```

For recorded runs, use `merlin experiment run` with `config.profiles_root` and
`config.descriptor`; the runner supplies its run-owned output. It freezes every
implementation Python file and the complete profiles directory, including hidden
sidecars. Neither private data nor target qualification is supplied by installing
this wheel. The old generator script remains only a checkout CLI launcher.
The staged public self-check client is package-owned and uses only the standard
library, so it needs no Merlin imports inside an agent workspace.

## Oracle metadata without evaluation

Core corpus derivation uses `merlin.targetgen.oracle_policy.oracle_tier_plan`, not
grader adapter construction. Explicit profile `required_oracle_tiers` remain
authoritative. Plans describe the selected/advertised tier inventory and engine
evidence, not necessarily the constructed adapter inventory. They are not
execution results or certificates.

Support plugins opt in through `plugin.sim_oracle_metadata`, a registration module
that imports `register_sim_oracle` and `OracleTierPlan` from core `oracle_policy`
and supplies a `tier_plan(target)` callback. It must not import the optional grader
or instantiate execution adapters. The same import-light module may serve both
`sim_oracle` and `sim_oracle_metadata`; both discovery paths share its identity.
Keep tier selection shared between this callback and the plugin's adapter factory.
Requirement inference additionally needs `requirements_inference_safe=True` on
the plan, explicitly promising that construction cannot silently change its tier
keys. The default is false. A plugin that catches constructor failures and drops
selected tiers cannot make this promise; it must require explicit profile tiers.
For example, Muon's selected RTL tier remains advisory metadata for inference
because its evaluator can drop that tier after a factory failure. The existing
Radiance profile's explicit `[L0, L1, L2]` requirements remain unchanged.

Legacy adapter-only plugins still evaluate unchanged. Core-only tier inference
refuses unknown declared engines, missing callbacks, or unproved inference safety with an actionable
error; it never guesses tiers or substitutes ARC for a missing declared plugin.
An existing plugin can add metadata or the corpus author can declare explicit
required tiers. Metadata callbacks are trusted support code, not candidate grants.
