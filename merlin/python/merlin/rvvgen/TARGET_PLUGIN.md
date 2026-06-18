# Target-plugin contract for the beam/fork engine (S5.8)

`rvvgen.beam.run_beam` is the **target-agnostic** beam-search engine: each generation it expands
every surviving parent into forks (via a *proposer* over the S4 divergences), certifies the whole
generation in parallel + isolated (`rvvgen.sweep.run_sweep`), ranks (correctness -> structural
match toward the expert -> cycles), keeps top-k as parents, and writes the full lineage to
`beam_tree.yaml`. RVV is the **first instantiation**, not a hard dependency — every target-specific
behavior is an injectable callable with an RVV default.

A new target reuses `run_beam` **unchanged** by supplying four callables. Nothing else needs to be
touched.

## The four seams

| `run_beam` param | RVV default | Contract |
| --- | --- | --- |
| `loader` | `registry.load_rvv_package` | `loader(package_dir) -> pkg`. `pkg` MUST expose `.run_id: str` and `.knobs: dict`. The knobs dict is the tunable surface, passed to the proposer and to the minter. |
| `minter` | `from_strategy.mint_fork` | `minter(parent_pkg, overrides, *, version, depth, timestamp, source_evidence, lever, target, out_root, generated_by_agent) -> Path`. Deep-copies the parent knobs, applies `overrides`, **renders the target artifact from the merged knobs**, and writes a lineage-stamped fork package dir. This is where the target's own *render/generator* lives. |
| `proposer` | `kernels.rvv_knobs.propose_forks` | `proposer(divergences, knobs) -> list[ForkProposal]`. The lever/tuning policy. The LLM tuning-agent (`rvvgen.tuning_agent.propose_forks_llm`) is a drop-in alternative. See `ForkProposal` below. |
| `certify_fn` | `runner.certify_rvv` | `certify_fn(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir) -> dict`. Build + run + gate one fork. Result dict consumed by the engine: `result["correctness"]["gate_ok"]: bool`, `result["measurement"]: [{target, cycles, ...}]`, and (mock path) optional `structural_match`/`divergences`. |

`target` (str) names the dialect; it only flows into `mint_run_id` (`<target>_tuned_v{v}_d{d}_{ts}`)
and the manifest, so fork packages for different targets accumulate under
`generated_targets/<target>/` without collision.

## ForkProposal (the proposer's return type)

`kernels.rvv_knobs.ForkProposal` — shared across all proposers (deterministic + LLM):

```python
ForkProposal(
    overrides: dict,   # knob overrides applied to the parent (empty for non-actionable)
    lever:     str,    # "knob" | "lowering_pattern" | "llvm_requirement" | "llm_suggestion" | ...
    targets:   str,    # which divergence/decision this addresses
    evidence:  list,   # mined-policy / kernel ids justifying it
    forkable:  bool,   # True => beam mints+certifies; False => recorded as a deferred work-item
    note:      str = "",
)
```

The engine mints + certifies only `forkable=True` proposals (up to `width` per parent); every
`forkable=False` proposal is recorded under `deferred_work_items` in `beam_tree.yaml` so a missing
lever (one the renderer can't express today) is surfaced, never silently dropped.

## What is intentionally RVV-specific (do NOT reuse cross-target)

- `from_strategy.render_schedule` + `from_strategy._VARS` — the **RVV renderer**: it emits the RVV
  transform-dialect schedule (per-op `tile`/`vectorize` + `transform.apply_patterns.vector.*`
  lowering, `contraction_strategy`, `dtype_strategy`). A different target supplies its own
  `render_schedule`/`minter`. The renderer knob vocabulary is enumerated in
  `tuning_agent._KNOWN_OVERRIDE_KEYS` (`op_match`, `contraction_strategy`, `lowering_patterns`,
  `dtype_strategy`); the LLM proposer clamps to it so an unrenderable knob is never emitted.
- `kernels.compare.RvvFingerprint` / `compare_fingerprints` — the structural scorer is RVV-objdump
  based. The engine uses it for the real (non-mock) `structural_match`; a new target with no objdump
  fingerprint can return `structural_match` directly from its `certify_fn` (the mock path), or wire
  in its own fingerprint scorer.
- `fork.write_fork` is largely generic (it writes `schedule.mlir` + `knobs.yaml` + `manifest.yaml`);
  its `family: "vector_schedule"` / `schedule_format: "transform_dialect_mlir"` manifest fields and
  the `schedule.mlir` filename are RVV-flavored conventions a different minter may override.

## Minimal new-target checklist

1. A **package loader** returning an object with `.run_id` + `.knobs`.
2. A **render/generator** + **minter** that turns merged knobs into the target artifact and writes a
   lineage-stamped fork dir (model `from_strategy.mint_fork` / `fork.write_fork`).
3. A **proposer** (deterministic gap-router and/or the LLM `propose_forks_llm`) over your target's
   knob vocabulary.
4. A **certify_fn** returning the gate/measurement dict shape above.

Then: `run_beam(seed, ..., target="mytarget", loader=..., minter=..., proposer=..., certify_fn=...)`.
