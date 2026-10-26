---
name: heavy-pass-recipes
description: "Runnable, verified recipes for the heavy scheduling-library passes, tiles derived from RTL-fact capacities and every emitted command stream checked on host integers against the source: readout overlap and ready readout for an accumulator-bound GEMM, the segmented pipeline for long K, convolution through window row runs with retained input and resident weights (no per-pixel one-row gather, no zero-fill-then-overwrite rows), a conv-requant-pool chain found through the composition catalog, and a shape_policy worked example where --panel results choose between registered library PipelineVariants instead of hand-written if cutoffs. Use before hand-writing a convolution or GEMM schedule, when a lowering issues a transfer per output pixel, when a dense path picks its plan with an if on M, N or K, or when the performance brief names one of these passes."
---

`SKILL_DIR` = the directory holding this file. Run from the workspace root. Every recipe reads
`rtl_facts/facts.json` if present (else a small synthetic body, labelled so in `"facts"`), imports
`submission/compiler_library` (or `--library DIR`), prints JSON, answers `--help`, and fails if an
emitted program does not reproduce the source result. The preflight runs all of them on every launch.

| recipe (`SKILL_DIR/scripts/`) | applies when | library chain | read |
|---|---|---|---|
| `dense_readout_recipe.py` | the output needs more accumulator rows than the store has | `schedule_dense_pipeline` on one K-segmented block: separate, batched, `disjoint` (`dense_readout_overlap`), `ready` (`dense_ready_readout`) | `reference/dense-recipes.md` |
| `long_k_recipe.py` | a full-K LHS+RHS panel pair no longer fits the operand store | `schedule_dense_pipeline`, K-segmented (`dense_segmented_pipeline`), LHS window = one full-K panel | `reference/dense-recipes.md` |
| `conv_window_recipe.py` | any convolution | `window_transfer.CountedConvolutionPlan` -> `convolution_exploration` (input retention, resident weights); `rolling_window` row staging | `reference/convolution.md` |
| `spatial_chain_recipe.py` | a convolution feeding a pool | `composition_catalog` -> `rolling_spatial` | `reference/composition.md` |
| `library_pipeline_policy.py` | choosing among library pipelines per shape | `PipelineVariant` + `merlin.transforms.shape_policy` | `reference/pipeline-policy.md` |

1. **Run the recipe for your operator** on its derived shape, then on your layer
   (`--shape`; `conv_window_recipe.py --count-only` for a large layer). Read the counts it prints
   next to the baseline: epoch drains, overlapped compute pairs, input transfers and rows per
   transfer, weight loads, and the count a per-pixel one-row gather would issue.
2. **Move the chain into your lowering.** Keep the derivation (capacities -> plan -> library calls)
   and the host check as a unit test; replace the neutral adapter (`encode`/`project`) and the
   test-only law identities with your target's own declarations. A pass that refuses names the
   premise that failed; that refusal is where your next alternative applies, not a bug to bypass.
3. **Convolution.** Interior taps move `row_tile` input rows in one strided transfer; weights load
   once per invocation when resident; one input transfer serves every live output-channel tile.
   Padding is either a row that is loaded OR zero-sourced (once, never zero-filled then
   overwritten), or a zero border materialized once on the host so every tap is in bounds. The
   recipe's hazard check fails on any local row written twice before a compute reads it.
4. **Register alternatives instead of `if`.** Each pipeline is a `PipelineVariant` of registered
   library passes with its legality predicate; per-shape choice comes from panel cycles through
   `assess_shape_policy`, and `AssessedShapePolicy.execute` runs the selected passes on the current
   source. Missing or incomplete evidence keeps the class incumbent.
5. **Cycles decide.** Command counts and drains explain a result; change a default only after the
   panel (`explore-schedules` skill) and `ab-library-pass` show it on every affected class.

Contracts and adapter: the `scheduling-api` skill. Choice tables over knobs: the
`shape-schedule-selection` skill, whose `capacity_pipeline_example.py` these recipes reuse.
