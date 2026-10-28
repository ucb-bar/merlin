---
name: heavy-pass-recipes
description: "Runnable, verified recipes for heavy scheduling-library passes, tiles from RTL-fact capacities and every command stream checked on host integers against the source: readout overlap and ready readout for an accumulator-bound GEMM, the segmented pipeline for long K, convolution through window row runs with retained input and resident weights (no per-pixel gather, no zero-fill-then-overwrite rows), convolution compute epochs and dense panel epochs that drain only where RTL ordering facts leave rows unordered, a conv-requant-pool chain via the composition catalog, a per-case list of composition routes and source_composition.json requests, and a shape_policy example choosing registered PipelineVariants from --panel results. Use before hand-writing a convolution or GEMM schedule, when a lowering issues a transfer per output pixel or drains after every compute or K step, when a dense path picks its plan with an if on M, N or K, or when the brief names one of these passes."
---

Workspace: `--facts FILE` (else synthetic), `--library DIR` (default
`submission/compiler_library`); JSON, `--help`.

| recipe (`SKILL_DIR/scripts/`) | applies when | library chain | read |
|---|---|---|---|
| `dense_readout_recipe.py` | serialized readouts; `--family retained` also accepts one K tile | `schedule_dense_pipeline`: separate, batched, `dense_readout_overlap`, `dense_ready_readout` | `reference/dense-recipes.md` |
| `source_observer_recipe.py` | closed i8 observations | typed observer -> interval/finite proofs | `reference/dense-recipes.md` |
| `projection_recipe.py` | shared immutable input | retained LHS -> `projection_views` -> `projection_dense` | `reference/dense-recipes.md` |
| `long_k_recipe.py` | full-K panels exceed operand store | `schedule_dense_pipeline`, `dense_segmented_pipeline`; resident LHS panel | `reference/dense-recipes.md` |
| `conv_window_recipe.py` | any convolution | `window_transfer.CountedConvolutionPlan` -> `convolution_exploration` (input retention, resident weights); `rolling_window` row staging | `reference/convolution.md` |
| `conv_epoch_recipe.py` | convolution draining after compute/store | ordering facts -> epochs; source-qualified readout law -> checked consumer; completion-order check | `reference/convolution.md` |
| `dense_epoch_recipe.py` | K drains | ordering facts -> `DensePanelEpochLaw` -> `schedule_dense_pipeline(epoch_law=)` | `reference/dense-recipes.md` |
| `shared_rhs_epoch_recipe.py --describe` | immutable shared RHS | V24 `schedule_dense_shared_rhs`; consume `schedule.plan` | script; `ab-library-pass` |
| `composition_routes.py --case NAME` | pool, epilogue, residual, siblings or heads | `composition_catalog` routes; `source_composition.json` options | `reference/source-composition.md` |
| `spatial_chain_recipe.py` | a convolution feeding a pool | `composition_catalog` -> `rolling_spatial` | `reference/composition.md` |
| `library_pipeline_policy.py` | pipeline choice per shape | `PipelineVariant` + `merlin.transforms.shape_policy` | `reference/pipeline-policy.md` |
| `counted_emission_recipe.py` | reuse encoder | checked `.emit_index_program`: same effects/completion, counted loops | `reference/dense-recipes.md` |
| `parallel_schedule_recipe.py --output DIR` | shape variants | registered heavy passes -> unique emission -> bounded `evaluate_emitted` | `reference/parallel-evaluation.md` |

1. **Run** (`--shape`; `--count-only` for large convolutions). Counts describe work; measure cycles.
2. **Bind your lowering.** Keep source checks. Replace neutral `encode`/`project` and test-only laws
   with selected declarations. Drains invoke real completion; refused/UNKNOWN keeps the original.
3. **Convolution.** `emit_checked_readout_epoch` consumes `ConvolutionReadoutEpochLaw` and checks
   actual grouped/wide consumer receipts. Derive ACC visibility, configuration capture, queue
   progress and ownership from selected contracts. Load OR zero-source each row; never overwrite
   an unfinished zero fill. Removed drains require ordering evidence.
   `conv_epoch_recipe.py --write-toy-contract FILE`, then `--adapter-contract FILE --output DIR`,
   emits ISA/SCF with toy laws and unranked alternatives. `generated_effect_callbacks`
   binds caller instructions, without backend or physical authority.
4. **Register alternatives.** A `PipelineVariant` binds passes and a legality predicate.
   `assess_shape_policy` consumes panel cycles; `AssessedShapePolicy.execute` applies the selected
   passes to current source. Missing evidence keeps the incumbent.
   `shared_rhs_epoch_recipe.prepare` shares authored `tools/shared_rhs_owner.py` across both arms.
   Bind products, aliases, outputs and laws; retain LHS reuse. `--describe` supplies metadata;
   execution uses ordinary A/B checks.
5. **Cycles decide.** Use the panel (`explore-schedules`) and `ab-library-pass` for affected classes.
   `parallel_schedule_recipe.py --adapter` uses ordinary shared CPU limits and caller-owned
   numerics/cycle callbacks. Neutral compilation is not a PASS; UNKNOWN ranks on `fast_estimate`.

Contracts: `scheduling-api`. Choices: `shape-schedule-selection` / `capacity_pipeline_example.py`.
