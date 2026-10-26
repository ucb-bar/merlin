# Dense recipes: accumulator-bound output and long K

Both recipes share one upstream with the shape-schedule-selection skill's
`capacity_pipeline_example.py`: RTL facts -> `merlin.targetgen.address_space` capacities (array edge,
operand and accumulator stores) -> `DenseProblem` with the array edge as the primitive tile ->
`DenseLoopStorage`/`DenseResources` whose byte budgets ARE the store windows, so `dense_blocks`
offers only blocks that fit. Nothing below is a number written into the lowering: every bound is a
property of the stores and the shape.

## Accumulator-bound GEMM (`dense_readout_recipe.py`)

**When.** The whole output needs `tiles_m * tile_m * tiles_n` accumulator rows and that exceeds the
accumulator store. The schedule must then compute, read out and drain output groups in turn; the
readouts and drains serialize behind compute.

**Chain.** One typed K-segmented plan through the single checked entry, once per readout mode (this is
`recipe_support.segmented_pipeline`; the shape-schedule-selection cookbook shows the same entry
for every family):

```python
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline

plan = make_dense_schedule_plan(p, rs, s, adapter, source_identity=src, dtypes=dtypes, static_budget=128,
                                block_tiles=(b.m, b.k, b.n), pipeline_kind="retained_lhs_segmented_k",
                                pipeline_prefetch=True)
result = schedule_dense_pipeline(p, rs, s, adapter, name="seg", dtypes=dtypes,
    schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)), source_identity=src,
    external_buffers_disjoint=True, pipeline_admission=segmented_law,   # DenseSegmentedAdmission
    batch_readouts=True,                                                # readouts grouped
    readout_mode="disjoint",          # or "ready" / "separate"
    readout_admission=readout_law,    # DenseReadoutComputeAdmission / DenseReadyReadoutAdmission
    readout_group_path=("outer_m", "outer_n"))   # disjoint only; () otherwise
result.emit_llvm(builder, lower_call)  # lower_call(instruction, operands, metadata=None)
```

`disjoint` is `dense_readout_overlap` (readout of one accumulator window overlaps compute into the
other); `ready` is `dense_ready_readout` (a store follows its final K reduction inside the compute
epoch). The entry executes or refuses every explicit decision; it never substitutes another block or
family.

**Block premise.** At least two K segments (`b.k < tiles_k`), the output block inside ONE of two
accumulator windows (`b.m * b.n * tile_m <= accumulator_rows // 2`), and more than one output group
along N (`tiles_n // b.n > 1`). The recipe takes the legal block with the fewest requested loads;
with real alternatives, time the legal blocks instead of fixing a rule.

**Reading the output** (synthetic facts: edge 4, 128 operand rows, 32 accumulator rows; shape
7x12x19 needs 40 accumulator rows):

```
segmented          epoch_drains 20
batched_readouts   epoch_drains 13
readout_overlap    epoch_drains 12   disjoint_readout_epochs 1
ready_readout      epoch_drains 10   causal_ready_epochs 2
```

Same products and stores in every variant (the host check compares each with the source product).
`disjoint_readout_epochs`: epochs where the readout of one window runs while compute fills the
other. `causal_ready_epochs`: epochs where a store follows its final K reduction without a drain.
Which is faster depends on your store path and queue depths: measure both. The two need DIFFERENT
target laws; declare only what your hardware guarantees (a readout that may observe a pending
accumulator write needs the causal law, not just ordered issue).

## Long K (`long_k_recipe.py`)

**Two bounds from the operand store** (R rows, array edge E):

| bound | value | meaning |
|---|---|---|
| `full_k_panel_pair_max_k_tiles` | `R // (E + E)` | one M tile's full-K LHS panel and one N tile's full-K RHS panel fit together: the retained full-K panel pipeline applies |
| `segmented_max_k_tiles` | `(R - 2E) // E` | a full-K LHS panel fits next to two minimal RHS K segments: the segmented pipeline applies |

Between the bounds, the K-segmented family of `schedule_dense_pipeline` keeps the LHS panel resident and
double-buffers RHS K segments; prefetch of the next segment overlaps compute
(`overlapped_compute_pairs`). The operand-store split is part of the plan: the LHS window is one
full-K M panel (`lhs_rows = tiles_k * tile_m`) and the RHS window is the rest. A fixed split written
into the lowering is what this replaces. Beyond the second bound the recipe reports a refusal
(split K into partial sums with the compact plan, or shrink the M panel), never a silent fallback.

Sample (synthetic facts, K one tile past the first bound):

```
retained_full_k: declined: a full-K panel pair needs 136 operand rows, the store has 128
segmented: operand_split_rows {lhs: 68, rhs: 60}  block (1, 2, 3)  k_segments 9
           overlapped_compute_pairs 48  epoch_drains 14 (12 with batched readouts)
```

The block choice there is "fewest loads, then the longest segment" (fewer segment boundaries to
drain at). That is a starting point to time, not a rule.

## Porting into your lowering

1. Keep `capacities -> dense_view -> dense_blocks` as is; reserve the rows your other live data needs
   before splitting the stores.
2. Replace `adapter()` (neutral instruction names) with your `DenseLoopAdapter(encode, project, ...)`
   and the `test-only-*` law identities with your target's declarations (the `scheduling-api` skill
   lists each law's fields).
3. Keep `execute_dense` (or your own host executor) as a unit test on a ragged shape on both sides of
   each bound.
4. Register the pipelines as alternatives (`reference/pipeline-policy.md`), not as an `if` on M, N
   or K.
