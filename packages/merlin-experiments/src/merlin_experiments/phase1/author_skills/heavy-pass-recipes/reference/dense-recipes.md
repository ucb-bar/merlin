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

### Full-K panels, including one K tile

`dense_readout_recipe.py --family retained --shape M,K,N` uses the same checked entry for a
full-K panel. It compares separate, batched and causal-ready readouts on the **same** capacity-legal
block. It requires the complete LHS/RHS reduction panels to fit their operand windows, and the
output block to fit one accumulator window. It does not require a second accumulator window,
multiple K segments, or an accumulator-bound whole output. When full K does not fit it refuses;
try the segmented family as a separate alternative.

`recipe_support.retained_pipeline` constructs and round-trips a `retained_full_k` schedule plan,
hoists the retained operand, verifies actual SCF, and returns the checked result for ordinary
`emit_llvm` / `emit_index_program` consumption. The recipe executes every emitted product on host
integers. Its laws are explicitly test-only; bind your actual adapter and separate reduction and
producer-to-readout completion laws before target execution. Host checks and drain counts do not
establish target performance: the report leaves cycles UNKNOWN for your normal cycle evaluator.

Add `--input-pipeline` to compare two advancing LHS slots with invariant weights. All controls
then retain weights on the same full-K block. The extra alternative derives its LHS slot offset
from the padded panel footprint and refuses if two panels do not fit the original owned window,
the combined operand budget, or at least two advancing groups. Its separate mixed-load/readout
law is test-only. Bind selected target laws and measure cycles before retaining this alternative.

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

## Drains between K steps (`dense_epoch_recipe.py`)

**When.** A two-window plan (`pipeline_prefetch=True`, no `pipeline_kind`) streams K through two
operand windows: the current window's compute and the next load into the other window share one
epoch, and every K step ends with a full drain. With `K / edge` K steps that is `K / edge + 2` drains
per output block (initial load, every step, readout), whatever the target already orders.

**What a K-step drain orders.** A compute after the load into its rows (read after write), a load
into a window after the computes that read it (write after read), and successive accumulator updates
of the block. A target whose command ordering waits for those pairs needs no drain for them.
`dense_panel_epochs.DensePanelEpochLaw` names each premise as a flag; the recipe maps every flag to the
`command_ordering` fact that backs it, to a program premise, or to `UNKNOWN`:

| flag | backed by |
|---|---|
| `local_read_after_write` | load write -> compute/descriptor read waits (`cross_queue`, scratchpad) |
| `local_write_after_read` | compute/descriptor read -> load write waits |
| `ordered_accumulator_writes` | the compute queue completes in order (`completion_order`) |
| `accumulator_result_visibility` | descriptor accumulator write -> readout read waits |
| `ordered_configuration_capture` | latched configuration codes enter the queue that uses them |
| `finite_queue_progress` | derived dispatch entries for the load and compute queues |
| `complete_device_drain`, `exclusive_private_state` | program: your completion callback, your windows |

Load completion order is not a premise: a load queue that lands writes out of order is fine, because
the library proves that every load of a merged K step is read by the next step, so a later load into
the same rows is ordered through that reader. A schedule that defers consumption (prefetch off with
`prefetch_rhs_between_columns`) is refused rather than left unordered.

**Chain.** One option on the same entry and plan:

```python
from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw

law = DensePanelEpochLaw(adapter.configuration_identity, adapter.numerical_policy_identity,
                         "your-ordering-facts-identity", **flags_from_your_facts)
result = schedule_dense_pipeline(p, r, s, adapter, ..., schedule_plan=two_window_plan,
    pipeline_admission=overlap_law, batch_readouts=True, epoch_law=law)
result.emit_llvm(builder, lower_call)   # lower_call sees only the remaining drains
```

The K stream of each output block (and, under result visibility, its readout) becomes one epoch;
initialization, output-block boundaries and the closing drain stay. Data calls and their order do not
change. A merged epoch must fit the admission's `max_issued_compute_pairs` / `max_issued_loads`
(compiler budgets; larger ones only change how much a finite queue backpressures); a stream over them
keeps its K-step drains. `epoch_law=None` returns the unchanged drained schedule.

**Reading the output** (synthetic facts: edge 4, 128 operand rows, 32 accumulator rows; `--small`):

```
shape (M,K,N)   block    output blocks   drains drained -> epochs   data commands
1 x 48 x 80     (1,1,8)  3               42 -> 3                    776 (both)
8 x 48 x 36     (2,1,4)  3               42 -> 3                    630 (both)
7 x 27 x 17     (2,1,4)  2               18 -> 2                    213 (both)
```

Both routes run on host integers under an ordering model built from the same facts (commands pend
until a drain, loads land newest first, only the facts' waits order commands) and through
`completion_order.check_commands`. An `UNKNOWN` flag in `law_flags` keeps its drains: without the
readout wait two drains per output block remain, without either operand wait all of them. Fewer
drains raise queue occupancy; time both routes (`ab-library-pass`) before keeping the epochs.

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

## Keep your encoder (`counted_emission_recipe.py`)

The checked panel, segmented, ready-readout and disjoint-readout result owners expose the SAME
entry. Your existing effect encoder receives the actual original instruction names and index
operands from your `DenseLoopAdapter`; it needs no schedule-specific rewrite:

```python
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter

control = FnBuilderControlAdapter(fn)  # your existing ordinary FnBuilder
result.emit_index_program(control, lower_effect, complete_epoch=full_device_completion)
# lower_effect(instruction, fields) emits actual target instructions, without an added per-call fence.
# full_device_completion() emits your original, source-qualified full completion.
# Finish fn through the SAME normal package emitter on both sides of the pass.
```

`counted_emission_recipe.py --shape 7,12,19 --output DIR` uses one neutral callback to emit actual
LLVM for the original segmented schedule, batched readouts, causal ready readout and disjoint
readout/compute epochs. It checks every output using the projected source interpreter. Replace its
external effect declarations and test-only admissions with your selected encoder/completion and
laws; those placeholders confer no hardware support. The recipe is an optional mechanical probe,
not a timing result or a new launch gate.

Loops remain counted. Only original explicit drains invoke completion; the checked owners
revalidate source, actual schedule, effects and library identities before and after emission.
The lower-level `llvm_program.emit_index_effect_program` provides the same mapping for a counted
`Kernel`, but supplies no source/effect admission by itself. Event/state annotations are refused
by this raw path: use `scf_commands` with an annotation-aware target callback for those programs.
Arbitrary private Python loops or opaque assembly still need a lossless `encode`/`project`
description and selected hardware premises. This bridge does not infer them.

## Size pipeline storage from actual operand lifetimes

For an already legal counted issue schedule, `operand_slots` connects the existing temporary
allocator to counted SCF emission. Group original private backing generations by operand:

```python
from merlin.compile.scheduling.operand_slots import PipelineOperand, allocate_pipeline_operand_slots

operands = tuple(PipelineOperand(name, tuple(generations))
                 for name, generations in original_operand_generations.items())
candidate = allocate_pipeline_operand_slots(
    original_kernel, operands, selected_banks, storage_adapter,
    reservations=live_reservations, external_windows=original_external_windows,
    private_ownership=True, max_calls=effect_expansion_budget,
)
chosen_kernel = original_kernel if candidate is None else candidate.kernel
```

`StorageAdapter` must decode actual initialization, access and launch/wait operands, then rewrite
and independently re-decode addresses. Invariant operands can use one slot; advancing operands use
as many live generations as their real completion dependencies require. This changes allocation,
not command order or overlap permission. Emit `chosen_kernel` with the existing
`scf_commands.materialize_scf_commands`/`emit_scf_commands` consumer and your actual target callbacks.
Compare complete-call cycles against the original schedule.

The current family accepts one bounded, nonnested source loop and reconstructs checked periodic
address rotations with exact tails. It refuses mutable generations, irregular rotations,
SSA-carried asynchronous boundaries and excessive effect expansion with
`StorageOptimizationUnavailable`; retain the incumbent for those unsupported alternatives.
Infeasible capacity returns `None`. Neither outcome grants extra hardware buffers or removes waits.

For an existing retained full-K schedule with causal ready readout, the same
`schedule_dense_pipeline(..., readout_mode="ready", input_pipeline_row_offset=...)`
entry can pipeline advancing input panels. Derive the second LHS row offset from
one complete padded input panel (`block.m * block.k * problem.tile_m` rows), check
both slots against the original owned view and operand byte budget, and keep the
immutable RHS in one slot. The input pipeline keeps the original accumulator
addresses and completes all loads, compute and stores before their reuse.

Select `DenseReadyReadoutAdmission.disjoint_load_qualification_identity` only
when the original target owner separately establishes the complete disjoint
load/compute/ready-store conjunction, configuration snapshots, finite queue
progress and full completion. The checked consumer proves actual local intervals
and every final wide-load band; an ordinary ready-readout law alone refuses this
composition. Other pipeline families reject the input-slot option. Retain the
incumbent for unsupported shapes, UNKNOWN cycles or a regression.

`projection_dense.bind_dense_projection` connects current shared-input
`projection_views` to an existing checked counted dense schedule. Retain the LHS
with the ordinary dense hoists, select a panel width dividing every member, and
name the complete column loops to partition. The returned counted kernel adds
member, local-column and original row-width operands to actual weight/output
transfers. Bind those operands to the original independent ABI pointers and
pitches; preserve every original readout word and revalidate the dense owner
around emission. No virtual tensor is allocated. Physical rectangles crossing
members, changed precision/scales and unsupported partitions refuse. Compare
whole-component cycles and retain the incumbent on UNKNOWN or regression.

Run `projection_recipe.py` for a neutral retained-input example that checks every
original pointer rectangle and source product, with fact-derived capacities and
explicit UNKNOWN timing. Replace its test-only laws and callback with the target
encoder and completion before native qualification.

## Closed host integer observations

`source_observer_recipe.py` runs from the V24 selected library. It binds typed
source arithmetic and saturated i8 observations to actual LLVM, including the
standard bounded roundeven form emitted by ordinary host rounding. It derives
exact observer cells, an interval table and a finite broadcast-scale witness;
its neutral boundary checks are not native execution or timing evidence.

Supply current-source RNE/gradual/nontrapping/unobserved-effect permission, an
actual incoming-mode guard, original source fallback, immutable scale spans and
fresh disjoint outputs. Consume generated callbacks and LLVM in the ordinary
compiler before object identity, preserving the existing late host legalizer.
`--output DIR` writes neutral source/LLVM/C/table files; bind original source
callbacks and runtime ownership before compiling them. Unknown arithmetic such
as exp has no polynomial permission and retains source. No runtime backend or
protected evaluation API is included. Full compiler access remains a separate
reviewed inventory; V23 and earlier selected-library inventories are unchanged.
