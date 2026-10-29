# Convolution without a per-pixel gather (`conv_window_recipe.py`)

## The pattern to replace

An implicit-GEMM lowering that, for every output pixel and every (ky, kx, channel chunk), issues a
one-row input transfer into the LHS rows, and for padded taps first zero-fills those rows and then
overwrites them, has two problems:

* **Command volume.** `pixels * kh * kw * chunks * output_tiles` input transfers (the recipe prints
  this as `one_row_gather_input_transfers`), plus a weight transfer per compute. On a 56x56x64 -> 64
  3x3 layer that is hundreds of thousands of input commands before any weight traffic.
* **A write-after-write race.** A zero fill and a data transfer to the same local rows, both in
  flight, complete in an order the program does not control unless every pair is fenced. The
  result depends on timing, so it can pass on one simulator and fail on another.

## The library route

```python
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource, ConvolutionRhsResidencyLaw, convolution_pass_space,
    emit_convolution_schedule, explore_convolution_schedules)
from merlin.compile.scheduling.window_transfer import ConvolutionInputRetentionLaw

window = Window2D((b, h, w, c), (kh, kw), (sh, sw), padding=(top, left, bottom, right))
edge = min(array_rows, array_cols)                       # from RTL facts
row_tile = min(edge, window.output_spatial[1])           # output pixels per tile = LHS rows
plan = CountedConvolutionPlan(window, cout, edge, row_tile, edge, min(edge, c),
                              operand_rows, accumulator_rows,   # the stores, from RTL facts
                              0, ceil(row_tile / edge) * edge, 0)  # LHS rows, RHS rows, ACC rows
source = ConvolutionExplorationSource(plan, column_intervals, source_identity)
space = convolution_pass_space(source, verify_source=reopen_your_source, realize=your_emitter,
    retention_law=ConvolutionInputRetentionLaw(...), rhs_law=ConvolutionRhsResidencyLaw(...),
    live_tile_limits=(1, None), issue_modes=(False,), rhs_modes=(False, True),
    private_operand_rows=operand_rows)
for candidate in explore_convolution_schedules(source, space, context=PipelineContext()):
    ...  # candidate.emission is YOUR emitted program; time the unique ones
```

`realize(schedule, rhs=None)` calls `emit_convolution_schedule(schedule, builder, lower,
complete=..., synchronous_completion=True, rhs=rhs)` with your IR builder and instruction
callbacks, and returns an `EmittedProgram` bound to `(rhs or schedule).identity`. The recipe's
builder is a host integer machine (`Integers`) so the effect stream can be executed and checked.

What each registered choice does:

| choice | effect in the emitted stream |
|---|---|
| interior row runs (always) | `Window2D.emit_row_run` proves the `row_tile` input rows of a tile form one in-bounds strided rectangle; ONE transfer moves them (`input_multi_row`) |
| `live_tiles` = capacity | one input transfer per (ky, kx, chunk) serves every output-channel tile whose accumulator rows are live (`ConvolutionInputRetentionLaw`) |
| `rhs_preparation` = on | every weight panel is loaded once per invocation into resident rows (`ConvolutionRhsResidencyLaw`, placement by `dense_rhs_residency`); computes read them in place (`weight_loads` drops to `kh*kw*chunks*column_tiles`) |

Capacity decides what is offered: residency declines (no candidate) when the weight panels do not
fit next to the LHS rows; `live_tiles` is clamped to the accumulator rows.

Where the original output columns are independent, immutable inputs and no intermediate observers
permit a separate capacity-derived alternative. `plan_convolution_rhs_phases(schedule,
operand_rows=assigned_rows, law=selected_rhs_law, phase_law=selected_phase_law,
input_rows=checked_resident_rows)` keeps a full-fit owner, otherwise returns `ConvolutionRhsPhases`
covering every original column with full-K residency, or `None` if even one interval cannot fit.
`ConvolutionRhsPhaseLaw` must bind those source premises and real completion before reprepare.
Pass the **complete owner** to `emit_convolution_schedule(..., rhs=owner, input_rows=the_same_rows)`;
never emit `owner.phases` individually. The consumer drains each phase's last use before replacement.
The registered alternative uses `rhs_modes=(False, True, 'phased')`, `rhs_phase_law` and `input_rows`.
Keep the incumbent where legality/capacity declines and compare actual complete-call cycles.

## Safe zero handling

Two correct options, both shown by the recipe (`routes`):

* **`border_rows`.** A tile whose row run leaves the image falls back to one transfer per row, and
  each row is EITHER loaded from the source OR zero-sourced (`zero: true`), exactly once before the
  compute that reads it. There is no zero fill followed by a data write.
* **`halo_source`.** The host copies the input once into a buffer with a zero border (padding becomes
  part of the source tensor; the zeros are written once, before any device transfer). Every tap is
  then an in-bounds row run: no per-row fallback at all. Cost: one extra host copy of the input.

The recipe's `Machine` fails if any local row is written twice before a compute consumes it, whatever
the order the writes were issued in, and compares every output with an independent source
convolution. Issue order is not completion order: two transfers into the same rows are only safe with
a real ordering point between them, so the recipe does not accept "zero first, data second" as a fix:
an in-order simulator can pass such a stream while hardware that completes loads out of order does not.

## Where a drain is needed, and compute epochs (`conv_epoch_recipe.py`)

A full drain is needed only between a write and a later conflicting access to the same local rows
that nothing else orders: the zero-fill-then-overwrite pair above, or a reload of rows a compute is
still reading. It is not needed after every compute. The library's routes differ only in where the
`complete` calls fall:

| route | `convolution_pass_space` choice | drains |
|---|---|---|
| synchronous | `issue_modes=(False,)` | every callback completes before it returns: one drain per effect |
| drain per compute | `issue_modes=(True,)`, `ConvolutionLoadIssueLaw` | after every compute and every store |
| compute epochs | `compute_pair_limits=(P,)`, `ConvolutionComputeEpochLaw` | at a runtime boundary every P compute pairs and after every store |
| final producer/readout epoch (separate opt-in) | `emit_convolution_schedule(..., readout_law=ConvolutionReadoutEpochLaw(...))` | intermediate compute boundaries and a real final drain after the last original store of each private column block |

The readout law is independent of the compute law and of checked dense readout-result objects.
Your source/target adapter must establish ACC RAW and result visibility, ordered store-configuration
snapshots, finite queue progress, disjoint original output/ACC intervals and private ownership.
Record the selected pending store capacity separately from the compiler's issued readout bound;
issuing more work relies on actual backpressure, never extra hardware slots. Keep rolling
input-ready/last-use drains and whole-call completion. Unknown premises retain the default.

The existing recipe provides the small adapter seam (placeholders are caller-owned):

```python
from conv_epoch_recipe import emit_checked_readout_epoch

applied = emit_checked_readout_epoch(checked_convolution_schedule, your_index_builder,
    your_original_effect_encoder, complete=your_real_full_drain,
    readout_law=your_selected_source_readout_law, rhs=your_checked_rhs_plan)
save_your_emission_receipt(applied)  # emission + consumer_receipt_matches
```

`verify_readout_application` compares the returned source, schedule, readout law and final-drain
scope with the actual requested objects. It refuses a default consumer receipt, wrong law or
unapplied driver-local option. This is a consumer declaration check: independently inspect emitted
instructions/order, all original numerical outputs and same-domain complete-call cycles before
promotion. The neutral recipe does not derive store snapshots or grant target qualification.
When using the phased RHS owner, the same adapter verifies **all** returned `phase_emissions`, their
original intervals and owner identities; a missing phase or a stale partial receipt refuses.

Each route needs every flag of its law. `conv_epoch_recipe.py` reads the target's RTL
`command_ordering` facts (`--ordering`, or the `MERLIN_COMPLETION_ORDER_SELECTION` pin) and prints one
row per flag: `fact` with the facts row that justifies it (for example a younger compute waits for the
completion of an older load into overlapping rows; a younger load waits for an older compute that
read those rows; a readout waits for the descriptor that wrote its accumulator rows; the queue holding
descriptor and compute issues in order and has entries for the pair), `program` for a property of
your allocation or completion callback, or `UNKNOWN`. A hardware flag without a fact removes the
route that needs it, so its drains stay: no issue law means the synchronous route, no epoch law means
a drain after every compute. Where a load queue may complete writes out of order, a rewrite of a row
is ordered only through the compute that consumed it (load -> compute wait, then compute -> load
wait), so a stream with two loads into one row and no consumer between them still needs its drain;
the row check and the ordering check below catch it.

Every admitted route is executed on host integers and checked against the source; its effect
stream is also encoded as accelerator commands from the same facts (local address, rows, columns)
and run through `merlin.targetgen.completion_order.check_commands` (write-write and write-read pairs
under the queue semantics) plus a read-then-write supplement. `check_scope` in the output lists what
is and is not checked; `uncertain` with `residual` names any pair whose dispatch tracking the probes
left open and what they observed. Replace the recipe's encoder with yours and keep the checks as a
unit test; your lowering must still implement every remaining `complete` as a real drain.

Sample, a 4x56x56x64 -> 64 3x3 layer with an edge-16 facts body and their ordering facts
(`--count-only`; 129,024 compute pairs on every route):

```
route              drains    drains/compute   hazard pairs   verdict
synchronous        228433    1.77             0              clear
drain per compute  132609    1.03             0              unknown (residual below)
epochs, P=144        4481    0.035            0              unknown (residual below)
```

The two issued routes share one residual: the facts leave open whether the dispatch tracks a load
write against a later compute read by address, because the probe saw the compute wait for the load's
completion whether or not their rows overlapped (`ordered_when_overlapping: true`). Both routes rely
on that wait; only the synchronous route avoids it, with a drain after every load. Drains and commands
explain a result; cycles decide. Time the routes on the same cases.

Sample, `--shape 1,56,56,64,64 --count-only` with an edge-16 facts body (counts per candidate):

```
route        live_tiles resident  input transfers (multi/one/zero)   rows/transfer  weight loads
border_rows  capacity   yes       23920 (6640 / 14608 / 2672)        4.72           144
halo_source  capacity   yes        8064 (8064 / 0 / 0)               14.0           144
one-row gather equivalent: 451584 input transfers
```

Border tiles of `border_rows` are where its one-row transfers come from (a tile crossing the border
by one pixel moves all its rows one at a time). Whether the halo copy pays for itself is a timing
question for your layer sizes: measure both.

## Row staging alternative: `rolling_window`

`propose_rolling_window(window, cache, source_identity=...)` -> `materialize_rolling_window` ->
`emit_rolling_window(module, plan, builder, lower)` stages whole input rows (`row_bytes` contiguous
bytes each) into a circular cache: `kh` rows for the first output row, then `stride` new rows per
output row; a padding row is staged as zeros (`valid: false`) once. `emit_rolling_window_lookup`
addresses one tap inside the resident rows. The recipe counts staged rows against
`unstaged_reference_rows` and checks that no cache slot is written twice before its consumer. The
spatial-chain recipe uses the same cache to feed a packed producer.

## Porting

Keep `plan()` (capacities -> tiles), replace `realize` with your emitter (your builder, your
`load_input`/`load_weight`/`compute`/`store` lowering, your completion callback), replace the
test-only law identities with your target's, and keep the host machine as a unit test on a padded,
ragged layer. Time the unique candidates on your panel; do not pick by command count alone.

## Cross-spatial reuse and wider readouts: separate source-bound choices

The current V23 source selection includes `convolution_output_groups`. Its planner accepts
caller-owned `max_row_blocks` and `max_output_rows` bounds. The latter defaults to one, preserving
the original realization. It returns an existing `ConvolutionOutputGroup`, an aligned tuple of
phase-local groups when `rhs` is a complete `ConvolutionRhsPhases` owner, or `None` to retain the
incumbent. A tuple entry of `None` retains that phase's ordinary route; a partial RHS member is
never a standalone entry. Primitive channel/column partitions and disjoint private ACC cells are
required; short source rows keep primitive-sized ACC slots and their original active extents.

```python
from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule

# A caller can read this bound with merlin.compile.tuning_knobs.knob from its
# own finite TUNING_KNOBS.json declaration. No bound selects a timing winner.
group = plan_rolling_convolution_output_group(schedule, rhs, input_rows,
    max_row_blocks=your_static_expansion_bound, max_output_rows=your_output_row_bound)
if group is None:
    emit_your_incumbent(schedule, rhs=rhs, input_rows=input_rows)
else:
    applied = emit_convolution_schedule(schedule, your_index_builder, your_effect_encoder,
        complete=your_real_full_drain, readout_law=selected_readout_law,
        rhs=rhs, input_rows=input_rows, output_group=group, array_law=selected_array_law)
```

The same options work through `conv_epoch_recipe.emit_checked_readout_epoch`. Its verifier
reopens each selected group, grouped transport IR and readout map, including every complete
RHS phase. A phase-local `None` keeps that phase's ordinary route.

RHS phase width is an independent joint-storage choice. The default retains maximal
complete-K RHS residency. An explicit `max_column_blocks` in
`plan_convolution_rhs_phases`, or `rhs_column_blocks` in
`plan_convolution_storage_alternative`, can keep narrower phases so more original
spatial accumulators fit. Every phase still covers complete K and the original
column intervals exactly once, and uses the actual declared operand capacity and
input/RHS placement. Rebinding an owner to a newly checked schedule must preserve
its `max_column_blocks` alongside its phases and source/input bindings.

The ordinary exploration knob is `convolution_storage_pass_space(...,
rhs_column_limits=(None, your_legal_column_bound), ...)`. Here `None` selects the
unchanged maximal-capacity policy. A caller can derive a candidate bound from
`available_ACC_rows // (requested_live_spatial_tiles * primitive_rows)` after
checking the original row footprint and expansion bound. Emit and compare the
resulting whole invocations; phase count, operand reads and spatial reuse are
diagnostics, and do not select a winner without completed timing observations.

Multiple output rows require an already admitted ring whose actual slots cover
`kernel_span_y + (group_output_rows - 1) * stride_y`. The planner clamps against that source-row
union, source extent, the caller's expansion bound and active phase ACC capacity; requesting a
larger bound cannot grant storage. Plan the original RHS phases against the same actual input
footprint. Products wait until their whole group is staged, exact per-row tile origins and tails
are retained, and all original ready/last-use drains remain real. Each phase keeps its original
column intervals, scalar readouts, scratch offsets and full final drain before RHS replacement.

`selected_array_law` is the existing `DenseArrayRetentionAdmission`, independently bound to actual
fresh/stay/output-descriptor behavior. The source retention law additionally needs
`independent_output_rows` and `no_intermediate_observers`; channel independence does not grant
spatial reordering. The encoder consumes the actual static Boolean `fresh_weights`: the first M
consume captures the original RHS, and subsequent M consumes retain it while changing only their
checked output descriptor. Preserve every original K order, partial row/channel extent and exact
per-N scale/activation. Bind issued initializer/compute/readout bounds to the real group work;
finite pending queue capacities are unchanged. The final real group drain closes ACC ownership
before reuse, in addition to the unchanged ring ready/last-use drains.

A distinct option is `window_transfer.ConvolutionReadoutTransferLaw` with
`plan_convolution_readout_transfers`: original scaled-INT8 stores can combine only when their
numerical states, output columns and actual physical ACC block spacing match a separately selected
controller map. Pass `readout_transfer_law=selected_transfer_law` to the same recipe adapter. No raw
full-ACC, pooling, capacity or wide-store support is inferred from shape. Short private row slots,
state changes and unsupported layouts retain their original transfers. It also composes with
spatial groups: transfer adjacency uses the group's actual disjoint ACC placement. Independently
opt into `grouped_transport=True` to stage each group's exact source-row union before products;
the existing load-issue law must cover its actual transfer count. This does not increase queue
capacity or remove the real final drain.

An explicit V24 `rolling_lookahead` consumer can issue future input strips between
complete original reduction traversals. Supply the original reduction-panel count;
the SCF contains typed start/advance/finish effects and the actual stage addresses.
The default with no `consumer_runs` retains the atomic consumer. A convolution
continuation initializes bias once, preserves K/N/M order and every fresh/stay
run and pair completion, and emits original readouts and the group drain once.

```python
from merlin.compile.scheduling.rolling_lookahead import (
    materialize_lookahead_rolling_window, emit_lookahead_rolling_window,
)
from merlin.compile.scheduling.convolution_output_groups import emit_rolling_convolution_output_group

p = schedule.source.plan
ci = p.window.input_shape[-1]
runs = p.window.kernel[0] * p.window.kernel[1] * ((ci + p.channel_tile - 1) // p.channel_tile)
module = materialize_lookahead_rolling_window(ring_plan,
    output_rows=group.output_rows, consumer_runs=runs)

def lower(kind, fields):
    if kind == "start_consumer":
        # This mapping uses this group's original source rows, column block,
        # row counts/origins, resident input, array/readout laws and ACC slots.
        return emit_rolling_convolution_output_group(schedule, group, builder,
            issue_original_effect, complete_original_effect,
            **original_group_operands(fields), defer_consumer=True)
    if kind == "advance_consumer":
        return fields["consumer"].advance(fields["first_run"], fields["run_count"])
    if kind == "finish_consumer":
        fields["consumer"].finish()
        return fields["consumer"]
    return lower_original_transport(kind, fields)

emit_lookahead_rolling_window(module, ring_plan, builder, lower,
    output_rows=group.output_rows, consumer_runs=runs, admission=selected_lookahead_admission)
```

The target supplies `selected_lookahead_admission` with separately bound ordered
configuration, finite queue progress, disjoint future/current local effects, full
completion and `retained_consumer_continuations=True`. Input/ACC state stays live
across every strip. The union of current and next group must fit the actual ring;
no slot or completion is inferred from a run count. The existing invocation owner
binds start/advance/finish for one live column block; phase replacement still waits
for its real final drain. Independent final instruction/address checks, exact
output execution and completed-call timing remain required.

For a small runnable interface example, use the existing recipe:

```sh
python conv_epoch_recipe.py --library submission/compiler_library --write-toy-contract toy.json
python conv_epoch_recipe.py --library submission/compiler_library --adapter-contract toy.json --output toy-artifacts
```

The passed contract specifies a **sequential toy machine**, original convolution/readout, private
storage, separate typed laws and bounded alternatives. The example generates abstract ISA definitions
with the ordinary ISA generator, binds complete effect fields with `generated_effect_callbacks`,
emits SCF and interprets that actual IR with bounded pending effects and real FIFO drains. Every
original output is compared exactly; `receipt.json` includes application receipts and artifact hashes.
Alternatives are unranked and cycles remain UNKNOWN. No target backend or physical law is supplied.
For a real target, author/review its ISA and semantic bindings from its selected source first;
call the same adapter with its generated `create`, scalar builder and operation insertion callback,
or retain your existing encoder when an effect requires several instructions. Completion bindings
are separate and must implement the actual device drain. ACC RAW/result visibility, captured
configuration, bounded queue progress, input/RHS preservation, fresh/stay behavior, independent
outputs and exact scaled-i8 block mapping remain explicit requirements of the selected hardware class.

Inspect the actual selected group/map and law hashes in each returned consumer receipt, including
every phase emission; a requested bound alone is not application evidence.
That proves request forwarding, not target execution, numeric success or faster cycles. Keep an
uninstrumented incumbent, compare every original output and rank only eligible complete-cycle
feedback. No neutral recipe supplies a target law or measured winning configuration.

For original scalar versus per-column integer exits, use
`DenseTiledIntegerReadout.from_source_scales(original_scales, columns=original_columns,
tile_columns=selected_tile_columns, relu=original_relu)`. It broadcasts an original scalar or retains
the complete original vector at exact F32 boundaries; it never averages gains. Each tile must be
source-scale compatible. This normalization does not grant a causal varying-scale snapshot law:
unsupported dense ready-readout combinations retain separate original readouts.

The initial author workspace can include `COMMAND_ORDERING_SELECTION.json`, an exact
source-bound copy of the operator's selected public ordering record. The epoch
recipe discovers it from the working directory or the staged facts root; no
operator environment or private path is required. A different facts identity is
refused. Derive each required law from the record; its presence is not a blanket
asynchronous-execution permission.
