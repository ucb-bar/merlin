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
a real ordering point (a completion) between them, so the recipe does not accept "zero first, data
second" as a fix. On the synchronous route used here every callback must complete before it returns
(that is the ordering point between a compute and the next transfer into its rows); the issued route
(`issue_modes=(True,)`, `ConvolutionLoadIssueLaw`) emits explicit `complete` calls instead, and your
lowering must implement each one as a real drain, not drop it.

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
