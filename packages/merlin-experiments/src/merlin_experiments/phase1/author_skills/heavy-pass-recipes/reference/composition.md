# A connected chain through the composition catalog (`spatial_chain_recipe.py`)

## Which routes apply to one case

```
python .agents/skills/heavy-pass-recipes/scripts/composition_routes.py --case NAME
```

reads the case's public declaration (and, for a model or program, its interface IR) and lists the
producer -> consumer chains it shows: product -> pool (`producer_pool`), product -> bias, scale,
activation or requantization (`producer_epilogue`), residual add (`residual_add`), sibling
products sharing an input (`shared_input_siblings`), per-head slices (`independent_heads`),
connected stages, and parameters prepared on every call. Each route names its catalog rows and
their roster status, one `module:symbol` to start from, whether a file under `submission/`
imports it, and a command (the route's recipe, else the symbol's contract). For models and
programs it also lists matching `source_composition.json` options
(`reference/source-composition.md`). Without `--case` it lists every route. Self-check rows carry
the same pointer as `optimization_next.composition`.

## Discover what you can compose

```python
from merlin.compile.composition_catalog import composition_capabilities
catalog = composition_capabilities(selected_modules=roster)   # exact module names you can import
for row in catalog["capabilities"]:
    row["name"], row["access"], row["source_availability"]["status"], row["symbols"], row["requires"]
```

`roster` is exact module membership: the recipe walks `submission/compiler_library/merlin/**.py`
when that tree exists, otherwise uses `merlin.compile.scheduling.library_sources.library_modules(N)`
for the inventory N your TASK.md names. `access` says who may use a row (`author_library` rows are
yours; `original_host_owner` rows need the original caller's selection). `source_availability`
lists modules the roster lacks; the catalog grants nothing, it tells you where to look. Read each
row's `requires` and `limitations` before building on it.

The recipe prints every `author_library` row with its status and uses
`bounded_spatial_packing_and_rolling`.

## The chain

Convolution + bias -> rounded int8 (Q1, with relu) -> spatial max with padding -> rounded int8 (Q2),
without a full intermediate image:

```python
producer = CountedConvolutionPlan(window, cout, edge, row_tile, edge, min(edge, c),
                                  operand_rows, accumulator_rows, 0, rhs_row, 0, True)  # bias
pool = SpatialMaxReadout((oh, ow), (kh, kw), (sh, sw), pool_padding, -128)
chains = tuple(SpatialReadoutChain(RoundedReadout(q1_scales, True), RoundedReadout(q2_scales), None)
               for each output-channel tile)
source = SpatialProducerPlan(producer, pool, chains, stripe, readout_mode="staged_readouts",
                             intermediate=AddressWindow("q1_stripe", 0, halo_rows * column_tile))
choices = propose_rolling_spatial_choices(source, workspace, panel_rows_limit=...,
                                          output_columns_limit=stripe, alignment=...)
receipt = emit_rolling_spatial_producer(choice.plan, builder, lower, synchronous_completion=True)
```

Capacity derivation in the recipe (all from RTL facts):

* tiles: the array edge; output pixels per conv tile `min(edge, output width)`;
* pool stripe: the widest stripe whose halo rows `kh * ((stripe - 1) * sw + kw)` fit the
  accumulator store;
* host panel rows: bounded by half the operand store divided by the K tiles of one packed row;
* alignment: the largest power of two dividing an operand-store row;
* host workspace: by default exactly what the widest stripe family needs (`--workspace-bytes`
  narrows it and drops families that no longer fit).

`propose_rolling_spatial_choices` returns each stripe family with `dense_invocations`,
`logical_macs`, `primitive_mac_extent` (MACs including tile padding) and `staged_input_bytes`. These
are inspectable costs, not latencies: the recipe takes the fewest dense invocations to have one to
emit; rank them by timing on your panel.

## Effects your backend implements

`lower(kind, fields)` receives, in order per stripe: `begin_transport`, `stage_row` (one contiguous
input row into the cache, or zeros when `valid` is false), `await_transport`,
`pack_cached_input_panel` (gather halo columns from the cache with `emit_rolling_window_lookup`),
`produce_packed_panel` (the dense product + bias + Q1 into the Q1 stripe), `reload_intermediate`
(Q1 rows back into accumulator rows), `store_spatial_max` (max over the window + Q2 into the
output), `consumer_issued`, `complete`. Every callback completes synchronously; `complete` must
finish all uses before the next frontier transport overwrites cache or Q1 rows.

The recipe's host callbacks implement these on Python lists and compare the pooled output with an
independent source evaluation (conv, Q1, max over valid cells, Q2), and fail if any output element
is written twice. The catalog row also names `merlin.llvmlower.convolution_packing` (the host LLVM
packer); when your roster lacks it, your backend supplies `pack_cached_input_panel` itself.

## Other chains

* Pointwise chains between contractions: `checked_pointwise_fusion` / `source_fusion_exploration`
  rows (the `fusion` skill).
* Sequential stages with preserved tensor identities: `connected_source_execution`
  (`connected_regions.plan_connected_region`); no cross-stage fusion is implied.
* Ordering and loop transforms consumed by emission: `real_transform_ordering` (the
  `compose-pass-pipeline` skill).
