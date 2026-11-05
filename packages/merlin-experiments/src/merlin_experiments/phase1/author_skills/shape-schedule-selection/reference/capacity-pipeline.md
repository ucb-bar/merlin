# capacity_pipeline_example.py: one dense kernel, capacity-derived tiles, three library plans

The example is the shape your dense lowering can take: one shared upstream (facts, problem, storage,
resources) and a downstream plan chosen per shape. It runs against `submission/compiler_library` (or
`--library DIR`) and prints JSON; the preflight runs it on every launch.

## 1. Facts to capacities (shared)

```python
from merlin.targetgen import address_space as AS
space = AS.derive_address_space(label, facts=json.load(open("rtl_facts/facts.json")))
operand, accumulator = AS.operand_store(space), AS.accumulator_store(space)
edge = space.array_rows, space.array_cols
rows, row_bytes = operand.capacity_rows(), operand.store.row_bytes      # same for accumulator
```

Stores are chosen by row width, never by name; a refusal carries its reason (`role.reason`), so report
it instead of falling back to a number. Without a facts file the example uses a small synthetic body and
says so in `"facts"`.

## 2. One problem, storage and resource view (shared)

* `DenseProblem(m, k, n, tile_m, tile_k, tile_n, lhs_bytes, rhs_bytes, acc_bytes)`: the primitive tile
  is the array edge.
* `DenseLoopStorage`: the LHS and RHS windows split the operand store (two `DenseMemoryView`s with
  different bases in the same memory); the accumulator view owns the accumulator store. Reserve what
  your other lowerings keep live before splitting. A fixed window size written into the lowering
  (some rows for LHS, some for RHS, regardless of shape) is the thing this replaces.
* `DenseResources(operand_bytes, accumulator_bytes, max_*_blocks=tile counts)`: the byte budgets are
  the windows, so `dense_blocks(problem, resources)` enumerates exactly the blocks that fit.
* `merlin.compile.capacity._capacity_fit_tile(M, K, N, edge, operand_elems, acc_elems)` answers the
  whole-layer question: the largest resident tile and how many cover the layer (`capacity_fit`). The
  example reports it; `shape_choice_table.py` uses it as the `T` column.

## 3. A plan per pipeline (specialized)

Each plan picks its block from the same `dense_blocks` list under its own premise, then states it as
editable decision IR in the `schedule` dialect (`make_dense_schedule_plan` -> `dense_schedule_text`:
`schedule.plan`, `schedule.bind`, `schedule.block_contraction`, `schedule.loop_transform`,
`schedule.panel_pipeline`). The example takes the legal block with the fewest requested loads; with
real alternatives, choose among legal blocks by measurement (`shape_choice_table.py`), not by a
constant.

| plan | premise on the block | library chain |
|---|---|---|
| `compact` | any legal block | `schedule_compact_dense` |
| `retained` | whole K panel resident (`b.k == tiles_k`) | `emit_compact_dense` -> `dense_retained_operand_hoists` + `dense_retained_compute_order(order=("k","n","m"))` as `loop_schedule` transforms (`propose_loop_transforms`) -> plan `pipeline_kind="retained_full_k"` -> `schedule_dense_panel_pipeline` -> `propose_dense_array_retention` + `verify_dense_array_retention` |
| `segmented` | at least two K segments, the output block in half the accumulator, at least two full output groups along N | plan `pipeline_kind="retained_lhs_segmented_k"` -> `schedule_dense_segmented_pipeline` -> `batch_dense_segmented_readouts` -> `schedule_dense_readout_overlap(group_path=("outer_m","outer_n"))` |

What each prints: `calls` (projected call counts with loop multiplicities), the chosen `block`,
`retained_pairs` (array keeps the RHS: `stay` calls replace fresh `compute`/`preload` pairs),
`overlapped_compute_pairs`, `disjoint_readout_epochs` and `epoch_drains` before/after readout overlap.
On a larger shape (`--shape`) the blocks change with the stores: the retained block grows until its
output block fills the accumulator, the segmented block until it fills one of two windows.

A premise that does not hold for a shape is a refusal (`no capacity-legal block satisfies this
pipeline's premise`), not a fallback. In a lowering that is exactly where the next plan in your
knob's order applies.

## 4. Laws and adapter (yours)

The example adapter is neutral: placeholder instruction names and `test-only-*` law identities.
Supply your own `encode`/`project` (projection must decode the emitted operands, including the
fresh-capture/STAY pair for array retention), and back every admission field with your target's
facts: queue depths for pending counts, the completion and snapshot behavior for readout overlap. The
`scheduling-api` skill prints each admission's fields (`dense_schedule_example.py contract SYMBOL`).

## 5. Check (every plan)

* `execute(kernel, adapter, problem)`: runs the projected calls on host integers and compares with the
  source product, independent of the pass.
* `result.revalidate()` and `result.materialize_scf().verify()` (counted SCF ready for `emit_llvm`).
* Then numerics and cycles on your cases: Spike numerics for every class you change, and the timing
  comparison of `ab-library-pass`.

## Transplanting into a lowering

```python
def dense_plan(m, k, n, settings, facts):
    problem, storage, resources = shared_upstream((m, k, n), capacities(facts))   # one place
    plan = PLANS[settings["dense_pipeline"]]          # knob, default = today's behavior
    block = pick_block(dense_blocks(problem, resources), plan.premise)            # refusal if none
    return plan.build(problem, storage, resources, block)                         # schedule IR -> pass
```

The per-shape decision is then a knob value plus a capacity premise, both visible to
`shape_choice_table.py`. Keep dimension cutoffs in explicit policy predicates rather than
scattered through lowering. A new shape follows the predicates and its actual resource
checks; it never selects the nearest measured capsule or model name.

## Start with a small set of shape classes

Use broad geometry classes to organize experiments: tiny, thin-M, thin-N, balanced,
tall and wide. Define them using primitive-tile counts and aspect ratios from the
selected hardware, with explicit precedence so a shape has an unambiguous coarse
class. These are exploration groups, not claims that every member has one winner.

Treat reduction depth, aligned/ragged tails, layout, output precision and residency
as modifiers. Derive residency from the actual assigned live windows, including
buffering and accumulator storage. Do not enumerate every cross-product of tags.

Start each class with a few legal schedule families and retain its incumbent.
Compare representative shapes in parallel, then refine a class only where paired
cycles show a useful crossover. Check neighboring dimensions and independent held-out
shapes around that rule. Prefer a simpler rule when timing cannot distinguish choices.
Use `pipeline_policy` dimension/ratio/tile predicates and `shape_policy` assessment.
The existing `propose_shape_policies` accepts raw `axes=` and exact `features=`
(`RatioFeature`/`TileFeature`), searching training-only cutoffs through the same proposal
and assessment flow. Its requests identify missing feature-boundary facts and source
geometries. Selection at compilation is a cheap predicate lookup, not another simulator
search. Reuse exact matching evidence.

### Convolution and attention use the same policy

Keep source-family distinctions before reusing contraction geometry:

| Family | Coarse groups | Relevant modifiers and alternatives |
|---|---|---|
| Convolution | Pointwise, ordinary spatial, grouped/depthwise | Output-position/channel tile counts; kernel/stride/dilation/padding; sliding-row reuse, gather versus explicit packing, channel/output grouping, buffering and readout |
| Attention | Single-query decode, short-query, balanced prefill, asymmetric query/KV lengths | Query/KV tile ratios, head dimension, batch/head groups, shared KV heads, resident versus streamed KV, head batching and storage reuse |

Use explicit precedence for overlapping geometry descriptions. Residency and tails
are modifiers, not new operator classes. Pointwise reuses the dense policy when its
original layout permits the view; other geometries retain their legal source route.
Spatial convolution retains window geometry and layout while using output-position,
channel and reduction counts to organize shared blocking choices. A group/depthwise
label does not establish that the selected target has an efficient legal lowering.

Attention QK and PV can reuse dense scheduling choices, but keep batch/head axes,
masks, softmax/requantization boundaries and KV-cache recurrence visible in connected
regions. A class never authorizes a different numerical recipe or an online-softmax
algorithm. Grouping, fusion and streaming alternatives require their original source
and storage checks. Distinguish isolated-product timing from complete attention-region
timing, including host work, intermediate movement and state updates.

These are starting groups for the existing policy, not additional selectors or
implemented target capabilities. Classify with shape/layout/resource arithmetic once,
try a small initial set of applicable alternatives on representative performance cases,
and reuse matching evidence. Expand only when a measured crossover or missed opportunity
justifies it. Do not add startup profiling, launch gates, per-class compiler copies or
simulations during ordinary compilation. Small correctness capsules remain controls;
large and connected regions drive performance exploration.


## One checked entry for your existing dense backend

```python
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline

result = schedule_dense_pipeline(
    problem, resources, storage, adapter,
    name=entry, dtypes=dtypes, schedule_plan=typed_plan,
    source_identity=original_source_sha256, external_buffers_disjoint=True,
    pipeline_admission=selected_pipeline_law,
    batch_readouts=try_batching,
    readout_mode=readout_mode,  # "separate", "ready", or "disjoint"
    readout_admission=selected_readout_law,
    readout_group_path=group_path,  # disjoint only; () for separate/ready
)
result.emit_llvm(builder, lower_call)
```

The same problem, assigned storage windows and actual-operation adapter feed all
families. The typed plan selects compact, two-window panel, retained full-K, or
segmented K; the original family checks every source product, tail and final
hazard. Use `None` for both plan/source identity to retain the original compact
planner; no asynchronous law is consumed on that route. Every explicit advanced
decision executes or refuses, with no silent block or family substitution.

`lower_call(instruction, operands, metadata=None)` handles the actual emitted
commands. Ready emission supplies its actual SCF metadata. Real ordered issue,
backpressure and completion remain your target's independently checked duty.
Implement every explicit drain; fencing every data command instead prevents the
intended overlap. Capacity does not grant an execution law. The neutral example
uses test-only laws, so its host product checks are not hardware qualification.

For an integer exit, carry its original semantics into this same consumer through
`output_readout=`. `DenseTiledIntegerReadout.from_source_scales(source_scales,
columns=problem.n, tile_columns=problem.tile_n, ...)` accepts the original scalar
broadcast or complete column vector and retains exact binary32 coefficients. The
instruction name/arguments come from your actual store adapter. This normalizes
representation; it does not replace source arithmetic or choose numerical bounds.

Retained full-K and segmented-K plans can both consume `readout_mode="ready"`
when their independently checked completion law and actual storage layout admit
it. Uniform coefficients permit the supported causal integer readout; varying
coefficients keep the separately completed route. Revalidate and inspect emitted
SCF/LLVM. Loop-carried operand/array retention can require a conflicting order:
preserve it as a separate alternative and compare whole-call cycles across shape
classes. An admissible overlap is not automatically faster.

When timing is UNKNOWN, keep the incumbent and alternative artifacts. Inspect
emitted IR/LLVM and actual operand/configuration/fence/packing effects; derive
legality from source, storage and RTL facts, remove proven redundant work and
check the changed output/effects. UNKNOWN does not freeze legal exploration.
Rank with matched complete cycles once available; smaller code or fewer commands
alone do not establish a speedup. There is no required named-API usage quota.
