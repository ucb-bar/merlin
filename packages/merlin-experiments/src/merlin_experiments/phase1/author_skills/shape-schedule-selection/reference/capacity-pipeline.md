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
`shape_choice_table.py`; no `if n % 32` or `m >= 128` remains in the lowering, and a new class of
shape is handled by the premise rather than by the nearest measured case.


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

When timing is UNKNOWN, keep the incumbent and alternative artifacts. Inspect
emitted IR/LLVM and actual operand/configuration/fence/packing effects; derive
legality from source, storage and RTL facts, remove proven redundant work and
check the changed output/effects. UNKNOWN does not freeze legal exploration.
Rank with matched complete cycles once available; smaller code or fewer commands
alone do not establish a speedup. There is no required named-API usage quota.
