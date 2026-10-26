---
name: scheduling-api
description: Call a library scheduling pass from your own backend. Open it when you want to reuse a pass from merlin.compile.scheduling (segmented-K pipelines, weight retention, readout overlap, loop transforms) instead of hand-writing a schedule, when the pass-library skill or the performance brief names a mechanism you want to try, or when a pass refuses your inputs. Covers discovery through scheduling_capabilities(N) for the inventory N your TASK.md names, a library inventory with a per-family pass status table (scripts/library_inventory.py), reading one pass's contract, the dense planning input form, the target adapter you supply, and checking the result.
---

# Scheduling API: discover, read a contract, call a pass

The passes take a fixed input form (dense planning objects or schedule IR) plus an **adapter** that
your backend supplies. The runnable version of every snippet below is
`scripts/dense_schedule_example.py` (run it from the workspace; it puts
`submission/compiler_library` on `sys.path`).

## 1. Discover

```
python library_inventory.py                    # -> submission/docs/library_inventory.md
python dense_schedule_example.py discover      # names, symbols, usage, requires
```
`library_inventory.py` lists capabilities, composition rows and available catalogue mechanisms, the
ones your files import and a per-family table to keep current. Both read `scheduling_capabilities(N)`
(`api_catalog`) for the `inventory_version` TASK.md names. A row's `requires` lists what your target
declares; nothing is inferred.

## 2. Read one pass's contract

```
python dense_schedule_example.py contract merlin.compile.scheduling.dense_segmented_pipeline.schedule_dense_segmented_pipeline
python dense_schedule_example.py contract merlin.compile.scheduling.dense_segmented_pipeline.DenseSegmentedAdmission
```
It prints signature, docstring, dataclass fields and source line. For each pass, note:
- its **inputs**: usually `(problem, resources, storage, adapter)`, a `source_identity` and a budget;
- the **admission/law** dataclass it needs (`*Admission`). Its identities name evidence your target
  provides. Its issued/pending counts come from your own queue facts;
- what it **refuses**. Each `ValueError` message names the premise that is missing.

## 3. Express the kernel in the planning form

```python
p = DenseProblem(m, k, n, tile_m, tile_k, tile_n, lhs_bytes, rhs_bytes, acc_bytes)
s = DenseLoopStorage(DenseMemoryView("left", "left_sram", row_bytes, base_row, rows), ...rhs..., ...acc...)
r = DenseResources(operand_bytes, accumulator_bytes, max_m_blocks=2, max_n_blocks=2, max_k_blocks=2,
                   partial_output_tiles=True, partial_output_blocks=True)
adapter = DenseLoopAdapter(encode, project, config_id, numeric_policy_id,
                           synchronous_completion, exact_accumulator_continuation, index_bits)
```
Take the primitive tile, row widths and capacities from your own workspace facts (`rtl_facts/`, the
capability manifest), after your other reservations. `encode(op) -> Call` lowers a generic
`Load/Preload/Compute/Store` to one of your instructions. `project(call) -> op` decodes an actual call
back. Projection has to read the emitted operands; returning the op that went in does not count.

## 4. Call the pass

```python
base = schedule_compact_dense(p, r, s, adapter, name="dense", dtypes=("i8","i8","i32"),
                              static_budget=128, external_buffers_disjoint=True)
plan = make_dense_schedule_plan(p, r, s, adapter, source_identity=src_sha, dtypes=("i8","i8","i32"),
                                static_budget=128, block_tiles=(2, 2, 2),
                                pipeline_kind="retained_lhs_segmented_k", pipeline_prefetch=True)
text = dense_schedule_text(plan)          # editable decision IR
result = schedule_dense_segmented_pipeline(p, r, s, adapter, admission, name="seg",
    dtypes=("i8","i8","i32"), schedule_plan=parse_dense_schedule_plan(text),
    source_identity=src_sha, external_buffers_disjoint=True)
```
`block_tiles` has to be one of `dense_blocks(p, r)`. Choose it by measurement, not by a constant.

## 5. Check before you keep it

- `result.revalidate()` raises if the kernel, the law or the library changed after checking.
- `result.materialize_scf().verify()` produces counted SCF. `result.emit_llvm(builder, lower_call)`
  hands each instruction, with i64 operands, to your lowering callback.
- Run the projected calls on host integers and compare with the source product (`execute` in the
  script does this). Then compare `result.realization` (loads, stores, overlapped pairs) and
  measured cycles with your previous version on the same cases.
- `read-command-buffer` shows whether the emitted program changed as the realization says.

## Composition

To chain passes, choose an order, or A/B two orders, see the `compose-pass-pipeline` skill. For
fusing producers and epilogues, see the `fusion` skill. `pass-library` lists every mechanism with
its module, its adapter obligations and a test that calls it. Capacity-derived dense plans: the
`shape-schedule-selection` skill. Host-checked recipes (readout overlap, long K, convolution, a
conv-pool chain, per-shape variants): the `heavy-pass-recipes` skill.
