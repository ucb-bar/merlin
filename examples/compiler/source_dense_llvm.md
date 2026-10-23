# Checked dense schedules into ordinary LLVM emission

`schedule_source_dense` already derives a compact counted-loop program from a
typed integer contraction and selected hardware/software contracts. It owns panel
iteration, all dimension tails, first-contribution initialization, exact
accumulator continuation and stores after the complete reduction. Target authors
should not reconstruct those loops around a small `schedule_contraction` result.

The added `DenseSourceResult.emit_llvm(builder, lower_call)` connects that existing
program to Merlin's ordinary `FnBuilder`. `CompactDenseResult.emit_llvm` provides
the same bridge at the checked schedule layer. Both preserve existing custody
checks; the source wrapper also binds original IR and selected declarations.

```python
result = schedule_source_dense(
    module, source_operation,
    selected_contracts=selected,
    storage=assigned_storage,
    adapter=qualified_instruction_adapter,
    name="contraction", static_budget=budget,
    external_buffers_disjoint=True,  # An actual runtime premise, not SSA inference.
)
if result is None:
    # Retain an existing correct route or report precise unsupported coverage.
    return fallback()

builder = FnBuilder([PTR, PTR, PTR])
pointer_bindings = dict(zip((arg.name for arg in result.compiled.kernel.args),
                            builder.entry.args))

def lower_call(instruction, operands):
    # operands contains the checked Call's actual i64 SSA values.
    # The target owns instruction encoding, physical interpretation and completion.
    target.emit_instruction(builder, pointer_bindings, instruction, operands)

receipt = result.emit_llvm(builder, lower_call)
# Finish the function; add it and target declarations to the ordinary module.
# Verify, translate, link and independently execute that emitted module.
```

`source_dense_llvm.py` provides the corresponding small function-building example.
It is callable by an offline driver or an automatic author using the same API.
There is no target implementation, instruction sequence, preferred tile size or
new optimizer in the example. Existing `schedule_source_dense` policy/order
arguments still drive the shared planner. The target callback receives complete
logical arguments and actual scheduled instruction operands; it must not move
stores, infer different tails or reset the reduction independently.

Supported domain remains the existing checked rectangular, zero-initialized,
source-order signed-integer contraction with selected source/readout semantics,
explicit storage, justified synchronous completion and exact continuation. The
new bridge supports selected **64-bit indices**, ordinary increasing counted
loops and positive steps; it rejects 32-bit mode rather than silently widening
its contract. Bias-seeded contractions, unknown effects/completion, arbitrary
asynchronous schedules and general connected graph realization are not added.
The LLVM index bridge continues to refuse execution/configuration annotations it
cannot preserve.

Any source, selected-contract or checked-program mutation makes emission fail.
Discard a partial builder after failure. The returned receipt binds source and
schedule identities; it does not qualify arbitrary callback machine semantics.

The focused native tests translate emitted LLVM MLIR, compile it with an
independent neutral C instruction runtime, and compare all outputs against a
separate scalar matrix product. Cases use primitive geometry 2×3×4 with shapes
M×K×N = 5×14×13 and 1×7×9, so all dimensions exercise tails. Inputs and outputs
occupy disjoint arrays; every output must be stored exactly once. A deliberately
incorrect callback clears reduction continuation: its LLVM remains structurally
valid but independent numerical checking rejects it. Additional cases reject
source/contract mutation, unsupported index width and invalid loop direction,
step or signed bounds. These are neutral mechanism checks, not Gemmini execution,
gSIM qualification or performance measurements.
