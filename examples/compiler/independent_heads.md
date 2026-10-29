# Independent head and plane products

For the ordinary owned-host/model request, a candidate may write
`source_composition.json`:

```json
{
  "schema": "merlin.source-composition-preferences.v1",
  "shared_input_siblings": false,
  "prepack_weights": false,
  "independent_heads": true
}
```

The original owner selects `HeadBatchSourceRoute` only when every composed head
has independently admitted source-operation evidence. It refolds the exact source
then uses ordinary group compilation and the existing counted batch-slice shim.
Unmatched admitted groups stay on their ordinary accelerator route; all original
host readouts stay intact. No candidate-supplied host LLVM enters this request.
Inspect the verified composition's `batch_abi_calls`, `matrix_products`, members,
and bridge invocation count. Requesting this flag alone establishes no application.

Add `"whole_head_batches": true` alongside `"independent_heads": true` to
request the existing full rank-three batched interface instead of a per-product
shim. The target compiler must implement the entire declared batch and return
only after every original output is complete. This enables checked storage and
phase schedules across products; the request supplies no hardware effect law.
The owner checks the complete object output budget and reports mathematical
products separately from kernel calls. Unknown target forms must refuse or use
the ordinary synchronous batch implementation. The default remains per-product.
Whole-batch selection currently defers requested sibling composition; it does
not claim a combined head/sibling scheduler.

If both `independent_heads` and `shared_input_siblings` are requested, the owner
can select independent head families and compatible sibling components together.
Unmatched admitted groups retain ordinary lowering. Inspect the composition
report to distinguish joint selection from either individual route; requesting
both does not establish joint fusion or overlap. Missing or unsupported source evidence
retains the ordinary route, and normal placement, object, work and numerical
checks remain required.

Use the additional compiler source selection for these APIs. The compact SDK's
older inventory does not acquire new modules by importing this example.

```python
from merlin.llvmlower.integer_batched_matmul_calls import lower_integer_batched_matmul_program

program = lower_integer_batched_matmul_program(
    original_tensor_source,
    output_directory,
    select=lambda call: supported_matrix_geometry(call.m, call.k, call.n),
    emit_matrix_kernel=emit_original_synchronous_matrix,
    logical_abi=selected_logical_borrowed_abi,
    data_layout=selected_data_layout,
    refold_heads=True,
)
```

`emit_original_synchronous_matrix(call)` returns LLVM and immutable `(Path,
sha256)` implementation pins for **one** matrix product. Link `program.host`,
`program.kernels`, and `program.bridges` through the ordinary toolchain and
compare every original output. No target implementation is supplied here.

The source transform recognizes complete matching head slices from original
rank-three operands. It preserves all original observers, quantizers, and
plane-combination arithmetic. Its inverse proof restores the complete original
SSA module. Missing heads, unknown crossed effects, or unsupported views retain
the ordinary source route. Check `program.preparation.calls` to distinguish an
applied route from that fallback.

For explicit plane geometry, `BatchedDenseProblem((2, 2, heads),
BatchMatrixView((2, heads, m, k), (0, 2)),
BatchMatrixView((2, heads, k, n), (1, 2)))` describes all four independent plane
products per head. Read-only inputs can omit independent axes; output storage
covers every product. This geometry alone does not refold arbitrary source.

Record three counts separately: outlined tensor ABI calls, matrix primitive
invocations, and actual target commands. A runtime head loop still executes the
synchronous primitive for every original product. It establishes neither
cross-head overlap nor a reduction in accelerator work. Inspect actual emitted
allocation/copy sites and runtime traffic; rank performance only with complete,
identity-matched cycle feedback. UNKNOWN timing does not prevent checking legal
structural alternatives, retaining the original route, and comparing outputs.

For coarse whole-forward diagnostics, add `"group_profile": true` to the same
request. It selects the ordinary group-boundary profiler and reports grouped
counter ticks and gaps between groups through the usual per-tier feedback.
The gaps include orchestration and instrumentation overhead; they are not a
measurement of scalar arithmetic alone. Functional Spike ticks are not elapsed
cycles. Keep an uninstrumented control for performance comparisons. Use
`"op_profile": true` separately when finer operation attribution is needed.

The executable native checks in `test_logical_borrowed_batch_abi.py` cover varying
head/plane/shape geometry, shared read-only operands, transpose, pitched output,
and aliases. `test_integer_batched_matmul_calls.py` exercises the ordinary source
lowering and linking path with an explicit independent matrix fixture.

## Host layouts and intermediate storage

The same `source_composition.json` accepts optional Boolean
`propagate_layouts` and `reuse_intermediate_storage` requests. The original owner
resolves permissions from the selected frozen numeric recipe's software
declaration. These flags grant no permissions: a declaration without the required
host-optimization premises retains the original code and reports
`permission_not_admitted`.

Layout propagation runs before complete device-group binding. For head/sibling
composition it runs on the checked projected host before normal host lowering;
the receipt binds that exact input and preserves every device call as an opaque
coordinate/effect boundary. This does not propagate layouts across device calls
or change their schedules. Storage reuse runs after binding on actual host
LLVM, using proven allocation/free lifetimes and the original sequential,
successful-allocation and non-reentrant-entry permissions. Escaping/loop-carried
allocations stay on the heap; KV recurrence is not inferred to be disposable.
The compact `source_composition.features` feedback distinguishes requests,
selected policies, actual rewrites, unchanged code and deferred reasons. Neither
reused storage bytes nor removed layout operations establish a cycle speedup.
