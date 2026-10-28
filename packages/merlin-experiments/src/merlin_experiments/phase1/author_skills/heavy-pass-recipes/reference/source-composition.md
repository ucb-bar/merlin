# Source-composition requests for model and program checks (`submission/source_composition.json`)

A model or program check builds the original host program around your device groups. That owner
can apply a few source-level compositions when you ask for them. You never select source
operations, change numerical boundaries or grant effects: the owner re-derives membership from the
source and keeps the ordinary route wherever a request does not apply.

## The file

```json
{
  "schema": "merlin.source-composition-preferences.v1",
  "shared_input_siblings": false,
  "prepack_weights": true,
  "share_exact_tensor_preparations": true
}
```

`schema`, `shared_input_siblings` and `prepack_weights` are required; every other field is an
optional Boolean. Unknown fields, non-Boolean values, a repeated field, a symlink or a file over
8 KB are refused. Leaving an optional field out keeps the earlier request identity.

| option | asks the owner to | when the case shows |
|---|---|---|
| `prepack_weights` | prepare immutable weight layouts once, outside the per-call path | weights that reach a product through layout, cast or quantization regions |
| `share_exact_tensor_preparations` | merge exact duplicate view, constant and integer-to-float preparations | the same tensor prepared more than once |
| `dense_input_views` | pass proven dense read-only views (offset, row pitch) instead of layout copies | products reading reshaped or permuted views |
| `propagate_layouts` | request checked layouts before device-group binding | explicit source layout regions; the original software must admit element-order and any required floating-environment changes |
| `reuse_intermediate_storage` | reuse proven final host allocation lifetimes | connected source tensor edges; the original software must admit LLVM storage reuse, sequential execution, successful allocation and non-reentrant entry |
| `shared_input_siblings` | compose sibling products that read one input into checked components | several products reading the same tensor |
| `independent_heads` | refold complete independent head slices through the batch-slice interface | products over slices of one tensor, or batched head operands |
| `whole_head_batches` | retain the complete ranked batch interface, together with `independent_heads` | a target consumer that implements every original product with explicit storage, effects and completion |
| `op_profile` | add an instrumented per-operation profile | host-placed operations you want to attribute (diagnostic) |

With both `independent_heads` and `shared_input_siblings`, heads are tried first and siblings use
the remaining products. `op_profile` markers can change fusion and scheduling: compare timing on an
uninstrumented build. Its counter ticks are not cycles.

The default head bridge invokes a synchronous primitive for each product. With
`whole_head_batches`, the backend receives the complete ranked batch instead.
`BatchedDenseProblem` and `counted_batched_dense_products` in
`merlin.compile.scheduling.batched_dense` construct counted `Kernel`/`Loop`/`Call`
control and exact **element offsets** for projected batch axes. The authored
consumer supplies element widths, instruction lowering, storage, effects and
completion. This helper does not combine products or establish legal overlap;
compare complete-call cycles and verify all original outputs for alternatives.

Layout/storage requests grant no permissions. Missing original software premises retain
unchanged code with `permission_not_admitted`. Selected head/sibling composition applies layout
planning after checked source projection. Source connections do not prove disposable intermediate or
KV state: final allocation/free custody must justify storage reuse; unsupported sites remain.

On the ordinary uncomposed route, `propagate_layouts` also considers supported ordered generic
contractions connected through exact pointwise/quantization regions. Input permutations change
index maps; initialized reduction destinations, reduction order, public layouts and extra observers
remain. Unknown operations keep explicit boundary conversions. Inspect the actual rewritten source
and layout report before deciding whether your device lowering benefits.

`reuse_intermediate_storage` carries the current native-checked synchronous borrowed-call effects
to the final arena allocator. Actual completion, aliases and release determine reuse; graph last use
alone is insufficient. Saved reports cannot recreate that live binding. The existing wide-interface
annotation preserves it only after unchanged-body conformance; other late LLVM changes retain
conservative allocation. `allocation_refusals` in feedback distinguishes escaping pointers,
descriptors, loop allocation sites and absent/nonunique releases. Arena bytes describe compiled
storage, not measured peak occupancy or elapsed cycles.

Ordinary single-output logical device groups can expose their checked destination descriptor
through the same request when the original caller explicitly supplies its live
`descriptor_effects` factory to `execute_owned_host_model` and `audit_owned_host_model`.
This uses the existing `BorrowedBufferCalls` interface for the actual prepared source, exact
argument/result positions and layouts, compiled shim and kernel objects. The candidate request
and host-storage policy grant no noescape, release or completion effects. Missing qualification
keeps the opaque route; spatial/sibling composition and later host LLVM rewrites still refuse this
bridge. Inspect `output.descriptor_results` alongside the final storage report.

## Find what applies, then check what happened

```
python .agents/skills/heavy-pass-recipes/scripts/composition_routes.py --case NAME
```

prints, for a model or program, the matching options with the IR observation behind each and a
complete example file. After the next model or program self-check, each row's `source_composition`
reports every option as `requested`, `selected` and a status: `applied` (with effect counts),
`selected_no_rewrite`, `selected_no_applied_evidence`, `applied_evidence_unavailable`,
`unchanged` (with reason), `deferred` (with reason), `not_selected`, `not_requested` or `request_unavailable` (`observed` for `op_profile`). Selection
is not application, and application is not a performance result: counts describe source and
emission effects, never elapsed cycles. Keep a request when the same cases still pass and their cycles of
the same evidence kind do not rise.

The library routes behind these requests (`HeadBatchSourceRoute`, `SiblingSourceRoute`,
`weight_prepack`, `tensor_provenance_cse`) are listed by `composition_routes.py` with their catalog
rows. They need modules from the additional compiler source selection; the compact SDK may not
hold them.
