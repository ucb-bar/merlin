# Shape normalization: whole-module conditional graph preservation

**Proposed paper sentence (independent review pending):** For the admitted static,
identity-layout view domain, our Lean checker proves that identical-type memref
cast elimination and static dimension materialization preserve every possible
complete returned value and the entire heap/public world, under the same partial
interpretation of retained operations. A trusted Python matcher binds the complete
actual source and result modules to the checked blueprint. Native descriptor
fidelity, projection review and ordinary compiler gating remain separate.

This is the registered `shape_view_normalization` from pinned Merlin producer
`4499eebb959c0f92cefd92cc6d80646ba66474ed`,
`src/merlin/transforms/normalization.py::normalize_shapes`. The old host broker
established local identity obligations. This extension derives preservation
through typed lexical SSA aliases, every nested region, internal calls, outputs,
caller memory and surrounding functions. It does not identify the frozen
MerlinVerify evaluator with these new graph semantics.

## Relation, domain and assumptions

`GraphLoopsShape.Model.checked_complete`, `checked_memory`, `checked_context`
and `Module.checked_calls` are the whole-boundary roots. `preservation` derives
block preservation; `Agreement.alias/define_many` relate private SSA stacks.
`identical_cast`, `cast_descriptor`, `cast_address` and `static_dimension` supply
the local operational laws. Existing StateEffects/ResourceGeometry and
GraphLoopsReuse/Segment/MemorySCF definitions are imported unchanged; imported
roots alone do not grant the unproved adjacent memory-model bridge.

The relation is equality of **possible complete outcomes**, for all typed entry
values and all common interpretations: returned Runtime values and complete
World, specialized to MemoryWorld(heap, public). It does not claim equality of
categorized exceptions or divergence. Complete finite call-depth derivations
cover internal functions; external primitives remain shared partial relations.
No source-success, retained primitive totality or numerical law is assumed.

Admitted actual IR has static ranked integer memrefs with identity layout,
default memory space and element widths 8/16/32/64; scalar values also admit i1.
Index width is explicitly selected as signed 32 or 64. Every folded dimension
and known index literal must fit that domain. Identical casts must preserve the
entire descriptor, including allocation identity/base/offset/strides. Runtime
shape/type consistency is represented by typed descriptor proofs, not inferred
from a hash of source metadata. Unknown layouts/encodings, dynamic views, unknown
view construction and unsupported region/control forms refuse the whole pair.
Zero extents support metadata-only examples; an empty-view read is not inhabited.

Retained action/region meanings receive explicit arguments and World, never the
private SSA environment. Typed result/carry conditions remain explicit. Concrete
load/store meaning checks every coordinate and logical element address; SCF
forwards all carries and the entire World in order. Logical element addresses
are not a native byte/pointer ABI proof. Standalone shape normalization permits
allocation and other retained operations to be partial and observable. The old
reuse edge additionally requires total/private/unobservable allocation; the
actual 8→16-byte peak-storage counterexample remains unchanged.

## Actual IR mapping

| Actual compiler field | Checked representation / binding |
| --- | --- |
| Complete module/function signatures, visibility and returns | Full source/result trees plus typed `Module.functions` and declared call targets; untouched functions matched |
| Every result/block argument and operand | Parsed stable SSA identifiers and types; scoped forward aliases, freshness, bound-name and return-type checks |
| Identical memref.cast source/result type | Exact type record equality; `Code.cast`, typed Runtime descriptor and alias update |
| memref.dim view, constant axis and candidate constant | `Code.dimension`, known-axis binding, static shape lookup and signed-width proof |
| Shape, layout, offset, strides, element and address space | Explicit ViewType; unsupported actual encodings refused; runtime Conforms remains a separate premise |
| Retained opcode, operands, attributes, properties and locations | Exact complete-tree matching and shared action key; loads/stores specialize to explicit descriptors |
| SCF controls, body parameters, captures, carries and yields | Lexical `Code.region`, checked typed body/results and concrete ordered SCF handler |
| Actual internal/external calls | Entire function table and target properties, finite-depth invocation with common external meaning |
| Full source/result bytes, model and emitted Lean query | Fresh parse from saved input buffers; postchecks of original and saved bytes, export/model/query, provider/parser/host and proof context |

The parser, exporter, tree matcher, operation classification and theorem emitter
remain trusted Python. Lean checks the resulting structural/type/domain facts
and the general preservation theorem. The Python implementation itself has not
been formally verified or independently approved.

## Worked actual example

The unabridged production input/output are [before.mlir](before.mlir) and
[after.mlir](after.mlir). The source casts `%a : memref<2x3xi32>` twice, reads
axis 1 through the aliases, reads axis 0 of `%b`, loads `%a[0,0]`, stores into a
caller sink and returns both dimensions plus the loaded scalar. The output
removes both casts, replaces the dimensions by 3 and 2, and redirects the load
to `%a`. The separate `@untouched` function remains in the checked boundary.

```mlir
// Source excerpt; see the complete files for constants and surrounding context.
%first = "memref.cast"(%a) : (memref<2x3xi32>) -> memref<2x3xi32>
%second = "memref.cast"(%first) : (memref<2x3xi32>) -> memref<2x3xi32>
%d0 = memref.dim %second, %axis : memref<2x3xi32>
%d1 = memref.dim %b, %other : memref<2x3xi32>
%value = memref.load %second[%zero, %zero] : memref<2x3xi32>
// Result excerpt:
%d0 = arith.constant 3 : index
%d1 = arith.constant 2 : index
%value = memref.load %a[%zero, %zero] : memref<2x3xi32>
```

[CHECKER_OUTPUT.json](CHECKER_OUTPUT.json) retains the complete receipt reference,
source/result/model/query digests, all four changed-site mappings, return code 0,
`lean_complete_shape_checked=true` and the exact 35-root standard-three audit.
The full receipt is under `standalone-mode-v1/qualification/pytest/`. Six actual
nonzero Lean witnesses audit 41 roots each, including a compact huge shape.
Finite xDSL execution independently returns `[3, 2, 17]` and writes 17 to the sink.

A compiler-valid candidate loading the same-shaped `%b` instead returns
`[3, 2, 29]`; [the finite counterexample](WRONG_BUFFER_COUNTEREXAMPLE.json)
retains both observations. The fresh matcher refuses that operand change.
Other valid negatives cover wrong dimensions, coordinates, captures, callees,
returns, extra effects, missed sites, changed types/layouts and early/double/missing
frees. Wrong dimension values and signed-range overflow reach Lean and are refused.
Malformed parser inputs are recorded separately from these valid negatives.

## Qualification and remaining work

`standalone-mode-v1/QUALIFICATION.json`: 78 original tests and **one** retained
fresh 78-test replay, zero failures/errors/skips, identical loaded runtimes and
unchanged source/tool/import bindings. Exactly 22 new sources and 38 project
proof modules belong to this standalone profile. The actual import inspection
covered 1110 modules, excluded the suspect Std.Time.Format.Basic module and found
zero missing/changed original bindings. The external SDK incident and the other
worker's elision crash remain unresolved; neither context was repaired or promoted.
A deterministic packaging-only prior-DELIVERY schema error is retained separately;
no qualification or Lean run was retried to repair it.

Eight changed actual families cover rank1/2/3, all four memref widths, explicit
32/64 index selection, nested captures, repeated casts, multiple dimensions,
zero extents, a huge compact shape, a changed callee and the two-epoch memory input.
Finite executions include aliased input descriptors and initialized private memory.
These finite executions are Python/xDSL evidence, not native/hardware proofs.
The structure gate retains the unrelated `zephyr_model.py` module-size failure.

[COMPOSITION_REFUSAL.json](COMPOSITION_REFUSAL.json) binds the actual
shape→reuse→segment buffers and accepted old-memory edge. The normalized middle
is byte-identical. The typed/full-heap to epoch/public/initializer adapter is
**still unproved**, so this packet refuses a combined semantic claim. A new
separately owned adapter module is development work and is not covered by this
standalone seal or its audit. It must derive scoped SSA, initializer, caller-heap
and retained-operation correspondence before composition is admitted.

Graham owns independent projection/contract review. Full Lean-MLIR selection,
native descriptor/operator/allocator/ABI fidelity, numerical laws, LLVM/OOT/backend,
hardware behavior, model captures and ordinary compiler receipt gating remain
separate. The panel-pipelining gSIM figure is independent execution evidence.
This row does not claim all graph transformations or the complete compiler correct.
Absolute Linux receipts are historical evidence on another machine, not laptop
qualification; copying sources does not establish portability or repair context.
