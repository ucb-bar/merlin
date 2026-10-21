# AGENT.md — src/merlin/semantic_compiler

This package owns target-independent typed semantic graphs, generated instruction
rules, native search, extraction, allocation, and independent checking. Keep
target-specific facts in explicitly selected out-of-tree support packages.
The production path must not import or invoke ACT. Reference adapters belong in
a separate module and may only enter through an explicit engine selection.

Kernel identities reuse canonical serialization only when every source field,
attribute pair and affine-map container is immutable. Legacy list-based callers
must rehash after mutation. Caches are excluded from records and reset by
dataclass replacement; never normalize away a semantic field to improve reuse.

The Rust `egg_bridge` is a general-purpose e-graph dependency boundary. Merlin
owns all rule eligibility, buffer-aware extraction, constraint construction,
fallback and result checking. Unknown semantics or solver results are failures,
not evidence that a target operation is unsupported.

Extraction reuses lazy choice streams within one graph's bounded search session.
The key includes storage, layout, remaining budget and ancestor contexts from
the current strongly connected component. Ancestors outside that component
cannot be revisited. Preserve the cycle context, candidate ordering, bundle
identity and serialized choice keys when changing this cache. A changed graph,
request, rule program or deadline requires a fresh session.
On a repeated storage request, an instruction may finish at a producer whose
operands leave the component. This retains finite spill/store/reload paths;
further traversal inside the same cycle remains excluded. Computational
realizations precede copy alternatives, which still require allocation checks.
Search tries the geometric work stages directly and accepts the first checked
candidate. It preserves the configured maximum instruction/candidate counts;
minimizing instruction count is a separate optimization policy.
Extraction skips a node budget only when a conservative component bound
exceeds it. Intersect obligations across all finite SCC exits, retain sharing
across roots, and count separately required boundary issues. Multi-result
bundles disable this single-result bound. Storage/layout uncertainty must
weaken the bound rather than discard a valid implementation.

The allocator rejects a live-capacity violation only when the affected backing
store has no views or in-place permissions. This is a necessary bound for one
candidate and order. All other cases still use the full placement formula;
neither a capacity rejection nor a timeout establishes source infeasibility.

`scalar_region.py` preserves closed pointwise source SSA, casts, Boolean
projections and literal-boundary counts. Binary64 remains binary64; source
math intrinsics remain unqualified. `scalar_reference.py` independently checks
the admitted arithmetic and indexing. Neither module grants target placement
or a host fallback. Runtime-sized masked compaction and general source control
flow still require separate translation.

`constant_indexing.py` specializes only declared constant signed-i64 indices.
It checks every output coordinate for bounds and agreement with an affine map;
runtime indices and non-affine reads cannot enter by this route. Keep its work
limit explicit and avoid materializing unrelated model weights.

`ordered_scan_source.py` recognizes the complete source loop for f32 prefix
scans with a separate f64 accumulator. Check initialization, lane reset, every
loop-carried tensor and both rounding boundaries. The private accumulator must
not escape. Target lowering and constant folding remain separate decisions.

`tensor_assembly.py` tracks static, disjoint full-axis writes into tensor.empty.
Only complete coverage produces a semantic value. Dead checked views do not
read elements; live partial reads, holes and unsupported overlapping writes
remain errors. Do not initialize unwritten lanes to zero or discard encodings.

Unannotated BF16 source arithmetic has its own IEEE RNE primitives. Do not
identify it with a target's similarly named VPU operation. The independent
reference covers finite values, subnormals, signed zeros and infinities; NaN
payload behavior remains explicitly unqualified.

Repeated generated splats share their semantic value and exact bytes. Their
cache key includes shape, dtype, numerical policy and element encoding. Do not
merge runtime inputs or assume a shared value needs only one physical buffer.

`constant_folding.py` specializes only declared byte-bound constants under its
exact integer/IEEE primitive policies. Retain node IDs and output and state wiring;
keep the original nodes and byte hashes in the stage receipt. Its replay checker
reconstructs the original source without duplicating parameter payloads. Work
limits, runtime values and unqualified math stay deferred. Keep the production
integer arithmetic separate from the independent rational reference evaluator.
The application path also eliminates unused pure computations. Its v2/v4 receipt
retains original positions and nodes for replay; declared input ports, constants,
ordered outputs and state dependencies are preserved. Never erase an effect or
evaluate an unqualified intrinsic to eliminate its dead result.
Receipts v3/v4 admit exact f32 arithmetic in addition to the earlier primitive
set. Replay v1/v2 with that older set; never apply newly available folds while
checking an existing receipt. Mathematical power remains unqualified.
`source_preparation.py` may apply a separately selected constant-math precision
policy between exact passes. Its MPFR helper accepts only typed positive power
operands and requires matching directed bounds after IEEE rounding. Pin the
helper and loaded libraries; never load an IR-supplied library path. Reconstruct
and replay every stage. A precision/work limit is inconclusive, and framework
equivalence remains unqualified. Generated programs do not depend on this helper.
Tensor constant folding copies exact bits through logical identity/reshape and
checked affine gathers. Ordered f32 scans keep an f64 accumulator and round each
output separately. Do not use runtime samples, weaken bounds, or turn semantic
movement into a physical alias. Historical receipts replay their original
primitive set; the v2 preparation sequence enables tensor folds explicitly.
Declared constant comparisons, Boolean operations and selects use the shared
bit-level predicate primitives. Ordered IEEE comparisons keep NaNs unordered
and signed zeros equal; select copies the chosen bits, including NaN payloads.
These folds require the exact types, policies and identity maps. The C emitter
shares their primitive arithmetic, while the semantic reference stays separate.

`guarded_softmax_source.py` recognizes Boolean exclusion masks and the explicit
zero output for fully masked rows. Check both maximum operands, literal bits,
affine broadcasts and all outside uses. Numerical qualification is separate.
The tiled zero-row policy floors the denominator by one before reciprocal;
the provider must establish exp(0)=1 and nonnegative, finite row sums. Do not
erase the source guard by assuming every runtime row has a kept element.

`affine_views.py` commutes exact BF16 widening with full-source dimension
projections and absorbs those gathers into checked F32 pointwise operands.
Complete source coverage preserves input validation. Keep partial/strided views
deferred, preserve escaping values and replay original/generated SSA nodes.
Source-preparation v3 wraps the earlier numerical receipt; historical receipts
retain their earlier pass sequence. The snapshot pins the indexing pass source.
