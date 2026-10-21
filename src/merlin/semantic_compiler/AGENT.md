# AGENT.md — src/merlin/semantic_compiler

This package owns target-independent typed semantic graphs, generated instruction
rules, native search, extraction, allocation, and independent checking. Keep
target-specific facts in explicitly selected out-of-tree support packages.
The production path must not import or invoke ACT. Reference adapters belong in
a separate module and may only enter through an explicit engine selection.

The Rust `egg_bridge` is a general-purpose e-graph dependency boundary. Merlin
owns all rule eligibility, buffer-aware extraction, constraint construction,
fallback and result checking. Unknown semantics or solver results are failures,
not evidence that a target operation is unsupported.

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
