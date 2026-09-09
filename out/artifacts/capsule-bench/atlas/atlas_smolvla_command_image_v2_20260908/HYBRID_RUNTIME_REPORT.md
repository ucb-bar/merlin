# Atlas SmolVLA hybrid schedule status

The artifact now has an executable schedule constructor, but it intentionally
fails closed for whole-model execution. `build_hybrid_schedule.py` reconstructs
the live capture inventory, emits 6,104 deterministic capture-order events, and
records every accelerator, host, layout, and quantization boundary. It does not
promote structural partitions to executable partitions.

The earlier 2,033 “layout bridge” count needed refinement. These are candidates,
not aliases: 1,675 `view`/`unsqueeze` regions are proven metadata-only from
equal element counts and reshape/cast operations. The remaining 358 regions are
now classified into 246 `expand` and 112 `copy` operations, spanning 25 exact
shape classes. Of those, 291 are identity materializations and 67 are
constant-zero-axis broadcasts. A fail-closed host bridge lane accepts all 358
only after proving one parallel `linalg.generic`, an identity output map,
singleton-only broadcast axes, unchanged dtype, and an exact input-value yield.
It produces a distinct C-contiguous tensor. This is valid host materialization,
not a zero-copy strided device descriptor or physical DMA implementation.
Independently, 2,430 regions require host semantics. The prior schedule treated
22 regions covered by one scoped bridge as implemented, leaving 2,408
unresolved.

The generic host semantic lane now extracts and validates complete operation
signatures instead of trusting provenance names. Its first tranche admitted
1,396 real-capture regions: 472 `dtype_cast`, 536 `mul`, 215 `add`, 68 `sub`,
56 `div`, four `compare`, and 45 conditional `select` regions. Admission
requires one terminal
`linalg.generic`, parallel iterators, an identity output map, input maps composed
only of distinct loop dimensions and constant-zero broadcast axes, supported
tensor dtypes, and a scalar body whose root matches the declared semantic.
Two other `select` regions are actually `aten.select.int` slice/reshape chains;
they are excluded from this pointwise tranche and handled by the exact movement
signature below.

The second tranche admits all 353 captured unary/scalar-expression regions by
their complete scalar DAG: 123 `pow`, 66 `rsqrt`, 57 each of `sin` and `cos`,
33 `sigmoid`, 12 exact-erf `gelu`, three reciprocal `elementwise`, and two
`minmax`. Each intermediate is cast to its declared scalar result dtype; BF16
uses round-to-nearest-even, and affine broadcast rules are unchanged. Composite
patterns such as sigmoid and GELU must match their full ordered operation DAG,
not merely their terminal arithmetic operation.

The constructor tranche additionally admits all 113 captured constructors: 63
`arange` and 50 `fill` regions. `arange` must be a rank-one, dimension-zero
index lowered through one of the two exact captured integer/floating scalar
DAGs; its step and start constants, casts, output dtype, and static extent are
part of the signature. `fill` must be exactly one captured scalar constant
consumed by one `tensor.splat`; scalar tensors, booleans, finite values, and
negative infinity are preserved in their declared dtype. Provenance alone is
not sufficient for either constructor.

The reduction/composite tranche admits all 150 captured regions: 66
`reduce_mean`, 44 stable `softmax`, 25 two-pass `layer_norm`, eight
`aten_min_dim`, four `cumsum`, and three boolean-to-integer `reduce_sum`.
Admission checks the complete top-level operation sequence, affine maps,
iterator kinds, reduction dimensions, scalar SSA dataflow, comparison
predicates, index dimensions, constants, dtypes, and reshape element counts.
Execution follows the captured row-major left fold and casts after every scalar
operation. In particular, `cumsum` direction and `aten_min_dim` first-tie
behavior come from their actual SSA operands rather than their provenance name.

The movement/indexing tranche admits 412 more regions: 129 static `slice`, 112
static `slice_scatter`, 95 `cat`, 56 two-output `split`, 16 exact boolean
`bitwise`, two static `select` slice/view chains, and two `bucketize` reductions.
Every slice records and bounds-checks its static offsets, sizes, and strides;
concat checks its axis and every non-concat extent; view steps must preserve
element count and dtype. The optional BF16-to-F32 concat cast and the bitwise and
not scalar bodies are extracted, not inferred from provenance. `bucketize`
admits only the captured ordered-less-equal/count reduction and executes as a
row-major reduction. Fresh real-capture chains exercise every admitted family,
and malformed multi-slice and unsupported-bitwise controls fail closed.

The final indexed tranche admits the remaining six regions from their complete
capture topology: two embeddings, one two-index boolean gather, a linked
mask-gather/index-put pair, and the patch-embedding im2col convolution. Indexed
loads sign their affine operand maps, table shapes and dtypes, scalar
index-cast/extract dataflow, and enforce runtime bounds. The mask pair signs the
complete count-reduction and `scf.for`/`scf.if` compaction/scatter topology;
its dynamic update count must equal the mask population. The convolution signs
the 16x16 no-overlap im2col affine map, every reshape/dataflow edge, f32 matmul,
and channel-bias epilogue. All six execute on fresh real capture shapes against
independent NumPy oracles, including the full 512x512 patch input. Malformed
topologies and runtime bounds/count violations fail closed.

The scheduler marks a host region executable only when that extracted
signature is accepted. It now qualifies all 2,430 regions. Missing host
semantics fall from the prior 2,408 to zero, an exact net reduction of 2,408.
The difference between 2,430 qualified signatures and the 2,408 net reduction is
the old 22-region scoped baseline; it is not hidden or double-counted.

The schedule covers all 391 structural accelerator partitions, with explicit
host-to-device quantization, device-to-host dequantization, and 88
device-to-device requantization boundaries. Four capture partitions are now
executable, so 387 partitions remain blocked on capture-specific
calibration/numeric qualification. There are 1,250 conversion events in total;
only the 15 around qualified partitions are marked executable.

The new accelerator-contract census separates 31 exact classes by kernel,
source semantic, and input-origin topology. A fail-closed static lane now proves
source operation, ABI, command dependency chain, allocation, and compiled-image
receipt agreement for 391/391 partitions: 303 rank-2 matmuls and all 88 batched
matmuls. The batched split is 24 BF16 and 64 F32 contracts. The corresponding
1,250 of 1,250 conversion boundaries have implemented f32/BF16-to-FP8 input
conversion, BF16 quant-domain bias conversion, device requantization, and scaled
BF16 output publication semantics. This is a static command contract and
executable host conversion implementation; it does not promote physical
partition execution. Physical coverage is 4/391 partitions and 15/1,250
conversion events.

The former 88-command rejection is repaired at its source. Batched command
buffers now declare `RES_PACK W -> W_resident`, execute `BATCHED_MATMUL` from
`W_resident`, and explicitly evict that handle. The contract lane independently
checks the captured batch-matmul maps, loop roles, scalar multiply-accumulate
body, zero initializer, ABI, allocation, image receipt, and complete command
chain. Raw-`W` and missing-evict negative controls fail closed. Five fresh
real-shape software witnesses cover rank-2 and batched BF16/F32 classes. The
assertion-enabled GSIM control for exact shape `15x50x64x113` remains bit-exact
at 4,114,764 cycles with 0/84,750 mismatches. It is shape-level evidence and
does not promote any physical occurrence. The patch-embedding contraction now
has an exact source contract for NCHW-to-im2col materialization, kernel reshape,
matrix ABI, output reshape, and per-channel bias. Affine-stride and ABI-origin
negative controls fail closed. Its exact `768x768x1024` command remained
assertion-clean through a bounded 1,000,000-cycle GSIM run but did not halt by
20,000,000 cycles, so it receives no numeric or physical promotion.

The newly qualified physical batched partition is `atlas_p0102`, the first
text-layer attention QK contraction. Its exact `15x113x64x113` captured
operands complete on assertion-enabled GSIM in 8,700,444 cycles. Its sole
immediate graph frontier, multiplication by 0.125, is included in the source
gate: max absolute error is 0.103215 and cosine similarity is 0.999278. The
independent quantized-domain comparison has 0.006253 max absolute error and
0.999998 cosine similarity. Raw-`W`, missing-evict, and source-perturbation
controls reject. This promotes one partition, not its later mask/softmax chain.

An inclusive-lifetime, 32-byte-aligned first-fit activation allocator assigns
all 391 partition outputs. Its deterministic symbolic arena peaks at 29,884,416
bytes, versus 550,404,032 bytes if every output had dedicated storage, saving
520,519,616 bytes. This is address planning, not proof of a physical Atlas DRAM
capacity or runtime speed.

The builder also executes the real host implementation for the bounded
`atlas_p0243 -> atlas_p0244` chain. It replays all 27 bridge regions and obtains
the exact saved p0244 activation hash
`23457ea06c994031379dcfc4491e866b1f9e5558a2ad4c5ad5a310366145bcce`.
The adjacent device steps are tied to their retained assertion-clean RTL
receipts; they are not rerun. This validates a two-partition hybrid chain, not
SmolVLA end to end.

Separately, the builder discovers the longest bounded consecutive run of
qualified host regions with a true SSA dependency; no region IDs or expected
outputs are encoded in the selection. The resulting 16-region real capture
chain starts with integer `arange`, extends through `pow` and reciprocal, and
has 17 SSA dependency edges plus one fresh 360-element integer input.
Additional signature-selected witnesses cover a cumsum-to-mean normalization
chain, masked softmax, arg-min and reduction-sum successor chains, sigmoid
gating, trigonometric fan-out, and separate real chains in which `arange` and
`fill` outputs feed successor regions. They now also cover static split/slice,
slice-scatter, concat, bitwise, bucketize, and static-select chains. Native
layer norm and GELU are isolated by accelerator partitions in this capture, so the
builder truthfully selects and executes the smallest real GELU standalone
rather than manufacturing a dependency, and does the same for layer norm.
Every witness replays to identical per-region hashes. Eleven additional
real-capture witnesses cover every bridge topology class and compare fresh
execution with an independently indexed NumPy oracle; together their class
counts cover all 358 materializations. This is host-only semantic evidence, not
device or whole-model execution.

Host semantic and layout-bridge coverage are now complete: all 2,033 bridge
candidates are qualified, including 358 real host materializations. The
remaining exact blockers are 387 unqualified accelerator partitions and the
absent physical event/DMA runtime. Full schedule and compact summary are in
`whole_capture_plan/hybrid_schedule.json` and
`whole_capture_plan/hybrid_schedule_summary.json`.

## Prioritized enablement ladder

1. Continue accelerator numeric qualification across the 28 emitted kernel
   variants. Fresh batch evidence now compiles all 28/28 variants and maps them
   onto all 391 occurrences. Direct capture-bound RTL numerics qualify four
   distinct kernel variants: three rank-2 variants and batched
   `15x113x64x113`. The separate
   batched `15x50x64x113` result is a passing shape-only control and is not
   transitive to a capture occurrence. Thus 24 variants lack direct
   capture-bound qualification, none of the four qualified variants is failing,
   and 387 physical partitions remain unqualified. Variant-level RTL testing
   can amortize kernel proof, but all 387
   occurrences still need real weight binding and an activation calibration
   record. The schedule currently has 1,235 unqualified conversions: 686
   host-to-device activations/weights, 74 bias quantizations, 88 device
   requantizations, and 387 device-to-host dequantizations.
2. Turn the symbolic schedule into a runtime: execute all 6,104 ordered events,
   materialize host/device conversions, invoke the qualified host bridge lane,
   launch split command images, propagate failures, and bind the 391 interval
   allocations. The proven allocator reuses 387 allocations and has a
   29,884,416-byte peak, but the current GSIM harness exposes only a 1 MiB
   alias-free DRAM window, so large partitions must remain sliced/staged unless
   that harness is changed and requalified. Zero-stride device descriptors can
   later replace 67 host broadcasts as a performance optimization; they are not
   needed to establish host semantic correctness.

The host-semantics and unrealized-layout gates are now zero. The first honest
E2E becomes possible only when all remaining gates are zero at the same time:
387 unqualified physical partitions (and their 1,235 physically unqualified
conversion events), and the absent physical
event/DMA runtime. At that point one fresh full input must
traverse the entire schedule and be compared with the source-model output.
Kernel-variant coverage alone, structural 391-partition coverage, or replaying
retained intermediates is not sufficient. Performance claims require a
subsequent timed hardware run.
