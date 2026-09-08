# Atlas SmolVLA hybrid schedule status

The artifact now has an executable schedule constructor, but it intentionally
fails closed for whole-model execution. `build_hybrid_schedule.py` reconstructs
the live capture inventory, emits 6,104 deterministic capture-order events, and
records every accelerator, host, layout, and quantization boundary. It does not
promote structural partitions to executable partitions.

The earlier 2,033 “layout bridge” count needed refinement. These are candidates,
not aliases: 1,675 `view`/`unsqueeze` regions are proven metadata-only from
equal element counts and reshape/cast operations; 246 `expand` regions require a
strided/zero-stride descriptor that the current contiguous ABI lacks; and 112
`copy` regions require materialization. Independently, 2,430 regions require
host semantics. The prior schedule treated 22 regions covered by one scoped
bridge as implemented, leaving 2,408 unresolved.

The generic host semantic lane now extracts and validates complete operation
signatures instead of trusting provenance names. Its first tranche admitted
1,396 real-capture regions: 472 `dtype_cast`, 536 `mul`, 215 `add`, 68 `sub`,
56 `div`, four `compare`, and 45 conditional `select` regions. Admission
requires one terminal
`linalg.generic`, parallel iterators, an identity output map, input maps composed
only of distinct loop dimensions and constant-zero broadcast axes, supported
tensor dtypes, and a scalar body whose root matches the declared semantic.
Two other `select` regions are actually `aten.select.int` slice/reshape chains;
they correctly remain rejected.

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

The scheduler marks a host region executable only when that extracted
signature is accepted. It now qualifies 1,862 regions. Missing host semantics
fall from the prior 2,408 to 568, an exact net reduction of 1,840. The
difference between 1,862 qualified signatures and the 1,840 net reduction is
the old 22-region scoped baseline; it is not hidden or double-counted.

The schedule covers all 391 structural accelerator partitions, with explicit
host-to-device quantization, device-to-host dequantization, and 88
device-to-device requantization boundaries. Only the three previously qualified
partitions are executable, so 388 partitions remain blocked on capture-specific
calibration/numeric qualification. There are 1,250 conversion events in total;
only the 12 around qualified partitions are marked executable.

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
Additional signature-selected witnesses cover an `rsqrt` normalization chain,
a sigmoid gating chain, a trigonometric fan-out, and separate real chains in
which `arange` and `fill` outputs feed successor regions. The fill-to-cast chain
needs no seeded input; its value is entirely determined by the captured
constructor. GELU is isolated by accelerator partitions in this capture, so the
builder truthfully selects and executes the smallest real GELU standalone
rather than manufacturing a dependency. Every witness replays to identical
per-region hashes. This is host-only pointwise evidence, not device or
whole-model execution.

The exact blockers are therefore measurable: 568 missing host semantic
implementations, 388 unqualified accelerator partitions, and 358 unrealized
layout bridges. Full schedule and compact summary are in
`whole_capture_plan/hybrid_schedule.json` and
`whole_capture_plan/hybrid_schedule_summary.json`.

## Prioritized enablement ladder

1. Add dependence-aware reduction/composite execution for 150 regions: 66
   `reduce_mean`, 44 `softmax`, 25 `layer_norm`, eight `aten_min_dim`, four
   `cumsum`, and three `reduce_sum`. Exact accumulation order and dtype rules
   are part of the signature and cannot be delegated to an arbitrary NumPy
   default.
2. Implement the remaining 418 movement/indexing regions: 129 `slice`, 112
   `slice_scatter`, 95 `cat`, 56 `split`, 16 `bitwise`, two `select` slices, two
   `bucketize`, two `embedding`, and one each of `mask_gather`, `index_put`,
   `index_gather`, and the host-refused im2col convolution. This reaches all
   568 currently missing host regions if the earlier groups qualify fully.
3. Realize all 358 blocking layout bridges. The 246 `expand` regions need
   zero-stride descriptors or exact materialization; the 112 `copy` regions
   require real storage and copy events. Host materialization is sufficient for
   a first correctness run; descriptor propagation is the later performance
   path. The 1,675 already proven metadata aliases require no data movement.
4. Continue accelerator numeric qualification across the 28 emitted kernel
   variants. Fresh batch evidence now compiles all 28/28 variants and maps them
   onto all 391 occurrences. RTL numerics tested four variants: three rank-2
   shapes pass, while batched `15x50x64x113` fails closed on VMEM capacity.
   Thus 24 variants are numerically untested, one is tested-failing, only three
   physical capture partitions have direct qualification, and 388 remain
   unqualified. Variant-level RTL testing can amortize kernel proof, but all 388
   occurrences still need real weight binding and an activation calibration
   record. The schedule currently has 1,238 unqualified conversions: 688
   host-to-device activations/weights, 74 bias quantizations, 88 device
   requantizations, and 388 device-to-host dequantizations.
5. Turn the symbolic schedule into a runtime: execute all 6,104 ordered events,
   materialize host/device conversions, launch split command images, propagate
   failures, and bind the 391 interval allocations. The proven allocator reuses
   387 allocations and has a 29,884,416-byte peak, but the current GSIM harness
   exposes only a 1 MiB alias-free DRAM window, so large partitions must remain
   sliced/staged unless that harness is changed and requalified.

The first honest E2E becomes possible only when all four gates are zero at the
same time: 568 missing host semantics, 358 unrealized bridges, 388
unqualified partitions (and their 1,238 conversion events), and the absent
physical event/DMA runtime. At that point one fresh full input must traverse the
entire schedule and be compared with the source-model output. Kernel-variant
coverage alone, structural 391-partition coverage, or replaying retained
intermediates is not sufficient. Performance claims require a subsequent timed
hardware run.
