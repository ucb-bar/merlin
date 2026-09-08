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
signatures instead of trusting provenance names. It admits 1,396 real-capture
regions: 472 `dtype_cast`, 536 `mul`, 215 `add`, 68 `sub`, 56 `div`, four
`compare`, and 45 conditional `select` regions. Admission requires one terminal
`linalg.generic`, parallel iterators, an identity output map, input maps composed
only of distinct loop dimensions and constant-zero broadcast axes, supported
tensor dtypes, and a scalar body whose root matches the declared semantic.
Two other `select` regions are actually `aten.select.int` slice/reshape chains;
they correctly remain rejected.

The scheduler now marks a host region executable only when that extracted
signature is accepted. Missing host semantics fall from the prior 2,408 to
1,034, an exact net reduction of 1,374. The difference between 1,396 qualified
signatures and the 1,374 net reduction is the old 22-region scoped baseline; it
is not hidden or double-counted.

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
outputs are encoded in the selection. The resulting nine-region real capture
chain is `compare -> dtype_cast -> mul -> add -> sub -> dtype_cast -> mul ->
sub -> select`, with eight SSA dependency edges and two fresh 360-element
integer inputs. It executes numerically and replays to identical per-region
hashes. This is host-only pointwise evidence, not device or whole-model
execution.

The exact blockers are therefore measurable: 1,034 missing host semantic
implementations, 388 unqualified accelerator partitions, and 358 unrealized
layout bridges. Full schedule and compact summary are in
`whole_capture_plan/hybrid_schedule.json` and
`whole_capture_plan/hybrid_schedule_summary.json`.

## Prioritized enablement ladder

1. Extend the same signature-driven scalar lane to the 353 remaining unary and
   scalar-expression regions: 123 `pow`, 66 `rsqrt`, 57 `sin`, 57 `cos`, 33
   `sigmoid`, 12 `gelu`, three miscellaneous `elementwise`, and two `minmax`.
   These have the largest coverage-to-runtime-complexity ratio, but each still
   needs its scalar body and legal numeric domain checked rather than admission
   by name.
2. Add constructors for 63 `arange` and 50 `fill` regions (113 total), deriving
   start/step/value and output dtype from the actual region body.
3. Add dependence-aware reduction/composite execution for 150 regions: 66
   `reduce_mean`, 44 `softmax`, 25 `layer_norm`, eight `aten_min_dim`, four
   `cumsum`, and three `reduce_sum`. Exact accumulation order and dtype rules
   are part of the signature and cannot be delegated to an arbitrary NumPy
   default.
4. Implement the remaining 418 movement/indexing regions: 129 `slice`, 112
   `slice_scatter`, 95 `cat`, 56 `split`, 16 `bitwise`, two `select` slices, two
   `bucketize`, two `embedding`, and one each of `mask_gather`, `index_put`,
   `index_gather`, and the host-refused im2col convolution. This reaches all
   1,034 currently missing host regions if the earlier groups qualify fully.
5. Realize all 358 blocking layout bridges. The 246 `expand` regions need
   zero-stride descriptors or exact materialization; the 112 `copy` regions
   require real storage and copy events. Host materialization is sufficient for
   a first correctness run; descriptor propagation is the later performance
   path. The 1,675 already proven metadata aliases require no data movement.
6. Batch accelerator qualification by the 28 emitted kernel variants: three
   variants/physical occurrences have retained numeric evidence, leaving 25
   variants and 388 of 391 physical partitions without capture-specific
   qualification. Variant-level RTL testing can amortize kernel proof, but all
   388 occurrences still need real weight binding and an activation calibration
   record. The schedule currently has 1,238 unqualified conversions: 688
   host-to-device activations/weights, 74 bias quantizations, 88 device
   requantizations, and 388 device-to-host dequantizations.
7. Turn the symbolic schedule into a runtime: execute all 6,104 ordered events,
   materialize host/device conversions, launch split command images, propagate
   failures, and bind the 391 interval allocations. The proven allocator reuses
   387 allocations and has a 29,884,416-byte peak, but the current GSIM harness
   exposes only a 1 MiB alias-free DRAM window, so large partitions must remain
   sliced/staged unless that harness is changed and requalified.

The first honest E2E becomes possible only when all four gates are zero at the
same time: 1,034 missing host semantics, 358 unrealized bridges, 388
unqualified partitions (and their 1,238 conversion events), and the absent
physical event/DMA runtime. At that point one fresh full input must traverse the
entire schedule and be compared with the source-model output. Kernel-variant
coverage alone, structural 391-partition coverage, or replaying retained
intermediates is not sufficient. Performance claims require a subsequent timed
hardware run.
