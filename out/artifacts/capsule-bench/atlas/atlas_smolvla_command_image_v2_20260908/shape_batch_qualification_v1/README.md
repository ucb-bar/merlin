# Atlas SmolVLA shape-batch qualification

This directory is an isolated, fail-closed advance toward SmolVLA E2E. It does not
modify or consume the live hybrid runtime as qualification authority.

`backend_baseline/` and `backend_fixed/` are complete, isolated copies of the exact
`mlir_oot` package (13 Python modules plus `atlas-opt`). Only the fixed copy's
`codegen.py` differs. Both copies carry the command-only batched residency repair,
so every batched command buffer is the three-command
`RES_PACK -> BATCHED_MATMUL(resident) -> EVICT` chain while the paired machine-code
negative control remains isolated to `codegen.py`. `qualify_shapes.py` freshly compiles all 28 distinct
FP8-contraction interfaces in the 391-partition capture plan with that copy. Each
compile receipt checks the source-interface hash, expected assembly-change scope,
32-Kword IMEM bound, compact-loop control targets, pair-bank rules, and
command-buffer shape. The receipts are mapped one-to-one onto all 391 physical
capture partitions.

Four exact-value numerical probes run fresh images on the assertion-enabled,
elaborated-RTL Atlas GSIM. They cover fused bias, rank-2 M/K tails, the real
`50x32x720` projection geometry, and the near-IMEM-limit batched compact loop with
B=15 and N=113. Inputs are sparse only to keep stimulus construction cheap; the
compiled loop nests and full outputs execute and are compared bit-exactly when RTL
accepts the image.

The original batched image issued row stores with byte sizes below one 32-byte DMA
beat and with 226-byte row strides that do not preserve beat alignment. AtlasCore24
derives transfer beats as `size >> 5`, VMEM line address as `local >> 3`, and asserts
that the final line is below `0xc000`. The fixed backend checks those bounds at every
DMA helper and stores each live row segment through a compact scalar loop of aligned
64-byte read/modify/write windows. That keeps the exact B=15, M=50, K=64, N=113
workload and its full 84,750-element output while avoiding both zero-beat launches
and row shear. Its image is 31,130 words, below the 32,768-word IMEM limit.

All four saved assertion-enabled L3 GSIM probes now halt cleanly and compare
bit-exactly. The real batched case executes 4,114,764 cycles and has 0/84,750
mismatches. A paired negative control compiles the exact copied pre-fix image
(matching the planned 32,458-word assembly hash) and runs the identical workload and
stimulus; it still exits `-6` with `DMA VMEM transfer range exceeds VMEM capacity`.
Thus the result demonstrates that the positive is sensitive to the bounded DMA fix,
without disabling assertions or shrinking the case.

The two evidence levels are intentionally non-transitive:

- shape compilation does not imply numeric qualification;
- a synthetic shape-level RTL result does not qualify any physical capture
  occurrence;
- only a direct capture-bound RTL result may qualify a physical partition.

Consequently, the saved run reports 28/28 shapes compile-qualified, 4/4 shapes
tested numerically (covering 11 mapped occurrences), and the negative control 1/1,
while still reporting only 3/391 physical partitions qualified and 388/391
unqualified. Shape compilation and synthetic shape numerics do not constitute a
runnable or numerically qualified whole model.

Run and verify:

```bash
python shape_batch_qualification_v1/qualify_shapes.py
python shape_batch_qualification_v1/verify.py
python shape_batch_qualification_v1/test_qualification.py
```

Saved outputs live under `evidence/`; `qualification.json` carries exact counts and
nanosecond timings. The recorded run took 16,286,901,303 ns overall: 6,989,067,370
ns in compile subprocesses, 7,487,073,562 ns in positive RTL probes, and
1,380,136,075 ns in the negative control. `partition_receipt_map.json` exposes every
physical mapping.
