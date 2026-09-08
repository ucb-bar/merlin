# Atlas SmolVLA shape-batch qualification

This directory is an isolated, fail-closed advance toward SmolVLA E2E. It does not
modify or consume the live hybrid runtime as qualification authority.

`qualify_shapes.py` freshly compiles all 28 distinct FP8-contraction interfaces in
the 391-partition capture plan. Each compile receipt checks the source-interface
hash, fresh assembly hash, instruction count, 32-Kword IMEM bound, compact-loop
control targets, pair-bank rules, and command-buffer shape. The receipts are mapped
one-to-one onto all 391 physical capture partitions.

Four exact-value numerical probes run fresh images on the assertion-enabled,
elaborated-RTL Atlas GSIM. They cover fused bias, rank-2 M/K tails, the real
`50x32x720` projection geometry, and the near-IMEM-limit batched compact loop with
B=15 and N=113. Inputs are sparse only to keep stimulus construction cheap; the
compiled loop nests and full outputs execute and are compared bit-exactly when RTL
accepts the image.

The saved run is intentionally not all-green. Three rank-2 shapes pass bit-exactly.
The batched `15x50x64x113` image aborts in assertion-enabled RTL with `DMA VMEM
transfer range exceeds VMEM capacity`; its one shape and all eight mapped capture
occurrences remain unqualified. The driver's nonzero exit is therefore expected for
this evidence set, while `verify.py` succeeds only if that negative receipt remains
intact and correctly scoped.

The two evidence levels are intentionally non-transitive:

- shape compilation does not imply numeric qualification;
- a synthetic shape-level RTL result does not qualify any physical capture
  occurrence;
- only a direct capture-bound RTL result may qualify a physical partition.

Consequently, the map can report 28/28 shapes compiled while still reporting only
3/391 physical partitions qualified and 388/391 unqualified. It does not claim a
runnable or numerically qualified whole model.

Run and verify:

```bash
python shape_batch_qualification_v1/qualify_shapes.py
python shape_batch_qualification_v1/verify.py
python shape_batch_qualification_v1/test_qualification.py
```

Saved outputs live under `evidence/`; `qualification.json` carries exact counts and
nanosecond timings, and `partition_receipt_map.json` exposes every physical mapping.
