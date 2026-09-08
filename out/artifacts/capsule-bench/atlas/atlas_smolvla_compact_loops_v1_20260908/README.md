# Atlas SmolVLA compact-loop recovery

This artifact removes the remaining instruction-memory blocker from the native-FP8 SmolVLA
contraction inventory. It is a **compile/code-size recovery candidate**, not a numerically certified
backend or a whole-model Atlas executable.

## Result

- Before: 13/28 unique contraction shapes fit Atlas' 32,768-word IMEM, covering 205/391 physical
  contraction layers. The largest program contained 2,774,543 words.
- After: 28/28 shapes fit, covering 391/391 physical contraction layers. The largest program contains
  32,458 words, leaving a 310-word minimum margin.
- All 15 previously overflowing shapes were recovered. Total instructions across the 28 unique
  programs fell from 11,384,956 to 379,142 (30.0x smaller).
- The bounded after-run accepted all shapes in 13.7 seconds wall time; summed backend subprocess time
  was 6.25 seconds.

The result is based on the same 28 interfaces and 391-layer census saved by
`smolvla_fp8_compile_inventory_20260908`. The datatype is genuinely `fp8_e4m3` with BF16 output; this
artifact does not relabel INT8 data as FP8.

## Changes

`submission/mlir_oot/codegen.py` makes two code-size changes:

1. `_emit_fp8_matmul` and `_emit_fp8_matmul_loop_region` now keep M/N/K traversal in runtime loops
   when K or N has a partial 32-wide tile. A shape-specialized K-tail body is emitted once, inactive
   VMEM lanes are cleared before the tail MAC, and partial N stores issue only the live byte count.
2. `_emit_batched_matmul` emits one FP8 matmul body. Scalar registers x29/x30/x31 carry the current
   LHS/RHS/output bases; a runtime batch loop advances them by the static batch strides. A focused
   regression proves B=1 and B=15 produce equal instruction counts.

The inherited two-unit Atlas branch-displacement recovery remains present. Every decoded relative
control-flow target in all 28 saved after-programs is checked to remain within that program.

## Evidence and provenance

- Capture: `out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir`, SHA-256
  `27256e2414be16eb0f783a547e79102c9b31afad76cbfd13c2f4b421ec7d8e4c`.
- Parent package: the repaired 49/61 fresh Atlas package recorded in
  `atlas_control_flow_recovery_v1_20260908T102627Z_ca7705d`; original source-only tree SHA-256
  `ce5f4e543988e294360ffffbdf723e977c1b73a635c42926047ecb6719695df0`.
- Recovery package source-only tree SHA-256:
  `5b038a25b4f9d24349c8b28714d9650e421499764bdf3137266066b87fab9c41`.
- Patched `codegen.py` SHA-256:
  `94c5c429201b298d696e08258af8eabb3c5589f160496f632353446982d248b6`.
- IMEM capacity: generated GSIM RTL `AtlasCore.h:18413`, SHA-256
  `902abb6b294991c7f25e6178a29292e419944cd828c4fb7d883305bb8d328d7d`.

The 49/61 score describes the parent package, not this recovery: this changed package has not been
regraded. No simulator was launched, and no halt, numeric-correctness, performance, capsule-score, or
single-whole-image claim is made. The 310-word worst-case margin is also narrow enough that future
prologue or instrumentation growth needs an explicit IMEM regression.

## Reproduce

From the repository root:

```bash
PYTHONPATH=merlin/python timeout --signal=TERM --kill-after=10s 600s \
  python out/artifacts/capsule-bench/atlas/smolvla_fp8_compile_inventory_20260908/run_inventory.py \
  --package out/artifacts/capsule-bench/atlas/atlas_smolvla_compact_loops_v1_20260908/submission \
  --output out/artifacts/capsule-bench/atlas/atlas_smolvla_compact_loops_v1_20260908/after_inventory

python out/artifacts/capsule-bench/atlas/atlas_smolvla_compact_loops_v1_20260908/validate_recovery.py

.venv/bin/python -m pytest -q \
  out/artifacts/capsule-bench/atlas/atlas_smolvla_compact_loops_v1_20260908/test_compact_loops.py
```

## Files

- `submission/`: isolated patched backend package.
- `before_summary.json`, `before_unique_shapes.json`: immutable baseline receipts.
- `after_inventory/`: complete rerun, including all emitted programs as deterministic gzip files.
- `validation.json`: independently recomputed fit and recovery counts.
- `test_compact_loops.py`: focused K/N-tail, runtime-batch, IMEM, and decoded-edge regressions.
- `interfaces_before_reference/`: exact interfaces used by the focused tests.

