# Atlas functional weight-buffer layout diagnosis

The independent FP8 matmul mismatch is in the functional model's weight-buffer
read view, not in the emitted program. RTL `PushWeight` consumes one MRF row per
cycle and writes it to `weightSlot(laneIdx)`, making the stored tile
`[output lane][reduction row]`. Both RTL compute engines then consume lane `j`
as output column `j`; the systolic wrapper states the corresponding transpose
explicitly. The functional model exposes the same lane-major buffer but passes
it directly to `activation @ weight`, interpreting it as `[K][N]`.

A reviewed process-local overlay now transposes only `read_wb_fp8` while one of
the four FP8 matmul instructions executes. It does not edit the target-owned
model, `VTRPOSE_XLU`, `VMATPUSH_WEIGHT`, emitted words, or output decoding, and
it restores the original method after each instruction.

The deterministic A/B uses the same saved kernels and fixtures:

| case | without layout erratum | with layout erratum | retained RTL |
|---|---:|---:|---:|
| independent Y0 | 25/32 mismatches | 0/32 | 0/32 |
| independent Y1 | 16/21 mismatches | 0/21 | 0/21 |
| chained Y0 | 0/32 | 0/32 | 0/32 |
| chained Y1 | 0/20 | 0/20 | 0/20 |

Both functional arms halt. The correction preserves the already-correct chained
case and repairs all 53 independent outputs. Exact source hashes, cycles, and
per-output comparisons are retained in
`evidence/functional_weight_layout_diagnosis.json`. This qualifies a functional
model erratum; it is not new RTL evidence, whole-model evidence, or performance
evidence.
