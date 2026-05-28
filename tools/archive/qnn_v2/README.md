# `tools/archive/qnn_v2/` — incomplete QNN-emitter v2 attempt

A second attempt at the MLIR → QNN graph emitter that was meant to
replace the regex-based v1 in `kernels/qnn/emit.py` with a
`mlir.ir`-bindings-based dispatcher. **Never reached parity with v1.**

The v1 path remains the production code at `kernels/qnn/emit.py`.

## Files

- `emit_v2.py` (143 LOC) — the v2 dispatcher.
- `tests/test_qnn_emit_v2_parity.py` (123 LOC) — byte-equality gate vs v1.
  Required 11 fixtures to converge; never did.
- `tests/test_qnn_emit_v2_phase2_gate.py` (270 LOC) — phase-2 recognizer
  gate.
- `tests/test_qnn_emit_v2_yolov8_build.py` (203 LOC) — yolov8 build via v2.
- `tests/test_qnn_emit_v2_yolov8_parser.py` (429 LOC) — yolov8 mlir → graph
  traversal under v2.

Total: ~1 168 LOC.

## Why archived

1. v1 (`kernels/qnn/emit.py`) is what the compile pipeline uses today.
2. v2 was a clean-room redesign that didn't reach feature parity.
3. No external code imports v2 — only its own tests do. Archiving is
   self-contained.

If you ever return to a bindings-based emitter, this is the starting
material. Read v1's recognizer set in `kernels/qnn/recognizers/` first —
that's the surface a replacement has to match.
