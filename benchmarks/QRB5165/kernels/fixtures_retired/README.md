# Fixtures slated for retirement

This directory is the **target** for retiring hand-authored
elementwise / activation / conv-relu-smoke fixtures whose function
the v2 emitter (`tools/kernels/qnn_emit_v2.py` +
`tools/kernels/qnn_emit_recognizers/`) now generates from MLIR.

As of Phase 6 (#131), the kernels listed below are *structurally
superseded* — the emitter produces byte-identical `.qnn.cpp` for them
through the parity gate at `tools/kernels/tests/test_qnn_emit_v2_parity.py`.
However, they are **still physically in `../abi/`** because the
existing kernel-embedding flow (`benchmarks/QRB5165/kernels/manifest.json`
→ `tools/kernels/spec_gen.py` → `iree-compile`) consumes them
directly. Until `tools/compile.py` is wired to invoke the v2 emitter
as the manifest-source-of-truth (Phase 6+ deliverable), the manifest
entries remain authoritative and the files stay in place.

## Currently superseded (still in `../abi/` for compatibility)

| Hand-authored | v2 recognizer | Parity test |
|---|---|---|
| `add_f32.qnn.cpp` | `elementwise_binary` | `test_v1_v2_emit_parity[…/add_f32_smoke.mlir-float32]` |
| `sub_f32.qnn.cpp` | `elementwise_binary` | (sub fixture pending) |
| `mul_f32.qnn.cpp` | `elementwise_binary` | `test_v1_v2_emit_parity[…/mul_f32_smoke.mlir-float32]` |
| `div_f32.qnn.cpp` | `elementwise_binary` | (div fixture pending) |
| `sigmoid_f32.qnn.cpp` | `elementwise_unary` | `test_v1_v2_emit_parity[…/sigmoid_f32_smoke.mlir-float32]` |
| `relu_f32.qnn.cpp` | `elementwise_unary` | (relu f32 fixture pending) |
| `relu6_f32.qnn.cpp` | `elementwise_unary` | (relu6 fixture pending) |
| `tanh_f32.qnn.cpp` | `elementwise_unary` | (tanh fixture pending) |
| `hardswish_f32.qnn.cpp` | `elementwise_unary` | (hardswish fixture pending) |
| `conv2d_relu_smoke_f32.qnn.cpp` | `f32_conv2d_relu` | `test_v1_v2_emit_parity[…/conv2d_relu_smoke.mlir-float32]` |

## Kept actively in `../abi/` (no v2 supersession)

| File | Why kept |
|---|---|
| `add_uint8.qnn.cpp`, `relu_int8_smoke.qnn.cpp` | uint8 elementwise — the v2 emitter currently only covers fp32 elementwise; uint8 path is a Phase-2-followup recognizer extension. |
| `bb_mlp.qnn-ctx`, `bb_vit.qnn-ctx`, `bb_mlp_hta.qnn-ctx`, `bb_vit_hta.qnn-ctx` | Pre-built ctxbins from `qairt-converter`; black-box passthrough proofs for the model-level perf comparison vs the v2 emitter's per-island QNN graphs. Not source files; never going to be emitter-generated. |
| `conv2d_int8_smoke.qnn.cpp` | Bare Conv2D (no fused activation) — reference for the recognizer's Conv2d-only path. |
| `conv2d_relu_int8_fused.qnn.cpp` | **Golden Conv+Relu fusion shape** — HTA's `fold_relu_activation_into_conv` reference. Phase 2's structural-match gate compares the emitter's lowering to this fixture. |
| `conv_layer_dronet_stem.qnn.cpp` | dronet stem-conv variant; kept for the dronet-specific Phase 6 regression. |

## Physical retirement procedure

When `tools/compile.py` is wired to invoke the v2 emitter as the
authoritative source for kernel embedding (replacing the manifest
entries that point to `../abi/*.qnn.cpp`):

1. Delete the corresponding manifest entries in
   `benchmarks/QRB5165/kernels/manifest.json`.
2. Move the `.qnn.cpp` files from `../abi/` into this directory.
3. Re-run the parity gate (`pytest
   tools/kernels/tests/test_qnn_emit_v2_parity.py`) — should still pass
   because the emitter generates equivalent output from MLIR fixtures
   independently of the hand-authored sources.
4. Re-run the Phase 2 structural-match gate
   (`pytest tools/kernels/tests/test_qnn_emit_v2_phase2_gate.py`).

Until then, this README is the *contract* — the listed files are
"candidate retirements" pending the wiring.
