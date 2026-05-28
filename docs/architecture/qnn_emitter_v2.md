# QNN emitter — v1 / v2 history (retrospective)

> **Status: historical.** This document records why `kernels/qnn/emit.py`
> (v1) remained the production path and the v2 bindings-based emitter
> attempt was archived. New work should target the v1 emitter and its
> recognizer registry at `kernels/qnn/recognizers/`. See
> `tools/archive/qnn_v2/README.md` for the archived v2 source and the
> parity gap that blocked promotion.

## Why a v2 was attempted

`kernels/qnn/emit.py` (v1) parses MLIR with regular expressions. It works
for hand-curated smoke fixtures (Conv+ReLU, elementwise, depthwise,
maxpool, concat, reshape, custom-op uint8 conv) but at the time it did not
scale to real IREE-emitted IR — yolov8 nano alone has 64
`linalg.conv_2d_nchw_fchw_q` ops, 763 `linalg.generic` ops, and many
`tensor.{pad, broadcast, extract_slice}` ops with structural variations
regex can't reliably match.

The v2 plan was to keep the v1 IR and source emitter
(`kernels/qnn/ir.py`) and replace only the parser frontend with the
`iree.compiler.ir` Python bindings (already available in-tree via
`third_party/iree-turbine`). Each pattern recognizer would be a free
function walking the parsed `mlir.ir.Module`.

## Why v2 was archived

The v1 emitter caught up: the recognizer registry under
`kernels/qnn/recognizers/` grew to 14 pattern matchers covering the
yolov8 anchor set. With v1 passing the production gates, v2 never
reached parity with v1 on the same fixture coverage and the maintenance
cost of two emitters wasn't justified. v2 was moved to
`tools/archive/qnn_v2/` during the May 2026 kernels-extraction pass.

## What v1 looks like today

The production path lives at:

- `kernels/qnn/emit.py` — entry point (v1 regex emitter).
- `kernels/qnn/ir.py` — compact intermediate representation, shared.
- `kernels/qnn/recognizers/` — 14 pattern matchers, one file per op
  family; registered in `recognizers/__init__.py:REGISTRY`.
- `kernels/qnn/build.py` — toolchain orchestration into `.qnn-ctx`.

For the per-module map and contributor entry points, see
`kernels/qnn/AGENTS.md` and the
[recognizer how-to](../how_to/add_qnn_recognizer.md).

## When this retrospective is worth re-reading

Before any attempt to restart a v1→v2 migration. The archived v2 code
captures the design but also the parity-gap failure modes that blocked
promotion; skipping that history is how second-system effects compound.
