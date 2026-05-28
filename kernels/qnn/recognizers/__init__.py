"""Bindings-based MLIR→QnnGraphDesc recognizers.

Each recognizer lives in its own module and exposes a single function
following the protocol defined in `base.py`. The v2 dispatcher
(`tools/archive/qnn_v2/emit_v2.py`) walks the registry in priority order and
returns the first non-None result.

Registry order matters: recognizers are tried most-specific first
(custom-op tokens, multi-op patterns, conv-anchored DAGs) and fall
through to the generic single-op recognizers. This mirrors the legacy
regex emitter's dispatcher.

Parity with the legacy `qnn_emit.py` regex emitter is gated by
`tools/archive/qnn_v2/tests/test_qnn_emit_v2_parity.py`.
"""

from __future__ import annotations

from . import (  # noqa: F401
    concat,
    depthwise_conv,
    elementwise_binary,
    elementwise_unary,
    f32_conv2d_relu,
    maxpool,
    nchw_int8_concat,
    nchw_int8_conv,
    nchw_int8_pool,
    nchw_int8_reshape,
    nchw_int8_transpose,
    nhwc_int8_conv,
    reshape,
    uint8_conv,
)

REGISTRY = (
    # NHWC int8 conv first: HTA-compatible (no Transpose adapter,
    # weights already HWIO). Used when upstream IR has been
    # layout-converted to NHWC (re-export, ONNX layout converter, or
    # IREE preprocessing pass).
    nhwc_int8_conv,
    # NCHW int8 conv: lowers with NCHW↔NHWC Transpose adapter, which
    # neither HTA (no Transpose op) nor Adreno GPU (rejects our
    # specific Transpose declaration) accept on QAIRT 2.45. Kept as
    # the structural reference; use the NHWC variant in practice.
    nchw_int8_conv,
    # NCHW int8 standalone-op recognizers. Each anchors on a distinct
    # named op so they don't conflict with each other or with the conv
    # recognizer.
    nchw_int8_pool,
    nchw_int8_concat,
    nchw_int8_reshape,
    nchw_int8_transpose,
    # f32 / uint8-fixture recognizers (mirrored from v1).
    f32_conv2d_relu,
    uint8_conv,
    depthwise_conv,
    maxpool,
    concat,
    reshape,
    elementwise_binary,
    elementwise_unary,
)
