"""A legal strided ``tensor.insert_slice`` must survive whole-model preprocessing.

``passes_xdsl._fix_insert_slices`` repairs a legacy model2MLIR artifact (``strides[dim]`` written
with the slice's END index) by resetting a stride that "overruns" the destination back to 1. Its
predicate is ``size * stride > extent``, which counts a step PAST the last written element as a
written element: a ``ConvTranspose2d(stride=2)`` upsample writes ``size=16`` elements at indices
0, 2, ..., 30 of a 31-wide destination -- entirely inside -- yet ``16 * 2 = 32 > 31`` trips it and
the scatter is silently rewritten into a dense copy.

Measured on ``deepjscc`` (two such ops, ``dec.model.0`` and ``dec.model.3``): the whole-model
compiled path scored ``fp32_cos 0.885366`` against the capture's own golden while the per-kernel
interpreter on the same bundle scored ``1.000000``. It is silent -- the module stays verifier-clean
and every downstream pass is happy -- and it only reaches a build whose prepared IR is still in
xDSL GENERIC form, so a build carrying a feature that round-trips through mlir-opt (which prints the
custom form the repair does not match) computes the right answer and an otherwise identical
baseline build does not.

The occupancy test a strided insert has to pass is ``offset + (size - 1) * stride < extent``.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.llvmlower import toolchain
from merlin.llvmlower.passes_xdsl import preprocess_text_textual

_STRIDES = "static_strides = array<i64: "


def _insert_slice_module(c: int, h: int, w: int, s: int, *, pad: int = 0) -> tuple[str, int, int]:
    """A ConvTranspose-shaped upsample: scatter ``1xCxHxW`` into ``1xCxOHxOW`` at stride ``s``.

    Spelled in GENERIC form because that is how xDSL prints ``tensor.insert_slice`` (it has no
    custom format), and the generic spelling is what every whole-model build hands the textual
    preprocessor.
    """
    oh, ow = s * (h - 1) + 1 + 2 * pad, s * (w - 1) + 1 + 2 * pad
    src, dst = f"1x{c}x{h}x{w}xf32", f"1x{c}x{oh}x{ow}xf32"
    return (
        f"builtin.module {{ func.func @forward(%s: tensor<{src}>) -> tensor<{dst}> {{ "
        f"%z = arith.constant 0.0 : f32 "
        f"%d = tensor.splat %z : tensor<{dst}> "
        f'%r = "tensor.insert_slice"(%s, %d) <{{'
        f"static_offsets = array<i64: 0, 0, {pad}, {pad}>, "
        f"static_sizes = array<i64: 1, {c}, {h}, {w}>, "
        f"{_STRIDES}1, 1, {s}, {s}>, "
        f"operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}}> "
        f": (tensor<{src}>, tensor<{dst}>) -> tensor<{dst}> "
        f"func.return %r : tensor<{dst}> }} }}", oh, ow)


@pytest.mark.parametrize("pad", [0, 1])
def test_legal_strided_insert_slice_keeps_its_stride(pad):
    """The upsample fits (``offset + (size-1)*stride < extent``), so nothing may be reset."""
    text, _oh, _ow = _insert_slice_module(2, 16, 16, 2, pad=pad)
    out, _stats = preprocess_text_textual(text)
    assert f"{_STRIDES}1, 1, 2, 2>" in out, (
        "a legal ConvTranspose upsample lost its stride in preprocessing:\n" + out)


def test_overrunning_stride_is_still_repaired():
    """The legacy artifact this repair exists for (stride = the slice's END index) still resets.

    ``size == extent`` with ``stride == extent`` cannot fit: element 1 would land at ``extent``.
    """
    src, dst = "1x2x4x4xf32", "1x2x4x4xf32"
    text = (f"builtin.module {{ func.func @forward(%s: tensor<{src}>) -> tensor<{dst}> {{ "
            f"%z = arith.constant 0.0 : f32 %d = tensor.splat %z : tensor<{dst}> "
            f'%r = "tensor.insert_slice"(%s, %d) <{{'
            f"static_offsets = array<i64: 0, 0, 0, 0>, "
            f"static_sizes = array<i64: 1, 2, 4, 4>, {_STRIDES}1, 1, 4, 4>, "
            f"operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}}> "
            f": (tensor<{src}>, tensor<{dst}>) -> tensor<{dst}> "
            f"func.return %r : tensor<{dst}> }} }}")
    out, _stats = preprocess_text_textual(text)
    assert f"{_STRIDES}1, 1, 1, 1>" in out, "the legacy overrunning-stride repair stopped firing"


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_strided_insert_slice_lowers_to_a_scatter(tmp_path):
    """End to end: the compiled module must scatter at the stride, not pack a dense block.

    The failure mode is not a rounding difference -- the source lands contiguously in the corner
    and every other output element stays zero, which is a different function.
    """
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    c, h, w, s = 2, 4, 4, 2
    text, oh, ow = _insert_slice_module(c, h, w, s)
    res = lower_model(text, tmp_path / "ins", targets=("host",), textual=True)
    a = np.arange(1, 1 + c * h * w, dtype=np.float32).reshape(1, c, h, w)
    y = np.full((1, c, oh, ow), -1.0, np.float32)
    HostModel.load(str(res.host_so))([(a.ctypes.data, a.shape), (y.ctypes.data, y.shape)])
    expected = np.zeros((1, c, oh, ow), np.float32)
    expected[:, :, ::s, ::s] = a
    assert np.array_equal(y, expected), (
        f"strided insert_slice did not scatter:\ngot\n{y[0, 0]}\nwant\n{expected[0, 0]}")
