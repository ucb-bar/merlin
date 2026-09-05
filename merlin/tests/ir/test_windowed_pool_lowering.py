"""A windowed pooling reduction must lower, and an unbound one must SAY WHY.

Max-pool is the op that kept the whole CNN family out of this compiler. A capture that emits it as
a two-operand ``linalg.generic`` over ``(d0, d1, d2 * 2 + d4, d3 * 2 + d5)`` is rejected by MLIR
before any pass runs -- the window dims are bound by no map, so the concatenated indexing map has no
inverse -- and the failure surfaced as an unreadable reader dump hundreds of lines from the op.

Two things are pinned here:

* the FIXED shape lowers end to end (the window operand mapped ``(d4, d5)``, which is what the
  upstream ``linalg.pooling_*`` named ops carry as ``K``), so the CNN path cannot silently close
  again;
* the BROKEN shape is diagnosed structurally by op, dim and captured layer, so the next capture
  that drops the window extent is a one-line read rather than a bisection.
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

# stride 2, 3x3 window, 1 element of -inf padding on each spatial edge: the resnet50 `model.maxpool`
# shape, at a size a unit test can compile.
_POOL_MAPS_FIXED = (
    "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>, "
    "affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, "
    "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>"
)
_POOL_MAPS_UNBOUND = (
    "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>, "
    "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>"
)


def _module(*, window: bool) -> str:
    ins = ("%in, %win : tensor<1x2x10x10xf32>, tensor<3x3xf32>" if window
           else "%in : tensor<1x2x10x10xf32>")
    block = "^bb0(%a: f32, %k: f32, %b: f32):" if window else "^bb0(%a: f32, %b: f32):"
    acc = "%b" if window else "%b"
    return f"""
builtin.module attributes {{prov.level = "linalg-on-tensors"}} {{
  func.func @forward(%arg0: tensor<1x2x8x8xf32>) -> tensor<1x2x4x4xf32> {{
    %ninf = arith.constant 0xff800000 : f32
    %pad = tensor.splat %ninf : tensor<1x2x10x10xf32>
    %in = "tensor.insert_slice"(%arg0, %pad) <{{static_offsets = array<i64: 0, 0, 1, 1>,
        static_sizes = array<i64: 1, 2, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>,
        operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}}>
        : (tensor<1x2x8x8xf32>, tensor<1x2x10x10xf32>) -> tensor<1x2x10x10xf32>
    %win = tensor.empty() : tensor<3x3xf32>
    %init = tensor.splat %ninf : tensor<1x2x4x4xf32>
    %out = linalg.generic {{indexing_maps = [{_POOL_MAPS_FIXED if window else _POOL_MAPS_UNBOUND}],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}}
        ins({ins}) outs(%init : tensor<1x2x4x4xf32>)
        attrs = {{prov.aten = "aten.max_pool2d.default", prov.fqn = "model.maxpool"}} {{
    {block}
      %m = arith.maximumf %a, {acc} : f32
      linalg.yield %m : f32
    }} -> tensor<1x2x4x4xf32>
    func.return %out : tensor<1x2x4x4xf32>
  }}
}}
"""


def test_unbound_window_dims_are_named_with_their_layer():
    """The diagnosis names the op, the unbound dims and the captured layer -- no MLIR line numbers."""
    from merlin.llvmlower.window_maps import explain, unbound_windows

    found = unbound_windows(_module(window=False))
    assert len(found) == 1, found
    assert found[0].dims == (4, 5)
    assert found[0].op == "linalg.generic"
    assert found[0].prov.get("fqn") == "model.maxpool"

    note = explain(_module(window=False))
    assert note is not None
    assert "d4, d5" in note and "model.maxpool" in note


def test_a_window_operand_binds_the_iteration_space():
    """The same reduction with a ``(d4, d5)`` window operand has nothing unbound -- and so verifies."""
    from merlin.llvmlower.window_maps import explain, unbound_windows

    assert unbound_windows(_module(window=True)) == []
    assert explain(_module(window=True)) is None


def test_windowed_pool_lowers_end_to_end(tmp_path):
    """The fixed shape survives the real pipeline: this is the CNN family's gate."""
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model

    if not toolchain.m2m_python().is_file():
        pytest.skip("model2MLIR venv (torch-mlir wheel) unavailable")
    result = lower_model(_module(window=True), tmp_path, targets=("host",), textual=True)
    assert result.ll_path.is_file() and result.ll_path.stat().st_size > 0
    assert result.host_so is not None and result.host_so.is_file()


def test_lowering_an_unbound_window_explains_itself(tmp_path):
    """The pipeline error carries the diagnosis, instead of only the reader's dump."""
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model

    if not toolchain.m2m_python().is_file():
        pytest.skip("model2MLIR venv (torch-mlir wheel) unavailable")
    with pytest.raises(Exception) as excinfo:
        lower_model(_module(window=False), tmp_path, targets=("host",), textual=True)
    message = str(excinfo.value)
    assert "non-invertible" in message
    assert "d4, d5" in message and "model.maxpool" in message
