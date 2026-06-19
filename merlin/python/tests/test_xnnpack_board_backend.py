"""Tests for the BOARD (RVV) XNNPACK kernel backend (default-off, additive).

Mirrors test_xnnpack_host_backend.py's intent on the board side: the routable-matmul classifier
and the MLIR matmul->external-call rewrite are faithful and default-off. The board-build/run leg
is exercised by scripts/k1_e2e_xnnpack.py (needs the K1 + SpacemiT toolchain) — not here.
"""
from __future__ import annotations

from merlin.runtime.backends import xnnpack_board as xb


def test_routable_filter_plain_2d_f32_only():
    assert xb._is_routable("32x256xf32", "256x256xf32", "32x256xf32", "32x256xf32")
    # rank != 2 (batched) -> not routed
    assert not xb._is_routable("2x32x256xf32", "256x256xf32", "2x32x256xf32", "2x32x256xf32")
    # non-f32 -> not routed
    assert not xb._is_routable("32x256xbf16", "256x256xbf16", "32x256xbf16", "32x256xbf16")
    # dynamic dim -> not routed (static only)
    assert not xb._is_routable("?x256xf32", "256x256xf32", "?x256xf32", "?x256xf32")


def test_rewrite_is_identity_without_matmul():
    t = ("module {\n  func.func @f(%a: tensor<2xf32>) -> tensor<2xf32> "
         "{ return %a : tensor<2xf32> }\n}\n")
    out, n = xb.rewrite_matmuls_to_xnn(t)
    assert n == 0
    assert out == t  # default path byte-unchanged


def test_rewrite_routes_plain_matmul_to_external_call():
    t = (
        'builtin.module attributes {x = 1 : i64} {\n'
        '  func.func @forward(%a: tensor<32x256xf32>, %b: tensor<256x128xf32>) '
        '-> tensor<32x128xf32> {\n'
        '    %0 = tensor.empty() : tensor<32x128xf32>\n'
        '    %1 = linalg.matmul {prov.op = "matmul"} ins(%a, %b : tensor<32x256xf32>, '
        'tensor<256x128xf32>) outs(%0 : tensor<32x128xf32>) -> tensor<32x128xf32>\n'
        '    return %1 : tensor<32x128xf32>\n'
        '  }\n}\n')
    out, n = xb.rewrite_matmuls_to_xnn(t)
    assert n == 1
    assert "linalg.matmul" not in out
    assert "call @merlin_xnn_gemm_f32_0(%a, %b, %0)" in out
    # one private decl per signature, with read/read/write access annotations + inside the body
    assert 'func.func private @merlin_xnn_gemm_f32_0(' in out
    assert out.count('bufferization.access = "read"') == 2
    assert out.count('bufferization.access = "write"') == 1
    # decl is placed before the forward func (inside the module body, not in the attr dict)
    assert out.index("private @merlin_xnn_gemm_f32_0") < out.index("func.func @forward")


def test_rewrite_skips_nonroutable_matmul():
    # bf16 matmul must fall through (NOT routed) — stays on the compiled runtime.
    t = (
        'module {\n  func.func @forward(%a: tensor<4x4xbf16>, %b: tensor<4x4xbf16>) '
        '-> tensor<4x4xbf16> {\n'
        '    %0 = tensor.empty() : tensor<4x4xbf16>\n'
        '    %1 = linalg.matmul ins(%a, %b : tensor<4x4xbf16>, tensor<4x4xbf16>) '
        'outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>\n'
        '    return %1 : tensor<4x4xbf16>\n  }\n}\n')
    out, n = xb.rewrite_matmuls_to_xnn(t)
    assert n == 0
    assert "linalg.matmul" in out  # untouched
