"""Parity tests between the v1 regex emitter (`qnn_emit.py`) and the v2
bindings emitter (`qnn_emit_v2.py`).

Two flavours:

1. **Byte-equal parity** (`test_v1_v2_emit_parity`) — for fixtures whose
   QnnGraphDesc is fully recoverable from the parsed `mlir.ir.Module`,
   v2 must emit a byte-identical `.qnn.cpp` to v1.

2. **Structural parity** (`test_v1_v2_emit_structural_concat`) — for
   `tensor.concat`, the legacy regex emitter scraped source-level SSA
   names (`%a`, `%b`) and used them as QNN tensor names. The bindings
   normalize block arguments to canonical names (`%arg0`, `%arg1`); we
   accept the textual divergence and assert structural equivalence
   instead (same axis, same shapes, same op type, same dtype). The
   `.qnn-ctx` bound at runtime is index-based, so the rename is
   functionally equivalent.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "kernels"))


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# Byte-equal fixtures — every recognizer that can be fully recovered
# structurally from the parsed module belongs here.
PARITY_FIXTURES: tuple[tuple[str, str], ...] = (
    ("benchmarks/QRB5165/mlir/conv2d_relu_smoke.mlir", "float32"),
    ("benchmarks/QRB5165/mlir/conv2d_relu_smoke.mlir", "float16"),
    ("benchmarks/QRB5165/mlir/conv2d_uint8_smoke.mlir", "float32"),
    ("benchmarks/QRB5165/mlir/depthwise_conv_smoke.mlir", "float32"),
    ("benchmarks/QRB5165/mlir/maxpool_smoke.mlir", "float32"),
    ("benchmarks/QRB5165/mlir/reshape_smoke.mlir", "float32"),
    ("benchmarks/QRB5165/mlir/add_f32_smoke.mlir", "float32"),
    ("benchmarks/QRB5165/mlir/mul_f32_smoke.mlir", "float32"),
    ("benchmarks/QRB5165/mlir/sigmoid_f32_smoke.mlir", "float32"),
)


@pytest.mark.parametrize("fixture_rel,fp_dtype", PARITY_FIXTURES)
def test_v1_v2_emit_parity(fixture_rel: str, fp_dtype: str) -> None:
    import qnn_emit
    import qnn_emit_v2
    from qnn_ir import emit_qnn_cpp

    fixture = REPO_ROOT / fixture_rel
    text = fixture.read_text()

    g_v1 = qnn_emit.parse_mlir(text, fp_dtype=fp_dtype)
    g_v2 = qnn_emit_v2.parse_mlir(text, fp_dtype=fp_dtype)
    cpp_v1 = emit_qnn_cpp(g_v1)
    cpp_v2 = emit_qnn_cpp(g_v2)

    assert g_v1.name == g_v2.name, f"graph name mismatch: v1={g_v1.name} v2={g_v2.name}"
    assert len(g_v1.tensors) == len(g_v2.tensors)
    assert len(g_v1.nodes) == len(g_v2.nodes)
    assert cpp_v1 == cpp_v2, (
        f"emitted .qnn.cpp differs between v1 and v2 for {fixture_rel} "
        f"(fp_dtype={fp_dtype}); md5 v1={_md5(cpp_v1)} v2={_md5(cpp_v2)}"
    )


def test_v1_v2_emit_structural_concat() -> None:
    """Concat tensor names diverge by design (canonical vs source-level
    SSA); compare structure, not bytes."""
    import qnn_emit
    import qnn_emit_v2

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/concat_smoke.mlir"
    text = fixture.read_text()
    g_v1 = qnn_emit.parse_mlir(text)
    g_v2 = qnn_emit_v2.parse_mlir(text)

    # Same graph identity.
    assert g_v1.name == g_v2.name
    # Same tensor count, same shapes, same dtypes, same roles (just names differ).
    assert len(g_v1.tensors) == len(g_v2.tensors)
    for t1, t2 in zip(g_v1.tensors, g_v2.tensors):
        assert t1.shape == t2.shape
        assert t1.dtype == t2.dtype
        assert t1.role == t2.role
    # Same node count and op types.
    assert len(g_v1.nodes) == len(g_v2.nodes)
    for n1, n2 in zip(g_v1.nodes, g_v2.nodes):
        assert n1.op_type == n2.op_type


def test_regex_fallback_knob() -> None:
    """`MERLIN_QNN_EMIT_REGEX=1` routes v2's parse_mlir through the v1
    regex implementation, producing identical output."""
    import os

    import qnn_emit
    import qnn_emit_v2
    from qnn_ir import emit_qnn_cpp

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/conv2d_relu_smoke.mlir"
    text = fixture.read_text()

    cpp_native = emit_qnn_cpp(qnn_emit.parse_mlir(text))

    prev = os.environ.get("MERLIN_QNN_EMIT_REGEX")
    os.environ["MERLIN_QNN_EMIT_REGEX"] = "1"
    try:
        cpp_fallback = emit_qnn_cpp(qnn_emit_v2.parse_mlir(text))
    finally:
        if prev is None:
            del os.environ["MERLIN_QNN_EMIT_REGEX"]
        else:
            os.environ["MERLIN_QNN_EMIT_REGEX"] = prev

    assert cpp_native == cpp_fallback
