"""Bufferization's layout relation remains typed and matches selected MLIR."""

import subprocess

import pytest
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text


def source(tensor="tensor<2x3xi8>", buffer="memref<2x3xi8, strided<[3, 1], offset: ?>>"):
    return f"""module {{func.func @read(%value:{tensor}) {{
      %buffer = bufferization.to_buffer %value read_only : {tensor} to {buffer}
      return
    }}}}"""


@pytest.mark.parametrize(
    "buffer",
    ["memref<2x3xi8>", "memref<2x3xi8, strided<[3, 1], offset: ?>>", "memref<2x3xi8, strided<[3, 1], offset: 3>>"],
)
def test_layout_roundtrip_preserves_shape_element_and_readonly(buffer):
    module = parse_mlir_text(source(buffer=buffer))
    module.verify()
    reopened = parse_mlir_text(text(module, generic=True))
    reopened.verify()
    assert text(reopened, generic=True) == text(module, generic=True)
    operation = next(op for op in reopened.walk() if op.name == "bufferization.to_buffer")
    assert "read_only" in operation.properties


@pytest.mark.parametrize(
    "tensor,buffer",
    [
        ("tensor<2x3xi8>", "memref<2x4xi8>"),
        ("tensor<2x3xi8>", "memref<2x3xi32>"),
        ("tensor<2x3xi8>", "memref<6xi8>"),
        ("tensor<2x3xi8>", "memref<*xi8>"),
        ("tensor<*xi8>", "memref<2x3xi8>"),
        ("tensor<2x3xi8>", "memref<2x3xi8, strided<[1], offset: ?>>"),
    ],
)
def test_incompatible_buffer_types_refuse(tensor, buffer):
    with pytest.raises(VerifyException):
        parse_mlir_text(source(tensor, buffer)).verify()


def test_unranked_pair_and_dynamic_ranked_pair_remain_supported():
    for tensor, buffer in (
        ("tensor<*xi8>", "memref<*xi8>"),
        ("tensor<?x3xi8>", "memref<?x3xi8, strided<[3, 1], offset: ?>>"),
    ):
        parse_mlir_text(source(tensor, buffer)).verify()


def test_selected_mlir_verifies_exact_readonly_layout(tmp_path):
    from merlin.llvmlower import toolchain

    module = parse_mlir_text(source())
    path = tmp_path / "layout.mlir"
    path.write_text(text(module, generic=True))
    result = subprocess.run(
        [str(toolchain.mlir_translate().with_name("mlir-opt")), str(path), "--verify-each"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
