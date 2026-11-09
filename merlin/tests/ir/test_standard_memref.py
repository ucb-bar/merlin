"""Ranked/unranked standard views keep native element/layout constraints."""

import pytest
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text


def program(source="memref<*xf32>", result="memref<4xf32, strided<[1], offset: ?>>", *, size=4, stride=1):
    return f'''builtin.module {{func.func @view(%p: {source}, %off: index) {{
      %v = "memref.reinterpret_cast"(%p, %off) <{{operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: {size}>, static_strides = array<i64: {stride}>}}>
        : ({source}, index) -> {result}
      return
    }}}}'''


@pytest.mark.parametrize("source,result", [("memref<*xf32>", "memref<4xf32, strided<[1], offset: ?>>"),
    ("memref<8xf32>", "memref<4xf32, strided<[1], offset: ?>>"),
    ("memref<*xf32>", "memref<?xf32, strided<[1], offset: ?>>")])
def test_native_view_can_reinterpret_raw_or_ranked_buffer(source, result):
    module = parse_mlir_text(program(source, result))
    module.verify()
    parse_mlir_text(text(module, generic=True)).verify()


@pytest.mark.parametrize("source,result,size,stride", [
    ("memref<*xi32>", "memref<4xf32, strided<[1], offset: ?>>", 4, 1),
    ("memref<*xf32, 3>", "memref<4xf32, strided<[1], offset: ?>>", 4, 1),
    ("memref<*xf32>", "memref<5xf32, strided<[1], offset: ?>>", 4, 1),
    ("memref<*xf32>", "memref<4xf32, strided<[2], offset: ?>>", 4, 1),
    ("memref<*xf32>", "memref<4xf32>", 4, 1),
])
def test_view_rejects_element_space_size_stride_or_offset_mismatch(source, result, size, stride):
    with pytest.raises(VerifyException):
        parse_mlir_text(program(source, result, size=size, stride=stride)).verify()
