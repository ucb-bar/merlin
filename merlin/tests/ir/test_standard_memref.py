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


@pytest.mark.parametrize("element", ["i32", "i64", "f32"])
def test_standard_generic_atomic_region_round_trips(element):
    source = f'''builtin.module {{func.func @atomic(%p:memref<?x{element}>,%i:index)->{element} {{
      %v="memref.generic_atomic_rmw"(%p,%i) ({{
      ^bb0(%old:{element}):
        "memref.atomic_yield"(%old):({element})->()
      }}):(memref<?x{element}>,index)->{element}
      return %v:{element}
    }}}}'''
    module = parse_mlir_text(source)
    module.verify()
    parse_mlir_text(text(module, generic=True)).verify()


@pytest.mark.parametrize("mutation", ["body-type", "result-type", "missing-index", "effect"])
def test_standard_atomic_region_rejects_mismatched_types_or_hidden_effects(mutation):
    source = '''builtin.module {func.func @atomic(%p:memref<?xi32>,%i:index)->i32 {
      %v="memref.generic_atomic_rmw"(%p,%i) ({
      ^bb0(%old:i32):
        "memref.atomic_yield"(%old):(i32)->()
      }):(memref<?xi32>,index)->i32
      return %v:i32
    }}'''
    if mutation == "body-type": source = source.replace("%old:i32", "%old:i64").replace("(%old):(i32)", "(%old):(i64)")
    elif mutation == "result-type": source = source.replace("->i32", "->i64").replace("%v:i32", "%v:i64")
    elif mutation == "missing-index": source = source.replace("(%p,%i)", "(%p)").replace("(memref<?xi32>,index)", "(memref<?xi32>)")
    else: source = source.replace('"memref.atomic_yield"', '"memref.store"(%old,%p,%i):(i32,memref<?xi32>,index)->()\n"memref.atomic_yield"')
    with pytest.raises(VerifyException):
        parse_mlir_text(source).verify()
