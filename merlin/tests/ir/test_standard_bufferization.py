"""Standard materialization keeps type checks without inferring buffer layout."""

import pytest
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text


def program(source="tensor<4xf32>", dest="memref<4xf32, strided<[2], offset: ?>>", *, attributes="writable", result=None):
    prefix = "%r = " if result else ""
    results = result or "()"
    return f'''builtin.module {{
      func.func @copy(%x: {source}, %out: {dest}) {{
        {prefix}"bufferization.materialize_in_destination"(%x, %out) <{{{attributes}}}> : ({source}, {dest}) -> {results}
        return
      }}
    }}'''


@pytest.mark.parametrize("source,dest,attributes,result", [
    ("tensor<4xf32>", "memref<4xf32, strided<[2], offset: ?>>", "writable", None),
    ("tensor<?xf32>", "memref<4xf32, strided<[1], offset: ?>>", "writable, restrict", None),
    ("tensor<*xf32>", "memref<*xf32>", "writable", None),
    ("tensor<4xf32>", "tensor<4xf32>", "", "tensor<4xf32>"),
])
def test_shape_and_element_compatibility_does_not_impose_identity_layout(source, dest, attributes, result):
    module = parse_mlir_text(program(source, dest, attributes=attributes, result=result))
    module.verify()
    parse_mlir_text(text(module, generic=True)).verify()
    assert all(op.name != "builtin.unregistered" for op in module.walk())


@pytest.mark.parametrize("source,dest,attributes,result", [
    ("tensor<4xf32>", "memref<5xf32>", "writable", None),
    ("tensor<4xf32>", "memref<4xf64>", "writable", None),
    ("tensor<4xf32>", "memref<2x2xf32>", "writable", None),
    ("tensor<4xf32>", "memref<4xf32>", "", None),
    ("tensor<4xf32>", "memref<4xf32>", "writable", "tensor<4xf32>"),
    ("tensor<4xf32>", "tensor<4xf32>", "writable", "tensor<4xf32>"),
    ("tensor<4xf32>", "tensor<4xf32>", "restrict", "tensor<4xf32>"),
    ("tensor<4xf32>", "tensor<4xf32>", "", None),
])
def test_malformed_shape_element_result_and_access_forms_still_refuse(source, dest, attributes, result):
    with pytest.raises(VerifyException):
        parse_mlir_text(program(source, dest, attributes=attributes, result=result)).verify()
