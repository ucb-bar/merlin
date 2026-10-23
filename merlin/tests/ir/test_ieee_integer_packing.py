"""Execute every packing shift, including zeros unconstrained by the grid."""

import struct
import sys
from fractions import Fraction

import pytest
from mlir_fixtures import run_ops
from xdsl.dialects.builtin import IntegerType

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.floating_component_ir import pack_ieee_integer


@pytest.mark.parametrize(
    "dtype,width,exponent,values",
    [
        ("f64", 512, 0, [0.0, -0.0, 1.0, -1.0, 2.0**63, -(2.0**400)]),
        ("f64", 512, 400, [0.0, -0.0, 2.0**400, -(2.0**800)]),
        ("f64", 512, 971, [0.0, -0.0, sys.float_info.max, -sys.float_info.max]),
        ("f64", 1024, 0, [0.0, -0.0]),
        ("f64", 2048, 0, [0.0, -0.0, sys.float_info.max]),
        ("f32", 512, -149, [0.0, -0.0, 2.0**-149, (2.0 - 2.0**-23) * 2.0**127]),
    ],
)
def test_exact_packing_has_no_out_of_range_shift(dtype, width, exponent, values):
    module = parse_mlir_text(f"module {{ func.func @f(%x: {dtype}) {{ return }} }}")
    argument = next(op for op in module.walk() if op.name == "func.func").body.block.args[0]
    operations = []

    def emit(op):
        operations.append(op)
        return op.results[0]

    result = pack_ieee_integer(argument, exponent, IntegerType(width), emit)
    assert {op.predicate.value.data for op in operations if op.name == "arith.cmpi"} <= {0, 1, 5}
    for value in values:
        raw = int.from_bytes(struct.pack("<d" if dtype == "f64" else "<f", value), "little")
        data = {argument: raw}
        run_ops(operations, data)
        actual = data[result]
        if actual >> (width - 1):
            actual -= 1 << width
        assert actual == Fraction.from_float(value) / Fraction(2) ** exponent
