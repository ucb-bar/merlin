"""Admit canonical integer contraction semantics, not metadata or op counts."""
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.parser import Parser

from merlin.xdsl_dialects.lowering.canonical_matmul import is_integer_matmul

SOURCE = """
builtin.module {
func.func @compute(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>, %c: tensor<2x4xi32>) -> tensor<2x4xi32> {
%r = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%c : tensor<2x4xi32>) {
^bb0(%x: i8, %y: i8, %acc: i32):
%xx = arith.extsi %x : i8 to i32
%yy = arith.extsi %y : i8 to i32
%prod = arith.muli %xx, %yy : i32
%sum = arith.addi %acc, %prod : i32
linalg.yield %sum : i32
} -> tensor<2x4xi32>
func.return %r : tensor<2x4xi32>
}
}
"""


def generic(text):
    ctx = Context()
    for dialect in (builtin.Builtin, func.Func, arith.Arith, linalg.Linalg, tensor.Tensor):
        ctx.load_dialect(dialect)
    module = Parser(ctx, text).parse_module()
    return next(op for op in module.walk() if op.name == "linalg.generic")


def test_canonical_signed_matmul_and_commuted_add():
    assert is_integer_matmul(generic(SOURCE))
    assert is_integer_matmul(generic(SOURCE.replace("%acc, %prod", "%prod, %acc")))


def test_similar_bodies_with_different_semantics_decline():
    assert not is_integer_matmul(generic(SOURCE.replace("arith.extsi", "arith.extui")))
    assert not is_integer_matmul(generic(SOURCE.replace("%acc, %prod", "%prod, %prod")))
    assert not is_integer_matmul(generic(SOURCE.replace("%xx, %yy", "%xx, %xx")))
    assert not is_integer_matmul(generic(SOURCE.replace('"reduction"', '"parallel"')))
    assert not is_integer_matmul(generic(SOURCE.replace("->(d2,d1)", "->(d1,d2)")))


def test_narrow_intermediate_and_changed_yield_are_not_matmul():
    narrowed = SOURCE.replace(
        "%sum = arith.addi %acc, %prod : i32",
        "%narrow = arith.trunci %prod : i32 to i8\n"
        "%wide = arith.extsi %narrow : i8 to i32\n"
        "%sum = arith.addi %acc, %wide : i32")
    assert not is_integer_matmul(generic(narrowed))
    assert not is_integer_matmul(generic(SOURCE.replace("linalg.yield %sum", "linalg.yield %prod")))
    assert not is_integer_matmul(generic(SOURCE.replace("linalg.yield %sum", "linalg.yield %acc")))
