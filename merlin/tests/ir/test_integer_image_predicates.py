"""Uniform predicates use the full rounded integer image, never sampled values."""

import pytest
from mlir_fixtures import assert_unchanged, forward, generic, returned

from merlin.llvmlower.image_folds import fold_integer_image_predicates


def source(integer="i64", floating="f32", threshold="0.5", predicate="oeq", conversion="sitofp"):
    typ, real = f"tensor<4x{integer}>", f"tensor<4x{floating}>"
    convert = generic([("%x", typ)], real, f"%v=arith.{conversion} %a : {integer} to {floating}", name="%f")
    threshold = f"%c=arith.constant {threshold} : {floating}\n%s=tensor.splat %c : {real}"
    cmp = f"%p=arith.cmpf {predicate},%a,%b : {floating}"
    compare = generic([("%f", real), ("%s", real)], "tensor<4xi1>", cmp, yields="%p")
    return forward(f"%x:{typ}", "tensor<4xi1>", f"{convert}\n{threshold}\n{compare}")


@pytest.mark.parametrize("integer", ["i8", "i32", "i64"])
@pytest.mark.parametrize("floating", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("predicate,value", [("oeq", 0), ("une", 1)])
def test_fractional_threshold_is_outside_every_integer_float_image(integer, floating, predicate, value):
    module = source(integer, floating, predicate=predicate)
    assert fold_integer_image_predicates(module) == 1
    module.verify()
    assert returned(module).owner.name == "tensor.splat"
    assert bool(returned(module).owner.operands[0].owner.value.value.data) == bool(value)


@pytest.mark.parametrize(
    "threshold,predicate,value",
    [("300.0", "oeq", False), ("300.0", "olt", True), ("-129.0", "oge", True), ("128.0", "one", True)],
)
def test_typed_endpoint_bound_closes_only_unattainable_integer_values(threshold, predicate, value):
    module = source("i8", threshold=threshold, predicate=predicate)
    assert fold_integer_image_predicates(module) == 1
    module.verify()
    assert bool(returned(module).owner.operands[0].owner.value.value.data) == value


# A real floating input is not assumed integral; INT_MAX rounds upward to 2**31 in binary32.
@pytest.mark.parametrize(
    "integer,threshold,predicate,conversion",
    [("i8", "127.0", "oeq", "sitofp"), ("i8", "127.0", "olt", "sitofp"), ("i8", "0.0", "one", "sitofp")]
    + [("f16", "0.5", "oeq", "extf"), ("i32", "2147483648.0", "oeq", "sitofp")],
)
def test_attainable_threshold_or_real_input_does_not_become_a_fill(integer, threshold, predicate, conversion):
    module = source(integer, threshold=threshold, predicate=predicate, conversion=conversion)
    assert_unchanged(module, fold_integer_image_predicates)
