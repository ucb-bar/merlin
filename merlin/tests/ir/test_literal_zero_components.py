# ruff: noqa: E501 -- static MLIR fixtures retain one operation per line
"""Integer zero components are source algebra; floating unknowns remain unknown."""

import pytest
from mlir_fixtures import forward, generic, pointwise

from merlin.llvmlower.literal_zero_components import fold_literal_zero_components

LIMB = '{test.limb="integer"}'


def source(literal):
    i8 = "tensor<2xi8>"
    prelude = "%z=arith.constant 0 : i8\n%zero=tensor.splat %z : tensor<2xi8>\n"
    prelude += "%s=arith.constant 0 : i32\n%seed=tensor.splat %s : tensor<2xi32>\n"
    body = """%x32=arith.extsi %a : i8 to i32
          %y32=arith.extsi %b : i8 to i32
          %p=arith.muli %x32,%y32 : i32
          %v=arith.addi %p,%acc : i32"""
    ins = [("%x", i8), ("%zero" if literal else "%y", i8)]
    product = generic(ins, "tensor<2xi32>", body, out="%seed", acc="%acc", attrs=LIMB)
    return forward(f"%x:{i8},%y:{i8}", "tensor<2xi32>", prelude + product)


@pytest.mark.parametrize("literal", [True, False])
def test_only_a_source_literal_zero_removes_a_symbolic_integer_product(literal):
    module = source(literal)
    assert fold_literal_zero_components(module, "test.limb") == int(literal)
    assert any("test.limb" in op.attributes for op in module.walk()) != literal
    module.verify()


def test_unknown_float_times_zero_is_not_an_integer_zero_proof():
    body = "%z=arith.constant 0.0 : f32\n%m=arith.mulf %a,%z : f32\n%i=arith.fptosi %m : f32 to i8"
    product = generic([("%x", "tensor<2xf32>")], "tensor<2xi8>", body, yields="%i", name="%p")
    widened = generic(
        [("%p", "tensor<2xi8>")], "tensor<2xi32>", "%i=arith.extsi %a : i8 to i32", yields="%i", attrs=LIMB
    )
    module = forward("%x:tensor<2xf32>", "tensor<2xi32>", product + "\n" + widened)
    assert fold_literal_zero_components(module, "test.limb") == 0


def test_index_arithmetic_is_not_treated_as_a_fixed_width_integer():
    body = """%zero=arith.constant 0 : index
          %one=arith.constant 1 : index
          %index=arith.addi %zero,%one : index
          %converted=arith.index_cast %index : index to i32"""
    module = pointwise(["tensor<2xi32>"], "tensor<2xi32>", body, yields="%converted", attrs=LIMB)
    assert fold_literal_zero_components(module, "test.limb") == 0
    assert any(op.name == "arith.index_cast" for op in module.walk())


def test_zero_result_does_not_erase_an_opaque_call():
    body = "%effect=func.call @effect() : () -> i8\n%zero=arith.constant 0 : i32"
    call = generic([("%x", "tensor<2xi8>")], "tensor<2xi32>", body, yields="%zero", attrs='{test.limb="unproved"}')
    module = forward("%x:tensor<2xi8>", "tensor<2xi32>", call, decls="func.func private @effect() -> i8")
    assert fold_literal_zero_components(module, "test.limb") == 0
    assert any(op.name == "func.call" for op in module.walk())


def test_deep_complete_limb_chain_does_not_exhaust_literal_fold_stack():
    t, prior, chain = "tensor<1xi32>", "%x", []
    for i in range(1100):
        add = "%one=arith.constant 1:i32\n%sum=arith.addi %a,%one:i32"
        chain.append(generic([(prior, t)], t, add, yields="%sum", name=f"%r{i}", attrs=LIMB if i == 1099 else ""))
        prior = f"%r{i}"
    module = forward(f"%x:{t}", t, "\n".join(chain), ret=prior)
    assert fold_literal_zero_components(module, "test.limb") == 0
    assert sum("test.limb" in op.attributes for op in module.walk()) == 1
