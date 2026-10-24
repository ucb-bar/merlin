"""Dropping an identity selector must preserve the source's NaN-quieting path."""

import pytest
from mlir_fixtures import assert_unchanged, forward, generic

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.image_folds import fold_ieee_identity_selectors, fold_zero_sign_selectors


@pytest.mark.parametrize("dtype,bits", [("f16", 16), ("bf16", 16), ("f32", 32), ("f64", 64)])
@pytest.mark.parametrize("correct", [False, True])
def test_zero_sign_selector_is_identity_only_with_the_complete_sign_word(dtype, bits, correct):
    mask = -(1 << (bits - 1)) if correct else (1 << (bits - 2))
    module = parse_mlir_text(f"""module {{func.func @forward(%x:{dtype}) -> {dtype} {{
      %zero=arith.constant 0.0 : {dtype}
      %mask=arith.constant {mask} : i{bits}
      %raw=arith.bitcast %x : {dtype} to i{bits}
      %sign=arith.andi %raw,%mask : i{bits}
      %signedzero=arith.bitcast %sign : i{bits} to {dtype}
      %p=arith.cmpf oeq,%x,%zero : {dtype}
      %out=arith.select %p,%signedzero,%x : {dtype}
      return %out : {dtype}
    }} }}""")
    assert fold_zero_sign_selectors(module) == int(correct)
    module.verify()


def source(predicate="ogt", factor="1.0", *, swapped=False, unknown=False):
    t, mask = "tensor<4xf32>", "tensor<4xi1>"
    prelude = f"""%z=arith.constant 0.0 : f32
      %one=arith.constant {factor} : f32
      %zt=tensor.splat %z : {t}
      %ot=tensor.splat %one : {t}
      %e=tensor.empty() : {t}
      %be=tensor.empty() : {mask}
"""
    compare = generic(
        [("%x", t), ("%zt", t)], mask, f"%p=arith.cmpf {predicate},%a,%b : f32", yields="%p", name="%cond", out="%be"
    )
    scale = generic([("%x", t), ("%ot", t)], t, "%p=arith.mulf %a,%b : f32", yields="%p", name="%m", out="%e")
    carrier = '\n%opaque="fixture.unknown"(%m) : (tensor<4xf32>) -> tensor<4xf32>' if unknown else ""
    choices = [("%m", t), ("%x", t)] if swapped else [("%x", t), ("%opaque" if unknown else "%m", t)]
    select = generic([("%cond", mask), *choices], t, "%s=arith.select %a,%b,%c : f32", yields="%s", out="%e")
    return forward(f"%x:{t}", t, f"{prelude}{compare}\n{scale}{carrier}\n{select}")


@pytest.mark.parametrize("predicate,swapped", [("ogt", False), ("ole", False), ("ugt", True), ("uno", True)])
def test_nan_path_keeps_the_original_unit_multiply(predicate, swapped):
    module = source(predicate, swapped=swapped)
    assert fold_ieee_identity_selectors(module) == 1
    assert any(op.name == "arith.mulf" for op in module.walk())
    assert not any(op.name in {"arith.cmpf", "arith.select"} for op in module.walk())


@pytest.mark.parametrize("options", [dict(predicate="ugt"), dict(factor="2.0"), dict(unknown=True)])
def test_different_nan_path_arithmetic_or_unknown_carrier_is_retained(options):
    assert_unchanged(source(**options), fold_ieee_identity_selectors)
