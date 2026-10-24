"""Discrete IEEE identities require the full unsigned Boolean conversion image."""

import pytest
from mlir_fixtures import assert_unchanged, forward, generic, pointwise

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.image_folds import fold_boolean_unit_image


def source(operation, *, element="f32", integer="i1", literal=None, unknown=False):
    coefficient = f"%c=arith.constant {literal} : {element}" if literal is not None else ""
    carrier = (
        '%opaque="fixture.unknown"(%x) : (tensor<4x' + integer + ">) -> tensor<4x" + integer + ">" if unknown else ""
    )
    args = "%f, %c" if literal is not None else "%f"
    calculation = (
        f'%v="arith.remf"(%f,%c) : ({element},{element}) -> {element}'
        if operation == "arith.remf"
        else f"%v={operation} {args} : {element}"
    )
    typ, result = f"tensor<4x{integer}>", f"tensor<4x{element}>"
    body = f"%f=arith.uitofp %a : {integer} to {element}\n{coefficient}\n{calculation}"
    return forward(f"%x:{typ}", result, carrier + "\n" + generic([("%opaque" if unknown else "%x", typ)], result, body))


@pytest.mark.parametrize("element", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize(
    "operation,literal",
    [
        ("math.sqrt", None),
        ("math.floor", None),
        ("arith.divf", 1.0),
        ("math.powf", 0.5),
        ("arith.remf", 0.5),
        ("arith.remf", 1.25),
        ("arith.remf", -2.0),
        ("arith.subf", 0.0),
    ],
)
def test_every_boolean_word_has_the_proved_ieee_result(element, operation, literal):
    module = source(operation, element=element, literal=literal)
    assert fold_boolean_unit_image(module) == 1
    module.verify()
    assert not any(op.name == operation for op in module.walk())


# A rounded reciprocal is not an exact remainder proof.
@pytest.mark.parametrize(
    "operation,options",
    [("math.sqrt", dict(integer="i8")), ("math.sqrt", dict(unknown=True))]
    + [("arith.remf", dict(element="f64", literal="0.3333333333333333"))],
)
def test_other_or_unknown_integer_images_are_not_boolean(operation, options):
    assert_unchanged(source(operation, **options), fold_boolean_unit_image)


@pytest.mark.parametrize(
    "integer,mask,folds", [("i1", -2147483648, True), ("i1", 1065353216, False), ("i8", -2147483648, False)]
)
def test_a_storage_mask_is_zero_only_over_the_proved_boolean_image(integer, mask, folds):
    module = parse_mlir_text(f"""module {{func.func @forward(%a:{integer}) -> i32 {{
      %f=arith.uitofp %a : {integer} to f32
      %raw=arith.bitcast %f : f32 to i32
      %mask=arith.constant {mask} : i32
      %v=arith.andi %raw,%mask : i32
      return %v : i32
    }} }}""")
    assert (fold_boolean_unit_image(module) > 0) is folds


@pytest.mark.parametrize("zero,identity", [("0.0", True), ("-0.0", False)])
def test_unit_selector_does_not_erase_a_negative_zero_word(zero, identity):
    body = f"""%f=arith.uitofp %a : i1 to f32
      %positive=arith.constant 0.0 : f32
      %z=arith.constant {zero} : f32
      %p=arith.cmpf oeq,%f,%positive : f32
      %v=arith.select %p,%z,%f : f32"""
    module = pointwise(["tensor<4xi1>"], "tensor<4xf32>", body)
    original = str(module)
    assert (fold_boolean_unit_image(module) > 0) == identity
    if not identity:
        assert str(module) == original
