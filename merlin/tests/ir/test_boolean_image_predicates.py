"""A comparison over Boolean conversions proves all possible source words."""

import math
import operator

import pytest
from mlir_fixtures import assert_unchanged, forward, generic, returned

from merlin.llvmlower.image_folds import fold_boolean_image_predicates

# Independent of the production predicate table: IEEE cmpf on ordered operands.
ORDERED = {"oeq": operator.eq, "one": operator.ne, "ogt": operator.gt, "olt": operator.lt, "uge": operator.ge}


def source(predicate, threshold, *, typ="f32", conversion="uitofp", element="i1", unknown=False):
    value = "0x7FC00000" if math.isnan(threshold) else str(float(threshold))
    word, real = f"tensor<2x{element}>", f"tensor<2x{typ}>"
    carrier = f'%opaque = "fixture.unknown"(%x) : ({word}) -> {word}\n' if unknown else ""
    convert = f"%f=arith.{conversion} %a : {element} to {typ}"
    image = generic([("%opaque" if unknown else "%x", word)], real, convert, yields="%f", name="%v")
    threshold = f"%c = arith.constant {value} : {typ}\n%t = tensor.splat %c : {real}\n"
    cmpf = f"%p = arith.cmpf {predicate}, %a, %b : {typ}"
    compare = generic([("%v", real), ("%t", real)], "tensor<2xi1>", cmpf, yields="%p")
    return forward(f"%x:{word}", "tensor<2xi1>", carrier + image + "\n" + threshold + compare)


@pytest.mark.parametrize("typ", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("predicate", sorted(ORDERED))
@pytest.mark.parametrize("threshold", [-1, 0, 0.5, 1, 2])
@pytest.mark.parametrize("conversion", ["uitofp", "sitofp"])
def test_only_identity_or_uniform_images_are_emitted(typ, predicate, threshold, conversion):
    answers = [ORDERED[predicate](v, threshold) for v in (0, -1 if conversion == "sitofp" else 1)]
    module = source(predicate, threshold, typ=typ, conversion=conversion)
    expected = answers[0] == answers[1] or answers == [False, True]
    assert fold_boolean_image_predicates(module) == int(expected)
    module.verify()
    if expected:
        producer = returned(module).owner
        assert producer.name == ("tensor.splat" if answers[0] == answers[1] else "linalg.generic")
        if producer.name == "linalg.generic":
            assert producer.inputs[0] is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]


def test_nan_threshold_is_uniform_for_an_integer_image():
    module = source("oeq", math.nan)
    assert fold_boolean_image_predicates(module) == 1


@pytest.mark.parametrize("options", [dict(element="i8"), dict(unknown=True)])
def test_larger_integer_or_unknown_payload_cannot_gain_a_two_word_proof(options):
    assert_unchanged(source("oeq", 1, **options), fold_boolean_image_predicates)
