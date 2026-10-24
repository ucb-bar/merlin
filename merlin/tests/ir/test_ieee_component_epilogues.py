# ruff: noqa: E501 -- MLIR fixtures retain one operation per line
"""Exact bit checks for signed-zero and separable invalid-result epilogues."""

import mlir_fixtures
import pytest
from mlir_fixtures import forward, generic

from merlin.llvmlower.fixed_point_ir import FORMAT
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.ieee_aggregate import propagation_source
from merlin.llvmlower.ieee_zero_division import rewrite_zero_division
from merlin.targetgen.core_aten_components import unlowered_data_reductions


def run_block(block, arguments):
    assert all(op.predicate.value.data == 1 for op in block.ops if op.name == "arith.cmpf")  # ordered equal
    assert all(op.predicate.value.data in {0, 1, 6, 8} for op in block.ops if op.name == "arith.cmpi")  # eq ne slt sgt
    return mlir_fixtures.run_block(block, arguments)


def module(token, operation, divisor=None):
    literal = (
        f"%c=arith.constant {divisor} : {token}\n%constant=tensor.splat %c : tensor<1x{token}>"
        if divisor is not None
        else ""
    )
    typ = f"tensor<1x{token}>"
    rhs = "%constant" if divisor is not None else "%y"
    computation = generic([("%x", typ), (rhs, typ)], typ, f"%v=arith.{operation} %a,%b : {token}")
    return forward(f"%x:{typ},%y:{typ}", typ, literal + "\n" + computation)


@pytest.mark.parametrize("token", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("operation", ["addf", "subf"])
def test_exact_zero_uses_ieee_rne_operand_signs(token, operation):
    source = module(token, operation)
    _, proofs = rewrite_floating_components(source, [(0, 1), (0, 1)])
    assert proofs
    assert unlowered_data_reductions(source, "test.limb") == []
    final = list(op for op in source.walk() if op.name == "linalg.generic")[-1]
    f, e, bias = FORMAT[token]
    sign = 1 << (f + e)
    for left in (0, sign):
        for right in (0, sign):
            expected_negative = bool(left) and (not bool(right) if operation == "subf" else bool(right))
            assert run_block(final.body.block, [0, 0, left, right, 0]) == (sign if expected_negative else 0)
    # Nonzero exact cancellation rounds to positive zero under RNE.
    one = bias << f
    right = one if operation == "subf" else one | sign
    assert run_block(final.body.block, [0, 0, one, right, 0]) == 0


@pytest.mark.parametrize("token", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("divisor", ["0.0", "-0.0"])
def test_zero_divisor_overlay_depends_on_the_computed_numerator(token, divisor):
    source = module(token, "divf", divisor)
    assert rewrite_zero_division(source) == 1
    producers = list(op for op in source.walk() if op.name == "linalg.generic")
    assert "merlin.exact_host_arithmetic" in producers[-1].attributes
    assert producers[-1].inputs[0].owner is producers[-2]
    f, e, bias = FORMAT[token]
    sign, inf = 1 << (f + e), ((1 << e) - 1) << f
    zero_sign = sign if divisor == "-0.0" else 0
    invalid = inf | 1 << (f - 1) | (0 if token == "bf16" else sign)
    for numerator in (0, sign):
        assert run_block(producers[-1].body.block, [numerator, zero_sign, 0]) == invalid
    for numerator in (bias << f, (bias << f) | sign):
        assert run_block(producers[-1].body.block, [numerator, zero_sign, 0]) == inf | ((numerator ^ zero_sign) & sign)


@pytest.mark.parametrize("token", ["f16", "bf16", "f32", "f64"])
def test_aggregate_propagation_keeps_special_words_and_discards_finite_terms(token):
    source = module(token, "addf")
    op = next(op for op in source.walk() if op.name == "linalg.generic")
    propagation, proof = propagation_source(op, lambda value: (0, 2))
    assert proof["finite_dependence_outputs"] == "retain_exact_device_integer_readout_without_overlay"
    assert not any(op.name in {"arith.addf", "arith.mulf"} for op in propagation.walk())
    f, e, bias = FORMAT[token]
    inf, sign = ((1 << e) - 1) << f, 1 << (f + e)
    one, nan = bias << f, inf | 1 << (f - 1)
    invalid = nan | (0 if token == "bf16" else sign)
    assert run_block(propagation.body.block, [one, one, 0]) == 0
    assert run_block(propagation.body.block, [inf, one, 0]) == inf
    assert run_block(propagation.body.block, [inf, inf | sign, 0]) == invalid
    assert run_block(propagation.body.block, [nan, one, 0]) == nan


def test_aggregate_finite_overflow_requires_host_instruction_order():
    source = module("f32", "addf")
    op = next(op for op in source.walk() if op.name == "linalg.generic")
    with pytest.raises(ValueError, match="host instruction order"):
        propagation_source(op, lambda value: (127, (1 << 24) - 1))


def test_pointwise_finite_overflow_has_its_original_typed_rne_boundary():
    source = module("f32", "addf")
    _, proofs = rewrite_floating_components(source, [(127, 1), (127, 1)], ieee_pointwise=True)
    assert proofs
    assert not any(op.name == "arith.addf" for op in source.walk())
