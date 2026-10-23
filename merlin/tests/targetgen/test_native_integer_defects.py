"""Integer and IEEE storage boundaries do not inherit unrelated proof gates."""

import pytest
from mlir_fixtures import forward, generic, pointwise

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_dtype_domain import component_semantics
from merlin.targetgen.native_unit_policy import construction_evidence


def quotient(divisor, *, unknown=False):
    operand = '%bad="opaque.carrier"(%x) : (tensor<4xi64>) -> tensor<4xi64>' if unknown else ""
    source = "%bad" if unknown else "%x"
    typ = "tensor<4xi64>"
    body = generic([(source, typ)], typ, "%v=arith.divsi %a,%d:i64")
    return forward(f"%x:{typ}", typ, f"{operand}\n%d=arith.constant {divisor} : i64\n{body}")


@pytest.mark.parametrize("divisor", [1, 3, 4, 9223372036854775807])
def test_positive_integer_quotient_needs_no_ieee_certificate(divisor, monkeypatch):
    def unavailable():
        raise AssertionError("integer arithmetic must not consult an IEEE proof")

    monkeypatch.setattr("merlin.targetgen.native_ieee_certificate.certificate", unavailable)
    proof = component_semantics(quotient(divisor))
    assert proof["status"] == "proven"
    assert proof["method"] == "typed_storage_and_modular_ring_v1"
    assert construction_evidence(quotient(divisor))["status"] == "eligible"


@pytest.mark.parametrize("divisor", [0, -1])
def test_undefined_quotient_domain_remains_unproved(divisor):
    assert component_semantics(quotient(divisor))["status"] == "refused"
    reasons = {x["reason"] for x in construction_evidence(quotient(divisor))["evidence"]}
    assert "unproved_payload_carrier" in reasons


def test_known_quotient_cannot_authenticate_an_unknown_input_carrier():
    proof = construction_evidence(quotient(4, unknown=True))
    assert proof["status"] == "host"
    assert any(x["reason"] == "unproved_payload_carrier" for x in proof["evidence"])


@pytest.mark.parametrize("target", ["f16", "bf16"])
def test_affine_data_followed_by_rne_storage_conversion_is_not_offset_rounding(target):
    module = pointwise(
        ["tensor<4xf32>"],
        f"tensor<4x{target}>",
        f"%c=arith.constant 0.5:f32\n%b=arith.addf %a,%c:f32\n%v=arith.truncf %b:f32 to {target}",
    )
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "eligible"


def test_biased_integer_rounding_still_refuses():
    module = parse_mlir_text("""module {
      func.func @forward(%x:f32) -> i32 {
        %c=arith.constant 0.5:f32
        %b=arith.addf %x,%c:f32
        %r=arith.fptosi %b:f32 to i32
        return %r:i32
      }}""")
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "host"


def reduction(dtype, shape, dimensions):
    """A zero-seeded union (i1) or sum (i64) of ``tensor<{shape}{dtype}>`` over ``dimensions``."""
    operator = "ori" if dtype == "i1" else "addi"
    literal = "false" if dtype == "i1" else "0:i64"
    return parse_mlir_text(
        """module {
      func.func @forward(%x:tensor<SHAPETYPE>) -> tensor<TYPE> {
        %z=arith.constant ZERO
        %s=tensor.splat %z:tensor<TYPE>
        %r=linalg.reduce ins(%x:tensor<SHAPETYPE>) outs(%s:tensor<TYPE>) dimensions=[DIMS]
          (%a:TYPE,%b:TYPE) { %v=arith.OP %a,%b:TYPE
            linalg.yield %v:TYPE }
        return %r:tensor<TYPE>
      }}""".replace("SHAPE", shape)
        .replace("TYPE", dtype)
        .replace("ZERO", literal)
        .replace("DIMS", dimensions)
        .replace("OP", operator)
    )


def test_boolean_union_reduction_is_a_hack_exclusion():
    module = reduction("i1", "4x", "0")
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "host"


def test_boolean_storage_carriers_remain_pure_movement():
    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<2x2xi1>) -> tensor<4xi1> {
        %r=tensor.collapse_shape %x [[0,1]]:tensor<2x2xi1> into tensor<4xi1>
        return %r:tensor<4xi1>
      }}""")
    assert construction_evidence(module)["status"] == "eligible"


@pytest.mark.parametrize("dtype,shape,dimensions", [("i1", "", ""), ("i1", "1x", "0"), ("i64", "", "")])
def test_single_contribution_zero_seed_reduction_is_storage_movement(dtype, shape, dimensions):
    from merlin.llvmlower.structural_folds import fold_singleton_reductions

    module = reduction(dtype, shape, dimensions)
    original = str(module)
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "eligible"
    assert str(module) == original
    assert fold_singleton_reductions(module) == 1
    assert not any(op.name in {"linalg.reduce", "arith.ori", "arith.addi"} for op in module.walk())
    assert any(op.name == "tensor.reshape" for op in module.walk())


def test_empty_shape_metadata_is_not_a_literal_element_splat():
    from merlin.targetgen.core_aten_exact_dtype import _expose_literal_movement

    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<1xi1>) -> tensor<i1> {
        %shape=arith.constant dense<>:tensor<0xi64>
        %r=tensor.reshape %x(%shape):(tensor<1xi1>,tensor<0xi64>) -> tensor<i1>
        return %r:tensor<i1>
      }}""")
    original = str(module)
    _expose_literal_movement(module)
    assert str(module) == original
