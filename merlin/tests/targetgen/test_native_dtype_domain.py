"""General component domains never consult example tensors or eager answers."""

from fractions import Fraction

import pytest
from mlir_fixtures import pointwise

from merlin.targetgen.native_dtype_domain import component_semantics, dtype_domain


@pytest.mark.parametrize("width", [1, 8, 16, 32, 64])
def test_complete_integer_domain(width):
    assert dtype_domain(f"i{width}")["absolute_bound"] == (1 if width == 1 else 1 << (width - 1))


@pytest.mark.parametrize(
    "dtype,minimum,maximum",
    [
        ("f16", Fraction(2) ** -24, Fraction(65504)),
        ("bf16", Fraction(2) ** -133, Fraction(255) * 2**120),
        ("f32", Fraction(2) ** -149, Fraction(2**24 - 1) * 2**104),
        ("f64", Fraction(2) ** -1074, Fraction(2**53 - 1) * 2**971),
    ],
)
def test_finite_grid_covers_full_dtype_without_omitting_special_values(dtype, minimum, maximum):
    domain = dtype_domain(dtype)
    exponent, bound = domain["finite_grid"]
    assert Fraction(2) ** exponent == minimum
    assert bound * minimum == maximum
    assert domain["ieee_special_values"] is True
    assert domain["absolute_bound"] is None


def unary(dtype, expression):
    return pointwise([f"tensor<4x{dtype}>"], f"tensor<4x{dtype}>", expression)


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_native_sign_theorem_is_about_all_input_words(dtype):
    proof = component_semantics(unary(dtype, f"%v=arith.negf %a : {dtype}"))
    assert proof["status"] == "proven"
    assert proof["input_domain"] == "all_storage_bit_patterns"
    assert proof["runtime_input_observations"] is False


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_reduced_precision_nan_dispatch_requires_its_own_theorem(dtype):
    proof = component_semantics(unary(dtype, f"%v=arith.negf %a : {dtype}"))
    assert proof["status"] == "proven"
    assert proof["method"] == "closed_typed_pointwise_IEEE_composition_v1"
    assert proof["certificate_sha256"]
    assert proof["runtime_input_observations"] is False


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
def test_finite_inputs_cannot_replace_the_general_ieee_certificate(dtype, monkeypatch):
    def unavailable():
        raise ValueError("missing full-domain certificate")

    monkeypatch.setattr("merlin.targetgen.native_ieee_certificate.certificate", unavailable)
    module = unary(dtype, f"%v=arith.addf %a,%a : {dtype}")
    assert component_semantics(module)["code"] == "native_cpu_certificate_unavailable"


def test_modular_ring_keeps_dtype_and_promotion_boundaries():
    proof = component_semantics(unary("i64", "%v=arith.addi %a,%a : i64"))
    assert proof["status"] == "proven"
    assert proof["integer_widths"] == [64]
    assert proof["integer_arithmetic"] == "modulo_each_source_storage_width"


def test_false_threshold_wiring_cannot_claim_signed_zero_or_nan_preservation():
    source = unary(
        "f32",
        """%z=arith.constant 0.0 : f32
          %c=arith.cmpf olt, %a,%z : f32
          %v=arith.select %c,%a,%z : f32""",
    )
    assert component_semantics(source)["code"] == "native_ieee_word_semantics_unproved"
