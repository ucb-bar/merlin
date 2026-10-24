"""Native invalid-result NaN bits require an explicit source policy."""

import numpy as np
from mlir_fixtures import forward, generic, pointwise
from targetgen_fixtures import assert_host_refusal

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_dtype_domain import component_semantics


def division_module(literal):
    prelude = f"%zero=arith.constant {literal} : f32\n%divisor=tensor.splat %zero : tensor<1xf32>\n"
    ins = [("%x", "tensor<1xi32>"), ("%divisor", "tensor<1xf32>")]
    body = "%w=arith.sitofp %a : i32 to f32\n%v=arith.divf %w,%b : f32"
    return forward("%x: tensor<1xi32>", "tensor<1xf32>", prelude + generic(ins, "tensor<1xf32>", body))


def test_certified_integer_conversion_zero_division():
    proof = component_semantics(division_module("0.0"))
    assert proof["status"] == "proven"
    assert proof["method"] == "closed_typed_pointwise_IEEE_composition_v1"


def test_uncertified_binary64_zero_divisor_falls_back_before_provider(tmp_path):
    source = str(division_module("0.0"))
    source = source.replace("i32", "f64").replace("f32", "f64")
    source = source.replace("arith.sitofp %a : f64 to f64", "arith.negf %a : f64")
    inputs, cases = [np.array([0], np.float64)], [dict(input_abi=[dict(dtype="f64")])]
    code = "native_binary64_arithmetic_form_unproved"
    assert_host_refusal(tmp_path, parse_mlir_text(source), inputs, cases, code)


def test_bf16_nan_negation_needs_explicit_source_policy(tmp_path):
    """Native reduced-precision NaN policy is not inferred from IEEE negf."""
    module = pointwise(["tensor<1xbf16>"], "tensor<1xbf16>", "%v=arith.negf %a : bf16")
    inputs, cases = [np.array([float("nan")], np.float32)], [dict(input_abi=[dict(dtype="bf16")])]
    assert_host_refusal(tmp_path, module, inputs, cases, "native_cpu_word_layout_unproved")
