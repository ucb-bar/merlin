"""Logical uniformity never exempts an input-indexed or unknown table carrier."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_unit_policy import construction_evidence


def source(values, *, opaque=False):
    return parse_mlir_text(f"""module {{func.func @forward() -> tensor<2x2xf32> {{
      %t=arith.constant dense<{values}> : tensor<2x2xf32>
      {'%u="fixture.unknown"(%t) : (tensor<2x2xf32>) -> tensor<2x2xf32>' if opaque else ""}
      return {"%u" if opaque else "%t"} : tensor<2x2xf32>
    }} }}""")


def test_uniform_complex_words_are_a_literal_fill_not_a_result_table():
    module = source("[[1.0,0.0],[1.0,0.0]]")
    assert construction_evidence(module)["status"] == "host"
    result = construction_evidence(module, result_abi={0: dict(dtype="complex<f32>", shape=[2])})
    assert result["status"] == "eligible"
    assert result["logical_literal_results"][0]["dtype"] == "complex<f32>"


@pytest.mark.parametrize(
    "abi",
    [dict(dtype="f32", shape=[2, 2]), dict(dtype="complex<f32>", shape=[1, 2]), dict(dtype="complex<f64>", shape=[2])],
)
def test_an_unbound_or_incompatible_abi_cannot_hide_a_result_table(abi):
    module = source("[[1.0,0.0],[1.0,0.0]]")
    assert construction_evidence(module, result_abi={0: abi})["status"] == "host"


def test_different_complex_words_remain_a_nonuniform_result_table():
    module = source("[[1.0,0.0],[2.0,0.0]]")
    result = construction_evidence(module, result_abi={0: dict(dtype="complex<f32>", shape=[2])})
    assert "nonuniform_literal_result_construction" in {e["reason"] for e in result["evidence"]}


def test_unknown_carrier_remains_closed_even_for_a_logically_uniform_literal():
    module = source("[[1.0,0.0],[1.0,0.0]]", opaque=True)
    result = construction_evidence(module, result_abi={0: dict(dtype="complex<f32>", shape=[2])})
    assert result["status"] == "host"
    assert "unproved_payload_carrier" in {e["reason"] for e in result["evidence"]}


def test_input_indexed_literal_slice_is_not_certified_as_a_logical_fill():
    module = parse_mlir_text("""module {func.func @forward(%offset:index) -> tensor<2x2xf32> {
      %t=arith.constant dense<[[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]]> : tensor<4x2xf32>
      %r="tensor.extract_slice"(%t,%offset) : (tensor<4x2xf32>,index) -> tensor<2x2xf32>
      return %r : tensor<2x2xf32>
    }}""")
    result = construction_evidence(module, result_abi={0: dict(dtype="complex<f32>", shape=[2])})
    assert result["status"] == "host"
    assert "nonuniform_literal_result_construction" in {row["reason"] for row in result["evidence"]}
