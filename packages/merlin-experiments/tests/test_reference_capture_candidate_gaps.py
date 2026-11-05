"""Unsupported candidate families name the missing proof adapter without granting axes."""

from __future__ import annotations

import pytest
from merlin_experiments.phase1.feedback.reference_capture_views import candidate_view, compare_candidate
from test_reference_views import inputs

from merlin.targetgen.contract.interface_emit import emit_interface_mlir


def observe(source, *, semantics=object()):
    # Every fixture stops at its actual source-contract boundary. No invented
    # decoder, execution or emitted function is needed to identify that boundary.
    return candidate_view(dict(source=source, emitted=b"not emitted IR"), entry_symbol="kernel_entry",
                          decoder=dict(target="selected", semantics=semantics))


def test_typed_multi_product_requires_complete_subviews_and_output_composition():
    source, _, _ = inputs()
    report, cca = observe(source)
    assert report["reason"] == "candidate_product_readout_cardinality_unsupported"
    assert report["source_counts"] == dict(products=2, commits=2)
    assert report["supported_counts"] == dict(products=1, commits=1)
    assert report["required_adapter"] == "typed_multi_product_view"
    assert report["required_obligations"] == ["per_product_operand_subviews", "complete_output_coordinates",
                                               "ordered_output_composition", "per_product_numeric_contract"]
    assert report["status"] == "unknown" and cca is None
    assert report["semantic_alignment"] == report["numerical_equivalence"] == "unknown"
    assert "command_counts" not in report and "cca" not in report
    comparison = compare_candidate(dict(problems=[], schedule_model=dict(instruction_roles={}, role_counts={})),
                                   dict(captured_reference_axes=report), None, cca)
    assert comparison["cca"]["denominator"] == 0
    gap, = comparison["cca"]["unknowns"]
    assert gap["reason"] == report["reason"] and gap["source_counts"] == report["source_counts"]
    assert gap["required_adapter"] == "typed_multi_product_view"


@pytest.mark.parametrize("stages", [["relu"], ["acc_scale", "relu"], ["relu", "acc_scale"]])
def test_ordered_numeric_stages_require_a_separate_adapter(stages):
    _, cb, _ = inputs()
    cb["commands"] = [command for command in cb["commands"]
                      if command.get("operands", {}).get("dst") not in ("acc1", "y1")]
    cb["commands"][2]["attributes"]["epilogue"] = stages
    report, cca = observe(emit_interface_mlir(cb).encode())
    assert report["reason"] == "candidate_numeric_epilogue_adapter_unavailable"
    assert report["required_adapter"] == "ordered_numeric_epilogue"
    assert report["ordered_stages"] == stages
    assert report["execution"] == "not_established" and cca is None


def test_generic_source_is_not_confused_with_a_closed_interface_or_zero_products():
    source = b"""module {
      func.func @product(%a: tensor<3x5xi8>, %b: tensor<5x7xi8>, %c: tensor<3x7xi32>) -> tensor<3x7xi32> {
        %0 = linalg.matmul ins(%a, %b : tensor<3x5xi8>, tensor<5x7xi8>)
                           outs(%c : tensor<3x7xi32>) -> tensor<3x7xi32>
        return %0 : tensor<3x7xi32>
      }
    }"""
    report, cca = observe(source)
    assert report["reason"] == "candidate_source_to_command_buffer_adapter_unavailable"
    assert report["required_adapter"] == "typed_source_to_command_buffer"
    assert report["supported_source_contract"] == "closed_merlin_interface"
    assert "source_counts" not in report and cca is None


def test_missing_selected_semantics_is_distinct_from_an_unsupported_source_family():
    source, _, _ = inputs()
    report, cca = observe(source, semantics=None)
    assert report["reason"] == "candidate_selected_operand_semantics_unavailable"
    assert report["required_adapter"] == "selected_operand_decoder" and cca is None
