"""Configuration-derived target layout choices; no simulator or RTL files required."""
from __future__ import annotations

import importlib.util
import sys

from merlin.common.paths import repo_root
from merlin.perf.physical_layout import (
    LayoutGraph, LayoutValue, PhysicalEncoding, plan_physical_layout,
)


SPEC = importlib.util.spec_from_file_location(
    "gemmini_physical_layout",
    repo_root() / "merlin/targets/gemmini/backend/gemmini_physical_layout.py",
)
TARGET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TARGET
assert SPEC.loader is not None
SPEC.loader.exec_module(TARGET)


def _request(batch: int = 1):
    return TARGET.NativeConvLayoutRequest(
        "conv", "x", ("w",), ("y",), batch, ("fixture source operation identity",))


def _facts(*, mesh_rows: int, evidence: str | None = "hardware-run-hash",
           evidence_configuration: str | None = "fixture-configuration",
           capabilities=("native_conv_non_transposed_input", "native_conv_transposed_input")):
    return TARGET.NativeConvLayoutFacts(
        mesh_rows, tuple(capabilities), "channels_last", "channels_first", "packed_parameter",
        "fixture-configuration", evidence, evidence_configuration)


def _value(name: str) -> LayoutValue:
    return LayoutValue(name, (1, 2, 2, 4), "i8", (
        PhysicalEncoding("channels_first", "channels_first", "external", 16),
        PhysicalEncoding("channels_last", "channels_last", "external", 16),
        PhysicalEncoding("packed_parameter", "packed_parameter", "external", 16),
    ))


def test_batch_one_transpose_is_rejected_from_supplied_mesh_geometry() -> None:
    decision = TARGET.native_convolution_layout_decision(_request(), _facts(mesh_rows=8))
    assert decision.op is not None
    assert [option.encoding_id for option in decision.op.options] == ["channels_last"]
    refusal = next(item for item in decision.refusals
                   if item.code == TARGET.UNDERFILLED_TRANSPOSED_INPUT_REFUSAL)
    assert "1 of 8" in refusal.detail
    receipt = dict(decision.receipt)
    assert receipt["active_transposed_systolic_rows"] == 1
    assert receipt["available_systolic_rows"] == 8
    assert receipt["transposed_systolic_row_utilization"] == 1 / 8


def test_guard_is_not_a_fixed_shape_or_fixed_mesh_rule() -> None:
    decision = TARGET.native_convolution_layout_decision(_request(batch=12), _facts(mesh_rows=8))
    assert decision.op is not None
    assert [option.encoding_id for option in decision.op.options] == [
        "channels_last", "channels_first"]
    assert not any(item.code == TARGET.UNDERFILLED_TRANSPOSED_INPUT_REFUSAL
                   for item in decision.refusals)
    assert dict(decision.receipt)["transposed_systolic_row_utilization"] == 1.0


def test_graph_propagated_non_transposed_layout_wins_when_both_are_legal() -> None:
    decision = TARGET.native_convolution_layout_decision(_request(batch=8), _facts(mesh_rows=8))
    assert decision.op is not None
    result = plan_physical_layout(LayoutGraph(
        tuple(_value(name) for name in ("x", "w", "y")),
        (decision.op,),
        ("native_conv_non_transposed_input", "native_conv_transposed_input"),
        (),
        target_refusals=decision.refusals,
    ))
    assert result.plan is not None
    assert dict(result.plan.assignments) == {
        "w": "packed_parameter", "x": "channels_last", "y": "channels_last"}
    assert dict(result.plan.selected_options) == {"conv": "native_conv_non_transposed_input"}
    assert result.plan.cost.preferred_weight == 1


def test_underfill_without_configuration_bound_evidence_fails_closed() -> None:
    decision = TARGET.native_convolution_layout_decision(
        _request(),
        _facts(mesh_rows=4, evidence=None, capabilities=("native_conv_transposed_input",)),
    )
    assert decision.op is None
    assert {item.code for item in decision.refusals} == {
        "native_non_transposed_input_unavailable", "missing_hardware_cost_guard_evidence"}
    assert dict(decision.receipt)["transposed_systolic_row_utilization"] == 0.25


def test_underfill_with_evidence_from_another_configuration_fails_closed() -> None:
    decision = TARGET.native_convolution_layout_decision(
        _request(),
        _facts(mesh_rows=4, evidence_configuration="different-configuration",
               capabilities=("native_conv_transposed_input",)),
    )
    assert decision.op is None
    assert "missing_hardware_cost_guard_evidence" in {
        item.code for item in decision.refusals}
