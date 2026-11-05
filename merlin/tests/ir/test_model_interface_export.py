"""Interface export reports every call, including failed or absent requests."""

from types import SimpleNamespace

from merlin.common import mlir_query as mq
from merlin.compile.model_interface import export_rewrite
from merlin.xdsl_dialects.lowering.compute_groups import Group
from merlin.xdsl_dialects.lowering.group_route_report import lowering_obligation


def test_missing_request_is_not_a_successful_model_export(tmp_path):
    module = mq.parse("module { func.func @forward() { return } }")
    rewrite = SimpleNamespace(entries={}, routed=[SimpleNamespace(symbol="missing")], skipped=(), to_sidecar=lambda: {})
    result = export_rewrite(module, rewrite, binder=None, output=tmp_path / "export", source_reference="source.mlir")
    assert result["status"] == "incomplete"
    assert result["missing_call_interfaces"] == ["missing"]
    assert result["calls_with_interface"] == 0
    assert not result["target_execution"]


def test_refused_interface_keeps_its_call_in_the_denominator(tmp_path):
    module = mq.parse("module { func.func @forward() { return } }")
    rewrite = SimpleNamespace(entries={"requested": {}}, routed=[SimpleNamespace(symbol="requested")], skipped=(), to_sidecar=lambda: {})
    result = export_rewrite(module, rewrite, binder=None, output=tmp_path / "export", source_reference="source.mlir")
    assert result["status"] == "incomplete"
    assert result["device_calls"] == 1
    assert result["requests"][0]["status"] == "refused"
    assert result["requests"][0]["error"] == "UnboundGroupCapsule"


def test_family_admission_does_not_disguise_dequantization_as_missing_integer_arithmetic():
    group = Group(0, "host", None, stages=["dequantize", "view", "movement"], refusal="no_device_form")
    assert lowering_obligation(group) == "numerical_boundary"
    group.stages = ["elementwise"]
    assert lowering_obligation(group) == "missing_device_form"
    group.refusal = "undeclared_family"
    assert lowering_obligation(group) == "unresolved_evidence"
