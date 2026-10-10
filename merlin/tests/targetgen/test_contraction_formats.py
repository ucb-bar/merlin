"""Actual parsed synthetic bodies; format observations confer no capture authority."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import yaml
from capture_contraction_inputs import _module, _pin, _save, _selection, _trace

from merlin.capture import contraction_formats as F


def test_complete_typed_integer_body_binds_original_count_and_macs(tmp_path):
    selection = _selection(tmp_path)
    observation = F.require_capture_contraction_formats(tmp_path, selection)
    census = observation["census"]
    assert census["eligible"] is True
    assert census["count_coverage"] == [1, 1]
    assert census["macs"] == {"original": 24, "integer": 24, "integer_coverage": [24, 24]}
    assert census["not_established"] == [
        "source_producer_authority",
        "numerical_equivalence",
        "precision_certification",
        "offload",
        "cycles",
        "runtime",
    ]


@pytest.mark.parametrize("floating", ["float", "bf16", "f16"])
def test_original_plain_or_preserved_float_is_in_complete_denominator(tmp_path, floating):
    selection = _selection(tmp_path, geometries=((1, 1, 1), (8, 16, 32)), formats=("integer", floating))
    census = F.observe_contraction_formats(selection)
    assert census["counts"]["integer"] == census["counts"]["float"] == 1
    assert census["count_coverage"] == [1, 2]
    assert census["macs"]["integer_coverage"] == [1, 4097]
    with pytest.raises(F.ContractionFormatError, match="coverage is not established"):
        F.require_capture_contraction_formats(tmp_path, selection)


@pytest.mark.parametrize("change", ["shape", "nested", "unknown_call", "duplicate_fragment", "bad_ssa", "tagged_float"])
def test_unproved_geometry_multiplicity_lineage_and_body_remain_unknown(tmp_path, change):
    selection = _selection(tmp_path)
    row = selection.programs[0]
    graph = json.loads(row.original_graph.path.read_text())
    text = row.final_mlir.path.read_text()
    if change == "shape":
        graph["nodes"][0]["results"][0]["shape"][1] = "dynamic"
    elif change == "nested":
        for node in graph["nodes"]:
            node["graph_id"] = "g:original:body"
    elif change == "unknown_call":
        graph["nodes"][2]["target"] = "custom.opaque"
        graph["by_target"] = {"custom.opaque": 1}
    elif change == "duplicate_fragment":
        text = _module(((2, 3, 4), (2, 3, 4)), ("integer", "integer")).replace('"mm1"', '"mm0"')
    elif change == "bad_ssa":
        text = text.replace("(%ex, %ey)", "(%ex, %ex)")
    elif change == "tagged_float":
        text = _module(((2, 3, 4),), ("float",)).replace('prov.op = "matmul"', 'prov.op = "int_matmul"')
    graph["sha256"] = F._digest({key: value for key, value in graph.items() if key != "sha256"})
    row.final_mlir.path.write_text(text)
    current = replace(
        row,
        original_graph=_save(row.original_graph.path, graph),
        frontend_trace=_save(row.frontend_trace.path, _trace(graph, text)),
        final_mlir=_pin(row.final_mlir.path),
    )
    census = F.observe_contraction_formats(replace(selection, programs=(current,)))
    assert census["eligible"] is False
    assert census["counts"]["unknown"] or census["counts"]["float"] or census["counts"]["other_unknowns"]
    if change in {"shape", "nested", "unknown_call"}:
        assert census["macs"]["integer_coverage"] is None


@pytest.mark.parametrize("member", ["original_graph", "frontend_trace", "final_mlir", "session_contract"])
def test_changed_selected_bytes_cannot_be_resigned_into_membership(tmp_path, member):
    selection = _selection(tmp_path)
    pin = getattr(selection, member) if member == "session_contract" else getattr(selection.programs[0], member)
    pin.path.write_bytes(pin.path.read_bytes() + b" ")
    with pytest.raises(F.ContractionFormatError, match="source bytes changed"):
        F.require_capture_contraction_formats(tmp_path, selection)


def test_trace_resigned_with_deleted_original_node_is_not_the_selected_original(tmp_path):
    selection = _selection(tmp_path)
    row = selection.programs[0]
    doc = json.loads(row.frontend_trace.path.read_text())
    doc["graphs"]["original"]["nodes"].pop()
    row = replace(row, frontend_trace=_save(row.frontend_trace.path, doc))
    with pytest.raises(F.ContractionFormatError, match="membership is incomplete"):
        F.observe_contraction_formats(replace(selection, programs=(row,)))


def test_session_unselected_program_and_no_recipe_float_stage_cannot_disappear(tmp_path):
    a = _selection(tmp_path / "first")
    b = _selection(tmp_path / "second", formats=("float",))
    contract = tmp_path / "session_contract.yaml"
    contract.write_text(
        yaml.safe_dump(
            {"version": 2, "programs": [{"name": "first", "bundle": "first"}, {"name": "second", "bundle": "second"}]}
        )
    )
    first = replace(a.programs[0], program="first")
    second = replace(b.programs[0], program="second")
    incomplete = F.CaptureContractionSelection(_pin(contract), (first,))
    with pytest.raises(F.ContractionFormatError, match="membership is incomplete"):
        F.require_capture_contraction_formats(tmp_path, incomplete)
    complete = replace(incomplete, programs=(first, second))
    assert F.observe_contraction_formats(complete)["counts"]["float"] == 1
    with pytest.raises(F.ContractionFormatError, match="coverage is not established"):
        F.require_capture_contraction_formats(tmp_path, complete)


@pytest.mark.parametrize(
    "limits",
    [
        F.ContractionFormatLimits(file_bytes=32),
        F.ContractionFormatLimits(metadata_items=8),
        F.ContractionFormatLimits(graph_nodes=2),
        F.ContractionFormatLimits(operations=2),
        F.ContractionFormatLimits(integer_bits=3),
    ],
)
def test_budgets_refuse_before_unbounded_metadata_or_work_expansion(tmp_path, limits):
    selection = replace(_selection(tmp_path), limits=limits)
    with pytest.raises(F.ContractionFormatError, match="budget|membership"):
        F.observe_contraction_formats(selection)


def test_source_quantized_subset_receipt_does_not_replace_original_census(tmp_path):
    selection = _selection(tmp_path, geometries=((1, 1, 1), (8, 16, 32)), formats=("integer", "float"))
    # An unrelated successful selected-subset receipt is deliberately ignored.
    (tmp_path / "integerization.json").write_text(json.dumps({"seen": 1, "integerized": 1, "status": "passed"}))
    assert F.observe_contraction_formats(selection)["counts"]["original_rows"] == 2
    with pytest.raises(F.ContractionFormatError, match="coverage is not established"):
        F.require_capture_contraction_formats(tmp_path, selection)


@pytest.mark.parametrize("target", ["aten.linear.default", "aten.conv2d.default", "aten.bmm.default"])
def test_unsupported_original_taxonomy_cannot_borrow_mm_geometry(tmp_path, target):
    selection = _selection(tmp_path)
    row = selection.programs[0]
    graph = json.loads(row.original_graph.path.read_text())
    graph["nodes"][2]["target"] = target
    graph["by_target"] = {target: 1}
    graph["sha256"] = F._digest({key: value for key, value in graph.items() if key != "sha256"})
    row = replace(
        row,
        original_graph=_save(row.original_graph.path, graph),
        frontend_trace=_save(row.frontend_trace.path, _trace(graph, row.final_mlir.path.read_text())),
    )
    census = F.observe_contraction_formats(replace(selection, programs=(row,)))
    assert census["counts"]["original_rows"] == census["counts"]["unknown"] == 1
    assert census["macs"]["original"] is None and census["eligible"] is False


@pytest.mark.parametrize("change", ["missing_origin", "fused_origin", "different_geometry"])
def test_actual_fragment_requires_one_original_member_and_exact_geometry(tmp_path, change):
    selection = _selection(tmp_path)
    row = selection.programs[0]
    text = row.final_mlir.path.read_text()
    if change == "missing_origin":
        text = text.replace('prov.origin_node_ids = ["mm0"]', "prov.origin_node_ids = []")
    elif change == "fused_origin":
        text = text.replace('prov.origin_node_ids = ["mm0"]', 'prov.origin_node_ids = ["mm0", "hidden"]')
    else:
        text = _module(((1, 3, 4),), ("integer",))
    graph = json.loads(row.original_graph.path.read_text())
    row.final_mlir.path.write_text(text)
    row = replace(
        row, final_mlir=_pin(row.final_mlir.path), frontend_trace=_save(row.frontend_trace.path, _trace(graph, text))
    )
    census = F.observe_contraction_formats(replace(selection, programs=(row,)))
    assert census["eligible"] is False
    assert census["counts"]["unknown"] == 1


@pytest.mark.parametrize("change", ["unknown_call", "unknown_node"])
def test_unknown_denominator_never_reports_known_only_coverage(tmp_path, change):
    selection = _selection(tmp_path)
    row = selection.programs[0]
    graph = json.loads(row.original_graph.path.read_text())
    if change == "unknown_call":
        graph["nodes"][2]["target"] = "unknown.original.call"
        graph["by_target"] = {"unknown.original.call": 1}
    else:
        graph["nodes"][0]["op"] = "unknown.original.node"
    graph["sha256"] = F._digest({key: value for key, value in graph.items() if key != "sha256"})
    row = replace(
        row,
        original_graph=_save(row.original_graph.path, graph),
        frontend_trace=_save(row.frontend_trace.path, _trace(graph, row.final_mlir.path.read_text())),
    )
    census = F.observe_contraction_formats(replace(selection, programs=(row,)))
    assert census["counts"]["complete_original_contraction_count"] is None
    assert census["macs"]["integer_coverage"] is None and census["count_coverage"] is None


@pytest.mark.parametrize("change", ["unknown_argument", "bool_ordinal", "bool_count", "bad_results", "bad_mlir_table"])
def test_resigned_structural_metadata_still_requires_actual_typed_membership(tmp_path, change):
    selection = _selection(tmp_path)
    row = selection.programs[0]
    graph = json.loads(row.original_graph.path.read_text())
    trace = json.loads(row.frontend_trace.path.read_text())
    if change == "unknown_argument":
        graph["nodes"][2]["args"][0]["value_id"] = "unselected"
    elif change == "bool_count":
        graph["call_count"] = True
    elif change == "bad_results":
        graph["nodes"][2]["results"] = None
    elif change == "bool_ordinal":
        trace["mlir"]["operations"][0]["ordinal"] = False
    else:
        trace["mlir"] = []
    graph["sha256"] = F._digest({key: value for key, value in graph.items() if key != "sha256"})
    trace["graphs"]["original"] = graph
    row = replace(
        row, original_graph=_save(row.original_graph.path, graph), frontend_trace=_save(row.frontend_trace.path, trace)
    )
    with pytest.raises(F.ContractionFormatError, match="membership is incomplete"):
        F.observe_contraction_formats(replace(selection, programs=(row,)))


@pytest.mark.parametrize("text", ["version: 1\nversion: 1\n", "version: 1\nx: &x []\ny: *x\n", "version: true\n"])
def test_ambiguous_or_non_integer_session_version_refuses(tmp_path, text):
    selection = _selection(tmp_path)
    selection.session_contract.path.write_text(text)
    selection = replace(selection, session_contract=_pin(selection.session_contract.path))
    with pytest.raises(F.ContractionFormatError, match="membership is incomplete"):
        F.require_capture_contraction_formats(tmp_path, selection)


def test_selected_file_growth_at_open_is_bounded_before_parse(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    limit = max(
        pin.path.stat().st_size
        for pin in (
            selection.session_contract,
            *[pin for row in selection.programs for pin in (row.original_graph, row.frontend_trace, row.final_mlir)],
        )
    )
    selection = replace(selection, limits=F.ContractionFormatLimits(file_bytes=limit))
    original_open = F.os.open

    def grow(path, flags):
        if path == selection.session_contract.path:
            path.write_bytes(b" " * (limit + 1))
        return original_open(path, flags)

    monkeypatch.setattr(F.os, "open", grow)
    with pytest.raises(F.ContractionFormatError, match="budget"):
        F.observe_contraction_formats(selection)


def test_whole_roster_byte_admission_precedes_any_program_parse(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    selection = replace(selection, limits=F.ContractionFormatLimits(total_bytes=1))

    def unexpected(*args):
        pytest.fail("program parsing occurred before whole-roster byte admission")

    monkeypatch.setattr(F, "_program", unexpected)
    with pytest.raises(F.ContractionFormatError, match="budget"):
        F.observe_contraction_formats(selection)


@pytest.mark.parametrize("change", ["unknown_dtype", "different_dtype", "scalar_operand", "zero_extent", "bool_extent"])
def test_original_operand_dtype_kind_and_geometry_are_required(tmp_path, change):
    selection = _selection(tmp_path)
    row = selection.programs[0]
    graph = json.loads(row.original_graph.path.read_text())
    operand = graph["nodes"][0]["results"][0]
    if change == "unknown_dtype":
        operand["dtype"] = {"unknown": "type"}
    elif change == "different_dtype":
        operand["dtype"] = "int8"
    elif change == "scalar_operand":
        operand["kind"] = "scalar"
    else:
        operand["shape"][0] = 0 if change == "zero_extent" else True
    graph["sha256"] = F._digest({key: value for key, value in graph.items() if key != "sha256"})
    row = replace(
        row,
        original_graph=_save(row.original_graph.path, graph),
        frontend_trace=_save(row.frontend_trace.path, _trace(graph, row.final_mlir.path.read_text())),
    )
    census = F.observe_contraction_formats(replace(selection, programs=(row,)))
    assert census["counts"]["unknown"] == 1 and census["eligible"] is False


def test_nested_public_graph_local_ordinals_retain_all_unknown_call_rows(tmp_path):
    selection = _selection(tmp_path)
    row = selection.programs[0]
    graph = json.loads(row.original_graph.path.read_text())
    nested = copy.deepcopy(graph["nodes"])

    def rename(value):
        if isinstance(value, dict):
            return {
                key: (
                    "nested:" + item if key in {"id", "node_id", "value_id"} and isinstance(item, str) else rename(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rename(item) for item in value]
        return value

    nested = rename(nested)
    for node in nested:
        node["graph_id"] = "g:original:body"
    graph["nodes"].extend(nested)
    graph["call_count"] = graph["by_target"]["aten.mm.default"] = 2
    graph["sha256"] = F._digest({key: value for key, value in graph.items() if key != "sha256"})
    row = replace(
        row,
        original_graph=_save(row.original_graph.path, graph),
        frontend_trace=_save(row.frontend_trace.path, _trace(graph, row.final_mlir.path.read_text())),
    )
    census = F.observe_contraction_formats(replace(selection, programs=(row,)))
    assert census["counts"]["original_rows"] == 2
    assert census["counts"]["integer"] == census["counts"]["unknown"] == 1
    assert census["macs"]["original"] is None and census["eligible"] is False
