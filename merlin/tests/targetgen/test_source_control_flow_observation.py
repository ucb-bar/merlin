"""Explicit complete CFG transport, never executed output or layout proof."""

import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from test_emitted_control_flow import _LOOP
from test_source_observation import _ABI, _CB, _SOURCE, Reader

from merlin.common import invocation_record as I
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.emitted_dataflow import DataflowUnavailable
from merlin.targetgen.contract.source_control_flow import (
    LAYOUT_SCHEMA,
    READER_SCHEMA,
    ControlFlowObservationPlan,
    _read,
)
from merlin.targetgen.contract.source_observation import ExplicitSourceObservation


class ControlFlowReader:
    """Synthetic structural reader; its false PASS label cannot issue a role."""

    def __init__(self):
        self.selection = "complete static diagnostic"
        self.calls = 0
        self.mutation = None
        self.graph = None

    def verify(self):
        pass

    def record(self):
        return {"schema": READER_SCHEMA, "selection": self.selection}

    def observe_control_flow(self, **kwargs):
        assert "dataflow" not in kwargs
        self.calls += 1
        self.graph = kwargs["control_flow"]
        if self.mutation == "callback":
            self.observe_control_flow = self.substitute
        elif self.mutation == "selection":
            self.selection = "replacement"
        elif self.mutation == "product":
            kwargs["lowered_mlir"].write_text("module {}")
        return {"claimed_stage": "PASS", "blocks": len(self.graph.blocks)}

    def substitute(self, **kwargs):
        return {"replacement": True}


def prepare(tmp_path, *, text=_LOOP, reader=None, overrides=None):
    source, selected, layout = (tmp_path / name for name in ("source.mlir", "selected.mlir", "layout.json"))
    emitted, command = (tmp_path / name for name in ("actual.mlir", "command.json"))
    source.write_text(_SOURCE)
    selected.write_text(text)
    emitted.write_text(text)
    layout.write_text(json.dumps({"schema": LAYOUT_SCHEMA, "data_layout": "e-p:64:64", "pointer_bits": 64}))
    command.write_text(json.dumps(_CB))
    plan = ControlFlowObservationPlan(
        selected,
        file_digest(selected),
        "entry",
        layout,
        file_digest(layout),
        64,
        **{
            "max_source_bytes": 10000,
            "max_nesting": 64,
            "max_blocks": 16,
            "max_operations": 100,
            "max_values": 100,
            "max_edges": 32,
            "max_integer_bits": 64,
            "max_layout_bytes": 1000,
            **(overrides or {}),
        },
    )
    reader = ControlFlowReader() if reader is None else reader
    owner = Path(inspect.getsourcefile(type(reader)))
    service = ExplicitSourceObservation(
        "diagnostic-target",
        source,
        _ABI,
        64,
        plan.max_operations,
        reader,
        tuple((str(path), file_digest(path)) for path in (source, selected, layout, owner)),
        control_flow_plan=plan,
    )
    arguments = {
        "source": source,
        "lowered_mlir": emitted,
        "command_buffer": _CB,
        "command_buffer_path": command,
        "entry_symbol": "entry",
        "evidence_root": tmp_path,
    }
    return service, reader, arguments


def test_complete_loop_graph_and_input_record_remain_conditional(tmp_path):
    service, reader, arguments = prepare(tmp_path)
    before = service.sha256
    result = service.observe(**arguments)
    assert service.sha256 == before and reader.calls == 1
    graph = result["control_flow"]
    assert len(graph["blocks"]) == 4 and len(graph["arguments"]) == 2
    incoming = [edge for op in graph["operations"] for edge in op["edges"] if edge["successor"] == 1]
    assert len(incoming) == 2 and incoming[0]["arguments"] != incoming[1]["arguments"]
    assert result["observations"]["claimed_stage"] == "PASS"
    assert {"same_object_data_layout", "layout", "source_equivalence", "instruction_effects", "runtime"} <= set(
        result["unknown"]
    )
    (record,) = [I.verify(path) for path in tmp_path.rglob("invocation.json")]
    assert record["stage"] == "explicit_original_source_cfg_observation"
    assert {row["path"] for row in record["inputs"]} == {
        str(service.original_source),
        str(arguments["lowered_mlir"]),
        str(arguments["command_buffer_path"]),
        str(service.control_flow_plan.lowered_source),
        str(service.control_flow_plan.layout_source),
    }
    persisted = json.loads((tmp_path / "source_observation.json").read_text())
    assert persisted["control_flow"] == json.loads(json.dumps(graph))


def test_branch_pointer_joins_and_dead_blocks_are_not_discarded(tmp_path):
    source = """module { llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr) {
      %cond = llvm.mlir.constant(1 : i1) : i1
      llvm.cond_br %cond, ^join(%a : !llvm.ptr), ^join(%b : !llvm.ptr)
    ^join(%chosen: !llvm.ptr):
      llvm.return
    ^dead:
      llvm.return } }"""
    service, reader, arguments = prepare(tmp_path, text=source)
    result = service.observe(**arguments)
    assert len(result["control_flow"]["blocks"]) == 3
    edges = reader.graph.operations[1].edges
    assert edges[0].parameters == edges[1].parameters and edges[0].arguments != edges[1].arguments
    assert {"dynamic_execution_paths", "alias", "complete_output_stores"} <= set(result["unknown"])


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_source_bytes", 4),
        ("max_nesting", 1),
        ("max_blocks", 1),
        ("max_operations", 1),
        ("max_values", 1),
        ("max_edges", 1),
        ("max_integer_bits", 8),
        ("max_layout_bytes", 4),
    ],
)
def test_every_selected_observation_budget_refuses_without_reader(tmp_path, name, value):
    with pytest.raises(ValueError):
        service, reader, arguments = prepare(tmp_path, overrides={name: value})
        service.observe(**arguments)
    if "reader" in locals():
        assert reader.calls == 0


@pytest.mark.parametrize(
    "name",
    [
        "max_source_bytes",
        "max_nesting",
        "max_blocks",
        "max_operations",
        "max_values",
        "max_edges",
        "max_integer_bits",
        "max_layout_bytes",
        "pointer_bits",
    ],
)
@pytest.mark.parametrize("value", [0, True])
def test_limits_are_positive_exact_integers(tmp_path, name, value):
    service, _, _ = prepare(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        replace(service.control_flow_plan, **{name: value})


@pytest.mark.parametrize("member", ["lowered_source", "layout_source"])
@pytest.mark.parametrize("mutation", ["missing", "changed", "symlink"])
def test_original_selected_files_must_remain_complete_current_regular_bytes(tmp_path, member, mutation):
    service, reader, arguments = prepare(tmp_path)
    path = getattr(service.control_flow_plan, member)
    if mutation == "changed":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        original = path.read_bytes()
        path.unlink()
        if mutation == "symlink":
            other = tmp_path / "substitution"
            other.write_bytes(original)
            path.symlink_to(other)
    with pytest.raises(ValueError):
        service.observe(**arguments)
    assert reader.calls == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("llvm.load %src", "llvm.call @opaque(%src)"),
        lambda text: text.replace(
            "llvm.store %value, %dst : i8, !llvm.ptr", "llvm.store %value, %dst {claimed_effect = true} : i8, !llvm.ptr"
        ),
        lambda text: text.replace("module {", 'module attributes {llvm.data_layout = "e-p:64:64"} {'),
        lambda text: text.replace("llvm.br ^loop(%zero : i64)", "llvm.br ^loop"),
    ],
)
def test_unknown_operation_effect_module_layout_and_partial_join_are_not_stripped(tmp_path, change):
    service, reader, arguments = prepare(tmp_path, text=change(_LOOP))
    with pytest.raises(DataflowUnavailable):
        service.observe(**arguments)
    assert reader.calls == 0


@pytest.mark.parametrize("change", ["entry", "emitted", "pointer_roster"])
def test_actual_caller_must_join_the_selected_complete_original_input(tmp_path, change):
    service, reader, arguments = prepare(tmp_path)
    if change == "entry":
        arguments["entry_symbol"] = "absent"
    else:
        text = _LOOP + "\n" if change == "emitted" else _LOOP.replace(", %output: !llvm.ptr", "")
        arguments["lowered_mlir"].write_text(text)
    with pytest.raises(ValueError, match="selected"):
        service.observe(**arguments)
    assert reader.calls == 0


@pytest.mark.parametrize("mutation", ["callback", "selection", "product"])
def test_selected_reader_and_actual_input_changes_are_rechecked_after_invocation(tmp_path, mutation):
    service, reader, arguments = prepare(tmp_path)
    reader.mutation = mutation
    with pytest.raises(ValueError, match="changed"):
        service.observe(**arguments)
    assert reader.calls == 1


def test_same_file_reader_code_replacement_refuses_before_call(tmp_path):
    service, reader, arguments = prepare(tmp_path)
    original = ControlFlowReader.observe_control_flow.__code__
    try:
        ControlFlowReader.observe_control_flow.__code__ = ControlFlowReader.substitute.__code__
        with pytest.raises(ValueError, match="callback identity"):
            service.observe(**arguments)
        assert reader.calls == 0
    finally:
        ControlFlowReader.observe_control_flow.__code__ = original


def test_legacy_reader_cannot_silently_receive_cfg_meanings(tmp_path):
    with pytest.raises(ValueError, match="bound methods"):
        prepare(tmp_path, reader=Reader())


def test_cfg_reader_must_explicitly_declare_its_versioned_contract(tmp_path):
    class WrongVersion(ControlFlowReader):
        def record(self):
            return {"schema": "unknown"}

    with pytest.raises(ValueError, match="versioned"):
        prepare(tmp_path, reader=WrongVersion())


def test_selected_entry_must_exist_in_actual_complete_graph(tmp_path):
    service, reader, arguments = prepare(tmp_path)
    changed = replace(service.control_flow_plan, entry_symbol="absent")
    service = replace(service, control_flow_plan=changed)
    with pytest.raises(DataflowUnavailable, match="entry"):
        service.observe(**{**arguments, "entry_symbol": "absent"})
    assert reader.calls == 0


def test_actual_pointer_roster_remains_the_complete_original_abi(tmp_path):
    service, reader, arguments = prepare(
        tmp_path,
        text=_LOOP.replace(
            "%output: !llvm.ptr",
            "%output: !llvm.ptr, %extra: !llvm.ptr",
        ),
    )
    with pytest.raises(ValueError, match="pointer slots"):
        service.observe(**arguments)
    assert reader.calls == 0


def test_plan_removal_mutation_and_missing_source_membership_refuse(tmp_path):
    service, reader, arguments = prepare(tmp_path)
    with pytest.raises(ValueError, match="membership"):
        replace(
            service,
            source_pins=tuple(
                pin for pin in service.source_pins if pin[0] != str(service.control_flow_plan.layout_source)
            ),
        )
    object.__setattr__(service, "control_flow_plan", None)
    with pytest.raises(ValueError, match="removed"):
        service.observe(**arguments)
    assert reader.calls == 0


def test_existing_product_is_preserved_on_repeat(tmp_path):
    service, reader, arguments = prepare(tmp_path)
    service.observe(**arguments)
    output = tmp_path / "source_observation.json"
    before = output.read_bytes()
    with pytest.raises(ValueError, match="new private product"):
        service.observe(**arguments)
    assert output.read_bytes() == before and reader.calls == 1


def test_fifo_source_is_refused_without_open(tmp_path, monkeypatch):
    service, _, _ = prepare(tmp_path)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("nonregular input was opened"))
    with pytest.raises(ValueError, match="regular"):
        _read(fifo, 100)


@pytest.mark.parametrize(
    "declaration",
    [
        '{"schema":"merlin.source_control_flow_layout_declaration.v1","pointer_bits":64}',
        '{"schema":"merlin.source_control_flow_layout_declaration.v1","pointer_bits":true,"data_layout":"e-p:64:64"}',
        '{"schema":"merlin.source_control_flow_layout_declaration.v1","pointer_bits":64,"data_layout":"e-p:64:64","extra":1}',
        '{"schema":"merlin.source_control_flow_layout_declaration.v1","pointer_bits":64,"data_layout":"e-p:64:64","pointer_bits":64}',
    ],
)
def test_layout_declaration_is_explicit_closed_and_not_scalar_aliases(tmp_path, declaration):
    service, _, _ = prepare(tmp_path)
    layout = service.control_flow_plan.layout_source
    layout.write_text(declaration)
    with pytest.raises(ValueError, match="declaration"):
        replace(service.control_flow_plan, layout_sha256=hashlib.sha256(layout.read_bytes()).hexdigest())
