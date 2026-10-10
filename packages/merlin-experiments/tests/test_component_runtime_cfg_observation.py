"""Actual grade/source callback chain, with native work explicitly unattempted."""

import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_runtime_support as support
from merlin_experiments.phase2.contracts import StageGateError
from test_component_runtime_source_selection import _CB, selected
from test_component_runtime_support import prepared as prepared

from merlin.common import invocation_record as I
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.source_control_flow import (
    LAYOUT_SCHEMA,
    READER_SCHEMA,
    ControlFlowObservationPlan,
)
from merlin.targetgen.contract.source_observation import ExplicitSourceObservation

_CFG = """module { llvm.func @control_entry(%input: !llvm.ptr, %output: !llvm.ptr) {
  %zero = llvm.mlir.constant(0 : i64) : i64
  %one = llvm.mlir.constant(1 : i64) : i64
  llvm.br ^loop(%zero : i64)
^loop(%index: i64):
  %condition = llvm.icmp "ult" %index, %one : i64
  llvm.cond_br %condition, ^body, ^done
^body:
  %value = llvm.load %input : !llvm.ptr -> i8
  llvm.store %value, %output : i8, !llvm.ptr
  %next = llvm.add %index, %one : i64
  llvm.br ^loop(%next : i64)
^done:
  llvm.return } }"""


class DiagnosticControlFlowReader:
    """Actual structural callback only, never an independent semantic owner."""

    def verify(self):
        pass

    def record(self):
        return {"schema": READER_SCHEMA, "scope": "synthetic complete CFG attribution only"}

    def observe_control_flow(self, **kwargs):
        assert "dataflow" not in kwargs
        return {"blocks": len(kwargs["control_flow"].blocks), "false_role_label": "PASS"}


def selected_cfg(prepared, tmp_path):
    context, source, lowered = selected(prepared, tmp_path)
    lowered.write_text(_CFG)
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"schema": LAYOUT_SCHEMA, "data_layout": "e-p:64:64", "pointer_bits": 64}))
    plan = ControlFlowObservationPlan(
        lowered,
        file_digest(lowered),
        "control_entry",
        layout,
        file_digest(layout),
        64,
        10000,
        64,
        16,
        100,
        100,
        32,
        64,
        1000,
    )
    owner = Path(inspect.getsourcefile(DiagnosticControlFlowReader))
    pins = tuple((str(path), file_digest(path)) for path in (source, lowered, layout, owner))
    service = ExplicitSourceObservation(
        "private_control",
        source,
        context.source_observation.original_abi,
        64,
        100,
        DiagnosticControlFlowReader(),
        pins,
        control_flow_plan=plan,
    )
    context = replace(
        context,
        source_observation=service,
        source_pins=(*context.source_pins, *((Path(path), digest) for path, digest in pins)),
    )
    return context, source, lowered


def test_actual_grade_invokes_selected_complete_cfg_callback_and_retains_unknowns(prepared, tmp_path, monkeypatch):
    context, source, lowered = selected_cfg(prepared, tmp_path)
    capsule = source.parent
    (capsule / "capsule.yaml").write_text('{"name":"original","numeric_policy":{"compare":"exact_int","dtype":"i8"}}')
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "manifest.yaml").write_text("name: synthetic-source-only\n")
    monkeypatch.setattr(
        support.package_runtime, "active_package_executor", lambda: SimpleNamespace(container_transport=None)
    )
    calls = []

    def diagnostic_compile(**kwargs):
        # Substitute build/execution work, while exercising the actual ordinary
        # grade -> source_verifier -> source selection -> CFG reader chain.
        assert kwargs["package_dir"] == package and kwargs["capsule_dir"] == capsule
        proof = kwargs["source_verifier"](source=source, command_buffer=_CB, lowered_mlir=lowered)
        calls.append(proof)
        raise StageGateError("pure diagnostic: native execution not attempted")

    monkeypatch.setattr(support, "execute_component", diagnostic_compile)
    with pytest.raises(StageGateError, match="native execution not attempted"):
        context.grade(
            package,
            capsules_root=[capsule],
            runs_root=tmp_path / "runs",
            contract=context.contract_root,
            target=context.build_service.target,
            timeout=30,
        )
    assert len(calls) == 1 and len(calls[0]["control_flow"]["blocks"]) == 4
    assert {"source_equivalence", "instruction_effects", "same_object_data_layout", "runtime"} <= set(
        calls[0]["unknown"]
    )
    persisted = json.loads((lowered.parent / "source_correspondence.json").read_text())
    assert persisted["status"] == "diagnostic_observed"
    assert persisted["proof"]["control_flow"] == json.loads(json.dumps(calls[0]["control_flow"]))
    records = [I.verify(path) for path in lowered.parent.rglob("invocation.json")]
    assert {row["stage"] for row in records} == {
        "primitive_source_verification",
        "explicit_original_source_cfg_observation",
    }
    assert not (tmp_path / "runs" / "original" / "result.json").exists()
    result = tmp_path / "capsule_result.json"
    result.write_text('{"status":"pass","numeric":"pass"}')
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        context.stage_verifier(result_path=result)


def test_cfg_selection_requires_complete_context_source_membership(prepared, tmp_path):
    context, _, _ = selected_cfg(prepared, tmp_path)
    with pytest.raises(StageGateError, match="membership"):
        support.source_selection.selection(replace(context, source_pins=prepared.source_pins))


def test_actual_source_verifier_refuses_stale_selected_lowered_bytes(prepared, tmp_path):
    context, source, lowered = selected_cfg(prepared, tmp_path)
    lowered.write_text(_CFG + "\n")
    with pytest.raises(StageGateError, match="selected|changed"):
        context._source_verifier(source=source, command_buffer=_CB, lowered_mlir=lowered)
    assert not (lowered.parent / "source_observation.json").exists()


def test_legacy_context_does_not_implicitly_select_cfg_reader(prepared, tmp_path):
    context, source, lowered = selected(prepared, tmp_path)
    lowered.write_text(_CFG)
    with pytest.raises(StageGateError, match="control-flow join"):
        context._source_verifier(source=source, command_buffer=_CB, lowered_mlir=lowered)
    assert not (lowered.parent / "source_observation.json").exists()
