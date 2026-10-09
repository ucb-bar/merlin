"""Prepared diagnostic observations retain original gates; no role is issued."""

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase2 import component_runtime_support as support
from merlin_experiments.phase2.contracts import StageGateError
from test_component_runtime_support import prepared as prepared

from merlin.common import invocation_record as I
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.source_observation import ExplicitSourceObservation

_SOURCE = """module { func.func @main(%a: tensor<1x1xi8>) -> tensor<1x1xi8> {
%d = tensor.empty() : tensor<1x1xi8>
%copy = linalg.copy ins(%a : tensor<1x1xi8>) outs(%d : tensor<1x1xi8>) -> tensor<1x1xi8>
func.return %copy : tensor<1x1xi8> } }"""
_CB = {
    "abi_version": "0.1",
    "operand_naming": "positional",
    "tensors": {
        "arg0": {"shape": [1, 1], "dtype": "i8", "role": "input"},
        "out0": {"shape": [1, 1], "dtype": "i8", "role": "output"},
    },
    "kernel_abi": {
        "kind": "whole_program",
        "args": [{"tensor": "arg0", "access": "read"}, {"tensor": "out0", "access": "write"}],
        "outputs": ["out0"],
    },
}


class DiagnosticReader:
    """Synthetic attribution only; deliberately returns a false role label."""

    def verify(self):
        pass

    def record(self):
        return {"scope": "synthetic source observation, never semantic proof"}

    def observe(self, **kwargs):
        return {"actions": [row.kind for row in kwargs["dataflow"].actions], "stage": "PASS"}


def selected(prepared, tmp_path):
    source_dir = tmp_path / "original"
    source_dir.mkdir()
    source = source_dir / "source.mlir"
    source.write_text(_SOURCE)
    (source_dir / "capsule.yaml").write_text('{"numeric_policy":{"compare":"exact_int","dtype":"i8"}}')
    owner = Path(inspect.getsourcefile(DiagnosticReader))
    abi = CompileOnlySourceAbi((CompileOnlyTensor("a", (1, 1), "i8"),), (CompileOnlyTensor("result", (1, 1), "i8"),))
    service = ExplicitSourceObservation(
        "private_control",
        source,
        abi,
        64,
        100,
        DiagnosticReader(),
        tuple((str(p), file_digest(p)) for p in (source, owner)),
    )
    context = replace(
        prepared,
        source_observation=service,
        source_pins=(*prepared.source_pins, *((Path(p), sha) for p, sha in service.source_pins)),
    )
    emitted = tmp_path / "emitted"
    emitted.mkdir()
    lowered = emitted / "lowered.llvm.mlir"
    lowered.write_text(
        "module { llvm.func @control_entry(%input: !llvm.ptr, %output: !llvm.ptr) {"
        "%v = llvm.load %input : !llvm.ptr -> i8 llvm.store %v, %output : i8, !llvm.ptr llvm.return } }"
    )
    (emitted / "command_buffer.bound.json").write_text(json.dumps(_CB))
    return context, source, lowered


def test_selected_actual_source_observation_does_not_become_source_or_stage_proof(prepared, tmp_path):
    context, source, lowered = selected(prepared, tmp_path)
    proof = context._source_verifier(source=source, command_buffer=_CB, lowered_mlir=lowered)
    assert proof["observations"]["stage"] == "PASS" and "compiler_stages" in proof["unknown"]
    persisted = json.loads((lowered.parent / "source_correspondence.json").read_text())
    assert persisted["status"] == "diagnostic_observed"
    assert persisted["proof"]["observations"]["actions"] == ["load", "store", "return"]
    records = [I.verify(path) for path in lowered.parent.rglob("invocation.json")]
    assert {row["stage"] for row in records} == {
        "primitive_source_verification",
        "explicit_original_source_dataflow_observation",
    }
    result = tmp_path / "capsule_result.json"
    result.write_text('{"status":"pass","numeric":"pass"}')
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        context.stage_verifier(result_path=result)


@pytest.mark.parametrize("case", ["source_correspondence.negative", "original_numeric_gate.negative"])
def test_selected_observer_cannot_replace_original_fixed_qualifier_controls(prepared, tmp_path, case):
    context, _, _ = selected(prepared, tmp_path)
    fixture = context.prepare_control(case, tmp_path / "control")
    source = fixture.capsule_root / "source.mlir"
    program = support.controls.parse_primitive(source.read_text())
    lowered = tmp_path / "actual-control.mlir"
    # The source-correspondence defect is genuinely wrong emitted semantics;
    # the independent fixed check must still reject it before any reader.
    if case.startswith("source"):
        program = replace(program, outputs=(("input", 0), *program.outputs[1:]))
    lowered.write_text(support.controls.emit_primitive_llvm(program, entry_symbol="control_entry"))
    result = context._evaluate_source(source, lowered, fixture)
    assert result["status"] == "refused"
    assert not (lowered.parent / "source_observation.json").exists()


def test_missing_selection_membership_and_helper_substitution_refuse(prepared, tmp_path):
    context, _, _ = selected(prepared, tmp_path)
    with pytest.raises(StageGateError, match="reader/source membership"):
        support.source_selection.selection(replace(context, source_pins=prepared.source_pins))
    with pytest.raises(StageGateError, match="copy-helper"):
        support.source_selection.selection(replace(context, copy_control_support=object()))


def test_foreign_transport_cannot_execute_a_supplied_factory(prepared):
    class Foreign:
        def verify(self):
            raise AssertionError("foreign callback was invoked")

    with pytest.raises(StageGateError, match="exact explicitly selected"):
        support.source_selection.selection(replace(prepared, source_observation=Foreign()))
