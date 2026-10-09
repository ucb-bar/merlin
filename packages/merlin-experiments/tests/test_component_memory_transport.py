"""Ordinary source lowering forwards only an explicit source-pinned memory reader.

The downstream build/execution return is a diagnostic fixture, not an ELF or
runtime authority. Real package subprocesses and original full numerical gates
still run, so dropped/reordered outputs and reader selection are observable.
"""

import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase2.component_runtime_control_execution import PrivateRuntimeControlExecutor
from merlin_experiments.phase2.contracts import exact_tree_record

from merlin.targetgen import native_component_execution as native
from merlin.targetgen import package_runtime as P
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.execution_service import FunctionalExecutionService

_spec = importlib.util.spec_from_file_location(
    "private_memory_transport_source_fixture", Path(__file__).with_name("test_component_runtime_support.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
prepared = _fixture.prepared


class SourceOwnedReader:
    def prepare(self, **_kwargs):
        raise AssertionError("diagnostic downstream stub must preserve, not invoke the reader")

    def decode(self, _console):
        raise AssertionError("diagnostic downstream stub must preserve, not invoke the reader")

    def replacement_prepare(self, **_kwargs):
        raise AssertionError("same-source replacement was not selected")


def _arguments(prepared, fixture, output, *, transport):
    return dict(
        package_dir=fixture.grade_arguments["package_dir"],
        capsule_dir=fixture.capsule_root,
        contract_root=prepared.contract_root,
        target=prepared.build_service.target,
        out_dir=output,
        build_service=prepared.build_service,
        execution_service=prepared.execution_service,
        source_verifier=prepared._source_verifier,
        readback_policy=RB.ReadbackPolicy(transport),
        timeout_s=30,
    )


@pytest.mark.parametrize("transport", [*RB.MEMORY_TRANSPORTS, RB.FULL_VALUES_B64])
@pytest.mark.parametrize("defect", [False, True])
def test_real_source_lowering_keeps_original_full_gate_and_selected_reader(
    prepared, tmp_path, monkeypatch, transport, defect
):
    from merlin.runtime.backends import base

    monkeypatch.setattr(base, "get_backend", lambda *_args: pytest.fail("explicit route discovered a backend"))
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "original")
    output = tmp_path / "ordinary"
    args = _arguments(prepared, fixture, output, transport=transport)
    reader = SourceOwnedReader()
    if transport in RB.MEMORY_TRANSPORTS:
        args["memory_readback"] = reader
        args["execution_service"] = replace(
            prepared.execution_service,
            source_pins=(
                *prepared.execution_service.source_pins,
                (str(Path(__file__).resolve()), file_digest(Path(__file__))),
            ),
        )
    calls = []

    def downstream(bound, artifact, **kwargs):
        calls.append(kwargs)
        assert "llvm.func" in artifact and bound["kernel_abi"]["outputs"] == ["out0", "out1", "out2"]
        assert kwargs["readback_policy"].transport == transport
        assert kwargs["execution_deadline"].remaining() <= 30
        if transport in RB.MEMORY_TRANSPORTS:
            assert kwargs["memory_readback"] is reader
            assert kwargs["oracle_revalidate"]() == args["execution_service"].verify(
                prepared.build_service.target, prepared.execution_service.simulator
            )
        else:
            assert "memory_readback" not in kwargs and "oracle_revalidate" not in kwargs
        work = kwargs["workdir"]
        work.mkdir()
        elf = work / "diagnostic-fixture.elf"
        elf.write_bytes(b"fixture: no linked artifact or physical authority")
        (work / "oracle_console.log").write_text("DONE\n")
        actual = {
            "out0": [[3.25, -3.0, -0.125]],
            "out1": [[2.0, 0.5, -0.25]],
            "out2": [[1.25, -3.5, 0.125]],
        }
        if defect:
            actual["out2"][0][2] = 0.0
        return {"elf": str(elf), "outputs": actual, "console": "DONE\n"}

    monkeypatch.setattr(compiler, "run_on_oracle", downstream)
    candidate = args["package_dir"]
    executor = PrivateRuntimeControlExecutor(candidate, output, exact_tree_record(candidate)["sha256"])
    with P.scoped_package_executor(executor):
        result = native.execute_component(**args)
    assert len(calls) == 1
    assert result["numeric_report"]["status"] == ("fail" if defect else "pass")
    assert set(result["emission"]) >= {"source_interface", "lowered_mlir", "command_buffer", "input_projection"}
    assert result["status"] == ("numeric_mismatch_diagnostic" if defect else "numeric_match_diagnostic")
    assert "unqualified" in result["scope"]
    original = json.loads((fixture.evidence_root / "original_policy.json").read_bytes())
    assert original == {"compare": "tolerance_float", "dtype": "f32", "atol": 0.0, "rtol": 0.0}
    if transport in RB.MEMORY_TRANSPORTS:
        assert result["memory_reader_source_pins"] == [(str(Path(__file__).resolve()), file_digest(Path(__file__)))]
    else:
        assert "memory_reader_source_pins" not in result


@pytest.mark.parametrize("defect", ["missing_reader", "unpinned_reader", "serial_reader", "native_method"])
def test_missing_or_unselected_reader_refuses_before_any_package_work(prepared, tmp_path, defect):
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "original")
    output = tmp_path / "refused"
    args = _arguments(prepared, fixture, output, transport=RB.COHERENT_DUMP_V1)
    args["memory_readback"] = None if defect == "missing_reader" else SourceOwnedReader()
    if defect == "serial_reader":
        args["readback_policy"] = RB.ReadbackPolicy(RB.FULL_VALUES_B64)
    elif defect == "native_method":
        args["memory_readback"] = type("NativeOnlyReader", (), {"prepare": len, "decode": len})()
    with pytest.raises(native.NativeComponentExecutionError, match="memory"):
        native.execute_component(**args)
    assert not output.exists()


@pytest.mark.parametrize("boundary", ["lowering", "execution"])
def test_same_source_callback_replacement_cannot_retain_a_numeric_result(prepared, tmp_path, monkeypatch, boundary):
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "original")
    output = tmp_path / "ordinary"
    args = _arguments(prepared, fixture, output, transport=RB.COHERENT_DUMP_V1)
    reader = SourceOwnedReader()
    args["memory_readback"] = reader
    args["execution_service"] = replace(
        prepared.execution_service,
        source_pins=(
            *prepared.execution_service.source_pins,
            (str(Path(__file__).resolve()), file_digest(Path(__file__))),
        ),
    )
    if boundary == "lowering":
        original = native.CC.lower_interface

        def changed_after_real_lowering(*values, **kwargs):
            result = original(*values, **kwargs)
            reader.prepare = reader.replacement_prepare
            return result

        monkeypatch.setattr(native.CC, "lower_interface", changed_after_real_lowering)
    else:

        def changed_during_execution(*_values, **kwargs):
            reader.prepare = reader.replacement_prepare
            kwargs["oracle_revalidate"]()
            raise AssertionError("changed reader must refuse before returning an output")

        monkeypatch.setattr(compiler, "run_on_oracle", changed_during_execution)
    candidate = args["package_dir"]
    executor = PrivateRuntimeControlExecutor(candidate, output, exact_tree_record(candidate)["sha256"])
    with (
        P.scoped_package_executor(executor),
        pytest.raises(native.NativeComponentExecutionError, match="callback changed"),
    ):
        native.execute_component(**args)
    record = json.loads((output / "result.json").read_bytes())
    assert record["status"] == "unavailable" and "numeric_report" not in record
    assert record["failure"]["type"] == "NativeComponentExecutionError"


class DiagnosticMemoryReader:
    """Stateful gate fixture; its bytes are not a linked ELF/runtime proof."""

    def __init__(self, fault, engine):
        self.fault, self.engine = fault, engine
        self.calls = []
        self.outputs = {
            "out0": [[3.25, -3.0, -0.125]],
            "out1": [[2.0, 0.5, -0.25]],
            "out2": [[1.25, -3.5, 0.125]],
        }

    def prepare(self, *, elf_path, **kwargs):
        self.calls.append("prepare")
        self.prepared = kwargs
        if self.fault == "changed_elf":
            elf_path.write_bytes(b"changed diagnostic artifact")
        return {"memory_readback": {"elf_sha256": file_digest(elf_path)}}

    def run(self, _elf, **kwargs):
        self.calls.append("run")
        assert set(kwargs) == {"simulator", "timeout", "memory_readback"}
        if self.fault == "changed_engine":
            self.engine.write_bytes(b"changed selected execution source")
        if self.fault == "mixed_serial":
            return "OUT out0 1 3 3.25 -3.0 -0.125\nDONE\n"
        if self.fault == "missing_done":
            return "diagnostic still running\n"
        if self.fault == "duplicate_done":
            return "DONE\nDONE\n"
        return "DONE\n"

    def parse(self, console):
        return ({"out0": self.outputs["out0"]} if console.startswith("OUT ") else {}), {}

    def decode(self, console):
        self.calls.append("decode")
        assert console == "DONE\n"
        outputs = deepcopy(self.outputs)
        if self.fault == "missing_output":
            del outputs["out2"]
        elif self.fault == "partial_values":
            outputs["out2"][0].pop()
        elif self.fault == "wrong_value":
            outputs["out2"][0][2] = 0.0
        return outputs, {"status": "unproved" if self.fault == "unproved" else "complete"}


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "changed_elf",
        "changed_engine",
        "missing_output",
        "partial_values",
        "unproved",
        "mixed_serial",
        "missing_done",
        "duplicate_done",
        "wrong_value",
    ],
)
def test_real_ordinary_oracle_gates_preserve_full_roster_and_selected_transport(prepared, tmp_path, monkeypatch, fault):
    """Actual lowering and run_on_oracle gates; only link/engine are diagnostic fixtures."""
    from merlin.runtime.backends import base

    monkeypatch.setattr(base, "get_backend", lambda *_args: pytest.fail("explicit route discovered a backend"))
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "original")
    output = tmp_path / "ordinary"
    args = _arguments(prepared, fixture, output, transport=RB.COHERENT_DUMP_V1)
    engine = tmp_path / "selected-diagnostic-engine"
    engine.write_bytes(b"diagnostic execution source; no hardware authority")
    reader = DiagnosticMemoryReader(fault, engine)
    owner = Path(__file__).resolve()
    args["memory_readback"] = reader
    args["execution_service"] = FunctionalExecutionService(
        prepared.build_service.target,
        "private_memory_gate",
        reader.run,
        reader.parse,
        ((str(owner), file_digest(owner)), (str(engine), file_digest(engine))),
        '{"scope":"ordinary memory gate diagnostic, not runtime proof"}',
    )
    selected_elf = {}

    def link_fixture(_cb, text, work, **kwargs):
        assert "llvm.func" in text and kwargs["_build_service"] is prepared.build_service
        work.mkdir()
        elf = work / "diagnostic-fixture.elf"
        elf.write_bytes(b"diagnostic link fixture; no ELF or ISA authority")
        selected_elf["sha256"] = file_digest(elf)
        (work / RB.BUILD_RECEIPT).write_text(
            json.dumps({"scope": "diagnostic build fixture; no ELF or ISA authority", **selected_elf}) + "\n"
        )
        return elf

    def reopen_build_fixture(*, elf_path, **_kwargs):
        if file_digest(elf_path) != selected_elf["sha256"]:
            raise ValueError("diagnostic completed build identity changed")
        return {"scope": "diagnostic build fixture", "elf_sha256": selected_elf["sha256"]}

    monkeypatch.setattr(compiler, "compile_lowered_to_elf", link_fixture)
    monkeypatch.setattr(RB, "require_current_build_receipt", reopen_build_fixture)
    candidate = args["package_dir"]
    executor = PrivateRuntimeControlExecutor(candidate, output, exact_tree_record(candidate)["sha256"])
    with P.scoped_package_executor(executor):
        if fault in {None, "wrong_value"}:
            result = native.execute_component(**args)
            assert reader.calls == ["prepare", "run", "decode"]
            assert reader.prepared["cb"]["kernel_abi"]["outputs"] == ["out0", "out1", "out2"]
            assert result["numeric_report"]["status"] == ("pass" if fault is None else "fail")
            assert "unqualified" in result["scope"]
            return
        with pytest.raises((ValueError, native.NativeComponentExecutionError)):
            native.execute_component(**args)
    record = json.loads((output / "result.json").read_bytes())
    assert record["status"] == "unavailable" and "numeric_report" not in record
    if fault == "changed_elf":
        assert reader.calls == ["prepare"]
    elif fault in {"changed_engine", "mixed_serial", "missing_done", "duplicate_done"}:
        assert reader.calls == ["prepare", "run"]
    else:
        assert reader.calls == ["prepare", "run", "decode"]
