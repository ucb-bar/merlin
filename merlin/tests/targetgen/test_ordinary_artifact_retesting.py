"""Harmless ordinary process dispatch with synthetic build artifacts only.

No engine, compiler, timing or physical support is qualified by these controls.
The missing normal runtime closure must execute, including unchanged artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common import invocation_record as I
from merlin.runtime.backends import base
from merlin.runtime.backends.base import parse_console
from merlin.targetgen.contract import artifact_retesting as A
from merlin.targetgen.contract import compile as C
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_service import file_digest


class _SyntheticOrdinaryBackend:
    ORACLE = {"gsim": {"kind": "synthetic_process_control_only"}}

    def __init__(self):
        self.executions = 0

    def run_elf(self, elf, *, simulator, timeout):
        assert simulator == "gsim"
        self.executions += 1
        return subprocess.run(
            [sys.executable, "-I", "-B", str(elf)], capture_output=True, text=True, check=True, timeout=timeout
        ).stdout

    def parse_output(self, console):
        return parse_console(console)


@pytest.fixture
def ordinary(tmp_path, monkeypatch):
    backend = _SyntheticOrdinaryBackend()
    state = SimpleNamespace(builds=0, elf_revision="", harness="harness", object="object", recipe="original")
    cb = {"tensors": {"out": {"shape": [1, 2], "dtype": "i8", "role": "output"}}}
    console = "OUT_B64_BEGIN v1 out 1 2 1 s\nOUT_B64_CHUNK 00000000 0002 AQI=\nOUT_B64_END\nDONE\n"

    def build(_cb, _lowered, work, **kwargs):
        state.builds += 1
        work.mkdir()
        elf = work / "complete-harness.py"
        elf.write_text("print(" + repr(console) + ", end='')\n# " + state.elf_revision)
        (work / "harness.c").write_text(state.harness)
        (work / "kernel.o").write_text(state.object)
        receipt = {
            "target": "synthetic", "readback_policy": kwargs["readback_policy"].record(),
            "elf_sha256": file_digest(elf), "harness_sha256": file_digest(work / "harness.c"),
            "kernel_object_sha256": file_digest(work / "kernel.o"), "recipe": state.recipe,
        }
        (work / "readback_build.json").write_text(json.dumps(receipt))
        return elf

    def receipt(**kwargs):
        work = kwargs["workdir"]
        result = json.loads((work / "readback_build.json").read_text())
        assert result["elf_sha256"] == file_digest(kwargs["elf_path"])
        assert result["harness_sha256"] == file_digest(work / "harness.c")
        assert result["kernel_object_sha256"] == file_digest(work / "kernel.o")
        return result

    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    monkeypatch.setattr(C, "compile_lowered_to_elf", build)
    monkeypatch.setattr(C, "simulator_provenance", lambda *args: {"scope": "synthetic only"})
    monkeypatch.setattr(C, "_counter_observations", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(RB, "require_current_build_receipt", receipt)

    def invoke(**kwargs):
        return C.run_on_oracle(
            cb, kwargs.pop("lowered", "lowered"), simulator="gsim", target="synthetic",
            workdir=tmp_path / f"build-{state.builds}", timeout=5,
            readback_policy=RB.ReadbackPolicy(RB.FULL_VALUES_B64),
            inputs=kwargs.pop("inputs", {"input": [1, 2]}), **kwargs,
        )

    return SimpleNamespace(**locals())


def test_unchanged_fresh_artifacts_still_execute_without_frozen_native_runtime(ordinary):
    f = ordinary
    with A.observe_public_artifact_retests(f.tmp_path, context={"normal_selection": "synthetic"}):
        first, second = f.invoke(), f.invoke()
    assert f.state.builds == f.backend.executions == 2
    report = second["artifact_retesting"]
    assert first["artifact_retesting"]["comparison"] == "first_observation"
    assert report["comparison"] == "unchanged_artifacts"
    assert report["execution_reused"] is False and report["decision"] == "execute"
    assert report["runtime_dependency_closure"]["status"] == "UNKNOWN"
    assert "native simulator" in report["runtime_dependency_closure"]["reason"]
    assert first["outputs"] == second["outputs"] == {"out": [[1, 2]]}
    records = [I.verify(path) for path in f.tmp_path.rglob("invocation.json")]
    assert len(records) == 2 and all(row["kind"] == "python_call" for row in records)
    assert json.loads((Path(second["elf"]).parent / "artifact_retesting.json").read_text()) == report


@pytest.mark.parametrize("drift", ["elf_revision", "harness", "object", "recipe", "input", "lowered", "environment"])
def test_changed_whole_build_or_numeric_context_executes(ordinary, monkeypatch, drift):
    f, kwargs = ordinary, {}
    with A.observe_public_artifact_retests(f.tmp_path, context={"normal_selection": "synthetic"}):
        f.invoke()
        if drift == "input":
            kwargs["inputs"] = {"input": [2, 3]}
        elif drift == "lowered":
            kwargs["lowered"] = "different lowered program"
        elif drift == "environment":
            monkeypatch.setenv("SYNTHETIC_RUNTIME_CONTROL", "changed")
        else:
            setattr(f.state, drift, "changed")
        result = f.invoke(**kwargs)
    assert f.state.builds == f.backend.executions == 2
    assert result["artifact_retesting"]["comparison"] == "changed_artifacts"
    assert result["artifact_retesting"]["execution_reused"] is False


@pytest.mark.parametrize("damage", ["stdout", "original_elf", "original_harness"])
def test_damaged_original_witness_or_artifact_is_uncertain_and_executes(ordinary, damage):
    f = ordinary
    with A.observe_public_artifact_retests(f.tmp_path, context={}):
        first = f.invoke()
        path = Path(first["elf"])
        if damage == "stdout":
            original = I.verify(next(f.tmp_path.rglob("invocation.json")))
            path = Path(original["stdout"]["path"])
        elif damage == "original_harness":
            path = path.parent / "harness.c"
        path.write_text("damaged old observation")
        second = f.invoke()
    assert f.backend.executions == 2
    assert second["artifact_retesting"]["comparison"] == "uncertain_previous_observation"


def test_unselected_final_has_no_retesting_archive(ordinary):
    f = ordinary
    with A.observe_public_artifact_retests(f.tmp_path, context={}):
        f.invoke()
    final = f.invoke()
    assert f.backend.executions == f.state.builds == 2
    assert "artifact_retesting" not in final
    assert A.active_artifact_retests() is None


def test_unknown_complete_harness_executes_without_comparison(ordinary, monkeypatch):
    f = ordinary
    # Missing comparison evidence is an ordinary miss, not a new gate.
    def unavailable(**kwargs):
        return {"unavailable": True}
    monkeypatch.setattr(RB, "require_current_build_receipt", unavailable)
    with A.observe_public_artifact_retests(f.tmp_path, context={}):
        result = f.invoke()
    assert f.backend.executions == 1
    assert result["artifact_retesting"]["comparison"] == "uncertain_artifact_observation"
