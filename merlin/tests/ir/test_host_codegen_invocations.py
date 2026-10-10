"""Diagnostic wiring controls; fake process products grant no native authority."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.llvmlower import codegen


def selected(tmp_path, monkeypatch, *, llc):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("cc", "clang", "llc"):
        tool = tools / name
        tool.write_bytes(b"diagnostic nonexecuted selected tool\n")
        tool.chmod(0o700)
    environment = {"PATH": str(tools), "LANG": "C.UTF-8"}
    monkeypatch.setattr(os, "environ", environment)
    source, runtime = tmp_path / "source.ll", tmp_path / "runtime.c"
    source.write_text("diagnostic LLVM source\n")
    runtime.write_text("diagnostic runtime source\n")
    monkeypatch.setattr(codegen, "clang", lambda: str(tools / "clang"))
    monkeypatch.setattr(codegen, "host_llc", lambda: str(tools / "llc") if llc else None)
    monkeypatch.setattr(codegen, "mlir_runtime_c", lambda: runtime)
    calls = []

    def fake_process(argv, **kwargs):
        argv = [str(token) for token in argv]
        calls.append((argv, kwargs))
        Path(argv[-1]).write_bytes(("diagnostic product: " + " ".join(argv)).encode())
        return subprocess.CompletedProcess(argv, 0, stdout="owned output", stderr="owned diagnostic")

    monkeypatch.setattr(codegen._proc, "run_checked", fake_process)
    return source, runtime, environment, calls


@pytest.mark.parametrize("llc", (False, True))
@pytest.mark.parametrize("explicit_cc", (False, True))
def test_exact_original_three_child_boundaries_and_input_product_roles(tmp_path, monkeypatch, llc, explicit_cc):
    source, runtime, environment, calls = selected(tmp_path, monkeypatch, llc=llc)
    selected_cc = str(tmp_path / "tools/clang") if explicit_cc else "cc"
    if explicit_cc:
        monkeypatch.setenv("CC", selected_cc)
    result = codegen.build_host_shared(source, tmp_path / "model.so")
    records = [I.verify(path) for path in tmp_path.rglob("invocation.json")]
    assert len(records) == 3
    by_stage = {row["stage"]: row for row in records}
    assert set(by_stage) == {"object", "runtime_object", "link"}
    model, runtime_object = tmp_path / "model.o", tmp_path / "mlir_runtime_host.o"
    roles = {
        "object": ((source,), (model,)),
        "runtime_object": ((runtime,), (runtime_object,)),
        "link": ((model, runtime_object), (result,)),
    }
    for stage, (inputs, outputs) in roles.items():
        record = by_stage[stage]
        assert {pin["path"] for pin in record["inputs"]} == {str(path) for path in inputs}
        assert {pin["path"] for pin in record["outputs"]} == {str(path) for path in outputs}
        I.require_environment(Path(record["stdout"]["path"]).parent / "invocation.json", environment=environment)
        actual = next(call for call in calls if call[0] == record["argv"])
        assert actual[1]["env"] == environment and str(actual[1]["cwd"]) == record["cwd"]
    assert calls[1][0] == [selected_cc, "-O2", "-fPIC", "-c", str(runtime), "-o", str(runtime_object)]
    assert calls[2][0] == [selected_cc, "-shared", str(model), str(runtime_object), "-lm", "-o", str(result)]


@pytest.mark.parametrize("drift", ("environment", "cwd", "argv"))
def test_observer_boundary_cannot_change_actual_selected_command_context(tmp_path, monkeypatch, drift):
    _, _, environment, calls = selected(tmp_path, monkeypatch, llc=True)
    before = dict(environment)
    cwd = Path.cwd()
    destination = tmp_path / "result.o"
    argv = [str(tmp_path / "tools/llc"), "-o", str(destination)]
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    original = I.Invocation.__init__

    def observed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if drift == "environment":
            environment["OWNED_DRIFT"] = "changed after selection"
        elif drift == "cwd":
            monkeypatch.chdir(alternate)
        else:
            argv[0] = str(tmp_path / "tools/clang")

    monkeypatch.setattr(I.Invocation, "__init__", observed)
    codegen._run(argv, outputs=(destination,))
    path = next(tmp_path.rglob("invocation.json"))
    document = I.verify(path)
    assert calls[0][0] == document["argv"]
    assert str(calls[0][1].get("cwd", Path.cwd())) == str(cwd) == document["cwd"]
    actual_environment = calls[0][1].get("env", environment)
    assert actual_environment == before
    I.require_environment(path, environment=actual_environment)


@pytest.mark.parametrize("defect", ("changed_runtime", "missing_runtime_object", "changed_model", "missing_link"))
def test_actual_record_replay_refuses_changed_or_missing_original_intermediate(tmp_path, monkeypatch, defect):
    source, runtime, _, _ = selected(tmp_path, monkeypatch, llc=True)
    result = codegen.build_host_shared(source, tmp_path / "model.so")
    paths = {
        "changed_runtime": runtime,
        "missing_runtime_object": tmp_path / "mlir_runtime_host.o",
        "changed_model": tmp_path / "model.o",
        "missing_link": result,
    }
    path = paths[defect]
    if defect.startswith("changed"):
        path.write_bytes(path.read_bytes() + b"changed original\n")
    else:
        path.unlink()
    stage = {
        "changed_runtime": "runtime_object",
        "missing_runtime_object": "link",
        "changed_model": "link",
        "missing_link": "link",
    }[defect]
    record = next(p for p in tmp_path.rglob("invocation.json") if json.loads(p.read_text())["stage"] == stage)
    with pytest.raises(ValueError):
        I.verify(record)


@pytest.mark.parametrize("error", (codegen.CodegenError("owned nonzero"), subprocess.TimeoutExpired("owned", 1)))
@pytest.mark.parametrize("boundary", (0, 1, 2), ids=("object", "runtime_object", "link"))
def test_existing_exception_does_not_become_a_completed_child(tmp_path, monkeypatch, error, boundary):
    source, _, _, calls = selected(tmp_path, monkeypatch, llc=True)
    original = codegen._proc.run_checked

    def failed(*args, **kwargs):
        if len(calls) == boundary:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(codegen._proc, "run_checked", failed)
    with pytest.raises(type(error)) as observed:
        codegen.build_host_shared(source, tmp_path / "model.so")
    assert observed.value is error
    records = list(tmp_path.rglob("invocation.json"))
    assert len(records) == boundary + 1 and len(calls) == boundary
    stage = ("object", "runtime_object", "link")[boundary]
    record = next(path for path in records if json.loads(path.read_text())["stage"] == stage)
    assert json.loads(record.read_text())["status"] == "interrupted"
    with pytest.raises(ValueError):
        I.verify(record)
    for completed in set(records) - {record}:
        I.verify(completed)
