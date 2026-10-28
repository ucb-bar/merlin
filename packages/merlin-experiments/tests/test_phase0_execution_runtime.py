"""Frozen execution paths reach the child and remain byte checked on resume."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from merlin_experiments.measured_launch import execution_environment
from merlin_experiments.phase0 import execution_runtime, freeze


@pytest.fixture
def selected(tmp_path, monkeypatch):
    for name in (*execution_runtime.FILES, *execution_runtime.ROOTS):
        monkeypatch.delenv(name, raising=False)
    from merlin.targetgen.rtl import semantic_facts

    monkeypatch.setattr(semantic_facts, "load_probes", lambda target: {})
    tools = {}
    for name in execution_runtime.FILES[:4]:
        tool = tmp_path / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
        monkeypatch.setenv(name, str(tool))
        tools[name] = tool
    return tools


def test_stage_launch_and_resume_keep_execution_selection(tmp_path, monkeypatch, selected):
    # Exercise stage's selection/environment wiring without copying a checkout
    # or launching a compiler, capture, simulator or generator.
    evidence = SimpleNamespace(source_paths=(), source_snapshots=(), views_json=b"{}", software_spec={}, contract={})
    monkeypatch.setattr(freeze, "selection", lambda *args: evidence)
    monkeypatch.setattr(freeze, "export_evidence", lambda *args: {"sources": []})
    monkeypatch.setattr(freeze, "_source_layout", lambda: (tmp_path, (), ()))
    monkeypatch.setattr(freeze, "module_source_path", lambda name: tmp_path / "core/__init__.py")
    monkeypatch.setattr(freeze.source_snapshot, "selected_provider", lambda target: {})
    monkeypatch.setattr(freeze.source_snapshot, "create", lambda *args, **kwargs: None)
    monkeypatch.setattr(freeze.source_snapshot, "verify", lambda *args: {})
    monkeypatch.setattr(freeze.source_snapshot, "remap_input", lambda snapshot, receipt, path, **kwargs: path)
    monkeypatch.setattr(freeze.source_snapshot, "provider_environment", lambda *args: {})
    monkeypatch.setattr(freeze.frozen_python, "python_command", lambda *args, **kwargs: ["frozen"])
    import merlin_experiments.runner as runner

    monkeypatch.setattr(runner, "_phase0_operator_inputs", lambda command: {})
    entrypoint = tmp_path / "entrypoint.py"
    entrypoint.write_text("pass\n")
    software = tmp_path / "software.yaml"
    software.write_text("{}\n")
    paths = {"phase0:entrypoint": str(entrypoint), "phase0:software_spec": str(software)}
    plan = {
        "target": "fixture",
        "run_dir": str(tmp_path / "run"),
        "storage_root": str(tmp_path / "out"),
        "inputs": {name: {"path": path, "sha256": freeze.fingerprint(path)} for name, path in paths.items()},
        "input_paths": paths,
        "phases": {
            "0": {
                "adapter": "capsule_derivation",
                "phase0_evidence": {"status": "diagnostic"},
                "inputs": {"software_spec": str(software), "descriptor": str(software)},
                "entrypoint": str(entrypoint),
                "argv": [sys.executable, str(entrypoint)],
                "env": {},
            }
        },
    }
    staged = freeze.stage(plan)
    command = staged["phases"]["0"]
    selection = command["phase0_execution_selection"]
    assert (
        selection["MERLIN_RISCV_GCC"]["members"][0]["digest"]
        == hashlib.sha256(selected["MERLIN_RISCV_GCC"].read_bytes()).hexdigest()
    )
    monkeypatch.setenv("MERLIN_RISCV_GCC", "/ambient/replacement")
    monkeypatch.setenv("UNSELECTED_SECRET", "must not enter child")
    environment = execution_environment(command)
    code = (
        "import json,os; "
        "assert 'UNSELECTED_SECRET' not in os.environ; "
        f"print(json.dumps({{k:os.environ[k] for k in {tuple(selected)!r}}}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == {name: str(path) for name, path in selected.items()}
    monkeypatch.setattr(freeze, "_verify_frozen_sources", lambda plan: {})
    freeze.verify(staged)
    selected["MERLIN_RISCV_GCC"].write_text("changed bytes\n")
    with pytest.raises(ValueError, match="execution selection changed: MERLIN_RISCV_GCC"):
        freeze.verify(staged)


def test_routing_and_unselected_injection_are_refused(selected):
    selection = execution_runtime.select("fixture", SimpleNamespace(source_paths=()))
    env = execution_runtime.environment(selection)
    for name, value in (("MERLIN_CLANG", "/other/clang"), ("MERLIN_MLC_DIR", "/unselected/mlc")):
        changed = {**env, name: value}
        with pytest.raises(ValueError, match="environment routing changed"):
            execution_runtime.verify(selection, changed)
    with pytest.raises(ValueError, match="freeze a new run"):
        execution_runtime.verify(None, env)


def test_external_rtl_members_and_absence_are_bound(tmp_path, monkeypatch, selected):
    from merlin.targetgen.rtl import semantic_facts

    rtl = tmp_path / "rtl"
    rtl.mkdir()
    (rtl / "present.fir").write_text("circuit Fixture\n")
    probes = {
        "sources": {"rtl": {"ext_root": "chipyard", "path": "present.fir"}},
        "inline": {"ext_root": "chipyard", "path": "absent.fir"},
    }
    monkeypatch.setattr(semantic_facts, "load_probes", lambda target: probes)
    monkeypatch.setenv("MERLIN_EXT_CHIPYARD", str(rtl))
    selection = execution_runtime.select("fixture", SimpleNamespace(source_paths=()))
    env = execution_runtime.environment(selection)
    execution_runtime.verify(selection, env)
    (rtl / "absent.fir").write_text("newly available source\n")
    with pytest.raises(ValueError, match="execution selection changed: MERLIN_EXT_CHIPYARD"):
        execution_runtime.verify(selection, env)


def test_mlc_and_compiler_dependencies_are_bound(tmp_path, monkeypatch, selected):
    from merlin.targetgen.rtl import mlc_bridge

    monkeypatch.setattr(mlc_bridge, "_arc_target", lambda target: target)
    mlc = tmp_path / "model-ladder"
    (mlc / "mlc").mkdir(parents=True)
    source = mlc / "mlc/__init__.py"
    source.write_text("# selected MLC\n")
    monkeypatch.setenv("MERLIN_MLC_DIR", str(mlc))
    compiler = tmp_path / "compiler-venv"
    (compiler / "bin").mkdir(parents=True)
    (compiler / "lib").mkdir()
    python = compiler / "bin/python"
    python.symlink_to(sys.executable)
    (compiler / "pyvenv.cfg").write_text("# selected compiler venv\n")
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", str(python))
    selection = execution_runtime.select("fixture", SimpleNamespace(source_paths=()))
    env = execution_runtime.environment(selection)
    assert env["MERLIN_COMPILER_PYTHON"] == str(python)  # retain venv identity
    execution_runtime.verify(selection, env)
    for member in (source, compiler / "lib/new_dependency.py"):
        original = member.read_bytes() if member.exists() else None
        member.write_text("changed dependency\n")
        with pytest.raises(ValueError, match="execution selection changed"):
            execution_runtime.verify(selection, env)
        if original is None:
            member.unlink()
        else:
            member.write_bytes(original)


def test_executable_alias_retargeting_is_refused(tmp_path, monkeypatch, selected):
    alias = tmp_path / "selected-clang"
    alias.symlink_to(selected["MERLIN_CLANG"])
    monkeypatch.setenv("MERLIN_CLANG", str(alias))
    selection = execution_runtime.select("fixture", SimpleNamespace(source_paths=()))
    alias.unlink()
    alias.symlink_to(selected["MERLIN_OBJDUMP"])  # identical bytes, different location
    with pytest.raises(ValueError, match="execution selection changed: MERLIN_CLANG"):
        execution_runtime.verify(selection, execution_runtime.environment(selection))
