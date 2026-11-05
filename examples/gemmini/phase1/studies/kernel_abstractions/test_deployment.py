"""Deployment paths are explicit and do not launch an author or evaluator."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("study_runner_deployment_test", Path(__file__).with_name("run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("bwrap", "codex", "python"):
        path = tools / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    llvm = tmp_path / "llvm"
    llvm.mkdir()
    auth = tmp_path / "private-auth.json"
    auth.write_text("{}")
    module.C = dict(bwrap=str(tools / "bwrap"), codex_binary=str(tools / "codex"),
                    python=str(tools / "python"), codex_auth_source=str(auth))
    module.ENV = {"MERLIN_MLIR_INSTALL": str(llvm)}
    return module


def test_paths_follow_selected_deployment(runner):
    selected = runner.deployment_paths()
    assert selected["codex"] == Path(runner.C["codex_binary"])
    assert selected["auth"] == Path(runner.C["codex_auth_source"])
    assert selected["llvm"] == Path(runner.ENV["MERLIN_MLIR_INSTALL"])


def test_missing_executable_refuses_before_author_call(runner):
    Path(runner.C["codex_binary"]).unlink()
    with pytest.raises(FileNotFoundError):
        runner.deployment_paths()


def test_nonexecutable_tool_refuses(runner):
    Path(runner.C["bwrap"]).chmod(0o644)
    with pytest.raises(ValueError, match="not executable"):
        runner.deployment_paths()


def test_sandbox_uses_selected_tools_and_auth(runner, tmp_path):
    runner.WS, runner.F = tmp_path / "workspace", tmp_path / "framework"
    runner.base_argv = lambda *args, **kwargs: ["bwrap", "--clearenv"]
    home = tmp_path / "provider-home"
    argv, readonly = runner.sandbox(["/usr/bin/true"], home)
    assert argv[0] == runner.C["bwrap"]
    assert runner.C["codex_binary"] in argv
    at = argv.index(runner.C["codex_auth_source"])
    assert argv[at - 1] == "--bind" and argv[at + 1] == str(home / "auth.json")
    assert runner.ENV["MERLIN_MLIR_INSTALL"] in readonly
