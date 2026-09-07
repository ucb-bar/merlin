"""Every machine-audit tool invocation uses the supplied isolation boundary."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends.base import get_backend


def test_machine_audit_requires_a_runner(tmp_path):
    with pytest.raises(TypeError, match="run_command"):
        get_backend("gemmini").analyze_machine_artifact("module {}", workdir=tmp_path / "work")


def test_machine_audit_routes_all_tools_through_runner(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import base
    tool = tmp_path / "tool"
    tool.write_text("inert test tool identity")
    translator_alias = tmp_path / "granted_translator"
    translator_alias.symlink_to(tool)
    monkeypatch.setattr(toolchain, "clang", lambda: tool)
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: translator_alias)
    monkeypatch.setattr(toolchain, "objdump", lambda: tool)
    monkeypatch.setattr(base, "harness_build_recipe", lambda target: SimpleNamespace(
        march=lambda: "-march=rv64gc"))
    calls = []

    def runner(argv, *, timeout_s):
        assert 0 < timeout_s <= 5
        calls.append(argv)
        if "-o" in argv:
            Path(argv[argv.index("-o")+1]).write_bytes(b"test artifact")
        return SimpleNamespace(returncode=0, stderr="", stdout=
            "00000000 <kernel>:\n 0: 13 00 00 00\tnop\n 4: 0f29307b\t<unknown>\n")

    result = get_backend("gemmini").analyze_machine_artifact(
        "module {}", workdir=tmp_path / "work", run_command=runner, timeout_seconds=5)
    assert len(calls) == 3
    assert calls[0][0] == str(translator_alias)
    assert calls[-1][1] == "-d"
    assert [arg for arg in calls[1] if arg.startswith("-march=")][-1] == "-march=rv64gc"
    assert result["compiler"]["flags"][-1] == "-march=rv64gc"
    assert result["status"] == "compiled"
    assert result["instruction_sites"]["total"] == 2
    assert result["instruction_sites"]["decoded_total"] == 1
    assert result["instruction_sites"]["undecoded"] == 1
    assert result["instruction_decode_complete"] is False
    assert result["full_model_executed"] is False
    assert result["timing_measured"] is False
    previous = result["build_policy_identity"]
    assert get_backend("gemmini").machine_artifact_policy_identity() == previous
    tool.write_text("changed tool bytes at same path")
    changed_tool = get_backend("gemmini").machine_artifact_policy_identity()
    assert changed_tool != previous
    monkeypatch.setattr(base, "harness_build_recipe", lambda target: SimpleNamespace(
        march=lambda: "-march=rv64gcv"))
    assert get_backend("gemmini").machine_artifact_policy_identity() != changed_tool
