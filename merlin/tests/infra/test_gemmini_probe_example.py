"""Pure probe CLI routing tests; no simulator or toolchain process is launched."""

import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def load_probe():
    source = repo_root() / "examples/gemmini/target/probe_oracles.py"
    spec = importlib.util.spec_from_file_location("gemmini_probe_example", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_never_launches_a_process(monkeypatch):
    module = load_probe()
    monkeypatch.setattr(sys, "argv", ["probe_oracles.py"])
    monkeypatch.setattr(module, "status", lambda: {"spike": False, "verilator": False})

    def forbidden(*args, **kwargs):
        raise AssertionError("default status must not execute a process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    assert module.main() == 0


@pytest.mark.parametrize("engine", ["spike", "verilator"])
def test_explicit_run_preserves_command_and_timeout(tmp_path, monkeypatch, engine):
    module = load_probe()
    monkeypatch.setenv("MERLIN_CHIPYARD", str(tmp_path))
    monkeypatch.setenv("MERLIN_GEMMINI_HARNESS_DIR", str(tmp_path / "harness"))
    monkeypatch.setenv("MERLIN_GEMMINI_SPIKE", str(tmp_path / "spike"))
    monkeypatch.setenv("MERLIN_GEMMINI_VERILATOR", str(tmp_path / "verilator"))
    elf = tmp_path / "harness/build" / module.KNOWN_GOOD[engine]
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"synthetic-not-an-executable")
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(stdout="fixture\n", returncode=7)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(module, "status", lambda: {"spike": True, "verilator": True})
    monkeypatch.setattr(sys, "argv", ["probe_oracles.py", "--run", engine, "--timeout", "19"])
    assert module.main() == 7
    command, options = observed[0]
    assert command == [str(tmp_path / engine), *(["--extension=gemmini"] if engine == "spike" else []), str(elf)]
    assert options["timeout"] == 19
    assert options["capture_output"] is True
    if engine == "spike":
        assert options["env"]["LD_LIBRARY_PATH"].startswith(str(tmp_path / ".conda-env/riscv-tools/lib") + ":")


def test_unavailable_requested_oracle_fails_without_execution(monkeypatch, capsys):
    module = load_probe()
    monkeypatch.setattr(sys, "argv", ["probe_oracles.py", "--run", "spike"])
    monkeypatch.setattr(module, "status", lambda: {"spike": False})
    monkeypatch.setattr(module, "run_known_good", lambda *args: pytest.fail("missing oracle must not run"))
    assert module.main() == 2
    assert "oracle unavailable" in capsys.readouterr().out
