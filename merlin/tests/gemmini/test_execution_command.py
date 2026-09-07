"""GSIM command construction and stale-pin refusal; no emulator is executed."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import pytest

from merlin.runtime.backends.base import get_backend


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    registered = get_backend("gemmini")
    backend = importlib.import_module(registered.__package__ + ".gemmini")
    helper = importlib.import_module(registered.__package__ + ".gemmini_execution_command")
    # This deliberately non-running file is only a provenance/argv test fixture.
    engine = tmp_path / "emulator"
    engine.write_bytes(b"never execute this command-construction fixture\n")
    engine.chmod(0o755)
    receipt = tmp_path / "build_receipt.json"
    receipt.write_text(json.dumps({"schema_version": "merlin.gsim-model-build.v2", "status": "complete",
                                   "binary_sha256": sha(engine.read_bytes())}))
    elf = tmp_path / "short.elf"
    elf.write_bytes(b"exact already-built artifact identity fixture")
    monkeypatch.setenv(backend.GSIM_EMU_ENV, str(engine))
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_MAXCYCLES", "12345")
    expected = helper._current_engine()
    assert expected["available"] and expected["receipt_status"] == "bound"
    return backend, helper, elf, engine, receipt, expected


def prepare(data, **kwargs):
    _, helper, elf, _, _, engine = data
    return helper.prepare_gsim_command(elf, expected_elf_sha256=sha(elf.read_bytes()),
                                       expected_engine_provenance=engine, **kwargs)


def test_same_emulator_argv_as_legacy_run_elf_without_execution(inputs, monkeypatch):
    backend, helper, elf, _, _, _ = inputs
    calls = []

    def no_execute(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "captured, not executed", "")

    monkeypatch.setattr(backend.subprocess, "run", no_execute)
    command = prepare(inputs)
    assert not calls, "command preparation must not invoke a subprocess"
    assert backend.run_elf(elf, simulator="gsim", timeout=7) == "captured, not executed"
    assert len(calls) == 1 and calls[0][0] == list(command.emulator_argv)
    assert calls[0][1]["preexec_fn"] is backend._unlimited_stack
    assert command.argv[1:4] == ("-I", "-S", "-c")
    assert command.argv[5:] == command.emulator_argv
    assert "os.execv(sys.argv[1], sys.argv[1:])" in command.argv[4]
    assert "start_new_session" not in command.argv[4]
    assert Path(command.argv[0]).is_absolute()
    assert command.env_overrides == ()
    assert command.revalidate()["status"] == "unchanged"
    evidence = command.to_evidence()
    assert not evidence["target_executed"] and not evidence["wall_time_admission"]
    assert not evidence["mount_authorization"] and not evidence["warm_protocol_verified"]
    assert set(command.required_readonly_paths) == {str(elf), command.emulator_argv[0], command.argv[0]}


def test_explicit_hang_bound_does_not_change_backend_configuration(inputs):
    backend, _, elf, _, _, _ = inputs
    command = prepare(inputs, max_cycles=73)
    assert command.emulator_argv == tuple(backend._gsim_argv(elf, max_cycles=73))
    assert command.emulator_argv[-2] == "+max-cycles=73"
    assert backend.gsim_max_cycles() == "12345"
    assert not command.to_evidence()["workload_scope_admitted"]


@pytest.mark.parametrize("value", [0, -1, True, "12;bad", 1 << 63, 1.5])
def test_invalid_hang_bounds_refuse(inputs, value):
    with pytest.raises(ValueError, match="max cycles"):
        prepare(inputs, max_cycles=value)


def test_wrong_expected_elf_or_engine_refuses(inputs):
    _, helper, elf, _, _, engine = inputs
    with pytest.raises(ValueError, match="expected bytes"):
        helper.prepare_gsim_command(elf, expected_elf_sha256="0" * 64, expected_engine_provenance=engine)
    wrong = copy.deepcopy(engine)
    wrong["binary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unexpected GSIM"):
        helper.prepare_gsim_command(elf, expected_elf_sha256=sha(elf.read_bytes()), expected_engine_provenance=wrong)


@pytest.mark.parametrize("changed", ["elf", "engine", "receipt"])
def test_changed_pins_refuse_before_or_after_consumption(inputs, changed):
    _, _, elf, engine, receipt, _ = inputs
    command = prepare(inputs)
    path = {"elf": elf, "engine": engine, "receipt": receipt}[changed]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        command.revalidate()


def test_available_but_unreceipted_engine_is_not_accepted(inputs, monkeypatch):
    _, helper, elf, _, receipt, _ = inputs
    receipt.unlink()
    monkeypatch.setenv("MERLIN_GSIM_REQUIRE_RECEIPT", "0")
    engine = helper._current_engine()
    assert engine["available"] and engine["receipt_status"] == "absent"
    with pytest.raises(ValueError, match="bound build receipt"):
        helper.prepare_gsim_command(elf, expected_elf_sha256=sha(elf.read_bytes()), expected_engine_provenance=engine)


def test_relative_interpreter_refuses(inputs):
    with pytest.raises(ValueError, match="absolute host-approved Python"):
        prepare(inputs, python_executable="python3")
