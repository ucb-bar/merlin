"""Pure provider smoke policy tests: no native tool or simulator is launched."""

import importlib
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def smoke(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic smoke must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(Path(__file__).resolve().parents[1]))
    from merlin.runtime.backends.base import get_backend

    backend = get_backend("muon")
    module = importlib.import_module(f"{backend.__name__}.codegen_smoke")
    assert backend.preflight_codegen_smoke is module.preflight_codegen_smoke
    monkeypatch.setattr(module, "selected_sim_via", lambda target: "cyclotron")
    monkeypatch.setattr(module.muon, "_model_for", lambda target: object())
    monkeypatch.setattr(module, "isa_model_for_target", lambda target: SimpleNamespace(is_fixed_format=lambda: True))
    monkeypatch.setattr(module.muon, "available", lambda simulator: True)
    monkeypatch.setattr(module.muon, "compile_kernel_forkfree", forbidden)
    monkeypatch.setattr(module.muon, "run_elf", forbidden)
    return module


def test_prerequisite_failure_precedes_fixed_format_and_availability(smoke, monkeypatch):
    def unavailable(target):
        raise RuntimeError("missing encoding fact")

    def forbidden(*args):
        pytest.fail("prerequisite must run first")

    monkeypatch.setattr(smoke.muon, "_model_for", unavailable)
    monkeypatch.setattr(smoke, "isa_model_for_target", forbidden)
    monkeypatch.setattr(smoke.muon, "available", forbidden)
    ok, reason = smoke.preflight_codegen_smoke(target="radiance")
    assert ok is False
    assert "missing encoding fact" in reason
    assert "MERLIN_MLC_DIR" in reason


@pytest.mark.parametrize("case", ["nonfixed", "missing_model", "other_sim", "absent_sim"])
def test_uncovered_or_unavailable_is_not_success(smoke, monkeypatch, case):
    if case == "nonfixed":
        monkeypatch.setattr(
            smoke, "isa_model_for_target", lambda target: SimpleNamespace(is_fixed_format=lambda: False)
        )
    elif case == "missing_model":

        def missing(target):
            raise RuntimeError("missing ISA")

        monkeypatch.setattr(smoke, "isa_model_for_target", missing)
    elif case == "other_sim":
        monkeypatch.setattr(smoke, "selected_sim_via", lambda target: "other")
        monkeypatch.setattr(smoke.muon, "_model_for", lambda target: pytest.fail("inapplicable prerequisite"))
    else:
        monkeypatch.setattr(smoke.muon, "available", lambda simulator: False)
    ok, reason = smoke.preflight_codegen_smoke(target="radiance")
    assert ok is None
    assert reason.startswith("n/a")


@pytest.mark.parametrize("failure", [None, "compile", "run", "output"])
def test_exact_pipeline_and_result_policy(smoke, monkeypatch, failure):
    events = []

    def model(target):
        events.append(("prerequisite", target))

    def compile_kernel(kernel, directory, *, target):
        events.append(("compile", target))
        assert "csrr %0,0xF14" in kernel
        assert "0xFF080000u" in kernel
        assert "C[i]=A[i]+B[i]" in kernel
        assert Path(directory).is_dir()
        if failure == "compile":
            raise RuntimeError("synthetic compile failure")
        return Path(directory) / "synthetic.elf"

    def run_elf(elf, *, simulator, timeout):
        events.append(("run", simulator, timeout))
        assert elf.endswith("synthetic.elf")
        if failure == "run":
            raise RuntimeError("synthetic run failure")
        values = range(1, 8 if failure == "output" else 9)
        return "\n".join(f"{11 * value:08x}" for value in values), 0, {}

    monkeypatch.setattr(smoke.muon, "_model_for", model)
    monkeypatch.setattr(smoke.muon, "compile_kernel_forkfree", compile_kernel)
    monkeypatch.setattr(smoke.muon, "run_elf", run_elf)
    ok, reason = smoke.preflight_codegen_smoke(target="radiance")
    assert ok is (failure is None)
    assert events[:2] == [("prerequisite", "radiance"), ("compile", "radiance")]
    if failure != "compile":
        assert events[2] == ("run", "cyclotron", 180)
    if failure == "output":
        assert "00000058" in reason
    elif failure:
        assert f"synthetic {failure} failure" in reason
