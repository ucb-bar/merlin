"""Selected backend smoke evidence is tri-state and fails closed on broken hooks."""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base as backends
from merlin.targetgen import capsule_runner as CR


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("synthetic smoke tests cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.mark.parametrize("value", [True, False, None])
@pytest.mark.parametrize("selected", [None, "selected_compiler"])
def test_selected_provider_receives_original_target_and_preserves_tristate(monkeypatch, value, selected):
    observed = []

    def hook(*, target):
        observed.append(("hook", target))
        return value, "exact provider evidence"

    def backend(name):
        observed.append(("backend", name))
        return SimpleNamespace(preflight_codegen_smoke=hook)

    monkeypatch.setattr(backends, "get_backend", backend)
    result = CR.codegen_smoke("experiment_target", backend_target=selected)
    assert result[0] is value
    assert result[1] == "exact provider evidence"
    assert observed == [("backend", selected or "experiment_target"), ("hook", "experiment_target")]


@pytest.mark.parametrize("backend", [SimpleNamespace(), SimpleNamespace(preflight_codegen_smoke=None)])
def test_absent_optional_hook_is_not_a_pass(monkeypatch, backend):
    monkeypatch.setattr(backends, "get_backend", lambda name: backend)
    ok, reason = CR.codegen_smoke("uncovered_target")
    assert ok is None and reason


@pytest.mark.parametrize("hook", [True, 0, "not callable", object()])
def test_noncallable_hook_is_broken_not_absent(monkeypatch, hook):
    monkeypatch.setattr(backends, "get_backend", lambda name: SimpleNamespace(preflight_codegen_smoke=hook))
    ok, reason = CR.codegen_smoke("fixture")
    assert ok is False and "non-callable" in reason


@pytest.mark.parametrize(
    "result",
    [
        None,
        True,
        False,
        0,
        1,
        "pass",
        [],
        (True,),
        (True, "ok", "extra"),
        [True, "ok"],
        (0, "ok"),
        (1, "ok"),
        ("yes", "ok"),
        (True, None),
        (None, 1),
    ],
)
def test_malformed_hook_evidence_fails_closed(monkeypatch, result):
    monkeypatch.setattr(
        backends,
        "get_backend",
        lambda name: SimpleNamespace(preflight_codegen_smoke=lambda **kwargs: result),
    )
    ok, reason = CR.codegen_smoke("fixture")
    assert ok is False and "malformed" in reason


def test_broken_hook_reports_failure_without_fallback(monkeypatch):
    selected = []

    def hook(*, target):
        raise RuntimeError("the selected compiler emitted a broken kernel")

    def backend(name):
        selected.append(name)
        return SimpleNamespace(preflight_codegen_smoke=hook)

    monkeypatch.setattr(backends, "get_backend", backend)
    ok, reason = CR.codegen_smoke("fixture", backend_target="declared_compiler")
    assert ok is False and "broken kernel" in reason
    assert selected == ["declared_compiler"]


@pytest.mark.parametrize(
    "explicit,error,expected",
    [
        (False, KeyError("fixture"), None),
        (True, KeyError("fixture"), False),
        (False, KeyError("backend module failed to load"), False),
        (False, KeyError("fixture", "load failure"), False),
        (False, RuntimeError("broken selected provider"), False),
        (True, RuntimeError("broken selected provider"), False),
        (False, ImportError("broken plugin import"), False),
    ],
)
def test_missing_backend_differs_from_broken_or_explicit_selection(monkeypatch, explicit, error, expected):
    seen = []

    def unavailable(name):
        seen.append(name)
        raise error

    monkeypatch.setattr(backends, "get_backend", unavailable)
    kwargs = {"backend_target": "fixture"} if explicit else {}
    ok, reason = CR.codegen_smoke("fixture", **kwargs)
    assert ok is expected and reason
    assert seen == ["fixture"]


@pytest.mark.parametrize("selected", ["", " ", 0, True, []])
def test_invalid_explicit_backend_cannot_fall_back_to_target(monkeypatch, selected):
    monkeypatch.setattr(backends, "get_backend", lambda name: pytest.fail("invalid selection reached loading"))
    ok, reason = CR.codegen_smoke("fixture", backend_target=selected)
    assert ok is False and "backend target" in reason


def test_gemmini_production_smoke_uses_the_shared_l3_engine_selection(monkeypatch):
    """A required GSIM run must not secretly execute a Verilator-only production smoke first."""
    from merlin.runtime.backends import base as backends
    from merlin.runtime.reference import reference_outputs
    from merlin.targetgen.target_registry import resolve

    if resolve("gemmini").external_root is None:
        pytest.skip("requires explicitly selected Gemmini support via MERLIN_TARGET_PATH")
    backend = backends.get_backend("gemmini")
    selected = {"engine": "gsim", "required_engine": "gsim"}
    monkeypatch.setattr(CR, "chipyard_l3_selection", lambda target: selected)
    seen: dict[str, str] = {}

    def fake_available(engine):
        seen["available"] = engine
        return engine == "gsim"

    def fake_run(cb, *, workdir, simulator, timeout):
        seen["run"] = simulator
        elf = Path(workdir) / "smoke.elf"
        elf.write_bytes(b"ELF")
        return {
            "correct": True,
            "outputs": reference_outputs(cb),
            "oracle": {"derived_from_rtl": True},
            "elf": str(elf),
        }

    # The public OOT backend package re-exports the implementation's functions.  Patch the globals in
    # the implementation module where ``preflight_codegen_smoke`` resolves them.
    monkeypatch.setattr(backend.gemmini, "available", fake_available)
    monkeypatch.setattr(backend.gemmini, "run_command_buffer", fake_run)
    ok, reason = backend.preflight_codegen_smoke(target="gemmini")

    assert ok is True
    assert seen == {"available": "gsim", "run": "gsim"}
    assert "bit-exact on gsim RTL" in reason
