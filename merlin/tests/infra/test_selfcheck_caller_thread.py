"""The in-sandbox self-check client names the Codex thread that ran it, and nothing else changes."""

from __future__ import annotations

import importlib.util
import json

import pytest

from merlin.common.paths import module_source_path


def _shim(tmp_path, monkeypatch):
    # Loaded from its own file under the staged name, as the sandbox runs it (no package imports).
    spec = importlib.util.spec_from_file_location(
        "agent_selfcheck", module_source_path("merlin_experiments.phase1.tools.selfcheck")
    )
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    monkeypatch.setattr(shim, "__file__", str(tmp_path / "agent_selfcheck.py"))
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    return shim


class _Published(Exception):
    def __init__(self, body):
        self.body = body


def _published(shim, monkeypatch, argv) -> dict:
    def capture(_path, text):
        raise _Published(json.loads(text))

    monkeypatch.setattr(shim, "_atomic_write", capture)
    with pytest.raises(_Published) as published:
        shim.main(argv)
    return published.value.body


def test_the_request_carries_the_codex_thread_id(tmp_path, monkeypatch):
    shim = _shim(tmp_path, monkeypatch)
    monkeypatch.setenv("CODEX_THREAD_ID", "01a14a7f-b0d1-7121-b977-83225f4fba52")
    body = _published(shim, monkeypatch, ["--capsules", "A,B"])
    assert body["caller_thread_id"] == "01a14a7f-b0d1-7121-b977-83225f4fba52"
    assert body["capsules"] == "A,B" and body["protocol"] == shim.PROTOCOL_VERSION


def test_without_a_thread_id_the_request_is_unchanged(tmp_path, monkeypatch):
    shim = _shim(tmp_path, monkeypatch)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    assert "caller_thread_id" not in _published(shim, monkeypatch, ["--capsules", "A"])
    monkeypatch.setenv("CODEX_THREAD_ID", "   ")
    assert "caller_thread_id" not in _published(shim, monkeypatch, ["--capsules", "A"])
    monkeypatch.setenv("CODEX_THREAD_ID", "x" * 500)
    assert len(_published(shim, monkeypatch, ["--capsules", "A"])["caller_thread_id"]) == 128
