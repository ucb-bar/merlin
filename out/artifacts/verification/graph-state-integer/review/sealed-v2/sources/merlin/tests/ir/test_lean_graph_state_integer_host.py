"""Mocked process classifications, without creating native crashes or SDK holds."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_lean_graph_state_integer import REFERENCE

from merlin.verify import lean_graph_state_integer as integer


@pytest.mark.parametrize(
    "kind",
    [
        "signal",
        "exit2",
        "setup",
        "unknown-marker",
        "bool-marker",
        "invalid-json",
        "success-missing",
        "success-invalid-json",
    ],
)
def test_unconfirmed_process_failure_is_unavailable(tmp_path, monkeypatch, kind):
    source = tmp_path / "input.mlir"
    source.write_text("builtin.module {}")
    observed = []

    def unavailable(destination, reason):
        observed.append((destination, reason))
        raise RuntimeError("mocked unavailable; no real blocker file written")

    def run(command, **_):
        destination = Path(command[command.index("--destination") + 1])
        destination.mkdir(parents=True)
        marker = {
            "schema": "merlin.graph-state-integer.host-domain-refusal.v1",
            "status": "unsupported",
            "reason": "deliberate mock",
            "kind": "ValueError",
            "qualified": False,
        }
        if kind == "unknown-marker":
            marker["kind"] = "ImportError"
        if kind == "bool-marker":
            marker["qualified"] = 0
        if kind in ("unknown-marker", "bool-marker", "exit2"):
            (destination / "REJECTION.json").write_text(json.dumps(marker))
        elif kind == "invalid-json":
            (destination / "REJECTION.json").write_text("not json")
        elif kind == "success-invalid-json":
            (destination / "CAPTURE.json").write_text("not json")
        return SimpleNamespace(
            returncode=-11 if kind == "signal" else 2 if kind == "exit2" else 0 if kind.startswith("success-") else 1
        )

    monkeypatch.setattr(integer, "run_owned", run)
    monkeypatch.setattr(integer, "_unavailable", unavailable)
    with pytest.raises(RuntimeError, match="mocked unavailable"):
        integer.intake(REFERENCE, tmp_path / "intake", source=source)
    assert len(observed) == 1


def test_only_structured_known_domain_refusal_is_value_error(tmp_path, monkeypatch):
    source = tmp_path / "input.mlir"
    source.write_text("builtin.module {}")

    def run(command, **_):
        destination = Path(command[command.index("--destination") + 1])
        destination.mkdir(parents=True)
        (destination / "REJECTION.json").write_text(
            json.dumps(
                {
                    "schema": "merlin.graph-state-integer.host-domain-refusal.v1",
                    "status": "unsupported",
                    "reason": "explicit supported-domain refusal",
                    "kind": "ValueError",
                    "qualified": False,
                }
            )
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(integer, "run_owned", run)
    with pytest.raises(ValueError, match="structured actual-IR"):
        integer.intake(REFERENCE, tmp_path / "intake", source=source)
