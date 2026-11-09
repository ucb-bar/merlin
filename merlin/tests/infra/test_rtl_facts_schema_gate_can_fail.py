from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from merlin.common.paths import repo_root

GATE_PATH = repo_root() / "build_tools" / "scripts" / "check_rtl_facts_schema.py"


def _gate():
    spec = importlib.util.spec_from_file_location("_rtl_facts_schema_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, name: str, doc: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_an_artifact_whose_body_contradicts_its_family_is_caught(tmp_path):
    """The whole point of the discriminator: a thread geometry is not a statically extracted body."""
    gate = _gate()
    wrong = _write(
        tmp_path,
        "facts.json",
        {"schema_version": "2.0", "family": "circt_static", "facts": {"target": "t", "simt": {"cores": 1}}},
    )
    problems, rc = gate.verdict(gate.artifacts([wrong]))
    assert rc == 1, "an artifact claiming the wrong family was accepted"
    assert any("circt_static" in p for p in problems)
