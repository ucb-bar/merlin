from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_no_target_name.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_check_no_target_name", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_target_identifier_planted_in_a_clean_module_fails_the_gate(gate, tmp_path, monkeypatch, capsys):
    """MUTATION, at the VERDICT: the gate's own ``main`` passes the clean module and fails it once an
    identifier naming a target is planted -- the spelling the literal check let through."""
    rel = "src/merlin/kernels/bench_ceiling.py"
    clean = (repo_root() / rel).read_text(encoding="utf-8")
    module = tmp_path / rel
    module.parent.mkdir(parents=True)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_iter_targets", lambda staged: [Path(rel)])
    module.write_text(clean, encoding="utf-8")
    assert gate.main([]) == 0
    module.write_text(clean + "\nFOO_SATURN_X = 1\n", encoding="utf-8")
    capsys.readouterr()
    assert gate.main([]) == 1
    assert "FOO_SATURN_X" in capsys.readouterr().out
