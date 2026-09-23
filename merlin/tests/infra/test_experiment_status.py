"""check_structure "experiment status": every experiment declares active / frozen / reference."""
from __future__ import annotations

import importlib.util

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_structure.py"


def _gate(root=None):
    spec = importlib.util.spec_from_file_location("check_structure_under_test", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if root is not None:
        mod.ROOT = str(root)
    return mod


def _experiment(root, name, status_line, findings=False):
    d = root / "merlin" / "experiments" / name
    d.mkdir(parents=True)
    body = f"# AGENT.md — {name}\n\n" + (f"{status_line}\n" if status_line else "") + "\n## Purpose\nx\n"
    (d / "AGENT.md").write_text(body, encoding="utf-8")
    if findings:
        (d / "FINDINGS.md").write_text("# findings\n", encoding="utf-8")


def _errors(root):
    errors: list[str] = []
    _gate(root).check_experiment_status(errors)
    return errors


def test_every_experiment_in_the_tree_declares_a_valid_status():
    errors: list[str] = []
    _gate().check_experiment_status(errors)
    assert errors == []


def test_a_missing_status_is_reported(tmp_path):
    _experiment(tmp_path, "quiet", None)
    assert any("declares no `Status:`" in e and "quiet" in e for e in _errors(tmp_path))


def test_an_unknown_status_is_reported(tmp_path):
    _experiment(tmp_path, "odd", "Status: abandoned — nobody touched it")
    assert any("'abandoned'" in e for e in _errors(tmp_path))


def test_frozen_requires_its_findings(tmp_path):
    _experiment(tmp_path, "done", "Status: frozen — results are in FINDINGS.md")
    assert any("no FINDINGS.md" in e for e in _errors(tmp_path))
    (tmp_path / "merlin/experiments/done/FINDINGS.md").write_text("# f\n", encoding="utf-8")
    assert _errors(tmp_path) == []


def test_active_and_reference_pass(tmp_path):
    _experiment(tmp_path, "live", "Status: active")
    _experiment(tmp_path, "scaffold", "Status: reference — copied by the others")
    assert _errors(tmp_path) == []
