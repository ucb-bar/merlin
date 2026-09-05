"""The unattended chain must propagate a refused/failed stage to its process status."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from merlin.common.paths import repo_root


def _load_runner():
    harness = repo_root() / "merlin/experiments/capsule_bench/harness"
    sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location("_run_overnight_gates", harness / "run_overnight.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RO = _load_runner()


class _Journal:
    def __init__(self, *_args, **_kwargs):
        self.path = Path("journal.json")
        self.report = Path("REPORT.md")
        self.stages: list[tuple[str, str]] = []

    def stage(self, name, status, _detail, **_extra):
        self.stages.append((name, status))


def _prepare(monkeypatch, tmp_path: Path, outcomes: dict[str, bool]):
    journal = _Journal()
    monkeypatch.setattr(RO, "Journal", lambda *_args, **_kwargs: journal)
    monkeypatch.setattr(RO, "artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(RO, "stage_preflight", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(RO, "stage_functional", lambda *_args, **_kwargs: tmp_path / "functional")
    calls: list[str] = []

    def stage(name):
        def run(*_args, **_kwargs):
            calls.append(name)
            return outcomes[name]
        return run

    monkeypatch.setattr(RO, "stage_grade_and_freeze", stage("grade"))
    monkeypatch.setattr(RO, "stage_calibration", stage("calibration"))
    monkeypatch.setattr(RO, "stage_performance", stage("performance"))
    return journal, calls


def test_grade_failure_stops_the_chain(monkeypatch, tmp_path: Path) -> None:
    journal, calls = _prepare(
        monkeypatch, tmp_path, {"grade": False, "calibration": True, "performance": True})
    assert RO.main(["--tag", "test"]) == 4
    assert calls == ["grade"]
    assert journal.stages[-1] == ("chain", "failed")


def test_calibration_failure_stops_the_chain(monkeypatch, tmp_path: Path) -> None:
    _journal, calls = _prepare(
        monkeypatch, tmp_path, {"grade": True, "calibration": False, "performance": True})
    assert RO.main(["--tag", "test"]) == 5
    assert calls == ["grade", "calibration"]


def test_performance_refusal_is_a_nonzero_chain_result(monkeypatch, tmp_path: Path) -> None:
    journal, calls = _prepare(
        monkeypatch, tmp_path, {"grade": True, "calibration": True, "performance": False})
    assert RO.main(["--tag", "test"]) == 6
    assert calls == ["grade", "calibration", "performance"]
    assert journal.stages[-1] == ("chain", "failed")
