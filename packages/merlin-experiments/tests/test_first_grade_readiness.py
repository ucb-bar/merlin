"""Incomplete package drafts never become a corpus-sized batch of tool failures."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import lifecycle as L
from merlin_experiments.phase1.feedback import loop_grading as LG

from merlin.targetgen import capsule_grade as CG
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.package_runtime import Package


def _package_bootstrap(monkeypatch):
    monkeypatch.setattr(CG, "source_experiment_env", lambda _: {})
    monkeypatch.setattr(
        CG,
        "load_package",
        lambda directory, **_: Package(Path(directory), {"language": "python"}, Path(directory) / "compiler.py"),
    )
    monkeypatch.setattr(CG, "integrity_scan", lambda *a, **k: None)


@pytest.mark.parametrize("tool_is_directory", [False, True])
def test_grade_rejects_missing_declared_tool_before_discovery_or_fanout(tmp_path, monkeypatch, tool_is_directory):
    _package_bootstrap(monkeypatch)
    package = tmp_path / "package"
    package.mkdir()
    if tool_is_directory:
        (package / "compiler.py").mkdir()
    monkeypatch.setattr(CR, "discover_capsules", lambda *a, **k: pytest.fail("unbuildable draft discovered cohort"))
    monkeypatch.setattr(CR, "run_suite", lambda *a, **k: pytest.fail("unbuildable draft issued capsule work"))
    result = CG.grade(package, capsules_root=tmp_path / "public", runs_root=tmp_path / "runs", target="fixture")
    assert result["package_readiness"] == {"status": "not_ready", "reason": "declared_tool_missing"}
    assert result["failure"]["plane"] == "build"
    assert "compiler.py" in result["failure"]["detail"]
    assert result["n_capsules"] == result["n_passed"] == 0
    assert result["per_capsule"] == [] and result["functional_pass"] == 0
    assert not (tmp_path / "runs").exists()


def test_declared_build_can_materialize_tool_before_readiness_check(tmp_path, monkeypatch):
    from merlin.targetgen import package_runtime
    from merlin.targetgen.contract import toolchain

    _package_bootstrap(monkeypatch)
    package = tmp_path / "package"
    package.mkdir()
    manifest = {
        "language": "cpp",
        "build": {
            "tool_output": "compiler.py",
            "command": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('compiler.py').write_text('generated compiler\\n')",
            ],
        },
    }
    monkeypatch.setattr(
        CG, "load_package", lambda directory, **_: Package(Path(directory), manifest, Path(directory) / "compiler.py")
    )
    monkeypatch.setattr(toolchain, "mlir_cmake_dir", lambda: tmp_path / "cmake")
    monkeypatch.setattr(toolchain, "mlir_install", lambda: tmp_path / "llvm")
    monkeypatch.setattr(package_runtime, "_usable_cmake", lambda: "/bin/false")

    class ReachedDiscovery(Exception):
        pass

    def discover(*a, **k):
        assert (package / "compiler.py").read_text() == "generated compiler\n"
        raise ReachedDiscovery

    monkeypatch.setattr(CR, "discover_capsules", discover)
    with pytest.raises(ReachedDiscovery):
        CG.grade(package, capsules_root=tmp_path / "public", runs_root=tmp_path / "runs", target="fixture")


def test_real_fast_grade_publishes_ungradeable_zero_execution_readiness(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import promotion, qa

    _package_bootstrap(monkeypatch)
    ws, run, public = (tmp_path / name for name in ("workspace", "run", "public"))
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    context = SimpleNamespace(descriptor=tmp_path / "descriptor", repo=tmp_path)
    inputs = LG.GradingInputs(context, "merlin_assisted", public, (), None)
    monkeypatch.setattr(LG, "load_target_experiment", lambda _: SimpleNamespace(target="fixture", sim_via=""))
    monkeypatch.setattr(LG, "compiler_execution", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(LG, "declared_loop_tiers", lambda *a: {"L2", "L3"})
    monkeypatch.setattr(LG, "cert_tiers_beyond_loop", lambda **k: ({"L3"}, {"L3"}))
    monkeypatch.setattr(CR, "qa_loop_adapters", lambda *a, **k: {"L2": object()})
    monkeypatch.setattr(LG, "oracle_kwargs", lambda _: {})
    monkeypatch.setattr(LG, "_record_plateau", lambda *a, **kw: None)
    monkeypatch.setattr(L, "record_channel_health", lambda *a: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        qa, "_per_capsule_from_results", lambda *a: pytest.fail("no capsule readback belongs to readiness")
    )
    monkeypatch.setattr(CR, "discover_capsules", lambda *a, **k: pytest.fail("no discovery for missing compiler"))
    monkeypatch.setattr(CR, "run_suite", lambda *a, **k: pytest.fail("no original child for missing compiler"))
    result = LG.fast_grade(ws, run, 900, 30, inputs=inputs)
    assert result["package_readiness"]["status"] == "not_ready"
    assert result["package_failure"]["plane"] == "build"
    assert result["n_capsules"] == result["n_passed"] == 0 and result["per_capsule"] == []
    assert result["gradeable"] is result["all_pass"] is result["mandatory_tiers_complete"] is False
    assert result["original_cohort_complete"] is result["structural_evidence_complete"] is False
    assert result["tiers_graded"] == [] and result["tiers_not_run"] == ["L2", "L3"]
    assert result["snapshot_current"] is True
    for path in (ws / "qa/verdict.json", run / "qa_history/verdict_fast_900.json"):
        saved = json.loads(path.read_text())
        assert saved["graded_source_sha256"] == result["graded_source_sha256"]
        assert saved["graded_request_sha256"] == result["graded_request_sha256"]
        assert saved["snapshot_current"] is True
        assert saved["n_capsules"] == 0 and saved["gradeable"] is False


def _synchronous_background(monkeypatch, waits, *, stop_on_wait):
    import threading

    class Event:
        done = False

        def is_set(self):
            return self.done

        def set(self):
            self.done = True

        def wait(self, delay):
            if self.done:
                return True
            waits.append(delay)
            self.done = stop_on_wait(delay)
            return self.done

    event = Event()

    class Thread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(threading, "Event", lambda: event)
    monkeypatch.setattr(threading, "Thread", Thread)
    return event


@pytest.mark.parametrize("previous_readiness", [False, True])
def test_missing_tool_readiness_retries_before_normal_interval(tmp_path, monkeypatch, previous_readiness):
    ws, waits, calls = tmp_path / "workspace", [], []
    not_ready = {"package_readiness": {"status": "not_ready"}, "gradeable": False, "n_capsules": 0}
    if previous_readiness:
        (ws / "qa").mkdir(parents=True)
        (ws / "qa/verdict.json").write_text(json.dumps(not_ready))
    _synchronous_background(monkeypatch, waits, stop_on_wait=lambda delay: delay == 300)

    def first(*args):
        calls.append(args[2])
        if len(calls) == 1:
            return not_ready
        return {"snapshot_current": True, "n_capsules": 1, "n_passed": 0, "all_pass": False}

    L.start_background(
        ws,
        tmp_path / "run",
        L.GradeCadence(30, False, 300),
        interval_grades=True,
        grade_callback=lambda *a, **k: pytest.fail("normal interval should still wait"),
        fast_grade_callback=first,
    )
    assert calls == [900, 900]
    assert waits == [30, 300]


def test_obsolete_buildable_first_grade_immediately_hands_off_to_current_grade(tmp_path, monkeypatch):
    waits, calls = [], []
    event = _synchronous_background(monkeypatch, waits, stop_on_wait=lambda _: False)

    def grade(*args, **kwargs):
        calls.append((args[2], kwargs["scratch_key"]))
        event.set()
        return {"snapshot_current": True, "all_pass": False}

    L.start_background(
        tmp_path / "workspace",
        tmp_path / "run",
        L.GradeCadence(30, False, 300),
        interval_grades=True,
        grade_callback=grade,
        source_identity=lambda: "new-source",
        fast_grade_callback=lambda *a: {"snapshot_current": False, "n_capsules": 1, "n_passed": 0},
    )
    assert waits == [0]
    assert len(calls) == 1 and calls[0][0] == 901


def test_first_grade_streams_completed_redaction_without_globally_gating_other_capsules(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import promotion, qa

    ws, run, public = (tmp_path / name for name in ("workspace", "run", "public"))
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    inputs = LG.GradingInputs(
        SimpleNamespace(descriptor=tmp_path / "descriptor", repo=tmp_path), "merlin_assisted", public, (), None
    )
    monkeypatch.setattr(LG, "load_target_experiment", lambda _: SimpleNamespace(target="fixture", sim_via=""))
    monkeypatch.setattr(LG, "compiler_execution", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(LG, "declared_loop_tiers", lambda *a: {"L2", "L3"})
    monkeypatch.setattr(LG, "cert_tiers_beyond_loop", lambda **k: ({"L3"}, {"L3"}))
    monkeypatch.setattr(CR, "qa_loop_adapters", lambda *a, **k: {"L2": object()})
    monkeypatch.setattr(LG, "oracle_kwargs", lambda _: {})
    monkeypatch.setattr(LG, "_record_plateau", lambda *a, **kw: None)
    monkeypatch.setattr(L, "record_channel_health", lambda *a: None)
    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **k: (None, None, None))
    redacted = {}
    monkeypatch.setattr(
        qa,
        "_per_capsule_from_results",
        lambda *a, only_capsule=None: {only_capsule: redacted[only_capsule]} if only_capsule else redacted.copy(),
    )

    def grade(*args, **kwargs):
        assert kwargs["defer_running"] is False
        kwargs["on_cohort_ready"](2)
        for name, status in (("individual-refusal", "fail"), ("legitimate-compute", "pass")):
            redacted[name] = {"status": status, "tiers": {"L2": status}}
            kwargs["on_capsule_complete"]({"capsule": name, "status": status, "private": "MUST NOT LEAK"})
            progress = json.loads((ws / "qa/progress.json").read_text())
            assert progress["state"] == "running"
            assert progress["all_pass"] is None and progress["certified_complete"] is False
            assert progress["original_cohort_complete"] is False
            assert progress["n_selected"] == 2
            assert progress["n_completed"] == len(redacted)
            assert "MUST NOT LEAK" not in json.dumps(progress)
            assert not (ws / "qa/verdict.json").exists()
            assert kwargs["defer_pending"]() is None  # one failure cannot gate an unrelated member
        return {"per_capsule": [{"capsule": name, **row} for name, row in redacted.items()]}

    monkeypatch.setattr(CG, "grade", grade)
    result = LG.fast_grade(ws, run, 900, 30, inputs=inputs)
    assert result["n_capsules"] == 2 and result["n_passed"] == 1
    progress = json.loads((ws / "qa/progress.json").read_text())
    assert progress["state"] == "complete" and progress["all_pass"] is None


def test_draining_snapshot_cannot_replace_newer_progress(tmp_path):
    from merlin_experiments.phase1.feedback.grade_progress import SnapshotProgress

    ws = tmp_path / "workspace"
    (ws / "qa").mkdir(parents=True)
    path = ws / "qa/progress.json"
    current = {"graded_source_sha256": "new", "state": "running"}
    path.write_text(json.dumps(current))
    guard = SimpleNamespace(current=lambda: False, reason=lambda: "background source superseded")
    SnapshotProgress(ws, tmp_path / "runs", "old", guard).finish({"snapshot_current": False})
    assert json.loads(path.read_text()) == current


@pytest.mark.parametrize("current", [True, False])
def test_progress_distinguishes_snapshot_passes_from_current_source(tmp_path, current):
    from merlin_experiments.phase1.feedback.grade_progress import SnapshotProgress

    ws = tmp_path / "workspace"
    ws.mkdir()
    guard = SimpleNamespace(current=lambda: current, reason=lambda: None if current else "source superseded")
    progress = SnapshotProgress(ws, tmp_path / "runs", "snapshot", guard)
    progress.rows["a"] = {"capsule": "a", "status": "pass", "tiers": {"L2": "pass"}}
    progress.finish({"snapshot_current": current})
    record = json.loads((ws / "qa/progress.json").read_text())
    assert record["n_passed"] == 1 and record["count_scope"] == "immutable_snapshot"
    assert record["snapshot_current"] is current
    assert record["current_source_n_passed"] == (1 if current else None)
    assert record["all_pass"] is None and record["certified_complete"] is False
