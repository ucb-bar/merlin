"""Phase 1 entrypoints lease the one shared CPU pool at their class, and caps default to its size."""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers import selfcheck as B
from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.feedback import background_admission as admission
from merlin_experiments.phase1.feedback import lifecycle as L
from merlin_experiments.phase1.feedback import loop_grading as LG

from merlin.common import cpu_pool as C


@pytest.fixture
def pool(monkeypatch, tmp_path):
    def select(size):
        monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, range(size))))
        monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))

    for key in (
        "MERLIN_COMPUTE_CPUS",
        "MERLIN_COMPUTE_CPU_SLOTS",
        "MERLIN_SELFCHECK_BROKER_JOBS",
        "MERLIN_SELFCHECK_ORDINARY_JOBS",
        C.PRIORITY_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    return select


def test_selfcheck_and_simjob_concurrency_default_to_the_pool(pool, monkeypatch):
    assert (B._dispatch_capacity(), B._ordinary_capacity(), simjob._default_max_jobs()) == (1, 1, 4)  # legacy
    pool(18)
    assert (B._dispatch_capacity(), B._ordinary_capacity(), simjob._default_max_jobs()) == (2, 18, 18)
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    monkeypatch.setenv("MERLIN_SELFCHECK_ORDINARY_JOBS", "2")
    assert (B._dispatch_capacity(), B._ordinary_capacity()) == (1, 2)  # explicit values still win
    for bad in ("0", "19", "-1", "2.0", ""):
        monkeypatch.setenv("MERLIN_SELFCHECK_ORDINARY_JOBS", bad)
        with pytest.raises(ValueError, match="MERLIN_SELFCHECK_ORDINARY_JOBS"):
            B._ordinary_capacity()


def test_first_grade_leases_at_grade_class_with_no_foreground_skip(monkeypatch, tmp_path):
    monkeypatch.setenv(C.PRIORITY_ENV, str(C.AUXILIARY))  # the grader thread's class wins over an inherited one
    observed = []

    def first(*args):
        observed.append(C.lease_priority())
        return {"all_pass": None}

    handle = L.start_background(
        tmp_path / "ws",
        tmp_path / "run",
        L.GradeCadence(60, False, 900),
        interval_grades=False,
        grade_callback=lambda *a, **k: pytest.fail("no interval"),
        fast_grade_callback=first,
    )
    handle[0].join(10)
    L.stop_background(handle)
    assert observed == [C.GRADE]
    with pytest.raises(TypeError):  # the foreground skip is gone, not silently ignored
        L.start_background(
            tmp_path / "ws",
            tmp_path / "run",
            L.GradeCadence(60, False, 900),
            interval_grades=False,
            grade_callback=first,
            fast_grade_callback=first,
            foreground_has_priority=lambda: True,
        )


def test_tick_telemetry_leases_at_auxiliary_class(monkeypatch, tmp_path):
    (tmp_path / "ws" / "qa").mkdir(parents=True)
    (tmp_path / "ws" / "qa" / "verdict.json").write_text("{}")
    seen = []

    class Stop:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            return self.calls > 1

        def set(self):
            self.calls = 99

        def wait(self, delay):
            self.calls += 1
            return self.calls > 1

    class Inline:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(threading, "Event", Stop)
    monkeypatch.setattr(threading, "Thread", Inline)
    L.start_background(
        tmp_path / "ws",
        tmp_path / "run",
        L.GradeCadence(60, False, 30),
        interval_grades=True,
        grade_callback=lambda *a, **k: seen.append(("grade", C.lease_priority())) or {},
        fast_grade_callback=lambda *a: pytest.fail("a verdict already exists"),
        on_tick=lambda index: seen.append(("tick", C.lease_priority())),
    )
    assert seen == [("grade", C.GRADE), ("tick", C.AUXILIARY)]


def test_background_guard_defers_only_a_superseded_source(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission" / "compiler.py").write_text("original\n")
    monkeypatch.setattr(admission, "foreground_covers", lambda *args: True)
    guard = admission.BackgroundGuard(ws, tmp_path, SimpleNamespace(), admission.source_digest(ws))
    assert guard.reason() is None  # a same-source author check no longer pauses the grade
    (ws / "submission" / "compiler.py").write_text("replacement\n")
    assert guard.reason() == "background source superseded"


def test_advisory_probes_lease_at_auxiliary_class(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability

    from merlin.targetgen import lowering_coverage

    seen = []

    def sweep(*args, **kwargs):
        seen.append(("shape", C.lease_priority()))
        return {"corners": [], "all_covered": True}

    def scalability(*args, **kwargs):
        seen.append(("scalability", C.lease_priority()))
        return {"samples": []}

    monkeypatch.setattr(lowering_coverage, "sweep", sweep)
    monkeypatch.setattr(codegen_scalability, "run", scalability)
    context = SimpleNamespace(target="fixture", repo=tmp_path)
    verdict = {"all_pass": True}
    with C.cpu_priority(C.GRADE):
        LG._attach_shape_generalization(verdict, tmp_path, tmp_path, 0, timeout=1, context=context)
        LG._attach_codegen_scalability(verdict, tmp_path, tmp_path, 0, timeout=1, context=context)
        assert C.lease_priority() == C.GRADE
    assert seen == [("shape", C.AUXILIARY), ("scalability", C.AUXILIARY)]
    assert verdict["shape_coverage"]["ran"] and verdict["codegen_scalability"]["ran"]


def test_run_preparation_leases_at_preparation_class(monkeypatch):
    from merlin_experiments.phase1 import session

    monkeypatch.delenv(C.PRIORITY_ENV, raising=False)
    monkeypatch.setattr(session, "_prepare", lambda *args, **kwargs: C.lease_priority())
    assert session.prepare(None, None, None, workspace_leases=[]) == C.PREPARATION
    assert C.lease_priority() == C.GRADE


@pytest.mark.parametrize("module", ["selfcheck", "simjob", "model_check"])
def test_author_broker_processes_export_the_author_class(module):
    """Each broker's process entrypoint (and so every child it launches) leases at AUTHOR."""
    from merlin.common.paths import repo_root

    source = repo_root() / "packages/merlin-experiments/src/merlin_experiments/phase1/brokers" / f"{module}.py"
    tree = ast.parse(Path(source).read_text())
    guards = [
        node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
    ]
    assert len(guards) == 1
    withs = [node for node in ast.walk(guards[0]) if isinstance(node, ast.With)]
    assert [ast.unparse(item.context_expr) for node in withs for item in node.items] == [
        "cpu_priority(AUTHOR, export=True)"
    ]
    assert "main()" in ast.unparse(withs[0])


def test_promoted_simulation_jobs_lease_below_author_requests(pool, monkeypatch):
    """A promotion certifies a verdict the author already has: it queues behind author work for CPUs."""
    from merlin_experiments.phase1 import joint_gsim_queue as Q

    pool(4)
    inherited = {
        C.PRIORITY_ENV: str(C.AUTHOR),
        "MERLIN_COMPUTE_CPUS": "0,1,2,3",
        "MERLIN_COMPUTE_CPU_SLOTS": "/nonexistent-slot-root",
    }
    promoted = simjob._job_classes(dict(inherited), promoted=True)
    authored = simjob._job_classes(dict(inherited), promoted=False)
    assert (promoted[C.PRIORITY_ENV], promoted[Q.CLASS_ENV]) == (str(C.GRADE), Q.BACKGROUND)
    assert (authored[C.PRIORITY_ENV], authored[Q.CLASS_ENV]) == (str(C.AUTHOR), Q.AUTHOR)
    with C.cpu_priority(C.AUTHOR):  # the broker's own class does not raise the promotion back
        assert C.child_environment(promoted)[C.PRIORITY_ENV] == str(C.GRADE)
    from merlin.common.paths import repo_root

    source = (repo_root() / "packages/merlin-experiments/src/merlin_experiments/phase1/brokers/simjob.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_job_classes"
    ]
    assert [ast.unparse(node) for node in calls] == ["_job_classes(worker_env, promoted=bool(r.get('promoted')))"]


def _lane(*, diagnostic=False, variant=False):
    return {"diagnostic": diagnostic, "variant": variant}


def test_author_requests_start_past_the_lanes_only_while_the_pool_serves_every_author_lease():
    full = [_lane() for _ in range(4)]
    lanes = dict(diagnostic=False, variant=False, ordinary_capacity=4, variant_capacity=2)
    assert not B._lane_admits(full, **lanes)
    # No author lease queued: one more author request would take a free CPU or a lower class's next one.
    assert B._overflow_admits(full, ordinary_capacity=4, author_backlog=0, pool=4)
    assert B._lane_admits(full, **lanes, overflow=True)
    # Author leases already queue in the pool, or no pool is selected: the dispatcher keeps the order.
    assert not B._overflow_admits(full, ordinary_capacity=4, author_backlog=1, pool=4)
    assert not B._overflow_admits(full, ordinary_capacity=4, author_backlog=None, pool=0)
    # At most one extra handler per pool CPU, for work that never shows up as a queued lease.
    assert not B._overflow_admits(full * 2, ordinary_capacity=4, author_backlog=0, pool=4)
    # The variant bound and the single diagnostic lane are unchanged by overflow.
    variants = [_lane(variant=True), _lane(variant=True), *full]
    assert not B._lane_admits(variants, **dict(lanes, variant=True), overflow=True)
    assert B._lane_admits(full * 2, **dict(lanes, diagnostic=True))
    assert not B._lane_admits([*full, _lane(diagnostic=True)], **dict(lanes, diagnostic=True), overflow=True)
