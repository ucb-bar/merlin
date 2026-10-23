"""Foreground priority never creates qualification; obsolete native work may be deferred."""

import contextlib
import json
import multiprocessing
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import background_admission as admission
from merlin_experiments.phase1.feedback import loop_grading as LG
from merlin_experiments.phase1.feedback import operator_child as OC
from merlin_experiments.phase1.feedback import selfcheck

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import capsule_scheduling as CS
from merlin.targetgen.capsule_scheduling import active_defer_probe, run_pipelined


@pytest.mark.parametrize("dispatched", [False, True])
def test_private_priority_is_bound_to_actual_live_broker_worker(tmp_path, monkeypatch, dispatched):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ws, public = tmp_path / "workspace", tmp_path / "public"
    ws.mkdir()
    public.mkdir()
    archive = tmp_path / "profile.yaml"
    descriptor = tmp_path / "descriptor.yaml"
    archive.write_text("original profile\n")
    descriptor.write_text("original descriptor\n")
    context = SimpleNamespace(descriptor=descriptor)
    inputs = SimpleNamespace(
        context=context, contract=None, compiler_profile_service=SimpleNamespace(bundle_manifest=archive)
    )
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda _: ("fixture", "chipyard"))
    monkeypatch.setattr(selfcheck, "_adapters", lambda *a, **k: ({"L2": object(), "L3": object()}, "gsim"))
    monkeypatch.setattr(CR, "qa_checkpoint_adapters", lambda *a: {"L2": object(), "L3": object()})
    ctx = multiprocessing.get_context("fork")
    ready, stop = ctx.Event(), ctx.Event()

    def broker():
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            with admission.active_full_selfcheck(
                ws=ws,
                process=worker,
                request={"capsules": "all"},
                source_sha256="a" * 64,
                public=public,
                context=context,
                contract=None,
                bundle_manifest=archive,
                sim="gsim",
                dispatcher_pid=os.getppid() if dispatched else None,
            ):
                ready.set()
                stop.wait(5)
        finally:
            worker.terminate()
            worker.wait(timeout=2)

    def dispatcher():
        handler = ctx.Process(target=broker)
        handler.start()
        handler.join(timeout=8)
        if handler.is_alive():
            handler.terminate()
            handler.join(timeout=2)

    process = ctx.Process(target=dispatcher if dispatched else broker)
    process.start()
    try:
        assert ready.wait(3)
        assert admission.foreground_covers(ws, "a" * 64, public, inputs)
        assert not admission.foreground_covers(ws, "b" * 64, public, inputs)
        assert not admission.foreground_covers(ws, "a" * 64, tmp_path / "other", inputs)
        path = admission._marker(ws)
        record = json.loads(path.read_text())
        assert record["manager"]["pid"] == os.getpid()
        assert ("dispatcher" in record) is dispatched
        if dispatched:
            assert record["dispatcher"]["pid"] == process.pid
            assert record["broker"]["parent"] == process.pid
        record["worker"]["birth"] = "wrong birth"
        path.write_text(json.dumps(record))
        assert not admission.foreground_covers(ws, "a" * 64, public, inputs)
    finally:
        stop.set()
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


@pytest.mark.parametrize("dispatched", [False, True])
@pytest.mark.parametrize("damage", ["expired", "birth", "parent", "namespace"])
def test_foreground_priority_reopens_entire_original_process_chain(tmp_path, monkeypatch, dispatched, damage):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    manager_pid, dispatcher_pid, broker_pid, worker_pid = 100, 101, 102, 103
    monkeypatch.setattr(admission.os, "getpid", lambda: manager_pid)
    owners = {
        "manager": (manager_pid, 99),
        "broker": (broker_pid, dispatcher_pid if dispatched else manager_pid),
        "worker": (worker_pid, broker_pid),
    }
    if dispatched:
        owners["dispatcher"] = (dispatcher_pid, manager_pid)
    identities = {
        name: {"pid": pid, "parent": parent, "birth": f"birth-{pid}",
               "pid_namespace": "pid:[original]", "mount_namespace": "mnt:[original]"}
        for name, (pid, parent) in owners.items()
    }
    live = {row["pid"]: dict(row) for row in identities.values()}

    def identity(pid):
        if pid not in live:
            raise ProcessLookupError("original process ended")
        return dict(live[pid])

    monkeypatch.setattr(admission, "_identity", identity)
    profile, descriptor = tmp_path / "profile", tmp_path / "descriptor"
    profile.write_text("original profile")
    descriptor.write_text("original descriptor")
    inputs = SimpleNamespace(context=SimpleNamespace(descriptor=descriptor), contract=None,
                             compiler_profile_service=SimpleNamespace(bundle_manifest=profile))
    marker = admission._marker(tmp_path, create=True)
    record = dict(identities, schema=admission.SCHEMA, source_sha256="a" * 64,
                  public=str(tmp_path.resolve()), descriptor=admission._archive(descriptor),
                  profile=admission._archive(profile), contract=None, tiers=["L2", "L3"])
    marker.write_text(json.dumps(record))
    assert admission.foreground_covers(tmp_path, "a" * 64, tmp_path, inputs)
    # Check each hop, including the extra dispatcher. Mutating both the saved
    # and current parent cannot make a different process tree gain priority.
    for owner in owners:
        saved = {pid: dict(row) for pid, row in live.items()}
        selected = live[identities[owner]["pid"]]
        if damage == "expired":
            del live[selected["pid"]]
        elif damage == "birth":
            selected["birth"] = "replacement-birth"
        elif damage == "namespace":
            selected["mount_namespace"] = "mnt:[foreign]"
        else:
            selected["parent"] = 999
        assert not admission.foreground_covers(tmp_path, "a" * 64, tmp_path, inputs), owner
        if damage == "parent" and owner != "manager":
            changed = json.loads(marker.read_text())
            changed[owner]["parent"] = 999
            marker.write_text(json.dumps(changed))
            assert not admission.foreground_covers(tmp_path, "a" * 64, tmp_path, inputs), owner
            marker.write_text(json.dumps(record))
        live = saved


def test_agent_visible_counts_do_not_admit_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ws = tmp_path / "workspace"
    channel = ws / ".qa_channel"
    channel.mkdir(parents=True)
    (channel / "progress_forged.json").write_text(json.dumps({"n_passed": 146, "all_pass": True}))
    inputs = SimpleNamespace(compiler_profile_service=None)
    assert not admission.foreground_covers(ws, "a" * 64, tmp_path, inputs)


@pytest.mark.parametrize(
    "request_doc",
    [
        {"capsules": "subset"},
        {"capsules": "all", "tiers": "L2"},
        {"capsules": "all", "tiers": "L2,L3,unknown"},
        {"capsules": "all", "offload_census": True},
    ],
)
def test_partial_or_diagnostic_worker_never_issues_foreground_admission(tmp_path, monkeypatch, request_doc):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda _: ("fixture", "chipyard"))
    monkeypatch.setattr(selfcheck, "_adapters", lambda *a, **k: ({"L2": object(), "L3": object()}, "gsim"))
    monkeypatch.setattr(CR, "qa_checkpoint_adapters", lambda *a: {"L2": object(), "L3": object()})
    monkeypatch.setattr(admission, "_identity", lambda _: pytest.fail("unapproved admission"))
    with admission.active_full_selfcheck(
        ws=tmp_path,
        process=SimpleNamespace(pid=123),
        request=request_doc,
        source_sha256="a" * 64,
        public=tmp_path,
        context=SimpleNamespace(),
        contract=None,
        bundle_manifest=tmp_path / "profile",
        sim="gsim",
    ):
        assert not admission._marker(tmp_path).exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_deferral_drains_running_cases_and_retains_original_denominator(tmp_path, workers):
    stop = threading.Event()
    second_started = threading.Event()
    release_second = threading.Event()
    finished = []
    caps = [{"name": name, "kind": "operator"} for name in ("a", "b", "c", "d")]

    def run(cap, parallel):
        if cap["name"] == "a":
            if workers == 2:
                assert second_started.wait(2)
            stop.set()
        elif cap["name"] == "b":
            second_started.set()
            assert release_second.wait(2)
        (tmp_path / f"{cap['name']}.result").write_text("actual completed case")
        return {"capsule": cap["name"], "status": "pass"}

    def completed(row):
        finished.append(row["capsule"])
        if row["capsule"] == "a":
            release_second.set()

    rows = run_pipelined(
        caps,
        run,
        max_workers=workers,
        on_complete=completed,
        defer_pending=lambda: "foreground priority" if stop.is_set() else None,
    )
    assert len(rows) == 4
    assert finished == ["a"] if workers == 1 else set(finished) == {"a", "b"}
    assert all(row["status"] == "pass" for row in rows[:workers])
    assert all(row["status"] == "gated" and row["background_deferred"] for row in rows[workers:])
    assert not (tmp_path / "c.result").exists() and not (tmp_path / "d.result").exists()


def test_background_scheduler_notifies_inflight_workers_without_a_completion():
    started = threading.Barrier(3)
    superseded = threading.Event()
    attempts = []

    def trigger():
        started.wait(timeout=2)
        superseded.set()

    observer = threading.Thread(target=trigger)
    observer.start()

    def run(cap, parallel):
        attempts.append(cap["name"])
        probe = active_defer_probe()
        assert probe is not None
        started.wait(timeout=2)
        while not (reason := probe()):
            superseded.wait(0.02)
        return {"capsule": cap["name"], "status": "gated", "background_deferred": reason}

    try:
        rows = run_pipelined(
            [{"name": name, "kind": "operator"} for name in ("a", "b", "c")], run,
            max_workers=2, defer_pending=lambda: "background source superseded" if superseded.is_set() else None,
        )
    finally:
        observer.join(timeout=2)
    assert set(attempts) == {"a", "b"}
    assert len(rows) == 3
    assert active_defer_probe() is None
    assert all(row["status"] == "gated" and row["background_deferred"] == "background source superseded"
               for row in rows)


def test_operator_child_wait_releases_superseded_work_and_preserves_direct_wait(monkeypatch):
    reason = {"value": None}

    class Process:
        def __init__(self):
            self.timeouts = []

        def communicate(self, *, timeout):
            self.timeouts.append(timeout)
            if len(self.timeouts) == 1:
                reason["value"] = "background source superseded"
                raise subprocess.TimeoutExpired("selected operator child", timeout)
            return ("", "")

    watched = Process()
    monkeypatch.setattr(CS, "active_defer_probe", lambda: lambda: reason["value"])
    assert OC._communicate_or_defer(watched, timeout_s=60) == "background source superseded"
    assert len(watched.timeouts) == 1 and watched.timeouts[0] <= 5

    direct = Process()
    direct.communicate = lambda *, timeout: direct.timeouts.append(timeout)
    monkeypatch.setattr(CS, "active_defer_probe", lambda: None)
    assert OC._communicate_or_defer(direct, timeout_s=60) is None
    assert direct.timeouts == [60]


def test_superseded_operator_child_is_killed_and_persisted_as_deferred(tmp_path, monkeypatch):
    @contextlib.contextmanager
    def prepared(document):
        yield tmp_path / "spec.json", tmp_path / "output.json", {}, SimpleNamespace(argv=(), pass_fds=())

    killed = []
    waited = []

    class Process:
        pid = 417

        def wait(self, *, timeout):
            waited.append(timeout)
            return -9

    monkeypatch.setattr(OC, "_prepared_child", prepared)
    monkeypatch.setattr(OC, "adapter_selection", lambda *args, **kwargs: {"L3": {"engine": "gsim"}})
    monkeypatch.setattr(OC.tier_policy, "observed_cost", lambda *args: None)
    monkeypatch.setattr(OC.cpu_pool, "worker_pool_command", lambda command, environment: command)
    monkeypatch.setattr(OC.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(OC, "_communicate_or_defer", lambda *args, **kwargs: "background source superseded")
    monkeypatch.setattr(CR, "_kill_tree", killed.append)
    monkeypatch.setattr(CR, "_config_for_target", lambda *args: SimpleNamespace(suite="fixture-suite"))
    row = OC.grade(
        {"name": "capacity", "kind": "layer"}, tmp_path / "submission", runs_root=tmp_path / "runs",
        contract=None, target="fixture", suite=None, dtype="i8", adapters={}, timeout=60,
        workers=2, no_oracle=False,
    )
    saved = tmp_path / "runs/runs/fixture-suite/capacity/capsule_result.json"
    assert row["status"] == "gated" and row["background_deferred"] == "background source superseded"
    assert row == json.loads(saved.read_text())
    assert killed == [417] and waited == [10]


def test_superseded_source_is_sticky_and_never_current(tmp_path):
    ws = tmp_path / "workspace"
    source = ws / "submission"
    source.mkdir(parents=True)
    program = source / "compiler.py"
    program.write_text("original\n")
    inputs = SimpleNamespace(compiler_profile_service=None)
    guard = admission.BackgroundGuard(ws, tmp_path, inputs, admission.source_digest(ws))
    assert guard.reason() is None and guard.current()
    program.write_text("replacement\n")
    assert guard.reason() == "background source superseded" and not guard.current()
    program.write_text("original\n")
    assert guard.reason() == "background source superseded"


def test_selected_snapshot_completion_cannot_keep_queuing_obsolete_work(tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_BACKGROUND_FINISH_SNAPSHOT', '1')
    ws = tmp_path / 'workspace'
    (ws/'submission').mkdir(parents=True)
    program = ws/'submission/compiler.py'
    program.write_text('original\n')
    guard = admission.BackgroundGuard(ws, tmp_path,
        SimpleNamespace(compiler_profile_service=None), admission.source_digest(ws))
    program.write_text('replacement\n')
    assert guard.reason() == "background source superseded"
    assert not guard.current()


@pytest.mark.parametrize("finish_snapshot", [False, True])
def test_fast_grade_archives_unknown_rows_without_overwriting_current_feedback(tmp_path, monkeypatch, finish_snapshot):
    monkeypatch.setenv("MERLIN_BACKGROUND_FINISH_SNAPSHOT", str(int(finish_snapshot)))
    ws, run, public = (tmp_path / name for name in ("workspace", "run", "public"))
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    (ws / "qa").mkdir()
    (ws / "qa/verdict.json").write_text('{"current_feedback": true}')
    context = SimpleNamespace(descriptor=tmp_path / "descriptor", repo=tmp_path)
    inputs = LG.GradingInputs(context, "merlin_assisted", public, (), None)
    monkeypatch.setattr(LG, "load_target_experiment", lambda _: SimpleNamespace(target="fixture", sim_via=""))
    monkeypatch.setattr(LG, "compiler_execution", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(LG, "declared_loop_tiers", lambda *a: {"L2", "L3"})
    monkeypatch.setattr(LG, "cert_tiers_beyond_loop", lambda **k: ({"L3"}, {"L3"}))
    monkeypatch.setattr(CR, "qa_loop_adapters", lambda *a, **k: {"L2": object()})
    monkeypatch.setattr(LG, "oracle_kwargs", lambda _: {})
    monkeypatch.setattr(LG, "_record_plateau", lambda *a, **kw: None)
    monkeypatch.setattr(LG.FL, "record_channel_health", lambda *a: None)
    from merlin_experiments.phase1.feedback import promotion, qa

    from merlin.targetgen import capsule_grade as CG

    monkeypatch.setattr(promotion, "resolve_tiers", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        qa, "_per_capsule_from_results", lambda *a: {"finished": {"status": "pass", "tiers": {"L2": "pass"}}}
    )

    def grade(*args, **kwargs):
        (ws / "submission/manifest.yaml").write_text("language: python\nrevision: replaced\n")
        assert kwargs["defer_pending"]() == "background source superseded"
        assert kwargs["defer_running"] is False
        return {
            "per_capsule": [
                {"capsule": "queued", "status": "gated", "background_deferred": "background source superseded"}
            ]
        }

    monkeypatch.setattr(CG, "grade", grade)
    result = LG.fast_grade(ws, run, 900, 30, inputs=inputs)
    assert result["n_capsules"] == 2 and result["all_pass"] is None
    assert result["background_grade"]["status"] == "deferred"
    assert result["snapshot_current"] is False
    assert json.loads((ws / "qa/verdict.json").read_text()) == {"current_feedback": True}
    archived = json.loads((run / "qa_history/verdict_fast_900.json").read_text())
    assert archived["snapshot_current"] is False
    assert archived["graded_request_sha256"] == result["graded_request_sha256"]
