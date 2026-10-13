"""Foreground priority never creates qualification or interrupts verified cases."""

import contextlib
import json
import multiprocessing
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import background_admission as admission
from merlin_experiments.phase1.feedback import loop_grading as LG
from merlin_experiments.phase1.feedback import selfcheck

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.capsule_scheduling import run_pipelined


def test_private_priority_is_bound_to_actual_live_broker_worker(tmp_path, monkeypatch):
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
            ):
                ready.set()
                stop.wait(5)
        finally:
            worker.terminate()
            worker.wait(timeout=2)

    process = ctx.Process(target=broker)
    process.start()
    try:
        assert ready.wait(3)
        assert admission.foreground_covers(ws, "a" * 64, public, inputs)
        assert not admission.foreground_covers(ws, "b" * 64, public, inputs)
        assert not admission.foreground_covers(ws, "a" * 64, tmp_path / "other", inputs)
        path = admission._marker(ws)
        record = json.loads(path.read_text())
        record["worker"]["birth"] = "wrong birth"
        path.write_text(json.dumps(record))
        assert not admission.foreground_covers(ws, "a" * 64, public, inputs)
    finally:
        stop.set()
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


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


def test_fast_grade_archives_unknown_rows_without_overwriting_current_feedback(tmp_path, monkeypatch):
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
    monkeypatch.setattr(LG, "readback_kwargs", lambda _: {})
    monkeypatch.setattr(LG, "_record_plateau", lambda *a: None)
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
        return {
            "per_capsule": [
                {"capsule": "queued", "status": "gated", "background_deferred": "background source superseded"}
            ]
        }

    monkeypatch.setattr(CG, "grade", grade)
    result = LG.fast_grade(ws, run, 900, 30, inputs=inputs)
    assert result["n_capsules"] == 2 and result["all_pass"] is None
    assert result["background_grade"]["status"] == "deferred"
    assert json.loads((ws / "qa/verdict.json").read_text()) == {"current_feedback": True}
    assert (run / "qa_history/verdict_fast_900.json").is_file()
