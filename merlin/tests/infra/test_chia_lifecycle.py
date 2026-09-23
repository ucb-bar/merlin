"""Driver resource ownership must not leak across canonical experiment runs."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.benchharness import chia_bridge
from merlin.common import artifacts


@pytest.fixture
def services(tmp_path, monkeypatch):
    events = []
    state = {"initialized": False, "collector": None, "failure": None, "resources": {}}
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: state["initialized"]
    ray.cluster_resources = lambda: state["resources"]

    def initialize(**kwargs):
        events.append(("ray.start", kwargs))
        if state["failure"] == "ray":
            raise RuntimeError("ray setup failed")
        state["initialized"] = True
        if state["failure"] == "ray_partial":
            raise RuntimeError("ray setup failed after init")

    def shutdown():
        events.append(("ray.stop",))
        state["initialized"] = False

    ray.init, ray.shutdown = initialize, shutdown
    chia = types.ModuleType("chia")
    trace = types.ModuleType("chia.trace")

    def start_collector(*, log_dir, **kwargs):
        if state["failure"] == "collector":
            raise RuntimeError("collector setup failed")
        # Match upstream start_collector: existing collector is reused, not retargeted.
        if state["collector"] is None:
            state["collector"] = Path(log_dir)
        if state["failure"] == "collector_partial":
            raise RuntimeError("collector setup failed after actor creation")
        events.append(("collector.start", state["collector"]))

    def stop_collector():
        events.append(("collector.stop", state["collector"], os.environ.get("CHIA_AET_SINK")))
        state["collector"] = None
        if state["failure"] == "stop":
            raise RuntimeError("collector cleanup failed")

    class Metrics:
        def __init__(self, **kwargs):
            if state["failure"] == "metrics":
                raise RuntimeError("metrics setup failed")
            events.append(("metrics.start",))

        def close(self):
            events.append(("metrics.stop",))
            if state["failure"] == "close":
                raise RuntimeError("metrics cleanup failed")

    trace.start_collector, trace.stop_collector, trace.MetricsLogger = start_collector, stop_collector, Metrics
    trace.get_collector = lambda: state["collector"]
    profiler = types.ModuleType("chia.trace.profiler")
    profiler.reset_profiler = lambda: events.append(("profiler.reset",))
    profiler.get_collector = trace.get_collector
    profiler.start_collector = start_collector
    profiler.stop_collector = stop_collector
    chia.trace = trace
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "chia", chia)
    monkeypatch.setitem(sys.modules, "chia.trace", trace)
    monkeypatch.setitem(sys.modules, "chia.trace.profiler", profiler)
    monkeypatch.delenv("CHIA_AET_SINK", raising=False)
    monkeypatch.setattr(chia_bridge, "require_chia", lambda: None)
    monkeypatch.setattr(chia_bridge, "_aet_backend_cls", lambda: Metrics)
    monkeypatch.setattr(
        chia_bridge,
        "_record_trace",
        lambda handle, **kwargs: events.append(("trace.record", handle, kwargs)),
    )

    def start_run(**kwargs):
        events.append(("aet.start", kwargs["run_id"]))
        run_dir = tmp_path / kwargs["run_id"]
        run_dir.mkdir()
        spec = SimpleNamespace(project="merlin", suite="fixture/test", target="fixture", method="test", seed=0)
        (run_dir / "run_record.json").write_text(
            json.dumps(
                {
                    "created_at": "original",
                    "git_sha": "commit",
                    "timestamp": "first",
                    "custom": {"nested": [1, 2]},
                    "run_id": kwargs["run_id"],
                }
            )
        )
        return SimpleNamespace(run_id=kwargs["run_id"], run_dir=run_dir, spec=spec)

    def finish_run(handle, **kwargs):
        events.append(("aet.finish", handle.run_id, kwargs))

    monkeypatch.setattr(artifacts, "start_run", start_run)
    monkeypatch.setattr(artifacts, "finish_run", finish_run)
    return state, events


def test_sequential_runs_have_distinct_profiler_owners(services):
    state, events = services
    for name in ("first", "second"):
        with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id=name) as run:
            assert state["collector"] == run.run_dir / "chia"
    assert len([row for row in events if row[0] == "collector.stop"]) == 2
    assert state["collector"] is None
    assert state["initialized"] is False


@pytest.mark.parametrize("failure", ["ray", "ray_partial", "collector", "collector_partial", "metrics"])
def test_setup_failures_finish_the_aet_run(services, failure):
    state, events = services
    state["failure"] = failure
    with pytest.raises(RuntimeError, match="setup failed"):
        with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id="failed"):
            pytest.fail("setup failure must not enter the workflow")
    finished = [row for row in events if row[0] == "aet.finish"]
    assert len(finished) == 1
    assert finished[0][2]["status"] == "error"
    assert state["collector"] is None
    assert state["initialized"] is False


def test_borrowed_ray_is_validated_and_not_stopped(services):
    state, events = services
    state.update(initialized=True, resources={"compiler": 2})
    with chia_bridge.chia_run(
        suite="test", method="test", target="fixture", run_id="borrowed", ray_resources={"compiler": 1}
    ):
        pass
    assert state["initialized"] is True
    assert not any(row[0] in {"ray.start", "ray.stop"} for row in events)
    with pytest.raises(RuntimeError, match="resources"):
        with chia_bridge.chia_run(
            suite="test", method="test", target="fixture", run_id="short", ray_resources={"compiler": 3}
        ):
            pytest.fail("cannot schedule requested capacity")
    assert state["initialized"] is True


def test_existing_collector_is_never_borrowed_or_stopped(services, tmp_path):
    state, events = services
    state.update(initialized=True, collector=tmp_path / "other")
    with pytest.raises(RuntimeError, match="collector"):
        with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id="refused"):
            pytest.fail("must not append to another workflow's profile")
    assert state["collector"] == tmp_path / "other"
    assert not any(row[0] in {"collector.stop", "profiler.reset", "ray.stop"} for row in events)
    assert events[-1][0] == "aet.finish"


@pytest.mark.parametrize("failure", ["close", "stop"])
@pytest.mark.parametrize("body_fails", [False, True])
def test_cleanup_failures_finish_and_preserve_primary_error(services, failure, body_fails):
    state, events = services
    state["failure"] = failure
    with pytest.raises(RuntimeError, match="body failed" if body_fails else "cleanup failed"):
        with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id="cleanup"):
            if body_fails:
                raise RuntimeError("body failed")
    assert state["initialized"] is False
    assert events[-1][0] == "aet.finish"
    assert events[-1][2]["status"] == "error"
    assert [row[0] for row in events].count("profiler.reset") == 2


def test_explicit_sink_preserves_canonical_identity_metadata_and_failed_status(services, monkeypatch):
    _, events = services
    monkeypatch.setenv("CHIA_AET_SINK", "1")
    monkeypatch.setenv("CHIA_AET_RUN_DIR", "/unrelated/no-write")
    monkeypatch.setenv("CHIA_AET_RUN_ID", "inherited-wrong")
    with pytest.raises(RuntimeError, match="body failed"):
        with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id="canonical") as run:
            raise RuntimeError("body failed")
    calls = [row[1] for row in events if row[0] == "trace.record"]
    assert len(calls) == 1
    assert calls[0] is run.handle
    record = json.loads((run.run_dir / "run_record.json").read_text())
    assert record["created_at"] == "original"
    assert record["custom"] == {"nested": [1, 2]}
    assert record["git_sha"] == "commit"
    assert record["timestamp"] == "first"
    assert events[-1] == ("aet.finish", "canonical", {"status": "error", "summary": None})
    assert next(row[2] for row in events if row[0] == "collector.stop") == "0"
    assert os.environ["CHIA_AET_SINK"] == "1"


def test_sink_is_opt_in_and_profiler_is_reset_around_collector(services):
    _, events = services
    with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id="plain"):
        pass
    names = [row[0] for row in events]
    assert "trace.record" not in names
    assert names.index("profiler.reset") < names.index("collector.start")
    assert names.index("collector.stop") < max(i for i, name in enumerate(names) if name == "profiler.reset")
    assert "CHIA_AET_SINK" not in os.environ


def test_partial_uncached_collector_in_borrowed_ray_is_killed(services, monkeypatch):
    state, events = services
    state.update(initialized=True, failure="collector_partial")
    # Upstream stop only knows its cached actor, but get_collector can find the
    # actor by name when readiness failed before that cache was populated.
    monkeypatch.setattr(sys.modules["chia.trace.profiler"], "stop_collector", lambda: None)

    def kill(actor):
        events.append(("orphan.kill", actor))
        state["collector"] = None

    monkeypatch.setattr(sys.modules["ray"], "kill", kill, raising=False)
    with pytest.raises(RuntimeError, match="actor creation"):
        with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id="partial"):
            pytest.fail("partial startup cannot enter workflow")
    assert state["collector"] is None
    assert state["initialized"] is True
    assert any(row[0] == "orphan.kill" for row in events)
    assert events[-1][0] == "aet.finish"


def test_finish_failure_does_not_mask_workflow_error(services, monkeypatch):
    state, _ = services

    def fail_finish(*args, **kwargs):
        raise ValueError("AET finalization failed")

    monkeypatch.setattr(artifacts, "finish_run", fail_finish)
    with pytest.raises(RuntimeError, match="workflow") as caught:
        with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id="finish"):
            raise RuntimeError("workflow failed")
    assert any("AET finalization" in note for note in caught.value.__notes__)
    assert state["initialized"] is False
    assert state["collector"] is None


def test_failed_child_outcome_survives_normal_context_exit_and_upstream_sink(services, monkeypatch):
    state, events = services
    monkeypatch.setenv("CHIA_AET_SINK", "1")
    with chia_bridge.chia_run(suite="test", method="test", target="fixture", run_id="child-failed") as run:
        run.summary["returncode"] = 7
        run.mark_failed()
        run.mark_failed()  # Multiple failed child tasks cannot reset the outcome.
    assert any(row[0] == "trace.record" for row in events)
    assert events[-1] == (
        "aet.finish",
        "child-failed",
        {"status": "error", "summary": {"returncode": 7}},
    )
    assert state["initialized"] is False
    assert state["collector"] is None
