"""Real pinned Chia/AET accounting, without agents, network calls or a Ray cluster.

Run in the isolated ``merlin-experiments[chia]`` environment. Only the cluster
transport is replaced; scalar logging, usage folding and AET files are real.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from tempfile import TemporaryDirectory

import pytest

pytest.importorskip("chia.trace.profiler")


@pytest.mark.parametrize("profiled", [False, True])
def test_actual_public_get_batch_and_callback_transport_without_ray(monkeypatch, profiled):
    """Exercise upstream scalar get, replacing only Ray and profiler transport.

    This is not a real-Ray profiled trampoline qualification.
    """
    from types import SimpleNamespace

    import ray
    from chia.base.ChiaFunction import ObjectRefCallback
    from chia.trace import profiler

    from merlin.benchharness.chia_bridge import chia_get

    seen = []

    def resolve(ref, **kwargs):
        seen.append(ref)
        return {"value": ref} if profiled else ref

    monkeypatch.setattr(ray, "get", resolve)
    monkeypatch.setattr(
        profiler,
        "get_profiler",
        lambda: SimpleNamespace(
            on_remote_complete=lambda value: value["value"] if profiled else value,
        ),
    )
    callback = ObjectRefCallback("b", lambda value: value.upper())
    assert chia_get(["a", callback, "a"], timeout=2, callback=tuple) == ("a", "B", "a")
    assert seen == ["a", "b", "a"]


@pytest.mark.parametrize("outcome", ["ok", "exception", "child_failed"])
def test_canonical_run_with_real_upstream_accounting(tmp_path, monkeypatch, outcome):
    from types import SimpleNamespace

    import ray
    from chia.trace import profiler as trace

    from merlin.benchharness import chia_bridge

    chia_bridge.require_chia()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CHIA_AET_SINK", "1")
    monkeypatch.setenv("CHIA_AET_RUN_DIR", str(tmp_path / "wrong-owner"))
    state = {"ray": False, "collector": None}
    monkeypatch.setattr(ray, "is_initialized", lambda: state["ray"])
    monkeypatch.setattr(ray, "init", lambda **kw: state.update(ray=True))
    monkeypatch.setattr(ray, "shutdown", lambda: state.update(ray=False))
    monkeypatch.setattr(trace, "get_collector", lambda: state["collector"])
    monkeypatch.setattr(trace, "start_collector", lambda **kw: state.update(collector=kw["log_dir"]))
    monkeypatch.setattr(trace, "stop_collector", lambda: state.update(collector=None))
    events = [
        {
            "type": "complete",
            "call_id": "metered",
            "ts": 2.0,
            "extra": {
                "input_tokens": 10,
                "output_tokens": 3,
                "cost_usd": 0.25,
                "cost_source": "billed",
                "billing_mode": "per_token",
                "model": "fixture-model",
            },
        },
        {
            "type": "local_end",
            "call_id": "subscription",
            "ts": 3.0,
            "extra": {
                "input_tokens": 20,
                "output_tokens": 4,
                "cost_usd": 1.25,
                "cost_source": "billed",
                "billing_mode": "subscription",
                "model": "fixture-model",
            },
        },
    ]
    events.append(events[0])  # Merlin deduplicates public call metadata; AET splits billing.
    collector = SimpleNamespace(get_events=SimpleNamespace(remote=lambda: events))
    monkeypatch.setattr(ray, "get", lambda value, **kwargs: value)
    monkeypatch.setattr(trace, "start_collector", lambda **kw: state.update(collector=collector))
    for index in range(2):
        try:
            with chia_bridge.chia_run(
                suite="integration",
                method="fixture",
                target="fixture",
                run_id=f"sequential-{index}",
                extra={"input_pin": "original"},
            ) as run:
                original = json.loads((run.run_dir / "run_record.json").read_text())
                run.metrics.log_scalar("fixture/wall_s", 0.5, 0)
                if outcome == "child_failed":
                    run.mark_failed()
                if outcome == "exception":
                    raise RuntimeError("workflow failed")
        except RuntimeError as exc:
            assert outcome == "exception" and str(exc) == "workflow failed"
        record = json.loads((run.run_dir / "run_record.json").read_text())
        for key, value in original.items():
            assert record[key] == value, key
        assert record["run_id"] == run.run_id
        accounting = json.loads((run.run_dir / "chia/accounting.json").read_text())
        assert accounting["coverage"] == "observed_only_not_complete"
        assert len(accounting["calls"]) == 2
        metrics = [json.loads(line) for line in (run.run_dir / "logs" / "metrics.jsonl").read_text().splitlines()]
        values = {item["name"]: item["value"] for item in metrics}
        assert values["aet.agent.cost_usd"] == 0.25
        assert values["chia.subscription.cost_equivalent_usd"] == 1.25
        logged = [json.loads(line) for line in (run.run_dir / "logs" / "events.jsonl").read_text().splitlines()]
        finished = [event for event in logged if event["event"] == "run.finished"]
        assert finished[-1]["payload"]["status"] == ("ok" if outcome == "ok" else "error")
        assert state == {"ray": False, "collector": None}
    assert not (tmp_path / "wrong-owner").exists()


@pytest.mark.skipif(os.environ.get("MERLIN_TEST_CHIA_RAY") != "1", reason="opt-in local Ray runtime qualification")
def test_real_local_ray_sequential_workflows(tmp_path, monkeypatch):
    """No LLMs or hardware: qualify collector ownership and public batched get."""
    if sys.platform != "linux" or {name for _, name in socket.if_nameindex()} != {"lo"}:
        pytest.fail("Ray qualification requires a loopback-only network namespace/container (--network=none)")
    import ray
    from chia.base.ChiaFunction import ChiaFunction
    from chia.trace.profiler import get_collector

    from merlin.benchharness.chia_bridge import chia_get, chia_run

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    monkeypatch.delenv("CHIA_AET_SINK", raising=False)
    if ray.is_initialized():
        pytest.fail("runtime qualification requires its own process")

    @ChiaFunction(max_retries=0, resources={"fixture_slot": 1})
    def increment(value):
        return value + 1

    # Ray uses Unix sockets; pytest's long scratch path can exceed their limit.
    with TemporaryDirectory(prefix="mc-", dir="/tmp") as ray_temp:
        monkeypatch.setenv("RAY_TMPDIR", ray_temp)
        try:
            ray.init(
                address="local",
                num_cpus=1,
                resources={"fixture_slot": 1},
                include_dashboard=False,
                object_store_memory=80 * 1024 * 1024,
            )
            for index in range(2):
                with chia_run(
                    suite="ray-contract",
                    method="fixture",
                    target="fixture",
                    run_id=f"local-{index}",
                    ray_resources={"fixture_slot": 1},
                ) as run:
                    assert chia_get([increment.chia_remote(1), increment.chia_remote(2)], timeout=30) == [2, 3]
                    run.metrics.log_scalar("fixture/completed", 2, 0)
                assert ray.is_initialized(), "bridge must leave borrowed cluster alive"
                assert get_collector() is None
                assert run.profile_path.is_file()
        finally:
            ray.shutdown()
