from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from merlin_experiments.phase0 import parallel_replay
from merlin_experiments.phase0.parallel_replay import WORKERS_ENV, Replays


def echo(selection_path, selected_sha256, model_path):
    # Later labels finish first, so any completion-order consumption would be visible.
    time.sleep(0.4 if str(selection_path).endswith("0") else 0.0)
    return {"label": Path(selection_path).name}, {"pid": os.getpid()}, {"sha": selected_sha256}


def fail_odd(selection_path, selected_sha256, model_path):
    name = Path(selection_path).name
    if int(name) % 2:
        raise ValueError(f"replay of {name} failed")
    return echo(selection_path, selected_sha256, model_path)


class Unportable(Exception):
    def __init__(self, a, b):  # two required args: the default exception pickle round trip fails
        super().__init__(a)


def unportable_once(selection_path, selected_sha256, model_path):
    if os.getpid() != int(os.environ["PARENT_PID"]):
        raise Unportable(1, 2)
    return echo(selection_path, selected_sha256, model_path)


def jobs(count=6):
    return [(Path(str(index)), f"{index:064d}", Path(f"m{index}")) for index in range(count)]


@pytest.mark.parametrize("workers", ["0", "1", "4"])
def test_results_follow_declared_order(monkeypatch, workers):
    monkeypatch.setenv(WORKERS_ENV, workers)
    with Replays(jobs(), replay=echo) as replays:
        got = [replays[index] for index in reversed(range(6))][::-1]
    assert [row[0]["label"] for row in got] == [str(index) for index in range(6)]
    assert [row[2]["sha"] for row in got] == [f"{index:064d}" for index in range(6)]
    if workers == "4":
        assert {row[1]["pid"] for row in got} != {os.getpid()}


@pytest.mark.parametrize("workers", ["0", "4"])
def test_failure_surfaces_at_its_label_with_the_original_type(monkeypatch, workers):
    monkeypatch.setenv(WORKERS_ENV, workers)
    with Replays(jobs(), replay=fail_odd) as replays:
        assert replays[0][0]["label"] == "0"
        with pytest.raises(ValueError, match="replay of 1 failed"):
            replays[1]
        assert replays[2][0]["label"] == "2"
        with pytest.raises(ValueError, match="replay of 3 failed"):
            replays[3]


def test_serial_mode_replays_lazily_in_the_caller(monkeypatch):
    monkeypatch.setenv(WORKERS_ENV, "1")
    seen = []

    def record(selection_path, selected_sha256, model_path):
        seen.append(selection_path.name)
        return {}, {}, {}

    with Replays(jobs(3), replay=record) as replays:
        assert seen == []
        replays[1]
        assert seen == ["1"]


def test_unportable_worker_error_is_repeated_in_process(monkeypatch):
    monkeypatch.setenv(WORKERS_ENV, "2")
    monkeypatch.setenv("PARENT_PID", str(os.getpid()))
    with Replays(jobs(2), replay=unportable_once) as replays:
        assert replays[1][0]["label"] == "1"
        assert replays[1][1]["pid"] == os.getpid()


def test_worker_count_bounds(monkeypatch):
    monkeypatch.delenv(WORKERS_ENV, raising=False)
    assert parallel_replay.worker_count(3) <= 3
    monkeypatch.setenv(WORKERS_ENV, "1000")
    assert parallel_replay.worker_count(5) <= 5
    monkeypatch.setenv(WORKERS_ENV, "x")
    with pytest.raises(ValueError):
        parallel_replay.worker_count(5)


def test_missing_selection_fails_exactly_as_before(monkeypatch, tmp_path):
    from merlin_experiments.phase0.parallel_replay import replay_one

    job = (tmp_path / "absent/capture-selection.json", "0" * 64, tmp_path / "model.mlir")
    with pytest.raises(ValueError) as serial:
        replay_one(*job)
    monkeypatch.setenv(WORKERS_ENV, "2")
    with Replays([job, job]) as replays, pytest.raises(type(serial.value)) as pooled:
        replays[0]
    assert str(pooled.value) == str(serial.value)
