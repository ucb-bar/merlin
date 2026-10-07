"""A measured run's heartbeat, the STALLED verdict ``status`` reads from its records, and the watchdog's
once-per-episode stall record and notification."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase2.whole_model_measured import launch as LAUNCH
from merlin_experiments.phase2.whole_model_measured import liveness as L
from merlin_experiments.phase2.whole_model_measured import progress as P
from merlin_experiments.phase2.whole_model_measured import watchdog as WD
from merlin_experiments.phase2.whole_model_measured.identity import epoch_of, write_json_atomic

from merlin.perf import whole_model_verdict as V

T0 = epoch_of("20261005T120000Z")
HOUR = 3600.0


def _measure(store: Path, key: str, at: str, status: str = V.TIMING_MEASURED) -> None:
    (store / key).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        store / key / "result.json",
        {"package_sha256": key, "timing_status": status, "finished_at": at, "verdict": {"whole_window_cycles": 9}},
    )


def _run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir, store = tmp_path / "run", tmp_path / "store"
    run_dir.mkdir()
    store.mkdir()
    write_json_atomic(run_dir / "resumed_seed.json", {"store_roots": {"screen": str(store)}})
    return run_dir, store


def test_the_heartbeat_names_its_launcher_its_activity_and_the_newest_measurement(tmp_path):
    run_dir, store = _run(tmp_path)
    _measure(store, "a" * 64, "20261005T100000Z")
    _measure(store, "b" * 64, "20261005T110000Z", status=V.TIMING_REFUSED)  # a refusal is not a measurement
    (store / "c" / "attempts" / "0").mkdir(parents=True)
    _measure(store / "c" / "attempts", "0", "20261005T103000Z", status=V.TIMING_MEASURED_INVALID)
    clock = iter([T0, T0 + 10, T0 + 120])
    beat = L.Heartbeat(run_dir, stores=[store], clock=lambda: next(clock))
    first = beat.tick("launched", force=True)
    assert first["launcher"]["pid"] == os.getpid() and first["launcher"]["start_ticks"] == L.start_ticks(os.getpid())
    assert first["last_measured"]["package_sha256"] == "0" and first["last_measured"]["at"] == "20261005T103000Z"
    assert beat.tick("poll") is None  # within the interval: no write
    again = beat.tick("poll")
    assert again["beats"] == 2 and again["last_activity"]["what"] == "poll"
    assert json.loads((run_dir / L.HEARTBEAT).read_text())["schema"] == L.HEARTBEAT_SCHEMA


def test_no_measurement_within_the_threshold_is_stalled_and_a_recent_one_is_live(tmp_path):
    run_dir, store = _run(tmp_path)
    _measure(store, "a" * 64, "20261005T100000Z")
    L.Heartbeat(run_dir, stores=[store], clock=lambda: T0).tick("poll", force=True)  # this process: alive
    live = L.assess(run_dir, stall_hours=6, clock=lambda: T0)
    assert live["state"] == L.LIVE and live["launcher_alive"] is True and live["hours_since_measured"] == 2.0
    stalled = L.assess(run_dir, stall_hours=6, clock=lambda: T0 + 5 * HOUR)
    assert stalled["state"] == L.STALLED and "7.0 h since the last measured candidate" in stalled["reasons"][0]
    # MUTATION: the same records under a longer threshold are live again -- the threshold is what decides.
    assert L.assess(run_dir, stall_hours=8, clock=lambda: T0 + 5 * HOUR)["state"] == L.LIVE


def test_a_dead_launcher_is_stalled_even_when_it_measured_recently(tmp_path):
    run_dir, store = _run(tmp_path)
    _measure(store, "a" * 64, "20261005T115000Z")
    write_json_atomic(run_dir / LAUNCH.LAUNCH_RECORD, {"pid": 2**22 + 12345, "started_at": "20261005T080000Z"})
    verdict = L.assess(run_dir, stall_hours=6, clock=lambda: T0)
    assert verdict["state"] == L.STALLED and verdict["launcher_alive"] is False
    assert "is not running" in verdict["reasons"][0]


def test_a_reused_pid_is_not_the_launcher(tmp_path):
    run_dir, store = _run(tmp_path)
    _measure(store, "a" * 64, "20261005T115000Z")
    write_json_atomic(
        run_dir / L.HEARTBEAT, {"launcher": {"pid": os.getpid(), "start_ticks": "not-this-process"}, "beats": 1}
    )
    assert L.assess(run_dir, stall_hours=6, clock=lambda: T0)["launcher_alive"] is False


def test_a_run_that_recorded_its_stop_is_stopped_not_stalled(tmp_path):
    run_dir, _store = _run(tmp_path)
    (run_dir / "stage").mkdir()
    write_json_atomic(run_dir / "stage" / "sessions.json", {"stopped": {"kind": "plateau", "reason": "6 h"}})
    assert L.assess(run_dir, stall_hours=1, clock=lambda: T0 + 100 * HOUR)["state"] == L.STOPPED


def test_with_nothing_measured_the_age_counts_from_the_launch(tmp_path):
    run_dir, _store = _run(tmp_path)
    L.Heartbeat(run_dir, stores=[], clock=lambda: T0).tick("launched", force=True)
    write_json_atomic(run_dir / LAUNCH.LAUNCH_RECORD, {"pid": os.getpid(), "started_at": "20261005T120000Z"})
    verdict = L.assess(run_dir, stall_hours=2, clock=lambda: T0 + 3 * HOUR)
    assert verdict["state"] == L.STALLED and "since the launch" in verdict["reasons"][-1]


def test_a_stall_is_recorded_once_per_episode_and_announced(tmp_path):
    run_dir, _store = _run(tmp_path)
    out = tmp_path / "notified.txt"
    command = [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write(sys.argv[2])", str(out), "{reason}"]
    stalled = {"state": L.STALLED, "reasons": ["7.0 h since the last measured candidate"]}
    first = L.record_transition(run_dir, stalled, notify_command=command)
    assert first["event"] == L.STALLED and first["notify"]["returncode"] == 0
    assert out.read_text() == "7.0 h since the last measured candidate"
    assert L.record_transition(run_dir, stalled, notify_command=command) is None  # the same episode
    assert L.record_transition(run_dir, {"state": L.LIVE})["event"] == "RECOVERED"
    assert L.record_transition(run_dir, stalled)["event"] == L.STALLED  # a new episode
    assert [e["event"] for e in L.events(run_dir)] == [L.STALLED, "RECOVERED", L.STALLED]


def test_the_watchdog_records_a_stall_while_the_launcher_lives(tmp_path, monkeypatch):
    run_dir, store = _run(tmp_path)
    _measure(store, "a" * 64, "20200101T000000Z")  # long ago
    alive = iter([True, True, False])
    WD.watch(
        run_dir,
        1,
        launch=lambda p: pytest.fail("relaunched"),
        why="x",
        policy=WD.WatchPolicy(stall_hours=1, max_relaunches=0),
        is_alive=lambda pid: next(alive),
        sleep=lambda s: None,
        log=lambda line: None,
    )
    kinds = [e["event"] for e in L.events(run_dir)]
    assert kinds[0] == L.STALLED and kinds.count(L.STALLED) == 1  # once, though it was seen at every poll


def test_a_launcher_the_watchdog_will_not_relaunch_is_a_stall_on_record(tmp_path):
    run_dir, _store = _run(tmp_path)
    document = WD.watch(
        run_dir,
        1,
        launch=lambda p: pytest.fail("relaunched"),
        why="x",
        is_alive=lambda pid: False,
        sleep=lambda s: None,
        clock=lambda: 0.0,
        log=lambda line: None,
    )
    assert document["stopped"]["kind"] == "crash_loop"
    (event,) = L.events(run_dir)
    assert event["event"] == L.STALLED and "was not relaunched" in event["reasons"][0]


def test_status_reports_the_stall_first(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    run_dir, store = _run(tmp_path)
    _measure(store, "a" * 64, "20200101T000000Z")
    write_json_atomic(run_dir / "run.json", {"target": "toy", "method": "m"})
    document = P.run_status(run_dir, stall_hours=1)
    assert document["liveness"]["state"] == L.STALLED
    assert P.format_status(document).splitlines()[2].startswith("  STALLED: last measured candidate 20200101T000000Z")


def test_the_objectives_poll_and_each_session_start_beat(tmp_path):
    from merlin_experiments.phase2.whole_model_measured import objective as O

    class _Screen:
        root = tmp_path / "store"
        machine = {"kind": "spike"}

        def poll(self):
            return []

        def jobs(self):
            return []

    class _Beat:
        def __init__(self):
            self.ticks = []

        def tick(self, what, *, force=False):
            self.ticks.append((what, force))

    objective = O.WholeModelObjective(screen=_Screen(), screen_reference=None)
    objective.heartbeat = _Beat()
    objective.mark_session(T0)
    objective.poll()
    assert objective.heartbeat.ticks == [("session started", True), ("poll", False)]
