"""The run watchdog samples a synthetic /proc and cgroup tree, holds warnings, and keeps the marker."""

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
UNIT = "merlinphase1fixture"


@pytest.fixture
def W(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_watchdog_test", HERE / "run_watchdog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROC", tmp_path / "proc")
    monkeypatch.setattr(module, "CGROUP", tmp_path / "cgroup")
    monkeypatch.setattr(module, "MARKER", tmp_path / "work/RUN_IN_PROGRESS.json")
    return module


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def stat_line(pid, comm, state, start_ticks=0):
    fields = [state] + ["0"] * 18 + [str(start_ticks)] + ["0"] * 10
    return f"{pid} ({comm}) {' '.join(fields)}\n"


def host(tmp_path, *, io_full=0.0, squashfs=0):
    proc = tmp_path / "proc"
    write(
        proc / "pressure/io",
        f"some avg10=1.00 avg60=2.00 avg300=1.00 total=1\nfull avg10=0 avg60={io_full} avg300=0 total=1\n",
    )
    write(
        proc / "pressure/cpu",
        "some avg10=50.00 avg60=50.00 avg300=50.00 total=1\nfull avg10=0 avg60=0 avg300=0 total=0\n",
    )
    mounts = ["22 1 259:1 / / rw - ext4 /dev/root rw"]
    mounts += [f"{30 + i} 22 7:{i} / /snap/x/{i} ro - squashfs /dev/loop{i} ro" for i in range(squashfs)]
    write(proc / "self/mountinfo", "\n".join(mounts) + "\n")
    write(proc / "uptime", "1000.00 1.00\n")
    # A lease table the watchdog must never open.
    write(proc / "locks", "1: LEASE ACTIVE READ 1 00:01:1 0 EOF\n")


def unit(tmp_path, *, usage_usec, processes):
    group = tmp_path / "cgroup/merlinphase1.slice" / f"{UNIT}.service"
    write(group / "cpu.stat", f"usage_usec {usage_usec}\nuser_usec 0\n")
    write(group / "cpuset.cpus.effective", "8-11\n")
    write(group / "cgroup.procs", "".join(f"{pid}\n" for pid in processes))
    for pid, (comm, state, wchan, start) in processes.items():
        write(tmp_path / "proc" / str(pid) / "stat", stat_line(pid, comm, state, start))
        write(tmp_path / "proc" / str(pid) / "wchan", wchan)


def test_sample_counts_waiters_and_busy_share(W, tmp_path, monkeypatch):
    host(tmp_path)
    tick = os.sysconf("SC_CLK_TCK")
    procs = {100 + i: ("python3.12", "D", "percpu_rwsem_wait", 0) for i in range(12)}
    procs |= {200 + i: ("bwrap", "D", "m_start", 900 * tick) for i in range(3)}  # started 100 s ago
    procs[300] = ("spike", "R", "0", 0)
    unit(tmp_path, usage_usec=0, processes=procs)
    first = W.sample_unit(UNIT, None, at=100.0)
    unit(tmp_path, usage_usec=40_000_000, processes=procs)  # 40 CPU-s over 4 CPUs in 100 s
    second = W.sample_unit(UNIT, first, at=200.0)
    assert first["busy_share"] is None and second["busy_share"] == 0.1
    assert second["lock_waiters"] == 12 and second["mount_waiters"] == 3
    assert second["sandbox_d_state"] == 3 and second["sandbox_d_median_age_s"] == 100.0
    assert second["states"] == {"D": 15, "R": 1}
    assert W.sample_unit("merlinphase1absent", None, at=0) is None


def test_conditions_name_each_problem_in_plain_language(W, tmp_path):
    host(tmp_path, io_full=42.0, squashfs=160)
    sample = W.host_sample()
    assert sample["mount_count"] == 160 and sample["io_full_avg60"] == 42.0
    unit_sample = {
        "unit": UNIT,
        "busy_share": 0.05,
        "lock_waiters": 20,
        "mount_waiters": 0,
        "sandbox_d_state": 0,
        "sandbox_d_median_age_s": None,
    }
    at = time.time()
    stale = [{"graded_at": W.dt.datetime.fromtimestamp(at - 1800, W.dt.timezone.utc).isoformat()}]
    found = W.conditions(unit_sample, stale, sample, started=at - 7200, at=at)
    assert set(found) == {"io_pressure", "mount_count", "lock_waiters", "idle_stale"}
    assert "30 min old" in found["idle_stale"] and "/proc/locks" in found["lock_waiters"]
    # A young run has no grade yet: its age counts from the reservation start, not from never.
    young = W.conditions(unit_sample, [{"graded_at": None}], sample, started=at - 600, at=at)
    assert "idle_stale" not in young


def test_watch_once_reports_live_fixture_without_reading_proc_locks(W, tmp_path, monkeypatch, capsys):
    host(tmp_path, squashfs=200)
    procs = {100: ("python3.12", "S", "do_poll", 0)}
    unit(tmp_path, usage_usec=0, processes=procs)
    reads = []
    original = W._read
    monkeypatch.setattr(W, "_read", lambda path: reads.append(path) or original(path))
    monkeypatch.setattr(W, "unit_state", lambda name: (None, {}))
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    assert W.watch([UNIT], interval=60, once=True, out=None) == 0
    out = capsys.readouterr().out
    assert "WARNING 200 squashfs/loop mounts" in out
    assert tmp_path / "proc/locks" not in reads
    with pytest.raises(PermissionError):
        W._read(tmp_path / "proc/locks")


def test_warning_needs_consecutive_samples_and_writes_status(W, tmp_path, monkeypatch, capsys):
    host(tmp_path, io_full=50.0)
    unit(tmp_path, usage_usec=0, processes={100: ("python3.12", "S", "do_poll", 0)})
    state = write(tmp_path / "task/.tmp/run/cpu-reservation.json", json.dumps({"argv": []}))
    monkeypatch.setattr(W, "unit_state", lambda name: (state, {}))
    seen = []

    def sleep(_):
        seen.append(capsys.readouterr().out)
        if len(seen) == W.HOLD_SAMPLES:
            (tmp_path / "cgroup/merlinphase1.slice" / f"{UNIT}.service").rename(tmp_path / "gone")

    monkeypatch.setattr(W.time, "sleep", sleep)
    assert W.watch([UNIT], interval=60, once=False, out=None) == 0
    assert all("WARNING" not in text for text in seen[: W.HOLD_SAMPLES - 1])
    assert "WARNING host I/O is saturated" in seen[W.HOLD_SAMPLES - 1]
    status = json.loads((state.parent / "operator/watchdog-status.json").read_text())
    assert status["warnings"] and status["unit"]["unit"] == UNIT


def test_runs_come_from_campaign_argv(W, tmp_path):
    argv = ["/usr/bin/env", "-i", "MERLIN_OUT_ROOT=/o", "TMPDIR=/t", "python", "x", "--run-id", "r1"]
    assert W.runs_of(tmp_path / "s.json", {"argv": argv}) == [{"run_id": "r1", "out_root": "/o"}]
    workspace = tmp_path / "out/build/agent-workspaces/p/t/m/r1/workspace"
    write(workspace / "qa/latest_completed_grade.json", json.dumps({"graded_at": "2026-10-27T19:36:20+00:00"}))
    write(tmp_path / "out/runs/t/p/r1/_qa_work/fruns_1/runs/b/case/capsule_result.json", "{}")
    progress = W.grade_progress({"run_id": "r1", "out_root": str(tmp_path / "out")}, time.time())
    assert progress["graded_at"].startswith("2026-10-27") and progress["cases_last_10min"] == 1


def test_marker_lives_for_the_reservation_and_check_reports_it(W, tmp_path, capsys):
    (tmp_path / "cgroup/merlinphase1.slice").mkdir(parents=True)
    (tmp_path / "proc" / str(os.getpid())).mkdir(parents=True)
    assert W.check() == 0
    with W.announced(UNIT, {8, 9}, 3600, ["r1"]):
        with W.announced("merlinphase1other", {10}, 3600, []):
            document = json.loads(W.MARKER.read_text())
            assert set(document["runs"]) == {UNIT, "merlinphase1other"}
        assert W.check() == 3
        assert "RUN IN PROGRESS: merlinphase1fixture ['r1'] on CPUs 8,9" in capsys.readouterr().out
    assert not W.MARKER.exists() and W.check() == 0
    # An active reservation cgroup counts even when no launcher wrote a marker.
    write(tmp_path / "cgroup/merlinphase1.slice" / f"{UNIT}.service/cpuset.cpus.effective", "8-27\n")
    assert W.check() == 3


def test_reservation_launcher_announces_its_run(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("reservation_test", HERE / "resource_reservation.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    marker = tmp_path / "RUN_IN_PROGRESS.json"
    monkeypatch.setenv("MERLIN_RUN_MARKER", str(marker))
    args = type("Args", (), {"unit": UNIT, "max_seconds": 60, "command": ["x", "--run-id", "r9"]})()
    with wrapper.announced(args, {3}):
        assert json.loads(marker.read_text())["runs"][UNIT]["run_ids"] == ["r9"]
    assert not marker.exists()


def test_preflight_prints_one_warning_line_and_never_refuses(W, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("campaign_test", HERE / "fresh_campaign.py")
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    monkeypatch.setattr(campaign, "load_module", lambda name, directory: W)
    host(tmp_path, io_full=35.0)
    (tmp_path / "cgroup/merlinphase1.slice").mkdir(parents=True)
    line = campaign.host_health_line()
    assert line.startswith("HOST HEALTH WARNING: host I/O is saturated") and "\n" not in line
    host(tmp_path)
    assert campaign.host_health_line() == "HOST HEALTH ok"
    (tmp_path / "proc/pressure/io").unlink()
    assert campaign.host_health_line().startswith("HOST HEALTH unknown")
