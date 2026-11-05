"""Capture admission uses available RAM while retaining CPU and memory ceilings."""

from pathlib import Path

import pytest
from merlin_experiments.phase0 import parallel_capture as P

GIB = 1024**3


def _free_pages(monkeypatch, memory):
    monkeypatch.setattr(P.os, "sysconf", lambda name: memory // 4096 if name == "SC_AVPHYS_PAGES" else 4096)


def test_available_cache_is_not_free_pages(tmp_path, monkeypatch):
    _free_pages(monkeypatch, 2 * GIB)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemFree: {2 * GIB // 1024} kB\nMemAvailable: {20 * GIB // 1024} kB\n")
    assert P._available_memory(meminfo) == 20 * GIB
    monkeypatch.setattr(P, "_available_memory", lambda: 20 * GIB)
    monkeypatch.setattr(P, "_cgroup_limits", lambda: (None, None))
    monkeypatch.setattr(P.os, "sched_getaffinity", lambda _pid: set(range(6)))
    assert P.worker_limit() == 6


@pytest.mark.parametrize(
    "text",
    [
        "",
        "MemFree: 123 kB\n",
        "MemAvailable: -1 kB\n",
        "MemAvailable: 12 MB\n",
        "MemAvailable: x kB\n",
        "MemAvailable: 3 kB extra\n",
        "MemAvailable: 12 kB\nMemAvailable: 120 kB\n",
    ],
)
def test_missing_invalid_or_duplicate_available_falls_back(tmp_path, monkeypatch, text):
    _free_pages(monkeypatch, 4 * GIB)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(text)
    assert P._available_memory(meminfo) == 4 * GIB


def test_nonlinux_fallback_and_unknown_memory(tmp_path, monkeypatch):
    _free_pages(monkeypatch, 4 * GIB)
    assert P._available_memory(tmp_path / "absent") == 4 * GIB
    monkeypatch.setattr(P.os, "sysconf", lambda _name: -1)
    assert P._available_memory(tmp_path / "absent") == 0

    def unavailable(_name):
        raise ValueError("not available")

    monkeypatch.setattr(P.os, "sysconf", unavailable)
    assert P._available_memory(tmp_path / "absent") == 0
    assert P._cgroup_limits(tmp_path / "no-proc") == (None, None)


@pytest.mark.parametrize(
    "affinity,memory,limits,expected",
    [
        (32, 100 * GIB, (None, None), 8),
        (3, 100 * GIB, (None, None), 3),
        (8, 7 * GIB, (None, None), 3),
        (8, 100 * GIB, (2, None), 2),
        (8, 100 * GIB, (None, 5 * GIB), 2),
        (8, 3 * GIB, (None, 100 * GIB), 1),
        (8, 0, (None, None), 1),
        (8, 100 * GIB, (1, 0), 1),
    ],
)
def test_original_policy_and_tighter_ceilings(monkeypatch, affinity, memory, limits, expected):
    monkeypatch.setattr(P.os, "sched_getaffinity", lambda _pid: set(range(affinity)))
    monkeypatch.setattr(P, "_available_memory", lambda: memory)
    monkeypatch.setattr(P, "_cgroup_limits", lambda: limits)
    assert P.worker_limit() == expected


def _cgroups(tmp_path, *, mount_root="/"):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    mount = tmp_path / "cgroup space"
    current = mount / "nested" / "worker"
    current.mkdir(parents=True)
    member = Path(mount_root) / "nested/worker"
    (proc / "self/cgroup").write_text(f"0::{member}\n")
    escaped = str(mount).replace(" ", "\\040")
    (proc / "self/mountinfo").write_text(f"1 2 0:1 {mount_root} {escaped} rw - cgroup2 cgroup rw\n")
    return proc, mount, current


def test_cgroup_ancestors_memory_headroom_and_cpu_bandwidth(tmp_path):
    proc, mount, current = _cgroups(tmp_path, mount_root="/namespace")
    (current / "memory.max").write_text(str(20 * GIB))
    (current / "memory.current").write_text(str(2 * GIB))
    (current / "cpu.max").write_text("350000 100000")
    (mount / "memory.max").write_text(str(8 * GIB))
    (mount / "memory.current").write_text(str(3 * GIB))
    (mount / "cpu.max").write_text("200000 100000")
    assert P._cgroup_limits(proc) == (2, 5 * GIB)


def test_cgroup_unlimited_and_consumed_ceiling(tmp_path):
    proc, _mount, current = _cgroups(tmp_path)
    (current / "memory.max").write_text("max")
    (current / "cpu.max").write_text("max 100000")
    assert P._cgroup_limits(proc) == (None, None)
    (current / "memory.max").write_text("12")
    (current / "memory.current").write_text("13")
    (current / "cpu.max").write_text("50000 100000")
    assert P._cgroup_limits(proc) == (1, 0)


@pytest.mark.parametrize(
    "filename,value",
    [
        ("memory.max", "-1"),
        ("memory.current", "bad"),
        ("cpu.max", "max 0"),
        ("cpu.max", "-1 100000"),
        ("cpu.max", "garbage"),
        ("cpu.max", "100000 100000 extra"),
    ],
)
def test_visible_malformed_cgroup_does_not_expand_budget(tmp_path, filename, value):
    proc, _mount, current = _cgroups(tmp_path)
    (current / "memory.max").write_text(str(8 * GIB))
    (current / "memory.current").write_text("0")
    (current / filename).write_text(value)
    assert P._cgroup_limits(proc) == (1, 0)


def test_incomplete_visible_memory_and_hidden_ancestor_refuse(tmp_path):
    proc, _mount, current = _cgroups(tmp_path)
    (current / "memory.max").write_text("1000")
    assert P._cgroup_limits(proc) == (1, 0)
    (proc / "self/cgroup").write_text("0::/outside\n")
    text = (proc / "self/mountinfo").read_text().replace("0:1 / ", "0:1 /hidden ")
    (proc / "self/mountinfo").write_text(text)
    assert P._cgroup_limits(proc) == (1, 0)


def test_zero_available_is_real_memory_limit(tmp_path, monkeypatch):
    _free_pages(monkeypatch, 40 * GIB)
    info = tmp_path / "meminfo"
    info.write_text("MemAvailable: 0 kB\n")
    assert P._available_memory(info) == 0


def test_affinity_unavailable_uses_original_cpu_count_fallback(monkeypatch):
    monkeypatch.delattr(P.os, "sched_getaffinity")
    monkeypatch.setattr(P.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(P, "_available_memory", lambda: 100 * GIB)
    monkeypatch.setattr(P, "_cgroup_limits", lambda: (None, None))
    assert P.worker_limit() == 2
    monkeypatch.setattr(P.os, "cpu_count", lambda: None)
    assert P.worker_limit() == 1


def test_missing_mount_for_visible_cgroup_is_conservative(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/nested\n")
    assert P._cgroup_limits(proc) == (1, 0)
