"""A round must die on INACTIVITY, not only on the wall clock.

Measured: a GLM-5 round read its task files, emitted ``step_start``, then produced zero bytes for 28
minutes while the process stayed perfectly healthy. The round timeout was 4h, so the run would have
held the budget open doing nothing for four hours. A wall-clock cap cannot distinguish a stalled agent
from a slow one; bytes produced can.

These drive the REAL ``_capture`` against REAL processes -- a mocked Popen would not exercise the
process-group reap, which is the part that previously leaked orphans.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import time

import pytest

from merlin.common.paths import merlin_dir

_SRC = merlin_dir() / "experiments/capsule_bench/harness/opencode_agent.py"


@pytest.fixture(scope="module")
def oc():
    if not _SRC.is_file():
        pytest.skip(f"{_SRC} not present")
    # the harness dir is not a package; its siblings import by bare name
    import sys
    sys.path.insert(0, str(_SRC.parent))
    try:
        spec = importlib.util.spec_from_file_location("opencode_agent_under_test", _SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def _run(oc, script, *, timeout, stall):
    return oc._capture(["bash", "-c", script], dict(os.environ), timeout, ".", stall_seconds=stall)


def test_a_silent_process_is_killed_long_before_the_wall_clock(oc):
    """The measured failure: alive, healthy, emitting nothing."""
    t0 = time.monotonic()
    with pytest.raises(oc.AgentStalled):
        _run(oc, "echo hello; sleep 120", timeout=120, stall=6)
    assert time.monotonic() - t0 < 40, "stall must fire on inactivity, not wait out the wall timeout"


def test_a_slow_but_producing_process_is_not_killed(oc):
    """The false positive that would silently truncate honest work."""
    rc, out, _ = _run(oc, "for i in 1 2 3; do echo x; sleep 4; done", timeout=90, stall=8)
    assert rc == 0 and out.count("x") == 3


def test_the_wall_timeout_still_fires_for_a_chatty_overrun(oc):
    """A process that never stops talking must still be capped."""
    with pytest.raises(subprocess.TimeoutExpired):
        _run(oc, "while true; do echo spam; sleep 0.2; done", timeout=5, stall=0)


def test_a_stall_is_distinguishable_from_an_honest_overrun(oc):
    """Callers map both to a failed round; the type is what tells them apart in the report."""
    assert issubclass(oc.AgentStalled, subprocess.TimeoutExpired)


def test_zero_disables_the_detector(oc):
    rc, _, _ = _run(oc, "echo hi; sleep 4", timeout=30, stall=0)
    assert rc == 0


def test_the_whole_process_tree_dies_on_a_stall(oc):
    """bash -> bash -> sleep mirrors bash -> bwrap -> opencode; a survivor hangs the next reap."""
    mark = "MERLIN_STALL_ORPHAN_PROBE_TEST"

    def alive():
        out = subprocess.run(["ps", "-u", os.environ.get("USER", ""), "-o", "cmd="],
                             capture_output=True, text=True).stdout
        return sum(1 for line in out.splitlines() if mark in line and "ps -u" not in line)

    assert alive() == 0
    with pytest.raises(oc.AgentStalled):
        _run(oc, f"echo start; bash -c 'exec -a {mark} sleep 200' & sleep 200", timeout=200, stall=6)
    time.sleep(2)
    assert alive() == 0, "a stalled tree left orphans behind"
