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


def _sh_quote(text):
    return "'" + text.replace("'", "'\\''") + "'"


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


def test_a_buffered_worker_that_prints_nothing_yet_is_not_killed(oc):
    """The regression this file exists for the second time.

    opencode block-buffers stdout when it is redirected to a FILE, so a round doing real but quiet work
    shows a byte count that does not move. Measured on a live GLM-5 run: the agent wrote three source
    files and ran selfchecks while its transcript sat at 109 bytes, and a bytes-only detector killed 4 of
    its 6 rounds. This child is the same shape -- it burns CPU and produces output, but nothing reaches
    the file until it exits.

    It must outlive TWO polls: the first poll sees the file appear (0 bytes vs the -1 sentinel) and
    counts that as progress, so only the second flat poll can trip the detector. A shorter child exits
    inside the first interval and proves nothing -- the first version of this test did exactly that.
    """
    burn = (
        "import sys, time\n"
        "end = time.monotonic() + 35\n"
        "n = 0\n"
        "while time.monotonic() < end:\n"
        "    n += sum(i * i for i in range(20000))\n"
        "print('done', n)\n"
    )
    t0 = time.monotonic()
    rc, out, _ = _run(oc, f"exec python3 -c {_sh_quote(burn)}", timeout=120, stall=5)
    assert rc == 0, "a working agent must not be killed just because its stdout is still buffered"
    assert "done" in out
    assert time.monotonic() - t0 >= 30, "the child must have been allowed to run to completion"


def test_cpu_progress_is_read_for_the_process_group_only(oc):
    """The CPU signal must not be satisfied by unrelated load elsewhere on this shared host."""
    mine = os.getpgid(0)
    assert oc._tree_cpu_seconds(mine) > 0.0, "this test's own group has certainly burned CPU"
    assert oc._tree_cpu_seconds(0x7FFFFFF0) == 0.0, "a group with no processes must report no CPU"


def test_a_process_blocked_on_io_still_counts_as_stalled(oc):
    """CPU must not become a blanket amnesty: a hang burns none, which is the whole point."""
    with pytest.raises(oc.AgentStalled):
        _run(oc, "exec python3 -c 'import time; time.sleep(120)'", timeout=120, stall=6)


def test_a_killed_round_keeps_the_record_of_what_it_did(oc):
    """A round killed on the wall clock must still report its actions and tokens.

    Measured: a live GLM-5 round worked for 40 minutes, was cut off at the round timeout, and left a
    two-line transcript reporting tool_calls=0 and no usage. The tokens were spent and the actions
    happened; discarding the stream made the round unbudgetable and made the agent look idle.
    """
    script = "printf 'line-one\\n'; printf 'line-two\\n'; sleep 120"
    with pytest.raises(subprocess.TimeoutExpired) as ei:
        _run(oc, script, timeout=8, stall=0)          # stall detector off: this is a WALL-CLOCK kill
    exc = ei.value
    assert hasattr(exc, "partial_stdout"), "the partial stream must be attached to the exception"
    assert "line-one" in exc.partial_stdout and "line-two" in exc.partial_stdout, \
        "output produced before the kill must survive the capture files being cleaned up"


def test_a_stalled_round_also_keeps_its_partial_stream(oc):
    """Same guarantee on the inactivity path, which is the other way a round dies."""
    with pytest.raises(oc.AgentStalled) as ei:
        _run(oc, "printf 'seen-before-stall\\n'; sleep 200", timeout=200, stall=6)
    assert "seen-before-stall" in (getattr(ei.value, "partial_stdout", "") or "")
