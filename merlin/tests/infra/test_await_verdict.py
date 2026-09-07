"""The blocking wait on a harness grade: it must return when one lands, and always terminate.

Every other wait in the agent's toolbox blocks -- the self-check returns when the broker answers,
`simjob wait` returns when a job lands. This one did not exist, so the only way to notice a new grade
was to look again, and an agent that must look again writes a polling loop. Measured on one 6.1 h run:
89 commands aimed at qa/verdict.json, ~15.5 min of sleeping, and each look also a model round trip.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

TOOL = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "await_verdict.py"
VERDICT = {"n_passed": 82, "n_capsules": 96, "all_pass": False,
           "per_capsule": {"A0": "pass", "B1": "fail", "C2": "fail"}}


def _run(cwd, *args, timeout=60):
    got = subprocess.run([sys.executable, str(TOOL), *args], cwd=str(cwd),
                         capture_output=True, text=True, timeout=timeout)
    try:
        return got.returncode, json.loads(got.stdout)
    except ValueError:
        raise AssertionError(f"not JSON: rc={got.returncode} out={got.stdout!r} err={got.stderr!r}")


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "qa").mkdir()
    (tmp_path / "qa" / "verdict.json").write_text(json.dumps(VERDICT))
    return tmp_path


def test_it_imports_nothing_from_merlin():
    """It is staged INSIDE the sandbox, where merlin is masked. An import would make it unstageable."""
    import ast
    tree = ast.parse(TOOL.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.split(".")[0] == "merlin" for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "merlin"


def _tool_module():
    """Import the staged script directly, to test its stamp resolution without a subprocess."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("await_verdict", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("await_verdict", mod)
    spec.loader.exec_module(mod)
    return mod


def test_two_grades_inside_one_second_are_distinguishable(ws):
    """SUB-SECOND resolution, and the reason is not pedantry.

    What is waited for is a GRADE, not a change of score: two consecutive grades may agree exactly and
    the agent still needs to know the second one ran. A whole-second stamp cannot see a re-grade that
    lands in the same second as the baseline read, and the consequence is not a late return but a
    MISSED one -- the wait reports "nothing new" and the agent sleeps another full grade interval.

    Written with explicit mtimes rather than two quick writes: this filesystem stamps writes at about
    millisecond granularity (measured: only 2 of 6 consecutive writes got distinct stamps), so a test
    that wrote twice and demanded different stamps failed on the storage rather than on the code.
    """
    import os
    mod = _tool_module()
    path = ws / "qa" / "verdict.json"
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    before = mod._stamp(path)
    os.utime(path, ns=(1_000_000_000, 1_001_000_000))          # one millisecond later
    after = mod._stamp(path)
    assert before is not None and after is not None
    assert after > before, (
        "two grades one millisecond apart were not distinguishable; a whole-second stamp would report "
        "no new grade and the agent would sleep through it")


def test_an_absent_verdict_has_no_stamp(ws):
    mod = _tool_module()
    assert mod._stamp(ws / "qa" / "nope.json") is None


def test_a_new_grade_returns_immediately(ws):
    """The paired direction for the timeout test: it must actually WAKE, and quickly."""
    def _grade():
        time.sleep(1.0)
        (ws / "qa" / "verdict.json").write_text(json.dumps({**VERDICT, "n_passed": 92}))
    t = threading.Thread(target=_grade, daemon=True)
    started = time.monotonic()
    t.start()
    rc, got = _run(ws, "--timeout", "30")
    elapsed = time.monotonic() - started
    t.join()
    assert rc == 0 and got["waited"] == "graded"
    assert got["n_passed"] == 92, "it returned the OLD verdict"
    assert elapsed < 15, f"woke {elapsed:.1f}s after the grade landed, not promptly"


def test_it_waits_for_the_NEXT_grade_not_the_one_already_there(ws):
    """Returning on the existing verdict would make the wait a no-op and the loop continue forever."""
    rc, got = _run(ws, "--timeout", "3")
    assert rc == 2 and got["waited"] == "timeout"


def test_a_timeout_is_reported_as_not_necessarily_a_fault(ws):
    rc, got = _run(ws, "--timeout", "3")
    assert rc == 2
    assert "not necessarily a fault" in got["note"]
    assert "wait again" in got["note"]


def test_it_always_terminates(ws):
    """Bounded by --timeout, with no path that blocks forever -- the failure mode of a hand-rolled
    poll loop this replaces."""
    start = time.monotonic()
    rc, _ = _run(ws, "--timeout", "2", timeout=30)
    assert rc == 2 and time.monotonic() - start < 25


def test_no_verdict_yet_is_absent_not_a_crash(tmp_path):
    rc, got = _run(tmp_path, "--timeout", "2")
    assert rc == 2 and got["waited"] == "absent"
    assert got["mtime_ns"] is None
    assert "first grade" in got["note"]


def test_it_reports_the_score_and_what_is_failing(ws):
    def _grade():
        time.sleep(0.8)
        (ws / "qa" / "verdict.json").write_text(json.dumps(
            {**VERDICT, "n_passed": 95, "per_capsule": {"A0": "pass", "B1": "pass", "C2": "fail"}}))
    threading.Thread(target=_grade, daemon=True).start()
    rc, got = _run(ws, "--timeout", "30")
    assert rc == 0 and got["n_passed"] == 95 and got["failing"] == ["C2"]


def test_an_unreadable_verdict_still_reports_the_grade(ws):
    """A grade DID land; that fact is what was waited for. Refusing to say so because the bytes could
    not be parsed would leave the agent polling again for an event that already happened."""
    def _grade():
        time.sleep(0.8)
        (ws / "qa" / "verdict.json").write_text("{ truncated")
    threading.Thread(target=_grade, daemon=True).start()
    rc, got = _run(ws, "--timeout", "30")
    assert rc == 0 and got["waited"] == "graded"
    assert got["readable"] is False and got["why"]


def test_a_caller_supplied_stamp_is_honoured(ws):
    """Lets an agent wait for a grade newer than one it already holds, without a race between reading
    the stamp and starting the wait."""
    rc, got = _run(ws, "--since-ns", "1", "--timeout", "10")
    assert rc == 0 and got["waited"] == "graded", "an old stamp must satisfy immediately"


def test_it_is_staged_into_the_workspace():
    """A tool the agent cannot invoke is not a tool. It must be copied in beside the other shims."""
    loop = (merlin_dir() / "experiments" / "capsule_bench" / "harness"
            / "run_baseline_qa_loop.py").read_text()
    assert '_stage_shim(ws, "await_verdict.py", "await_verdict.py")' in loop


def test_the_agent_is_told_it_exists():
    """The launch-generated task block is authoritative over bundled prose, so the tool is named there
    -- and the same block corrects the stale "you are relaunched each round" framing that made an
    agent look for a new verdict instead of waiting for one."""
    loop = (merlin_dir() / "experiments" / "capsule_bench" / "harness"
            / "run_baseline_qa_loop.py").read_text()
    i = loop.index("_task_runtime_scope_block")
    block = loop[i:i + 4000]
    assert "await_verdict.py" in block
    assert "not relaunched" in block.lower() or "NOT relaunched" in block
