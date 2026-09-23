"""Recovery policy runs independently of native controllers and target discovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase1 import recovery as R

from merlin.common import access
from merlin.common.paths import merlin_dir, python_import_roots


@pytest.mark.parametrize("reason", ["weekly", "daily", "five_hour", "timeout", "complete"])
def test_canonical_and_compatibility_cli_share_recorded_policy(tmp_path, reason):
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    event = {"type": "result", "result": "done"}
    if reason == "weekly":
        event = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "seven_day"}}
    elif reason == "daily":
        event = {"type": "result", "result": "429 daily quota limit"}
    elif reason == "five_hour":
        event = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour"}}
    elif reason == "timeout":
        event = {"type": "system", "subtype": "init"}
    (rounds / "round_00.transcript.jsonl").write_text(json.dumps(event) + "\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_import_roots())))
    native = merlin_dir() / "experiments/capsule_bench/harness/resume_on_quota.py"
    results = [
        subprocess.run(
            [sys.executable, *command, str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
        )
        for command in (["-m", "merlin_experiments.phase1.recovery"], [str(native)])
    ]
    assert all(result.returncode == 0 for result in results), [result.stderr for result in results]
    assert results[0].stdout == results[1].stdout
    expected = {
        "weekly": R.REASON_WEEKLY,
        "daily": R.REASON_DAILY,
        "five_hour": R.REASON_FIVE_HOUR,
        "timeout": R.REASON_TIMEOUT,
        "complete": "(finished cleanly)",
    }[reason]
    assert f"reason       : {expected}" in results[0].stdout
    assert list(tmp_path.iterdir()) == [rounds], "classification must not mutate run evidence"


def test_import_and_policy_are_target_inert(tmp_path):
    program = """
import importlib.abc, json, os, subprocess, sys
sys.path[:0] = json.loads(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name in {'_common', '_ratelimit', 'resume_on_quota', 'run_baseline_qa_loop'}:
            raise AssertionError('native import: ' + name)
sys.meta_path.insert(0, NoNative())
before = dict(os.environ)
def unexpected(*args, **kwargs): raise AssertionError('process at import')
subprocess.run = unexpected
from merlin_experiments.phase1 import recovery as R
assert R.resume_policy(R.REASON_WEEKLY) == R.EXIT_WITH_STATUS
assert R.resume_policy(R.REASON_TIMEOUT) == R.RESUME_IN_BUDGET
assert dict(os.environ) == before
"""
    roots = list(map(str, python_import_roots())) + [
        path for path in sys.path if path and "site-packages" in path and Path(path).is_dir()
    ]
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, json.dumps(roots)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_precedence_and_zero_work_guard_preserve_reporting_denominator(tmp_path):
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    five = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour"}}
    daily = {"type": "result", "result": "429 daily quota"}
    weekly = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "seven_day"}}
    work = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}}
    first, second = rounds / "round_00.transcript.jsonl", rounds / "round_01.transcript.jsonl"
    first.write_text("\n".join(map(json.dumps, [five, daily, weekly])))
    second.write_text("\n".join(map(json.dumps, [five, work])))
    assert R.classify(first, rc=124) == R.REASON_WEEKLY
    assert R.rounds_rate_limited(tmp_path) == (1, 1)
    assert not R.round_rejected(second)
    assert R.agent_turn_dead(second) == (False, "")
    assert "merlin_experiments.phase1.recovery" in access.declared_modules("grader")
