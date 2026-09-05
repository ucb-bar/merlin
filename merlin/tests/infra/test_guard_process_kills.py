"""The process-kill guard: what it must block, and what it must not.

This guard exists because a MEMORY did not stop the failure. `no-broad-pkill-shared-host` documents
both mechanisms precisely — including the misleading `exit 144` — and the mistake was still made
three times (2026-07-14, 2026-08-28, 2026-09-05). A gate is the difference between advice and
enforcement, which is the same argument the artifact-layout guard beside it makes.

The allow cases matter as much as the deny cases. A guard that fired on the substring "pkill" would
block writing this file, or any note explaining the rule, and would be turned off within a day.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from merlin.common.paths import repo_root

HOOK = repo_root() / ".claude" / "hooks" / "guard_process_kills.py"

ALLOW, DENY = 0, 2


def _verdict(command: str) -> int:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    proc = subprocess.run(["python3", str(HOOK)], input=payload, capture_output=True, text=True)
    return proc.returncode


@pytest.mark.parametrize("command", [
    'pkill -f "land.sh"',                                   # the one that killed a shell
    'pkill -f pytest',
    'nohup pkill -f thing &',                               # prefix must not hide it
    'killall python3',
    'until ! pgrep -f test_x >/dev/null; do sleep 5; done',  # the 34-minute hang
    'while ! pgrep -f foo; do sleep 1; done',
    'kill merlin',                                          # not a PID
])
def test_denies_the_shapes_that_have_actually_bitten(command):
    assert _verdict(command) == DENY, f"guard allowed a known-harmful command: {command!r}"


@pytest.mark.parametrize("command", [
    'kill 12345',                                           # exact PID: the safe form
    'kill -9 12345',
    'pgrep -af pytest | head -5',                           # inspection is how you find the PIDs
    # NOTE: an unbounded `ps -p` wait used to be listed here as "the safe wait". It is safe from the
    # self-match, and still unbounded -- if the process hangs, the loop does too. It moved to the
    # deny list when the bound rule landed, which is the rule correcting an earlier, weaker claim.
    '.venv/bin/python -m pytest merlin/tests/ir -q',
    'echo "never use pkill -f on this host"',               # a MENTION, not an invocation
])
def test_allows_the_safe_and_the_merely_mentioned(command):
    assert _verdict(command) == ALLOW, f"guard blocked a legitimate command: {command!r}"


def test_the_denial_explains_the_mechanism_and_names_a_remedy():
    """A block that does not say what to do instead gets bypassed rather than obeyed."""
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": 'pkill -f x'}})
    proc = subprocess.run(["python3", str(HOOK)], input=payload, capture_output=True, text=True)
    assert proc.returncode == DENY
    err = proc.stderr
    assert "own argv" in err, "the message must explain WHY it self-matches"
    assert "kill <pid>" in err, "the message must name the safe alternative"
    assert "MERLIN_ALLOW_PROCESS_KILL" in err, "the message must name the escape hatch"


def test_the_escape_hatch_works():
    """A genuine one-off must be possible, or the guard invites disabling the whole hook."""
    import os

    env = dict(os.environ, MERLIN_ALLOW_PROCESS_KILL="1")
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": 'pkill -f x'}})
    proc = subprocess.run(["python3", str(HOOK)], input=payload, capture_output=True,
                          text=True, env=env)
    assert proc.returncode == ALLOW


def test_it_ignores_tools_other_than_bash():
    payload = json.dumps({"tool_name": "Write", "tool_input": {"command": "pkill -f x"}})
    proc = subprocess.run(["python3", str(HOOK)], input=payload, capture_output=True, text=True)
    assert proc.returncode == ALLOW


@pytest.mark.parametrize("command", [
    'until [ -s /tmp/x.log ]; do sleep 2; done',        # the 2.5-hour zombie, measured
    'until ! ps -p 999 >/dev/null; do sleep 5; done',   # unbounded even on the "safe" wait
    'while [ ! -f /tmp/marker ]; do sleep 10; done',
])
def test_denies_an_unbounded_polling_loop(command):
    """A wait with no bound spins forever, silently, and reads as a slow job.

    Measured 2026-09-05: a loop polling for a file whose writer had already been stopped ran 2.5
    hours producing no output. The self-matching pgrep loop is the same family with a different
    trigger; requiring a bound catches every variant, because it stops asking WHY the condition
    never holds.
    """
    assert _verdict(command) == DENY, f"unbounded poll allowed: {command!r}"


@pytest.mark.parametrize("command", [
    'for i in $(seq 1 60); do ps -p 999 >/dev/null || break; sleep 5; done',
    'timeout 300 bash -c "until [ -s /tmp/x ]; do sleep 2; done"',
    'while read l; do echo "$l"; done < f.txt',          # a read loop is not a poll
    'sleep 5 && echo done',
])
def test_allows_a_bounded_wait_and_a_non_poll(command):
    assert _verdict(command) == ALLOW, f"guard blocked a bounded or non-polling loop: {command!r}"


def test_a_glued_separator_does_not_hide_an_invocation():
    """`echo hi;<killer> -f x` is an invocation; the segmenter must split a glued `;`."""
    assert _verdict('echo hi;' + "pkill" + ' -f thing') == DENY


def test_a_separator_inside_quotes_stays_a_mention():
    """The other half of the same rule: padding blindly turns a quoted string into a fake invocation.

    An earlier draft of this guard did exactly that and blocked its own test suite.
    """
    assert _verdict('echo "hi;' + "pkill" + ' -f thing"') == ALLOW
