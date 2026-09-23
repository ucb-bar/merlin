"""Every blocking wait offered to the agent must RETURN inside the agent's shell ceiling.

The calling agent's shell tool aborts any command at 600s. A wait whose default exceeds that is killed
rather than returning, so the caller gets nothing back -- and the lesson it draws is that the blocking
wait "does not work", which is precisely how a run ends up hand-writing ``until ... sleep`` loops that
the same ceiling then truncates too. Measured on one run: six such loops, 59.4 min, no information.

The irony this pins down: ``await_verdict.py`` is the tool the task brief offers AS the cure for
polling, and its default timeout was 1200s -- double the ceiling. It could not survive its own
recommended use.

A timeout is explicitly not an error in either tool (they say so in their output), so a bound under the
ceiling costs one cheap turn to re-issue. Being killed costs the entire wait.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"

#: What the agent's shell kills at. Not ours to change; the waits must fit under it.
SHELL_CEILING_S = 600


def _module(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("module", "attr"),
    [
        ("await_verdict", "_DEFAULT_TIMEOUT_S"),
        ("selfcheck_shim", "_DEFAULT_WAIT_BUDGET_S"),
    ],
)
def test_default_wait_returns_before_the_shell_kills_it(module, attr):
    mod = _module(module)
    value = getattr(mod, attr)
    assert 0 < value < SHELL_CEILING_S, (
        f"{module}.{attr} is {value}s; the calling agent's shell kills at {SHELL_CEILING_S}s, so this "
        "wait is killed instead of returning and the caller learns that blocking waits do not work"
    )
