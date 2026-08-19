"""Every driver in a campaign must run at the arm's declared reasoning effort.

Measured: `--effort` defaults to high and was threaded to the claude and codex drivers but not to
opencode, whose `run_round` absorbed it into `**_ignored`. So a campaign that compared a commercial model
against open ones compared `high` against each provider's default -- partly a comparison of reasoning
budgets rather than of models. The harness already states the rule in a comment on the codex branch; this
is that comment given teeth.
"""
from __future__ import annotations

import importlib.util
import inspect

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _load(name):
    src = _HARNESS / f"{name}.py"
    if not src.is_file():
        pytest.skip(f"{src} not present")
    import sys
    sys.path.insert(0, str(_HARNESS))
    try:
        spec = importlib.util.spec_from_file_location(f"{name}_under_test", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("driver", ["opencode_agent", "codex_agent"])
def test_the_driver_accepts_effort_by_name(driver):
    """A driver that only accepts effort via **kwargs drops it silently -- which is what happened."""
    mod = _load(driver)
    params = inspect.signature(mod.run_round).parameters
    assert "effort" in params, (
        f"{driver}.run_round has no named `effort` parameter, so a caller passing effort= is absorbed by "
        f"**kwargs and the round runs at the provider default")


def test_opencode_puts_the_effort_on_the_command_line():
    """opencode spells reasoning effort `--variant`; the flag must actually reach argv."""
    src = (_HARNESS / "opencode_agent.py").read_text()
    assert "--variant" in src, "opencode_agent never passes --variant, so effort cannot reach the model"
    i, j = src.index("--variant"), src.index("run_cmd")
    assert i > j, "--variant must be appended to the command opencode is invoked with"


def test_the_qa_loop_hands_effort_to_every_driver_it_dispatches():
    """The dispatch site is where the parity is won or lost -- check each branch forwards it."""
    src = (_HARNESS / "run_baseline_qa_loop.py").read_text()
    start = src.index("def launch_agent(")
    body = src[start:src.index("\ndef ", start + 10)]
    for driver in ("_OA.run_round", "_CA.run_round"):
        call = body.index(driver)
        nxt = body.find("return ", call)
        segment = body[call:nxt if nxt > call else call + 400]
        assert "effort=effort" in segment, f"{driver} is dispatched without the arm's effort"
    assert "--effort {effort}" in body, "the claude branch must keep passing --effort"
