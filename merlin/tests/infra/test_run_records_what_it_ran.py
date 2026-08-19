"""A run's artifact must record what it ACTUALLY ran, not what the launcher intended.

Measured: every opencode round in the 2026-08 gemmini campaign wrote `effort: high` into
environment.yaml while executing at the provider default, because the driver never received the flag. The
artifact asserted something untrue about its own run, and nothing in the run directory could contradict
it. The driver now stamps the effort it actually passed into the transcript's init record -- the codex
driver already does this -- so the claim is checkable against the run rather than trusted.
"""
from __future__ import annotations

import importlib.util
import inspect

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _src(name):
    p = _HARNESS / f"{name}.py"
    if not p.is_file():
        pytest.skip(f"{p} not present")
    return p.read_text()


def test_the_init_record_stamps_the_effort_actually_passed():
    s = _src("opencode_agent")
    init = s[s.index('"subtype": "init"'):]
    init = init[:init.index("})") + 2]
    for field in ("effort_requested", "variant_passed", "sandbox", "delegate_model"):
        assert field in init, f"the init record omits {field}, so the round is not auditable from its own artifact"


def test_reasoning_tokens_are_carried_not_dropped():
    """Providers that bill thinking separately report it separately; dropping it reads as 'did not think'."""
    s = _src("opencode_agent")
    assert 'tok["reasoning"]' in s, "reasoning tokens are never accumulated"
    assert "reasoning_tokens" in s, "reasoning tokens never reach the usage event the accounting reads"


def test_the_delegate_is_reported_truthfully():
    """The record must name the delegate that was configured, or None -- never a hardcoded placeholder."""
    s = _src("opencode_agent")
    assert '"delegate_model": _delegate' in s, \
        "delegate_model must come from the resolved delegate, not a literal"
    assert s.index("_delegate =") < s.index('"delegate_model"'), \
        "the delegate must be resolved before the record that reports it"
