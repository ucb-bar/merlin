"""Readiness's Verilator timing probe and a campaign's RTL-engine pin do not conflict.

The oracle-timing record calibrates the launcher with Verilator's per-capsule cost, so its probe runs
Verilator by design. A campaign that pins certification to another engine (``.env``
``MERLIN_REQUIRED_RTL_ENGINE=gsim``) made selfcheck refuse that probe outright -- a NO-GO that blamed
the oracle and blocked the timing record. The pin is lifted for the probe only and returned, so
readiness still proves the pinned engine on its own.
"""

from __future__ import annotations

import sys

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))

from readiness_reference import REQUIRED_ENGINE_ENV, timing_probe_environment  # noqa: E402


def test_a_pin_to_another_engine_is_lifted_for_the_probe_and_reported():
    env = {REQUIRED_ENGINE_ENV: "gsim", "KEEP": "1"}
    probe, pinned = timing_probe_environment(env, engine="verilator")
    assert REQUIRED_ENGINE_ENV not in probe and probe["KEEP"] == "1"
    assert pinned == "gsim", "the caller must still grade the pinned engine"
    assert env == {REQUIRED_ENGINE_ENV: "gsim", "KEEP": "1"}, "the run's environment is never mutated"


def test_no_pin_or_a_matching_pin_changes_nothing():
    assert timing_probe_environment({"KEEP": "1"}, engine="verilator") == ({"KEEP": "1"}, None)
    same = {REQUIRED_ENGINE_ENV: "verilator"}
    assert timing_probe_environment(same, engine="verilator") == (same, None)
    assert timing_probe_environment({REQUIRED_ENGINE_ENV: "  "}, engine="verilator")[1] is None


def test_readiness_grades_the_timing_probe_with_the_scoped_environment_and_the_pin_separately():
    source = (repo_root() / "merlin/experiments/capsule_bench/harness/readiness_check.py").read_text()
    assert 'timing_probe_environment(env, engine="verilator")' in source
    assert 'grade_env=probe_env' in source
    assert "_grade(ref, pinned_engine, 900" in source
