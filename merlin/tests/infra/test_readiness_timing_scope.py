"""Legacy probe helper behavior and selected-engine readiness integration."""

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


def test_readiness_measures_the_selected_engine_without_lifting_its_policy():
    source = (repo_root() / "merlin/experiments/capsule_bench/harness/readiness_check.py").read_text()
    assert "before = selected_engine_binding(descriptor=C.DESCRIPTOR, target=TARGET)" in source
    assert 'engine = before["engine"]' in source
    assert '_grade(ref, engine, 900, cap="A2_single_tile_matmul")' in source
    assert "write_observed_timing(" in source
    assert "grade_env=probe_env" not in source
