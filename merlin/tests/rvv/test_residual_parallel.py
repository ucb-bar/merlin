"""Residual-loop parallelism is a named, grain-bounded lowering choice."""
import pytest

from merlin.llvmlower import impr_features
from merlin.llvmlower.parallel_grain import feature_name as grain_name
from merlin.llvmlower.parallel_team import feature_name as team_name
from merlin.llvmlower.pipeline import build_rvv_pipeline
from merlin.llvmlower.residual_parallel import (
    ensure_registered,
    feature_name,
    threshold_of,
)


def test_family_resolves_and_implies_the_same_grain():
    name = ensure_registered(10_000)
    enabled = impr_features.normalize([name])
    assert name == feature_name(10_000)
    assert name in enabled
    assert grain_name(10_000) in enabled
    assert threshold_of(enabled) == 10_000


def test_zero_threshold_names_the_full_residual_path_without_a_grain_pass():
    name = ensure_registered(0)
    enabled = impr_features.normalize([name])
    assert name == feature_name(0)
    assert threshold_of(enabled) == 0
    assert impr_features.get(name).implies == frozenset()


def test_conflicting_residual_points_fail_closed():
    with pytest.raises(ValueError, match="one residual policy"):
        threshold_of([feature_name(1000), feature_name(10_000)])


def test_negative_threshold_is_refused():
    with pytest.raises(ValueError, match=">= 0"):
        feature_name(-1)


def test_feature_selects_parallel_residue_while_default_stays_serial(tmp_path):
    schedule = tmp_path / "schedule.mlir"
    schedule.write_text("module attributes {transform.with_named_sequence} {}", encoding="utf-8")
    baseline = build_rvv_pipeline(schedule, features=frozenset(),
                                  par_sched_path=schedule, perop_parallel=True)
    enabled = impr_features.normalize([ensure_registered(10_000)])
    parallel = build_rvv_pipeline(schedule, features=enabled,
                                  par_sched_path=schedule, perop_parallel=True)
    assert "func.func(convert-linalg-to-loops)" in baseline
    assert "func.func(convert-linalg-to-parallel-loops)" not in baseline
    assert "func.func(convert-linalg-to-parallel-loops)" in parallel


def test_unrelated_team_policy_does_not_enable_residual_parallelism():
    enabled = impr_features.normalize([team_name(10_000)])
    assert threshold_of(enabled) is None
