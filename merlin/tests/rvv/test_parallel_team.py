"""Per-region OpenMP team sizing is explicit, structural, and default-off."""
import pytest

from merlin.llvmlower import impr_features
from merlin.llvmlower.parallel_team import (
    apply_for_test,
    ensure_registered,
    feature_name,
    team_width,
    work_of,
)


def test_team_width_rounds_through_powers_and_caps():
    assert [team_width(cost, 100, 8) for cost in (0, 100, 101, 201, 401, 801)] == [
        1, 1, 2, 4, 8, 8,
    ]
    assert team_width(1000, 100, 6) == 6


def test_feature_name_is_derived_and_conflicts_fail_closed():
    name = ensure_registered(100)
    assert name == feature_name(100)
    assert name in impr_features.normalize([name])
    assert work_of([name]) == 100
    with pytest.raises(ValueError, match="one team policy"):
        work_of([feature_name(10), feature_name(20)])
    with pytest.raises(ValueError):
        feature_name(0)


def test_rewrite_serializes_tiny_region_and_emits_per_region_num_threads():
    module, report = apply_for_test(r'''module {
      func.func @forward() {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c128 = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.parallel (%i) = (%c0) to (%c8) step (%c1) {
          %x = arith.index_cast %i : index to i64
          scf.reduce
        }
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          %x = arith.index_cast %i : index to i64
          scf.reduce
        }
        return
      }
    }''', work_per_thread=16, max_team=8)
    assert report["plan"] == [8]
    assert module.count("omp.parallel") == 1
    assert "num_threads(%c8_i32 : i32)" in module
    assert "scf.parallel" not in module


def test_empty_feature_set_does_not_edit_cflags_or_pipeline():
    flags = ["-O2", "-fno-vectorize"]
    pipeline = ["canonicalize", "convert-scf-to-openmp"]
    assert impr_features.apply_cflags(flags, frozenset()) == flags
    assert impr_features.apply_pipeline(pipeline, frozenset()) == pipeline
