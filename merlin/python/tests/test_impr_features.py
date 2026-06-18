"""R1: fork-scoped default-off compiler features must not perturb the baseline.

The baseline RVV compiler is frozen; an ``impr_`` fork enables named features. With no features
enabled, the emitted pipeline string and schedule are byte-identical to today's, so any fork can
be measured against an unchanged baseline.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import impr_features as F
from merlin.llvmlower import pipeline as P


def test_empty_features_pipeline_byte_identical():
    base = P.build_rvv_pipeline("/tmp/s.mlir")
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == base


def test_empty_features_schedule_byte_identical():
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE


def test_feature_changes_only_when_enabled():
    on = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["fused_vfmacc_contraction"]))
    assert on != P.RVV_TRANSFORM_SCHEDULE
    assert "[4, 8, 4]" in on  # K-vector tile injected


def test_unknown_feature_rejected():
    with pytest.raises(KeyError):
        F.normalize(["does_not_exist"])


def test_registered_feature_is_typed():
    f = F.get("fused_vfmacc_contraction")
    assert f.action_class in ("PASS", "HEURISTIC", "PATTERN")


def test_baseline_package_has_no_features():
    # The immutable baseline must carry zero compiler_features -> byte-identical lowering.
    from pathlib import Path

    from merlin.rvvgen.registry import load_rvv_package
    base = Path(__file__).resolve().parents[3] / "generated_targets" / "rvv" / "hand_v0"
    if not base.is_dir():
        pytest.skip("hand_v0 package not present")
    assert load_rvv_package(base).compiler_features == []
