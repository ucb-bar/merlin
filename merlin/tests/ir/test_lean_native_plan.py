"""Native proof phases come from an authored roster, including unchanged occurrences."""

from dataclasses import FrozenInstanceError, replace

import pytest

from merlin.llvmlower import fusion_guard, pipeline
from merlin.verify.lean import native_plan


@pytest.fixture
def ordinary(monkeypatch):
    for key in ["MERLIN_GENERALIZE_BEFORE_FUSE", "MERLIN_FUSE_POST", "MERLIN_NO_DEALLOC", "MERLIN_PASSES"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pipeline, "_sink_deallocs", lambda: False)
    monkeypatch.setattr(fusion_guard, "enabled", lambda: True)

    def resolve(**changes):
        return native_plan.ordinary(
            **{"features": frozenset(), "index_bits": 64, "data_layout": "e-p:64:64", **changes}
        )

    return resolve


@pytest.mark.parametrize("index_bits", [32, 64])
def test_authored_plan_renders_the_actual_ordinary_pipeline(ordinary, index_bits):
    plan = ordinary(index_bits=index_bits)
    assert [p.name for p in plan.boundaries.boundaries] == [
        "native-parse",
        "native-canonicalize",
        "native-cse",
        "native-guarded-fusion",
        "native-generalize",
    ]
    assert plan.pipeline() == pipeline._bind_index_width(pipeline._upstream_pipeline(frozenset()), index_bits)
    assert plan.suffix[0].startswith("one-shot-bufferize{")
    assert plan.sha256 == ordinary(index_bits=index_bits).sha256
    with pytest.raises(FrozenInstanceError):
        plan.index_bits = 16


@pytest.mark.parametrize("fault", ["omit", "duplicate", "reorder", "unexpected", "missing-barrier", "empty"])
def test_production_roster_cannot_define_its_own_denominator(ordinary, monkeypatch, fault):
    values = list(pipeline._UPSTREAM_PASSES)
    if fault == "omit":
        del values[1]
    elif fault == "duplicate":
        values.insert(1, values[0])
    elif fault == "reorder":
        values[0], values[1] = values[1], values[0]
    elif fault == "unexpected":
        values.insert(2, "symbol-dce")
    elif fault == "missing-barrier":
        del values[4]
    else:
        values = []
    monkeypatch.setattr(pipeline, "_UPSTREAM_PASSES", values)
    with pytest.raises(ValueError, match="production tensor roster"):
        ordinary()


@pytest.mark.parametrize("index_bits", [0, 1, 16, 128, True, "64"])
def test_unsupported_index_domain_refuses(ordinary, index_bits):
    with pytest.raises(ValueError, match="index domains"):
        ordinary(index_bits=index_bits)


@pytest.mark.parametrize(
    "changes",
    [
        {"features": frozenset({"any-feature"})},
        {"features": set()},
        {"custom_pipeline": "canonicalize"},
        {"vectorize": True},
        {"parallel": True},
        {"parallel_harts": 2},
        {"parallel_chunks": []},
        {"transform_schedule": "schedule"},
        {"source_scalar_carrier": object()},
        {"source_observation_effects": object()},
        {"masked_contraction_effects": object()},
    ],
)
def test_nonordinary_effective_selection_refuses_before_execution(ordinary, changes):
    with pytest.raises(ValueError, match="initial native verification"):
        ordinary(**changes)


@pytest.mark.parametrize("switch", ["MERLIN_GENERALIZE_BEFORE_FUSE", "MERLIN_FUSE_POST", "MERLIN_NO_DEALLOC"])
def test_legacy_pipeline_switch_refuses(ordinary, monkeypatch, switch):
    monkeypatch.setenv(switch, "1")
    with pytest.raises(ValueError, match="pipeline switches"):
        ordinary()


def test_actual_fusion_wrapper_is_part_of_the_plan(ordinary, monkeypatch):
    plan = ordinary()
    assert "fusion-includes-control-hooks" in plan.boundaries.granularity
    monkeypatch.setattr(fusion_guard, "enabled", lambda: False)
    with pytest.raises(ValueError, match="pipeline switches"):
        ordinary()
    with pytest.raises(ValueError, match="guarded fusion"):
        replace(plan, fusion_guard=False)


def test_barrier_is_unique_and_explicit(ordinary):
    plan = ordinary()
    for suffix in [(), plan.suffix[1:], (plan.suffix[0], *plan.suffix)]:
        with pytest.raises(ValueError, match="barrier"):
            replace(plan, suffix=suffix)
    with pytest.raises(ValueError, match="authored ordinary roster"):
        replace(plan, prefix=plan.prefix[:-1])
    assert plan.sha256 != replace(plan, data_layout="e-p:32:32").sha256


def test_converter_entry_changes_refuse(ordinary, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_UPSTREAM_PASSES",
        [
            entry + "{index-bitwidth=32}" if entry == "convert-arith-to-llvm" else entry
            for entry in pipeline._UPSTREAM_PASSES
        ],
    )
    with pytest.raises(ValueError, match="unique bare convert-arith-to-llvm"):
        ordinary()


def test_converter_roster_cannot_silently_drop_width_binding(ordinary, monkeypatch):
    monkeypatch.setattr(pipeline, "_INDEX_CONVERSION_PASSES", pipeline._INDEX_CONVERSION_PASSES[:-1])
    with pytest.raises(ValueError, match="converter roster"):
        ordinary()
