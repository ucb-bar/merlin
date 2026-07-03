"""Pipeline registry + builder: named passes compose into a runnable plan transform."""
from merlin.dse.pipelines import registry
from merlin.dse.pipelines.builder import build_pipeline, parse_spec


def _plan():
    return {"pack_count": 8, "weight_loads": 8, "per_step_intermediate": True,
            "dispatch_count": 8, "resident_setup": False, "accumulator_setup": False}


def test_registered_passes_present():
    for name in ("hoist-pack", "make-resident", "defer-commit", "batch-dispatch", "hw-cache"):
        assert registry.has_pass(name)


def test_make_resident_pass_effect():
    out = registry.get_pass("make-resident")(_plan())
    assert out["pack_count"] == 1 and out["weight_loads"] == 1 and out["resident_setup"] is True


def test_pipeline_runs_passes_in_order():
    pipe = build_pipeline("merlin-contract,hoist-pack,make-resident,defer-commit,toynpu-lower")
    out = pipe.run(_plan())
    assert out["pack_count"] == 1
    assert out["per_step_intermediate"] is False and out["accumulator_setup"] is True
    assert pipe.unknown == ()


def test_unknown_pass_tolerated_then_strict_raises():
    pipe = build_pipeline("make-resident,not-a-real-pass")
    assert pipe.unknown == ("not-a-real-pass",)
    assert pipe.run(_plan())["weight_loads"] == 1  # unknown is identity
    import pytest
    with pytest.raises(KeyError):
        build_pipeline("make-resident,not-a-real-pass", strict=True)


def test_parse_spec_tolerates_whitespace():
    assert parse_spec(" a, b ,, c ") == ["a", "b", "c"]
