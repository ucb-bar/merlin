"""Source-only ordinary Code width admission; no SDK or native proof claims."""

import pytest

from merlin.llvmlower import passes_xdsl
from merlin.transforms.certificates import ir_text
from merlin.verify import compiler_ir_source_profile as profile
from merlin.verify import memory_scalar_nan_admission
from merlin.verify.buffer_reuse_bridge import parse
from merlin.verify.compiler_memory_stage_intake import prepare
from merlin.verify.compiler_profile import SourceOutsideProfile
from merlin.verify.gate_errors import GateUnavailable

MAX63 = (1 << 63) - 1


def constant(value, *, used=False):
    return (
        "builtin.module {func.func @f(%view:memref<3xi32>)->"
        + ("index" if used else "i32")
        + " {%value=arith.constant "
        + str(value)
        + ":index "
        + ("func.return %value:index}}" if used else "%r=arith.constant 0:i32 func.return %r:i32}}")
    )


def dimension(shape, axis=0):
    return (
        "builtin.module {func.func @f(%view:memref<"
        + str(shape)
        + "xi32>)->index {%dead=arith.constant 0:index %axis=arith.constant "
        + str(axis)
        + ":index %dim=memref.dim %view,%axis:memref<"
        + str(shape)
        + "xi32> func.return %dim:index}}"
    )


def no_producers(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("source profile reached an optimizer")

    monkeypatch.setattr(passes_xdsl, "prune_dead_pure_tensor_ops", fail)
    monkeypatch.setattr("merlin.transforms.normalization.normalize_shapes", fail)
    monkeypatch.setattr("merlin.transforms.planning.plan_storage", fail)


@pytest.mark.parametrize("value", [0, 1, MAX63])
def test_fitting_original_constants_admit_without_production(value, monkeypatch):
    no_producers(monkeypatch)
    raw = constant(value)
    assert profile.admit_memory_dce(raw) == "ordinary-memory-context-dead-index-constants-index64-v1"
    assert str(value) in raw


@pytest.mark.parametrize("value", [1 << 63, (1 << 64) - 1, 1 << 64])
@pytest.mark.parametrize("used", [False, True], ids=["dead", "returned"])
def test_overflow_constant_excluded_from_original_source_even_without_change(value, used, monkeypatch):
    no_producers(monkeypatch)
    with pytest.raises(SourceOutsideProfile) as failure:
        profile.admit_memory_dce(constant(value, used=used))
    assert failure.value.code == "index-constant-width"


@pytest.mark.parametrize("placement", ["entry-rest", "nested-body", "helper-body"])
def test_width_walk_visits_rest_nested_regions_and_all_functions(placement, monkeypatch):
    raw = constant(0)
    if placement == "entry-rest":
        raw = raw.replace("%r=", "%huge=arith.constant 9223372036854775808:index %r=", 1)
    elif placement == "nested-body":
        raw = raw.replace(
            "%r=",
            "%zero=arith.constant 0:index %one=arith.constant 1:index "
            "scf.for %i=%zero to %one step %one {"
            "%huge=arith.constant 9223372036854775808:index scf.yield} %r=",
            1,
        )
    else:
        raw = (
            raw[:-1]
            + "func.func @helper()->index {%huge=arith.constant 9223372036854775808:index func.return %huge:index}}"
        )
    no_producers(monkeypatch)
    with pytest.raises(SourceOutsideProfile) as failure:
        profile.admit_memory_dce(raw)
    assert failure.value.code == "index-constant-width"


@pytest.mark.parametrize("shape", [0, 3, MAX63])
def test_known_dimension_result_at_fitting_bound_remains_admitted(shape, monkeypatch):
    no_producers(monkeypatch)
    assert profile.admit_memory_dce(dimension(shape)) == "ordinary-memory-context-dead-index-constants-index64-v1"


def test_nonfitting_known_dimension_is_original_source_exclusion(monkeypatch):
    no_producers(monkeypatch)
    with pytest.raises(SourceOutsideProfile) as failure:
        profile.admit_memory_dce(dimension(1 << 63))
    assert failure.value.code == "dimension-result-width"


def test_large_view_without_intrinsic_dimension_has_no_new_shape_limit(monkeypatch):
    raw = constant(0).replace("memref<3xi32>", "memref<9223372036854775808xi32>")
    no_producers(monkeypatch)
    assert profile.admit_memory_dce(raw) == "ordinary-memory-context-dead-index-constants-index64-v1"


@pytest.mark.parametrize("raw", [constant(-1), dimension(3, 1)])
def test_existing_negative_constant_and_axis_exclusions_remain(raw, monkeypatch):
    no_producers(monkeypatch)
    with pytest.raises(SourceOutsideProfile) as failure:
        profile.admit_memory_dce(raw)
    assert failure.value.code == "source-model"


@pytest.mark.parametrize("value", [0, MAX63, 1 << 63])
def test_real_dce_pair_confirms_width_boundary_is_source_policy(value):
    raw = constant(value)
    module = parse(raw)
    before = ir_text(module)
    assert passes_xdsl.prune_dead_pure_tensor_ops(module) == 1
    after = ir_text(module)
    assert before != after
    actual = prepare(before, after)
    code = actual["projection"]["before"]["functions"][0]["code"]
    assert code["tag"] == "constant" and code["value"] == value
    # This mirrors the explicit ConstantFacts64 inequality; no Lean execution.
    assert (value < 1 << 63) == (value <= MAX63)
    if value > MAX63:
        with pytest.raises(SourceOutsideProfile, match="index64"):
            profile.admit_memory_dce(raw)
    else:
        assert profile.admit_memory_dce(raw)


def test_dense_parser_budget_is_still_blocking_unavailability(monkeypatch):
    monkeypatch.setattr(memory_scalar_nan_admission, "MAX_DENSE_FLOAT_BYTES", 4)
    raw = constant(0).replace("{%value=", "attributes {tag=dense<0.0>:tensor<2xf32>} {%value=", 1)
    with pytest.raises(GateUnavailable):
        profile.admit_memory_dce(raw)
