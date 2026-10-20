"""Explicit public typed source intake grants no capture or target qualification."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i32

from merlin.common import mlir_query as mq
from merlin.compile import route_before_build as RB
from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, _problem
from merlin.targetgen import routing
from merlin.targetgen.capsule_source import ModelDemandIncomplete, model_op_demands_checked
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry

MODE = "typed_static_integer_v1"


@pytest.fixture
def source():
    return (Path(__file__).resolve().parents[1] / "ir/producer_packing_connected.mlir").read_text()


def demands(source, requested="int8"):
    return model_op_demands_checked(source, requested, source_kind=MODE)


def test_explicit_typed_source_observes_both_dots_and_keeps_host_work_without_provenance(source):
    before = ir_text(mq.parse(source))
    observed = demands(source)
    dots = [row for row in observed if row.family == "contraction"]
    assert [(row.batch, row.m, row.k, row.n, row.rank) for row in dots] == [(1, 2, 4, 12, 2), (1, 2, 12, 12, 2)]
    assert all(row.op == "matmul" and row.carrier_op == "linalg.generic" for row in dots)
    assert all(row.elem_fmt == "int8" and row.captured_input_formats == ("int8", "int8") for row in dots)
    assert all(row.region_id is None and row.source_family is None for row in observed)
    assert [row.op for row in observed] == ["matmul", "linalg.generic", "linalg.transpose", "matmul"]
    assert all(row.family is None for row in observed if row not in dots)
    assert ir_text(mq.parse(source)) == before and "prov.op" not in before
    assert len({row.site for row in observed}) == len(observed)
    assert [(row.elem_fmt, row.admission_input_fmt) for row in demands(source, "fp32") if row.family] == [
        ("int8", "int8"),
        ("int8", "int8"),
    ]
    with pytest.raises(ModelDemandIncomplete, match="untagged"):
        model_op_demands_checked(source, "int8")


def test_captured_default_keeps_exact_old_demands_and_lost_tag_refusal(source):
    module = mq.parse(source).clone()
    dots = [op for op in module.walk() if op.name == "linalg.generic" and len(tuple(op.iterator_types)) == 3]
    for op in dots:
        op.attributes["prov.op"] = StringAttr("matmul")
        op.attributes["prov.family"] = StringAttr("contraction")
    source = ir_text(module)
    observed = model_op_demands_checked(source, "int8")
    assert [(row.op, row.m, row.k, row.n) for row in observed] == [("matmul", 2, 4, 12), ("matmul", 2, 12, 12)]
    assert observed == model_op_demands_checked(source, "int8", source_kind="captured_provenance_v1")
    del dots[0].attributes["prov.op"]
    source = ir_text(module)
    with pytest.raises(ModelDemandIncomplete, match="untagged"):
        model_op_demands_checked(source, "int8")


@pytest.mark.parametrize(
    "mutation",
    [
        "unsigned",
        "product",
        "accumulator",
        "extra_scalar",
        "nonzero",
        "initializer",
        "map",
        "extent",
        "dynamic",
        "zero_extent",
        "width",
        "flags",
        "unknown_attribute",
    ],
)
def test_typed_reductions_never_disappear_when_the_exact_contract_is_broken(source, mutation):
    changes = {
        "unsigned": ("arith.extsi %a_wide", "arith.extui %a_wide"),
        "product": ("arith.muli %ae_wide,%be_wide", "arith.addi %ae_wide,%be_wide"),
        "accumulator": ("arith.addi %c_wide,%mul_wide", "arith.addi %mul_wide,%mul_wide"),
        "extra_scalar": (
            "%acc_wide=arith.addi",
            "%extra = arith.addi %mul_wide, %mul_wide : i32\n        %acc_wide=arith.addi",
        ),
        "nonzero": ("%z = arith.constant 0", "%z = arith.constant 1"),
        "initializer": ("outs(%zero1:tensor<2x12xi32>)", "outs(%state:tensor<2x12xi32>)"),
        "map": ("affine_map<(m,n,k)->(m,k)>", "affine_map<(m,n,k)->(m,n)>"),
        "extent": ("tensor<4x12xi8>", "tensor<5x12xi8>"),
        "dynamic": ("tensor<2x4xi8>", "tensor<2x?xi8>"),
        "zero_extent": ("tensor<2x4xi8>", "tensor<2x0xi8>"),
        "width": ("i32", "i16"),
    }
    if mutation in changes:
        old, new = changes[mutation]
        source = source.replace(old, new)
    else:
        module = mq.parse(source).clone()
        dot = next(op for op in module.walk() if op.name == "linalg.generic")
        if mutation == "flags":
            multiply = next(op for op in dot.walk() if op.name == "arith.muli")
            multiply.properties["overflowFlags"] = arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
        else:
            from xdsl.dialects.builtin import StringAttr

            dot.attributes["unrecognized_effect"] = StringAttr("not a capability")
        source = ir_text(module)
    with pytest.raises(ModelDemandIncomplete):
        demands(source)


def test_generic_serialization_and_source_order_keep_exact_demand_correspondence(source):
    module = mq.parse(source).clone()
    observed = demands(ir_text(module))
    assert [(row.m, row.k, row.n) for row in observed if row.family] == [(2, 4, 12), (2, 12, 12)]
    dots = [op for op in module.walk() if op.name == "linalg.generic" and len(tuple(op.iterator_types)) == 3]
    # A same-shaped second product cannot inherit the first product's observed K.
    assert static_zero_integer_matmul_geometry(dots[0]) == (2, 4, 12)
    assert static_zero_integer_matmul_geometry(dots[1]) == (2, 12, 12)


@pytest.mark.parametrize("metadata", ["effect", "iterator", "library_call"])
def test_unsupported_host_generic_metadata_is_not_silently_classified_as_host(source, metadata):
    from xdsl.dialects.builtin import ArrayAttr

    module = mq.parse(source).clone()
    host = next(op for op in module.walk() if op.name == "linalg.generic" and len(tuple(op.iterator_types)) == 2)
    if metadata == "effect":
        host.attributes["executable_unknown"] = StringAttr("opaque")
    elif metadata == "iterator":
        host.properties["iterator_types"] = ArrayAttr([StringAttr("reduction"), StringAttr("parallel")])
    else:
        host.properties["library_call"] = StringAttr("opaque")
    with pytest.raises(ModelDemandIncomplete):
        demands(ir_text(module))


def test_source_geometry_refactor_preserves_dense_problem_and_initializer_refusals(source):
    module = mq.parse(source).clone()
    operation = next(op for op in module.walk() if op.name == "linalg.generic")
    selected = SimpleNamespace(dense=SimpleNamespace(dtypes=("i8", "i8", "i32"), problem=lambda *shape: shape))
    assert _problem(operation, selected, 64 << 20) == static_zero_integer_matmul_geometry(operation)
    with pytest.raises(DenseSourceUnavailable, match="element budget"):
        _problem(operation, selected, 1)
    zero = operation.outputs[0].owner.inputs[0].owner
    zero.properties["value"] = IntegerAttr(1, i32)
    with pytest.raises(ValueError, match="exact integer zero"):
        static_zero_integer_matmul_geometry(operation)
    with pytest.raises(DenseSourceUnavailable, match="exact integer zero"):
        _problem(operation, selected, 64 << 20)


def test_ordinary_placement_reaches_original_router_only_with_explicit_typed_intake(source, monkeypatch):
    from merlin.llvmlower import group_offload
    from merlin.system import derive

    seen = []
    real_route = routing.route_plan_on

    def route(rows, target):
        seen.append((rows, target))
        return real_route(rows, [])

    monkeypatch.setattr(routing, "route_plan", route)

    def unavailable(*_args, **_kwargs):
        raise ValueError("source-only test has no selected target capability")

    monkeypatch.setattr(derive, "system_for_experiment", unavailable)
    monkeypatch.setattr(group_offload, "plan", unavailable)
    original = RB.plan_before_build("neutral", source, datapath="int8", source_kind=MODE)
    assert len(seen) == 1 and seen[0][1] == "neutral"
    assert original["source_kind"] == MODE
    assert len(original["plan"]["scalar_rvv"]) == 4
    with pytest.raises(ModelDemandIncomplete, match="untagged"):
        RB.plan_before_build("no_lookup", source, datapath="int8")
    assert len(seen) == 1
    with pytest.raises(ModelDemandIncomplete):
        RB.plan_before_build(
            "no_lookup", source.replace("arith.extsi", "arith.extui"), datapath="int8", source_kind=MODE
        )
    assert len(seen) == 1


@pytest.mark.parametrize("mode", [None, True, "auto", "", "typed_static_integer_v2"])
def test_unknown_intake_modes_refuse_before_target_lookup(source, monkeypatch, mode):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid intake must not query a target")

    monkeypatch.setattr(routing, "route_plan", forbidden)
    with pytest.raises(ValueError):
        RB.plan_before_build("neutral", source, datapath="int8", source_kind=mode)
    with pytest.raises(ModelDemandIncomplete):
        model_op_demands_checked(source, "int8", source_kind=mode)
