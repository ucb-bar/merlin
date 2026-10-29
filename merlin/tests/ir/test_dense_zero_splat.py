"""Exact splat initialization proves tensor values, never buffer ownership."""

from types import SimpleNamespace

import pytest
from test_dense_capture_metadata import annotate
from test_dense_source import compile_source
from test_dense_source import source as dense_source
from test_selected_dense_contracts import inputs
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.ir import Block, Region

from merlin.compile import route_before_build as RB
from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, _problem, schedule_source_dense
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from merlin.targetgen import routing
from merlin.targetgen.capsule_source import model_op_demands_checked
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry
from merlin.transforms.pipeline_state import PassSpec, PipelineSession


def source(shape=(2, 3, 4), dtypes=("i8", "i8", "i32")):
    module, dot = dense_source(shape, dtypes)
    annotate(module)
    fill = dot.outputs[0].owner
    empty = fill.outputs[0].owner
    splat = tensor.SplatOp(fill.inputs[0], [], fill.results[0].type)
    splat.attributes.update(fill.attributes)
    fill.parent.insert_op_before(splat, fill)
    fill.results[0].replace_all_uses_with(splat.result)
    fill.parent.erase_op(fill)
    empty.parent.erase_op(empty)
    module.verify()
    return module, dot, splat


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 5, 2)])
@pytest.mark.parametrize(
    "dtypes", [("i8", "i8", "i32"), ("i16", "i8", "i32"), ("i16", "i16", "i64"), ("i32", "i32", "i32")]
)
def test_static_splat_zero_reads_actual_rectangular_mixed_width_geometry_without_source_edits(shape, dtypes):
    module, dot, _ = source(shape, dtypes)
    original = ir_text(module)
    assert static_zero_integer_matmul_geometry(dot, dtypes) == shape
    selected = SimpleNamespace(dense=SimpleNamespace(dtypes=dtypes, problem=lambda *values: values))
    assert _problem(dot, selected, 1024) == shape
    assert ir_text(module) == original
    assert not any(op.name in ("tensor.empty", "linalg.fill") for op in module.walk())


@pytest.mark.parametrize(
    "mutation",
    [
        "nonzero",
        "negative",
        "wrong_literal_width",
        "wrong_result_width",
        "nonconstant",
        "late_constant",
        "detached_constant",
        "dynamic_operand",
        "missing_scalar",
        "missing_result",
        "unknown_property",
        "constant_property",
        "unknown_attribute",
        "constant_attribute",
        "unknown_provenance",
        "region",
        "dynamic_shape",
        "zero_extent",
        "encoding",
    ],
)
def test_splat_source_rejects_incomplete_or_unknown_value_contracts_without_mutation(mutation):
    module, dot, splat = source()
    constant = splat.input.owner
    if mutation in ("nonzero", "negative", "wrong_literal_width"):
        constant.properties["value"] = builtin.IntegerAttr(
            -1 if mutation == "negative" else 1 if mutation == "nonzero" else 0,
            builtin.i16 if mutation == "wrong_literal_width" else builtin.i32,
        )
    elif mutation == "wrong_result_width":
        constant.result._type = builtin.i16
    elif mutation == "nonconstant":
        addition = arith.AddiOp(constant.result, constant.result)
        splat.parent.insert_op_before(addition, splat)
        splat.operands = (addition.result,)
    elif mutation == "late_constant":
        constant.detach()
        splat.parent.insert_op_after(constant, splat)
    elif mutation == "detached_constant":
        constant.detach()
    elif mutation == "dynamic_operand":
        splat.operands = (constant.result, constant.result)
    elif mutation == "missing_scalar":
        splat.operands = ()
    elif mutation == "missing_result":
        splat.results = ()
    elif mutation in ("unknown_property", "constant_property"):
        (constant if mutation == "constant_property" else splat).properties["unknown_effect"] = builtin.StringAttr(
            "opaque"
        )
    elif mutation in ("unknown_attribute", "constant_attribute", "unknown_provenance"):
        (constant if mutation == "constant_attribute" else splat).attributes[
            "prov.unknown_effect" if mutation == "unknown_provenance" else "unknown_effect"
        ] = builtin.StringAttr("opaque")
    elif mutation == "region":
        splat.add_region(Region(Block()))
    elif mutation in ("dynamic_shape", "zero_extent"):
        typ = builtin.TensorType(builtin.i32, (-1 if mutation == "dynamic_shape" else 0, 4))
        splat.result._type = typ
        dot.results[0]._type = typ
    else:
        typ = builtin.TensorType(builtin.i32, (2, 4), builtin.StringAttr("unqualified-layout"))
        splat.result._type = typ
        dot.results[0]._type = typ
    before = tuple(
        (op, tuple(op.operands), tuple(op.results), dict(op.attributes), dict(op.properties)) for op in module.walk()
    )
    with pytest.raises(ValueError):
        static_zero_integer_matmul_geometry(dot)
    assert (
        tuple(
            (op, tuple(op.operands), tuple(op.results), dict(op.attributes), dict(op.properties))
            for op in module.walk()
        )
        == before
    )


def test_splat_observer_and_current_source_analysis_are_preserved_independently():
    module, dot, splat = source()
    function = module.body.block.first_op
    returned = function.body.block.last_op
    returned.operands = (dot.results[0], splat.result)
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        [v.type for v in function.body.block.args], [v.type for v in returned.operands]
    )
    module.verify()
    original = ir_text(module)
    assert len(tuple(splat.result.uses)) == 2
    session = PipelineSession(module, ir_level="linalg")
    session.run(
        PassSpec("zero-value", 0, produces=("geometry",), mutates_ir=False, ir_level="linalg"),
        lambda: {"geometry": static_zero_integer_matmul_geometry(dot)},
    )
    assert session.analysis("geometry") == (2, 3, 4)
    assert ir_text(module) == original and returned.operands[-1] is splat.result
    splat.attributes["prov.origin_node_ids"] = builtin.ArrayAttr([builtin.StringAttr("changed")])
    assert static_zero_integer_matmul_geometry(dot) == (2, 3, 4)
    with pytest.raises(ValueError, match="stale"):
        session.analysis("geometry")


def test_splat_value_never_qualifies_a_fresh_writable_call_argument():
    module, dot, splat = source()
    declaration = func.FuncOp.external("unqualified_writer", [v.type for v in dot.operands], [dot.results[0].type])
    declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    declaration.properties["arg_attrs"] = builtin.ArrayAttr(
        [builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr(v)}) for v in ("read", "read", "write")]
    )
    module.body.block.add_op(declaration)
    call = func.CallOp("unqualified_writer", tuple(dot.operands), [dot.results[0].type])
    dot.parent.insert_op_before(call, dot)
    dot.results[0].replace_all_uses_with(call.results[0])
    dot.parent.erase_op(dot)
    module.verify()
    assert len(tuple(splat.result.uses)) == 1
    original = ir_text(module)
    with pytest.raises(ValueError, match="sole-use tensor.empty"):
        rewrite_fresh_tensor_writers(
            module, [FreshTensorWriterContract("unqualified_writer", 2, (2,), "unqualified_borrowed", 64)]
        )
    assert ir_text(module) == original


def test_dense_storage_disjointness_and_issued_source_custody_remain_required(tmp_path):
    model = inputs(tmp_path)
    selected, _, storage, adapter = model
    module, dot, splat = source((5, 14, 13), selected.dense.dtypes)
    with pytest.raises(ValueError, match="possibly aliased"):
        schedule_source_dense(
            module,
            dot,
            selected_contracts=selected,
            storage=storage,
            adapter=adapter,
            name="unqualified",
            static_budget=80,
            external_buffers_disjoint=False,
        )
    result = compile_source(model, module, dot)
    assert result is not None
    result.revalidate()
    splat.input.owner.properties["value"] = builtin.IntegerAttr(1, builtin.i32)
    with pytest.raises(DenseSourceUnavailable, match="zero"):
        _problem(dot, selected, 1024)
    with pytest.raises(ValueError, match="changed; replan"):
        result.revalidate()


def test_ordinary_typed_placement_reads_splat_source_before_any_target_qualification(monkeypatch):
    from merlin.llvmlower import group_offload
    from merlin.system import derive

    module, _, _ = source()
    original = ir_text(module)
    demands = model_op_demands_checked(original, "int8", source_kind="typed_static_integer_v1")
    assert len(demands) == 1 and (demands[0].m, demands[0].k, demands[0].n) == (2, 3, 4)
    observed = []

    def route(rows, target):
        observed.append(rows)
        return routing.route_plan_on(rows, [])

    def unavailable(*_args, **_kwargs):
        raise ValueError("no original target capability supplied in source-only test")

    monkeypatch.setattr(routing, "route_plan", route)
    monkeypatch.setattr(derive, "system_for_experiment", unavailable)
    monkeypatch.setattr(group_offload, "plan", unavailable)
    result = RB.plan_before_build("unqualified", original, datapath="int8", source_kind="typed_static_integer_v1")
    assert len(observed) == 1 and len(result["plan"]["scalar_rvv"]) == 1
    assert ir_text(module) == original


def observed_native_source(m, k, dtype, *, source_factory=None):
    """Reuse the ordinary native model, retaining its first zero value as output."""
    from test_native_offload_model import source as ordinary_source

    module = (ordinary_source if source_factory is None else source_factory)(m, k, dtype)
    reductions = [op for op in module.walk() if op.name == "linalg.generic" and len(op.iterator_types) == 3]
    assert len(reductions) == 2
    observed_zero = None
    for ordinal, dot in enumerate(reductions):
        assert static_zero_integer_matmul_geometry(dot, (dtype, dtype, "i32")) == (m, k, k)
        fill = dot.outputs[0].owner
        empty = fill.outputs[0].owner
        ancestry = {
            "prov.family": builtin.StringAttr("contraction"),
            "prov.op": builtin.StringAttr("int_matmul"),
            "prov._pattern_hint": builtin.StringAttr("int_matmul"),
            "prov.aten": builtin.StringAttr("aten._int_mm.default"),
            "prov.orig_dtype": builtin.StringAttr("int32"),
            "prov.source_node_ids": builtin.ArrayAttr([builtin.StringAttr(f"neutral:prepared:{ordinal}")]),
            "prov.origin_node_ids": builtin.ArrayAttr([builtin.StringAttr(f"neutral:original:{ordinal}")]),
            "prov.trace_role": builtin.StringAttr("lowering"),
        }
        for operation in (*tuple(dot.walk()), fill.inputs[0].owner):
            operation.attributes.update(ancestry)
        splat = tensor.SplatOp(fill.inputs[0], [], fill.results[0].type)
        splat.attributes.update(ancestry)
        fill.parent.insert_op_before(splat, fill)
        fill.results[0].replace_all_uses_with(splat.result)
        fill.parent.erase_op(fill)
        empty.parent.erase_op(empty)
        assert static_zero_integer_matmul_geometry(dot, (dtype, dtype, "i32")) == (m, k, k)
        if observed_zero is None:
            observed_zero = splat.result
    function = module.body.block.first_op
    returned = function.body.block.last_op
    returned.operands = (*returned.operands, observed_zero)
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        [value.type for value in function.body.block.args], [value.type for value in returned.operands]
    )
    module.verify()
    assert len(tuple(observed_zero.uses)) == 2
    return module


def observed_native_reference(m, k, width, *, reference_factory=None):
    import numpy as np
    from test_native_offload_model import reference as ordinary_reference

    arguments, expected, trace = (ordinary_reference if reference_factory is None else reference_factory)(m, k, width)
    return arguments, [(*row, np.zeros((m, k), np.int32)) for row in expected], trace


def test_native_fixture_declares_two_splats_and_complete_additional_zero_observer():
    module = observed_native_source(3, 5, "i8")
    original = ir_text(module)
    splats = [op for op in module.walk() if type(op) is tensor.SplatOp]
    assert len(splats) == 2
    assert sorted(len(tuple(op.result.uses)) for op in splats) == [1, 2]
    assert len(module.body.block.first_op.function_type.outputs) == 8
    demands = model_op_demands_checked(original, "int8", source_kind="typed_static_integer_v1")
    assert [(d.m, d.k, d.n) for d in demands if d.family == "contraction"] == [(3, 5, 5), (3, 5, 5)]
    _, expected, _ = observed_native_reference(3, 5, 1)
    assert len(expected) == 3 and all(len(epoch) == 8 and not epoch[-1].any() for epoch in expected)
    assert ir_text(module) == original


def install_observed_native_case(monkeypatch):
    import test_native_offload_model as original

    source_factory, reference_factory = original.source, original.reference

    def current_source(m, k, dtype):
        return observed_native_source(m, k, dtype, source_factory=source_factory)

    def current_reference(m, k, width):
        return observed_native_reference(m, k, width, reference_factory=reference_factory)

    monkeypatch.setattr(original, "source", current_source)
    monkeypatch.setattr(original, "reference", current_reference)
    return original


def test_native_fixture_installs_the_original_factories_without_recursing(monkeypatch):
    original = install_observed_native_case(monkeypatch)
    module = original.source(3, 5, "i8")
    module.verify()
    assert len(module.body.block.first_op.function_type.outputs) == 8
    _, expected, _ = original.reference(3, 5, 1)
    assert len(expected) == 3 and all(len(row) == 8 for row in expected)


def test_native_ordinary_source_model_preserves_observed_zero_splat(tmp_path, monkeypatch):
    """Actual execution only when the existing native toolchain is selected.

    This extends the existing three-arm ordinary model test, not a geometry or
    C-emitter shortcut. Parent qualification must run this explicit node.
    """
    import json

    original = install_observed_native_case(monkeypatch)
    original.test_complete_native_source_plan_selects_exact_same_shape_operations(
        tmp_path, monkeypatch, (3, 5), (2, 3, 4), 1, False
    )
    pair = json.loads((tmp_path / "pair.json").read_bytes())
    assert len(pair) == 3
    for arm in pair:
        assert len(arm["history"]) == 3
        for epoch in arm["history"]:
            assert len(epoch) == 8 and epoch[-1] == [[0] * 5 for _ in range(3)]
