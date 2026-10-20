"""Capture attribution stays inert in exact typed integer dense recognition."""

from types import SimpleNamespace

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, i32

from merlin.common import mlir_query as mq
from merlin.common.paths import merlin_dir
from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, _problem
from merlin.targetgen.capsule_source import ModelDemandIncomplete, model_op_demands_checked
from merlin.transforms.certificates import ir_digest, ir_text
from merlin.transforms.dense_ir import recognize_signed_integer_matmul, static_zero_integer_matmul_geometry
from merlin.transforms.pipeline_state import PassSpec, PipelineSession


def source(dtypes=("i8", "i8", "i32")):
    text = (merlin_dir() / "tests/fixtures/shared_dense_source.mlir").read_text()
    # Independent rectangular M=2,K=3,N=4, unequal input formats supported.
    for old, new in zip(("%x: i8", "%y: i8"), (f"%x: {dtypes[0]}", f"%y: {dtypes[1]}"), strict=True):
        text = text.replace(old, new)
    text = text.replace("2x3xi8", f"2x3x{dtypes[0]}").replace("3x4xi8", f"3x4x{dtypes[1]}")
    text = text.replace("arith.extsi %x : i8", f"arith.extsi %x : {dtypes[0]}")
    text = text.replace("arith.extsi %y : i8", f"arith.extsi %y : {dtypes[1]}")
    if dtypes[0] == "i32":
        text = text.replace("%u = arith.extsi %x : i32 to i32", "").replace("%u, %v", "%x, %v")
    if dtypes[1] == "i32":
        text = (
            text.replace("%v = arith.extsi %y : i32 to i32", "").replace("%u, %v", "%u, %y").replace("%x, %v", "%x, %y")
        )
    module = mq.parse(text)
    module.verify()
    return module


def operations(module):
    result = {}
    for op in module.walk():
        if op.name in ("arith.constant", "tensor.empty", "linalg.fill", "linalg.generic", "arith.muli", "arith.addi"):
            result[op.name] = op
        if op.name == "arith.extsi":
            result.setdefault(op.name, op)
    result["fill_yield"] = result["linalg.fill"].regions[0].block.last_op
    result["dot_yield"] = result["linalg.generic"].regions[0].block.last_op
    return result


def annotate(module, *, suffix="one"):
    # Same fields and types emitted by import_fx._forward_fx_meta and recursive
    # capture.trace.stamp_mlir_sources; values deliberately contradict the math.
    for op in operations(module).values():
        op.attributes.update(
            {
                "prov.region_id": StringAttr(f"unrelated-region-{suffix}"),
                "prov.fqn": StringAttr(f"source.module.{suffix}"),
                "prov.module": StringAttr("unrelated-source-module"),
                "prov.family": StringAttr("not-a-contraction"),
                "prov.op": StringAttr("not-a-matmul"),
                "prov._pattern_hint": StringAttr("not-a-matmul-hint"),
                "prov.aten": StringAttr("aten.unrelated.default"),
                "prov.orig_dtype": StringAttr("not-the-stored-dtype"),
                "prov.trace_role": StringAttr("support"),
                "prov.source_node_ids": ArrayAttr([StringAttr(f"prepared:{suffix}")]),
                "prov.origin_node_ids": ArrayAttr([StringAttr(f"original:{suffix}")]),
            }
        )


@pytest.mark.parametrize(
    "dtypes", [("i8", "i8", "i32"), ("i16", "i8", "i32"), ("i16", "i16", "i32"), ("i32", "i32", "i32")]
)
def test_captured_annotations_preserve_source_and_do_not_choose_formats_or_geometry(dtypes):
    module = source(dtypes)
    dot = operations(module)["linalg.generic"]
    expected = static_zero_integer_matmul_geometry(dot, dtypes)
    annotate(module)
    original = ir_text(module)
    module.verify()
    assert recognize_signed_integer_matmul(dot, dtypes) == (*dot.inputs, dot.outputs[0])
    assert static_zero_integer_matmul_geometry(dot, dtypes) == expected == (2, 3, 4)
    selected = SimpleNamespace(dense=SimpleNamespace(dtypes=dtypes, problem=lambda *shape: shape))
    assert _problem(dot, selected, 1024) == expected
    assert ir_text(module) == original
    assert 'prov.orig_dtype = "not-the-stored-dtype"' in original
    assert 'prov.source_node_ids = ["prepared:one"]' in original


def test_typed_intake_derives_actual_integer_demands_without_rewriting_capture_tags():
    module = source()
    annotate(module)
    original = ir_text(module)
    demands = model_op_demands_checked(original, "fp32", source_kind="typed_static_integer_v1")
    assert len(demands) == 1
    demand = demands[0]
    assert (demand.op, demand.family, demand.m, demand.k, demand.n) == ("matmul", "contraction", 2, 3, 4)
    assert demand.captured_input_formats == ("int8", "int8") and demand.admission_input_fmt == "int8"
    assert ir_text(module) == original


@pytest.mark.parametrize(
    "location",
    [
        "arith.constant",
        "tensor.empty",
        "linalg.fill",
        "linalg.generic",
        "arith.extsi",
        "arith.muli",
        "arith.addi",
        "fill_yield",
        "dot_yield",
    ],
)
@pytest.mark.parametrize(
    "name",
    [
        "unknown_effect",
        "prov.unknown_effect",
        "prov.placement",
        "prov.dispatch_id",
        "prov.quantization",
        "prov.transpose_absorbed",
        "prov.fuse_dequant",
    ],
)
def test_unknown_annotations_are_still_barriers_at_every_dense_component(location, name):
    module = source()
    annotate(module)
    ops = operations(module)
    ops[location].attributes[name] = StringAttr("not-a-permission")
    original = ir_text(module)
    with pytest.raises(ValueError):
        static_zero_integer_matmul_geometry(ops["linalg.generic"])
    with pytest.raises(DenseSourceUnavailable):
        _problem(ops["linalg.generic"], SimpleNamespace(dense=SimpleNamespace(dtypes=("i8", "i8", "i32"))), 1024)
    assert ir_text(module) == original


@pytest.mark.parametrize(
    "name,value",
    [
        ("prov.family", IntegerAttr.from_bool(True)),
        ("prov.region_id", ArrayAttr([])),
        ("prov.source_node_ids", StringAttr("wrong-array")),
        ("prov.origin_node_ids", ArrayAttr([IntegerAttr(1, i32)])),
    ],
)
@pytest.mark.parametrize("location", ["linalg.generic", "arith.extsi", "arith.muli", "linalg.fill", "dot_yield"])
def test_known_metadata_requires_its_actual_capture_type(name, value, location):
    module = source()
    annotate(module)
    ops = operations(module)
    ops[location].attributes[name] = value
    with pytest.raises(ValueError):
        static_zero_integer_matmul_geometry(ops["linalg.generic"])


@pytest.mark.parametrize(
    "mutation",
    [
        "unsigned",
        "duplicate_product",
        "wrong_accumulator",
        "wrong_yield",
        "nonzero",
        "fill_body",
        "flags_nsw",
        "flags_nuw",
        "generic_property",
        "scalar_property",
        "map",
    ],
)
def test_annotations_never_rescue_wrong_mathematics_or_executable_properties(mutation):
    module = source()
    annotate(module)
    ops = operations(module)
    if mutation == "unsigned":
        module = mq.parse(ir_text(module).replace('"arith.extsi"', '"arith.extui"', 1))
        ops = operations(module)
    elif mutation == "duplicate_product":
        op = ops["arith.muli"]
        op.operands = (op.operands[0], op.operands[0])
    elif mutation == "wrong_accumulator":
        op = ops["arith.addi"]
        op.operands = (op.operands[1], op.operands[1])
    elif mutation == "wrong_yield":
        ops["dot_yield"].operands = (ops["arith.muli"].results[0],)
    elif mutation == "nonzero":
        ops["arith.constant"].properties["value"] = IntegerAttr(1, i32)
    elif mutation == "fill_body":
        ops["fill_yield"].operands = (ops["linalg.fill"].regions[0].block.args[1],)
    elif mutation.startswith("flags_"):
        ops["arith.addi"].properties["overflowFlags"] = arith.IntegerOverflowAttr(
            [getattr(arith.IntegerOverflowFlag, mutation.removeprefix("flags_").upper())]
        )
    elif mutation.endswith("property"):
        ops["linalg.generic" if mutation == "generic_property" else "arith.extsi"].properties["prov.op"] = StringAttr(
            "not-executable-permission"
        )
    else:
        text = ir_text(module).replace("(d0, d2)", "(d0, d1)", 1)
        assert text != ir_text(module)
        ops = operations(mq.parse(text))
    original = ir_text(module)
    with pytest.raises(ValueError):
        static_zero_integer_matmul_geometry(ops["linalg.generic"])
    assert ir_text(module) == original


def test_changed_inert_ids_still_invalidate_current_source_analysis():
    module = source()
    annotate(module)
    dot = operations(module)["linalg.generic"]
    session = PipelineSession(module, ir_level="linalg")
    session.run(
        PassSpec("observe", 0, produces=("shape",), mutates_ir=False, ir_level="linalg"),
        lambda: {"shape": static_zero_integer_matmul_geometry(dot)},
    )
    assert session.analysis("shape") == (2, 3, 4)
    previous = ir_digest(module)
    annotate(module, suffix="two")
    assert static_zero_integer_matmul_geometry(dot) == (2, 3, 4) and ir_digest(module) != previous
    with pytest.raises(ValueError, match="stale"):
        session.analysis("shape")


def test_splat_initialization_remains_outside_this_metadata_only_extension():
    module = source()
    annotate(module)
    ops = operations(module)
    from xdsl.dialects import tensor

    fill = ops["linalg.fill"]
    splat = tensor.SplatOp(fill.inputs[0], [], fill.results[0].type)
    fill.parent.insert_op_before(splat, fill)
    fill.results[0].replace_all_uses_with(splat.result)
    fill.parent.erase_op(fill)
    module.verify()
    with pytest.raises(ValueError, match="zero fill"):
        static_zero_integer_matmul_geometry(ops["linalg.generic"])
    with pytest.raises(ModelDemandIncomplete, match="zero fill"):
        model_op_demands_checked(ir_text(module), "int8", source_kind="typed_static_integer_v1")


@pytest.mark.parametrize("location", ["linalg.generic", "arith.extsi"])
def test_issued_dense_result_refuses_changed_ids_even_when_geometry_is_unchanged(tmp_path, location):
    from test_dense_source import compile_source
    from test_dense_source import source as dense_source
    from test_selected_dense_contracts import inputs

    model = inputs(tmp_path)
    module, dot = dense_source((5, 14, 13), model[0].dense.dtypes)
    annotate(module)
    result = compile_source(model, module, dot)
    assert result is not None
    result.revalidate()
    before = static_zero_integer_matmul_geometry(dot)
    operations(module)[location].attributes["prov.origin_node_ids"] = ArrayAttr([StringAttr("new-origin")])
    assert static_zero_integer_matmul_geometry(dot) == before
    with pytest.raises(ValueError, match="changed; replan"):
        result.revalidate()
