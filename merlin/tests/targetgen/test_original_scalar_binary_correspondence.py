"""Independent one-call IR fixtures challenge scalar contents and real SSA uses.

Synthetic trace/registry fixtures are reader controls, never native observations
or original protected-corpus coverage. Mutations regenerate the trace hashes so
semantic refusals cannot be satisfied by only a digest or type-roster check.
"""

import copy
import hashlib

import pytest
from test_original_scalar_binary_sources import declarations, form
from xdsl.dialects import arith, linalg, tensor
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, ModuleOp, StringAttr, TensorType, f32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import IteratorType, IteratorTypeAttr
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap

from merlin.targetgen import original_scalar_binary_correspondence as C
from merlin.targetgen import original_scalar_binary_sources as S
from merlin.targetgen.application_graph import _program_graph
from merlin.targetgen.frontend_trace import _digest
from merlin.xdsl_dialects._common import text

LIMITS = {"max_source_bytes": 100000, "max_nesting": 32, "max_operations": 20, "max_tensor_elements": 100}
SOURCES = {"/selected/decompositions.py": "a" * 64, "/selected/import_fx.py": "b" * 64}


def fixture(target="aten.div.Tensor", scalar=2.23606797749979):
    original = form(target, scalar)
    source = S.scalar_binary_source(original, extent=2, max_tensor_elements=100)
    value_type = TensorType(f32, [2, 3])
    block = Block(arg_types=[value_type])
    constant = arith.ConstantOp(FloatAttr(scalar, f32))
    splat = tensor.SplatOp(constant.result, [], value_type)
    empty = tensor.EmptyOp([], value_type)
    scalar_block = Block(arg_types=[f32, f32, f32])
    arithmetic = arith.MulfOp if target == "aten.mul.Tensor" else arith.DivfOp
    calculation = arithmetic(*scalar_block.args[:2])
    scalar_block.add_ops([calculation, linalg.YieldOp(calculation.result)])
    generic = linalg.GenericOp(
        inputs=[block.args[0], splat.result],
        outputs=[empty.results[0]],
        body=Region(scalar_block),
        indexing_maps=[AffineMapAttr(AffineMap.identity(2))] * 3,
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * 2,
        result_types=[value_type],
    )
    block.add_ops([constant, splat, empty, generic, ReturnOp(generic.results[0])])
    module = ModuleOp([FuncOp("original_scalar", ([value_type], [value_type]), Region(block))])
    graphs = {}
    for stage in ("original", "quantized", "prepared"):
        graph = declarations(target, scalar, shape=(2, 3))[0]["graphs"]["original"]

        def remap(value):
            if type(value) is str and value in {"input", "input:0", "binary", "binary:0", "output"}:
                return stage + ":" + value
            if type(value) is dict:
                return {key: item if key in {"op", "target"} else remap(item) for key, item in value.items()}
            if type(value) is list:
                return [remap(item) for item in value]
            return value

        graph = remap(graph)
        graph.update(stage=stage)
        graph.pop("sha256")
        graph["sha256"] = _digest(graph)
        graphs[stage] = graph
    origins = ["original:binary", "quantized:binary"]
    for op in (constant, splat, *generic.walk()):
        op.attributes.update(
            {
                "prov.source_node_ids": ArrayAttr([StringAttr("prepared:binary")]),
                "prov.origin_node_ids": ArrayAttr([StringAttr(value) for value in origins]),
                "prov.trace_role": StringAttr("semantic"),
            }
        )
    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "blockers": [],
        "graphs": graphs,
        "transformations": [
            {
                "from_stage": left,
                "to_stage": right,
                "status": "complete",
                "relations": [{"source_ids": [left + ":binary"], "destination_ids": [right + ":binary"]}],
                "unresolved_source_ids": [],
                "unresolved_destination_ids": [],
            }
            for left, right in (("original", "quantized"), ("quantized", "prepared"))
        ],
    }
    registry = {
        "schema": C.REGISTRY_SCHEMA,
        "target": target,
        "function": {
            "module": "m2m.ir.decompositions",
            "name": C.REGISTRY_FUNCTIONS[target],
            "path": "/selected/decompositions.py",
            "sha256": SOURCES["/selected/decompositions.py"],
        },
        "importer": {
            "module": "m2m.ir.import_fx",
            "name": "FXImporter.import_graph",
            "path": "/selected/import_fx.py",
            "sha256": SOURCES["/selected/import_fx.py"],
        },
        "events": [
            {
                "target": target,
                "literal": copy.deepcopy(original["parameters"]["other"]),
                "operand_types": [str(value_type)],
                "result_type": str(value_type),
                "source_node_id": "prepared:binary",
                "origin_node_ids": origins,
                "emitted_operations": ["arith.constant", "tensor.splat", "tensor.empty", "linalg.generic"],
                "dynamic_overrides": [],
            }
        ],
    }
    return original, source, module, trace, registry


def products(module, trace):
    payload = text(module)
    raw = _program_graph(module, hashlib.sha256(payload.encode()).hexdigest())
    trace = copy.deepcopy(trace)
    trace["mlir"] = {
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "bytes": len(payload.encode()),
        "operations": [
            {
                "ordinal": row["ordinal"],
                "operation": row["mlir_operation"],
                "operand_types": [value["type"] for value in row["operands"]],
                "result_types": [value["type"] for value in row["results"]],
                "source_node_ids": row["source_node_ids"],
                "origin_node_ids": row["origin_node_ids"],
                "role": row["trace_role"]
                or (
                    "structural"
                    if row["mlir_operation"] in {"builtin.module", "func.func", "func.return"}
                    else "unresolved"
                ),
            }
            for row in raw["operations"]
        ],
        "source_correspondence": [
            {
                "node_id": "prepared:binary",
                "status": "lowered",
                "mlir_ordinals": [
                    row["ordinal"] for row in raw["operations"] if "prepared:binary" in row["source_node_ids"]
                ],
            }
        ],
    }
    return payload, trace


def checked(values, **kwargs):
    original, source, module, trace, registry = values
    payload, trace = products(module, trace)
    return C.verify(
        original,
        source,
        extent=2,
        text=payload,
        trace=trace,
        registry=registry,
        source_inventory=SOURCES,
        limits={**LIMITS, **kwargs},
    )


@pytest.mark.parametrize("target", ["aten.mul.Tensor", "aten.div.Tensor"])
@pytest.mark.parametrize("scalar", [1.0, 2.23606797749979, -0.0, 0.1, 1.0 + 2**-24, 2**-149])
def test_complete_literal_order_result_and_registered_construction_is_separate_from_admission(target, scalar):
    record = checked(fixture(target, scalar))
    assert record["coefficient_f32_bits"] == C.coefficient_bits({"kind": "float", "value_hex": scalar.hex()})
    assert record["ordered_abi"] == {"inputs": ["tensor<2x3xf32>"], "outputs": ["tensor<2x3xf32>"]}
    assert "no numerical" in record["scope"]


@pytest.mark.parametrize(
    "change",
    [
        "coefficient",
        "generic_order",
        "empty_output",
        "div_order",
        "yield",
        "return",
        "map",
        "iterator",
        "fastmath",
        "opaque_attr",
    ],
)
def test_type_and_digest_equivalent_products_cannot_substitute_actual_scalar_body_relations(change):
    values = fixture()
    block = next(iter(values[2].body.block.ops)).body.block
    constant, splat, _, generic, returned = tuple(block.ops)
    calculation, yielded = tuple(generic.body.block.ops)
    if change == "coefficient":
        constant.properties["value"] = FloatAttr(3.0, f32)
    elif change == "generic_order":
        generic.operands = [splat.result, block.args[0], generic.outputs[0]]
    elif change == "empty_output":
        generic.operands = [block.args[0], splat.result, splat.result]
    elif change == "div_order":
        calculation.operands = list(reversed(calculation.operands))
    elif change == "yield":
        yielded.operands = [generic.body.block.args[0]]
    elif change == "return":
        returned.operands = [block.args[0]]
    elif change == "map":
        maps = list(generic.indexing_maps)
        maps[0] = AffineMapAttr(AffineMap(2, 0, (AffineExpr.dimension(1), AffineExpr.dimension(0))))
        generic.properties["indexing_maps"] = ArrayAttr(maps)
    elif change == "iterator":
        generic.properties["iterator_types"] = ArrayAttr([IteratorTypeAttr(IteratorType.REDUCTION)] * 2)
    elif change == "fastmath":
        calculation.properties["fastmath"] = arith.FastMathFlagsAttr("fast")
    else:
        calculation.attributes["unproved_semantics"] = StringAttr("different")
    with pytest.raises(ValueError):
        checked(values)


@pytest.mark.parametrize("change", ["literal", "kind", "second_ssa", "dtype", "schema", "result", "transition"])
def test_complete_actual_frontend_stage_semantics_must_match_the_original_call(change):
    values = fixture()
    graph = values[3]["graphs"]["prepared"]
    call = graph["nodes"][1]
    if change == "literal":
        call["args"][1] = 3.0
    elif change == "kind":
        call["args"][1] = 2
    elif change == "second_ssa":
        call["args"][1] = copy.deepcopy(call["args"][0])
    elif change == "dtype":
        call["results"][0]["storage_dtype"] = "float64"
    elif change == "schema":
        graph["operator_schemas"][call["target"]] = "changed"
    elif change == "result":
        graph["nodes"][2]["args"][0]["value_id"] = "prepared:input:0"
    else:
        values[3]["transformations"].pop()
    graph.pop("sha256")
    graph["sha256"] = _digest(graph)
    with pytest.raises(ValueError):
        checked(values)


@pytest.mark.parametrize("change", ["callable", "source", "literal", "override", "result", "duplicate", "missing"])
def test_registry_source_and_actual_invocation_joins_cannot_be_replaced_by_matching_ids(change):
    values = fixture()
    registry = values[4]
    event = registry["events"][0]
    if change == "callable":
        registry["function"]["name"] = "substitute"
    elif change == "source":
        registry["function"]["sha256"] = "c" * 64
    elif change == "literal":
        event["literal"]["value_hex"] = (3.0).hex()
    elif change == "override":
        event["dynamic_overrides"] = [registry["target"]]
    elif change == "result":
        event["result_type"] = "tensor<2x3xf64>"
    elif change == "duplicate":
        registry["events"].append(copy.deepcopy(event))
    else:
        registry["events"].clear()
    with pytest.raises(ValueError):
        checked(values)


@pytest.mark.parametrize("field,value", [("max_tensor_elements", 11), ("max_operations", 8), ("max_source_bytes", 100)])
def test_complete_preallocation_and_source_roster_budgets_cannot_be_relaxed(field, value):
    with pytest.raises(ValueError):
        checked(fixture(), **{field: value})


@pytest.mark.parametrize("field", sorted(LIMITS))
def test_boolean_is_not_a_parser_or_logical_budget(field):
    with pytest.raises(ValueError):
        checked(fixture(), **{field: True})


def test_literal_underflow_preserves_signed_zero_and_overflow_is_not_licensed():
    assert C.coefficient_bits({"kind": "float", "value_hex": (-(2**-150)).hex()}) == "80000000"
    with pytest.raises(ValueError, match="outside finite"):
        C.coefficient_bits({"kind": "float", "value_hex": (2**128 * 1.0).hex()})
