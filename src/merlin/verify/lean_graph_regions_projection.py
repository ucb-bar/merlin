"""Independent actual-SSA projection for the initial connected integer region domain.

This host parser is a named trusted correspondence component. Lean recomputes plan
bindings/liveness from its full projected nodes; it does not verify this xDSL parser
or infer native primitive correctness from an operation name.
"""

from __future__ import annotations

import hashlib
import io

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.printer import Printer


class UnsupportedGraphRegion(ValueError):
    """The source requires an execution/effect domain not modeled by this leaf."""


def ir_text(operation: Operation) -> str:
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(operation)
    return stream.getvalue() + "\n"


def parse_source(text: str):
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, func.Func, linalg.Linalg, tensor.Tensor):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return module


def _integer_tensor(value):
    typ = value.type
    if (
        not isinstance(typ, builtin.TensorType)
        or not isinstance(typ.element_type, builtin.IntegerType)
        or typ.element_type.width.data <= 0
        or typ.element_type.width.data % 8
        or any(d <= 0 for d in typ.get_shape())
        or not isinstance(typ.encoding, builtin.NoneAttr)
    ):
        raise UnsupportedGraphRegion("requires positive static byte-addressable integer tensors")
    return typ


def _logical_operands(operation):
    if operation.name != "linalg.generic" or len(operation.results) != 1 or len(operation.regions) != 1:
        raise UnsupportedGraphRegion("one-result pointwise linalg.generic required")
    if operation.attributes or set(operation.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise UnsupportedGraphRegion("unknown generic metadata or execution contract")
    blocks = tuple(operation.regions[0].blocks)
    if len(blocks) != 1:
        raise UnsupportedGraphRegion("pointwise source must have one scalar block")
    inputs, outputs = tuple(operation.inputs), tuple(operation.outputs)
    if len(outputs) != 1:
        raise UnsupportedGraphRegion("one unread destination initializer required")
    shape = _integer_tensor(operation.results[0]).get_shape()
    if any(_integer_tensor(v).get_shape() != shape for v in (*inputs, *outputs)):
        raise UnsupportedGraphRegion("pointwise input/destination shapes differ")
    maps = operation.properties.get("indexing_maps")
    iterators = operation.properties.get("iterator_types")
    if (
        maps is None
        or len(maps.data) != len(inputs) + 1
        or any(m.data != m.data.identity(len(shape)) for m in maps.data)
        or iterators is None
        or len(iterators.data) != len(shape)
        or any(i.data.value != "parallel" for i in iterators.data)
    ):
        raise UnsupportedGraphRegion("requires exact parallel identity-indexed tensor maps")
    block = blocks[0]
    if len(block.args) != len(inputs) + 1:
        raise UnsupportedGraphRegion("scalar block argument arity differs")
    allowed = {
        "arith.constant",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.maxsi",
        "arith.minsi",
        "arith.trunci",
        "arith.extsi",
        "arith.extui",
    }
    body = tuple(block.ops)
    if not body or body[-1].name != "linalg.yield" or len(body[-1].operands) != 1:
        raise UnsupportedGraphRegion("one explicit pointwise yield required")
    if body[-1].attributes or body[-1].properties or body[-1].regions or body[-1].successors:
        raise UnsupportedGraphRegion("unknown pointwise yield metadata/effects")
    available = set(block.args[:-1])
    for op in body[:-1]:
        if (
            op.name not in allowed
            or op.regions
            or op.successors
            or len(op.results) != 1
            or op.attributes
            or any(v not in available for v in op.operands)
            or any(not isinstance(v.type, builtin.IntegerType) for v in (*op.operands, *op.results))
        ):
            raise UnsupportedGraphRegion("unknown, captured, initializer-reading or effectful scalar body")
        for name, value in op.properties.items():
            if name == "value" and op.name == "arith.constant" and isinstance(value, builtin.IntegerAttr):
                continue
            # Flags capable of making initialized integer operations poison are
            # outside the initial total-function domain. Names are not proof facts.
            if name == "overflowFlags" and isinstance(value, arith.IntegerOverflowAttr) and not value.data:
                continue
            raise UnsupportedGraphRegion(f"unsupported scalar modifier: {op.name}.{name}")
        available.update(op.results)
    if body[-1].operands[0] not in available:
        raise UnsupportedGraphRegion("pointwise yield reads initializer or external state")
    return inputs, outputs, block.args[-1]


def project(module) -> tuple[dict, dict, dict]:
    """Read actual definitions, all use positions, observers and unread initializers."""
    module.verify()
    top = tuple(module.body.block.ops)
    if len(top) != 1 or not isinstance(top[0], func.FuncOp) or len(top[0].body.blocks) != 1:
        raise UnsupportedGraphRegion("initial domain is one defined single-block function")
    function = top[0]
    body = tuple(function.body.block.ops)
    if not body or body[-1].name != "func.return":
        raise UnsupportedGraphRegion("explicit function return required")
    if any(op.name not in {"tensor.empty", "linalg.generic", "func.return"} for op in body):
        raise UnsupportedGraphRegion("unknown surrounding operation/effects require another domain")
    if any(op.name == "func.return" for op in body[:-1]):
        raise UnsupportedGraphRegion("early return is unsupported")
    members = tuple(module.walk())
    ordinals = {op: i for i, op in enumerate(members)}
    values, value_ids = [], {}

    def register(value, origin):
        if value in value_ids:
            raise UnsupportedGraphRegion("SSA definition has multiple source identities")
        ident = len(values)
        value_ids[value] = ident
        values.append({"id": ident, "type": str(value.type), "origin": origin})

    for op in members:
        for index, value in enumerate(op.results):
            register(value, {"operation": ordinals[op], "result": index})
        for r, region in enumerate(op.regions):
            for b, block in enumerate(region.blocks):
                for index, value in enumerate(block.args):
                    register(value, {"block_owner": ordinals[op], "region": r, "block": b, "argument": index})
    nodes, initializers = [], []
    for op in body[:-1]:
        if op.name == "tensor.empty":
            if op.operands or len(op.results) != 1:
                raise UnsupportedGraphRegion("dynamic allocation/shape effects are unsupported")
            _integer_tensor(op.results[0])
            initializers.append({"ordinal": ordinals[op], "results": [value_ids[v] for v in op.results]})
            continue
        inputs, outputs, unused = _logical_operands(op)
        nodes.append(
            {
                "ordinal": ordinals[op],
                "semanticKey": hashlib.sha256(ir_text(op).encode()).hexdigest(),
                "operands": [value_ids[v] for v in inputs],
                "results": [value_ids[v] for v in op.results],
                "operandTypes": [str(v.type) for v in inputs],
                "resultTypes": [str(v.type) for v in op.results],
                "rawOperands": [value_ids[v] for v in op.operands],
                "unreadInitializers": [value_ids[v] for v in outputs],
                "unusedScalarArgument": value_ids[unused],
            }
        )
    uses = [
        {"value": ident, "operation": ordinals[use.operation], "operand": use.index}
        for value, ident in value_ids.items()
        for use in value.uses
    ]
    uses.sort(key=lambda u: (u["value"], u["operation"], u["operand"]))
    result = {
        "schema": "merlin.graph-regions.actual-ssa.v1",
        "function": function.sym_name.data,
        "functionArguments": [value_ids[v] for v in function.body.block.args],
        "values": values,
        "nodes": nodes,
        "initializers": initializers,
        "allUses": uses,
        "observations": [value_ids[v] for v in body[-1].operands],
        "returnOrdinal": ordinals[body[-1]],
        "semanticDomain": "typed total pure logical integer tensor stages, conditional on action premises",
        "projectionTrustedPython": True,
        "nativePrimitiveCorrectnessProved": False,
    }
    return result, ordinals, value_ids


def planner_record(region, ordinals, value_ids, projection) -> dict:
    """Serialize the actual planner proposal; the Lean checker must rederive its facts."""
    nodes = {n["ordinal"]: n for n in projection["nodes"]}
    return {
        "values": [value_ids[v] for v in region.values],
        "stages": [
            {
                "ordinal": s.ordinal,
                "semanticKey": nodes[s.ordinal]["semanticKey"],
                "inputs": list(s.inputs),
                "outputs": [s.output],
                "operandTypes": nodes[s.ordinal]["operandTypes"],
                "resultTypes": nodes[s.ordinal]["resultTypes"],
            }
            for s in region.stages
        ],
        "inputs": list(region.inputs),
        "outputs": list(region.outputs),
        "lifetimes": [list(life) for life in region.lifetimes],
    }
