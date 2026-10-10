"""Bounded original scalar calls joined to actual registered standard IR.

Provenance and registry observations bind products to a call. The separate SSA,
coefficient, map and body checks establish the selected construction relation.
Neither relation licenses numerical policy, effects or compiled semantics.
"""

from __future__ import annotations

import hashlib
import math
import struct

from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_json

from . import original_scalar_binary_sources as S
from .frontend_trace import _graph, join_frontend_trace

SCHEMA = "merlin.original_scalar_binary_correspondence.v1"
INTEGER_SCHEMA = "merlin.original_scalar_binary_correspondence.v2"
REGISTRY_SCHEMA = "merlin.native_scalar_binary_registry.v1"
INTEGER_REGISTRY_SCHEMA = "merlin.native_scalar_binary_registry.v2"
REGISTRY_FUNCTIONS = {
    "aten.mul.Tensor": "decompose_mul_tensor",
    "aten.div.Tensor": "decompose_div_tensor",
}
_LIMITS = {"max_source_bytes", "max_nesting", "max_operations", "max_tensor_elements"}
_SCOPE = "exact registered scalar source construction only; no numerical, effect, compiled or hardware admission"


def coefficient_bits(literal, *, version=1):
    """Represent the original finite Python binary64 FloatLiteral in f32.

    This checks typed constant representation, not the operator's arithmetic or
    a framework promotion policy. Unrepresentable finite coefficients refuse.
    """
    from .frontend_original_call import default_value

    value = default_value(literal)
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("scalar coefficient needs its explicit representation version")
    if version == 2 and type(value) is int:
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError("integer scalar coefficient is outside signed64")
        # IEEE f32 RNE from exact integer bits, without an intermediate f64.
        magnitude = abs(value)
        if not magnitude:
            return "00000000"
        exponent = magnitude.bit_length() - 1
        shift = max(0, exponent - 23)
        significant, remainder = divmod(magnitude, 1 << shift)
        if shift and (remainder > (1 << (shift - 1)) or remainder == (1 << (shift - 1)) and significant % 2):
            significant += 1
        if significant == 1 << 24:
            significant >>= 1
            exponent += 1
        significant <<= max(0, 23 - exponent)
        bits = (int(value < 0) << 31) | ((exponent + 127) << 23) | (significant - (1 << 23))
        return bits.to_bytes(4, "big").hex()
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("registered scalar correspondence needs its finite FloatLiteral")
    try:
        bits = struct.pack(">f", value)
    except OverflowError as error:
        raise ValueError("original scalar is outside finite f32 constant representation") from error
    if not math.isfinite(struct.unpack(">f", bits)[0]):
        raise ValueError("original scalar is outside finite f32 constant representation")
    return bits.hex()


def validate_limits(limits):
    if (
        type(limits) is not dict
        or set(limits) != _LIMITS
        or any(type(value) is not int or value < 1 for value in limits.values())
    ):
        raise ValueError("scalar correspondence requires complete positive parser and logical budgets")
    return limits


def _parse(text, limits):
    from xdsl.dialects import arith, linalg, tensor
    from xdsl.parser import Parser

    from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source
    from merlin.xdsl_dialects._common import make_context

    admit_mlir_source(
        text,
        max_source_bytes=limits["max_source_bytes"],
        max_nesting=limits["max_nesting"],
        max_integer_bits=64,
        allow_dense=False,
        allow_dense_resource=False,
    )
    from xdsl.utils.exceptions import ParseError, VerifyException

    try:
        module = Parser(make_context(arith.Arith, linalg.Linalg, tensor.Tensor), text).parse_module()
        operations = []
        for operation in module.walk():
            if len(operations) == limits["max_operations"]:
                raise ValueError("scalar correspondence exceeds its complete operation budget")
            operations.append(operation)
        module.verify()
    except (ParseError, VerifyException) as error:
        raise ValueError("scalar conversion product lacks supported complete typed IR") from error
    return module, operations


def _trace_call(snapshot, stage, form, shape):
    errors = []
    graph = _graph(snapshot, stage, errors)
    if errors or graph["status"] != "verified" or len(graph["calls"]) != 1:
        raise ValueError("scalar conversion needs one complete original call in each frontend stage")
    call = graph["nodes"][next(iter(graph["calls"]))]
    inputs = [row for row in graph["nodes"].values() if row["op"] == "placeholder"]
    outputs = [row for row in graph["nodes"].values() if row["op"] == "output"]
    scalar = S.default_value(form["parameters"]["other"])
    if (
        len(graph["nodes"]) != 3
        or len(inputs) != 1
        or len(outputs) != 1
        or call["op"] != "call_function"
        or call["target"] != form["target"]
        or snapshot.get("operator_schemas", {}).get(form["target"]) != form["schema"]
        or call.get("kwargs") != {}
        or type(call.get("args")) is not list
        or len(call["args"]) != 2
        or type(call["args"][1]) is not type(scalar)
        or canonical_json(call["args"][1]) != canonical_json(scalar)
    ):
        raise ValueError("actual frontend call changed original target, schema, scalar kind or argument order")
    for row in (inputs[0], call):
        results = row.get("results")
        if type(results) is not list or len(results) != 1:
            raise ValueError("actual frontend omits an original input/result slot")
        value = results[0]
        if (
            value.get("kind") != "tensor"
            or value.get("dtype") != "float32"
            or value.get("storage_dtype") != "float32"
            or canonical_json(value.get("shape")) != canonical_json(shape)
            or value.get("layout") != "torch.strided"
            or value.get("device") != "cpu"
        ):
            raise ValueError("actual frontend promoted or changed original f32 storage/geometry")
    expected = {"node_id": inputs[0]["id"], "value_id": inputs[0]["results"][0]["id"]}
    if canonical_json(call["args"][0]) != canonical_json(expected):
        raise ValueError("actual scalar call lost its first and only original Tensor SSA")
    # Export represents a function's complete tensor readout as a one-element tuple.
    readout = outputs[0].get("args")
    expected = {"node_id": call["id"], "value_id": call["results"][0]["id"]}
    if readout not in ([expected], [[expected]]):
        raise ValueError("actual frontend omitted or substituted its complete original readout")
    return call["id"]


def _registry(registry, *, form, shape, tensor_type, prepared, origins, source_inventory):
    version = 2 if form["form_schema"] == S.INTEGER_FORM_SCHEMA else 1
    fields = {"schema", "target", "function", "importer", "events"}
    if version == 2:
        fields.add("tensor_argument")
    if (
        type(registry) is not dict
        or set(registry) != fields
        or registry["schema"] != (INTEGER_REGISTRY_SCHEMA if version == 2 else REGISTRY_SCHEMA)
        or registry["target"] != form["target"]
    ):
        raise ValueError("scalar correspondence lacks its actual registered conversion observation")
    wanted = (
        ("function", "m2m.ir.decompositions", REGISTRY_FUNCTIONS[form["target"]]),
        ("importer", "m2m.ir.import_fx", "FXImporter.import_graph"),
    )
    for role, module, name in wanted:
        actual = registry[role]
        if (
            type(actual) is not dict
            or set(actual) != {"module", "name", "path", "sha256"}
            or actual["module"] != module
            or actual["name"] != name
            or not is_sha256(actual["sha256"])
            or source_inventory.get(actual["path"]) != actual["sha256"]
        ):
            raise ValueError("registered callable/importer differs from the independently selected public source")
    expected = {
        "target": form["target"],
        "literal": form["parameters"]["other"],
        "operand_types": [tensor_type],
        "result_type": tensor_type,
        "source_node_id": prepared,
        "origin_node_ids": origins,
        "emitted_operations": ["arith.constant"]
        + (["arith.sitofp"] if form["parameters"]["other"]["kind"] == "int" else [])
        + ["tensor.splat", "tensor.empty", "linalg.generic"],
        "dynamic_overrides": [],
    }
    if version == 2:
        binding = form.get("tensor_binding")
        if binding is None:
            if registry["tensor_argument"] is not None:
                raise ValueError("floating source cannot inherit an integer wrapped-number observation")
        else:
            descriptor = {
                "kind": "tensor",
                "dtype": "torch.float32",
                "shape": shape,
                "layout": "torch.strided",
                "device": "cpu",
            }
            promotion = {
                "original_binding": binding,
                "native_binding": binding["native"],
                "common_dtype": "torch.float32",
                "input": descriptor,
                "outputs": [descriptor],
                "boxed_outputs": [descriptor],
            }
            if canonical_json(registry["tensor_argument"]) != canonical_json(promotion):
                raise ValueError(
                    "integer scalar lacks exact native boxing, promotion and complete readout observations"
                )
    if canonical_json(registry["events"]) != canonical_json([expected]):
        raise ValueError(
            "actual registered invocation changed original literal, operands, result or registry selection"
        )


def verify(form, source, *, extent, text, trace, registry, source_inventory, limits):
    """Check actual products, never source IDs or an ABI alone.

    The native observation owner separately reopens selected source bytes and
    actual invocation pins. This reader cannot issue such execution authority.
    """
    limits = validate_limits(limits)
    expected = S.scalar_binary_source(form, extent=extent, max_tensor_elements=limits["max_tensor_elements"])
    if source.loader != expected.loader or canonical_json(source.metadata()) != canonical_json(expected.metadata()):
        raise ValueError("scalar correspondence source differs from the complete original factory")
    version = 2 if form["form_schema"] == S.INTEGER_FORM_SCHEMA else 1
    integer = form["parameters"]["other"]["kind"] == "int"
    bits = coefficient_bits(form["parameters"]["other"], version=version)
    if len(canonical_json(trace)) > limits["max_source_bytes"]:
        raise ValueError("scalar frontend trace exceeds its complete byte budget")
    module, operations = _parse(text, limits)
    from xdsl.dialects import arith, linalg, tensor
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, NoneAttr, TensorType, f32, i64
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import IteratorType
    from xdsl.ir.affine import AffineMap

    shape = expected.metadata()["inputs"][0]["shape"]
    value_type = TensorType(f32, shape)
    functions = tuple(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not FuncOp or len(functions[0].body.blocks) != 1:
        raise ValueError("scalar conversion requires one defined original function")
    entry = functions[0]
    block = entry.body.block
    body = tuple(block.ops)
    if (
        len(block.args) != 1
        or block.args[0].type != value_type
        or type(block.args[0].type.encoding) is not NoneAttr
        or tuple(entry.function_type.inputs) != (value_type,)
        or tuple(entry.function_type.outputs) != (value_type,)
        or tuple(type(op) for op in body)
        != (arith.ConstantOp,)
        + ((arith.SIToFPOp,) if integer else ())
        + (tensor.SplatOp, tensor.EmptyOp, linalg.GenericOp, ReturnOp)
        or len(operations) != (10 if integer else 9)
    ):
        raise ValueError("scalar conversion changed its complete original ABI or supported operation roster")
    constant = body[0]
    converted = body[1] if integer else None
    splat, empty, generic, returned = body[-4:]
    value = constant.value
    if integer:
        if (
            type(value) is not IntegerAttr
            or value.type != i64
            or value.value.data != S.default_value(form["parameters"]["other"])
            or constant.result.type != i64
            or tuple(converted.operands) != (constant.result,)
            or converted.result.type != f32
        ):
            raise ValueError("integer scalar conversion changed original signed64 constant or direct f32 cast")
    elif (
        type(value) is not FloatAttr
        or value.type != f32
        or struct.pack(">f", value.value.data).hex() != bits
        or constant.result.type != f32
    ):
        raise ValueError("scalar conversion changed original f32 coefficient bits")
    if (
        tuple(splat.operands) != ((converted.result if integer else constant.result),)
        or splat.result.type != value_type
        or tuple(empty.operands)
        or empty.results[0].type != value_type
        or tuple(generic.inputs) != (block.args[0], splat.result)
        or tuple(generic.outputs) != (empty.results[0],)
        or generic.results[0].type != value_type
        or tuple(returned.arguments) != (generic.results[0],)
    ):
        raise ValueError("scalar conversion changed coefficient bits or ordered input/splat/output SSA bindings")
    if (
        tuple(item.data for item in generic.indexing_maps) != (AffineMap.identity(len(shape)),) * 3
        or tuple(item.data for item in generic.iterator_types) != (IteratorType.PARALLEL,) * len(shape)
        or generic.doc is not None
        or generic.library_call is not None
        or len(generic.body.blocks) != 1
    ):
        raise ValueError("scalar conversion changed its exact original identity maps or parallel body")
    scalar_body = generic.body.block
    scalar_ops = tuple(scalar_body.ops)
    calculation, yielded = scalar_ops if len(scalar_ops) == 2 else (None, None)
    arithmetic = arith.MulfOp if form["target"] == "aten.mul.Tensor" else arith.DivfOp
    if (
        tuple(arg.type for arg in scalar_body.args) != (f32, f32, f32)
        or type(calculation) is not arithmetic
        or type(yielded) is not linalg.YieldOp
        or tuple(calculation.operands) != tuple(scalar_body.args[:2])
        or tuple(yielded.operands) != (calculation.result,)
        or calculation.result.type != f32
        or calculation.fastmath.data
    ):
        raise ValueError(
            "scalar conversion changed original arithmetic, operand order, flags or complete yielded value"
        )
    allowed_properties = (
        (constant, {"value"}),
        (splat, set()),
        (empty, set()),
        (generic, {"indexing_maps", "iterator_types", "operandSegmentSizes"}),
        (calculation, {"fastmath"}),
        (yielded, set()),
        (returned, set()),
    )
    if integer:
        allowed_properties += ((converted, set()),)
    for op, allowed in allowed_properties:
        if any(not key.startswith("prov.") for key in op.attributes) or set(op.properties) - allowed:
            raise ValueError("scalar conversion carries unsupported semantic attributes or properties")
    snapshots = trace.get("graphs", {})
    calls = {
        stage: _trace_call(snapshots.get(stage), stage, form, shape) for stage in ("original", "quantized", "prepared")
    }
    from .application_graph import _program_graph

    digest = hashlib.sha256(text.encode()).hexdigest()
    raw = _program_graph(module, digest)
    joined = join_frontend_trace(
        trace,
        {
            "schema": "merlin.application_graph.v1",
            "capture_sha256": digest,
            "capture_bytes": len(text.encode()),
            "capture_graph": raw,
            "normalization_correspondence": {"status": "identity"},
        },
        capture_sha256=digest,
    )
    if joined["status"] != "complete":
        raise ValueError(
            "scalar conversion lost complete actual source/type/trace correspondence: " + repr(joined["errors"])
        )
    origins = [calls["original"], calls["quantized"]]
    # The selected tensor.empty custom printer omits decoration. Its exact
    # typed output allocation and SSA use are checked above; absent IDs confer
    # no separate effect or ownership correspondence.
    for op in (constant, splat, generic, calculation, yielded) + ((converted,) if integer else ()):
        if [item.data for item in op.attributes.get("prov.source_node_ids", ())] != [calls["prepared"]] or [
            item.data for item in op.attributes.get("prov.origin_node_ids", ())
        ] != origins:
            raise ValueError("scalar conversion computation lost its exact original/prepared trace bindings")
    _registry(
        registry,
        form=form,
        shape=shape,
        tensor_type=str(value_type),
        prepared=calls["prepared"],
        origins=origins,
        source_inventory=source_inventory,
    )
    return {
        "schema": INTEGER_SCHEMA if version == 2 else SCHEMA,
        "target": form["target"],
        "original_node": form["node"],
        "frontend_nodes": calls,
        "coefficient_f32_bits": bits,
        "ordered_abi": {"inputs": [str(value_type)], "outputs": [str(value_type)]},
        "source_sha256": hashlib.sha256(source.loader.encode()).hexdigest(),
        "mlir_sha256": digest,
        "trace_sha256": hashlib.sha256(canonical_json(trace)).hexdigest(),
        "scope": _SCOPE,
    }
