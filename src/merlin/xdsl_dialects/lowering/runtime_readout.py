"""Source-bound, tensor-wide runtime f32 readouts, proved without sampling a scale.

The only symbolic value licensed here is a direct f32[1] function argument.
Every arithmetic operation and broadcast is checked against the source SSA.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from xdsl.dialects import builtin, func
from xdsl.ir import BlockArgument
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

from merlin.common import mlir_query as mq
from merlin.kernels import shapes as KS

from . import output_fusion as OF
from .group_numerics import GroupNumericsError as GNError


def scale_operand_record(value) -> dict:
    if not isinstance(value, BlockArgument) or not isinstance(value.type, builtin.TensorType):
        raise ValueError("runtime scale must be a direct tensor function argument")
    block = value.owner
    function = block.parent_op()
    if not isinstance(function, func.FuncOp) or block is not function.body.blocks.first:
        raise ValueError("runtime scale must belong to the original function entry block")
    if value.type.get_shape() != (1,) or value.type.element_type != builtin.f32:
        raise ValueError("runtime scale storage must be tensor<1xf32>")
    return {
        "schema": "merlin.readout_scale_operand.v1",
        "kind": "runtime_input",
        "dtype": "f32",
        "storage_shape": [1],
        "source": {"func_arg": value.index, "element_index": 0},
        "granularity": "tensor",
        "domain": "finite_f32",
    }


def _type(value, shape, dtype):
    return (
        isinstance(value.type, builtin.TensorType)
        and value.type.get_shape() == tuple(shape)
        and str(value.type.element_type) == dtype
    )


def _plain(op):
    # Even apparently harmless fast-math flags change the theorem's domain.
    for table in (op.attributes, op.properties):
        for key, value in table.items():
            if "fastmath" in key.lower() or "fast_math" in key.lower():
                if getattr(value, "data", None) != frozenset():
                    return False
    return True


def _identity(expressions, rank):
    return len(expressions) == rank and all(
        isinstance(e, AffineDimExpr) and e.position == i for i, e in enumerate(expressions)
    )


def _generic(op, data, *, inputs, output_dtype, scale_map=None):
    if mq.op_name(op) != "linalg.generic" or len(op.operands) != inputs + 1 or len(op.results) != 1:
        return None
    shape, _ = mq.type_shape_dtype(data.type)
    if len(shape) != 2 or any(int(d) <= 0 for d in shape) or op.operands[0] is not data:
        return None
    if not _type(op.results[0], shape, output_dtype) or op.operands[-1].type != op.results[0].type:
        return None
    raw_maps = op.properties.get("indexing_maps", op.attributes.get("indexing_maps"))
    if raw_maps is None or any(m.data.num_dims != 2 or m.data.num_symbols != 0 for m in raw_maps):
        return None
    maps = KS.indexing_maps(op)
    if not maps or len(maps) != inputs + 1 or not _identity(maps[0], 2) or not _identity(maps[-1], 2):
        return None
    if inputs == 2 and not scale_map(maps[1]):
        return None
    if KS._iterator_types(op) != ["parallel", "parallel"] or not _plain(op) or len(op.regions) != 1:
        return None
    blocks = tuple(op.regions[0].blocks)
    if len(blocks) != 1 or len(blocks[0].args) != inputs + 1 or blocks[0].args[-1].uses:
        return None
    block = blocks[0]
    expected = [v.type.element_type for v in (*op.operands,)]
    if [v.type for v in block.args] != expected or any(not _plain(child) for child in block.ops):
        return None
    return block


def _zero_map(expressions):
    return len(expressions) == 1 and isinstance(expressions[0], AffineConstantExpr) and expressions[0].value == 0


def _constant(value, number):
    owner = value.owner
    if not hasattr(owner, "name") or mq.op_name(owner) != "arith.constant" or str(value.type) != "f32":
        return False
    attr = owner.properties.get("value", owner.attributes.get("value"))
    raw = getattr(getattr(attr, "value", None), "data", None)
    return type(raw) in (int, float) and raw == number and (number != 0 or math.copysign(1, raw) > 0)


def _binary(op, name, value, other=None, constant=None):
    if mq.op_name(op) != name or len(op.operands) != 2 or len(op.results) != 1 or str(op.results[0].type) != "f32":
        return False
    a, b = op.operands
    return (a is value and (b is other if constant is None else _constant(b, constant))) or (
        b is value and (a is other if constant is None else _constant(a, constant))
    )


def _closed_body(block):
    operations = [op for op in block.ops if mq.op_name(op) != "arith.constant"]
    names = [mq.op_name(op) for op in operations]
    relu = len(operations) == 8
    expected = (
        ["arith.sitofp"]
        + (["arith.maximumf"] if relu else [])
        + ["arith.mulf", "math.roundeven", "arith.maximumf", "arith.minimumf", "arith.fptosi", "linalg.yield"]
    )
    if names != expected:
        return None
    convert = operations.pop(0)
    if tuple(convert.operands) != (block.args[0],) or str(convert.results[0].type) != "f32":
        return None
    value = convert.results[0]
    if relu:
        activation = operations.pop(0)
        if not _binary(activation, "arith.maximumf", value, constant=0):
            return None
        value = activation.results[0]
    product, rounding, low, high, narrow, yielding = operations
    if not _binary(product, "arith.mulf", value, block.args[1]):
        return None
    if tuple(rounding.operands) != (product.results[0],) or str(rounding.results[0].type) != "f32":
        return None
    if not _binary(low, "arith.maximumf", rounding.results[0], constant=-128) or not _binary(
        high, "arith.minimumf", low.results[0], constant=127
    ):
        return None
    if (
        tuple(narrow.operands) != (high.results[0],)
        or str(narrow.results[0].type) != "i8"
        or tuple(yielding.operands) != (narrow.results[0],)
    ):
        return None
    # No unused constants or hidden operations/effects are allowed in a closed body.
    constants = [op for op in block.ops if mq.op_name(op) == "arith.constant"]
    if len(constants) != 2 + int(relu) or any(len(op.results) != 1 or not op.results[0].uses for op in constants):
        return None
    return relu


def _closed(root, *, constant=False):
    classified = OF.CG.classify(root)
    if classified is None or classified.kind != OF.CG.CONTRACTION or len(root.results) != 1:
        return None
    shape, dtype = mq.type_shape_dtype(root.results[0].type)
    if dtype != "i32" or len(shape) != 2:
        return None
    uses = list(root.results[0].uses)
    if len(uses) != 1:
        return None
    op = uses[0].operation
    if len(op.operands) != 3:
        return None
    scale = op.operands[1]
    if constant:
        if isinstance(scale, BlockArgument):
            return None
        scale_shape, scale_dtype = mq.type_shape_dtype(scale.type)
        if scale_dtype != "f32" or list(scale_shape) != [shape[1]]:
            return None

        def scale_map(exprs):
            return len(exprs) == 1 and isinstance(exprs[0], AffineDimExpr) and exprs[0].position == 1
    else:
        try:
            scale_operand_record(scale)
        except ValueError:
            return None
        scale_map = _zero_map
    block = _generic(op, root.results[0], inputs=2, output_dtype="i8", scale_map=scale_map)
    relu = _closed_body(block) if block is not None else None
    if relu is None:
        return None
    return OF.Chain(
        root, int(shape[1]), [OF.Stage("runtime_readout", op, detail={"scale_value": scale, "relu": relu})], [op], "f32"
    )


def match_closed_readout(root):
    return _closed(root)


def match_constant_readout(root, values):
    """Expand the same closed scalar body into the existing constant proof vocabulary."""
    import numpy as np

    chain = _closed(root, constant=True)
    if chain is None:
        return None
    detail, op = chain.stages[0].detail, chain.ops[0]
    numbers = values.array(detail["scale_value"])
    if numbers is None:
        return None
    numbers = np.asarray(numbers).reshape(-1)
    if numbers.size != chain.channels:
        return None
    chain.stages = [
        OF.Stage("convert", op, detail={"closed_readout": True}),
        OF.Stage("scale", op, numbers, {"closed_readout": True}),
    ]
    if detail["relu"]:
        chain.stages.append(OF.Stage("relu", op, detail={"closed_readout": True}))
    chain.stages.append(OF.Stage("quantize", op, 1.0, {"zero_point": 0, "min": -128, "max": 127, "dtype": "i8"}))
    return chain


def split_scale_value(op, data):
    if len(op.operands) != 3:
        return None
    try:
        scale_operand_record(op.operands[1])
    except ValueError:
        return None
    block = _generic(op, data, inputs=2, output_dtype="f32", scale_map=_zero_map)
    if block is None:
        return None
    ops = list(block.ops)
    if len(ops) != 2 or not _binary(ops[0], "arith.mulf", block.args[0], block.args[1]):
        return None
    if mq.op_name(ops[1]) != "linalg.yield" or tuple(ops[1].operands) != (ops[0].results[0],):
        return None
    return op.operands[1]


def _split_unary(op, data, kind):
    integer_relu = kind == "relu" and str(data.type.element_type) == "i32"
    output_dtype = "i32" if integer_relu else "f32"
    block = _generic(op, data, inputs=1, output_dtype=output_dtype)
    if block is None:
        return False
    ops = [child for child in block.ops if mq.op_name(child) != "arith.constant"]
    if len(ops) != 2 or mq.op_name(ops[1]) != "linalg.yield" or tuple(ops[1].operands) != (ops[0].results[0],):
        return False
    if kind == "convert":
        return (
            len(list(block.ops)) == 2
            and mq.op_name(ops[0]) == "arith.sitofp"
            and tuple(ops[0].operands) == (block.args[0],)
            and str(block.args[0].type) == "i32"
            and str(ops[0].results[0].type) == "f32"
        )
    if integer_relu:
        rectifier = ops[0]
        if len(list(block.ops)) != 3 or mq.op_name(rectifier) != "arith.maxsi" or len(rectifier.operands) != 2:
            return False
        other = [v for v in rectifier.operands if v is not block.args[0]]
        if len(other) != 1:
            return False
        owner = other[0].owner
        if not hasattr(owner, "name") or mq.op_name(owner) != "arith.constant" or str(other[0].type) != "i32":
            return False
        attr = owner.properties.get("value", owner.attributes.get("value"))
        return getattr(getattr(attr, "value", None), "data", None) == 0
    return len(list(block.ops)) == 3 and _binary(ops[0], "arith.maximumf", block.args[0], constant=0)


def _runtime_source(chain):
    shape, dtype = mq.type_shape_dtype(chain.root.results[0].type)
    classified = OF.CG.classify(chain.root)
    if (
        classified is None
        or classified.kind != OF.CG.CONTRACTION
        or len(shape) != 2
        or dtype != "i32"
        or chain.channels != shape[1]
    ):
        raise ValueError("runtime source is not a rank-2 i32 contraction with the stated channel axis")
    if chain.unread or chain.float_dtype != "f32":
        raise ValueError("runtime readout is not a closed f32 source")
    if [s.kind for s in chain.stages] == ["runtime_readout"]:
        fresh = match_closed_readout(chain.root)
        if (
            fresh is None
            or fresh.channels != chain.channels
            or fresh.ops != chain.ops
            or fresh.stages[0].detail != chain.stages[0].detail
        ):
            raise ValueError("runtime composite differs from source")
        return fresh.stages[0].detail
    kinds = [s.kind for s in chain.stages]
    if kinds not in (
        ["convert", "scale", "quantize"],
        ["convert", "relu", "scale", "quantize"],
        ["relu", "convert", "scale", "quantize"],
    ):
        raise ValueError("runtime source requires convert, optional pre-product ReLU, scale and quantize")
    data = chain.root.results[0]
    for stage in chain.stages[:-1]:
        if len(list(data.uses)) != 1 or list(data.uses)[0].operation is not stage.op:
            raise ValueError("runtime intermediate has an observer")
        valid = (
            stage.detail.get("scale_value") is not None
            and split_scale_value(stage.op, data) is stage.detail["scale_value"]
            if stage.kind == "scale"
            else _split_unary(stage.op, data, stage.kind)
        )
        if not valid:
            raise ValueError("runtime split stage differs from exact primitive source")
        data = stage.op.results[0]
    quant = chain.stages[-1]
    from . import group_numerics as GN

    source = GN._scale_source(quant.op)
    quantizer_attributes = {**quant.op.attributes, **quant.op.properties}
    dtype = getattr(quantizer_attributes.get("output_dtype"), "data", None)
    unknown = {
        key
        for key in quantizer_attributes
        if key not in {"op_name__", "output_dtype", "quant_min", "quant_max"} and not key.startswith("prov.")
    }
    if (
        mq.op_name(quant.op) != "quant_ext.quantize_per_tensor"
        or len(quant.op.operands) != 3
        or unknown
        or dtype != "int8"
        or not _type(quant.op.operands[1], [], "f32")
        or not _type(quant.op.operands[2], [], "i64")
        or quant.op.operands[0] is not data
        or len(list(data.uses)) != 1
        or source.value != 1.0
        or source.zero_point != 0
        or GN._int_attr(quant.op, "quant_min") != -128
        or GN._int_attr(quant.op, "quant_max") != 127
        or not _type(quant.op.results[0], mq.type_shape_dtype(data.type)[0], "i8")
        or not _plain(quant.op)
    ):
        raise ValueError("runtime quantizer must be unit-scale zero-point-zero signed i8 RNE saturation")
    if chain.ops != [s.op for s in chain.stages]:
        raise ValueError("runtime split consumed members differ from source")
    return {
        "scale_value": next(s.detail["scale_value"] for s in chain.stages if s.kind == "scale"),
        "relu": "relu" in kinds,
    }


def _transport(value):
    """Validate the derived transport record; its slot position is the facts', not assumed here.

    derive_runtime_readout_scale places the slot from the register layout and
    checks it lies inside its bundle; this only requires one whole f32 field.
    """
    from merlin.common.quant_formats import storage_bits

    keys = {"schema", "dtype", "granularity", "operand", "bundle", "field", "offset", "width", "encoding"}
    return (
        isinstance(value, Mapping)
        and set(value) == keys
        and value["schema"] == "merlin.runtime_readout_scale.v1"
        and value["dtype"] == "f32"
        and value["granularity"] == "per_command"
        and value["operand"] == "rs2"
        and isinstance(value["bundle"], str)
        and bool(value["bundle"])
        and isinstance(value["field"], str)
        and bool(value["field"])
        and type(value["offset"]) is int
        and value["offset"] >= 0
        and type(value["width"]) is int
        and value["width"] == storage_bits(value["dtype"])
        and value["encoding"] == "ieee754_binary32_bits"
    )


def prove_runtime_readout(chain, path: Mapping, *, readout_strategy: str, runtime_scale_facts: Mapping):
    """Prove primitive equality for every finite f32 scale; bounds are the caller's obligation."""
    from merlin.verify.lean_contract import document_digest

    try:
        detail = _runtime_source(chain)
        if readout_strategy != "f32_product_rne" or not _transport(runtime_scale_facts):
            raise ValueError("runtime readout requires selected f32 strategy and derived runtime transport")
        if path.get("accumulator", {}).get("dtype") != "i32" or not path.get("exits", {}).get("requant"):
            raise ValueError("runtime readout requires an i32 accumulator and narrow exit")
        readout = OF._readout(path, "requant")
        OF._single_product_readout({"readout": readout})
        OF._require_models(readout, chain)
        activation = next((s for s in readout if s["kind"] == "activation"), None)
        if detail["relu"] and (
            activation is None or activation.get("domain") != "accumulator" or readout[0] is not activation
        ):
            raise ValueError("signed runtime scale requires accumulator ReLU before the product")
        convert = next(s for s in readout if s["kind"] == "convert")
        scale = next(s for s in readout if s["kind"] == "scale")
        requant = next(s for s in readout if s["kind"] == "requant")
        if (
            convert.get("to") != "f32"
            or scale.get("dtype") != "f32"
            or scale.get("granularity") not in ("tensor", "per_command")
            or requant.get("to") != "i8"
            or requant.get("clamp") != [-128, 127]
        ):
            raise ValueError("runtime device path differs from f32 product RNE i8 saturation")
        operand = scale_operand_record(detail["scale_value"])
    except (ValueError, GNError) as error:
        return OF.Decision("bare", [], [], [{"cut": "requant", "code": OF.UNREAD, "reason": str(error)}])
    except OF._Refused as error:
        return OF.Decision("bare", [], [], [{"cut": "requant", "code": error.code, "reason": error.reason}])
    stages = ["convert"] + (["relu"] if detail["relu"] else []) + ["scale", "quantize"]
    proof = {
        "definitional": "f32_product_rne",
        "scale_operand": operand,
        "offsets": [0] * int(chain.channels),
        "channels": int(chain.channels),
        "readout": "requant",
        "accumulator": dict(path["accumulator"]),
        "preload": next((dict(s) for s in path.get("accumulator_stages", ()) if s["kind"] == "preload"), None),
        "readout_stages": readout,
        "runtime_scale_facts_sha256": document_digest(runtime_scale_facts),
    }
    return OF.Decision("requant", list(chain.ops), stages, [], proof)
