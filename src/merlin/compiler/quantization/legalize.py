"""Explicitly legalize a closed captured quantization form using shared passes.

This owner supplies orchestration and source correspondence, not another
quantization algorithm or permission to change a preparation/quality policy.
Other unregistered forms retain their existing refusal at public source intake.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _owners():
    from merlin.compiler.frontend.preprocess import implementation_paths
    from merlin.compiler.quantization.linalg import lower_quant_ext

    paths = tuple(dict.fromkeys((
        Path(__file__), *implementation_paths(), Path(lower_quant_ext.__code__.co_filename),
    )))
    return {str(path.absolute()): _sha(path.read_bytes()) for path in paths}


def revalidate(record):
    """Reopen fixed actual owners; record paths cannot authorize arbitrary reads."""
    if (not isinstance(record, dict) or record.get("schema") != "merlin.captured-quantization-legalization.v1"
            or record.get("source_owners") != _owners()):
        raise ValueError("captured quantization legalization implementation changed after selection")


def _admit(operation):
    from xdsl.dialects import builtin

    if (getattr(getattr(operation, "op_name", None), "data", None) != "quant_ext.quantize_per_tensor"
            or operation.regions or operation.successors or len(operation.operands) != 3
            or len(operation.results) != 1
            or set(operation.properties) != {"quant_min", "quant_max", "output_dtype"}
            or any(key != "op_name__" and not key.startswith("prov.") for key in operation.attributes)):
        raise ValueError("captured quantization legalization requires the exact closed per-tensor marker")
    value, scale, zero = (operand.type for operand in operation.operands)
    result = operation.results[0].type
    if (any(type(typ) is not builtin.TensorType or not typ.has_static_shape()
            or type(typ.encoding) is not builtin.NoneAttr for typ in (value, scale, zero, result))
            or value.get_shape() != result.get_shape() or scale.get_shape() != () or zero.get_shape() != ()
            or value.element_type not in (builtin.f16, builtin.bf16, builtin.f32)
            or scale.element_type != builtin.f32 or zero.element_type not in (builtin.i32, builtin.i64)
            or result.element_type != builtin.i8
            or operation.properties["output_dtype"] != builtin.StringAttr("int8")):
        raise ValueError("captured quantization marker has unsupported typed arithmetic/storage")
    bounds = [operation.properties[name] for name in ("quant_min", "quant_max")]
    if (any(type(bound) is not builtin.IntegerAttr or type(bound.value.data) is not int for bound in bounds)
            or not -128 <= bounds[0].value.data < bounds[1].value.data <= 127):
        raise ValueError("captured quantization marker has invalid signed-i8 clamp bounds")


def _source_arithmetic(operation, trace):
    """Admit only source qparams whose precision survived externalization.

    The pinned primitive takes a Python scalar reciprocal before f32 multiply;
    the importer materializes scalar qparams as f32. A nonexact-f32 scalar can
    cross a rounding boundary. Unknown/runtime qparam domains need a separate
    admitted guard/producer contract and are conservatively refused here.
    """
    from xdsl.dialects import arith, builtin, tensor

    source_ids = operation.attributes.get("prov.source_node_ids")
    if (not isinstance(trace, dict) or trace.get("status") != "complete"
            or type(source_ids) is not builtin.ArrayAttr or len(source_ids) != 1
            or any(type(value) is not builtin.StringAttr for value in source_ids)):
        raise ValueError("captured quantization legalization needs exact prepared source correspondence")
    nodes = (trace.get("graphs", {}).get("prepared") or {}).get("nodes", ())
    selected = [node for node in nodes if node.get("id") == source_ids.data[0].data]
    if len(selected) != 1 or selected[0].get("target") != "quantized_decomposed.quantize_per_tensor.default":
        raise ValueError("captured quantization has an unproved source primitive/qparam producer")
    node = selected[0]
    args = node.get("args")
    if (not isinstance(args, list) or len(args) != 6 or node.get("kwargs") != {}
            or type(args[1]) not in (float, int) or type(args[2]) is not int
            or any(type(args[index]) is not int for index in (3, 4))
            or args[5] != {"kind": "dtype", "value": "torch.int8"}):
        raise ValueError("captured quantization has unknown or runtime qparam arithmetic/domain")
    scale = args[1]
    try:
        exact = struct.unpack("<f", struct.pack("<f", scale))[0]
    except (OverflowError, struct.error):
        exact = None
    if not math.isfinite(scale) or scale <= 0 or exact != scale:
        raise ValueError("captured scalar scale precision/domain did not survive f32 externalization")
    if not args[3] <= args[2] <= args[4] or any(
            operation.properties[name].value.data != args[index]
            for name, index in (("quant_min", 3), ("quant_max", 4))):
        raise ValueError("captured quantization zero-point/clamp order differs from its source")
    values = []
    for operand in operation.operands[1:]:
        splat = operand.owner
        if type(splat) is not tensor.SplatOp or type(splat.input.owner) is not arith.ConstantOp:
            raise ValueError("captured scalar qparams lost their actual literal producers")
        values.append(splat.input.owner.value.value.data)
    if values != [scale, args[2]]:
        raise ValueError("captured scalar qparams differ from their prepared source values")
    return {"source_node_id": node["id"], "primitive": node["target"], "scale": scale,
            "scale_precision": "exact_f32_python_scalar", "zero_point": args[2],
            "quant_min": args[3], "quant_max": args[4],
            "order": "f32 reciprocal; f32 multiply; nearest-even; add zero point; clamp; signed-i8"}


def legalize(module, *, trace=None):
    """Return a fresh legalized module and optional exact before/after receipt.

    All source markers and scalar precision are validated before rewriting.
    This intake admits traced exact scalar qparams; runtime qparam producers
    need their own domain/guard admission. Other existing lowering routes are
    unchanged. The caller still admits numerical meaning.
    """
    from xdsl.dialects.builtin import UnregisteredOp

    from merlin.compiler.quantization.linalg import lower_quant_ext
    from merlin.xdsl_dialects._common import text

    markers = tuple(op for op in module.walk() if isinstance(op, UnregisteredOp))
    if not markers:
        return module, None
    arithmetic = []
    for operation in markers:
        _admit(operation)
        arithmetic.append(_source_arithmetic(operation, trace))
    owners = _owners()
    original = text(module, generic=True)
    normalized = module.clone()
    rewrites = []
    def record(old, replacements, replacement):
        rewrites.append({"marker": old.op_name.data,
                         "source_attributes": {key: str(value) for key, value in old.attributes.items()
                                               if key.startswith("prov.")},
                         "properties": {key: str(value) for key, value in old.properties.items()},
                         "replacement_operations": [op.name for op in replacements],
                         "result_operation": replacement.name})
    if lower_quant_ext(normalized, on_rewrite=record) != len(markers):
        raise ValueError("captured quantization legalization did not cover every admitted marker")
    normalized.verify()
    if any(isinstance(op, UnregisteredOp) for op in normalized.walk()):
        raise ValueError("captured quantization legalization retained unknown operations")
    if text(module, generic=True) != original or _owners() != owners:
        raise ValueError("captured quantization source/implementation changed during legalization")
    return normalized, {
        "schema": "merlin.captured-quantization-legalization.v1", "input_ir_sha256": _sha(original.encode()),
        "output_ir_sha256": _sha(text(normalized, generic=True).encode()), "source_owners": owners,
        "pass": "merlin.compiler.quantization.linalg.lower_quant_ext", "rewrites": rewrites,
        "source_arithmetic": arithmetic,
        "scope": "explicit shared marker legalization; no new precision choice or numerical qualification",
    }
