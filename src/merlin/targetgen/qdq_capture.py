"""Fail-closed source correspondence for the supported Q/DQ capture profile.

This checks exact scalar literals, typed SSA operands, overloads and capture bytes.
It does not approve the proposed Lean specification or prove concrete IEEE laws.
Saved reports are evidence; callers must run the check again on selected artifacts.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from .qdq_capture_identity import digest, recheck, sha


class QDQCaptureError(ValueError):
    """The selected source cannot be bound to the supported Q/DQ model."""


def scalar_scale(value: object, *, quantize: bool) -> dict:
    if type(value) is not float or not math.isfinite(value):
        raise QDQCaptureError("scale must be an original finite Python float")
    try:
        raw = struct.pack(">f", value)
    except OverflowError as exc:
        raise QDQCaptureError("scale is outside f32") from exc
    materialized = struct.unpack(">f", raw)[0]
    if value.hex() != materialized.hex():
        raise QDQCaptureError(
            f"scalar scale loses precision on f32 materialization: {value.hex()} -> {materialized.hex()}"
        )
    if quantize:
        if value <= 0:
            raise QDQCaptureError("quantizer scale must be positive; zero raises in the Python source")
        try:
            inverse = struct.unpack(">f", struct.pack(">f", 1.0 / value))[0]
        except OverflowError as exc:
            raise QDQCaptureError("quantizer reciprocal is outside finite f32") from exc
        if not math.isfinite(inverse):
            raise QDQCaptureError("quantizer reciprocal is outside finite f32")
    return {"python_hex": value.hex(), "f32_bits": raw.hex()}


def _name(op) -> str:
    from xdsl.dialects.builtin import UnregisteredOp

    return op.op_name.data if isinstance(op, UnregisteredOp) else getattr(op, "name", "")


def _ids(op, key: str) -> list[str]:
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    values = op.attributes.get(key)
    if not isinstance(values, ArrayAttr) or any(not isinstance(v, StringAttr) for v in values):
        raise QDQCaptureError(f"missing exact {key}")
    return [v.data for v in values]


def _tensor(ty, shape: list[int], dtype: str) -> bool:
    from xdsl.dialects.builtin import TensorType

    types = {"float32": "f32", "int8": "i8", "int32": "i32", "int64": "i64"}
    return isinstance(ty, TensorType) and list(ty.get_shape()) == shape and str(ty.element_type) == types.get(dtype)


def _spec(node: dict) -> dict:
    results = node.get("results", [])
    if len(results) != 1:
        raise QDQCaptureError("source must have exactly one tensor result")
    result = results[0]
    if result.get("id") != node["id"] + ":v0":
        raise QDQCaptureError("source result identity differs from its node")
    shape = result.get("shape")
    if result.get("kind") != "tensor" or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise QDQCaptureError("source must have a static tensor result")
    return result


def _literal(operand):
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    splat = operand.owner
    if _name(splat) != "tensor.splat" or len(splat.operands) != 1:
        raise QDQCaptureError("scalar qparam is not a direct tensor.splat")
    constant = splat.operands[0].owner
    if _name(constant) != "arith.constant" or constant.operands:
        raise QDQCaptureError("scalar qparam is not an original arith.constant")
    attr = constant.properties.get("value")
    if isinstance(attr, (FloatAttr, IntegerAttr)):
        return attr.value.data
    raise QDQCaptureError("scalar qparam has an unsupported literal")


def check_source(trace: dict, module) -> list[dict]:
    """Recompute local source correspondence from parsed IR and original/prepared graphs.

    The initial supported capture is one root function containing Q/DQ and return.
    Other computations need their own source correspondence, rather than an ID guess.
    Tensor qparams remain typed symbolic inputs, never substituted sample values.
    """
    from xdsl.dialects.builtin import StringAttr
    from xdsl.utils.exceptions import VerifyException

    if module is None:
        raise QDQCaptureError("emitted MLIR cannot be parsed")
    try:
        module.verify()
    except VerifyException as exc:
        raise QDQCaptureError(f"emitted module is invalid: {exc}") from exc
    graphs = trace.get("graphs", {})
    original = graphs.get("original", {})
    prepared = graphs.get("prepared", {})
    if original.get("status") != "complete" or prepared.get("status") != "complete":
        raise QDQCaptureError("complete original and same-conversion prepared graphs are required")
    nodes = prepared.get("nodes", [])
    old_nodes = original.get("nodes", [])
    if len({n["id"] for n in nodes}) != len(nodes) or len({n["id"] for n in old_nodes}) != len(old_nodes):
        raise QDQCaptureError("duplicate frontend node identity")
    originals = {n["id"]: n for n in old_nodes}
    functions = [op for op in module.walk() if _name(op) == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise QDQCaptureError("supported capture requires one root function and one block")
    block = functions[0].body.block
    placeholders = [n for n in nodes if n["op"] == "placeholder"]
    if len(placeholders) != len(block.args):
        raise QDQCaptureError("frontend placeholder and function argument arity differ")
    old_placeholders = [n for n in old_nodes if n["op"] == "placeholder"]
    if len(old_placeholders) != len(placeholders):
        raise QDQCaptureError("original/prepared placeholder arity differs")
    bound, original_values = {}, {}
    for node, old, arg in zip(placeholders, old_placeholders, block.args, strict=True):
        spec = _spec(node)
        old_spec = _spec(old)
        if old["id"] not in node.get("origin_node_ids", []) or (spec["shape"], spec["dtype"]) != (
            old_spec["shape"],
            old_spec["dtype"],
        ):
            raise QDQCaptureError("original/prepared placeholder binding differs")
        if not _tensor(arg.type, spec["shape"], spec["dtype"]):
            raise QDQCaptureError("frontend placeholder and function argument types differ")
        bound[spec["id"]] = arg
        original_values[spec["id"]] = {"node_id": old["id"], "value_id": old_spec["id"]}
    targets = {
        "quantized_decomposed.quantize_per_tensor.default": "quant_ext.quantize_per_tensor",
        "quantized_decomposed.dequantize_per_tensor.default": "quant_ext.dequantize_per_tensor",
        "quantized_decomposed.dequantize_per_channel.default": "quant_ext.dequantize_per_channel",
    }
    calls = [n for n in nodes if n["op"] not in {"placeholder", "output"}]
    if not calls or any(n["target"] not in targets for n in calls):
        raise QDQCaptureError("capture contains unsupported source calls (including TorchAO or other overloads)")
    emitted = [op for op in block.ops if _name(op).startswith("quant_ext.")]
    if len(emitted) != len(calls):
        raise QDQCaptureError("source/emitted Q/DQ site counts differ")
    allowed = {*targets.values(), "arith.constant", "tensor.splat", "func.return"}
    if any(_name(op) not in allowed for op in block.ops):
        raise QDQCaptureError("capture contains computations without supported source correspondence")
    sites, consumed = [], set()

    def ref(argument):
        if not isinstance(argument, dict) or argument.get("value_id") not in bound:
            raise QDQCaptureError("source tensor operand is not bound to an exact SSA value")
        expected = {"node_id": argument["value_id"].rsplit(":v", 1)[0], "value_id": argument["value_id"]}
        if argument != expected:
            raise QDQCaptureError("source tensor reference node/value identities differ")
        return bound[argument["value_id"]]

    def original_argument(argument):
        if isinstance(argument, dict) and "value_id" in argument:
            ref(argument)
            if argument["value_id"] not in original_values:
                raise QDQCaptureError("prepared argument has no original value binding")
            return original_values[argument["value_id"]]
        if isinstance(argument, list):
            return [original_argument(item) for item in argument]
        return argument

    for node, op in zip(calls, emitted, strict=True):
        if _ids(op, "prov.source_node_ids") != [node["id"]]:
            raise QDQCaptureError("Q/DQ source identity or operation order differs")
        origins = [identity for identity in _ids(op, "prov.origin_node_ids") if identity in originals]
        if len(origins) != 1 or origins[0] in consumed:
            raise QDQCaptureError("Q/DQ has no unique original source call")
        old = originals[origins[0]]
        consumed.add(old["id"])
        target = node["target"]
        if old["target"] != target or _name(op) != targets[target]:
            raise QDQCaptureError("original, prepared and emitted overloads differ")
        if any(not key.startswith("prov.") and key != "op_name__" for key in op.attributes) or set(op.properties) - {
            "axis",
            "quant_min",
            "quant_max",
            "output_dtype",
        }:
            raise QDQCaptureError("Q/DQ contains unsupported semantic attributes")
        aten = op.attributes.get("prov.aten")
        if not isinstance(aten, StringAttr) or aten.data != target:
            raise QDQCaptureError("emitted overload provenance differs")
        args, old_args = node.get("args"), old.get("args")
        channel = target.endswith("dequantize_per_channel.default")
        if (
            not isinstance(args, list)
            or len(args) != (7 if channel else 6)
            or not isinstance(old_args, list)
            or len(old_args) != len(args)
            or node.get("kwargs")
            or old.get("kwargs")
        ):
            raise QDQCaptureError("unsupported Q/DQ source argument schema")
        # Compare complete ordered source arguments, including tensor identities
        # and exact float spellings. Equal types alone cannot bind a qparam.
        if json.dumps(original_argument(args), sort_keys=True) != json.dumps(old_args, sort_keys=True):
            raise QDQCaptureError("original call arguments changed during frontend preparation")
        if len(op.operands) != 3 or len(op.results) != 1 or op.operands[0] is not ref(args[0]):
            raise QDQCaptureError("Q/DQ data operand binding differs")
        spec = _spec(node)
        old_spec = _spec(old)
        if (spec["shape"], spec["dtype"]) != (old_spec["shape"], old_spec["dtype"]):
            raise QDQCaptureError("original/prepared result type differs")
        if not _tensor(op.results[0].type, spec["shape"], spec["dtype"]):
            raise QDQCaptureError("source and emitted result types differ")
        quantize = target == "quantized_decomposed.quantize_per_tensor.default"
        shape = spec["shape"]
        if not _tensor(op.operands[0].type, shape, "float32" if quantize else "int8"):
            raise QDQCaptureError("unsupported Q/DQ activation storage")
        if spec["dtype"] != ("int8" if quantize else "float32"):
            raise QDQCaptureError("unsupported Q/DQ result storage")
        scale = None
        if channel:
            axis, qmin, qmax, dtype = args[3:]
            if type(axis) is not int or not 0 <= axis < len(shape):
                raise QDQCaptureError("unsupported per-channel axis")
            axis_attr = op.properties.get("axis")
            if axis_attr is None or axis_attr.value.data != axis:
                raise QDQCaptureError("emitted axis differs from source")
            for i, dtype_name in ((1, "float32"), (2, "int32")):
                if op.operands[i] is not ref(args[i]) or not _tensor(op.operands[i].type, [shape[axis]], dtype_name):
                    raise QDQCaptureError("per-channel qparam SSA binding or type differs")
        else:
            qmin, qmax, dtype = args[3:]
            scale = scalar_scale(args[1], quantize=quantize)
            zero = args[2]
            if type(zero) is not int or not -(1 << 63) <= zero < (1 << 63):
                raise QDQCaptureError("zero point must be an original signed64 Python integer")
            if not _tensor(op.operands[1].type, [], "float32") or not _tensor(op.operands[2].type, [], "int64"):
                raise QDQCaptureError("unsupported scalar qparam storage")
            if _literal(op.operands[1]).hex() != args[1].hex() or _literal(op.operands[2]) != zero:
                raise QDQCaptureError("emitted scalar qparams differ from original source")
        if (
            type(qmin) is not int
            or type(qmax) is not int
            or (qmin, qmax) != (-128, 127)
            or dtype != {"kind": "dtype", "value": "torch.int8"}
        ):
            raise QDQCaptureError("unsupported source storage dtype or quantization range")
        for key, value in (("quant_min", qmin), ("quant_max", qmax)):
            attr = op.properties.get(key)
            if attr is None or attr.value.data != value:
                raise QDQCaptureError("emitted quantization bounds differ from source")
        bound[spec["id"]] = op.results[0]
        original_values[spec["id"]] = {"node_id": old["id"], "value_id": old_spec["id"]}
        sites.append(
            {
                "source_id": old["id"],
                "prepared_id": node["id"],
                "overload": target,
                "scale": scale,
                "zero_point": args[2],
                "quant_min": qmin,
                "quant_max": qmax,
            }
        )
    original_calls = {n["id"] for n in old_nodes if n["op"] not in {"placeholder", "output"}}
    if consumed != original_calls:
        raise QDQCaptureError("original source contains dropped or unsupported calls")
    outputs = [n for n in nodes if n["op"] == "output"]
    terminator = block.last_op
    if len(outputs) != 1 or _name(terminator) != "func.return":
        raise QDQCaptureError("source/output return schema differs")
    returned = outputs[0]["args"]
    old_outputs = [n for n in old_nodes if n["op"] == "output"]
    if len(old_outputs) != 1 or original_argument(returned) != old_outputs[0]["args"]:
        raise QDQCaptureError("original/prepared output binding differs")
    if len(returned) != 1 or not isinstance(returned[0], list):
        raise QDQCaptureError("unsupported source return structure")
    if tuple(ref(arg) for arg in returned[0]) != tuple(terminator.operands):
        raise QDQCaptureError("source/output return operand binding differs")
    return sites


def write_check(result, identity: dict, output: Path) -> dict:
    """Write fresh evidence for this conversion, preserving explicit refusals."""
    trace = result.capture_trace or {}
    raw_trace = (json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (output / "qdq-frontend.json").write_bytes(raw_trace)
    (output / "qdq-producer.json").write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n")
    (output / "qdq-source.mlir").write_text(result.mlir_text)
    report = {
        "schema": "merlin.qdq_capture.v1",
        "status": "unsupported",
        "sites": [],
        "identity_sha256": digest(identity),
        "trace_sha256": sha(raw_trace),
        "mlir_sha256": sha(result.mlir_text.encode()),
        "source_closure_verified": False,
        "scope": "supported Q/DQ source correspondence; not source-contract approval or numerical execution",
        "quantizer_definedness": "not_established",
    }
    try:
        recheck(identity)
        if not result.ok or result.module is None:
            raise QDQCaptureError("same-conversion parsed module is unavailable")
        # Parse the bytes that downstream lowering receives, rather than trusting
        # a different live module or the trace's self-reported MLIR digest.
        from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl

        parsed = _parse_mlir_text_to_xdsl(result.mlir_text)
        if parsed is None:
            raise QDQCaptureError("same-conversion emitted MLIR cannot be parsed")
        report["sites"] = check_source(trace, parsed)
        report["status"] = "matched"
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report["reason"] = str(exc)
    (output / "qdq-source-check.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def begin_worker(args, output: Path) -> dict | None:
    """Retain an explicit source/tool refusal before the worker starts conversion."""
    from .qdq_capture_identity import begin

    try:
        return begin(Path(args.m2m_dir), Path(args.qdq_torch_wheel))
    except (ValueError, OSError) as exc:
        refusal = {"status": "unsupported", "reason": str(exc), "source_closure_verified": False}
        (output / "qdq-source-check.json").write_text(json.dumps(refusal, indent=2) + "\n")
        (output / "meta.json").write_text(json.dumps({"ok": False, "opaque": 0, "qdq_source": refusal}))
        return None


def check_worker(result, identity: dict, output: Path, opaque: dict) -> dict:
    """Preserve a refused conversion before later reference evaluation can fail."""
    report = write_check(result, identity, output)
    if report["status"] != "matched":
        (output / "linalg.mlir").write_text(result.mlir_text)
        (output / "meta.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "opaque": sum(opaque.values()),
                    "opaque_detail": opaque,
                    "qdq_source": report,
                }
            )
        )
    return report


def replay(output: Path) -> dict:
    """Reopen sources and recompute a saved correspondence check; no cached verdict."""
    from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl

    record = json.loads((output / "qdq-source-check.json").read_bytes())
    identity = json.loads((output / "qdq-producer.json").read_bytes())
    trace_raw = (output / "qdq-frontend.json").read_bytes()
    mlir_raw = (output / "qdq-source.mlir").read_bytes()
    if (
        digest(identity) != record["identity_sha256"]
        or sha(trace_raw) != record["trace_sha256"]
        or sha(mlir_raw) != record["mlir_sha256"]
    ):
        raise QDQCaptureError("saved capture evidence bytes changed")
    recheck(identity)
    sites = check_source(json.loads(trace_raw), _parse_mlir_text_to_xdsl(mlir_raw.decode()))
    return {"status": "matched", "sites": sites, "mlir_sha256": sha(mlir_raw)}
