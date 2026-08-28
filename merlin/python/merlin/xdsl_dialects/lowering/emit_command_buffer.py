"""``runtime`` module -> command-buffer dict (merlin-runtime-to-command-buffer stage).

A pure function of the runtime module: reads the device/backend, the create op's
target + resource table, and the ordered appends, and produces the dict the Python
engine (``merlin.runtime``) executes — conforming to command_buffer.schema.yaml.
"""
from __future__ import annotations

from typing import Any

from .._common import HAS_XDSL
from .interface_lowering import LoweringError

ABI_VERSION = "0.1"


def _attr_to_py(attr) -> Any:
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerAttr, StringAttr

    if isinstance(attr, StringAttr):
        return attr.data
    if isinstance(attr, IntegerAttr):
        return attr.value.data
    if isinstance(attr, ArrayAttr):
        return [_attr_to_py(a) for a in attr]
    if isinstance(attr, DictionaryAttr):
        return {k: _attr_to_py(v) for k, v in attr.data.items()}
    raise LoweringError("cannot lower attribute %r to a command buffer value" % attr)


def _parse_shape(spec: str) -> tuple[list[int], str]:
    shape_s, dtype = spec.split(":")
    return [int(d) for d in shape_s.split("x")], dtype


def emit_command_buffers(module) -> list[dict[str, Any]]:
    """One executable command buffer per ``command_buffer.create`` in the module.

    A configuration with two accelerators is a normal one, and the IR has always been able to say so:
    ``create`` takes its device as an OPERAND and ``append`` takes its buffer as one, so which device
    runs which commands is already written down. The single-buffer cap was an artificial one.

    Lifting it is a correctness fix independently of multi-device work. The previous code collected
    EVERY append in the module regardless of which buffer it was appended to, so a second buffer would
    not merely have been rejected -- had the cap ever been relaxed without this, both buffers would
    have been emitted carrying all of both their commands.
    """
    if not HAS_XDSL:
        raise LoweringError("xDSL is required to emit a command buffer")
    from .. import runtime as r
    from . import analyses

    problems = analyses.check_command_buffer_consistency(module)
    if problems:
        raise LoweringError("; ".join(problems))

    creates = [op for op in module.walk() if isinstance(op, r.CommandBufferCreateOp)]
    if not creates:
        raise LoweringError("no runtime.command_buffer.create in the module")
    return [_emit_one(module, create) for create in creates]


def emit_command_buffer(module) -> dict[str, Any]:
    """The single command buffer this module describes.

    Kept for callers that are single-device by nature (an oracle grading one capsule). A module
    describing several devices is a real thing, not an error, so it is directed to
    :func:`emit_command_buffers` rather than rejected as malformed.
    """
    buffers = emit_command_buffers(module)
    if len(buffers) != 1:
        raise LoweringError(f"this module describes {len(buffers)} command buffers; "
                            f"use emit_command_buffers() to get them all")
    return buffers[0]


def _emit_one(module, create) -> dict[str, Any]:
    """One buffer, carrying only the commands appended to IT and its own device."""
    from .. import runtime as r

    dev = create.dev.owner if isinstance(create.dev.owner, r.DeviceGetOp) else None
    if dev is None:
        raise LoweringError("runtime.command_buffer.create's device operand is not a device.get")

    commands: list[dict[str, Any]] = []
    bias_names: set[str] = set()
    for op in module.walk():
        if not isinstance(op, r.CommandBufferAppendOp):
            continue
        if op.cb.owner is not create:
            continue                      # belongs to another buffer; see the docstring above
        cmd: dict[str, Any] = {"opcode": op.opcode.data,
                               "operands": _attr_to_py(op.args)}
        attrs = _attr_to_py(op.attrs) if op.attrs is not None else {}
        if attrs:
            cmd["attributes"] = attrs
        if "bias" in cmd["operands"]:
            bias_names.add(cmd["operands"]["bias"])
        commands.append(cmd)

    weights = {c["operands"]["src"] for c in commands if c["opcode"] == "RES_PACK"}
    output_names = [s.data for s in create.outputs] if create.outputs is not None else []
    # Vector-family destinations are RESULTS too — a vector workload declares no create.outputs, so also
    # collect VECTOR_MAP/VREDUCE dsts by role, else such a result is mislabelled an input and silently
    # not read back. The matmul path names its result through create.outputs; union covers both engines.
    produced = {c["operands"]["dst"] for c in commands
                if c["opcode"] in ("VECTOR_MAP", "VREDUCE") and "dst" in c["operands"]}
    outputs_set = set(output_names) | produced
    tensors: dict[str, Any] = {}
    table = create.tensors.data if create.tensors is not None else {}
    for name, spec in table.items():
        shape, dtype = _parse_shape(spec.data)
        if name in outputs_set:
            role = "output"
        elif name in weights:
            role = "weight"
        elif name in bias_names:
            role = "bias"
        else:
            role = "input"
        tensors[name] = {"shape": shape, "dtype": dtype, "role": role}

    metrics_requested: list[str] = []
    for op in module.walk():
        if isinstance(op, r.MetricsReadOp):
            metrics_requested = _attr_to_py(op.metrics)

    cb: dict[str, Any] = {
        "abi_version": ABI_VERSION,
        "target": create.target.data,
        "backend": dev.backend.data.value,
        "tensors": tensors,
        "commands": commands,
    }
    if output_names:
        cb["outputs"] = output_names
    if metrics_requested:
        cb["metrics_requested"] = metrics_requested
    return cb
