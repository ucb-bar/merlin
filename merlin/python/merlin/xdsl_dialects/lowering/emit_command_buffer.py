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


def emit_command_buffer(module) -> dict[str, Any]:
    """Walk the runtime module and emit the executable command-buffer dict."""
    if not HAS_XDSL:
        raise LoweringError("xDSL is required to emit a command buffer")
    from .. import runtime as r
    from . import analyses

    problems = analyses.check_command_buffer_consistency(module)
    if problems:
        raise LoweringError("; ".join(problems))

    devices = [op for op in module.walk() if isinstance(op, r.DeviceGetOp)]
    creates = [op for op in module.walk() if isinstance(op, r.CommandBufferCreateOp)]
    if len(devices) != 1 or len(creates) != 1:
        raise LoweringError("expected exactly one device.get and one "
                            "command_buffer.create")
    dev, create = devices[0], creates[0]

    commands: list[dict[str, Any]] = []
    bias_names: set[str] = set()
    for op in module.walk():
        if not isinstance(op, r.CommandBufferAppendOp):
            continue
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
