"""target dialect -> ``runtime`` (merlin-target-to-runtime stage).

Encodes each selected target op as an abstract command appended to one command buffer:
device.get -> command_buffer.create -> append* -> submit -> wait -> metrics.read.
Tensor naming is deterministic: the pack source is ``W``, other tensor args are
``A0..``, pack results are ``W_res``, accumulators ``acc{i}``, commits ``Y{i}``. The
command buffer's resource table (leaf tensor shapes/dtypes) rides on
``command_buffer.create`` so the terminal emit stage is a pure function of this module.
"""

from __future__ import annotations

from typing import Any

from merlin.compiler.dialects.lowering.interface_stage import LoweringError
from merlin.compiler.dialects.xdsl_context import HAS_XDSL
from merlin.target.capabilities.families import DEFAULT_EXAMPLE_TARGET

# ---- runtime_lowering --------------------------------------------------------------------------

# Target op -> shared command opcode comes from the authored example's dialect plan, a selected
# support plugin, or the isolated/generated package supplied through ``opcodes``.

METRICS_TO_CAPTURE = [
    "cycles",
    "bytes_moved",
    "command_count",
    "pack_count",
    "resident_hits",
    "evictions",
    "accumulator_commits",
]


def _dtype_str(t) -> str:
    from xdsl.dialects.builtin import IntegerType

    elem = t.element_type
    if isinstance(elem, IntegerType):
        return "i%d" % elem.width.data
    # Float element types carry their own mnemonic (f16/f32/f64/bf16) for a float-model layer.
    tok = str(elem)
    if tok in ("f16", "f32", "f64", "bf16"):
        return tok
    raise LoweringError("unsupported element type %s" % elem)


def _shape_str(t) -> str:
    return "x".join(str(d) for d in t.get_shape()) + ":" + _dtype_str(t)


def lower_to_runtime(
    module, target: str = DEFAULT_EXAMPLE_TARGET, backend: str = "simulator", opcodes: dict | None = None
):
    """Rebuild the target module as runtime command-buffer IR.

    ``opcodes`` (target-op name -> command-buffer opcode) is merged over selected mappings so an
    isolated/generated target package can supply its own encoding without editing this module.
    """
    if not HAS_XDSL:
        return module
    # Authored checkout example maps + explicitly selected support plugins + generated package maps.
    from merlin.compiler.dialects.lowering.target_stage import plugin_opcodes

    opcode_map = {}
    for _m in plugin_opcodes().values():
        opcode_map.update(_m)
    opcode_map.update(opcodes or {})
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FunctionType, ModuleOp, StringAttr, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from merlin.compiler.dialects import runtime as r

    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        raise LoweringError("no func.func in module")
    fn = fns[0]
    src_block = fn.body.blocks[0]

    def kind(op) -> str | None:
        """Merlin opcode for a target op (reference + generated dialects share field names)."""
        return opcode_map.get(op.name)

    # Deterministic names, assigned in ONE pre-pass so every value (incl. committed outputs)
    # is named before the create op is built. Pack sources are weights; other args activations;
    # accumulators acc%d; committed tensors Y%d.
    pack_srcs = [op.src for op in src_block.ops if kind(op) == "RES_PACK"]
    names: dict = {}
    tensors: dict[str, str] = {}
    n_w = 0
    for v in pack_srcs:
        names[v] = "W" if n_w == 0 else "W%d" % n_w
        n_w += 1
    n_a = 0
    for arg in src_block.args:
        if arg in names:
            continue
        if isinstance(arg.type, TensorType):
            names[arg] = "A%d" % n_a
            n_a += 1
    n_acc = n_y = n_v = 0
    for op in src_block.ops:
        k = kind(op)
        if k == "RES_PACK":
            names[op.res] = names[op.src] + "_res"
        elif k in ("MATMUL_RESIDENT", "MATMUL"):
            names[op.acc] = "acc%d" % n_acc
            n_acc += 1
        elif k == "COMMIT":
            names[op.out] = "Y%d" % n_y
            n_y += 1
        elif k in ("VECTOR_MAP", "VREDUCE"):
            names[op.out] = "V%d" % n_v
            n_v += 1
    for arg in src_block.args:
        if arg in names:
            tensors[names[arg]] = _shape_str(arg.type)
    # The function's RETURN values are the model's output tensors — declare them (with shape)
    # and name them explicitly so the engine surfaces exactly these, not every commit. A vector-family
    # result reaches here through the return, so this covers both the matmul and the vector engine.
    ret_ops = [o for o in src_block.ops if o.name == "func.return"]
    output_names: list[str] = []
    for operand in ret_ops[0].operands if ret_ops else []:
        nm = names.get(operand)
        if nm is None:
            raise LoweringError("return operand has no runtime name")
        tensors[nm] = _shape_str(operand.type)
        output_names.append(nm)

    blk = Block()
    dev = r.DeviceGetOp(
        result_types=[r.DeviceType()],
        properties={"device": StringAttr("%s0" % target), "backend": r.BackendAttr(r.Backend(backend))},
    )
    cb = r.CommandBufferCreateOp(
        operands=[dev.dev],
        result_types=[r.CommandBufferType()],
        properties={
            "target": StringAttr(target),
            "mode": r.SubmitModeAttr(r.SubmitMode.BATCHED),
            "tensors": DictionaryAttr({k: StringAttr(v) for k, v in sorted(tensors.items())}),
            "outputs": ArrayAttr([StringAttr(n) for n in output_names]),
        },
    )
    ops = [dev, cb]

    for op in src_block.ops:
        opcode = kind(op)
        if opcode == "RES_PACK":
            args = {"src": names[op.src], "dst": names[op.res]}
            attrs = {"layout": op.layout}
            # A generated target's pack op may not declare the optional dequant-scale operand/axis
            # (only the factory-built resident packs do) — getattr keeps this total for both.
            scale = getattr(op, "scale", None)
            if scale is not None:  # int8 weight-only dequant pack
                args["scale"] = names[scale]
                dq_axis = getattr(op, "dequant_axis", None)
                if dq_axis is not None:
                    attrs["dequant_axis"] = dq_axis
        elif opcode in ("MATMUL_RESIDENT", "MATMUL"):
            args = {"lhs": names[op.lhs], "rhs": names[op.rhs], "dst": names[op.acc]}
            attrs = {}
            # Honest opcode: only a pack-produced RHS is a resident matmul.
            if not names[op.rhs].endswith("_res"):
                opcode = "MATMUL"
        elif opcode == "COMMIT":
            args = {"src": names[op.acc], "dst": names[op.out]}
            if op.bias is not None:
                args["bias"] = op.bias.data
            attrs = {"epilogue": op.epilogue}
            if op.requant_shift is not None:
                attrs["requant_shift"] = op.requant_shift
            if op.output_dtype is not None:
                attrs["output_dtype"] = op.output_dtype
        elif opcode == "VECTOR_MAP":
            # The destination was named and declared in the pre-pass above (the tensor table is
            # already frozen into the create op by this point).
            args = {"lhs": names[op.lhs], "rhs": names[op.rhs], "dst": names[op.out]}
            attrs = {"combine": op.combine}
            if op.activation is not None:
                attrs["activation"] = op.activation
        elif opcode == "EVICT":
            args = {"handle": names[op.handle]}
            attrs = {}
        elif opcode == "VECTOR_MAP":
            args = {"lhs": names[op.lhs], "rhs": names[op.rhs], "dst": names[op.out]}
            attrs = {"combine": op.combine}
            if op.activation is not None:
                attrs["activation"] = op.activation
        elif opcode == "VREDUCE":
            args = {"src": names[op.src], "dst": names[op.out]}
            attrs = {"op": op.reduce}
        elif op.name == "func.return":
            continue
        else:
            raise LoweringError("no runtime encoding for %s" % op.name)
        ops.append(
            r.CommandBufferAppendOp(
                operands=[cb.cb],
                properties={
                    "opcode": StringAttr(opcode),
                    "args": DictionaryAttr({k: StringAttr(v) for k, v in args.items()}),
                    "attrs": DictionaryAttr(attrs),
                    "queue": r.QueueKindAttr(r.QueueKind.COMPUTE),
                },
            )
        )

    sub = r.SubmitOp(
        operands=[dev.dev, cb.cb],
        result_types=[r.EventType()],
        properties={"mode": r.SubmitModeAttr(r.SubmitMode.BLOCKING)},
    )
    wait = r.WaitOp(operands=[sub.event])
    met = r.MetricsReadOp(
        operands=[dev.dev],
        result_types=[r.MetricsType()],
        properties={"metrics": ArrayAttr([StringAttr(m) for m in METRICS_TO_CAPTURE])},
    )
    ops += [sub, wait, met, ReturnOp()]
    blk.add_ops(ops)
    new_fn = FuncOp(fn.sym_name.data, FunctionType.from_lists([], []), Region([blk]))
    return ModuleOp([new_fn])


# ---- emit_command_buffer -----------------------------------------------------------------------
# ``runtime`` module -> command-buffer dict (merlin-runtime-to-command-buffer stage).
#
# A pure function of the runtime module: reads the device/backend, the create op's
# target + resource table, and the ordered appends, and produces the dict the Python
# engine (``merlin.runtime``) executes — conforming to command_buffer.schema.yaml.

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
    from merlin.compiler.dialects import runtime as r
    from merlin.compiler.dialects.lowering import analyses

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
        raise LoweringError(
            f"this module describes {len(buffers)} command buffers; use emit_command_buffers() to get them all"
        )
    return buffers[0]


def _emit_one(module, create) -> dict[str, Any]:
    """One buffer, carrying only the commands appended to IT and its own device."""
    from merlin.compiler.dialects import runtime as r

    dev = create.dev.owner if isinstance(create.dev.owner, r.DeviceGetOp) else None
    if dev is None:
        raise LoweringError("runtime.command_buffer.create's device operand is not a device.get")

    commands: list[dict[str, Any]] = []
    bias_names: set[str] = set()
    for op in module.walk():
        if not isinstance(op, r.CommandBufferAppendOp):
            continue
        if op.cb.owner is not create:
            continue  # belongs to another buffer; see the docstring above
        cmd: dict[str, Any] = {"opcode": op.opcode.data, "operands": _attr_to_py(op.args)}
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
    produced = {
        c["operands"]["dst"] for c in commands if c["opcode"] in ("VECTOR_MAP", "VREDUCE") and "dst" in c["operands"]
    }
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
