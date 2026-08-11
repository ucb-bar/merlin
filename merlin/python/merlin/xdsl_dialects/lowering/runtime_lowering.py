"""target dialect -> ``runtime`` (merlin-target-to-runtime stage).

Encodes each toynpu op as an abstract command appended to one command buffer:
device.get -> command_buffer.create -> append* -> submit -> wait -> metrics.read.
Tensor naming is deterministic: the pack source is ``W``, other tensor args are
``A0..``, pack results are ``W_res``, accumulators ``acc{i}``, commits ``Y{i}``. The
command buffer's resource table (leaf tensor shapes/dtypes) rides on
``command_buffer.create`` so the terminal emit stage is a pure function of this module.
"""
from __future__ import annotations

from .._common import HAS_XDSL
from .interface_lowering import LoweringError

# Target op -> Merlin-owned abstract opcode (the command buffer is Merlin's; every
# target encodes onto the same opcode set — that is what keeps metrics comparable).
TARGET_OPCODES = {
    "toynpu.res_pack": "RES_PACK",
    "toynpu.matmul": "MATMUL_RESIDENT",
    "toynpu.commit": "COMMIT",
    "toynpu.evict": "EVICT",
    "saturn.pack": "RES_PACK",
    "saturn.matmul": "MATMUL_RESIDENT",
    "saturn.commit": "COMMIT",
    "saturn.release": "EVICT",
}
# Generated targets (e.g. gemmini) supply their own target-op -> opcode map from their isolated
# package (merlin.targetgen.registry); it is merged in via the ``opcodes`` arg of
# lower_to_runtime rather than hardcoded here.

METRICS_TO_CAPTURE = ["cycles", "bytes_moved", "command_count", "pack_count",
                      "resident_hits", "evictions", "accumulator_commits"]


def _dtype_str(t) -> str:
    from xdsl.dialects.builtin import IntegerType

    elem = t.element_type
    if isinstance(elem, IntegerType):
        return "i%d" % elem.width.data
    raise LoweringError("unsupported element type %s" % elem)


def _shape_str(t) -> str:
    return "x".join(str(d) for d in t.get_shape()) + ":" + _dtype_str(t)


def lower_to_runtime(module, target: str = "toy_npu", backend: str = "simulator",
                     opcodes: dict | None = None):
    """Rebuild the target module as runtime command-buffer IR.

    ``opcodes`` (target-op name -> command-buffer opcode) is merged over the built-in map so an
    isolated/generated target package can supply its own encoding without editing this module.
    """
    if not HAS_XDSL:
        return module
    opcode_map = {**TARGET_OPCODES, **(opcodes or {})}
    from xdsl.ir import Block, Region
    from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, FunctionType,
                                       ModuleOp, StringAttr, TensorType)
    from xdsl.dialects.func import FuncOp, ReturnOp

    from .. import runtime as r

    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        raise LoweringError("no func.func in module")
    fn = fns[0]
    src_block = fn.body.blocks[0]

    def kind(op) -> str | None:
        """Merlin opcode for a target op (reference + generated dialects share field names)."""
        return opcode_map.get(op.name)

    # Deterministic tensor names. Pack sources are weights; other args activations.
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
    for arg in src_block.args:
        if arg in names:
            tensors[names[arg]] = _shape_str(arg.type)
    # Vector-family destinations are RESULTS, not arguments, so they have to be named and declared
    # here — the tensor table is frozen into the command-buffer-create op below, before the command
    # loop runs, and a destination added later would never reach it. Both engines find vector results
    # by scanning the table, so the omission produced a buffer that ran and returned nothing.
    n_declared_results = 0
    for op in src_block.ops:
        if kind(op) == "VECTOR_MAP":
            names[op.out] = f"Y{n_declared_results}"
            tensors[names[op.out]] = _shape_str(op.out.type)
            n_declared_results += 1

    blk = Block()
    dev = r.DeviceGetOp(result_types=[r.DeviceType()], properties={
        "device": StringAttr("%s0" % target),
        "backend": r.BackendAttr(r.Backend(backend))})
    cb = r.CommandBufferCreateOp(operands=[dev.dev], result_types=[r.CommandBufferType()],
                                 properties={
        "target": StringAttr(target),
        "mode": r.SubmitModeAttr(r.SubmitMode.BATCHED),
        "tensors": DictionaryAttr({k: StringAttr(v) for k, v in sorted(tensors.items())}),
    })
    ops = [dev, cb]

    n_acc = 0
    # Continue the Y numbering past anything the pre-pass already claimed, so commits and vector
    # results can never be handed the same name. They cannot co-occur today (a mixed payload is
    # refused at the interface stage), and a silent collision is not the way to find out that changed.
    n_y = n_declared_results
    for op in src_block.ops:
        opcode = kind(op)
        if opcode == "RES_PACK":
            handle = names[op.src] + "_res"
            names[op.res] = handle
            args = {"src": names[op.src], "dst": handle}
            attrs = {"layout": op.layout}
        elif opcode in ("MATMUL_RESIDENT", "MATMUL"):
            acc = "acc%d" % n_acc
            n_acc += 1
            names[op.acc] = acc
            args = {"lhs": names[op.lhs], "rhs": names[op.rhs], "dst": acc}
            attrs = {}
            # Honest opcode: only a pack-produced RHS is a resident matmul.
            if not names[op.rhs].endswith("_res"):
                opcode = "MATMUL"
        elif opcode == "COMMIT":
            y = "Y%d" % n_y
            n_y += 1
            names[op.out] = y
            args = {"src": names[op.acc], "dst": y}
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
        elif op.name == "func.return":
            continue
        else:
            raise LoweringError("no runtime encoding for %s" % op.name)
        ops.append(r.CommandBufferAppendOp(operands=[cb.cb], properties={
            "opcode": StringAttr(opcode),
            "args": DictionaryAttr({k: StringAttr(v) for k, v in args.items()}),
            "attrs": DictionaryAttr(attrs),
            "queue": r.QueueKindAttr(r.QueueKind.COMPUTE)}))

    sub = r.SubmitOp(operands=[dev.dev, cb.cb], result_types=[r.EventType()],
                     properties={"mode": r.SubmitModeAttr(r.SubmitMode.BLOCKING)})
    wait = r.WaitOp(operands=[sub.event])
    met = r.MetricsReadOp(operands=[dev.dev], result_types=[r.MetricsType()],
                          properties={"metrics": ArrayAttr(
                              [StringAttr(m) for m in METRICS_TO_CAPTURE])})
    ops += [sub, wait, met, ReturnOp()]
    blk.add_ops(ops)
    new_fn = FuncOp(fn.sym_name.data, FunctionType.from_lists([], []), Region([blk]))
    return ModuleOp([new_fn])
