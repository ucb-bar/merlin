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
# target encodes onto the same opcode set — that is what keeps metrics comparable). Only the ONE
# built-in reference dialect (toynpu) is baked here. A reference target's dialect that lives in its own
# package (e.g. saturn) ships its op->opcode map WITH the dialect (its plugin self-registers it); those
# discovered maps are merged in via ``plugin_opcodes()`` below — no target name hardcoded here.
TARGET_OPCODES = {
    "toynpu.res_pack": "RES_PACK",
    "toynpu.matmul": "MATMUL_RESIDENT",
    "toynpu.commit": "COMMIT",
    "toynpu.evict": "EVICT",
    "toynpu.vector_map": "VECTOR_MAP",
    "toynpu.vector_reduce": "VREDUCE",
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
    # Float element types carry their own mnemonic (f16/f32/f64/bf16) for a float-model layer.
    tok = str(elem)
    if tok in ("f16", "f32", "f64", "bf16"):
        return tok
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
    # built-in (toynpu) + every discovered reference dialect's map (e.g. saturn, keyed by full op name so
    # cross-target merges never collide) + the explicit ``opcodes`` a generated package supplies.
    from .target_lowering import plugin_opcodes
    opcode_map = {**TARGET_OPCODES}
    for _m in plugin_opcodes().values():
        opcode_map.update(_m)
    opcode_map.update(opcodes or {})
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
    for operand in (ret_ops[0].operands if ret_ops else []):
        nm = names.get(operand)
        if nm is None:
            raise LoweringError("return operand has no runtime name")
        tensors[nm] = _shape_str(operand.type)
        output_names.append(nm)

    blk = Block()
    dev = r.DeviceGetOp(result_types=[r.DeviceType()], properties={
        "device": StringAttr("%s0" % target),
        "backend": r.BackendAttr(r.Backend(backend))})
    cb = r.CommandBufferCreateOp(operands=[dev.dev], result_types=[r.CommandBufferType()],
                                 properties={
        "target": StringAttr(target),
        "mode": r.SubmitModeAttr(r.SubmitMode.BATCHED),
        "tensors": DictionaryAttr({k: StringAttr(v) for k, v in sorted(tensors.items())}),
        "outputs": ArrayAttr([StringAttr(n) for n in output_names]),
    })
    ops = [dev, cb]

    for op in src_block.ops:
        opcode = kind(op)
        if opcode == "RES_PACK":
            args = {"src": names[op.src], "dst": names[op.res]}
            attrs = {"layout": op.layout}
            # A generated target's pack op may not declare the optional dequant-scale operand/axis
            # (only the factory-built resident packs do) — getattr keeps this total for both.
            scale = getattr(op, "scale", None)
            if scale is not None:                        # int8 weight-only dequant pack
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
