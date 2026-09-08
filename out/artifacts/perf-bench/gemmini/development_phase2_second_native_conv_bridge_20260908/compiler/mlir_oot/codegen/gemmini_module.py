"""Materialise the scheduled command stream as a verified `gemmini`-dialect module.

This is what `--convert-iface-to-gemmini` prints: one target-dialect op per accelerator command,
inside a `func.func @gemmini_kernel` whose pointer arguments are the kernel ABI's own operand
list.  Building it through IRDL means the RTL-derived limits (mesh DIM, scratchpad depth,
accumulator depth) are checked by `verify()` before anything is encoded.
"""
from __future__ import annotations

from typing import Any

from xdsl.dialects import func, llvm
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    Float32Type,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    i64,
)
from xdsl.ir import Block, Region

from ..ir import gemmini_dialect as G
from ..frontend.linalg_reader import HOST_LANE, MESH_LANE
from ..lowering.plan import Plan
from ..lowering.schedule import Instr
from ..tables import rtl_facts as F

PTR = llvm.LLVMPointerType()

_OP_FOR_KIND = {
    "flush": G.FlushOp,
    "config_ex": G.ConfigExOp,
    "config_ld": G.ConfigLdOp,
    "config_st": G.ConfigStOp,
    "mvin": G.MvinOp,
    "mvout": G.MvoutOp,
    "preload": G.PreloadOp,
    "compute": G.ComputeOp,
    "fence": G.FenceOp,
    "host_epilogue": G.HostEpilogueOp,
    "host_transpose": G.HostTransposeOp,
    "host_bias_add": G.HostEpilogueOp,
    "host_linalg": G.HostLaneProgramOp,
    "host_segment": G.HostLaneProgramOp,
    "im2col_row": G.Im2colRowOp,
    "loop_ws_block": G.LoopWsBlockOp,
    "loop_conv_ws": G.LoopConvWsOp,
}


#: Attributes that carry compiler-internal IR handles rather than target facts; they drive
#: codegen and have no place in the printed target module.
_OPAQUE_ATTRS = frozenset({"func", "segment"})


def _attr(value: Any):
    if isinstance(value, bool):
        return IntegerAttr(1 if value else 0, 64)
    if isinstance(value, int):
        return IntegerAttr(value, 64)
    if isinstance(value, float):
        return FloatAttr(value, Float32Type())
    if isinstance(value, str):
        return StringAttr(value)
    if isinstance(value, (list, tuple)):
        return ArrayAttr([_attr(v) for v in value])
    return StringAttr(str(value))


def build(plan: Plan, instrs: list[Instr], staging: dict[str, Any]) -> ModuleOp:
    args = list(plan.kernel_args)
    block = Block(arg_types=[PTR] * len(args))
    slot = {name: block.args[i] for i, name in enumerate(args)}
    staged: dict[str, Any] = {}
    ops = []
    for ins in instrs:
        cls = _OP_FOR_KIND.get(ins.kind)
        if cls is None:                                        # pragma: no cover - guarded
            raise ValueError(f"no gemmini-dialect op for {ins.kind!r}")
        operands = []
        for name in ins.bufs:
            if name in slot:
                operands.append(slot[name])
            else:
                if name not in staged:
                    buf = staging.get(name) or plan.buffers[name]
                    sc = G.ScratchOp(operands=[[]], result_types=[[PTR]])
                    sc.attributes["bytes"] = IntegerAttr(buf.nbytes, 64)
                    sc.attributes["name"] = StringAttr(name)
                    ops.append(sc)
                    staged[name] = sc.results[0]
                operands.append(staged[name])
        op = cls(operands=[operands], result_types=[[]])
        for key, value in ins.attrs.items():
            if value is None or key in _OPAQUE_ATTRS:
                continue
            op.attributes[key] = _attr(value)
        if ins.bufs:
            op.attributes["buffers"] = ArrayAttr([StringAttr(n) for n in ins.bufs])
        if ins.kind == "host_bias_add":
            op.attributes["stages"] = ArrayAttr([StringAttr("bias_add")])
        ops.append(op)
    ops.append(func.ReturnOp())
    block.add_ops(ops)
    fn = func.FuncOp("gemmini_kernel", ([PTR] * len(args), []), Region([block]))
    module = ModuleOp([fn])
    module.attributes["gemmini.dim"] = IntegerAttr(F.DIM, 64)
    module.attributes["gemmini.scratchpad_rows"] = IntegerAttr(F.SPAD_ROWS, 64)
    module.attributes["gemmini.accumulator_rows"] = IntegerAttr(F.ACC_ROWS, 64)
    module.attributes["gemmini.kernel_args"] = ArrayAttr([StringAttr(a) for a in args])
    placement = (plan.command_buffer.get("params") or {}).get("lane_placement")
    if placement is not None:
        module.attributes["gemmini.host_lane_regions"] = ArrayAttr(
            [StringAttr(r["region"]) for r in placement if r["lane"] == HOST_LANE])
        module.attributes["gemmini.mesh_regions"] = ArrayAttr(
            [StringAttr(r["region"]) for r in placement if r["lane"] == MESH_LANE])
    module.verify()
    return module


def declined_module(reason: str, op: str = "", shape=None) -> ModuleOp:
    """The target module for a capsule this backend DECLINED.

    `lower_interface_to_target` has to answer with target MLIR whatever the answer is: a decline
    that arrives as an empty stdout is indistinguishable from a crashed tool, and the runner reads
    it as a protocol failure rather than as the coverage gap the command buffer declares.  The
    module carries no accelerator op -- the decline is the whole content -- and says why.
    """
    block = Block(arg_types=[])
    block.add_ops([func.ReturnOp()])
    fn = func.FuncOp("gemmini_kernel", ([], []), Region([block]))
    module = ModuleOp([fn])
    module.attributes["gemmini.dim"] = IntegerAttr(F.DIM, 64)
    module.attributes["gemmini.declined"] = StringAttr(reason)
    if op:
        module.attributes["gemmini.declined_op"] = StringAttr(op)
    if shape:
        module.attributes["gemmini.declined_shape"] = ArrayAttr(
            [IntegerAttr(int(d), 64) for d in shape])
    module.verify()
    return module
