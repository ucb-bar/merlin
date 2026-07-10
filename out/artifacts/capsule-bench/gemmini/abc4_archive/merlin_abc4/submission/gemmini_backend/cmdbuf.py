"""Emit the target-independent command buffer (command_buffer.json) from the program.

Structure mirrors the frozen ``merlin_iface`` -> command-buffer mapping
(``bench_contract/command_buffer_abi.yaml``): leaf tensors up front, then ordered
commands with the contract opcodes. Operand values reference leaf tensor names and the
SSA handle/accumulator names exactly as the interface grammar does, so the reference
engine resolves them by name.
"""
from __future__ import annotations

from typing import Any

from . import iface_ir as IR


def build_command_buffer(prog: IR.Program) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for name, t in prog.tensors.items():
        tensors[name] = {"shape": list(t.shape), "dtype": t.dtype, "role": t.role}

    commands: list[dict[str, Any]] = []
    for op in prog.ops:
        if isinstance(op, IR.Pack):
            commands.append({
                "opcode": "RES_PACK",
                "operands": {"src": op.src, "dst": op.dst},
                "attributes": {"layout": op.layout}})
        elif isinstance(op, IR.Matmul):
            commands.append({
                "opcode": "MATMUL_RESIDENT",
                "operands": {"lhs": op.lhs, "rhs": op.rhs, "dst": op.dst}})
        elif isinstance(op, IR.Commit):
            attrs: dict[str, Any] = {
                "epilogue": list(op.epilogue),
                "output_dtype": op.output_dtype}
            if op.acc_scale is not None:
                attrs["acc_scale"] = float(op.acc_scale)
            commands.append({
                "opcode": "COMMIT",
                "operands": {"src": op.src, "dst": op.dst},
                "attributes": attrs})
        elif isinstance(op, IR.Movement):
            st = prog.tensors[op.src]
            tensors[op.dst] = {"shape": list(st.shape), "dtype": st.dtype,
                               "role": "output"}
            commands.append({
                "opcode": "VECTOR_MAP",
                "operands": {"lhs": op.src, "src": op.src, "dst": op.dst},
                "attributes": {"op": "identity", "combine": "identity",
                               "activation": [], "output_dtype": st.dtype}})
        elif isinstance(op, IR.Conv2d):
            from . import passes as _P
            ifm_t = prog.tensors[op.ifm]
            np_, co = _P.conv_out_shape(prog, op)
            kh, kw = int(op.kernel[0]), int(op.kernel[1])
            ci = int(op.kernel[2]) if len(op.kernel) > 2 else int(ifm_t.shape[-1])
            patch = kh * kw * ci
            # PROBE: declare the conv input leaf directly as the 2D im2col matrix.
            tensors[op.ifm] = {"shape": [np_, patch], "dtype": ifm_t.dtype,
                               "role": "input"}
            acc = "convacc_" + op.dst
            commands.append({
                "opcode": "MATMUL_RESIDENT",
                "operands": {"lhs": op.ifm, "rhs": op.rhs, "dst": acc}})
            cattrs = {"epilogue": list(op.epilogue), "output_dtype": op.output_dtype}
            if op.acc_scale is not None:
                cattrs["acc_scale"] = float(op.acc_scale)
            commands.append({
                "opcode": "COMMIT",
                "operands": {"src": acc, "dst": op.dst},
                "attributes": cattrs})
        elif isinstance(op, IR.Evict):
            commands.append({
                "opcode": "EVICT",
                "operands": {"handle": op.handle}})

    return {
        "abi_version": prog.abi_version or "0.1",
        "target": prog.target or "gemmini",
        "backend": "gemmini_oot_xdsl",
        "tensors": tensors,
        "commands": commands,
    }
