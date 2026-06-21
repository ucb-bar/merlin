"""Emit the target-independent command buffer (``command_buffer.json``) from the gemmini module.

Opcode/operand/attribute encoding follows ``bench_contract/command_buffer_abi.yaml`` so the
abstract program the runner interprets matches the one the device executes. Conv carries an
``im2col`` recipe under ``params`` so the runner materializes the im2col activation identically
on every side.
"""
from __future__ import annotations

from typing import Any

from . import dialects as D

ABI = "0.1"


def _dt(tensor_type) -> str:
    return "i" + str(tensor_type.get_element_type().width.data)


def _name(value) -> str:
    return value.owner.sym.data


def _ints(arr) -> list[int]:
    return [int(x.value.data) for x in arr]


def build_command_buffer(gem: "D.ModuleOp") -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    commands: list[dict[str, Any]] = []
    recipes: list[dict[str, Any]] = []

    for op in gem.body.block.ops:
        if isinstance(op, D.GTensorOp):
            tensors[op.sym.data] = {
                "shape": [int(d) for d in op.res.type.get_shape()],
                "dtype": _dt(op.res.type), "role": op.role.data}
        elif isinstance(op, D.GPackOp):
            commands.append({"opcode": "RES_PACK",
                             "operands": {"src": _name(op.src), "dst": op.sym.data},
                             "attributes": {"layout": op.layout.data}})
        elif isinstance(op, D.GMatmulOp):
            commands.append({"opcode": "MATMUL_RESIDENT",
                             "operands": {"lhs": _name(op.lhs), "rhs": _name(op.rhs),
                                          "dst": op.sym.data}})
            if op.im2col is not None:
                d = op.im2col.data
                recipes.append({
                    "source": d["source"].data, "target": d["target"].data,
                    "kh": int(d["kh"].value.data), "kw": int(d["kw"].value.data),
                    "ci": int(d["ci"].value.data),
                    "stride": _ints(d["stride"]), "padding": _ints(d["padding"]),
                    "dilation": _ints(d["dilation"]), "layout": d["layout"].data})
        elif isinstance(op, D.GCommitOp):
            attrs: dict[str, Any] = {
                "epilogue": [e.data for e in op.epilogue],
                "output_dtype": op.output_dtype.data}
            if op.acc_scale is not None:
                attrs["acc_scale"] = float(op.acc_scale.value.data)
            commands.append({"opcode": "COMMIT",
                             "operands": {"src": _name(op.acc), "dst": op.sym.data},
                             "attributes": attrs})
        elif isinstance(op, D.GMoveOp):
            commands.append({"opcode": "VECTOR_MAP",
                             "operands": {"lhs": _name(op.src), "dst": op.sym.data},
                             "attributes": {"combine": "identity"}})
        elif isinstance(op, D.GReleaseOp):
            commands.append({"opcode": "EVICT",
                             "operands": {"handle": _name(op.handle)}})

    cb: dict[str, Any] = {"abi_version": ABI, "target": "gemmini",
                          "backend": "gemmini_oot", "tensors": tensors, "commands": commands}
    if recipes:
        cb["params"] = {"im2col_recipes": recipes}
    return cb
