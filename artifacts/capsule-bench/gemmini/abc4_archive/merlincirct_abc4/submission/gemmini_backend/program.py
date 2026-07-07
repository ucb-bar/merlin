"""Extract a plain-Python ``Program`` from a verified gemmini-dialect module.

Walks the target IR, resolves tensor shapes/names through the def-use chain, and
produces a small ordered record list that the command-buffer and RoCC emitters
consume.  This keeps the emitters decoupled from xDSL traversal details.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import StringAttr, TensorType

from .dialects import (GCommitOp, GConvOp, GMatmulOp, GMovementOp, GPackOp,
                       GReleaseOp, IfTensorOp)


@dataclass
class Program:
    target: str = "gemmini"
    abi_version: str = "0.1"
    tensors: dict[str, dict] = field(default_factory=dict)   # name -> {shape,dtype,role}
    ops: list[dict] = field(default_factory=list)            # ordered records


def _ten_type(val) -> TensorType:
    t = val.type
    assert isinstance(t, TensorType), f"expected tensor type, got {t}"
    return t


def _shape(val) -> list[int]:
    return [int(d) for d in _ten_type(val).get_shape()]


def _dtype(val) -> str:
    return "i" + str(_ten_type(val).get_element_type().width.data)


def _leaf(val):
    """Return the IfTensorOp that defines a leaf value (following 0-hop)."""
    owner = val.owner
    if isinstance(owner, IfTensorOp):
        return owner
    raise ValueError(f"value not a leaf tensor: {owner}")


def _name(op: IfTensorOp) -> str:
    return op.tname.data


def extract(module) -> Program:
    prog = Program()

    def reg_leaf(val, role_default="input"):
        lt = _leaf(val)
        nm = _name(lt)
        if nm not in prog.tensors:
            prog.tensors[nm] = {"shape": _shape(val), "dtype": _dtype(val),
                                "role": lt.role.data}
        return nm

    for op in module.body.block.ops:
        if isinstance(op, IfTensorOp):
            continue  # registered lazily when referenced
        if isinstance(op, GPackOp):
            wname = reg_leaf(op.src)
            prog.ops.append({"kind": "pack", "src": wname,
                             "dst": _ssa(op.results[0]),
                             "layout": op.layout.data,
                             "shape": prog.tensors[wname]["shape"]})
        elif isinstance(op, GMatmulOp):
            lhs_name = reg_leaf(op.lhs)
            wname = _pack_src_name(op.rhs)
            prog.ops.append({"kind": "matmul", "lhs": lhs_name,
                             "rhs": _ssa(op.rhs), "dst": _ssa(op.results[0]),
                             "lhs_shape": _shape(op.lhs),
                             "weight": wname,
                             "weight_shape": prog.tensors[wname]["shape"]})
        elif isinstance(op, GMovementOp):
            src_name = reg_leaf(op.src)
            prog.ops.append({"kind": "movement", "src": src_name,
                             "dst": op.tname.data,
                             "shape": prog.tensors[src_name]["shape"],
                             "dtype": prog.tensors[src_name]["dtype"]})
        elif isinstance(op, GConvOp):
            ifm_name = reg_leaf(op.ifm)
            wname = _pack_src_name(op.weight)
            prog.ops.append({
                "kind": "conv2d", "ifm": ifm_name, "weight": wname,
                "dst": op.tname.data,
                "ifm_shape": prog.tensors[ifm_name]["shape"],
                "weight_shape": prog.tensors[wname]["shape"],
                "out_shape": _shape(op.results[0]),
                "kernel": [int(x.value.data) for x in op.kernel],
                "stride": [int(x.value.data) for x in op.stride],
                "padding": [int(x.value.data) for x in op.padding],
                "dilation": [int(x.value.data) for x in op.dilation],
                "epilogue": [e.data for e in op.epilogue],
                "output_dtype": op.output_dtype.data,
                "layout": op.layout.data,
            })
        elif isinstance(op, GCommitOp):
            mm = op.acc.owner
            assert isinstance(mm, GMatmulOp)
            prog.ops.append({
                "kind": "commit", "acc": _ssa(op.acc),
                "dst": op.tname.data,
                "epilogue": [e.data for e in op.epilogue],
                "output_dtype": op.output_dtype.data,
                "acc_scale": (op.acc_scale.value.data
                              if op.acc_scale is not None else None),
                "out_shape": _shape(op.results[0]),
            })
        elif isinstance(op, GReleaseOp):
            prog.ops.append({"kind": "evict", "handle": _ssa(op.handle)})
    return prog


def _ssa(val) -> str:
    return val.name_hint or str(id(val))


def _pack_src_name(resident_val) -> str:
    owner = resident_val.owner
    assert isinstance(owner, GPackOp), f"resident not from pack: {owner}"
    return _name(_leaf(owner.src))
