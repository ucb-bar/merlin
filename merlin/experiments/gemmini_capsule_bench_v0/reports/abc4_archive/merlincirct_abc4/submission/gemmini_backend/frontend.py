"""Frontend: read the frozen ``merlin_iface`` v0.1 text and build an xDSL module.

The interface grammar is *deliberately* a small regular text format (the contract
calls it "regex-parseable, decoupled from xDSL").  We read it with a few regexes
and then **construct a verified xDSL ``merlin_iface`` module** — the IR the
lowering pass and emitters operate on.  Parsing text is not the backend; the
xDSL IR + rewrite passes are.
"""
from __future__ import annotations

import re
from typing import Any

from xdsl.dialects.builtin import (ArrayAttr, Float32Type, FloatAttr, IntegerType,
                                   ModuleOp, StringAttr, TensorType)
from xdsl.ir import Block, Region, SSAValue

from .dialects import (AccType, IfCommitOp, IfConv2dOp, IfEvictOp, IfMatmulOp,
                       IfMovementOp, IfResidentPackOp, IfTensorOp, ResidentType)

_RE_MOD = re.compile(r"module\s+attributes\s*\{([^}]*)\}")
_RE_TENSOR = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.tensor\s*\{([^}]*)\}\s*:\s*(tensor<[^>]+>)")
_RE_PACK = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.resident_pack\s*%(\S+)\s*\{([^}]*)\}")
_RE_MATMUL = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.matmul\s*%(\S+),\s*%(\S+)\s*:")
_RE_MOVE = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.movement\s*%(\S+)\s*\{([^}]*)\}\s*:\s*"
    r"\(([^)]*)\)\s*->\s*(tensor<[^>]+>)")
_RE_CONV = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.conv2d\s*%(\S+),\s*%(\S+)\s*\{([^}]*)\}\s*:\s*"
    r"\(([^)]*)\)\s*->\s*(tensor<[^>]+>)")
_RE_COMMIT = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.commit\s*%(\S+)\s*\{([^}]*)\}\s*:\s*\(.*?\)\s*"
    r"->\s*(tensor<[^>]+>)")
_RE_EVICT = re.compile(r"merlin_iface\.evict\s*%(\S+)")
_TENSOR_TY = re.compile(r"tensor<([0-9x]+)x(i\d+)>")


def _parse_attr_block(s: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    depth, cur, parts = 0, "", []
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = _parse_value(v.strip())
    return out


def _parse_value(v: str) -> Any:
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        body = v[1:-1].strip()
        return [] if not body else [_parse_value(x) for x in body.split(",")]
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    num = v.split(":")[0].strip()
    if re.fullmatch(r"[-+]?\d+", num):
        return int(num)
    try:
        return float(num)
    except ValueError:
        return num


def _dtype(s: str) -> IntegerType:
    return IntegerType(int(s[1:]))


def _tensor_type(ttype: str) -> TensorType:
    m = _TENSOR_TY.search(ttype)
    if not m:
        raise ValueError(f"unparseable tensor type {ttype!r}")
    dims = [int(d) for d in m.group(1).split("x")]
    return TensorType(_dtype(m.group(2)), dims)


def _strarr(xs) -> ArrayAttr:
    return ArrayAttr([StringAttr(str(x)) for x in xs])


def _intarr(xs) -> ArrayAttr:
    from xdsl.dialects.builtin import IntegerAttr, i64
    return ArrayAttr([IntegerAttr(int(x), i64) for x in xs])


def build_module(text: str) -> ModuleOp:
    """Parse interface text into a verified xDSL ``merlin_iface`` ModuleOp."""
    mod = _RE_MOD.search(text)
    mod_attrs = _parse_attr_block(mod.group(1)) if mod else {}
    ver = mod_attrs.get("merlin_iface.version", "0.1")
    if str(ver) != "0.1":
        raise ValueError(f"unsupported merlin_iface.version {ver!r}")

    block = Block()
    vals: dict[str, SSAValue] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        m = _RE_TENSOR.search(line)
        if m:
            attrs = _parse_attr_block(m.group(2))
            op = IfTensorOp(
                properties={"tname": StringAttr(attrs.get("name", m.group(1))),
                            "role": StringAttr(attrs.get("role", "input"))},
                result_types=[_tensor_type(m.group(3))])
            block.add_op(op)
            op.results[0].name_hint = m.group(1)
            vals[m.group(1)] = op.results[0]
            continue

        m = _RE_PACK.search(line)
        if m:
            attrs = _parse_attr_block(m.group(3))
            op = IfResidentPackOp(
                operands=[vals[m.group(2)]],
                properties={"layout": StringAttr(attrs.get("layout", "packed_rhs"))},
                result_types=[ResidentType()])
            block.add_op(op)
            op.results[0].name_hint = m.group(1)
            vals[m.group(1)] = op.results[0]
            continue

        m = _RE_MATMUL.search(line)
        if m:
            op = IfMatmulOp(
                operands=[vals[m.group(2)], vals[m.group(3)]],
                result_types=[AccType(IntegerType(32))])
            block.add_op(op)
            op.results[0].name_hint = m.group(1)
            vals[m.group(1)] = op.results[0]
            continue

        m = _RE_MOVE.search(line)
        if m:
            attrs = _parse_attr_block(m.group(3))
            op = IfMovementOp(
                operands=[vals[m.group(2)]],
                properties={"tname": StringAttr(attrs.get("name", m.group(1)))},
                result_types=[_tensor_type(m.group(5))])
            block.add_op(op)
            op.results[0].name_hint = m.group(1)
            vals[m.group(1)] = op.results[0]
            continue

        m = _RE_CONV.search(line)
        if m:
            attrs = _parse_attr_block(m.group(4))
            op = IfConv2dOp(
                operands=[vals[m.group(2)], vals[m.group(3)]],
                properties={
                    "tname": StringAttr(attrs.get("name", m.group(1))),
                    "kernel": _intarr(attrs.get("kernel", [])),
                    "stride": _intarr(attrs.get("stride", [1, 1])),
                    "padding": _intarr(attrs.get("padding", [0, 0, 0, 0])),
                    "dilation": _intarr(attrs.get("dilation", [1, 1])),
                    "epilogue": _strarr(attrs.get("epilogue", [])),
                    "output_dtype": StringAttr(attrs.get("output_dtype", "i32")),
                    "layout": StringAttr(attrs.get("layout", "nhwc")),
                },
                result_types=[_tensor_type(m.group(6))])
            block.add_op(op)
            op.results[0].name_hint = m.group(1)
            vals[m.group(1)] = op.results[0]
            continue

        m = _RE_COMMIT.search(line)
        if m:
            attrs = _parse_attr_block(m.group(3))
            props: dict[str, Any] = {
                "tname": StringAttr(attrs.get("name", m.group(1))),
                "epilogue": _strarr(attrs.get("epilogue", [])),
                "output_dtype": StringAttr(attrs.get("output_dtype", "i8")),
            }
            if "acc_scale" in attrs:
                props["acc_scale"] = FloatAttr(float(attrs["acc_scale"]), Float32Type())
            op = IfCommitOp(operands=[vals[m.group(2)]], properties=props,
                            result_types=[_tensor_type(m.group(4))])
            block.add_op(op)
            op.results[0].name_hint = m.group(1)
            vals[m.group(1)] = op.results[0]
            continue

        m = _RE_EVICT.search(line)
        if m:
            block.add_op(IfEvictOp(operands=[vals[m.group(1)]]))
            continue

    module = ModuleOp(Region([block]))
    module.verify()
    return module
