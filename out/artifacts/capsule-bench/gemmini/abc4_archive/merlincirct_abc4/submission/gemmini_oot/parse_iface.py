"""Parse a ``merlin_iface`` interface.mlir into a verified xDSL ``merlin_iface`` module.

The grammar is the small, regular, regex-readable MLIR the contract freezes
(``bench_contract/interface_grammar.md``); we read it line by line and build genuine xDSL
ops (with verifiers), preserving the SSA value names so the downstream command buffer keys
match the reference round-trip exactly.
"""
from __future__ import annotations

import re

from xdsl.dialects.builtin import (ArrayAttr, FloatAttr, IntegerAttr, ModuleOp, StringAttr,
                                    TensorType, f32, i8, i16, i32)
from xdsl.ir import Block, Region, SSAValue

from .dialects import (AccType, IfaceCommitOp, IfaceConvOp, IfaceEvictOp, IfaceMatmulOp,
                       IfaceMoveOp, IfaceResidentPackOp, IfaceTensorOp, ResidentType)

SUPPORTED_VERSION = "0.1"

_DT = {"i8": i8, "i16": i16, "i32": i32}

_RE_MOD = re.compile(r"module\s+attributes\s*\{([^}]*)\}")
_RE_TENSOR = re.compile(r'%(\S+)\s*=\s*merlin_iface\.tensor\s*\{([^}]*)\}\s*:\s*(tensor<[^>]+>)')
_RE_PACK = re.compile(r'%(\S+)\s*=\s*merlin_iface\.resident_pack\s*%(\S+)\s*\{([^}]*)\}')
_RE_MATMUL = re.compile(r'%(\S+)\s*=\s*merlin_iface\.matmul\s*%(\S+),\s*%(\S+)\s*:')
_RE_COMMIT = re.compile(
    r'%(\S+)\s*=\s*merlin_iface\.commit\s*%(\S+)\s*\{([^}]*)\}\s*:\s*\(.*?\)\s*->\s*(tensor<[^>]+>)')
_RE_MOVE = re.compile(
    r'%(\S+)\s*=\s*merlin_iface\.movement\s*%(\S+)\s*\{([^}]*)\}\s*:\s*\(.*?\)\s*->\s*(tensor<[^>]+>)')
_RE_CONV = re.compile(
    r'%(\S+)\s*=\s*merlin_iface\.conv2d\s*%(\S+),\s*%(\S+)\s*\{([^}]*)\}\s*:\s*\(.*?\)\s*->\s*(tensor<[^>]+>)')
_RE_EVICT = re.compile(r'merlin_iface\.evict\s*%(\S+)')


class ParseError(Exception):
    pass


def _tensor_type(text: str) -> TensorType:
    m = re.fullmatch(r"tensor<([0-9x]+)x(i\d+)>", text.strip())
    if not m:
        raise ParseError(f"unparseable tensor type {text!r}")
    dims = [int(d) for d in m.group(1).split("x")]
    dt = _DT.get(m.group(2))
    if dt is None:
        raise ParseError(f"unsupported dtype {m.group(2)!r}")
    return TensorType(dt, dims)


def _split_top(s: str) -> list[str]:
    parts, cur, depth = [], "", 0
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
    return parts


def _value(v: str):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        body = v[1:-1].strip()
        return [] if not body else [_value(x) for x in _split_top(body)]
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    num = v.split(":")[0].strip()
    if re.fullmatch(r"[-+]?\d+", num):
        return int(num)
    try:
        return float(num)
    except ValueError:
        return num


def _attrs(block: str) -> dict:
    out = {}
    for part in _split_top(block):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = _value(v)
    return out


def _arr(xs) -> ArrayAttr:
    return ArrayAttr([IntegerAttr(int(x), 64) for x in xs])


def _epi(xs) -> ArrayAttr:
    return ArrayAttr([StringAttr(str(x)) for x in xs])


def parse_module(text: str) -> ModuleOp:
    """Build a verified xDSL ``merlin_iface`` module from interface text."""
    m = _RE_MOD.search(text)
    mod_attrs = _attrs(m.group(1)) if m else {}
    version = mod_attrs.get("merlin_iface.version")
    if version != SUPPORTED_VERSION:
        raise ParseError(f"unsupported merlin_iface.version {version!r} (want {SUPPORTED_VERSION})")

    blk = Block()
    env: dict[str, SSAValue] = {}
    ops = []

    def add(op):
        ops.append(op)
        return op

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("module") or line == "}":
            continue
        mt = _RE_TENSOR.search(line)
        if mt:
            a = _attrs(mt.group(2))
            op = add(IfaceTensorOp(properties={
                "sym": StringAttr(a.get("name", mt.group(1))),
                "role": StringAttr(a.get("role", "input"))},
                result_types=[_tensor_type(mt.group(3))]))
            env[mt.group(1)] = op.res
            continue
        mp = _RE_PACK.search(line)
        if mp:
            a = _attrs(mp.group(3))
            op = add(IfaceResidentPackOp(operands=[env[mp.group(2)]], properties={
                "sym": StringAttr(mp.group(1)),
                "layout": StringAttr(a.get("layout", "packed_rhs"))},
                result_types=[ResidentType()]))
            env[mp.group(1)] = op.res
            continue
        mm = _RE_MATMUL.search(line)
        if mm:
            op = add(IfaceMatmulOp(operands=[env[mm.group(2)], env[mm.group(3)]],
                                   properties={"sym": StringAttr(mm.group(1))},
                                   result_types=[AccType()]))
            env[mm.group(1)] = op.res
            continue
        mc = _RE_COMMIT.search(line)
        if mc:
            a = _attrs(mc.group(3))
            props = {"sym": StringAttr(a.get("name", mc.group(1))),
                     "epilogue": _epi(a.get("epilogue", [])),
                     "output_dtype": StringAttr(a.get("output_dtype", "i32"))}
            if "acc_scale" in a:
                props["acc_scale"] = FloatAttr(float(a["acc_scale"]), f32)
            op = add(IfaceCommitOp(operands=[env[mc.group(2)]], properties=props,
                                   result_types=[_tensor_type(mc.group(4))]))
            env[mc.group(1)] = op.res
            continue
        mv = _RE_MOVE.search(line)
        if mv:
            a = _attrs(mv.group(3))
            op = add(IfaceMoveOp(operands=[env[mv.group(2)]],
                                 properties={"sym": StringAttr(a.get("name", mv.group(1)))},
                                 result_types=[_tensor_type(mv.group(4))]))
            env[mv.group(1)] = op.res
            continue
        mn = _RE_CONV.search(line)
        if mn:
            a = _attrs(mn.group(4))
            props = {"sym": StringAttr(a.get("name", mn.group(1))),
                     "kernel": _arr(a.get("kernel", [])),
                     "stride": _arr(a.get("stride", [1, 1])),
                     "padding": _arr(a.get("padding", [0, 0, 0, 0])),
                     "dilation": _arr(a.get("dilation", [1, 1])),
                     "layout": StringAttr(a.get("layout", "nhwc")),
                     "epilogue": _epi(a.get("epilogue", [])),
                     "output_dtype": StringAttr(a.get("output_dtype", "i32"))}
            if "acc_scale" in a:
                props["acc_scale"] = FloatAttr(float(a["acc_scale"]), f32)
            op = add(IfaceConvOp(operands=[env[mn.group(2)], env[mn.group(3)]],
                                 properties=props, result_types=[_tensor_type(mn.group(5))]))
            env[mn.group(1)] = op.res
            continue
        me = _RE_EVICT.search(line)
        if me:
            add(IfaceEvictOp(operands=[env[me.group(1)]]))
            continue
        raise ParseError(f"unrecognized interface op: {line!r}")

    blk.add_ops(ops)
    module = ModuleOp(Region([blk]))
    module.verify()
    return module
