"""Front-end parsing of the frozen ``merlin_iface`` grammar (v0.1) into a small
normalized program IR.

This is the *front end* only: it reads the contract text into Python dataclasses.
The genuine compiler work (IRDL dialects + verifiers + a rewrite pass) happens on
top of this IR (see ``dialect_iface``/``dialect_gemmini``/``lower``).

The grammar is documented in ``bench_contract/interface_grammar.md``. We parse it with
a few regexes (the grammar is deliberately regular). We support the five core ops
(tensor / resident_pack / matmul / commit / evict) plus the two Gemmini extension ops
that appear in the capsule corpus: ``movement`` (pure mvin/mvout copy) and ``conv2d``
(im2col-lowered convolution).

No ``merlin`` import: this module depends only on the stdlib, so the shipped package
stays integrity-clean.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

GRAMMAR_VERSION = "0.1"


@dataclass
class Tensor:
    name: str
    shape: list[int]
    dtype: str
    role: str


@dataclass
class Pack:
    dst: str          # resident handle ssa name
    src: str          # weight tensor name
    layout: str


@dataclass
class Matmul:
    dst: str          # accumulator ssa name
    lhs: str          # input tensor name
    rhs: str          # resident handle ssa name


@dataclass
class Commit:
    dst: str          # output tensor name
    src: str          # accumulator ssa name
    epilogue: list[str]
    output_dtype: str
    acc_scale: float | None = None


@dataclass
class Movement:
    dst: str          # output tensor name
    src: str          # input tensor name


@dataclass
class Conv2d:
    dst: str          # output tensor name (im2col matmul result, [n_patches, out_ch])
    ifm: str          # input feature map tensor name (NHWC)
    rhs: str          # resident weight handle ssa name
    kernel: list[int]  # [kh, kw, ci, co]
    stride: list[int]
    padding: list[int]
    dilation: list[int]
    layout: str
    epilogue: list[str]
    output_dtype: str
    acc_scale: float | None = None


@dataclass
class Evict:
    handle: str


@dataclass
class Program:
    version: str
    target: str
    abi_version: str
    tensors: dict[str, Tensor] = field(default_factory=dict)
    ops: list[Any] = field(default_factory=list)
    # ssa name -> tensor name it refers to (handles map to their packed weight)
    pack_src: dict[str, str] = field(default_factory=dict)


_RE_MOD = re.compile(r"module\s+attributes\s*\{([^}]*)\}")
_RE_TENSOR = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.tensor\s*\{([^}]*)\}\s*:\s*tensor<([^>]+)>")
_RE_PACK = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.resident_pack\s*%(\S+)\s*\{([^}]*)\}")
_RE_MATMUL = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.matmul\s*%(\S+),\s*%(\S+)\s*:")
_RE_COMMIT = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.commit\s*%(\S+)\s*\{([^}]*)\}\s*:")
_RE_MOVEMENT = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.movement\s*%(\S+)\s*\{([^}]*)\}\s*:")
_RE_CONV = re.compile(
    r"%(\S+)\s*=\s*merlin_iface\.conv2d\s*%(\S+),\s*%(\S+)\s*\{([^}]*)\}\s*:")
_RE_EVICT = re.compile(r"merlin_iface\.evict\s*%(\S+)")


def _shape(s: str) -> tuple[list[int], str]:
    parts = s.split("x")
    dtype = parts[-1]
    dims = [int(p) for p in parts[:-1]]
    return dims, dtype


def _parse_attrs(body: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    depth = 0
    cur = ""
    parts: list[str] = []
    for ch in body:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
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
        if not body:
            return []
        return [_parse_value(x) for x in body.split(",")]
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    num = v.split(":")[0].strip()
    if re.fullmatch(r"[-+]?\d+", num):
        return int(num)
    try:
        return float(num)
    except ValueError:
        return num


class ParseError(Exception):
    pass


def parse_program(text: str) -> Program:
    mod = _RE_MOD.search(text)
    mattrs = _parse_attrs(mod.group(1)) if mod else {}
    prog = Program(
        version=str(mattrs.get("merlin_iface.version", "")),
        target=str(mattrs.get("merlin_iface.target", "")),
        abi_version=str(mattrs.get("merlin_iface.abi_version", "0.1")),
    )

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        m = _RE_TENSOR.search(line)
        if m:
            attrs = _parse_attrs(m.group(2))
            dims, dtype = _shape(m.group(3))
            name = attrs.get("name", m.group(1))
            prog.tensors[name] = Tensor(name, dims, dtype, attrs.get("role", "input"))
            continue

        m = _RE_PACK.search(line)
        if m:
            dst, src = m.group(1), m.group(2)
            attrs = _parse_attrs(m.group(3))
            prog.ops.append(Pack(dst=dst, src=src, layout=attrs.get("layout", "packed_rhs")))
            prog.pack_src[dst] = src
            continue

        m = _RE_MATMUL.search(line)
        if m:
            prog.ops.append(Matmul(dst=m.group(1), lhs=m.group(2), rhs=m.group(3)))
            continue

        m = _RE_COMMIT.search(line)
        if m:
            ssa, src = m.group(1), m.group(2)
            attrs = _parse_attrs(m.group(3))
            dst = attrs.get("name", ssa)
            prog.ops.append(Commit(
                dst=dst, src=src,
                epilogue=list(attrs.get("epilogue", [])),
                output_dtype=str(attrs.get("output_dtype", "i32")),
                acc_scale=attrs.get("acc_scale")))
            continue

        m = _RE_MOVEMENT.search(line)
        if m:
            ssa, src = m.group(1), m.group(2)
            attrs = _parse_attrs(m.group(3))
            prog.ops.append(Movement(dst=attrs.get("name", ssa), src=src))
            continue

        m = _RE_CONV.search(line)
        if m:
            ssa, ifm, rhs = m.group(1), m.group(2), m.group(3)
            attrs = _parse_attrs(m.group(4))
            prog.ops.append(Conv2d(
                dst=attrs.get("name", ssa), ifm=ifm, rhs=rhs,
                kernel=list(attrs.get("kernel", [])),
                stride=list(attrs.get("stride", [1, 1])),
                padding=list(attrs.get("padding", [0, 0, 0, 0])),
                dilation=list(attrs.get("dilation", [1, 1])),
                layout=str(attrs.get("layout", "nhwc")),
                epilogue=list(attrs.get("epilogue", [])),
                output_dtype=str(attrs.get("output_dtype", "i32")),
                acc_scale=attrs.get("acc_scale")))
            continue

        m = _RE_EVICT.search(line)
        if m:
            prog.ops.append(Evict(handle=m.group(1)))
            continue

    return prog
