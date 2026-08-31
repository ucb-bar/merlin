"""Parse linalg-on-tensors MLIR (model2MLIR output) and inventory its matmuls.

Works on the real artifacts: ``workloads/smolvla/smolvla.mlir`` (25k lines) parses in
a few seconds. Two model2MLIR/xDSL impedance notes, both handled here:

- xDSL's linalg parser rejects the parenthesized multi-result form
  ``} -> (tensor<...>, tensor<...>)`` that MLIR prints for multi-result
  ``linalg.generic`` (re-verified on the pinned xDSL 0.68: ``Expected '->'``);
  :func:`strip_paren_results` normalizes it textually before parsing.
- model2MLIR's *section splitter* can emit use-before-def SSA references
  (e.g. ``sections/smolvla.model.mlir`` references ``%2034`` which is never defined
  in that file) — invalid SSACFG IR. Parse the **full** artifact, not the sections,
  until that upstream bug is fixed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def strip_paren_results(text: str) -> str:
    """Drop the parentheses from a multi-result ``linalg`` region terminator.

    MLIR prints a multi-result ``linalg.generic`` as (real line from
    ``out/runs/rvv/beam/matmul/.../generated/v/model.prepared.mlir``)::

        } -> (tensor<1x32xf32>, tensor<1x32xi64>)

    and xDSL's linalg custom syntax rejects the parenthesized form — still true on the
    xDSL pinned here (0.68), verified by parsing that exact text: ``Expected '->'``. So the
    parens are removed BEFORE the parse; this repair is what makes the module parseable at
    all, which is why it is textual.

    Scanned structurally: a ``}``, optional whitespace, ``->``, optional whitespace, then a
    parenthesized group holding no further parentheses. Anything else is left verbatim —
    in particular a nested-paren type list is NOT something this normalizer understands, so
    it reaches the parser unchanged and fails there loudly instead of being mangled here.
    """
    out: list[str] = []
    i = 0
    while True:
        brace = text.find("}", i)
        if brace < 0:
            out.append(text[i:])
            return "".join(out)
        k = brace + 1
        while k < len(text) and text[k].isspace():
            k += 1
        if text[k:k + 2] == "->":
            k += 2
            while k < len(text) and text[k].isspace():
                k += 1
        else:
            k = -1
        if k < 0 or k >= len(text) or text[k] != "(":
            out.append(text[i:brace + 1])   # not a `} -> (...)` terminator
            i = brace + 1
            continue
        close = k + 1
        while close < len(text) and text[close] not in "()":
            close += 1
        if close >= len(text) or text[close] != ")" or close == k + 1:
            out.append(text[i:brace + 1])   # unbalanced / nested / empty — leave alone
            i = brace + 1
            continue
        out.append(text[i:k])               # `} -> ` verbatim, whitespace included
        out.append(text[k + 1:close])       # the result-type list, parens dropped
        i = close + 1


def make_context():
    """A permissive xDSL context for linalg-on-tensors modules."""
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.cf import Cf
    from xdsl.dialects.func import Func
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.tensor import Tensor

    ctx = Context(allow_unregistered=True)
    for d in (Builtin, Func, Arith, Linalg, Tensor, Scf, Math, Cf):
        ctx.load_dialect(d)
    return ctx


def parse_mlir_text(text: str, ctx=None):
    """Parse linalg-on-tensors MLIR text into an xDSL module.

    ``ctx`` lets a caller supply a context that loads extra dialects (e.g. the quant-aware
    context in :mod:`merlin.frontends.quant_ext`); it defaults to the permissive
    :func:`make_context`.
    """
    from xdsl.parser import Parser

    from ..common.ir_lock import IR_LOCK

    text = strip_paren_results(text)
    # THE serialization point for xDSL parsing, held here rather than at call sites because the call
    # sites are not discoverable by inspection: locking the two obvious ones in `build_app` still left
    # `c_runtime.generate` parsing twice for the @forward signature, and the resulting race surfaced as
    # a *mutation* invariant ("Can't add to a block an operation already attached to a block") on
    # perfectly valid IR, in whichever image happened to lose the race. Parsing a whole model is tens
    # of seconds against builds and simulations that take minutes, so the cost of serializing here is
    # small and the alternative is a build whose success depends on scheduling.
    with IR_LOCK:
        return Parser(ctx or make_context(), text).parse_module()


def parse_mlir_file(path: str | Path, ctx=None):
    return parse_mlir_text(Path(path).read_text(encoding="utf-8"), ctx=ctx)


def load_manifest(path: str | Path) -> dict[int, dict[str, Any]]:
    """safetensors manifest: func-arg index -> {weight, kind, dtype, shape}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


@dataclass
class MatmulRecord:
    """One matmul-family op found in the module."""

    kind: str                       # e.g. "linalg.matmul"
    m: int | None
    k: int | None
    n: int | None
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    dtype: str
    weight_arg_index: int | None    # func-arg index the RHS traces back to (if any)
    weight_name: str | None         # resolved via the safetensors manifest
    prov: dict[str, str] = field(default_factory=dict)


def _shape(t) -> tuple[int, ...]:
    from xdsl.dialects.builtin import TensorType

    return tuple(t.get_shape()) if isinstance(t, TensorType) else ()


def _dtype(t) -> str:
    from xdsl.dialects.builtin import TensorType

    return str(t.element_type) if isinstance(t, TensorType) else str(t)


def _prov(op) -> dict[str, str]:
    from xdsl.dialects.builtin import StringAttr

    out = {}
    for table in (op.attributes, getattr(op, "properties", {}) or {}):
        for key, val in table.items():
            if key.startswith("prov.") and isinstance(val, StringAttr):
                out[key] = val.data
    return out


def _trace_to_func_arg(value, func_args) -> int | None:
    """Follow simple view/layout chains (transpose, reshape, cast) to a func arg."""
    from xdsl.ir import BlockArgument

    seen = 0
    while seen < 32:
        seen += 1
        if isinstance(value, BlockArgument):
            return func_args.index(value) if value in func_args else None
        owner = value.owner
        name = getattr(owner, "name", "")
        if name in ("tensor.expand_shape", "tensor.collapse_shape", "tensor.cast"):
            value = owner.operands[0]
        elif name == "linalg.transpose":
            value = owner.inputs[0]
        else:
            return None
    return None


def matmul_inventory(module, manifest: dict[int, dict[str, Any]] | None = None
                     ) -> list[MatmulRecord]:
    """All linalg matmul-family ops, with weights resolved through the manifest."""
    fns = [op for op in module.walk() if op.name == "func.func"]
    if not fns:
        return []
    func_args = list(fns[0].body.blocks[0].args)

    records: list[MatmulRecord] = []
    for op in module.walk():
        if op.name not in ("linalg.matmul", "linalg.batch_matmul",
                           "linalg.quantized_matmul"):
            continue
        lhs, rhs = op.inputs[0], op.inputs[1]
        ls, rs = _shape(lhs.type), _shape(rhs.type)
        m = k = n = None
        if len(ls) == 2 and len(rs) == 2:
            m, k = ls
            _, n = rs
        idx = _trace_to_func_arg(rhs, func_args)
        name = None
        if idx is not None and manifest and idx in manifest:
            name = manifest[idx].get("weight")
        records.append(MatmulRecord(
            kind=op.name, m=m, k=k, n=n, lhs_shape=ls, rhs_shape=rs,
            dtype=_dtype(rhs.type), weight_arg_index=idx, weight_name=name,
            prov=_prov(op)))
    return records
