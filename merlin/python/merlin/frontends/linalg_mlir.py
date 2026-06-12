"""Parse linalg-on-tensors MLIR (model2MLIR output) and inventory its matmuls.

Works on the real artifacts: ``workloads/smolvla/smolvla.mlir`` (25k lines) parses in
a few seconds. Two model2MLIR/xDSL impedance notes, both handled here:

- xDSL 0.65's linalg parser rejects the parenthesized multi-result form
  ``} -> (tensor<...>, tensor<...>)`` that MLIR prints for multi-result
  ``linalg.generic``; :data:`PAREN_RESULTS` normalizes it textually before parsing.
- model2MLIR's *section splitter* can emit use-before-def SSA references
  (e.g. ``sections/smolvla.model.mlir`` references ``%2034`` which is never defined
  in that file) — invalid SSACFG IR. Parse the **full** artifact, not the sections,
  until that upstream bug is fixed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# xDSL 0.65 linalg custom syntax does not accept `} -> (T1, T2)`.
PAREN_RESULTS = re.compile(r"(\}\s*->\s*)\(([^()]+)\)")


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


def parse_mlir_text(text: str):
    """Parse linalg-on-tensors MLIR text into an xDSL module."""
    from xdsl.parser import Parser

    text = PAREN_RESULTS.sub(r"\1\2", text)
    return Parser(make_context(), text).parse_module()


def parse_mlir_file(path: str | Path):
    return parse_mlir_text(Path(path).read_text(encoding="utf-8"))


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
