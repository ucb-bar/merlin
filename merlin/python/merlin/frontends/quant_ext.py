"""Opt-in, quant-aware parsing of model2MLIR bundles.

model2MLIR emits quantized weights using its ``quant_ext`` dialect (``m2m.ir.quant``:
``quant_ext.dequantize_per_channel`` & friends, plus the metadata types
``affine_tensor`` / ``packed_int_tensor`` / ``mx_tensor`` / ``nvfp4_tensor``). The default
frontend context (:func:`merlin.frontends.linalg_mlir.make_context`) parses those as *unregistered*
ops on purpose — the battle-tested int8 lowering (`llvmlower.passes_xdsl.lower_quant_ext`) and
`common.mlir_query` match on the unregistered op name, and must keep doing so. This module adds an
**opt-in** context that additionally registers ``quant_ext`` so callers that want *structured* access
to quantization metadata (element bits, scale kind, granularity) get typed ops/attributes instead of
having to scrape text — without changing the default path.

The m2m ``quant_ext`` dialect is pure xDSL (no torch); we import it without triggering m2m's
torch-bound package init (m2m's ``__init__`` is lazy). Everything here fails **soft**: if the m2m
checkout is not resolvable, :func:`available` is ``False`` and callers fall back to the untyped path.

Metadata is reported against the canonical, target-agnostic :mod:`merlin.common.quant_formats`
registry so the rest of the tooling speaks one vocabulary of formats.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from merlin.common import quant_formats as qf
from merlin.frontends import linalg_mlir as _fl


def _m2m_dir() -> Path | None:
    """Resolve the model2MLIR checkout (``MERLIN_M2M_DIR`` then ``MERLIN_MODEL2MLIR``)."""
    for var in ("MERLIN_M2M_DIR", "MERLIN_MODEL2MLIR"):
        val = os.environ.get(var)
        if val and (Path(val) / "m2m" / "ir" / "quant" / "dialect.py").is_file():
            return Path(val)
    return None


@lru_cache(maxsize=1)
def load_dialect() -> Any | None:
    """Return the m2m ``quant_ext`` ``Dialect`` object, or ``None`` if unavailable.

    Adds the m2m checkout to ``sys.path`` (idempotent) and imports the pure-xDSL dialect. Any
    failure (no checkout, import error) resolves to ``None`` so the caller degrades gracefully.
    """
    base = _m2m_dir()
    if base is None:
        return None
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    try:
        from m2m.ir.quant import Quant
    except Exception:  # noqa: BLE001 — soft dependency; untyped path still works
        return None
    return Quant


def available() -> bool:
    """True when the ``quant_ext`` dialect can be registered (m2m checkout resolvable)."""
    return load_dialect() is not None


def make_quant_context():
    """A frontend context that additionally registers ``quant_ext`` (typed quant ops/types).

    Falls back to the plain :func:`merlin.frontends.linalg_mlir.make_context` if the dialect is
    unavailable, so parsing still succeeds (quant ops stay unregistered).
    """
    ctx = _fl.make_context()
    dialect = load_dialect()
    if dialect is not None:
        ctx.load_dialect(dialect)
    return ctx


def parse_quant_mlir(source: str | Path):
    """Parse a linalg-on-tensors + ``quant_ext`` module with the quant-aware context.

    ``source`` is a path or MLIR text. Reuses the frontend's parser (and its sanctioned
    text-normalization) so the only quant-specific part here is which dialects the context loads.
    """
    ctx = make_quant_context()
    if isinstance(source, str) and "\n" in source:
        return _fl.parse_mlir_text(source, ctx=ctx)
    return _fl.parse_mlir_file(source, ctx=ctx)


# --- structured metadata accessors ----------------------------------------------------------------


@dataclass(frozen=True)
class QuantizedTensor:
    """One quantized operand/result found in a quant-aware module."""

    op_name: str                       # e.g. "quant_ext.dequantize_per_channel"
    storage_dtype: str                 # element type of the stored tensor, e.g. "i8"
    shape: tuple[int, ...]
    granularity: str | None            # per_tensor | per_channel | per_group | per_block | None
    fmt: qf.QuantFormat | None         # canonical format, when resolvable


def _prop_str(op: Any, key: str) -> str | None:
    """Read a string-ish property (``input_dtype``) off a typed op, without text scraping."""
    from xdsl.dialects.builtin import StringAttr

    attr = op.properties.get(key)
    if isinstance(attr, StringAttr):
        return attr.data
    return None


def _granularity_from_op_name(op_name: str) -> str | None:
    tail = op_name.rsplit(".", 1)[-1]           # dequantize_per_channel -> per_channel
    for g in ("per_tensor", "per_channel", "per_group", "per_token", "per_row"):
        if tail.endswith(g):
            return g
    return None


def _int_width(ty: Any) -> int | None:
    from xdsl.dialects.builtin import IntegerType

    return ty.width.data if isinstance(ty, IntegerType) else None


def format_from_quant_type(attr: Any) -> qf.QuantFormat | None:
    """Map a model2MLIR ``quant_ext`` metadata *type* onto a canonical :class:`QuantFormat`.

    Uses the registry's structural fields (element bits, kind), never string matching. Returns
    ``None`` when a type is ambiguous or has no registered counterpart.
    """
    dialect = load_dialect()
    if dialect is None:
        return None
    from m2m.ir.quant.types import (
        AffineQuantizedTensorType,
        MXQuantizedTensorType,
        NVFP4TensorType,
        PackedIntTensorType,
    )

    if isinstance(attr, NVFP4TensorType):
        return qf.get("nvfp4")
    if isinstance(attr, MXQuantizedTensorType):
        bits = attr.element_bit_width.value.data
        for fmt in qf.by_kind("mx_block"):
            if fmt.element_bits == bits:
                return fmt
        return None
    if isinstance(attr, PackedIntTensorType):
        bits = attr.bit_width.value.data
        # A packed *integer* tensor: prefer an int_affine format of that width, then a packed sub-byte.
        for kind in ("int_affine", "packed_sub_byte"):
            for fmt in qf.by_kind(kind):
                if fmt.element_bits == bits:
                    return fmt
        return None
    if isinstance(attr, AffineQuantizedTensorType):
        bits = _int_width(attr.storage_type)
        for fmt in qf.by_kind("int_affine"):
            if fmt.element_bits == bits:
                return fmt
    return None


def _format_from_int_dtype(dtype: str | None, granularity: str | None) -> qf.QuantFormat | None:
    """Map an integer storage dtype token (``i8``/``i4``) to a canonical int format."""
    if not dtype:
        return None
    bits = {"i8": 8, "i4": 4}.get(dtype)
    if bits is None:
        return None
    for fmt in qf.by_kind("int_affine"):
        if fmt.element_bits == bits:
            return fmt
    return None


def quantized_tensors(module: Any) -> list[QuantizedTensor]:
    """Walk a quant-aware module and describe each quantized weight structurally.

    Handles the current op-level form (``quant_ext.dequantize_*`` with an ``input_dtype`` attribute
    over a plain ``tensor<...xiN>``) and resolves the canonical format via
    :mod:`merlin.common.quant_formats`.
    """
    dialect = load_dialect()
    if dialect is None:
        return []
    from m2m.ir.quant.ops import (
        DequantizePerChannelOp,
        DequantizePerGroupOp,
        DequantizePerTensorOp,
    )
    from xdsl.dialects.builtin import ShapedType

    out: list[QuantizedTensor] = []
    dequants = (DequantizePerTensorOp, DequantizePerChannelOp, DequantizePerGroupOp)
    for op in module.walk():
        if not isinstance(op, dequants):
            continue
        stored = op.operands[0].type
        dtype = _prop_str(op, "input_dtype")
        gran = _granularity_from_op_name(op.name)
        shape = tuple(stored.get_shape()) if isinstance(stored, ShapedType) else ()
        out.append(
            QuantizedTensor(
                op_name=op.name,
                storage_dtype=dtype or "",
                shape=shape,
                granularity=gran,
                fmt=_format_from_int_dtype(dtype, gran),
            )
        )
    return out
