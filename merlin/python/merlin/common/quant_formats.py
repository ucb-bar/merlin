"""Target-agnostic quantization / numeric **format** registry.

A :class:`QuantFormat` describes WHAT a number is — its element encoding (bit width, and the
exponent/mantissa split for floats), sub-byte packing, and scale representation — and never WHICH
hardware runs it. That datatype -> compute-unit mapping belongs in a target contract's
``compute_units`` (see :mod:`merlin.targetgen.routing`), not here. Keeping the two apart is what lets
the rest of the tooling (framework ingestion, mixed-precision split, routing, verification) stay
**format-agnostic**: it consults the descriptor, never a hardcoded list of formats.

The single source of truth is ``merlin/schemas/quant_formats.registry.yaml`` (bundled read-only via
:func:`merlin.common.paths.schemas_dir`), each entry validated against ``quant_format.schema.yaml``.
A new format plugs in as a data entry — no code change. An optional ``MERLIN_QUANT_FORMATS`` overlay
(a YAML file with the same ``{version, formats}`` shape) is merged on top, so a downstream target repo
can contribute formats without editing this tree.

This module is dependency-light (stdlib + the shared YAML/schema helpers) and side-effect free.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from merlin.common import schemas as _schemas
from merlin.common.paths import schemas_dir
from merlin.common.yaml import load_yaml

_REGISTRY_FILENAME = "quant_formats.registry.yaml"
_ENV_OVERLAY = "MERLIN_QUANT_FORMATS"

#: Allowed ``kind`` values. ``float_ieee``/``fp_ocp`` elements carry an exp/mantissa split;
#: ``mx_block``/``nvfp4`` are block-scaled floats whose *element* also carries exp/mantissa;
#: ``int_affine``/``packed_sub_byte`` are integers (optionally sub-byte packed).
KINDS: frozenset[str] = frozenset(
    {"float_ieee", "int_affine", "fp_ocp", "packed_sub_byte", "mx_block", "nvfp4"}
)

#: Kinds whose *element* is a float with an explicit exponent/mantissa split.
_FLOAT_ELEMENT_KINDS: frozenset[str] = frozenset({"float_ieee", "fp_ocp", "mx_block", "nvfp4"})

#: Allowed ``scale.kind`` values.
SCALE_KINDS: frozenset[str] = frozenset(
    {
        "none",
        "per_tensor",
        "per_channel",
        "per_group",
        "block_affine",
        "block_e8m0",
        "nvfp4_block",
        "kquant_superblock",
    }
)

#: ``scale.kind`` values that carry a block dimension.
_BLOCK_SCALE_KINDS: frozenset[str] = frozenset(
    {"per_group", "block_affine", "block_e8m0", "nvfp4_block", "kquant_superblock"}
)


@dataclass(frozen=True)
class Scale:
    """How the scale (and optional zero point) that dequantizes the format is laid out."""

    kind: str = "none"
    block: int | None = None
    dtype: str | None = None

    @property
    def is_block(self) -> bool:
        return self.kind in _BLOCK_SCALE_KINDS


@dataclass(frozen=True)
class QuantFormat:
    """A single, target-agnostic numeric/quantization format descriptor.

    Attributes:
        name: canonical identifier (e.g. ``"fp4_e2m1"``, ``"int8"``, ``"gguf_q6_k"``).
        kind: one of :data:`KINDS`.
        element_bits: logical bit width of one element.
        exp_bits / mant_bits: exponent/mantissa split for float-valued kinds (else ``None``).
        signed: whether the element is signed.
        pack_bits: logical width when sub-byte packed into bytes (``None`` = byte-aligned storage).
        pack_dim: dimension packed along (``None`` unless packed).
        scale: the :class:`Scale` describing dequantization metadata.
        granularity: coarse granularity tag (per_tensor/per_channel/per_group/per_block/...).
        aliases: alternate names resolvable via :func:`get`.
        torchao_scheme / ggml_type: cross-reference to the producing source's own name.
        quant_ext_type: the model2MLIR ``quant_ext`` metadata type used to carry it in IR
            (``affine_tensor`` | ``packed_int_tensor`` | ``mx_tensor`` | ``nvfp4_tensor``).
    """

    name: str
    kind: str
    element_bits: int
    exp_bits: int | None = None
    mant_bits: int | None = None
    signed: bool = True
    pack_bits: int | None = None
    pack_dim: int | None = None
    scale: Scale = field(default_factory=Scale)
    granularity: str | None = None
    aliases: tuple[str, ...] = ()
    torchao_scheme: str | None = None
    ggml_type: str | None = None
    quant_ext_type: str | None = None
    notes: str = ""

    @property
    def is_float(self) -> bool:
        """True when the element is a float carrying an exponent/mantissa split."""
        return self.exp_bits is not None and self.mant_bits is not None

    @property
    def is_sub_byte(self) -> bool:
        return self.element_bits < 8

    @property
    def is_block_scaled(self) -> bool:
        return self.scale.is_block


def _as_scale(raw: Any) -> Scale:
    if raw is None:
        return Scale()
    if not isinstance(raw, dict):
        raise ValueError(f"scale must be a mapping, got {type(raw).__name__}")
    return Scale(kind=raw.get("kind", "none"), block=raw.get("block"), dtype=raw.get("dtype"))


def _validate_entry(name: str, d: dict[str, Any]) -> None:
    """Structured per-entry validation (presence via schema, then field semantics). No regex.

    ``d`` is the entry body; the format's ``name`` is the registry key, injected here so the
    presence check (which expects a ``name`` field) sees it.
    """
    problems = _schemas.validate({"name": name, **d}, "quant_format")
    if problems:
        raise ValueError(f"quant format {name!r}: {'; '.join(problems)}")

    kind = d["kind"]
    if kind not in KINDS:
        raise ValueError(f"quant format {name!r}: kind {kind!r} not in {sorted(KINDS)}")

    bits = d["element_bits"]
    if not isinstance(bits, int) or bits <= 0:
        raise ValueError(f"quant format {name!r}: element_bits must be a positive int, got {bits!r}")

    exp, mant = d.get("exp_bits"), d.get("mant_bits")
    if kind in _FLOAT_ELEMENT_KINDS:
        if not isinstance(exp, int) or not isinstance(mant, int):
            raise ValueError(f"quant format {name!r}: {kind} requires int exp_bits + mant_bits")
        # A self-describing float element: sign + exponent + mantissa fills the element width.
        if 1 + exp + mant != bits:
            raise ValueError(
                f"quant format {name!r}: 1 + exp_bits({exp}) + mant_bits({mant}) "
                f"!= element_bits({bits})"
            )
    elif exp is not None or mant is not None:
        raise ValueError(f"quant format {name!r}: exp/mant_bits only valid for float kinds")

    scale = _as_scale(d.get("scale"))
    if scale.kind not in SCALE_KINDS:
        raise ValueError(f"quant format {name!r}: scale.kind {scale.kind!r} not in {sorted(SCALE_KINDS)}")
    if scale.is_block and not isinstance(scale.block, int):
        raise ValueError(f"quant format {name!r}: scale.kind {scale.kind!r} requires an int block size")

    pack = d.get("pack")
    if pack is not None:
        if not isinstance(pack, dict) or not isinstance(pack.get("bits"), int):
            raise ValueError(f"quant format {name!r}: pack must be a mapping with an int 'bits'")


def _build(name: str, d: dict[str, Any]) -> QuantFormat:
    _validate_entry(name, d)
    pack = d.get("pack") or {}
    aliases = tuple(d.get("aliases", ()) or ())
    return QuantFormat(
        name=name,
        kind=d["kind"],
        element_bits=d["element_bits"],
        exp_bits=d.get("exp_bits"),
        mant_bits=d.get("mant_bits"),
        signed=bool(d.get("signed", True)),
        pack_bits=pack.get("bits"),
        pack_dim=pack.get("dim"),
        scale=_as_scale(d.get("scale")),
        granularity=d.get("granularity"),
        aliases=aliases,
        torchao_scheme=d.get("torchao_scheme"),
        ggml_type=d.get("ggml_type"),
        quant_ext_type=d.get("quant_ext_type"),
        notes=d.get("notes", ""),
    )


def _load_file(path: Path) -> dict[str, dict[str, Any]]:
    doc = load_yaml(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("formats"), dict):
        raise ValueError(f"quant-format registry {path} must be a mapping with a 'formats' mapping")
    return doc["formats"]


@lru_cache(maxsize=1)
def registry() -> dict[str, QuantFormat]:
    """Return ``{name: QuantFormat}`` for every canonical format, plus overlay entries.

    The base registry ships at ``schemas_dir()/quant_formats.registry.yaml``; an optional
    ``MERLIN_QUANT_FORMATS`` file is merged on top (overlay names override base names).
    """
    raw: dict[str, dict[str, Any]] = dict(_load_file(schemas_dir() / _REGISTRY_FILENAME))
    overlay = os.environ.get(_ENV_OVERLAY)
    if overlay:
        raw.update(_load_file(Path(overlay)))
    return {name: _build(name, entry) for name, entry in raw.items()}


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for fmt in registry().values():
        idx[fmt.name] = fmt.name
        for a in fmt.aliases:
            idx[a] = fmt.name
    return idx


def get(name: str) -> QuantFormat:
    """Resolve a format by canonical name or alias. Raises ``KeyError`` with the known names."""
    reg = registry()
    resolved = _alias_index().get(name)
    if resolved is None:
        raise KeyError(f"unknown quant format {name!r}; known: {sorted(reg)}")
    return reg[resolved]


def has(name: str) -> bool:
    return name in _alias_index()


def names() -> list[str]:
    return sorted(registry())


def by_kind(kind: str) -> list[QuantFormat]:
    return [f for f in registry().values() if f.kind == kind]


def machine_bits(token: str) -> int | None:
    """Bit width for a PLAIN machine scalar spelling (``i8`` / ``i32`` / ``f32`` / ``int8``), else None.

    These are deliberately NOT registry entries: an accumulator width is a machine type, not a way of
    encoding a quantized value, and the registry describes the latter. Parsed structurally (known prefix
    + decimal width) rather than pattern-matched, and a spelling this does not recognize returns None so
    the caller fails closed instead of assuming a width. Note this rejects MLIR's float spellings
    (``f8E4M3FN``) on purpose — those name a registry format and resolve through :func:`get`.
    """
    for prefix in ("float", "uint", "int", "f", "u", "i"):
        if token.startswith(prefix):
            suffix = token[len(prefix):]
            if suffix.isdigit():
                return int(suffix)
    return None


def is_element_dtype(token: str) -> bool:
    """True if ``token`` names an element type this tooling can reason about — a registered format (by
    canonical name or alias) or a plain machine width. The vocabulary check for artifacts that declare a
    dtype as data (capsules, contracts), so none of them has to carry its own copy of the list."""
    return has(token) or machine_bits(token) is not None


def from_torchao(scheme_name: str) -> QuantFormat | None:
    """Return the canonical format a torchAO scheme produces, if one is registered."""
    for f in registry().values():
        if f.torchao_scheme == scheme_name:
            return f
    return None


def from_ggml(ggml_type: str) -> QuantFormat | None:
    """Return the canonical format for a GGUF/GGML quantization type name, if registered."""
    for f in registry().values():
        if f.ggml_type == ggml_type:
            return f
    return None
