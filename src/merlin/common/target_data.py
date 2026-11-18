"""Target-neutral readers of target data: the numeric format registry, RTL-facts block views and
deterministic stimulus.

Nothing here names a target; every value is read from the target's own declarations.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from merlin.common import record_formats as _schemas
from merlin.common.paths import schemas_dir
from merlin.common.record_formats import load_yaml

# ---- quant_formats -----------------------------------------------------------------------------
# Target-agnostic quantization / numeric **format** registry.
#
# A :class:`QuantFormat` describes WHAT a number is — its element encoding (bit width, and the
# exponent/mantissa split for floats), sub-byte packing, and scale representation — and never WHICH
# hardware runs it. That datatype -> compute-unit mapping belongs in a target contract's
# ``compute_units`` (see :mod:`merlin.target.capabilities.routing`), not here. Keeping the two apart is what lets
# the rest of the tooling (framework ingestion, mixed-precision split, routing, verification) stay
# **format-agnostic**: it consults the descriptor, never a hardcoded list of formats.
#
# The single source of truth is ``src/merlin/resources/schemas/quant_formats.registry.yaml`` (bundled read-only via
# :func:`merlin.common.paths.schemas_dir`), each entry validated against ``quant_format.schema.yaml``.
# A new format plugs in as a data entry — no code change. An optional ``MERLIN_QUANT_FORMATS`` overlay
# (a YAML file with the same ``{version, formats}`` shape) is merged on top, so a downstream target repo
# can contribute formats without editing this tree.
#
# This module is dependency-light (stdlib + the shared YAML/schema helpers) and side-effect free.

_REGISTRY_FILENAME = "quant_formats.registry.yaml"
_ENV_OVERLAY = "MERLIN_QUANT_FORMATS"

#: Allowed ``kind`` values. ``float_ieee``/``fp_ocp`` elements carry an exp/mantissa split;
#: ``mx_block``/``nvfp4`` are block-scaled floats whose *element* also carries exp/mantissa;
#: ``int_affine``/``packed_sub_byte`` are integers (optionally sub-byte packed).
KINDS: frozenset[str] = frozenset({"float_ieee", "int_affine", "fp_ocp", "packed_sub_byte", "mx_block", "nvfp4"})

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
        # A self-describing float element: sign + exponent + mantissa fills the element width. The
        # sign bit is counted only when the element HAS one -- an unsigned float element (the OCP
        # E8M0 block-scale type, whose name ends FNU for "finite, no sign, unsigned") spends all
        # eight of its bits on the exponent. Hardcoding the sign bit made that format unregisterable,
        # which pushed its width back into hand-written per-module tables -- the second copy this
        # registry exists to retire. Every signed format is unaffected.
        sign_bits = 1 if bool(d.get("signed", True)) else 0
        if sign_bits + exp + mant != bits:
            raise ValueError(
                f"quant format {name!r}: {sign_bits} (sign) + exp_bits({exp}) + mant_bits({mant}) "
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
            suffix = token[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    return None


def storage_bits(dtype: str) -> int:
    """Element storage width, without any capsule address-layout dependency.

    Packed formats use their declared packed width; plain machine spellings
    are parsed structurally. Scale-plane placement is NOT part of this width.
    The historical capsule helper delegates here, preserving its exceptions.
    """
    key = str(dtype)
    if key.startswith("torch."):
        key = key[len("torch.") :]
    if has(key):
        fmt = get(key)
        return int(fmt.pack_bits or fmt.element_bits)
    bits = machine_bits(key)
    if bits is None:
        raise KeyError(
            f"capsule_dram: cannot size dtype {dtype!r} — it is neither a format registered in "
            f"src/merlin/resources/schemas/quant_formats.registry.yaml ({names()}) nor a machine width; "
            f"register the format rather than assuming a width"
        )
    return bits


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


# ---- facts_view --------------------------------------------------------------------------------
# Read one named block out of an RTL-facts body -- the lookup every facts consumer used to hand-roll.
#
# A facts body lists its interfaces as ``[{"name": ..., ...}, ...]``. Measured 2026-09-14: about a dozen
# modules found a block by name with their own ``next((i for i in ... if i.get("name") == ...), ...)``,
# each choosing its own answer for a body with no ``interfaces`` key, a ``None`` list, or a non-dict entry.
# This is the one answer: absent means ``None``, and malformed entries are skipped rather than crashed on.
# Dependency-free on purpose, so any module can import it at top level without an import cycle.


def interface(body: Mapping[str, Any] | None, name: str) -> dict[str, Any] | None:
    """The interface block named ``name`` in a facts ``body``, or ``None`` when the body declares none."""
    for block in (body or {}).get("interfaces") or ():
        if isinstance(block, Mapping) and block.get("name") == name:
            return dict(block) if not isinstance(block, dict) else block
    return None


def _blocks(body: Mapping[str, Any] | None, key: str) -> tuple[dict[str, Any], ...]:
    """Every well-formed mapping in the list at ``key``; ``()`` when the body declares none.

    ABSENT and EMPTY are deliberately the same answer here. They are different facts -- "this device has
    no such thing" versus "the extractor found none" -- but nothing in the body distinguishes them, so
    reporting them apart would be inventing the distinction. A caller that needs it asks the extractor's
    own status field, never the length of this tuple.
    """
    return tuple(b for b in (body or {}).get(key) or () if isinstance(b, Mapping))


def interfaces(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """Every interface block, for a caller that enumerates rather than looks one up by name."""
    return _blocks(body, "interfaces")


def arrays(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The compute arrays a body describes. Absent on every family that declares no array."""
    return _blocks(body, "arrays")


def memories(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The on-chip stores a body describes, unfiltered and unscored.

    ``targetgen.address_space`` is the reader that turns these into stores with roles and capacities;
    this is only the shape-safe way to reach the list.
    """
    return _blocks(body, "memories")


def datapaths(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The datapaths a body describes.

    Note for any caller tempted to key off ``name``: the naming AXIS differs between families -- some
    name a datapath by its ROLE (``input``, ``accumulator``) and some by its FORMAT (``int8``,
    ``float8``) -- so a name is not a role and must not be read as one.
    """
    return _blocks(body, "datapaths")


def timing(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The per-module timing records a body describes.

    These are register-chain depths of a MODULE, not per-instruction latencies. They corroborate a
    declared cost; they do not supply one.
    """
    return _blocks(body, "timing")


def simt(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The SIMT geometry block, or ``None`` on a body that declares none."""
    block = (body or {}).get("simt")
    return block if isinstance(block, Mapping) else None


def source(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What the facts were extracted FROM, normalised to a mapping.

    One family records this as a bare string rather than a block, so every caller that reached for
    ``body["source"]["kind"]`` crashed on it. A string is returned as ``{"kind": <the string>}`` so the
    shape is one shape; absent stays ``None``.
    """
    block = (body or {}).get("source")
    if isinstance(block, Mapping):
        return dict(block)
    if isinstance(block, str) and block:
        return {"kind": block}
    return None


# ---- stimulus ----------------------------------------------------------------------------------
# Deterministic test-stimulus synthesis — the single source of truth for capsule leaf data.
#
# Every capsule operand, every C corroboration program and every perf-bench harness must fill the
# SAME bytes, or a "three-way bit-exact" agreement is comparing different problems. That shared fill
# lives here, in one place, and the C emitters below are generated from the same parameters rather
# than hand-restated at each call site.
#
# **Why the indexing matters.** The earlier fill was ``a[k] = (seed*(k+1) + k*k) % span`` over the
# FLAT row-major index. Modulo a span of 4 that expression has period 4 in ``k``, so any 2-D operand
# whose row length is a multiple of 4 — i.e. every ML shape — got *every row identical* and only two
# distinct values. Operands like that cannot expose a compiler bug: a wrong row stride, a swapped
# base offset or a transposed load all produce the same result on a matrix whose rows repeat. A
# mutation test on the gemmini C0 matmul found 4 of 6 deliberately wrong implementations PASSING.
#
# The fill is therefore indexed by ``(row, col)`` through an avalanche hash instead of by flat
# position. Determinism, the ``lo..hi`` range and the pure-integer construction are unchanged; what
# changes is that rows and columns now differ from one another, so row/column/stride/transpose bugs
# alter the output and the grader can see them.

_U32 = 0xFFFFFFFF

#: The DEFAULT range, kept for byte-compatibility with every existing capsule golden.
DEFAULT_LO, DEFAULT_HI = 0, 3

#: A range that spans BOTH SIGNS. Not the default -- changing the default would invalidate every
#: committed golden -- but the range a sign-sensitive operand must be filled from.
#:
#: WHY THIS EXISTS, measured. The default range is four of the 256 i8 values and every one of them
#: is non-negative, so sign, saturation and domain-error paths are untested BY CONSTRUCTION. That
#: is not theoretical: the generated host lane's `powf` was `exp(y * log(x))`, valid only for
#: x > 0, applied unconditionally. Its own docstring said "for x > 0" and nothing enforced it.
#: Because that `log` is a generated polynomial rather than a library call it cannot produce NaN,
#: so every negative base returned a FIXED FINITE 1.6516362661361307e+38 regardless of magnitude.
#: `x ** 2` inside RMSNorm therefore squared each negative activation to +1.65e38, the mean
#: overflowed to +inf, and the quantizer's clamp -- whose comparisons are all false against the
#: resulting NaN -- pinned every element to -127. That shipped in 655 compiler packages and
#: presented as a whole-model language-model miscompile (eight identical argmaxes), three layers
#: of indirection from the arithmetic that caused it. One negative stimulus value would have
#: caught it.
SIGNED_LO, SIGNED_HI = -3, 3


def mix(r: int, c: int, s: int) -> int:
    """A deterministic 32-bit avalanche hash of ``(row, col, salt)``.

    Structural and integer-only (no regex, no float, no RNG), so the value is identical across
    runs, machines and Python versions — and is cheap to restate in C (see :func:`c_fill_lines`).
    """
    x = ((r + 1) * 0x9E3779B1) ^ ((c + 1) * 0x85EBCA77) ^ ((s + 1) * 0xC2B2AE3D)
    x &= _U32
    x ^= x >> 15
    x = (x * 0x2545F491) & _U32
    x ^= x >> 13
    x = (x * 0x27D4EB2F) & _U32
    x ^= x >> 16
    return x


def det_seed(name: str) -> int:
    """The per-tensor salt derived from the tensor's name. Never zero (a zero salt would make two
    differently-named tensors collide)."""
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) or 1


def grid_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    """Collapse any rank to ``(rows, cols)``: the last dimension is the column axis and the product
    of the leading dimensions is the row axis. A rank-1 operand becomes a single row, so its
    elements still vary along the column axis."""
    dims = tuple(int(d) for d in shape)
    if not dims:
        return (1, 1)
    cols = dims[-1]
    rows = 1
    for d in dims[:-1]:
        rows *= d
    return (max(rows, 1), max(cols, 1))


def fill(name: str, shape: tuple[int, ...], lo: int = DEFAULT_LO, hi: int = DEFAULT_HI) -> list[int]:
    """Flat row-major integer stimulus for ``shape``, deterministic in ``name``.

    Values lie in ``lo..hi`` inclusive. The result is indexed by ``(row, col)``, so distinct rows
    and distinct columns are the norm rather than the exception — see the module docstring.
    """
    rows, cols = grid_shape(shape)
    span = hi - lo + 1
    if span <= 0:
        raise ValueError(f"empty stimulus range: lo={lo} hi={hi}")
    s = det_seed(name)
    if rows * cols >= _ARRAY_FILL_MIN:
        filled = _array_fill(rows, cols, s, lo, span)
        if filled is not None:
            return filled
    return [lo + (mix(r, c, s) % span) for r in range(rows) for c in range(cols)]


#: Fills smaller than this stay on the scalar :func:`mix` loop.
_ARRAY_FILL_MIN = 1 << 14
_ARRAY_FILL_CHUNK = 1 << 22


def _array_fill(rows: int, cols: int, s: int, lo: int, span: int) -> list[int] | None:
    """:func:`fill`'s exact values computed over uint64 arrays, or ``None`` without numpy.

    Generating a large operand one :func:`mix` call at a time cost seconds per tensor in every grade.
    Each step of :func:`mix` keeps only the low 32 bits, and the low 32 bits of a product or XOR do
    not depend on bits above 2**64, so uint64 wraparound leaves every value unchanged. Rows are
    processed in bounded chunks so a large operand never needs several full-size temporaries.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    if s + 1 >= 1 << 32 or rows + 1 >= 1 << 32 or cols + 1 >= 1 << 32:
        return None
    u = np.uint64
    column = (np.arange(1, cols + 1, dtype=u) * u(0x85EBCA77)) ^ u(((s + 1) * 0xC2B2AE3D) & _U32)
    chunks = []
    step = max(1, _ARRAY_FILL_CHUNK // cols)
    for start in range(0, rows, step):
        row = np.arange(start + 1, min(rows, start + step) + 1, dtype=u)[:, None] * u(0x9E3779B1)
        x = (row ^ column) & u(_U32)
        x ^= x >> u(15)
        x = (x * u(0x2545F491)) & u(_U32)
        x ^= x >> u(13)
        x = (x * u(0x27D4EB2F)) & u(_U32)
        x ^= x >> u(16)
        chunks.append((x % u(span)).astype(np.int64).ravel() + lo)
    return np.concatenate(chunks).tolist()


def fill_signed(name: str, shape: tuple[int, ...]) -> list[int]:
    """:func:`fill` over :data:`SIGNED_LO`..:data:`SIGNED_HI` -- values of BOTH signs.

    Use this for any operand that reaches a sign-sensitive primitive: a power, a logarithm, a
    square root, a reciprocal, a saturating narrow, or an absolute value. See :data:`SIGNED_LO`
    for what the non-negative default cost once.
    """
    return fill(name, shape, lo=SIGNED_LO, hi=SIGNED_HI)


def sign_coverage(values: "list[int]") -> dict[str, bool]:
    """Which sign classes ``values`` actually contains.

    A stimulus that claims to exercise a sign-sensitive path must be CHECKED to, rather than
    assumed to: `fill_signed` draws from a hash, so a small or narrow operand can legitimately
    miss a class, and a test asserting "this covers negatives" is worthless if it never looks.
    """
    return {
        "negative": any(v < 0 for v in values),
        "zero": any(v == 0 for v in values),
        "positive": any(v > 0 for v in values),
    }


# --------------------------------------------------------------------------------------------
# C emission — the same fill, for the baremetal reference programs
# --------------------------------------------------------------------------------------------
# A C program that fills its own leaves must produce byte-identical data to `fill` above, so the
# expression is emitted from here rather than restated at each call site. Kept as plain C99 with
# uint32_t arithmetic so the wraparound matches the Python `& 0xFFFFFFFF` exactly.

C_MIX_FN = r"""
/* Deterministic (row,col,salt) avalanche hash -- byte-identical to merlin.common.target_data.mix. */
static uint32_t merlin_mix(uint32_t r, uint32_t c, uint32_t s) {
  uint32_t x = ((r + 1u) * 0x9E3779B1u) ^ ((c + 1u) * 0x85EBCA77u) ^ ((s + 1u) * 0xC2B2AE3Du);
  x ^= x >> 15; x *= 0x2545F491u;
  x ^= x >> 13; x *= 0x27D4EB2Fu;
  x ^= x >> 16;
  return x;
}
"""


def _c_value(seed: str, span: int, lo: int, cast: str) -> str:
    """The C expression for one stimulus element, in SIGNED arithmetic.

    `lo + hash % span` must not be written as `{lo} + merlin_mix(...)%{span}u`: `%u` yields an
    UNSIGNED result, so a negative `lo` is promoted to unsigned and the sum becomes ~4.29e9. That
    happens to survive a cast to an integer type, because two's-complement truncation restores the
    value -- but `(float)(-3 + 0u)` is 4294967293.0, not -3.0. The whole point of this module is
    that the C program and `fill` produce IDENTICAL bytes; a stimulus that is right for `elem_t`
    and wrong for `float` makes a "three-way bit-exact" agreement compare different problems.
    Casting the modulus to `int32_t` before adding `lo` keeps the arithmetic signed for every cast.
    """
    return f"({cast})((int32_t)(merlin_mix(r,c,(uint32_t)({seed}))%{span}u) + ({lo}))"


def c_fill_loop(
    dest: str, rows: str, cols: str, seed: str, *, cast: str = "elem_t", lo: int = 0, hi: int = 3, indent: str = "  "
) -> str:
    """A C statement filling ``dest`` (a flat row-major buffer) with the same values :func:`fill`
    produces. ``rows``/``cols``/``seed`` are C expressions so the caller can pass macros."""
    span = hi - lo + 1
    return (
        f"{indent}for (uint32_t r=0;r<(uint32_t)({rows});r++) "
        f"for (uint32_t c=0;c<(uint32_t)({cols});c++) "
        f"{{ ({dest})[r*(uint32_t)({cols})+c] = {_c_value(seed, span, lo, cast)}; }}"
    )


def c_fill_loop_2d(
    dest: str, rows: str, cols: str, seed: str, *, cast: str = "elem_t", lo: int = 0, hi: int = 3, indent: str = "  "
) -> str:
    """Like :func:`c_fill_loop` but for a C array declared as ``dest[rows][cols]``."""
    span = hi - lo + 1
    return (
        f"{indent}for (uint32_t r=0;r<(uint32_t)({rows});r++) "
        f"for (uint32_t c=0;c<(uint32_t)({cols});c++) "
        f"{{ ({dest})[r][c] = {_c_value(seed, span, lo, cast)}; }}"
    )
